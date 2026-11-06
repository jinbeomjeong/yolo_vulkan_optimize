#include "decode/ffmpeg_drm_decoder.hpp"

#include <drm_fourcc.h>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_drm.h>
#include <libavutil/pixdesc.h>
}

#include <cstdio>
#include <memory>
#include "decode/sw_sand_frames.hpp"
namespace yvo {

namespace {

// get_format callback: only DRM_PRIME (HW decode, dma-buf output) is allowed.
// Silently falling back to a SW format would break the pipeline's assumptions, so return NONE to make it fail.
AVPixelFormat pick_drm_prime(AVCodecContext* /*ctx*/, const AVPixelFormat* fmts) {
    for (const AVPixelFormat* p = fmts; *p != AV_PIX_FMT_NONE; ++p) {
        if (*p == AV_PIX_FMT_DRM_PRIME) return *p;
    }
    return AV_PIX_FMT_NONE;
}

std::string av_err_str(int err) {
    char buf[AV_ERROR_MAX_STRING_SIZE] = {};
    av_strerror(err, buf, sizeof(buf));
    return buf;
}

} // namespace

bool drm_descriptor_to_frame(const AVDRMFrameDescriptor& desc, int width, int height, int coded_height,
                             DrmFrame& out, std::string* why) {
    auto reject = [why](const char* msg) {
        if (why) *why = msg;
        return false;
    };
    if (width <= 0 || height <= 0 || (width % 2) != 0 || (height % 2) != 0) return reject("bad frame size");
    if (desc.nb_objects < 1 || desc.nb_objects > 2) return reject("nb_objects not in [1, 2]");
    if (desc.nb_layers != 1) return reject("nb_layers != 1");
    const AVDRMLayerDescriptor& layer = desc.layers[0];
    if (layer.format != DRM_FORMAT_NV12) return reject("layer format is not NV12");
    if (layer.nb_planes != 2) return reject("NV12 layer nb_planes != 2");
    const AVDRMPlaneDescriptor& py = layer.planes[0];
    const AVDRMPlaneDescriptor& puv = layer.planes[1];
    if (py.object_index < 0 || py.object_index >= desc.nb_objects || puv.object_index < 0 ||
        puv.object_index >= desc.nb_objects)
        return reject("plane object_index out of range");
    const AVDRMObjectDescriptor& oy = desc.objects[py.object_index];
    const AVDRMObjectDescriptor& ouv = desc.objects[puv.object_index];
    if (fourcc_mod_broadcom_mod(oy.format_modifier) != DRM_FORMAT_MOD_BROADCOM_SAND128 ||
        fourcc_mod_broadcom_mod(ouv.format_modifier) != DRM_FORMAT_MOD_BROADCOM_SAND128)
        return reject("modifier is not Broadcom SAND128");
    if (py.offset < 0 || puv.offset < 0 || py.pitch <= 0 || puv.pitch <= 0)
        return reject("negative offset or non-positive pitch");

    // SAND column height (rows): if the modifier parameter is nonzero, use that value (kernel DRM convention, common to all planes).
    // If 0, per the rpi ffmpeg convention each plane's pitch field is that plane's column height.
    // Measured on Pi 5 HEVC: separate Y/UV objects, param=0, pitch Y=1088 / UV=544.
    const std::uint64_t param_y = static_cast<std::uint64_t>(fourcc_mod_broadcom_param(oy.format_modifier));
    const std::uint64_t param_uv = static_cast<std::uint64_t>(fourcc_mod_broadcom_param(ouv.format_modifier));
    const std::uint64_t col_y = param_y ? param_y : static_cast<std::uint64_t>(py.pitch);
    const std::uint64_t col_uv = param_uv ? param_uv : static_cast<std::uint64_t>(puv.pitch);

    const std::uint64_t w = static_cast<std::uint64_t>(width);
    const std::uint64_t h = static_cast<std::uint64_t>(height);
    if (col_y < h || col_uv < h / 2) return reject("column height smaller than plane height");

    // Check that the maximum byte position read from each plane lies inside the object
    const std::uint64_t last_col = (w - 1) / 128;
    const std::uint64_t last_x = (w - 1) % 128;
    const std::uint64_t y_end = static_cast<std::uint64_t>(py.offset) + last_col * 128 * col_y + (h - 1) * 128 + last_x + 1;
    const std::uint64_t uv_end =
        static_cast<std::uint64_t>(puv.offset) + last_col * 128 * col_uv + (h / 2 - 1) * 128 + last_x + 1;
    if (y_end > oy.size) return reject("Y plane exceeds its object");
    if (uv_end > ouv.size) return reject("UV plane exceeds its object");

    out.fd = oy.fd;
    out.size = oy.size;
    out.modifier = oy.format_modifier;
    out.drm_format = layer.format;
    out.width = width;
    out.height = height;
    out.coded_height = coded_height > 0 ? coded_height : height;
    out.y_offset = static_cast<std::size_t>(py.offset);
    out.y_pitch = static_cast<std::size_t>(py.pitch);
    out.uv_offset = static_cast<std::size_t>(puv.offset);
    out.uv_pitch = static_cast<std::size_t>(puv.pitch);
    out.sand_col_height = static_cast<std::uint32_t>(col_y);
    out.uv_fd = ouv.fd;
    out.uv_size = ouv.size;
    out.uv_sand_col_height = static_cast<std::uint32_t>(col_uv);
    return true;
}

std::string describe_drm_descriptor(const AVDRMFrameDescriptor& d) {
    std::string s;
    char line[256];
    std::snprintf(line, sizeof(line), "nb_objects=%d nb_layers=%d\n", d.nb_objects, d.nb_layers);
    s += line;
    for (int o = 0; o < d.nb_objects && o < AV_DRM_MAX_PLANES; ++o) {
        const auto& ob = d.objects[o];
        std::snprintf(line, sizeof(line), "  object[%d]: fd=%d size=%zu modifier=0x%016llx (param=%d)\n", o, ob.fd,
                      ob.size, static_cast<unsigned long long>(ob.format_modifier),
                      fourcc_mod_broadcom_param(ob.format_modifier));
        s += line;
    }
    for (int l = 0; l < d.nb_layers && l < AV_DRM_MAX_PLANES; ++l) {
        const auto& ly = d.layers[l];
        char cc[5] = {static_cast<char>(ly.format & 0xff), static_cast<char>((ly.format >> 8) & 0xff),
                      static_cast<char>((ly.format >> 16) & 0xff), static_cast<char>((ly.format >> 24) & 0xff), 0};
        std::snprintf(line, sizeof(line), "  layer[%d]: format=%s(0x%08x) nb_planes=%d\n", l, cc, ly.format,
                      ly.nb_planes);
        s += line;
        for (int p = 0; p < ly.nb_planes && p < AV_DRM_MAX_PLANES; ++p) {
            std::snprintf(line, sizeof(line), "    plane[%d]: object=%d offset=%td pitch=%td\n", p,
                          ly.planes[p].object_index, ly.planes[p].offset, ly.planes[p].pitch);
            s += line;
        }
    }
    return s;
}

FfmpegDrmDecoder::~FfmpegDrmDecoder() { close(); }

void FfmpegDrmDecoder::close() {
    if (codec_ && !eof_) avcodec_flush_buffers(codec_); // stream closed mid-way: discard frames being decoded
    av_packet_free(&pkt_);
    avcodec_free_context(&codec_);
    av_buffer_unref(&hw_device_);
    avformat_close_input(&fmt_);
    stream_index_ = -1;
    input_done_ = false;
    eof_ = false;
    packet_gate_ = nullptr; frame_filter_ = nullptr; packets_sent_ = 0; frame_counter_ = 0; sw_pool_.reset(); software_ = false;  // SW frames still held keep the pool alive; sw_stats_ stays readable
}

bool FfmpegDrmDecoder::fail(const std::string& msg) {
    error_ = msg;
    return false;
}

bool FfmpegDrmDecoder::fail_av(const char* what, int err) {
    return fail(std::string(what) + ": " + av_err_str(err));
}

bool FfmpegDrmDecoder::open(const std::string& path, const Options& opt) {
    close();
    error_.clear();

    int ret = avformat_open_input(&fmt_, path.c_str(), nullptr, nullptr);
    if (ret < 0) return fail_av("avformat_open_input", ret);
    ret = avformat_find_stream_info(fmt_, nullptr);
    if (ret < 0) return fail_av("avformat_find_stream_info", ret);

    const AVCodec* dec = nullptr;
    ret = av_find_best_stream(fmt_, AVMEDIA_TYPE_VIDEO, -1, -1, &dec, 0);
    if (ret < 0) return fail_av("av_find_best_stream", ret);
    stream_index_ = ret;
    const AVStream* st = fmt_->streams[stream_index_];
    if (st->codecpar->codec_id != AV_CODEC_ID_HEVC) return fail("video stream is not HEVC");

    codec_ = avcodec_alloc_context3(opt.software ? avcodec_find_decoder_by_name("hevc") : dec);
    if (!codec_) return fail("avcodec_alloc_context3 failed");
    ret = avcodec_parameters_to_context(codec_, st->codecpar);
    if (ret < 0) return fail_av("avcodec_parameters_to_context", ret);

    // HW: DRM hw device, get_format accepts DRM_PRIME only. SW: no hw device, native libavcodec HEVC; next() copies each frame to
    // a dma-heap SAND128 buffer. Neither path falls back to the other.
    const bool backend_ok = opt.software ? configure_sw_codec() : configure_hw_codec(opt);
    if (!backend_ok) return false;
    software_ = opt.software;
    sw_stats_ = {};
    codec_->pkt_timebase = st->time_base;
    // The library default is 1 thread. Enabling frame threading greatly raises HW decode throughput (1 thread 266 → 4 threads ~400 FPS).
    codec_->thread_count = opt.threads > 0 ? opt.threads : 0;

    ret = avcodec_open2(codec_, nullptr, nullptr);
    if (ret < 0) return fail_av("avcodec_open2", ret);

    pkt_ = av_packet_alloc();
    if (!pkt_) return fail("av_packet_alloc failed");
    return !software_ || create_sw_pool(opt);
}

std::optional<DrmFrame> FfmpegDrmDecoder::next() {
    if (!codec_ || eof_) return std::nullopt;
    auto wait_begin = software_ ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};

    std::unique_ptr<AVFrame, void (*)(AVFrame*)> frame(av_frame_alloc(),
                                                        [](AVFrame* f) { av_frame_free(&f); });
    if (!frame) {
        fail("av_frame_alloc failed");
        return std::nullopt;
    }

    for (;;) {
        int ret = avcodec_receive_frame(codec_, frame.get());
        if (ret == 0) break;
        if (ret == AVERROR_EOF) {
            eof_ = true;
            return std::nullopt;
        }
        if (ret != AVERROR(EAGAIN)) {
            fail_av("avcodec_receive_frame", ret);
            return std::nullopt;
        }
        // The decoder wants more input: send one video packet (flush at the end)
        if (input_done_) {
            fail("decoder returned EAGAIN after flush");
            return std::nullopt;
        }
        for (;;) {
            if (packet_gate_) {   // live input: the packet does not exist before its arrival time
                const auto g0 = std::chrono::steady_clock::now();
                if (!packet_gate_(packets_sent_)) return std::nullopt;
                if (software_) wait_begin += std::chrono::steady_clock::now() - g0;
            }
            ret = av_read_frame(fmt_, pkt_);
            if (ret == AVERROR_EOF) {
                input_done_ = true;
                ret = avcodec_send_packet(codec_, nullptr);
                if (ret < 0 && ret != AVERROR_EOF) {
                    fail_av("avcodec_send_packet(flush)", ret);
                    return std::nullopt;
                }
                break;
            }
            if (ret < 0) {
                fail_av("av_read_frame", ret);
                return std::nullopt;
            }
            const bool video = pkt_->stream_index == stream_index_;
            if (video) ret = avcodec_send_packet(codec_, pkt_);
            av_packet_unref(pkt_);
            if (!video) continue;
            if (ret < 0) {
                fail_av("avcodec_send_packet", ret);
                return std::nullopt;
            }
            ++packets_sent_;
            break;
        }
    }

    if (software_) return sw_frame_to_drm(*frame, wait_begin);

    if (frame->format != AV_PIX_FMT_DRM_PRIME || !frame->data[0]) {
        const char* name = av_get_pix_fmt_name(static_cast<AVPixelFormat>(frame->format));
        fail(std::string("unexpected frame format: ") + (name ? name : "?"));
        return std::nullopt;
    }
    const auto* desc = reinterpret_cast<const AVDRMFrameDescriptor*>(frame->data[0]);

    DrmFrame out;
    std::string why;
    if (!drm_descriptor_to_frame(*desc, frame->width, frame->height, codec_->coded_height, out, &why)) {
        fail("unsupported DRM descriptor: " + why + "\n" + describe_drm_descriptor(*desc));
        return std::nullopt;
    }
    out.pts = frame_counter_++;
    // Hand AVFrame ownership to keepalive. When the last reference is released, the buffer returns to the decoder pool.
    AVFrame* raw = frame.release();
    out.keepalive = std::shared_ptr<void>(raw, [](void* p) {
        auto* f = static_cast<AVFrame*>(p);
        av_frame_free(&f);
    });
    if (frame_filter_) {   // --decode-copy: the CPU copy replaces the frame; the AVFrame goes back to the pool here
        why.clear();
        if (!frame_filter_(out, why)) {
            fail(why.empty() ? std::string("frame filter failed") : why);
            return std::nullopt;
        }
    }
    return out;
}

namespace {
// get_format callback of the SW path: never lets a hwaccel format win (first format without AV_PIX_FMT_FLAG_HWACCEL).
AVPixelFormat pick_software_format(AVCodecContext* /*ctx*/, const AVPixelFormat* fmts) {
    for (; *fmts != AV_PIX_FMT_NONE; ++fmts) {
        const AVPixFmtDescriptor* d = av_pix_fmt_desc_get(*fmts);
        if (d && !(d->flags & AV_PIX_FMT_FLAG_HWACCEL)) return *fmts;
    }
    return AV_PIX_FMT_NONE;
}
} // namespace

bool FfmpegDrmDecoder::configure_hw_codec(const Options& opt) {
    const int ret = av_hwdevice_ctx_create(&hw_device_, AV_HWDEVICE_TYPE_DRM, nullptr, nullptr, 0);
    if (ret < 0) return fail_av("av_hwdevice_ctx_create(DRM)", ret);
    codec_->hw_device_ctx = av_buffer_ref(hw_device_);
    if (!codec_->hw_device_ctx) return fail("av_buffer_ref(hw_device) failed");
    codec_->get_format = pick_drm_prime;
    codec_->extra_hw_frames = opt.extra_hw_frames;
    return true;
}

bool FfmpegDrmDecoder::configure_sw_codec() {
    codec_->get_format = pick_software_format;
    codec_->thread_type = FF_THREAD_FRAME | FF_THREAD_SLICE;
    return true;
}

bool FfmpegDrmDecoder::create_sw_pool(const Options& opt) {
    if (codec_->width <= 0 || codec_->height <= 0) return fail("--decoder sw: stream size is unknown after avcodec_open2");
    SandFramePool::Config cfg;
    cfg.width = codec_->width;
    cfg.height = codec_->height;
    // The same number of frames the HW pool lets the user hold (Options::extra_hw_frames comment: about extra + 4).
    cfg.count = (opt.extra_hw_frames >= 0 ? opt.extra_hw_frames : 2 * kNumSlots) + 4;
    cfg.heaps = opt.sw_heaps;
    try {
        sw_pool_ = std::make_shared<SandFramePool>(cfg);
    } catch (const std::exception& e) {
        return fail(std::string("--decoder sw: frame pool: ") + e.what());
    }
    sw_stats_.pool_buffers = sw_pool_->count();
    sw_stats_.heap = sw_pool_->heap();
    sw_stats_.threads = codec_->thread_count;
    sw_stats_.thread_type = codec_->active_thread_type == FF_THREAD_FRAME   ? "frame"
                            : codec_->active_thread_type == FF_THREAD_SLICE ? "slice"
                                                                            : "none";
    return true;
}

std::optional<DrmFrame> FfmpegDrmDecoder::sw_frame_to_drm(const AVFrame& frame, std::chrono::steady_clock::time_point wait_begin) {
    using Clock = std::chrono::steady_clock;
    const auto ms_of = [](Clock::duration d) { return std::chrono::duration<double, std::milli>(d).count(); };
    const auto t_decoded = Clock::now();
    if (frame.format != AV_PIX_FMT_YUV420P) {
        const char* name = av_get_pix_fmt_name(static_cast<AVPixelFormat>(frame.format));
        fail(std::string("--decoder sw: only 8-bit yuv420p is supported, stream decodes to ") + (name ? name : "?"));
        return std::nullopt;
    }
    if (frame.width != codec_->width || frame.height != codec_->height || frame.linesize[0] <= 0 || frame.linesize[1] <= 0 ||
        frame.linesize[2] <= 0) {
        fail("--decoder sw: frame " + std::to_string(frame.width) + "x" + std::to_string(frame.height) +
             " does not fit the pool or has a negative stride (mid-stream size changes are not supported)");
        return std::nullopt;
    }
    auto lease = sw_pool_->acquire();
    if (!lease) {
        fail("--decoder sw: frame pool exhausted, all " + std::to_string(sw_pool_->count()) +
             " buffers are held downstream (raise Options::extra_hw_frames)");
        return std::nullopt;
    }
    try {
        // dma_sync_begin/end instead of DmaCpuAccess: the two sections wrap one copy, and a failure must reach this catch.
        dma_sync_begin(lease->y_fd, DmaSyncDir::Write);
        dma_sync_begin(lease->uv_fd, DmaSyncDir::Write);
        const auto t_copy0 = Clock::now();
        const std::size_t y_col = sand_y_col_height(frame.height), uv_col = sand_uv_col_height(frame.height);
        const bool copied = i420_to_sand128(frame.data[0], static_cast<std::size_t>(frame.linesize[0]), frame.data[1],
                                            static_cast<std::size_t>(frame.linesize[1]), frame.data[2],
                                            static_cast<std::size_t>(frame.linesize[2]), frame.width, frame.height, lease->y,
                                            lease->y_size, y_col, lease->uv, lease->uv_size, uv_col);
        const auto t_copy1 = Clock::now();
        dma_sync_end(lease->y_fd, DmaSyncDir::Write);
        dma_sync_end(lease->uv_fd, DmaSyncDir::Write);
        const auto t_end = Clock::now();
        if (!copied) {
            fail("--decoder sw: I420 to SAND128 copy rejected the frame layout");
            return std::nullopt;
        }

        // Same validation as a HW frame: what fails here would fail there.
        AVDRMFrameDescriptor desc{};
        desc.nb_objects = 2;
        desc.objects[0] = {lease->y_fd, lease->y_size, DRM_FORMAT_MOD_BROADCOM_SAND128};
        desc.objects[1] = {lease->uv_fd, lease->uv_size, DRM_FORMAT_MOD_BROADCOM_SAND128};
        desc.nb_layers = 1;
        desc.layers[0].format = DRM_FORMAT_NV12;
        desc.layers[0].nb_planes = 2;
        desc.layers[0].planes[0] = {0, 0, static_cast<std::ptrdiff_t>(y_col)};   // object, offset, pitch = column height (param 0)
        desc.layers[0].planes[1] = {1, 0, static_cast<std::ptrdiff_t>(uv_col)};
        DrmFrame out;
        std::string why;
        if (!drm_descriptor_to_frame(desc, frame.width, frame.height, codec_->coded_height, out, &why)) {
            fail("--decoder sw: bad frame layout: " + why + "\n" + describe_drm_descriptor(desc));
            return std::nullopt;
        }
        out.pts = frame_counter_++;
        out.keepalive = std::move(lease->keepalive);

        ++sw_stats_.frames;
        sw_stats_.decode_wait_ms += ms_of(t_decoded - wait_begin);
        sw_stats_.memcpy_ms += ms_of(t_copy1 - t_copy0);
        sw_stats_.sync_ms += ms_of((t_copy0 - t_decoded) + (t_end - t_copy1));
        sw_stats_.pool_peak_in_use = sw_pool_->peak_in_use();
        return out;
    } catch (const std::exception& e) {
        fail(std::string("--decoder sw: ") + e.what());
        return std::nullopt;
    }
}

int FfmpegDrmDecoder::width() const { return codec_ ? codec_->width : 0; }
int FfmpegDrmDecoder::height() const { return codec_ ? codec_->height : 0; }
int FfmpegDrmDecoder::thread_count() const { return codec_ ? codec_->thread_count : 0; }
int FfmpegDrmDecoder::reorder_frames() const { return codec_ ? codec_->has_b_frames : 0; }

double FfmpegDrmDecoder::frame_rate() const {
    if (!fmt_ || stream_index_ < 0) return 0.0;
    return av_q2d(fmt_->streams[stream_index_]->avg_frame_rate);
}

std::int64_t FfmpegDrmDecoder::nb_frames() const {
    if (!fmt_ || stream_index_ < 0) return 0;
    return fmt_->streams[stream_index_]->nb_frames;
}

} // namespace yvo
