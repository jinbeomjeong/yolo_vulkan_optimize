// libavcodec HEVC decoder → DrmFrame. HW (V4L2 stateless, DRM_PRIME output) or SW (CPU copy into dma-heap SAND128 buffers)
#pragma once

#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "core/frame_types.hpp"

struct AVFormatContext;
struct AVCodecContext;
struct AVBufferRef;
struct AVPacket;
struct AVFrame;
struct AVDRMFrameDescriptor;

namespace yvo {

class SandFramePool;

class FfmpegDrmDecoder {
public:
    struct Options {
        // Extra frames to allocate in the decoder pool. Needed for as many frames as the user holds at once (ring slots).
        // Measured (1080p HEVC): pool size = 6 + thread count (frame threading) + extra; frames that can safely be held ≈ extra + 4.
        // Holding more than that makes the decoder stall waiting for a free buffer (hangs without an error). -1 lets the pool grow as needed.
        int extra_hw_frames = 2 * kNumSlots;
        int threads = 0; // 0 = auto (core count), 1 = frame threading off
        // Software HEVC, no hw device: each frame is copied by the CPU to a dma-heap SAND128 NV12 buffer with the layout of the
        // HW output, so consumers cannot tell the two apart. Pool = extra_hw_frames + 4 pairs (-1 uses 2 * kNumSlots), fixed:
        // holding them all fails next() with an error instead of stalling. decode/sw_sand_frames.hpp has the reasons.
        bool software = false;
        std::vector<std::string> sw_heaps;  // dma-heap nodes tried in order; empty = DmaHeapConfig default (system heap)
    };

    // Software path only (zero on the HW path). Stays readable after close() until the next open().
    struct SwStats {
        long frames = 0;             // frames returned by next()
        double decode_wait_ms = 0;   // total: entering next() until the decoded frame is in hand (read, send, receive)
        double memcpy_ms = 0;        // total: i420_to_sand128 alone
        double sync_ms = 0;          // total: DMA_BUF_IOCTL_SYNC START + END of both planes (the cache maintenance)
        int pool_buffers = 0;
        int pool_peak_in_use = 0;
        int threads = 0;             // resolved libavcodec thread count
        std::string thread_type;     // frame / slice / none
        std::string heap;            // dma-heap node in use
    };

    FfmpegDrmDecoder() = default;
    ~FfmpegDrmDecoder();
    FfmpegDrmDecoder(const FfmpegDrmDecoder&) = delete;
    FfmpegDrmDecoder& operator=(const FfmpegDrmDecoder&) = delete;

    bool open(const std::string& path) { return open(path, Options{}); }
    bool open(const std::string& path, const Options& opt);
    void close();

    // Next frame. nullopt on EOF or error (distinguish with error()).
    // The dma-buf contents stay valid while the returned DrmFrame.keepalive is alive.
    std::optional<DrmFrame> next();

    bool eof() const { return eof_; }
    const std::string& error() const { return error_; }
    int width() const;
    int height() const;
    double frame_rate() const;      // stream avg_frame_rate
    std::int64_t nb_frames() const; // frame count reported by the container (0 if unknown)

    const SwStats& sw_stats() const { return sw_stats_; }

    // Live input (pipeline/live_input.hpp): next() calls the gate with the number of packets sent so far right before it reads
    // another packet (also before the read that finds the end of the file). The gate may sleep; its time is not counted in
    // SwStats::decode_wait_ms. false makes next() return nullopt without an error. Set after open(); close() clears it.
    void set_packet_gate(std::function<bool(std::int64_t)> gate) { packet_gate_ = std::move(gate); }
    // Post-decode hook of the HW path (pipeline/decode_copy.hpp, yvo_pipeline --decode-copy): next() passes each frame through it
    // right before returning. The filter may replace the frame (the old keepalive is then dropped); false makes next() return
    // nullopt with error() = the reason it wrote. Runs on the thread that calls next(). Set after open(); close() clears it.
    using FrameFilter = std::function<bool(DrmFrame&, std::string&)>;
    void set_frame_filter(FrameFilter filter) { frame_filter_ = std::move(filter); }
    std::int64_t packets_sent() const { return packets_sent_; }
    bool input_flushed() const { return input_done_; }   // the end of the file was reached and the flush was sent
    int thread_count() const;     // resolved libavcodec thread count
    int reorder_frames() const;   // frames the stream may hold back for display order (AVCodecContext::has_b_frames)

    // Extract the AVFrame (DRM_PRIME) held in keepalive. For inspection tools. HW path only: an SW frame holds no AVFrame.
    static AVFrame* av_frame(const DrmFrame& f) { return static_cast<AVFrame*>(f.keepalive.get()); }

private:
    bool fail(const std::string& msg);
    bool fail_av(const char* what, int err);
    bool configure_hw_codec(const Options& opt);
    bool configure_sw_codec();
    bool create_sw_pool(const Options& opt);
    std::optional<DrmFrame> sw_frame_to_drm(const AVFrame& frame, std::chrono::steady_clock::time_point wait_begin);

    AVFormatContext* fmt_ = nullptr;
    AVCodecContext* codec_ = nullptr;
    AVBufferRef* hw_device_ = nullptr;
    AVPacket* pkt_ = nullptr;
    int stream_index_ = -1;
    bool input_done_ = false; // end of packets (flush sent)
    std::function<bool(std::int64_t)> packet_gate_;
    FrameFilter frame_filter_;
    std::int64_t packets_sent_ = 0;
    bool eof_ = false;
    std::int64_t frame_counter_ = 0;
    std::string error_;
    bool software_ = false;
    std::shared_ptr<SandFramePool> sw_pool_;
    SwStats sw_stats_;
};

// AVDRMFrameDescriptor → DrmFrame fields (except keepalive/pts). Handles Y/UV in one object or two.
// For an unsupported layout (not SAND128 NV12, size mismatch) returns false with the reason.
bool drm_descriptor_to_frame(const AVDRMFrameDescriptor& desc, int width, int height, int coded_height,
                             DrmFrame& out, std::string* why = nullptr);

// Raw descriptor values as a human-readable string (for logs and inspection tools)
std::string describe_drm_descriptor(const AVDRMFrameDescriptor& desc);

} // namespace yvo
