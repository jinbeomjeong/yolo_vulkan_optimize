#include "isp/isp_resizer.hpp"

#include <cerrno>
#include <chrono>
#include <cstring>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <sys/stat.h>
#include <unistd.h>

#include <linux/media.h>

// libpisp public headers. These types never leak into isp_resizer.hpp (PIMPL).
#include "backend/backend.hpp"
#include "common/logging.hpp"
#include "common/utils.hpp"
#include "helpers/backend_device.hpp"
#include "helpers/media_device.hpp"
#include "variants/variant.hpp"

namespace yvo::isp {

namespace {

constexpr int kSandCol = 128; // SAND128: width of one column (pixels = bytes, 8bpp)

// libpisp's default boost::log sink dumps everything down to debug to stderr.
// logging_init() applies a filter at LIBPISP_LOG_LEVEL (default fatal). Once per process.
void init_libpisp_logging() {
    static std::once_flag once;
    std::call_once(once, [] { libpisp::logging_init(); });
}

const libpisp::PiSPVariant& pick_variant(unsigned int hw_revision, const std::string& media_device) {
    for (const auto& v : libpisp::get_variants()) {
        if (v.BackEndVersion() == hw_revision) return v;
    }
    std::ostringstream os;
    os << "IspResizer: no libpisp variant matches " << media_device << " hw_revision 0x" << std::hex << hw_revision
       << " (known:";
    for (const auto& v : libpisp::get_variants()) os << " " << v.Name() << "=0x" << v.BackEndVersion();
    os << ")";
    throw std::runtime_error(os.str());
}

std::string node_name(unsigned int branch) { return "pispbe-output" + std::to_string(branch); }

} // namespace

// ---------------------------------------------------------------------------

struct IspResizer::Impl {
    // Input dma-buf cache. libpisp::helpers::Buffer takes ownership of the fd and closes it on destruction,
    // so the decoder fd cannot be passed as-is. Build the Buffer from a dup'ed fd and cache it by the decoder fd number.
    // (Decoder pool fds are fixed, but reopening the decoder can reuse numbers, so verify with fstat.)
    struct InputEntry {
        libpisp::helpers::Buffer buffer;
        dev_t y_dev = 0, uv_dev = 0;
        ino_t y_ino = 0, uv_ino = 0;
        std::size_t y_size = 0, uv_size = 0;
    };

    explicit Impl(const IspResizerConfig& c) : cfg(c) {}

    const libpisp::helpers::Buffer& input_buffer(const DrmFrame& f);

    IspResizerConfig cfg;
    LetterboxParams lb;
    int content_w = 0, content_h = 0; // size of the downscaled result (excluding padding)
    int out_w = 0, out_h = 0;         // output buffer size
    std::size_t out_bytes = 0;
    std::string variant_name;
    unsigned int hw_revision = 0;
    std::string node;

    std::unique_ptr<libpisp::BackEnd> be;
    pisp_be_tiles_config tiles{};
    std::unique_ptr<libpisp::helpers::BackendDevice> dev;
    std::vector<libpisp::helpers::BufferRef> out_bufs;
    std::vector<std::uint8_t*> out_ptrs;
    std::unordered_map<int, InputEntry> in_cache;
    std::uint64_t in_hits = 0, in_imports = 0;
    double last_isp_ms = -1.0;
    std::uint32_t out_sizeimage = 0;
};

const libpisp::helpers::Buffer& IspResizer::Impl::input_buffer(const DrmFrame& f) {
    struct stat ys{}, uvs{};
    if (::fstat(f.fd, &ys) != 0 || ::fstat(f.uv_fd, &uvs) != 0)
        throw std::runtime_error(std::string("IspResizer: fstat on decoder fd failed: ") + std::strerror(errno));

    auto it = in_cache.find(f.fd);
    if (it != in_cache.end()) {
        const InputEntry& e = it->second;
        if (e.y_dev == ys.st_dev && e.y_ino == ys.st_ino && e.uv_dev == uvs.st_dev && e.uv_ino == uvs.st_ino &&
            e.y_size == f.size && e.uv_size == f.uv_size) {
            ++in_hits;
            return e.buffer;
        }
        in_cache.erase(it); // fd number reused: drop the stale entry
    }

    const int y_dup = ::dup(f.fd);
    if (y_dup < 0) throw std::runtime_error(std::string("IspResizer: dup(y_fd) failed: ") + std::strerror(errno));
    const int uv_dup = ::dup(f.uv_fd);
    if (uv_dup < 0) {
        ::close(y_dup);
        throw std::runtime_error(std::string("IspResizer: dup(uv_fd) failed: ") + std::strerror(errno));
    }
    InputEntry e;
    // Buffer takes ownership of the fd here (the destructor closes it).
    e.buffer = libpisp::helpers::Buffer(std::array<int, 3> { y_dup, uv_dup, -1 },
                                        std::array<std::size_t, 3> { f.size, f.uv_size, 0 });
    e.y_dev = ys.st_dev;
    e.y_ino = ys.st_ino;
    e.uv_dev = uvs.st_dev;
    e.uv_ino = uvs.st_ino;
    e.y_size = f.size;
    e.uv_size = f.uv_size;
    ++in_imports;
    return in_cache.emplace(f.fd, std::move(e)).first->second.buffer;
}

// ---------------------------------------------------------------------------

IspResizer::IspResizer(const IspResizerConfig& cfg) : impl_(std::make_unique<Impl>(cfg)) {
    Impl& m = *impl_;
    init_libpisp_logging();

    if (m.cfg.num_slots <= 0) throw std::invalid_argument("IspResizer: num_slots must be > 0");
    if (m.cfg.buffer_count < m.cfg.num_slots)
        throw std::invalid_argument("IspResizer: buffer_count must be >= num_slots");
    if (m.cfg.branch >= PISP_BACK_END_NUM_OUTPUTS)
        throw std::invalid_argument("IspResizer: branch out of range");
    if (m.cfg.src_w <= 0 || m.cfg.src_h <= 0 || m.cfg.dst_w <= 0 || m.cfg.dst_h <= 0)
        throw std::invalid_argument("IspResizer: invalid source/destination size");
    if ((m.cfg.src_w % kSandCol) != 0)
        throw std::invalid_argument("IspResizer: src_w must be a multiple of 128 (SAND column width)");
    if (m.cfg.coded_height < m.cfg.src_h)
        throw std::invalid_argument("IspResizer: coded_height must be >= src_h");

    m.lb = make_letterbox(m.cfg.src_w, m.cfg.src_h, m.cfg.dst_w, m.cfg.dst_h);
    m.lb.pad_value = m.cfg.pad_value;
    m.content_w = static_cast<int>(static_cast<float>(m.cfg.src_w) * m.lb.scale + 0.5f);
    m.content_h = static_cast<int>(static_cast<float>(m.cfg.src_h) * m.lb.scale + 0.5f);
    if (m.cfg.letterbox) {
        m.out_w = m.cfg.dst_w;
        m.out_h = m.cfg.dst_h;
    } else {
        m.out_w = m.content_w;
        m.out_h = m.content_h;
        m.lb.dst_w = m.out_w;
        m.lb.dst_h = m.out_h;
        m.lb.pad_x = 0;
        m.lb.pad_y = 0;
    }
    // libpisp does not range-check offset_x/offset_y. On overflow it silently writes outside the buffer.
    if (m.lb.pad_x + m.content_w > m.out_w || m.lb.pad_y + m.content_h > m.out_h)
        throw std::invalid_argument("IspResizer: letterbox window " + std::to_string(m.content_w) + "x" +
                                    std::to_string(m.content_h) + " at (" + std::to_string(m.lb.pad_x) + "," +
                                    std::to_string(m.lb.pad_y) + ") does not fit in " + std::to_string(m.out_w) + "x" +
                                    std::to_string(m.out_h));
    m.out_bytes = static_cast<std::size_t>(m.out_w) * 3u * static_cast<std::size_t>(m.out_h);

    // --- variant selection: match the media node's hw revision to the libpisp variant's BackEndVersion ---
    libpisp::helpers::MediaDevice media;
    const media_device_info info = media.DeviceInfo(m.cfg.media_device);
    m.hw_revision = info.hw_revision;
    const libpisp::PiSPVariant& variant = pick_variant(info.hw_revision, m.cfg.media_device);
    m.variant_name = variant.Name();
    m.node = node_name(m.cfg.branch);

    const unsigned int downscale = static_cast<unsigned int>(m.cfg.src_w / m.content_w);
    if (downscale > 2 && !variant.BackEndDownscalerAvailable(0, m.cfg.branch))
        throw std::runtime_error("IspResizer: branch " + std::to_string(m.cfg.branch) +
                                 " has no downscaler but a " + std::to_string(downscale) +
                                 "x reduction needs one (resampler alone handles at most 2x)");

    m.be = std::make_unique<libpisp::BackEnd>(
        libpisp::BackEnd::Config(0, 0, libpisp::BackEnd::Config::NONE, m.cfg.defaults_file), variant);
    if (static_cast<unsigned int>(m.cfg.src_w / m.content_w) > m.be->GetMaxDownscale())
        throw std::runtime_error("IspResizer: reduction exceeds GetMaxDownscale() = " +
                                 std::to_string(m.be->GetMaxDownscale()));

    // --- input format: SAND128 NV12. In the wallpaper roll format, stride is the byte count of one column. ---
    pisp_image_format_config in{};
    in.width = static_cast<std::uint16_t>(m.cfg.src_w);
    in.height = static_cast<std::uint16_t>(m.cfg.coded_height);
    in.format = libpisp::get_pisp_image_format("YUV420SP_COL128");
    in.stride = kSandCol * m.cfg.sand_col_height;
    in.stride2 = kSandCol * m.cfg.uv_sand_col_height;
    m.be->SetInputFormat(in);

    // --- global enable: whole Bayer path off, RGB uses only input/ccm/output. ---
    // The DOWNSCALE/RESAMPLE bits are filled by SetSmartResize in Prepare. CSC is RGB→YCbCr, so it is not used.
    pisp_be_global_config global{};
    global.bayer_enables = 0;
    global.rgb_enables = PISP_BE_RGB_ENABLE_INPUT | PISP_BE_RGB_ENABLE_CCM |
                         static_cast<std::uint32_t>(PISP_BE_RGB_ENABLE_OUTPUT(m.cfg.branch));
    m.be->SetGlobal(global);

    // --- YUV→RGB: in the RGB block order ycbcr_inverse comes after sharpen, so it cannot be used at the input. Load the inverse matrix into the leading ccm. ---
    pisp_be_ccm_config ccm{};
    m.be->InitialiseYcbcrInverse(ccm, m.cfg.colour_space);
    m.be->SetCcm(ccm);

    // --- crop: use only the valid src_h (1080) rows out of coded_height (1088) ---
    pisp_be_crop_config crop{};
    crop.offset_x = 0;
    crop.offset_y = 0;
    crop.width = static_cast<std::uint16_t>(m.cfg.src_w);
    crop.height = static_cast<std::uint16_t>(m.cfg.src_h);
    m.be->SetCrop(m.cfg.branch, crop);

    // --- downscale: leave the downscale + resample combination to SmartResize ---
    m.be->SetSmartResize(m.cfg.branch, { static_cast<std::uint16_t>(m.content_w),
                                         static_cast<std::uint16_t>(m.content_h) });
    if (!m.cfg.resample_filter.empty()) {
        pisp_be_resample_config resample{};
        m.be->InitialiseResample(resample, m.cfg.resample_filter);
        pisp_be_resample_extra extra{};
        extra.scaled_width = static_cast<std::uint16_t>(m.content_w);
        extra.scaled_height = static_cast<std::uint16_t>(m.content_h);
        m.be->SetResample(m.cfg.branch, resample, extra);
    }

    // --- output: packed RGB888. letterbox is built with offset_x/offset_y. ---
    pisp_be_output_format_config of{};
    of.image.width = static_cast<std::uint16_t>(m.content_w);
    of.image.height = static_cast<std::uint16_t>(m.content_h);
    of.image.format = libpisp::get_pisp_image_format("RGB888");
    of.image.stride = m.out_w * 3;
    of.image.stride2 = 0;
    pisp_be_output_format_extra extra{};
    extra.offset_x = static_cast<std::uint16_t>(m.lb.pad_x);
    extra.offset_y = static_cast<std::uint16_t>(m.lb.pad_y);
    m.be->SetOutputFormat(m.cfg.branch, of, extra);

    m.be->Prepare(&m.tiles);

    // Prepare() overwrites output_format[i].image with the resample result size (content_w x content_h).
    // The hardware writes only by tile address, so height is used only for the V4L2 format's sizeimage calculation.
    // The whole letterbox buffer (out_h rows) must also be told to the node format so sizeimage matches the real buffer size.
    if (m.tiles.config.output_format[m.cfg.branch].image.stride != m.out_w * 3)
        throw std::runtime_error("IspResizer: Prepare() changed the output stride to " +
                                 std::to_string(m.tiles.config.output_format[m.cfg.branch].image.stride));
    m.tiles.config.output_format[m.cfg.branch].image.height = static_cast<std::uint16_t>(m.out_h);
    m.tiles.config.output_format[m.cfg.branch].image.width = static_cast<std::uint16_t>(m.out_w);

    // Verify on the host that tile write addresses actually fall inside the buffer.
    for (std::uint32_t t = 0; t < m.tiles.num_tiles; ++t) {
        const pisp_tile& tile = m.tiles.tiles[t];
        const std::size_t x = static_cast<std::size_t>(tile.output_offset_x[m.cfg.branch]) +
                              static_cast<std::size_t>(m.lb.pad_x);
        const std::size_t y = static_cast<std::size_t>(tile.output_offset_y[m.cfg.branch]) +
                              static_cast<std::size_t>(m.lb.pad_y);
        const std::size_t end_x = x + tile.output_width[m.cfg.branch];
        const std::size_t end_y = y + tile.output_height[m.cfg.branch];
        if (end_x > static_cast<std::size_t>(m.out_w) || end_y > static_cast<std::size_t>(m.out_h))
            throw std::runtime_error("IspResizer: tile " + std::to_string(t) + " writes outside the output buffer");
    }

    m.out_sizeimage = static_cast<std::uint32_t>(m.out_bytes);
    if (!m.cfg.open_device) return;

    m.dev = std::make_unique<libpisp::helpers::BackendDevice>(m.cfg.media_device);
    if (!m.dev->Valid()) throw std::runtime_error("IspResizer: BackendDevice(" + m.cfg.media_device + ") is not valid");
    m.dev->Setup(m.tiles, static_cast<unsigned int>(m.cfg.buffer_count));

    auto buffers = m.dev->GetBuffers();
    const auto it = buffers.find(m.node);
    if (it == buffers.end() || it->second.size() < static_cast<std::size_t>(m.cfg.num_slots))
        throw std::runtime_error("IspResizer: " + m.node + " gave fewer buffers than num_slots");
    for (int s = 0; s < m.cfg.num_slots; ++s) {
        const libpisp::helpers::Buffer& b = it->second[static_cast<std::size_t>(s)].get();
        if (b.Size()[0] < m.out_bytes)
            throw std::runtime_error("IspResizer: output buffer " + std::to_string(s) + " is " +
                                     std::to_string(b.Size()[0]) + " B, need " + std::to_string(m.out_bytes));
        m.out_bufs.push_back(it->second[static_cast<std::size_t>(s)]);
        // Fill with the padding value only once right after allocation. Afterwards the ISP touches only the content area.
        libpisp::helpers::Buffer::Sync sync(m.out_bufs.back(), libpisp::helpers::Buffer::Sync::Access::Write);
        std::uint8_t* p = sync.Get()[0];
        if (p == nullptr) throw std::runtime_error("IspResizer: failed to mmap output buffer");
        std::memset(p, m.cfg.pad_value, m.out_bytes);
        m.out_ptrs.push_back(p);
    }
    m.out_sizeimage = static_cast<std::uint32_t>(it->second[0].get().Size()[0]);
}

IspResizer::~IspResizer() = default;

PreprocOutput IspResizer::process(const DrmFrame& frame, int slot) {
    Impl& m = *impl_;
    if (!m.dev) throw std::runtime_error("IspResizer::process: device was not opened (open_device = false)");
    if (slot < 0 || slot >= m.cfg.num_slots)
        throw std::out_of_range("IspResizer::process: slot " + std::to_string(slot) + " out of range");
    if (frame.width != m.cfg.src_w || frame.height != m.cfg.src_h || frame.coded_height != m.cfg.coded_height)
        throw std::runtime_error("IspResizer::process: frame is " + std::to_string(frame.width) + "x" +
                                 std::to_string(frame.height) + " (coded " + std::to_string(frame.coded_height) +
                                 "), configured for " + std::to_string(m.cfg.src_w) + "x" +
                                 std::to_string(m.cfg.src_h) + " (coded " + std::to_string(m.cfg.coded_height) + ")");
    if (static_cast<int>(frame.sand_col_height) != m.cfg.sand_col_height ||
        static_cast<int>(frame.uv_sand_col_height) != m.cfg.uv_sand_col_height)
        throw std::runtime_error("IspResizer::process: SAND column heights " +
                                 std::to_string(frame.sand_col_height) + "/" +
                                 std::to_string(frame.uv_sand_col_height) + " do not match the configured " +
                                 std::to_string(m.cfg.sand_col_height) + "/" +
                                 std::to_string(m.cfg.uv_sand_col_height));
    // pisp_image_format_config has no plane start offset field. A nonzero offset cannot be expressed.
    if (frame.y_offset != 0 || frame.uv_offset != 0)
        throw std::runtime_error("IspResizer::process: non-zero plane offsets (y " + std::to_string(frame.y_offset) +
                                 ", uv " + std::to_string(frame.uv_offset) + ") cannot be expressed to the ISP");
    if (frame.fd < 0 || frame.uv_fd < 0) throw std::runtime_error("IspResizer::process: frame has no dma-buf fd");

    const libpisp::helpers::Buffer& in = m.input_buffer(frame);

    std::map<std::string, libpisp::helpers::BufferRef> bufs;
    bufs.emplace("pispbe-input", std::cref(in));
    bufs.emplace(m.node, m.out_bufs[static_cast<std::size_t>(slot)]);

    const auto t0 = std::chrono::steady_clock::now();
    const int ret = m.dev->Run(bufs);
    const auto t1 = std::chrono::steady_clock::now();
    m.last_isp_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    if (ret != 0) throw std::runtime_error("IspResizer::process: BackendDevice::Run failed: " + std::to_string(ret));

    PreprocOutput out;
    out.host_ptr = m.out_ptrs[static_cast<std::size_t>(slot)];
    out.dmabuf_fd = m.out_bufs[static_cast<std::size_t>(slot)].get().Fd()[0];
    out.size = m.out_bytes;
    out.layout = TensorLayout::Nhwc; // the ISP only produces packed RGB
    out.lb = m.lb;
    out.pts = frame.pts;
    out.slot = slot;
    return out;
}

int IspResizer::num_slots() const { return impl_->cfg.num_slots; }
std::size_t IspResizer::output_size() const { return impl_->out_bytes; }
int IspResizer::out_width() const { return impl_->out_w; }
int IspResizer::out_height() const { return impl_->out_h; }
const LetterboxParams& IspResizer::letterbox() const { return impl_->lb; }
double IspResizer::last_isp_ms() const { return impl_->last_isp_ms; }
const IspResizerConfig& IspResizer::config() const { return impl_->cfg; }
void IspResizer::clear_input_cache() { impl_->in_cache.clear(); }

int IspResizer::dmabuf_fd(int slot) const {
    if (slot < 0 || slot >= static_cast<int>(impl_->out_bufs.size()))
        throw std::out_of_range("IspResizer::dmabuf_fd: slot out of range");
    return impl_->out_bufs[static_cast<std::size_t>(slot)].get().Fd()[0];
}

std::uint8_t* IspResizer::host_ptr(int slot) const {
    if (slot < 0 || slot >= static_cast<int>(impl_->out_ptrs.size()))
        throw std::out_of_range("IspResizer::host_ptr: slot out of range");
    return impl_->out_ptrs[static_cast<std::size_t>(slot)];
}

std::string IspResizer::describe() const {
    const Impl& m = *impl_;
    const pisp_be_config& c = m.tiles.config;
    const pisp_image_format_config in = c.input_format;
    const pisp_image_format_config of = c.output_format[m.cfg.branch].image;
    const pisp_be_downscale_config ds = c.downscale[m.cfg.branch];
    const pisp_be_resample_config rs = c.resample[m.cfg.branch];

    std::ostringstream os;
    os << std::dec;
    os << "variant            : " << m.variant_name << " (hw_revision 0x" << std::hex << m.hw_revision << std::dec
       << ")\n";
    os << "media device       : " << m.cfg.media_device << (m.dev ? " (open)" : " (not opened)") << "\n";
    os << "output node        : " << m.node << " (branch " << m.cfg.branch << ")\n";
    os << "bayer_enables      : 0x" << std::hex << c.global.bayer_enables << std::dec << "\n";
    os << "rgb_enables        : 0x" << std::hex << c.global.rgb_enables << std::dec << "  (INPUT|CCM|DOWNSCALE"
       << m.cfg.branch << "|RESAMPLE" << m.cfg.branch << "|OUTPUT" << m.cfg.branch << " expected)\n";
    os << "colour space (ccm) : " << m.cfg.colour_space << "  coeffs";
    for (int i = 0; i < 9; ++i) os << " " << c.ccm.coeffs[i];
    os << "  offsets " << c.ccm.offsets[0] << " " << c.ccm.offsets[1] << " " << c.ccm.offsets[2] << "\n";
    os << "input format       : " << in.width << "x" << in.height << " fmt 0x" << std::hex << in.format << std::dec
       << " stride " << in.stride << " stride2 " << in.stride2 << " (SAND cols " << (m.cfg.src_w / kSandCol) << ")\n";
    os << "input plane sizes  : y " << (static_cast<std::size_t>(m.cfg.src_w / kSandCol) * kSandCol *
                                        static_cast<std::size_t>(m.cfg.sand_col_height))
       << " uv "
       << (static_cast<std::size_t>(m.cfg.src_w / kSandCol) * kSandCol *
           static_cast<std::size_t>(m.cfg.uv_sand_col_height))
       << "\n";
    // crop is not a pisp_be_config register field but lives in BeConfigExtra, so it cannot be read back. Print the configured value.
    os << "crop[" << m.cfg.branch << "]            : (0,0) " << m.cfg.src_w << "x" << m.cfg.src_h << "\n";
    os << "downscale          : scale_factor (" << ds.scale_factor_h << "," << ds.scale_factor_v << ") recip ("
       << ds.scale_recip_h << "," << ds.scale_recip_v << ")\n";
    os << "resample           : scale_factor (" << rs.scale_factor_h << "," << rs.scale_factor_v << ")"
       << (m.cfg.resample_filter.empty() ? "  filter: smart selection" : "  filter: " + m.cfg.resample_filter) << "\n";
    os << "output format      : " << of.width << "x" << of.height << " fmt 0x" << std::hex << of.format << std::dec
       << " stride " << of.stride << "  (content " << m.content_w << "x" << m.content_h << ")\n";
    os << "letterbox          : offset (" << m.lb.pad_x << "," << m.lb.pad_y << ") scale " << m.lb.scale
       << " pad_value " << static_cast<int>(m.lb.pad_value) << (m.cfg.letterbox ? "" : "  (disabled)") << "\n";
    os << "output buffer      : " << m.out_w << "x" << m.out_h << "x3 = " << m.out_bytes << " B, "
       << (m.dev ? "node sizeimage " : "expected sizeimage ") << m.out_sizeimage << "\n";
    os << "slots / buffers    : " << m.cfg.num_slots << " slots, " << m.cfg.buffer_count << " V4L2 buffers per node\n";
    os << "num_tiles          : " << m.tiles.num_tiles << "\n";
    for (std::uint32_t t = 0; t < m.tiles.num_tiles; ++t) {
        const pisp_tile& tile = m.tiles.tiles[t];
        os << "  tile " << t << ": input (" << tile.input_offset_x << "," << tile.input_offset_y << ") "
           << tile.input_width << "x" << tile.input_height << " -> output (" << tile.output_offset_x[m.cfg.branch]
           << "," << tile.output_offset_y[m.cfg.branch] << ") " << tile.output_width[m.cfg.branch] << "x"
           << tile.output_height[m.cfg.branch] << "\n";
    }
    if (m.dev) {
        os << "output dma-buf fds :";
        for (const auto& b : m.out_bufs) os << " " << b.get().Fd()[0];
        os << "\n";
        os << "input cache        : " << m.in_cache.size() << " entries, " << m.in_imports << " imports, " << m.in_hits
           << " hits\n";
    }
    return os.str();
}

// ---------------------------------------------------------------------------

struct IspResizer::OutputRead::Impl {
    Impl(libpisp::helpers::BufferRef b, std::size_t n)
        : sync(b, libpisp::helpers::Buffer::Sync::Access::Read), size(n) {}
    libpisp::helpers::Buffer::Sync sync;
    std::size_t size;
};

IspResizer::OutputRead::OutputRead(const IspResizer& r, int slot) {
    if (slot < 0 || slot >= static_cast<int>(r.impl_->out_bufs.size()))
        throw std::out_of_range("IspResizer::OutputRead: slot out of range");
    impl_ = std::make_unique<Impl>(r.impl_->out_bufs[static_cast<std::size_t>(slot)], r.impl_->out_bytes);
}

IspResizer::OutputRead::~OutputRead() = default;

const std::uint8_t* IspResizer::OutputRead::data() const { return impl_->sync.Get()[0]; }
std::size_t IspResizer::OutputRead::size() const { return impl_->size; }

} // namespace yvo::isp
