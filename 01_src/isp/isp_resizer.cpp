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

// libpisp 공개 헤더. 이 타입들은 절대 isp_resizer.hpp로 새어 나가지 않는다(PIMPL).
#include "backend/backend.hpp"
#include "common/logging.hpp"
#include "common/utils.hpp"
#include "helpers/backend_device.hpp"
#include "helpers/media_device.hpp"
#include "variants/variant.hpp"

namespace yvo::isp {

namespace {

constexpr int kSandCol = 128; // SAND128: 열 하나의 폭(픽셀 = 바이트, 8bpp)

// libpisp의 boost::log 기본 싱크는 debug까지 전부 stderr로 쏟아낸다.
// logging_init()이 LIBPISP_LOG_LEVEL(기본 fatal)로 필터를 건다. 프로세스당 한 번만.
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
    // 입력 dma-buf 캐시. libpisp::helpers::Buffer는 fd 소유권을 가져가고 소멸 시 close하므로
    // 디코더 fd를 그대로 넘길 수 없다. dup한 fd로 Buffer를 만들고 디코더 fd 번호로 캐시한다.
    // (디코더 풀의 fd는 고정이지만 디코더를 다시 열면 번호가 재사용될 수 있어 fstat으로 확인한다.)
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
    int content_w = 0, content_h = 0; // 축소 결과(패딩 제외) 크기
    int out_w = 0, out_h = 0;         // 출력 버퍼 크기
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
        in_cache.erase(it); // fd 번호 재사용: 옛 항목 버림
    }

    const int y_dup = ::dup(f.fd);
    if (y_dup < 0) throw std::runtime_error(std::string("IspResizer: dup(y_fd) failed: ") + std::strerror(errno));
    const int uv_dup = ::dup(f.uv_fd);
    if (uv_dup < 0) {
        ::close(y_dup);
        throw std::runtime_error(std::string("IspResizer: dup(uv_fd) failed: ") + std::strerror(errno));
    }
    InputEntry e;
    // Buffer는 여기서 fd 소유권을 가져간다(소멸자가 close).
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
    // libpisp은 offset_x/offset_y의 범위를 검사하지 않는다. 넘치면 조용히 버퍼 밖을 쓴다.
    if (m.lb.pad_x + m.content_w > m.out_w || m.lb.pad_y + m.content_h > m.out_h)
        throw std::invalid_argument("IspResizer: letterbox window " + std::to_string(m.content_w) + "x" +
                                    std::to_string(m.content_h) + " at (" + std::to_string(m.lb.pad_x) + "," +
                                    std::to_string(m.lb.pad_y) + ") does not fit in " + std::to_string(m.out_w) + "x" +
                                    std::to_string(m.out_h));
    m.out_bytes = static_cast<std::size_t>(m.out_w) * 3u * static_cast<std::size_t>(m.out_h);

    // --- variant 선택: media 노드의 hw revision과 libpisp variant의 BackEndVersion을 맞춘다 ---
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

    // --- 입력 포맷: SAND128 NV12. wallpaper roll 포맷에서 stride는 열 하나의 바이트 수다. ---
    pisp_image_format_config in{};
    in.width = static_cast<std::uint16_t>(m.cfg.src_w);
    in.height = static_cast<std::uint16_t>(m.cfg.coded_height);
    in.format = libpisp::get_pisp_image_format("YUV420SP_COL128");
    in.stride = kSandCol * m.cfg.sand_col_height;
    in.stride2 = kSandCol * m.cfg.uv_sand_col_height;
    m.be->SetInputFormat(in);

    // --- 전역 enable: Bayer 경로 전체 off, RGB는 input/ccm/output만. ---
    // DOWNSCALE/RESAMPLE 비트는 SetSmartResize가 Prepare에서 채운다. CSC는 RGB→YCbCr이라 쓰지 않는다.
    pisp_be_global_config global{};
    global.bayer_enables = 0;
    global.rgb_enables = PISP_BE_RGB_ENABLE_INPUT | PISP_BE_RGB_ENABLE_CCM |
                         static_cast<std::uint32_t>(PISP_BE_RGB_ENABLE_OUTPUT(m.cfg.branch));
    m.be->SetGlobal(global);

    // --- YUV→RGB: RGB 블록 순서상 ycbcr_inverse는 sharpen 뒤라 입력단에 못 쓴다. 맨 앞 ccm에 역행렬을 싣는다. ---
    pisp_be_ccm_config ccm{};
    m.be->InitialiseYcbcrInverse(ccm, m.cfg.colour_space);
    m.be->SetCcm(ccm);

    // --- crop: coded_height(1088) 중 유효한 src_h(1080)행만 쓴다 ---
    pisp_be_crop_config crop{};
    crop.offset_x = 0;
    crop.offset_y = 0;
    crop.width = static_cast<std::uint16_t>(m.cfg.src_w);
    crop.height = static_cast<std::uint16_t>(m.cfg.src_h);
    m.be->SetCrop(m.cfg.branch, crop);

    // --- 축소: downscale + resample 조합은 SmartResize에 맡긴다 ---
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

    // --- 출력: packed RGB888. letterbox는 offset_x/offset_y로 만든다. ---
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

    // Prepare()는 output_format[i].image를 리샘플 결과 크기(content_w x content_h)로 덮어쓴다.
    // 하드웨어는 타일 주소로만 쓰므로 height는 V4L2 포맷의 sizeimage 계산에만 쓰인다.
    // letterbox 버퍼(out_h행) 전체를 노드 포맷에도 알려야 sizeimage가 실제 버퍼 크기와 맞는다.
    if (m.tiles.config.output_format[m.cfg.branch].image.stride != m.out_w * 3)
        throw std::runtime_error("IspResizer: Prepare() changed the output stride to " +
                                 std::to_string(m.tiles.config.output_format[m.cfg.branch].image.stride));
    m.tiles.config.output_format[m.cfg.branch].image.height = static_cast<std::uint16_t>(m.out_h);
    m.tiles.config.output_format[m.cfg.branch].image.width = static_cast<std::uint16_t>(m.out_w);

    // 타일 쓰기 주소가 실제로 버퍼 안에 있는지 호스트에서 직접 확인한다.
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
        // 할당 직후 한 번만 패딩 값으로 채운다. 이후 ISP는 content 영역만 건드린다.
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
    // pisp_image_format_config에는 plane 시작 오프셋 필드가 없다. 0이 아니면 표현할 방법이 없다.
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
    out.layout = TensorLayout::Nhwc; // ISP는 packed RGB만 낸다
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
    // crop은 pisp_be_config의 레지스터 필드가 아니라 BeConfigExtra에 있어 되읽을 수 없다. 설정값을 적는다.
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
