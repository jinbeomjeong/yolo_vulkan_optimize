#include "display/display_pipeline.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

#include "vulkan/vk_check.hpp"
#include "vulkan/vulkan_context.hpp"

namespace yvo::display {
namespace {

using Clock = DisplayPipeline::Clock;

double ms_between(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

// VulkanSwapchain 을 DisplayBackend 인터페이스에 맞춘다. 로직은 전부 VulkanSwapchain 에 있고
// 여기서는 이름만 이어 준다(빌드 시점에 wayland 가 없으면 이 파일 자체가 빌드되지 않는다).
class SwapchainBackend final : public DisplayBackend {
public:
    SwapchainBackend(yvo::vk::VulkanContext& ctx, const SwapchainOptions& opt) : sc_(ctx, opt) {}

    void create(int w, int h) { sc_.create(w, h); }

    bool acquire(SwapchainFrame* out) override { return sc_.acquire(out); }
    void record_acquire_transition(VkCommandBuffer cb, const SwapchainFrame& f) const override {
        sc_.record_acquire_transition(cb, f);
    }
    void record_present_transition(VkCommandBuffer cb, const SwapchainFrame& f) const override {
        sc_.record_present_transition(cb, f);
    }
    void present(const SwapchainFrame& f) override { (void)sc_.present(f); }

    bool needs_recreate() const override { return sc_.needs_recreate(); }
    void request_recreate(int w, int h) override { sc_.request_recreate(w, h); }
    void recreate(int w, int h) override { sc_.create(w, h); }
    void collect_retired() override { sc_.collect_retired(); }

    VkExtent2D extent() const override { return sc_.extent(); }
    std::uint32_t image_count() const override { return sc_.image_count(); }
    bool blit_mode() const override { return sc_.blit_mode(); }

    bool present_wait_available() const override { return sc_.present_wait_available(); }
    bool wait_present(std::uint64_t id, std::uint64_t timeout_ns) override {
        return sc_.wait_present(id, timeout_ns);
    }

    std::uint64_t presented() const override { return sc_.presented(); }
    std::uint64_t display_dropped() const override { return sc_.display_dropped(); }
    std::uint64_t recreates() const override { return sc_.swapchain_recreates(); }
    const char* name() const override { return "wayland"; }

private:
    VulkanSwapchain sc_;
};

} // namespace

const char* sync_name(DisplaySync s) { return s == DisplaySync::Latest ? "latest" : "matched"; }
const char* pace_name(DisplayPace p) {
    switch (p) {
    case DisplayPace::Source: return "source";
    case DisplayPace::Asap: return "asap";
    default: return "none";
    }
}
bool parse_backend_kind(const std::string& s, DisplayBackendKind* out) {
    if (s == "auto") *out = DisplayBackendKind::Auto;
    else if (s == "wayland") *out = DisplayBackendKind::Wayland;
    else if (s == "offscreen") *out = DisplayBackendKind::Offscreen;
    else return false;
    return true;
}
bool parse_display_sync(const std::string& s, DisplaySync* out) {
    if (s == "matched") *out = DisplaySync::Matched;
    else if (s == "latest") *out = DisplaySync::Latest;
    else return false;
    return true;
}
bool parse_display_pace(const std::string& s, DisplayPace* out) {
    if (s == "source") *out = DisplayPace::Source;
    else if (s == "asap") *out = DisplayPace::Asap;
    else if (s == "none" || s == "off") *out = DisplayPace::None;
    else return false;
    return true;
}

bool DisplayPipeline::open_window(WaylandWindow& win, const DisplayPipelineConfig& cfg) {
    if (cfg.backend == DisplayBackendKind::Offscreen) return false;
    WaylandWindowOptions wo;
    wo.width = cfg.width;
    wo.height = cfg.height;
    wo.fullscreen = cfg.fullscreen;
    wo.title = cfg.title;
    wo.app_id = cfg.app_id;
    wo.verbose = cfg.verbose;
    return win.open(wo);
}

DisplayPipeline::DisplayPipeline(yvo::vk::VulkanContext& ctx, WaylandWindow* win,
                                 const DisplayPipelineConfig& cfg)
    : ctx_(ctx), cfg_(cfg) {
    const bool want_wayland = cfg_.backend != DisplayBackendKind::Offscreen;
    const bool can_wayland = want_wayland && win != nullptr && win->is_open() && ctx_.present_enabled();
    if (cfg_.backend == DisplayBackendKind::Wayland && !can_wayland) {
        // strict: 조용한 폴백을 금지한다(측정이 몰래 다른 경로로 새는 것을 막는 유일한 방법이다).
        std::string why = win == nullptr || !win->is_open() ? "Wayland 창을 열지 못했다"
                                                            : ctx_.present_unavailable_reason();
        throw std::runtime_error("--display-backend wayland: " + (why.empty() ? std::string("present 불가") : why));
    }

    if (can_wayland) {
        win_ = win;
        SwapchainOptions so;
        so.wayland_display = win_->native_display();
        so.wayland_surface = win_->native_surface();
        so.width = win_->width();
        so.height = win_->height();
        so.present_mode = cfg_.present_mode;
        so.force_blit = cfg_.force_blit;
        so.verbose = cfg_.verbose;
        auto sb = std::make_unique<SwapchainBackend>(ctx_, so);
        sb->create(win_->width(), win_->height());
        backend_ = std::move(sb);
    } else {
        offscreen_ = true;
        // 오프스크린은 창이 없으므로 요청 크기를 그대로 쓴다. fullscreen 은 1920x1080 을 뜻한다.
        const int w = cfg_.fullscreen ? cfg_.src_w : cfg_.width;
        const int h = cfg_.fullscreen ? cfg_.src_h : cfg_.height;
        backend_ = std::make_unique<OffscreenBackend>(ctx_, w, h, 3);
    }

    retained_.resize(static_cast<std::size_t>(std::max(cfg_.num_slots, 1)));
    pace_blocking_ok_ = cfg_.pace != DisplayPace::Asap;
    build_passes();
}

DisplayPipeline::~DisplayPipeline() {
    // 예외가 밖으로 나가면 안 된다(소멸자). 남은 GPU 작업은 반드시 끝내고 자원을 놓는다.
    try {
        drain();
    } catch (...) {
    }
    conv_.reset();
    ov_.reset();
    backend_.reset();
}

void DisplayPipeline::build_passes() {
    const VkExtent2D ext = backend_->extent();
    uniform_scale_ = !(static_cast<int>(ext.width) == cfg_.src_w && static_cast<int>(ext.height) == cfg_.src_h &&
                       (cfg_.src_w % 4) == 0);

    DisplayConvertConfig cc;
    cc.num_slots = static_cast<int>(backend_->image_count());
    cc.uniform_scale = uniform_scale_;
    cc.max_dst_w = std::max<int>(static_cast<int>(ext.width), 1920);
    conv_ = std::make_unique<DisplayConvert>(ctx_, cc);

    OverlayConfig oc;
    oc.num_slots = static_cast<int>(backend_->image_count());
    oc.max_dets = cfg_.max_dets;
    oc.line_width = cfg_.overlay_line_width;
    oc.text_scale = cfg_.overlay_text_scale;
    ov_ = std::make_unique<OverlayRenderer>(ctx_, oc);
}

void DisplayPipeline::apply_extent() {
    const VkExtent2D ext = backend_->extent();
    const bool us = !(static_cast<int>(ext.width) == cfg_.src_w && static_cast<int>(ext.height) == cfg_.src_h &&
                      (cfg_.src_w % 4) == 0);
    // 이미지 수가 늘었거나 창이 LUT 상한을 넘어 커졌으면 패스를 다시 만든다(인플라이트는 이미 비었다).
    const bool rebuild = static_cast<int>(backend_->image_count()) > conv_->num_slots() ||
                         static_cast<int>(ext.width) > conv_->config().max_dst_w;
    if (rebuild) {
        conv_.reset();
        ov_.reset();
        build_passes();
        return;
    }
    if (us != uniform_scale_) {
        conv_->set_uniform_scale(us);
        uniform_scale_ = us;
    }
}

void DisplayPipeline::set_source_size(int w, int h) {
    if (w <= 0 || h <= 0) return;
    cfg_.src_w = w;
    cfg_.src_h = h;
    apply_extent();
}

void DisplayPipeline::retain(int slot, const DrmFrame& f) {
    Retained& r = retained_.at(static_cast<std::size_t>(slot));
    r.frame = f;   // keepalive shared_ptr 추가 참조 — main 의 f.reset() 이후에도 프레임이 산다
    r.t = Clock::now();
    r.valid = true;
}

std::optional<Clock::time_point> DisplayPipeline::pace_deadline(std::int64_t frame_index) {
    if (cfg_.pace != DisplayPace::Source || cfg_.source_fps <= 0.0) return std::nullopt;
    if (!pace_started_) {
        pace_t0_ = Clock::now();
        pace_started_ = true;
    }
    const double ms = 1000.0 * static_cast<double>(frame_index) / cfg_.source_fps;
    return pace_t0_ + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double, std::milli>(ms));
}

bool DisplayPipeline::poll_window() {
    if (!win_) return true;
    const bool ok = win_->poll();
    int nw = 0, nh = 0;
    if (win_->take_pending_size(&nw, &nh)) backend_->request_recreate(nw, nh);
    return ok && !win_->should_close();
}

void DisplayPipeline::recreate_if_needed() {
    if (!backend_->needs_recreate()) return;
    // 구 스왑체인의 이미지·세마포어를 참조하는 제출이 남아 있으면 안 된다 → 전부 끝낸다(블로킹).
    while (!inflight_.empty()) {
        const Pending p = inflight_.front();
        inflight_.pop_front();
        collect(p);
    }
    present_ids_.clear();
    have_last_present_ = false;
    const int w = win_ ? win_->width() : static_cast<int>(backend_->extent().width);
    const int h = win_ ? win_->height() : static_cast<int>(backend_->extent().height);
    backend_->recreate(w, h);
    apply_extent();
}

bool DisplayPipeline::show(int slot, const std::vector<Detection>& dets, std::string_view hud) {
    Retained& r = retained_.at(static_cast<std::size_t>(slot));
    if (!r.valid) return false;
    // 어떤 경로로 나가든 retained 참조는 여기서 놓는다(표시하면 DisplayConvert 가 따로 잡는다).
    const DrmFrame frame = std::move(r.frame);
    const Clock::time_point t_retain = r.t;
    r = Retained{};

    recreate_if_needed();

    const bool skip = cfg_.every > 1 && (shown_counter_ % cfg_.every) != 0;
    ++shown_counter_;
    if (skip) {
        ++skipped_;
        return false;
    }

    // ⚠ **acquire() 보다 먼저** 끝난 표시 슬롯을 회수한다.
    //   백엔드의 acquire() 는 슬롯을 내주기 전에 그 슬롯의 submit 펜스를 `vkResetFences` 한다
    //   (VulkanSwapchain::acquire / OffscreenBackend::acquire). 그런데 `DisplayConvert::finish` 는
    //   그 **같은 펜스**를 무한 대기한다 → acquire 뒤에 finish 하면 이미 리셋된 펜스를 영원히 기다린다.
    //   (실제로 오프스크린 백엔드에서 교착했다. 스왑체인 경로도 원리상 같은 위험이 있었다.)
    //   백엔드는 라운드로빈으로 **가장 오래된** 슬롯을 내주고 inflight_ 도 제출 순서라, 앞에서부터
    //   끝난 것을 걷어내면 acquire 가 내줄 슬롯은 반드시 비어 있다.
    while (!inflight_.empty() && conv_->ready(inflight_.front().disp)) {
        const Pending p = inflight_.front();
        inflight_.pop_front();
        collect(p);
    }
    if (inflight_.size() >= backend_->image_count()) {
        // 모든 표시 슬롯이 아직 GPU 에 걸려 있다. **기다리지 않고 버린다** — 표시가 파이프라인을 막으면 안 된다.
        ++busy_dropped_;
        return false;
    }

    SwapchainFrame fr;
    if (!backend_->acquire(&fr)) return false;   // ★ 표시를 버려도 파이프라인은 막히지 않는다
    const int d = static_cast<int>(fr.sync_index);
    if (conv_->pending(d))
        throw std::logic_error("DisplayPipeline::show: acquire 가 아직 제출 중인 표시 슬롯을 내줬다 "
                               "(위 불변식이 깨졌다 — 조용히 교착하느니 여기서 멈춘다)");

    VkCommandBuffer cb = conv_->begin(d);
    backend_->record_acquire_transition(cb, fr);
    conv_->record(d, frame, fr.target_view, fr.extent);
    if (ov_) {
        ov_->set_frame(d, dets, hud);
        if (ov_->overflow(d)) ++overlay_overflow_;
        ov_->record(d, cb, fr.target_view, fr.extent,
                    make_overlay_affine(cfg_.src_w, cfg_.src_h, static_cast<int>(fr.extent.width),
                                        static_cast<int>(fr.extent.height)),
                    true);
    }
    backend_->record_present_transition(cb, fr);
    conv_->end_and_submit(d, fr.wait_semaphore, fr.signal_semaphore, fr.submit_fence);
    backend_->present(fr);

    Pending p;
    p.disp = d;
    // latest 모드에서는 호출자가 show() 직후 스스로 display_done 을 세운다 → 콜백을 두 번 부르면 안 된다.
    p.pipe = cfg_.sync == DisplaySync::Latest ? -1 : slot;
    p.t_retain = t_retain;
    p.have_retain = true;
    inflight_.push_back(p);
    if (fr.present_id != 0) present_ids_.push_back(PresentWait{fr.present_id, t_retain, true});
    return true;
}

void DisplayPipeline::collect(const Pending& p) {
    conv_->finish(p.disp);
    if (conv_->last_shader_ms() >= 0) convert_ms_.push_back(conv_->last_shader_ms());
    if (conv_->last_total_ms() >= 0) convert_total_ms_.push_back(conv_->last_total_ms());
    if (ov_) {
        ov_->collect(p.disp);
        if (ov_->last_shader_ms() >= 0) overlay_ms_.push_back(ov_->last_shader_ms());
    }
    if (p.pipe >= 0 && done_cb_) done_cb_(p.pipe);
}

void DisplayPipeline::drain_present_waits(bool blocking) {
    if (!backend_ || !backend_->present_wait_available()) {
        present_ids_.clear();
        return;
    }
    while (!present_ids_.empty()) {
        // 논블로킹이 기본이다. pace=asap 에서 present_wait 로 기다리면 **처리량 측정이 vsync 에 묶인다.**
        // pace=source(또는 drain)에서는 큐가 프로브와 같은 길이가 됐을 때만 블로킹해 같은 수치를 낸다
        // (display_probe.cpp:1201 의 image_count-1 규칙과 동일).
        const bool deep = present_ids_.size() >= std::max<std::size_t>(backend_->image_count() - 1, 1);
        const bool block = blocking || (pace_blocking_ok_ && deep);
        const std::uint64_t timeout = block ? 1000ull * 1000 * 1000 : 0;
        const PresentWait w = present_ids_.front();
        if (!backend_->wait_present(w.id, timeout)) {
            // 논블로킹 폴링이 계속 실패하는데 큐만 자라면(드롭·재생성) 측정을 포기하고 버린다.
            if (!block && present_ids_.size() > backend_->image_count() + 4) {
                present_ids_.pop_front();
                continue;
            }
            break;
        }
        present_ids_.pop_front();
        const Clock::time_point t = Clock::now();
        if (have_last_present_) interval_ms_.push_back(ms_between(last_present_, t));
        last_present_ = t;
        have_last_present_ = true;
        if (w.have_retain) glass_ms_.push_back(ms_between(w.t_retain, t));
    }
}

void DisplayPipeline::pump() {
    if (!conv_ || !backend_) return;
    while (!inflight_.empty() && conv_->ready(inflight_.front().disp)) {
        const Pending p = inflight_.front();
        inflight_.pop_front();
        collect(p);
    }
    drain_present_waits(false);
    backend_->collect_retired();
}

void DisplayPipeline::drain() {
    if (!conv_ || !backend_) return;
    while (!inflight_.empty()) {
        const Pending p = inflight_.front();
        inflight_.pop_front();
        collect(p);
    }
    drain_present_waits(true);
    // 아직 표시되지 않은 채 붙들고 있던 프레임을 놓는다(호출자는 이미 슬롯을 회수했다).
    for (Retained& r : retained_) r = Retained{};
    conv_->clear_import_cache();
    backend_->collect_retired();
}

int DisplayPipeline::frames_held() const {
    int n = 0;
    for (const Retained& r : retained_)
        if (r.valid) ++n;
    // DisplayConvert 가 잡고 있는 프레임은 제출 중인 표시 슬롯 수와 같다.
    return n + static_cast<int>(inflight_.size());
}

} // namespace yvo::display
