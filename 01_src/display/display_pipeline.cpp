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

} // namespace

const char* sync_name(DisplaySync s) { return s == DisplaySync::Latest ? "latest" : "matched"; }
const char* pace_name(DisplayPace p) {
    switch (p) {
    case DisplayPace::Source: return "source";
    case DisplayPace::Asap: return "asap";
    default: return "none";
    }
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
    WaylandWindowOptions wo;
    wo.width = cfg.width;
    wo.height = cfg.height;
    wo.fullscreen = cfg.fullscreen;
    wo.title = cfg.title;
    wo.app_id = cfg.app_id;
    wo.verbose = cfg.verbose;
    return win.open(wo);
}

DisplayPipeline::DisplayPipeline(yvo::vk::VulkanContext& ctx, WaylandWindow& win,
                                 const DisplayPipelineConfig& cfg)
    : ctx_(ctx), cfg_(cfg) {
    if (!win.is_open() || !ctx_.present_enabled()) {
        // strict: silent fallback is forbidden (the only way to keep measurements from quietly leaking onto another path).
        std::string why = !win.is_open() ? "could not open the Wayland window" : ctx_.present_unavailable_reason();
        throw std::runtime_error("--display on: " + (why.empty() ? std::string("present unavailable") : why));
    }

    win_ = &win;
    SwapchainOptions so;
    so.wayland_display = win_->native_display();
    so.wayland_surface = win_->native_surface();
    so.width = win_->width();
    so.height = win_->height();
    so.present_mode = cfg_.present_mode;
    so.force_blit = cfg_.force_blit;
    so.verbose = cfg_.verbose;
    swapchain_ = std::make_unique<VulkanSwapchain>(ctx_, so);
    swapchain_->create(win_->width(), win_->height());

    retained_.resize(static_cast<std::size_t>(std::max(cfg_.num_slots, 1)));
    pace_blocking_ok_ = cfg_.pace != DisplayPace::Asap;
    build_passes();
}

DisplayPipeline::~DisplayPipeline() {
    // Exceptions must not escape (destructor). Remaining GPU work must be finished before releasing resources.
    try {
        drain();
    } catch (...) {
    }
    conv_.reset();
    ov_.reset();
    swapchain_.reset();
}

void DisplayPipeline::build_passes() {
    const VkExtent2D ext = swapchain_->extent();
    uniform_scale_ = !(static_cast<int>(ext.width) == cfg_.src_w && static_cast<int>(ext.height) == cfg_.src_h &&
                       (cfg_.src_w % 4) == 0);

    DisplayConvertConfig cc;
    cc.num_slots = static_cast<int>(swapchain_->image_count());
    cc.uniform_scale = uniform_scale_;
    cc.max_dst_w = std::max<int>(static_cast<int>(ext.width), 1920);
    conv_ = std::make_unique<DisplayConvert>(ctx_, cc);

    OverlayConfig oc;
    oc.num_slots = static_cast<int>(swapchain_->image_count());
    oc.max_dets = cfg_.max_dets;
    oc.line_width = cfg_.overlay_line_width;
    oc.text_scale = cfg_.overlay_text_scale;
    ov_ = std::make_unique<OverlayRenderer>(ctx_, oc);
}

void DisplayPipeline::apply_extent() {
    const VkExtent2D ext = swapchain_->extent();
    const bool us = !(static_cast<int>(ext.width) == cfg_.src_w && static_cast<int>(ext.height) == cfg_.src_h &&
                      (cfg_.src_w % 4) == 0);
    // If the image count grew or the window grew beyond the LUT upper bound, rebuild the passes (in-flight is already empty).
    const bool rebuild = static_cast<int>(swapchain_->image_count()) > conv_->num_slots() ||
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

void DisplayPipeline::retain(int slot, const DrmFrame& f, Clock::time_point t_decoded) {
    Retained& r = retained_.at(static_cast<std::size_t>(slot));
    r.frame = f;   // extra reference to the keepalive shared_ptr — the frame survives even after main's f.reset()
    r.t = Clock::now();
    r.t_decoded = t_decoded;
    r.src = -1;
    r.valid = true;
}

void DisplayPipeline::set_source_index(int slot, std::int64_t src_index) {
    retained_.at(static_cast<std::size_t>(slot)).src = src_index;
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
    const bool ok = win_->poll();
    int nw = 0, nh = 0;
    if (win_->take_pending_size(&nw, &nh)) swapchain_->request_recreate(nw, nh);
    return ok && !win_->should_close() && !win_->quit_requested();
}

bool DisplayPipeline::quit_requested() const { return win_->quit_requested(); }

void DisplayPipeline::recreate_if_needed() {
    if (!swapchain_->needs_recreate()) return;
    // No submission referencing the old swapchain's images/semaphores may remain → finish everything (blocking).
    while (!inflight_.empty()) {
        const Pending p = inflight_.front();
        inflight_.pop_front();
        collect(p);
    }
    present_ids_.clear();
    have_last_present_ = false;
    swapchain_->create(win_->width(), win_->height());
    apply_extent();
}

bool DisplayPipeline::show(int slot, const std::vector<Detection>& dets, std::string_view hud) {
    Retained& r = retained_.at(static_cast<std::size_t>(slot));
    if (!r.valid) return false;
    // Whichever path exits, the retained reference is released here (if displayed, DisplayConvert holds its own).
    const DrmFrame frame = std::move(r.frame);
    const Clock::time_point t_retain = r.t, t_decoded = r.t_decoded;
    const std::int64_t src = r.src;
    r = Retained{};

    recreate_if_needed();

    const bool skip = cfg_.every > 1 && (shown_counter_ % cfg_.every) != 0;
    ++shown_counter_;
    if (skip) {
        ++skipped_;
        return false;
    }

    // ⚠ Collect finished display slots **before acquire()**.
    //   VulkanSwapchain::acquire() does `vkResetFences` on that slot's submit fence before handing the slot out.
    //   But `DisplayConvert::finish` waits indefinitely on that **same fence** → finishing after acquire waits
    //   forever on an already-reset fence (it actually deadlocked on a former headless backend that shared this loop).
    //   The swapchain hands out the **oldest** slot round-robin and inflight_ is also in submit order, so if
    //   finished entries are cleared from the front, the slot acquire hands out is guaranteed to be free.
    while (!inflight_.empty() && conv_->ready(inflight_.front().disp)) {
        const Pending p = inflight_.front();
        inflight_.pop_front();
        collect(p);
    }
    if (inflight_.size() >= swapchain_->image_count()) {
        // All display slots are still tied up on the GPU. **Drop without waiting** — display must not block the pipeline.
        ++busy_dropped_;
        return false;
    }

    SwapchainFrame fr;
    if (!swapchain_->acquire(&fr)) return false;   // ★ dropping the display does not block the pipeline
    const int d = static_cast<int>(fr.sync_index);
    if (conv_->pending(d))
        throw std::logic_error("DisplayPipeline::show: acquire handed out a display slot that is still being submitted "
                               "(the invariant above is broken — stop here rather than deadlock silently)");

    VkCommandBuffer cb = conv_->begin(d);
    swapchain_->record_acquire_transition(cb, fr);
    conv_->record(d, frame, fr.target_view, fr.extent);
    if (ov_) {
        ov_->set_frame(d, dets, hud);
        if (ov_->overflow(d)) ++overlay_overflow_;
        ov_->record(d, cb, fr.target_view, fr.extent,
                    make_overlay_affine(cfg_.src_w, cfg_.src_h, static_cast<int>(fr.extent.width),
                                        static_cast<int>(fr.extent.height)),
                    true);
    }
    swapchain_->record_present_transition(cb, fr);
    conv_->end_and_submit(d, fr.wait_semaphore, fr.signal_semaphore, fr.submit_fence);
    (void)swapchain_->present(fr);

    Pending p;
    p.disp = d;
    // In latest mode the caller raises display_done itself right after show() → the callback must not be called twice.
    p.pipe = cfg_.sync == DisplaySync::Latest ? -1 : slot;
    p.t_retain = t_retain;
    p.have_retain = true;
    inflight_.push_back(p);
    if (fr.present_id != 0) present_ids_.push_back(PresentWait{fr.present_id, t_retain, t_decoded, src, true});
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
    if (!swapchain_ || !swapchain_->present_wait_available()) {
        present_ids_.clear();
        return;
    }
    while (!present_ids_.empty()) {
        // Non-blocking is the default. Waiting via present_wait with pace=asap **ties the throughput measurement to vsync.**
        // With pace=source (or drain) it blocks only when the queue reaches the same length as the probe, to give the same numbers
        // (same as the image_count-1 rule at display_probe.cpp:1201).
        const bool deep = present_ids_.size() >= std::max<std::size_t>(swapchain_->image_count() - 1, 1);
        const bool block = blocking || (pace_blocking_ok_ && deep);
        const std::uint64_t timeout = block ? 1000ull * 1000 * 1000 : 0;
        const PresentWait w = present_ids_.front();
        if (!swapchain_->wait_present(w.id, timeout)) {
            // If non-blocking polling keeps failing while the queue only grows (drops/recreation), give up measuring and drop.
            if (!block && present_ids_.size() > swapchain_->image_count() + 4) {
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
        if (w.have_retain) {   // same screen time t for both series → equal length and order (main.cpp cuts both alike)
            glass_ms_.push_back(ms_between(w.t_retain, t));
            d2s_ms_.push_back(ms_between(w.t_decoded, t));
            present_log_.push_back(PresentRecord{w.src, t});
        }
    }
}

void DisplayPipeline::pump() {
    if (!conv_ || !swapchain_) return;
    while (!inflight_.empty() && conv_->ready(inflight_.front().disp)) {
        const Pending p = inflight_.front();
        inflight_.pop_front();
        collect(p);
    }
    drain_present_waits(false);
    swapchain_->collect_retired();
}

void DisplayPipeline::drain() {
    if (!conv_ || !swapchain_) return;
    while (!inflight_.empty()) {
        const Pending p = inflight_.front();
        inflight_.pop_front();
        collect(p);
    }
    drain_present_waits(true);
    // Release frames that were held without being displayed yet (the caller has already reclaimed the slots).
    for (Retained& r : retained_) r = Retained{};
    conv_->clear_import_cache();
    swapchain_->collect_retired();
}

int DisplayPipeline::frames_held() const {
    int n = 0;
    for (const Retained& r : retained_)
        if (r.valid) ++n;
    // Frames held by DisplayConvert equal the number of display slots being submitted.
    return n + static_cast<int>(inflight_.size());
}

} // namespace yvo::display
