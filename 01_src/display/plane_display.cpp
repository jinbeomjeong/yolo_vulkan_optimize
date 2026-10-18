#include "display/plane_display.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <utility>

#include "vulkan/vulkan_context.hpp"

namespace yvo::display {

namespace {

using Clock = PlaneDisplay::Clock;

// steady_clock is CLOCK_MONOTONIC on Linux (libstdc++). The presentation time given by wp_presentation in CLOCK_MONOTONIC
// and the retain() time are subtracted on the same axis.
std::int64_t steady_ns(Clock::time_point t) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(t.time_since_epoch()).count();
}

} // namespace

PlaneDisplay::PlaneDisplay(yvo::vk::VulkanContext& ctx, std::unique_ptr<PlaneScanout> scanout,
                           const DisplayConfig& cfg)
    : ctx_(ctx), cfg_(cfg), scanout_(std::move(scanout)) {
    if (!scanout_) throw std::invalid_argument("PlaneDisplay: scanout is null");
    scanout_->set_listener(this);
    const VkExtent2D ext = scanout_->output_extent();
    if (ext.width == 0 || ext.height == 0) throw std::runtime_error("PlaneDisplay: output size is 0");

    OverlayPlaneConfig oc;
    oc.ring = kOverlayRing;
    oc.width = static_cast<int>(ext.width);
    oc.height = static_cast<int>(ext.height);
    oc.src_w = cfg_.src_w;
    oc.src_h = cfg_.src_h;
    oc.overlay.max_dets = cfg_.max_dets;
    oc.overlay.line_width = cfg_.overlay_line_width;
    oc.overlay.text_scale = cfg_.overlay_text_scale;
    ov_ = std::make_unique<OverlayPlane>(ctx_, oc);
    scanout_->register_overlay(*ov_);

    retained_.resize(static_cast<std::size_t>(std::max(cfg_.num_slots, 1)));
    set_source_fps(cfg_.source_fps);
}

PlaneDisplay::~PlaneDisplay() {
    // Do not throw out of the destructor.
    try {
        drain();
    } catch (...) {
    }
    // Tear down the transport first (wl_buffer points at the OverlayPlane dma-buf), then release the box ring.
    // Detach listeners first so the transport being destroyed does not send events to a half-dismantled this.
    if (scanout_) scanout_->set_listener(nullptr);
    scanout_.reset();
    ov_.reset();
}

void PlaneDisplay::set_source_size(int w, int h) {
    if (w <= 0 || h <= 0) return;
    cfg_.src_w = w;
    cfg_.src_h = h;
    ov_->set_source_size(w, h);
}

void PlaneDisplay::set_source_fps(double fps) {
    // pace=source commit gate = source interval - 8 ms (half a vblank). <= 0 / NaN keeps the current value.
    if (!(fps > 0.0)) return;
    cfg_.source_fps = fps;
    if (cfg_.pace == DisplayPace::Source)
        min_commit_interval_ = std::chrono::duration_cast<Clock::duration>(
            std::chrono::duration<double, std::milli>(std::max(1000.0 / fps - 8.0, 0.0)));
}

void PlaneDisplay::retain(int slot, const DrmFrame& f, Clock::time_point t_decoded) {
    Retained& r = retained_.at(static_cast<std::size_t>(slot));
    r.frame = f;   // extra keepalive reference
    r.t = Clock::now();
    r.t_decoded = t_decoded;
    r.src = -1;
    r.valid = true;
}

void PlaneDisplay::set_source_index(int slot, std::int64_t src_index) {
    retained_.at(static_cast<std::size_t>(slot)).src = src_index;
}

std::optional<Clock::time_point> PlaneDisplay::pace_deadline(std::int64_t frame_index) {
    // Source fps timetable (frame k is due at t0 + k / fps).
    if (cfg_.pace != DisplayPace::Source || cfg_.source_fps <= 0.0) return std::nullopt;
    if (!pace_started_) {
        pace_t0_ = Clock::now();
        pace_started_ = true;
    }
    const double ms = 1000.0 * static_cast<double>(frame_index) / cfg_.source_fps;
    return pace_t0_ + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double, std::milli>(ms));
}

bool PlaneDisplay::show(int slot, const std::vector<Detection>& dets, std::string_view hud) {
    Retained& r = retained_.at(static_cast<std::size_t>(slot));
    if (!r.valid) return false;
    Item it;
    it.frame = std::move(r.frame);
    it.t_retain = r.t;
    it.t_decoded = r.t_decoded;
    it.src = r.src;
    r = Retained{};

    // Source frames between this show() and the previous one never reached the display (leaky input queue drops).
    // Counted before the --display-every skip: a skipped frame did reach show(), so it is not missing.
    if (it.src >= 0) {
        if (last_show_src_ >= 0 && it.src > last_show_src_ + 1)
            upstream_missing_ += static_cast<std::uint64_t>(it.src - last_show_src_ - 1);
        last_show_src_ = it.src;
    }
    it.missing = upstream_missing_;

    const bool skip = cfg_.every > 1 && (shown_counter_ % cfg_.every) != 0;
    ++shown_counter_;
    if (skip) {
        ++skipped_;
        return false;   // it releases the keepalive as it leaves scope
    }
    if (drained_) return false;
    // matched-detach: the Item owns the frame from here, so the caller returns the slot right after show() in both sync
    // modes (retained_[slot] was emptied above — a new retain() into this slot cannot touch this frame).

    // Boxes: non-blocking submit. If the ring is empty, this frame keeps the previous boxes.
    const int idx = ov_->acquire();
    if (idx < 0) {
        ++overlay_skipped_;
    } else {
        ov_->draw(idx, dets, hud);
        it.overlay = idx;
    }

    // If pending is full, drop the oldest one. The new frame goes to the back.
    while (static_cast<int>(pending_.size()) >= kMaxPending) {
        drop_item(pending_.front());
        pending_.pop_front();
        ++replaced_;
    }
    pending_.push_back(std::move(it));
    try_commit(false);
    return true;
}

bool PlaneDisplay::overlay_ready(Item& it) {
    if (it.overlay < 0) return true;
    if (!ov_->ready(it.overlay)) return false;
    if (!it.overlay_stat) {
        // The call where ready() first returns true is the call that reaped the timestamp (acquire does not touch Claimed entries).
        it.overlay_stat = true;
        const double o = ov_->last_overlay_ms();
        const double c = ov_->last_clear_ms();
        if (o >= 0.0) overlay_ms_.push_back(o + std::max(c, 0.0));
    }
    return true;
}

void PlaneDisplay::drop_item(Item& it) {
    if (it.overlay >= 0) {
        ov_->release(it.overlay);   // Claimed → Free (acquire skips it if the GPU is still using it)
        it.overlay = -1;
    }
    it.frame = DrmFrame{};
}

void PlaneDisplay::finish_item(Item& it) {
    it.frame = DrmFrame{};          // release keepalive → returns to the decoder pool
}

void PlaneDisplay::try_commit(bool force) {
    if (pending_.empty()) return;
    Item& p = pending_.front();
    const Clock::time_point now = Clock::now();
    if (force) {
        // Only at shutdown: wait for box GPU completion (bounded, OverlayPlane::wait 5 s) and ignore the cap and vblank gate.
        if (p.overlay >= 0) {
            ov_->wait(p.overlay);
            overlay_ready(p);
        }
    } else {
        if (!overlay_ready(p)) return;                                     // box GPU in progress
        if (static_cast<int>(committed_.size()) >= kMaxCommitted) return;  // compositor has not released yet
        // pace=source: do not commit faster than the source frame interval (half-vblank margin). With matched + batch 2, detections
        // come out two at a time, so without this gate two frames go out on consecutive vblanks (16.7 ms) and the next one 66.7 ms later,
        // giving a jagged cadence (measured present interval median 16.7 / p90 66.7).
        if (min_commit_interval_.count() > 0 && commits_ > 0 && now - last_commit_ < min_commit_interval_) return;
        if (frame_cb_pending_) {
            if (now - last_commit_ < kFrameCallbackTimeout) return;        // wait for the next vblank
            // The callback does not arrive (window occluded or compositor stalled). Commit without waiting — commit is non-blocking.
            ++frame_cb_timeouts_;
            frame_cb_pending_ = false;
        }
    }
    p.token = next_token_++;
    if (!scanout_->commit(p.frame, p.token, p.overlay)) {
        ++rejected_;
        stall_tokens_.push_back(p.token);   // the previous frame stays on screen longer: a real stall (seq gap)
        drop_item(p);
        pending_.pop_front();
        return;
    }
    if (p.overlay >= 0) {
        ov_->mark_held(p.overlay);
        overlay_ever_committed_ = true;
    }
    ++commits_;
    if (!scanout_->reports_presentation()) ++presented_;
    frame_cb_pending_ = true;
    last_commit_ = now;
    await_present_[p.token] = AwaitPresent{p.t_retain, p.t_decoded, p.src, p.missing};
    const std::uint64_t tok = p.token;
    committed_.emplace(tok, std::move(p));
    pending_.pop_front();
}

// ── PlaneScanoutListener ──────────────────────────────────────────────────────────────────────
void PlaneDisplay::on_video_released(std::uint64_t token) {
    auto it = committed_.find(token);
    if (it == committed_.end()) return;
    finish_item(it->second);
    committed_.erase(it);
}

void PlaneDisplay::on_overlay_released(int overlay_idx) {
    if (ov_ && ov_->held(overlay_idx)) ov_->release(overlay_idx);
}

void PlaneDisplay::on_frame_done() { frame_cb_pending_ = false; }

void PlaneDisplay::on_presented(std::uint64_t token, std::int64_t t_ns, bool clock_monotonic, bool zero_copy,
                                std::uint64_t seq, std::uint32_t refresh_ns) {
    // Vblank seq gap ending at this presentation (index = feedback_presented_ before the increment). Gaps that touch the
    // first cfg_.warmup presentations are skipped — the same first-N cut main.cpp applies to the interval series.
    // A gap is taken only if this token directly follows the previous presented one in commit order. Tokens that
    // were discarded (wp_presentation) or rejected by the transport are real screen stalls, so they stay **inside** the
    // gap (stall_tokens_). Any other hole — a presentation without feedback, a presentation without a vsync
    // seq, a non-increasing seq — breaks the chain: counted in seq_breaks_, no gap (it would inflate seq_gap_max).
    if (refresh_ns > 0) refresh_ns_ = refresh_ns;
    std::uint64_t expect = last_chain_token_ + 1;
    while (std::find(stall_tokens_.begin(), stall_tokens_.end(), expect) != stall_tokens_.end()) ++expect;
    stall_tokens_.erase(std::remove_if(stall_tokens_.begin(), stall_tokens_.end(),
                                       [&](std::uint64_t t) { return t <= token; }),
                        stall_tokens_.end());
    auto it = await_present_.find(token);
    // Upstream drops inside this gap: difference of the cumulative counters (0 when the token is unknown).
    const std::uint64_t missing = it != await_present_.end() ? it->second.missing : last_chain_missing_;
    if (seq != 0) seq_seen_ = true;
    if (last_seq_ != 0 && feedback_presented_ > static_cast<std::uint64_t>(std::max(cfg_.warmup, 0))) {
        if (seq > last_seq_ && token == expect) {
            seq_gaps_.push_back(static_cast<std::uint32_t>(std::min<std::uint64_t>(seq - last_seq_, UINT32_MAX)));
            seq_gap_missing_.push_back(missing >= last_chain_missing_ ? missing - last_chain_missing_ : 0);
        } else {
            ++seq_breaks_;
        }
    }
    last_seq_ = seq;   // seq 0 also ends the chain: the next presentation starts a new one
    last_chain_token_ = token;
    last_chain_missing_ = missing;
    ++presented_;
    ++feedback_presented_;
    if (zero_copy) ++zero_copy_fb_;
    if (clock_monotonic) {
        if (last_present_ns_ != 0) interval_ms_.push_back(static_cast<double>(t_ns - last_present_ns_) / 1e6);
        last_present_ns_ = t_ns;
        if (it != await_present_.end()) {   // same t_ns for all three series → equal length and order
            glass_ms_.push_back(static_cast<double>(t_ns - steady_ns(it->second.t_retain)) / 1e6);
            d2s_ms_.push_back(static_cast<double>(t_ns - steady_ns(it->second.t_decoded)) / 1e6);
            present_log_.push_back(
                PresentRecord{it->second.src, Clock::time_point(std::chrono::duration_cast<Clock::duration>(
                                                  std::chrono::nanoseconds(t_ns)))});
        }
    }
    if (it != await_present_.end()) await_present_.erase(it);
}

void PlaneDisplay::on_discarded(std::uint64_t token) {
    ++discarded_;
    stall_tokens_.push_back(token);   // never reached the screen: a real stall, kept inside the seq gap
    await_present_.erase(token);
}

// ── Loop ──────────────────────────────────────────────────────────────────────────────────────
void PlaneDisplay::pump() {
    if (drained_) return;
    (void)scanout_->dispatch(0);   // connection errors are reported by poll_window()
    try_commit(false);
    if (!snap_start_taken_ && presented() >= kSnapshotAfterPresented) {
        snap_start_taken_ = true;
        snapshot("start", &snap_start_);
    }
}

bool PlaneDisplay::poll_window() {
    if (drained_) return true;
    const bool ok = scanout_->dispatch(0);
    try_commit(false);
    return ok && !scanout_->should_close() && !scanout_->quit_requested();
}

bool PlaneDisplay::quit_requested() const { return scanout_->quit_requested(); }

void PlaneDisplay::drain() {
    if (drained_ || !scanout_) return;
    (void)scanout_->dispatch(0);
    // Emit the remaining pending (matched's final frames were just shown in flush_results) with the usual gate for up to 200 ms,
    // and if some still remain, emit them ignoring the gate (earlier ones may never reach the screen and be discarded).
    const auto flush_deadline = Clock::now() + std::chrono::milliseconds(200);
    while (!pending_.empty() && Clock::now() < flush_deadline) {
        if (!scanout_->dispatch(5)) break;
        try_commit(false);
    }
    while (!pending_.empty()) try_commit(true);
    // Briefly wait (up to 100 ms) for the last presentation feedback — for the glass-to-glass samples and the debugfs end snapshot.
    const auto deadline = Clock::now() + std::chrono::milliseconds(100);
    while (!await_present_.empty() && Clock::now() < deadline)
        if (!scanout_->dispatch(10)) break;
    snapshot("end", &snap_end_);   // read **before** taking it off the screen

    // Take it off the screen and wait a bounded time for the remaining releases. Whatever still remains is force-released (shutdown).
    scanout_->shutdown(kDrainTimeoutMs);
    for (auto& [tok, item] : committed_) finish_item(item);
    committed_.clear();
    await_present_.clear();
    for (Item& it : pending_) {   // remains only when the force commit was rejected
        drop_item(it);
        ++shutdown_dropped_;
    }
    pending_.clear();
    for (Retained& r : retained_) r = Retained{};
    scanout_->clear_video_cache();
    drained_ = true;
}

int PlaneDisplay::frames_held() const {
    int n = 0;
    for (const Retained& r : retained_)
        if (r.valid) ++n;
    return n + static_cast<int>(pending_.size()) + static_cast<int>(committed_.size());
}

void PlaneDisplay::snapshot(const char* tag, DrmStateSnapshot* out) {
    *out = take_drm_state_snapshot(dri_debug_dir_);
    if (!out->readable) {
        std::printf("[scanout] debugfs %s: unverified (%s)\n", tag, out->error.c_str());
    } else {
        std::printf("[scanout] debugfs %s (presented %llu): %s | hvs_underrun=%lld\n", tag,
                    static_cast<unsigned long long>(presented()), out->verdict.text.c_str(), out->hvs_underrun);
        if (cfg_.verbose) std::printf("%s", describe_planes(out->planes).c_str());
    }
    std::fflush(stdout);
}

std::string PlaneDisplay::describe() const {
    const OverlayPlaneBuffer& b = ov_->buffer(0);
    char buf[160];
    std::snprintf(buf, sizeof(buf), " | boxes=OverlayPlane AB24 LINEAR %ux%u pitch=%u x%d clear=%s", b.width, b.height,
                  b.pitch, ov_->ring_size(), clear_mode_name(ov_->clear_mode()));
    return scanout_->describe() + buf;
}

DisplaySink::ScanoutReport PlaneDisplay::scanout_report() const {
    ScanoutReport r;
    r.overlay_skipped = overlay_skipped_;
    r.frame_cb_timeouts = frame_cb_timeouts_;
    r.zero_copy_feedback = zero_copy_fb_;
    r.feedback_presented = feedback_presented_;
    r.busy_reuse = scanout_ ? scanout_->busy_reuse() : 0;
    r.discarded = discarded_;
    r.refresh_hz = refresh_ns_ > 0 ? 1e9 / static_cast<double>(refresh_ns_) : 0.0;
    // Expected gap = the largest gap of the normal cadence (late_gap_threshold, plane_display.hpp). show() puts up 1 of
    // every K frames (--display-every), so a shown frame lasts K × refresh / fps vblanks — plus one source interval per
    // frame dropped upstream inside that gap (per-gap threshold below; late_expected_gap is the no-drop value).
    r.late_expected_gap = late_gap_threshold(cfg_.every, 0, r.refresh_hz, cfg_.source_fps);
    r.upstream_missing = upstream_missing_;
    r.seq_gaps = seq_gaps_.size();
    r.seq_breaks = seq_breaks_;
    for (std::uint32_t g : seq_gaps_) r.seq_gap_max = std::max<std::uint64_t>(r.seq_gap_max, g);
    if (seq_seen_ && r.late_expected_gap > 0 && !seq_gaps_.empty()) {   // no gap left (warmup >= presented) = unverified
        long long late = 0;
        for (std::size_t i = 0; i < seq_gaps_.size(); ++i) {
            const int thr = late_gap_threshold(cfg_.every, seq_gap_missing_[i], r.refresh_hz, cfg_.source_fps);
            if (seq_gaps_[i] > static_cast<std::uint32_t>(thr)) ++late;
        }
        r.late_frames = late;
    }
    const bool expect_ov = overlay_ever_committed_;   // boxes committed at least once → an AB24 plane above the video is expected
    std::vector<const DrmStateSnapshot*> snaps;
    if (snap_start_taken_) snaps.push_back(&snap_start_);
    snaps.push_back(&snap_end_);
    bool any = false, zc = true, col = true;
    for (const DrmStateSnapshot* s : snaps) {
        if (!s->readable) continue;
        any = true;
        zc = zc && s->verdict.zero_copy(expect_ov);
        col = col && s->verdict.color_ok();
    }
    r.zero_copy = any ? (zc ? "1" : "0") : "unverified";
    r.color_ok = any ? (col ? "1" : "0") : "unverified";
    if (snap_start_taken_ && snap_start_.readable && snap_end_.readable && snap_start_.hvs_underrun >= 0 &&
        snap_end_.hvs_underrun >= 0)
        r.underrun_delta = snap_end_.hvs_underrun - snap_start_.hvs_underrun;
    auto line = [](const char* tag, const DrmStateSnapshot& s) {
        return std::string(tag) + ": " +
               (s.readable ? s.verdict.text + " | hvs_underrun=" + std::to_string(s.hvs_underrun)
                           : "unverified (" + s.error + ")");
    };
    if (snap_start_taken_) r.lines.push_back(line("start", snap_start_));
    r.lines.push_back(line("end", snap_end_));
    return r;
}

} // namespace yvo::display
