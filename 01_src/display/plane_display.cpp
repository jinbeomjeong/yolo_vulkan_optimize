#include "display/plane_display.hpp"

#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <utility>

#include "vulkan/vulkan_context.hpp"

namespace yvo::display {

namespace {

using Clock = PlaneDisplay::Clock;

// steady_clock 은 Linux(libstdc++)에서 CLOCK_MONOTONIC 이다. wp_presentation 이 CLOCK_MONOTONIC 으로 준
// 표시 시각과 retain() 시각을 같은 축에서 뺀다.
std::int64_t steady_ns(Clock::time_point t) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(t.time_since_epoch()).count();
}

} // namespace

PlaneDisplay::PlaneDisplay(yvo::vk::VulkanContext& ctx, std::unique_ptr<PlaneScanout> scanout,
                           const DisplayPipelineConfig& cfg)
    : ctx_(ctx), cfg_(cfg), scanout_(std::move(scanout)) {
    if (!scanout_) throw std::invalid_argument("PlaneDisplay: scanout is null");
    scanout_->set_listener(this);
    const VkExtent2D ext = scanout_->output_extent();
    if (ext.width == 0 || ext.height == 0) throw std::runtime_error("PlaneDisplay: 출력 크기가 0 이다");

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
    if (cfg_.pace == DisplayPace::Source && cfg_.source_fps > 0.0) {
        const double period_ms = 1000.0 / cfg_.source_fps;
        min_commit_interval_ = std::chrono::duration_cast<Clock::duration>(
            std::chrono::duration<double, std::milli>(std::max(period_ms - 8.0, 0.0)));
    }
}

PlaneDisplay::~PlaneDisplay() {
    // 호출자(SlotExchange)가 이미 사라졌을 수 있다 → 콜백을 끊고 정리한다. 소멸자 밖으로 예외를 내지 않는다.
    done_cb_ = nullptr;
    try {
        drain();
    } catch (...) {
    }
    // 전송층(wl_buffer 가 OverlayPlane 의 dma-buf 를 가리킨다)을 먼저 없애고 박스 링을 푼다.
    // 소멸 중인 전송층이 반쯤 해체된 this 로 이벤트를 보내지 않도록 리스너를 먼저 끊는다.
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

void PlaneDisplay::retain(int slot, const DrmFrame& f) {
    Retained& r = retained_.at(static_cast<std::size_t>(slot));
    r.frame = f;   // keepalive 추가 참조
    r.t = Clock::now();
    r.valid = true;
}

std::optional<Clock::time_point> PlaneDisplay::pace_deadline(std::int64_t frame_index) {
    // DisplayPipeline::pace_deadline 과 같은 의미(소스 fps 시간표).
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
    r = Retained{};

    const bool skip = cfg_.every > 1 && (shown_counter_ % cfg_.every) != 0;
    ++shown_counter_;
    if (skip) {
        ++skipped_;
        return false;   // it 가 스코프를 벗어나며 keepalive 를 놓는다
    }
    if (drained_) return false;
    it.pipe = cfg_.sync == DisplaySync::Latest ? -1 : slot;

    // 박스: 비차단 submit. 링이 비었으면 이번 프레임은 직전 박스를 유지한다.
    const int idx = ov_->acquire();
    if (idx < 0) {
        ++overlay_skipped_;
    } else {
        ov_->draw(idx, dets, hud);
        it.overlay = idx;
    }

    // pending 이 가득 차 있으면 가장 오래된 것을 버린다(드롭). 새 프레임은 뒤에 선다.
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
        // ready() 가 처음 true 를 낸 호출이 timestamp 를 회수한 호출이다(Claimed 항목은 acquire 가 건드리지 않는다).
        it.overlay_stat = true;
        const double o = ov_->last_overlay_ms();
        const double c = ov_->last_clear_ms();
        if (o >= 0.0) overlay_ms_.push_back(o + std::max(c, 0.0));
    }
    return true;
}

void PlaneDisplay::drop_item(Item& it) {
    if (it.overlay >= 0) {
        ov_->release(it.overlay);   // Claimed → Free (GPU 가 아직 쓰는 중이면 acquire 가 건너뛴다)
        it.overlay = -1;
    }
    it.frame = DrmFrame{};
    if (it.pipe >= 0 && done_cb_) done_cb_(it.pipe);
    it.pipe = -1;
}

void PlaneDisplay::finish_item(Item& it) {
    it.frame = DrmFrame{};          // keepalive 해제 → 디코더 풀로 돌아간다
    if (it.pipe >= 0 && done_cb_) done_cb_(it.pipe);
    it.pipe = -1;
}

void PlaneDisplay::try_commit(bool force) {
    if (pending_.empty()) return;
    Item& p = pending_.front();
    const Clock::time_point now = Clock::now();
    if (force) {
        // 종료 때만: 박스 GPU 완료를 기다리고(유한, OverlayPlane::wait 5 s), 상한·vblank 게이트를 무시한다.
        if (p.overlay >= 0) {
            ov_->wait(p.overlay);
            overlay_ready(p);
        }
    } else {
        if (!overlay_ready(p)) return;                                     // 박스 GPU 진행 중
        if (static_cast<int>(committed_.size()) >= kMaxCommitted) return;  // 컴포지터가 아직 놓지 않았다
        // pace=source: 소스 프레임 간격보다 빨리 커밋하지 않는다(반 vblank 여유). matched + batch 2 는 검출이
        // 두 장씩 한꺼번에 나와서, 이 게이트가 없으면 두 장이 연속 vblank(16.7 ms)에 나가고 다음은 66.7 ms 뒤에
        // 나가는 들쭉날쭉한 cadence 가 된다(실측 present 간격 median 16.7 / p90 66.7).
        if (min_commit_interval_.count() > 0 && commits_ > 0 && now - last_commit_ < min_commit_interval_) return;
        if (frame_cb_pending_) {
            if (now - last_commit_ < kFrameCallbackTimeout) return;        // 다음 vblank 대기
            // 콜백이 안 온다(창이 가려졌거나 컴포지터가 멈췄다). 기다리지 않고 커밋한다 — 커밋은 비차단이다.
            ++frame_cb_timeouts_;
            frame_cb_pending_ = false;
        }
    }
    p.token = next_token_++;
    if (!scanout_->commit(p.frame, p.token, p.overlay)) {
        ++rejected_;
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
    await_present_[p.token] = p.t_retain;
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

void PlaneDisplay::on_presented(std::uint64_t token, std::int64_t t_ns, bool clock_monotonic, bool zero_copy) {
    ++presented_;
    ++feedback_presented_;
    if (zero_copy) ++zero_copy_fb_;
    auto it = await_present_.find(token);
    if (clock_monotonic) {
        if (last_present_ns_ != 0) interval_ms_.push_back(static_cast<double>(t_ns - last_present_ns_) / 1e6);
        last_present_ns_ = t_ns;
        if (it != await_present_.end()) glass_ms_.push_back(static_cast<double>(t_ns - steady_ns(it->second)) / 1e6);
    }
    if (it != await_present_.end()) await_present_.erase(it);
}

void PlaneDisplay::on_discarded(std::uint64_t token) {
    ++discarded_;
    await_present_.erase(token);
}

// ── 루프 ──────────────────────────────────────────────────────────────────────────────────────
void PlaneDisplay::pump() {
    if (drained_) return;
    (void)scanout_->dispatch(0);   // 연결 오류는 poll_window() 가 보고한다
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
    // 남은 pending(matched 의 끝 프레임들이 flush_results 에서 방금 show 됐다)을 평소 게이트대로 최대 200 ms
    // 동안 내보내고, 그래도 남으면 게이트를 무시하고 낸다(앞 것은 화면에 안 나가고 discarded 될 수 있다).
    const auto flush_deadline = Clock::now() + std::chrono::milliseconds(200);
    while (!pending_.empty() && Clock::now() < flush_deadline) {
        if (!scanout_->dispatch(5)) break;
        try_commit(false);
    }
    while (!pending_.empty()) try_commit(true);
    // 마지막 표시 피드백을 잠깐(최대 100 ms) 기다린다 — glass-to-glass 표본과 debugfs 끝 스냅샷을 위해.
    const auto deadline = Clock::now() + std::chrono::milliseconds(100);
    while (!await_present_.empty() && Clock::now() < deadline)
        if (!scanout_->dispatch(10)) break;
    snapshot("end", &snap_end_);   // 화면에서 내리기 **전에** 읽는다

    // 화면에서 내리고 남은 release 를 유한 시간 기다린다. 그래도 남은 것은 강제로 놓는다(종료).
    scanout_->shutdown(kDrainTimeoutMs);
    for (auto& [tok, item] : committed_) finish_item(item);
    committed_.clear();
    await_present_.clear();
    for (Item& it : pending_) {   // force 커밋이 거부됐을 때만 남는다
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
    r.applicable = true;
    r.overlay_skipped = overlay_skipped_;
    r.frame_cb_timeouts = frame_cb_timeouts_;
    r.zero_copy_feedback = zero_copy_fb_;
    r.feedback_presented = feedback_presented_;
    r.busy_reuse = scanout_ ? scanout_->busy_reuse() : 0;
    if (scanout_) {
        r.transport_lines = scanout_->report_lines();
        r.fatal = scanout_->fatal_error();
    }
    const bool expect_ov = overlay_ever_committed_ && scanout_ && scanout_->expects_debugfs_overlay();
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
