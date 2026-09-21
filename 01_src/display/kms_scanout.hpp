// KmsScanout: KMS atomic implementation of PlaneScanout (--display-backend kms).
//
// Commits two planes directly as DRM master of card1 (vc4) without a compositor. Plan atomic-munching-kite "step 4".
//   Video plane : decoder DrmFrame (NV12, SAND128, separate Y/UV dma-bufs) → drmPrimeFDToHandle ×2 → AddFB2WithModifiers.
//               pitch field = SAND column height (DrmFrame::y_pitch/uv_pitch, kernel vc4 HVS6 convention). SRC = frame.width/height.
//               COLOR_ENCODING=BT.709 and COLOR_RANGE are set on **every commit** (auto = limited: KMS has no fullhack).
//               If zpos is mutable, its min is set explicitly on every commit.
//   Box plane : OverlayPlane ring entry (AB24 LINEAR dma-buf) → one AddFB2 per entry in register_overlay().
//               zpos above video, pixel blend mode=Pre-multiplied, alpha=0xffff.
//   Origin: 06_tools/probes/hvs_probe/hvs_kms_probe.cpp. Defects fixed — hardcoded SRC size (1920x1080), Y GEM handle leak
//   on UV handle failure, COLOR properties set only on the first commit, double close of GEM handles of the same dma-buf (solved with handle refcounts).
//
// ── Selection ─────────────────────────────────────────────────────────────────────────────────
//   Connector: first connected HDMI-A (set with --display-connector). Mode: interlaced excluded, 1920x1080@60 → preferred → first mode.
//   CRTC: the CRTC the connector's encoder currently uses → else a **not-in-use** CRTC among possible_crtcs → else the first bit.
//   Box plane: prefer an AB24 LINEAR Overlay whose current CRTC_ID is 0 or our CRTC (do not steal a plane used by another output).
//
// ── Event contract (same meaning as WaylandPlaneScanout) ──────────────────────────────────────
//   commit(token N) **always** attaches DRM_MODE_PAGE_FLIP_EVENT (user_data = token).
//   Only the first commit is an ALLOW_MODESET **blocking** commit after a TEST_ONLY check — the flip event still queues on the fd and
//   dispatch() reads it, so the event flow is the same as for later non-blocking commits. Listeners are called **only inside dispatch()**
//   (not inside commit(): PlaneDisplay records mark_held/committed_ after commit returns).
//   On the flip(N, tv) event they are called in this order:
//     1. on_frame_done()                               — the next commit may be made (corresponds to wl_surface.frame)
//     2. on_presented(N, tv→ns, TIMESTAMP_MONOTONIC, zero_copy=true)   (corresponds to wp_presentation.presented)
//     3. on_video_released(token of the previous on-screen commit)        — the HVS no longer reads that buffer (corresponds to wl_buffer.release)
//     4. if commit N changed the boxes, on_overlay_released(previous on-screen box entry)
//   on_discarded is not called (a non-blocking commit is rejected with EBUSY, not replaced).
//   At most 1 in-flight. Committing before the previous flip arrives returns false without an ioctl (= caller drops, counted in busy_reuse).
//   Flip watchdog: if in-flight does not finish within kFlipTimeoutMs, warn once + treat it as flipped (call 1·3·4 without
//   on_presented, counted in flip_timeouts). That event arriving late is ignored because its token differs.
//
// ── Measurement caveats (B8/B10) ──────────────────────────────────────────────────────────────
//   The first commit is a blocking modeset, so the call takes tens of ms (main loop stalls), and its flip time is the modeset completion time,
//   not the normal vblank period. In glass-to-glass·present interval statistics **exclude the first N frames (right after modeset)**
//   (PlaneDisplay's debugfs start snapshot is also taken after presented 60). To be handled in the 07_results measurement scripts.
//
// ── master ────────────────────────────────────────────────────────────────────────────────────
//   Right after open, drmIsMaster → otherwise drmSetMaster. On failure, KmsMasterError (finds who is master from debugfs clients and
//   puts it in the message). Operation: leave lightdm as is and deactivate the desktop session with `loginctl activate <non-desktop session id>` (no sudo needed, or
//   `sudo chvt N`); master becomes free, and the kernel grants it automatically when card1 is opened.
//   Do not switch VT while running (master is lost — see master_lost below).
//   If a commit returns EACCES while running, master was lost (VT return, etc.): warn once, reject (drop) later commits without an ioctl, and
//   mark it as master_lost in report_lines().
//   KmsScanoutOptions::acquire_master=false (tests) **never takes** master: it does not call drmSetMaster,
//   and if open granted it automatically, it calls drmDropMaster immediately. commit() in this state returns false without an ioctl.
//
// ── Shutdown·restore ──────────────────────────────────────────────────────────────────────────
//   **Right before** the first modeset, the current KMS state is saved (capture_state): CRTC MODE_ID·ACTIVE·GAMMA_LUT·CTM, connector CRTC_ID·
//   Broadcast RGB, and FB_ID·CRTC_ID·SRC_*·CRTC_*·zpos·color·blend properties of the planes on our CRTC and the two planes we will use.
//   shutdown() first tries a blocking commit that restores that state. With the session-deactivation method, the previous master (labwc) keeps card1
//   open, so there is no lastclose and fbcon does not restore by itself. If the restore commit is rejected (measured EINVAL —
//   presumably the previous master's FB changed in the meantime), it falls back to turning off both planes → turning off the CRTC (planes-off). The result is left in a one-line log and report_lines(). Then drmDropMaster.
//   Even if the process terminates abnormally, the kernel reclaims master (restoring the screen is up to the next master).
//
// Threads: everything on the main thread. There is no window, so quit_requested is always false (shutdown via SIGINT/SIGTERM).
// should_close is true only on an unrecoverable error (the first modeset's TEST_ONLY failed kMaxModesetTestFailures times in a row).
#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "display/display_pipeline.hpp"   // ColorRangeRequest
#include "display/plane_display.hpp"

namespace yvo::display {

// Failed to obtain DRM master (main exits with code 2).
class KmsMasterError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

struct KmsScanoutOptions {
    std::string card = "/dev/dri/card1";
    std::string connector;              // "" = first connected HDMI-A. e.g. "HDMI-A-2"
    ColorRangeRequest color_range = ColorRangeRequest::Auto;   // auto = limited
    bool acquire_master = true;         // false = discovery·FB only (tests). Never takes master
    std::string dri_debug_dir = kDefaultDriDebugDir;   // location of the clients file used to find the master owner's name
    bool verbose = false;
};

class KmsScanout final : public PlaneScanout {
public:
    static constexpr int kFlipTimeoutMs = 1000;
    static constexpr int kMaxModesetTestFailures = 3;

    // Open device → caps → master → select connector·mode·CRTC·planes. On failure std::runtime_error,
    // on master failure (acquire_master=true) KmsMasterError.
    explicit KmsScanout(const KmsScanoutOptions& opt);
    ~KmsScanout() override;
    KmsScanout(const KmsScanout&) = delete;
    KmsScanout& operator=(const KmsScanout&) = delete;

    const char* name() const override { return "kms"; }
    void set_listener(PlaneScanoutListener* l) override { listener_ = l; }
    VkExtent2D output_extent() const override;
    void register_overlay(const OverlayPlane& ov) override;
    bool commit(const DrmFrame& video, std::uint64_t token, int overlay_idx) override;
    bool dispatch(int timeout_ms) override;
    bool quit_requested() const override { return false; }
    bool should_close() const override { return !fatal_.empty(); }
    void shutdown(int timeout_ms) override;
    void clear_video_cache() override;
    std::string describe() const override;
    std::uint64_t busy_reuse() const override { return busy_; }
    bool reports_presentation() const override { return true; }
    std::vector<std::string> report_lines() const override;
    std::string fatal_error() const override { return fatal_; }

    // ── For diagnostics·tests ──
    bool is_master() const { return master_; }
    bool master_lost() const { return master_lost_; }
    bool timestamp_monotonic() const { return ts_monotonic_; }
    std::uint32_t connector_id() const { return conn_id_; }
    const std::string& connector_name() const { return conn_name_; }
    std::uint32_t crtc_id() const { return crtc_id_; }
    std::uint32_t video_plane_id() const { return vplane_ ? vplane_->id : 0; }
    std::uint32_t overlay_plane_id() const { return oplane_ ? oplane_->id : 0; }
    bool video_plane_nv12_sand128() const { return vplane_ && vplane_->nv12_sand; }
    bool overlay_plane_ab24_linear() const { return oplane_ && oplane_->ab24_lin; }
    std::uint32_t overlay_fb(int idx) const;          // FB id created by register_overlay (0 if none)
    std::uint32_t video_fb_for(const DrmFrame& f);    // creates / looks up in cache the video FB (0 on failure, last_error)
    std::size_t video_fb_count() const { return video_.size(); }
    const std::string& last_error() const { return last_error_; }
    std::uint64_t commit_failures() const { return commit_fail_; }
    std::uint64_t flip_timeouts() const { return flip_timeouts_; }

    // One entry of the saved state for restore (restores value to property prop_id of obj)
    struct SavedProp {
        std::string kind;               // "crtc" / "connector" / "plane"
        std::uint32_t obj = 0;
        std::string name;
        std::uint32_t prop_id = 0;
        std::uint64_t value = 0;
    };
    // Reads and saves the current KMS state (called automatically right before the first modeset). Works without master (read-only).
    void capture_state();
    const std::vector<SavedProp>& saved_state() const { return saved_; }
    // One "crtc[94].MODE_ID=123" per line (for tests·verbose log)
    std::string saved_state_text() const;
    const std::string& shutdown_result() const { return shutdown_result_; }

    // Test guard: opens the card and checks the master state. Does not touch the screen.
    //   Unavailable = cannot open, OtherMaster = another process is master (desktop active),
    //   NoMaster = nobody was master (open granted master automatically and we **immediately called drmDropMaster**).
    enum class MasterProbe { Unavailable, OtherMaster, NoMaster };
    static MasterProbe probe_master(const std::string& card);

    // List of "name(pid)" of processes that are master(y) in debugfs clients. "" if unreadable (for tests·error messages)
    static std::string find_master_holders(const std::string& dri_debug_dir);

private:
    struct Prop {
        std::uint32_t id = 0;
        std::uint64_t value = 0;
        std::uint32_t flags = 0;
        std::uint64_t min = 0, max = 0;                    // RANGE
        std::map<std::string, std::uint64_t> enums;        // ENUM
    };
    struct Props {
        std::map<std::string, Prop> m;
        std::uint32_t id(const std::string& n) const;
        const Prop* get(const std::string& n) const;
    };
    struct Plane {
        std::uint32_t id = 0;
        std::uint32_t possible_crtcs = 0;
        std::uint32_t cur_crtc = 0;     // CRTC attached at discovery time (0 = none)
        std::string type;               // Primary / Overlay / Cursor
        bool nv12_sand = false, ab24_lin = false;
        Props props;
    };
    using VideoKey = std::array<std::uint64_t, 4>;   // (Y dev, Y ino, UV dev, UV ino)
    struct VideoFb {
        std::uint32_t fb = 0, hy = 0, huv = 0;
        int w = 0, h = 0;
        std::uint64_t modifier = 0;
    };
    struct Shown {                      // commit on screen or awaiting flip
        bool valid = false;
        std::uint64_t token = 0;
        int overlay = -1;               // box entry raised by this commit (-1 = boxes unchanged)
        std::uint32_t video_fb = 0;
    };
    // Color pipeline defaults set on the CRTC/connector at modeset (only if the property exists)
    struct ModesetExtra {
        const char* kind;
        std::uint32_t obj;
        std::string name;
        std::uint64_t value;
        std::string text;               // "GAMMA_LUT=0" for describe()
    };

    Props load_props(std::uint32_t obj, std::uint32_t type) const;
    void load_planes();
    void select_output();
    void select_planes();
    void acquire_master_or_throw();
    void prepare_modeset_extras();

    bool import_handle(int fd, std::uint32_t* handle, int* err);   // drmPrimeFDToHandle + refcount
    void release_handle(std::uint32_t handle);
    void destroy_video_fb(VideoFb& v);

    bool add(void* req, std::uint32_t obj, const Props& p, const char* name, std::uint64_t value, bool required);
    bool build_request(void* req, std::uint32_t video_fb, int vw, int vh, int overlay_idx, bool modeset);
    std::uint64_t color_encoding_value() const;
    std::uint64_t color_range_value() const;
    const char* color_range_text() const;

    // Shutdown commits. Returns 0 = success, otherwise errno (-1 if building the request failed)
    int commit_restore();
    int commit_planes_off(bool crtc_off);

    static void on_flip(int fd, unsigned seq, unsigned sec, unsigned usec, unsigned crtc, void* data);
    void handle_flip(std::uint64_t token, bool have_time, unsigned sec, unsigned usec);
    void check_flip_watchdog();
    void fail(const std::string& msg);
    void on_master_lost(const std::string& where);

    KmsScanoutOptions opt_;
    PlaneScanoutListener* listener_ = nullptr;
    int fd_ = -1;
    bool master_ = false;
    bool master_lost_ = false;
    bool ts_monotonic_ = false;

    std::uint32_t conn_id_ = 0, crtc_id_ = 0, mode_blob_ = 0;
    int crtc_idx_ = -1;
    std::string conn_name_;
    int mode_w_ = 0, mode_h_ = 0, mode_hz_ = 0;
    std::string mode_name_;
    std::vector<std::uint8_t> mode_raw_;   // drmModeModeInfo bytes (to avoid pulling xf86drmMode.h into the header)
    Props conn_props_, crtc_props_;
    std::vector<Plane> planes_;
    const Plane* vplane_ = nullptr;
    const Plane* oplane_ = nullptr;
    std::uint64_t video_zpos_ = 0;
    bool video_zpos_settable_ = false;
    std::uint64_t overlay_zpos_ = 0;
    bool overlay_zpos_settable_ = false;
    std::vector<ModesetExtra> modeset_extras_;

    std::map<std::uint32_t, int> handle_refs_;        // GEM handle → refcount (same dma-buf gives same handle)
    std::map<VideoKey, VideoFb> video_;
    struct OverlayFb {
        std::uint32_t fb = 0, handle = 0, w = 0, h = 0;
    };
    std::vector<OverlayFb> overlay_;

    bool modeset_done_ = false;
    bool shut_ = false;
    int modeset_test_failures_ = 0;     // consecutive TEST_ONLY failures of the first modeset
    std::string fatal_;                 // unrecoverable error (should_close)
    Shown on_screen_, in_flight_;
    std::chrono::steady_clock::time_point in_flight_since_{};
    int shown_overlay_ = -1;            // entry currently on the box plane (-1 = off)

    std::vector<SavedProp> saved_;
    bool state_captured_ = false;
    std::string shutdown_result_ = "not-run";

    std::uint64_t busy_ = 0, commit_fail_ = 0, flips_ = 0, flip_timeouts_ = 0, lost_drops_ = 0;
    std::string last_error_;
};

} // namespace yvo::display
