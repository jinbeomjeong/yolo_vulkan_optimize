// KmsScanout: PlaneScanout 의 KMS atomic 구현 (--display-backend kms).
//
// 컴포지터 없이 card1(vc4) 의 DRM master 로 평면 두 장을 직접 커밋한다. 계획서 atomic-munching-kite "4단계".
//   영상 평면 : 디코더 DrmFrame(NV12, SAND128, Y/UV 별도 dma-buf) → drmPrimeFDToHandle ×2 → AddFB2WithModifiers.
//               pitch 필드 = SAND 열 높이(DrmFrame::y_pitch/uv_pitch, 커널 vc4 HVS6 관례). SRC = frame.width/height.
//               COLOR_ENCODING=BT.709, COLOR_RANGE 를 **매 커밋** 넣는다(auto = limited: KMS 에는 fullhack 이 없다).
//               zpos 가 mutable 이면 그 min 을 매 커밋 명시한다.
//   박스 평면 : OverlayPlane 링 항목(AB24 LINEAR dma-buf) → register_overlay() 에서 항목마다 AddFB2 한 번.
//               zpos 영상 위, pixel blend mode=Pre-multiplied, alpha=0xffff.
//   원본: 06_tools/probes/hvs_probe/hvs_kms_probe.cpp. 고친 결함 — SRC 크기 하드코딩(1920x1080), UV 핸들 실패 시
//   Y GEM 핸들 누수, COLOR 속성을 첫 커밋에만 넣던 것, 같은 dma-buf 의 GEM 핸들 중복 close(핸들 참조 계수로 해결).
//
// ── 선택 ──────────────────────────────────────────────────────────────────────────────────────
//   커넥터: 연결된 첫 HDMI-A (--display-connector 로 지정). 모드: 인터레이스 제외, 1920x1080@60 → preferred → 첫 모드.
//   CRTC: 커넥터 인코더가 지금 쓰는 CRTC → 없으면 possible_crtcs 중 **사용 중이 아닌** CRTC → 그래도 없으면 첫 비트.
//   박스 평면: 현재 CRTC_ID 가 0 이거나 우리 CRTC 인 AB24 LINEAR Overlay 우선(다른 출력이 쓰는 평면을 뺏지 않는다).
//
// ── 이벤트 계약 (WaylandPlaneScanout 과 같은 의미) ──────────────────────────────────────────────
//   commit(token N) 은 **항상 DRM_MODE_PAGE_FLIP_EVENT** 를 붙인다(user_data = token).
//   첫 커밋만 TEST_ONLY 확인 뒤 ALLOW_MODESET **블로킹** 커밋이다 — 그래도 flip 이벤트는 fd 에 쌓이고
//   dispatch() 가 읽으므로 이벤트 흐름이 이후 비차단 커밋과 같다. 리스너 호출은 **dispatch() 안에서만** 한다
//   (commit() 안에서 부르지 않는다: PlaneDisplay 는 commit 이 돌아온 뒤에 mark_held/committed_ 를 기록한다).
//   flip(N, tv) 이벤트에서 이 순서로 부른다:
//     1. on_frame_done()                               — 다음 커밋을 해도 된다 (wl_surface.frame 대응)
//     2. on_presented(N, tv→ns, TIMESTAMP_MONOTONIC, zero_copy=true)   (wp_presentation.presented 대응)
//     3. on_video_released(직전 화면 커밋 token)        — 이제 HVS 가 그 버퍼를 읽지 않는다 (wl_buffer.release 대응)
//     4. 커밋 N 이 박스를 바꿨으면 on_overlay_released(직전 화면 박스 항목)
//   on_discarded 는 부르지 않는다(비차단 커밋은 EBUSY 로 거부되지 대체되지 않는다).
//   in-flight 는 최대 1개다. 이전 flip 이 오기 전에 commit 하면 ioctl 없이 false(= 호출자가 드롭, busy_reuse 에 센다).
//   flip 워치독: in-flight 가 kFlipTimeoutMs 넘게 안 끝나면 경고 1회 + flip 된 것으로 간주한다(on_presented 없이
//   1·3·4 를 부른다, flip_timeouts 에 센다). 늦게 온 그 이벤트는 token 이 달라 무시된다.
//
// ── 측정 주의 (B8/B10) ────────────────────────────────────────────────────────────────────────
//   첫 커밋은 블로킹 모드셋이라 호출이 수십 ms 걸리고(메인 루프 정지), 그 flip 시각은 모드셋 완료 시각이지 정상
//   vblank 주기가 아니다. glass-to-glass·present 간격 통계에서 **첫 N 프레임(모드셋 직후)은 빼고** 봐야 한다
//   (PlaneDisplay 의 debugfs 시작 스냅샷도 presented 60 뒤에 뜬다). 07_results 측정 스크립트에서 처리할 것.
//
// ── master ────────────────────────────────────────────────────────────────────────────────────
//   open 직후 drmIsMaster → 아니면 drmSetMaster. 실패하면 KmsMasterError(누가 master 인지 debugfs clients 에서 찾아
//   메시지에 넣는다). 운용: lightdm 은 그대로 두고 `loginctl activate <비데스크톱 세션 id>`(sudo 불필요, 또는
//   `sudo chvt N`)로 데스크톱 세션을 비활성화하면 master 가 비고, card1 을 open 할 때 커널이 자동으로 준다.
//   실행 중에는 VT 를 전환하지 않는다(master 를 잃는다 — 아래 master_lost).
//   실행 중 커밋이 EACCES 면 master 를 잃은 것이다(VT 복귀 등): 경고 1회, 이후 커밋은 ioctl 없이 거부(드롭)하고
//   report_lines() 에 master_lost 로 표시한다.
//   KmsScanoutOptions::acquire_master=false(테스트)는 master 를 **절대 가져가지 않는다**: drmSetMaster 를 부르지
//   않고, open 에서 자동으로 받았으면 즉시 drmDropMaster 한다. 이 상태의 commit() 은 ioctl 없이 false 다.
//
// ── 종료·복원 ─────────────────────────────────────────────────────────────────────────────────
//   첫 모드셋 **직전에** 현재 KMS 상태를 저장한다(capture_state): CRTC MODE_ID·ACTIVE·GAMMA_LUT·CTM, 커넥터 CRTC_ID·
//   Broadcast RGB, 우리 CRTC 에 붙은 평면과 우리가 쓸 두 평면의 FB_ID·CRTC_ID·SRC_*·CRTC_*·zpos·색·블렌드 속성.
//   shutdown() 은 그 상태를 되돌리는 블로킹 커밋을 먼저 시도한다. 세션 비활성화 방식에서는 이전 master(labwc)가 card1 을
//   계속 열어 두어 lastclose 가 없고 fbcon 이 스스로 복원하지 않기 때문이다. 복원 커밋이 거부되면(실측 EINVAL —
//   이전 master 의 FB 가 그사이 바뀐 것으로 추정) 평면 두 장 끄기 → CRTC 끄기(planes-off)로 폴백한다. 결과는 한 줄 로그와 report_lines() 에 남는다. 그다음 drmDropMaster.
//   프로세스가 비정상 종료해도 커널이 master 를 회수한다(화면 복원은 다음 master 몫).
//
// 스레드: 전부 메인 스레드. 창이 없으므로 quit_requested 는 항상 false(종료는 SIGINT/SIGTERM).
// should_close 는 복구 불가 오류(첫 모드셋 TEST_ONLY 가 kMaxModesetTestFailures 번 연속 실패)일 때만 true.
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

// DRM master 를 얻지 못했다 (main 은 종료 코드 2).
class KmsMasterError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

struct KmsScanoutOptions {
    std::string card = "/dev/dri/card1";
    std::string connector;              // "" = 연결된 첫 HDMI-A. 예: "HDMI-A-2"
    ColorRangeRequest color_range = ColorRangeRequest::Auto;   // auto = limited
    bool acquire_master = true;         // false = 탐색·FB 만 (테스트). master 를 절대 가져가지 않는다
    std::string dri_debug_dir = kDefaultDriDebugDir;   // master 소유자 이름을 찾을 clients 파일 위치
    bool verbose = false;
};

class KmsScanout final : public PlaneScanout {
public:
    static constexpr int kFlipTimeoutMs = 1000;
    static constexpr int kMaxModesetTestFailures = 3;

    // 장치 열기 → 캡 → master → 커넥터·모드·CRTC·평면 선택. 실패하면 std::runtime_error,
    // master 실패(acquire_master=true)면 KmsMasterError.
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

    // ── 진단·테스트용 ──
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
    std::uint32_t overlay_fb(int idx) const;          // register_overlay 가 만든 FB id (없으면 0)
    std::uint32_t video_fb_for(const DrmFrame& f);    // 영상 FB 를 만든다/캐시에서 찾는다 (실패 0, last_error)
    std::size_t video_fb_count() const { return video_.size(); }
    const std::string& last_error() const { return last_error_; }
    std::uint64_t commit_failures() const { return commit_fail_; }
    std::uint64_t flip_timeouts() const { return flip_timeouts_; }

    // 복원용 저장 상태의 한 항목 (obj 의 속성 prop_id 에 value 를 되돌린다)
    struct SavedProp {
        std::string kind;               // "crtc" / "connector" / "plane"
        std::uint32_t obj = 0;
        std::string name;
        std::uint32_t prop_id = 0;
        std::uint64_t value = 0;
    };
    // 현재 KMS 상태를 읽어 저장한다(첫 모드셋 직전에 자동 호출). master 없이도 된다(읽기만 한다).
    void capture_state();
    const std::vector<SavedProp>& saved_state() const { return saved_; }
    // "crtc[94].MODE_ID=123" 한 줄씩 (테스트·verbose 로그용)
    std::string saved_state_text() const;
    const std::string& shutdown_result() const { return shutdown_result_; }

    // 테스트 가드: card 를 열어 master 상태를 본다. 화면은 건드리지 않는다.
    //   Unavailable = 열 수 없다, OtherMaster = 다른 프로세스가 master(데스크톱 활성),
    //   NoMaster = 아무도 master 가 아니었다(open 이 자동으로 master 를 받았고 **즉시 drmDropMaster** 했다).
    enum class MasterProbe { Unavailable, OtherMaster, NoMaster };
    static MasterProbe probe_master(const std::string& card);

    // debugfs clients 에서 master(y) 인 프로세스 "이름(pid)" 목록. 읽지 못하면 "" (테스트·오류 메시지용)
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
        std::uint32_t cur_crtc = 0;     // 탐색 시점에 붙어 있던 CRTC (0 = 없음)
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
    struct Shown {                      // 화면 또는 flip 대기 중인 커밋
        bool valid = false;
        std::uint64_t token = 0;
        int overlay = -1;               // 이 커밋이 올린 박스 항목 (-1 = 박스 그대로)
        std::uint32_t video_fb = 0;
    };
    // 모드셋 때 CRTC/커넥터에 넣는 색 파이프라인 기본값 (속성이 있을 때만)
    struct ModesetExtra {
        const char* kind;
        std::uint32_t obj;
        std::string name;
        std::uint64_t value;
        std::string text;               // describe() 용 "GAMMA_LUT=0"
    };

    Props load_props(std::uint32_t obj, std::uint32_t type) const;
    void load_planes();
    void select_output();
    void select_planes();
    void acquire_master_or_throw();
    void prepare_modeset_extras();

    bool import_handle(int fd, std::uint32_t* handle, int* err);   // drmPrimeFDToHandle + 참조 계수
    void release_handle(std::uint32_t handle);
    void destroy_video_fb(VideoFb& v);

    bool add(void* req, std::uint32_t obj, const Props& p, const char* name, std::uint64_t value, bool required);
    bool build_request(void* req, std::uint32_t video_fb, int vw, int vh, int overlay_idx, bool modeset);
    std::uint64_t color_encoding_value() const;
    std::uint64_t color_range_value() const;
    const char* color_range_text() const;

    // 종료 커밋들. 반환 0 = 성공, 아니면 errno(요청 구성 실패는 -1)
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
    std::vector<std::uint8_t> mode_raw_;   // drmModeModeInfo 바이트 (헤더에 xf86drmMode.h 를 끌어오지 않으려고)
    Props conn_props_, crtc_props_;
    std::vector<Plane> planes_;
    const Plane* vplane_ = nullptr;
    const Plane* oplane_ = nullptr;
    std::uint64_t video_zpos_ = 0;
    bool video_zpos_settable_ = false;
    std::uint64_t overlay_zpos_ = 0;
    bool overlay_zpos_settable_ = false;
    std::vector<ModesetExtra> modeset_extras_;

    std::map<std::uint32_t, int> handle_refs_;        // GEM 핸들 → 참조 수 (같은 dma-buf 는 같은 핸들)
    std::map<VideoKey, VideoFb> video_;
    struct OverlayFb {
        std::uint32_t fb = 0, handle = 0, w = 0, h = 0;
    };
    std::vector<OverlayFb> overlay_;

    bool modeset_done_ = false;
    bool shut_ = false;
    int modeset_test_failures_ = 0;     // 첫 모드셋 TEST_ONLY 연속 실패 수
    std::string fatal_;                 // 복구 불가 오류 (should_close)
    Shown on_screen_, in_flight_;
    std::chrono::steady_clock::time_point in_flight_since_{};
    int shown_overlay_ = -1;            // 지금 박스 평면에 올라가 있는 항목 (-1 = 꺼짐)

    std::vector<SavedProp> saved_;
    bool state_captured_ = false;
    std::string shutdown_result_ = "not-run";

    std::uint64_t busy_ = 0, commit_fail_ = 0, flips_ = 0, flip_timeouts_ = 0, lost_drops_ = 0;
    std::string last_error_;
};

} // namespace yvo::display
