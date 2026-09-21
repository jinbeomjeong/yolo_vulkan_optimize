// drm_state: debugfs `/sys/kernel/debug/dri/<n>/state` 의 활성 평면을 읽어 **실제 스캔아웃 배치**를 판정한다.
//
// 왜 필요한가: 평면 스캔아웃 백엔드(wayland-planes, kms)는 "HVS 가 디코더 NV12 SAND 버퍼를 직접 읽고
// 박스 평면을 그 위에 블렌딩한다"는 것이 전제다. Wayland 에서는 컴포지터(labwc/wlroots)가 조용히 GL 합성으로
// 떨어질 수 있고(창 모드, 포맷 거부 …), 색 범위도 labwc 0.20.1 에서 요청과 뒤집혀 나간다(wayland_plane_scanout.hpp
// "색"). 프로토콜 응답으로는 이것을 알 수 없다 — 커널 atomic 상태를 직접 읽는 것이 유일한 확인 수단이다.
//
// 원본: 06_tools/probes/hvs_probe/probe_common.hpp:137-216 (parse_active_planes / judge_placement).
// 옮기면서 바꾼 것:
//   · modifier·fourcc·zpos 를 숫자로 해석한다(문자열 prefix 비교 대신). SAND128 판정은 열 높이 파라미터를 뺀
//     modifier 로 한다(fourcc_mod_broadcom_mod 와 같은 규칙).
//   · "RGB 평면 수" 대신 **영상 평면(NV12 SAND) 위(zpos 큰 쪽)에 AB24 평면이 있는가**를 본다 — labwc 는
//     커서를 AR24 평면(zpos 가장 위)에 올리므로 RGB 평면 수로는 박스 평면을 구분할 수 없다.
//   · 영상 평면의 COLOR_ENCODING / COLOR_RANGE 를 판정한다.
//   · hvs_underrun 카운터를 함께 읽는다(누적값 — 시작·끝 차이를 본다).
//   · 읽기 실패는 예외가 아니라 readable=false ("unverified") 다. debugfs 는 root:sudo 0750 이라 권한이 없을 수 있다.
//
// 이 파일은 Wayland·Vulkan·libdrm 을 쓰지 않는다(문자열 파서). 단위 테스트: 05_tests/test_overlay.cpp DrmState.*
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace yvo::display {

inline constexpr const char* kDefaultDriDebugDir = "/sys/kernel/debug/dri/1";   // vc4 (axi:gpu)

// 활성 평면 하나 (crtc 가 붙은 평면만)
struct DrmPlaneState {
    int id = 0;                 // plane[83] → 83
    std::string name;           // "plane-2"
    std::string crtc;           // "crtc-2"
    std::string alloc;          // "labwc" / "[fbcon]" / 우리 프로세스 이름
    std::string format;         // "NV12 little-endian (0x3231564e)"
    std::uint32_t fourcc = 0;   // 괄호 안 16진수. 없으면 0
    std::string modifier_str;   // "0x700000000000004"
    std::uint64_t modifier = 0;
    std::string size;           // fb 크기 "1920x1080"
    std::string crtc_pos, src_pos;
    int zpos = -1;              // normalized-zpos
    std::string encoding;       // "ITU-R BT.709 YCbCr"
    std::string range;          // "YCbCr limited range"
};

// state 텍스트에서 crtc 가 붙은 평면만 뽑는다(fb=0 / crtc=(null) 은 버린다).
std::vector<DrmPlaneState> parse_drm_active_planes(std::string_view state);

bool is_nv12_sand128(const DrmPlaneState& p);   // NV12 + Broadcom SAND128 (열 높이 파라미터 무시)
bool is_ab24_linear(const DrmPlaneState& p);    // ABGR8888 + LINEAR

// 평면 스캔아웃 판정
struct PlaneScanoutVerdict {
    int active_planes = 0;
    int video_plane = -1;       // NV12 SAND 평면 id (없으면 -1)
    int video_zpos = -1;
    int overlay_plane = -1;     // 영상 평면보다 위의 AB24 LINEAR 평면 id (없으면 -1)
    int overlay_zpos = -1;
    bool video_on_plane = false;   // HVS 가 NV12 SAND 를 직접 읽는다 (zero-copy 영상)
    bool overlay_above = false;    // 박스 평면이 영상 위에 있다
    bool encoding_bt709 = false;   // 영상 평면 COLOR_ENCODING
    bool range_limited = false;    // 영상 평면 COLOR_RANGE
    std::string text;              // 한 줄 요약

    // expect_overlay: 박스 평면을 한 번이라도 커밋했으면 true (그 평면이 영상 위에 있어야 zero-copy 다)
    bool zero_copy(bool expect_overlay) const { return video_on_plane && (!expect_overlay || overlay_above); }
    bool color_ok() const { return video_on_plane && encoding_bt709 && range_limited; }
};

PlaneScanoutVerdict judge_plane_scanout(const std::vector<DrmPlaneState>& planes);

std::string describe_planes(const std::vector<DrmPlaneState>& planes);

// 스냅샷 = state + hvs_underrun 한 번 읽기
struct DrmStateSnapshot {
    bool readable = false;         // state 를 읽었다 (false = "unverified")
    std::string error;             // readable=false 일 때 이유
    std::vector<DrmPlaneState> planes;
    PlaneScanoutVerdict verdict;
    long long hvs_underrun = -1;   // 누적 카운터. 읽지 못하면 -1
};

DrmStateSnapshot take_drm_state_snapshot(const std::string& dri_debug_dir = kDefaultDriDebugDir);

// hvs_underrun 파일 내용 파서 (숫자 한 개). 실패하면 -1.
long long parse_hvs_underrun(std::string_view text);

} // namespace yvo::display
