// OverlayPlane: 박스·라벨·HUD 를 **내보낼 수 있는(dma-buf) 투명 RGBA 평면**에 GPU 로 그리는 링.
//
// HVS 두 평면 표시 백엔드(계획서 atomic-munching-kite §2)의 "박스 평면"이다. 영상은 디코더 NV12 SAND128
// dma-buf 를 HVS 영상 평면에 그대로 올리고, 이 클래스가 만든 AB24 dma-buf 를 그 위 평면(zpos 위)에 올려
// HVS 가 스캔아웃 중에 블렌딩한다. 전송(Wayland linux-dmabuf subsurface / KMS atomic)은 이 클래스 밖이다.
//
// ── 왜 이미지가 아니라 버퍼인가 (V3D 실측 + 스펙) ─────────────────────────────────────────────
// 계획은 LINEAR RGBA8 storage image 를 export 하는 것이었지만 V3D 7.1(Mesa V3DV)에서 불가능하다.
//   · vkGetPhysicalDeviceFormatProperties2(R8G8B8A8_UNORM): linearTilingFeatures = 0xc980
//     (STORAGE_IMAGE_BIT 없음). optimalTilingFeatures = 0xdd83 (있음).
//   · VkDrmFormatModifierPropertiesListEXT: MOD_LINEAR(0x0) = 0xc980 (STORAGE 없음),
//     BROADCOM_UIF(0x0700000000000006) = 0xdd83 (STORAGE 있음 — 그러나 AB24 LINEAR 평면용이 아니다).
//   · vkGetPhysicalDeviceImageFormatProperties2(LINEAR 또는 DRM_FORMAT_MODIFIER(LINEAR), STORAGE|TRANSFER_DST,
//     + VkPhysicalDeviceExternalImageFormatInfo(DMA_BUF)) = VK_ERROR_FORMAT_NOT_SUPPORTED.
//   → 그 조합으로 VkImage 를 만들면 VUID-VkImageCreateInfo-imageCreateMaxMipLevels-02251, 뷰를 만들면
//     VUID-VkImageViewCreateInfo-usage-02275 ("usage 에 STORAGE 가 있으면 뷰 포맷 기능에 STORAGE_IMAGE_BIT")
//     위반이다. 그래서 overlay_boxes.comp 의 **storage-buffer 변형**(overlay_boxes_buf.spv)으로 packed RGBA8
//     버퍼에 pitch 로 직접 쓰고, 버퍼는 vulkan/dmabuf_export.hpp 의 DmaBufExporter 로 할당·export 한다.
//
// ── 픽셀 포맷 ─────────────────────────────────────────────────────────────────────────────────
// 메모리 바이트 순서 R,G,B,A = DRM_FORMAT_ABGR8888 ("AB24", 리틀엔디언 32비트 A:B:G:R), modifier LINEAR.
// 박스·라벨·HUD 픽셀은 alpha 255(R7), 나머지는 (0,0,0,0). 불투명 아니면 완전 투명이라 HVS 블렌드 모드가
// Pre-multiplied 든 Coverage 든 결과가 같다.
//
// ── 지우기 (링 항목을 재사용하므로 직전 내용을 지워야 한다) ──────────────────────────────────
//   · Full  : vkCmdFillBuffer(0) 로 전체를 지운다.
//   · Erase : 이 항목에 **직전에 그린 검출·HUD 를 같은 셰이더로 다시 돌리되 (0,0,0,0) 을 쓴다**(push erase=1).
//             같은 입력·같은 affine 이면 픽셀 집합이 결정적이므로 정확히 그 픽셀만 지워진다(박스 테두리·
//             라벨·HUD 만 — 박스 내부 면적은 건드리지 않는다). 항목마다 OverlayRenderer 슬롯 두 개를 번갈아
//             쓰므로(그리기 슬롯 / 직전 내용을 가진 슬롯) DetBuf 복사가 없다.
//   Auto 는 Erase 다: 1920x1080 전체 fill 의 V3D GPU 시간이 기준(0.3 ms)을 크게 넘었기 때문이다.
//   실측(05_tests/test_overlay.cpp OverlayPlane.ClearCost, timestamp median, 50프레임, 2026-09-22):
//     Full  vkCmdFillBuffer 8.3 MB           : 2.80 ms (검출 수와 무관)
//     Erase 검출 8 + 라벨 + HUD              : 0.44 ms (= 그 프레임 오버레이 0.43 ms 와 같다)
//     Erase 검출 80, 텍스트 끔               : 1.12 ms (오버레이 1.11 ms)
//   Erase 비용은 "직전 프레임의 오버레이 비용"과 같다. 전형적인 프레임에서도 0.3 ms 를 넘으므로 더 줄이려면
//   라벨 글리프 대신 라벨 사각형을 채우는 전용 지우기 셰이더가 후보다(남은 과제).
//
// ── 동기화 ────────────────────────────────────────────────────────────────────────────────────
// 한 프레임 = (지우기 + 오버레이) 커맨드 버퍼 1개, submit 1회, 펜스 1개. `ready(idx)` 는 비차단 폴링이다.
// 스캔아웃(HVS/컴포지터)은 Vulkan 밖의 소비자라 **외부 큐 패밀리 소유권 전환**을 한다(스펙 synchronization
// 장 "Queue Family Ownership Transfer": 외부 메모리로 공유한 자원은 VkSharingMode 와 무관하게 로컬·외부 큐 사이
// 소유권 전환을 명시해야 한다). 프레임 끝에 우리 큐 → VK_QUEUE_FAMILY_FOREIGN_EXT(확장이 없으면 EXTERNAL)
// release, 다음 재사용 시작에 FOREIGN → 우리 큐 acquire. 버퍼라 레이아웃이 없다.
//
// ── 링 항목 수명 (Wayland wl_buffer.release 까지 기다린다) ─────────────────────────────────────
//   Free --acquire()--> Claimed --draw()--> (GPU 진행 중) --ready()==true--> 그려짐
//        --mark_held()--> Held(소비자가 스캔아웃 중) --release()--> Free
// 그린 뒤 표시하지 않기로 하면(드롭) mark_held 없이 release() 를 부른다. acquire() 는 Free 항목만 준다.
// 즉 재사용은 "GPU 펜스 완료" **그리고** "소비자 해제(wl_buffer.release / 다음 flip)" 둘 다 필요하다.
//
// 스레드: 전부 메인 스레드(VkQueue 하나).
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string_view>
#include <vector>

#include <vulkan/vulkan.h>

#include "core/frame_types.hpp"
#include "display/overlay_renderer.hpp"
#include "ref/overlay_reference.hpp"

namespace yvo::vk {
class VulkanContext;
class DmaBufExporter;
}

namespace yvo::display {

// DRM fourcc/modifier (drm_fourcc.h 를 끌어오지 않으려고 값만 둔다)
inline constexpr std::uint32_t kDrmFormatAbgr8888 =
    static_cast<std::uint32_t>('A') | (static_cast<std::uint32_t>('B') << 8) |
    (static_cast<std::uint32_t>('2') << 16) | (static_cast<std::uint32_t>('4') << 24);
inline constexpr std::uint64_t kDrmFormatModLinear = 0;

enum class OverlayClearMode { Auto, Full, Erase };

struct OverlayPlaneConfig {
    int ring = 3;                      // 링 항목 수 (on-screen + pending + 그리는 중)
    int width = 1920;                  // 평면 크기 (보통 화면 = 원본 크기)
    int height = 1080;
    std::uint32_t pitch_align = 64;    // 바이트 pitch 정렬 (DRM/HVS 에 넘기는 stride)
    int src_w = 1920;                  // 검출 좌표계 (make_overlay_affine 의 원본). 1:1 이면 항등
    int src_h = 1080;
    OverlayClearMode clear = OverlayClearMode::Auto;
    // 오버레이 스타일·상한. target/num_slots 는 OverlayPlane 이 덮어쓴다(Buffer, 2*ring).
    OverlayConfig overlay{};
    bool timestamps = true;
};

// 소비자(Wayland zwp_linux_buffer_params_v1 / KMS AddFB2)에 넘길 한 항목의 기술
struct OverlayPlaneBuffer {
    int fd = -1;                       // OverlayPlane 소유 (넘겨줄 때는 dup_fd())
    std::uint32_t fourcc = kDrmFormatAbgr8888;
    std::uint64_t modifier = kDrmFormatModLinear;
    std::uint32_t offset = 0;          // plane 0 오프셋 (항상 0)
    std::uint32_t pitch = 0;           // 바이트 stride (pitch_align 배수)
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::size_t size = 0;              // pitch * height (유효 바이트)
    std::size_t dmabuf_size = 0;       // lseek 로 잰 dma-buf 객체 크기 (>= size)
};

class OverlayPlane {
public:
    OverlayPlane(yvo::vk::VulkanContext& ctx, const OverlayPlaneConfig& cfg = {});
    ~OverlayPlane();
    OverlayPlane(const OverlayPlane&) = delete;
    OverlayPlane& operator=(const OverlayPlane&) = delete;

    // 검출 좌표계(원본 크기). 다음 draw() 부터 affine 이 바뀐다.
    void set_source_size(int w, int h);

    // Free 항목 하나를 잡는다. 없으면 -1 (모두 GPU 진행 중이거나 소비자가 쥐고 있다).
    int acquire();

    // 잡은 항목에 지우기 + 오버레이를 기록·제출한다(submit 1회 + 펜스). 반환값 = 실제로 기록한 검출 수.
    int draw(int idx, const std::vector<Detection>& dets, std::string_view hud = {});

    // 비차단: 이 항목의 GPU 작업이 끝났는가. 끝났으면 timestamp 를 회수한다. 제출이 없었으면 true.
    bool ready(int idx);
    // 블로킹 대기 (테스트·종료용).
    void wait(int idx);

    // 소비자가 이 항목을 스캔아웃에 쥐었다(wl_surface.commit / atomic commit 직후).
    void mark_held(int idx);
    // 소비자가 놓았다(wl_buffer.release / 다음 flip) 또는 표시하지 않기로 했다 → Free.
    void release(int idx);

    int ring_size() const { return static_cast<int>(entries_.size()); }
    const OverlayPlaneBuffer& buffer(int idx) const;
    int dup_fd(int idx) const;          // 호출자가 close 한다
    OverlayClearMode clear_mode() const { return clear_mode_; }
    bool held(int idx) const;
    bool claimed(int idx) const;

    // 마지막으로 ready() 가 회수한 GPU 시간(ms). 측정 불가면 < 0.
    double last_clear_ms() const { return last_clear_ms_; }
    double last_overlay_ms() const { return last_overlay_ms_; }
    std::uint64_t overlay_overflows() const { return overflows_; }
    OverlayStyle style() const { return ov_->style(); }
    OverlayAffine affine() const { return affine_; }

private:
    enum class State { Free, Claimed, Held };
    struct Entry {
        OverlayPlaneBuffer desc;
        VkBuffer buffer = VK_NULL_HANDLE;
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;
        State state = State::Free;
        bool submitted = false;        // 펜스가 걸린 제출이 있다(아직 회수 안 됨)
        bool released_foreign = false; // 마지막 제출이 FOREIGN 으로 release 했다 → 다음에 acquire 필요
        int draw_slot = 0;             // OverlayRenderer 슬롯 (2*idx 또는 2*idx+1)
        bool has_prev = false;         // 이전에 그린 내용이 있다 (Erase 모드가 지울 대상)
        OverlayAffine prev_affine{};
        bool pending_erase = false;    // 이번 제출에 erase 패스가 들어 있다 (timestamp 회수용)
        bool pending_fill = false;
        std::uint64_t seq = 0;         // acquire 순서 (가장 오래 쉰 항목 우선)
    };

    void create();
    void destroy();
    void initial_clear();              // 생성 직후 전체 0 (한 번, 블로킹)
    void collect(Entry& e, int idx);
    Entry& entry(int idx);
    const Entry& entry(int idx) const;

    yvo::vk::VulkanContext& ctx_;
    OverlayPlaneConfig cfg_;
    OverlayClearMode clear_mode_ = OverlayClearMode::Erase;
    OverlayAffine affine_{};
    std::uint32_t pitch_px_ = 0;
    std::uint32_t foreign_qf_ = VK_QUEUE_FAMILY_EXTERNAL;

    std::unique_ptr<yvo::vk::DmaBufExporter> exporter_;
    std::unique_ptr<OverlayRenderer> ov_;
    VkCommandPool pool_ = VK_NULL_HANDLE;
    VkQueryPool fill_query_ = VK_NULL_HANDLE;   // Full 모드 fill 구간 (항목당 2개)
    std::vector<Entry> entries_;
    std::uint64_t seq_ = 0;

    double last_clear_ms_ = -1.0;
    double last_overlay_ms_ = -1.0;
    std::uint64_t overflows_ = 0;
};

const char* clear_mode_name(OverlayClearMode m);

} // namespace yvo::display
