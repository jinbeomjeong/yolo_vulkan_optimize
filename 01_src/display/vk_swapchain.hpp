// VulkanSwapchain: Wayland surface + 스왑체인 + present. 계획서 §5.6 을 그대로 구현한다.
//
// **VulkanContext 는 VkSurfaceKHR 를 영원히 모른다** — surface 는 이 클래스가 만든다(계획서 §5.2).
// 생성 순서: WaylandWindow::open() → VulkanContext(wayland_display=native_display()) → VulkanSwapchain.
//
// ── 확정 사항과 근거 (계획서 §5.6) ────────────────────────────────────────────────────────────
//  · 포맷 R8G8B8A8_UNORM + SRGB_NONLINEAR **고정**. 이유 셋이 같은 답을 가리킨다:
//      ① Vulkan 에서 _SRGB 포맷은 storage image 가 될 수 없다
//      ② shaderStorageImageWriteWithoutFormat=false 라 GLSL 포맷 한정자가 필수고, GLSL 에 bgra8 이
//         없으므로 컴퓨트 직접 쓰기는 rgba8 = R8G8B8A8_UNORM 뿐이다
//      ③ 전처리 셰이더(sand_letterbox.comp:94-101)가 내놓는 값은 이미 감마 인코딩된 sRGB 0~255 →
//         UNORM 이면 아무 변환 없이 통과한다
//  · usage = STORAGE | COLOR_ATTACHMENT | TRANSFER_DST.
//    supportedUsageFlags 가 STORAGE 를 보고해도 **포맷별 optimalTilingFeatures 는 별개 질의**이므로
//    vkGetPhysicalDeviceImageFormatProperties2(STORAGE) 로 런타임 확인하고, 불가하면
//    **자체 이미지 + vkCmdBlitImage 폴백**으로 내려간다. blit 폴백은 force_blit 으로 강제할 수 있어
//    (프로브 --display-blit) 항상 테스트된다.
//  · 이미지 개수는 VK_EXT_surface_maintenance1 로 **present mode 별 질의**한다
//    (실측: 기본 caps 4 지만 모드별로는 FIFO 3 / MAILBOX 4 / IMMEDIATE 3).
//  · currentExtent = 0xFFFFFFFF, supportedPresentScaling = None → **이미지 크기 = 창 크기를 우리가 맞춘다.**
//
// ── acquire 규칙: 절대 블로킹하지 않는다 ──────────────────────────────────────────────────────
//   acquire(timeout = 0). VK_NOT_READY 면 **이 프레임 표시를 버린다**(display_dropped++).
//   디코드가 표시 때문에 막히면 안 되기 때문이다. 획득만 하고 안 그린 이미지는
//   VK_EXT_swapchain_maintenance1 의 vkReleaseSwapchainImagesEXT 로 되돌린다
//   (확장이 없으면 다음 acquire 가 재사용하도록 stash 에 보관한다 — 이 하드웨어에서는 쓰이지 않는 안전망).
//
// ── 재생성: vkDeviceWaitIdle 없이 ─────────────────────────────────────────────────────────────
//   OUT_OF_DATE / SUBOPTIMAL 또는 WaylandWindow::take_pending_size() 가 트리거다.
//   present 마다 VkSwapchainPresentFenceInfoEXT 로 펜스를 붙이므로 **제출한 이미지가 다시 쓸 수 있게 된
//   시점을 정확히 안다** → 구 스왑체인은 그 펜스들이 모두 signaled 될 때 폐기한다.
//   흔한 per-image-fence 해킹도 vkDeviceWaitIdle 도 쓰지 않는다. 세마포어·펜스는 **스왑체인 인스턴스마다
//   따로 소유**한다 — 그래야 구 스왑체인에 걸린 펜스를 신 스왑체인이 덮어쓰지 않는다.
//
// ── 동기화 ────────────────────────────────────────────────────────────────────────────────────
//   인플라이트(= 이미지 개수)당 binary 세마포어 2개(acquire, render-finished) + 펜스 2개(submit, present).
//   timeline 세마포어는 WSI acquire/present 에 쓸 수 없다.
//
// ── 지연 계측 ─────────────────────────────────────────────────────────────────────────────────
//   VK_KHR_present_id + VK_KHR_present_wait. vkWaitForPresentKHR(presentId) 가 **실제 화면 도달 시각**을
//   준다(GPU 시간이 아니다). wp_presentation 폴백은 **만들지 않는다** — 1단계 프로브가 present_id=1,
//   present_wait=1 을 확인했고, 목적이 겹치는 두 수단을 이중으로 두지 않는다.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <vulkan/vulkan.h>

namespace yvo::vk {
class VulkanContext;
}

namespace yvo::display {

struct SwapchainOptions {
    void* wayland_display = nullptr;   // struct wl_display*  (WaylandWindow::native_display())
    void* wayland_surface = nullptr;   // struct wl_surface*  (WaylandWindow::native_surface())
    int width = 1920;                  // 이미지 크기 = 창 크기 (컴포지터가 스케일해 주지 않는다)
    int height = 1080;
    VkPresentModeKHR present_mode = VK_PRESENT_MODE_FIFO_KHR;
    bool force_blit = false;           // --display-blit: STORAGE 가 가능해도 blit 폴백을 쓴다
    std::uint32_t extra_images = 0;    // minImageCount 에 더할 여유 (기본 0 = 최소로)
    // ⚠ present mode 별 minImageCount(FIFO 3)를 실제로 쓸 것인가. **기본 false 다.**
    //   실측(V3D, 검증 레이어 1.4.309): 기본 caps 는 4, FIFO 모드별로는 3 이다. 그런데 3 으로 만들면
    //   VUID-VkSwapchainCreateInfoKHR-presentMode-02839 가 error 로 걸린다 —
    //   "minImageCount 는 vkGetPhysicalDeviceSurfaceCapabilitiesKHR 의 minImageCount 이상이어야 한다"
    //   가 조건 없이 적용되고, VK_EXT_swapchain_maintenance1 의 VkSwapchainPresentModesCreateInfoEXT 를
    //   체인에 넣어도 레이어가 완화해 주지 않는다(VK_KHR_surface_maintenance1 은 이 드라이버·헤더에 없다).
    //   "검증 레이어 errors 0" 이 이 프로젝트의 하드 게이트이므로 **기본값은 안전한 4** 이고,
    //   이미지 1장(1080p RGBA 8.29 MB)을 아끼는 선택은 이 플래그로 명시적 opt-in 으로 남긴다.
    bool lean_images = false;
    bool verbose = false;
};

// present mode 별 이미지 개수 (VK_EXT_surface_maintenance1)
struct SurfaceModeCount {
    VkPresentModeKHR mode = VK_PRESENT_MODE_FIFO_KHR;
    std::uint32_t min_image_count = 0;
    std::uint32_t max_image_count = 0;
};

// P4(--probe surface) 가 그대로 덤프하는 surface 질의 결과
struct SurfaceInfo {
    VkBool32 wayland_presentation_support = VK_FALSE;  // vkGetPhysicalDeviceWaylandPresentationSupportKHR
    VkBool32 surface_support = VK_FALSE;               // vkGetPhysicalDeviceSurfaceSupportKHR
    std::uint32_t queue_family = 0;
    VkSurfaceCapabilitiesKHR caps{};
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> present_modes;
    std::vector<SurfaceModeCount> mode_counts;   // surface_maintenance1 이 없으면 비어 있다
    bool format_available = false;               // R8G8B8A8_UNORM + SRGB_NONLINEAR 가 목록에 있는가
    bool usage_storage = false;                  // supportedUsageFlags 에 STORAGE 가 있는가
    bool fifo_available = false;
    // vkGetPhysicalDeviceImageFormatProperties2(R8G8B8A8_UNORM, OPTIMAL, STORAGE|COLOR_ATTACHMENT|TRANSFER_DST)
    VkResult storage_image_result = VK_ERROR_FORMAT_NOT_SUPPORTED;
    VkImageFormatProperties storage_image_props{};
    bool blit_src_supported = false;             // 자체 이미지 → 스왑체인 blit 폴백 가능 여부
    bool blit_dst_supported = false;
};

// 한 프레임의 표시 대상. acquire() 가 채워 준다.
struct SwapchainFrame {
    bool valid = false;
    std::uint32_t image_index = 0;
    std::uint32_t sync_index = 0;          // 호출자의 슬롯별 자원(커맨드 버퍼·디스크립터)도 이 번호로 고른다
    VkImage swapchain_image = VK_NULL_HANDLE;
    // 컴퓨트가 실제로 쓰는 대상. storage 모드면 스왑체인 이미지, blit 모드면 자체 이미지다.
    VkImage target_image = VK_NULL_HANDLE;
    VkImageView target_view = VK_NULL_HANDLE;   // rgba8 storage image 뷰
    VkExtent2D extent{};
    VkSemaphore wait_semaphore = VK_NULL_HANDLE;    // 제출이 대기할 acquire 세마포어
    VkSemaphore signal_semaphore = VK_NULL_HANDLE;  // 제출이 신호할 render-finished 세마포어
    VkFence submit_fence = VK_NULL_HANDLE;          // 제출에 붙일 펜스 (스왑체인이 소유, unsignaled 상태)
    std::uint64_t present_id = 0;                   // present_wait 로 기다릴 id (0 = present_id 미사용)
};

class VulkanSwapchain {
public:
    // surface 를 만들고 caps·포맷·모드·이미지 개수·STORAGE 가능 여부를 질의한다. **스왑체인은 만들지 않는다**
    // (P4 는 여기까지만 필요하다). 실패하면 예외.
    VulkanSwapchain(yvo::vk::VulkanContext& ctx, const SwapchainOptions& opt);
    ~VulkanSwapchain();
    VulkanSwapchain(const VulkanSwapchain&) = delete;
    VulkanSwapchain& operator=(const VulkanSwapchain&) = delete;

    const SurfaceInfo& surface_info() const { return info_; }
    VkSurfaceKHR surface() const { return surface_; }

    // 스왑체인을 만든다(또는 새 크기로 다시 만든다). vkDeviceWaitIdle 을 쓰지 않는다.
    void create(int width, int height);
    // OUT_OF_DATE/SUBOPTIMAL 을 만났거나 창 크기가 바뀌었는가
    bool needs_recreate() const { return needs_recreate_; }
    void request_recreate(int width, int height);

    bool created() const { return cur_.handle != VK_NULL_HANDLE; }
    VkExtent2D extent() const { return cur_.extent; }
    std::uint32_t image_count() const { return static_cast<std::uint32_t>(cur_.images.size()); }
    bool blit_mode() const { return blit_; }
    VkFormat format() const { return format_.format; }

    // timeout = 0 으로 획득한다. 준비된 이미지가 없으면 false(display_dropped++) — **절대 블로킹하지 않는다.**
    bool acquire(SwapchainFrame* out);
    // 획득했지만 그리지 않기로 했다. 이미지를 되돌린다(또는 다음 acquire 가 재사용하도록 보관).
    // 반납(vkReleaseSwapchainImagesEXT)은 세마포어를 소비한 빈 제출의 펜스가 신호된 뒤로 **미뤄진다**
    // — spec 이 "이미지의 모든 사용이 실행을 완료해야 한다"고 요구하고, 호스트를 블로킹하지 않기 위해서다.
    void discard(const SwapchainFrame& f);

    // 컴퓨트 쓰기 전: 대상 이미지를 UNDEFINED → GENERAL 로 전환한다.
    // (스왑체인 이미지는 acquire 후 내용이 정의되지 않아도 되고 셰이더가 전 픽셀을 덮어쓰므로 UNDEFINED 가 가장 싸다)
    void record_acquire_transition(VkCommandBuffer cb, const SwapchainFrame& f) const;
    // 컴퓨트 쓰기 후: (blit 모드면 자체 이미지 → 스왑체인 이미지 복사 후) PRESENT_SRC 로 전환한다.
    void record_present_transition(VkCommandBuffer cb, const SwapchainFrame& f) const;

    // 제출이 끝난 뒤 호출. present_id + present fence 를 붙인다. OUT_OF_DATE/SUBOPTIMAL 이면 재생성을 예약한다.
    VkResult present(const SwapchainFrame& f);
    // vkWaitForPresentKHR: presentId 가 **실제로 화면에 나간** 시점까지 기다린다. present_wait 이 없으면 false.
    bool wait_present(std::uint64_t present_id, std::uint64_t timeout_ns);
    bool present_wait_available() const;

    // 재사용 대기 중인 구 스왑체인을 폐기할 수 있으면 폐기한다(펜스 폴링, 블로킹 없음). 매 프레임 호출.
    void collect_retired();

    // ── 통계 ──
    std::uint64_t presented() const { return presented_; }
    std::uint64_t display_dropped() const { return dropped_; }
    std::uint64_t acquire_not_ready() const { return not_ready_; }
    std::uint64_t sync_busy() const { return sync_busy_; }          // 인플라이트가 꽉 차서 버린 수
    std::uint64_t swapchain_recreates() const { return recreates_; }
    std::uint64_t retired_alive() const { return retired_.size(); }
    std::uint64_t images_released() const { return released_; }        // vkReleaseSwapchainImagesEXT 로 반납한 수
    std::size_t pending_releases() const { return pending_release_.size(); }

private:
    struct SyncSet {
        VkSemaphore acquire = VK_NULL_HANDLE;
        VkSemaphore render = VK_NULL_HANDLE;
        VkFence submit = VK_NULL_HANDLE;
        VkFence present = VK_NULL_HANDLE;
        bool in_flight = false;      // submit/present 펜스가 사용 중인가
        bool present_fenced = false; // present 펜스에 실제로 present 를 걸었는가
    };
    struct Instance {
        VkSwapchainKHR handle = VK_NULL_HANDLE;
        VkExtent2D extent{};
        VkPresentModeKHR mode = VK_PRESENT_MODE_FIFO_KHR;
        std::vector<VkImage> images;
        std::vector<VkImageView> views;       // storage 모드: 스왑체인 이미지의 rgba8 뷰
        std::vector<VkImage> own_images;      // blit 모드: 컴퓨트가 쓰는 자체 이미지
        std::vector<VkDeviceMemory> own_mems;
        std::vector<VkImageView> own_views;
        std::vector<SyncSet> sync;
        std::uint64_t next_present_id = 1;
    };

    void query_surface();
    void load_entry_points();
    void destroy_instance(Instance& in);
    bool instance_idle(const Instance& in) const;
    void create_own_images(Instance& in);
    void flush_pending_releases();
    void create_sync(Instance& in, std::uint32_t n);
    std::uint32_t choose_image_count(VkPresentModeKHR mode) const;

    yvo::vk::VulkanContext& ctx_;
    SwapchainOptions opt_;
    VkSurfaceKHR surface_ = VK_NULL_HANDLE;
    SurfaceInfo info_{};
    VkSurfaceFormatKHR format_{VK_FORMAT_R8G8B8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR};
    bool blit_ = false;
    bool blit_use_copy_ = false;   // blit 이 지원되지 않으면 vkCmdCopyImage 로 내려간다
    bool needs_recreate_ = false;
    int pending_w_ = 0, pending_h_ = 0;

    Instance cur_;
    std::vector<Instance> retired_;
    std::uint32_t next_sync_ = 0;

    // stash: swapchain_maintenance1 이 없을 때만 쓰는 안전망 (획득했지만 안 그린 이미지)
    SwapchainFrame stash_{};
    // discard 로 되돌릴 이미지: 펜스가 신호되면 반납한다(위 discard() 주석 참고)
    struct PendingRelease {
        std::uint32_t image_index = 0;
        VkFence fence = VK_NULL_HANDLE;   // 소유하지 않는다(현재 인스턴스의 sync 슬롯 펜스)
    };
    std::vector<PendingRelease> pending_release_;

    std::uint64_t presented_ = 0, dropped_ = 0, not_ready_ = 0, sync_busy_ = 0, recreates_ = 0, released_ = 0;

    // 확장 진입점 (로더 정적 export 에 의존하지 않는다)
    PFN_vkGetPhysicalDeviceSurfaceCapabilities2KHR p_get_caps2_ = nullptr;
    PFN_vkReleaseSwapchainImagesEXT p_release_images_ = nullptr;
    PFN_vkWaitForPresentKHR p_wait_present_ = nullptr;
};

const char* present_mode_name(VkPresentModeKHR m);
bool parse_present_mode(const std::string& s, VkPresentModeKHR* out);
const char* format_name(VkFormat f);
const char* color_space_name(VkColorSpaceKHR c);

} // namespace yvo::display
