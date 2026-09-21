// BCM2712 PiSP Back End(ISP) 하드웨어 리사이즈: SAND128 NV12 dma-buf → packed RGB888 letterbox
//
// Vulkan 경로(vulkan/sand_letterbox.hpp)와 같은 일을 ISP 블록으로 대신한다. 두 경로를 비교·교체할 수 있도록
// 슬롯 규약과 이름을 맞췄다. 기존 GPU 경로에는 영향이 없다(별도 라이브러리 yvo_isp).
//
// 파이프라인(RGB 블록 순서는 pisp_be_config의 필드 순서와 같다):
//   input(SAND128 NV12) → ccm(BT.709 limited YUV→RGB 역행렬) → crop(1920x1080)
//   → downscale(1920x1080 → 960x540) → resample(→ 640x360) → output1(packed RGB888)
//   - YUV→RGB는 맨 앞 ccm에서 한다. ycbcr_inverse 블록은 sharpen 뒤라 입력단 변환에 쓸 수 없다.
//   - 다운스케일러는 branch 1에만 있다(BCM2712_D0 실측). 1920→640은 3배라 downscale이 필수다.
//   - letterbox는 pisp_be_output_format_extra::offset_y로 만든다. 모든 타일의 쓰기 주소가 pad_y행만큼
//     내려가므로 위/아래 패딩 영역은 ISP가 물리적으로 건드리지 않는다. 출력 버퍼는 만들 때 한 번
//     pad_value로 memset한다.
//
// 슬롯 규약 (SandLetterbox와 같은 슬롯 번호를 쓴다):
//   process(frame, slot) → PreprocOutput. BackendDevice::Run()이 동기라 submit/wait 분리는 없다.
//   슬롯 출력 버퍼는 반환 후 호출자(HailoRT) 소유이며, 다 쓴 뒤 같은 슬롯을 다시 process한다.
//   DrmFrame::keepalive는 process가 반환하는 시점에 ISP가 입력을 다 읽은 상태이므로 따로 잡지 않는다.
//   스레드 안전하지 않다.
//
// 제약:
//   - 입력은 디코더 dma-buf fd를 그대로 쓴다(복사 없음). y_offset / uv_offset이 0이 아니면 예외
//     (pisp_image_format_config에 plane 오프셋을 표현할 수단이 없다).
//   - 출력은 packed RGB888(NHWC)뿐이다. ISP는 NHCW를 만들 수 없다.
//   - CPU가 출력을 읽을 때는 OutputRead(dma-buf sync)로 감싼다.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "core/frame_types.hpp"

namespace yvo::isp {

struct IspResizerConfig {
    std::string media_device = "/dev/media0"; // pispbe media 노드 (/dev/media0, /dev/media1 둘 다 유효)
    unsigned int branch = 1;                  // 출력 branch. 다운스케일러가 있는 1을 쓴다.
    int num_slots = kNumSlots;                // 출력 링 슬롯 수
    // V4L2 노드마다 잡는 버퍼 수(= import 슬롯 수). num_slots 이상이어야 한다.
    // 디코더 풀(1080p에서 약 18개)보다 작아도 된다 — libpisp이 쓰이지 않는 슬롯을 재사용한다.
    // 12/20/24로 재 봤을 때 ISP 시간 차이는 없었다(median 1.59 ms). 늘리면 dma-heap만 더 먹는다.
    int buffer_count = 12;
    int src_w = 1920;                         // 유효 입력 폭
    int src_h = 1080;                         // 유효 입력 높이 (crop 대상)
    int coded_height = 1088;                  // 정렬된 높이 (SAND 열 높이와 같다)
    int sand_col_height = 1088;               // Y plane SAND 열 높이(행 수)
    int uv_sand_col_height = 544;             // UV plane SAND 열 높이(행 수)
    int dst_w = 640;
    int dst_h = 640;
    bool letterbox = true;                    // false면 패딩 없이 축소 결과 크기(예: 640x360)로만 출력
    std::uint8_t pad_value = 114;
    std::string colour_space = "rec709";      // backend_default_config.json의 colour_encoding 키
    std::string resample_filter;              // 비우면 축소율 기반 smart selection
    bool open_device = true;                  // false면 설정만 만들고 장치를 열지 않는다(describe 전용)
    std::string defaults_file = "/usr/share/libpisp/backend_default_config.json";
};

class IspResizer {
public:
    explicit IspResizer(const IspResizerConfig& cfg = {});
    ~IspResizer();
    IspResizer(const IspResizer&) = delete;
    IspResizer& operator=(const IspResizer&) = delete;

    // 동기 처리: 입력 dma-buf를 queue → ISP 실행 → dequeue. 실패하면 예외.
    PreprocOutput process(const DrmFrame& frame, int slot);

    int num_slots() const;
    std::size_t output_size() const;   // 슬롯 출력 버퍼 바이트 수 (stride * out_h)
    int out_width() const;
    int out_height() const;
    // 슬롯 출력 dma-buf fd. 소유권은 IspResizer에 있다(닫지 말 것). IspResizer가 살아 있는 동안 유효.
    int dmabuf_fd(int slot) const;
    // 슬롯 출력의 CPU 매핑(캐시됨). 읽기 전에 OutputRead로 감쌀 것.
    std::uint8_t* host_ptr(int slot) const;
    const LetterboxParams& letterbox() const;
    // 마지막 process()의 Run() 구간 시간(ms). 아직 없으면 < 0.
    double last_isp_ms() const;
    const IspResizerConfig& config() const;
    // 입력 dma-buf 캐시(디코더 fd의 dup)를 비운다. 디코더를 닫거나 다시 열기 전에 호출할 것
    // (SandLetterbox::clear_import_cache()와 같은 규약).
    void clear_input_cache();
    // variant, 노드, enable 비트, 타일, 축소 파라미터, stride, sizeimage를 사람이 읽는 문자열로.
    std::string describe() const;

    // 출력 dma-buf를 CPU로 읽는 동안의 캐시 동기화(DMA_BUF_IOCTL_SYNC) RAII 래퍼.
    class OutputRead {
    public:
        OutputRead(const IspResizer& r, int slot);
        ~OutputRead();
        OutputRead(const OutputRead&) = delete;
        OutputRead& operator=(const OutputRead&) = delete;
        const std::uint8_t* data() const;
        std::size_t size() const;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace yvo::isp
