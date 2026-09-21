// 파이프라인 단계 간 인터페이스: ffmpeg(DRM_PRIME) → Vulkan 전처리 → HailoRT
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

namespace yvo {

// 디코더가 내보낸 HW 프레임 (dma-buf, Broadcom SAND128 NV12)
// 값은 AVDRMFrameDescriptor에서 채운다 (yvo_drm_dump로 확인한 Pi 5 rpi-hevc-dec 출력 기준).
// - Y와 UV는 서로 다른 dma-buf 객체다(nb_objects=2): Y는 fd/size, UV는 uv_fd/uv_size.
//   (단일 객체 레이아웃이면 uv_fd == fd, uv_offset은 같은 객체 안의 오프셋)
// - modifier 파라미터는 0이고, 각 plane의 pitch 필드에 그 plane의 SAND 열 높이(행 수)가 들어 있다.
// - plane별 SAND128 주소: 열 간격 = 128 * 열 높이
//     Y(x, y)   = Y객체 [y_offset  + (x/128) * 128 * sand_col_height    + y  * 128 + x%128]
//     UV(x, cy) = UV객체[uv_offset + (x/128) * 128 * uv_sand_col_height + cy * 128 + x%128]
//   1080p: sand_col_height=1088, uv_sand_col_height=544, 열 15개(마지막 열은 1920-1792=128px 꽉 참)
struct DrmFrame {
    int fd = -1;                  // Y plane dma-buf fd (소유하지 않음, keepalive가 살아 있는 동안만 유효)
    std::size_t size = 0;         // fd 객체 전체 크기(바이트)
    std::uint64_t modifier = 0;   // DRM format modifier (0x0700000000000004 = SAND128, 열 높이 파라미터 0)
    std::uint32_t drm_format = 0; // DRM fourcc (NV12)
    int width = 0;                // 표시 크기 (1920)
    int height = 0;               // 표시 크기 (1080)
    int coded_height = 0;         // 정렬된 높이 (1088)
    std::size_t y_offset = 0;     // Y plane 시작 오프셋 (fd 객체 안)
    std::size_t y_pitch = 0;      // Y plane pitch 필드 원값 (SAND: Y 열 높이 = 1088, 바이트 pitch 아님)
    std::size_t uv_offset = 0;    // UV plane 시작 오프셋 (uv_fd 객체 안)
    std::size_t uv_pitch = 0;     // UV plane pitch 필드 원값 (SAND: UV 열 높이 = 544)
    std::uint32_t sand_col_height = 0; // Y plane SAND 열 높이(행 수), 해석된 값
    int uv_fd = -1;                    // UV plane dma-buf fd (단일 객체면 fd와 같음)
    std::size_t uv_size = 0;           // uv_fd 객체 전체 크기(바이트)
    std::uint32_t uv_sand_col_height = 0; // UV plane SAND 열 높이(행 수), 해석된 값
    std::int64_t pts = 0;         // 프레임 순번 또는 pts
    std::shared_ptr<void> keepalive; // AVFrame 참조. GPU 사용이 끝날 때까지 보관
};

// letterbox 파라미터: 원본 좌표 = (letterbox 좌표 - pad) / scale
struct LetterboxParams {
    int dst_w = 640;
    int dst_h = 640;
    float scale = 1.0f;    // min(dst_w/src_w, dst_h/src_h)
    int pad_x = 0;         // 좌측 패딩(픽셀)
    int pad_y = 0;         // 상단 패딩(픽셀)
    std::uint8_t pad_value = 114;
};

// 640x640x3 UINT8 텐서의 바이트 배치. 두 레이아웃 모두 크기는 dst_w*dst_h*3이고 채널 순서는 R,G,B다.
//   Nhwc: packed RGB888     idx = (dy * dst_w + dx) * 3 + c   (HailoRT 기본 입력 형식)
//   Nhcw: 행마다 R dst_w개, G dst_w개, B dst_w개
//                           idx = (dy * 3 + c) * dst_w + dx   (yolov11m.hef의 장치 입력 형식)
// Nhcw로 넘기면 HailoRT의 CPU 입력 변환(약 14 ms/frame)이 사라진다. 단 변환이 없어지는 만큼
// HailoRT가 입력 버퍼를 직접 DMA 매핑하므로, V3D GEM mmap 포인터는 쓸 수 없고 dma-buf만 가능하다.
enum class TensorLayout {
    Nhwc,
    Nhcw,
};

// Vulkan 전처리 결과: 640x640x3 UINT8 텐서 (레이아웃은 layout 필드)
struct PreprocOutput {
    std::uint8_t* host_ptr = nullptr; // 1단계(ptr 모드): host-visible 매핑 포인터
    int dmabuf_fd = -1;               // 2단계(dmabuf 모드): export된 fd
    std::size_t size = 0;             // 640*640*3
    TensorLayout layout = TensorLayout::Nhwc; // 버퍼 안의 바이트 배치
    LetterboxParams lb;
    std::int64_t pts = 0;
    int slot = -1;                    // 링 버퍼 슬롯 번호 (HailoRT 완료 후 반납)
};

// 원본 좌표계 검출 결과
struct Detection {
    int class_id = 0;
    float score = 0.0f;
    float x1 = 0, y1 = 0, x2 = 0, y2 = 0; // 원본 프레임 픽셀 좌표
};

// 공통 규칙
// - letterbox: 비율 유지 bilinear 축소, 중앙 정렬, 패딩 114, BT.709 limited range YUV→RGB
// - 동기화: 디코더가 프레임을 반환하면 디코드 완료로 간주. Vulkan fence 완료 후 HailoRT에 전달.
//   슬롯 4개 링 버퍼, HailoRT 완료 콜백이 슬롯을 반납한다.
inline constexpr int kNumSlots = 4;

inline LetterboxParams make_letterbox(int src_w, int src_h, int dst_w = 640, int dst_h = 640) {
    LetterboxParams p;
    p.dst_w = dst_w;
    p.dst_h = dst_h;
    const float sx = static_cast<float>(dst_w) / static_cast<float>(src_w);
    const float sy = static_cast<float>(dst_h) / static_cast<float>(src_h);
    p.scale = sx < sy ? sx : sy;
    const int new_w = static_cast<int>(src_w * p.scale + 0.5f);
    const int new_h = static_cast<int>(src_h * p.scale + 0.5f);
    p.pad_x = (dst_w - new_w) / 2;
    p.pad_y = (dst_h - new_h) / 2;
    return p;
}

} // namespace yvo
