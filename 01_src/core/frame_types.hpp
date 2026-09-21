// Interfaces between pipeline stages: ffmpeg(DRM_PRIME) → Vulkan preprocessing → HailoRT
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

namespace yvo {

// HW frame exported by the decoder (dma-buf, Broadcom SAND128 NV12)
// Values are filled from AVDRMFrameDescriptor (based on Pi 5 rpi-hevc-dec output confirmed with yvo_drm_dump).
// - Y and UV are separate dma-buf objects (nb_objects=2): Y is fd/size, UV is uv_fd/uv_size.
//   (for a single-object layout, uv_fd == fd and uv_offset is the offset within the same object)
// - The modifier parameter is 0, and each plane's pitch field holds that plane's SAND column height (row count).
// - Per-plane SAND128 address: column stride = 128 * column height
//     Y(x, y)   = Yobject [y_offset  + (x/128) * 128 * sand_col_height    + y  * 128 + x%128]
//     UV(x, cy) = UVobject[uv_offset + (x/128) * 128 * uv_sand_col_height + cy * 128 + x%128]
//   1080p: sand_col_height=1088, uv_sand_col_height=544, 15 columns (the last column is fully used, 1920-1792=128px)
struct DrmFrame {
    int fd = -1;                  // Y plane dma-buf fd (not owned, valid only while keepalive is alive)
    std::size_t size = 0;         // total size of the fd object (bytes)
    std::uint64_t modifier = 0;   // DRM format modifier (0x0700000000000004 = SAND128, column height parameter 0)
    std::uint32_t drm_format = 0; // DRM fourcc (NV12)
    int width = 0;                // display size (1920)
    int height = 0;               // display size (1080)
    int coded_height = 0;         // aligned height (1088)
    std::size_t y_offset = 0;     // Y plane start offset (within the fd object)
    std::size_t y_pitch = 0;      // raw Y plane pitch field (SAND: Y column height = 1088, not a byte pitch)
    std::size_t uv_offset = 0;    // UV plane start offset (within the uv_fd object)
    std::size_t uv_pitch = 0;     // raw UV plane pitch field (SAND: UV column height = 544)
    std::uint32_t sand_col_height = 0; // Y plane SAND column height (row count), interpreted value
    int uv_fd = -1;                    // UV plane dma-buf fd (same as fd for a single object)
    std::size_t uv_size = 0;           // total size of the uv_fd object (bytes)
    std::uint32_t uv_sand_col_height = 0; // UV plane SAND column height (row count), interpreted value
    std::int64_t pts = 0;         // frame sequence number or pts
    std::shared_ptr<void> keepalive; // AVFrame reference. Kept until GPU use is finished
};

// letterbox parameters: source coordinate = (letterbox coordinate - pad) / scale
struct LetterboxParams {
    int dst_w = 640;
    int dst_h = 640;
    float scale = 1.0f;    // min(dst_w/src_w, dst_h/src_h)
    int pad_x = 0;         // left padding (pixels)
    int pad_y = 0;         // top padding (pixels)
    std::uint8_t pad_value = 114;
};

// Byte layout of a 640x640x3 UINT8 tensor. Both layouts have size dst_w*dst_h*3 and channel order R,G,B.
//   Nhwc: packed RGB888     idx = (dy * dst_w + dx) * 3 + c   (HailoRT default input format)
//   Nhcw: per row, dst_w R values, dst_w G values, dst_w B values
//                           idx = (dy * 3 + c) * dst_w + dx   (device input format of yolov11m.hef)
// Passing Nhcw removes HailoRT's CPU input conversion (about 14 ms/frame). However, since the conversion is gone,
// HailoRT DMA-maps the input buffer directly, so a V3D GEM mmap pointer cannot be used; only a dma-buf works.
enum class TensorLayout {
    Nhwc,
    Nhcw,
};

// Vulkan preprocessing result: 640x640x3 UINT8 tensor (layout given by the layout field)
struct PreprocOutput {
    std::uint8_t* host_ptr = nullptr; // stage 1 (ptr mode): host-visible mapped pointer
    int dmabuf_fd = -1;               // stage 2 (dmabuf mode): exported fd
    std::size_t size = 0;             // 640*640*3
    TensorLayout layout = TensorLayout::Nhwc; // byte layout inside the buffer
    LetterboxParams lb;
    std::int64_t pts = 0;
    int slot = -1;                    // ring buffer slot index (returned after HailoRT completes)
};

// Detection result in source coordinates
struct Detection {
    int class_id = 0;
    float score = 0.0f;
    float x1 = 0, y1 = 0, x2 = 0, y2 = 0; // source frame pixel coordinates
};

// Common rules
// - letterbox: aspect-preserving bilinear downscale, centered, padding 114, BT.709 limited range YUV→RGB
// - Synchronization: decode is considered complete when the decoder returns a frame. Handed to HailoRT after the Vulkan fence completes.
//   4-slot ring buffer; the HailoRT completion callback returns the slot.
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
