// BCM2712 PiSP Back End (ISP) hardware resize: SAND128 NV12 dma-buf → packed RGB888 letterbox
//
// Does the same job as the Vulkan path (vulkan/sand_letterbox.hpp) using the ISP block instead. The slot convention and names
// match so the two paths can be compared and swapped. The existing GPU path is unaffected (separate library yvo_isp).
//
// Pipeline (the RGB block order equals the field order of pisp_be_config):
//   input(SAND128 NV12) → ccm(BT.709 limited YUV→RGB inverse matrix) → crop(1920x1080)
//   → downscale(1920x1080 → 960x540) → resample(→ 640x360) → output1(packed RGB888)
//   - YUV→RGB is done in the leading ccm. The ycbcr_inverse block comes after sharpen, so it cannot convert at the input.
//   - The downscaler exists only on branch 1 (measured on BCM2712_D0). 1920→640 is 3x, so downscale is mandatory.
//   - letterbox is built with pisp_be_output_format_extra::offset_y. Every tile's write address moves down by pad_y rows,
//     so the ISP physically never touches the top/bottom padding areas. The output buffer is memset once to
//     pad_value at creation.
//
// Slot convention (uses the same slot numbers as SandLetterbox):
//   process(frame, slot) → PreprocOutput. BackendDevice::Run() is synchronous, so there is no submit/wait split.
//   After return the slot output buffer is owned by the caller (HailoRT); when done, process the same slot again.
//   DrmFrame::keepalive need not be held separately: by the time process returns the ISP has finished reading the input.
//   Not thread-safe.
//
// Constraints:
//   - The input uses the decoder dma-buf fd directly (no copy). Throws if y_offset / uv_offset is nonzero
//     (pisp_image_format_config has no way to express a plane offset).
//   - The output is packed RGB888 (NHWC) only. The ISP cannot produce NHCW.
//   - Wrap CPU reads of the output in OutputRead (dma-buf sync).
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "core/frame_types.hpp"

namespace yvo::isp {

struct IspResizerConfig {
    std::string media_device = "/dev/media0"; // pispbe media node (both /dev/media0 and /dev/media1 are valid)
    unsigned int branch = 1;                  // output branch. Use 1, which has the downscaler.
    int num_slots = kNumSlots;                // number of output ring slots
    // Buffers allocated per V4L2 node (= number of import slots). Must be >= num_slots.
    // May be smaller than the decoder pool (about 18 at 1080p) — libpisp reuses unused slots.
    // Measured with 12/20/24: no difference in ISP time (median 1.59 ms). Raising it only consumes more dma-heap.
    int buffer_count = 12;
    int src_w = 1920;                         // valid input width
    int src_h = 1080;                         // valid input height (crop target)
    int coded_height = 1088;                  // aligned height (equals the SAND column height)
    int sand_col_height = 1088;               // Y plane SAND column height (rows)
    int uv_sand_col_height = 544;             // UV plane SAND column height (rows)
    int dst_w = 640;
    int dst_h = 640;
    bool letterbox = true;                    // if false, output only the downscaled size without padding (e.g. 640x360)
    std::uint8_t pad_value = 114;
    std::string colour_space = "rec709";      // colour_encoding key in backend_default_config.json
    std::string resample_filter;              // empty = smart selection based on the downscale ratio
    bool open_device = true;                  // if false, only build the config without opening the device (describe only)
    std::string defaults_file = "/usr/share/libpisp/backend_default_config.json";
};

class IspResizer {
public:
    explicit IspResizer(const IspResizerConfig& cfg = {});
    ~IspResizer();
    IspResizer(const IspResizer&) = delete;
    IspResizer& operator=(const IspResizer&) = delete;

    // Synchronous processing: queue the input dma-buf → run the ISP → dequeue. Throws on failure.
    PreprocOutput process(const DrmFrame& frame, int slot);

    int num_slots() const;
    std::size_t output_size() const;   // slot output buffer size in bytes (stride * out_h)
    int out_width() const;
    int out_height() const;
    // Slot output dma-buf fd. Owned by IspResizer (do not close). Valid while IspResizer is alive.
    int dmabuf_fd(int slot) const;
    // CPU mapping of the slot output (cached). Wrap reads in OutputRead.
    std::uint8_t* host_ptr(int slot) const;
    const LetterboxParams& letterbox() const;
    // Run() interval time (ms) of the last process(). < 0 if none yet.
    double last_isp_ms() const;
    const IspResizerConfig& config() const;
    // Clear the input dma-buf cache (dups of decoder fds). Call before closing or reopening the decoder
    // (same convention as SandLetterbox::clear_import_cache()).
    void clear_input_cache();
    // variant, nodes, enable bits, tiles, downscale parameters, stride, sizeimage as a human-readable string.
    std::string describe() const;

    // RAII wrapper for cache synchronization (DMA_BUF_IOCTL_SYNC) while the CPU reads the output dma-buf.
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
