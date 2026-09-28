#include "gstelem/gst_drm_frame.hpp"

#include <gst/allocators/allocators.h>
#include <gst/gst.h>
#include <gst/video/video.h>

#include <drm_fourcc.h>
#include <sys/types.h>
#include <unistd.h>

#include <cstdio>
#include <memory>

namespace yvo::gst {

namespace {

// SAND128 tile geometry in the RPi NV12_128C8 COLUMN tile-mode stride encoding:
// y_tiles counts 128-byte-wide tiles of 8 luma rows (Y) or 4 chroma rows (UV) per column.
constexpr std::uint32_t kYTileRows = 8;
constexpr std::uint32_t kUvTileRows = 4;
constexpr std::size_t kSandCol = 128;

bool reject(std::string* why, const std::string& msg) {
    if (why) *why = msg;
    return false;
}

// fd of a plane memory. GstDmaBufMemory is a GstFdMemory, so both getters return the same fd;
// is_dmabuf only reports which allocator type the memory advertises.
int memory_fd(GstMemory* mem, bool* is_dmabuf) {
    *is_dmabuf = gst_is_dmabuf_memory(mem) != FALSE;
    if (*is_dmabuf) return gst_dmabuf_memory_get_fd(mem);
    if (gst_is_fd_memory(mem)) return gst_fd_memory_get_fd(mem);
    return -1;
}

// True dma-buf object size. dma-buf only supports SEEK_END/SEEK_SET with offset 0; the file offset is restored.
std::size_t dmabuf_size(int fd) {
    const off_t end = lseek(fd, 0, SEEK_END);
    if (end < 0) return 0;
    (void)lseek(fd, 0, SEEK_SET);
    return static_cast<std::size_t>(end);
}

struct PlaneRef {
    int fd = -1;
    std::size_t fd_offset = 0; // plane start within the fd object = mem->offset + skip
    std::size_t obj_size = 0;  // lseek size of the fd object
};

bool find_plane(GstBuffer* buf, const GstVideoMeta* vmeta, guint plane, PlaneRef& out, std::string* why) {
    guint idx = 0, len = 0;
    gsize skip = 0;
    if (!gst_buffer_find_memory(buf, vmeta->offset[plane], 1, &idx, &len, &skip))
        return reject(why, "plane " + std::to_string(plane) + ": no memory covers VideoMeta offset " +
                               std::to_string(vmeta->offset[plane]));
    GstMemory* mem = gst_buffer_peek_memory(buf, idx);
    bool is_dmabuf = false;
    const int fd = memory_fd(mem, &is_dmabuf);
    if (fd < 0) return reject(why, "plane " + std::to_string(plane) + ": memory is neither dmabuf nor fd memory");
    out.fd = fd;
    out.fd_offset = static_cast<std::size_t>(mem->offset) + static_cast<std::size_t>(skip);
    out.obj_size = dmabuf_size(fd);
    if (out.obj_size == 0) return reject(why, "plane " + std::to_string(plane) + ": lseek(SEEK_END) on fd failed");
    return true;
}

} // namespace

bool sand_caps_from_caps(const GstCaps* caps, SandCaps& out, std::string* why) {
    if (!caps || gst_caps_get_size(caps) < 1) return reject(why, "empty caps");
    if (!gst_video_is_dma_drm_caps(caps)) return reject(why, "caps are not DMA_DRM");
    const GstStructure* s = gst_caps_get_structure(caps, 0);
    int w = 0, h = 0;
    if (!gst_structure_get_int(s, "width", &w) || !gst_structure_get_int(s, "height", &h))
        return reject(why, "caps without width/height");
    const gchar* drm = gst_structure_get_string(s, "drm-format");
    if (!drm) return reject(why, "caps without drm-format");
    guint64 modifier = 0;
    const guint32 fourcc = gst_video_dma_drm_fourcc_from_string(drm, &modifier);
    if (fourcc != DRM_FORMAT_NV12) return reject(why, std::string("drm-format is not NV12: ") + drm);
    if (fourcc_mod_broadcom_mod(modifier) != DRM_FORMAT_MOD_BROADCOM_SAND128)
        return reject(why, std::string("drm-format modifier is not Broadcom SAND128: ") + drm);
    if (w <= 0 || h <= 0 || (w % 2) || (h % 2)) return reject(why, "width/height must be positive and even");
    out.width = w;
    out.height = h;
    out.drm_fourcc = fourcc;
    out.drm_modifier = modifier;
    return true;
}

bool drm_frame_from_buffer(GstBuffer* buf, const SandCaps& caps, DrmFrame& out, std::string* why) {
    if (!buf) return reject(why, "null buffer");
    if (caps.drm_fourcc != DRM_FORMAT_NV12 || fourcc_mod_broadcom_mod(caps.drm_modifier) != DRM_FORMAT_MOD_BROADCOM_SAND128)
        return reject(why, "caps are not NV12 + Broadcom SAND128");
    const GstVideoMeta* vmeta = gst_buffer_get_video_meta(buf);
    if (!vmeta) return reject(why, "buffer has no GstVideoMeta");
    if (vmeta->n_planes != 2) return reject(why, "VideoMeta n_planes != 2");
    // VideoMeta carries the padded (coded) size, e.g. 1920x1088 for 1080p caps; it must cover the display size.
    if (static_cast<int>(vmeta->width) < caps.width || static_cast<int>(vmeta->height) < caps.height)
        return reject(why, "VideoMeta size smaller than caps size");

    PlaneRef y, uv;
    if (!find_plane(buf, vmeta, 0, y, why) || !find_plane(buf, vmeta, 1, uv, why)) return false;

    // Column heights from the tile-mode stride (x_tiles: 128-byte columns, y_tiles: 8/4-row tiles per column)
    const std::uint32_t y_xt = GST_VIDEO_TILE_X_TILES(vmeta->stride[0]);
    const std::uint32_t uv_xt = GST_VIDEO_TILE_X_TILES(vmeta->stride[1]);
    const std::uint32_t col_y = static_cast<std::uint32_t>(GST_VIDEO_TILE_Y_TILES(vmeta->stride[0])) * kYTileRows;
    const std::uint32_t col_uv = static_cast<std::uint32_t>(GST_VIDEO_TILE_Y_TILES(vmeta->stride[1])) * kUvTileRows;
    const std::size_t w = static_cast<std::size_t>(caps.width), h = static_cast<std::size_t>(caps.height);
    if (y_xt * kSandCol < w || uv_xt * kSandCol < w) return reject(why, "stride x_tiles do not cover the width");
    if (col_y < h || col_uv < h / 2) return reject(why, "column height (from stride y_tiles) smaller than plane height");

    // Maximum byte read per plane must lie inside the dma-buf object (same rule as drm_descriptor_to_frame)
    const std::size_t last_col = (w - 1) / kSandCol, last_x = (w - 1) % kSandCol;
    const std::size_t y_end = y.fd_offset + last_col * kSandCol * col_y + (h - 1) * kSandCol + last_x + 1;
    const std::size_t uv_end = uv.fd_offset + last_col * kSandCol * col_uv + (h / 2 - 1) * kSandCol + last_x + 1;
    if (y_end > y.obj_size) return reject(why, "Y plane exceeds its dma-buf");
    if (uv_end > uv.obj_size) return reject(why, "UV plane exceeds its dma-buf");

    DrmFrame f;
    f.fd = y.fd;
    f.size = y.obj_size;
    f.modifier = caps.drm_modifier;
    f.drm_format = caps.drm_fourcc;
    f.width = caps.width;
    f.height = caps.height;
    f.coded_height = static_cast<int>(vmeta->height); // padded height from VideoMeta (1080p: 1088)
    f.y_offset = y.fd_offset;
    f.y_pitch = col_y;                        // raw pitch field holds the column height (rpi ffmpeg convention)
    f.uv_offset = uv.fd_offset;
    f.uv_pitch = col_uv;
    f.sand_col_height = col_y;
    f.uv_fd = uv.fd;
    f.uv_size = uv.obj_size;
    f.uv_sand_col_height = col_uv;
    f.pts = GST_BUFFER_PTS_IS_VALID(buf) ? static_cast<std::int64_t>(GST_BUFFER_PTS(buf)) : -1;
    f.keepalive = std::shared_ptr<void>(gst_buffer_ref(buf), [](void* p) { gst_buffer_unref(static_cast<GstBuffer*>(p)); });
    out = std::move(f);
    return true;
}

std::string describe_buffer(GstBuffer* buf) {
    if (!buf) return "null buffer\n";
    std::string s;
    char line[320];
    const guint n = gst_buffer_n_memory(buf);
    std::snprintf(line, sizeof(line), "n_memory=%u buffer_size=%zu pts=%" GST_TIME_FORMAT "\n", n,
                  gst_buffer_get_size(buf), GST_TIME_ARGS(GST_BUFFER_PTS(buf)));
    s += line;
    for (guint i = 0; i < n; ++i) {
        GstMemory* mem = gst_buffer_peek_memory(buf, i);
        bool is_dmabuf = false;
        const int fd = memory_fd(mem, &is_dmabuf);
        const std::size_t lsz = fd >= 0 ? dmabuf_size(fd) : 0;
        std::snprintf(line, sizeof(line),
                      "  mem[%u]: allocator=%s is_dmabuf=%d is_fd=%d fd=%d offset=%zu size=%zu maxsize=%zu lseek_size=%zu\n",
                      i, mem->allocator ? mem->allocator->mem_type : "(none)", is_dmabuf ? 1 : 0,
                      gst_is_fd_memory(mem) ? 1 : 0, fd, static_cast<std::size_t>(mem->offset),
                      static_cast<std::size_t>(mem->size), static_cast<std::size_t>(mem->maxsize), lsz);
        s += line;
    }
    const GstVideoMeta* vmeta = gst_buffer_get_video_meta(buf);
    if (!vmeta) {
        s += "  VideoMeta: none\n";
        return s;
    }
    std::snprintf(line, sizeof(line), "  VideoMeta: format=%s %ux%u n_planes=%u\n",
                  gst_video_format_to_string(vmeta->format), vmeta->width, vmeta->height, vmeta->n_planes);
    s += line;
    const GstVideoFormatInfo* nv128 = gst_video_format_get_info(GST_VIDEO_FORMAT_NV12_128C8);
    for (guint p = 0; p < vmeta->n_planes && p < GST_VIDEO_MAX_PLANES; ++p) {
        guint idx = 0, len = 0;
        gsize skip = 0;
        const gboolean found = gst_buffer_find_memory(buf, vmeta->offset[p], 1, &idx, &len, &skip);
        const gint st = vmeta->stride[p];
        std::snprintf(line, sizeof(line),
                      "    plane[%u]: offset=%zu stride=0x%08x (x_tiles=%d y_tiles=%d) -> mem=%d skip=%zu"
                      " | NV12_128C8 tile_info: w=%u h=%u stride=%u size=%u\n",
                      p, static_cast<std::size_t>(vmeta->offset[p]), static_cast<unsigned>(st),
                      GST_VIDEO_TILE_X_TILES(st), GST_VIDEO_TILE_Y_TILES(st), found ? static_cast<int>(idx) : -1,
                      static_cast<std::size_t>(skip), nv128 ? nv128->tile_info[p].width : 0,
                      nv128 ? nv128->tile_info[p].height : 0, nv128 ? nv128->tile_info[p].stride : 0,
                      nv128 ? nv128->tile_info[p].size : 0);
        s += line;
    }
    return s;
}

} // namespace yvo::gst
