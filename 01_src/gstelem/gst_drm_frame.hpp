// GStreamer DMA_DRM (NV12 + Broadcom SAND128) GstBuffer → yvo::DrmFrame adapter (research path, stage 1)
//
// Source: v4l2slh265dec (gst-plugins-bad v4l2codecs, Raspberry Pi build) with src caps
//   video/x-raw(memory:DMABuf), format=DMA_DRM, drm-format=NV12:0x0700000000000004
// Buffer layout as delivered by that decoder (driver NV12MT_COL128, num_planes=2, VIDIOC_EXPBUF per plane):
//   - one GstMemory per plane (Y, UV), each backed by its own dma-buf fd
//   - GstVideoMeta is mandatory: offset[i] is the plane offset in the buffer-wide byte space (memories concatenated),
//     stride[i] = GST_VIDEO_TILE_MAKE_STRIDE(x_tiles, y_tiles) (RPi downstream NV12_128C8 COLUMN tile mode),
//     width/height = padded (coded) size (1080p caps: VideoMeta 1920x1088)
// Derivation (see gst_drm_frame.cpp for details):
//   plane i memory      = gst_buffer_find_memory(buf, offset[i], 1, &idx, &len, &skip)
//   plane i fd          = gst_dmabuf_memory_get_fd (or gst_fd_memory_get_fd)
//   plane i fd offset   = mem->offset + skip
//   Y  column height    = GST_VIDEO_TILE_Y_TILES(stride[0]) * 8   (1080p: 1088)
//   UV column height    = GST_VIDEO_TILE_Y_TILES(stride[1]) * 4   (1080p: 544)
//   object size         = lseek(fd, 0, SEEK_END) (true dma-buf size; GstMemory size may be smaller)
// The y_pitch/uv_pitch raw fields follow the rpi ffmpeg convention (= column height), so a GStreamer DrmFrame
// is field-for-field comparable with one from FfmpegDrmDecoder.
//
// GStreamer headers stay private to yvo_gstelem: this header only forward-declares the GStreamer types.
#pragma once

#include <cstdint>
#include <string>

#include "core/frame_types.hpp"

typedef struct _GstBuffer GstBuffer;
typedef struct _GstCaps GstCaps;

namespace yvo::gst {

// Scalars taken from the negotiated DMA_DRM caps (parse once per caps change, not per buffer).
struct SandCaps {
    int width = 0;                 // display size (vinfo width)
    int height = 0;                // display size (vinfo height)
    std::uint32_t drm_fourcc = 0;  // DRM_FORMAT_NV12 expected
    std::uint64_t drm_modifier = 0; // Broadcom SAND128 expected (param bits may be set)
};

// Parse DMA_DRM caps. Returns false with a reason if the caps are not DMA_DRM NV12 + Broadcom SAND128.
bool sand_caps_from_caps(const GstCaps* caps, SandCaps& out, std::string* why = nullptr);

// Fill a DrmFrame from a DMA_DRM (NV12 + SAND128) GstBuffer. On success out.keepalive holds a new ref on buf
// (released when the last DrmFrame copy dies); the fds are not owned and stay valid while keepalive is alive.
// out.pts = GST_BUFFER_PTS (ns, or -1 if none). Returns false with a reason for any unsupported layout
// (no VideoMeta, wrong format/modifier, plane memory not found, non-fd memory, layout outside the dma-buf...).
bool drm_frame_from_buffer(GstBuffer* buf, const SandCaps& caps, DrmFrame& out, std::string* why = nullptr);

// Human-readable dump of the buffer's memories and VideoMeta (n_memory, per-memory is_dmabuf/fd/size/offset/lseek size,
// VideoMeta planes, offsets, raw strides and decoded x/y tiles). For probes and anomaly reports.
std::string describe_buffer(GstBuffer* buf);

} // namespace yvo::gst
