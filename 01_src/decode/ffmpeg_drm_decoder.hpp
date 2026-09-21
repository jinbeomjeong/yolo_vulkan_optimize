// libavcodec HW HEVC decoder (V4L2 stateless, DRM_PRIME output) → DrmFrame
#pragma once

#include <cstdint>
#include <optional>
#include <string>

#include "core/frame_types.hpp"

struct AVFormatContext;
struct AVCodecContext;
struct AVBufferRef;
struct AVPacket;
struct AVFrame;
struct AVDRMFrameDescriptor;

namespace yvo {

class FfmpegDrmDecoder {
public:
    struct Options {
        // Extra frames to allocate in the decoder pool. Needed for as many frames as the user holds at once (ring slots).
        // Measured (1080p HEVC): pool size = 6 + thread count (frame threading) + extra; frames that can safely be held ≈ extra + 4.
        // Holding more than that makes the decoder stall waiting for a free buffer (hangs without an error). -1 lets the pool grow as needed.
        int extra_hw_frames = 2 * kNumSlots;
        int threads = 0; // 0 = auto (core count), 1 = frame threading off
    };

    FfmpegDrmDecoder() = default;
    ~FfmpegDrmDecoder();
    FfmpegDrmDecoder(const FfmpegDrmDecoder&) = delete;
    FfmpegDrmDecoder& operator=(const FfmpegDrmDecoder&) = delete;

    bool open(const std::string& path) { return open(path, Options{}); }
    bool open(const std::string& path, const Options& opt);
    void close();

    // Next frame. nullopt on EOF or error (distinguish with error()).
    // The dma-buf contents stay valid while the returned DrmFrame.keepalive is alive.
    std::optional<DrmFrame> next();

    bool eof() const { return eof_; }
    const std::string& error() const { return error_; }
    int width() const;
    int height() const;
    double frame_rate() const;      // stream avg_frame_rate
    std::int64_t nb_frames() const; // frame count reported by the container (0 if unknown)

    // Extract the AVFrame (DRM_PRIME) held in keepalive. For inspection tools.
    static AVFrame* av_frame(const DrmFrame& f) { return static_cast<AVFrame*>(f.keepalive.get()); }

private:
    bool fail(const std::string& msg);
    bool fail_av(const char* what, int err);

    AVFormatContext* fmt_ = nullptr;
    AVCodecContext* codec_ = nullptr;
    AVBufferRef* hw_device_ = nullptr;
    AVPacket* pkt_ = nullptr;
    int stream_index_ = -1;
    bool input_done_ = false; // end of packets (flush sent)
    bool eof_ = false;
    std::int64_t frame_counter_ = 0;
    std::string error_;
};

// AVDRMFrameDescriptor → DrmFrame fields (except keepalive/pts). Handles Y/UV in one object or two.
// For an unsupported layout (not SAND128 NV12, size mismatch) returns false with the reason.
bool drm_descriptor_to_frame(const AVDRMFrameDescriptor& desc, int width, int height, int coded_height,
                             DrmFrame& out, std::string* why = nullptr);

// Raw descriptor values as a human-readable string (for logs and inspection tools)
std::string describe_drm_descriptor(const AVDRMFrameDescriptor& desc);

} // namespace yvo
