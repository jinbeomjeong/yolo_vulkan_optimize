// libavcodec HW HEVC 디코더(V4L2 stateless, DRM_PRIME 출력) → DrmFrame
#pragma once

#include <cstdint>
#include <optional>
#include <string>

#include "common/frame_types.hpp"

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
        // 디코더 풀에 추가로 잡을 프레임 수. 사용자가 동시에 들고 있는 프레임(링 슬롯)만큼 필요하다.
        // 실측(1080p HEVC): 풀 크기 = 6 + 스레드 수(프레임 스레딩) + extra, 안전하게 들고 있을 수 있는 프레임 ≈ extra + 4.
        // 초과해서 들고 있으면 디코더가 빈 버퍼를 기다리며 멈춘다(에러 없이 행). -1이면 풀이 필요에 따라 늘어난다.
        int extra_hw_frames = 2 * kNumSlots;
        int threads = 0; // 0 = 자동(코어 수), 1 = 프레임 스레딩 끔
    };

    FfmpegDrmDecoder() = default;
    ~FfmpegDrmDecoder();
    FfmpegDrmDecoder(const FfmpegDrmDecoder&) = delete;
    FfmpegDrmDecoder& operator=(const FfmpegDrmDecoder&) = delete;

    bool open(const std::string& path) { return open(path, Options{}); }
    bool open(const std::string& path, const Options& opt);
    void close();

    // 다음 프레임. EOF 또는 에러면 nullopt (에러 여부는 error()로 구분).
    // 반환된 DrmFrame.keepalive가 살아 있는 동안 dma-buf 내용이 유지된다.
    std::optional<DrmFrame> next();

    bool eof() const { return eof_; }
    const std::string& error() const { return error_; }
    int width() const;
    int height() const;
    double frame_rate() const;      // 스트림 avg_frame_rate
    std::int64_t nb_frames() const; // 컨테이너가 알려 준 프레임 수(모르면 0)

    // keepalive에 담긴 AVFrame(DRM_PRIME)을 꺼낸다. 확인 도구용.
    static AVFrame* av_frame(const DrmFrame& f) { return static_cast<AVFrame*>(f.keepalive.get()); }

private:
    bool fail(const std::string& msg);
    bool fail_av(const char* what, int err);

    AVFormatContext* fmt_ = nullptr;
    AVCodecContext* codec_ = nullptr;
    AVBufferRef* hw_device_ = nullptr;
    AVPacket* pkt_ = nullptr;
    int stream_index_ = -1;
    bool input_done_ = false; // 패킷 끝(flush 전송 완료)
    bool eof_ = false;
    std::int64_t frame_counter_ = 0;
    std::string error_;
};

// AVDRMFrameDescriptor → DrmFrame 필드(keepalive/pts 제외). Y/UV가 한 객체든 두 객체든 처리한다.
// 지원하지 않는 레이아웃(SAND128 NV12가 아님, 크기 불일치)이면 false와 사유를 반환한다.
bool drm_descriptor_to_frame(const AVDRMFrameDescriptor& desc, int width, int height, int coded_height,
                             DrmFrame& out, std::string* why = nullptr);

// descriptor 원값을 사람이 읽을 문자열로 (로그·확인 도구용)
std::string describe_drm_descriptor(const AVDRMFrameDescriptor& desc);

} // namespace yvo
