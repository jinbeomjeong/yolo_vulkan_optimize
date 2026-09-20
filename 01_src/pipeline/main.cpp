// yvo_pipeline: HEVC HW 디코드(DRM_PRIME) → Vulkan SAND128 letterbox → HailoRT yolov11m 추론
//
// 스레드 구조
//   - 메인 스레드: 디코드 → 빈 슬롯 확보 → GPU submit → (펜스 완료된 슬롯) wait → HailoRT submit.
//   - HailoRT 콜백 스레드: 결과를 완료 큐에 넣고 슬롯을 free 목록에 반납(mutex + condvar).
//   - 결과 기록(JSON Lines, PNG, 통계)은 메인 스레드가 pts 순서로 한다.
// 슬롯 규약 (GPU·NPU·표시 공용 슬롯 번호, 0..slots-1) — **반납 조건이 둘이다**
//   free → [GPU submit] → gpu 대기열 → [SandLetterbox::wait: 펜스 완료, AVFrame keepalive 해제]
//        → [HailoRunner::submit] → [HailoRT 완료 콜백 = compute_done]
//        → [DisplayPipeline 표시 완료 = display_done]  → 둘 다 서면 free
//   --display off 면 display 조건이 아예 없다(SlotExchange(need_display=false)) → 오늘과 완전히 같다.
//   ⚠ display_done 을 세우는 주체는 **메인 스레드 자신**이다. 그래서 빈 슬롯을 기다릴 때 블로킹하면
//     교착한다 — 아래 슬롯 확보 루프는 acquire_poll + pump 로만 돌린다(acquire_wait 를 쓰지 않는다).
//   GPU에 동시에 걸린 프레임은 --gpu-depth(기본 3) 이하. 디코더는 약 13프레임 이상 들고 있으면
//   **에러 없이 조용히 멈춘다** → 시작 시 slots+gpu_depth <= 10 을 검사하고, 런타임에 실제 보유 수
//   (held_frames_max)를 계측해 CSV 로 낸다. 이것이 유일한 감지 수단이다.
// 모드
//   ptr    : HailoRT set_buffer(MemoryView(V3D 출력 매핑 포인터))
//   dmabuf : 초기화 때 슬롯별 출력 메모리를 dma-buf fd로 export → HailoRunner::map_dmabuf 사전 매핑
//            → PreprocOutput.dmabuf_fd로 set_dma_buffer
// 표시 (--display, YVO_HAVE_DISPLAY 로 조건부 컴파일)
//   off  : 오늘까지의 경로. DisplayPipeline 을 만들지도 않는다(기본값).
//   on   : 디코더 SAND dma-buf → display_convert.comp → overlay_boxes.comp → 스왑체인 present.
//          matched = 그 프레임의 검출과 함께 표시(flush_results 에서 show),
//          latest  = 전처리가 끝나는 대로 표시하고 검출은 가장 최근 것을 얹는다(finish_gpu 에서 show).
// 후처리 (--postproc)
//   host : HEF 내장 NMS 출력 1개 → HailoRT가 호스트 CPU에서 파싱 (기준 경로, 기본값)
//   gpu  : 원시 출력 6스트림(dma-buf) → yolo_decode.comp pass 1 → CPU greedy 억제.
//          NPU 완료분을 GPU로 미는 일도 **메인 스레드**가 한다(InferStage::pump/drain_one) —
//          VkQueue가 하나뿐이라 vkQueueSubmit을 다른 스레드에서 하면 안 되기 때문이다.
// 종료 순서: InferStage::drain → flush_results → DisplayPipeline::drain → SandLetterbox::clear_import_cache
//            → 디코더 close → 단계 해제(러너가 dma-buf unmap) → 표시 해제(스왑체인·surface)
//            → export fd close → VulkanContext → WaylandWindow
//            (창은 컨텍스트보다 늦게 죽어야 하므로 컨텍스트보다 먼저 선언한다)
// 시그널: SIGINT/SIGTERM 은 sigaction(SA_RESTART) 로 받아 **atomic 플래그만** 세운다.
//         메인 루프·표시 페이싱 루프·슬롯 확보 루프가 매 바퀴 이 플래그를 보고 빠져나와
//         위의 **종료 순서를 그대로** 탄다 → 통계 요약이 정상 종료와 같은 형식으로 나온다.
//         요약 머리에 [interrupted] 를 덧붙이고 RESULT_CSV/RESULT_CSV2 의 ok 는 0,
//         종료 코드는 128+signo(SIGINT 130 / SIGTERM 143). 두 번째 시그널은 _exit(130).
#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <sys/resource.h>
#include <unistd.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "common/coco_labels.hpp"
#include "common/cpu_letterbox.hpp"
#include "hailo/hailo_runner.hpp"
#include "pipeline/ffmpeg_drm_decoder.hpp"
#include "pipeline/gpu_postproc_stage.hpp"
#include "pipeline/infer_stage.hpp"
#include "vulkan/sand_letterbox.hpp"
#include "vulkan/vulkan_context.hpp"

#ifdef YVO_HAVE_DISPLAY
#include "display/display_pipeline.hpp"
#include "common/stats.hpp"
#include "common/tool_util.hpp"

using yvo::Stats;
using yvo::stats;
#endif

namespace {

using yvo::tool::soc_temperature;

using Clock = std::chrono::steady_clock;

// ── SIGINT/SIGTERM 정상 종료 ────────────────────────────────────────────────────────────
// 핸들러가 하는 일은 atomic 플래그를 세우는 것이 **전부**다(async-signal-safe: printf·malloc 금지).
// 실제 종료는 메인 스레드가 플래그를 보고 기존 종료 경로를 그대로 타면서 한다.
std::atomic<bool> g_stop{false};
std::atomic<int> g_stop_signal{0};
static_assert(std::atomic<bool>::is_always_lock_free,
              "종료 플래그는 lock-free 여야 한다(시그널 핸들러 안에서 쓴다)");
static_assert(std::atomic<int>::is_always_lock_free,
              "시그널 번호 저장도 lock-free 여야 한다(시그널 핸들러 안에서 쓴다)");

void on_stop_signal(int sig) {
    // exchange 가 true 를 돌려주면 두 번째 시그널이다 → 드레인을 기다리지 않고 즉시 끊는다.
    if (g_stop.exchange(true, std::memory_order_relaxed)) _exit(130);
    g_stop_signal.store(sig, std::memory_order_relaxed);
}

bool stop_requested() { return g_stop.load(std::memory_order_relaxed); }

// SA_RESTART 를 켠다: 블로킹 시스템 콜이 EINTR 로 깨지면 안 된다(디코더의 av_read_frame).
// 설치 실패는 치명적이지 않다 — 경고만 내고 계속한다(시그널이 기본 동작으로 떨어질 뿐이다).
bool install_stop_handlers() {
    struct sigaction sa;
    std::memset(&sa, 0, sizeof(sa));
    sa.sa_handler = on_stop_signal;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_RESTART;
    bool ok = true;
    for (const int sig : {SIGINT, SIGTERM}) {
        if (sigaction(sig, &sa, nullptr) != 0) {
            std::fprintf(stderr, "[warn] sigaction(%d) 실패: %s — 이 시그널은 기본 동작으로 떨어진다\n", sig,
                         std::strerror(errno));
            ok = false;
        }
    }
    return ok;
}

// COCO 라벨·클래스 색은 common/coco_labels.hpp 에 하나로 모았다(GPU 오버레이와 공유한다).
using yvo::class_name;

enum class Mode { Ptr, Dmabuf };

// 후처리를 누가 하는가.
//   Host: HEF 내장 NMS 출력 1개 → HailoRT 가 호스트 CPU 에서 파싱 (오늘의 기준 경로)
//   Gpu : 원시 출력 6스트림 → GPU pass 1(yolo_decode.comp) + CPU greedy 억제
enum class Postproc { Host, Gpu };

struct Args {
    std::string video = "04_data/videos/video_h265.mp4";
    std::string hef = "03_models/yolov11m.hef";
    long frames = 0;  // 0 = 전체
    Mode mode = Mode::Ptr;
    // 전처리 출력·NPU 입력 텐서 레이아웃. Nhcw면 HailoRT의 CPU 입력 변환이 사라진다(dmabuf 모드 전용).
    yvo::TensorLayout layout = yvo::TensorLayout::Nhwc;
    int batch = 8;
    int slots = 0;      // 0 = batch*2 (최소 4)
    int gpu_depth = 3;  // GPU에 동시에 걸어 둘 프레임 수
    float score = -1.0f;
    Postproc postproc = Postproc::Host;
    // GPU 후처리의 원시 출력 버퍼 할당 방식. 기본 export(= Vulkan 이 할당·export).
    // heap(dma-heap import)은 3자 일관성(PCIe DMA ↔ CPU 캐시 ↔ V3D)이 아직 미검증이다.
    std::string postproc_alloc = "export";
    std::string dump_dir;
    int dump_every = 100;
    bool quiet = false;
    std::string csv_label;  // 지정하면 마지막에 CSV 한 줄(RESULT_CSV,...) 출력

    // ── 표시 (기본값은 전부 "끔": 인자 없이 돌리면 오늘까지와 같은 경로다) ──
    bool display = false;
#ifdef YVO_HAVE_DISPLAY
    yvo::display::DisplayPipelineConfig disp;
    bool pace_explicit = false;   // --display-pace 를 사용자가 직접 줬는가
#endif
};

void usage(const char* a0) {
    std::fprintf(stderr,
                 "usage: %s [--video PATH] [--hef PATH] [--frames N] [--mode ptr|dmabuf] [--layout nhwc|nhcw]\n"
                 "          [--batch N] [--slots N] [--postproc host|gpu] [--postproc-alloc export|heap]\n"
                 "          [--gpu-depth N] [--score T] [--dump-dir DIR] [--dump-every K] [--quiet] [--csv-label L]\n"
                 "  --layout     전처리 출력·NPU 입력 텐서 레이아웃 (기본 nhwc)\n"
                 "               nhcw는 HailoRT의 CPU 입력 변환(약 14 ms/frame)을 없앤다. --mode dmabuf 전용\n"
                 "  --slots      GPU·NPU 공용 슬롯 수 (기본 max(4, batch*2))\n"
                 "  --gpu-depth  GPU에 동시에 걸린 프레임 수 상한 (기본 3, 2~4 권장: 디코더 버퍼 한도)\n"
                 "  --postproc   후처리 위치 (기본 host). gpu는 원시 출력 6스트림 + yolo_decode.comp + CPU 억제\n"
                 "               --mode dmabuf --layout nhcw 전용\n"
                 "  --postproc-alloc  gpu 후처리의 원시 출력 버퍼 할당 (기본 export)\n"
                 "  --dump-dir   detections.jsonl(프레임당 한 줄) + K 프레임마다 박스 PNG(letterbox 입력 위)\n"
#ifdef YVO_HAVE_DISPLAY
                 "  --display {off,on}                       HDMI 실시간 표시 (기본 off)\n"
                 "  --display-backend {auto,wayland,offscreen}  기본 auto. wayland 는 strict(조용한 폴백 금지)\n"
                 "  --display-sync {matched,latest}          기본 matched (그 프레임의 검출과 함께 표시)\n"
                 "  --display-pace {source,asap}             표시 on 이면 기본 source. asap 는 페이싱 없음\n"
                 "  --display-fullscreen | --display-size WxH | --display-every K\n"
                 "  --present-mode {fifo,mailbox,immediate}  기본 fifo\n"
                 "  --display-blit                           STORAGE 가 가능해도 blit 폴백을 강제한다\n"
                 "  --overlay-line-width N | --overlay-text-scale N   기본 2 / 2\n"
                 "  ⚠ 측정은 반드시 --display-fullscreen 으로 한다. 창 모드는 균일 스케일 경로로 넘어가\n"
                 "    에러도 경고도 없이 표시 변환이 2.6배(4.25 → 10.9 ms) 느려진다\n"
#endif
                 ,
                 a0);
}

bool parse(int argc, char** argv, Args& a) {
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        auto next = [&]() -> const char* { return i + 1 < argc ? argv[++i] : nullptr; };
        const char* v = nullptr;
        if (s == "--video" && (v = next())) a.video = v;
        else if (s == "--hef" && (v = next())) a.hef = v;
        else if (s == "--frames" && (v = next())) a.frames = std::atol(v);
        else if (s == "--mode" && (v = next())) {
            const std::string m = v;
            if (m == "ptr") a.mode = Mode::Ptr;
            else if (m == "dmabuf") a.mode = Mode::Dmabuf;
            else return false;
        } else if (s == "--layout" && (v = next())) {
            const std::string l = v;
            if (l == "nhwc") a.layout = yvo::TensorLayout::Nhwc;
            else if (l == "nhcw") a.layout = yvo::TensorLayout::Nhcw;
            else return false;
        } else if (s == "--batch" && (v = next())) a.batch = std::atoi(v);
        else if (s == "--slots" && (v = next())) a.slots = std::atoi(v);
        else if (s == "--gpu-depth" && (v = next())) a.gpu_depth = std::atoi(v);
        else if (s == "--postproc" && (v = next())) {
            const std::string p = v;
            if (p == "host") a.postproc = Postproc::Host;
            else if (p == "gpu") a.postproc = Postproc::Gpu;
            else return false;
        } else if (s == "--postproc-alloc" && (v = next())) {
            const std::string p = v;
            if (p != "export" && p != "heap") return false;
            a.postproc_alloc = p;
        }
        else if (s == "--score" && (v = next())) a.score = static_cast<float>(std::atof(v));
        else if (s == "--dump-dir" && (v = next())) a.dump_dir = v;
        else if (s == "--dump-every" && (v = next())) a.dump_every = std::atoi(v);
        else if (s == "--csv-label" && (v = next())) a.csv_label = v;
        else if (s == "--quiet") a.quiet = true;
#ifdef YVO_HAVE_DISPLAY
        else if (s == "--display" && (v = next())) {
            const std::string d = v;
            if (d == "on") a.display = true;
            else if (d == "off") a.display = false;
            else return false;
        } else if (s == "--display-backend" && (v = next())) {
            if (!yvo::display::parse_backend_kind(v, &a.disp.backend)) return false;
        } else if (s == "--display-sync" && (v = next())) {
            if (!yvo::display::parse_display_sync(v, &a.disp.sync)) return false;
        } else if (s == "--display-pace" && (v = next())) {
            if (!yvo::display::parse_display_pace(v, &a.disp.pace)) return false;
            a.pace_explicit = true;
        } else if (s == "--display-fullscreen") a.disp.fullscreen = true;
        else if (s == "--display-size" && (v = next())) {
            int w = 0, h = 0;
            if (std::sscanf(v, "%dx%d", &w, &h) != 2 || w <= 0 || h <= 0) return false;
            a.disp.width = w;
            a.disp.height = h;
        } else if (s == "--display-every" && (v = next())) a.disp.every = std::atoi(v);
        else if (s == "--present-mode" && (v = next())) {
            if (!yvo::display::parse_present_mode(v, &a.disp.present_mode)) return false;
        } else if (s == "--display-blit") a.disp.force_blit = true;
        else if (s == "--overlay-line-width" && (v = next())) a.disp.overlay_line_width = std::atoi(v);
        else if (s == "--overlay-text-scale" && (v = next())) a.disp.overlay_text_scale = std::atoi(v);
#endif
        else if (s == "-h" || s == "--help") return false;
        else return false;
    }
    if (a.slots <= 0) a.slots = std::max(4, a.batch * 2);
    if (a.batch < 1 || a.batch > 64 || a.gpu_depth < 1 || a.dump_every < 1) return false;
    a.gpu_depth = std::min(a.gpu_depth, a.slots);
#ifdef YVO_HAVE_DISPLAY
    if (a.disp.every < 1 || a.disp.overlay_line_width < 1 || a.disp.overlay_text_scale < 1) return false;
    // 페이싱 기본값: 표시 on 이면 source(소스 30 fps 를 따라간다), off 면 없음.
    if (!a.pace_explicit)
        a.disp.pace = a.display ? yvo::display::DisplayPace::Source : yvo::display::DisplayPace::None;
    a.disp.num_slots = a.slots;
    // ⚠ 디코더 13프레임 제약: 동시에 약 13프레임을 넘게 잡으면 디코더가 **에러 없이 멈춘다.**
    //   matched 는 slots + gpu_depth 를, latest 는 gpu_depth + image_count 를 잡는다. 여유를 두고 10 에서 막는다.
    if (a.display && a.slots + a.gpu_depth > 10) {
        std::fprintf(stderr,
                     "--display on 에서는 slots + gpu_depth <= 10 이어야 한다 (지금 %d + %d = %d).\n"
                     "디코더는 동시에 약 13프레임 넘게 잡히면 에러 없이 멈춘다. --slots 또는 --gpu-depth 를 줄여라.\n",
                     a.slots, a.gpu_depth, a.slots + a.gpu_depth);
        return false;
    }
#endif
    return true;
}

double cpu_ms() {
    rusage ru{};
    getrusage(RUSAGE_SELF, &ru);
    return static_cast<double>(ru.ru_utime.tv_sec + ru.ru_stime.tv_sec) * 1e3 +
           static_cast<double>(ru.ru_utime.tv_usec + ru.ru_stime.tv_usec) / 1e3;
}

// 스레드 안전 슬롯 풀 + 완료 큐 (HailoRT 콜백 ↔ 메인 스레드)
struct Completion {
    yvo::InferResult result;
    Clock::time_point t_done;
};

// 슬롯 반납 조건이 둘이다(계산 완료 + 표시 완료). need_display == false 면 표시 조건이 없어
// **오늘까지와 완전히 같은 동작**이 된다(--display off 회귀 경로).
class SlotExchange {
public:
    SlotExchange(int n, bool need_display)
        : need_display_(need_display), ready_(static_cast<std::size_t>(n), {false, false}),
          out_(static_cast<std::size_t>(n), false) {
        for (int i = 0; i < n; ++i) free_.push_back(i);
    }
    // HailoRT 콜백 스레드(host) 또는 메인 스레드(gpu 후처리)
    void complete(yvo::InferResult&& r) {
        Completion c{std::move(r), Clock::now()};
        {
            std::lock_guard<std::mutex> lk(m_);
            const int slot = c.result.slot;
            done_.push_back(std::move(c));
            mark(slot, 0);
        }
        cv_.notify_all();
    }
    // 메인 스레드 전용. 표시가 끝났거나(표시 완료 콜백) 표시를 건너뛰었다.
    void display_done(int slot) {
        {
            std::lock_guard<std::mutex> lk(m_);
            mark(slot, 1);
        }
        cv_.notify_all();
    }
    std::optional<int> try_acquire() {
        std::lock_guard<std::mutex> lk(m_);
        return pop_locked();
    }
    // slice 동안만 기다린다(블로킹 금지 — display_done 을 세울 주체가 호출자 자신이다).
    std::optional<int> acquire_poll(std::chrono::milliseconds slice) {
        std::unique_lock<std::mutex> lk(m_);
        if (free_.empty()) cv_.wait_for(lk, slice, [&] { return !free_.empty(); });
        return pop_locked();
    }
    std::vector<Completion> take_done() {
        std::lock_guard<std::mutex> lk(m_);
        std::vector<Completion> out(std::make_move_iterator(done_.begin()), std::make_move_iterator(done_.end()));
        done_.clear();
        return out;
    }

private:
    // m_ 를 잡은 상태로 부른다. 나가 있지 않은 슬롯의 신호는 무시한다(중복 호출을 무해하게 만든다).
    void mark(int slot, int which) {
        const auto i = static_cast<std::size_t>(slot);
        if (i >= out_.size() || !out_[i]) return;
        ready_[i][static_cast<std::size_t>(which)] = true;
        if (ready_[i][0] && (ready_[i][1] || !need_display_)) {
            ready_[i] = {false, false};
            out_[i] = false;
            free_.push_back(slot);
        }
    }
    std::optional<int> pop_locked() {
        if (free_.empty()) return std::nullopt;
        const int s = free_.front();
        free_.pop_front();
        const auto i = static_cast<std::size_t>(s);
        ready_[i] = {false, false};
        out_[i] = true;
        return s;
    }

    std::mutex m_;
    std::condition_variable cv_;
    bool need_display_;
    std::vector<std::array<bool, 2>> ready_;   // [0] = 계산 완료, [1] = 표시 완료
    std::vector<bool> out_;                    // 슬롯이 나가 있는가 (중복 신호 방어)
    std::deque<int> free_;
    std::deque<Completion> done_;
};

void draw_and_save(const std::string& path, const std::vector<std::uint8_t>& rgb, const yvo::LetterboxParams& lb,
                   const std::vector<yvo::Detection>& dets) {
    cv::Mat m(lb.dst_h, lb.dst_w, CV_8UC3, const_cast<std::uint8_t*>(rgb.data()));
    cv::Mat bgr;
    cv::cvtColor(m, bgr, cv::COLOR_RGB2BGR);
    for (const auto& d : dets) {
        // 원본 좌표 → letterbox 좌표
        const int x1 = static_cast<int>(std::lround(d.x1 * lb.scale + static_cast<float>(lb.pad_x)));
        const int y1 = static_cast<int>(std::lround(d.y1 * lb.scale + static_cast<float>(lb.pad_y)));
        const int x2 = static_cast<int>(std::lround(d.x2 * lb.scale + static_cast<float>(lb.pad_x)));
        const int y2 = static_cast<int>(std::lround(d.y2 * lb.scale + static_cast<float>(lb.pad_y)));
        // class_color()는 RGB를 돌려주므로 OpenCV(BGR)로 뒤집는다. 색 정의는 coco_labels.hpp 하나뿐이다.
        const auto rgb_col = yvo::class_color(d.class_id);
        const cv::Scalar color(rgb_col[2], rgb_col[1], rgb_col[0]);
        cv::rectangle(bgr, cv::Point(x1, y1), cv::Point(x2, y2), color, 1);
        char label[64];
        std::snprintf(label, sizeof(label), "%s %.2f", class_name(d.class_id), static_cast<double>(d.score));
        cv::putText(bgr, label, cv::Point(x1, std::max(10, y1 - 3)), cv::FONT_HERSHEY_SIMPLEX, 0.35, color, 1);
    }
    if (!cv::imwrite(path, bgr)) std::fprintf(stderr, "[warn] PNG 저장 실패: %s\n", path.c_str());
}

struct FdCloser {
    std::vector<int> fds;
    ~FdCloser() {
        for (int fd : fds)
            if (fd >= 0) ::close(fd);
    }
};

const char* layout_name(yvo::TensorLayout l) { return l == yvo::TensorLayout::Nhcw ? "nhcw" : "nhwc"; }

int run(const Args& a) {
    const bool dmabuf = a.mode == Mode::Dmabuf;
    // NHCW에서는 HailoRT가 입력 버퍼를 직접 DMA 매핑하는데, V3D GEM mmap 포인터는 매핑에 실패한다.
    // 첫 submit에서 러너가 던지게 두지 말고 시작 시점에 거른다.
    if (!dmabuf && a.layout == yvo::TensorLayout::Nhcw) {
        std::fprintf(stderr,
                     "--mode ptr --layout nhcw 조합은 쓸 수 없다: NHCW에서는 HailoRT가 입력 버퍼를 직접 DMA 매핑하는데\n"
                     "V3D GEM mmap 포인터는 DMA 매핑에 실패한다. --mode dmabuf --layout nhcw를 쓸 것.\n");
        return 2;
    }
    // GPU 후처리는 입력도 dma-buf(NHCW)여야 한다: HailoRawRunner 는 호스트 포인터 경로가 없고,
    // 원시 출력 6스트림은 NHCW 입력 전제로 검증했다.
    if (a.postproc == Postproc::Gpu && (!dmabuf || a.layout != yvo::TensorLayout::Nhcw)) {
        std::fprintf(stderr,
                     "--postproc gpu는 --mode dmabuf --layout nhcw 전용이다: HailoRawRunner는 입력·출력을\n"
                     "dma-buf로만 받고(호스트 포인터 경로 없음), 원시 출력 6스트림 경로는 NHCW 입력으로만 검증됐다.\n");
        return 2;
    }
    const double soc_t0 = soc_temperature();

    // export fd는 러너(매핑 해제)보다 늦게 닫아야 하므로 먼저 선언한다(역순 소멸).
    FdCloser export_fds;

    // ── 초기화 순서: WaylandWindow → VulkanContext → SandLetterbox → InferStage → DisplayPipeline → 디코더 ──
    // ⚠ 창이 컨텍스트보다 먼저다: vkGetPhysicalDeviceWaylandPresentationSupportKHR 가 wl_display* 를
    //   요구하기 때문이다(vulkan_context.hpp 의 닭-달걀 설명). 그리고 창은 컨텍스트보다 **늦게** 소멸해야
    //   하므로 여기서 선언한다(역순 소멸).
    yvo::vk::VulkanContextOptions copt;
#ifdef YVO_HAVE_DISPLAY
    yvo::display::WaylandWindow win;
    bool have_win = false;
    if (a.display) {
        have_win = yvo::display::DisplayPipeline::open_window(win, a.disp);
        if (!have_win && a.disp.backend == yvo::display::DisplayBackendKind::Wayland) {
            std::fprintf(stderr, "--display-backend wayland: 창을 열지 못했다 — %s\n", win.error().c_str());
            return 1;
        }
        copt.enable_present = have_win;
        copt.wayland_display = have_win ? win.native_display() : nullptr;
        // strict: 조용한 폴백 금지. auto 면 present 를 못 켜도 오프스크린으로 계속한다.
        copt.require_present = have_win && a.disp.backend == yvo::display::DisplayBackendKind::Wayland;
    }
#endif
    yvo::vk::VulkanContext ctx(copt);
    yvo::vk::SandLetterboxConfig pcfg;
    pcfg.num_slots = a.slots;
    pcfg.export_dmabuf = dmabuf;
    pcfg.out_layout = a.layout;
    yvo::vk::SandLetterbox pre(ctx, pcfg);

    std::unique_ptr<yvo::pipeline::InferStage> stage;
    yvo::pipeline::GpuPostprocStage* gpu_stage = nullptr;
    if (a.postproc == Postproc::Gpu) {
        yvo::pipeline::GpuPostprocStageConfig gcfg;
        gcfg.hef_path = a.hef;
        gcfg.batch_size = static_cast<std::uint16_t>(a.batch);
        gcfg.num_slots = a.slots;
        gcfg.score_thr = a.score >= 0.0f ? a.score : 0.2f;
        gcfg.input_alloc = a.postproc_alloc == "heap" ? yvo::vk::YoloInputAlloc::ImportHeap
                                                      : yvo::vk::YoloInputAlloc::ExportDmaBuf;
        auto g = std::make_unique<yvo::pipeline::GpuPostprocStage>(ctx, gcfg);
        gpu_stage = g.get();
        stage = std::move(g);
    } else {
        yvo::HailoRunnerConfig hcfg;
        hcfg.hef_path = a.hef;
        hcfg.input_layout = a.layout;  // allow_host_ptr_nhcw는 그대로 false: V3D 포인터 제출은 러너가 막는다
        hcfg.batch_size = static_cast<std::uint16_t>(a.batch);
        hcfg.num_slots = a.slots;
        hcfg.score_threshold = a.score;
        stage = std::make_unique<yvo::pipeline::HostNmsStage>(hcfg);
    }
    if (stage->input_frame_size() != pre.output_size()) {
        std::fprintf(stderr, "입력 크기 불일치: NPU %zu, GPU %zu\n", stage->input_frame_size(), pre.output_size());
        return 1;
    }

    std::vector<int> slot_fd(static_cast<std::size_t>(a.slots), -1);
    if (dmabuf) {
        for (int s = 0; s < a.slots; ++s) {
            const int fd = pre.export_dmabuf_fd(s);
            export_fds.fds.push_back(fd);
            stage->bind_input_dmabuf(s, fd);
            slot_fd[static_cast<std::size_t>(s)] = fd;
        }
    }

    // 표시 경로. 디코더보다 **먼저** 만든다 — extra_hw_frames 가 image_count() 에 의존한다.
    // 원본 크기는 디코더를 연 뒤 set_source_size() 로 알려 준다.
#ifdef YVO_HAVE_DISPLAY
    std::unique_ptr<yvo::display::DisplayPipeline> display;
    if (a.display) display = std::make_unique<yvo::display::DisplayPipeline>(ctx, have_win ? &win : nullptr, a.disp);
#endif

    yvo::FfmpegDrmDecoder dec;
    yvo::FfmpegDrmDecoder::Options dopt;
    dopt.extra_hw_frames = std::max(2 * yvo::kNumSlots, a.gpu_depth + 4);
#ifdef YVO_HAVE_DISPLAY
    if (display) {
        // 표시가 프레임을 더 오래 잡는다. matched 는 슬롯마다 한 장(= slots), latest 는 스왑체인
        // 이미지마다 한 장(= image_count)을 GPU 가 다 읽을 때까지 붙든다. 여기에 gpu_depth 와 여유 4.
        const int need = a.disp.sync == yvo::display::DisplaySync::Matched
                             ? a.slots + a.gpu_depth + 4
                             : a.gpu_depth + static_cast<int>(display->image_count()) + 4;
        dopt.extra_hw_frames = std::max(8, need);
    }
#endif
    if (!dec.open(a.video, dopt)) {
        std::fprintf(stderr, "decoder open failed: %s\n", dec.error().c_str());
        return 1;
    }
    const int src_w = dec.width(), src_h = dec.height();
    stage->set_source_size(src_w, src_h);
#ifdef YVO_HAVE_DISPLAY
    if (display) display->set_source_size(src_w, src_h);
#endif

    std::FILE* jsonl = nullptr;
    if (!a.dump_dir.empty()) {
        std::filesystem::create_directories(a.dump_dir);
        const std::string p = a.dump_dir + "/detections.jsonl";
        jsonl = std::fopen(p.c_str(), "w");
        if (!jsonl) {
            std::fprintf(stderr, "cannot open %s\n", p.c_str());
            return 1;
        }
    }

    const float hailo_t0 = stage->chip_temperature();
    if (!a.quiet) {
        const std::string pp_desc = gpu_stage ? std::string("gpu/") + gpu_stage->alloc_name() : std::string("host");
        std::printf("[cfg] video=%s %dx%d mode=%s layout=%s batch=%d slots=%d gpu_depth=%d postproc=%s device=%s\n",
                    a.video.c_str(), src_w, src_h, dmabuf ? "dmabuf" : "ptr", layout_name(a.layout), a.batch, a.slots,
                    a.gpu_depth, pp_desc.c_str(), ctx.device_name());
#ifdef YVO_HAVE_DISPLAY
        if (display)
            std::printf("[display] backend=%s %ux%u %s / %s images=%u sync=%s pace=%s every=%d hw_frames=%d\n",
                        display->offscreen() ? "offscreen" : "wayland", display->extent().width,
                        display->extent().height, display->uniform_scale() ? "uniform-scale(bilinear)" : "1:1",
                        display->blit_mode() ? "blit fallback" : "compute -> storage image", display->image_count(),
                        yvo::display::sync_name(a.disp.sync), yvo::display::pace_name(a.disp.pace), a.disp.every,
                        dopt.extra_hw_frames);
#endif
        std::fflush(stdout);
    }

    bool need_display_ack = false;
#ifdef YVO_HAVE_DISPLAY
    need_display_ack = display != nullptr;
#endif
    SlotExchange ex(a.slots, need_display_ack);
#ifdef YVO_HAVE_DISPLAY
    // 표시 완료(= DisplayConvert 의 submit 펜스)가 슬롯 반납의 두 번째 조건이다. 메인 스레드에서 불린다.
    if (display) display->set_display_done_callback([&ex](int s) { ex.display_done(s); });
#endif
    std::deque<int> gpu_q;  // GPU에 걸린 슬롯 (submit 순서)
    std::vector<double> decode_ms, gpu_ms, npu_ms, e2e_ms, wait_block_ms;
    std::unordered_map<std::int64_t, Clock::time_point> t_decoded;  // pts → 디코드 반환 시각
    std::map<std::int64_t, Completion> reorder;                      // pts 순서 기록용
    std::unordered_map<std::int64_t, std::vector<std::uint8_t>> png_src;  // dump 대상 프레임의 letterbox 사본
    std::unordered_map<std::int64_t, yvo::LetterboxParams> png_lb;
    std::int64_t next_pts = 0;
    long submitted_npu = 0, completed = 0, total_dets = 0, errors = 0;

    // 디코더 13프레임 제약 계측: 우리가 실제로 붙들고 있는 AVFrame 수를 weak_ptr 로 센다.
    // (SandLetterbox 의 keepalive 와 DisplayPipeline 의 retained 가 같은 shared_ptr 를 공유하므로
    //  이중으로 세지 않는다. 이것이 "조용히 멈추는" 실패 모드의 유일한 감지 수단이다.)
    std::deque<std::weak_ptr<void>> live_frames;
    long held_frames_max = 0, slot_timeout_count = 0;
    std::vector<yvo::Detection> latest_dets;   // --display-sync latest 가 얹을 가장 최근 검출
    double fps_now = 0.0;

    auto on_result = [&ex](yvo::InferResult&& r) { ex.complete(std::move(r)); };

    // 완료 결과를 pts 순서로 기록
    auto flush_results = [&]() {
        for (auto& c : ex.take_done()) {
            const std::int64_t pts = c.result.pts;
            reorder.emplace(pts, std::move(c));
        }
        while (!reorder.empty() && reorder.begin()->first == next_pts) {
            Completion& c = reorder.begin()->second;
            const yvo::InferResult& r = c.result;
            ++completed;
            if (r.status != HAILO_SUCCESS) {
                ++errors;
                std::fprintf(stderr, "[error] frame %lld: HailoRT status %d\n", static_cast<long long>(r.pts),
                             static_cast<int>(r.status));
            }
            npu_ms.push_back(r.latency_ms);
            if (auto it = t_decoded.find(r.pts); it != t_decoded.end()) {
                e2e_ms.push_back(std::chrono::duration<double, std::milli>(c.t_done - it->second).count());
                t_decoded.erase(it);
            }
            total_dets += static_cast<long>(r.detections.size());
            if (jsonl) {
                std::fprintf(jsonl, "{\"frame\":%lld,\"detections\":[", static_cast<long long>(r.pts));
                for (std::size_t i = 0; i < r.detections.size(); ++i) {
                    const auto& d = r.detections[i];
                    std::fprintf(jsonl, "%s{\"class\":%d,\"name\":\"%s\",\"score\":%.4f,\"box\":[%.1f,%.1f,%.1f,%.1f]}",
                                 i ? "," : "", d.class_id, class_name(d.class_id), static_cast<double>(d.score),
                                 static_cast<double>(d.x1), static_cast<double>(d.y1), static_cast<double>(d.x2),
                                 static_cast<double>(d.y2));
                }
                std::fprintf(jsonl, "]}\n");
            }
            if (auto it = png_src.find(r.pts); it != png_src.end()) {
                char name[64];
                std::snprintf(name, sizeof(name), "/frame%05lld.png", static_cast<long long>(r.pts));
                draw_and_save(a.dump_dir + name, it->second, png_lb[r.pts], r.detections);
                png_src.erase(it);
                png_lb.erase(r.pts);
            }
#ifdef YVO_HAVE_DISPLAY
            // 표시는 기록이 전부 끝난 뒤에 한다(드레인 끝). matched 에서는 여기가 유일한 show 지점이다.
            if (display && a.disp.sync == yvo::display::DisplaySync::Latest) latest_dets = r.detections;
            if (display && a.disp.sync == yvo::display::DisplaySync::Matched) {
                char hud[64];
                std::snprintf(hud, sizeof(hud), "FPS %4.1f DET %d NPU %3.0fms", fps_now,
                              static_cast<int>(r.detections.size()), r.latency_ms);
                // show() 가 false 면 그 프레임은 화면에 나가지 않는다 → 슬롯을 **즉시** 돌려줘야 한다.
                if (!display->show(r.slot, r.detections, hud)) ex.display_done(r.slot);
            }
#endif
            reorder.erase(reorder.begin());
            ++next_pts;
        }
    };

    // GPU 대기열 맨 앞 슬롯을 끝내고 NPU에 넘긴다
    auto finish_gpu = [&]() {
        const int slot = gpu_q.front();
        gpu_q.pop_front();
        yvo::PreprocOutput out = pre.wait(slot);
        if (pre.last_gpu_ms() >= 0) gpu_ms.push_back(pre.last_gpu_ms());
        if (jsonl && out.pts % a.dump_every == 0) {
            // draw_and_save()는 packed RGB를 기대하므로 NHCW 출력은 되돌려서 보관한다.
            std::vector<std::uint8_t>& buf = png_src[out.pts];
            buf.resize(out.size);
            if (out.layout == yvo::TensorLayout::Nhcw)
                yvo::nhcw_to_rgb_packed(out.host_ptr, out.lb.dst_w, out.lb.dst_h, buf.data());
            else
                std::memcpy(buf.data(), out.host_ptr, out.size);
            png_lb[out.pts] = out.lb;
        }
        if (dmabuf) out.dmabuf_fd = slot_fd[static_cast<std::size_t>(slot)];
        stage->submit(out, on_result);
        ++submitted_npu;
#ifdef YVO_HAVE_DISPLAY
        // latest: 검출을 기다리지 않고 **전처리가 끝나는 대로** 화면에 낸다(가장 최근 검출을 얹는다).
        // 표시가 슬롯 수명을 늘리지 않도록 display_done 을 여기서 바로 세운다.
        if (display && a.disp.sync == yvo::display::DisplaySync::Latest) {
            char hud[64];
            std::snprintf(hud, sizeof(hud), "FPS %4.1f DET %d", fps_now, static_cast<int>(latest_dets.size()));
            display->show(slot, latest_dets, hud);
            ex.display_done(slot);
        }
#endif
        // submit()은 NPU 백프레셔에서 블록할 수 있다. 그동안 쌓인 완료분을 바로 거둬
        // 슬롯을 돌려준다(gpu 경로에서 슬롯 보유 시간이 NPU 점유를 그대로 깎는다). host는 no-op.
        stage->pump();
    };

    const double c0 = cpu_ms();
    const auto t0 = Clock::now();
    long n = 0;
    bool fail = false;
    try {
        while (!stop_requested() && (a.frames <= 0 || n < a.frames)) {
#ifdef YVO_HAVE_DISPLAY
            if (display) {
                if (!display->poll_window()) break;   // 창이 닫혔다
                // --display-pace source: 소스 프레임 레이트를 따라간다. **블로킹 sleep 을 쓰지 않는다** —
                // 그동안 NPU 완료·GPU 후처리·표시 완료를 계속 거둬야 지연이 왜곡되지 않는다.
                if (auto dl = display->pace_deadline(n)) {
                    while (Clock::now() < *dl && !stop_requested()) {
                        stage->pump();
                        while (!gpu_q.empty() && pre.ready(gpu_q.front())) finish_gpu();
                        display->pump();
                        flush_results();
                        std::this_thread::sleep_for(std::chrono::microseconds(500));
                    }
                }
            }
#endif
            const auto d0 = Clock::now();
            auto f = dec.next();
            const auto d1 = Clock::now();
            stage->pump();   // 디코드 대기 중에 도착한 완료분 (host는 no-op)
            if (!f) break;
            decode_ms.push_back(std::chrono::duration<double, std::milli>(d1 - d0).count());
            f->pts = n;  // 프레임 순번으로 통일
            t_decoded[n] = d1;

            // 빈 슬롯 확보. 없으면 GPU 대기열을 먼저 NPU로 넘기고, 그래도 없으면 콜백을 기다린다.
            std::optional<int> slot = ex.try_acquire();
            const auto w0 = Clock::now();
            // ⚠ **블로킹 대기를 쓰지 않는다.** 슬롯 반납의 두 번째 조건(display_done)을 세울 주체가
            //   메인 스레드 자신이므로, 여기서 condvar 를 오래 잡으면 그대로 교착한다.
            //   대신 한 바퀴에 세 가지를 전부 진행시킨다:
            //     stage->pump()  NPU 완료 → GPU 후처리 → compute_done (host 는 no-op)
            //     finish_gpu()   전처리 결과를 NPU 로 밀어 슬롯이 다음 단계로 넘어가게 한다
            //     flush_results() **표시를 거는 곳이 여기다**(matched). 빠뜨리면 compute_done 만 서고
            //                    display_done 이 영원히 오지 않아 5 초 만에 타임아웃한다(실제로 그랬다)
            //     display->pump() 표시 펜스 회수 → display_done
            //   --display off && --postproc host 면 두 pump 가 no-op 이고 flush_results 는 원래 매 프레임
            //   부르던 것이라 오늘과 동치다.
            if (!slot) {
                const auto deadline = Clock::now() + std::chrono::seconds(5);
                while (!slot) {
                    stage->pump();
                    if (!gpu_q.empty()) finish_gpu();
                    flush_results();
#ifdef YVO_HAVE_DISPLAY
                    if (display) display->pump();
#endif
                    slot = ex.acquire_poll(std::chrono::milliseconds(2));
                    // 종료 신호. 한 바퀴가 2 ms 라 즉시 빠져나온다(5 초 타임아웃을 기다리지 않는다).
                    // **예외를 쓰지 않는다** — 아래 catch 는 진짜 실패용이고, 사용자의 중단은 실패가 아니다.
                    if (!slot && stop_requested()) break;
                    if (!slot && Clock::now() > deadline) {
                        ++slot_timeout_count;
                        throw std::runtime_error("슬롯 대기 타임아웃(5 s): 계산·표시 완료가 오지 않음");
                    }
                }
            }
            // 슬롯 없이 빠져나왔다 = 종료 신호. 이미 디코드한 이 프레임은 제출하지 않고 버린다.
            // n 도 올리지 않으므로 decoded/submitted/completed 카운트는 그대로 일치한다.
            if (!slot) break;
            wait_block_ms.push_back(std::chrono::duration<double, std::milli>(Clock::now() - w0).count());

#ifdef YVO_HAVE_DISPLAY
            // ⚠ f.reset() **앞에서** 붙든다. 여기가 glass-to-glass 의 시작점이기도 하다.
            if (display) display->retain(*slot, *f);
#endif
            pre.submit(*f, *slot);
            gpu_q.push_back(*slot);
            live_frames.push_back(f->keepalive);
            f.reset();  // keepalive는 SandLetterbox(와 표시 경로)가 들고 있다

            // GPU 깊이 제한 + 이미 끝난 슬롯은 바로 NPU로
            while (static_cast<int>(gpu_q.size()) > a.gpu_depth) finish_gpu();
            while (!gpu_q.empty() && pre.ready(gpu_q.front())) finish_gpu();

            stage->pump();
            flush_results();
#ifdef YVO_HAVE_DISPLAY
            if (display) display->pump();
#endif
            ++n;
            // 실제로 붙들고 있는 디코더 프레임 수(약 13이 한계). 만료된 weak_ptr 을 걷어내고 최대치를 남긴다.
            for (auto it = live_frames.begin(); it != live_frames.end();)
                it = it->expired() ? live_frames.erase(it) : it + 1;
            held_frames_max = std::max(held_frames_max, static_cast<long>(live_frames.size()));
            {
                const double el = std::chrono::duration<double>(Clock::now() - t0).count();
                fps_now = el > 0 ? static_cast<double>(n) / el : 0.0;
            }
            if (!a.quiet && n % 300 == 0) {
                const double el = std::chrono::duration<double>(Clock::now() - t0).count();
                std::printf("[progress] %ld frames, %.1f FPS, hailo %.1f C\n", n, static_cast<double>(n) / el,
                            static_cast<double>(stage->chip_temperature()));
                std::fflush(stdout);
            }
        }
        while (!gpu_q.empty()) finish_gpu();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] frame %ld: %s\n", n, e.what());
        fail = true;
        while (!gpu_q.empty()) {  // 남은 GPU 작업은 NPU로 넘기지 않고 정리
            pre.wait(gpu_q.front());
            gpu_q.pop_front();
        }
    }
    stage->drain();
    flush_results();   // matched 의 마지막 show 들이 여기서 나간다
#ifdef YVO_HAVE_DISPLAY
    if (display) display->drain();   // 표시 펜스를 전부 기다려 디코더 프레임·import pin 을 놓는다
#endif
    // 중단 여부는 드레인까지 끝난 뒤에 확정한다(드레인 도중에 받은 시그널도 중단으로 친다).
    int stop_signal = 0;
    if (g_stop.load(std::memory_order_relaxed)) {
        stop_signal = g_stop_signal.load(std::memory_order_relaxed);
        if (stop_signal == 0) stop_signal = SIGINT;  // 핸들러가 번호를 쓰기 직전이면 SIGINT 로 본다
    }
    const bool interrupted = stop_signal != 0;
    if (interrupted) fail = true;  // RESULT_CSV/RESULT_CSV2 의 ok 를 0 으로 (벤치 집계에 섞이면 안 된다)

    const auto t1 = Clock::now();
    const double c1 = cpu_ms();

    if (!dec.error().empty()) {
        std::fprintf(stderr, "decode error after %ld frames: %s\n", n, dec.error().c_str());
        fail = true;
    }
    const float hailo_t1 = stage->chip_temperature();
    // GPU 후처리 통계는 stage 를 놓기 전에 떠 둔다.
    Stats sp, sq, sc, sh;
    long cand_overflow = 0;
    if (gpu_stage) {
        sp = stats(gpu_stage->post_ms());
        sq = stats(gpu_stage->nms_us());
        sc = stats(gpu_stage->cand());
        sh = stats(gpu_stage->hold_ms());
        cand_overflow = gpu_stage->overflow_frames();
    }
    // 표시 통계도 display 를 놓기 전에 떠 둔다.
    Stats sdc, sdo, sdi, sdg;
    unsigned long long presented = 0, disp_dropped = 0, recreates = 0, ov_overflow = 0;
    const char* disp_backend = "off";
#ifdef YVO_HAVE_DISPLAY
    if (display) {
        sdc = stats(display->convert_ms());
        sdo = stats(display->overlay_ms());
        sdi = stats(display->present_interval_ms());
        sdg = stats(display->glass_ms());
        presented = display->presented();
        disp_dropped = display->display_dropped();
        recreates = display->recreates();
        ov_overflow = display->overlay_overflows();
        disp_backend = display->offscreen() ? "offscreen" : "wayland";
    }
#endif
    // 종료 순서: drain(위) → import 캐시 비움 → 디코더 close → 러너 해제(dma-buf unmap)
    //            → 표시 해제(스왑체인·창) → export fd close(FdCloser) → ctx → 창
    pre.clear_import_cache();
    dec.close();
    gpu_stage = nullptr;
    stage.reset();
#ifdef YVO_HAVE_DISPLAY
    display.reset();
#endif
    if (jsonl) std::fclose(jsonl);
    const double soc_t1 = soc_temperature();

    const double wall_s = std::chrono::duration<double>(t1 - t0).count();
    const double fps = static_cast<double>(completed) / wall_s;
    const double cpu_per_frame = completed ? (c1 - c0) / static_cast<double>(completed) : 0.0;
    const Stats sd = stats(decode_ms), sg = stats(gpu_ms), sn = stats(npu_ms), se = stats(e2e_ms),
                sw = stats(wait_block_ms);
    if (static_cast<long>(completed) != n || static_cast<long>(completed) != submitted_npu) {
        std::fprintf(stderr, "[error] 프레임 수 불일치: decoded=%ld npu_submitted=%ld completed=%ld\n", n, submitted_npu,
                     completed);
        fail = true;
    }
    if (errors) fail = true;

    const std::string postproc_desc =
        a.postproc == Postproc::Gpu ? std::string("gpu/") + a.postproc_alloc : std::string("host");
    if (interrupted)
        std::printf("[interrupted] %s 수신 — %ld 프레임에서 중단 (아래 통계는 부분 실행 값이다, ok=0)\n",
                    stop_signal == SIGTERM ? "SIGTERM" : "SIGINT", n);
    std::printf("== yvo_pipeline (%s, %s, batch %d, slots %d, gpu_depth %d, postproc %s) ==\n",
                dmabuf ? "dmabuf" : "ptr", layout_name(a.layout), a.batch, a.slots, a.gpu_depth,
                postproc_desc.c_str());
    std::printf("frames          %ld (decoded %ld, errors %ld)\n", completed, n, errors);
    std::printf("wall            %.3f s, %.2f FPS\n", wall_s, fps);
    std::printf("cpu             %.3f ms/frame (process user+sys %.1f ms, %.0f%% of one core)\n", cpu_per_frame,
                c1 - c0, (c1 - c0) / (wall_s * 10.0));
    std::printf("decode call ms  mean %.3f  median %.3f  p90 %.3f  max %.3f\n", sd.mean, sd.median, sd.p90, sd.max);
    std::printf("gpu shader ms   mean %.3f  median %.3f  p90 %.3f  max %.3f\n", sg.mean, sg.median, sg.p90, sg.max);
    std::printf("npu latency ms  mean %.3f  median %.3f  p90 %.3f  max %.3f\n", sn.mean, sn.median, sn.p90, sn.max);
    std::printf("e2e latency ms  mean %.3f  median %.3f  p90 %.3f  max %.3f  (decode 반환 → NPU 완료 콜백)\n", se.mean,
                se.median, se.p90, se.max);
    std::printf("slot wait ms    mean %.3f  p90 %.3f  (메인 스레드가 빈 슬롯을 기다린 시간)\n", sw.mean, sw.p90);
    if (a.postproc == Postproc::Gpu) {
        std::printf("gpu post ms     mean %.3f  median %.3f  p90 %.3f  max %.3f  (yolo_decode.comp pass 1)\n", sp.mean,
                    sp.median, sp.p90, sp.max);
        std::printf("cpu nms us      mean %.3f  median %.3f  p90 %.3f  max %.3f  (cpu_nms_greedy_by_class)\n", sq.mean,
                    sq.median, sq.p90, sq.max);
        std::printf("candidates      mean %.2f  max %.0f  overflow frames %ld\n", sc.mean, sc.max, cand_overflow);
        std::printf("slot hold ms    mean %.3f  median %.3f  p90 %.3f  max %.3f  (NPU 완료 콜백 → 슬롯 반납)\n",
                    sh.mean, sh.median, sh.p90, sh.max);
    }
    if (a.display) {
        std::printf("disp convert ms mean %.3f  median %.3f  p90 %.3f  (display_convert.comp)\n", sdc.mean,
                    sdc.median, sdc.p90);
        std::printf("disp overlay ms mean %.3f  median %.3f  p90 %.3f  (overlay_boxes.comp)\n", sdo.mean, sdo.median,
                    sdo.p90);
        std::printf("present 간격 ms median %.3f  p90 %.3f  (n=%zu)\n", sdi.median, sdi.p90, sdi.n);
        std::printf("glass-to-glass  mean %.2f  median %.2f  p90 %.2f  (n=%zu, 디코드 → 화면 도달)\n", sdg.mean,
                    sdg.median, sdg.p90, sdg.n);
        std::printf("display         presented %llu, dropped %llu, recreates %llu, overlay overflow %llu\n",
                    presented, disp_dropped, recreates, ov_overflow);
    }
    std::printf("held frames max %ld (디코더 한계 약 13), slot timeout %ld\n", held_frames_max, slot_timeout_count);
    std::printf("detections      %ld total, %.2f / frame\n", total_dets,
                completed ? static_cast<double>(total_dets) / static_cast<double>(completed) : 0.0);
    std::printf("hailo chip temp %.1f -> %.1f C\n", static_cast<double>(hailo_t0), static_cast<double>(hailo_t1));
    std::printf("soc temp        %.1f -> %.1f C\n", soc_t0, soc_t1);
    if (ctx.debug_messenger_active())
        std::printf("vk validation   errors %llu, warnings %llu\n",
                    static_cast<unsigned long long>(ctx.validation_errors()),
                    static_cast<unsigned long long>(ctx.validation_warnings()));
    if (!a.csv_label.empty()) {
        // label,mode,layout,batch,slots,frames,wall_s,fps,cpu_ms_per_frame,decode_mean,gpu_mean,gpu_p90,npu_mean,
        // npu_p90,e2e_mean,e2e_p90,detections,hailo_t0,hailo_t1,soc_t0,soc_t1,ok
        std::printf(
            "RESULT_CSV,%s,%s,%s,%d,%d,%ld,%.3f,%.2f,%.3f,%.3f,%.3f,%.3f,%.2f,%.2f,%.2f,%.2f,%ld,%.1f,%.1f,%.1f,%.1f,%d\n",
            a.csv_label.c_str(), dmabuf ? "dmabuf" : "ptr", layout_name(a.layout), a.batch, a.slots, completed, wall_s,
            fps, cpu_per_frame, sd.mean, sg.mean, sg.p90, sn.mean, sn.p90, se.mean, se.p90, total_dets,
            static_cast<double>(hailo_t0), static_cast<double>(hailo_t1), soc_t0, soc_t1, fail ? 0 : 1);
        // 신규 지표는 **별도 줄**이다. RESULT_CSV 의 22필드는 순서·개수를 바꾸지 않는다
        // (06_tools/bench/pipeline_bench.sh:120 이 헤더를 grep -qx 로 비교한다).
        // label,postproc,alloc,gpu_post_mean,gpu_post_p90,cpu_nms_us_mean,cand_mean,cand_max,cand_overflow,ok
        // + 표시 지표는 **뒤로만** 덧붙인다(앞의 10필드는 순서·개수를 바꾸지 않는다):
        //   display,sync,pace,disp_convert_mean,disp_convert_p90,disp_overlay_mean,present_interval_median,
        //   e2e_glass_mean,e2e_glass_p90,presented,display_dropped,recreates,held_frames_max,slot_timeout_count
        const char* sync_s = "-";
        const char* pace_s = "-";
#ifdef YVO_HAVE_DISPLAY
        if (a.display) {
            sync_s = yvo::display::sync_name(a.disp.sync);
            pace_s = yvo::display::pace_name(a.disp.pace);
        }
#endif
        std::printf("RESULT_CSV2,%s,%s,%s,%.4f,%.4f,%.3f,%.2f,%.0f,%ld,%d"
                    ",%s,%s,%s,%.4f,%.4f,%.4f,%.3f,%.2f,%.2f,%llu,%llu,%llu,%ld,%ld\n",
                    a.csv_label.c_str(), a.postproc == Postproc::Gpu ? "gpu" : "host",
                    a.postproc == Postproc::Gpu ? a.postproc_alloc.c_str() : "-", sp.mean, sp.p90, sq.mean, sc.mean,
                    sc.max, cand_overflow, fail ? 0 : 1,
                    disp_backend, sync_s, pace_s, sdc.mean, sdc.p90, sdo.mean, sdi.median, sdg.mean, sdg.p90,
                    presented, disp_dropped, recreates, held_frames_max, slot_timeout_count);
    }
    std::fflush(stdout);
    if (ctx.debug_messenger_active() && ctx.validation_errors() > 0) fail = true;
    // 중단된 실행은 셸 관례대로 128+signo (SIGINT 130, SIGTERM 143).
    if (interrupted) return 128 + stop_signal;
    return fail ? 1 : 0;
}

} // namespace

int main(int argc, char** argv) {
    Args a;
    if (!parse(argc, argv, a)) {
        usage(argv[0]);
        return 2;
    }
    install_stop_handlers();  // 실패해도 계속한다(경고는 함수 안에서 낸다)
    try {
        return run(a);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "fatal: %s\n", e.what());
        return 1;
    }
}
