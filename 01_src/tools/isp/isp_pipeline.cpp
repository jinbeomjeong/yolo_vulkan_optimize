// yvo_isp_pipeline: HEVC HW 디코드(DRM_PRIME) → PiSP BE(ISP) letterbox → HailoRT yolov11m 추론
//
// 01_src/pipeline/main.cpp(yvo_pipeline)와 같은 일을 하되 전처리만 Vulkan 셰이더 → ISP로 바꿨다.
// main.cpp는 건드리지 않는다. 검출 파싱은 HailoRunner 안의 parse_nms_by_class + letterbox 역변환을
// 그대로 쓰므로 두 도구의 검출 경로는 완전히 같다.
//
//   --layout nhwc : ISP 출력(packed RGB888)을 그대로 HailoRT에 넘긴다(A안).
//                   HailoRT가 CPU에서 NHWC→NHCW 전치를 한다. 그 비용이 이 측정의 핵심 미지수다.
//   --layout nhcw : ISP 출력을 우리가 NHCW로 전치해 별도 dma-heap 버퍼로 넘긴다(C안).
//
// 스레드 구조 (main.cpp와 다른 점)
//   - IspResizer::process()가 동기다(BackendDevice::Run()에 비동기 API가 없다). 그래서 디코드와
//     전처리가 겹치지 않는다. NPU 상한이 51.7 FPS라 처리량 병목은 아니지만 e2e 지연에는 불리하다.
//   - HailoRT 콜백 스레드가 슬롯을 반납하는 구조는 main.cpp와 같다.
// 슬롯 규약: ISP 출력 슬롯 = HailoRT 슬롯 = NHCW 버퍼 슬롯 (0..slots-1)
//   free → [ISP process] → [(nhcw면) 전치] → [HailoRunner::submit] → [완료 콜백] → free
#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
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
#include <unordered_map>
#include <vector>

#include <sys/resource.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "core/coco_labels.hpp"
#include "ref/cpu_letterbox.hpp"
#include "hailo/hailo_runner.hpp"
#include "core/dma_heap_alloc.hpp"
#include "isp/isp_resizer.hpp"
#include "isp/nhcw_repack.hpp"
#include "decode/ffmpeg_drm_decoder.hpp"
#include "toolkit/stats.hpp"
#include "toolkit/tool_util.hpp"

using yvo::Stats;
using yvo::stats;

namespace {

using yvo::tool::cpu_ms;
using yvo::tool::soc_temperature;

using Clock = std::chrono::steady_clock;
using yvo::class_name;

enum class Mode { Ptr, Dmabuf };

struct Args {
    std::string video = YVO_DEFAULT_VIDEO;
    std::string hef = YVO_DEFAULT_HEF;
    std::string media = "/dev/media0";
    std::string filter; // libpisp resample 필터(비면 smart selection: 1.5배 리샘플 구간 = lanczos2)
    long frames = 0; // 0 = 전체
    Mode mode = Mode::Dmabuf;
    yvo::TensorLayout layout = yvo::TensorLayout::Nhwc;
    yvo::isp::RepackMode repack = yvo::isp::RepackMode::Cpu; // layout nhcw일 때만 의미 있다
    int batch = 8;
    int slots = 0; // 0 = max(4, batch*2)
    unsigned int branch = 1;
    float score = -1.0f;
    std::string dump_dir;
    int dump_every = 100;
    bool quiet = false;
    std::string csv_label;
};

void usage(const char* a0) {
    std::fprintf(stderr,
                 "usage: %s [--video PATH] [--hef PATH] [--frames N] [--mode ptr|dmabuf] [--layout nhwc|nhcw]\n"
                 "          [--repack cpu|neon] [--batch N] [--slots N] [--branch N] [--media /dev/mediaN]\n"
                 "          [--filter lanczos3|lanczos2|bicubic-spline|michel-netravali]\n"
                 "          [--score T] [--dump-dir DIR] [--dump-every K] [--quiet] [--csv-label L]\n"
                 "  --layout nhwc : ISP 출력을 그대로 HailoRT에 (HailoRT가 CPU 전치) — A안\n"
                 "  --layout nhcw : 우리가 전치해서 별도 dma-heap 버퍼로 넘김 — C안\n"
                 "  --mode        : HailoRT 입력 전달 방식. dmabuf는 fd(set_dma_buffer), ptr는 매핑 포인터\n",
                 a0);
}

bool parse(int argc, char** argv, Args& a) {
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        auto next = [&]() -> const char* { return i + 1 < argc ? argv[++i] : nullptr; };
        const char* v = nullptr;
        if (s == "--video" && (v = next())) a.video = v;
        else if (s == "--hef" && (v = next())) a.hef = v;
        else if (s == "--media" && (v = next())) a.media = v;
        else if (s == "--filter" && (v = next())) a.filter = v;
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
        } else if (s == "--repack" && (v = next())) {
            if (!yvo::isp::parse_repack(v, a.repack) || a.repack == yvo::isp::RepackMode::None) return false;
        }
        else if (s == "--batch" && (v = next())) a.batch = std::atoi(v);
        else if (s == "--slots" && (v = next())) a.slots = std::atoi(v);
        else if (s == "--branch" && (v = next())) a.branch = static_cast<unsigned int>(std::atoi(v));
        else if (s == "--score" && (v = next())) a.score = static_cast<float>(std::atof(v));
        else if (s == "--dump-dir" && (v = next())) a.dump_dir = v;
        else if (s == "--dump-every" && (v = next())) a.dump_every = std::atoi(v);
        else if (s == "--csv-label" && (v = next())) a.csv_label = v;
        else if (s == "--quiet") a.quiet = true;
        else return false;
    }
    if (a.slots <= 0) a.slots = std::max(4, a.batch * 2);
    return a.batch >= 1 && a.batch <= 64 && a.dump_every >= 1;
}

// 스레드 안전 슬롯 풀 + 완료 큐 (main.cpp의 SlotExchange와 같은 규약)
struct Completion {
    yvo::InferResult result;
    Clock::time_point t_done;
};

class SlotExchange {
public:
    explicit SlotExchange(int n) {
        for (int i = 0; i < n; ++i) free_.push_back(i);
    }
    void complete(yvo::InferResult&& r) {
        Completion c{std::move(r), Clock::now()};
        {
            std::lock_guard<std::mutex> lk(m_);
            const int slot = c.result.slot;
            done_.push_back(std::move(c));
            free_.push_back(slot);
        }
        cv_.notify_all();
    }
    std::optional<int> acquire_wait(std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lk(m_);
        if (!cv_.wait_for(lk, timeout, [&] { return !free_.empty(); })) return std::nullopt;
        const int s = free_.front();
        free_.pop_front();
        return s;
    }
    std::vector<Completion> take_done() {
        std::lock_guard<std::mutex> lk(m_);
        std::vector<Completion> out(std::make_move_iterator(done_.begin()), std::make_move_iterator(done_.end()));
        done_.clear();
        return out;
    }

private:
    std::mutex m_;
    std::condition_variable cv_;
    std::deque<int> free_;
    std::deque<Completion> done_;
};

// main.cpp의 draw_and_save와 같은 그림(비교용 PNG가 같은 형식이어야 한다)
void draw_and_save(const std::string& path, const std::vector<std::uint8_t>& rgb, const yvo::LetterboxParams& lb,
                   const std::vector<yvo::Detection>& dets) {
    cv::Mat m(lb.dst_h, lb.dst_w, CV_8UC3, const_cast<std::uint8_t*>(rgb.data()));
    cv::Mat bgr;
    cv::cvtColor(m, bgr, cv::COLOR_RGB2BGR);
    for (const auto& d : dets) {
        const int x1 = static_cast<int>(std::lround(d.x1 * lb.scale + static_cast<float>(lb.pad_x)));
        const int y1 = static_cast<int>(std::lround(d.y1 * lb.scale + static_cast<float>(lb.pad_y)));
        const int x2 = static_cast<int>(std::lround(d.x2 * lb.scale + static_cast<float>(lb.pad_x)));
        const int y2 = static_cast<int>(std::lround(d.y2 * lb.scale + static_cast<float>(lb.pad_y)));
        const auto rgb_col = yvo::class_color(d.class_id);
        const cv::Scalar color(rgb_col[2], rgb_col[1], rgb_col[0]);
        cv::rectangle(bgr, cv::Point(x1, y1), cv::Point(x2, y2), color, 1);
        char label[64];
        std::snprintf(label, sizeof(label), "%s %.2f", class_name(d.class_id), static_cast<double>(d.score));
        cv::putText(bgr, label, cv::Point(x1, std::max(10, y1 - 3)), cv::FONT_HERSHEY_SIMPLEX, 0.35, color, 1);
    }
    if (!cv::imwrite(path, bgr)) std::fprintf(stderr, "[warn] PNG 저장 실패: %s\n", path.c_str());
}

const char* layout_name(yvo::TensorLayout l) { return l == yvo::TensorLayout::Nhcw ? "nhcw" : "nhwc"; }

int run(const Args& a) {
    const bool dmabuf = a.mode == Mode::Dmabuf;
    const bool nhcw = a.layout == yvo::TensorLayout::Nhcw;
    if (!dmabuf && nhcw) {
        std::fprintf(stderr, "--mode ptr --layout nhcw는 쓰지 않는다: NHCW는 HailoRT가 입력을 직접 DMA 매핑하므로\n"
                             "dma-buf로 넘겨야 사전 매핑(map_dmabuf)이 된다. --mode dmabuf를 쓸 것.\n");
        return 2;
    }
    const double soc_t0 = soc_temperature();

    yvo::FfmpegDrmDecoder dec;
    yvo::FfmpegDrmDecoder::Options dopt;
    dopt.extra_hw_frames = 2 * yvo::kNumSlots;
    if (!dec.open(a.video, dopt)) {
        std::fprintf(stderr, "decoder open failed: %s\n", dec.error().c_str());
        return 1;
    }
    const int src_w = dec.width(), src_h = dec.height();

    // 첫 프레임의 SAND 기하로 ISP 설정을 확정한다(프레임은 바로 놓고 디코더를 다시 연다).
    yvo::isp::IspResizerConfig icfg;
    icfg.media_device = a.media;
    icfg.branch = a.branch;
    icfg.num_slots = a.slots;
    icfg.buffer_count = std::max(12, a.slots);
    icfg.resample_filter = a.filter;
    {
        const auto first = dec.next();
        if (!first) {
            std::fprintf(stderr, "no frames in %s\n", a.video.c_str());
            return 1;
        }
        icfg.src_w = first->width;
        icfg.src_h = first->height;
        icfg.coded_height = first->coded_height;
        icfg.sand_col_height = static_cast<int>(first->sand_col_height);
        icfg.uv_sand_col_height = static_cast<int>(first->uv_sand_col_height);
    }
    dec.close();
    if (!dec.open(a.video, dopt)) {
        std::fprintf(stderr, "decoder reopen failed: %s\n", dec.error().c_str());
        return 1;
    }

    yvo::isp::IspResizer isp(icfg);

    yvo::HailoRunnerConfig hcfg;
    hcfg.hef_path = a.hef;
    hcfg.input_layout = a.layout;
    hcfg.batch_size = static_cast<std::uint16_t>(a.batch);
    hcfg.num_slots = a.slots;
    hcfg.score_threshold = a.score;
    auto runner = std::make_unique<yvo::HailoRunner>(hcfg);
    if (runner->input_frame_size() != isp.output_size()) {
        std::fprintf(stderr, "입력 크기 불일치: NPU %zu, ISP %zu\n", runner->input_frame_size(), isp.output_size());
        return 1;
    }

    // C안: NHCW 전치 결과를 담을 dma-heap 버퍼(슬롯마다 하나) + 사전 DMA 매핑
    std::unique_ptr<yvo::DmaHeapPool> nhcw_bufs;
    if (nhcw) {
        // core/dma_heap_alloc.hpp 의 풀을 쓴다. 2026-09-21 이전의 isp 전용 DmaHeapBuffers 에는
        // HailoRT read_async 가 요구하는 16384 정렬이 없었고 READ 방향 sync 도 없었다.
        yvo::DmaHeapConfig hc;
        hc.name = "yvo_nhcw";
        nhcw_bufs = std::make_unique<yvo::DmaHeapPool>(hc);
        for (int s = 0; s < a.slots; ++s) {
            const int i = nhcw_bufs->allocate(isp.output_size());
            // dma_map_dmabuf 에는 dma-buf 실크기를 준다(HailoRT 크기 계약)
            runner->map_dmabuf(nhcw_bufs->at(i).fd, nhcw_bufs->at(i).dmabuf_size);
        }
    } else if (dmabuf) {
        for (int s = 0; s < a.slots; ++s) runner->map_dmabuf(isp.dmabuf_fd(s), isp.output_size());
    }

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

    const float hailo_t0 = runner->chip_temperature();
    if (!a.quiet) {
        std::printf("[cfg] video=%s %dx%d mode=%s layout=%s repack=%s filter=%s batch=%d slots=%d isp=%s\n",
                    a.video.c_str(), src_w, src_h, dmabuf ? "dmabuf" : "ptr", layout_name(a.layout),
                    nhcw ? yvo::isp::repack_name(a.repack) : "-",
                    a.filter.empty() ? "smart" : a.filter.c_str(), a.batch, a.slots, a.media.c_str());
        if (nhcw_bufs)
            std::printf("[cfg] nhcw dma-heap: %s x%zu\n", nhcw_bufs->heap().c_str(), nhcw_bufs->count());
        std::fflush(stdout);
    }

    SlotExchange ex(a.slots);
    std::vector<double> decode_ms, pre_ms, repack_ms, npu_ms, e2e_ms, wait_block_ms;
    std::unordered_map<std::int64_t, Clock::time_point> t_decoded;
    std::map<std::int64_t, Completion> reorder;
    std::unordered_map<std::int64_t, std::vector<std::uint8_t>> png_src;
    std::unordered_map<std::int64_t, yvo::LetterboxParams> png_lb;
    std::int64_t next_pts = 0;
    long submitted_npu = 0, completed = 0, total_dets = 0, errors = 0;

    auto on_result = [&ex](yvo::InferResult&& r) { ex.complete(std::move(r)); };

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
            reorder.erase(reorder.begin());
            ++next_pts;
        }
    };

    const double c0 = cpu_ms();
    const auto t0 = Clock::now();
    long n = 0;
    bool fail = false;
    try {
        while (a.frames <= 0 || n < a.frames) {
            const auto d0 = Clock::now();
            auto f = dec.next();
            const auto d1 = Clock::now();
            if (!f) break;
            decode_ms.push_back(std::chrono::duration<double, std::milli>(d1 - d0).count());
            f->pts = n;
            t_decoded[n] = d1;

            const auto w0 = Clock::now();
            std::optional<int> slot = ex.acquire_wait(std::chrono::milliseconds(5000));
            if (!slot) throw std::runtime_error("슬롯 대기 타임아웃(5 s): HailoRT 콜백이 오지 않음");
            wait_block_ms.push_back(std::chrono::duration<double, std::milli>(Clock::now() - w0).count());

            yvo::PreprocOutput out = isp.process(*f, *slot);
            pre_ms.push_back(isp.last_isp_ms());
            f.reset(); // ISP Run()은 동기라 여기서 입력 프레임을 놓아도 된다

            const bool want_png = jsonl && out.pts % a.dump_every == 0;
            if (nhcw || want_png) {
                // ISP 출력은 캐시 매핑 dma-heap이다. CPU로 읽기 전에 DMA_BUF_IOCTL_SYNC로 감싼다.
                const yvo::isp::IspResizer::OutputRead lock(isp, *slot);
                if (nhcw) {
                    auto acc = nhcw_bufs->cpu_access(*slot, yvo::DmaSyncDir::Write);
                    const auto r0 = Clock::now();
                    yvo::isp::rgb_packed_to_nhcw(a.repack, lock.data(), out.lb.dst_w, out.lb.dst_h,
                                                 nhcw_bufs->at(*slot).ptr);
                    repack_ms.push_back(std::chrono::duration<double, std::milli>(Clock::now() - r0).count());
                    acc.finish();
                }
                if (want_png) {
                    std::vector<std::uint8_t>& buf = png_src[out.pts];
                    buf.assign(lock.data(), lock.data() + out.size);
                    png_lb[out.pts] = out.lb;
                }
            }

            if (nhcw) {
                out.layout = yvo::TensorLayout::Nhcw;
                out.dmabuf_fd = nhcw_bufs->at(*slot).fd;
                out.host_ptr = nhcw_bufs->at(*slot).ptr;
            } else if (!dmabuf) {
                out.dmabuf_fd = -1; // host 매핑 포인터로 제출
            }
            runner->submit(out, on_result);
            ++submitted_npu;

            flush_results();
            ++n;
            if (!a.quiet && n % 300 == 0) {
                const double el = std::chrono::duration<double>(Clock::now() - t0).count();
                std::printf("[progress] %ld frames, %.1f FPS, hailo %.1f C\n", n, static_cast<double>(n) / el,
                            static_cast<double>(runner->chip_temperature()));
                std::fflush(stdout);
            }
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] frame %ld: %s\n", n, e.what());
        fail = true;
    }
    runner->drain();
    flush_results();
    const auto t1 = Clock::now();
    const double c1 = cpu_ms();

    if (!dec.error().empty()) {
        std::fprintf(stderr, "decode error after %ld frames: %s\n", n, dec.error().c_str());
        fail = true;
    }
    const float hailo_t1 = runner->chip_temperature();
    // 종료 순서: drain(위) → ISP 입력 캐시 비움 → 디코더 close → 러너 해제(dma-buf unmap) → 버퍼 해제
    isp.clear_input_cache();
    dec.close();
    runner.reset();
    nhcw_bufs.reset();
    if (jsonl) std::fclose(jsonl);
    const double soc_t1 = soc_temperature();

    const double wall_s = std::chrono::duration<double>(t1 - t0).count();
    const double fps = static_cast<double>(completed) / wall_s;
    const double cpu_per_frame = completed ? (c1 - c0) / static_cast<double>(completed) : 0.0;
    const Stats sd = stats(decode_ms), sp = stats(pre_ms), sr = stats(repack_ms), sn = stats(npu_ms),
                se = stats(e2e_ms), sw = stats(wait_block_ms);
    if (static_cast<long>(completed) != n || static_cast<long>(completed) != submitted_npu) {
        std::fprintf(stderr, "[error] 프레임 수 불일치: decoded=%ld npu_submitted=%ld completed=%ld\n", n,
                     submitted_npu, completed);
        fail = true;
    }
    if (errors) fail = true;

    std::printf("== yvo_isp_pipeline (%s, %s%s, batch %d, slots %d) ==\n", dmabuf ? "dmabuf" : "ptr",
                layout_name(a.layout), nhcw ? (std::string("/") + yvo::isp::repack_name(a.repack)).c_str() : "",
                a.batch, a.slots);
    std::printf("frames          %ld (decoded %ld, errors %ld)\n", completed, n, errors);
    std::printf("wall            %.3f s, %.2f FPS\n", wall_s, fps);
    std::printf("cpu             %.3f ms/frame (process user+sys %.1f ms, %.0f%% of one core)\n", cpu_per_frame,
                c1 - c0, (c1 - c0) / (wall_s * 10.0));
    std::printf("decode call ms  mean %.3f  median %.3f  p90 %.3f  max %.3f\n", sd.mean, sd.median, sd.p90, sd.max);
    std::printf("isp wall ms     mean %.3f  median %.3f  p90 %.3f  max %.3f\n", sp.mean, sp.median, sp.p90, sp.max);
    if (sr.n)
        std::printf("repack ms       mean %.3f  median %.3f  p90 %.3f  max %.3f  (%s)\n", sr.mean, sr.median, sr.p90,
                    sr.max, yvo::isp::repack_name(a.repack));
    std::printf("npu latency ms  mean %.3f  median %.3f  p90 %.3f  max %.3f\n", sn.mean, sn.median, sn.p90, sn.max);
    std::printf("e2e latency ms  mean %.3f  median %.3f  p90 %.3f  max %.3f  (decode 반환 → NPU 완료 콜백)\n",
                se.mean, se.median, se.p90, se.max);
    std::printf("slot wait ms    mean %.3f  p90 %.3f\n", sw.mean, sw.p90);
    std::printf("detections      %ld total, %.2f / frame\n", total_dets,
                completed ? static_cast<double>(total_dets) / static_cast<double>(completed) : 0.0);
    std::printf("hailo chip temp %.1f -> %.1f C\n", static_cast<double>(hailo_t0), static_cast<double>(hailo_t1));
    std::printf("soc temp        %.1f -> %.1f C\n", soc_t0, soc_t1);
    if (!a.csv_label.empty()) {
        // yvo_pipeline과 같은 22열이다. ISP 행에서는 gpu_* 열이 ISP wall 시간을 담는다(전처리 시간 열).
        std::printf(
            "RESULT_CSV,%s,%s,%s,%d,%d,%ld,%.3f,%.2f,%.3f,%.3f,%.3f,%.3f,%.2f,%.2f,%.2f,%.2f,%ld,%.1f,%.1f,%.1f,%.1f,%d\n",
            a.csv_label.c_str(), dmabuf ? "dmabuf" : "ptr", layout_name(a.layout), a.batch, a.slots, completed, wall_s,
            fps, cpu_per_frame, sd.mean, sp.mean, sp.p90, sn.mean, sn.p90, se.mean, se.p90, total_dets,
            static_cast<double>(hailo_t0), static_cast<double>(hailo_t1), soc_t0, soc_t1, fail ? 0 : 1);
    }
    std::fflush(stdout);
    return fail ? 1 : 0;
}

} // namespace

int main(int argc, char** argv) {
    Args a;
    if (!parse(argc, argv, a)) {
        usage(argv[0]);
        return 2;
    }
    try {
        return run(a);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "fatal: %s\n", e.what());
        return 1;
    }
}
