// yvo_hailo_selftest: CPU letterbox (NV12 BT.709) → HailoRunner (run_async) verification tool
//
// 1) Read the first N frames of the video as NV12 raw via the ffmpeg CLI.
// 2) Build 640x640 RGB with cpu_letterbox_nv12_bt709 and run inference with the runner.
// 3) Print detections (class, score, source coordinates) and draw boxes on the first frame, saving it as PNG.
// 4) Throughput: cycle through slot buffers and submit --bench-frames frames back-to-back via run_async to measure FPS and latency.
//    Then submit one frame at a time sequentially (queue depth 1) to measure single-frame latency.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "core/coco_labels.hpp"
#include "ref/cpu_letterbox.hpp"
#include "core/frame_types.hpp"
#include "hailo/hailo_runner.hpp"
#include "toolkit/stats.hpp"
#include "toolkit/tool_util.hpp"

using yvo::percentile;

namespace {

using yvo::tool::Nv12Frame;
using yvo::tool::read_nv12_frames;

// COCO labels are consolidated in core/coco_labels.hpp (duplicate definitions removed).
using yvo::kCoco;

struct Options {
    std::string video = YVO_DEFAULT_VIDEO;
    std::string hef = YVO_DEFAULT_HEF;
    std::string out_dir = YVO_REPO_ROOT "/build/out/yvo_hailo_selftest";
    int frames = 8;          // Number of frames for detection check
    int bench_frames = 300;  // Number of frames for back-to-back submission measurement
    int warmup = 30;
    int seq_frames = 50;     // Number of frames for sequential (queue depth 1) latency measurement
    int batch = 1;
    int slots = 0;           // 0 = max(kNumSlots, 2*batch)
    float score = -1.0f;
    yvo::TensorLayout layout = yvo::TensorLayout::Nhwc;  // Input tensor layout passed to the runner
};

std::shared_ptr<std::uint8_t> alloc_aligned(std::size_t size) {
    const std::size_t rounded = (size + 4095) / 4096 * 4096;
    auto* p = static_cast<std::uint8_t*>(std::aligned_alloc(4096, rounded));
    if (!p) throw std::bad_alloc();
    return std::shared_ptr<std::uint8_t>(p, [](std::uint8_t* q) { std::free(q); });
}

// Pool of available slots (returned by the callback thread)
class SlotPool {
public:
    explicit SlotPool(int n) {
        for (int i = 0; i < n; ++i) free_.push_back(i);
    }
    int acquire() {
        std::unique_lock<std::mutex> lk(m_);
        cv_.wait(lk, [&] { return !free_.empty(); });
        int s = free_.front();
        free_.pop_front();
        return s;
    }
    void release(int s) {
        {
            std::lock_guard<std::mutex> lk(m_);
            free_.push_back(s);
        }
        cv_.notify_one();
    }

private:
    std::mutex m_;
    std::condition_variable cv_;
    std::deque<int> free_;
};

const char* layout_str(yvo::TensorLayout l) {
    return l == yvo::TensorLayout::Nhcw ? "nhcw" : "nhwc";
}

void usage() {
    std::cerr << "usage: yvo_hailo_selftest [--video P] [--hef P] [--out-dir D] [--frames N] [--bench-frames N]\n"
                 "                          [--warmup N] [--seq-frames N] [--batch B] [--slots S] [--score T]\n"
                 "                          [--layout nhwc|nhcw]\n";
}

} // namespace

int main(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                usage();
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--video") o.video = next();
        else if (a == "--hef") o.hef = next();
        else if (a == "--out-dir") o.out_dir = next();
        else if (a == "--frames") o.frames = std::stoi(next());
        else if (a == "--bench-frames") o.bench_frames = std::stoi(next());
        else if (a == "--warmup") o.warmup = std::stoi(next());
        else if (a == "--seq-frames") o.seq_frames = std::stoi(next());
        else if (a == "--batch") o.batch = std::stoi(next());
        else if (a == "--slots") o.slots = std::stoi(next());
        else if (a == "--score") o.score = std::stof(next());
        else if (a == "--layout") {
            const std::string v = next();
            if (v == "nhwc") o.layout = yvo::TensorLayout::Nhwc;
            else if (v == "nhcw") o.layout = yvo::TensorLayout::Nhcw;
            else {
                usage();
                return 2;
            }
        }
        else {
            usage();
            return 2;
        }
    }
    if (o.slots <= 0) o.slots = std::max(yvo::kNumSlots, 2 * o.batch);

    try {
        constexpr int kSrcW = 1920, kSrcH = 1080;
        const auto t_read0 = std::chrono::steady_clock::now();
        auto frames = read_nv12_frames(o.video, o.frames, kSrcW, kSrcH);
        if (frames.empty()) throw std::runtime_error("failed to read frames: " + o.video);
        const double read_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_read0).count();

        yvo::HailoRunnerConfig cfg;
        cfg.hef_path = o.hef;
        cfg.batch_size = static_cast<std::uint16_t>(o.batch);
        cfg.num_slots = o.slots;
        cfg.score_threshold = o.score;
        cfg.input_layout = o.layout;
        // This tool uses malloc (page-aligned) host buffers, so it attempts host_ptr submission even with NHCW.
        cfg.allow_host_ptr_nhcw = true;
        cfg.src_width = kSrcW;
        cfg.src_height = kSrcH;
        yvo::HailoRunner runner(cfg);

        const float temp_start = runner.chip_temperature();
        std::printf("[info] layout=%s batch=%d slots=%d async_queue_size=%zu in=%dx%d (%zu B) out=%zu B classes=%d max/cls=%d\n",
                    layout_str(o.layout), o.batch, o.slots, runner.async_queue_size(), runner.input_width(), runner.input_height(),
                    runner.input_frame_size(), runner.output_frame_size(), runner.num_classes(),
                    runner.max_bboxes_per_class());
        std::printf("[info] read %zu frames (NV12) in %.1f ms, chip temp start %.1f C\n", frames.size(), read_ms, temp_start);

        const auto lb = yvo::make_letterbox(kSrcW, kSrcH, runner.input_width(), runner.input_height());
        std::printf("[info] letterbox scale=%.6f pad_x=%d pad_y=%d\n", lb.scale, lb.pad_x, lb.pad_y);

        // Per-slot input buffers (page-aligned)
        std::vector<std::shared_ptr<std::uint8_t>> in_bufs;
        for (int s = 0; s < o.slots; ++s) in_bufs.push_back(alloc_aligned(runner.input_frame_size()));

        // ---- 1) Detection check ----
        std::vector<std::vector<std::uint8_t>> lb_frames(frames.size());
        double lb_ms_total = 0.0;
        for (std::size_t i = 0; i < frames.size(); ++i) {
            lb_frames[i].resize(runner.input_frame_size());
            const auto t0 = std::chrono::steady_clock::now();
            const std::uint8_t* y = frames[i].data.data();
            const std::uint8_t* uv = y + static_cast<std::size_t>(kSrcW) * kSrcH;
            yvo::cpu_letterbox_nv12_bt709(y, kSrcW, uv, kSrcW, kSrcW, kSrcH, lb, lb_frames[i].data());
            lb_ms_total += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        }
        std::printf("[info] cpu letterbox: %.2f ms/frame (reference only, scalar reference implementation)\n", lb_ms_total / frames.size());

        // Tensor fed to the runner. With NHCW, transpose packed RGB888 (lb_frames stays packed for the PNG dump).
        std::vector<std::vector<std::uint8_t>> nhcw_frames;
        if (o.layout == yvo::TensorLayout::Nhcw) {
            double t_ms = 0.0;
            nhcw_frames.resize(lb_frames.size());
            for (std::size_t i = 0; i < lb_frames.size(); ++i) {
                nhcw_frames[i].resize(runner.input_frame_size());
                const auto t0 = std::chrono::steady_clock::now();
                yvo::rgb_packed_to_nhcw(lb_frames[i].data(), runner.input_width(), runner.input_height(),
                                        nhcw_frames[i].data());
                t_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            }
            std::printf("[info] nhcw transpose: %.2f ms/frame (CPU reference implementation)\n", t_ms / lb_frames.size());
        }
        const auto& net_frames = (o.layout == yvo::TensorLayout::Nhcw) ? nhcw_frames : lb_frames;

        std::vector<yvo::InferResult> results(frames.size());
        SlotPool pool(o.slots);
        for (std::size_t i = 0; i < frames.size(); ++i) {
            const int slot = pool.acquire();
            std::memcpy(in_bufs[slot].get(), net_frames[i].data(), runner.input_frame_size());
            yvo::PreprocOutput in;
            in.host_ptr = in_bufs[slot].get();
            in.size = runner.input_frame_size();
            in.layout = o.layout;
            in.lb = lb;
            in.pts = static_cast<std::int64_t>(i);
            in.slot = slot;
            runner.submit(in, [&results, &pool](yvo::InferResult&& r) {
                const int s = r.slot;
                results[static_cast<std::size_t>(r.pts)] = std::move(r);
                pool.release(s);
            });
        }
        runner.drain();

        int bad = 0;
        for (const auto& r : results) {
            if (r.status != HAILO_SUCCESS) {
                ++bad;
                std::printf("[frame %lld] status=%d\n", static_cast<long long>(r.pts), static_cast<int>(r.status));
                continue;
            }
            int hist[80] = {};
            for (const auto& d : r.detections) hist[d.class_id]++;
            std::printf("[frame %lld] %zu dets:", static_cast<long long>(r.pts), r.detections.size());
            for (int c = 0; c < 80; ++c)
                if (hist[c]) std::printf(" %s=%d", kCoco[c], hist[c]);
            std::printf("\n");
            if (r.pts == 0) {
                auto sorted = r.detections;
                std::sort(sorted.begin(), sorted.end(), [](auto& a, auto& b) { return a.score > b.score; });
                for (std::size_t k = 0; k < std::min<std::size_t>(sorted.size(), 15); ++k) {
                    const auto& d = sorted[k];
                    std::printf("    %-14s %.3f  (%.1f, %.1f)-(%.1f, %.1f)\n", kCoco[d.class_id], d.score, d.x1, d.y1,
                                d.x2, d.y2);
                }
            }
        }

        // Save PNG: first source frame + boxes, letterbox input
        std::filesystem::create_directories(o.out_dir);
        {
            cv::Mat nv12(kSrcH * 3 / 2, kSrcW, CV_8UC1, frames[0].data.data());
            cv::Mat bgr;
            cv::cvtColor(nv12, bgr, cv::COLOR_YUV2BGR_NV12);
            for (const auto& d : results[0].detections) {
                const cv::Scalar color = cv::Scalar((d.class_id * 67) % 256, (d.class_id * 131 + 80) % 256, 255 - (d.class_id * 29) % 256);
                cv::rectangle(bgr, cv::Point(static_cast<int>(d.x1), static_cast<int>(d.y1)),
                              cv::Point(static_cast<int>(d.x2), static_cast<int>(d.y2)), color, 2);
                char label[64];
                std::snprintf(label, sizeof(label), "%s %.2f", kCoco[d.class_id], d.score);
                cv::putText(bgr, label, cv::Point(static_cast<int>(d.x1), std::max(15, static_cast<int>(d.y1) - 4)),
                            cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
            }
            const std::string p1 = o.out_dir + "/frame0_detections.png";
            cv::imwrite(p1, bgr);
            cv::Mat rgb(runner.input_height(), runner.input_width(), CV_8UC3, lb_frames[0].data());
            cv::Mat lb_bgr;
            cv::cvtColor(rgb, lb_bgr, cv::COLOR_RGB2BGR);
            const std::string p2 = o.out_dir + "/frame0_letterbox_input.png";
            cv::imwrite(p2, lb_bgr);
            std::printf("[info] saved %s, %s\n", p1.c_str(), p2.c_str());
        }

        // ---- 2) Back-to-back submission throughput ----
        // Pre-fill slot buffers with different letterbox frames and reuse them (excludes preprocessing cost, measures only the NPU path).
        for (int s = 0; s < o.slots; ++s)
            std::memcpy(in_bufs[s].get(), net_frames[s % net_frames.size()].data(), runner.input_frame_size());

        const int total = o.warmup + o.bench_frames;
        std::vector<double> lat(static_cast<std::size_t>(total), 0.0);
        std::vector<std::chrono::steady_clock::time_point> t_done(static_cast<std::size_t>(total));
        std::atomic<int> errors{0};
        std::chrono::steady_clock::time_point t_bench0;
        for (int i = 0; i < total; ++i) {
            const int slot = pool.acquire();
            if (i == o.warmup) t_bench0 = std::chrono::steady_clock::now();
            yvo::PreprocOutput in;
            in.host_ptr = in_bufs[slot].get();
            in.size = runner.input_frame_size();
            in.layout = o.layout;
            in.lb = lb;
            in.pts = i;
            in.slot = slot;
            runner.submit(in, [&, slot](yvo::InferResult&& r) {
                const auto idx = static_cast<std::size_t>(r.pts);
                lat[idx] = r.latency_ms;
                t_done[idx] = std::chrono::steady_clock::now();
                if (r.status != HAILO_SUCCESS) errors.fetch_add(1);
                pool.release(slot);
            });
        }
        runner.drain();
        const auto t_bench1 = *std::max_element(t_done.begin() + o.warmup, t_done.end());
        const double bench_s = std::chrono::duration<double>(t_bench1 - t_bench0).count();
        std::vector<double> lat_meas(lat.begin() + o.warmup, lat.end());
        const float temp_after_stream = runner.chip_temperature();
        std::printf("[stream] layout=%s batch=%d slots=%d frames=%d fps=%.2f latency(ms) median=%.2f p90=%.2f max=%.2f errors=%d chip_temp=%.1fC\n",
                    layout_str(o.layout), o.batch, o.slots, o.bench_frames, o.bench_frames / bench_s, percentile(lat_meas, 0.5),
                    percentile(lat_meas, 0.9), percentile(lat_meas, 1.0), errors.load(), temp_after_stream);

        // ---- 3) Sequential submission (queue depth 1) single-frame latency ----
        if (o.seq_frames > 0) {
            std::vector<double> seq_lat;
            for (int i = 0; i < o.seq_frames + 5; ++i) {
                yvo::PreprocOutput in;
                in.host_ptr = in_bufs[0].get();
                in.size = runner.input_frame_size();
                in.layout = o.layout;
                in.lb = lb;
                in.pts = i;
                in.slot = 0;
                double l = 0.0;
                runner.submit(in, [&l](yvo::InferResult&& r) { l = r.latency_ms; });
                runner.drain();
                if (i >= 5) seq_lat.push_back(l);
            }
            std::printf("[seq] batch=%d frames=%d latency(ms) median=%.2f p90=%.2f (batch>1 may be affected by scheduler timeout)\n",
                        o.batch, o.seq_frames, percentile(seq_lat, 0.5), percentile(seq_lat, 0.9));
        }
        std::printf("[info] chip temp end %.1f C\n", runner.chip_temperature());
        return bad == 0 && errors.load() == 0 ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 1;
    }
}
