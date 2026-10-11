// yvo_postproc_probe: verification probe for GPU pass 1 (`02_shaders/yolo_decode.comp`). Plan §6.2 P3.
//
// ── Modes ─────────────────────────────────────────────────────────────────────────────────────
//  --probe fixture (default, **no NPU needed**)
//      memcpy the `04_data/raw_tensors/*.bin` dumped by P1 into GPU buffers, run pass 1, and
//      compare against `cpu_yolov8_candidates()`. **All shader bugs are caught here (fast loop).**
//      P3-① (G5: identical candidate count / bit-exact scores / coordinates < 0.25 px)
//      P3-⑥ (median pass 1 GPU timestamp, **continuous stream** of N frames — default 300)
//      G6  (over 10 repeats of the same fixture, the candidate set is fully identical under the total order `(score, anchor_key)`)
//      The byte-sum (kChecksum pipeline) cross-check is also done here.
//  --probe sweep (no NPU needed)
//      local_size_x × local_size_y sweep. The plan's "measure and pick" discipline.
//  --probe nms (no NPU needed)
//      GPU NMS (efficient_nms.comp) cost alone: fixture continuous stream + synthetic worst case (run_nms()).
//  --probe live (**exclusive NPU**)
//      Allocate/export 6 tensors × slots with `DmaBufExporter` → HailoRT maps them D2H → receive the raw output
//      directly and feed it to pass 1. P3-② (byte sum computed by shader == sum computed by CPU).
//  --probe heapimport (**no NPU needed**, Appendix B step 1 = P0 gate)
//      Take 6 tensors from /dev/dma_heap, import them into Vulkan via `DmaBufImportCache::get()`,
//      record memoryTypeBits / req.size / dedicated flags / range, then check a CPU↔GPU byte-sum round trip.
//      **Whether V3DV imports a dma-heap fd** is the gate for all of Appendix B (B.14 S1).
//  --probe stale (**exclusive NPU**)
//      Receive frames A and B back to back into **the same slot buffer** (num_slots=1) and run pass 1 on each.
//      `--stale-barrier {on,off,both}` turns the FOREIGN_EXT→compute acquire barrier on / off / runs both.
//      P3-③ (A/B each correct with the barrier) + P3-④ (record whether it goes wrong without the barrier)
//
// ── Size contract (Appendix A.13) ─────────────────────────────────────────────────────────────
//   `dma_map_dmabuf(fd, size, D2H)` → **dmabuf_size** (covering the whole buffer is the safe side)
//   `read_async(fd, size, cb)` / `hailo_dma_buffer_t.size` → **hw_frame_size only**
//   ⚠ If `infer_async` is given a wrong size it returns `HAILO_SUCCESS` and **never invokes a single callback.**
//     So waiting for frame completion **must have a deadline**, and exceeding it is raised as a fatal error.
#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <unistd.h>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <fcntl.h>
#include <linux/dma-buf.h>
#include <linux/dma-heap.h>
#include <sys/ioctl.h>
#include <sys/mman.h>

#include "ref/cpu_letterbox.hpp"
#include "ref/cpu_yolov8_decode.hpp"
#include "core/dma_heap_alloc.hpp"
#include "core/frame_types.hpp"
#include "vulkan/dmabuf_export.hpp"
#include "vulkan/dmabuf_import.hpp"
#include "vulkan/vk_check.hpp"
#include "vulkan/vk_storage_headroom.hpp"
#include "vulkan/vulkan_context.hpp"
#include "vulkan/yolo_postproc.hpp"

#include "heap_sum_spv.h"
#include "toolkit/tool_util.hpp"
#include "testkit/nms_harness.hpp"
#include "testkit/postproc_fixture.hpp"
#include "toolkit/stats.hpp"

#ifdef YVO_HAVE_HAILO
#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>

#include <hailo/hailort.hpp>
#endif

namespace {

using yvo::tool::fmt;
using yvo::tool::count_open_fds;
using namespace yvo::fixture;

using yvo::DetectionKeyed;
using yvo::LetterboxParams;
using yvo::vk::kYoloNumScales;
using yvo::vk::kYoloNumTensors;
using yvo::vk::YoloInputAlloc;
using yvo::vk::YoloInputSync;
using yvo::vk::YoloPostproc;
using yvo::vk::YoloPostprocConfig;
using yvo::vk::YoloTensorLayout;
using yvo::vk::yolo_box_tensor;
using yvo::vk::yolo_cls_tensor;

constexpr int kSrcW = 1920;
constexpr int kSrcH = 1080;
constexpr float kHefScoreThreshold = 0.2f;   // YOLOV8 post-process setting of the HEF

// ─────────────────────────────────────────────────────────────────── output (stdout + file)

class Tee {
public:
    explicit Tee(const std::string& path) {
        if (!path.empty()) {
            std::filesystem::create_directories(std::filesystem::path(path).parent_path());
            f_.open(path, std::ios::trunc);
        }
    }
    void operator()(const std::string& s) {
        std::fputs(s.c_str(), stdout);
        std::fputc('\n', stdout);
        std::fflush(stdout);
        if (f_) f_ << s << '\n' << std::flush;
    }

private:
    std::ofstream f_;
};

// Capture the output of a one-line shell command. Defined in the heapimport section (gpuab uses it first to record temperature/throttling).
std::string run_capture(const char* cmd);

// Empty samples are printed as NaN ("no value") — to distinguish them from 0. The interpolation itself lives only in toolkit/stats.hpp.
double percentile(std::vector<double> v, double q) {
    return v.empty() ? std::nan("") : yvo::percentile(std::move(v), q);
}

// ─────────────────────────────────────────────────────────── minimal JSON reader (fixtures.json only)
//
// On this system nlohmann_json exists only in the conda prefix, so a single tool would pull in the conda include path
// (risk of shadowing other system headers). The schema is fixed, so we read only what we need by hand.
// Fixture reading and comparison live only in testkit/postproc_fixture.hpp (shared with the tests).

// Byte sum of the **valid region** the shader runs over (h < grid_h, c < features, w < grid_w). Same range as kChecksum.
std::uint32_t cpu_valid_sum(const std::uint8_t* p, int grid_h, int features, int grid_w, int w_padded) {
    std::uint32_t sum = 0;
    for (int h = 0; h < grid_h; ++h)
        for (int c = 0; c < features; ++c) {
            const std::uint8_t* row = p + yvo::nhcw_index(h, c, 0, features, w_padded);
            for (int w = 0; w < grid_w; ++w) sum += row[w];
        }
    return sum;
}

// ─────────────────────────────────────────────────────────────────── options

struct Options {
    std::string probe = "fixture";
    std::string fixture_dir = YVO_FIXTURE_DIR;
    std::string out_dir = YVO_REPO_ROOT "/build/out/yvo_postproc_probe";
    std::string out_file;                 // if empty, derived from the probe name
    std::string hef = YVO_DEFAULT_HEF;
    std::string video = YVO_DEFAULT_VIDEO;
    int perf_frames = 300;                // P3-⑥ (>= 200)
    int warmup = 30;
    int repeats = 10;                     // G6
    int live_frames = 20;
    int slots = 5;
    std::uint32_t lx = 10, ly = 6;   // kept equal to the measured defaults of YoloPostprocConfig
    int max_dets = 4096;
    std::string stale_barrier = "both";
    std::string heap;                     // heapimport: dma-heap path. If empty, vidbuf_cached → system fallback
    bool no_headroom = false;             // heapimport: disables the B.12 headroom (negative control to see that the enforced check really fires)
    // Appendix B steps 4·5: allocation path of the raw output buffers.
    //   plain = today's default path (fixture/gpuab → Plain, live/stale/v3 → ExportDmaBuf)
    //   heap  = ImportHeap (dma-heap allocation + Vulkan import)
    std::string alloc = "plain";
    std::string heap_sync = "on";         // on | off | both — DMA_BUF_IOCTL_SYNC of ImportHeap (B.10 R2/R3)
    int rounds = 3;                       // gpuab/stages: number of interleaved rounds
    std::string variants = "S1,S2,S3,Full";     // stages: which yolo_decode builds to run (Full is always added)
    int v3_window = 4096;                 // v3: per-tensor window size (B) used by line-dirty/stale-read
    std::string nms_cands = "16,64,256,1024";   // nms: synthetic candidate counts (1024 = kMaxSelected, worst case)
    float nms_score = kHefScoreThreshold;       // nms/stages: decode score threshold (--score = --nms-score)
    bool out_dir_set = false;             // whether --out-dir was given explicitly (decides the per-probe default path)
    std::string api = "read_async";       // live: read_async | infer_async
    std::uint16_t batch = 1;
    int timeout_ms = 10000;               // frame completion deadline (prevents the permanent hang of Appendix A.13)
    bool validation = false;
    bool sync_validation = false;
    bool verbose = false;
};

void usage() {
    std::puts(
        "yvo_postproc_probe --probe {fixture|sweep|nms|stages|groups|topk|live|stale|heapimport|gpuab|v3} [options]\n"
        "  --fixture-dir DIR      fixture directory (default <repo>/04_data/raw_tensors = YVO_FIXTURE_DIR)\n"
        "  --out-dir DIR          evidence file directory (default <repo>/build/out/yvo_postproc_probe)\n"
        "  --out-file NAME        evidence file name (default p3_<probe>.txt)\n"
        "  --perf-frames N        frames for continuous-stream measurement (default 300, P3-⑥ needs >= 200)\n"
        "  --warmup N             warm-up before measurement (default 30)\n"
        "  --repeats N            G6 determinism repeats (default 10)\n"
        "  --slots N              YoloPostproc slot count (default 5)\n"
        "  --local N,M            local_size_x,local_size_y (default 10,6; stages always uses the config default)\n"
        "  --max-dets N           candidate cap (default 4096)\n"
        "  --live-frames N        frame count in live mode (default 20)\n"
        "  --api NAME             live: read_async | infer_async (default read_async)\n"
        "  --batch N              live: HEF batch_size (default 1)\n"
        "  --timeout-ms N         frame completion deadline (default 10000)\n"
        "  --stale-barrier X      on | off | both (default both)\n"
        "  --alloc X              plain | heap  raw output allocation path (default plain, Appendix B step 4)\n"
        "  --heap-sync X          on | off | both  DMA_BUF_IOCTL_SYNC of ImportHeap (default on)\n"
        "  --rounds N             gpuab: plain↔heap / stages: per-variant interleaved rounds (default 3)\n"
        "  --variants LIST        stages: yolo_decode builds to run, comma-separated (default S1,S2,S3,Full;\n"
        "                         S1|S2|S3 = stage cuts, noatomic|norescan = ⑤ emit split, legacy|stop1_legacy =\n"
        "                         pre-argmax-shortcut build (+ multiset check vs Full); Full is always added)\n"
        "  --v3-window N          v3: per-tensor window size B for line-dirty/stale-read (default 4096)\n"
        "  --nms-cand LIST        nms: synthetic all-survive candidate counts, comma-separated (default 16,64,256,1024)\n"
        "  --score T              nms/stages: decode score threshold of the fixture stream (default 0.2 = HEF; alias --nms-score)\n"
        "  --heap PATH            heapimport: dma-heap node (default vidbuf_cached → system fallback)\n"
        "  --no-headroom          heapimport: set the B.12 headroom to 0 (negative control — conv71_178 is expected to throw)\n"
        "  --validation           explicitly enable the validation layer in the context\n"
        "  --sync-validation      synchronization validation (together with --validation)\n"
        "  --verbose\n");
}

// ─────────────────────────────────────────────────────────────── fixture / sweep

struct FixtureRig {
    FixtureSet fs;
    YoloTensorLayout layout;
    std::vector<FrameTensors> frames;
};

FixtureRig load_rig(const Options& o) {
    FixtureRig r;
    r.fs = load_fixtures(o.fixture_dir);
    r.layout = layout_from_fixtures(r.fs);
    for (int i = 0; i < r.fs.frames; ++i) r.frames.push_back(load_frame(r.fs, i));
    return r;
}

// ── Appendix B.10 protocol helpers ────────────────────────────────────────────────────────────
// Opens an R2 (CPU read) section via RAII. Does nothing unless ImportHeap + heap_sync.
class RawRead {
public:
    RawRead(YoloPostproc& pp, int slot) : pp_(pp), slot_(slot) {
        pp_.raw_cpu_sync_begin(slot_, yvo::DmaSyncDir::Read);
    }
    ~RawRead() {
        try {
            pp_.raw_cpu_sync_end(slot_, yvo::DmaSyncDir::Read);
        } catch (...) {
        }
    }
    RawRead(const RawRead&) = delete;
    RawRead& operator=(const RawRead&) = delete;

private:
    YoloPostproc& pp_;
    int slot_;
};

// Forces a sync **regardless of settings**. Used when V3 (the 0xA5 harness) takes the reference snapshot —
// because the reference must be correct even in the `--heap-sync off` cell.
void forced_sync(YoloPostproc& pp, int slot, yvo::DmaSyncDir dir, bool start) {
    for (int t = 0; t < kYoloNumTensors; ++t) {
        const int fd = pp.raw_dmabuf_fd(slot, t);
        if (start)
            yvo::dma_sync_begin(fd, dir, /*strict=*/true);
        else
            yvo::dma_sync_end(fd, dir, /*strict=*/true);
    }
}

// Fill the slot buffers with frame data (keeping one frame resident per slot makes the continuous-stream measurement clean).
// **Appendix B.10 R3**: CPU writes that dirty the cached mapping are wrapped in START|WRITE … END|WRITE.
// Without it, dirty cache lines are evicted later and **overwrite the data written by the device** (B.6(c) defect type).
void fill_slot(YoloPostproc& pp, int slot, const FrameTensors& f) {
    pp.raw_cpu_sync_begin(slot, yvo::DmaSyncDir::Write);
    for (int t = 0; t < kYoloNumTensors; ++t)
        std::memcpy(pp.raw_host_ptr(slot, t), f[static_cast<std::size_t>(t)].data(),
                    f[static_cast<std::size_t>(t)].size());
    pp.raw_cpu_sync_end(slot, yvo::DmaSyncDir::Write);
}

// --alloc string → YoloInputAlloc. The meaning of plain differs per probe (fixture=Plain, live=ExportDmaBuf).
YoloInputAlloc alloc_of(const std::string& a, YoloInputAlloc plain_means) {
    if (a == "heap") return YoloInputAlloc::ImportHeap;
    if (a == "plain") return plain_means;
    throw std::runtime_error("--alloc must be plain | heap: " + a);
}

std::vector<bool> heap_sync_cells(const std::string& v) {
    if (v == "on") return {true};
    if (v == "off") return {false};
    if (v == "both") return {true, false};
    throw std::runtime_error("--heap-sync must be on | off | both: " + v);
}

int run_fixture(const Options& o, Tee& log) {
    FixtureRig rig = load_rig(o);
    log(fmt("[fixture] dir=%s frames=%d  letterbox scale=%.9g pad=(%d,%d) src=%dx%d", o.fixture_dir.c_str(),
            rig.fs.frames, static_cast<double>(rig.fs.lb.scale), rig.fs.lb.pad_x, rig.fs.lb.pad_y, rig.fs.src_w,
            rig.fs.src_h));

    yvo::vk::VulkanContextOptions co;
    co.enable_validation = o.validation;
    co.validation_sync = o.sync_validation;
    co.verbose = o.verbose;
    yvo::vk::VulkanContext ctx(co);
    log(fmt("[vk] device=%s  timestamps=%d period=%.3f ns valid_bits=%u", ctx.device_name(),
            ctx.timestamps_supported() ? 1 : 0, static_cast<double>(ctx.timestamp_period()),
            ctx.timestamp_valid_bits()));

    YoloPostprocConfig pc;
    pc.num_slots = std::max(1, std::min(o.slots, rig.fs.frames));
    pc.max_dets = o.max_dets;
    pc.local_size_x = o.lx;
    pc.local_size_y = o.ly;
    pc.score_thr = kHefScoreThreshold;
    pc.input_alloc = alloc_of(o.alloc, YoloInputAlloc::Plain);
    pc.input_sync = YoloInputSync::HostWrite;
    pc.heap_sync = heap_sync_cells(o.heap_sync).front();
    pc.checksum_pipeline = true;
    pc.verbose = o.verbose;
    YoloPostproc pp(ctx, rig.layout, pc);
    log(fmt("[cfg] slots=%d max_dets=%d local=%ux%u  q_thr_low = P3 %u / P4 %u / P5 %u", pp.num_slots(),
            pc.max_dets, pc.local_size_x, pc.local_size_y, pp.q_thr_low(0), pp.q_thr_low(1), pp.q_thr_low(2)));
    log(fmt("[cfg] alloc=%s heap_sync=%s headroom=%zu B", o.alloc.c_str(),
            pp.raw_heap_sync_active() ? "on" : (pc.input_alloc == YoloInputAlloc::ImportHeap ? "off" : "n/a"),
            pp.heap_headroom()));

    const int nslots = pp.num_slots();
    for (int s = 0; s < nslots; ++s) fill_slot(pp, s, rig.frames[static_cast<std::size_t>(s)]);

    // ── P3-① : G5 (per-frame comparison against the CPU reference) ───────────────────────────
    log("");
    log("== P3-(1) G5: GPU pass 1 vs cpu_yolov8_candidates() ==");
    log("frame  cpu  gpu  matched  onlyCPU  onlyGPU  score_bit_mismatch  max_coord_err_px  overflow");
    bool g5_ok = true;
    float worst_coord = 0.0f;
    std::size_t total_cpu = 0, total_gpu = 0;
    std::vector<std::string> canon_ref(static_cast<std::size_t>(nslots));
    for (int s = 0; s < nslots; ++s) {
        pp.submit(s, rig.fs.lb, rig.fs.src_w, rig.fs.src_h);
        pp.wait(s);
        const auto gpu = pp.read_candidates(s);
        const std::uint8_t* ptr[kYoloNumTensors];
        for (int t = 0; t < kYoloNumTensors; ++t) ptr[t] = pp.raw_host_ptr(s, t);
        std::vector<DetectionKeyed> cpu;
        {
            const RawRead rr(pp, s);   // R2: section where the CPU reads the cached mapping
            cpu = yvo::cpu_yolov8_candidates(make_raw_set(rig.layout, ptr), kHefScoreThreshold, rig.fs.lb,
                                             rig.fs.src_w, rig.fs.src_h);
        }
        const Compare c = compare_candidates(cpu, gpu);
        log(fmt("%5d  %3zu  %3zu  %7zu  %7zu  %7zu  %18zu  %16.6f  %8u", s, c.cpu, c.gpu, c.matched, c.only_cpu,
                c.only_gpu, c.score_bit_mismatch, static_cast<double>(c.max_coord_err), pp.overflow(s)));
        g5_ok = g5_ok && c.ok() && pp.overflow(s) == 0;
        worst_coord = std::max(worst_coord, c.max_coord_err);
        total_cpu += c.cpu;
        total_gpu += c.gpu;
        canon_ref[static_cast<std::size_t>(s)] = canonical(gpu);
    }
    log(fmt("[G5] %s  total candidates cpu=%zu gpu=%zu  max coordinate error %.6f px (limit < %.2f)",
            g5_ok ? "PASS" : "FAIL", total_cpu, total_gpu, static_cast<double>(worst_coord),
            static_cast<double>(kCoordTolPx)));

    // ── G6 : determinism ─────────────────────────────────────────────────────────────────────
    log("");
    log(fmt("== G6: same fixture repeated %d times, fully identical after sorting by total order (score, anchor_key) ==", o.repeats));
    bool g6_ok = true;
    for (int rep = 1; rep < o.repeats; ++rep) {
        for (int s = 0; s < nslots; ++s) {
            pp.submit(s, rig.fs.lb, rig.fs.src_w, rig.fs.src_h);
            pp.wait(s);
            if (canonical(pp.read_candidates(s)) != canon_ref[static_cast<std::size_t>(s)]) {
                log(fmt("[G6] FAIL: rep=%d slot=%d result differs from the first run", rep, s));
                g6_ok = false;
            }
        }
    }
    log(fmt("[G6] %s (%d runs x %d frames)", g6_ok ? "PASS" : "FAIL", o.repeats, nslots));

    // ── byte sum (kChecksum pipeline) ───────────────────────────────────────────────────────
    log("");
    log("== byte sum: shader (kChecksum) vs CPU (valid region only, excluding P5 columns 20~23) ==");
    log("slot tensor      shader_sum         cpu_sum  match");
    bool sum_ok = true;
    const auto order = tensor_order(rig.fs);
    for (int s = 0; s < nslots; ++s) {
        pp.submit(s, rig.fs.lb, rig.fs.src_w, rig.fs.src_h, /*checksum=*/true);
        pp.wait(s);
        std::uint32_t shader[kYoloNumTensors];
        pp.checksums(s, shader);
        const RawRead rr(pp, s);   // R2
        for (int t = 0; t < kYoloNumTensors; ++t) {
            const FixtureStream& m = *order[static_cast<std::size_t>(t)];
            const std::uint32_t cs = cpu_valid_sum(pp.raw_host_ptr(s, t), m.grid_h, m.features, m.grid_w,
                                                   m.w_padded);
            const bool eq = cs == shader[t];
            sum_ok = sum_ok && eq;
            if (s == 0 || !eq)
                log(fmt("%4d %6d  %14u  %14u  %s", s, t, shader[t], cs, eq ? "ok" : "MISMATCH"));
        }
    }
    log(fmt("[checksum] %s", sum_ok ? "PASS" : "FAIL"));

    // ── hybrid (§4.4 A) host-side cost: candidate read + greedy NMS ──────────────────────────
    log("");
    log("== hybrid (plan §4.4 A): GPU candidates → host read + cpu_nms_greedy_by_class ==");
    log("frame  cand  read_us  nms_us  final_dets  cpu_ref_dets  match");
    bool hyb_ok = true;
    for (int s = 0; s < nslots; ++s) {
        pp.submit(s, rig.fs.lb, rig.fs.src_w, rig.fs.src_h);
        pp.wait(s);
        const auto t0 = std::chrono::steady_clock::now();
        auto cand = pp.read_candidates(s);
        const auto t1 = std::chrono::steady_clock::now();
        const std::size_t n_before = cand.size();
        yvo::cpu_nms_greedy_by_class(cand, 0.7f, 100);   // HEF settings: IoU 0.7, 100 per class
        const auto t2 = std::chrono::steady_clock::now();
        const std::uint8_t* ptr[kYoloNumTensors];
        for (int t = 0; t < kYoloNumTensors; ++t) ptr[t] = pp.raw_host_ptr(s, t);
        std::vector<yvo::Detection> ref;
        {
            const RawRead rr(pp, s);   // R2
            ref = yvo::cpu_yolov8_postprocess(make_raw_set(rig.layout, ptr), kHefScoreThreshold, 0.7f, 100,
                                              rig.fs.lb, rig.fs.src_w, rig.fs.src_h);
        }
        const bool eq = cand.size() == ref.size();
        hyb_ok = hyb_ok && eq;
        log(fmt("%5d  %4zu  %7.1f  %6.1f  %10zu  %12zu  %s", s, n_before,
                std::chrono::duration<double, std::micro>(t1 - t0).count(),
                std::chrono::duration<double, std::micro>(t2 - t1).count(), cand.size(), ref.size(),
                eq ? "ok" : "MISMATCH"));
    }
    log(fmt("[hybrid] final detection count %s (checks the plan §4.4 claim of '5~20 µs' against read_us+nms_us above)",
            hyb_ok ? "PASS" : "FAIL"));

    // ── P3-⑥ : continuous-stream pass 1 GPU time ────────────────────────────────────────────
    log("");
    log(fmt("== P3-(6) pass 1 GPU timestamp, continuous stream of %d frames (warm-up %d) ==", o.perf_frames, o.warmup));
    std::vector<double> pass1, total;
    for (int i = 0; i < o.warmup + o.perf_frames; ++i) {
        const int s = i % nslots;
        pp.submit(s, rig.fs.lb, rig.fs.src_w, rig.fs.src_h);
        pp.wait(s);
        if (i >= o.warmup) {
            pass1.push_back(pp.last_pass1_ms());
            total.push_back(pp.last_total_ms());
        }
    }
    log(fmt("pass1_ms  median %.4f  p90 %.4f  min %.4f  max %.4f  n=%zu", percentile(pass1, 0.5),
            percentile(pass1, 0.9), *std::min_element(pass1.begin(), pass1.end()),
            *std::max_element(pass1.begin(), pass1.end()), pass1.size()));
    log(fmt("cmdbuf_ms median %.4f  p90 %.4f", percentile(total, 0.5), percentile(total, 0.9)));
    log(fmt("[§4.7 estimate 0.4~0.9 ms] measured median %.4f ms → %s", percentile(pass1, 0.5),
            (percentile(pass1, 0.5) >= 0.4 && percentile(pass1, 0.5) <= 0.9) ? "within estimated range"
                                                                            : "outside estimated range (state it in the report)"));

    log("");
    log(fmt("[validation] errors=%llu warnings=%llu",
            static_cast<unsigned long long>(ctx.validation_errors()),
            static_cast<unsigned long long>(ctx.validation_warnings())));
    const bool pass = g5_ok && g6_ok && sum_ok && hyb_ok && ctx.validation_errors() == 0;
    log(fmt("[RESULT] fixture %s", pass ? "PASS" : "FAIL"));
    return pass ? 0 : 1;
}

int run_sweep(const Options& o, Tee& log) {
    FixtureRig rig = load_rig(o);
    yvo::vk::VulkanContextOptions co;
    co.verbose = o.verbose;
    yvo::vk::VulkanContext ctx(co);
    const std::uint32_t maxinv = ctx.properties().limits.maxComputeWorkGroupInvocations;
    log(fmt("[sweep] maxComputeWorkGroupInvocations=%u, frames %d, repeats %d (3 interleaved rounds)", maxinv,
            o.perf_frames, 3));
    log("local_x local_y  inv   pass1_median_ms  p90     cand/frame");

    // x is the word column, so 20(P3)/10(P4)/5(P5) is the upper bound. Together with y, keep the invocation count at 256 or less.
    const std::vector<std::pair<std::uint32_t, std::uint32_t>> combos{
        {5, 4},   {5, 8},   {5, 12},  {5, 48},  {8, 4},   {8, 8},   {10, 2},  {10, 4},
        {10, 6},  {10, 8},  {10, 12}, {10, 16}, {10, 24}, {12, 5},  {16, 2},  {16, 4},
        {16, 8},  {16, 16}, {20, 3},  {20, 6},  {20, 12}, {24, 10}, {32, 8},  {40, 6},
        {64, 4},  {80, 3}};
    struct Row {
        std::uint32_t x, y;
        std::vector<double> ms;
        std::size_t cand = 0;
    };
    std::vector<Row> rows;
    for (const auto& c : combos)
        if (c.first * c.second <= maxinv && c.first <= ctx.properties().limits.maxComputeWorkGroupSize[0] &&
            c.second <= ctx.properties().limits.maxComputeWorkGroupSize[1])
            rows.push_back(Row{c.first, c.second, {}, 0});

    for (int round = 0; round < 3; ++round) {   // interleaving: rules out DVFS drift
        for (Row& r : rows) {
            YoloPostprocConfig pc;
            pc.num_slots = std::max(1, std::min(o.slots, rig.fs.frames));
            pc.max_dets = o.max_dets;
            pc.local_size_x = r.x;
            pc.local_size_y = r.y;
            pc.score_thr = kHefScoreThreshold;
            YoloPostproc pp(ctx, rig.layout, pc);
            for (int s = 0; s < pp.num_slots(); ++s) fill_slot(pp, s, rig.frames[static_cast<std::size_t>(s)]);
            for (int i = 0; i < o.warmup + o.perf_frames; ++i) {
                const int s = i % pp.num_slots();
                pp.submit(s, rig.fs.lb, rig.fs.src_w, rig.fs.src_h);
                pp.wait(s);
                if (i >= o.warmup) r.ms.push_back(pp.last_pass1_ms());
            }
            if (round == 0) r.cand = pp.read_candidates(0).size();
        }
    }
    for (const Row& r : rows)
        log(fmt("%7u %7u %4u  %15.4f  %6.4f  %zu", r.x, r.y, r.x * r.y, percentile(r.ms, 0.5),
                percentile(r.ms, 0.9), r.cand));
    const Row* best = nullptr;
    for (const Row& r : rows)
        if (!best || percentile(r.ms, 0.5) < percentile(best->ms, 0.5)) best = &r;
    if (best) log(fmt("[best] local=%ux%u  median %.4f ms", best->x, best->y, percentile(best->ms, 0.5)));
    log(fmt("[validation] errors=%llu", static_cast<unsigned long long>(ctx.validation_errors())));
    return 0;
}


// ───────────────────────────────────────── gpuab (Appendix B step 4: GPU read A/B, **no NPU needed**)
//
// Within the same run, `plain` (Vulkan memory) and `heap` (dma-heap + Vulkan import) are run **interleaved**
// to compare pass 1 GPU time. The verdict is **not absolute ms but the ratio within the same run** —
// insensitive to DVFS/temperature drift (Appendix B.13 acceptance bands).
//   heap ≤ plain ×1.05  → A: no penalty (buffers read by the GPU are also heap candidates)
//   ×1.05 ~ ×1.30       → B: partial penalty (heap only for "buffers the CPU reads heavily")
//   > ×1.30             → C: penalty (S3. not used for GPU-accessed buffers)
// **Correctness (G5) is judged in the same run at the same time.** If it fails here, the performance numbers are not worth looking at (S2).
//
// `cpu_read_1.22MB` is measured as well (plain/heap/malloc three-way, same run, interleaved) — in B.18
// the cause of heap improving 0.396 → 0.292 ms is unconfirmed, so the conditions are unified to settle it again.

struct AbMode {
    const char* name = "";
    YoloInputAlloc alloc = YoloInputAlloc::Plain;
    std::unique_ptr<YoloPostproc> pp;
    std::vector<double> pass1;
    std::vector<double> cmdbuf;
    std::vector<double> round_median;   // per-round median (for the ratio computation)
    std::vector<double> cpu_read_ms;
    bool g5_ok = false;
    std::size_t g5_cpu = 0, g5_gpu = 0;
    float g5_coord = 0.0f;
    std::size_t g5_score_bits = 0;
    bool sum_ok = false;
    std::string canon;                  // canonical form of the candidate set (for plain↔heap cross-check)
};

// G5 + byte-sum verification for one mode. The same fixture frames are used, so the plain and heap results
// **must be bit-identical** — memory type is a visibility/ordering issue, not a value issue (B.7).
void ab_correctness(AbMode& m, FixtureRig& rig, const std::vector<const FixtureStream*>& order, Tee& log) {
    YoloPostproc& pp = *m.pp;
    const int nslots = pp.num_slots();
    m.g5_ok = true;
    m.sum_ok = true;
    std::string canon_all;
    for (int s = 0; s < nslots; ++s) {
        pp.submit(s, rig.fs.lb, rig.fs.src_w, rig.fs.src_h);
        pp.wait(s);
        const auto gpu = pp.read_candidates(s);
        const std::uint8_t* ptr[kYoloNumTensors];
        for (int t = 0; t < kYoloNumTensors; ++t) ptr[t] = pp.raw_host_ptr(s, t);
        std::vector<DetectionKeyed> cpu;
        {
            const RawRead rr(pp, s);   // R2
            cpu = yvo::cpu_yolov8_candidates(make_raw_set(rig.layout, ptr), kHefScoreThreshold, rig.fs.lb,
                                             rig.fs.src_w, rig.fs.src_h);
        }
        const Compare c = compare_candidates(cpu, gpu);
        m.g5_ok = m.g5_ok && c.ok() && pp.overflow(s) == 0;
        m.g5_cpu += c.cpu;
        m.g5_gpu += c.gpu;
        m.g5_score_bits += c.score_bit_mismatch;
        m.g5_coord = std::max(m.g5_coord, c.max_coord_err);
        canon_all += canonical(gpu) + "|";
        log(fmt("  [%s] slot %d  cpu=%zu gpu=%zu matched=%zu onlyCPU=%zu onlyGPU=%zu score_bit_mismatch=%zu "
                "max_coord_err=%.6f px overflow=%u",
                m.name, s, c.cpu, c.gpu, c.matched, c.only_cpu, c.only_gpu, c.score_bit_mismatch,
                static_cast<double>(c.max_coord_err), pp.overflow(s)));

        // byte sum (kChecksum): bytes the shader actually read == bytes the CPU sees through the mapping
        pp.submit(s, rig.fs.lb, rig.fs.src_w, rig.fs.src_h, /*checksum=*/true);
        pp.wait(s);
        std::uint32_t shader[kYoloNumTensors];
        pp.checksums(s, shader);
        const RawRead rr(pp, s);   // R2
        for (int t = 0; t < kYoloNumTensors; ++t) {
            const FixtureStream& mm = *order[static_cast<std::size_t>(t)];
            const std::uint32_t cs =
                cpu_valid_sum(pp.raw_host_ptr(s, t), mm.grid_h, mm.features, mm.grid_w, mm.w_padded);
            if (cs != shader[t]) {
                m.sum_ok = false;
                log(fmt("  [%s] slot %d tensor %d byte sum mismatch: shader %u != cpu %u", m.name, s, t,
                        shader[t], cs));
            }
        }
    }
    m.canon = canon_all;
}

// Time (ms) for the CPU to read all 6 tensors with sequential memcpy. **Includes** the sync cost
// (same method as read_all_ms of hailo_raw_probe — aligned so it is comparable with P2-⑥).
double cpu_read_all_ms(YoloPostproc& pp, int slot, std::uint8_t* dst) {
    const auto t0 = std::chrono::steady_clock::now();
    pp.raw_cpu_sync_begin(slot, yvo::DmaSyncDir::Read);
    std::size_t off = 0;
    for (int t = 0; t < kYoloNumTensors; ++t) {
        const std::size_t n = pp.raw_size(t);
        std::memcpy(dst + off, pp.raw_host_ptr(slot, t), n);
        off += n;
    }
    pp.raw_cpu_sync_end(slot, yvo::DmaSyncDir::Read);
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

int run_gpuab(const Options& o, Tee& log) {
    FixtureRig rig = load_rig(o);
    const auto order = tensor_order(rig.fs);
    log(fmt("[gpuab] dir=%s frames=%d  interleaved %d rounds x (warm-up %d + measure %d)", o.fixture_dir.c_str(),
            rig.fs.frames, o.rounds, o.warmup, o.perf_frames));
    log(fmt("[env] SoC %s  throttled %s", run_capture("vcgencmd measure_temp").c_str(),
            run_capture("vcgencmd get_throttled").c_str()));

    yvo::vk::VulkanContextOptions co;
    co.enable_validation = o.validation;
    co.validation_sync = o.sync_validation;
    co.verbose = o.verbose;
    yvo::vk::VulkanContext ctx(co);
    log(fmt("[vk] device=%s timestamps=%d period=%.3f ns", ctx.device_name(),
            ctx.timestamps_supported() ? 1 : 0, static_cast<double>(ctx.timestamp_period())));

    // fd leak baseline: fds opened by VulkanContext are not counted (the context outlives this block).
    const long fd_base = count_open_fds();

    std::array<AbMode, 2> modes{};
    modes[0].name = "plain";
    modes[0].alloc = YoloInputAlloc::Plain;
    modes[1].name = "heap";
    modes[1].alloc = YoloInputAlloc::ImportHeap;
    for (AbMode& m : modes) {
        YoloPostprocConfig pc;
        pc.num_slots = std::max(1, std::min(o.slots, rig.fs.frames));
        pc.max_dets = o.max_dets;
        pc.local_size_x = o.lx;
        pc.local_size_y = o.ly;
        pc.score_thr = kHefScoreThreshold;
        pc.input_alloc = m.alloc;
        pc.input_sync = YoloInputSync::HostWrite;
        pc.heap_sync = true;
        pc.checksum_pipeline = true;
        pc.verbose = o.verbose;
        m.pp = std::make_unique<YoloPostproc>(ctx, rig.layout, pc);
        for (int s = 0; s < m.pp->num_slots(); ++s) fill_slot(*m.pp, s, rig.frames[static_cast<std::size_t>(s)]);
        log(fmt("[%s] slots=%d  headroom=%zu B  heap_sync=%s  tensor0 range=%zu", m.name, m.pp->num_slots(),
                m.pp->heap_headroom(), m.pp->raw_heap_sync_active() ? "on" : "n/a",
                static_cast<std::size_t>(m.pp->raw_size(0))));
    }

    // ── ① correctness (G5 + byte sum) — judged before performance ─────────────────────────────
    log("");
    log("== ① G5 correctness (same run, same fixture) ==");
    for (AbMode& m : modes) ab_correctness(m, rig, order, log);
    const bool cross_same = modes[0].canon == modes[1].canon;
    for (const AbMode& m : modes)
        log(fmt("[G5 %s] %s  candidates cpu=%zu gpu=%zu  max coordinate error %.6f px (limit < %.2f)  score bit mismatches %zu  "
                "byte sum %s",
                m.name, m.g5_ok ? "PASS" : "FAIL", m.g5_cpu, m.g5_gpu, static_cast<double>(m.g5_coord),
                static_cast<double>(kCoordTolPx), m.g5_score_bits, m.sum_ok ? "match" : "mismatch"));
    log(fmt("[cross] are the plain and heap candidate sets byte-identical: %s",
            cross_same ? "yes (memory type is not a value issue — B.7)" : "**no — hard stop (S2)**"));
    const bool acc_ok = modes[0].g5_ok && modes[1].g5_ok && modes[0].sum_ok && modes[1].sum_ok && cross_same;
    if (!acc_ok) {
        log("[S2] correctness failure — performance numbers are not examined. Fix with the step-5 sync protocol and re-measure.");
        log("[RESULT] gpuab FAIL");
        return 1;
    }

    // ── ② performance: plain ↔ heap interleaved ──────────────────────────────────────────────
    log("");
    log(fmt("== ② pass 1 GPU timestamp — plain ↔ heap interleaved %d rounds (continuous stream) ==", o.rounds));
    log("round  plain_median_ms  heap_median_ms  heap/plain");
    for (int round = 0; round < o.rounds; ++round) {
        for (AbMode& m : modes) {
            std::vector<double> r;
            YoloPostproc& pp = *m.pp;
            const int ns = pp.num_slots();
            for (int i = 0; i < o.warmup + o.perf_frames; ++i) {
                const int s = i % ns;
                pp.submit(s, rig.fs.lb, rig.fs.src_w, rig.fs.src_h);
                pp.wait(s);
                if (i >= o.warmup) {
                    r.push_back(pp.last_pass1_ms());
                    m.pass1.push_back(pp.last_pass1_ms());
                    m.cmdbuf.push_back(pp.last_total_ms());
                }
            }
            m.round_median.push_back(percentile(r, 0.5));
        }
        log(fmt("%5d  %15.4f  %14.4f  %10.4f", round, modes[0].round_median.back(),
                modes[1].round_median.back(), modes[1].round_median.back() / modes[0].round_median.back()));
    }
    for (const AbMode& m : modes)
        log(fmt("[%s] pass1_ms median %.4f  p90 %.4f  min %.4f  max %.4f  n=%zu | cmdbuf median %.4f", m.name,
                percentile(m.pass1, 0.5), percentile(m.pass1, 0.9),
                *std::min_element(m.pass1.begin(), m.pass1.end()),
                *std::max_element(m.pass1.begin(), m.pass1.end()), m.pass1.size(),
                percentile(m.cmdbuf, 0.5)));

    // Use the median of per-round ratios (temperature conditions are nearly the same within a round)
    std::vector<double> ratios;
    for (int r = 0; r < o.rounds; ++r) ratios.push_back(modes[1].round_median[r] / modes[0].round_median[r]);
    const double ratio = percentile(ratios, 0.5);
    const double ratio_all = percentile(modes[1].pass1, 0.5) / percentile(modes[0].pass1, 0.5);
    const char* band = ratio <= 1.05 ? "A: no penalty — buffers read by the GPU are also heap candidates"
                                     : (ratio <= 1.30 ? "B: partial penalty — heap only for 'buffers the CPU reads heavily'"
                                                      : "C: penalty (S3) — not used for GPU-accessed buffers");
    log(fmt("[ratio] per-round heap/plain median = %.4f (%.4f based on overall medians)", ratio, ratio_all));
    log(fmt("[band verdict] %s", band));

    // ── ③ cpu_read_1.22MB: plain / heap / malloc three-way, same-run interleaved ────────────
    log("");
    log(fmt("== ③ cpu_read_1.22MB — plain / heap / malloc three-way interleaved %d rounds x %d times ==", o.rounds,
            o.repeats));
    std::size_t total_bytes = 0;
    for (int t = 0; t < kYoloNumTensors; ++t) total_bytes += modes[0].pp->raw_size(t);
    std::vector<std::uint8_t> dst(total_bytes), mal_src(total_bytes);
    for (std::size_t i = 0; i < mal_src.size(); ++i) mal_src[i] = static_cast<std::uint8_t>(i * 131u);
    std::vector<double> malloc_ms;
    for (int i = 0; i < 3; ++i) {   // warm-up
        for (AbMode& m : modes) cpu_read_all_ms(*m.pp, 0, dst.data());
        std::memcpy(dst.data(), mal_src.data(), total_bytes);
    }
    for (int round = 0; round < o.rounds; ++round)
        for (int i = 0; i < o.repeats; ++i) {
            for (AbMode& m : modes) m.cpu_read_ms.push_back(cpu_read_all_ms(*m.pp, 0, dst.data()));
            const auto t0 = std::chrono::steady_clock::now();
            std::memcpy(dst.data(), mal_src.data(), total_bytes);
            malloc_ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
                                    .count());
        }
    log(fmt("total bytes read = %zu B (sum of hw_frame_size of 6 tensors)", total_bytes));
    log("alloc    median_ms   p90_ms    min_ms    GB/s(median)  note");
    auto rowline = [&](const char* nm, const std::vector<double>& v, const char* note) {
        const double med = percentile(v, 0.5);
        log(fmt("%-7s  %9.3f  %8.3f  %8.3f  %12.3f  %s", nm, med, percentile(v, 0.9),
                *std::min_element(v.begin(), v.end()),
                static_cast<double>(total_bytes) / (med * 1e-3) / 1e9, note));
    };
    rowline("malloc", malloc_ms, "host malloc (reference)");
    rowline("heap", modes[1].cpu_read_ms, "dma-heap cached mapping, including DMA_BUF_IOCTL_SYNC");
    rowline("plain", modes[0].cpu_read_ms, "V3D HOST_COHERENT (uncached)");
    log(fmt("[cpu_read] heap/plain = %.3fx (compare with P2 measured 0.396/0.826 = 0.48x)",
            percentile(modes[1].cpu_read_ms, 0.5) / percentile(modes[0].cpu_read_ms, 0.5)));
    log("⚠ the malloc row is an **optimistic reference**: src and dst are both malloc and the same buffer is read repeatedly, so it");
    log("  stays in L2/SLC. P2's malloc 0.244 ms read the buffer HailoRT received the output into, so the conditions differ.");
    log("  **What the verdict uses is the heap vs plain comparison** — these two share the same run, method and size.");
    log("  The sync cost of the heap row (START|READ + END|READ x 6 tensors) is included in the time (Appendix B.10 R6).");

    log("");
    log(fmt("[validation] errors=%llu warnings=%llu",
            static_cast<unsigned long long>(ctx.validation_errors()),
            static_cast<unsigned long long>(ctx.validation_warnings())));
    // Make the destruction order explicit (import cache → heap pool is handled by YoloPostproc::destroy())
    for (AbMode& m : modes) m.pp.reset();
    const long fd_after = count_open_fds();
    log(fmt("[fd] baseline %ld → after release %ld → leak %ld  (heap slots %d x tensors %d dma-buf + import dup fd)",
            fd_base, fd_after, fd_after - fd_base, std::max(1, std::min(o.slots, rig.fs.frames)),
            kYoloNumTensors));
    const bool pass = acc_ok && ctx.validation_errors() == 0 && fd_after == fd_base;
    log(fmt("[RESULT] gpuab %s  (band %c)", pass ? "PASS" : "FAIL",
            ratio <= 1.05 ? 'A' : (ratio <= 1.30 ? 'B' : 'C')));
    return pass ? 0 : 1;
}

// ─────────────────────────────────────────────────────────────── live / stale (NPU)

#ifdef YVO_HAVE_HAILO

std::string status_name(hailo_status s) { return std::to_string(static_cast<int>(s)); }

void hcheck(hailo_status s, const char* where) {
    if (s != HAILO_SUCCESS) throw std::runtime_error(std::string(where) + " failed: status " + status_name(s));
}

template <typename T>
T unwrap(hailort::Expected<T>&& e, const char* where) {
    if (!e) throw std::runtime_error(std::string(where) + " failed: status " + status_name(e.status()));
    return e.release();
}

std::vector<std::vector<std::uint8_t>> make_nhcw_inputs(const std::string& video, int n, const LetterboxParams& lb,
                                                        std::size_t frame_size) {
    const std::string cmd = "ffmpeg -nostdin -v error -i '" + video + "' -frames:v " + std::to_string(n) +
                            " -f rawvideo -pix_fmt nv12 -";
    FILE* p = popen(cmd.c_str(), "r");
    if (!p) throw std::runtime_error("popen(ffmpeg) failed");
    const std::size_t fsz = static_cast<std::size_t>(kSrcW) * kSrcH * 3 / 2;
    std::vector<std::uint8_t> nv12(fsz), packed(static_cast<std::size_t>(lb.dst_w) * lb.dst_h * 3);
    std::vector<std::vector<std::uint8_t>> out;
    for (int i = 0; i < n; ++i) {
        if (std::fread(nv12.data(), 1, fsz, p) != fsz) break;
        yvo::cpu_letterbox_nv12_bt709(nv12.data(), kSrcW, nv12.data() + static_cast<std::size_t>(kSrcW) * kSrcH,
                                      kSrcW, kSrcW, kSrcH, lb, packed.data());
        std::vector<std::uint8_t> nhcw(frame_size);
        yvo::rgb_packed_to_nhcw(packed.data(), lb.dst_w, lb.dst_h, nhcw.data());
        out.push_back(std::move(nhcw));
    }
    pclose(p);
    if (out.empty()) throw std::runtime_error("could not read frames from the video: " + video);
    return out;
}

// Minimal session that receives raw output into YoloPostproc's dma-bufs.
// Follows the same discipline as RawSession in hailo_raw_probe.cpp (frame completion has a **deadline** — Appendix A.13).
class LiveSession {
public:
    LiveSession(const std::string& hef_path, YoloPostproc& pp, const std::vector<std::string>& tensor_stream_name,
                bool use_infer_async, std::uint16_t batch, bool verbose)
        : pp_(pp), infer_async_(use_infer_async) {
        hef_ = std::make_unique<hailort::Hef>(unwrap(hailort::Hef::create(hef_path), "Hef::create"));
        hailo_vdevice_params_t vp;
        hcheck(hailo_init_vdevice_params(&vp), "hailo_init_vdevice_params");
        vp.scheduling_algorithm = HAILO_SCHEDULING_ALGORITHM_ROUND_ROBIN;
        vdevice_ = unwrap(hailort::VDevice::create(vp), "VDevice::create");
        auto params = unwrap(vdevice_->create_configure_params(*hef_), "create_configure_params");
        for (auto& kv : params) {
            kv.second.batch_size = batch;
            for (auto& sp : kv.second.stream_params_by_name) sp.second.flags = HAILO_STREAM_FLAGS_ASYNC;
        }
        auto ngs = unwrap(vdevice_->configure(*hef_, params), "VDevice::configure");
        if (ngs.size() != 1) throw std::runtime_error("network group count is not 1");
        ng_ = ngs[0];

        auto ins = ng_->get_input_streams();
        if (ins.size() != 1) throw std::runtime_error("input stream count is not 1");
        input_ = &ins[0].get();
        in_size_ = input_->get_frame_size();
        in_name_ = input_->name();

        // Tensor index 0..5 ↔ OutputStream mapping. The names come from fixtures.json via the caller.
        auto outs = ng_->get_output_streams();
        for (int t = 0; t < kYoloNumTensors; ++t) {
            hailort::OutputStream* found = nullptr;
            for (auto& r : outs)
                if (r.get().name() == tensor_stream_name[static_cast<std::size_t>(t)]) found = &r.get();
            if (!found)
                throw std::runtime_error("output stream not found: " + tensor_stream_name[static_cast<std::size_t>(t)]);
            out_[t] = found;
            hw_size_[t] = found->get_frame_size();
            if (hw_size_[t] != pp_.raw_size(t))
                throw std::runtime_error("hw_frame_size differs from the fixture metadata: " + found->name() + " " +
                                         std::to_string(hw_size_[t]) + " vs " + std::to_string(pp_.raw_size(t)));
        }

        // dma-buf mapping: map with **dmabuf_size** (Appendix A.13). Only once for the lifetime of the process.
        const auto t0 = std::chrono::steady_clock::now();
        for (int s = 0; s < pp_.num_slots(); ++s)
            for (int t = 0; t < kYoloNumTensors; ++t) {
                const int fd = pp_.raw_dmabuf_fd(s, t);
                const std::size_t sz = pp_.raw_dmabuf_size(s, t) ? pp_.raw_dmabuf_size(s, t) : hw_size_[t];
                const auto st = vdevice_->dma_map_dmabuf(fd, sz, HAILO_DMA_BUFFER_DIRECTION_D2H);
                ++map_calls_;
                if (st != HAILO_SUCCESS) throw std::runtime_error("dma_map_dmabuf failed: status " + status_name(st));
                mapped_.push_back({fd, sz});
            }
        map_ms_ = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        in_buf_.assign(static_cast<std::size_t>(pp_.num_slots()) * in_size_, 0);
        if (verbose) std::fprintf(stderr, "[LiveSession] dma_map_dmabuf %ld calls %.3f ms\n", map_calls_, map_ms_);
    }

    ~LiveSession() {
        try {
            ng_.reset();
        } catch (...) {
        }
        for (const auto& m : mapped_) vdevice_->dma_unmap_dmabuf(m.fd, m.size, HAILO_DMA_BUFFER_DIRECTION_D2H);
    }
    LiveSession(const LiveSession&) = delete;
    LiveSession& operator=(const LiveSession&) = delete;

    long map_calls() const { return map_calls_; }
    double map_ms() const { return map_ms_; }

    // Receive one frame into the dma-buf of slot `slot`. **Throws if not done within the deadline** (Appendix A.13).
    void run_frame(int slot, const std::vector<std::uint8_t>& nhcw, int timeout_ms) {
        if (nhcw.size() != in_size_) throw std::runtime_error("input frame size differs");
        std::uint8_t* in = in_buf_.data() + static_cast<std::size_t>(slot) * in_size_;
        std::memcpy(in, nhcw.data(), in_size_);
        remaining_.store(kYoloNumTensors + 1);
        status_.store(HAILO_SUCCESS);
        if (infer_async_)
            issue_infer_async(slot, in);
        else
            issue_read_async(slot, in);
        std::unique_lock<std::mutex> lk(m_);
        if (!cv_.wait_for(lk, std::chrono::milliseconds(timeout_ms), [&] { return remaining_.load() == 0; }))
            throw std::runtime_error(
                "frame completion deadline exceeded — no callback arrived (Appendix A.13: on a size contract violation infer_async "
                "returns HAILO_SUCCESS and silently stalls)");
        if (status_.load() != HAILO_SUCCESS)
            throw std::runtime_error("callback status error: " + status_name(status_.load()));
    }

private:
    struct Mapping {
        int fd;
        std::size_t size;
    };

    void note(hailo_status st) {
        if (st != HAILO_SUCCESS) status_.store(st);
        if (remaining_.fetch_sub(1) == 1) {
            std::lock_guard<std::mutex> lk(m_);
            cv_.notify_all();
        }
    }

    void issue_infer_async(int slot, std::uint8_t* in) {
        hailort::NamedBuffersCallbacks named;
        {
            hailort::BufferRepresentation br{};   // it is a union: set buffer_type first
            br.buffer_type = hailort::BufferType::VIEW;
            br.view = hailort::MemoryView(in, in_size_);
            named.emplace(in_name_, std::make_pair(br, [this](hailo_status st) { note(st); }));
        }
        for (int t = 0; t < kYoloNumTensors; ++t) {
            hailort::BufferRepresentation br{};
            br.buffer_type = hailort::BufferType::DMA_BUFFER;
            // ⚠ the read-side size is **hw_frame_size only** (Appendix A.13)
            br.dma_buffer = hailo_dma_buffer_t{pp_.raw_dmabuf_fd(slot, t), hw_size_[t]};
            named.emplace(out_[t]->name(), std::make_pair(br, [this](hailo_status st) { note(st); }));
        }
        hcheck(ng_->wait_for_ongoing_callbacks_count_under(1), "wait_for_ongoing_callbacks_count_under");
        ng_->increase_ongoing_callbacks();
        const auto st = ng_->infer_async(named, [this](hailo_status) { ng_->decrease_ongoing_callbacks(); });
        if (st != HAILO_SUCCESS) {
            ng_->decrease_ongoing_callbacks();
            remaining_.store(0);
            throw std::runtime_error("infer_async failed: status " + status_name(st));
        }
    }

    void issue_read_async(int slot, std::uint8_t* in) {
        // FIFO pairing: issue 1 write + 6 reads inside one mutex (sufficient since frames run sequentially).
        std::lock_guard<std::mutex> issue(issue_m_);
        const auto st_in = input_->write_async(
            in, in_size_, [this](const hailort::InputStream::CompletionInfo& ci) { note(ci.status); });
        if (st_in != HAILO_SUCCESS) {
            remaining_.store(0);
            throw std::runtime_error("write_async failed: status " + status_name(st_in));
        }
        for (int t = 0; t < kYoloNumTensors; ++t) {
            const auto st = out_[t]->read_async(
                pp_.raw_dmabuf_fd(slot, t), hw_size_[t],
                [this](const hailort::OutputStream::CompletionInfo& ci) { note(ci.status); });
            if (st != HAILO_SUCCESS) {
                const int not_issued = kYoloNumTensors - t;
                if (remaining_.fetch_sub(not_issued) == not_issued) {
                    std::lock_guard<std::mutex> lk(m_);
                    cv_.notify_all();
                }
                throw std::runtime_error("read_async failed: status " + status_name(st));
            }
        }
    }

    YoloPostproc& pp_;
    bool infer_async_;
    std::unique_ptr<hailort::Hef> hef_;
    std::unique_ptr<hailort::VDevice> vdevice_;
    std::shared_ptr<hailort::ConfiguredNetworkGroup> ng_;
    hailort::InputStream* input_ = nullptr;
    std::string in_name_;
    std::size_t in_size_ = 0;
    hailort::OutputStream* out_[kYoloNumTensors]{};
    std::size_t hw_size_[kYoloNumTensors]{};
    std::vector<std::uint8_t> in_buf_;
    std::vector<Mapping> mapped_;
    long map_calls_ = 0;
    double map_ms_ = 0.0;
    std::atomic<int> remaining_{0};
    std::atomic<hailo_status> status_{HAILO_SUCCESS};
    std::mutex m_, issue_m_;
    std::condition_variable cv_;
};

float read_chip_temp() {
    hailo_vdevice_params_t vp;
    if (hailo_init_vdevice_params(&vp) != HAILO_SUCCESS) return std::nanf("");
    vp.scheduling_algorithm = HAILO_SCHEDULING_ALGORITHM_ROUND_ROBIN;
    auto vd = hailort::VDevice::create(vp);
    if (!vd) return std::nanf("");
    auto devs = vd.value()->get_physical_devices();
    if (!devs || devs->empty()) return std::nanf("");
    auto t = devs->at(0).get().get_chip_temperature();
    if (!t) return std::nanf("");
    return std::max(t->ts0_temperature, t->ts1_temperature);
}

// Cooling rule of the measure skill: above 75 °C, wait down to 65 °C, at most 120 s.
int cool_down(Tee& log) {
    const float before = read_chip_temp();
    log(fmt("[thermal] chip temp before = %.1f °C", static_cast<double>(before)));
    if (!(before > 75.0f)) return 0;
    int waited = 0;
    float now = before;
    while (waited < 120) {
        std::this_thread::sleep_for(std::chrono::seconds(5));
        waited += 5;
        now = read_chip_temp();
        if (!(now > 65.0f)) break;
    }
    if (std::isnan(now))
        log(fmt("[thermal] ⚠ could not read the chip temperature even after %d s of cooling wait (another process holds the device) — "
                "performance numbers of this run are unreliable", waited));
    else
        log(fmt("[thermal] cooling wait %d s → %.1f °C", waited, static_cast<double>(now)));
    return waited;
}

std::string read_first_line(const std::string& path) {
    std::ifstream f(path);
    std::string l;
    std::getline(f, l);
    return l;
}

// The PCIe path of the Hailo-8 differs per system (on this Pi 5 it is 0001:01:00.0). Find it by vendor id 0x1e60.
std::string pcie_link_speed() {
    std::error_code ec;
    for (auto it = std::filesystem::directory_iterator("/sys/bus/pci/devices", ec);
         !ec && it != std::filesystem::directory_iterator(); it.increment(ec)) {
        const std::string dir = it->path().string();
        if (read_first_line(dir + "/vendor") != "0x1e60") continue;
        return read_first_line(dir + "/current_link_speed") + " x" + read_first_line(dir + "/current_link_width") +
               " (" + it->path().filename().string() + ")";
    }
    return "(Hailo PCIe device not found)";
}

int count_hailort_log_errors() {
    std::ifstream f("hailort.log");
    std::string line;
    int n = 0;
    while (std::getline(f, line))
        if (line.find("[error]") != std::string::npos) ++n;
    return n;
}

int count_hailort_log_thermal() {
    std::ifstream f("hailort.log");
    std::string line;
    int n = 0;
    while (std::getline(f, line)) {
        if (line.find("[warning]") == std::string::npos) continue;
        if (line.find("emperature") != std::string::npos || line.find("hrottl") != std::string::npos ||
            line.find("verheat") != std::string::npos)
            ++n;
    }
    return n;
}

std::vector<std::string> stream_names_by_tensor(const FixtureSet& fs) {
    const auto order = tensor_order(fs);
    std::vector<std::string> v;
    for (const auto* p : order) v.push_back(p->name);
    return v;
}

// If another process holds /dev/hailo0, VDevice::create fails with HAILO_OUT_OF_PHYSICAL_DEVICES(74).
// Check first and report down to the process name so the cause is not guessed.
std::string npu_holder() {
    std::error_code ec;
    for (auto it = std::filesystem::directory_iterator("/proc", ec);
         !ec && it != std::filesystem::directory_iterator(); it.increment(ec)) {
        const std::string name = it->path().filename().string();
        if (name.empty() || !std::isdigit(static_cast<unsigned char>(name[0]))) continue;
        // Skip ourselves — libhailort already opens /dev/hailo0 at process start.
        if (name == std::to_string(::getpid())) continue;
        std::error_code ec2;
        for (auto f = std::filesystem::directory_iterator(it->path() / "fd", ec2);
             !ec2 && f != std::filesystem::directory_iterator(); f.increment(ec2)) {
            std::error_code ec3;
            const auto target = std::filesystem::read_symlink(f->path(), ec3);
            if (!ec3 && target.string().rfind("/dev/hailo", 0) == 0)   // distinguish from files whose path contains "hailo"
                return name + " (" + read_first_line(it->path().string() + "/comm") + ")";
        }
    }
    return std::string();
}

int run_live_inner(const Options& o, Tee& log, long fd0);

// fd leak count: only meaningful after the session, YoloPostproc and VulkanContext are **all destroyed**.
int run_live(const Options& o, Tee& log) {
    const long fd0 = count_open_fds();
    const int rc = run_live_inner(o, log, fd0);
    const long fd1 = count_open_fds();
    log(fmt("[fd] after release start=%ld end=%ld  %s", fd0, fd1, fd1 == fd0 ? "no leak" : "⚠ leak suspected"));
    return (rc == 0 && fd1 == fd0) ? 0 : (rc != 0 ? rc : 1);
}

int run_live_inner(const Options& o, Tee& log, long fd0) {
    FixtureSet fs = load_fixtures(o.fixture_dir);
    const YoloTensorLayout L = layout_from_fixtures(fs);
    log(fmt("[live] hef=%s video=%s frames=%d api=%s batch=%u slots=%d", o.hef.c_str(), o.video.c_str(),
            o.live_frames, o.api.c_str(), o.batch, o.slots));
    log(fmt("[pcie] %s  fd(start)=%ld", pcie_link_speed().c_str(), fd0));
    if (const std::string holder = npu_holder(); !holder.empty())
        throw std::runtime_error("the NPU is already held by another process: pid " + holder +
                                 " — there is only one Hailo-8, so it must be used exclusively");
    cool_down(log);
    const int err0 = count_hailort_log_errors();
    const int th0 = count_hailort_log_thermal();

    yvo::vk::VulkanContextOptions co;
    co.enable_validation = o.validation;
    co.validation_sync = o.sync_validation;
    co.verbose = o.verbose;
    yvo::vk::VulkanContext ctx(co);

    YoloPostprocConfig pc;
    pc.num_slots = std::max(1, o.slots);
    pc.max_dets = o.max_dets;
    pc.local_size_x = o.lx;
    pc.local_size_y = o.ly;
    pc.score_thr = kHefScoreThreshold;
    pc.input_alloc = alloc_of(o.alloc, YoloInputAlloc::ExportDmaBuf);   // plain = option adopted in plan §4.2
    pc.input_sync = YoloInputSync::Foreign;          // external memory written by Hailo PCIe DMA
    pc.heap_sync = heap_sync_cells(o.heap_sync).front();
    pc.checksum_pipeline = true;
    pc.verbose = o.verbose;
    YoloPostproc pp(ctx, L, pc);
    log(fmt("[alloc] %s  heap_sync=%s  headroom=%zu B", o.alloc.c_str(),
            pp.raw_heap_sync_active() ? "on"
                                      : (pc.input_alloc == YoloInputAlloc::ImportHeap ? "off" : "n/a"),
            pp.heap_headroom()));
    for (int t = 0; t < kYoloNumTensors; ++t)
        log(fmt("[dmabuf] tensor %d  hw_frame_size=%zu  dmabuf_size=%zu  fd=%d", t, pp.raw_size(t),
                pp.raw_dmabuf_size(0, t), pp.raw_dmabuf_fd(0, t)));

    LiveSession sess(o.hef, pp, stream_names_by_tensor(fs), o.api == "infer_async", o.batch, o.verbose);
    log(fmt("[live] dma_map_dmabuf %ld calls %.3f ms", sess.map_calls(), sess.map_ms()));

    // prepare input frames
    std::size_t in_sz = 0;
    {
        // Already validated by LiveSession. The input size is 640*640*3.
        in_sz = static_cast<std::size_t>(fs.lb.dst_w) * fs.lb.dst_h * 3;
    }
    const auto inputs = make_nhcw_inputs(o.video, o.live_frames, fs.lb, in_sz);
    log(fmt("[live] prepared %zu input frames (NHCW %zu B)", inputs.size(), in_sz));

    log("");
    log("== P3-(2) / V1 three-way check: Hailo DMA write → GPU computed sum ↔ CPU cached-mapping read sum ↔ malloc snapshot sum ==");
    log("  · shader : value read and summed by V3D through the kChecksum pipeline");
    log("  · mapped : value the CPU read and summed directly from the cached mapping of the heap fd inside the READ sync section");
    log("  · malloc : value summed from a copy memcpy'd into a malloc buffer inside the same READ section");
    log("  ⚠ the malloc copy is taken from the same mapping, so it is **not an independent path** — it only catches cases where");
    log("    the mapping read wobbles midway (torn reads); it differs from the P2 reference where HailoRT received directly into a malloc buffer.");
    log("frame slot  tensor mismatch  cand  cpu_cand  matched  max_coord_err_px");
    bool sum_ok = true, dec_ok = true, v1_ok = true;
    std::size_t v1_cmp = 0, v1_bad_mapped = 0, v1_bad_malloc = 0;
    std::vector<double> pass1;
    std::vector<std::uint8_t> snap;
    const auto order = tensor_order(fs);
    for (std::size_t f = 0; f < inputs.size(); ++f) {
        const int slot = static_cast<int>(f) % pp.num_slots();
        sess.run_frame(slot, inputs[f], o.timeout_ms);

        // ① byte-sum check (kChecksum pipeline)
        pp.submit(slot, fs.lb, fs.src_w, fs.src_h, /*checksum=*/true);
        pp.wait(slot);
        std::uint32_t shader[kYoloNumTensors];
        pp.checksums(slot, shader);
        int mism = 0;
        {
            // R2: open the READ section **after** the completion signal (vkWaitForFences + HailoRT callback).
            const RawRead rr(pp, slot);
            for (int t = 0; t < kYoloNumTensors; ++t) {
                const FixtureStream& m = *order[static_cast<std::size_t>(t)];
                const std::uint8_t* src = pp.raw_host_ptr(slot, t);
                const std::uint32_t cs = cpu_valid_sum(src, m.grid_h, m.features, m.grid_w, m.w_padded);
                snap.assign(src, src + pp.raw_size(t));
                const std::uint32_t ms = cpu_valid_sum(snap.data(), m.grid_h, m.features, m.grid_w, m.w_padded);
                ++v1_cmp;
                if (cs != shader[t]) {
                    ++mism;
                    ++v1_bad_mapped;
                    log(fmt("   tensor %d: shader %u != mapped %u", t, shader[t], cs));
                }
                if (ms != shader[t]) {
                    ++v1_bad_malloc;
                    log(fmt("   tensor %d: shader %u != malloc %u", t, shader[t], ms));
                }
            }
        }
        sum_ok = sum_ok && mism == 0;
        v1_ok = v1_ok && mism == 0 && v1_bad_malloc == 0;

        // ② same buffer through the normal pipeline → comparison against the CPU reference
        pp.submit(slot, fs.lb, fs.src_w, fs.src_h);
        pp.wait(slot);
        pass1.push_back(pp.last_pass1_ms());
        const auto gpu = pp.read_candidates(slot);
        const std::uint8_t* ptr[kYoloNumTensors];
        for (int t = 0; t < kYoloNumTensors; ++t) ptr[t] = pp.raw_host_ptr(slot, t);
        std::vector<DetectionKeyed> cpu;
        {
            const RawRead rr(pp, slot);   // R2
            cpu = yvo::cpu_yolov8_candidates(make_raw_set(L, ptr), kHefScoreThreshold, fs.lb, fs.src_w,
                                             fs.src_h);
        }
        const Compare c = compare_candidates(cpu, gpu);
        dec_ok = dec_ok && c.ok();
        log(fmt("%5zu %4d %8d  %5zu  %8zu  %7zu  %16.6f", f, slot, mism, c.gpu, c.cpu, c.matched,
                static_cast<double>(c.max_coord_err)));
    }
    log(fmt("[P3-(2)] byte sum %s / decode match %s", sum_ok ? "PASS" : "FAIL", dec_ok ? "PASS" : "FAIL"));
    log(fmt("[V1] three-way check %s — %zu comparisons (per tensor, %zu frames x 6) · shader!=mapped %zu · shader!=malloc %zu "
            "(criterion: 0 mismatches)",
            v1_ok ? "PASS" : "FAIL", v1_cmp, inputs.size(), v1_bad_mapped, v1_bad_malloc));
    if (!pass1.empty())
        log(fmt("[live] pass1_ms median %.4f p90 %.4f (with the NPU running concurrently — Appendix A.8: higher than the standalone bench)",
                percentile(pass1, 0.5), percentile(pass1, 0.9)));

    const int err1 = count_hailort_log_errors();
    const int th1 = count_hailort_log_thermal();
    log(fmt("[hailort.log] error +%d  temperature/clock warning +%d %s", err1 - err0, th1 - th0,
            (th1 - th0) ? "→ performance numbers of this run are unreliable" : ""));
    log(fmt("[validation] errors=%llu warnings=%llu",
            static_cast<unsigned long long>(ctx.validation_errors()),
            static_cast<unsigned long long>(ctx.validation_warnings())));
    log(fmt("[fd] start=%ld  while session held=%ld (= slots %d x tensors %d dma-buf + HailoRT/Vulkan internal)", fd0,
            count_open_fds(), pp.num_slots(), kYoloNumTensors));
    const bool pass = sum_ok && dec_ok && v1_ok && ctx.validation_errors() == 0 && (err1 - err0) == 0;
    log(fmt("[RESULT] live %s", pass ? "PASS" : "FAIL"));
    return pass ? 0 : 1;
}

// Receive frames A and B back to back into the same slot buffer and run pass 1 on each (barrier on/off).
int run_stale(const Options& o, Tee& log) {
    FixtureSet fs = load_fixtures(o.fixture_dir);
    const YoloTensorLayout L = layout_from_fixtures(fs);
    log(fmt("[stale] receive different frames A/B back to back into the same slot buffer (num_slots=1). barrier=%s",
            o.stale_barrier.c_str()));
    log(fmt("[pcie] %s", pcie_link_speed().c_str()));
    if (const std::string holder = npu_holder(); !holder.empty())
        throw std::runtime_error("the NPU is already held by another process: pid " + holder +
                                 " — there is only one Hailo-8, so it must be used exclusively");
    cool_down(log);
    const int err0 = count_hailort_log_errors();

    std::vector<bool> modes;
    if (o.stale_barrier == "on" || o.stale_barrier == "both") modes.push_back(true);
    if (o.stale_barrier == "off" || o.stale_barrier == "both") modes.push_back(false);
    // V2 (Appendix B.10): with ImportHeap, add the `--heap-sync` axis and run every {sync} x {barrier} cell.
    const YoloInputAlloc stale_alloc = alloc_of(o.alloc, YoloInputAlloc::ExportDmaBuf);
    const std::vector<bool> syncs =
        stale_alloc == YoloInputAlloc::ImportHeap ? heap_sync_cells(o.heap_sync) : std::vector<bool>{true};
    log(fmt("[stale] alloc=%s  cells %zu(heap_sync) x %zu(barrier)", o.alloc.c_str(), syncs.size(), modes.size()));

    // Input decoding does not need the NPU → finish it **before creating the session** (minimizes the time the NPU is held).
    const std::size_t in_sz = static_cast<std::size_t>(fs.lb.dst_w) * fs.lb.dst_h * 3;
    const auto inputs = make_nhcw_inputs(o.video, std::max(2, o.live_frames), fs.lb, in_sz);
    const std::vector<std::vector<std::uint8_t>> ab{inputs.front(), inputs.back()};
    log(fmt("[stale] frames A/B ready (first/last of the first %zu frames of the video)", inputs.size()));

    bool barrier_on_ok = true;
    bool barrier_off_wrong = false;
    bool sync_off_wrong = false;
    bool sync_on_ok = true;
    std::uint64_t vk_errors = 0;
    for (const bool heap_sync : syncs)
    for (const bool barrier : modes) {
        log("");
        log(fmt("---- heap_sync %s / acquire barrier %s ----",
                stale_alloc == YoloInputAlloc::ImportHeap ? (heap_sync ? "ON" : "OFF") : "n/a",
                barrier ? "ON (FOREIGN_EXT → compute)" : "OFF (no barrier)"));
        yvo::vk::VulkanContextOptions co;
        co.enable_validation = o.validation;
        co.validation_sync = o.sync_validation;
        co.verbose = o.verbose;
        yvo::vk::VulkanContext ctx(co);

        YoloPostprocConfig pc;
        pc.num_slots = 1;   // **force reuse of the same buffer** — the condition under which staleness shows up
        pc.max_dets = o.max_dets;
        pc.local_size_x = o.lx;
        pc.local_size_y = o.ly;
        pc.score_thr = kHefScoreThreshold;
        pc.input_alloc = stale_alloc;
        pc.input_sync = barrier ? YoloInputSync::Foreign : YoloInputSync::None;
        pc.heap_sync = heap_sync;
        pc.checksum_pipeline = true;
        YoloPostproc pp(ctx, L, pc);
        LiveSession sess(o.hef, pp, stream_names_by_tensor(fs), o.api == "infer_async", o.batch, o.verbose);

        log("pass frame  cand  cpu_cand  matched  onlyCPU  onlyGPU  score_bit_mismatch  max_coord_err_px  verdict");
        const auto order = tensor_order(fs);
        for (int rep = 0; rep < 2; ++rep) {   // run A, B twice to also check order dependence
            for (std::size_t k = 0; k < ab.size(); ++k) {
                sess.run_frame(0, ab[k], o.timeout_ms);
                pp.submit(0, fs.lb, fs.src_w, fs.src_h);
                pp.wait(0);
                const auto gpu = pp.read_candidates(0);
                const std::uint8_t* ptr[kYoloNumTensors];
                for (int t = 0; t < kYoloNumTensors; ++t) ptr[t] = pp.raw_host_ptr(0, t);
                std::vector<DetectionKeyed> cpu;
                {
                    const RawRead rr(pp, 0);   // R2 (a no-op in the heap_sync=off cell — that is the point of this cell)
                    cpu = yvo::cpu_yolov8_candidates(make_raw_set(L, ptr), kHefScoreThreshold, fs.lb, fs.src_w,
                                                     fs.src_h);
                }
                const Compare c = compare_candidates(cpu, gpu);
                // also check the byte sum (are the bytes the shader actually read the same as what the host sees)
                pp.submit(0, fs.lb, fs.src_w, fs.src_h, /*checksum=*/true);
                pp.wait(0);
                std::uint32_t shader[kYoloNumTensors];
                pp.checksums(0, shader);
                int mism = 0;
                {
                    const RawRead rr(pp, 0);   // R2
                    for (int t = 0; t < kYoloNumTensors; ++t) {
                        const FixtureStream& m = *order[static_cast<std::size_t>(t)];
                        if (cpu_valid_sum(pp.raw_host_ptr(0, t), m.grid_h, m.features, m.grid_w, m.w_padded) !=
                            shader[t])
                            ++mism;
                    }
                }
                const bool good = c.ok() && mism == 0;
                log(fmt("%4d %5s  %4zu  %8zu  %7zu  %7zu  %7zu  %18zu  %16.6f  %s  (sum_mismatch=%d)", rep,
                        k == 0 ? "A" : "B", c.gpu, c.cpu, c.matched, c.only_cpu, c.only_gpu,
                        c.score_bit_mismatch, static_cast<double>(c.max_coord_err), good ? "ok" : "WRONG", mism));
                if (barrier)
                    barrier_on_ok = barrier_on_ok && good;
                else if (!good)
                    barrier_off_wrong = true;
                if (heap_sync)
                    sync_on_ok = sync_on_ok && good;
                else if (!good)
                    sync_off_wrong = true;
            }
        }
        vk_errors += ctx.validation_errors();
        log(fmt("[validation] errors=%llu", static_cast<unsigned long long>(ctx.validation_errors())));
    }
    const int err1 = count_hailort_log_errors();
    log("");
    log(fmt("[P3-(3)] A/B each correct with barrier ON: %s", barrier_on_ok ? "PASS" : "FAIL"));
    log(fmt("[P3-(4)] did a wrong result appear with barrier OFF: %s (not a verdict — recorded)",
            barrier_off_wrong ? "yes (evidence that the barrier is required)"
                              : "no (correct even without the barrier on this hardware — may be by chance)"));
    if (stale_alloc == YoloInputAlloc::ImportHeap) {
        log(fmt("[V2] A/B each correct in the heap_sync ON cell: %s", sync_on_ok ? "PASS" : "FAIL"));
        log(fmt("[V2] did a wrong result appear with heap_sync OFF: %s",
                sync_off_wrong ? "yes" : "no"));
        log("     ⚠ **Appendix A P3-④ lesson**: not going wrong with OFF is **not evidence that sync is unnecessary.**");
        log("       This test cannot rule out the cache lines happening to be clean. Forcing the failure is V3's job.");
        log("       The structural basis stands — the device tree has 0 `dma-coherent` entries, so the kernel treats V3D");
        log("       as a non-coherent device. The experiment merely failed to refute that.");
    }
    log(fmt("[hailort.log] error +%d", err1 - err0));
    const bool pass = barrier_on_ok && sync_on_ok && vk_errors == 0 && (err1 - err0) == 0;
    log(fmt("[RESULT] stale %s", pass ? "PASS" : "FAIL"));
    return pass ? 0 : 1;
}


// ───────────────────────────────── V3: tests that force failure (Appendix B.10, **exclusive NPU**)
//
// V2 may "luckily not go wrong". V3 deliberately puts the cache in a bad state to **force** the failure.
// Three variants are run — the first two are the **write direction** (dirty lines), the third is the **read direction** (stale clean lines).
// The read direction is exactly the hole Appendix B.18 describes as "never demonstrated yet".
//
//   ① full-dirty  : the CPU memsets 0xA5 over the **whole** buffer — `END|WRITE` deliberately omitted → Hailo DMA
//                   writes frame A → read. If dirty lines remain, 0xA5 survives.
//                   ⚠ 1.22 MB mostly goes down to DRAM before inference finishes, due to A76 write-streaming (no line allocation for long
//                     sequential writes) and capacity eviction → low reproducibility. ② compensates for that.
//   ② line-dirty  : write **1 byte per 64 B line** only within a **small window** (--v3-window, default 4096 B).
//                   Partial-line writes cannot use write-streaming and fetch the line, making it dirty.
//                   The window is small, so it is far more likely to stay in L1/L2. `END|WRITE` is again omitted.
//   ③ stale-read  : **read direction**. Receive frame B and read the window to fill the cache with clean lines (priming) →
//                   Hailo DMA writes a **different** frame A → read without `START|READ`.
//                   If stale lines remain, **frame B's bytes** show up. Counting them is the verdict.
//
// The reference is taken with a **forced sync regardless of the cell** (`forced_sync`) — if the reference is contaminated too, the verdict is meaningless.
// Expected: contamination > 0 with `--heap-sync off`, 0 with `on`. **Only zero contamination in the `on` cell is the PASS condition.**
constexpr std::uint8_t kDirtyByte = 0xA5;
constexpr std::size_t kCacheLine = 64;   // getconf LEVEL1_DCACHE_LINESIZE = 64 (Cortex-A76)

struct V3Sub {
    const char* name = "";
    std::size_t bytes_cmp = 0;      // number of bytes compared (cumulative)
    std::size_t diff = 0;           // number of bytes differing from the reference
    std::size_t a5 = 0;             // of those, 0xA5 (residue of ① ②)
    std::size_t stale = 0;          // of those, **values from the other frame** (stale data of ③)
    int gpu_mismatch = 0;           // number of (trial x tensor) where the GPU byte sum differs from the reference
    int gpu_cmp = 0;
    std::size_t detectable = 0;     // bytes where A and B actually differ in ③ = detection budget
};

int run_v3(const Options& o, Tee& log) {
    FixtureSet fs = load_fixtures(o.fixture_dir);
    const YoloTensorLayout L = layout_from_fixtures(fs);
    log("[v3] failure-forcing harness, 3 variants: ① full-dirty ② line-dirty ③ stale-read (read direction)");
    log(fmt("[pcie] %s", pcie_link_speed().c_str()));
    if (const std::string holder = npu_holder(); !holder.empty())
        throw std::runtime_error("the NPU is already held by another process: pid " + holder +
                                 " — there is only one Hailo-8, so it must be used exclusively");
    cool_down(log);
    const int err0 = count_hailort_log_errors();

    const std::size_t in_sz = static_cast<std::size_t>(fs.lb.dst_w) * fs.lb.dst_h * 3;
    const auto inputs = make_nhcw_inputs(o.video, std::max(2, o.live_frames), fs.lb, in_sz);
    const std::vector<std::uint8_t>& frame_a = inputs.front();
    const std::vector<std::uint8_t>& frame_b = inputs.back();
    log(fmt("[v3] frames A/B ready (first/last of the first %zu frames of the video). %d repeats per variant per cell, window %d B, line %zu B",
            inputs.size(), o.repeats, o.v3_window, kCacheLine));

    std::vector<std::pair<bool, std::array<V3Sub, 3>>> cells;
    std::uint64_t vk_errors = 0;
    for (const bool heap_sync : heap_sync_cells(o.heap_sync)) {
        std::array<V3Sub, 3> sub{};
        sub[0].name = "full-dirty";
        sub[1].name = "line-dirty";
        sub[2].name = "stale-read";
        log("");
        log(fmt("======== heap_sync %s ========", heap_sync ? "ON (START/END|READ per protocol)" : "OFF (no sync)"));

        yvo::vk::VulkanContextOptions co;
        co.enable_validation = o.validation;
        co.validation_sync = o.sync_validation;
        co.verbose = o.verbose;
        yvo::vk::VulkanContext ctx(co);

        YoloPostprocConfig pc;
        pc.num_slots = 1;                                   // **force reuse of the same buffer**
        pc.max_dets = o.max_dets;
        pc.local_size_x = o.lx;
        pc.local_size_y = o.ly;
        pc.score_thr = kHefScoreThreshold;
        pc.input_alloc = YoloInputAlloc::ImportHeap;        // V3 is heap-only
        pc.input_sync = YoloInputSync::Foreign;             // the barrier is always on (it is not an axis)
        pc.heap_sync = heap_sync;
        pc.checksum_pipeline = true;
        pc.verbose = o.verbose;
        YoloPostproc pp(ctx, L, pc);
        LiveSession sess(o.hef, pp, stream_names_by_tensor(fs), o.api == "infer_async", o.batch, o.verbose);

        // Receive one frame and take the reference (full snapshot + GPU byte sum). The sync is **forced**.
        auto capture = [&](const std::vector<std::uint8_t>& frame,
                           std::array<std::vector<std::uint8_t>, kYoloNumTensors>& snap,
                           std::uint32_t sum[kYoloNumTensors]) {
            sess.run_frame(0, frame, o.timeout_ms);
            pp.submit(0, fs.lb, fs.src_w, fs.src_h, /*checksum=*/true);
            pp.wait(0);
            pp.checksums(0, sum);
            forced_sync(pp, 0, yvo::DmaSyncDir::Read, /*start=*/true);
            for (int t = 0; t < kYoloNumTensors; ++t) {
                const std::uint8_t* src = pp.raw_host_ptr(0, t);
                snap[static_cast<std::size_t>(t)].assign(src, src + pp.raw_size(t));
            }
            forced_sync(pp, 0, yvo::DmaSyncDir::Read, /*start=*/false);
        };
        std::array<std::vector<std::uint8_t>, kYoloNumTensors> refA, refB;
        std::uint32_t sumA[kYoloNumTensors], sumB[kYoloNumTensors];
        capture(frame_a, refA, sumA);
        capture(frame_b, refB, sumB);

        std::size_t a5_noise = 0, ab_diff_full = 0, ab_diff_win = 0;
        std::size_t win_total = 0;
        for (int t = 0; t < kYoloNumTensors; ++t) {
            const auto& a = refA[static_cast<std::size_t>(t)];
            const auto& b = refB[static_cast<std::size_t>(t)];
            const std::size_t w = std::min<std::size_t>(static_cast<std::size_t>(o.v3_window), a.size());
            win_total += w;
            for (std::size_t i = 0; i < a.size(); ++i) {
                if (a[i] == kDirtyByte) ++a5_noise;
                if (a[i] != b[i]) {
                    ++ab_diff_full;
                    if (i < w) ++ab_diff_win;
                }
            }
        }
        log(fmt("[reference] 0xA5 bytes in frame A snapshot %zu (noise floor for ①②) · A≠B bytes total %zu / window %zu "
                "(③ detection budget, window total %zu B)", a5_noise, ab_diff_full, ab_diff_win, win_total));
        if (ab_diff_win == 0)
            log("  ⚠ A and B are identical within the window — ③ cannot detect anything with this window. Increase --v3-window");

        std::vector<std::uint8_t> buf;
        // Read per the cell setting and compare with the reference. Only cmp_bytes are examined.
        auto read_and_compare = [&](V3Sub& st, const std::array<std::vector<std::uint8_t>, kYoloNumTensors>& ref,
                                    const std::array<std::vector<std::uint8_t>, kYoloNumTensors>* other,
                                    std::size_t cmp_bytes) {
            std::size_t d = 0, a5 = 0, stale = 0;
            const RawRead rr(pp, 0);   // a no-op in the heap_sync=off cell — that is the point of this cell
            for (int t = 0; t < kYoloNumTensors; ++t) {
                const auto& r = ref[static_cast<std::size_t>(t)];
                const std::size_t n = std::min(cmp_bytes, r.size());
                const std::uint8_t* src = pp.raw_host_ptr(0, t);
                buf.assign(src, src + n);
                st.bytes_cmp += n;
                for (std::size_t i = 0; i < n; ++i)
                    if (buf[i] != r[i]) {
                        ++d;
                        if (buf[i] == kDirtyByte) ++a5;
                        if (other && buf[i] == (*other)[static_cast<std::size_t>(t)][i]) ++stale;
                    }
            }
            st.diff += d;
            st.a5 += a5;
            st.stale += stale;
            return d;
        };
        auto gpu_compare = [&](V3Sub& st, const std::uint32_t ref_sum[kYoloNumTensors]) {
            pp.submit(0, fs.lb, fs.src_w, fs.src_h, /*checksum=*/true);
            pp.wait(0);
            std::uint32_t sum[kYoloNumTensors];
            pp.checksums(0, sum);
            int m = 0;
            for (int t = 0; t < kYoloNumTensors; ++t) {
                ++st.gpu_cmp;
                if (sum[t] != ref_sum[t]) {
                    ++m;
                    ++st.gpu_mismatch;
                }
            }
            return m;
        };

        const std::size_t win = static_cast<std::size_t>(o.v3_window);
        sub[2].detectable = ab_diff_win;

        // ── ① full-dirty ─────────────────────────────────────────────────────────────────────
        log("");
        log("---- ① full-dirty: full memset(0xA5), END|WRITE omitted ----");
        log("trial  gpu_mismatch  diff_bytes  0xA5_left");
        for (int trial = 0; trial < o.repeats; ++trial) {
            forced_sync(pp, 0, yvo::DmaSyncDir::Write, /*start=*/true);
            for (int t = 0; t < kYoloNumTensors; ++t)
                std::memset(pp.raw_host_ptr(0, t), kDirtyByte, pp.raw_size(t));
            // (no END|WRITE — this is the core of this variant)
            sess.run_frame(0, frame_a, o.timeout_ms);
            const int gm = gpu_compare(sub[0], sumA);
            const std::size_t before = sub[0].a5;
            const std::size_t d = read_and_compare(sub[0], refA, nullptr, SIZE_MAX);
            log(fmt("%5d  %12d  %10zu  %9zu", trial, gm, d, sub[0].a5 - before));
            forced_sync(pp, 0, yvo::DmaSyncDir::Write, /*start=*/false);   // pair up the ioctls
        }

        // ── ② line-dirty ─────────────────────────────────────────────────────────────────────
        log("");
        log(fmt("---- ② line-dirty: within a %zu B window, write only 1 byte per %zu B line (partial line → forced dirty), "
                "END|WRITE omitted ----", win, kCacheLine));
        log("trial  gpu_mismatch  diff_bytes(win)  0xA5_left");
        for (int trial = 0; trial < o.repeats; ++trial) {
            forced_sync(pp, 0, yvo::DmaSyncDir::Write, /*start=*/true);
            for (int t = 0; t < kYoloNumTensors; ++t) {
                std::uint8_t* pw = pp.raw_host_ptr(0, t);
                const std::size_t n = std::min(win, pp.raw_size(t));
                for (std::size_t i = 0; i < n; i += kCacheLine) pw[i] = kDirtyByte;
            }
            sess.run_frame(0, frame_a, o.timeout_ms);
            const int gm = gpu_compare(sub[1], sumA);
            const std::size_t before = sub[1].a5;
            const std::size_t d = read_and_compare(sub[1], refA, nullptr, win);
            log(fmt("%5d  %12d  %14zu  %9zu", trial, gm, d, sub[1].a5 - before));
            forced_sync(pp, 0, yvo::DmaSyncDir::Write, /*start=*/false);
        }

        // ── ③ stale-read (read direction) ────────────────────────────────────────────────────
        log("");
        log(fmt("---- ③ stale-read: read a %zu B window with frame B to fill the cache, then DMA writes frame A ----",
                win));
        log("trial  gpu_mismatch  diff_bytes(win)  of_which_B_values(stale_data)");
        std::vector<std::uint8_t> prime;
        for (int trial = 0; trial < o.repeats; ++trial) {
            // (a) Receive frame B and read the window several times to fill the cache with **clean lines** (forced sync).
            sess.run_frame(0, frame_b, o.timeout_ms);
            forced_sync(pp, 0, yvo::DmaSyncDir::Read, /*start=*/true);
            std::uint64_t acc = 0;
            for (int pass = 0; pass < 3; ++pass)
                for (int t = 0; t < kYoloNumTensors; ++t) {
                    const std::uint8_t* src = pp.raw_host_ptr(0, t);
                    const std::size_t n = std::min(win, pp.raw_size(t));
                    for (std::size_t i = 0; i < n; ++i) acc += src[i];
                }
            forced_sync(pp, 0, yvo::DmaSyncDir::Read, /*start=*/false);
            // (b) DMA writes a **different** frame A into the same buffer
            sess.run_frame(0, frame_a, o.timeout_ms);
            // (c) Read per the cell setting. In the off cell there is no START|READ → if stale lines show, B values appear
            const int gm = gpu_compare(sub[2], sumA);
            const std::size_t before = sub[2].stale;
            const std::size_t d = read_and_compare(sub[2], refA, &refB, win);
            log(fmt("%5d  %12d  %14zu  %21zu  (acc=%llu)", trial, gm, d, sub[2].stale - before,
                    static_cast<unsigned long long>(acc & 0xffffu)));
        }

        vk_errors += ctx.validation_errors();
        log(fmt("[validation] errors=%llu", static_cast<unsigned long long>(ctx.validation_errors())));
        cells.emplace_back(heap_sync, sub);
    }

    const int err1 = count_hailort_log_errors();
    log("");
    log("== V3 verdict ==");
    log("heap_sync  variant       bytes_cmp     diff  0xA5left  staleB  gpu_mismatch/cmp  budget");
    bool on_clean = true, off_dirty = false;
    for (const auto& [heap_sync, sub] : cells)
        for (const V3Sub& st : sub) {
            log(fmt("%-9s  %-11s  %10zu  %7zu  %8zu  %7zu  %10d/%-7d  %zu", heap_sync ? "on" : "off", st.name,
                    st.bytes_cmp, st.diff, st.a5, st.stale, st.gpu_mismatch, st.gpu_cmp, st.detectable));
            if (heap_sync)
                on_clean = on_clean && st.diff == 0 && st.gpu_mismatch == 0;
            else if (st.diff > 0 || st.gpu_mismatch > 0)
                off_dirty = true;
        }
    log(fmt("[V3] zero contamination with heap_sync ON (all 3 variants): %s", on_clean ? "PASS" : "**FAIL**"));
    log(fmt("[V3] was contamination reproduced with heap_sync OFF: %s", off_dirty ? "yes — the need for sync is demonstrated" : "no"));
    if (!off_dirty) {
        log("     ⚠ none of the three variants reproduced it. **Sync is kept as is anyway.**");
        log("       The kernel treating V3D as a non-coherent device because the device tree has 0 `dma-coherent` entries");
        log("       is the structural basis for 'required'; this experiment merely failed to refute it (Appendix A P3-④).");
        log("       Possible mechanisms (unconfirmed): ⓐ begin/end_cpu_access of the system heap walks the attached devices and");
        log("       already performs the needed cache maintenance, ⓑ PCIe DMA snoops the CPU caches (ACP/IO-coherent),");
        log("       ⓒ the cache is naturally evicted during the inference latency (ms), so the condition itself does not hold.");
        log("       To reduce ⓒ, ②③ used a small window (partial line/priming), yet it still did not reproduce.");
    }
    log(fmt("[hailort.log] error +%d", err1 - err0));
    const bool pass = on_clean && vk_errors == 0 && (err1 - err0) == 0;
    log(fmt("[RESULT] v3 %s", pass ? "PASS" : "FAIL"));
    return pass ? 0 : 1;
}

#else   // !defined(YVO_HAVE_HAILO)

int run_live(const Options&, Tee& log) {
    log("[live] built without HailoRT — live mode is unavailable");
    return 2;
}
int run_stale(const Options&, Tee& log) {
    log("[stale] built without HailoRT — stale mode is unavailable");
    return 2;
}
int run_v3(const Options&, Tee& log) {
    log("[v3] built without HailoRT — v3 mode is unavailable");
    return 2;
}

#endif

// ───────────────────────────────────────────── heapimport (Appendix B step 1 = P0 gate, S1)
//
// **Does not use the NPU.** For each of the 6 tensors, take align_up(hw_frame_size, 16384) from /dev/dma_heap,
// import it into Vulkan via `DmaBufImportCache::get()`, then record every value needed for the verdict.
//
// What is recorded (Appendix B.13 step-1 gate + B.12 design basis):
//   ① VkMemoryFdPropertiesKHR::memoryTypeBits      — **if 0, it ends right there (S1)**
//   ② VkMemoryRequirements2::size vs dma-buf size  — measured V3DV slack (+256 B). The B.12 headroom comes from here
//   ③ VkMemoryDedicatedRequirements two flags      — if prefers=true, over-allocation is a spec violation
//                                                    (VUID-VkMemoryDedicatedAllocateInfo-buffer-02965)
//   ④ ImportedDmaBuf::range vs size                — range < size means the -256 B shrink kicked in
//   ⑤ VkResult of vkAllocateMemory (dmesg on failure)
//   ⑥ 16384 alignment of the mmap pointer (HailoRT requirement, Appendix A.4) — **P0 exists for the verdict, so it only records, no throw**
//   ⑦ round trip: byte sum of the pattern the CPU wrote via mmap == sum computed by the GPU (heap_sum.comp)
//
// ── After steps 2·3 (Appendix B.13) ──────────────────────────────────────────────────────────
// Allocation, mapping and sync are all done by **`yvo::DmaHeapPool` in `core/dma_heap_alloc.hpp`** (step-2 unification).
// The inline implementation that lived in the probe (`HeapAlloc`/`HeapBuf`/`dma_sync`) was removed.
// The allocation size is **`align_up(hw + device_headroom, 16384)`** (step 3, B.12) — `device_headroom` is
// obtained by querying `vk::StorageHeadroomQuery`. **No hard-coded constant 256.**
// import is given `min_range = hw` so it throws if `range < hw` (B.12 enforced check).

constexpr std::size_t kHeapAlign = yvo::vk::kHailoAsyncAlign;   // 16384
constexpr std::uint32_t kSumGroups = 64;                        // workgroup count of the heap_sum.comp dispatch (fixed)
constexpr std::uint32_t kPatternMul = 2654435761u;              // Knuth. value of word i = i * kPatternMul

std::size_t align_up_sz(std::size_t v, std::size_t a) { return a <= 1 ? v : (v + a - 1) / a * a; }

std::string run_capture(const char* cmd) {
    std::string out;
    FILE* p = ::popen(cmd, "r");
    if (!p) return "(popen failed)";
    char buf[512];
    while (std::fgets(buf, sizeof(buf), p)) out += buf;
    ::pclose(p);
    return out.empty() ? std::string("(no output)") : out;
}

// Calls DMA_BUF_IOCTL_SYNC non-strictly and only records errno (the probe exists for the verdict).
// The real implementation is `yvo::dma_sync_*` (core/dma_heap_alloc.hpp) — Appendix B.10 R5.
int dma_sync_note(int fd, yvo::DmaSyncDir dir, bool start) {
    const std::uint64_t flags =
        (start ? DMA_BUF_SYNC_START : DMA_BUF_SYNC_END) | yvo::dma_sync_dir_bits(dir);
    return yvo::detail::dma_sync_raw(fd, flags);
}

// heap_sum.comp runner. Creates the pipeline, descriptors and command buffer once and reuses them for the 6 tensors.
class HeapSumRig {
public:
    explicit HeapSumRig(yvo::vk::VulkanContext& ctx) : ctx_(ctx) {
        VkDevice dev = ctx_.device();
        std::array<VkDescriptorSetLayoutBinding, 2> b{};
        for (std::uint32_t i = 0; i < b.size(); ++i) {
            b[i].binding = i;
            b[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            b[i].descriptorCount = 1;
            b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo sli{};
        sli.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        sli.bindingCount = static_cast<std::uint32_t>(b.size());
        sli.pBindings = b.data();
        YVO_VK_CHECK(vkCreateDescriptorSetLayout(dev, &sli, nullptr, &set_layout_));

        VkPushConstantRange pcr{};
        pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcr.offset = 0;
        pcr.size = sizeof(std::uint32_t);
        VkPipelineLayoutCreateInfo pli{};
        pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pli.setLayoutCount = 1;
        pli.pSetLayouts = &set_layout_;
        pli.pushConstantRangeCount = 1;
        pli.pPushConstantRanges = &pcr;
        YVO_VK_CHECK(vkCreatePipelineLayout(dev, &pli, nullptr, &pipe_layout_));

        VkShaderModuleCreateInfo smi{};
        smi.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        smi.codeSize = sizeof(kHeapSumSpv);
        smi.pCode = kHeapSumSpv;
        VkShaderModule mod = VK_NULL_HANDLE;
        YVO_VK_CHECK(vkCreateShaderModule(dev, &smi, nullptr, &mod));
        VkComputePipelineCreateInfo cpi{};
        cpi.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        cpi.stage.module = mod;
        cpi.stage.pName = "main";
        cpi.layout = pipe_layout_;
        const VkResult pr = vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpi, nullptr, &pipeline_);
        vkDestroyShaderModule(dev, mod, nullptr);
        YVO_VK_CHECK(pr);

        VkDescriptorPoolSize ps{};
        ps.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        ps.descriptorCount = 2;
        VkDescriptorPoolCreateInfo dpi{};
        dpi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpi.maxSets = 1;
        dpi.poolSizeCount = 1;
        dpi.pPoolSizes = &ps;
        YVO_VK_CHECK(vkCreateDescriptorPool(dev, &dpi, nullptr, &pool_));
        VkDescriptorSetAllocateInfo dsi{};
        dsi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dsi.descriptorPool = pool_;
        dsi.descriptorSetCount = 1;
        dsi.pSetLayouts = &set_layout_;
        YVO_VK_CHECK(vkAllocateDescriptorSets(dev, &dsi, &set_));

        // Result buffer {sum, words} 8 B. V3D has only one memory type, so it is HOST_VISIBLE|HOST_COHERENT.
        VkBufferCreateInfo bci{};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size = sizeof(std::uint32_t) * 2;
        bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        YVO_VK_CHECK(vkCreateBuffer(dev, &bci, nullptr, &dst_buf_));
        VkMemoryRequirements req{};
        vkGetBufferMemoryRequirements(dev, dst_buf_, &req);
        const std::uint32_t type = ctx_.find_memory_type(
            req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (type == UINT32_MAX) throw std::runtime_error("HeapSumRig: no HOST_VISIBLE|HOST_COHERENT memory type");
        VkMemoryAllocateInfo mai{};
        mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mai.allocationSize = req.size;
        mai.memoryTypeIndex = type;
        YVO_VK_CHECK(vkAllocateMemory(dev, &mai, nullptr, &dst_mem_));
        YVO_VK_CHECK(vkBindBufferMemory(dev, dst_buf_, dst_mem_, 0));
        void* p = nullptr;
        YVO_VK_CHECK(vkMapMemory(dev, dst_mem_, 0, VK_WHOLE_SIZE, 0, &p));
        dst_host_ = static_cast<std::uint32_t*>(p);

        VkCommandPoolCreateInfo cpci{};
        cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        cpci.queueFamilyIndex = ctx_.queue_family();
        YVO_VK_CHECK(vkCreateCommandPool(dev, &cpci, nullptr, &cmd_pool_));
        VkCommandBufferAllocateInfo cbi{};
        cbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cbi.commandPool = cmd_pool_;
        cbi.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbi.commandBufferCount = 1;
        YVO_VK_CHECK(vkAllocateCommandBuffers(dev, &cbi, &cmd_));
        VkFenceCreateInfo fci{};
        fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        YVO_VK_CHECK(vkCreateFence(dev, &fci, nullptr, &fence_));
    }

    ~HeapSumRig() {
        VkDevice dev = ctx_.device();
        if (fence_) vkDestroyFence(dev, fence_, nullptr);
        if (cmd_pool_) vkDestroyCommandPool(dev, cmd_pool_, nullptr);
        if (dst_host_) vkUnmapMemory(dev, dst_mem_);
        if (dst_buf_) vkDestroyBuffer(dev, dst_buf_, nullptr);
        if (dst_mem_) vkFreeMemory(dev, dst_mem_, nullptr);
        if (pool_) vkDestroyDescriptorPool(dev, pool_, nullptr);
        if (pipeline_) vkDestroyPipeline(dev, pipeline_, nullptr);
        if (pipe_layout_) vkDestroyPipelineLayout(dev, pipe_layout_, nullptr);
        if (set_layout_) vkDestroyDescriptorSetLayout(dev, set_layout_, nullptr);
    }
    HeapSumRig(const HeapSumRig&) = delete;
    HeapSumRig& operator=(const HeapSumRig&) = delete;

    // Byte-wise sum of the first n_words words of src. groups receives the number of workgroups actually run.
    std::uint32_t run(VkBuffer src, VkDeviceSize range, std::uint32_t n_words, std::uint32_t& groups) {
        VkDevice dev = ctx_.device();
        dst_host_[0] = 0;
        dst_host_[1] = 0;

        std::array<VkDescriptorBufferInfo, 2> bi{};
        bi[0].buffer = src;
        bi[0].offset = 0;
        bi[0].range = range;
        bi[1].buffer = dst_buf_;
        bi[1].offset = 0;
        bi[1].range = VK_WHOLE_SIZE;
        std::array<VkWriteDescriptorSet, 2> w{};
        for (std::uint32_t i = 0; i < w.size(); ++i) {
            w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[i].dstSet = set_;
            w[i].dstBinding = i;
            w[i].descriptorCount = 1;
            w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            w[i].pBufferInfo = &bi[i];
        }
        vkUpdateDescriptorSets(dev, static_cast<std::uint32_t>(w.size()), w.data(), 0, nullptr);

        YVO_VK_CHECK(vkResetCommandBuffer(cmd_, 0));
        VkCommandBufferBeginInfo cbb{};
        cbb.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        cbb.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        YVO_VK_CHECK(vkBeginCommandBuffer(cmd_, &cbb));

        const std::uint32_t q = ctx_.queue_family();
        const std::uint32_t ext = ctx_.external_queue_family();

        // ① external (memory the CPU wrote through the dma-buf mapping) → compute queue acquire + make result buffer host writes available.
        //    Appendix B.10: the sync ioctl handles only the **CPU caches**, the Vulkan barrier only the **V3D caches**. Both are needed.
        std::array<VkBufferMemoryBarrier2, 2> acq{};
        acq[0].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
        acq[0].srcStageMask = VK_PIPELINE_STAGE_2_NONE;
        acq[0].srcAccessMask = VK_ACCESS_2_NONE;
        acq[0].dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        acq[0].dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
        acq[0].srcQueueFamilyIndex = ext;
        acq[0].dstQueueFamilyIndex = q;
        acq[0].buffer = src;
        acq[0].offset = 0;
        acq[0].size = VK_WHOLE_SIZE;
        acq[1].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
        acq[1].srcStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
        acq[1].srcAccessMask = VK_ACCESS_2_HOST_WRITE_BIT;
        acq[1].dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        acq[1].dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        acq[1].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        acq[1].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        acq[1].buffer = dst_buf_;
        acq[1].offset = 0;
        acq[1].size = VK_WHOLE_SIZE;
        VkDependencyInfo di{};
        di.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        di.bufferMemoryBarrierCount = static_cast<std::uint32_t>(acq.size());
        di.pBufferMemoryBarriers = acq.data();
        vkCmdPipelineBarrier2(cmd_, &di);

        vkCmdBindPipeline(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
        vkCmdBindDescriptorSets(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_layout_, 0, 1, &set_, 0, nullptr);
        vkCmdPushConstants(cmd_, pipe_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(n_words), &n_words);
        vkCmdDispatch(cmd_, kSumGroups, 1, 1);

        // ② release input (return to external) + result shader write → make visible to host read
        std::array<VkBufferMemoryBarrier2, 2> rel{};
        rel[0] = acq[0];
        rel[0].srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        rel[0].srcAccessMask = VK_ACCESS_2_NONE;   // only read, so no writes to make available
        rel[0].dstStageMask = VK_PIPELINE_STAGE_2_NONE;
        rel[0].dstAccessMask = VK_ACCESS_2_NONE;
        rel[0].srcQueueFamilyIndex = q;
        rel[0].dstQueueFamilyIndex = ext;
        rel[1] = acq[1];
        rel[1].srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        rel[1].srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        rel[1].dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
        rel[1].dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;
        di.bufferMemoryBarrierCount = static_cast<std::uint32_t>(rel.size());
        di.pBufferMemoryBarriers = rel.data();
        vkCmdPipelineBarrier2(cmd_, &di);
        YVO_VK_CHECK(vkEndCommandBuffer(cmd_));

        YVO_VK_CHECK(vkResetFences(dev, 1, &fence_));
        VkSubmitInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd_;
        YVO_VK_CHECK(vkQueueSubmit(ctx_.queue(), 1, &si, fence_));
        const VkResult wr = vkWaitForFences(dev, 1, &fence_, VK_TRUE, 5ull * 1000 * 1000 * 1000);
        YVO_VK_CHECK(wr);
        groups = dst_host_[1];
        return dst_host_[0];
    }

private:
    yvo::vk::VulkanContext& ctx_;
    VkDescriptorSetLayout set_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout pipe_layout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    VkDescriptorPool pool_ = VK_NULL_HANDLE;
    VkDescriptorSet set_ = VK_NULL_HANDLE;
    VkBuffer dst_buf_ = VK_NULL_HANDLE;
    VkDeviceMemory dst_mem_ = VK_NULL_HANDLE;
    std::uint32_t* dst_host_ = nullptr;
    VkCommandPool cmd_pool_ = VK_NULL_HANDLE;
    VkCommandBuffer cmd_ = VK_NULL_HANDLE;
    VkFence fence_ = VK_NULL_HANDLE;
};

// P0 record for one tensor
struct HeapImportRow {
    std::string name;
    std::size_t hw = 0, want = 0, dmabuf = 0;
    std::size_t headroom = 0;   // B.12: slack the driver required for this tensor size (queried value)
    bool mmap_aligned = false;
    std::uint32_t fd_type_bits = 0;
    // requirements of a dummy VkBuffer created with the dma-buf size (step 2 of B.12)
    VkDeviceSize req_size_dmabuf = 0, req_align = 0;
    std::uint32_t req_type_bits = 0;
    bool requires_ded = false, prefers_ded = false;
    // requirements of a dummy created with hw_frame_size → headroom = req - hw
    VkDeviceSize req_size_hw = 0;
    // import result
    bool imported = false;
    VkDeviceSize imp_size = 0, imp_range = 0;
    std::string import_error;
    // round trip
    bool roundtrip_ok = false;
    std::uint32_t gpu_sum = 0, cpu_sum = 0, groups = 0;
    std::uint32_t n_words = 0;
    int sync_err_begin = 0, sync_err_end = 0;
};

// Queries the memory requirements of a dummy VkBuffer for external memory of `requested` bytes (does not allocate).
// The implementation moved to vulkan/vk_storage_headroom.hpp (Appendix B.12 — other callers use it too).
void query_dummy_req(yvo::vk::VulkanContext& ctx, VkDeviceSize requested, VkMemoryRequirements& out,
                     VkMemoryDedicatedRequirements& ded) {
    ded = VkMemoryDedicatedRequirements{};
    ded.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS;
    out = yvo::vk::query_dmabuf_storage_requirements(ctx.device(), requested, &ded);
}

int run_heapimport(const Options& o, Tee& log) {
    const FixtureSet fs = load_fixtures(o.fixture_dir);
    const long fd0 = count_open_fds();

    yvo::vk::VulkanContextOptions co;
    co.enable_validation = o.validation;
    co.validation_sync = o.sync_validation;
    co.verbose = o.verbose;
    yvo::vk::VulkanContext ctx(co);
    log(fmt("[vk] device=%s  dmabuf import=%d (dedicated_only=%d) export=%d", ctx.device_name(),
            ctx.dmabuf_import_supported() ? 1 : 0, ctx.dmabuf_import_dedicated_only() ? 1 : 0,
            ctx.dmabuf_export_supported() ? 1 : 0));
    log(fmt("[vk] validation layer=%d sync=%d  external_queue_family=%u queue_family=%u",
            ctx.validation_layer_active() ? 1 : 0, o.sync_validation ? 1 : 0, ctx.external_queue_family(),
            ctx.queue_family()));
    {
        const VkPhysicalDeviceMemoryProperties& mp = ctx.memory_properties();
        log(fmt("[vk] memoryTypeCount=%u heapCount=%u", mp.memoryTypeCount, mp.memoryHeapCount));
        for (std::uint32_t i = 0; i < mp.memoryTypeCount; ++i)
            log(fmt("      type %u: heap %u flags 0x%03x", i, mp.memoryTypes[i].heapIndex,
                    static_cast<unsigned>(mp.memoryTypes[i].propertyFlags)));
    }

    // fd leak baseline: fds opened by VulkanContext are not counted (the context outlives this block).
    const long fd_base = count_open_fds();

    std::vector<HeapImportRow> rows;
    bool hard_stop = false;
    std::uint64_t imports = 0, hits = 0;
    // Put heap, import and pipeline in an inner scope, and check that the fd count returns to the baseline after leaving it.
    {
    // ── step 3 (B.12): obtain the headroom by **query**. No hard-coded constant. ──────────────────
    // DmaHeapConfig::device_headroom is one value per pool, so the max over the 6 tensors is used
    // (V3D is 256 regardless of size, so all 6 are equal. Over-allocation is safe — requires/prefers = 0).
    yvo::vk::StorageHeadroomQuery hq(ctx.device());
    std::vector<std::size_t> tensor_sizes;
    tensor_sizes.reserve(fs.streams.size());
    for (const FixtureStream& t : fs.streams) tensor_sizes.push_back(t.bytes);
    const std::size_t queried = hq.max_headroom(tensor_sizes.begin(), tensor_sizes.end());
    const std::size_t headroom = o.no_headroom ? 0 : queried;
    log(fmt("[headroom] %zu queries → queried value %zu B (req.alignment = %llu B)%s. allocation size = align_up(hw + %zu, %zu)",
            hq.queries(), queried, static_cast<unsigned long long>(hq.last_alignment()),
            o.no_headroom ? "  **--no-headroom: forced to 0 (negative control)**" : "", headroom, kHeapAlign));

    yvo::DmaHeapConfig hc;
    if (!o.heap.empty()) hc.heaps = {o.heap};
    hc.align = kHeapAlign;
    hc.device_headroom = headroom;
    hc.map = true;
    hc.strict_sync = true;
    hc.zero_init = true;
    hc.name = "yvo_p0_heapimport";
    yvo::DmaHeapPool heap(hc);
    log(fmt("[heap] %s (yvo::DmaHeapPool — Appendix B.11 unified API)", heap.heap().c_str()));

    HeapSumRig rig(ctx);
    yvo::vk::DmaBufImportCache cache(ctx, 16);

    for (const FixtureStream& t : fs.streams) {
        HeapImportRow r;
        r.name = t.file;
        r.hw = t.bytes;
        r.headroom = headroom;   // actually applied value (0 with --no-headroom). The queried value is on the [headroom] line
        r.want = align_up_sz(t.bytes + headroom, kHeapAlign);
        const int hi = heap.allocate(t.bytes);
        const yvo::DmaHeapBuf hb = heap.at(hi);   // POD copy — safe even if the next allocate() reallocates the vector
        r.dmabuf = hb.dmabuf_size;
        r.mmap_aligned = reinterpret_cast<std::uintptr_t>(hb.ptr) % kHeapAlign == 0;

        // ① memory types that can import this fd — **if 0, S1 hard stop**
        VkMemoryFdPropertiesKHR fdp{};
        fdp.sType = VK_STRUCTURE_TYPE_MEMORY_FD_PROPERTIES_KHR;
        const VkResult fr = ctx.vkGetMemoryFdPropertiesKHR(
            ctx.device(), VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT, hb.fd, &fdp);
        if (fr != VK_SUCCESS) {
            r.import_error = std::string("vkGetMemoryFdPropertiesKHR: ") + yvo::vk::result_str(fr);
            hard_stop = true;
            rows.push_back(r);
            continue;
        }
        r.fd_type_bits = fdp.memoryTypeBits;
        if (fdp.memoryTypeBits == 0) {
            r.import_error = "memoryTypeBits == 0 (V3DV cannot import this dma-heap fd)";
            hard_stop = true;
            rows.push_back(r);
            continue;
        }

        // ②·③ size requirements and dedicated flags via dummy buffers (steps 1~2 of B.12)
        {
            VkMemoryRequirements req{};
            VkMemoryDedicatedRequirements ded{};
            query_dummy_req(ctx, r.dmabuf, req, ded);
            r.req_size_dmabuf = req.size;
            r.req_align = req.alignment;
            r.req_type_bits = req.memoryTypeBits;
            r.requires_ded = ded.requiresDedicatedAllocation != VK_FALSE;
            r.prefers_ded = ded.prefersDedicatedAllocation != VK_FALSE;
            VkMemoryRequirements req_hw{};
            VkMemoryDedicatedRequirements ded_hw{};
            query_dummy_req(ctx, r.hw, req_hw, ded_hw);
            r.req_size_hw = req_hw.size;
        }

        // ④·⑤ actual import — **uses the existing DmaBufImportCache::get() as is**
        const yvo::vk::ImportedDmaBuf* imp = nullptr;
        try {
            // min_range = hw → throws if range < hw (B.12 enforced check)
            imp = &cache.get(hb.fd, hb.dmabuf_size, r.hw);
            r.imported = true;
            r.imp_size = imp->size;
            r.imp_range = imp->range;
        } catch (const std::exception& ex) {
            r.import_error = ex.what();
            hard_stop = true;
            rows.push_back(r);
            continue;
        }

        // ⑦ round trip: does the GPU read the pattern the CPU wrote as the same value
        r.sync_err_begin = dma_sync_note(hb.fd, yvo::DmaSyncDir::Write, /*start=*/true);
        const std::size_t total_words = hb.dmabuf_size / 4;
        auto* wp = reinterpret_cast<std::uint32_t*>(hb.ptr);
        for (std::size_t i = 0; i < total_words; ++i)
            wp[i] = static_cast<std::uint32_t>(i) * kPatternMul;
        r.sync_err_end = dma_sync_note(hb.fd, yvo::DmaSyncDir::Write, /*start=*/false);

        r.n_words = static_cast<std::uint32_t>(std::min<VkDeviceSize>(r.imp_range, hb.dmabuf_size) / 4);
        std::uint32_t cpu = 0;
        for (std::uint32_t i = 0; i < r.n_words; ++i) {
            const std::uint32_t v = wp[i];
            cpu += (v & 0xffu) + ((v >> 8) & 0xffu) + ((v >> 16) & 0xffu) + (v >> 24);
        }
        r.cpu_sum = cpu;
        r.gpu_sum = rig.run(imp->buffer, r.imp_range, r.n_words, r.groups);
        r.roundtrip_ok = r.gpu_sum == r.cpu_sum && r.groups == kSumGroups;
        rows.push_back(r);
    }
    imports = cache.imports();
    hits = cache.hits();
    cache.clear();   // close every imported dup fd (vkFreeMemory closes them together)
    }
    const long fd_after = count_open_fds();

    // ── report ───────────────────────────────────────────────────────────────────────────────
    log("");
    log("== per-tensor records ==");
    for (const HeapImportRow& r : rows) {
        log(fmt("[%s] hw=%zu headroom=%zu want=align_up(hw+hr,%zu)=%zu dmabuf=%zu  mmap 16384-aligned=%s",
                r.name.c_str(), r.hw, r.headroom, kHeapAlign, r.want, r.dmabuf,
                r.mmap_aligned ? "yes" : "**no**"));
        log(fmt("    (1) VkMemoryFdPropertiesKHR::memoryTypeBits = 0x%08x", r.fd_type_bits));
        if (!r.import_error.empty()) {
            log(fmt("    !! failed: %s", r.import_error.c_str()));
            continue;
        }
        log(fmt("    (2) req(size=dmabuf %zu) = %llu  → slack %+lld B | align=%llu typeBits=0x%08x", r.dmabuf,
                static_cast<unsigned long long>(r.req_size_dmabuf),
                static_cast<long long>(r.req_size_dmabuf) - static_cast<long long>(r.dmabuf),
                static_cast<unsigned long long>(r.req_align), r.req_type_bits));
        log(fmt("        req(size=hw %zu) = %llu  → **headroom %+lld B** (value B.12 will cache)", r.hw,
                static_cast<unsigned long long>(r.req_size_hw),
                static_cast<long long>(r.req_size_hw) - static_cast<long long>(r.hw)));
        log(fmt("    (3) requiresDedicatedAllocation=%d prefersDedicatedAllocation=%d", r.requires_ded ? 1 : 0,
                r.prefers_ded ? 1 : 0));
        const bool shrunk = r.imp_range < r.imp_size;
        log(fmt("    (4) ImportedDmaBuf: size=%llu range=%llu  %s",
                static_cast<unsigned long long>(r.imp_size), static_cast<unsigned long long>(r.imp_range),
                shrunk ? "***** range < size — shrink kicked in *****" : "range == size (no shrink)"));
        if (shrunk)
            log(fmt("        ⇒ shrink %lld B. vs hw: %s (range %llu vs hw %zu, slack %+lld B)",
                    static_cast<long long>(r.imp_size) - static_cast<long long>(r.imp_range),
                    r.imp_range >= r.hw ? "still within slack" : "***** cuts into hw *****",
                    static_cast<unsigned long long>(r.imp_range), r.hw,
                    static_cast<long long>(r.imp_range) - static_cast<long long>(r.hw)));
        log(fmt("    (5) vkAllocateMemory = VK_SUCCESS"));
        log(fmt("    (6) DMA_BUF_IOCTL_SYNC START|WRITE=%d END|WRITE=%d (0 = success)", r.sync_err_begin,
                r.sync_err_end));
        log(fmt("    (7) round trip: n_words=%u groups=%u  gpu=0x%08x cpu=0x%08x  %s", r.n_words, r.groups, r.gpu_sum,
                r.cpu_sum, r.roundtrip_ok ? "match" : "***** mismatch *****"));
    }

    // ── verdict (Appendix B.14 S1) ───────────────────────────────────────────────────────────
    int n_imported = 0, n_shrunk = 0, n_roundtrip = 0, n_unaligned = 0, n_range_ok = 0;
    for (const HeapImportRow& r : rows) {
        if (r.imported) ++n_imported;
        if (r.imported && r.imp_range < r.imp_size) ++n_shrunk;
        if (r.imported && r.imp_range >= r.hw) ++n_range_ok;   // **step-3 gate**
        if (r.roundtrip_ok) ++n_roundtrip;
        if (!r.mmap_aligned) ++n_unaligned;
    }
    const std::uint64_t vk_errors = ctx.validation_errors();
    log("");
    log(fmt("[summary] import %d/%zu · **range >= hw %d/%zu** · round trip match %d/%zu · range<size %d · "
            "mmap unaligned %d · validation layer errors %llu",
            n_imported, rows.size(), n_range_ok, rows.size(), n_roundtrip, rows.size(), n_shrunk, n_unaligned,
            static_cast<unsigned long long>(vk_errors)));

    std::string verdict;
    int rc = 0;
    if (hard_stop || n_imported != static_cast<int>(rows.size())) {
        verdict = o.no_headroom ? "FAIL — **--no-headroom negative control**. Without headroom it breaks with range < hw (expected result)"
                                : "FAIL (hard stop) — S1. Buffer migration fully halted";
        rc = 1;
        log("");
        log("== dmesg (v3d/dma) ==");
        log(run_capture("dmesg 2>&1 | grep -iE 'v3d|dma-?buf|dma_heap' | tail -n 30"));
    } else if (n_roundtrip != static_cast<int>(rows.size())) {
        verdict = "coherency problem — round trip mismatch. Retry sync direction/timing";
        rc = 1;
    } else if (n_range_ok != static_cast<int>(rows.size())) {
        // step-3 gate failure = Appendix B.14 S4. The headroom did not solve it.
        verdict = "FAIL — there is a tensor with range < hw (S4). Headroom is insufficient";
        rc = 1;
    } else if (n_shrunk > 0) {
        // range < size is normal — the headroom was added in advance, so the shrink does not encroach on hw.
        verdict = "PASS — 6/6 range >= hw (headroom applied), round trip match. range < size is by design";
    } else {
        verdict = "PASS — all succeeded, range == size, round trip match";
    }
    if (vk_errors != 0) {
        verdict += " / validation layer errors != 0";
        rc = 1;
    }
    log(fmt("[S1/S4 verdict] %s", verdict.c_str()));
    log(fmt("[import cache] imports=%llu hits=%llu", static_cast<unsigned long long>(imports),
            static_cast<unsigned long long>(hits)));
    log(fmt("[fd] entry %ld · baseline after context %ld · after heap/import scope ends %ld → leak %ld", fd0, fd_base,
            fd_after, fd_after - fd_base));
    if (fd_after != fd_base) rc = 1;
    log(fmt("[RESULT] heapimport %s", rc == 0 ? "PASS" : "FAIL"));
    return rc;
}

// ───────────────────────────────────── nms (GPU NMS cost on its own, **no NPU needed**, plan A3)
//
// formal_measurement_plan.md §4 A3: the GPU cost of efficient_nms.comp without the NPU.
//   NMS-(1) bit-exact sanity: GPU NMS == cpu_nms_greedy_by_class() on the same GPU candidates (fixture frames)
//   NMS-(2) the P3-⑥ continuous-stream loop with enable_nms — pass1 / nms / cmdbuf timestamps per frame.
//           nms = last_nms_ms() = ts3→ts4, the same interval the pipeline reports as `gpu nms ms`.
//   NMS-(3) synthetic worst case through testkit/nms_harness.hpp (the EfficientNms.* test harness):
//           n candidates that all survive (non-overlapping boxes, <= max_per_class per class), so each of the
//           n main-loop iterations is a broadcast plus a same-class IoU sweep. n = 1024 is kMaxSelected.
// A single submit/wait (EfficientNmsBitExact's `gpu nms max`) is inflated by DVFS, so only the back-to-back
// streams here are measured. Not a pipeline number: the NPU is idle, so there is no memory-bandwidth contention.

// Order-sensitive bit comparison (count, order, class, anchor_key, score, box) — the rule of same_bits() in
// 01_src/pipeline/gpu_postproc_stage.cpp, which decides `nms verify mismatch`.
bool same_bits_ordered(const std::vector<DetectionKeyed>& a, const std::vector<DetectionKeyed>& b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const DetectionKeyed& x = a[i];
        const DetectionKeyed& y = b[i];
        if (x.anchor_key != y.anchor_key || x.class_id != y.class_id || bits_of(x.score) != bits_of(y.score) ||
            bits_of(x.x1) != bits_of(y.x1) || bits_of(x.y1) != bits_of(y.y1) || bits_of(x.x2) != bits_of(y.x2) ||
            bits_of(x.y2) != bits_of(y.y2))
            return false;
    }
    return true;
}

// n non-overlapping boxes (IoU 0 for every pair) on a 40-column grid over 1920x1080, class = i % ceil(n/max_per_class)
// so no class exceeds its quota → every candidate survives. Scores are quantized like the fixture (ties occur;
// anchor_key breaks them). The input order is shuffled with a fixed seed (the bitonic network is data-independent).
std::vector<DetectionKeyed> nms_worst_case(int n, int max_per_class) {
    const int classes = std::max(1, (n + max_per_class - 1) / max_per_class);
    const int cols = 40;
    const int rows = std::max(1, (n + cols - 1) / cols);
    const float cw = static_cast<float>(kSrcW) / static_cast<float>(cols);
    const float ch = static_cast<float>(kSrcH) / static_cast<float>(rows);
    std::vector<DetectionKeyed> v;
    v.reserve(static_cast<std::size_t>(n));
    std::uint32_t lcg = 12345u;
    for (int i = 0; i < n; ++i) {
        lcg = lcg * 1664525u + 1013904223u;
        DetectionKeyed d;
        d.x1 = static_cast<float>(i % cols) * cw + 2.0f;
        d.y1 = static_cast<float>(i / cols) * ch + 2.0f;
        d.x2 = d.x1 + cw - 4.0f;
        d.y2 = d.y1 + ch - 4.0f;
        d.score = 0.2f + static_cast<float>((lcg >> 8) % 60u) * 0.0125f;
        d.class_id = i % classes;
        d.anchor_key = static_cast<std::uint32_t>(i);
        v.push_back(d);
    }
    for (std::size_t i = v.size(); i > 1; --i) {   // Fisher–Yates with the same LCG (no <random> dependency)
        lcg = lcg * 1664525u + 1013904223u;
        std::swap(v[i - 1], v[(lcg >> 8) % i]);
    }
    return v;
}

std::vector<int> parse_int_list(const std::string& s) {
    std::vector<int> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ','))
        if (!tok.empty()) out.push_back(std::stoi(tok));
    return out;
}

std::vector<std::string> parse_name_list(const std::string& s) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ','))
        if (!tok.empty()) out.push_back(tok);
    return out;
}

int run_nms(const Options& o, Tee& log) {
    constexpr float kIouThr = 0.7f;      // HEF setting (gpu_postproc_stage.hpp)
    constexpr int kMaxPerClass = 100;    // HEF setting
    if (o.perf_frames < 1) throw std::runtime_error("--perf-frames must be >= 1");
    FixtureRig rig = load_rig(o);
    log(fmt("[fixture] dir=%s frames=%d  score_thr=%.4f (HEF %.2f)", o.fixture_dir.c_str(), rig.fs.frames,
            static_cast<double>(o.nms_score), static_cast<double>(kHefScoreThreshold)));

    yvo::vk::VulkanContextOptions co;
    co.enable_validation = o.validation;
    co.validation_sync = o.sync_validation;
    co.verbose = o.verbose;
    yvo::vk::VulkanContext ctx(co);
    log(fmt("[vk] device=%s  timestamps=%d period=%.3f ns valid_bits=%u", ctx.device_name(),
            ctx.timestamps_supported() ? 1 : 0, static_cast<double>(ctx.timestamp_period()),
            ctx.timestamp_valid_bits()));
    if (!ctx.timestamps_supported()) throw std::runtime_error("--probe nms needs timestamp queries");

    YoloPostprocConfig pc;
    pc.num_slots = std::max(1, std::min(o.slots, rig.fs.frames));
    pc.max_dets = o.max_dets;
    pc.local_size_x = o.lx;
    pc.local_size_y = o.ly;
    pc.score_thr = o.nms_score;
    pc.input_alloc = YoloInputAlloc::Plain;
    pc.input_sync = YoloInputSync::HostWrite;
    pc.enable_nms = true;
    pc.nms.iou_thr = kIouThr;
    pc.nms.max_per_class = kMaxPerClass;
    pc.verbose = o.verbose;
    YoloPostproc pp(ctx, rig.layout, pc);
    const yvo::vk::EfficientNmsConfig& nc = pp.nms()->config();
    log(fmt("[cfg] slots=%d max_dets=%d local=%ux%u  nms local=%u max_selected=%u iou=%.2f max_per_class=%d "
            "shared=%zu B",
            pp.num_slots(), pc.max_dets, pc.local_size_x, pc.local_size_y, nc.local_size, nc.max_selected,
            static_cast<double>(nc.iou_thr), nc.max_per_class, pp.nms()->shared_bytes()));
    const int nslots = pp.num_slots();
    for (int s = 0; s < nslots; ++s) fill_slot(pp, s, rig.frames[static_cast<std::size_t>(s)]);

    // ── NMS-(1) bit-exact sanity ────────────────────────────────────────────────────────────
    log("");
    log("== NMS-(1) GPU NMS vs cpu_nms_greedy_by_class() on the same GPU candidates (ordered bit comparison) ==");
    log("frame  cand  gpu_kept  cpu_kept  fallback  match");
    bool exact_ok = true;
    int fallback_frames = 0;
    for (int s = 0; s < nslots; ++s) {
        pp.submit(s, rig.fs.lb, rig.fs.src_w, rig.fs.src_h);
        pp.wait(s);
        std::vector<DetectionKeyed> gpu;
        const bool ok = pp.read_nms(s, gpu);
        auto cpu = pp.read_candidates(s);
        const std::size_t cand = cpu.size();
        yvo::cpu_nms_greedy_by_class(cpu, kIouThr, kMaxPerClass);
        const bool eq = ok && same_bits_ordered(gpu, cpu);
        if (!ok) ++fallback_frames;
        exact_ok = exact_ok && (eq || !ok);   // a fallback frame is the CPU's job by contract, not a mismatch
        log(fmt("%5d  %4zu  %8zu  %8zu  %8u  %s", s, cand, gpu.size(), cpu.size(), pp.nms_fallback_reason(s),
                !ok ? "fallback" : (eq ? "ok" : "MISMATCH")));
    }
    log(fmt("[NMS-(1)] %s  fallback frames %d/%d", exact_ok ? "PASS" : "FAIL", fallback_frames, nslots));

    // ── NMS-(2) continuous stream (P3-⑥ loop + enable_nms) ──────────────────────────────────
    log("");
    log(fmt("== NMS-(2) decode + NMS in one command buffer, continuous stream of %d frames (warm-up %d) ==",
            o.perf_frames, o.warmup));
    std::vector<double> pass1, nms, total;
    std::size_t cand_sum = 0, cand_max = 0;
    int stream_fallback = 0;
    for (int i = 0; i < o.warmup + o.perf_frames; ++i) {
        const int s = i % nslots;
        pp.submit(s, rig.fs.lb, rig.fs.src_w, rig.fs.src_h);
        pp.wait(s);
        if (i >= o.warmup) {
            pass1.push_back(pp.last_pass1_ms());
            nms.push_back(pp.last_nms_ms());
            total.push_back(pp.last_total_ms());
            const std::size_t c = pp.raw_count(s);
            cand_sum += c;
            cand_max = std::max(cand_max, c);
            if (pp.nms_fallback_reason(s) != 0) ++stream_fallback;
        }
    }
    auto line = [&](const char* name, const std::vector<double>& v) {
        log(fmt("%-9s median %.4f  p90 %.4f  min %.4f  max %.4f  n=%zu", name, percentile(v, 0.5), percentile(v, 0.9),
                *std::min_element(v.begin(), v.end()), *std::max_element(v.begin(), v.end()), v.size()));
    };
    line("pass1_ms", pass1);
    line("nms_ms", nms);
    line("cmdbuf_ms", total);
    log(fmt("candidates/frame mean %.2f max %zu  fallback frames %d", static_cast<double>(cand_sum) /
            static_cast<double>(std::max<std::size_t>(1, pass1.size())), cand_max, stream_fallback));

    // ── NMS-(3) synthetic worst case (testkit NmsHarness) ───────────────────────────────────
    log("");
    log(fmt("== NMS-(3) synthetic all-survive input via NmsHarness, continuous stream of %d runs per n (warm-up %d) ==",
            o.perf_frames, o.warmup));
    log("    n  classes  cpu_kept  fallback  match  nms_median_ms  p90     min     max");
    yvo::vk::EfficientNmsConfig hc;
    hc.iou_thr = kIouThr;
    hc.max_per_class = kMaxPerClass;
    NmsHarness h(ctx, hc);
    bool synth_ok = true;
    for (const int n : parse_int_list(o.nms_cands)) {
        if (n < 0 || n > static_cast<int>(NmsHarness::kInputMax))
            throw std::runtime_error(fmt("--nms-cand %d outside 0..%u", n, NmsHarness::kInputMax));
        const auto in = nms_worst_case(n, kMaxPerClass);
        std::vector<DetectionKeyed> out;
        std::uint32_t why = 0;
        const bool ok = h.run(in, out, &why);
        auto cpu = in;
        yvo::cpu_nms_greedy_by_class(cpu, kIouThr, kMaxPerClass);
        const bool eq = ok && same_bits_ordered(out, cpu);
        // n <= kMaxSelected must finish on the GPU and match; above it the capacity fallback is the expected result.
        const bool expect_gpu = n <= static_cast<int>(nc.max_selected);
        synth_ok = synth_ok && (expect_gpu ? eq : (!ok && (why & yvo::vk::kNmsFallbackCapacity) != 0));
        std::vector<double> ms;
        for (int i = 0; i < o.warmup + o.perf_frames; ++i) {
            h.run(in, out, nullptr);
            if (i >= o.warmup) ms.push_back(h.last_ms());
        }
        log(fmt("%5d  %7d  %8zu  %8u  %5s  %13.4f  %.4f  %.4f  %.4f", n,
                std::max(1, (n + kMaxPerClass - 1) / kMaxPerClass), cpu.size(), why,
                !ok ? "fb" : (eq ? "ok" : "MISM"), percentile(ms, 0.5), percentile(ms, 0.9),
                *std::min_element(ms.begin(), ms.end()), *std::max_element(ms.begin(), ms.end())));
    }
    log(fmt("[NMS-(3)] %s (n <= max_selected must match the CPU bit for bit; n > max_selected must fall back)",
            synth_ok ? "PASS" : "FAIL"));

    log("");
    log(fmt("[validation] errors=%llu warnings=%llu", static_cast<unsigned long long>(ctx.validation_errors()),
            static_cast<unsigned long long>(ctx.validation_warnings())));
    const bool pass = exact_ok && synth_ok && ctx.validation_errors() == 0;
    log(fmt("[RESULT] nms %s", pass ? "PASS" : "FAIL"));
    return pass ? 0 : 1;
}

// ───────────────────────────────────────── stages (pass-1 stage cost by differencing, **no NPU needed**)
//  --probe stages
//      Runs the same fixture stream through four builds of `yolo_decode.comp` and takes differences of the pass 1 GPU time:
//        S1 = stop after ② (class scan + group rejection), S2 = stop after ③ DFL, S3 = stop after ④ letterbox inverse,
//        Full = the embedded production SPIR-V. ①② = S1, ③ = S2 − S1, ④ = S3 − S2, ⑤ emit = Full − S3.
//      S1..S3 are `build/02_shaders/diag/yolo_decode_stop<N>.spv` (-DYVO_DECODE_STOP=N), injected with
//      `YVO_YOLO_DECODE_SPV` (read once in the YoloPostproc constructor). Each cut sinks every component of the
//      stage's result into the header's text[31] with one atomicXor per surviving group, so ⑤ = emit − that atomic.
//      `--variants` selects which builds run (default S1,S2,S3,Full; Full is always added because every difference
//      is taken against it). Two further builds split ⑤ itself, both **emitting intentionally wrong values**:
//        noatomic = non-atomic slot reservation → `Full - noatomic` = the device-atomic share of ⑤
//        norescan = no per-anchor class rescan   → `Full - norescan` = the class-rescan share of ⑤
//      Those two still emit, so their emitted counts are reported and differ from Full's (norescan emits once per
//      passing **anchor**, noatomic undercounts because the increments race) -- correct the difference with the
//      per-candidate marginal cost before reading it as a pure share.
//      `legacy` / `stop1_legacy` are the pre-argmax-shortcut ① + ⑤ (-DYVO_DECODE_LEGACY_EMIT, correct output):
//        `Full - legacy` = the shortcut's net Δ, `S1 - stop1_legacy` = the ① cost of the packed max + SWAR counter.
//      With `legacy` selected, every fixture frame is also decoded by both builds and the emitted candidates are
//      compared as maps anchor_key -> (score bits, class_id, coordinate bits): the shortcut emits the same multiset,
//      so this must match exactly (no NMS permutation to hide behind).
//      Variants are interleaved within each of `--rounds` rounds (DVFS drift), each round `--warmup` + `--perf-frames`
//      submissions cycling the slots. Local size is the YoloPostprocConfig default (the production value).
//      Gate: Full's candidate count == `cpu_yolov8_candidates()` on every fixture frame, stage cuts emit 0
//      candidates, and the emit-split variants emit **non-zero** (that is the dead-code-elimination guard).
int run_stages(const Options& o, Tee& log) {
    if (o.perf_frames < 1) throw std::runtime_error("--perf-frames must be >= 1");
    if (o.rounds < 1) throw std::runtime_error("--rounds must be >= 1");
    FixtureRig rig = load_rig(o);
    log(fmt("[fixture] dir=%s frames=%d  score_thr=%.4f (HEF %.2f)", o.fixture_dir.c_str(), rig.fs.frames,
            static_cast<double>(o.nms_score), static_cast<double>(kHefScoreThreshold)));

    struct Variant {
        const char* name;
        std::string spv;           // empty = embedded production SPIR-V
        bool expect_emit;          // false = stage cut (must emit 0); true = must emit (dead-code-elimination guard)
        std::vector<double> ms;    // pass 1 samples over all rounds
        std::size_t emitted = 0;   // candidates read back from slot 0 after the last round
    };
    // Table order is the interleaving order. 02_shaders/diag/CMakeLists.txt builds every .spv listed here.
    struct VariantDef {
        const char* name;
        const char* spv_name;      // nullptr = embedded production SPIR-V
        bool expect_emit;
    };
    static constexpr VariantDef kDefs[] = {{"S1", "yolo_decode_stop1", false},
                                           {"stop1_legacy", "yolo_decode_stop1_legacy", false},
                                           {"S2", "yolo_decode_stop2", false},
                                           {"S3", "yolo_decode_stop3", false},
                                           {"noatomic", "yolo_decode_noatomic", true},
                                           {"norescan", "yolo_decode_norescan", true},
                                           {"legacy", "yolo_decode_legacy", true},
                                           {"Full", nullptr, true}};
    const std::string dir = YVO_DECODE_STAGE_SPV_DIR;
    const std::vector<std::string> want = parse_name_list(o.variants);
    if (want.empty()) throw std::runtime_error("--variants must not be empty");
    for (const std::string& w : want) {
        bool known = false;
        for (const VariantDef& d : kDefs) known = known || w == d.name;
        if (!known)
            throw std::runtime_error("--variants: unknown variant '" + w + "' (S1,S2,S3,noatomic,norescan,legacy,stop1_legacy,Full)");
    }
    std::vector<Variant> vars;
    for (const VariantDef& d : kDefs) {
        const bool last = d.spv_name == nullptr;   // Full always runs: every difference is taken against it
        if (!last && std::find(want.begin(), want.end(), std::string(d.name)) == want.end()) continue;
        vars.push_back({d.name, last ? std::string() : dir + "/" + d.spv_name + ".spv", d.expect_emit, {}, 0});
    }
    for (const Variant& v : vars)
        if (!v.spv.empty() && !std::filesystem::exists(v.spv))
            throw std::runtime_error("stage SPIR-V missing (build yvo_postproc_probe first): " + v.spv);
    if (const char* pre = std::getenv("YVO_YOLO_DECODE_SPV"); pre && *pre)
        log(fmt("[warn] YVO_YOLO_DECODE_SPV=%s was set; it is ignored (Full always uses the embedded SPIR-V)", pre));

    yvo::vk::VulkanContextOptions co;
    co.enable_validation = o.validation;
    co.validation_sync = o.sync_validation;
    co.verbose = o.verbose;
    yvo::vk::VulkanContext ctx(co);
    log(fmt("[vk] device=%s  timestamps=%d period=%.3f ns valid_bits=%u  validation=%d", ctx.device_name(),
            ctx.timestamps_supported() ? 1 : 0, static_cast<double>(ctx.timestamp_period()),
            ctx.timestamp_valid_bits(), ctx.validation_layer_active() ? 1 : 0));
    if (!ctx.timestamps_supported()) throw std::runtime_error("--probe stages needs timestamp queries");

    YoloPostprocConfig pc;   // local_size_x/y stay at the production default on purpose
    pc.num_slots = std::max(1, std::min(o.slots, rig.fs.frames));
    pc.max_dets = o.max_dets;
    pc.score_thr = o.nms_score;
    pc.input_alloc = YoloInputAlloc::Plain;
    pc.input_sync = YoloInputSync::HostWrite;
    pc.verbose = o.verbose;
    log(fmt("[cfg] slots=%d (stream frames 0..%d) max_dets=%d local=%ux%u rounds=%d warmup=%d perf_frames=%d",
            pc.num_slots, pc.num_slots - 1, pc.max_dets, pc.local_size_x, pc.local_size_y, o.rounds, o.warmup,
            o.perf_frames));
    std::string vnames;
    for (const Variant& v : vars) vnames += (vnames.empty() ? "" : ",") + std::string(v.name);
    log(fmt("[cfg] variants=%s", vnames.c_str()));

    // Builds YoloPostproc with the variant's SPIR-V. The environment variable is read only in the constructor,
    // so it is cleared right after construction (Full never sees a leftover path).
    auto make_pp = [&](const Variant& v) {
        if (v.spv.empty())
            unsetenv("YVO_YOLO_DECODE_SPV");
        else
            setenv("YVO_YOLO_DECODE_SPV", v.spv.c_str(), 1);
        std::unique_ptr<YoloPostproc> pp;
        try {
            pp = std::make_unique<YoloPostproc>(ctx, rig.layout, pc);
        } catch (...) {
            unsetenv("YVO_YOLO_DECODE_SPV");
            throw;
        }
        unsetenv("YVO_YOLO_DECODE_SPV");
        return pp;
    };

    // ── gate: Full vs cpu_yolov8_candidates() on every fixture frame (slot 0 refilled per frame) ──
    log("");
    log("== gate: Full (embedded SPIR-V) candidate count vs cpu_yolov8_candidates(), every fixture frame ==");
    log("frame  cpu  gpu  overflow  g5_detail  count");
    bool gate_ok = true;
    std::size_t total_cand = 0, stream_cand = 0;
    {
        auto pp = make_pp(vars.back());
        for (int f = 0; f < rig.fs.frames; ++f) {
            fill_slot(*pp, 0, rig.frames[static_cast<std::size_t>(f)]);
            pp->submit(0, rig.fs.lb, rig.fs.src_w, rig.fs.src_h);
            pp->wait(0);
            const auto gpu = pp->read_candidates(0);
            const std::uint8_t* ptr[kYoloNumTensors];
            for (int t = 0; t < kYoloNumTensors; ++t) ptr[t] = pp->raw_host_ptr(0, t);
            std::vector<DetectionKeyed> cpu;
            {
                const RawRead rr(*pp, 0);   // R2 (no-op for Plain; kept for symmetry with run_fixture)
                cpu = yvo::cpu_yolov8_candidates(make_raw_set(rig.layout, ptr), o.nms_score, rig.fs.lb,
                                                 rig.fs.src_w, rig.fs.src_h);
            }
            const Compare c = compare_candidates(cpu, gpu);
            const bool eq = cpu.size() == gpu.size() && pp->overflow(0) == 0;
            gate_ok = gate_ok && eq;
            total_cand += gpu.size();
            if (f < pc.num_slots) stream_cand += gpu.size();
            log(fmt("%5d  %4zu %4zu  %8u  %9s  %s", f, cpu.size(), gpu.size(), pp->overflow(0),
                    c.ok() ? "ok" : "DIFF", eq ? "ok" : "MISMATCH"));
        }
    }
    log(fmt("[gate] %s  candidates/frame: all %d frames %.2f, stream frames %.2f", gate_ok ? "PASS" : "FAIL",
            rig.fs.frames, static_cast<double>(total_cand) / rig.fs.frames,
            static_cast<double>(stream_cand) / pc.num_slots));

    // ── multiset check: Full vs legacy on every fixture frame (only when legacy is selected) ───────
    bool multiset_ok = true;
    if (const auto it = std::find_if(vars.begin(), vars.end(),
                                     [](const Variant& v) { return std::strcmp(v.name, "legacy") == 0; });
        it != vars.end()) {
        using Rec = std::array<std::uint32_t, 6>;   // score, class_id, x1, y1, x2, y2 (float bits)
        auto decode_all = [&](const Variant& v) {
            auto pp = make_pp(v);
            std::vector<std::map<std::uint32_t, Rec>> out;
            std::size_t dup = 0;
            for (int f = 0; f < rig.fs.frames; ++f) {
                fill_slot(*pp, 0, rig.frames[static_cast<std::size_t>(f)]);
                pp->submit(0, rig.fs.lb, rig.fs.src_w, rig.fs.src_h);
                pp->wait(0);
                if (pp->overflow(0) != 0) multiset_ok = false;
                std::map<std::uint32_t, Rec> m;
                for (const DetectionKeyed& d : pp->read_candidates(0)) {
                    const Rec r = {bits_of(d.score), static_cast<std::uint32_t>(d.class_id), bits_of(d.x1),
                                   bits_of(d.y1), bits_of(d.x2), bits_of(d.y2)};
                    if (!m.emplace(d.anchor_key, r).second) ++dup;
                }
                out.push_back(std::move(m));
            }
            if (dup != 0) multiset_ok = false;   // anchor_key is unique per (anchor, class) by construction
            return out;
        };
        const auto full = decode_all(vars.back());
        const auto legacy = decode_all(*it);
        log("");
        log("== multiset: Full vs legacy, anchor_key -> (score bits, class_id, coordinate bits), every fixture frame ==");
        log("frame  full  legacy  only_full  only_legacy  value_diff");
        std::size_t tot = 0, bad = 0;
        for (int f = 0; f < rig.fs.frames; ++f) {
            const auto& a = full[static_cast<std::size_t>(f)];
            const auto& b = legacy[static_cast<std::size_t>(f)];
            std::size_t only_a = 0, only_b = 0, diff = 0;
            for (const auto& kv : a) {
                const auto jt = b.find(kv.first);
                if (jt == b.end())
                    ++only_a;
                else if (jt->second != kv.second)
                    ++diff;
            }
            for (const auto& kv : b) only_b += a.count(kv.first) == 0 ? 1u : 0u;
            tot += a.size();
            bad += only_a + only_b + diff;
            log(fmt("%5d  %4zu  %6zu  %9zu  %11zu  %10zu", f, a.size(), b.size(), only_a, only_b, diff));
        }
        multiset_ok = multiset_ok && bad == 0 && tot > 0;
        log(fmt("[multiset] %s  %zu candidates, %zu mismatches", multiset_ok ? "PASS" : "FAIL", tot, bad));
    }

    // ── continuous stream per variant, interleaved over rounds ──────────────────────────────────
    for (int round = 0; round < o.rounds; ++round) {
        for (Variant& v : vars) {
            auto pp = make_pp(v);
            for (int s = 0; s < pp->num_slots(); ++s) fill_slot(*pp, s, rig.frames[static_cast<std::size_t>(s)]);
            for (int i = 0; i < o.warmup + o.perf_frames; ++i) {
                const int s = i % pp->num_slots();
                pp->submit(s, rig.fs.lb, rig.fs.src_w, rig.fs.src_h);
                pp->wait(s);
                if (i >= o.warmup) v.ms.push_back(pp->last_pass1_ms());
            }
            v.emitted = pp->read_candidates(0).size();
        }
    }

    log("");
    log(fmt("== pass 1 GPU time per variant (%d rounds x %d frames, ms) ==", o.rounds, o.perf_frames));
    log("variant   median     p90      min    n     emitted(slot0)");
    bool cut_ok = true;
    for (const Variant& v : vars) {
        log(fmt("%-7s  %7.4f  %7.4f  %7.4f  %4zu  %zu", v.name, percentile(v.ms, 0.5), percentile(v.ms, 0.9),
                *std::min_element(v.ms.begin(), v.ms.end()), v.ms.size(), v.emitted));
        // a stage cut must never reach ⑤; an emit-split variant must emit, or it was dead-code eliminated
        if (v.expect_emit != (v.emitted != 0)) cut_ok = false;
    }
    auto med = [&](const char* name) -> double {
        for (const Variant& v : vars)
            if (std::strcmp(v.name, name) == 0) return percentile(v.ms, 0.5);
        return std::numeric_limits<double>::quiet_NaN();
    };
    auto emitted_of = [&](const char* name) -> std::size_t {
        for (const Variant& v : vars)
            if (std::strcmp(v.name, name) == 0) return v.emitted;
        return 0;
    };
    auto has = [&](const char* name) { return !std::isnan(med(name)); };
    const double mf = med("Full");
    log("");
    log("== stage differences (medians, ms) ==");
    if (has("S1") && has("S2") && has("S3")) {
        const double m1 = med("S1"), m2 = med("S2"), m3 = med("S3");
        log(fmt("①② class scan + rejection = S1        %8.4f", m1));
        log(fmt("③  DFL                    = S2 - S1   %8.4f", m2 - m1));
        log(fmt("④  letterbox inverse      = S3 - S2   %8.4f", m3 - m2));
        log(fmt("⑤  emit                   = Full - S3 %8.4f", mf - m3));
    }
    log(fmt("   Full                                %8.4f", mf));
    // ⑤ split. The emitted counts differ from Full's, so these are upper bounds until corrected by the
    // per-candidate marginal cost (07_results/postproc_stage_cost).
    if (has("noatomic") || has("norescan")) {
        log("");
        log("== ⑤ emit split (medians, ms; emitted = slot 0 candidates after the last round) ==");
        if (has("noatomic"))
            log(fmt("atomic share = Full - noatomic   %8.4f   emitted Full %zu / noatomic %zu", mf - med("noatomic"),
                    emitted_of("Full"), emitted_of("noatomic")));
        if (has("norescan"))
            log(fmt("rescan share = Full - norescan   %8.4f   emitted Full %zu / norescan %zu", mf - med("norescan"),
                    emitted_of("Full"), emitted_of("norescan")));
    }
    if (has("legacy") || has("stop1_legacy")) {
        log("");
        log("== argmax shortcut (medians, ms; legacy = pre-shortcut build) ==");
        if (has("legacy"))
            log(fmt("shortcut net = Full - legacy      %8.4f   emitted Full %zu / legacy %zu", mf - med("legacy"),
                    emitted_of("Full"), emitted_of("legacy")));
        if (has("S1") && has("stop1_legacy"))
            log(fmt("① cost       = S1 - stop1_legacy  %8.4f", med("S1") - med("stop1_legacy")));
    }
    if (!cut_ok)
        log("[cut] FAIL: a stage cut emitted candidates or an emit-split variant emitted none "
            "(the injected SPIR-V was not used, or the emit was dead-code eliminated?)");

    log("");
    log(fmt("[validation] errors=%llu warnings=%llu", static_cast<unsigned long long>(ctx.validation_errors()),
            static_cast<unsigned long long>(ctx.validation_warnings())));
    const bool pass = gate_ok && cut_ok && multiset_ok && ctx.validation_errors() == 0;
    log(fmt("[RESULT] stages %s", pass ? "PASS" : "FAIL"));
    return pass ? 0 : 1;
}

// Same formula as YoloPostproc::q_thr_low() (01_src/vulkan/yolo_postproc.cpp) -- the groups probe builds no
// YoloPostproc, so it is mirrored here. floor(thr / scale + zp), clamped to 0..255.
std::uint32_t mirror_q_thr_low(float thr, const yvo::vk::YoloScaleSpec& s) {
    const double q = static_cast<double>(thr) / static_cast<double>(s.cls_scale) + static_cast<double>(s.cls_zp);
    if (q <= 0.0) return 0;
    if (q >= 255.0) return 255;
    return static_cast<std::uint32_t>(std::floor(q));
}

// ⑤ argmax-shortcut hit rate (yolo_decode.comp). The shortcut gate is the **prefilter** count
// #{c : q_c >= t_sw} == 1 (t_sw = min(q_thr_low, 128)), one notch looser than the definitive judgement.
// Every anchor that reaches ⑤ (prefilter count >= 1) is classified, straight from the raw class tensors.
// It also replays two SWAR per-byte counters bit for bit against the scalar count #{c : q_c >= t_sw}:
//   borrow : lanes below t_sw, ((w - t4) & ~w & 0x80808080) -- the rejected design. Borrows cross byte lanes, so a
//            lane equal to t_sw above a lane below t_sw is counted as "below" (passes are UNDER-counted)
//   exact  : lanes at or above t_sw, (((w | 0x80808080) - t4) | w) & 0x80808080 -- the shader's ge_cnt, literally.
//            The high bit stops every borrow (t_sw <= 128)
struct PrefilterHist {
    std::array<std::size_t, 5> n{};   // [k] = anchors with k prefilter-passing classes (k = 1..3, 4 = 4 or more)
    std::size_t hit_emit = 0;          // k == 1 and the class also passes the definitive test (shortcut emits)
    std::size_t hit_none = 0;          // k == 1 but the definitive test fails (shortcut emits nothing)
    std::size_t swar_bad[2] = {0, 0};  // anchors whose SWAR count differs from the scalar count (borrow, exact)
    std::size_t swar_unsafe[2] = {0, 0};   // ... where the SWAR gate says 1 but the true prefilter count is not 1
    std::size_t swar_drop[2] = {0, 0};     // ... and >= 2 classes pass the definitive test (the shortcut loses emits)
};

void accumulate_prefilter(const FixtureRig& rig, int f, float thr, PrefilterHist& ph) {
    const int C = rig.layout.num_classes;
    for (int sc = 0; sc < kYoloNumScales; ++sc) {
        const yvo::vk::YoloScaleSpec& s = rig.layout.s[sc];
        const std::uint8_t* cls =
            rig.frames[static_cast<std::size_t>(f)][static_cast<std::size_t>(yolo_cls_tensor(sc))].data();
        const std::uint32_t qlow = mirror_q_thr_low(thr, s);
        const std::uint32_t t_sw = std::min(qlow, 128u);
        const std::uint32_t t4 = t_sw * 0x01010101u;
        for (int h = 0; h < s.grid_h; ++h)
            for (int w0 = 0; w0 < s.grid_w; w0 += 4) {
                std::array<int, 4> k{}, kdef{};
                std::uint32_t lt = 0u, ge = 0u;   // borrow form counts lanes below, exact form lanes at/above
                for (int c = 0; c < C; ++c) {
                    const std::size_t base =
                        (static_cast<std::size_t>(h) * static_cast<std::size_t>(C) + static_cast<std::size_t>(c)) *
                            static_cast<std::size_t>(s.w_padded) +
                        static_cast<std::size_t>(w0);
                    std::uint32_t word = 0;
                    for (int a = 0; a < 4; ++a) {
                        const std::uint32_t q = cls[base + static_cast<std::size_t>(a)];
                        word |= q << (8 * a);
                        if (q >= qlow) {
                            ++k[static_cast<std::size_t>(a)];
                            const float score = (static_cast<float>(q) - s.cls_zp) * s.cls_scale;   // shader's float form
                            if (score >= thr) ++kdef[static_cast<std::size_t>(a)];
                        }
                    }
                    lt += ((word - t4) & ~word & 0x80808080u) >> 7;
                    ge += ((((word | 0x80808080u) - t4) | word) & 0x80808080u) >> 7;
                }
                for (int a = 0; a < 4; ++a) {
                    const auto ua = static_cast<std::size_t>(a);
                    // scalar prefilter count against t_sw (what the SWAR counter must reproduce)
                    int k_sw = 0;
                    for (int c = 0; c < C; ++c)
                        k_sw += cls[(static_cast<std::size_t>(h) * static_cast<std::size_t>(C) +
                                     static_cast<std::size_t>(c)) *
                                        static_cast<std::size_t>(s.w_padded) +
                                    static_cast<std::size_t>(w0 + a)] >= t_sw;
                    for (int v = 0; v < 2; ++v) {
                        const int pass = v == 0 ? C - static_cast<int>((lt >> (8 * a)) & 0xffu)
                                                : static_cast<int>((ge >> (8 * a)) & 0xffu);
                        if (pass != k_sw) ++ph.swar_bad[v];
                        if (pass == 1 && k[ua] != 1 && k[ua] > 0) ++ph.swar_unsafe[v];
                        if (pass == 1 && kdef[ua] >= 2) ++ph.swar_drop[v];
                    }
                    if (k[ua] == 0) continue;   // never reaches ⑤
                    ++ph.n[static_cast<std::size_t>(std::min(k[ua], 4))];
                    if (k[ua] == 1) ++(kdef[ua] == 1 ? ph.hit_emit : ph.hit_none);
                }
            }
    }
}

// ───────────────────────────────────── groups (passing anchors per invocation group, **no NPU, no GPU**)
//  --probe groups
//      Histogram of **passing anchors per invocation group**, which bounds the gain of exchanging the ⑤ loops in
//      `yolo_decode.comp`. Today ⑤ rescans all `num_classes` class words once **per passing anchor** of a group,
//      so a group with n passing anchors loads n x num_classes words where a class-outer loop would load
//      num_classes. The mean of n is therefore the load-reduction factor, and it is 1.00 (no gain at all) when
//      every group has exactly one passing anchor.
//      One group = one shader invocation = (scale_id, h, w >> 2), because 1 class word covers 4 anchors in the w
//      direction. Anchor id = anchor_key >> 7 = (scale << 14) | (h << 7) | w (ref/cpu_yolov8_decode.hpp
//      make_anchor_key), so the group key is (scale << 12) | (h << 5) | (w >> 2).
//      Everything is derived from `cpu_yolov8_candidates()`, the same reference the ⑤ emit rule follows, and no
//      Vulkan context is created -- it runs on any fixture directory (the 5-frame default or a crowd fixture).
int run_groups(const Options& o, Tee& log) {
    FixtureRig rig = load_rig(o);
    const int C = rig.layout.num_classes;
    std::size_t groups_total = 0;   // invocation groups of one frame over all 3 scales (① reads C words in each)
    for (int sc = 0; sc < kYoloNumScales; ++sc)
        groups_total += static_cast<std::size_t>(rig.layout.s[sc].grid_w / 4) *
                        static_cast<std::size_t>(rig.layout.s[sc].grid_h);
    const std::size_t stage1_words = groups_total * static_cast<std::size_t>(C);
    log(fmt("[groups] dir=%s frames=%d  score_thr=%.4f (HEF %.2f)  num_classes=%d", o.fixture_dir.c_str(),
            rig.fs.frames, static_cast<double>(o.nms_score), static_cast<double>(kHefScoreThreshold), C));
    log(fmt("[groups] invocation groups per frame = %zu, so ① loads %zu class words;"
            " ⑤ rescan loads (passing anchors) x %d",
            groups_total, stage1_words, C));
    log("frame  cand  anchors  groups   a/grp   n=1   n=2   n=3   n=4   cls/anchor 1/2/3/4+   rescan_words  %of①");
    std::array<std::size_t, 5> ahist{};   // [n] = groups with n passing anchors (n = 1..4)
    std::array<std::size_t, 5> chist{};   // [k] = anchors with k candidates (k = 1..3, 4 = 4 or more)
    std::size_t tot_cand = 0, tot_anchor = 0, tot_group = 0;
    bool ok = true;
    PrefilterHist ph;
    for (int f = 0; f < rig.fs.frames; ++f) {
        accumulate_prefilter(rig, f, o.nms_score, ph);
        const std::uint8_t* ptr[kYoloNumTensors];
        for (int t = 0; t < kYoloNumTensors; ++t)
            ptr[t] = rig.frames[static_cast<std::size_t>(f)][static_cast<std::size_t>(t)].data();
        const auto cand = yvo::cpu_yolov8_candidates(make_raw_set(rig.layout, ptr), o.nms_score, rig.fs.lb,
                                                     rig.fs.src_w, rig.fs.src_h);
        std::map<std::uint32_t, int> per_anchor;   // anchor id -> candidates emitted for that anchor
        for (const auto& d : cand) ++per_anchor[d.anchor_key >> 7];
        std::map<std::uint32_t, int> per_group;    // group key -> passing anchors of that group
        std::array<std::size_t, 5> fc{};
        for (const auto& kv : per_anchor) {
            const std::uint32_t a = kv.first;
            ++per_group[((a >> 14) << 12) | (((a >> 7) & 0x7fu) << 5) | ((a & 0x7fu) >> 2)];
            if (kv.second < 1) ok = false;
            const std::size_t k = static_cast<std::size_t>(std::min(kv.second, 4));
            ++fc[k];
            ++chist[k];
        }
        std::array<std::size_t, 5> fa{};
        for (const auto& kv : per_group) {
            if (kv.second < 1 || kv.second > 4) {   // impossible: 1 class word covers exactly 4 anchors
                ok = false;
                continue;
            }
            const std::size_t n = static_cast<std::size_t>(kv.second);
            ++fa[n];
            ++ahist[n];
        }
        const double mean =
            per_group.empty() ? 0.0
                              : static_cast<double>(per_anchor.size()) / static_cast<double>(per_group.size());
        const std::size_t rescan = per_anchor.size() * static_cast<std::size_t>(C);
        tot_cand += cand.size();
        tot_anchor += per_anchor.size();
        tot_group += per_group.size();
        log(fmt("%5d  %4zu  %7zu  %6zu  %6.3f  %4zu  %4zu  %4zu  %4zu   %5zu %4zu %4zu %4zu       %8zu  %5.2f%%", f,
                cand.size(), per_anchor.size(), per_group.size(), mean, fa[1], fa[2], fa[3], fa[4], fc[1], fc[2],
                fc[3], fc[4], rescan, 100.0 * static_cast<double>(rescan) / static_cast<double>(stage1_words)));
    }
    const std::size_t gsum = ahist[1] + ahist[2] + ahist[3] + ahist[4];
    const double mean_all = gsum == 0 ? 0.0 : static_cast<double>(tot_anchor) / static_cast<double>(gsum);
    log("");
    log(fmt("[groups] totals over %d frames: candidates %zu, passing anchors %zu, passing groups %zu",
            rig.fs.frames, tot_cand, tot_anchor, tot_group));
    log("passing anchors per group    count     share");
    for (std::size_t n = 1; n <= 4; ++n)
        log(fmt("  n = %zu                    %7zu  %7.2f%%", n, ahist[n],
                gsum == 0 ? 0.0 : 100.0 * static_cast<double>(ahist[n]) / static_cast<double>(gsum)));
    log(fmt("  mean n                   %7.3f           <- ⑤ rescan load-reduction factor of a class-outer loop",
            mean_all));
    const std::size_t csum = chist[1] + chist[2] + chist[3] + chist[4];
    log("candidates per passing anchor  count   share");
    for (std::size_t k = 1; k <= 4; ++k)
        log(fmt("  %s %zu                      %7zu  %7.2f%%", k == 4 ? ">=" : " =", k, chist[k],
                csum == 0 ? 0.0 : 100.0 * static_cast<double>(chist[k]) / static_cast<double>(csum)));
    const std::size_t psum = ph.n[1] + ph.n[2] + ph.n[3] + ph.n[4];
    log(fmt("prefilter-passing classes per anchor reaching ⑤ (q >= q_thr_low = P3 %u / P4 %u / P5 %u)",
            mirror_q_thr_low(o.nms_score, rig.layout.s[0]), mirror_q_thr_low(o.nms_score, rig.layout.s[1]),
            mirror_q_thr_low(o.nms_score, rig.layout.s[2])));
    for (std::size_t k = 1; k <= 4; ++k)
        log(fmt("  %s %zu                      %7zu  %7.2f%%", k == 4 ? ">=" : " =", k, ph.n[k],
                psum == 0 ? 0.0 : 100.0 * static_cast<double>(ph.n[k]) / static_cast<double>(psum)));
    log(fmt("[shortcut] hit rate (prefilter count == 1) %7.2f%%  of %zu anchors  (emit %zu / no emit %zu)",
            psum == 0 ? 0.0 : 100.0 * static_cast<double>(ph.n[1]) / static_cast<double>(psum), psum, ph.hit_emit,
            ph.hit_none));
    log(fmt("[shortcut] SWAR counter vs scalar: borrow form %zu mismatched anchors (%zu unsafe gate=1, %zu lose emits), "
            "exact form %zu (%zu unsafe, %zu lose emits)",
            ph.swar_bad[0], ph.swar_unsafe[0], ph.swar_drop[0], ph.swar_bad[1], ph.swar_unsafe[1], ph.swar_drop[1]));
    if (ph.swar_bad[1] != 0 || ph.swar_drop[1] != 0) ok = false;
    log("");
    log(fmt("[RESULT] groups %s", ok && tot_cand > 0 ? "PASS" : "FAIL"));
    return ok && tot_cand > 0 ? 0 : 1;
}

// ─────────────────────────────────────────────────────────── topk (CPU only, offline)
//
//  --probe topk (**no NPU, no GPU** — no Vulkan context is created)
//      Compares candidate-reduction rules against the reference rule (every (anchor, class) with score >= T →
//      per-class greedy NMS IoU 0.7, 100 per class) on **every** fixture frame. All rules are built from
//      cpu_yolov8_candidates() output, so decode and coordinates are byte-identical to the reference; only the
//      candidate selection differs. Scores are compared as decoded floats (each scale has its own quantization step).
//        REF(T)        every (anchor, class) with score >= T                    (= cpu_yolov8_candidates(raw, T))
//        MAX(T)        per anchor only the best class (ties → lowest class id), kept if its score >= T
//        SNIP(K)       per anchor the best class, top K anchors by (score desc, anchor id asc), no threshold
//        SNIP_T(T,K)   SNIP(K), then score >= T (filtering before NMS = after NMS for scores >= T: greedy NMS
//                      never lets a lower-scored box affect a higher-scored one, and the class quota counts in order)
//        MC_HIST(T,K)  s_K = K-th largest per-anchor best score (0 if fewer anchors), cut = max(T, s_K);
//                      anchors whose best score >= cut, and in them every class with score >= T
//                      (ties at s_K are all kept, so the anchor count can exceed K)
//      Anchor id = anchor_key >> 7 (make_anchor_key: scale<<21 | h<<14 | w<<7 | class).

struct TopkOptions {
    std::string scores = "0.2,0.05,0.02,0.003";
    std::string ks = "100,300,1024";
};

void usage_topk() {
    std::puts(
        "topk mode (--probe topk: CPU only, no NPU/GPU, reads every frame in --fixture-dir):\n"
        "  --scores LIST          score thresholds T, comma-separated (default 0.2,0.05,0.02,0.003)\n"
        "  --ks LIST              top-K anchor counts, comma-separated (default 100,300,1024)");
}

std::vector<float> parse_float_list(const std::string& s) {
    std::vector<float> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ','))
        if (!tok.empty()) out.push_back(std::stof(tok));
    return out;
}

constexpr float kTopkIou = 0.7f;           // HEF setting
constexpr int kTopkMaxPerClass = 100;      // HEF setting
constexpr int kTopkReps = 5;               // timing repeats per frame (median)
constexpr std::size_t kTopkGpuNmsCap = 1024;   // GPU NMS capacity (kMaxSelected) → above it the CPU fallback runs
constexpr std::size_t kTopkCandCap = 4096;     // pass 1 candidate buffer (max_dets)

std::uint32_t topk_anchor(const DetectionKeyed& d) { return d.anchor_key >> 7; }

bool topk_same_bits(const DetectionKeyed& a, const DetectionKeyed& b) {
    return a.anchor_key == b.anchor_key && a.class_id == b.class_id && bits_of(a.score) == bits_of(b.score) &&
           bits_of(a.x1) == bits_of(b.x1) && bits_of(a.y1) == bits_of(b.y1) && bits_of(a.x2) == bits_of(b.x2) &&
           bits_of(a.y2) == bits_of(b.y2);
}

// Runs f() kTopkReps times and returns the last result; us = median wall time (µs).
template <class F>
auto topk_timed(F&& f, double& us) -> decltype(f()) {
    std::vector<double> t;
    decltype(f()) out{};
    for (int r = 0; r < kTopkReps; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        out = f();
        const auto t1 = std::chrono::steady_clock::now();
        t.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    }
    us = percentile(t, 0.5);
    return out;
}

// cpu_nms_greedy_by_class() on a copy (the copy is outside the timed region). us = median of kTopkReps.
std::vector<DetectionKeyed> topk_nms(const std::vector<DetectionKeyed>& cand, double& us) {
    std::vector<double> t;
    std::vector<DetectionKeyed> work;
    for (int r = 0; r < kTopkReps; ++r) {
        work = cand;
        const auto t0 = std::chrono::steady_clock::now();
        yvo::cpu_nms_greedy_by_class(work, kTopkIou, kTopkMaxPerClass);
        const auto t1 = std::chrono::steady_clock::now();
        t.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    }
    us = percentile(t, 0.5);
    return work;
}

std::size_t topk_distinct_anchors(const std::vector<DetectionKeyed>& v) {
    std::vector<std::uint32_t> a;
    a.reserve(v.size());
    for (const auto& d : v) a.push_back(topk_anchor(d));
    std::sort(a.begin(), a.end());
    return static_cast<std::size_t>(std::unique(a.begin(), a.end()) - a.begin());
}

struct TopkRow {
    std::string name;
    std::vector<double> cand, anchors, nms_us, sel_us;   // per frame
    std::size_t over_gpu = 0, over_cap = 0;
    std::size_t dets = 0, matched = 0, coord_diff = 0, lost = 0, extra = 0, lost_nonmax = 0, iou_pairs = 0;
    float lost_smax = -1.0f, extra_smax = -1.0f;
    bool has_sel = false;

    void add_frame(const std::vector<DetectionKeyed>& c, double nms, double sel) {
        cand.push_back(static_cast<double>(c.size()));
        anchors.push_back(static_cast<double>(topk_distinct_anchors(c)));
        if (c.size() > kTopkGpuNmsCap) ++over_gpu;
        if (c.size() > kTopkCandCap) ++over_cap;
        nms_us.push_back(nms);
        if (sel >= 0.0) {
            sel_us.push_back(sel);
            has_sel = true;
        }
    }

    // Final detections of this case vs REF(T) final detections of the same frame.
    // best_class(anchor id) = the anchor's highest class (MAX-family loss cause).
    template <class BestClass>
    void compare(const std::vector<DetectionKeyed>& ref, const std::vector<DetectionKeyed>& got, BestClass best_class) {
        std::map<std::uint32_t, const DetectionKeyed*> refm;
        for (const auto& d : ref) refm[d.anchor_key] = &d;
        std::map<std::uint32_t, char> hit;
        std::vector<const DetectionKeyed*> lv, ev;
        dets += got.size();
        for (const auto& d : got) {
            const auto it = refm.find(d.anchor_key);
            if (it == refm.end()) {
                ev.push_back(&d);
                continue;
            }
            ++matched;
            hit[d.anchor_key] = 1;
            if (!topk_same_bits(d, *it->second)) ++coord_diff;
        }
        for (const auto& d : ref)
            if (!hit.count(d.anchor_key)) lv.push_back(&d);
        lost += lv.size();
        extra += ev.size();
        for (const auto* d : lv) {
            if (best_class(topk_anchor(*d)) != d->class_id) ++lost_nonmax;
            lost_smax = std::max(lost_smax, d->score);
        }
        for (const auto* d : ev) extra_smax = std::max(extra_smax, d->score);
        // Secondary pairing: lost ↔ extra of the same class with IoU >= 0.5 (greedy, lost in score order, best IoU).
        std::vector<char> used(ev.size(), 0);
        for (const auto* l : lv) {
            int best = -1;
            float best_iou = 0.5f;
            for (std::size_t j = 0; j < ev.size(); ++j) {
                if (used[j] || ev[j]->class_id != l->class_id) continue;
                const float iou = yvo::iou_xyxy(*l, *ev[j]);
                if (iou >= best_iou) {
                    best_iou = iou;
                    best = static_cast<int>(j);
                }
            }
            if (best >= 0) {
                used[static_cast<std::size_t>(best)] = 1;
                ++iou_pairs;
            }
        }
    }
};

std::string topk_smax(float v) { return v < 0.0f ? std::string("-") : fmt("%.4f", static_cast<double>(v)); }

int run_topk(const Options& o, const TopkOptions& to, Tee& log) {
    const std::vector<float> ts = parse_float_list(to.scores);
    const std::vector<int> ks = parse_int_list(to.ks);
    if (ts.empty() || ks.empty()) throw std::runtime_error("--scores and --ks must not be empty");
    for (const float t : ts)
        if (!(t > 0.0f)) throw std::runtime_error(fmt("--scores values must be > 0: %g", static_cast<double>(t)));
    for (const int k : ks)
        if (k < 1) throw std::runtime_error(fmt("--ks values must be >= 1: %d", k));

    FixtureRig rig = load_rig(o);
    const int nframes = rig.fs.frames;
    const auto& lb = rig.fs.lb;
    const int sw = rig.fs.src_w, sh = rig.fs.src_h;

    // Threshold just below one quantization step: q >= 1 of every class tensor (requires zero point 0).
    float min_step = std::numeric_limits<float>::infinity();
    for (int s = 0; s < kYoloNumScales; ++s) {
        if (rig.layout.s[s].cls_zp != 0.0f)
            throw std::runtime_error(fmt("scale %d class zero point is %g, topk assumes 0", s,
                                         static_cast<double>(rig.layout.s[s].cls_zp)));
        min_step = std::min(min_step, rig.layout.s[s].cls_scale);
    }
    const float thr_q1 = 0.5f * min_step;
    log(fmt("[topk] dir=%s frames=%d  IoU %.2f  %d per class  timing: median of %d repeats per frame", o.fixture_dir.c_str(),
            nframes, static_cast<double>(kTopkIou), kTopkMaxPerClass, kTopkReps));
    log(fmt("[topk] class quant step per scale: %.6g / %.6g / %.6g  → all-class decode threshold %.6g (q >= 1)",
            static_cast<double>(rig.layout.s[0].cls_scale), static_cast<double>(rig.layout.s[1].cls_scale),
            static_cast<double>(rig.layout.s[2].cls_scale), static_cast<double>(thr_q1)));
    log(fmt("[topk] T = %s   K = %s", to.scores.c_str(), to.ks.c_str()));

    const std::size_t nT = ts.size(), nK = ks.size();
    const std::size_t nrows = 2 + 3 * nK;
    auto i_snip = [&](std::size_t k) { return 2 + k; };
    auto i_snipt = [&](std::size_t k) { return 2 + nK + k; };
    auto i_mc = [&](std::size_t k) { return 2 + 2 * nK + k; };
    std::vector<std::vector<TopkRow>> rows(nT, std::vector<TopkRow>(nrows));
    for (std::size_t ti = 0; ti < nT; ++ti) {
        rows[ti][0].name = "REF";
        rows[ti][1].name = "MAX";
        for (std::size_t k = 0; k < nK; ++k) {
            rows[ti][i_snip(k)].name = fmt("SNIP(%d)", ks[k]);
            rows[ti][i_snipt(k)].name = fmt("SNIP_T(%d)", ks[k]);
            rows[ti][i_mc(k)].name = fmt("MC_HIST(%d)", ks[k]);
        }
    }
    // Multi-class statistics of REF(T)
    struct Multi {
        std::size_t ref_cand = 0, multi_anchors = 0, multi_cand = 0, extra_classes = 0, ref_dets = 0, ref_dets_nonmax = 0;
        std::size_t max_cand = 0;
    };
    std::vector<Multi> multi(nT);
    std::vector<std::size_t> mc_eq_frames(nT * nK, 0), mc_eq_fail(nT * nK, 0);

    bool ok_pp = true, ok_filter = true, ok_argmax = true, ok_maxdiff = true, ok_snip = true, ok_mc = true;

    yvo::DecodeOptions arg_opt;
    arg_opt.argmax_only = true;
    const float kNoThr = -std::numeric_limits<float>::infinity();
    std::vector<int> best_idx(1u << 16, -1);   // anchor id (< 3 << 14) → index into A
    std::vector<int> cls_count(1u << 16, 0);
    auto snip_cmp = [](const DetectionKeyed& a, const DetectionKeyed& b) {
        if (a.score != b.score) return a.score > b.score;
        return topk_anchor(a) < topk_anchor(b);
    };

    for (int f = 0; f < nframes; ++f) {
        const std::uint8_t* ptr[kYoloNumTensors];
        for (int t = 0; t < kYoloNumTensors; ++t)
            ptr[t] = rig.frames[static_cast<std::size_t>(f)][static_cast<std::size_t>(t)].data();
        const yvo::RawOutputSet raw = make_raw_set(rig.layout, ptr);

        // A: one entry per anchor (best class, ties → lowest class id). B: every (anchor, class) with q >= 1.
        const auto A = yvo::cpu_yolov8_candidates(raw, kNoThr, lb, sw, sh, arg_opt);
        const auto B = yvo::cpu_yolov8_candidates(raw, thr_q1, lb, sw, sh);
        std::fill(best_idx.begin(), best_idx.end(), -1);
        for (std::size_t i = 0; i < A.size(); ++i) best_idx[topk_anchor(A[i])] = static_cast<int>(i);
        auto best_of = [&](std::uint32_t anchor) -> const DetectionKeyed& {
            return A[static_cast<std::size_t>(best_idx[anchor])];
        };
        auto best_class = [&](std::uint32_t anchor) { return best_of(anchor).class_id; };
        // Decode consistency: the per-anchor best of A dominates every B entry of that anchor and equals it bit for bit.
        for (const auto& b : B) {
            if (best_idx[topk_anchor(b)] < 0) {
                ok_argmax = false;
                continue;
            }
            const auto& a = best_of(topk_anchor(b));
            if (b.score > a.score || (b.score == a.score && b.class_id < a.class_id)) ok_argmax = false;
            if (b.class_id == a.class_id && !topk_same_bits(a, b)) ok_argmax = false;
        }

        // SNIP(K): independent of T
        std::vector<std::vector<DetectionKeyed>> snip_cand(nK), snip_final(nK);
        std::vector<double> snip_sel(nK), snip_nms(nK), sk(nK), sk_us(nK);
        for (std::size_t k = 0; k < nK; ++k) {
            const std::size_t n = std::min(static_cast<std::size_t>(ks[k]), A.size());
            snip_cand[k] = topk_timed(
                [&] {
                    std::vector<DetectionKeyed> v = A;
                    std::partial_sort(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(n), v.end(), snip_cmp);
                    v.resize(n);
                    return v;
                },
                snip_sel[k]);
            if (snip_cand[k].size() != n) ok_snip = false;
            snip_final[k] = topk_nms(snip_cand[k], snip_nms[k]);
            // s_K = K-th largest per-anchor best score (0 if fewer anchors)
            sk[k] = topk_timed(
                [&] {
                    if (A.size() < static_cast<std::size_t>(ks[k])) return 0.0;
                    std::vector<float> sc;
                    sc.reserve(A.size());
                    for (const auto& a : A) sc.push_back(a.score);
                    std::nth_element(sc.begin(), sc.begin() + (ks[k] - 1), sc.end(), [](float x, float y) { return x > y; });
                    return static_cast<double>(sc[static_cast<std::size_t>(ks[k] - 1)]);
                },
                sk_us[k]);
        }

        for (std::size_t ti = 0; ti < nT; ++ti) {
            const float T = ts[ti];
            auto& R = rows[ti];

            // REF(T)
            const auto ref = yvo::cpu_yolov8_candidates(raw, T, lb, sw, sh);
            double us = 0.0;
            const auto ref_final = topk_nms(ref, us);
            R[0].add_frame(ref, us, -1.0);
            R[0].compare(ref_final, ref_final, best_class);
            {
                const auto pp = yvo::cpu_yolov8_postprocess(raw, T, kTopkIou, kTopkMaxPerClass, lb, sw, sh);
                bool same = pp.size() == ref_final.size();
                for (std::size_t i = 0; same && i < pp.size(); ++i) {
                    const auto& a = pp[i];
                    const auto& b = ref_final[i];
                    same = a.class_id == b.class_id && bits_of(a.score) == bits_of(b.score) &&
                           bits_of(a.x1) == bits_of(b.x1) && bits_of(a.y1) == bits_of(b.y1) &&
                           bits_of(a.x2) == bits_of(b.x2) && bits_of(a.y2) == bits_of(b.y2);
                }
                if (!same) {
                    ok_pp = false;
                    log(fmt("[check] frame %d T=%g: REF final != cpu_yolov8_postprocess (%zu vs %zu)", f,
                            static_cast<double>(T), ref_final.size(), pp.size()));
                }
            }
            {   // REF(T) == B filtered by T (same order, same bits)
                std::size_t j = 0;
                bool same = true;
                for (const auto& b : B) {
                    if (!(b.score >= T)) continue;
                    if (j >= ref.size() || !topk_same_bits(b, ref[j])) same = false;
                    ++j;
                }
                if (!same || j != ref.size()) ok_filter = false;
            }
            // multi-class statistics
            Multi& M = multi[ti];
            M.ref_cand += ref.size();
            for (const auto& d : ref) ++cls_count[topk_anchor(d)];
            for (const auto& d : ref) {
                const int c = cls_count[topk_anchor(d)];
                if (c >= 2) ++M.multi_cand;
            }
            std::size_t extra_classes = 0;
            for (const auto& d : ref) {
                int& c = cls_count[topk_anchor(d)];
                if (c >= 2) {
                    ++M.multi_anchors;
                    extra_classes += static_cast<std::size_t>(c - 1);
                }
                c = 0;   // count each anchor once, and leave the table zeroed
            }
            M.extra_classes += extra_classes;
            M.ref_dets += ref_final.size();
            for (const auto& d : ref_final)
                if (best_class(topk_anchor(d)) != d.class_id) ++M.ref_dets_nonmax;

            // MAX(T)
            double sel = 0.0;
            const auto mx = topk_timed(
                [&] {
                    std::vector<DetectionKeyed> v;
                    for (const auto& a : A)
                        if (a.score >= T) v.push_back(a);
                    return v;
                },
                sel);
            const auto mx_final = topk_nms(mx, us);
            R[1].add_frame(mx, us, sel);
            R[1].compare(ref_final, mx_final, best_class);
            M.max_cand += mx.size();
            if (ref.size() - mx.size() != extra_classes || mx.size() != topk_distinct_anchors(ref)) ok_maxdiff = false;

            for (std::size_t k = 0; k < nK; ++k) {
                // SNIP(K)
                R[i_snip(k)].add_frame(snip_cand[k], snip_nms[k], snip_sel[k]);
                R[i_snip(k)].compare(ref_final, snip_final[k], best_class);
                // SNIP_T(T,K)
                const auto st = topk_timed(
                    [&] {
                        std::vector<DetectionKeyed> v;
                        for (const auto& a : snip_cand[k])
                            if (a.score >= T) v.push_back(a);
                        return v;
                    },
                    sel);
                const auto st_final = topk_nms(st, us);
                R[i_snipt(k)].add_frame(st, us, snip_sel[k] + sel);
                R[i_snipt(k)].compare(ref_final, st_final, best_class);
                // MC_HIST(T,K)
                const float cut = std::max(T, static_cast<float>(sk[k]));
                const auto mc = topk_timed(
                    [&] {
                        std::vector<DetectionKeyed> v;
                        for (const auto& b : B)
                            if (b.score >= T && best_of(topk_anchor(b)).score >= cut) v.push_back(b);
                        return v;
                    },
                    sel);
                const auto mc_final = topk_nms(mc, us);
                R[i_mc(k)].add_frame(mc, us, sk_us[k] + sel);
                R[i_mc(k)].compare(ref_final, mc_final, best_class);
                if (cut == T) {   // s_K <= T: the anchor cut is T itself → must be exactly REF(T)
                    ++mc_eq_frames[ti * nK + k];
                    bool same = mc.size() == ref.size();
                    for (std::size_t i = 0; same && i < mc.size(); ++i) same = topk_same_bits(mc[i], ref[i]);
                    if (!same) {
                        ++mc_eq_fail[ti * nK + k];
                        ok_mc = false;
                    }
                }
            }
        }
    }

    // ── tables ──
    log("");
    log("columns: cand/f = candidates per frame (mean), cmax = max, >1024 = frames over GPU NMS capacity (CPU fallback),");
    log("  >4096 = frames over the pass 1 candidate buffer, anc/f = distinct anchors per frame (mean), dets = final detections (sum),");
    log("  match/lost/extra vs REF(T) by (class, anchor_key), lostNM = lost whose class is not the anchor's best class,");
    log("  iouP = lost↔extra re-paired by same class IoU>=0.5, Smax = max score, cdiff = matched with different bits,");
    log("  nms_* = cpu_nms_greedy_by_class us per frame (median/p90/max over frames), sel = selection us per frame (median):");
    log("  MAX = filter anchors, SNIP = partial_sort of anchors, SNIP_T = SNIP + filter, MC_HIST = s_K (nth_element) + filter of");
    log("  the q>=1 decode (CPU emulation; a GPU would apply the cut inside pass 1)");
    bool ok_coord = true;
    for (std::size_t ti = 0; ti < nT; ++ti) {
        log("");
        log(fmt("== T = %g  (%d frames) ==", static_cast<double>(ts[ti]), nframes));
        log("case            cand/f   cmax  >1024 >4096   anc/f    dets  match   lost  extra lostNM  iouP  lostSmax extraSmax cdiff  nms_med  nms_p90  nms_max  sel_med");
        for (const TopkRow& r : rows[ti]) {
            double sum = 0.0, cmax = 0.0, asum = 0.0;
            for (const double c : r.cand) {
                sum += c;
                cmax = std::max(cmax, c);
            }
            for (const double a : r.anchors) asum += a;
            const double n = static_cast<double>(std::max<std::size_t>(1, r.cand.size()));
            double nmax = 0.0;
            for (const double v : r.nms_us) nmax = std::max(nmax, v);
            if (r.coord_diff) ok_coord = false;
            log(fmt("%-14s %7.1f %6.0f %6zu %5zu %7.1f %7zu %6zu %6zu %6zu %6zu %5zu %9s %9s %5zu %8.1f %8.1f %8.1f %8s",
                    r.name.c_str(), sum / n, cmax, r.over_gpu, r.over_cap, asum / n, r.dets, r.matched, r.lost, r.extra,
                    r.lost_nonmax, r.iou_pairs, topk_smax(r.lost_smax).c_str(), topk_smax(r.extra_smax).c_str(),
                    r.coord_diff, percentile(r.nms_us, 0.5), percentile(r.nms_us, 0.9), nmax,
                    r.has_sel ? fmt("%.1f", percentile(r.sel_us, 0.5)).c_str() : "-"));
        }
        for (std::size_t k = 0; k < nK; ++k)
            log(fmt("  MC_HIST(%d): frames with s_K <= T (cut = T) %zu/%d, of which != REF %zu", ks[k],
                    mc_eq_frames[ti * nK + k], nframes, mc_eq_fail[ti * nK + k]));
    }

    log("");
    log("== multi-class statistics of REF(T) (sums over frames) ==");
    log("T          ref_cand  max_cand  multi_anchors  multi_cand  multi_cand%  extra_classes  ref_dets  dets_nonmax");
    for (std::size_t ti = 0; ti < nT; ++ti) {
        const Multi& M = multi[ti];
        log(fmt("%-9g %9zu %9zu %14zu %11zu %11.2f%% %14zu %9zu %12zu", static_cast<double>(ts[ti]), M.ref_cand, M.max_cand,
                M.multi_anchors, M.multi_cand,
                M.ref_cand ? 100.0 * static_cast<double>(M.multi_cand) / static_cast<double>(M.ref_cand) : 0.0,
                M.extra_classes, M.ref_dets, M.ref_dets_nonmax));
    }

    log("");
    auto pf = [](bool b) { return b ? "PASS" : "FAIL"; };
    log(fmt("[check] ref == cpu_yolov8_postprocess %s", pf(ok_pp)));
    log(fmt("[check] REF(T) == q>=1 decode filtered by T %s", pf(ok_filter)));
    log(fmt("[check] per-anchor best (argmax decode) consistent with all-class decode %s", pf(ok_argmax)));
    log(fmt("[check] REF - MAX candidates == extra classes of multi-class anchors %s", pf(ok_maxdiff)));
    log(fmt("[check] SNIP(K) candidates == min(K, anchors) %s", pf(ok_snip)));
    log(fmt("[check] MC_HIST == REF whenever s_K <= T %s", pf(ok_mc)));
    log(fmt("[check] matched detections bit-identical to REF %s", pf(ok_coord)));
    const bool pass = ok_pp && ok_filter && ok_argmax && ok_maxdiff && ok_snip && ok_mc && ok_coord;
    log("[RESULT] topk DONE");
    return pass ? 0 : 1;
}

} // namespace

int main(int argc, char** argv) {
    Options o;
    TopkOptions to;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* what) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(std::string("missing argument: ") + what);
            return argv[++i];
        };
        try {
            if (a == "--probe") o.probe = next("--probe");
            else if (a == "--fixture") o.probe = "fixture";
            else if (a == "--live") o.probe = "live";
            else if (a == "--stale") o.probe = "stale";
            else if (a == "--sweep") o.probe = "sweep";
            else if (a == "--fixture-dir") o.fixture_dir = next("--fixture-dir");
            else if (a == "--out-dir") { o.out_dir = next("--out-dir"); o.out_dir_set = true; }
            else if (a == "--out-file") o.out_file = next("--out-file");
            else if (a == "--hef") o.hef = next("--hef");
            else if (a == "--video") o.video = next("--video");
            else if (a == "--perf-frames") o.perf_frames = std::stoi(next("--perf-frames"));
            else if (a == "--warmup") o.warmup = std::stoi(next("--warmup"));
            else if (a == "--repeats") o.repeats = std::stoi(next("--repeats"));
            else if (a == "--live-frames") o.live_frames = std::stoi(next("--live-frames"));
            else if (a == "--slots") o.slots = std::stoi(next("--slots"));
            else if (a == "--max-dets") o.max_dets = std::stoi(next("--max-dets"));
            else if (a == "--api") o.api = next("--api");
            else if (a == "--batch") o.batch = static_cast<std::uint16_t>(std::stoi(next("--batch")));
            else if (a == "--timeout-ms") o.timeout_ms = std::stoi(next("--timeout-ms"));
            else if (a == "--stale-barrier") o.stale_barrier = next("--stale-barrier");
            else if (a == "--heap") o.heap = next("--heap");
            else if (a == "--alloc") o.alloc = next("--alloc");
            else if (a == "--heap-sync") o.heap_sync = next("--heap-sync");
            else if (a == "--rounds") o.rounds = std::stoi(next("--rounds"));
            else if (a == "--variants") o.variants = next("--variants");
            else if (a == "--v3-window") o.v3_window = std::stoi(next("--v3-window"));
            else if (a == "--nms-cand") o.nms_cands = next("--nms-cand");
            else if (a == "--nms-score" || a == "--score") o.nms_score = std::stof(next(a.c_str()));
            else if (a == "--scores") to.scores = next("--scores");
            else if (a == "--ks") to.ks = next("--ks");
            else if (a == "--no-headroom") o.no_headroom = true;
            else if (a == "--heapimport") o.probe = "heapimport";
            else if (a == "--validation") o.validation = true;
            else if (a == "--sync-validation") { o.validation = true; o.sync_validation = true; }
            else if (a == "--verbose") o.verbose = true;
            else if (a == "--local") {
                const std::string v = next("--local");
                const auto comma = v.find(',');
                if (comma == std::string::npos) throw std::runtime_error("--local must be in X,Y form");
                o.lx = static_cast<std::uint32_t>(std::stoul(v.substr(0, comma)));
                o.ly = static_cast<std::uint32_t>(std::stoul(v.substr(comma + 1)));
            } else if (a == "-h" || a == "--help") {
                usage();
                usage_topk();
                return 0;
            } else {
                std::fprintf(stderr, "unknown argument: %s\n", a.c_str());
                usage();
                return 2;
            }
        } catch (const std::exception& e) {
            std::fprintf(stderr, "argument error: %s\n", e.what());
            return 2;
        }
    }
    // heapimport is an Appendix B (dmaheap_unify) deliverable, so its default output location differs.
    // Appendix B (dmaheap_unify) deliverables have a different default output location.
    // The default output is build/out/ so it does not overwrite archived evidence. When writing to 07_results, pass --out-dir/--out-file explicitly.
    if (o.probe == "heapimport" || o.probe == "gpuab" || o.probe == "v3") {
        if (!o.out_dir_set) o.out_dir = YVO_REPO_ROOT "/build/out/yvo_postproc_probe/dmaheap";
        if (o.out_file.empty())
            o.out_file = o.probe == "heapimport" ? "p0_heapimport.txt" : ("stage_" + o.probe + ".txt");
    }
    if (o.out_file.empty()) o.out_file = "p3_" + o.probe + ".txt";
    const std::string out_path = o.out_dir.empty() ? std::string() : o.out_dir + "/" + o.out_file;

    try {
        Tee log(out_path);
        log(fmt("yvo_postproc_probe --probe %s   (%s)", o.probe.c_str(),
                o.probe == "heapimport" ? "plan Appendix B step 1 = P0 gate" : "plan §6.2 P3"));
        if (o.probe == "heapimport") return run_heapimport(o, log);
        if (o.probe == "gpuab") return run_gpuab(o, log);
        if (o.probe == "v3") return run_v3(o, log);
        if (o.probe == "fixture") return run_fixture(o, log);
        if (o.probe == "sweep") return run_sweep(o, log);
        if (o.probe == "nms") return run_nms(o, log);
        if (o.probe == "stages") return run_stages(o, log);
        if (o.probe == "groups") return run_groups(o, log);
        if (o.probe == "topk") return run_topk(o, to, log);
        if (o.probe == "live") return run_live(o, log);
        if (o.probe == "stale") return run_stale(o, log);
        log(fmt("unknown probe: %s", o.probe.c_str()));
        usage();
        return 2;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[fatal] %s\n", e.what());
        return 1;
    }
}
