// yvo_postproc_probe: GPU pass 1(`02_shaders/yolo_decode.comp`) 검증 프로브. 계획서 §6.2 P3.
//
// ── 모드 ──────────────────────────────────────────────────────────────────────────────────────
//  --probe fixture (기본, **NPU 불필요**)
//      P1 이 덤프한 `04_data/raw_tensors/*.bin` 을 GPU 버퍼에 memcpy 하고 pass 1 을 돌려
//      `cpu_yolov8_candidates()` 와 비교한다. **셰이더 버그를 여기서 다 잡는다(빠른 루프).**
//      P3-① (G5: 후보 개수 동일 / 점수 비트 일치 / 좌표 < 0.25 px)
//      P3-⑥ (pass 1 GPU timestamp 중앙값, **연속 스트림** N 프레임 — 기본 300)
//      G6  (같은 픽스처 10회 반복에서 후보 집합이 `(score, anchor_key)` 전순서로 완전 동일)
//      바이트 합(kChecksum 파이프라인) 대조도 여기서 함께 한다.
//  --probe sweep (NPU 불필요)
//      local_size_x × local_size_y 스윕. 계획서의 "측정해서 고른다" 규율.
//  --probe live (**NPU 독점**)
//      `DmaBufExporter` 로 6 텐서 × 슬롯을 할당·export → HailoRT 가 D2H 로 매핑 → 원시 출력을
//      직접 받아 pass 1 에 먹인다. P3-② (셰이더 계산 바이트 합 == CPU 계산 합).
//  --probe heapimport (**NPU 불필요**, 부록 B 1단계 = P0 관문)
//      /dev/dma_heap 에서 6텐서를 잡아 `DmaBufImportCache::get()` 으로 Vulkan 에 import 하고,
//      memoryTypeBits / req.size / dedicated 플래그 / range 를 기록한 뒤 CPU↔GPU 바이트 합 왕복을 본다.
//      **V3DV 가 dma-heap fd 를 import 하는가**가 부록 B 전체의 관문이다(B.14 S1).
//  --probe stale (**NPU 독점**)
//      **같은 슬롯 버퍼**(num_slots=1)에 프레임 A, B 를 연속으로 받아 각각 pass 1 을 돌린다.
//      `--stale-barrier {on,off,both}` 로 FOREIGN_EXT→컴퓨트 acquire 배리어를 켜고/끄고 둘 다 실행한다.
//      P3-③ (배리어 있을 때 A/B 각각 정확) + P3-④ (배리어 없을 때 틀리는지 기록)
//
// ── 크기 계약 (부록 A.13) ─────────────────────────────────────────────────────────────────────
//   `dma_map_dmabuf(fd, size, D2H)` → **dmabuf_size** (버퍼 전체를 덮는 쪽이 안전하다)
//   `read_async(fd, size, cb)` / `hailo_dma_buffer_t.size` → **hw_frame_size 뿐이다**
//   ⚠ `infer_async` 에 틀린 크기를 넘기면 `HAILO_SUCCESS` 를 돌려주고 **콜백을 하나도 호출하지 않는다.**
//     그래서 프레임 완료 대기에 **반드시 시한**을 두고 초과를 치명적 오류로 올린다.
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

#include "common/cpu_letterbox.hpp"
#include "common/cpu_yolov8_decode.hpp"
#include "common/dma_heap_alloc.hpp"
#include "common/frame_types.hpp"
#include "vulkan/dmabuf_export.hpp"
#include "vulkan/dmabuf_import.hpp"
#include "vulkan/vk_check.hpp"
#include "vulkan/vk_storage_headroom.hpp"
#include "vulkan/vulkan_context.hpp"
#include "vulkan/yolo_postproc.hpp"

#include "heap_sum_spv.h"
#include "common/tool_util.hpp"
#include "vulkan/postproc_fixture.hpp"
#include "common/stats.hpp"

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
constexpr float kHefScoreThreshold = 0.2f;   // HEF 의 YOLOV8 post-process 설정

// ─────────────────────────────────────────────────────────────────── 출력 (stdout + 파일)

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

// 셸 한 줄 실행 결과 캡처. 정의는 heapimport 절에 있다(gpuab 가 온도·스로틀 기록에 먼저 쓴다).
std::string run_capture(const char* cmd);

// 빈 표본은 NaN("값 없음")으로 찍는다 — 0 과 구분하려고. 보간 자체는 common/stats.hpp 하나뿐이다.
double percentile(std::vector<double> v, double q) {
    return v.empty() ? std::nan("") : yvo::percentile(std::move(v), q);
}

// ─────────────────────────────────────────────────────────── 최소 JSON 리더 (fixtures.json 전용)
//
// nlohmann_json 은 이 시스템에서 conda prefix 에만 있어 tool 하나를 위해 conda include 경로를
// 끌어오게 된다(다른 시스템 헤더를 가릴 위험). 스키마가 고정돼 있으므로 필요한 만큼만 직접 읽는다.
// 픽스처 읽기·비교는 vulkan/postproc_fixture.hpp 하나뿐이다(테스트와 공유).

// 셰이더가 도는 **유효 영역**(h < grid_h, c < features, w < grid_w)의 바이트 합. kChecksum 과 같은 범위다.
std::uint32_t cpu_valid_sum(const std::uint8_t* p, int grid_h, int features, int grid_w, int w_padded) {
    std::uint32_t sum = 0;
    for (int h = 0; h < grid_h; ++h)
        for (int c = 0; c < features; ++c) {
            const std::uint8_t* row = p + yvo::nhcw_index(h, c, 0, features, w_padded);
            for (int w = 0; w < grid_w; ++w) sum += row[w];
        }
    return sum;
}

// ─────────────────────────────────────────────────────────────────── 옵션

struct Options {
    std::string probe = "fixture";
    std::string fixture_dir = "04_data/raw_tensors";
    std::string out_dir = "07_results/gpu_postprocess/evidence";
    std::string out_file;                 // 비면 probe 이름으로 정한다
    std::string hef = "03_models/yolov11m.hef";
    std::string video = "04_data/videos/video_h265.mp4";
    int perf_frames = 300;                // P3-⑥ (>= 200)
    int warmup = 30;
    int repeats = 10;                     // G6
    int live_frames = 20;
    int slots = 5;
    std::uint32_t lx = 10, ly = 6;   // YoloPostprocConfig 의 실측 기본값과 같게 둔다
    int max_dets = 4096;
    std::string stale_barrier = "both";
    std::string heap;                     // heapimport: dma-heap 경로. 비면 vidbuf_cached → system 폴백
    bool no_headroom = false;             // heapimport: B.12 헤드룸을 끈다(강제 검증이 실제로 터지는지 보는 음성 대조)
    // 부록 B 4·5단계: 원시 출력 버퍼의 할당 경로.
    //   plain = 오늘의 기본 경로 (fixture/gpuab → Plain, live/stale/v3 → ExportDmaBuf)
    //   heap  = ImportHeap (dma-heap 할당 + Vulkan import)
    std::string alloc = "plain";
    std::string heap_sync = "on";         // on | off | both — ImportHeap 의 DMA_BUF_IOCTL_SYNC (B.10 R2/R3)
    int rounds = 3;                       // gpuab: plain↔heap 인터리브 라운드 수
    int v3_window = 4096;                 // v3: line-dirty/stale-read 가 쓰는 텐서당 창 크기(B)
    bool out_dir_set = false;             // --out-dir 을 명시했는가 (probe 별 기본 경로 결정)
    std::string api = "read_async";       // live: read_async | infer_async
    std::uint16_t batch = 1;
    int timeout_ms = 10000;               // 프레임 완료 시한 (부록 A.13 의 영구 정지 방지)
    bool validation = false;
    bool sync_validation = false;
    bool verbose = false;
};

void usage() {
    std::puts(
        "yvo_postproc_probe --probe {fixture|sweep|live|stale|heapimport|gpuab|v3} [옵션]\n"
        "  --fixture-dir DIR      픽스처 디렉터리 (기본 04_data/raw_tensors)\n"
        "  --out-dir DIR          증거 파일 디렉터리 (기본 07_results/gpu_postprocess/evidence)\n"
        "  --out-file NAME        증거 파일 이름 (기본 p3_<probe>.txt)\n"
        "  --perf-frames N        연속 스트림 측정 프레임 수 (기본 300, P3-⑥ 는 >= 200)\n"
        "  --warmup N             측정 전 워밍업 (기본 30)\n"
        "  --repeats N            G6 결정성 반복 (기본 10)\n"
        "  --slots N              YoloPostproc 슬롯 수 (기본 5)\n"
        "  --local N,M            local_size_x,local_size_y (기본 10,6)\n"
        "  --max-dets N           후보 상한 (기본 4096)\n"
        "  --live-frames N        live 모드 프레임 수 (기본 20)\n"
        "  --api NAME             live: read_async | infer_async (기본 read_async)\n"
        "  --batch N              live: HEF batch_size (기본 1)\n"
        "  --timeout-ms N         프레임 완료 시한 (기본 10000)\n"
        "  --stale-barrier X      on | off | both (기본 both)\n"
        "  --alloc X              plain | heap  원시 출력 할당 경로 (기본 plain, 부록 B 4단계)\n"
        "  --heap-sync X          on | off | both  ImportHeap 의 DMA_BUF_IOCTL_SYNC (기본 on)\n"
        "  --rounds N             gpuab: plain↔heap 인터리브 라운드 (기본 3)\n"
        "  --v3-window N          v3: line-dirty/stale-read 의 텐서당 창 크기 B (기본 4096)\n"
        "  --heap PATH            heapimport: dma-heap 노드 (기본 vidbuf_cached → system 폴백)\n"
        "  --no-headroom          heapimport: B.12 헤드룸을 0 으로 (음성 대조 — conv71_178 이 예외로 터져야 정상)\n"
        "  --validation           검증 레이어를 컨텍스트에서 명시 활성화\n"
        "  --sync-validation      synchronization validation (--validation 과 함께)\n"
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

// ── 부록 B.10 규약 헬퍼 ───────────────────────────────────────────────────────────────────────
// R2(CPU 읽기) 구간을 RAII 로 연다. ImportHeap + heap_sync 가 아니면 아무것도 하지 않는다.
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

// **설정과 무관하게** 강제로 sync 를 건다. V3(0xA5 하네스)가 기준 스냅샷을 뜰 때 쓴다 —
// `--heap-sync off` 셀에서도 기준만은 정확해야 하기 때문이다.
void forced_sync(YoloPostproc& pp, int slot, yvo::DmaSyncDir dir, bool start) {
    for (int t = 0; t < kYoloNumTensors; ++t) {
        const int fd = pp.raw_dmabuf_fd(slot, t);
        if (start)
            yvo::dma_sync_begin(fd, dir, /*strict=*/true);
        else
            yvo::dma_sync_end(fd, dir, /*strict=*/true);
    }
}

// 프레임 데이터를 슬롯 버퍼에 채운다(슬롯당 프레임 하나를 상주시켜 연속 스트림 측정을 깨끗하게 만든다).
// **부록 B.10 R3**: 캐시 매핑을 더럽히는 CPU 쓰기는 START|WRITE … END|WRITE 로 감싼다.
// 감싸지 않으면 더러운 캐시 라인이 나중에 evict 되어 **장치가 쓴 데이터를 덮어쓴다**(B.6(c) 결함 유형).
void fill_slot(YoloPostproc& pp, int slot, const FrameTensors& f) {
    pp.raw_cpu_sync_begin(slot, yvo::DmaSyncDir::Write);
    for (int t = 0; t < kYoloNumTensors; ++t)
        std::memcpy(pp.raw_host_ptr(slot, t), f[static_cast<std::size_t>(t)].data(),
                    f[static_cast<std::size_t>(t)].size());
    pp.raw_cpu_sync_end(slot, yvo::DmaSyncDir::Write);
}

// --alloc 문자열 → YoloInputAlloc. plain 의 의미는 프로브마다 다르다(fixture=Plain, live=ExportDmaBuf).
YoloInputAlloc alloc_of(const std::string& a, YoloInputAlloc plain_means) {
    if (a == "heap") return YoloInputAlloc::ImportHeap;
    if (a == "plain") return plain_means;
    throw std::runtime_error("--alloc 은 plain | heap 이다: " + a);
}

std::vector<bool> heap_sync_cells(const std::string& v) {
    if (v == "on") return {true};
    if (v == "off") return {false};
    if (v == "both") return {true, false};
    throw std::runtime_error("--heap-sync 는 on | off | both 이다: " + v);
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

    // ── P3-① : G5 (프레임별 CPU 기준 비교) ───────────────────────────────────────────────────
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
            const RawRead rr(pp, s);   // R2: CPU 가 캐시 매핑을 읽는 구간
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
    log(fmt("[G5] %s  후보 합계 cpu=%zu gpu=%zu  좌표 최대오차 %.6f px (기준 < %.2f)",
            g5_ok ? "PASS" : "FAIL", total_cpu, total_gpu, static_cast<double>(worst_coord),
            static_cast<double>(kCoordTolPx)));

    // ── G6 : 결정성 ──────────────────────────────────────────────────────────────────────────
    log("");
    log(fmt("== G6: 같은 픽스처 %d회 반복, (score, anchor_key) 전순서 정렬 후 완전 동일 ==", o.repeats));
    bool g6_ok = true;
    for (int rep = 1; rep < o.repeats; ++rep) {
        for (int s = 0; s < nslots; ++s) {
            pp.submit(s, rig.fs.lb, rig.fs.src_w, rig.fs.src_h);
            pp.wait(s);
            if (canonical(pp.read_candidates(s)) != canon_ref[static_cast<std::size_t>(s)]) {
                log(fmt("[G6] FAIL: rep=%d slot=%d 결과가 1회차와 다르다", rep, s));
                g6_ok = false;
            }
        }
    }
    log(fmt("[G6] %s (%d회 x %d프레임)", g6_ok ? "PASS" : "FAIL", o.repeats, nslots));

    // ── 바이트 합 (kChecksum 파이프라인) ────────────────────────────────────────────────────
    log("");
    log("== 바이트 합: 셰이더(kChecksum) vs CPU (유효 영역만, P5 열 20~23 제외) ==");
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

    // ── 하이브리드(§4.4 A) 호스트 쪽 비용: 후보 읽기 + greedy NMS ──────────────────────────
    log("");
    log("== 하이브리드(계획서 §4.4 A): GPU 후보 → 호스트 read + cpu_nms_greedy_by_class ==");
    log("frame  cand  read_us  nms_us  final_dets  cpu_ref_dets  일치");
    bool hyb_ok = true;
    for (int s = 0; s < nslots; ++s) {
        pp.submit(s, rig.fs.lb, rig.fs.src_w, rig.fs.src_h);
        pp.wait(s);
        const auto t0 = std::chrono::steady_clock::now();
        auto cand = pp.read_candidates(s);
        const auto t1 = std::chrono::steady_clock::now();
        const std::size_t n_before = cand.size();
        yvo::cpu_nms_greedy_by_class(cand, 0.7f, 100);   // HEF 설정: IoU 0.7, 클래스당 100
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
    log(fmt("[hybrid] 최종 검출 수 %s (계획서 §4.4 의 '5~20 µs' 주장을 위 read_us+nms_us 로 대조한다)",
            hyb_ok ? "PASS" : "FAIL"));

    // ── P3-⑥ : 연속 스트림 pass 1 GPU 시간 ──────────────────────────────────────────────────
    log("");
    log(fmt("== P3-(6) pass 1 GPU timestamp, 연속 스트림 %d프레임 (워밍업 %d) ==", o.perf_frames, o.warmup));
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
    log(fmt("[§4.7 추정 0.4~0.9 ms] 실측 중앙값 %.4f ms → %s", percentile(pass1, 0.5),
            (percentile(pass1, 0.5) >= 0.4 && percentile(pass1, 0.5) <= 0.9) ? "추정 범위 안"
                                                                            : "추정 범위 밖(보고에 명시)"));

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
    log(fmt("[sweep] maxComputeWorkGroupInvocations=%u, 프레임 %d, 반복 %d (인터리브 3회)", maxinv,
            o.perf_frames, 3));
    log("local_x local_y  inv   pass1_median_ms  p90     cand/frame");

    // x 는 워드 열이라 20(P3)/10(P4)/5(P5) 가 상한이다. y 와 함께 invocation 수를 256 이하로 유지한다.
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

    for (int round = 0; round < 3; ++round) {   // 인터리브: DVFS 드리프트를 배제한다
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


// ───────────────────────────────────────── gpuab (부록 B 4단계: GPU 읽기 A/B, **NPU 불필요**)
//
// 같은 실행 안에서 `plain`(Vulkan 메모리)과 `heap`(dma-heap + Vulkan import) 을 **인터리브**로 돌려
// pass 1 GPU 시간을 비교한다. 판정은 **절대 ms 가 아니라 같은 실행 안의 비율**이다 —
// DVFS·온도 드리프트에 둔감하다(부록 B.13 합격 밴드).
//   heap ≤ plain ×1.05  → A: 손해 없음 (GPU 가 읽는 버퍼도 heap 후보)
//   ×1.05 ~ ×1.30       → B: 부분 손해 (heap 은 "CPU 가 크게 읽는 버퍼" 전용)
//   > ×1.30             → C: 손해 (S3. GPU 접근 버퍼에 쓰지 않는다)
// **정확성(G5)을 같은 실행에서 동시에 판정한다.** 여기서 틀리면 성능 숫자는 볼 필요도 없다(S2).
//
// `cpu_read_1.22MB` 도 같이 잰다(plain/heap/malloc 3자, 같은 실행·인터리브) — B.18 에서
// heap 이 0.396 → 0.292 ms 로 개선된 원인이 미확인이라 조건을 통일해 다시 확정한다.

struct AbMode {
    const char* name = "";
    YoloInputAlloc alloc = YoloInputAlloc::Plain;
    std::unique_ptr<YoloPostproc> pp;
    std::vector<double> pass1;
    std::vector<double> cmdbuf;
    std::vector<double> round_median;   // 라운드별 중앙값 (비율 계산용)
    std::vector<double> cpu_read_ms;
    bool g5_ok = false;
    std::size_t g5_cpu = 0, g5_gpu = 0;
    float g5_coord = 0.0f;
    std::size_t g5_score_bits = 0;
    bool sum_ok = false;
    std::string canon;                  // 후보 집합 정준형 (plain↔heap 교차 대조용)
};

// 한 모드의 G5 + 바이트 합 검증. 같은 픽스처 프레임을 쓰므로 plain 과 heap 의 결과는
// **비트 단위로 같아야 한다** — 메모리 타입은 가시성·순서 문제이지 값 문제가 아니다(B.7).
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

        // 바이트 합(kChecksum): 셰이더가 실제로 읽은 바이트 == CPU 가 매핑으로 보는 바이트
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
                log(fmt("  [%s] slot %d tensor %d 바이트 합 불일치: shader %u != cpu %u", m.name, s, t,
                        shader[t], cs));
            }
        }
    }
    m.canon = canon_all;
}

// 6 텐서 전체를 CPU 가 순차 memcpy 로 읽는 시간(ms). sync 비용을 **포함**한다
// (hailo_raw_probe 의 read_all_ms 와 같은 방법 — P2-⑥ 와 비교 가능하게 맞췄다).
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
    log(fmt("[gpuab] dir=%s frames=%d  인터리브 %d라운드 x (워밍업 %d + 측정 %d)", o.fixture_dir.c_str(),
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

    // fd 누수 기준선: VulkanContext 가 연 fd 는 세지 않는다(컨텍스트는 이 블록보다 오래 산다).
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

    // ── ① 정확성 (G5 + 바이트 합) — 성능보다 먼저 판정한다 ─────────────────────────────────
    log("");
    log("== ① G5 정확성 (같은 실행, 같은 픽스처) ==");
    for (AbMode& m : modes) ab_correctness(m, rig, order, log);
    const bool cross_same = modes[0].canon == modes[1].canon;
    for (const AbMode& m : modes)
        log(fmt("[G5 %s] %s  후보 cpu=%zu gpu=%zu  좌표 최대오차 %.6f px (기준 < %.2f)  점수 비트 불일치 %zu  "
                "바이트합 %s",
                m.name, m.g5_ok ? "PASS" : "FAIL", m.g5_cpu, m.g5_gpu, static_cast<double>(m.g5_coord),
                static_cast<double>(kCoordTolPx), m.g5_score_bits, m.sum_ok ? "일치" : "불일치"));
    log(fmt("[교차] plain 과 heap 의 후보 집합이 바이트 단위로 동일한가: %s",
            cross_same ? "예 (메모리 타입은 값 문제가 아니다 — B.7)" : "**아니오 — 하드 스톱(S2)**"));
    const bool acc_ok = modes[0].g5_ok && modes[1].g5_ok && modes[0].sum_ok && modes[1].sum_ok && cross_same;
    if (!acc_ok) {
        log("[S2] 정확성 실패 — 성능 수치를 보지 않는다. 5단계 sync 규약으로 고친 뒤 재측정한다.");
        log("[RESULT] gpuab FAIL");
        return 1;
    }

    // ── ② 성능: plain ↔ heap 인터리브 ────────────────────────────────────────────────────────
    log("");
    log(fmt("== ② pass 1 GPU timestamp — plain ↔ heap 인터리브 %d라운드 (연속 스트림) ==", o.rounds));
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

    // 라운드별 비율의 중앙값을 쓴다(라운드 안에서는 온도 조건이 거의 같다)
    std::vector<double> ratios;
    for (int r = 0; r < o.rounds; ++r) ratios.push_back(modes[1].round_median[r] / modes[0].round_median[r]);
    const double ratio = percentile(ratios, 0.5);
    const double ratio_all = percentile(modes[1].pass1, 0.5) / percentile(modes[0].pass1, 0.5);
    const char* band = ratio <= 1.05 ? "A: 손해 없음 — GPU 가 읽는 버퍼도 heap 후보"
                                     : (ratio <= 1.30 ? "B: 부분 손해 — heap 은 'CPU 가 크게 읽는 버퍼' 전용"
                                                      : "C: 손해(S3) — GPU 접근 버퍼에 쓰지 않는다");
    log(fmt("[비율] 라운드별 heap/plain 중앙값 = %.4f (전체 중앙값 기준 %.4f)", ratio, ratio_all));
    log(fmt("[밴드 판정] %s", band));

    // ── ③ cpu_read_1.22MB: plain / heap / malloc 3자, 같은 실행 인터리브 ────────────────────
    log("");
    log(fmt("== ③ cpu_read_1.22MB — plain / heap / malloc 3자 인터리브 %d라운드 x %d회 ==", o.rounds,
            o.repeats));
    std::size_t total_bytes = 0;
    for (int t = 0; t < kYoloNumTensors; ++t) total_bytes += modes[0].pp->raw_size(t);
    std::vector<std::uint8_t> dst(total_bytes), mal_src(total_bytes);
    for (std::size_t i = 0; i < mal_src.size(); ++i) mal_src[i] = static_cast<std::uint8_t>(i * 131u);
    std::vector<double> malloc_ms;
    for (int i = 0; i < 3; ++i) {   // 워밍업
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
    log(fmt("총 읽기 바이트 = %zu B (6 텐서 hw_frame_size 합)", total_bytes));
    log("alloc    median_ms   p90_ms    min_ms    GB/s(median)  비고");
    auto rowline = [&](const char* nm, const std::vector<double>& v, const char* note) {
        const double med = percentile(v, 0.5);
        log(fmt("%-7s  %9.3f  %8.3f  %8.3f  %12.3f  %s", nm, med, percentile(v, 0.9),
                *std::min_element(v.begin(), v.end()),
                static_cast<double>(total_bytes) / (med * 1e-3) / 1e9, note));
    };
    rowline("malloc", malloc_ms, "호스트 malloc (기준)");
    rowline("heap", modes[1].cpu_read_ms, "dma-heap 캐시 매핑, DMA_BUF_IOCTL_SYNC 포함");
    rowline("plain", modes[0].cpu_read_ms, "V3D HOST_COHERENT (캐시 없음)");
    log(fmt("[cpu_read] heap/plain = %.3f배 (P2 실측 0.396/0.826 = 0.48배와 대조)",
            percentile(modes[1].cpu_read_ms, 0.5) / percentile(modes[0].cpu_read_ms, 0.5)));
    log("⚠ malloc 행은 **낙관적 기준**이다: src·dst 가 둘 다 malloc 이고 같은 버퍼를 반복해서 읽으므로");
    log("  L2/SLC 에 남는다. P2 의 malloc 0.244 ms 는 HailoRT 가 출력을 받은 버퍼를 읽은 것이라 조건이 다르다.");
    log("  **판정에 쓰는 것은 heap 과 plain 의 비교다** — 이 둘은 같은 실행·같은 방법·같은 크기다.");
    log("  heap 행의 sync 비용(START|READ + END|READ x 6텐서)은 시간에 포함돼 있다(부록 B.10 R6).");

    log("");
    log(fmt("[validation] errors=%llu warnings=%llu",
            static_cast<unsigned long long>(ctx.validation_errors()),
            static_cast<unsigned long long>(ctx.validation_warnings())));
    // 파괴 순서를 명시한다(import 캐시 → 힙 풀은 YoloPostproc::destroy() 가 처리한다)
    for (AbMode& m : modes) m.pp.reset();
    const long fd_after = count_open_fds();
    log(fmt("[fd] 기준선 %ld → 해제 후 %ld → 누수 %ld  (heap 슬롯 %d x 텐서 %d dma-buf + import dup fd)",
            fd_base, fd_after, fd_after - fd_base, std::max(1, std::min(o.slots, rig.fs.frames)),
            kYoloNumTensors));
    const bool pass = acc_ok && ctx.validation_errors() == 0 && fd_after == fd_base;
    log(fmt("[RESULT] gpuab %s  (밴드 %c)", pass ? "PASS" : "FAIL",
            ratio <= 1.05 ? 'A' : (ratio <= 1.30 ? 'B' : 'C')));
    return pass ? 0 : 1;
}

// ─────────────────────────────────────────────────────────────── live / stale (NPU)

#ifdef YVO_HAVE_HAILO

std::string status_name(hailo_status s) { return std::to_string(static_cast<int>(s)); }

void hcheck(hailo_status s, const char* where) {
    if (s != HAILO_SUCCESS) throw std::runtime_error(std::string(where) + " 실패: status " + status_name(s));
}

template <typename T>
T unwrap(hailort::Expected<T>&& e, const char* where) {
    if (!e) throw std::runtime_error(std::string(where) + " 실패: status " + status_name(e.status()));
    return e.release();
}

std::vector<std::vector<std::uint8_t>> make_nhcw_inputs(const std::string& video, int n, const LetterboxParams& lb,
                                                        std::size_t frame_size) {
    const std::string cmd = "ffmpeg -nostdin -v error -i '" + video + "' -frames:v " + std::to_string(n) +
                            " -f rawvideo -pix_fmt nv12 -";
    FILE* p = popen(cmd.c_str(), "r");
    if (!p) throw std::runtime_error("popen(ffmpeg) 실패");
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
    if (out.empty()) throw std::runtime_error("영상에서 프레임을 못 읽었다: " + video);
    return out;
}

// YoloPostproc 의 dma-buf 로 원시 출력을 받는 최소 세션.
// hailo_raw_probe.cpp 의 RawSession 과 같은 규율을 따른다(프레임 완료에 **시한**을 둔다 — 부록 A.13).
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
        if (ngs.size() != 1) throw std::runtime_error("network group 이 1개가 아니다");
        ng_ = ngs[0];

        auto ins = ng_->get_input_streams();
        if (ins.size() != 1) throw std::runtime_error("입력 스트림이 1개가 아니다");
        input_ = &ins[0].get();
        in_size_ = input_->get_frame_size();
        in_name_ = input_->name();

        // 텐서 번호 0..5 ↔ OutputStream 대응. 이름은 호출자가 fixtures.json 에서 가져온다.
        auto outs = ng_->get_output_streams();
        for (int t = 0; t < kYoloNumTensors; ++t) {
            hailort::OutputStream* found = nullptr;
            for (auto& r : outs)
                if (r.get().name() == tensor_stream_name[static_cast<std::size_t>(t)]) found = &r.get();
            if (!found)
                throw std::runtime_error("출력 스트림을 못 찾았다: " + tensor_stream_name[static_cast<std::size_t>(t)]);
            out_[t] = found;
            hw_size_[t] = found->get_frame_size();
            if (hw_size_[t] != pp_.raw_size(t))
                throw std::runtime_error("hw_frame_size 가 픽스처 메타와 다르다: " + found->name() + " " +
                                         std::to_string(hw_size_[t]) + " vs " + std::to_string(pp_.raw_size(t)));
        }

        // dma-buf 매핑: **dmabuf_size** 로 매핑한다(부록 A.13). 프로세스 수명 동안 한 번만.
        const auto t0 = std::chrono::steady_clock::now();
        for (int s = 0; s < pp_.num_slots(); ++s)
            for (int t = 0; t < kYoloNumTensors; ++t) {
                const int fd = pp_.raw_dmabuf_fd(s, t);
                const std::size_t sz = pp_.raw_dmabuf_size(s, t) ? pp_.raw_dmabuf_size(s, t) : hw_size_[t];
                const auto st = vdevice_->dma_map_dmabuf(fd, sz, HAILO_DMA_BUFFER_DIRECTION_D2H);
                ++map_calls_;
                if (st != HAILO_SUCCESS) throw std::runtime_error("dma_map_dmabuf 실패: status " + status_name(st));
                mapped_.push_back({fd, sz});
            }
        map_ms_ = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        in_buf_.assign(static_cast<std::size_t>(pp_.num_slots()) * in_size_, 0);
        if (verbose) std::fprintf(stderr, "[LiveSession] dma_map_dmabuf %ld회 %.3f ms\n", map_calls_, map_ms_);
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

    // 프레임 1장을 슬롯 slot 의 dma-buf 로 받는다. **시한 안에 못 끝내면 예외**(부록 A.13).
    void run_frame(int slot, const std::vector<std::uint8_t>& nhcw, int timeout_ms) {
        if (nhcw.size() != in_size_) throw std::runtime_error("입력 프레임 크기가 다르다");
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
                "프레임 완료 시한 초과 — 콜백이 오지 않았다(부록 A.13: 크기 계약 위반 시 infer_async 가 "
                "HAILO_SUCCESS 를 돌려주고 조용히 멈춘다)");
        if (status_.load() != HAILO_SUCCESS)
            throw std::runtime_error("콜백 status 오류: " + status_name(status_.load()));
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
            hailort::BufferRepresentation br{};   // union 이다: buffer_type 을 먼저 세팅한다
            br.buffer_type = hailort::BufferType::VIEW;
            br.view = hailort::MemoryView(in, in_size_);
            named.emplace(in_name_, std::make_pair(br, [this](hailo_status st) { note(st); }));
        }
        for (int t = 0; t < kYoloNumTensors; ++t) {
            hailort::BufferRepresentation br{};
            br.buffer_type = hailort::BufferType::DMA_BUFFER;
            // ⚠ read 쪽 크기는 **hw_frame_size 뿐이다**(부록 A.13)
            br.dma_buffer = hailo_dma_buffer_t{pp_.raw_dmabuf_fd(slot, t), hw_size_[t]};
            named.emplace(out_[t]->name(), std::make_pair(br, [this](hailo_status st) { note(st); }));
        }
        hcheck(ng_->wait_for_ongoing_callbacks_count_under(1), "wait_for_ongoing_callbacks_count_under");
        ng_->increase_ongoing_callbacks();
        const auto st = ng_->infer_async(named, [this](hailo_status) { ng_->decrease_ongoing_callbacks(); });
        if (st != HAILO_SUCCESS) {
            ng_->decrease_ongoing_callbacks();
            remaining_.store(0);
            throw std::runtime_error("infer_async 실패: status " + status_name(st));
        }
    }

    void issue_read_async(int slot, std::uint8_t* in) {
        // FIFO 짝맞추기: write 1회 + read 6회를 한 mutex 안에서 발행한다(프레임을 순차로 돌리므로 충분하다).
        std::lock_guard<std::mutex> issue(issue_m_);
        const auto st_in = input_->write_async(
            in, in_size_, [this](const hailort::InputStream::CompletionInfo& ci) { note(ci.status); });
        if (st_in != HAILO_SUCCESS) {
            remaining_.store(0);
            throw std::runtime_error("write_async 실패: status " + status_name(st_in));
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
                throw std::runtime_error("read_async 실패: status " + status_name(st));
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

// measure 스킬의 냉각 규칙: 75 °C 를 넘으면 65 °C 까지, 최대 120 초.
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
        log(fmt("[thermal] ⚠ 냉각 대기 %d s 뒤에도 칩 온도를 읽지 못했다(장치를 다른 프로세스가 쥐었다) — "
                "이 실행의 성능 수치는 신뢰 불가", waited));
    else
        log(fmt("[thermal] 냉각 대기 %d s → %.1f °C", waited, static_cast<double>(now)));
    return waited;
}

std::string read_first_line(const std::string& path) {
    std::ifstream f(path);
    std::string l;
    std::getline(f, l);
    return l;
}

// Hailo-8 의 PCIe 경로는 시스템마다 다르다(이 Pi 5 는 0001:01:00.0). vendor id 0x1e60 으로 찾는다.
std::string pcie_link_speed() {
    std::error_code ec;
    for (auto it = std::filesystem::directory_iterator("/sys/bus/pci/devices", ec);
         !ec && it != std::filesystem::directory_iterator(); it.increment(ec)) {
        const std::string dir = it->path().string();
        if (read_first_line(dir + "/vendor") != "0x1e60") continue;
        return read_first_line(dir + "/current_link_speed") + " x" + read_first_line(dir + "/current_link_width") +
               " (" + it->path().filename().string() + ")";
    }
    return "(Hailo PCIe 장치를 못 찾았다)";
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

// /dev/hailo0 을 다른 프로세스가 쥐고 있으면 VDevice::create 가 HAILO_OUT_OF_PHYSICAL_DEVICES(74) 로
// 실패한다. 원인을 추측하지 않도록 먼저 확인해 이름까지 보고한다.
std::string npu_holder() {
    std::error_code ec;
    for (auto it = std::filesystem::directory_iterator("/proc", ec);
         !ec && it != std::filesystem::directory_iterator(); it.increment(ec)) {
        const std::string name = it->path().filename().string();
        if (name.empty() || !std::isdigit(static_cast<unsigned char>(name[0]))) continue;
        // 자기 자신은 건너뛴다 — libhailort 가 프로세스 시작 시 /dev/hailo0 을 이미 열어 둔다.
        if (name == std::to_string(::getpid())) continue;
        std::error_code ec2;
        for (auto f = std::filesystem::directory_iterator(it->path() / "fd", ec2);
             !ec2 && f != std::filesystem::directory_iterator(); f.increment(ec2)) {
            std::error_code ec3;
            const auto target = std::filesystem::read_symlink(f->path(), ec3);
            if (!ec3 && target.string().rfind("/dev/hailo", 0) == 0)   // 경로에 "hailo" 가 든 파일과 구분한다
                return name + " (" + read_first_line(it->path().string() + "/comm") + ")";
        }
    }
    return std::string();
}

int run_live_inner(const Options& o, Tee& log, long fd0);

// fd 누수 계수: 세션·YoloPostproc·VulkanContext 가 **전부 파괴된 뒤**에 세야 의미가 있다.
int run_live(const Options& o, Tee& log) {
    const long fd0 = count_open_fds();
    const int rc = run_live_inner(o, log, fd0);
    const long fd1 = count_open_fds();
    log(fmt("[fd] 해제 후 start=%ld end=%ld  %s", fd0, fd1, fd1 == fd0 ? "누수 없음" : "⚠ 누수 의심"));
    return (rc == 0 && fd1 == fd0) ? 0 : (rc != 0 ? rc : 1);
}

int run_live_inner(const Options& o, Tee& log, long fd0) {
    FixtureSet fs = load_fixtures(o.fixture_dir);
    const YoloTensorLayout L = layout_from_fixtures(fs);
    log(fmt("[live] hef=%s video=%s frames=%d api=%s batch=%u slots=%d", o.hef.c_str(), o.video.c_str(),
            o.live_frames, o.api.c_str(), o.batch, o.slots));
    log(fmt("[pcie] %s  fd(start)=%ld", pcie_link_speed().c_str(), fd0));
    if (const std::string holder = npu_holder(); !holder.empty())
        throw std::runtime_error("NPU 가 이미 다른 프로세스에 잡혀 있다: pid " + holder +
                                 " — Hailo-8 은 1개뿐이라 독점해야 한다");
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
    pc.input_alloc = alloc_of(o.alloc, YoloInputAlloc::ExportDmaBuf);   // plain = 계획서 §4.2 채택안
    pc.input_sync = YoloInputSync::Foreign;          // Hailo PCIe DMA 가 쓴 외부 메모리다
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
    log(fmt("[live] dma_map_dmabuf %ld회 %.3f ms", sess.map_calls(), sess.map_ms()));

    // 입력 프레임 준비
    std::size_t in_sz = 0;
    {
        // LiveSession 이 이미 검증했다. 입력 크기는 640*640*3 이다.
        in_sz = static_cast<std::size_t>(fs.lb.dst_w) * fs.lb.dst_h * 3;
    }
    const auto inputs = make_nhcw_inputs(o.video, o.live_frames, fs.lb, in_sz);
    log(fmt("[live] 입력 프레임 %zu장 준비 (NHCW %zu B)", inputs.size(), in_sz));

    log("");
    log("== P3-(2) / V1 3자 대조: Hailo DMA 쓰기 → GPU 계산 합 ↔ CPU 캐시 매핑 읽기 합 ↔ malloc 스냅샷 합 ==");
    log("  · shader : kChecksum 파이프라인이 V3D 로 읽어 더한 값");
    log("  · mapped : CPU 가 힙 fd 의 캐시 매핑을 READ sync 구간 안에서 직접 읽어 더한 값");
    log("  · malloc : 같은 READ 구간 안에서 malloc 버퍼로 memcpy 한 뒤 그 사본을 더한 값");
    log("  ⚠ malloc 사본은 같은 매핑에서 떠온 것이라 **독립 경로가 아니다** — 매핑 읽기가 도중에 흔들리는");
    log("    경우(찢어진 읽기)를 잡을 뿐, HailoRT 가 malloc 버퍼로 직접 받은 P2 의 기준과는 다르다.");
    log("frame slot  tensor mismatch  cand  cpu_cand  matched  max_coord_err_px");
    bool sum_ok = true, dec_ok = true, v1_ok = true;
    std::size_t v1_cmp = 0, v1_bad_mapped = 0, v1_bad_malloc = 0;
    std::vector<double> pass1;
    std::vector<std::uint8_t> snap;
    const auto order = tensor_order(fs);
    for (std::size_t f = 0; f < inputs.size(); ++f) {
        const int slot = static_cast<int>(f) % pp.num_slots();
        sess.run_frame(slot, inputs[f], o.timeout_ms);

        // ① 바이트 합 대조 (kChecksum 파이프라인)
        pp.submit(slot, fs.lb, fs.src_w, fs.src_h, /*checksum=*/true);
        pp.wait(slot);
        std::uint32_t shader[kYoloNumTensors];
        pp.checksums(slot, shader);
        int mism = 0;
        {
            // R2: 완료 신호(vkWaitForFences + HailoRT 콜백) **뒤에** READ 구간을 연다.
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

        // ② 같은 버퍼로 평상시 파이프라인 → CPU 기준 비교
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
    log(fmt("[P3-(2)] 바이트 합 %s / 디코드 일치 %s", sum_ok ? "PASS" : "FAIL", dec_ok ? "PASS" : "FAIL"));
    log(fmt("[V1] 3자 대조 %s — 비교 %zu건(텐서 단위, %zu프레임 x 6) · shader!=mapped %zu · shader!=malloc %zu "
            "(기준: 불일치 0)",
            v1_ok ? "PASS" : "FAIL", v1_cmp, inputs.size(), v1_bad_mapped, v1_bad_malloc));
    if (!pass1.empty())
        log(fmt("[live] pass1_ms median %.4f p90 %.4f (NPU 동시 실행 상태 — 부록 A.8: 단독 벤치보다 높게 나온다)",
                percentile(pass1, 0.5), percentile(pass1, 0.9)));

    const int err1 = count_hailort_log_errors();
    const int th1 = count_hailort_log_thermal();
    log(fmt("[hailort.log] error +%d  온도·클럭 warning +%d %s", err1 - err0, th1 - th0,
            (th1 - th0) ? "→ 이 실행의 성능 수치는 신뢰 불가" : ""));
    log(fmt("[validation] errors=%llu warnings=%llu",
            static_cast<unsigned long long>(ctx.validation_errors()),
            static_cast<unsigned long long>(ctx.validation_warnings())));
    log(fmt("[fd] start=%ld  세션 보유 중=%ld (= 슬롯 %d x 텐서 %d dma-buf + HailoRT/Vulkan 내부)", fd0,
            count_open_fds(), pp.num_slots(), kYoloNumTensors));
    const bool pass = sum_ok && dec_ok && v1_ok && ctx.validation_errors() == 0 && (err1 - err0) == 0;
    log(fmt("[RESULT] live %s", pass ? "PASS" : "FAIL"));
    return pass ? 0 : 1;
}

// 같은 슬롯 버퍼에 프레임 A, B 를 연속으로 받아 각각 pass 1 을 돌린다(배리어 on/off).
int run_stale(const Options& o, Tee& log) {
    FixtureSet fs = load_fixtures(o.fixture_dir);
    const YoloTensorLayout L = layout_from_fixtures(fs);
    log(fmt("[stale] 같은 슬롯 버퍼(num_slots=1)에 서로 다른 프레임 A/B 를 연속으로 받는다. barrier=%s",
            o.stale_barrier.c_str()));
    log(fmt("[pcie] %s", pcie_link_speed().c_str()));
    if (const std::string holder = npu_holder(); !holder.empty())
        throw std::runtime_error("NPU 가 이미 다른 프로세스에 잡혀 있다: pid " + holder +
                                 " — Hailo-8 은 1개뿐이라 독점해야 한다");
    cool_down(log);
    const int err0 = count_hailort_log_errors();

    std::vector<bool> modes;
    if (o.stale_barrier == "on" || o.stale_barrier == "both") modes.push_back(true);
    if (o.stale_barrier == "off" || o.stale_barrier == "both") modes.push_back(false);
    // V2(부록 B.10): ImportHeap 이면 `--heap-sync` 축을 더해 {sync} x {barrier} 셀을 전부 돈다.
    const YoloInputAlloc stale_alloc = alloc_of(o.alloc, YoloInputAlloc::ExportDmaBuf);
    const std::vector<bool> syncs =
        stale_alloc == YoloInputAlloc::ImportHeap ? heap_sync_cells(o.heap_sync) : std::vector<bool>{true};
    log(fmt("[stale] alloc=%s  셀 %zu(heap_sync) x %zu(barrier)", o.alloc.c_str(), syncs.size(), modes.size()));

    // 입력 디코드는 NPU 가 필요 없다 → **세션을 만들기 전에** 끝내 둔다(NPU 보유 시간을 최소화한다).
    const std::size_t in_sz = static_cast<std::size_t>(fs.lb.dst_w) * fs.lb.dst_h * 3;
    const auto inputs = make_nhcw_inputs(o.video, std::max(2, o.live_frames), fs.lb, in_sz);
    const std::vector<std::vector<std::uint8_t>> ab{inputs.front(), inputs.back()};
    log(fmt("[stale] 프레임 A/B 준비 완료 (영상 앞 %zu장 중 첫/마지막)", inputs.size()));

    bool barrier_on_ok = true;
    bool barrier_off_wrong = false;
    bool sync_off_wrong = false;
    bool sync_on_ok = true;
    std::uint64_t vk_errors = 0;
    for (const bool heap_sync : syncs)
    for (const bool barrier : modes) {
        log("");
        log(fmt("---- heap_sync %s / acquire 배리어 %s ----",
                stale_alloc == YoloInputAlloc::ImportHeap ? (heap_sync ? "ON" : "OFF") : "n/a",
                barrier ? "ON (FOREIGN_EXT → 컴퓨트)" : "OFF (배리어 없음)"));
        yvo::vk::VulkanContextOptions co;
        co.enable_validation = o.validation;
        co.validation_sync = o.sync_validation;
        co.verbose = o.verbose;
        yvo::vk::VulkanContext ctx(co);

        YoloPostprocConfig pc;
        pc.num_slots = 1;   // **같은 버퍼를 재사용하게 만든다** — 스테일이 드러나는 조건이다
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
        for (int rep = 0; rep < 2; ++rep) {   // A, B 를 두 번 돌려 순서 의존성까지 본다
            for (std::size_t k = 0; k < ab.size(); ++k) {
                sess.run_frame(0, ab[k], o.timeout_ms);
                pp.submit(0, fs.lb, fs.src_w, fs.src_h);
                pp.wait(0);
                const auto gpu = pp.read_candidates(0);
                const std::uint8_t* ptr[kYoloNumTensors];
                for (int t = 0; t < kYoloNumTensors; ++t) ptr[t] = pp.raw_host_ptr(0, t);
                std::vector<DetectionKeyed> cpu;
                {
                    const RawRead rr(pp, 0);   // R2 (heap_sync=off 셀에서는 no-op 이다 — 그것이 이 셀의 요점이다)
                    cpu = yvo::cpu_yolov8_candidates(make_raw_set(L, ptr), kHefScoreThreshold, fs.lb, fs.src_w,
                                                     fs.src_h);
                }
                const Compare c = compare_candidates(cpu, gpu);
                // 바이트 합도 같이 본다(셰이더가 실제로 읽은 바이트가 호스트가 보는 것과 같은가)
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
    log(fmt("[P3-(3)] 배리어 ON 에서 A/B 각각 정확: %s", barrier_on_ok ? "PASS" : "FAIL"));
    log(fmt("[P3-(4)] 배리어 OFF 에서 틀린 결과가 나왔는가: %s (판정 아님 — 기록)",
            barrier_off_wrong ? "예 (배리어가 필수임을 보이는 증거)"
                              : "아니오 (이 하드웨어에서는 배리어 없이도 맞았다 — 우연일 수 있다)"));
    if (stale_alloc == YoloInputAlloc::ImportHeap) {
        log(fmt("[V2] heap_sync ON 셀에서 A/B 각각 정확: %s", sync_on_ok ? "PASS" : "FAIL"));
        log(fmt("[V2] heap_sync OFF 에서 틀린 결과가 나왔는가: %s",
                sync_off_wrong ? "예" : "아니오"));
        log("     ⚠ **부록 A P3-④ 교훈**: OFF 에서 안 틀렸다고 해서 **sync 가 불필요하다는 증거가 아니다.**");
        log("       이 테스트는 캐시 라인이 우연히 깨끗했을 때를 배제하지 못한다. 실패를 강제하는 것은 V3 다.");
        log("       구조적 근거는 그대로다 — 디바이스 트리에 `dma-coherent` 가 0건이라 커널이 V3D 를");
        log("       비일관 장치로 취급한다. 실험이 그것을 반증하지 못한 것일 뿐이다.");
    }
    log(fmt("[hailort.log] error +%d", err1 - err0));
    const bool pass = barrier_on_ok && sync_on_ok && vk_errors == 0 && (err1 - err0) == 0;
    log(fmt("[RESULT] stale %s", pass ? "PASS" : "FAIL"));
    return pass ? 0 : 1;
}


// ───────────────────────────────── V3: 실패를 강제하는 테스트 (부록 B.10, **NPU 독점**)
//
// V2 는 "운 좋게 안 틀릴" 수 있다. V3 는 캐시를 의도적으로 나쁜 상태로 만들어 실패를 **강제**한다.
// 세 가지 변종을 돌린다 — 앞의 둘은 **쓰기 방향**(더러운 라인), 셋째는 **읽기 방향**(낡은 깨끗한 라인)이다.
// 읽기 방향이 부록 B.18 이 "아직 한 번도 실증되지 않았다"고 적은 바로 그 구멍이다.
//
//   ① full-dirty  : CPU 가 버퍼 **전체**에 0xA5 를 memset — `END|WRITE` 를 일부러 생략 → Hailo DMA 가
//                   프레임 A 를 씀 → 읽는다. 더러운 라인이 남아 있으면 0xA5 가 살아남는다.
//                   ⚠ 1.22 MB 는 A76 의 write-streaming(긴 순차 쓰기에 라인 할당을 끔)과 용량 축출 때문에
//                     추론이 끝나기 전에 대부분 DRAM 으로 내려간다 → 재현력이 낮다. ②가 그 보완이다.
//   ② line-dirty  : **작은 창**(--v3-window, 기본 4096 B)에만, **64 B 라인마다 1바이트씩** 쓴다.
//                   부분 라인 쓰기는 write-streaming 을 못 쓰고 라인을 fetch 해서 dirty 로 만든다.
//                   창이 작아 L1/L2 에 머무를 가능성이 훨씬 높다. `END|WRITE` 는 역시 생략한다.
//   ③ stale-read  : **읽기 방향**. 프레임 B 를 받아 창을 읽어 캐시를 깨끗한 라인으로 채운다(priming) →
//                   Hailo DMA 가 **다른** 프레임 A 를 씀 → `START|READ` 없이 읽는다.
//                   낡은 라인이 남아 있으면 **프레임 B 의 바이트**가 보인다. 그것을 세는 것이 판정이다.
//
// 기준(reference)은 **셀과 무관하게 강제 sync** 로 뜬다(`forced_sync`) — 기준까지 오염되면 판정이 무의미하다.
// 기대: `--heap-sync off` 에서 오염 > 0, `on` 에서 0. **`on` 셀의 오염 0 만 PASS 조건이다.**
constexpr std::uint8_t kDirtyByte = 0xA5;
constexpr std::size_t kCacheLine = 64;   // getconf LEVEL1_DCACHE_LINESIZE = 64 (Cortex-A76)

struct V3Sub {
    const char* name = "";
    std::size_t bytes_cmp = 0;      // 비교한 바이트 수 (누적)
    std::size_t diff = 0;           // 기준과 다른 바이트 수
    std::size_t a5 = 0;             // 그중 0xA5 (① ② 의 잔존)
    std::size_t stale = 0;          // 그중 **다른 프레임의 값** (③ 의 낡은 데이터)
    int gpu_mismatch = 0;           // GPU 바이트 합이 기준과 다른 (trial x 텐서) 수
    int gpu_cmp = 0;
    std::size_t detectable = 0;     // ③ 에서 A 와 B 가 실제로 다른 바이트 수 = 검출 가능 예산
};

int run_v3(const Options& o, Tee& log) {
    FixtureSet fs = load_fixtures(o.fixture_dir);
    const YoloTensorLayout L = layout_from_fixtures(fs);
    log("[v3] 실패 강제 하네스 3변종: ① full-dirty ② line-dirty ③ stale-read(읽기 방향)");
    log(fmt("[pcie] %s", pcie_link_speed().c_str()));
    if (const std::string holder = npu_holder(); !holder.empty())
        throw std::runtime_error("NPU 가 이미 다른 프로세스에 잡혀 있다: pid " + holder +
                                 " — Hailo-8 은 1개뿐이라 독점해야 한다");
    cool_down(log);
    const int err0 = count_hailort_log_errors();

    const std::size_t in_sz = static_cast<std::size_t>(fs.lb.dst_w) * fs.lb.dst_h * 3;
    const auto inputs = make_nhcw_inputs(o.video, std::max(2, o.live_frames), fs.lb, in_sz);
    const std::vector<std::uint8_t>& frame_a = inputs.front();
    const std::vector<std::uint8_t>& frame_b = inputs.back();
    log(fmt("[v3] 프레임 A/B 준비 (영상 앞 %zu장 중 첫/마지막). 셀당 변종별 %d회 반복, 창 %d B, 라인 %zu B",
            inputs.size(), o.repeats, o.v3_window, kCacheLine));

    std::vector<std::pair<bool, std::array<V3Sub, 3>>> cells;
    std::uint64_t vk_errors = 0;
    for (const bool heap_sync : heap_sync_cells(o.heap_sync)) {
        std::array<V3Sub, 3> sub{};
        sub[0].name = "full-dirty";
        sub[1].name = "line-dirty";
        sub[2].name = "stale-read";
        log("");
        log(fmt("======== heap_sync %s ========", heap_sync ? "ON (규약대로 START/END|READ)" : "OFF (sync 없음)"));

        yvo::vk::VulkanContextOptions co;
        co.enable_validation = o.validation;
        co.validation_sync = o.sync_validation;
        co.verbose = o.verbose;
        yvo::vk::VulkanContext ctx(co);

        YoloPostprocConfig pc;
        pc.num_slots = 1;                                   // **같은 버퍼를 강제로 재사용한다**
        pc.max_dets = o.max_dets;
        pc.local_size_x = o.lx;
        pc.local_size_y = o.ly;
        pc.score_thr = kHefScoreThreshold;
        pc.input_alloc = YoloInputAlloc::ImportHeap;        // V3 는 heap 전용이다
        pc.input_sync = YoloInputSync::Foreign;             // 배리어는 항상 켠다(축이 아니다)
        pc.heap_sync = heap_sync;
        pc.checksum_pipeline = true;
        pc.verbose = o.verbose;
        YoloPostproc pp(ctx, L, pc);
        LiveSession sess(o.hef, pp, stream_names_by_tensor(fs), o.api == "infer_async", o.batch, o.verbose);

        // 한 프레임을 받아 (전체 스냅샷 + GPU 바이트 합) 기준을 뜬다. sync 는 **강제**다.
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
        log(fmt("[기준] 프레임 A 스냅샷의 0xA5 바이트 %zu개(①② 잡음 하한) · A≠B 바이트 전체 %zu / 창 %zu "
                "(③ 검출 가능 예산, 창 합계 %zu B)", a5_noise, ab_diff_full, ab_diff_win, win_total));
        if (ab_diff_win == 0)
            log("  ⚠ 창 안에서 A 와 B 가 같다 — ③ 은 이 창으로 아무것도 검출할 수 없다. --v3-window 를 키울 것");

        std::vector<std::uint8_t> buf;
        // 셀 설정대로 읽어 기준과 비교한다. cmp_bytes 만큼만 본다.
        auto read_and_compare = [&](V3Sub& st, const std::array<std::vector<std::uint8_t>, kYoloNumTensors>& ref,
                                    const std::array<std::vector<std::uint8_t>, kYoloNumTensors>* other,
                                    std::size_t cmp_bytes) {
            std::size_t d = 0, a5 = 0, stale = 0;
            const RawRead rr(pp, 0);   // heap_sync=off 셀이면 no-op 이다 — 그것이 이 셀의 요점이다
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
        log("---- ① full-dirty: 전체 memset(0xA5), END|WRITE 생략 ----");
        log("trial  gpu_mismatch  diff_bytes  0xA5_잔존");
        for (int trial = 0; trial < o.repeats; ++trial) {
            forced_sync(pp, 0, yvo::DmaSyncDir::Write, /*start=*/true);
            for (int t = 0; t < kYoloNumTensors; ++t)
                std::memset(pp.raw_host_ptr(0, t), kDirtyByte, pp.raw_size(t));
            // (END|WRITE 없음 — 이것이 이 변종의 핵심이다)
            sess.run_frame(0, frame_a, o.timeout_ms);
            const int gm = gpu_compare(sub[0], sumA);
            const std::size_t before = sub[0].a5;
            const std::size_t d = read_and_compare(sub[0], refA, nullptr, SIZE_MAX);
            log(fmt("%5d  %12d  %10zu  %9zu", trial, gm, d, sub[0].a5 - before));
            forced_sync(pp, 0, yvo::DmaSyncDir::Write, /*start=*/false);   // ioctl 짝을 맞춘다
        }

        // ── ② line-dirty ─────────────────────────────────────────────────────────────────────
        log("");
        log(fmt("---- ② line-dirty: 창 %zu B 안에서 %zu B 라인마다 1바이트만 쓴다(부분 라인 → dirty 강제), "
                "END|WRITE 생략 ----", win, kCacheLine));
        log("trial  gpu_mismatch  diff_bytes(창)  0xA5_잔존");
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

        // ── ③ stale-read (읽기 방향) ─────────────────────────────────────────────────────────
        log("");
        log(fmt("---- ③ stale-read: 프레임 B 로 창 %zu B 를 읽어 캐시를 채운 뒤 DMA 가 프레임 A 를 쓴다 ----",
                win));
        log("trial  gpu_mismatch  diff_bytes(창)  그중_B값(낡은_데이터)");
        std::vector<std::uint8_t> prime;
        for (int trial = 0; trial < o.repeats; ++trial) {
            // (a) 프레임 B 를 받고, 창을 여러 번 읽어 **깨끗한 라인**으로 캐시를 채운다(강제 sync).
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
            // (b) DMA 가 **다른** 프레임 A 를 같은 버퍼에 쓴다
            sess.run_frame(0, frame_a, o.timeout_ms);
            // (c) 셀 설정대로 읽는다. off 셀이면 START|READ 가 없다 → 낡은 라인이 보이면 B 값이 나온다
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
    log("== V3 판정 ==");
    log("heap_sync  변종         비교바이트     diff  0xA5잔존  낡은B값  gpu_mismatch/비교  검출예산");
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
    log(fmt("[V3] heap_sync ON 에서 오염 0 (3변종 전부): %s", on_clean ? "PASS" : "**FAIL**"));
    log(fmt("[V3] heap_sync OFF 에서 오염이 재현됐는가: %s", off_dirty ? "예 — sync 필요가 실증됐다" : "아니오"));
    if (!off_dirty) {
        log("     ⚠ 세 변종 모두 재현하지 못했다. **그래도 sync 는 그대로 유지한다.**");
        log("       디바이스 트리에 `dma-coherent` 가 0건이라 커널이 V3D 를 비일관 장치로 취급하는 것이");
        log("       '필요하다' 쪽의 구조적 근거이고, 이 실험은 그것을 반증하지 못한 것뿐이다(부록 A P3-④).");
        log("       가능한 기제(미확정): ⓐ system heap 의 begin/end_cpu_access 가 attach 된 장치를 순회하며");
        log("       이미 필요한 캐시 관리를 하고 있다, ⓑ PCIe DMA 가 CPU 캐시를 스누핑한다(ACP/IO-coherent),");
        log("       ⓒ 추론 지연(ms) 동안 캐시가 자연 축출돼 조건 자체가 성립하지 않는다.");
        log("       ⓒ 를 줄이려고 ②③ 은 창을 작게(부분 라인/priming) 잡았으나 그래도 재현되지 않았다.");
    }
    log(fmt("[hailort.log] error +%d", err1 - err0));
    const bool pass = on_clean && vk_errors == 0 && (err1 - err0) == 0;
    log(fmt("[RESULT] v3 %s", pass ? "PASS" : "FAIL"));
    return pass ? 0 : 1;
}

#else   // !defined(YVO_HAVE_HAILO)

int run_live(const Options&, Tee& log) {
    log("[live] HailoRT 없이 빌드됐다 — live 모드를 쓸 수 없다");
    return 2;
}
int run_stale(const Options&, Tee& log) {
    log("[stale] HailoRT 없이 빌드됐다 — stale 모드를 쓸 수 없다");
    return 2;
}
int run_v3(const Options&, Tee& log) {
    log("[v3] HailoRT 없이 빌드됐다 — v3 모드를 쓸 수 없다");
    return 2;
}

#endif

// ───────────────────────────────────────────── heapimport (부록 B 1단계 = P0 관문, S1)
//
// **NPU 를 쓰지 않는다.** 6텐서 각각에 대해 /dev/dma_heap 에서 align_up(hw_frame_size, 16384) 를 잡고
// `DmaBufImportCache::get()` 으로 Vulkan 에 import 한 뒤, 판정에 필요한 값을 전부 기록한다.
//
// 기록하는 것 (부록 B.13 1단계 게이트 + B.12 설계 근거):
//   ① VkMemoryFdPropertiesKHR::memoryTypeBits      — **0 이면 그 자리에서 끝이다(S1)**
//   ② VkMemoryRequirements2::size vs dma-buf 크기  — V3DV 여유(+256 B)의 실측. B.12 headroom 이 여기서 나온다
//   ③ VkMemoryDedicatedRequirements 두 플래그      — prefers=true 면 과할당이 spec 위반이다
//                                                    (VUID-VkMemoryDedicatedAllocateInfo-buffer-02965)
//   ④ ImportedDmaBuf::range vs size                — range < size 면 -256 B 축소가 발동한 것이다
//   ⑤ vkAllocateMemory 의 VkResult (실패 시 dmesg)
//   ⑥ mmap 포인터의 16384 정렬 (HailoRT 요구, 부록 A.4) — **P0 는 판정이 목적이라 throw 하지 않고 기록만 한다**
//   ⑦ 왕복: CPU 가 mmap 으로 쓴 패턴의 바이트 합 == GPU(heap_sum.comp)가 계산한 합
//
// ── 2·3단계 이후 (부록 B.13) ────────────────────────────────────────────────────────────────
// 할당·매핑·sync 는 전부 **`common/dma_heap_alloc.hpp` 의 `yvo::DmaHeapPool`** 이 한다(2단계 일원화).
// 프로브 안에 있던 인라인 구현(`HeapAlloc`/`HeapBuf`/`dma_sync`)은 없앴다.
// 할당 크기는 **`align_up(hw + device_headroom, 16384)`** 다(3단계 B.12) — `device_headroom` 은
// `vk::StorageHeadroomQuery` 가 질의로 구한다. **상수 256 을 박지 않는다.**
// import 에는 `min_range = hw` 를 넘겨 `range < hw` 면 예외가 나게 한다(B.12 강제 검증).

constexpr std::size_t kHeapAlign = yvo::vk::kHailoAsyncAlign;   // 16384
constexpr std::uint32_t kSumGroups = 64;                        // heap_sum.comp 디스패치 워크그룹 수(고정)
constexpr std::uint32_t kPatternMul = 2654435761u;              // Knuth. 워드 i 의 값 = i * kPatternMul

std::size_t align_up_sz(std::size_t v, std::size_t a) { return a <= 1 ? v : (v + a - 1) / a * a; }

std::string run_capture(const char* cmd) {
    std::string out;
    FILE* p = ::popen(cmd, "r");
    if (!p) return "(popen 실패)";
    char buf[512];
    while (std::fgets(buf, sizeof(buf), p)) out += buf;
    ::pclose(p);
    return out.empty() ? std::string("(출력 없음)") : out;
}

// DMA_BUF_IOCTL_SYNC 를 non-strict 로 불러 errno 를 기록만 한다(프로브는 판정이 목적이다).
// 실제 구현은 `yvo::dma_sync_*`(common/dma_heap_alloc.hpp)다 — 부록 B.10 R5.
int dma_sync_note(int fd, yvo::DmaSyncDir dir, bool start) {
    const std::uint64_t flags =
        (start ? DMA_BUF_SYNC_START : DMA_BUF_SYNC_END) | yvo::dma_sync_dir_bits(dir);
    return yvo::detail::dma_sync_raw(fd, flags);
}

// heap_sum.comp 실행기. 파이프라인·디스크립터·커맨드버퍼를 한 번 만들고 6텐서에 재사용한다.
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

        // 결과 버퍼 {sum, words} 8 B. V3D 는 메모리 타입이 하나뿐이라 HOST_VISIBLE|HOST_COHERENT 다.
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
        if (type == UINT32_MAX) throw std::runtime_error("HeapSumRig: HOST_VISIBLE|HOST_COHERENT 메모리 타입 없음");
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

    // src 의 앞 n_words 워드를 바이트 단위로 더한 값. groups 에는 실제로 돈 워크그룹 수가 들어간다.
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

        // ① 외부(CPU 가 dma-buf 매핑으로 쓴 메모리) → 컴퓨트 큐 acquire + 결과 버퍼 호스트 쓰기 가용화.
        //    부록 B.10: sync ioctl 은 **CPU 캐시**만, Vulkan 배리어는 **V3D 캐시**만 다룬다. 둘 다 필요하다.
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

        // ② 입력 release(외부로 반납) + 결과 셰이더 쓰기 → 호스트 읽기 가시화
        std::array<VkBufferMemoryBarrier2, 2> rel{};
        rel[0] = acq[0];
        rel[0].srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        rel[0].srcAccessMask = VK_ACCESS_2_NONE;   // 읽기만 했으므로 가용화할 쓰기 없음
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

// 한 텐서의 P0 기록
struct HeapImportRow {
    std::string name;
    std::size_t hw = 0, want = 0, dmabuf = 0;
    std::size_t headroom = 0;   // B.12: 이 텐서 크기에 대해 드라이버가 요구한 여유 (질의값)
    bool mmap_aligned = false;
    std::uint32_t fd_type_bits = 0;
    // dma-buf 크기로 만든 더미 VkBuffer 의 요구 (B.12 의 2번 단계)
    VkDeviceSize req_size_dmabuf = 0, req_align = 0;
    std::uint32_t req_type_bits = 0;
    bool requires_ded = false, prefers_ded = false;
    // hw_frame_size 로 만든 더미의 요구 → headroom = req - hw
    VkDeviceSize req_size_hw = 0;
    // import 결과
    bool imported = false;
    VkDeviceSize imp_size = 0, imp_range = 0;
    std::string import_error;
    // 왕복
    bool roundtrip_ok = false;
    std::uint32_t gpu_sum = 0, cpu_sum = 0, groups = 0;
    std::uint32_t n_words = 0;
    int sync_err_begin = 0, sync_err_end = 0;
};

// requested 바이트짜리 외부 메모리용 더미 VkBuffer 의 메모리 요구를 묻는다 (할당하지 않는다).
// 구현은 vulkan/vk_storage_headroom.hpp 로 옮겼다(부록 B.12 — 다른 호출자도 쓴다).
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

    // fd 누수 판정 기준선: VulkanContext 가 연 fd 는 세지 않는다(컨텍스트는 이 블록보다 오래 산다).
    const long fd_base = count_open_fds();

    std::vector<HeapImportRow> rows;
    bool hard_stop = false;
    std::uint64_t imports = 0, hits = 0;
    // heap·import·파이프라인을 안쪽 스코프에 두어, 스코프를 벗어난 뒤 fd 가 기준선으로 돌아오는지 본다.
    {
    // ── 3단계(B.12): 헤드룸을 **질의**로 구한다. 상수를 박지 않는다. ────────────────────────
    // DmaHeapConfig::device_headroom 은 풀 하나에 값 하나라 6텐서 중 최대값을 쓴다
    // (V3D 는 크기와 무관하게 256 이라 6개 모두 같다. 과할당은 안전하다 — requires/prefers = 0).
    yvo::vk::StorageHeadroomQuery hq(ctx.device());
    std::vector<std::size_t> tensor_sizes;
    tensor_sizes.reserve(fs.streams.size());
    for (const FixtureStream& t : fs.streams) tensor_sizes.push_back(t.bytes);
    const std::size_t queried = hq.max_headroom(tensor_sizes.begin(), tensor_sizes.end());
    const std::size_t headroom = o.no_headroom ? 0 : queried;
    log(fmt("[headroom] 질의 %zu회 → 질의값 %zu B (req.alignment = %llu B)%s. 할당 크기 = align_up(hw + %zu, %zu)",
            hq.queries(), queried, static_cast<unsigned long long>(hq.last_alignment()),
            o.no_headroom ? "  **--no-headroom: 0 으로 강제(음성 대조)**" : "", headroom, kHeapAlign));

    yvo::DmaHeapConfig hc;
    if (!o.heap.empty()) hc.heaps = {o.heap};
    hc.align = kHeapAlign;
    hc.device_headroom = headroom;
    hc.map = true;
    hc.strict_sync = true;
    hc.zero_init = true;
    hc.name = "yvo_p0_heapimport";
    yvo::DmaHeapPool heap(hc);
    log(fmt("[heap] %s (yvo::DmaHeapPool — 부록 B.11 일원화 API)", heap.heap().c_str()));

    HeapSumRig rig(ctx);
    yvo::vk::DmaBufImportCache cache(ctx, 16);

    for (const FixtureStream& t : fs.streams) {
        HeapImportRow r;
        r.name = t.file;
        r.hw = t.bytes;
        r.headroom = headroom;   // 실제 적용값(--no-headroom 이면 0). 질의값은 [headroom] 줄에 있다
        r.want = align_up_sz(t.bytes + headroom, kHeapAlign);
        const int hi = heap.allocate(t.bytes);
        const yvo::DmaHeapBuf hb = heap.at(hi);   // POD 복사 — 다음 allocate() 가 벡터를 재할당해도 안전
        r.dmabuf = hb.dmabuf_size;
        r.mmap_aligned = reinterpret_cast<std::uintptr_t>(hb.ptr) % kHeapAlign == 0;

        // ① 이 fd 를 import 할 수 있는 메모리 타입 — **0 이면 S1 하드 스톱**
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
            r.import_error = "memoryTypeBits == 0 (V3DV 가 이 dma-heap fd 를 import 하지 못한다)";
            hard_stop = true;
            rows.push_back(r);
            continue;
        }

        // ②·③ 더미 버퍼로 크기 요구와 dedicated 플래그 (B.12 의 1~2번 단계)
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

        // ④·⑤ 실제 import — **기존 DmaBufImportCache::get() 을 그대로 쓴다**
        const yvo::vk::ImportedDmaBuf* imp = nullptr;
        try {
            // min_range = hw → range < hw 면 예외다(B.12 강제 검증)
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

        // ⑦ 왕복: CPU 가 쓴 패턴을 GPU 가 같은 값으로 읽는가
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
    cache.clear();   // import 한 dup fd 를 전부 닫는다(vkFreeMemory 가 같이 닫는다)
    }
    const long fd_after = count_open_fds();

    // ── 보고 ─────────────────────────────────────────────────────────────────────────────────
    log("");
    log("== 텐서별 기록 ==");
    for (const HeapImportRow& r : rows) {
        log(fmt("[%s] hw=%zu headroom=%zu want=align_up(hw+hr,%zu)=%zu dmabuf=%zu  mmap 16384정렬=%s",
                r.name.c_str(), r.hw, r.headroom, kHeapAlign, r.want, r.dmabuf,
                r.mmap_aligned ? "예" : "**아니오**"));
        log(fmt("    (1) VkMemoryFdPropertiesKHR::memoryTypeBits = 0x%08x", r.fd_type_bits));
        if (!r.import_error.empty()) {
            log(fmt("    !! 실패: %s", r.import_error.c_str()));
            continue;
        }
        log(fmt("    (2) req(size=dmabuf %zu) = %llu  → 여유 %+lld B | align=%llu typeBits=0x%08x", r.dmabuf,
                static_cast<unsigned long long>(r.req_size_dmabuf),
                static_cast<long long>(r.req_size_dmabuf) - static_cast<long long>(r.dmabuf),
                static_cast<unsigned long long>(r.req_align), r.req_type_bits));
        log(fmt("        req(size=hw %zu) = %llu  → **headroom %+lld B** (B.12 가 캐시할 값)", r.hw,
                static_cast<unsigned long long>(r.req_size_hw),
                static_cast<long long>(r.req_size_hw) - static_cast<long long>(r.hw)));
        log(fmt("    (3) requiresDedicatedAllocation=%d prefersDedicatedAllocation=%d", r.requires_ded ? 1 : 0,
                r.prefers_ded ? 1 : 0));
        const bool shrunk = r.imp_range < r.imp_size;
        log(fmt("    (4) ImportedDmaBuf: size=%llu range=%llu  %s",
                static_cast<unsigned long long>(r.imp_size), static_cast<unsigned long long>(r.imp_range),
                shrunk ? "***** range < size — 축소가 발동했다 *****" : "range == size (축소 없음)"));
        if (shrunk)
            log(fmt("        ⇒ 축소 %lld B. hw 대비 %s (range %llu vs hw %zu, 여유 %+lld B)",
                    static_cast<long long>(r.imp_size) - static_cast<long long>(r.imp_range),
                    r.imp_range >= r.hw ? "아직 여유 안" : "***** hw 를 잘라먹는다 *****",
                    static_cast<unsigned long long>(r.imp_range), r.hw,
                    static_cast<long long>(r.imp_range) - static_cast<long long>(r.hw)));
        log(fmt("    (5) vkAllocateMemory = VK_SUCCESS"));
        log(fmt("    (6) DMA_BUF_IOCTL_SYNC START|WRITE=%d END|WRITE=%d (0 = 성공)", r.sync_err_begin,
                r.sync_err_end));
        log(fmt("    (7) 왕복: n_words=%u groups=%u  gpu=0x%08x cpu=0x%08x  %s", r.n_words, r.groups, r.gpu_sum,
                r.cpu_sum, r.roundtrip_ok ? "일치" : "***** 불일치 *****"));
    }

    // ── 판정 (부록 B.14 S1) ──────────────────────────────────────────────────────────────────
    int n_imported = 0, n_shrunk = 0, n_roundtrip = 0, n_unaligned = 0, n_range_ok = 0;
    for (const HeapImportRow& r : rows) {
        if (r.imported) ++n_imported;
        if (r.imported && r.imp_range < r.imp_size) ++n_shrunk;
        if (r.imported && r.imp_range >= r.hw) ++n_range_ok;   // **3단계 게이트**
        if (r.roundtrip_ok) ++n_roundtrip;
        if (!r.mmap_aligned) ++n_unaligned;
    }
    const std::uint64_t vk_errors = ctx.validation_errors();
    log("");
    log(fmt("[요약] import %d/%zu · **range >= hw %d/%zu** · 왕복 일치 %d/%zu · range<size %d · "
            "mmap 미정렬 %d · 검증레이어 errors %llu",
            n_imported, rows.size(), n_range_ok, rows.size(), n_roundtrip, rows.size(), n_shrunk, n_unaligned,
            static_cast<unsigned long long>(vk_errors)));

    std::string verdict;
    int rc = 0;
    if (hard_stop || n_imported != static_cast<int>(rows.size())) {
        verdict = o.no_headroom ? "불합격 — **--no-headroom 음성 대조**. 헤드룸 없이는 range < hw 로 터진다(기대한 결과)"
                                : "불합격(하드 스톱) — S1. 버퍼 전환 전면 중단";
        rc = 1;
        log("");
        log("== dmesg (v3d/dma) ==");
        log(run_capture("dmesg 2>&1 | grep -iE 'v3d|dma-?buf|dma_heap' | tail -n 30"));
    } else if (n_roundtrip != static_cast<int>(rows.size())) {
        verdict = "일관성 문제 — 왕복 불일치. sync 방향·시점 재시도 필요";
        rc = 1;
    } else if (n_range_ok != static_cast<int>(rows.size())) {
        // 3단계 게이트 실패 = 부록 B.14 S4. 헤드룸으로 해결하지 못한 것이다.
        verdict = "불합격 — range < hw 인 텐서가 있다(S4). 헤드룸이 부족하다";
        rc = 1;
    } else if (n_shrunk > 0) {
        // range < size 는 정상이다 — 헤드룸을 미리 더해 잡았으므로 축소가 hw 를 침범하지 않는다.
        verdict = "합격 — 6/6 range >= hw (헤드룸 적용), 왕복 일치. range < size 는 설계대로다";
    } else {
        verdict = "합격 — 전부 성공, range == size, 왕복 일치";
    }
    if (vk_errors != 0) {
        verdict += " / 검증 레이어 errors != 0";
        rc = 1;
    }
    log(fmt("[S1/S4 판정] %s", verdict.c_str()));
    log(fmt("[import cache] imports=%llu hits=%llu", static_cast<unsigned long long>(imports),
            static_cast<unsigned long long>(hits)));
    log(fmt("[fd] 진입 %ld · 컨텍스트 뒤 기준선 %ld · heap/import 스코프 종료 후 %ld → 누수 %ld", fd0, fd_base,
            fd_after, fd_after - fd_base));
    if (fd_after != fd_base) rc = 1;
    log(fmt("[RESULT] heapimport %s", rc == 0 ? "PASS" : "FAIL"));
    return rc;
}

} // namespace

int main(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* what) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(std::string("인자 누락: ") + what);
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
            else if (a == "--v3-window") o.v3_window = std::stoi(next("--v3-window"));
            else if (a == "--no-headroom") o.no_headroom = true;
            else if (a == "--heapimport") o.probe = "heapimport";
            else if (a == "--validation") o.validation = true;
            else if (a == "--sync-validation") { o.validation = true; o.sync_validation = true; }
            else if (a == "--verbose") o.verbose = true;
            else if (a == "--local") {
                const std::string v = next("--local");
                const auto comma = v.find(',');
                if (comma == std::string::npos) throw std::runtime_error("--local 은 X,Y 형식이다");
                o.lx = static_cast<std::uint32_t>(std::stoul(v.substr(0, comma)));
                o.ly = static_cast<std::uint32_t>(std::stoul(v.substr(comma + 1)));
            } else if (a == "-h" || a == "--help") {
                usage();
                return 0;
            } else {
                std::fprintf(stderr, "알 수 없는 인자: %s\n", a.c_str());
                usage();
                return 2;
            }
        } catch (const std::exception& e) {
            std::fprintf(stderr, "인자 오류: %s\n", e.what());
            return 2;
        }
    }
    // heapimport 는 부록 B(dmaheap_unify) 산출물이라 기본 출력 위치가 다르다.
    // 부록 B(dmaheap_unify) 산출물은 기본 출력 위치가 다르다.
    // ⚠ heapimport 의 기본 파일명은 1단계 증거(`p0_heapimport.txt`)를 덮어쓴다 — `--out-file` 을 쓸 것.
    if (o.probe == "heapimport" || o.probe == "gpuab" || o.probe == "v3") {
        if (!o.out_dir_set) o.out_dir = "07_results/dmaheap_unify/evidence";
        if (o.out_file.empty())
            o.out_file = o.probe == "heapimport" ? "p0_heapimport.txt" : ("stage_" + o.probe + ".txt");
    }
    if (o.out_file.empty()) o.out_file = "p3_" + o.probe + ".txt";
    const std::string out_path = o.out_dir.empty() ? std::string() : o.out_dir + "/" + o.out_file;

    try {
        Tee log(out_path);
        log(fmt("yvo_postproc_probe --probe %s   (%s)", o.probe.c_str(),
                o.probe == "heapimport" ? "계획서 부록 B 1단계 = P0 관문" : "계획서 §6.2 P3"));
        if (o.probe == "heapimport") return run_heapimport(o, log);
        if (o.probe == "gpuab") return run_gpuab(o, log);
        if (o.probe == "v3") return run_v3(o, log);
        if (o.probe == "fixture") return run_fixture(o, log);
        if (o.probe == "sweep") return run_sweep(o, log);
        if (o.probe == "live") return run_live(o, log);
        if (o.probe == "stale") return run_stale(o, log);
        log(fmt("알 수 없는 probe: %s", o.probe.c_str()));
        usage();
        return 2;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[fatal] %s\n", e.what());
        return 1;
    }
}
