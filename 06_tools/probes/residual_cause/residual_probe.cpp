// residual_probe: same per-frame flow as yvo_preproc_bench RefImpl/RefSplitImpl (decode, map, de-SAND, letterbox stage,
// NHCW transpose to cached dma-heap output), with the letterbox stage chosen by --variant. Diagnostic only.
//   --accuracy       every frame: memcmp of the variant's RGB buffer and NHCW output against the ref, untimed.
//   --counters 1|2   PMU group (user mode only) read at each stage mark; adds read() syscalls between stages.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>

#include "core/dma_heap_alloc.hpp"
#include "core/dmabuf_map.hpp"
#include "decode/ffmpeg_drm_decoder.hpp"
#include "ref/cpu_letterbox.hpp"
#include "ref/sand.hpp"
#include "toolkit/cpu_preproc.hpp"
#include "variants.hpp"

using Clock = std::chrono::steady_clock;
static double ms_between(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

// ── PMU group ──────────────────────────────────────────────────────────────────────────────────
struct Pmu {
    std::vector<int> fds;
    std::vector<std::string> names;
    void add(const char* name, std::uint32_t type, std::uint64_t config) {
        perf_event_attr a{};
        a.size = sizeof(a);
        a.type = type;
        a.config = config;
        a.exclude_kernel = 1;
        a.exclude_hv = 1;
        a.read_format = PERF_FORMAT_GROUP;
        a.disabled = fds.empty() ? 1 : 0;
        a.pinned = fds.empty() ? 1 : 0;
        const int fd = static_cast<int>(syscall(SYS_perf_event_open, &a, 0, -1, fds.empty() ? -1 : fds[0], 0));
        if (fd < 0) throw std::runtime_error(std::string("perf_event_open failed: ") + name + ": " + std::strerror(errno));
        fds.push_back(fd);
        names.emplace_back(name);
    }
    void open_set(int set) {
        add("cycles", PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES);
        if (set == 1) {
            add("inst_retired", PERF_TYPE_RAW, 0x08);
            add("br_retired", PERF_TYPE_RAW, 0x21);
            add("br_mis_pred_retired", PERF_TYPE_RAW, 0x22);
            add("inst_spec", PERF_TYPE_RAW, 0x1b);
            add("stall_frontend", PERF_TYPE_RAW, 0x23);
            add("stall_backend", PERF_TYPE_RAW, 0x24);
        } else {
            add("l1d_cache", PERF_TYPE_RAW, 0x04);
            add("l1d_cache_refill", PERF_TYPE_RAW, 0x03);
            add("l1i_cache_refill", PERF_TYPE_RAW, 0x01);
            add("l2d_cache", PERF_TYPE_RAW, 0x16);
            add("l2d_cache_refill", PERF_TYPE_RAW, 0x17);
            add("l1d_tlb_refill", PERF_TYPE_RAW, 0x05);
        }
        ioctl(fds[0], PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP);
        ioctl(fds[0], PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP);
    }
    void read_all(std::vector<std::uint64_t>& out) const {
        std::uint64_t buf[16];
        const ssize_t n = read(fds[0], buf, sizeof(buf));
        if (n < 8 || buf[0] != fds.size()) throw std::runtime_error("perf read failed");
        out.assign(buf + 1, buf + 1 + fds.size());
    }
};

// ── Same mapping and output ring as preproc_bench.cpp ──────────────────────────────────────────
struct InputView {
    const std::uint8_t* y = nullptr;
    const std::uint8_t* uv = nullptr;
};
class InputMapper {
public:
    InputView open(const yvo::DrmFrame& f) {
        const bool two = f.uv_fd >= 0 && f.uv_fd != f.fd;
        InputView v;
        v.y = cache_.map(f.fd, f.size);
        v.uv = two ? cache_.map(f.uv_fd, f.uv_size) : v.y;
        yvo::dma_sync_begin(f.fd, yvo::DmaSyncDir::Read);
        if (two) yvo::dma_sync_begin(f.uv_fd, yvo::DmaSyncDir::Read);
        return v;
    }
    void close(const yvo::DrmFrame& f) {
        const bool two = f.uv_fd >= 0 && f.uv_fd != f.fd;
        yvo::dma_sync_end(f.fd, yvo::DmaSyncDir::Read);
        if (two) yvo::dma_sync_end(f.uv_fd, yvo::DmaSyncDir::Read);
    }
    void clear() { cache_.clear(); }

private:
    yvo::tool::DmaBufMapCache cache_;
};

static double median(std::vector<double> v) {
    if (v.empty()) return 0;
    std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
    return v[v.size() / 2];
}

int main(int argc, char** argv) {
    std::string variant = "ref", video = YVO_DEFAULT_VIDEO;
    int frames = 600, warmup = 60, counters = 0;
    bool accuracy = false;
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        if (s == "--variant") variant = argv[++i];
        else if (s == "--frames") frames = std::atoi(argv[++i]);
        else if (s == "--warmup") warmup = std::atoi(argv[++i]);
        else if (s == "--counters") counters = std::atoi(argv[++i]);
        else if (s == "--accuracy") accuracy = true;
        else if (s == "--video") video = argv[++i];
        else {
            std::fprintf(stderr, "unknown arg %s\n", s.c_str());
            return 2;
        }
    }
    yvo::FfmpegDrmDecoder dec;
    yvo::FfmpegDrmDecoder::Options dopt;
    dopt.threads = 1;
    dopt.extra_hw_frames = std::max(2 * yvo::kNumSlots, 8 + 4);
    if (!dec.open(video, dopt)) {
        std::fprintf(stderr, "decoder open failed: %s\n", dec.error().c_str());
        return 1;
    }
    const yvo::LetterboxParams lb0 = yvo::make_letterbox(dec.width(), dec.height());
    const std::size_t out_size = static_cast<std::size_t>(lb0.dst_w) * lb0.dst_h * 3;
    yvo::DmaHeapConfig cfg;
    cfg.name = "residual_probe_out";
    yvo::DmaHeapPool pool(cfg);
    for (int i = 0; i < 4; ++i) pool.allocate(out_size);

    Pmu pmu;
    if (counters) pmu.open_set(counters);
    std::vector<std::uint64_t> c0, c1;

    InputMapper in;
    std::vector<std::uint8_t> y_, uv_, rgb_, rgb_ref, nhcw_var(out_size), nhcw_ref(out_size);
    std::vector<float> py_, pu_, pv_;
    std::vector<double> v_map, v_flat, v_sync, v_trn, v_total;
    std::map<std::string, std::vector<double>> v_stage;
    std::map<std::string, std::vector<double>> v_cnt;
    long identical_rgb = 0, identical_nhcw = 0, n_acc = 0;

    const long total = accuracy ? frames : frames + warmup;
    long n = 0;
    for (; n < total; ++n) {
        auto f = dec.next();
        if (!f) break;
        const int w = f->width, h = f->height;
        const std::size_t sw = static_cast<std::size_t>(w);
        const int slot = static_cast<int>(n % 4);
        y_.resize(sw * static_cast<std::size_t>(h));
        uv_.resize(sw * static_cast<std::size_t>(h) / 2);
        const yvo::LetterboxParams lb = yvo::make_letterbox(w, h);
        if (lb.pad_x != 0) throw std::runtime_error("needs pad_x == 0");
        rgb_.resize(out_size);
        const int new_w = static_cast<int>(w * lb.scale + 0.5f), new_h = static_cast<int>(h * lb.scale + 0.5f);
        const std::size_t np = static_cast<std::size_t>(new_w) * new_h;
        const bool uses_buf = variant == "split" || variant == "split_nohoist";
        if (uses_buf) {
            py_.resize(np);
            pu_.resize(np);
            pv_.resize(np);
        }
        const yvo::SandNv12Layout l = yvo::make_sand_layout(*f);

        const std::uint8_t* Y = y_.data();
        const std::uint8_t* UV = uv_.data();
        std::uint8_t* dst = rgb_.data();

        // Stage marks: name of the interval that ends at the mark.
        struct Mark {
            const char* name;
            Clock::time_point t;
        };
        Mark marks[6];
        int nm = 0;
        std::vector<std::vector<std::uint64_t>> cmarks;
        const auto mark = [&](const char* name) {
            marks[nm++] = {name, Clock::now()};
        };
        const auto mark_c = [&](const char* name) {
            mark(name);
            if (counters) {
                pmu.read_all(c1);
                cmarks.push_back(c1);
            }
        };

        const auto t0 = Clock::now();
        const InputView v = in.open(*f);
        const auto t1 = Clock::now();
        const bool ok = yvo::tool::ref_desand_rows(v.y, f->size, v.uv, f->uv_fd >= 0 && f->uv_fd != f->fd ? f->uv_size : f->size,
                                                   l, 0, h, y_.data(), uv_.data(), sw);
        const auto t2 = Clock::now();
        in.close(*f);
        if (!ok) throw std::runtime_error("desand failed");
        mark_c("start");
        if (variant == "ref") {
            yvo::tool::ref_letterbox_rows(Y, sw, UV, sw, w, h, lb, 0, lb.dst_h, dst);
            mark_c("all");
        } else if (variant == "ref3") { // unmodified ref code, three calls: same loop, boundaries timed
            yvo::tool::ref_letterbox_rows(Y, sw, UV, sw, w, h, lb, 0, lb.pad_y, dst);
            mark_c("pad_top");
            yvo::tool::ref_letterbox_rows(Y, sw, UV, sw, w, h, lb, lb.pad_y, lb.pad_y + new_h, dst);
            mark_c("content");
            yvo::tool::ref_letterbox_rows(Y, sw, UV, sw, w, h, lb, lb.pad_y + new_h, lb.dst_h, dst);
            mark_c("pad_bot");
        } else if (variant == "split" || variant == "split_nohoist") {
            if (variant == "split") yvo::tool::split_resize(Y, sw, UV, sw, w, h, lb, py_.data(), pu_.data(), pv_.data());
            else rc::split_nohoist_resize(Y, sw, UV, sw, w, h, lb, py_.data(), pu_.data(), pv_.data());
            mark_c("resize");
            yvo::tool::split_color(py_.data(), pu_.data(), pv_.data(), new_w, new_h, lb, dst);
            mark_c("color");
            yvo::tool::ref_letterbox_rows(Y, sw, UV, sw, w, h, lb, 0, lb.pad_y, dst);
            yvo::tool::ref_letterbox_rows(Y, sw, UV, sw, w, h, lb, lb.pad_y + new_h, lb.dst_h, dst);
            mark_c("pad");
        } else if (variant == "nobranch" || variant == "nobranch_nohoist" || variant == "rowbuf") {
            yvo::tool::ref_letterbox_rows(Y, sw, UV, sw, w, h, lb, 0, lb.pad_y, dst);
            mark_c("pad_top");
            if (variant == "nobranch") rc::nobranch_content(Y, sw, UV, sw, w, h, lb, dst);
            else if (variant == "nobranch_nohoist") rc::nobranch_nohoist_content(Y, sw, UV, sw, w, h, lb, dst);
            else rc::rowbuf_content(Y, sw, UV, sw, w, h, lb, dst);
            mark_c("content");
            yvo::tool::ref_letterbox_rows(Y, sw, UV, sw, w, h, lb, lb.pad_y + new_h, lb.dst_h, dst);
            mark_c("pad_bot");
        } else {
            void (*fn)(const std::uint8_t*, std::size_t, const std::uint8_t*, std::size_t, int, int,
                       const yvo::LetterboxParams&, std::uint8_t*) = nullptr;
            if (variant == "local_params") fn = rc::local_params;
            else if (variant == "hoist_y") fn = rc::hoist_y;
            else if (variant == "hoist_all") fn = rc::hoist_all;
            else if (variant == "local_hoist") fn = rc::local_hoist;
            else if (variant == "rowsplit") fn = rc::rowsplit;
            else if (variant == "ref_notrap") fn = rc::ref_notrap;
            else if (variant == "ref_copy0") fn = rc::ref_copy0;
            else if (variant == "ref_copy1") fn = rc::ref_copy1;
            else if (variant == "ref_copy2") fn = rc::ref_copy2;
            else if (variant == "ref_copy3") fn = rc::ref_copy3;
            else throw std::runtime_error("unknown variant " + variant);
            fn(Y, sw, UV, sw, w, h, lb, dst);
            mark_c("all");
        }
        const auto t4 = marks[nm - 1].t;
        const yvo::DmaHeapBuf& ob = pool.at(slot);
        yvo::dma_sync_begin(ob.fd, yvo::DmaSyncDir::Write);
        const auto t5 = Clock::now();
        yvo::tool::ref_nhcw_rows(rgb_.data(), lb.dst_w, 0, lb.dst_h, ob.ptr);
        const auto t6 = Clock::now();
        yvo::dma_sync_end(ob.fd, yvo::DmaSyncDir::Write);
        const auto t7 = Clock::now();

        if (accuracy) {
            ++n_acc;
            rgb_ref.resize(out_size);
            yvo::cpu_letterbox_nv12_bt709(Y, sw, UV, sw, w, h, lb, rgb_ref.data());
            identical_rgb += std::memcmp(rgb_ref.data(), rgb_.data(), out_size) == 0;
            yvo::rgb_packed_to_nhcw(rgb_ref.data(), lb.dst_w, lb.dst_h, nhcw_ref.data());
            yvo::dma_sync_begin(ob.fd, yvo::DmaSyncDir::Read);
            std::memcpy(nhcw_var.data(), ob.ptr, out_size);
            yvo::dma_sync_end(ob.fd, yvo::DmaSyncDir::Read);
            identical_nhcw += std::memcmp(nhcw_ref.data(), nhcw_var.data(), out_size) == 0;
            continue;
        }
        if (n < warmup) continue;
        v_map.push_back(ms_between(t0, t1));
        v_flat.push_back(ms_between(t1, t2));
        v_trn.push_back(ms_between(t5, t6));
        v_sync.push_back(ms_between(t4, t5) + ms_between(t6, t7));
        v_total.push_back(ms_between(marks[0].t, t4));
        for (int i = 1; i < nm; ++i) v_stage[marks[i].name].push_back(ms_between(marks[i - 1].t, marks[i].t));
        if (counters)
            for (int i = 1; i < nm; ++i)
                for (std::size_t k = 0; k < pmu.names.size(); ++k)
                    v_cnt[std::string(marks[i].name) + "." + pmu.names[k]].push_back(
                        static_cast<double>(cmarks[static_cast<std::size_t>(i)][k] - cmarks[static_cast<std::size_t>(i) - 1][k]));
    }
    in.clear();
    if (!dec.error().empty()) {
        std::fprintf(stderr, "decode error: %s\n", dec.error().c_str());
        return 1;
    }
    if (accuracy) {
        std::printf("[ACCURACY] variant=%s frames=%ld identical_rgb=%ld identical_nhcw=%ld\n", variant.c_str(), n_acc,
                    identical_rgb, identical_nhcw);
        return identical_rgb == n_acc && identical_nhcw == n_acc ? 0 : 3;
    }
    std::printf("[RESULT] variant=%s frames=%zu lb_total=%.4f flatten=%.4f transpose=%.4f sync=%.4f map=%.4f", variant.c_str(),
                v_total.size(), median(v_total), median(v_flat), median(v_trn), median(v_sync), median(v_map));
    for (auto& [k, val] : v_stage) std::printf(" %s=%.4f", k.c_str(), median(val));
    std::printf("\n");
    if (counters)
        for (auto& [k, val] : v_cnt) {
            double sum = 0;
            for (double d : val) sum += d;
            std::printf("[COUNT] variant=%s set=%d %s mean_per_frame=%.0f median=%.0f\n", variant.c_str(), counters, k.c_str(),
                        sum / static_cast<double>(val.size()), median(val));
        }
    return 0;
}
