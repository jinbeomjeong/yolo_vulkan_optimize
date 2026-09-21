// ongoing_callbacks semantics measurer (one-off experiment, outside the project tree).
//   · runs infer_async with host malloc buffers and measures the per-frame in-flight ceiling.
//   · --manual 0/1 toggles whether increase/decrease pairs are used.
//   · --threshold k changes k in wait_for_ongoing_callbacks_count_under(k).
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <vector>

#include <hailo/hailort.hpp>

using namespace hailort;

static void die(const char* where, hailo_status st) {
    std::fprintf(stderr, "FAIL %s status=%d\n", where, (int)st);
    std::exit(1);
}

int main(int argc, char** argv) {
    std::string hef_path = "03_models/yolov11m.hef";
    int manual = 1, threshold = 4, slots = 4, frames = 200, batch = 2;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { return std::string(argv[++i]); };
        if (a == "--hef") hef_path = next();
        else if (a == "--manual") manual = std::stoi(next());
        else if (a == "--threshold") threshold = std::stoi(next());
        else if (a == "--slots") slots = std::stoi(next());
        else if (a == "--frames") frames = std::stoi(next());
        else if (a == "--batch") batch = std::stoi(next());
    }

    auto hef = Hef::create(hef_path);
    if (!hef) die("Hef::create", hef.status());

    hailo_vdevice_params_t vp;
    hailo_init_vdevice_params(&vp);
    vp.scheduling_algorithm = HAILO_SCHEDULING_ALGORITHM_ROUND_ROBIN;
    auto vdev = VDevice::create(vp);
    if (!vdev) die("VDevice::create", vdev.status());

    auto params = vdev.value()->create_configure_params(hef.value());
    if (!params) die("create_configure_params", params.status());
    for (auto& kv : params.value()) {
        kv.second.batch_size = (uint16_t)batch;
        for (auto& sp : kv.second.stream_params_by_name) sp.second.flags = HAILO_STREAM_FLAGS_ASYNC;
    }
    auto ngs = vdev.value()->configure(hef.value(), params.value());
    if (!ngs) die("configure", ngs.status());
    auto ng = ngs.value()[0];

    auto in_refs = ng->get_input_streams();
    auto out_refs = ng->get_output_streams();
    InputStream& in = in_refs[0].get();
    std::vector<OutputStream*> outs;
    for (auto& r : out_refs) outs.push_back(&r.get());
    std::sort(outs.begin(), outs.end(), [](const OutputStream* a, const OutputStream* b) { return a->name() < b->name(); });

    const std::size_t in_size = in.get_frame_size();
    std::vector<std::size_t> out_size;
    for (auto* o : outs) out_size.push_back(o->get_frame_size());

    // per-slot host buffers (16 KB aligned — avoids unaligned warning)
    struct SlotBuf { void* in; std::vector<void*> out; };
    std::vector<SlotBuf> bufs((std::size_t)slots);
    auto ali = [](std::size_t n) {
        std::size_t r = (n + 16383) / 16384 * 16384;
        void* p = std::aligned_alloc(16384, r);
        std::memset(p, 0, r);
        return p;
    };
    for (auto& b : bufs) {
        b.in = ali(in_size);
        for (std::size_t i = 0; i < outs.size(); ++i) b.out.push_back(ali(out_size[i]));
    }

    std::mutex m;
    std::condition_variable cv;
    std::deque<int> freeq;
    for (int s = 0; s < slots; ++s) freeq.push_back(s);
    std::atomic<int> inflight{0}, peak{0}, done{0};
    std::vector<std::atomic<int>> remaining((std::size_t)slots);
    for (auto& r : remaining) r.store(0);

    auto submit = [&](int slot, bool count_time) {
        (void)count_time;
        SlotBuf& b = bufs[(std::size_t)slot];
        NamedBuffersCallbacks named;
        auto on_buf = [&, slot](hailo_status) {
            if (remaining[(std::size_t)slot].fetch_sub(1) == 1) {
                inflight.fetch_sub(1);
                {
                    std::lock_guard<std::mutex> lk(m);
                    freeq.push_back(slot);
                }
                cv.notify_one();
                done.fetch_add(1);
            }
        };
        {
            BufferRepresentation br{};
            br.buffer_type = BufferType::VIEW;
            br.view = MemoryView(b.in, in_size);
            named.emplace(in.name(), std::make_pair(br, on_buf));
        }
        for (std::size_t i = 0; i < outs.size(); ++i) {
            BufferRepresentation br{};
            br.buffer_type = BufferType::VIEW;
            br.view = MemoryView(b.out[i], out_size[i]);
            named.emplace(outs[i]->name(), std::make_pair(br, on_buf));
        }
        remaining[(std::size_t)slot].store((int)outs.size() + 1);

        const auto st_wait = ng->wait_for_ongoing_callbacks_count_under((std::size_t)threshold);
        if (st_wait != HAILO_SUCCESS) die("wait_for_ongoing_callbacks_count_under", st_wait);
        const int now = inflight.fetch_add(1) + 1;
        int prev = peak.load();
        while (now > prev && !peak.compare_exchange_weak(prev, now)) {}
        if (manual) ng->increase_ongoing_callbacks();
        const auto st = ng->infer_async(named, [&](hailo_status) { if (manual) ng->decrease_ongoing_callbacks(); });
        if (st != HAILO_SUCCESS) die("infer_async", st);
    };

    auto run = [&](int n) {
        for (int i = 0; i < n; ++i) {
            int slot;
            {
                std::unique_lock<std::mutex> lk(m);
                if (!cv.wait_for(lk, std::chrono::seconds(10), [&] { return !freeq.empty(); })) {
                    std::fprintf(stderr, "FAIL: no free slot within 10 s\n");
                    std::exit(2);
                }
                slot = freeq.front();
                freeq.pop_front();
            }
            submit(slot, true);
        }
        // drain
        std::unique_lock<std::mutex> lk(m);
        if (!cv.wait_for(lk, std::chrono::seconds(10), [&] { return inflight.load() == 0; })) {
            std::fprintf(stderr, "FAIL: drain timeout exceeded inflight=%d\n", inflight.load());
            std::exit(3);
        }
    };

    run(std::min(20, frames));   // warm-up
    peak.store(0);
    done.store(0);
    const auto t0 = std::chrono::steady_clock::now();
    run(frames);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

    std::printf("RESULT manual=%d threshold=%d slots=%d batch=%d frames=%d peak_inflight=%d fps=%.2f\n",
                manual, threshold, slots, batch, frames, peak.load(), 1000.0 * frames / ms);
    return 0;
}
