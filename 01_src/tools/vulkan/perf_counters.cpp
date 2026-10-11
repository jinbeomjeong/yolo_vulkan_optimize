// yvo_perf_counters: enumerate VK_KHR_performance_query counters of the V3D GPU, analyse pass counts, and run
// small experiments on how the counters behave (granularity, contention, overhead).
//
// Modes (no NPU, no display, no decoder):
//   default           Enumeration + pass analysis + sanity check.
//     1. Picks the physical device whose name contains "V3D" (llvmpipe is also enumerated, so selection by name
//        and deviceType is mandatory). Checks the extension and the performanceCounterQueryPools feature.
//     2. For every queue family with VK_QUEUE_COMPUTE_BIT, lists all counters with
//        vkEnumeratePhysicalDeviceQueueFamilyPerformanceQueryCountersKHR (name, category, unit, scope, storage,
//        flags, uuid, description). Counter names are **only** taken from this runtime enumeration.
//     3. Pass analysis with vkGetPhysicalDeviceQueueFamilyPerformanceQueryPassesKHR:
//        (a) passes per single counter, (b) passes for all counters, (c) longest index-order prefix that fits in
//        1 pass, plus the size of a skip-greedy 1-pass set (add in index order, skip counters that break 1 pass).
//     4. Sanity check (--no-sanity to skip): performance query pool for a counter set, profiling lock, a
//        heap_sum.comp dispatch (02_shaders/diag) wrapped in the query, results read back.
//   --exp granularity E1: k dispatches in one command buffer vs k command buffers with one dispatch each.
//   --exp contention  E2: fixed job queried N times; run under different conditions (--label) and compare spread.
//   --exp overhead    E3: timestamp only vs timestamp + 32 counters vs timestamp + 4 counters (+ one 3-pass run).
//   --load <sec>      Background load: submits heap_sum dispatches back to back without any query.
//
// Command buffer layout used by every mode (counter set is 1 pass unless stated otherwise):
//   [ts0 TOP] [perf begin] { [compute→compute barrier] [dispatch] } × k [perf end] [compute→host barrier] [ts1 BOTTOM]
// The timestamps are outside the performance query, so the GPU interval includes the query's own cost (E3).
//
// Spec notes (Vulkan 1.4.309 registry, validusage.json):
//   - VUID-vkCmdBeginQuery-queryPool-03224 / vkCmdEndQuery-queryPool-03227: with a COMMAND_BUFFER-scope counter
//     the begin must be the first and the end the last recorded command. The layout above is only legal when no
//     selected counter has COMMAND_BUFFER scope, which is checked at runtime (V3DV reports COMMAND scope only).
//   - VUID-vkCmdBeginQuery-None-02863: no vkCmdResetQueryPool for the same query in the same command buffer
//     → queries are reset on the host with vkResetQueryPool (hostQueryReset feature).
//   - VUID-vkCmdBeginQuery-queryPool-03223 / vkQueueSubmit-pCommandBuffers-03220: the profiling lock is held
//     before vkBeginCommandBuffer and continuously until the last submission finished → held for the rig lifetime.
//   - VUID-vkGetQueryPoolResults-queryType-09440: no 64_BIT / AVAILABILITY / PARTIAL flags for performance queries.
//   - VUID-VkPerformanceQuerySubmitInfoKHR-counterPassIndex-03221: each pass is submitted with its pass index.
#include <algorithm>
#include <array>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <vulkan/vulkan.h>

#include "vulkan/vk_check.hpp"

#include "heap_sum_spv.h"
#include "toolkit/cli_parse.hpp"
#include "toolkit/stats.hpp"
#include "toolkit/tool_util.hpp"

namespace {

using yvo::tool::fmt;
using Clock = std::chrono::steady_clock;

struct Options {
    std::string csv_path;
    std::string device_substr = "V3D";
    bool sanity = true;
    std::vector<std::uint32_t> sanity_counters;  // empty → longest 1-pass prefix
    std::string exp;                             // granularity | contention | overhead
    std::string exp_csv;                         // long-format raw samples of --exp
    std::string label = "-";                     // condition label written to the output (E2: A/B/C)
    int iters = 0;                               // 0 → per-experiment default
    int warmup = 30;
    double load_sec = 0;
};

void print_help(const char* argv0) {
    std::printf(
        "Usage: %s [options]\n"
        "Enumerate VK_KHR_performance_query counters of the V3D GPU, analyse pass counts, run counter experiments.\n"
        "\n"
        "Enumeration (default mode):\n"
        "  --csv <path>              also write all counters as CSV (LF line endings)\n"
        "  --no-sanity               skip the query-pool / profiling-lock / dispatch sanity check\n"
        "  --sanity-counters <list>  comma-separated counter indices for the sanity check\n"
        "                            (default: longest index-order prefix that fits in 1 pass)\n"
        "Experiments (continuous submit/wait stream, heap_sum.comp 4 MiB per dispatch):\n"
        "  --exp granularity         E1: k=1,2,4,8 dispatches in one CB vs k CBs (default iters 100)\n"
        "  --exp contention          E2: fixed job, N queries, gpu_stats before/after (default iters 100)\n"
        "  --exp overhead            E3: ts only / ts+32 counters / ts+4 counters, + one 3-pass run (default 200)\n"
        "  --exp-csv <path>          long-format raw samples (exp,label,config,iter,metric,value)\n"
        "  --label <text>            condition label printed with the results (e.g. A, B, C)\n"
        "  --iters <n>               measured iterations per configuration\n"
        "  --warmup <n>              warm-up iterations per configuration (default 30)\n"
        "Load generator:\n"
        "  --load <sec>              submit heap_sum dispatches back to back for <sec> seconds, no queries\n"
        "Common:\n"
        "  --device <substr>         physical device name substring (default: V3D)\n"
        "  --help                    show this help\n",
        argv0);
}

std::vector<std::uint32_t> parse_index_list(const std::string& s) {
    std::vector<std::uint32_t> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (tok.empty()) continue;
        char* end = nullptr;
        const unsigned long v = std::strtoul(tok.c_str(), &end, 10);
        if (end == tok.c_str() || *end != '\0') throw std::runtime_error("invalid counter index: " + tok);
        out.push_back(static_cast<std::uint32_t>(v));
    }
    if (out.empty()) throw std::runtime_error("--sanity-counters: empty list");
    return out;
}

Options parse_args(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* name) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(std::string(name) + " requires a value");
            return argv[++i];
        };
        if (a == "--help" || a == "-h") {
            print_help(argv[0]);
            std::exit(0);
        } else if (a == "--csv") {
            o.csv_path = next("--csv");
        } else if (a == "--device") {
            o.device_substr = next("--device");
        } else if (a == "--no-sanity") {
            o.sanity = false;
        } else if (a == "--sanity-counters") {
            o.sanity_counters = parse_index_list(next("--sanity-counters"));
        } else if (a == "--exp") {
            o.exp = next("--exp");
            if (o.exp != "granularity" && o.exp != "contention" && o.exp != "overhead")
                throw std::runtime_error("--exp must be granularity, contention or overhead");
        } else if (a == "--exp-csv") {
            o.exp_csv = next("--exp-csv");
        } else if (a == "--label") {
            o.label = next("--label");
        } else if (a == "--iters") {
            o.iters = yvo::tool::arg_number<int>(argv[0], a, next("--iters").c_str());
            if (o.iters <= 0) throw std::runtime_error("--iters must be > 0");
        } else if (a == "--warmup") {
            o.warmup = yvo::tool::arg_number<int>(argv[0], a, next("--warmup").c_str());
            if (o.warmup < 0) throw std::runtime_error("--warmup must be >= 0");
        } else if (a == "--load") {
            o.load_sec = yvo::tool::arg_number<double>(argv[0], a, next("--load").c_str());
            if (o.load_sec <= 0) throw std::runtime_error("--load must be > 0");
        } else {
            throw std::runtime_error("unknown option: " + a + " (see --help)");
        }
    }
    if (!o.exp.empty() && o.load_sec > 0) throw std::runtime_error("--exp and --load are exclusive");
    return o;
}

const char* unit_str(VkPerformanceCounterUnitKHR u) {
    switch (u) {
    case VK_PERFORMANCE_COUNTER_UNIT_GENERIC_KHR: return "GENERIC";
    case VK_PERFORMANCE_COUNTER_UNIT_PERCENTAGE_KHR: return "PERCENTAGE";
    case VK_PERFORMANCE_COUNTER_UNIT_NANOSECONDS_KHR: return "NANOSECONDS";
    case VK_PERFORMANCE_COUNTER_UNIT_BYTES_KHR: return "BYTES";
    case VK_PERFORMANCE_COUNTER_UNIT_BYTES_PER_SECOND_KHR: return "BYTES_PER_SECOND";
    case VK_PERFORMANCE_COUNTER_UNIT_KELVIN_KHR: return "KELVIN";
    case VK_PERFORMANCE_COUNTER_UNIT_WATTS_KHR: return "WATTS";
    case VK_PERFORMANCE_COUNTER_UNIT_VOLTS_KHR: return "VOLTS";
    case VK_PERFORMANCE_COUNTER_UNIT_AMPS_KHR: return "AMPS";
    case VK_PERFORMANCE_COUNTER_UNIT_HERTZ_KHR: return "HERTZ";
    case VK_PERFORMANCE_COUNTER_UNIT_CYCLES_KHR: return "CYCLES";
    default: return "UNKNOWN_UNIT";
    }
}

const char* scope_str(VkPerformanceCounterScopeKHR s) {
    switch (s) {
    case VK_PERFORMANCE_COUNTER_SCOPE_COMMAND_BUFFER_KHR: return "COMMAND_BUFFER";
    case VK_PERFORMANCE_COUNTER_SCOPE_RENDER_PASS_KHR: return "RENDER_PASS";
    case VK_PERFORMANCE_COUNTER_SCOPE_COMMAND_KHR: return "COMMAND";
    default: return "UNKNOWN_SCOPE";
    }
}

const char* storage_str(VkPerformanceCounterStorageKHR s) {
    switch (s) {
    case VK_PERFORMANCE_COUNTER_STORAGE_INT32_KHR: return "INT32";
    case VK_PERFORMANCE_COUNTER_STORAGE_INT64_KHR: return "INT64";
    case VK_PERFORMANCE_COUNTER_STORAGE_UINT32_KHR: return "UINT32";
    case VK_PERFORMANCE_COUNTER_STORAGE_UINT64_KHR: return "UINT64";
    case VK_PERFORMANCE_COUNTER_STORAGE_FLOAT32_KHR: return "FLOAT32";
    case VK_PERFORMANCE_COUNTER_STORAGE_FLOAT64_KHR: return "FLOAT64";
    default: return "UNKNOWN_STORAGE";
    }
}

std::string flags_str(VkPerformanceCounterDescriptionFlagsKHR f) {
    std::string s;
    if (f & VK_PERFORMANCE_COUNTER_DESCRIPTION_PERFORMANCE_IMPACTING_BIT_KHR) s += "PERFORMANCE_IMPACTING";
    if (f & VK_PERFORMANCE_COUNTER_DESCRIPTION_CONCURRENTLY_IMPACTED_BIT_KHR) {
        if (!s.empty()) s += "|";
        s += "CONCURRENTLY_IMPACTED";
    }
    const VkPerformanceCounterDescriptionFlagsKHR known =
        VK_PERFORMANCE_COUNTER_DESCRIPTION_PERFORMANCE_IMPACTING_BIT_KHR |
        VK_PERFORMANCE_COUNTER_DESCRIPTION_CONCURRENTLY_IMPACTED_BIT_KHR;
    if (f & ~known) {
        if (!s.empty()) s += "|";
        s += fmt("0x%x", static_cast<unsigned>(f & ~known));
    }
    return s.empty() ? "none" : s;
}

std::string uuid_str(const std::uint8_t (&u)[VK_UUID_SIZE]) {
    std::string s;
    for (std::uint32_t i = 0; i < VK_UUID_SIZE; ++i) {
        if (i == 4 || i == 6 || i == 8 || i == 10) s += '-';
        s += fmt("%02x", u[i]);
    }
    return s;
}

std::string queue_flags_str(VkQueueFlags f) {
    std::string s;
    auto add = [&](VkQueueFlags bit, const char* name) {
        if (!(f & bit)) return;
        if (!s.empty()) s += "|";
        s += name;
    };
    add(VK_QUEUE_GRAPHICS_BIT, "GRAPHICS");
    add(VK_QUEUE_COMPUTE_BIT, "COMPUTE");
    add(VK_QUEUE_TRANSFER_BIT, "TRANSFER");
    add(VK_QUEUE_SPARSE_BINDING_BIT, "SPARSE");
    add(VK_QUEUE_PROTECTED_BIT, "PROTECTED");
    return s.empty() ? "none" : s;
}

// CSV field quoting (RFC 4180): quote when the field has a comma, quote or newline.
std::string csv_field(const std::string& v) {
    if (v.find_first_of(",\"\n\r") == std::string::npos) return v;
    std::string o = "\"";
    for (char c : v) {
        if (c == '"') o += '"';
        o += c;
    }
    return o + "\"";
}

std::string result_str(const VkPerformanceCounterResultKHR& r, VkPerformanceCounterStorageKHR st) {
    switch (st) {
    case VK_PERFORMANCE_COUNTER_STORAGE_INT32_KHR: return fmt("%" PRId32, r.int32);
    case VK_PERFORMANCE_COUNTER_STORAGE_INT64_KHR: return fmt("%" PRId64, r.int64);
    case VK_PERFORMANCE_COUNTER_STORAGE_UINT32_KHR: return fmt("%" PRIu32, r.uint32);
    case VK_PERFORMANCE_COUNTER_STORAGE_UINT64_KHR: return fmt("%" PRIu64, r.uint64);
    case VK_PERFORMANCE_COUNTER_STORAGE_FLOAT32_KHR: return fmt("%g", static_cast<double>(r.float32));
    case VK_PERFORMANCE_COUNTER_STORAGE_FLOAT64_KHR: return fmt("%g", r.float64);
    default: return "?";
    }
}

double result_double(const VkPerformanceCounterResultKHR& r, VkPerformanceCounterStorageKHR st) {
    switch (st) {
    case VK_PERFORMANCE_COUNTER_STORAGE_INT32_KHR: return r.int32;
    case VK_PERFORMANCE_COUNTER_STORAGE_INT64_KHR: return static_cast<double>(r.int64);
    case VK_PERFORMANCE_COUNTER_STORAGE_UINT32_KHR: return r.uint32;
    case VK_PERFORMANCE_COUNTER_STORAGE_UINT64_KHR: return static_cast<double>(r.uint64);
    case VK_PERFORMANCE_COUNTER_STORAGE_FLOAT32_KHR: return r.float32;
    case VK_PERFORMANCE_COUNTER_STORAGE_FLOAT64_KHR: return r.float64;
    default: return std::nan("");
    }
}

// Instance-level (physical-device) entry points of VK_KHR_performance_query.
struct PerfFns {
    PFN_vkEnumeratePhysicalDeviceQueueFamilyPerformanceQueryCountersKHR enumerate = nullptr;
    PFN_vkGetPhysicalDeviceQueueFamilyPerformanceQueryPassesKHR passes = nullptr;
};

struct Counter {
    VkPerformanceCounterKHR c{};
    VkPerformanceCounterDescriptionKHR d{};
};

std::vector<Counter> enumerate_counters(const PerfFns& fn, VkPhysicalDevice pd, std::uint32_t qf) {
    std::uint32_t n = 0;
    YVO_VK_CHECK(fn.enumerate(pd, qf, &n, nullptr, nullptr));
    std::vector<VkPerformanceCounterKHR> cs(n);
    std::vector<VkPerformanceCounterDescriptionKHR> ds(n);
    for (auto& c : cs) c.sType = VK_STRUCTURE_TYPE_PERFORMANCE_COUNTER_KHR;
    for (auto& d : ds) d.sType = VK_STRUCTURE_TYPE_PERFORMANCE_COUNTER_DESCRIPTION_KHR;
    YVO_VK_CHECK(fn.enumerate(pd, qf, &n, cs.data(), ds.data()));
    std::vector<Counter> out(n);
    for (std::uint32_t i = 0; i < n; ++i) out[i] = {cs[i], ds[i]};
    return out;
}

std::uint32_t num_passes(const PerfFns& fn, VkPhysicalDevice pd, std::uint32_t qf,
                         const std::vector<std::uint32_t>& idx) {
    VkQueryPoolPerformanceCreateInfoKHR ci{};
    ci.sType = VK_STRUCTURE_TYPE_QUERY_POOL_PERFORMANCE_CREATE_INFO_KHR;
    ci.queueFamilyIndex = qf;
    ci.counterIndexCount = static_cast<std::uint32_t>(idx.size());
    ci.pCounterIndices = idx.data();
    std::uint32_t p = 0;
    fn.passes(pd, &ci, &p);
    return p;
}

// /sys/class/drm/renderD128/device/gpu_stats: per V3D queue, cumulative job count and runtime [ns] of all clients.
struct QueueStat {
    std::uint64_t jobs = 0, runtime_ns = 0;
};
using GpuStats = std::map<std::string, QueueStat>;

GpuStats read_gpu_stats() {
    GpuStats s;
    std::ifstream f("/sys/class/drm/renderD128/device/gpu_stats");
    std::string line;
    while (std::getline(f, line)) {
        std::istringstream ls(line);
        std::string q;
        std::uint64_t ts = 0, jobs = 0, rt = 0;
        if (ls >> q >> ts >> jobs >> rt) s[q] = {jobs, rt};  // header line fails to parse and is skipped
    }
    return s;
}

void print_gpu_stats_delta(const char* title, const GpuStats& a, const GpuStats& b, double iters) {
    std::printf("  gpu_stats delta [%s] (all V3D clients; per_iter = delta / %.0f):\n", title, iters);
    std::printf("    %-12s %10s %14s %12s %14s\n", "queue", "jobs", "runtime_ms", "jobs/iter", "runtime_us/iter");
    for (const auto& [q, sb] : b) {
        const auto it = a.find(q);
        const QueueStat sa = it == a.end() ? QueueStat{} : it->second;
        const double dj = static_cast<double>(sb.jobs - sa.jobs);
        const double dr = static_cast<double>(sb.runtime_ns - sa.runtime_ns);
        std::printf("    %-12s %10.0f %14.3f %12.2f %14.2f\n", q.c_str(), dj, dr / 1e6, dj / iters, dr / 1e3 / iters);
    }
}

void print_gpu_stats_raw(const char* title, const GpuStats& s) {
    std::printf("  gpu_stats raw [%s]:", title);
    for (const auto& [q, v] : s) std::printf(" %s=%" PRIu64 "/%" PRIu64, q.c_str(), v.jobs, v.runtime_ns);
    std::printf("  (queue=jobs/runtime_ns)\n");
}

// ── Rig: device + heap_sum.comp pipeline + command buffers + timestamp / performance query pools ────────────
// Everything is created once; a counter set change recreates only the performance query pool.
class Rig {
public:
    static constexpr std::uint32_t kMaxWords = 4u * 1024u * 1024u;  // 16 MiB source buffer
    static constexpr std::uint32_t kGroups = 256;                    // heap_sum.comp workgroups (grid-stride)
    static constexpr std::uint32_t kMaxSlots = 8;                    // command buffers = query slots

    Rig(VkPhysicalDevice pd, std::uint32_t qf, bool profiling_lock) : pd_(pd), qf_(qf) {
        // Device: VK_KHR_performance_query + performanceCounterQueryPools (VUID-...-performanceCounterQueryPools-03237)
        //         + hostQueryReset (VUID-vkResetQueryPool-None-02665).
        VkPhysicalDevicePerformanceQueryFeaturesKHR pqf{};
        pqf.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PERFORMANCE_QUERY_FEATURES_KHR;
        pqf.performanceCounterQueryPools = VK_TRUE;
        VkPhysicalDeviceVulkan12Features f12{};
        f12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        f12.hostQueryReset = VK_TRUE;
        f12.pNext = &pqf;
        VkPhysicalDeviceFeatures2 f2{};
        f2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        f2.pNext = &f12;

        const float prio = 1.0f;
        VkDeviceQueueCreateInfo qci{};
        qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qci.queueFamilyIndex = qf_;
        qci.queueCount = 1;
        qci.pQueuePriorities = &prio;
        const char* exts[] = {VK_KHR_PERFORMANCE_QUERY_EXTENSION_NAME};
        VkDeviceCreateInfo dci{};
        dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        dci.pNext = &f2;
        dci.queueCreateInfoCount = 1;
        dci.pQueueCreateInfos = &qci;
        dci.enabledExtensionCount = 1;
        dci.ppEnabledExtensionNames = exts;
        YVO_VK_CHECK(vkCreateDevice(pd_, &dci, nullptr, &dev_));
        try {
            vkGetDeviceQueue(dev_, qf_, 0, &queue_);

            VkPhysicalDeviceProperties props{};
            vkGetPhysicalDeviceProperties(pd_, &props);
            ts_period_ns_ = static_cast<double>(props.limits.timestampPeriod);
            std::uint32_t nqf = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(pd_, &nqf, nullptr);
            std::vector<VkQueueFamilyProperties> qfs(nqf);
            vkGetPhysicalDeviceQueueFamilyProperties(pd_, &nqf, qfs.data());
            const std::uint32_t vb = qfs[qf_].timestampValidBits;
            if (vb == 0) throw std::runtime_error("queue family has no timestamp support");
            ts_mask_ = vb >= 64 ? ~0ull : ((1ull << vb) - 1ull);

            acquire_ = reinterpret_cast<PFN_vkAcquireProfilingLockKHR>(
                vkGetDeviceProcAddr(dev_, "vkAcquireProfilingLockKHR"));
            release_ = reinterpret_cast<PFN_vkReleaseProfilingLockKHR>(
                vkGetDeviceProcAddr(dev_, "vkReleaseProfilingLockKHR"));
            if (!acquire_ || !release_) throw std::runtime_error("profiling lock entry points not found");
            if (profiling_lock) {
                // The lock is held for the whole lifetime of the rig: it must be held before vkBeginCommandBuffer
                // (VUID-vkCmdBeginQuery-queryPool-03223) and continuously throughout recording and submission
                // (VUID-vkQueueSubmit-pCommandBuffers-03220). Releasing it between runs and re-recording the same
                // command buffer was flagged by the validation layer as 03220, so it is taken once here.
                VkAcquireProfilingLockInfoKHR li{};
                li.sType = VK_STRUCTURE_TYPE_ACQUIRE_PROFILING_LOCK_INFO_KHR;
                li.timeout = 2'000'000'000ull;  // 2 s
                YVO_VK_CHECK(acquire_(dev_, &li));
                lock_held_ = true;
            }

            create_pipeline();
            create_buffers();

            VkCommandPoolCreateInfo cpi{};
            cpi.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
            cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
            cpi.queueFamilyIndex = qf_;  // = the perf pool's queueFamilyIndex (VUID-vkCmdBeginQuery-queryPool-07289)
            YVO_VK_CHECK(vkCreateCommandPool(dev_, &cpi, nullptr, &cmd_pool_));
            VkCommandBufferAllocateInfo cai{};
            cai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            cai.commandPool = cmd_pool_;
            cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            cai.commandBufferCount = kMaxSlots;
            YVO_VK_CHECK(vkAllocateCommandBuffers(dev_, &cai, cmds_.data()));
            VkFenceCreateInfo fci{};
            fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            YVO_VK_CHECK(vkCreateFence(dev_, &fci, nullptr, &fence_));

            VkQueryPoolCreateInfo tpi{};
            tpi.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
            tpi.queryType = VK_QUERY_TYPE_TIMESTAMP;
            tpi.queryCount = 2 * kMaxSlots;
            YVO_VK_CHECK(vkCreateQueryPool(dev_, &tpi, nullptr, &ts_pool_));
        } catch (...) {
            destroy();
            throw;
        }
    }

    ~Rig() { destroy(); }
    Rig(const Rig&) = delete;
    Rig& operator=(const Rig&) = delete;

    // Replace the performance query pool (empty → no performance query). Command buffers must be re-recorded.
    void set_counters(const std::vector<std::uint32_t>& counters, const std::vector<Counter>& all, std::uint32_t passes) {
        YVO_VK_CHECK(vkDeviceWaitIdle(dev_));
        if (perf_pool_) vkDestroyQueryPool(dev_, perf_pool_, nullptr);
        perf_pool_ = VK_NULL_HANDLE;
        counters_ = counters;
        passes_ = counters.empty() ? 1 : passes;
        storage_.clear();
        for (std::uint32_t i : counters) {
            if (all[i].c.scope == VK_PERFORMANCE_COUNTER_SCOPE_COMMAND_BUFFER_KHR)
                throw std::runtime_error("COMMAND_BUFFER-scope counter not supported by this command buffer layout");
            storage_.push_back(all[i].c.storage);
        }
        if (counters.empty()) return;
        if (!lock_held_) throw std::runtime_error("performance queries need the profiling lock");
        VkQueryPoolPerformanceCreateInfoKHR pci{};
        pci.sType = VK_STRUCTURE_TYPE_QUERY_POOL_PERFORMANCE_CREATE_INFO_KHR;
        pci.queueFamilyIndex = qf_;
        pci.counterIndexCount = static_cast<std::uint32_t>(counters_.size());
        pci.pCounterIndices = counters_.data();
        VkQueryPoolCreateInfo qpi{};
        qpi.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        qpi.pNext = &pci;  // VUID-VkQueryPoolCreateInfo-queryType-03222
        qpi.queryType = VK_QUERY_TYPE_PERFORMANCE_QUERY_KHR;
        qpi.queryCount = kMaxSlots;
        YVO_VK_CHECK(vkCreateQueryPool(dev_, &qpi, nullptr, &perf_pool_));
    }

    // Record command buffer `slot` (query slot = slot) with `k` dispatches of `n_words` words each.
    // n_words == 0 or k == 0 → no dispatch (idle query).
    void record(std::uint32_t slot, std::uint32_t k, std::uint32_t n_words, bool with_ts) {
        if (slot >= kMaxSlots || n_words > kMaxWords) throw std::runtime_error("record: bad slot or size");
        VkCommandBuffer cb = cmds_[slot];
        YVO_VK_CHECK(vkResetCommandBuffer(cb, 0));
        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        // No ONE_TIME_SUBMIT: the same command buffer is resubmitted every iteration and once per counter pass.
        YVO_VK_CHECK(vkBeginCommandBuffer(cb, &bi));
        if (with_ts) vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, ts_pool_, 2 * slot);
        if (perf_pool_) vkCmdBeginQuery(cb, perf_pool_, slot, 0);
        if (n_words > 0) {
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_layout_, 0, 1, &set_, 0, nullptr);
            vkCmdPushConstants(cb, pipe_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(n_words), &n_words);
            for (std::uint32_t d = 0; d < k; ++d) {
                // dst is updated with atomics by every dispatch (also across command buffers of one submission):
                // earlier shader writes → later shader read/write.
                VkMemoryBarrier cc{};
                cc.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
                cc.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                cc.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
                vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                                     1, &cc, 0, nullptr, 0, nullptr);
                vkCmdDispatch(cb, kGroups, 1, 1);
            }
        }
        if (perf_pool_) vkCmdEndQuery(cb, perf_pool_, slot);
        if (n_words > 0) {
            // Shader writes of dst → host reads after the fence wait.
            VkMemoryBarrier mb{};
            mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0,
                                 nullptr, 0, nullptr);
        }
        if (with_ts) vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, ts_pool_, 2 * slot + 1);
        YVO_VK_CHECK(vkEndCommandBuffer(cb));
    }

    struct Iter {
        std::vector<std::vector<VkPerformanceCounterResultKHR>> perf;  // [slot][counter]
        std::vector<double> gpu_ns;                                    // [slot] ts1 - ts0
        double span_ns = 0;                                            // ts1(last slot) - ts0(slot 0)
        double host_submit_us = 0;                                     // vkQueueSubmit → fence signalled, all passes
        double host_perf_read_us = 0;                                  // vkGetQueryPoolResults (performance)
        double host_ts_read_us = 0;                                    // vkGetQueryPoolResults (timestamp)
        std::uint32_t submits = 0;
        bool sum_ok = true;
        std::uint32_t gpu_sum = 0, gpu_groups = 0;
    };

    // Submit command buffers [0, n_cb) as one batch per counter pass, wait, read the queries.
    // `dispatches` = total dispatches in the batch (for the heap_sum check), `n_words` = words per dispatch.
    Iter run(std::uint32_t n_cb, std::uint32_t dispatches, std::uint32_t n_words, bool with_ts) {
        Iter it;
        if (perf_pool_) vkResetQueryPool(dev_, perf_pool_, 0, n_cb);  // host reset: resets every pass
        const std::uint32_t exp_sum = expected_sum(n_words) * dispatches;  // uint32 wraparound, like the shader
        const std::uint32_t exp_groups = n_words ? kGroups * dispatches : 0;
        for (std::uint32_t pass = 0; pass < passes_; ++pass) {
            // Timestamps must be unavailable before every write (VUID-vkCmdWriteTimestamp-None-00830), and the same
            // command buffer is resubmitted per pass → reset them per pass (gpu_ns reports the last pass). The
            // performance query is reset only once above: it must stay open across its passes.
            if (with_ts) vkResetQueryPool(dev_, ts_pool_, 0, 2 * n_cb);
            dst_map_[0] = 0;
            dst_map_[1] = 0;
            VkPerformanceQuerySubmitInfoKHR psi{};
            psi.sType = VK_STRUCTURE_TYPE_PERFORMANCE_QUERY_SUBMIT_INFO_KHR;
            psi.counterPassIndex = pass;  // VUID-VkPerformanceQuerySubmitInfoKHR-counterPassIndex-03221
            VkSubmitInfo si{};
            si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            si.pNext = perf_pool_ ? &psi : nullptr;
            si.commandBufferCount = n_cb;
            si.pCommandBuffers = cmds_.data();
            YVO_VK_CHECK(vkResetFences(dev_, 1, &fence_));
            const auto t0 = Clock::now();
            YVO_VK_CHECK(vkQueueSubmit(queue_, 1, &si, fence_));
            YVO_VK_CHECK(vkWaitForFences(dev_, 1, &fence_, VK_TRUE, 5'000'000'000ull));
            it.host_submit_us += std::chrono::duration<double, std::micro>(Clock::now() - t0).count();
            ++it.submits;
            it.gpu_sum = dst_map_[0];
            it.gpu_groups = dst_map_[1];
            if (it.gpu_sum != exp_sum || it.gpu_groups != exp_groups) it.sum_ok = false;
        }
        if (perf_pool_) {
            const std::size_t stride = counters_.size() * sizeof(VkPerformanceCounterResultKHR);
            std::vector<VkPerformanceCounterResultKHR> buf(counters_.size() * n_cb);
            const auto t0 = Clock::now();
            // VUID-vkGetQueryPoolResults-queryType-09440: no 64_BIT / AVAILABILITY / PARTIAL for performance queries.
            // VUID-...-03229 / 04519: stride = counterIndexCount * sizeof(VkPerformanceCounterResultKHR).
            YVO_VK_CHECK(vkGetQueryPoolResults(dev_, perf_pool_, 0, n_cb, stride * n_cb, buf.data(), stride,
                                               VK_QUERY_RESULT_WAIT_BIT));
            it.host_perf_read_us = std::chrono::duration<double, std::micro>(Clock::now() - t0).count();
            it.perf.resize(n_cb);
            for (std::uint32_t s = 0; s < n_cb; ++s)
                it.perf[s].assign(buf.begin() + s * counters_.size(), buf.begin() + (s + 1) * counters_.size());
        }
        if (with_ts) {
            std::vector<std::uint64_t> ts(2 * n_cb);
            const auto t0 = Clock::now();
            YVO_VK_CHECK(vkGetQueryPoolResults(dev_, ts_pool_, 0, 2 * n_cb, ts.size() * sizeof(std::uint64_t),
                                               ts.data(), sizeof(std::uint64_t),
                                               VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
            it.host_ts_read_us = std::chrono::duration<double, std::micro>(Clock::now() - t0).count();
            for (std::uint32_t s = 0; s < n_cb; ++s)
                it.gpu_ns.push_back(static_cast<double>((ts[2 * s + 1] - ts[2 * s]) & ts_mask_) * ts_period_ns_);
            it.span_ns = static_cast<double>((ts[2 * n_cb - 1] - ts[0]) & ts_mask_) * ts_period_ns_;
        }
        return it;
    }

    // Byte sum of the first n_words source words (uint32 wraparound, same as heap_sum.comp).
    std::uint32_t expected_sum(std::uint32_t n_words) {
        auto it = sum_cache_.find(n_words);
        if (it != sum_cache_.end()) return it->second;
        std::uint32_t s = 0;
        for (std::uint32_t i = 0; i < n_words; ++i) {
            const std::uint32_t v = src_map_[i];
            s += (v & 0xffu) + ((v >> 8) & 0xffu) + ((v >> 16) & 0xffu) + (v >> 24);
        }
        sum_cache_[n_words] = s;
        return s;
    }

    const std::vector<VkPerformanceCounterStorageKHR>& storage() const { return storage_; }
    std::uint32_t passes() const { return passes_; }

private:
    // Reverse creation order. Also called when the constructor throws halfway.
    void destroy() {
        if (dev_ == VK_NULL_HANDLE) return;
        vkDeviceWaitIdle(dev_);
        if (lock_held_) release_(dev_);
        lock_held_ = false;
        if (perf_pool_) vkDestroyQueryPool(dev_, perf_pool_, nullptr);
        if (ts_pool_) vkDestroyQueryPool(dev_, ts_pool_, nullptr);
        if (fence_) vkDestroyFence(dev_, fence_, nullptr);
        if (cmd_pool_) vkDestroyCommandPool(dev_, cmd_pool_, nullptr);
        if (dst_) vkDestroyBuffer(dev_, dst_, nullptr);
        if (src_) vkDestroyBuffer(dev_, src_, nullptr);
        if (dst_mem_) vkFreeMemory(dev_, dst_mem_, nullptr);  // unmapped implicitly by vkFreeMemory
        if (src_mem_) vkFreeMemory(dev_, src_mem_, nullptr);
        if (desc_pool_) vkDestroyDescriptorPool(dev_, desc_pool_, nullptr);
        if (pipeline_) vkDestroyPipeline(dev_, pipeline_, nullptr);
        if (pipe_layout_) vkDestroyPipelineLayout(dev_, pipe_layout_, nullptr);
        if (set_layout_) vkDestroyDescriptorSetLayout(dev_, set_layout_, nullptr);
        vkDestroyDevice(dev_, nullptr);
        dev_ = VK_NULL_HANDLE;
    }

    void create_pipeline() {
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
        YVO_VK_CHECK(vkCreateDescriptorSetLayout(dev_, &sli, nullptr, &set_layout_));

        VkPushConstantRange pcr{};
        pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcr.size = sizeof(std::uint32_t);
        VkPipelineLayoutCreateInfo pli{};
        pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pli.setLayoutCount = 1;
        pli.pSetLayouts = &set_layout_;
        pli.pushConstantRangeCount = 1;
        pli.pPushConstantRanges = &pcr;
        YVO_VK_CHECK(vkCreatePipelineLayout(dev_, &pli, nullptr, &pipe_layout_));

        VkShaderModuleCreateInfo smi{};
        smi.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        smi.codeSize = sizeof(kHeapSumSpv);
        smi.pCode = kHeapSumSpv;
        VkShaderModule mod = VK_NULL_HANDLE;
        YVO_VK_CHECK(vkCreateShaderModule(dev_, &smi, nullptr, &mod));
        VkComputePipelineCreateInfo cpi{};
        cpi.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        cpi.stage.module = mod;
        cpi.stage.pName = "main";
        cpi.layout = pipe_layout_;
        const VkResult pr = vkCreateComputePipelines(dev_, VK_NULL_HANDLE, 1, &cpi, nullptr, &pipeline_);
        vkDestroyShaderModule(dev_, mod, nullptr);
        YVO_VK_CHECK(pr);

        VkDescriptorPoolSize ps{};
        ps.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        ps.descriptorCount = 2;
        VkDescriptorPoolCreateInfo dpi{};
        dpi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpi.maxSets = 1;
        dpi.poolSizeCount = 1;
        dpi.pPoolSizes = &ps;
        YVO_VK_CHECK(vkCreateDescriptorPool(dev_, &dpi, nullptr, &desc_pool_));
        VkDescriptorSetAllocateInfo dsi{};
        dsi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dsi.descriptorPool = desc_pool_;
        dsi.descriptorSetCount = 1;
        dsi.pSetLayouts = &set_layout_;
        YVO_VK_CHECK(vkAllocateDescriptorSets(dev_, &dsi, &set_));
    }

    // HOST_VISIBLE | HOST_COHERENT buffer (UMA). Non-coherent memory is rejected rather than flushed: this is a
    // diagnostic tool and V3D exposes a single coherent memory type.
    void make_buffer(VkDeviceSize size, VkBuffer& buf, VkDeviceMemory& mem, void** map) {
        VkBufferCreateInfo bci{};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size = size;
        bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        YVO_VK_CHECK(vkCreateBuffer(dev_, &bci, nullptr, &buf));
        VkMemoryRequirements req{};
        vkGetBufferMemoryRequirements(dev_, buf, &req);
        VkPhysicalDeviceMemoryProperties mp{};
        vkGetPhysicalDeviceMemoryProperties(pd_, &mp);
        const VkMemoryPropertyFlags want = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        std::uint32_t type = UINT32_MAX;
        for (std::uint32_t i = 0; i < mp.memoryTypeCount; ++i)
            if ((req.memoryTypeBits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) {
                type = i;
                break;
            }
        if (type == UINT32_MAX) throw std::runtime_error("no HOST_VISIBLE|HOST_COHERENT memory type");
        VkMemoryAllocateInfo mai{};
        mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mai.allocationSize = req.size;
        mai.memoryTypeIndex = type;
        YVO_VK_CHECK(vkAllocateMemory(dev_, &mai, nullptr, &mem));
        YVO_VK_CHECK(vkBindBufferMemory(dev_, buf, mem, 0));
        YVO_VK_CHECK(vkMapMemory(dev_, mem, 0, VK_WHOLE_SIZE, 0, map));
    }

    void create_buffers() {
        void* p = nullptr;
        make_buffer(VkDeviceSize{kMaxWords} * 4u, src_, src_mem_, &p);
        src_map_ = static_cast<std::uint32_t*>(p);
        make_buffer(8, dst_, dst_mem_, &p);
        dst_map_ = static_cast<std::uint32_t*>(p);
        std::uint32_t x = 0x12345678u;  // xorshift32 pattern (deterministic)
        for (std::uint32_t i = 0; i < kMaxWords; ++i) {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            src_map_[i] = x;
        }
        std::array<VkDescriptorBufferInfo, 2> bi{};
        bi[0] = {src_, 0, VK_WHOLE_SIZE};
        bi[1] = {dst_, 0, VK_WHOLE_SIZE};
        std::array<VkWriteDescriptorSet, 2> w{};
        for (std::uint32_t i = 0; i < 2; ++i) {
            w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[i].dstSet = set_;
            w[i].dstBinding = i;
            w[i].descriptorCount = 1;
            w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            w[i].pBufferInfo = &bi[i];
        }
        vkUpdateDescriptorSets(dev_, 2, w.data(), 0, nullptr);
    }

    VkPhysicalDevice pd_;
    std::uint32_t qf_;
    std::vector<std::uint32_t> counters_;
    std::vector<VkPerformanceCounterStorageKHR> storage_;
    std::uint32_t passes_ = 1;
    double ts_period_ns_ = 1.0;
    std::uint64_t ts_mask_ = ~0ull;
    std::map<std::uint32_t, std::uint32_t> sum_cache_;
    VkDevice dev_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    PFN_vkAcquireProfilingLockKHR acquire_ = nullptr;
    PFN_vkReleaseProfilingLockKHR release_ = nullptr;
    bool lock_held_ = false;
    VkDescriptorSetLayout set_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout pipe_layout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    VkDescriptorPool desc_pool_ = VK_NULL_HANDLE;
    VkDescriptorSet set_ = VK_NULL_HANDLE;
    VkBuffer src_ = VK_NULL_HANDLE, dst_ = VK_NULL_HANDLE;
    VkDeviceMemory src_mem_ = VK_NULL_HANDLE, dst_mem_ = VK_NULL_HANDLE;
    std::uint32_t* src_map_ = nullptr;
    std::uint32_t* dst_map_ = nullptr;
    VkCommandPool cmd_pool_ = VK_NULL_HANDLE;
    std::array<VkCommandBuffer, kMaxSlots> cmds_{};
    VkFence fence_ = VK_NULL_HANDLE;
    VkQueryPool ts_pool_ = VK_NULL_HANDLE;
    VkQueryPool perf_pool_ = VK_NULL_HANDLE;
};

// Instance RAII (destroyed last).
struct Instance {
    VkInstance h = VK_NULL_HANDLE;
    Instance() {
        VkApplicationInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        ai.pApplicationName = "yvo_perf_counters";
        ai.apiVersion = VK_API_VERSION_1_3;
        VkInstanceCreateInfo ici{};
        ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        ici.pApplicationInfo = &ai;
        YVO_VK_CHECK(vkCreateInstance(&ici, nullptr, &h));
    }
    ~Instance() {
        if (h) vkDestroyInstance(h, nullptr);
    }
    Instance(const Instance&) = delete;
    Instance& operator=(const Instance&) = delete;
};

// Shared state of the experiment modes.
struct ExpCtx {
    VkPhysicalDevice pd = VK_NULL_HANDLE;
    std::uint32_t qf = 0;
    PerfFns fn;
    std::vector<Counter> all;
    const Options* opt = nullptr;
    std::ofstream csv;
    bool write_csv = false;
    std::uint32_t sum_failures = 0;

    std::uint32_t find(const std::string& name) const {
        for (std::uint32_t i = 0; i < all.size(); ++i)
            if (name == all[i].d.name) return i;
        throw std::runtime_error("counter not found in the runtime enumeration: " + name);
    }
    std::vector<std::uint32_t> find_all(const std::vector<std::string>& names) const {
        std::vector<std::uint32_t> v;
        for (const auto& n : names) v.push_back(find(n));
        return v;
    }
    void row(const char* exp, const std::string& config, int iter, const std::string& metric, double value) {
        if (!write_csv) return;
        // Counter values exceed 6 significant digits (e.g. 16885760): integers are written in full.
        const bool integral = std::fabs(value) < 1e15 && value == std::floor(value);
        csv << exp << ',' << csv_field(opt->label) << ',' << csv_field(config) << ',' << iter << ','
            << csv_field(metric) << ',' << (integral ? fmt("%.0f", value) : fmt("%.6f", value)) << '\n';
    }
};

// 1-pass default experiment set: compute side + graphics controls. Names come from the runtime enumeration
// (see 11_logs/2026-09-29_v3d_perf_counters/counters.csv); a missing name is an error, not a silent skip.
const std::vector<std::string> kExpCounters = {
    // compute side
    "cycle-count", "core-active", "compute-active-cycles", "QPU-active", "QPU-stalls", "QPU-wait-bubble",
    "QPU-stalls-TMU", "TMU-active-cycles", "TMU-stalled-cycles", "L2T-total-cache-hit", "L2T-total-cache-miss",
    "L2T-stalls-read-data", "core-memory-reads", "core-memory-writes", "AXI-read-wait-cycles",
    // graphics controls (expected 0 for our compute job)
    "CLE-bin-thread-active-cycles", "CLE-render-thread-active-cycles", "FEP-valid-quads",
    "FEP-valid-primitives-rendered-pixels", "TLB-quads-written-to-color-buffer", "PTB-primitives-binned"};

const std::vector<std::string> kExp4Counters = {"cycle-count", "compute-active-cycles", "QPU-active",
                                                "core-memory-reads"};

constexpr std::uint32_t kExpWords = 1024u * 1024u;  // 4 MiB per dispatch

// Per-configuration sample series: metric name → values.
using Series = std::map<std::string, std::vector<double>>;

void collect(ExpCtx& x, Rig& rig, const char* exp, const std::string& config, int iter, const Rig::Iter& it,
             const std::vector<std::uint32_t>& set, bool sum_slots, Series& s) {
    if (!it.sum_ok) ++x.sum_failures;
    auto put = [&](const std::string& m, double v) {
        s[m].push_back(v);
        x.row(exp, config, iter, m, v);
    };
    double gpu = 0;
    for (double g : it.gpu_ns) gpu += g;
    if (!it.gpu_ns.empty()) {
        put("gpu_ns", sum_slots ? gpu : it.gpu_ns[0]);
        if (it.gpu_ns.size() > 1) put("span_ns", it.span_ns);
    }
    put("host_submit_us", it.host_submit_us);
    if (!it.perf.empty()) put("host_perf_read_us", it.host_perf_read_us);
    if (!it.gpu_ns.empty()) put("host_ts_read_us", it.host_ts_read_us);
    put("submits", it.submits);
    put("sum_ok", it.sum_ok ? 1 : 0);
    for (std::size_t c = 0; c < set.size() && !it.perf.empty(); ++c) {
        double v = 0;
        const std::size_t nslots = sum_slots ? it.perf.size() : 1;
        for (std::size_t sl = 0; sl < nslots; ++sl) v += result_double(it.perf[sl][c], rig.storage()[c]);
        put(x.all[set[c]].d.name, v);
    }
}

// Warm-up + measured loop on already recorded command buffers.
Series measure(ExpCtx& x, Rig& rig, const char* exp, const std::string& config, std::uint32_t n_cb,
               std::uint32_t dispatches, const std::vector<std::uint32_t>& set, bool sum_slots, int iters,
               GpuStats* before = nullptr, GpuStats* after = nullptr) {
    for (int i = 0; i < x.opt->warmup; ++i) {
        const auto it = rig.run(n_cb, dispatches, kExpWords, true);
        if (!it.sum_ok) ++x.sum_failures;
    }
    Series s;
    if (before) *before = read_gpu_stats();
    for (int i = 0; i < iters; ++i) collect(x, rig, exp, config, i, rig.run(n_cb, dispatches, kExpWords, true), set,
                                            sum_slots, s);
    if (after) *after = read_gpu_stats();
    return s;
}

double stdev(const std::vector<double>& v) {
    if (v.size() < 2) return 0;
    double m = 0;
    for (double a : v) m += a;
    m /= static_cast<double>(v.size());
    double q = 0;
    for (double a : v) q += (a - m) * (a - m);
    return std::sqrt(q / static_cast<double>(v.size() - 1));
}

void print_series(const std::string& config, const Series& s, const std::vector<std::string>& order) {
    std::printf("  [%s]\n", config.c_str());
    std::printf("    %-34s %14s %14s %14s %14s %14s %8s\n", "metric", "median", "p90", "min", "max", "stdev", "cv%");
    for (const auto& m : order) {
        const auto it = s.find(m);
        if (it == s.end()) continue;
        const auto st = yvo::stats(it->second);
        const double sd = stdev(it->second);
        std::printf("    %-34s %14.1f %14.1f %14.1f %14.1f %14.1f %8.2f\n", m.c_str(), st.median, st.p90, st.min,
                    st.max, sd, st.mean != 0 ? 100.0 * sd / st.mean : 0.0);
    }
}

std::vector<std::string> metric_order(const ExpCtx& x, const std::vector<std::uint32_t>& set) {
    std::vector<std::string> o = {"gpu_ns",   "span_ns", "host_submit_us", "host_perf_read_us", "host_ts_read_us",
                                  "submits", "sum_ok"};
    for (std::uint32_t i : set) o.push_back(x.all[i].d.name);
    return o;
}

double med(const Series& s, const std::string& m) {
    const auto it = s.find(m);
    return it == s.end() ? std::nan("") : yvo::stats(it->second).median;
}
double p90(const Series& s, const std::string& m) {
    const auto it = s.find(m);
    return it == s.end() ? std::nan("") : yvo::stats(it->second).p90;
}

// E1: does one performance query cover every job of the command buffer?
void exp_granularity(ExpCtx& x) {
    const int iters = x.opt->iters > 0 ? x.opt->iters : 100;
    const auto set = x.find_all(kExpCounters);
    const std::uint32_t passes = num_passes(x.fn, x.pd, x.qf, set);
    if (passes != 1) throw std::runtime_error(fmt("experiment counter set needs %u passes, expected 1", passes));
    Rig rig(x.pd, x.qf, true);
    rig.set_counters(set, x.all, passes);
    std::printf("== E1 granularity: heap_sum 4 MiB per dispatch, warmup %d, iters %d, %zu counters (1 pass) ==\n",
                x.opt->warmup, iters, set.size());
    const auto order = metric_order(x, set);
    std::map<std::string, Series> res;
    for (std::uint32_t k : {1u, 2u, 4u, 8u}) {
        // (a) one command buffer, k dispatches, one query
        rig.record(0, k, kExpWords, true);
        const std::string ca = fmt("onecb_k%u", k);
        GpuStats b, a;
        res[ca] = measure(x, rig, "granularity", ca, 1, k, set, false, iters, &b, &a);
        print_series(ca, res[ca], order);
        print_gpu_stats_delta(ca.c_str(), b, a, iters);
        // (b) k command buffers with one dispatch each, one query each, one vkQueueSubmit; values summed over CBs
        for (std::uint32_t s = 0; s < k; ++s) rig.record(s, 1, kExpWords, true);
        const std::string cbn = fmt("kcb_k%u", k);
        res[cbn] = measure(x, rig, "granularity", cbn, k, k, set, true, iters, &b, &a);
        print_series(cbn, res[cbn], order);
        print_gpu_stats_delta(cbn.c_str(), b, a, iters);
    }
    std::printf("\n-- E1 ratios (median / median of onecb_k1) --\n");
    const std::vector<std::string> keys = {"gpu_ns", "cycle-count", "compute-active-cycles", "QPU-active",
                                           "core-memory-reads", "TMU-active-cycles"};
    std::printf("  %-12s", "config");
    for (const auto& k : keys) std::printf(" %22s", k.c_str());
    std::printf("\n");
    for (const auto& [cfg, s] : res) {
        std::printf("  %-12s", cfg.c_str());
        for (const auto& k : keys) std::printf(" %22.3f", med(s, k) / med(res["onecb_k1"], k));
        std::printf("\n");
    }
    std::printf("\n-- E1 effective clock (median of per-iteration ratio) --\n");
    std::printf("  %-12s %22s %28s\n", "config", "cycle-count/gpu_ns MHz", "compute-active/gpu_ns MHz");
    for (const auto& [cfg, s] : res) {
        std::vector<double> c1, c2;
        const auto& g = s.at("gpu_ns");
        for (std::size_t i = 0; i < g.size(); ++i) {
            c1.push_back(s.at("cycle-count")[i] / g[i] * 1e3);
            c2.push_back(s.at("compute-active-cycles")[i] / g[i] * 1e3);
        }
        std::printf("  %-12s %22.1f %28.1f\n", cfg.c_str(), yvo::stats(c1).median, yvo::stats(c2).median);
    }
}

// E2: spread of a fixed job under the condition named by --label (idle / vkcube / second process).
void exp_contention(ExpCtx& x) {
    const int iters = x.opt->iters > 0 ? x.opt->iters : 100;
    const auto set = x.find_all(kExpCounters);
    const std::uint32_t passes = num_passes(x.fn, x.pd, x.qf, set);
    if (passes != 1) throw std::runtime_error(fmt("experiment counter set needs %u passes, expected 1", passes));
    Rig rig(x.pd, x.qf, true);
    rig.set_counters(set, x.all, passes);
    std::printf("== E2 contention [label %s]: heap_sum 4 MiB x1, warmup %d, iters %d, %zu counters (1 pass) ==\n",
                x.opt->label.c_str(), x.opt->warmup, iters, set.size());
    rig.record(0, 1, kExpWords, true);
    GpuStats b, a;
    const Series s = measure(x, rig, "contention", "fixed", 1, 1, set, false, iters, &b, &a);
    print_series("fixed", s, metric_order(x, set));
    print_gpu_stats_raw("before", b);
    print_gpu_stats_raw("after", a);
    print_gpu_stats_delta("fixed", b, a, iters);
}

// E3: cost of the performance query itself.
void exp_overhead(ExpCtx& x) {
    const int iters = x.opt->iters > 0 ? x.opt->iters : 200;
    std::vector<std::uint32_t> set32;
    for (std::uint32_t i = 0; i < 32 && i < x.all.size(); ++i) set32.push_back(i);
    const auto set4 = x.find_all(kExp4Counters);
    std::vector<std::uint32_t> set_all;
    for (std::uint32_t i = 0; i < x.all.size(); ++i) set_all.push_back(i);
    const std::uint32_t p32 = num_passes(x.fn, x.pd, x.qf, set32);
    const std::uint32_t p4 = num_passes(x.fn, x.pd, x.qf, set4);
    const std::uint32_t pall = num_passes(x.fn, x.pd, x.qf, set_all);
    std::printf("== E3 overhead: heap_sum 4 MiB x1, warmup %d, iters %d; passes: 32 counters=%u, 4 counters=%u, "
                "all %zu=%u ==\n",
                x.opt->warmup, iters, p32, p4, set_all.size(), pall);
    Rig rig(x.pd, x.qf, true);
    struct Cfg {
        const char* name;
        const std::vector<std::uint32_t>* set;
        std::uint32_t passes;
    };
    const std::vector<std::uint32_t> none;
    const std::array<Cfg, 4> cfgs{{{"i_ts_only", &none, 1},
                                   {"ii_ts_perf32", &set32, p32},
                                   {"iii_ts_perf4", &set4, p4},
                                   {"i_ts_only_repeat", &none, 1}}};
    std::map<std::string, Series> res;
    for (const auto& c : cfgs) {
        rig.set_counters(*c.set, x.all, c.passes);
        rig.record(0, 1, kExpWords, true);
        GpuStats b, a;
        res[c.name] = measure(x, rig, "overhead", c.name, 1, 1, *c.set, false, iters, &b, &a);
        print_series(c.name, res[c.name], metric_order(x, *c.set));
        print_gpu_stats_delta(c.name, b, a, iters);
    }
    // One multi-pass run (all counters): count the submissions and time them. Not a distribution.
    // No timestamps here: with timestamps in a command buffer resubmitted per counterPassIndex, the validation layer
    // reported VUID-vkCmdWriteTimestamp-None-00830 for passes >= 1 even with a host reset before every pass, and
    // then crashed inside the layer (no crash without layers). Host time is all this run needs.
    rig.set_counters(set_all, x.all, pall);
    rig.record(0, 1, kExpWords, false);
    Series s3;
    const auto it = rig.run(1, 1, kExpWords, false);
    collect(x, rig, "overhead", "all_multipass_once", 0, it, set_all, false, s3);
    std::printf("  [all_multipass_once] passes=%u submits=%u host_submit_us=%.1f host_perf_read_us=%.1f sum_ok=%d\n",
                pall, it.submits, it.host_submit_us, it.host_perf_read_us, it.sum_ok ? 1 : 0);

    std::printf("\n-- E3 judgement: overhead vs noise of (i) (noise = p90 - median) --\n");
    const Series& base = res["i_ts_only"];
    for (const char* m : {"gpu_ns", "host_submit_us"}) {
        const double noise = p90(base, m) - med(base, m);
        for (const char* c : {"ii_ts_perf32", "iii_ts_perf4", "i_ts_only_repeat"}) {
            const double d = med(res[c], m) - med(base, m);
            std::printf("  %-16s %-15s median %12.1f  delta %+10.1f (%+6.2f%%)  noise(i) %8.1f  -> %s\n", c, m,
                        med(res[c], m), d, 100.0 * d / med(base, m), noise,
                        std::fabs(d) < noise ? "negligible" : "NOT negligible");
        }
    }
}

// --load: background V3D compute load for E2 condition C. No queries, no profiling lock.
void run_load(VkPhysicalDevice pd, std::uint32_t qf, double sec) {
    Rig rig(pd, qf, false);
    rig.record(0, 1, kExpWords, false);
    std::printf("load: heap_sum 4 MiB dispatches back to back for %.1f s\n", sec);
    std::fflush(stdout);
    const auto t0 = Clock::now();
    std::uint64_t n = 0, bad = 0;
    while (std::chrono::duration<double>(Clock::now() - t0).count() < sec) {
        if (!rig.run(1, 1, kExpWords, false).sum_ok) ++bad;
        ++n;
    }
    const double el = std::chrono::duration<double>(Clock::now() - t0).count();
    std::printf("load: %" PRIu64 " submissions in %.2f s (%.1f /s), heap_sum mismatches %" PRIu64 "\n", n, el,
                static_cast<double>(n) / el, bad);
}

void run_sanity(VkPhysicalDevice pd, std::uint32_t qf, const PerfFns& fn, const std::vector<Counter>& cs,
                std::vector<std::uint32_t> sel) {
    std::printf("\n== Sanity check (queue family %u) ==\n", qf);
    for (std::uint32_t i : sel)
        if (i >= cs.size()) throw std::runtime_error(fmt("--sanity-counters: index %u out of range", i));
    const std::uint32_t passes = num_passes(fn, pd, qf, sel);
    std::printf("counter set: %zu counters, %u pass(es)\n", sel.size(), passes);

    Rig rig(pd, qf, true);
    rig.set_counters(sel, cs, passes);
    struct Load {
        const char* name;
        std::uint32_t words;
    };
    const std::array<Load, 3> loads{{{"idle (no dispatch)", 0u},
                                     {"small (1 MiB heap_sum)", 256u * 1024u},
                                     {"large (16 MiB heap_sum)", Rig::kMaxWords}}};
    std::vector<Rig::Iter> res;
    for (const auto& l : loads) {
        rig.record(0, 1, l.words, false);
        res.push_back(rig.run(1, l.words ? 1 : 0, l.words, false));
        const auto& r = res.back();
        std::printf("run %-24s : VK_SUCCESS, gpu_sum=0x%08x cpu_sum=0x%08x groups=%u/%u %s\n", l.name, r.gpu_sum,
                    l.words ? rig.expected_sum(l.words) : 0u, r.gpu_groups, l.words ? Rig::kGroups : 0u,
                    r.sum_ok ? "OK" : "MISMATCH");
    }
    std::printf("\n%5s  %-40s %-12s %20s %20s %20s\n", "index", "name", "unit", "idle", "small", "large");
    for (std::size_t k = 0; k < sel.size(); ++k) {
        const Counter& c = cs[sel[k]];
        std::printf("%5u  %-40s %-12s %20s %20s %20s\n", sel[k], c.d.name, unit_str(c.c.unit),
                    result_str(res[0].perf[0][k], c.c.storage).c_str(),
                    result_str(res[1].perf[0][k], c.c.storage).c_str(),
                    result_str(res[2].perf[0][k], c.c.storage).c_str());
    }
}

int run(const Options& opt) {
    Instance inst;
    std::uint32_t npd = 0;
    YVO_VK_CHECK(vkEnumeratePhysicalDevices(inst.h, &npd, nullptr));
    std::vector<VkPhysicalDevice> pds(npd);
    YVO_VK_CHECK(vkEnumeratePhysicalDevices(inst.h, &npd, pds.data()));
    VkPhysicalDevice pd = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties props{};
    for (VkPhysicalDevice p : pds) {
        VkPhysicalDeviceProperties pp{};
        vkGetPhysicalDeviceProperties(p, &pp);
        std::printf("physical device: %s (type %d)\n", pp.deviceName, static_cast<int>(pp.deviceType));
        if (pd == VK_NULL_HANDLE && pp.deviceType != VK_PHYSICAL_DEVICE_TYPE_CPU &&
            std::strstr(pp.deviceName, opt.device_substr.c_str()) != nullptr) {
            pd = p;
            props = pp;
        }
    }
    if (pd == VK_NULL_HANDLE) throw std::runtime_error("no non-CPU physical device matching '" + opt.device_substr + "'");
    std::printf("selected: %s, apiVersion %u.%u.%u, driverVersion 0x%x, timestampPeriod %g ns\n", props.deviceName,
                VK_API_VERSION_MAJOR(props.apiVersion), VK_API_VERSION_MINOR(props.apiVersion),
                VK_API_VERSION_PATCH(props.apiVersion), props.driverVersion,
                static_cast<double>(props.limits.timestampPeriod));

    // Extension support
    std::uint32_t next = 0;
    YVO_VK_CHECK(vkEnumerateDeviceExtensionProperties(pd, nullptr, &next, nullptr));
    std::vector<VkExtensionProperties> exts(next);
    YVO_VK_CHECK(vkEnumerateDeviceExtensionProperties(pd, nullptr, &next, exts.data()));
    std::uint32_t ext_rev = 0;
    bool has_ext = false;
    for (const auto& e : exts)
        if (std::strcmp(e.extensionName, VK_KHR_PERFORMANCE_QUERY_EXTENSION_NAME) == 0) {
            has_ext = true;
            ext_rev = e.specVersion;
        }
    if (!has_ext) throw std::runtime_error("VK_KHR_performance_query not supported by the selected device");

    VkPhysicalDevicePerformanceQueryFeaturesKHR pqf{};
    pqf.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PERFORMANCE_QUERY_FEATURES_KHR;
    VkPhysicalDeviceVulkan12Features f12{};
    f12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    f12.pNext = &pqf;
    VkPhysicalDeviceFeatures2 f2{};
    f2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    f2.pNext = &f12;
    vkGetPhysicalDeviceFeatures2(pd, &f2);
    VkPhysicalDevicePerformanceQueryPropertiesKHR pqp{};
    pqp.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PERFORMANCE_QUERY_PROPERTIES_KHR;
    VkPhysicalDeviceProperties2 p2{};
    p2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    p2.pNext = &pqp;
    vkGetPhysicalDeviceProperties2(pd, &p2);
    std::printf("%s rev %u: performanceCounterQueryPools=%u performanceCounterMultipleQueryPools=%u "
                "allowCommandBufferQueryCopies=%u hostQueryReset=%u\n",
                VK_KHR_PERFORMANCE_QUERY_EXTENSION_NAME, ext_rev, pqf.performanceCounterQueryPools,
                pqf.performanceCounterMultipleQueryPools, pqp.allowCommandBufferQueryCopies, f12.hostQueryReset);

    PerfFns fn;
    fn.enumerate = reinterpret_cast<PFN_vkEnumeratePhysicalDeviceQueueFamilyPerformanceQueryCountersKHR>(
        vkGetInstanceProcAddr(inst.h, "vkEnumeratePhysicalDeviceQueueFamilyPerformanceQueryCountersKHR"));
    fn.passes = reinterpret_cast<PFN_vkGetPhysicalDeviceQueueFamilyPerformanceQueryPassesKHR>(
        vkGetInstanceProcAddr(inst.h, "vkGetPhysicalDeviceQueueFamilyPerformanceQueryPassesKHR"));
    if (!fn.enumerate || !fn.passes) throw std::runtime_error("performance query entry points not found");

    std::uint32_t nqf = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &nqf, nullptr);
    std::vector<VkQueueFamilyProperties> qfs(nqf);
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &nqf, qfs.data());

    if (opt.load_sec > 0 || !opt.exp.empty()) {
        // Experiments and load use the first compute queue family.
        std::uint32_t qf = UINT32_MAX;
        for (std::uint32_t i = 0; i < nqf && qf == UINT32_MAX; ++i)
            if (qfs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) qf = i;
        if (qf == UINT32_MAX) throw std::runtime_error("no compute queue family");
        if (opt.load_sec > 0) {
            run_load(pd, qf, opt.load_sec);
            return 0;
        }
        ExpCtx x;
        x.pd = pd;
        x.qf = qf;
        x.fn = fn;
        x.opt = &opt;
        x.all = enumerate_counters(fn, pd, qf);
        x.write_csv = !opt.exp_csv.empty();
        if (x.write_csv) {
            x.csv.open(opt.exp_csv, std::ios::binary);  // binary → LF line endings as written
            if (!x.csv) throw std::runtime_error("cannot open CSV: " + opt.exp_csv);
            x.csv << "exp,label,config,iter,metric,value\n";
        }
        std::printf("queue family %u, %zu counters enumerated, label %s\n", qf, x.all.size(), opt.label.c_str());
        if (opt.exp == "granularity") exp_granularity(x);
        else if (opt.exp == "contention") exp_contention(x);
        else exp_overhead(x);
        std::printf("\nheap_sum GPU/CPU mismatches (warm-up included): %u\n", x.sum_failures);
        if (x.write_csv) {
            x.csv.flush();
            if (!x.csv) throw std::runtime_error("CSV write failed: " + opt.exp_csv);
            std::printf("CSV written: %s\n", opt.exp_csv.c_str());
        }
        return x.sum_failures == 0 ? 0 : 2;
    }

    std::ofstream csv;
    const bool write_csv = !opt.csv_path.empty();
    if (write_csv) {
        csv.open(opt.csv_path, std::ios::binary);  // binary → LF line endings as written
        if (!csv) throw std::runtime_error("cannot open CSV: " + opt.csv_path);
        csv << "queue_family,index,name,category,unit,scope,storage,flags,passes_alone,uuid,description\n";
    }

    bool sanity_done = false;
    for (std::uint32_t qf = 0; qf < nqf; ++qf) {
        std::printf("\n== Queue family %u: flags=%s, queueCount=%u ==\n", qf, queue_flags_str(qfs[qf].queueFlags).c_str(),
                    qfs[qf].queueCount);
        if (!(qfs[qf].queueFlags & VK_QUEUE_COMPUTE_BIT)) {
            std::printf("(no COMPUTE bit — skipped)\n");
            continue;
        }
        const std::vector<Counter> cs = enumerate_counters(fn, pd, qf);
        std::printf("counters: %zu\n\n", cs.size());

        std::vector<std::uint32_t> alone(cs.size());
        std::map<std::string, int> by_cat, by_unit, by_scope, by_storage, by_flags;
        for (std::uint32_t i = 0; i < cs.size(); ++i) {
            const Counter& c = cs[i];
            alone[i] = num_passes(fn, pd, qf, {i});
            std::printf("[%3u] %s\n", i, c.d.name);
            std::printf("      category=%s unit=%s scope=%s storage=%s flags=%s passes_alone=%u\n", c.d.category,
                        unit_str(c.c.unit), scope_str(c.c.scope), storage_str(c.c.storage),
                        flags_str(c.d.flags).c_str(), alone[i]);
            std::printf("      uuid=%s\n", uuid_str(c.c.uuid).c_str());
            std::printf("      %s\n", c.d.description);
            ++by_cat[c.d.category];
            ++by_unit[unit_str(c.c.unit)];
            ++by_scope[scope_str(c.c.scope)];
            ++by_storage[storage_str(c.c.storage)];
            ++by_flags[flags_str(c.d.flags)];
            if (write_csv) {
                csv << qf << ',' << i << ',' << csv_field(c.d.name) << ',' << csv_field(c.d.category) << ','
                    << unit_str(c.c.unit) << ',' << scope_str(c.c.scope) << ',' << storage_str(c.c.storage) << ','
                    << flags_str(c.d.flags) << ',' << alone[i] << ',' << uuid_str(c.c.uuid) << ','
                    << csv_field(c.d.description) << '\n';
            }
        }

        auto dump = [](const char* title, const std::map<std::string, int>& m) {
            std::printf("  by %-9s:", title);
            for (const auto& [k, v] : m) std::printf(" %s=%d", k.empty() ? "(empty)" : k.c_str(), v);
            std::printf("\n");
        };
        std::printf("\n-- Summary (queue family %u) --\n", qf);
        std::printf("  total    : %zu\n", cs.size());
        dump("category", by_cat);
        dump("unit", by_unit);
        dump("scope", by_scope);
        dump("storage", by_storage);
        dump("flags", by_flags);

        if (cs.empty()) continue;

        // Pass analysis
        std::map<std::uint32_t, int> alone_hist;
        for (std::uint32_t p : alone) ++alone_hist[p];
        std::vector<std::uint32_t> all(cs.size());
        for (std::uint32_t i = 0; i < cs.size(); ++i) all[i] = i;
        const std::uint32_t all_passes = num_passes(fn, pd, qf, all);

        std::vector<std::uint32_t> prefix;
        for (std::uint32_t i = 0; i < cs.size(); ++i) {
            prefix.push_back(i);
            if (num_passes(fn, pd, qf, prefix) != 1) {
                prefix.pop_back();
                break;
            }
        }
        std::vector<std::uint32_t> greedy;
        for (std::uint32_t i = 0; i < cs.size(); ++i) {
            greedy.push_back(i);
            if (num_passes(fn, pd, qf, greedy) != 1) greedy.pop_back();
        }
        // Pass count as a function of set size (index-order prefixes) — shows the per-pass capacity.
        std::printf("\n-- Pass analysis (queue family %u) --\n", qf);
        std::printf("  (a) passes per single counter:");
        for (const auto& [p, n] : alone_hist) std::printf(" %u pass=%d counters", p, n);
        std::printf("\n");
        std::printf("  (b) all %zu counters: %u pass(es)\n", cs.size(), all_passes);
        std::printf("  (c) longest index-order prefix in 1 pass: %zu counters (indices 0..%zu)\n", prefix.size(),
                    prefix.empty() ? 0 : prefix.size() - 1);
        std::printf("      skip-greedy 1-pass set (index order): %zu counters\n", greedy.size());
        std::printf("  passes vs prefix length:");
        std::uint32_t last = 0;
        for (std::uint32_t n = 1; n <= cs.size(); ++n) {
            std::vector<std::uint32_t> pre(all.begin(), all.begin() + n);
            const std::uint32_t p = num_passes(fn, pd, qf, pre);
            if (p != last) {
                std::printf(" n=%u→%u", n, p);
                last = p;
            }
        }
        std::printf("\n");

        if (opt.sanity && !sanity_done) {
            run_sanity(pd, qf, fn, cs, opt.sanity_counters.empty() ? prefix : opt.sanity_counters);
            sanity_done = true;
        }
    }
    if (write_csv) {
        csv.flush();
        if (!csv) throw std::runtime_error("CSV write failed: " + opt.csv_path);
    }
    if (write_csv) std::printf("\nCSV written: %s\n", opt.csv_path.c_str());
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    try {
        return run(parse_args(argc, argv));
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
