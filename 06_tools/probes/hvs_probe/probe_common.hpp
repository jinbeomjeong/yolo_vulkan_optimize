// hvs_probe 공통 도구 (헤더 전용). 본체(yvo_pipeline)는 이 파일을 쓰지 않는다.
//   - 시간·CPU 측정, V3D gpu_stats 읽기, 다른 프로세스(labwc) CPU 시간
//   - debugfs DRM state 스냅샷과 평면 배치 해석
//   - detections.jsonl 재생(프레임별 검출)
//   - SAND128 NV12 → RGB CPU 기준 변환(색 비교용), PPM 쓰기
#pragma once

#include <dirent.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "common/dma_heap_alloc.hpp"
#include "common/frame_types.hpp"

namespace hvsprobe {

inline std::int64_t now_ns(clockid_t clk = CLOCK_MONOTONIC) {
    timespec ts{};
    clock_gettime(clk, &ts);
    return static_cast<std::int64_t>(ts.tv_sec) * 1000000000LL + ts.tv_nsec;
}

inline double ms(std::int64_t ns) { return static_cast<double>(ns) / 1e6; }

// 이 프로세스의 CPU 시간(user+sys, 모든 스레드)
inline std::int64_t self_cpu_ns() { return now_ns(CLOCK_PROCESS_CPUTIME_ID); }

inline std::string read_text(const std::string& path) {
    std::ifstream f(path);
    if (!f) return {};
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

inline bool write_text(const std::string& path, const std::string& s) {
    FILE* f = std::fopen(path.c_str(), "w");
    if (!f) return false;
    const bool ok = std::fwrite(s.data(), 1, s.size(), f) == s.size();
    return std::fclose(f) == 0 && ok;
}

// ── V3D 누적 통계 (/sys/class/drm/renderD128/device/gpu_stats) ──────────────────────────────
// 형식: "queue timestamp jobs runtime" (runtime 은 ns 누적, 시스템 전체 = 모든 프로세스 합)
struct V3dQueue {
    std::uint64_t jobs = 0;
    std::uint64_t runtime_ns = 0;
};
using V3dStats = std::map<std::string, V3dQueue>;

inline const char* kV3dStatsPath = "/sys/class/drm/renderD128/device/gpu_stats";

inline V3dStats read_v3d_stats(const std::string& path = kV3dStatsPath) {
    V3dStats s;
    std::istringstream in(read_text(path));
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream ls(line);
        std::string q;
        unsigned long long ts = 0, jobs = 0, rt = 0;
        if (!(ls >> q >> ts >> jobs >> rt)) continue; // 머리줄은 숫자가 아니라 건너뛴다
        s[q] = V3dQueue{jobs, rt};
    }
    return s;
}

// 구간 [a, b] 동안 큐별 사용률(%)과 job 수를 한 줄로
inline std::string v3d_delta_str(const V3dStats& a, const V3dStats& b, std::int64_t wall_ns, double frames,
                                 double* render_ms_per_frame = nullptr, double* csd_ms_per_frame = nullptr) {
    std::string out;
    char buf[160];
    for (const auto& [q, vb] : b) {
        auto it = a.find(q);
        if (it == a.end()) continue;
        const double drt = static_cast<double>(vb.runtime_ns - it->second.runtime_ns);
        const double djobs = static_cast<double>(vb.jobs - it->second.jobs);
        const double busy = wall_ns > 0 ? 100.0 * drt / static_cast<double>(wall_ns) : 0.0;
        const double per_frame = frames > 0 ? drt / 1e6 / frames : 0.0;
        std::snprintf(buf, sizeof(buf), "%s=%.2f%%(%.3fms/f,%.0fjobs) ", q.c_str(), busy, per_frame, djobs);
        out += buf;
        if (q == "render" && render_ms_per_frame) *render_ms_per_frame = per_frame;
        if (q == "csd" && csd_ms_per_frame) *csd_ms_per_frame = per_frame;
    }
    return out;
}

// ── 다른 프로세스의 CPU 시간 ────────────────────────────────────────────────────────────────
inline int find_pid_by_comm(const std::string& comm) {
    DIR* d = opendir("/proc");
    if (!d) return -1;
    int found = -1;
    while (dirent* e = readdir(d)) {
        const int pid = std::atoi(e->d_name);
        if (pid <= 0) continue;
        std::string c = read_text("/proc/" + std::to_string(pid) + "/comm");
        while (!c.empty() && (c.back() == '\n' || c.back() == ' ')) c.pop_back();
        if (c == comm) {
            found = pid;
            break;
        }
    }
    closedir(d);
    return found;
}

// /proc/<pid>/stat 의 utime+stime(ns). 실패하면 -1.
inline std::int64_t proc_cpu_ns(int pid) {
    if (pid <= 0) return -1;
    const std::string s = read_text("/proc/" + std::to_string(pid) + "/stat");
    const auto rp = s.rfind(')');
    if (rp == std::string::npos) return -1;
    std::istringstream in(s.substr(rp + 2));
    std::string tok;
    unsigned long long ut = 0, st = 0;
    // ')' 다음 필드는 3번(state)부터. utime=14, stime=15 → 3..13 을 건너뛴다(11개)
    for (int i = 3; i <= 13; ++i) in >> tok;
    in >> ut >> st;
    const long hz = sysconf(_SC_CLK_TCK);
    return static_cast<std::int64_t>((ut + st) * (1000000000ULL / static_cast<unsigned long long>(hz > 0 ? hz : 100)));
}

// ── debugfs state 해석 ─────────────────────────────────────────────────────────────────────
struct PlaneState {
    std::string header;   // "plane[83]: plane-2"
    int plane_id = 0;
    std::string crtc, alloc, format, modifier, size, crtc_pos, src_pos, zpos, encoding, range;
};

inline std::vector<PlaneState> parse_active_planes(const std::string& state) {
    std::vector<PlaneState> out;
    std::istringstream in(state);
    std::string line;
    PlaneState cur;
    bool in_plane = false;
    auto flush = [&]() {
        if (in_plane && !cur.crtc.empty() && cur.crtc != "(null)") out.push_back(cur);
        cur = PlaneState{};
        in_plane = false;
    };
    auto val = [](const std::string& t) {
        const auto p = t.find('=');
        return p == std::string::npos ? std::string() : t.substr(p + 1);
    };
    while (std::getline(in, line)) {
        std::string t = line;
        t.erase(0, t.find_first_not_of(" \t"));
        if (line.rfind("plane[", 0) == 0) {
            flush();
            in_plane = true;
            cur.header = line;
            cur.plane_id = std::atoi(line.c_str() + 6);
        } else if (line.rfind("crtc[", 0) == 0 || line.rfind("connector[", 0) == 0) {
            flush();
        } else if (in_plane) {
            if (t.rfind("crtc=", 0) == 0) cur.crtc = val(t);
            else if (t.rfind("allocated by", 0) == 0) cur.alloc = val(t);
            else if (t.rfind("format=", 0) == 0) cur.format = val(t);
            else if (t.rfind("modifier=", 0) == 0) cur.modifier = val(t);
            else if (t.rfind("size=", 0) == 0 && cur.size.empty()) cur.size = val(t);
            else if (t.rfind("crtc-pos=", 0) == 0) cur.crtc_pos = val(t);
            else if (t.rfind("src-pos=", 0) == 0) cur.src_pos = val(t);
            else if (t.rfind("normalized-zpos=", 0) == 0) cur.zpos = val(t);
            else if (t.rfind("color-encoding=", 0) == 0) cur.encoding = val(t);
            else if (t.rfind("color-range=", 0) == 0) cur.range = val(t);
        }
    }
    flush();
    return out;
}

inline std::string describe_planes(const std::vector<PlaneState>& ps) {
    std::string s;
    for (const auto& p : ps) {
        s += "  " + p.header + " crtc=" + p.crtc + " alloc=" + p.alloc + " fmt=" + p.format + " mod=" + p.modifier +
             " size=" + p.size + " crtc-pos=" + p.crtc_pos + " src-pos=" + p.src_pos + " zpos=" + p.zpos +
             " enc=" + p.encoding + " range=" + p.range + "\n";
    }
    return s;
}

// 평면 배치 판정: NV12 fb 가 올라간 평면이 있으면 direct scanout(HVS 가 YUV 를 읽는다)
struct PlacementVerdict {
    int nv12_planes = 0;
    int argb_planes = 0;
    int total = 0;
    std::string text;
};

inline PlacementVerdict judge_placement(const std::vector<PlaneState>& ps) {
    PlacementVerdict v;
    v.total = static_cast<int>(ps.size());
    for (const auto& p : ps) {
        if (p.format.rfind("NV12", 0) == 0) ++v.nv12_planes;
        else ++v.argb_planes;
    }
    char buf[256];
    std::snprintf(buf, sizeof(buf), "active_planes=%d nv12_planes=%d rgb_planes=%d -> %s", v.total, v.nv12_planes,
                  v.argb_planes,
                  v.nv12_planes > 0 ? "NV12 on HW plane (HVS scanout)" : "no NV12 plane (compositor GL composition)");
    v.text = buf;
    return v;
}

// ── detections.jsonl 재생 ──────────────────────────────────────────────────────────────────
// 형식: {"frame":N,"detections":[{"class":c,"name":"..","score":s,"box":[x1,y1,x2,y2]},...]}
inline std::map<std::int64_t, std::vector<yvo::Detection>> load_detections(const std::string& path, std::size_t* total) {
    std::map<std::int64_t, std::vector<yvo::Detection>> out;
    std::ifstream f(path);
    std::string line;
    std::size_t n = 0;
    while (std::getline(f, line)) {
        long long frame = -1;
        if (std::sscanf(line.c_str(), "{\"frame\":%lld", &frame) != 1) continue;
        auto& v = out[frame];
        std::size_t pos = 0;
        while ((pos = line.find("{\"class\":", pos)) != std::string::npos) {
            yvo::Detection d;
            int c = 0;
            std::sscanf(line.c_str() + pos, "{\"class\":%d", &c);
            d.class_id = c;
            const auto sp = line.find("\"score\":", pos);
            const auto bp = line.find("\"box\":[", pos);
            if (sp == std::string::npos || bp == std::string::npos) break;
            d.score = std::strtof(line.c_str() + sp + 8, nullptr);
            std::sscanf(line.c_str() + bp + 7, "%f,%f,%f,%f", &d.x1, &d.y1, &d.x2, &d.y2);
            v.push_back(d);
            ++n;
            pos = bp + 7;
        }
    }
    if (total) *total = n;
    return out;
}

// ── 통계 요약 한 줄 ────────────────────────────────────────────────────────────────────────
struct Summary {
    double mean = 0, median = 0, p90 = 0, p99 = 0, min = 0, max = 0;
    std::size_t n = 0;
};

inline Summary summarize(std::vector<double> v) {
    Summary s;
    s.n = v.size();
    if (v.empty()) return s;
    std::sort(v.begin(), v.end());
    auto pct = [&](double q) {
        if (v.size() == 1) return v[0];
        const double idx = q * static_cast<double>(v.size() - 1);
        const auto lo = static_cast<std::size_t>(idx);
        const std::size_t hi = std::min(lo + 1, v.size() - 1);
        return v[lo] + (v[hi] - v[lo]) * (idx - static_cast<double>(lo));
    };
    double sum = 0;
    for (double x : v) sum += x;
    s.mean = sum / static_cast<double>(v.size());
    s.median = pct(0.5);
    s.p90 = pct(0.9);
    s.p99 = pct(0.99);
    s.min = v.front();
    s.max = v.back();
    return s;
}

inline std::string summary_str(const Summary& s) {
    char b[200];
    std::snprintf(b, sizeof(b), "mean %.3f median %.3f p90 %.3f p99 %.3f min %.3f max %.3f (n=%zu)", s.mean, s.median,
                  s.p90, s.p99, s.min, s.max, s.n);
    return b;
}

// ── SAND128 NV12 → RGB CPU 기준 변환 (색 비교용) ──────────────────────────────────────────────
// display_convert.comp 1:1 경로와 같은 수식: BT.709 limited → full, 크로마 left-sited
// (가로: 짝수 dx 는 그 열, 홀수 dx 는 좌우 평균 / 세로: cy = (dy+0.5)/2-0.5 선형 보간).
// dma-buf 를 읽기 전용으로 매핑한다(DMA_BUF_SYNC READ 로 감싼다). 실패하면 빈 벡터.
inline std::vector<std::uint8_t> sand_to_rgb(const yvo::DrmFrame& f) {
    std::vector<std::uint8_t> rgb;
    const std::size_t ysz = f.size;
    const std::size_t uvsz = f.uv_size;
    void* ym = mmap(nullptr, ysz, PROT_READ, MAP_SHARED, f.fd, 0);
    if (ym == MAP_FAILED) return rgb;
    void* uvm = (f.uv_fd == f.fd) ? ym : mmap(nullptr, uvsz, PROT_READ, MAP_SHARED, f.uv_fd, 0);
    if (uvm == MAP_FAILED) {
        munmap(ym, ysz);
        return rgb;
    }
    yvo::dma_sync_begin(f.fd, yvo::DmaSyncDir::Read, false);
    if (f.uv_fd != f.fd) yvo::dma_sync_begin(f.uv_fd, yvo::DmaSyncDir::Read, false);
    const auto* Y = static_cast<const std::uint8_t*>(ym) + f.y_offset;
    const auto* UV = static_cast<const std::uint8_t*>(uvm) + f.uv_offset;
    const std::size_t ycs = 128u * f.sand_col_height;
    const std::size_t uvcs = 128u * f.uv_sand_col_height;
    const int w = f.width, h = f.height, ch = h / 2;
    rgb.resize(static_cast<std::size_t>(w) * static_cast<std::size_t>(h) * 3);
    auto yat = [&](int x, int y) { return Y[(x / 128) * ycs + static_cast<std::size_t>(y) * 128 + (x % 128)]; };
    auto uvat = [&](int cx, int cy, int c) {
        const int bx = cx * 2 + c;
        return UV[(bx / 128) * uvcs + static_cast<std::size_t>(cy) * 128 + (bx % 128)];
    };
    auto to_u8 = [](float v) { return static_cast<std::uint8_t>(std::clamp(v, 0.0f, 255.0f) + 0.5f); };
    for (int dy = 0; dy < h; ++dy) {
        const float cy = std::clamp((static_cast<float>(dy) + 0.5f) * 0.5f - 0.5f, 0.0f, static_cast<float>(ch - 1));
        const int cy0 = static_cast<int>(cy);
        const int cy1 = std::min(cy0 + 1, ch - 1);
        const float fcy = cy - static_cast<float>(cy0);
        for (int dx = 0; dx < w; ++dx) {
            const int cx0 = dx / 2;
            const int cx1 = std::min(cx0 + 1, w / 2 - 1);
            const float fcx = (dx & 1) ? 0.5f : 0.0f;
            float uvv[2];
            for (int c = 0; c < 2; ++c) {
                const float a = uvat(cx0, cy0, c), b = uvat(cx1, cy0, c);
                const float cc = uvat(cx0, cy1, c), d = uvat(cx1, cy1, c);
                const float top = a + (b - a) * fcx;
                const float bot = cc + (d - cc) * fcx;
                uvv[c] = top + (bot - top) * fcy;
            }
            const float yf = (static_cast<float>(yat(dx, dy)) - 16.0f) * (255.0f / 219.0f);
            const float uf = (uvv[0] - 128.0f) * (255.0f / 224.0f);
            const float vf = (uvv[1] - 128.0f) * (255.0f / 224.0f);
            std::uint8_t* p = &rgb[(static_cast<std::size_t>(dy) * static_cast<std::size_t>(w) + static_cast<std::size_t>(dx)) * 3];
            p[0] = to_u8(yf + 1.5748f * vf);
            p[1] = to_u8(yf - 0.187324f * uf - 0.468124f * vf);
            p[2] = to_u8(yf + 1.8556f * uf);
        }
    }
    if (f.uv_fd != f.fd) yvo::dma_sync_end(f.uv_fd, yvo::DmaSyncDir::Read, false);
    yvo::dma_sync_end(f.fd, yvo::DmaSyncDir::Read, false);
    if (uvm != ym) munmap(uvm, uvsz);
    munmap(ym, ysz);
    return rgb;
}

inline bool write_ppm(const std::string& path, const std::uint8_t* rgb, int w, int h) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    std::fprintf(f, "P6\n%d %d\n255\n", w, h);
    const std::size_t n = static_cast<std::size_t>(w) * static_cast<std::size_t>(h) * 3;
    const bool ok = std::fwrite(rgb, 1, n, f) == n;
    return std::fclose(f) == 0 && ok;
}

} // namespace hvsprobe
