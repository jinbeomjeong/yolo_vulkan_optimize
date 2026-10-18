#include "display/drm_state.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>

namespace yvo::display {

namespace {

// DRM fourcc (only the values are kept, to avoid pulling in drm_fourcc.h)
constexpr std::uint32_t fourcc(char a, char b, char c, char d) {
    return static_cast<std::uint32_t>(static_cast<unsigned char>(a)) |
           (static_cast<std::uint32_t>(static_cast<unsigned char>(b)) << 8) |
           (static_cast<std::uint32_t>(static_cast<unsigned char>(c)) << 16) |
           (static_cast<std::uint32_t>(static_cast<unsigned char>(d)) << 24);
}
constexpr std::uint32_t kNv12 = fourcc('N', 'V', '1', '2');
constexpr std::uint32_t kAb24 = fourcc('A', 'B', '2', '4');
// Broadcom modifier: vendor 0x07 (upper 8 bits), lower 8 bits = type (SAND128 = 4), the 48 bits in between = column height parameter.
constexpr std::uint64_t kVendorBroadcom = 0x07;
constexpr std::uint64_t kBroadcomSand128 = 4;

bool starts_with(std::string_view s, std::string_view p) { return s.substr(0, p.size()) == p; }

std::string_view trim_left(std::string_view s) {
    const auto p = s.find_first_not_of(" \t");
    return p == std::string_view::npos ? std::string_view{} : s.substr(p);
}

std::string value_after_eq(std::string_view t) {
    const auto p = t.find('=');
    if (p == std::string_view::npos) return {};
    std::string v(t.substr(p + 1));
    // some lines have spaces around '=', like "allocated by = labwc"
    while (!v.empty() && (v.front() == ' ' || v.front() == '\t')) v.erase(v.begin());
    while (!v.empty() && (v.back() == ' ' || v.back() == '\r')) v.pop_back();
    return v;
}

std::uint32_t parse_fourcc_paren(const std::string& fmt) {
    // "NV12 little-endian (0x3231564e)"
    const auto p = fmt.find("(0x");
    if (p == std::string::npos) return 0;
    return static_cast<std::uint32_t>(std::strtoul(fmt.c_str() + p + 1, nullptr, 16));
}

bool read_file(const std::string& path, std::string* out) {
    std::ifstream f(path);
    if (!f) return false;
    std::ostringstream ss;
    ss << f.rdbuf();
    *out = ss.str();
    return !f.bad();
}

} // namespace

std::vector<DrmPlaneState> parse_drm_active_planes(std::string_view state) {
    std::vector<DrmPlaneState> out;
    DrmPlaneState cur;
    bool in_plane = false;
    bool have_fb_size = false;
    auto flush = [&]() {
        if (in_plane && !cur.crtc.empty() && cur.crtc != "(null)") out.push_back(cur);
        cur = DrmPlaneState{};
        in_plane = false;
        have_fb_size = false;
    };
    std::size_t pos = 0;
    while (pos <= state.size()) {
        const auto nl = state.find('\n', pos);
        const std::string_view line = state.substr(pos, nl == std::string_view::npos ? std::string_view::npos : nl - pos);
        pos = nl == std::string_view::npos ? state.size() + 1 : nl + 1;

        if (starts_with(line, "plane[")) {
            flush();
            in_plane = true;
            cur.id = std::atoi(std::string(line.substr(6)).c_str());
            const auto colon = line.find(": ");
            if (colon != std::string_view::npos) cur.name = std::string(line.substr(colon + 2));
            continue;
        }
        // an unindented header line of another object (crtc[..], connector[..], ...) ends the plane block
        if (!line.empty() && line.front() != '\t' && line.front() != ' ') {
            flush();
            continue;
        }
        if (!in_plane) continue;
        const std::string_view t = trim_left(line);
        if (starts_with(t, "crtc=")) cur.crtc = value_after_eq(t);
        else if (starts_with(t, "allocated by")) cur.alloc = value_after_eq(t);
        else if (starts_with(t, "format=")) {
            cur.format = value_after_eq(t);
            cur.fourcc = parse_fourcc_paren(cur.format);
        } else if (starts_with(t, "modifier=")) {
            cur.modifier_str = value_after_eq(t);
            cur.modifier = std::strtoull(cur.modifier_str.c_str(), nullptr, 16);
        } else if (starts_with(t, "size=") && !have_fb_size) {
            // the first "size=" is the fb size. The later obj block also has "size=<bytes>".
            cur.size = value_after_eq(t);
            have_fb_size = true;
        } else if (starts_with(t, "crtc-pos=")) cur.crtc_pos = value_after_eq(t);
        else if (starts_with(t, "src-pos=")) cur.src_pos = value_after_eq(t);
        else if (starts_with(t, "normalized-zpos=")) cur.zpos = std::atoi(value_after_eq(t).c_str());
        else if (starts_with(t, "color-encoding=")) cur.encoding = value_after_eq(t);
        else if (starts_with(t, "color-range=")) cur.range = value_after_eq(t);
    }
    flush();
    return out;
}

bool is_nv12_sand128(const DrmPlaneState& p) {
    const bool nv12 = p.fourcc == kNv12 || starts_with(p.format, "NV12");
    const std::uint64_t vendor = p.modifier >> 56;
    const std::uint64_t kind = p.modifier & 0xffu;
    return nv12 && vendor == kVendorBroadcom && kind == kBroadcomSand128;
}

bool is_ab24_linear(const DrmPlaneState& p) {
    const bool ab24 = p.fourcc == kAb24 || starts_with(p.format, "AB24");
    return ab24 && p.modifier == 0;
}

PlaneScanoutVerdict judge_plane_scanout(const std::vector<DrmPlaneState>& planes) {
    PlaneScanoutVerdict v;
    v.active_planes = static_cast<int>(planes.size());
    const DrmPlaneState* video = nullptr;
    for (const auto& p : planes)
        if (is_nv12_sand128(p) && (video == nullptr || p.zpos < video->zpos)) video = &p;
    if (video != nullptr) {
        v.video_on_plane = true;
        v.video_plane = video->id;
        v.video_zpos = video->zpos;
        v.encoding_bt709 = video->encoding.find("BT.709") != std::string::npos;
        v.range_limited = video->range.find("limited") != std::string::npos;
        // the lowest AB24 above the video (= the box plane right above the video)
        const DrmPlaneState* ov = nullptr;
        for (const auto& p : planes)
            if (is_ab24_linear(p) && p.zpos > video->zpos && (ov == nullptr || p.zpos < ov->zpos)) ov = &p;
        if (ov != nullptr) {
            v.overlay_above = true;
            v.overlay_plane = ov->id;
            v.overlay_zpos = ov->zpos;
        }
    }
    char buf[320];
    if (video != nullptr) {
        const std::string boxes = v.overlay_above ? "plane[" + std::to_string(v.overlay_plane) +
                                                        "] AB24 zpos=" + std::to_string(v.overlay_zpos)
                                                  : std::string("none above video");
        std::snprintf(buf, sizeof(buf), "active=%d video=plane[%d] NV12/SAND128 zpos=%d enc='%s' range='%s' | boxes=%s",
                      v.active_planes, v.video_plane, v.video_zpos, video->encoding.c_str(), video->range.c_str(),
                      boxes.c_str());
    } else {
        std::snprintf(buf, sizeof(buf), "active=%d no NV12/SAND128 plane (compositor GL composition or not shown)",
                      v.active_planes);
    }
    v.text = buf;
    return v;
}

std::string describe_planes(const std::vector<DrmPlaneState>& planes) {
    std::string s;
    for (const auto& p : planes) {
        s += "  plane[" + std::to_string(p.id) + "] " + p.name + " crtc=" + p.crtc + " alloc=" + p.alloc +
             " fmt=" + p.format + " mod=" + p.modifier_str + " size=" + p.size + " crtc-pos=" + p.crtc_pos +
             " zpos=" + std::to_string(p.zpos) + " enc=" + p.encoding + " range=" + p.range + "\n";
    }
    return s;
}

long long parse_hvs_underrun(std::string_view text) {
    const std::string s(trim_left(text));
    if (s.empty() || s.front() < '0' || s.front() > '9') return -1;
    return std::strtoll(s.c_str(), nullptr, 10);
}

DrmStateSnapshot take_drm_state_snapshot(const std::string& dri_debug_dir) {
    DrmStateSnapshot snap;
    std::string st;
    if (!read_file(dri_debug_dir + "/state", &st) || st.empty()) {
        snap.error = dri_debug_dir + "/state could not be read (debugfs permission: root:sudo 0750)";
        return snap;
    }
    snap.readable = true;
    snap.planes = parse_drm_active_planes(st);
    snap.verdict = judge_plane_scanout(snap.planes);
    std::string ur;
    if (read_file(dri_debug_dir + "/hvs_underrun", &ur)) snap.hvs_underrun = parse_hvs_underrun(ur);
    return snap;
}

} // namespace yvo::display
