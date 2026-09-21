// hvs_caps: card1(vc4 KMS/HVS6)의 평면 능력을 읽기 전용으로 덤프한다.
//   - 평면별 IN_FORMATS blob 을 drmModeFormatModifierBlobIterNext 로 해독
//   - NV12/P030 x SAND128, AR24 x LINEAR 지원 평면 목록, zpos 범위, 색 속성(enum) 목록
//   - 커넥터 목록(Writeback 포함), CRTC 상태
//   - debugfs state 요약(평면별 crtc/fb/format/modifier/zpos)
// DRM master 가 필요 없다(labwc 실행 중에도 된다). 아무것도 바꾸지 않는다.
//
// 사용: hvs_caps [--card /dev/dri/card1] [--state /sys/kernel/debug/dri/1/state]
#include <drm_fourcc.h>
#include <fcntl.h>
#include <unistd.h>
#include <xf86drm.h>
#include <xf86drmMode.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace {

std::string fourcc_str(std::uint32_t f) {
    char s[5] = {static_cast<char>(f & 0xff), static_cast<char>((f >> 8) & 0xff),
                 static_cast<char>((f >> 16) & 0xff), static_cast<char>((f >> 24) & 0xff), 0};
    return s;
}

struct PlaneInfo {
    std::uint32_t id = 0;
    std::uint32_t possible_crtcs = 0;
    std::string type = "?";
    std::uint64_t zpos_min = 0, zpos_max = 0, zpos_cur = 0;
    bool zpos_immutable = false;
    bool has_zpos = false;
    std::vector<std::string> enums;  // "COLOR_ENCODING: ..." 같은 사람이 읽을 줄
    std::map<std::uint32_t, std::set<std::uint64_t>> fmt_mods; // format -> modifiers
    std::vector<std::string> prop_names;
};

bool mod_is_sand128(std::uint64_t m) { return fourcc_mod_broadcom_mod(m) == DRM_FORMAT_MOD_BROADCOM_SAND128; }

} // namespace

int main(int argc, char** argv) {
    std::string card = "/dev/dri/card1";
    std::string state_path = "/sys/kernel/debug/dri/1/state";
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--card" && i + 1 < argc) card = argv[++i];
        else if (a == "--state" && i + 1 < argc) state_path = argv[++i];
        else {
            std::fprintf(stderr, "usage: %s [--card PATH] [--state PATH]\n", argv[0]);
            return 2;
        }
    }

    const int fd = open(card.c_str(), O_RDWR | O_CLOEXEC);
    if (fd < 0) {
        std::fprintf(stderr, "open %s: %s\n", card.c_str(), std::strerror(errno));
        return 1;
    }
    if (drmSetClientCap(fd, DRM_CLIENT_CAP_UNIVERSAL_PLANES, 1) != 0)
        std::fprintf(stderr, "[warn] UNIVERSAL_PLANES cap 실패: %s\n", std::strerror(errno));
    const int atomic_ok = drmSetClientCap(fd, DRM_CLIENT_CAP_ATOMIC, 1);
    if (drmSetClientCap(fd, DRM_CLIENT_CAP_WRITEBACK_CONNECTORS, 1) != 0)
        std::fprintf(stderr, "[warn] WRITEBACK_CONNECTORS cap 실패: %s\n", std::strerror(errno));

    drmVersionPtr ver = drmGetVersion(fd);
    if (ver) {
        std::printf("driver          %s (%s) %d.%d.%d\n", ver->name, ver->desc, ver->version_major,
                    ver->version_minor, ver->version_patchlevel);
        drmFreeVersion(ver);
    }
    std::printf("atomic cap      %s\n", atomic_ok == 0 ? "yes" : "no");
    std::uint64_t cap = 0;
    if (drmGetCap(fd, DRM_CAP_ADDFB2_MODIFIERS, &cap) == 0)
        std::printf("ADDFB2_MODIFIERS %llu\n", static_cast<unsigned long long>(cap));
    // master 여부: drmIsMaster 는 labwc 가 master 인 동안 0 이어야 한다.
    std::printf("is master       %s (labwc 실행 중이면 no 가 정상)\n", drmIsMaster(fd) ? "yes" : "no");

    drmModeResPtr res = drmModeGetResources(fd);
    if (!res) {
        std::fprintf(stderr, "drmModeGetResources: %s\n", std::strerror(errno));
        close(fd);
        return 1;
    }
    std::printf("\n== CRTC (%d) ==\n", res->count_crtcs);
    for (int i = 0; i < res->count_crtcs; ++i) {
        drmModeCrtcPtr c = drmModeGetCrtc(fd, res->crtcs[i]);
        if (!c) continue;
        std::printf("crtc idx %d id %u  fb %u  mode %s %ux%u@%u  active=%d\n", i, c->crtc_id, c->buffer_id,
                    c->mode_valid ? c->mode.name : "-", c->mode_valid ? c->mode.hdisplay : 0,
                    c->mode_valid ? c->mode.vdisplay : 0, c->mode_valid ? c->mode.vrefresh : 0, c->mode_valid);
        drmModeFreeCrtc(c);
    }
    std::printf("\n== CONNECTOR (%d) ==\n", res->count_connectors);
    for (int i = 0; i < res->count_connectors; ++i) {
        drmModeConnectorPtr c = drmModeGetConnector(fd, res->connectors[i]);
        if (!c) continue;
        const char* tn = drmModeGetConnectorTypeName(c->connector_type);
        std::printf("connector id %u  %s-%u  %s  encoder %u  modes %d%s\n", c->connector_id, tn ? tn : "?",
                    c->connector_type_id,
                    c->connection == DRM_MODE_CONNECTED ? "connected" : "disconnected", c->encoder_id,
                    c->count_modes, c->connector_type == DRM_MODE_CONNECTOR_WRITEBACK ? "  <-- WRITEBACK" : "");
        if (c->connector_type == DRM_MODE_CONNECTOR_WRITEBACK) {
            for (int e = 0; e < c->count_encoders; ++e) {
                if (drmModeEncoderPtr enc = drmModeGetEncoder(fd, c->encoders[e])) {
                    std::printf("    encoder %u possible_crtcs 0x%x\n", enc->encoder_id, enc->possible_crtcs);
                    drmModeFreeEncoder(enc);
                }
            }
            drmModeObjectPropertiesPtr props =
                drmModeObjectGetProperties(fd, c->connector_id, DRM_MODE_OBJECT_CONNECTOR);
            for (std::uint32_t p = 0; props && p < props->count_props; ++p) {
                drmModePropertyPtr pr = drmModeGetProperty(fd, props->props[p]);
                if (!pr) continue;
                if (std::strcmp(pr->name, "WRITEBACK_PIXEL_FORMATS") == 0) {
                    drmModePropertyBlobPtr b = drmModeGetPropertyBlob(fd, static_cast<std::uint32_t>(props->prop_values[p]));
                    if (b) {
                        std::printf("    WRITEBACK_PIXEL_FORMATS:");
                        const auto* f = static_cast<const std::uint32_t*>(b->data);
                        for (std::uint32_t k = 0; k < b->length / 4; ++k) std::printf(" %s", fourcc_str(f[k]).c_str());
                        std::printf("\n");
                        drmModeFreePropertyBlob(b);
                    }
                } else {
                    std::printf("    prop %s\n", pr->name);
                }
                drmModeFreeProperty(pr);
            }
            drmModeFreeObjectProperties(props);
        }
        drmModeFreeConnector(c);
    }

    drmModePlaneResPtr pres = drmModeGetPlaneResources(fd);
    if (!pres) {
        std::fprintf(stderr, "drmModeGetPlaneResources: %s\n", std::strerror(errno));
        drmModeFreeResources(res);
        close(fd);
        return 1;
    }
    std::vector<PlaneInfo> planes;
    for (std::uint32_t i = 0; i < pres->count_planes; ++i) {
        drmModePlanePtr pl = drmModeGetPlane(fd, pres->planes[i]);
        if (!pl) continue;
        PlaneInfo pi;
        pi.id = pl->plane_id;
        pi.possible_crtcs = pl->possible_crtcs;
        drmModeObjectPropertiesPtr props = drmModeObjectGetProperties(fd, pl->plane_id, DRM_MODE_OBJECT_PLANE);
        for (std::uint32_t p = 0; props && p < props->count_props; ++p) {
            drmModePropertyPtr pr = drmModeGetProperty(fd, props->props[p]);
            if (!pr) continue;
            const std::string name = pr->name;
            pi.prop_names.push_back(name);
            const std::uint64_t val = props->prop_values[p];
            if (name == "type") {
                for (int e = 0; e < pr->count_enums; ++e)
                    if (pr->enums[e].value == val) pi.type = pr->enums[e].name;
            } else if (name == "zpos") {
                pi.has_zpos = true;
                pi.zpos_immutable = (pr->flags & DRM_MODE_PROP_IMMUTABLE) != 0;
                if (pr->count_values >= 2) {
                    pi.zpos_min = pr->values[0];
                    pi.zpos_max = pr->values[1];
                }
                pi.zpos_cur = val;
            } else if (name == "IN_FORMATS") {
                drmModePropertyBlobPtr blob = drmModeGetPropertyBlob(fd, static_cast<std::uint32_t>(val));
                if (blob) {
                    drmModeFormatModifierIterator it{};
                    while (drmModeFormatModifierBlobIterNext(blob, &it)) pi.fmt_mods[it.fmt].insert(it.mod);
                    drmModeFreePropertyBlob(blob);
                }
            } else if (drm_property_type_is(pr, DRM_MODE_PROP_ENUM) || drm_property_type_is(pr, DRM_MODE_PROP_BITMASK)) {
                if (name == "COLOR_ENCODING" || name == "COLOR_RANGE" || name == "pixel blend mode" ||
                    name == "rotation" || name == "chroma_siting_h" || name == "chroma_siting_v") {
                    std::string line = name + " {";
                    for (int e = 0; e < pr->count_enums; ++e) {
                        line += (e ? ", " : "");
                        line += std::to_string(pr->enums[e].value) + "=" + pr->enums[e].name;
                    }
                    line += "} cur=" + std::to_string(val);
                    pi.enums.push_back(line);
                }
            } else if (drm_property_type_is(pr, DRM_MODE_PROP_RANGE) && name == "alpha") {
                pi.enums.push_back("alpha range [" + std::to_string(pr->values[0]) + "," +
                                   std::to_string(pr->values[1]) + "] cur=" + std::to_string(val));
            }
            drmModeFreeProperty(pr);
        }
        drmModeFreeObjectProperties(props);
        drmModeFreePlane(pl);
        planes.push_back(std::move(pi));
    }
    drmModeFreePlaneResources(pres);

    // CRTC idx -> id 표 (possible_crtcs 는 비트마스크)
    std::vector<std::uint32_t> crtc_ids(res->crtcs, res->crtcs + res->count_crtcs);
    auto crtc_list = [&](std::uint32_t mask) {
        std::string s;
        for (std::size_t k = 0; k < crtc_ids.size(); ++k)
            if (mask & (1u << k)) s += (s.empty() ? "" : ",") + std::to_string(k) + "(" + std::to_string(crtc_ids[k]) + ")";
        return s;
    };

    std::printf("\n== PLANE (%zu) ==\n", planes.size());
    int n_nv12_sand = 0, n_p030_sand = 0, n_ar24_lin = 0;
    for (const auto& p : planes) {
        const bool nv12_sand = p.fmt_mods.count(DRM_FORMAT_NV12) &&
                               std::any_of(p.fmt_mods.at(DRM_FORMAT_NV12).begin(), p.fmt_mods.at(DRM_FORMAT_NV12).end(), mod_is_sand128);
        const bool p030_sand = p.fmt_mods.count(DRM_FORMAT_P030) &&
                               std::any_of(p.fmt_mods.at(DRM_FORMAT_P030).begin(), p.fmt_mods.at(DRM_FORMAT_P030).end(), mod_is_sand128);
        const bool ar24_lin = p.fmt_mods.count(DRM_FORMAT_ARGB8888) &&
                              p.fmt_mods.at(DRM_FORMAT_ARGB8888).count(DRM_FORMAT_MOD_LINEAR);
        n_nv12_sand += nv12_sand;
        n_p030_sand += p030_sand;
        n_ar24_lin += ar24_lin;
        std::printf("plane %u  type %-7s crtcs %-28s zpos %s[%llu..%llu] cur %llu  NV12xSAND128 %s  P030xSAND128 %s  AR24xLINEAR %s  formats %zu\n",
                    p.id, p.type.c_str(), crtc_list(p.possible_crtcs).c_str(),
                    p.has_zpos ? (p.zpos_immutable ? "immutable" : "mutable") : "none",
                    static_cast<unsigned long long>(p.zpos_min), static_cast<unsigned long long>(p.zpos_max),
                    static_cast<unsigned long long>(p.zpos_cur), nv12_sand ? "Y" : "n", p030_sand ? "Y" : "n",
                    ar24_lin ? "Y" : "n", p.fmt_mods.size());
    }
    std::printf("\n요약: NV12xSAND128 평면 %d개, P030xSAND128 %d개, AR24xLINEAR %d개 (총 %zu)\n", n_nv12_sand,
                n_p030_sand, n_ar24_lin, planes.size());

    // 대표 평면(primary 1개, overlay 1개)의 전체 format/modifier 표와 속성
    bool shown_primary = false, shown_overlay = false;
    for (const auto& p : planes) {
        const bool is_primary = p.type == "Primary";
        const bool is_overlay = p.type == "Overlay";
        if ((is_primary && shown_primary) || (is_overlay && shown_overlay) || (!is_primary && !is_overlay)) continue;
        (is_primary ? shown_primary : shown_overlay) = true;
        std::printf("\n-- plane %u (%s) 상세 --\n  props:", p.id, p.type.c_str());
        for (const auto& n : p.prop_names) std::printf(" [%s]", n.c_str());
        std::printf("\n");
        for (const auto& e : p.enums) std::printf("  %s\n", e.c_str());
        for (const auto& [fmt, mods] : p.fmt_mods) {
            std::printf("  %s(0x%08x):", fourcc_str(fmt).c_str(), fmt);
            for (auto m : mods) std::printf(" 0x%016llx", static_cast<unsigned long long>(m));
            std::printf("\n");
        }
    }
    drmModeFreeResources(res);
    close(fd);

    // debugfs state 요약: crtc 가 붙은 평면만 요약한다.
    std::ifstream st(state_path);
    if (!st) {
        std::printf("\n(debugfs %s 를 읽을 수 없다)\n", state_path.c_str());
        return 0;
    }
    std::printf("\n== debugfs %s: 활성 평면 ==\n", state_path.c_str());
    std::string line, cur_plane, crtc, fmt, mod, size, zpos, alloc, crtcpos, srcpos, enc, range;
    auto flush = [&]() {
        if (!cur_plane.empty() && crtc != "(null)" && !crtc.empty())
            std::printf("%s crtc=%s alloc=%s fmt=%s mod=%s size=%s crtc-pos=%s src-pos=%s zpos=%s enc=%s range=%s\n",
                        cur_plane.c_str(), crtc.c_str(), alloc.c_str(), fmt.c_str(), mod.c_str(), size.c_str(),
                        crtcpos.c_str(), srcpos.c_str(), zpos.c_str(), enc.c_str(), range.c_str());
        crtc = fmt = mod = size = zpos = alloc = crtcpos = srcpos = enc = range = "";
    };
    auto val_of = [](const std::string& l) {
        auto pos = l.find('=');
        return pos == std::string::npos ? std::string() : l.substr(pos + 1);
    };
    while (std::getline(st, line)) {
        std::string t = line;
        t.erase(0, t.find_first_not_of(" \t"));
        if (line.rfind("plane[", 0) == 0) {
            flush();
            cur_plane = line;
        } else if (line.rfind("crtc[", 0) == 0 || line.rfind("connector[", 0) == 0) {
            flush();
            cur_plane.clear();
            std::printf("%s\n", line.c_str());
        } else if (cur_plane.empty()) {
            if (t.rfind("enable=", 0) == 0 || t.rfind("active=", 0) == 0 || t.rfind("mode:", 0) == 0 ||
                t.rfind("crtc=", 0) == 0)
                std::printf("    %s\n", t.c_str());
        } else if (t.rfind("crtc=", 0) == 0) crtc = val_of(t);
        else if (t.rfind("allocated by", 0) == 0) alloc = val_of(t);
        else if (t.rfind("format=", 0) == 0) fmt = val_of(t);
        else if (t.rfind("modifier=", 0) == 0) mod = val_of(t);
        else if (t.rfind("size=", 0) == 0 && size.empty()) size = val_of(t);
        else if (t.rfind("crtc-pos=", 0) == 0) crtcpos = val_of(t);
        else if (t.rfind("src-pos=", 0) == 0) srcpos = val_of(t);
        else if (t.rfind("normalized-zpos=", 0) == 0) zpos = val_of(t);
        else if (t.rfind("color-encoding=", 0) == 0) enc = val_of(t);
        else if (t.rfind("color-range=", 0) == 0) range = val_of(t);
    }
    flush();
    return 0;
}
