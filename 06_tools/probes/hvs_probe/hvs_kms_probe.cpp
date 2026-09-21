// hvs_kms_probe: 디코더 SAND128 NV12 dma-buf 를 KMS 평면(HVS6)에 직접 올리는 경로를 측정한다.
//
//   비디오 평면 : drmPrimeFDToHandle → drmModeAddFB2WithModifiers(NV12, SAND128, pitch=열 높이)
//                 COLOR_ENCODING=BT.709, COLOR_RANGE=limited
//   오버레이    : AB24(메모리 R,G,B,A) dumb buffer 3개 링, zpos 비디오 위, pixel blend mode=Pre-multiplied
//   페이싱      : 60 Hz 출력에서 30 fps. page-flip 이벤트(seq S) → vblank 이벤트(S+1) → 다음 commit → flip(S+2)
//   보유        : flip 완료 전까지 이전 프레임 keepalive 를 들고 있는다(held 최대치 기록)
//   측정        : flip 시각 간격, seq 간격(누락), commit→flip, CPU, V3D gpu_stats(0 이어야 함), debugfs 스냅샷
//   색 확인     : --writeback-at K 이면 Writeback 커넥터로 HVS 출력을 캡처해 CPU 기준 변환과 비교할 PPM 을 쓴다
//
// ⚠ DRM master 가 필요하다. labwc(lightdm)가 떠 있으면 master 를 얻지 못한다. 그때는 master 없이 가능한 단계
//   (FB 생성, dumb buffer, 평면 선택)까지만 검사하고 종료 코드 3 으로 끝난다. 이 프로그램은 어떤 세션도
//   멈추거나 VT 를 바꾸지 않는다. 실행 절차는 tools/hvs_probe/run_kms_probe.sh 참고.
#include <drm_fourcc.h>
#include <fcntl.h>
#include <poll.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <xf86drm.h>
#include <xf86drmMode.h>

#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <deque>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "overlay_painter.hpp"
#include "pipeline/ffmpeg_drm_decoder.hpp"
#include "probe_common.hpp"

using namespace hvsprobe;

namespace {

volatile std::sig_atomic_t g_stop = 0;
void on_signal(int) { g_stop = 1; }

struct Args {
    std::string card = "/dev/dri/card1";
    std::string video = "04_data/videos/video_h265.mp4";
    std::string dets;
    std::string out_dir = ".";
    std::string label = "kms";
    int frames = 300;
    bool overlay = true;
    int dst_w = 0, dst_h = 0; // 0 = 모드 크기(1:1)
    int video_plane = 0, overlay_plane = 0;
    int sand_param = 0;
    long pitch_y = -1, pitch_uv = -1;
    int writeback_at = -1;
    bool check_only = false;  // master 가 있어도 commit 하지 않고 TEST_ONLY 까지만
    int enc = -1, range = -1; // -1 = BT.709 / limited 를 enum 이름으로 찾는다
};

void usage(const char* p) {
    std::fprintf(stderr,
                 "usage: %s [--card /dev/dri/card1] [--video F] [--dets J] [--frames N] [--overlay on|off]\n"
                 "          [--dst WxH] [--video-plane ID] [--overlay-plane ID] [--sand-param N] [--pitch-y N]\n"
                 "          [--pitch-uv N] [--writeback-at K] [--check-only] [--out-dir D] [--label L]\n",
                 p);
}

bool parse_args(int argc, char** argv, Args& a) {
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        auto next = [&]() -> const char* { return i + 1 < argc ? argv[++i] : nullptr; };
        const char* v = nullptr;
        if (s == "--card" && (v = next())) a.card = v;
        else if (s == "--video" && (v = next())) a.video = v;
        else if (s == "--dets" && (v = next())) a.dets = v;
        else if (s == "--frames" && (v = next())) a.frames = std::atoi(v);
        else if (s == "--overlay" && (v = next())) a.overlay = std::string(v) == "on";
        else if (s == "--dst" && (v = next())) {
            if (std::sscanf(v, "%dx%d", &a.dst_w, &a.dst_h) != 2) return false;
        } else if (s == "--video-plane" && (v = next())) a.video_plane = std::atoi(v);
        else if (s == "--overlay-plane" && (v = next())) a.overlay_plane = std::atoi(v);
        else if (s == "--sand-param" && (v = next())) a.sand_param = std::atoi(v);
        else if (s == "--pitch-y" && (v = next())) a.pitch_y = std::atol(v);
        else if (s == "--pitch-uv" && (v = next())) a.pitch_uv = std::atol(v);
        else if (s == "--writeback-at" && (v = next())) a.writeback_at = std::atoi(v);
        else if (s == "--check-only") a.check_only = true;
        else if (s == "--out-dir" && (v = next())) a.out_dir = v;
        else if (s == "--label" && (v = next())) a.label = v;
        else return false;
    }
    return a.frames > 0;
}

// ── KMS 객체 속성 표 ────────────────────────────────────────────────────────────────────────
struct Props {
    std::map<std::string, std::uint32_t> id;
    std::map<std::string, std::uint64_t> val;
    std::map<std::string, std::map<std::string, std::uint64_t>> enums;
    std::uint32_t get(const std::string& n) const {
        auto it = id.find(n);
        return it == id.end() ? 0 : it->second;
    }
};

Props load_props(int fd, std::uint32_t obj, std::uint32_t type) {
    Props p;
    drmModeObjectPropertiesPtr props = drmModeObjectGetProperties(fd, obj, type);
    for (std::uint32_t i = 0; props && i < props->count_props; ++i) {
        drmModePropertyPtr pr = drmModeGetProperty(fd, props->props[i]);
        if (!pr) continue;
        p.id[pr->name] = pr->prop_id;
        p.val[pr->name] = props->prop_values[i];
        if (drm_property_type_is(pr, DRM_MODE_PROP_ENUM))
            for (int e = 0; e < pr->count_enums; ++e) p.enums[pr->name][pr->enums[e].name] = pr->enums[e].value;
        drmModeFreeProperty(pr);
    }
    drmModeFreeObjectProperties(props);
    return p;
}

struct PlaneCaps {
    std::uint32_t id = 0;
    std::uint32_t possible_crtcs = 0;
    std::string type;
    bool nv12_sand = false, ab24_lin = false, ar24_lin = false;
    Props props;
};

std::vector<PlaneCaps> load_planes(int fd) {
    std::vector<PlaneCaps> out;
    drmModePlaneResPtr pres = drmModeGetPlaneResources(fd);
    for (std::uint32_t i = 0; pres && i < pres->count_planes; ++i) {
        drmModePlanePtr pl = drmModeGetPlane(fd, pres->planes[i]);
        if (!pl) continue;
        PlaneCaps pc;
        pc.id = pl->plane_id;
        pc.possible_crtcs = pl->possible_crtcs;
        pc.props = load_props(fd, pl->plane_id, DRM_MODE_OBJECT_PLANE);
        for (const auto& [n, v] : pc.props.enums["type"])
            if (v == pc.props.val["type"]) pc.type = n;
        const std::uint32_t blob_id = static_cast<std::uint32_t>(pc.props.val["IN_FORMATS"]);
        if (drmModePropertyBlobPtr blob = drmModeGetPropertyBlob(fd, blob_id)) {
            drmModeFormatModifierIterator it{};
            while (drmModeFormatModifierBlobIterNext(blob, &it)) {
                if (it.fmt == DRM_FORMAT_NV12 && fourcc_mod_broadcom_mod(it.mod) == DRM_FORMAT_MOD_BROADCOM_SAND128)
                    pc.nv12_sand = true;
                if (it.fmt == DRM_FORMAT_ABGR8888 && it.mod == DRM_FORMAT_MOD_LINEAR) pc.ab24_lin = true;
                if (it.fmt == DRM_FORMAT_ARGB8888 && it.mod == DRM_FORMAT_MOD_LINEAR) pc.ar24_lin = true;
            }
            drmModeFreePropertyBlob(blob);
        }
        drmModeFreePlane(pl);
        out.push_back(std::move(pc));
    }
    drmModeFreePlaneResources(pres);
    return out;
}

// ── dumb buffer ────────────────────────────────────────────────────────────────────────────
struct DumbBuf {
    std::uint32_t handle = 0, pitch = 0, fb = 0;
    std::uint64_t size = 0;
    std::uint8_t* map = nullptr;
    int w = 0, h = 0;
};

bool create_dumb(int fd, int w, int h, std::uint32_t fmt, DumbBuf& b, std::string& err) {
    drm_mode_create_dumb c{};
    c.width = static_cast<std::uint32_t>(w);
    c.height = static_cast<std::uint32_t>(h);
    c.bpp = 32;
    if (drmIoctl(fd, DRM_IOCTL_MODE_CREATE_DUMB, &c) != 0) {
        err = std::string("CREATE_DUMB: ") + std::strerror(errno);
        return false;
    }
    b.handle = c.handle;
    b.pitch = c.pitch;
    b.size = c.size;
    b.w = w;
    b.h = h;
    drm_mode_map_dumb m{};
    m.handle = c.handle;
    if (drmIoctl(fd, DRM_IOCTL_MODE_MAP_DUMB, &m) != 0) {
        err = std::string("MAP_DUMB: ") + std::strerror(errno);
        return false;
    }
    void* p = mmap(nullptr, c.size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, static_cast<off_t>(m.offset));
    if (p == MAP_FAILED) {
        err = std::string("mmap dumb: ") + std::strerror(errno);
        return false;
    }
    b.map = static_cast<std::uint8_t*>(p);
    std::memset(b.map, 0, c.size);
    std::uint32_t handles[4] = {b.handle}, pitches[4] = {b.pitch}, offsets[4] = {0};
    std::uint64_t mods[4] = {DRM_FORMAT_MOD_LINEAR};
    if (drmModeAddFB2WithModifiers(fd, static_cast<std::uint32_t>(w), static_cast<std::uint32_t>(h), fmt, handles,
                                   pitches, offsets, mods, &b.fb, DRM_MODE_FB_MODIFIERS) != 0) {
        err = std::string("AddFB2(dumb): ") + std::strerror(errno);
        return false;
    }
    return true;
}

void destroy_dumb(int fd, DumbBuf& b) {
    if (b.fb) drmModeRmFB(fd, b.fb);
    if (b.map) munmap(b.map, b.size);
    if (b.handle) {
        drm_mode_destroy_dumb d{};
        d.handle = b.handle;
        drmIoctl(fd, DRM_IOCTL_MODE_DESTROY_DUMB, &d);
    }
    b = DumbBuf{};
}

// ── 비디오 FB 캐시 ─────────────────────────────────────────────────────────────────────────
struct VideoFb {
    std::uint32_t fb = 0, hy = 0, huv = 0;
};

struct FrameRec {
    std::int64_t pts = -1;
    std::int64_t decoded_ns = 0, commit_ns = 0, flip_ns = 0;
    std::uint32_t seq = 0;
    int held_at_commit = 0;
    double overlay_draw_ms = 0;
    double commit_ms = 0; // drmModeAtomicCommit 호출 시간
    int ndets = 0;
};

struct Kms {
    Args args;
    int fd = -1;
    bool master = false;
    std::uint32_t crtc_id = 0, conn_id = 0;
    int crtc_idx = -1;
    drmModeModeInfo mode{};
    std::uint32_t mode_blob = 0;
    Props crtc_props, conn_props;
    std::vector<PlaneCaps> planes;
    PlaneCaps* vplane = nullptr;
    PlaneCaps* oplane = nullptr;
    std::uint32_t primary_id = 0;
    std::map<std::pair<ino_t, ino_t>, VideoFb> fbs;
    std::vector<DumbBuf> ov;
    std::unique_ptr<OverlayPainter> painter;
    std::map<std::int64_t, std::vector<yvo::Detection>> dets;
    yvo::FfmpegDrmDecoder dec;

    // 표시 상태
    struct Held {
        yvo::DrmFrame f;
        int ov_idx = -1;
    };
    std::unique_ptr<Held> on_screen, pending;
    std::deque<yvo::DrmFrame> ahead;
    int held = 0, held_max = 0;
    bool flip_pending = false;
    bool vblank_pending = false;
    std::uint32_t last_flip_seq = 0;
    bool first = true;
    int committed = 0;
    std::vector<FrameRec> recs;
    std::int64_t commit_fail = 0;
    std::vector<std::string> log;
    bool done = false;
    int ov_next = 0;
};

std::uint64_t enum_or(const Props& p, const std::string& prop, const std::string& name, std::uint64_t dflt) {
    auto it = p.enums.find(prop);
    if (it == p.enums.end()) return dflt;
    auto jt = it->second.find(name);
    return jt == it->second.end() ? dflt : jt->second;
}

std::pair<ino_t, ino_t> buf_key(const yvo::DrmFrame& f) {
    struct stat sy{}, suv{};
    fstat(f.fd, &sy);
    fstat(f.uv_fd, &suv);
    return {sy.st_ino, suv.st_ino};
}

// 디코더 프레임 → KMS FB (dma-buf inode 로 캐시). 실패하면 0.
std::uint32_t fb_for(Kms& k, const yvo::DrmFrame& f, std::string* err) {
    const auto key = buf_key(f);
    auto it = k.fbs.find(key);
    if (it != k.fbs.end()) return it->second.fb;
    VideoFb v;
    if (drmPrimeFDToHandle(k.fd, f.fd, &v.hy) != 0) {
        if (err) *err = std::string("drmPrimeFDToHandle(Y): ") + std::strerror(errno);
        return 0;
    }
    if (drmPrimeFDToHandle(k.fd, f.uv_fd, &v.huv) != 0) {
        if (err) *err = std::string("drmPrimeFDToHandle(UV): ") + std::strerror(errno);
        return 0;
    }
    const std::uint64_t mod = k.args.sand_param
                                  ? DRM_FORMAT_MOD_BROADCOM_SAND128_COL_HEIGHT(static_cast<std::uint64_t>(k.args.sand_param))
                                  : f.modifier;
    std::uint32_t handles[4] = {v.hy, v.huv};
    std::uint32_t pitches[4] = {static_cast<std::uint32_t>(k.args.pitch_y >= 0 ? k.args.pitch_y : static_cast<long>(f.y_pitch)),
                                static_cast<std::uint32_t>(k.args.pitch_uv >= 0 ? k.args.pitch_uv : static_cast<long>(f.uv_pitch))};
    std::uint32_t offsets[4] = {static_cast<std::uint32_t>(f.y_offset), static_cast<std::uint32_t>(f.uv_offset)};
    std::uint64_t mods[4] = {mod, mod};
    if (drmModeAddFB2WithModifiers(k.fd, static_cast<std::uint32_t>(f.width), static_cast<std::uint32_t>(f.height),
                                   DRM_FORMAT_NV12, handles, pitches, offsets, mods, &v.fb, DRM_MODE_FB_MODIFIERS) != 0) {
        if (err) *err = std::string("drmModeAddFB2WithModifiers(NV12 SAND128): ") + std::strerror(errno);
        return 0;
    }
    char b[256];
    std::snprintf(b, sizeof(b), "[fb] new NV12 fb %u handles %u/%u pitches %u/%u offsets %u/%u mod 0x%016llx", v.fb,
                  v.hy, v.huv, pitches[0], pitches[1], offsets[0], offsets[1], static_cast<unsigned long long>(mod));
    k.log.push_back(b);
    if (k.fbs.size() < 2) std::printf("%s\n", b);
    k.fbs[key] = v;
    return v.fb;
}

void add_plane(drmModeAtomicReqPtr req, const PlaneCaps& p, std::uint32_t crtc, std::uint32_t fb, int sx, int sy,
               int sw, int sh, int cx, int cy, int cw, int ch) {
    const Props& pp = p.props;
    drmModeAtomicAddProperty(req, p.id, pp.get("FB_ID"), fb);
    drmModeAtomicAddProperty(req, p.id, pp.get("CRTC_ID"), crtc);
    drmModeAtomicAddProperty(req, p.id, pp.get("SRC_X"), static_cast<std::uint64_t>(sx) << 16);
    drmModeAtomicAddProperty(req, p.id, pp.get("SRC_Y"), static_cast<std::uint64_t>(sy) << 16);
    drmModeAtomicAddProperty(req, p.id, pp.get("SRC_W"), static_cast<std::uint64_t>(sw) << 16);
    drmModeAtomicAddProperty(req, p.id, pp.get("SRC_H"), static_cast<std::uint64_t>(sh) << 16);
    drmModeAtomicAddProperty(req, p.id, pp.get("CRTC_X"), static_cast<std::uint64_t>(cx));
    drmModeAtomicAddProperty(req, p.id, pp.get("CRTC_Y"), static_cast<std::uint64_t>(cy));
    drmModeAtomicAddProperty(req, p.id, pp.get("CRTC_W"), static_cast<std::uint64_t>(cw));
    drmModeAtomicAddProperty(req, p.id, pp.get("CRTC_H"), static_cast<std::uint64_t>(ch));
}

void disable_plane(drmModeAtomicReqPtr req, const PlaneCaps& p) {
    drmModeAtomicAddProperty(req, p.id, p.props.get("FB_ID"), 0);
    drmModeAtomicAddProperty(req, p.id, p.props.get("CRTC_ID"), 0);
}

// 한 프레임의 atomic 요청을 만든다
drmModeAtomicReqPtr build_req(Kms& k, std::uint32_t vfb, int ov_idx, bool modeset) {
    drmModeAtomicReqPtr req = drmModeAtomicAlloc();
    const int mw = k.mode.hdisplay, mh = k.mode.vdisplay;
    const int dw = k.args.dst_w > 0 ? k.args.dst_w : mw;
    const int dh = k.args.dst_h > 0 ? k.args.dst_h : mh;
    const int dx = (mw - dw) / 2, dy = (mh - dh) / 2;
    if (modeset) {
        drmModeAtomicAddProperty(req, k.conn_id, k.conn_props.get("CRTC_ID"), k.crtc_id);
        drmModeAtomicAddProperty(req, k.crtc_id, k.crtc_props.get("MODE_ID"), k.mode_blob);
        drmModeAtomicAddProperty(req, k.crtc_id, k.crtc_props.get("ACTIVE"), 1);
        // 이 CRTC 에 붙어 있을 수 있는 나머지 평면(labwc 가 쓰던 primary 등)을 끈다
        for (const auto& p : k.planes) {
            if (&p == k.vplane || &p == k.oplane) continue;
            if (!(p.possible_crtcs & (1u << k.crtc_idx))) continue;
            disable_plane(req, p);
        }
        const Props& vp = k.vplane->props;
        if (vp.get("COLOR_ENCODING"))
            drmModeAtomicAddProperty(req, k.vplane->id, vp.get("COLOR_ENCODING"),
                                     k.args.enc >= 0 ? static_cast<std::uint64_t>(k.args.enc)
                                                     : enum_or(vp, "COLOR_ENCODING", "ITU-R BT.709 YCbCr", 1));
        if (vp.get("COLOR_RANGE"))
            drmModeAtomicAddProperty(req, k.vplane->id, vp.get("COLOR_RANGE"),
                                     k.args.range >= 0 ? static_cast<std::uint64_t>(k.args.range)
                                                       : enum_or(vp, "COLOR_RANGE", "YCbCr limited range", 0));
        // primary 의 zpos 는 immutable(0) 이라 건드리지 않는다
        if (vp.get("zpos") && k.vplane->type != "Primary") drmModeAtomicAddProperty(req, k.vplane->id, vp.get("zpos"), 1);
        if (k.oplane) {
            const Props& op = k.oplane->props;
            if (op.get("zpos")) drmModeAtomicAddProperty(req, k.oplane->id, op.get("zpos"), 2);
            if (op.get("pixel blend mode"))
                drmModeAtomicAddProperty(req, k.oplane->id, op.get("pixel blend mode"),
                                         enum_or(op, "pixel blend mode", "Pre-multiplied", 0));
            if (op.get("alpha")) drmModeAtomicAddProperty(req, k.oplane->id, op.get("alpha"), 0xffff);
        }
    }
    add_plane(req, *k.vplane, k.crtc_id, vfb, 0, 0, 1920, 1080, dx, dy, dw, dh);
    if (k.oplane) {
        if (ov_idx >= 0) {
            const DumbBuf& b = k.ov[static_cast<std::size_t>(ov_idx)];
            add_plane(req, *k.oplane, k.crtc_id, b.fb, 0, 0, b.w, b.h, dx, dy, dw, dh);
        } else if (modeset) {
            disable_plane(req, *k.oplane);
        }
    }
    return req;
}

void snapshot_state(Kms& k, const std::string& tag) {
    const std::string st = read_text("/sys/kernel/debug/dri/1/state");
    write_text(k.args.out_dir + "/" + k.args.label + "_state_" + tag + ".txt", st);
    const auto planes = parse_active_planes(st);
    const std::string line = "[state " + tag + "] " + judge_placement(planes).text + "\n" + describe_planes(planes) +
                             "  hvs_underrun: " + read_text("/sys/kernel/debug/dri/1/hvs_underrun");
    k.log.push_back(line);
    std::printf("%s", line.c_str());
}

bool decode_ahead(Kms& k) {
    if (!k.ahead.empty()) return true;
    auto f = k.dec.next();
    if (!f) return false;
    k.ahead.push_back(std::move(*f));
    k.held++;
    k.held_max = std::max(k.held_max, k.held);
    return true;
}

// 다음 프레임을 commit (nonblocking + page flip event)
bool commit_next(Kms& k) {
    if (k.committed >= k.args.frames || g_stop) {
        k.done = true;
        return false;
    }
    if (!decode_ahead(k)) {
        k.done = true;
        return false;
    }
    yvo::DrmFrame f = std::move(k.ahead.front());
    k.ahead.pop_front();
    FrameRec r;
    r.pts = f.pts;
    std::string err;
    const std::uint32_t vfb = fb_for(k, f, &err);
    if (!vfb) {
        std::fprintf(stderr, "[kms] %s\n", err.c_str());
        k.held--;
        k.done = true;
        return false;
    }
    int ov_idx = -1;
    if (k.oplane && !k.ov.empty()) {
        // 화면·대기 중이 아닌 버퍼를 고른다 (링 3개면 항상 하나는 빈다)
        for (int t = 0; t < static_cast<int>(k.ov.size()); ++t) {
            const int c = (k.ov_next + t) % static_cast<int>(k.ov.size());
            if ((k.on_screen && k.on_screen->ov_idx == c) || (k.pending && k.pending->ov_idx == c)) continue;
            ov_idx = c;
            break;
        }
        k.ov_next = (ov_idx + 1) % static_cast<int>(k.ov.size());
        static const std::vector<yvo::Detection> kEmpty;
        auto it = k.dets.find(f.pts);
        const auto& d = it != k.dets.end() ? it->second : kEmpty;
        char hud[96];
        std::snprintf(hud, sizeof(hud), "FRAME %lld DET %zu HVS-KMS", static_cast<long long>(f.pts), d.size());
        const std::int64_t t0 = now_ns();
        k.painter->paint(k.ov[static_cast<std::size_t>(ov_idx)].map, ov_idx, d, hud);
        r.overlay_draw_ms = ms(now_ns() - t0);
        r.ndets = static_cast<int>(d.size());
    }
    drmModeAtomicReqPtr req = build_req(k, vfb, ov_idx, k.first);
    std::uint32_t flags = DRM_MODE_PAGE_FLIP_EVENT | DRM_MODE_ATOMIC_NONBLOCK;
    if (k.first) flags = DRM_MODE_PAGE_FLIP_EVENT | DRM_MODE_ATOMIC_ALLOW_MODESET;
    const std::int64_t t0 = now_ns();
    const int ret = drmModeAtomicCommit(k.fd, req, flags, &k);
    const std::int64_t t1 = now_ns();
    drmModeAtomicFree(req);
    if (ret != 0) {
        std::fprintf(stderr, "[kms] atomic commit 실패 (frame %lld): %s\n", static_cast<long long>(f.pts), std::strerror(errno));
        k.commit_fail++;
        k.held--;
        k.done = true;
        return false;
    }
    r.commit_ns = t1;
    r.commit_ms = ms(t1 - t0);
    r.held_at_commit = k.held;
    k.recs.push_back(r);
    k.pending = std::make_unique<Kms::Held>(Kms::Held{std::move(f), ov_idx});
    k.flip_pending = true;
    k.first = false;
    k.committed++;
    return true;
}

void on_flip(int, unsigned int seq, unsigned int sec, unsigned int usec, unsigned int, void* data) {
    auto* k = static_cast<Kms*>(data);
    k->flip_pending = false;
    FrameRec& r = k->recs.back();
    r.flip_ns = static_cast<std::int64_t>(sec) * 1000000000LL + static_cast<std::int64_t>(usec) * 1000;
    r.seq = seq;
    k->last_flip_seq = seq;
    // 이전 화면 프레임 반납 (이제 스캔아웃하지 않는다)
    if (k->on_screen) {
        k->on_screen.reset();
        k->held--;
    }
    k->on_screen = std::move(k->pending);
    const int n = k->committed;
    if (n == k->args.frames / 4 || n == k->args.frames / 2 || n == (3 * k->args.frames) / 4)
        snapshot_state(*k, "f" + std::to_string(n));
    // 다음 프레임을 미리 디코드해 둔다 (commit 은 다음 vblank 에서)
    decode_ahead(*k);
    // 30 fps: 다음 vblank(seq+1) 이벤트를 받아 commit 하면 flip 은 seq+2 에 일어난다.
    drmVBlank vbl{};
    vbl.request.type = static_cast<drmVBlankSeqType>(DRM_VBLANK_ABSOLUTE | DRM_VBLANK_EVENT |
                                                     (k->crtc_idx > 1 ? ((k->crtc_idx << DRM_VBLANK_HIGH_CRTC_SHIFT) &
                                                                         DRM_VBLANK_HIGH_CRTC_MASK)
                                                                      : (k->crtc_idx == 1 ? DRM_VBLANK_SECONDARY : 0)));
    vbl.request.sequence = seq + 1;
    vbl.request.signal = reinterpret_cast<unsigned long>(k);
    if (drmWaitVBlank(k->fd, &vbl) != 0) {
        // vblank 이벤트를 못 걸면 바로 commit (60 fps 가 될 수 있다 — 로그로 남긴다)
        k->log.push_back(std::string("[warn] drmWaitVBlank: ") + std::strerror(errno));
        commit_next(*k);
    } else {
        k->vblank_pending = true;
    }
}

void on_vblank(int, unsigned int, unsigned int, unsigned int, void* data) {
    auto* k = static_cast<Kms*>(data);
    k->vblank_pending = false;
    commit_next(*k);
}

// Writeback 커넥터로 현재 비디오 FB(+오버레이)를 한 번 합성해 캡처한다. 성공하면 PPM 을 쓴다.
bool writeback_capture(Kms& k, std::uint32_t vfb, int ov_idx, const std::string& path) {
    drmModeResPtr res = drmModeGetResources(k.fd);
    if (!res) return false;
    std::uint32_t wb_conn = 0, wb_crtc = 0;
    int wb_crtc_idx = -1;
    for (int i = 0; i < res->count_connectors && !wb_conn; ++i) {
        drmModeConnectorPtr c = drmModeGetConnector(k.fd, res->connectors[i]);
        if (!c) continue;
        if (c->connector_type == DRM_MODE_CONNECTOR_WRITEBACK) {
            for (int e = 0; e < c->count_encoders && !wb_conn; ++e) {
                drmModeEncoderPtr enc = drmModeGetEncoder(k.fd, c->encoders[e]);
                if (!enc) continue;
                for (int ci = 0; ci < res->count_crtcs; ++ci)
                    if ((enc->possible_crtcs & (1u << ci)) && res->crtcs[ci] != k.crtc_id) {
                        wb_conn = c->connector_id;
                        wb_crtc = res->crtcs[ci];
                        wb_crtc_idx = ci;
                        break;
                    }
                drmModeFreeEncoder(enc);
            }
        }
        drmModeFreeConnector(c);
    }
    drmModeFreeResources(res);
    if (!wb_conn) {
        std::printf("[wb] HDMI CRTC 와 다른 CRTC 에 붙는 writeback 커넥터가 없다\n");
        return false;
    }
    // 이 CRTC 에 붙일 수 있는 빈 평면 2개(NV12 SAND 1, AB24 1)
    PlaneCaps *pv = nullptr, *po = nullptr;
    for (auto& p : k.planes) {
        if (&p == k.vplane || &p == k.oplane || !(p.possible_crtcs & (1u << wb_crtc_idx))) continue;
        if (!pv && p.nv12_sand) pv = &p;
        else if (!po && p.ab24_lin) po = &p;
    }
    if (!pv) {
        std::printf("[wb] writeback CRTC 에 쓸 NV12 평면이 없다\n");
        return false;
    }
    Props wprops = load_props(k.fd, wb_conn, DRM_MODE_OBJECT_CONNECTOR);
    Props cprops = load_props(k.fd, wb_crtc, DRM_MODE_OBJECT_CRTC);
    DumbBuf cap;
    std::string err;
    if (!create_dumb(k.fd, k.mode.hdisplay, k.mode.vdisplay, DRM_FORMAT_XBGR8888, cap, err) &&
        !create_dumb(k.fd, k.mode.hdisplay, k.mode.vdisplay, DRM_FORMAT_XRGB8888, cap, err)) {
        std::printf("[wb] capture buffer: %s\n", err.c_str());
        return false;
    }
    int out_fence = -1;
    drmModeAtomicReqPtr req = drmModeAtomicAlloc();
    drmModeAtomicAddProperty(req, wb_conn, wprops.get("CRTC_ID"), wb_crtc);
    drmModeAtomicAddProperty(req, wb_conn, wprops.get("WRITEBACK_FB_ID"), cap.fb);
    drmModeAtomicAddProperty(req, wb_conn, wprops.get("WRITEBACK_OUT_FENCE_PTR"),
                             static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(&out_fence)));
    drmModeAtomicAddProperty(req, wb_crtc, cprops.get("MODE_ID"), k.mode_blob);
    drmModeAtomicAddProperty(req, wb_crtc, cprops.get("ACTIVE"), 1);
    const int mw = k.mode.hdisplay, mh = k.mode.vdisplay;
    const int dw = k.args.dst_w > 0 ? k.args.dst_w : mw, dh = k.args.dst_h > 0 ? k.args.dst_h : mh;
    const int dx = (mw - dw) / 2, dy = (mh - dh) / 2;
    add_plane(req, *pv, wb_crtc, vfb, 0, 0, 1920, 1080, dx, dy, dw, dh);
    const Props& vp = pv->props;
    if (vp.get("COLOR_ENCODING"))
        drmModeAtomicAddProperty(req, pv->id, vp.get("COLOR_ENCODING"), enum_or(vp, "COLOR_ENCODING", "ITU-R BT.709 YCbCr", 1));
    if (vp.get("COLOR_RANGE"))
        drmModeAtomicAddProperty(req, pv->id, vp.get("COLOR_RANGE"), enum_or(vp, "COLOR_RANGE", "YCbCr limited range", 0));
    if (po && ov_idx >= 0) {
        const DumbBuf& b = k.ov[static_cast<std::size_t>(ov_idx)];
        add_plane(req, *po, wb_crtc, b.fb, 0, 0, b.w, b.h, dx, dy, dw, dh);
        if (po->props.get("zpos")) drmModeAtomicAddProperty(req, po->id, po->props.get("zpos"), 3);
        if (po->props.get("pixel blend mode"))
            drmModeAtomicAddProperty(req, po->id, po->props.get("pixel blend mode"),
                                     enum_or(po->props, "pixel blend mode", "Pre-multiplied", 0));
    }
    const int ret = drmModeAtomicCommit(k.fd, req, DRM_MODE_ATOMIC_ALLOW_MODESET, nullptr);
    drmModeAtomicFree(req);
    if (ret != 0) {
        std::printf("[wb] writeback commit 실패: %s\n", std::strerror(errno));
        destroy_dumb(k.fd, cap);
        return false;
    }
    if (out_fence >= 0) {
        pollfd p{out_fence, POLLIN, 0};
        const int pr = poll(&p, 1, 1000);
        std::printf("[wb] out fence poll=%d\n", pr);
        close(out_fence);
    }
    std::vector<std::uint8_t> rgb(static_cast<std::size_t>(mw) * static_cast<std::size_t>(mh) * 3);
    // XB24: 메모리 R,G,B,X / XR24: B,G,R,X — 실제로 만든 fb 포맷을 다시 읽는다
    drmModeFB2Ptr fbi = drmModeGetFB2(k.fd, cap.fb);
    const bool is_xr24 = fbi && fbi->pixel_format == DRM_FORMAT_XRGB8888;
    if (fbi) drmModeFreeFB2(fbi);
    for (int y = 0; y < mh; ++y)
        for (int x = 0; x < mw; ++x) {
            const std::uint8_t* s = cap.map + static_cast<std::size_t>(y) * cap.pitch + static_cast<std::size_t>(x) * 4;
            std::uint8_t* d = &rgb[(static_cast<std::size_t>(y) * static_cast<std::size_t>(mw) + static_cast<std::size_t>(x)) * 3];
            if (is_xr24) {
                d[0] = s[2]; d[1] = s[1]; d[2] = s[0];
            } else {
                d[0] = s[0]; d[1] = s[1]; d[2] = s[2];
            }
        }
    write_ppm(path, rgb.data(), mw, mh);
    std::printf("[wb] captured -> %s (%s)\n", path.c_str(), is_xr24 ? "XR24" : "XB24");
    // writeback CRTC 끄기
    req = drmModeAtomicAlloc();
    drmModeAtomicAddProperty(req, wb_conn, wprops.get("CRTC_ID"), 0);
    drmModeAtomicAddProperty(req, wb_crtc, cprops.get("MODE_ID"), 0);
    drmModeAtomicAddProperty(req, wb_crtc, cprops.get("ACTIVE"), 0);
    disable_plane(req, *pv);
    if (po) disable_plane(req, *po);
    drmModeAtomicCommit(k.fd, req, DRM_MODE_ATOMIC_ALLOW_MODESET, nullptr);
    drmModeAtomicFree(req);
    destroy_dumb(k.fd, cap);
    return true;
}

} // namespace

int main(int argc, char** argv) {
    Kms k;
    std::setvbuf(stdout, nullptr, _IOLBF, 0); // 로그를 줄 단위로 (tee/리다이렉트 중 중단돼도 남도록)
    if (!parse_args(argc, argv, k.args)) {
        usage(argv[0]);
        return 2;
    }
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);
    Args& a = k.args;

    k.fd = open(a.card.c_str(), O_RDWR | O_CLOEXEC);
    if (k.fd < 0) {
        std::fprintf(stderr, "open %s: %s\n", a.card.c_str(), std::strerror(errno));
        return 1;
    }
    drmSetClientCap(k.fd, DRM_CLIENT_CAP_UNIVERSAL_PLANES, 1);
    const int atomic_ok = drmSetClientCap(k.fd, DRM_CLIENT_CAP_ATOMIC, 1);
    drmSetClientCap(k.fd, DRM_CLIENT_CAP_WRITEBACK_CONNECTORS, 1);
    // open 시점에 master 가 없었다면 이미 master 다. 아니면 SET_MASTER 를 시도한다(labwc 가 있으면 EACCES/EBUSY).
    bool master = drmIsMaster(k.fd);
    int set_master_errno = 0;
    if (!master) {
        if (drmSetMaster(k.fd) == 0) master = true;
        else set_master_errno = errno;
    }
    k.master = master;
    std::printf("[kms] %s atomic=%s master=%s%s\n", a.card.c_str(), atomic_ok == 0 ? "yes" : "no", master ? "yes" : "NO",
                master ? "" : (std::string(" (drmSetMaster: ") + std::strerror(set_master_errno) + ")").c_str());

    // 커넥터·CRTC·모드
    drmModeResPtr res = drmModeGetResources(k.fd);
    if (!res) {
        std::fprintf(stderr, "drmModeGetResources: %s\n", std::strerror(errno));
        return 1;
    }
    for (int i = 0; i < res->count_connectors && !k.conn_id; ++i) {
        drmModeConnectorPtr c = drmModeGetConnector(k.fd, res->connectors[i]);
        if (!c) continue;
        if (c->connector_type == DRM_MODE_CONNECTOR_HDMIA && c->connection == DRM_MODE_CONNECTED && c->count_modes > 0) {
            k.conn_id = c->connector_id;
            // 모드: 현재 CRTC 모드가 있으면 그것, 없으면 1920x1080@60 우선, 그다음 preferred
            std::uint32_t cur_crtc = 0;
            if (c->encoder_id) {
                if (drmModeEncoderPtr e = drmModeGetEncoder(k.fd, c->encoder_id)) {
                    cur_crtc = e->crtc_id;
                    drmModeFreeEncoder(e);
                }
            }
            drmModeModeInfo* pick = nullptr;
            for (int m = 0; m < c->count_modes; ++m)
                if (c->modes[m].hdisplay == 1920 && c->modes[m].vdisplay == 1080 && c->modes[m].vrefresh == 60) {
                    pick = &c->modes[m];
                    break;
                }
            for (int m = 0; m < c->count_modes && !pick; ++m)
                if (c->modes[m].type & DRM_MODE_TYPE_PREFERRED) pick = &c->modes[m];
            k.mode = pick ? *pick : c->modes[0];
            for (int ci = 0; ci < res->count_crtcs; ++ci)
                if (res->crtcs[ci] == cur_crtc) {
                    k.crtc_id = cur_crtc;
                    k.crtc_idx = ci;
                }
            if (!k.crtc_id && c->count_encoders > 0) {
                if (drmModeEncoderPtr e = drmModeGetEncoder(k.fd, c->encoders[0])) {
                    for (int ci = 0; ci < res->count_crtcs; ++ci)
                        if (e->possible_crtcs & (1u << ci)) {
                            k.crtc_id = res->crtcs[ci];
                            k.crtc_idx = ci;
                            break;
                        }
                    drmModeFreeEncoder(e);
                }
            }
        }
        drmModeFreeConnector(c);
    }
    drmModeFreeResources(res);
    if (!k.conn_id || !k.crtc_id) {
        std::fprintf(stderr, "연결된 HDMI 커넥터/CRTC 를 찾지 못했다\n");
        return 1;
    }
    std::printf("[kms] connector %u crtc %u (idx %d) mode %s %ux%u@%u\n", k.conn_id, k.crtc_id, k.crtc_idx, k.mode.name,
                k.mode.hdisplay, k.mode.vdisplay, k.mode.vrefresh);
    k.crtc_props = load_props(k.fd, k.crtc_id, DRM_MODE_OBJECT_CRTC);
    k.conn_props = load_props(k.fd, k.conn_id, DRM_MODE_OBJECT_CONNECTOR);

    // 평면 선택
    k.planes = load_planes(k.fd);
    for (auto& p : k.planes) {
        if (!(p.possible_crtcs & (1u << k.crtc_idx))) continue;
        if (p.type == "Primary") k.primary_id = p.id;
        if (a.video_plane ? static_cast<int>(p.id) == a.video_plane : (!k.vplane && p.nv12_sand && p.type == "Primary"))
            k.vplane = &p;
    }
    if (!k.vplane)
        for (auto& p : k.planes)
            if ((p.possible_crtcs & (1u << k.crtc_idx)) && p.nv12_sand && p.type == "Overlay") {
                k.vplane = &p;
                break;
            }
    if (a.overlay)
        for (auto& p : k.planes) {
            if (&p == k.vplane || !(p.possible_crtcs & (1u << k.crtc_idx))) continue;
            if (a.overlay_plane ? static_cast<int>(p.id) == a.overlay_plane : (p.type == "Overlay" && p.ab24_lin)) {
                k.oplane = &p;
                break;
            }
        }
    if (!k.vplane) {
        std::fprintf(stderr, "NV12xSAND128 을 받는 평면이 이 CRTC 에 없다\n");
        return 1;
    }
    std::printf("[kms] video plane %u (%s) COLOR_ENCODING %s COLOR_RANGE %s | overlay plane %u (%s)\n", k.vplane->id,
                k.vplane->type.c_str(), k.vplane->props.get("COLOR_ENCODING") ? "yes" : "no",
                k.vplane->props.get("COLOR_RANGE") ? "yes" : "no", k.oplane ? k.oplane->id : 0,
                k.oplane ? k.oplane->type.c_str() : "-");

    // 모드 blob (master 가 없어도 된다: CREATEPROPBLOB 은 DRM_MASTER 가 아니다)
    if (drmModeCreatePropertyBlob(k.fd, &k.mode, sizeof(k.mode), &k.mode_blob) != 0)
        std::printf("[kms] CreatePropertyBlob(mode) 실패: %s\n", std::strerror(errno));

    // 검출·디코더
    if (!a.dets.empty()) {
        std::size_t total = 0;
        k.dets = load_detections(a.dets, &total);
        std::printf("[dets] %zu frames, %zu detections\n", k.dets.size(), total);
    }
    if (!k.dec.open(a.video)) {
        std::fprintf(stderr, "decoder open failed: %s\n", k.dec.error().c_str());
        return 1;
    }

    // 오버레이 dumb buffer 링
    const int mw = k.mode.hdisplay, mh = k.mode.vdisplay;
    const int dw = a.dst_w > 0 ? a.dst_w : mw, dh = a.dst_h > 0 ? a.dst_h : mh;
    if (k.oplane) {
        const std::uint32_t fmt = k.oplane->ab24_lin ? DRM_FORMAT_ABGR8888 : DRM_FORMAT_ARGB8888;
        for (int i = 0; i < 3; ++i) {
            DumbBuf b;
            std::string err;
            if (!create_dumb(k.fd, dw, dh, fmt, b, err)) {
                std::printf("[kms] overlay dumb buffer: %s\n", err.c_str());
                break;
            }
            if (b.pitch != static_cast<std::uint32_t>(dw * 4))
                std::printf("[kms] 경고: dumb pitch %u != w*4 — 페인터는 패킹 버퍼를 가정한다\n", b.pitch);
            k.ov.push_back(b);
        }
        std::printf("[kms] overlay dumb buffers %zu x %dx%d %s\n", k.ov.size(), dw, dh,
                    fmt == DRM_FORMAT_ABGR8888 ? "AB24" : "AR24(주의: 채널 순서 R/B 바뀜)");
        k.painter = std::make_unique<OverlayPainter>(dw, dh, 3);
    }

    // 첫 프레임 FB (master 없이 되는지 확인)
    if (!decode_ahead(k)) {
        std::fprintf(stderr, "첫 프레임 디코드 실패: %s\n", k.dec.error().c_str());
        return 1;
    }
    {
        const yvo::DrmFrame& f = k.ahead.front();
        std::printf("[video] %dx%d mod=0x%016llx Y off=%zu pitch=%zu size=%zu | UV off=%zu pitch=%zu size=%zu (nb objects %s)\n",
                    f.width, f.height, static_cast<unsigned long long>(f.modifier), f.y_offset, f.y_pitch, f.size,
                    f.uv_offset, f.uv_pitch, f.uv_size, f.fd == f.uv_fd ? "1" : "2");
        std::string err;
        const std::uint32_t fb = fb_for(k, f, &err);
        std::printf("[kms] first NV12 SAND128 FB: %s\n", fb ? ("OK fb " + std::to_string(fb)).c_str() : err.c_str());
        if (!fb) return 1;
        // TEST_ONLY
        drmModeAtomicReqPtr req = build_req(k, fb, k.ov.empty() ? -1 : 0, true);
        const int ret = drmModeAtomicCommit(k.fd, req, DRM_MODE_ATOMIC_TEST_ONLY | DRM_MODE_ATOMIC_ALLOW_MODESET, nullptr);
        const int e = errno;
        drmModeAtomicFree(req);
        std::printf("[kms] TEST_ONLY atomic (video %s + overlay %s, dst %dx%d): %s\n", k.vplane->type.c_str(),
                    k.oplane ? "on" : "off", dw, dh, ret == 0 ? "OK" : std::strerror(e));
        if (!master) {
            std::printf("\nRESULT_KMS_CHECK,%s,master=no,addfb2=ok,test_only=%s\n", a.label.c_str(), ret == 0 ? "ok" : std::strerror(e));
            std::printf("DRM master 가 없다(labwc/lightdm 이 잡고 있음). 실제 표시 측정은 run_kms_probe.sh 절차로 사용자가 실행해야 한다.\n");
            return 3;
        }
        if (ret != 0 || a.check_only) return ret != 0 ? 1 : 0;
        if (a.writeback_at == 0) {
            const auto rgb = sand_to_rgb(f);
            write_ppm(a.out_dir + "/" + a.label + "_ref_f0.ppm", rgb.data(), f.width, f.height);
        }
    }

    // ── 표시 루프 ──
    drmEventContext ev{};
    ev.version = 3;
    ev.page_flip_handler2 = on_flip;
    ev.vblank_handler = on_vblank;
    const V3dStats v3d_a0 = read_v3d_stats();
    std::int64_t meas_t0 = 0, cpu0 = 0;
    V3dStats v3d_a;
    int meas_f0 = 0;
    commit_next(k);
    while (!k.done || k.flip_pending || k.vblank_pending) {
        if (k.committed == 30 && meas_t0 == 0) {
            meas_t0 = now_ns();
            cpu0 = self_cpu_ns();
            v3d_a = read_v3d_stats();
            meas_f0 = k.committed;
        }
        if (a.writeback_at >= 0 && k.committed == a.writeback_at + 1 && !k.flip_pending && k.on_screen) {
            // on_screen 프레임을 writeback 으로 캡처 (한 번만)
            static bool once = false;
            if (!once) {
                once = true;
                std::string err;
                const std::uint32_t vfb = fb_for(k, k.on_screen->f, &err);
                const auto rgb = sand_to_rgb(k.on_screen->f);
                const std::string base = a.out_dir + "/" + a.label + "_f" + std::to_string(k.on_screen->f.pts);
                write_ppm(base + "_ref.ppm", rgb.data(), k.on_screen->f.width, k.on_screen->f.height);
                writeback_capture(k, vfb, k.on_screen->ov_idx, base + "_writeback.ppm");
            }
        }
        pollfd p{k.fd, POLLIN, 0};
        const int pr = poll(&p, 1, 1000);
        if (pr < 0 && errno != EINTR) break;
        if (pr == 0) {
            std::fprintf(stderr, "[kms] 1초 동안 이벤트 없음 (flip_pending=%d vblank_pending=%d)\n", k.flip_pending,
                         k.vblank_pending);
            if (g_stop) break;
            continue;
        }
        if (pr > 0) drmHandleEvent(k.fd, &ev);
        if (g_stop && !k.flip_pending && !k.vblank_pending) break;
    }
    const std::int64_t meas_t1 = now_ns();
    const std::int64_t cpu1 = self_cpu_ns();
    const V3dStats v3d_b = read_v3d_stats();

    // ── 요약 ──
    std::vector<double> iv, gap, commit_to_flip, draw, commit_call;
    int late = 0;
    for (std::size_t i = 1; i < k.recs.size(); ++i) {
        const auto& r = k.recs[i];
        const auto& pr = k.recs[i - 1];
        if (!r.flip_ns || !pr.flip_ns || i < 30) continue;
        iv.push_back(ms(r.flip_ns - pr.flip_ns));
        const double g = static_cast<double>(r.seq - pr.seq);
        gap.push_back(g);
        if (g > 2) ++late;
        commit_to_flip.push_back(ms(r.flip_ns - r.commit_ns));
        commit_call.push_back(r.commit_ms);
        if (k.oplane) draw.push_back(r.overlay_draw_ms);
    }
    const int nf = k.committed - meas_f0;
    const std::int64_t wall = meas_t1 - meas_t0;
    double render_ms = 0, csd_ms = 0;
    const std::string v3d = v3d_delta_str(v3d_a, v3d_b, wall, nf, &render_ms, &csd_ms);
    std::printf("\n== hvs_kms_probe [%s] overlay=%s dst=%dx%d video plane %u overlay plane %u ==\n", a.label.c_str(),
                k.oplane ? "on" : "off", dw, dh, k.vplane->id, k.oplane ? k.oplane->id : 0);
    std::printf("frames committed %d (measured %d frames, %.3f s, %.2f FPS), commit failures %lld\n", k.committed, nf,
                wall / 1e9, nf > 0 ? nf / (wall / 1e9) : 0.0, static_cast<long long>(k.commit_fail));
    std::printf("video FBs %zu (cache by dmabuf inode)\n", k.fbs.size());
    std::printf("flip interval ms   %s\n", summary_str(summarize(iv)).c_str());
    std::printf("flip seq gap       %s (2 = 30fps on 60Hz; late(>2) %d)\n", summary_str(summarize(gap)).c_str(), late);
    std::printf("commit->flip ms    %s\n", summary_str(summarize(commit_to_flip)).c_str());
    std::printf("atomic commit call %s\n", summary_str(summarize(commit_call)).c_str());
    if (k.oplane) std::printf("overlay draw ms    %s\n", summary_str(summarize(draw)).c_str());
    std::printf("held frames max %d (decoder limit ~13)\n", k.held_max);
    std::printf("V3D (system-wide) %s\n", v3d.c_str());
    std::printf("V3D whole-run delta: %s\n", v3d_delta_str(v3d_a0, v3d_b, meas_t1 - meas_t0, nf).c_str());
    std::printf("CPU probe ms/frame %.3f\n", nf > 0 ? ms(cpu1 - cpu0) / nf : 0.0);
    std::printf("RESULT_KMS,%s,%s,%dx%d,%d,%.2f,%d,%.3f,%.3f,%.3f,%.3f,%d,%.3f,%.3f,%.3f\n", a.label.c_str(),
                k.oplane ? "on" : "off", dw, dh, nf, nf > 0 ? nf / (wall / 1e9) : 0.0, late, summarize(iv).median,
                summarize(iv).p99, summarize(commit_to_flip).median, summarize(draw).median, k.held_max, render_ms, csd_ms,
                nf > 0 ? ms(cpu1 - cpu0) / nf : 0.0);
    if (FILE* f = std::fopen((a.out_dir + "/" + a.label + "_frames.csv").c_str(), "w")) {
        std::fprintf(f, "idx,pts,commit_ns,flip_ns,seq,held_at_commit,overlay_draw_ms,commit_call_ms,ndets\n");
        for (std::size_t i = 0; i < k.recs.size(); ++i) {
            const auto& r = k.recs[i];
            std::fprintf(f, "%zu,%lld,%lld,%lld,%u,%d,%.4f,%.4f,%d\n", i, static_cast<long long>(r.pts),
                         static_cast<long long>(r.commit_ns), static_cast<long long>(r.flip_ns), r.seq, r.held_at_commit,
                         r.overlay_draw_ms, r.commit_ms, r.ndets);
        }
        std::fclose(f);
    }
    if (FILE* f = std::fopen((a.out_dir + "/" + a.label + "_log.txt").c_str(), "w")) {
        for (const auto& l : k.log) std::fprintf(f, "%s\n", l.c_str());
        std::fclose(f);
    }

    // 정리: 평면을 끄고 FB 를 지운다. master 를 놓으면 fbcon/다음 master 가 화면을 다시 잡는다.
    {
        drmModeAtomicReqPtr req = drmModeAtomicAlloc();
        disable_plane(req, *k.vplane);
        if (k.oplane) disable_plane(req, *k.oplane);
        drmModeAtomicCommit(k.fd, req, DRM_MODE_ATOMIC_ALLOW_MODESET, nullptr);
        drmModeAtomicFree(req);
    }
    k.on_screen.reset();
    k.pending.reset();
    k.ahead.clear();
    for (auto& [key, v] : k.fbs) {
        drmModeRmFB(k.fd, v.fb);
        drmCloseBufferHandle(k.fd, v.hy);
        if (v.huv != v.hy) drmCloseBufferHandle(k.fd, v.huv);
    }
    for (auto& b : k.ov) destroy_dumb(k.fd, b);
    if (k.mode_blob) drmModeDestroyPropertyBlob(k.fd, k.mode_blob);
    drmDropMaster(k.fd);
    close(k.fd);
    return 0;
}
