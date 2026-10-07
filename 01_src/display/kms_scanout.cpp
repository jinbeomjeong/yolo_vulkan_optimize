#include "display/kms_scanout.hpp"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>

#include <drm_fourcc.h>
#include <fcntl.h>
#include <poll.h>
#include <sys/stat.h>
#include <unistd.h>
#include <xf86drm.h>
#include <xf86drmMode.h>

namespace yvo::display {

namespace {

using SteadyClock = std::chrono::steady_clock;

std::string errno_text(int e) {
    if (e < 0) return "request construction failed (property missing or allocation failed)";
    return std::string(std::strerror(e)) + " (errno " + std::to_string(e) + ")";
}

const char* connector_type_name(std::uint32_t t) {
    switch (t) {
    case DRM_MODE_CONNECTOR_HDMIA: return "HDMI-A";
    case DRM_MODE_CONNECTOR_HDMIB: return "HDMI-B";
    case DRM_MODE_CONNECTOR_DSI: return "DSI";
    case DRM_MODE_CONNECTOR_DPI: return "DPI";
    case DRM_MODE_CONNECTOR_Composite: return "Composite";
    case DRM_MODE_CONNECTOR_WRITEBACK: return "Writeback";
    case DRM_MODE_CONNECTOR_VIRTUAL: return "Virtual";
    default: return "Unknown";
    }
}

drmModeAtomicReqPtr req_of(void* p) { return static_cast<drmModeAtomicReqPtr>(p); }

// RAII for drmModeAtomicReq
struct AtomicReq {
    drmModeAtomicReqPtr p = drmModeAtomicAlloc();
    ~AtomicReq() {
        if (p) drmModeAtomicFree(p);
    }
    AtomicReq() = default;
    AtomicReq(const AtomicReq&) = delete;
    AtomicReq& operator=(const AtomicReq&) = delete;
};

// Calls drmModeAtomicCommit and returns errno **immediately** (0 = success). Captured before later calls (free, etc.) overwrite errno.
int atomic_commit(int fd, drmModeAtomicReqPtr req, std::uint32_t flags, void* user) {
    if (drmModeAtomicCommit(fd, req, flags, user) == 0) return 0;
    const int e = errno;
    return e != 0 ? e : EIO;
}

// drmHandleEvent only passes user_data. user_data carries the token, and the instance is found via this pointer during dispatch.
thread_local KmsScanout* t_dispatching = nullptr;

bool mode_interlaced(const drmModeModeInfo& m) { return (m.flags & DRM_MODE_FLAG_INTERLACE) != 0; }

} // namespace

// ── Property table ────────────────────────────────────────────────────────────────────────────
std::uint32_t KmsScanout::Props::id(const std::string& n) const {
    auto it = m.find(n);
    return it == m.end() ? 0 : it->second.id;
}

const KmsScanout::Prop* KmsScanout::Props::get(const std::string& n) const {
    auto it = m.find(n);
    return it == m.end() ? nullptr : &it->second;
}

KmsScanout::Props KmsScanout::load_props(std::uint32_t obj, std::uint32_t type) const {
    Props p;
    drmModeObjectPropertiesPtr props = drmModeObjectGetProperties(fd_, obj, type);
    for (std::uint32_t i = 0; props && i < props->count_props; ++i) {
        drmModePropertyPtr pr = drmModeGetProperty(fd_, props->props[i]);
        if (!pr) continue;
        Prop q;
        q.id = pr->prop_id;
        q.value = props->prop_values[i];
        q.flags = pr->flags;
        if (drm_property_type_is(pr, DRM_MODE_PROP_ENUM))
            for (int e = 0; e < pr->count_enums; ++e) q.enums[pr->enums[e].name] = pr->enums[e].value;
        if ((drm_property_type_is(pr, DRM_MODE_PROP_RANGE) || drm_property_type_is(pr, DRM_MODE_PROP_SIGNED_RANGE)) &&
            pr->count_values >= 2) {
            q.min = pr->values[0];
            q.max = pr->values[1];
        }
        p.m[pr->name] = std::move(q);
        drmModeFreeProperty(pr);
    }
    drmModeFreeObjectProperties(props);
    return p;
}

void KmsScanout::load_planes() {
    planes_.clear();
    drmModePlaneResPtr pres = drmModeGetPlaneResources(fd_);
    if (!pres) {
        const int e = errno;
        throw std::runtime_error("kms: drmModeGetPlaneResources failed: " + errno_text(e));
    }
    for (std::uint32_t i = 0; i < pres->count_planes; ++i) {
        drmModePlanePtr pl = drmModeGetPlane(fd_, pres->planes[i]);
        if (!pl) continue;
        Plane pc;
        pc.id = pl->plane_id;
        pc.possible_crtcs = pl->possible_crtcs;
        pc.cur_crtc = pl->crtc_id;
        drmModeFreePlane(pl);
        pc.props = load_props(pc.id, DRM_MODE_OBJECT_PLANE);
        if (const Prop* t = pc.props.get("type"))
            for (const auto& [n, v] : t->enums)
                if (v == t->value) pc.type = n;
        const Prop* inf = pc.props.get("IN_FORMATS");
        if (inf != nullptr && inf->value != 0) {
            if (drmModePropertyBlobPtr blob = drmModeGetPropertyBlob(fd_, static_cast<std::uint32_t>(inf->value))) {
                drmModeFormatModifierIterator it{};
                while (drmModeFormatModifierBlobIterNext(blob, &it)) {
                    // SAND128 is judged on the modifier with the column-height parameter removed (same rule as drm_state.cpp)
                    if (it.fmt == DRM_FORMAT_NV12 && fourcc_mod_broadcom_mod(it.mod) == DRM_FORMAT_MOD_BROADCOM_SAND128)
                        pc.nv12_sand = true;
                    if (it.fmt == DRM_FORMAT_ABGR8888 && it.mod == DRM_FORMAT_MOD_LINEAR) pc.ab24_lin = true;
                }
                drmModeFreePropertyBlob(blob);
            }
        }
        planes_.push_back(std::move(pc));
    }
    drmModeFreePlaneResources(pres);
}

// ── Construction ──────────────────────────────────────────────────────────────────────────────
KmsScanout::KmsScanout(const KmsScanoutOptions& opt) : opt_(opt) {
    fd_ = ::open(opt_.card.c_str(), O_RDWR | O_CLOEXEC);
    if (fd_ < 0) {
        const int e = errno;
        throw std::runtime_error("kms: failed to open " + opt_.card + ": " + errno_text(e));
    }
    try {
        if (drmSetClientCap(fd_, DRM_CLIENT_CAP_UNIVERSAL_PLANES, 1) != 0) {
            const int e = errno;
            throw std::runtime_error("kms: DRM_CLIENT_CAP_UNIVERSAL_PLANES failed: " + errno_text(e));
        }
        if (drmSetClientCap(fd_, DRM_CLIENT_CAP_ATOMIC, 1) != 0) {
            const int e = errno;
            throw std::runtime_error("kms: DRM_CLIENT_CAP_ATOMIC failed (atomic not supported): " + errno_text(e));
        }
        std::uint64_t cap = 0;
        ts_monotonic_ = drmGetCap(fd_, DRM_CAP_TIMESTAMP_MONOTONIC, &cap) == 0 && cap != 0;

        acquire_master_or_throw();
        select_output();
        load_planes();
        select_planes();

        crtc_props_ = load_props(crtc_id_, DRM_MODE_OBJECT_CRTC);
        conn_props_ = load_props(conn_id_, DRM_MODE_OBJECT_CONNECTOR);
        if (!conn_props_.id("CRTC_ID")) throw std::runtime_error("kms: connector property CRTC_ID missing");
        for (const char* n : {"MODE_ID", "ACTIVE"})
            if (!crtc_props_.id(n)) throw std::runtime_error(std::string("kms: CRTC property ") + n + " missing");
        prepare_modeset_extras();
        // The mode blob can be created without master (CREATEPROPBLOB is not a DRM_MASTER ioctl)
        if (drmModeCreatePropertyBlob(fd_, mode_raw_.data(), mode_raw_.size(), &mode_blob_) != 0) {
            const int e = errno;
            throw std::runtime_error("kms: drmModeCreatePropertyBlob(mode) failed: " + errno_text(e));
        }
    } catch (...) {
        if (mode_blob_) drmModeDestroyPropertyBlob(fd_, mode_blob_);
        if (master_) drmDropMaster(fd_);
        ::close(fd_);
        fd_ = -1;
        throw;
    }
    if (opt_.verbose) std::printf("[kms] %s\n", describe().c_str());
}

KmsScanout::MasterProbe KmsScanout::probe_master(const std::string& card) {
    const int fd = ::open(card.c_str(), O_RDWR | O_CLOEXEC);
    if (fd < 0) return MasterProbe::Unavailable;
    MasterProbe r = MasterProbe::OtherMaster;
    if (drmIsMaster(fd)) {
        drmDropMaster(fd);
        r = MasterProbe::NoMaster;
    }
    ::close(fd);
    return r;
}

std::string KmsScanout::find_master_holders(const std::string& dri_debug_dir) {
    std::ifstream in(dri_debug_dir + "/clients");
    if (!in) return "";
    std::string line, out;
    bool header = true;
    while (std::getline(in, line)) {
        if (header) {   // "command  tgid dev master a uid magic ..."
            header = false;
            continue;
        }
        if (line.size() < 21) continue;
        // command is right-aligned in 20 columns (%20s) — some comm values contain spaces, so cut by width
        std::string cmd = line.substr(0, 20);
        cmd.erase(0, cmd.find_first_not_of(' '));
        std::istringstream rest(line.substr(20));
        long tgid = 0;
        int dev = 0;
        std::string master;
        if (!(rest >> tgid >> dev >> master)) continue;
        if (master != "y") continue;
        if (!out.empty()) out += ", ";
        out += cmd + "(" + std::to_string(tgid) + ")";
    }
    return out.empty() ? "(none)" : out;
}

void KmsScanout::acquire_master_or_throw() {
    // If master was free at open time, the kernel has already made this file master.
    master_ = drmIsMaster(fd_) != 0;
    if (!opt_.acquire_master) {
        // Test mode: never hold master. If granted automatically, release it immediately (= restore the pre-open state).
        if (master_) drmDropMaster(fd_);
        master_ = false;
        return;
    }
    if (master_) return;
    if (drmSetMaster(fd_) == 0) {
        master_ = true;
        return;
    }
    const int e = errno;
    std::string holders = find_master_holders(opt_.dri_debug_dir);
    if (holders.empty()) holders = "unknown (could not read " + opt_.dri_debug_dir + "/clients — root:sudo 0750)";
    throw KmsMasterError("kms: failed to obtain DRM master of " + opt_.card + " (drmSetMaster: " + errno_text(e) +
                         "). Current master: " + holders +
                         ". Run after deactivating the desktop session: loginctl activate <non-desktop session id> "
                         "(no sudo needed, or sudo chvt N). lightdm does not need to be stopped. Do not switch VT while running");
}

void KmsScanout::select_output() {
    drmModeResPtr res = drmModeGetResources(fd_);
    if (!res) {
        const int e = errno;
        throw std::runtime_error("kms: drmModeGetResources failed: " + errno_text(e));
    }
    // possible_crtcs is a 32-bit mask — CRTCs with index 32 or higher cannot be represented, so they are skipped (B7).
    const int n_crtcs = std::min(res->count_crtcs, 32);
    auto crtc_unused = [&](std::uint32_t id) {
        bool unused = false;
        if (drmModeCrtcPtr c = drmModeGetCrtc(fd_, id)) {
            unused = !c->mode_valid && c->buffer_id == 0;
            drmModeFreeCrtc(c);
        }
        return unused;
    };
    std::string seen;
    for (int i = 0; i < res->count_connectors && !conn_id_; ++i) {
        drmModeConnectorPtr c = drmModeGetConnector(fd_, res->connectors[i]);
        if (!c) continue;
        const std::string nm = std::string(connector_type_name(c->connector_type)) + "-" +
                               std::to_string(c->connector_type_id);
        const bool connected = c->connection == DRM_MODE_CONNECTED && c->count_modes > 0;
        seen += (seen.empty() ? "" : ", ") + nm + (connected ? "(connected)" : "");
        const bool want = opt_.connector.empty() ? (c->connector_type == DRM_MODE_CONNECTOR_HDMIA && connected)
                                                 : nm == opt_.connector;
        if (!want) {
            drmModeFreeConnector(c);
            continue;
        }
        if (!connected) {
            drmModeFreeConnector(c);
            drmModeFreeResources(res);
            throw std::runtime_error("kms: connector " + nm + " is not connected (no modes)");
        }
        // Mode: interlaced excluded (B3). 1920x1080@60 → preferred → first progressive mode
        const drmModeModeInfo* pick = nullptr;
        for (int m = 0; m < c->count_modes && !pick; ++m)
            if (!mode_interlaced(c->modes[m]) && c->modes[m].hdisplay == 1920 && c->modes[m].vdisplay == 1080 &&
                c->modes[m].vrefresh == 60)
                pick = &c->modes[m];
        for (int m = 0; m < c->count_modes && !pick; ++m)
            if (!mode_interlaced(c->modes[m]) && (c->modes[m].type & DRM_MODE_TYPE_PREFERRED)) pick = &c->modes[m];
        for (int m = 0; m < c->count_modes && !pick; ++m)
            if (!mode_interlaced(c->modes[m])) pick = &c->modes[m];
        if (!pick) {
            drmModeFreeConnector(c);
            drmModeFreeResources(res);
            throw std::runtime_error("kms: connector " + nm + " has no progressive mode (interlaced only)");
        }
        conn_id_ = c->connector_id;
        conn_name_ = nm;
        mode_raw_.assign(reinterpret_cast<const std::uint8_t*>(pick),
                         reinterpret_cast<const std::uint8_t*>(pick) + sizeof(drmModeModeInfo));
        mode_w_ = pick->hdisplay;
        mode_h_ = pick->vdisplay;
        mode_hz_ = static_cast<int>(pick->vrefresh);
        // Exact refresh period from the timings (vrefresh is rounded); feeds the late-frame seq gap rule. Same terms as
        // the kernel's drm_mode_vrefresh(): DBLSCAN doubles and vscan > 1 multiplies the denominator (interlaced modes
        // are never picked above). Out of u32 range or a zero term = 0 (unknown).
        {
            std::uint64_t den = static_cast<std::uint64_t>(pick->htotal) * pick->vtotal;
            if (pick->flags & DRM_MODE_FLAG_DBLSCAN) den *= 2;
            if (pick->vscan > 1) den *= pick->vscan;
            const std::uint64_t ns = pick->clock && den ? (1000000ull * den + pick->clock / 2) / pick->clock : 0;   // clock kHz
            refresh_ns_ = ns > 0 && ns <= UINT32_MAX ? static_cast<std::uint32_t>(ns) : 0u;
        }
        mode_name_ = pick->name;
        // CRTC: CRTC the encoder currently uses → possible CRTC not in use → first possible CRTC (B4)
        std::uint32_t cur_crtc = 0;
        if (c->encoder_id) {
            if (drmModeEncoderPtr e = drmModeGetEncoder(fd_, c->encoder_id)) {
                cur_crtc = e->crtc_id;
                drmModeFreeEncoder(e);
            }
        }
        for (int ci = 0; ci < n_crtcs && cur_crtc; ++ci)
            if (res->crtcs[ci] == cur_crtc) {
                crtc_id_ = cur_crtc;
                crtc_idx_ = ci;
            }
        std::uint32_t possible = 0;
        for (int ei = 0; ei < c->count_encoders; ++ei)
            if (drmModeEncoderPtr e = drmModeGetEncoder(fd_, c->encoders[ei])) {
                possible |= e->possible_crtcs;
                drmModeFreeEncoder(e);
            }
        for (int pass = 0; pass < 2 && !crtc_id_; ++pass)
            for (int ci = 0; ci < n_crtcs; ++ci)
                if ((possible & (1u << ci)) && (pass == 1 || crtc_unused(res->crtcs[ci]))) {
                    crtc_id_ = res->crtcs[ci];
                    crtc_idx_ = ci;
                    break;
                }
        drmModeFreeConnector(c);
    }
    drmModeFreeResources(res);
    if (!conn_id_)
        throw std::runtime_error("kms: " +
                                 (opt_.connector.empty() ? std::string("connected HDMI-A connector") : "connector " + opt_.connector) +
                                 " not found (available: " + seen + ")");
    if (!crtc_id_ || crtc_idx_ < 0 || crtc_idx_ >= 32)
        throw std::runtime_error("kms: no usable CRTC for connector " + conn_name_);
}

void KmsScanout::select_planes() {
    const std::uint32_t bit = 1u << crtc_idx_;
    // Do not steal planes used by another output (CRTC): prefer planes that are free now or attached to our CRTC, then any (B2).
    auto free_or_ours = [&](const Plane& p) { return p.cur_crtc == 0 || p.cur_crtc == crtc_id_; };
    // Video: prefer an NV12 SAND128 Primary, else an Overlay
    for (int pass = 0; pass < 2 && !vplane_; ++pass)
        for (const char* want : {"Primary", "Overlay"}) {
            for (const Plane& p : planes_)
                if ((p.possible_crtcs & bit) && p.nv12_sand && p.type == want && (pass == 1 || free_or_ours(p))) {
                    vplane_ = &p;
                    break;
                }
            if (vplane_) break;
        }
    if (!vplane_) throw std::runtime_error("kms: no NV12 SAND128 plane for CRTC " + std::to_string(crtc_id_));
    for (int pass = 0; pass < 2 && !oplane_; ++pass)
        for (const Plane& p : planes_)
            if (&p != vplane_ && (p.possible_crtcs & bit) && p.ab24_lin && p.type == "Overlay" &&
                (pass == 1 || free_or_ours(p))) {
                oplane_ = &p;
                break;
            }
    if (!oplane_) throw std::runtime_error("kms: no AB24 LINEAR Overlay plane for CRTC " + std::to_string(crtc_id_));
    for (const Plane* p : {vplane_, oplane_})
        for (const char* n : {"FB_ID", "CRTC_ID", "SRC_X", "SRC_Y", "SRC_W", "SRC_H", "CRTC_X", "CRTC_Y", "CRTC_W", "CRTC_H"})
            if (!p->props.id(n))
                throw std::runtime_error("kms: plane " + std::to_string(p->id) + " has no property " + n);

    // zpos: box plane above video. The Primary's zpos is usually immutable (its current value is used as the reference).
    // If mutable, its min is set explicitly on every commit (B1 — do not rely on values left by the previous master).
    std::uint64_t vz = 0;
    if (const Prop* z = vplane_->props.get("zpos")) {
        video_zpos_settable_ = (z->flags & DRM_MODE_PROP_IMMUTABLE) == 0;
        vz = video_zpos_settable_ ? z->min : z->value;
        video_zpos_ = vz;
    }
    if (const Prop* z = oplane_->props.get("zpos")) {
        if (z->flags & DRM_MODE_PROP_IMMUTABLE) {
            overlay_zpos_ = z->value;
            overlay_zpos_settable_ = false;
            if (overlay_zpos_ <= vz)
                throw std::runtime_error("kms: box plane zpos (immutable " + std::to_string(overlay_zpos_) +
                                         ") is not above video plane zpos " + std::to_string(vz));
        } else {
            overlay_zpos_ = std::clamp<std::uint64_t>(vz + 1, z->min, z->max);
            overlay_zpos_settable_ = true;
            if (overlay_zpos_ <= vz)
                throw std::runtime_error("kms: box plane zpos range cannot go above the video plane zpos");
        }
    }
}

// At modeset, put the color pipeline to known defaults (B11). Even if the previous master changed gamma/CTM/Broadcast RGB,
// our judgement (BT.709 limited) goes to the screen as is. Set only if the property exists and is mutable. Restoring is done by capture_state.
void KmsScanout::prepare_modeset_extras() {
    modeset_extras_.clear();
    for (const char* n : {"GAMMA_LUT", "CTM"}) {
        const Prop* p = crtc_props_.get(n);
        if (p && !(p->flags & DRM_MODE_PROP_IMMUTABLE))
            modeset_extras_.push_back({"crtc", crtc_id_, n, 0, std::string(n) + "=0"});
    }
    if (const Prop* p = conn_props_.get("Broadcast RGB"); p && !(p->flags & DRM_MODE_PROP_IMMUTABLE)) {
        auto it = p->enums.find("Automatic");
        if (it != p->enums.end())
            modeset_extras_.push_back({"connector", conn_id_, "Broadcast RGB", it->second, "Broadcast RGB=Automatic"});
    }
}

KmsScanout::~KmsScanout() {
    try {
        if (!shut_ && master_) shutdown(0);
    } catch (...) {
    }
    if (fd_ < 0) return;
    clear_video_cache();
    for (OverlayFb& o : overlay_) {
        if (o.fb) drmModeRmFB(fd_, o.fb);
        if (o.handle) release_handle(o.handle);
    }
    overlay_.clear();
    for (const auto& [h, n] : handle_refs_) drmCloseBufferHandle(fd_, h);   // prevent leaks if anything remains (bug)
    handle_refs_.clear();
    if (mode_blob_) drmModeDestroyPropertyBlob(fd_, mode_blob_);
    if (master_) drmDropMaster(fd_);
    ::close(fd_);
    fd_ = -1;
}

VkExtent2D KmsScanout::output_extent() const {
    return VkExtent2D{static_cast<std::uint32_t>(mode_w_), static_cast<std::uint32_t>(mode_h_)};
}

// ── GEM handles ───────────────────────────────────────────────────────────────────────────────
// drmPrimeFDToHandle returns the same handle for the same dma-buf, and handles have no refcount (one close makes them disappear).
// So we count them — so they are not closed too early even if Y/UV are one object or two FBs point at the same dma-buf.
bool KmsScanout::import_handle(int fd, std::uint32_t* handle, int* err) {
    if (drmPrimeFDToHandle(fd_, fd, handle) != 0) {
        *err = errno;
        return false;
    }
    ++handle_refs_[*handle];
    return true;
}

void KmsScanout::release_handle(std::uint32_t handle) {
    auto it = handle_refs_.find(handle);
    if (it == handle_refs_.end()) return;
    if (--it->second <= 0) {
        drmCloseBufferHandle(fd_, handle);
        handle_refs_.erase(it);
    }
}

void KmsScanout::destroy_video_fb(VideoFb& v) {
    if (v.fb) drmModeRmFB(fd_, v.fb);
    if (v.hy) release_handle(v.hy);
    if (v.huv) release_handle(v.huv);
    v = VideoFb{};
}

void KmsScanout::fail(const std::string& msg) {
    last_error_ = msg;
    // If the same error occurs every frame the log floods — print only the first few times and every 100th after that.
    if (commit_fail_ <= 5 || commit_fail_ % 100 == 0) std::fprintf(stderr, "[kms] %s\n", msg.c_str());
}

void KmsScanout::on_master_lost(const std::string& where) {
    if (master_lost_) return;
    master_lost_ = true;
    master_ = false;
    last_error_ = "lost DRM master (" + where + ": EACCES)";
    std::fprintf(stderr,
                 "[kms] warning: %s — another process became master (VT switch, etc.). All later commits are dropped"
                 " (current master: %s)\n",
                 last_error_.c_str(), find_master_holders(opt_.dri_debug_dir).c_str());
}

// ── FB ────────────────────────────────────────────────────────────────────────────────────────
void KmsScanout::register_overlay(const OverlayPlane& ov) {
    for (OverlayFb& o : overlay_) {
        if (o.fb) drmModeRmFB(fd_, o.fb);
        if (o.handle) release_handle(o.handle);
    }
    overlay_.clear();
    for (int i = 0; i < ov.ring_size(); ++i) {
        const OverlayPlaneBuffer& b = ov.buffer(i);
        OverlayFb o;
        int e = 0;
        // The fd stays owned by OverlayPlane. drmPrimeFDToHandle does not consume the fd (GEM holds its own dma-buf reference).
        if (!import_handle(b.fd, &o.handle, &e))
            throw std::runtime_error("kms: box buffer #" + std::to_string(i) + " drmPrimeFDToHandle failed: " + errno_text(e));
        std::uint32_t handles[4] = {o.handle, 0, 0, 0};
        std::uint32_t pitches[4] = {b.pitch, 0, 0, 0};
        std::uint32_t offsets[4] = {b.offset, 0, 0, 0};
        std::uint64_t mods[4] = {b.modifier, 0, 0, 0};
        if (drmModeAddFB2WithModifiers(fd_, b.width, b.height, b.fourcc, handles, pitches, offsets, mods, &o.fb,
                                       DRM_MODE_FB_MODIFIERS) != 0) {
            e = errno;
            release_handle(o.handle);
            throw std::runtime_error("kms: box buffer #" + std::to_string(i) + " AddFB2(AB24 LINEAR " +
                                     std::to_string(b.width) + "x" + std::to_string(b.height) + " pitch " +
                                     std::to_string(b.pitch) + ") failed: " + errno_text(e));
        }
        o.w = b.width;
        o.h = b.height;
        overlay_.push_back(o);
    }
}

std::uint32_t KmsScanout::overlay_fb(int idx) const {
    if (idx < 0 || idx >= static_cast<int>(overlay_.size())) return 0;
    return overlay_[static_cast<std::size_t>(idx)].fb;
}

std::uint32_t KmsScanout::video_fb_for(const DrmFrame& f) {
    struct stat sy{}, suv{};
    if (f.fd < 0 || f.uv_fd < 0 || fstat(f.fd, &sy) != 0 || fstat(f.uv_fd, &suv) != 0) {
        last_error_ = "video frame fd is invalid (fd=" + std::to_string(f.fd) + "/" + std::to_string(f.uv_fd) + ")";
        return 0;
    }
    const VideoKey key{static_cast<std::uint64_t>(sy.st_dev), static_cast<std::uint64_t>(sy.st_ino),
                       static_cast<std::uint64_t>(suv.st_dev), static_cast<std::uint64_t>(suv.st_ino)};
    auto it = video_.find(key);
    if (it != video_.end()) {
        VideoFb& v = it->second;
        if (v.w == f.width && v.h == f.height && v.modifier == f.modifier) return v.fb;
        // Same dma-buf but the descriptor changed (resolution change). Do not remove FBs on screen or awaiting flip (RmFB turns the plane off).
        if ((on_screen_.valid && on_screen_.video_fb == v.fb) || (in_flight_.valid && in_flight_.video_fb == v.fb)) {
            last_error_ = "video buffer with changed descriptor is still being scanned out";
            return 0;
        }
        destroy_video_fb(v);
        video_.erase(it);
    }
    if (f.drm_format != DRM_FORMAT_NV12) {
        char b[64];
        std::snprintf(b, sizeof(b), "unsupported video format fourcc=0x%08x", f.drm_format);
        last_error_ = b;
        return 0;
    }
    VideoFb v;
    int e = 0;
    if (!import_handle(f.fd, &v.hy, &e)) {
        last_error_ = "drmPrimeFDToHandle(Y) failed: " + errno_text(e);
        return 0;
    }
    if (!import_handle(f.uv_fd, &v.huv, &e)) {
        last_error_ = "drmPrimeFDToHandle(UV) failed: " + errno_text(e);
        release_handle(v.hy);   // probe defect fix: do not leak the Y handle
        return 0;
    }
    // SAND128: pitch field of each plane = SAND column height (Y 1088 / UV 544). Kernel vc4 HVS6 convention (frame_types.hpp DrmFrame).
    std::uint32_t handles[4] = {v.hy, v.huv, 0, 0};
    std::uint32_t pitches[4] = {static_cast<std::uint32_t>(f.y_pitch), static_cast<std::uint32_t>(f.uv_pitch), 0, 0};
    std::uint32_t offsets[4] = {static_cast<std::uint32_t>(f.y_offset), static_cast<std::uint32_t>(f.uv_offset), 0, 0};
    std::uint64_t mods[4] = {f.modifier, f.modifier, 0, 0};
    if (drmModeAddFB2WithModifiers(fd_, static_cast<std::uint32_t>(f.width), static_cast<std::uint32_t>(f.height),
                                   DRM_FORMAT_NV12, handles, pitches, offsets, mods, &v.fb, DRM_MODE_FB_MODIFIERS) != 0) {
        e = errno;   // capture before snprintf overwrites errno (B5)
        char b[160];
        std::snprintf(b, sizeof(b), "AddFB2(NV12 %dx%d mod=0x%016llx pitch(colh)=%zu/%zu) failed: ", f.width, f.height,
                      static_cast<unsigned long long>(f.modifier), f.y_pitch, f.uv_pitch);
        last_error_ = b + errno_text(e);
        release_handle(v.hy);
        release_handle(v.huv);
        return 0;
    }
    v.w = f.width;
    v.h = f.height;
    v.modifier = f.modifier;
    video_.emplace(key, v);
    if (opt_.verbose)
        std::printf("[kms] new video FB %u (#%zu) %dx%d mod=0x%016llx pitch(colh)=%zu/%zu handles %u/%u\n", v.fb,
                    video_.size(), f.width, f.height, static_cast<unsigned long long>(f.modifier), f.y_pitch,
                    f.uv_pitch, v.hy, v.huv);
    return v.fb;
}

// ── State save·restore ────────────────────────────────────────────────────────────────────────
void KmsScanout::capture_state() {
    saved_.clear();
    auto save = [&](const char* kind, std::uint32_t obj, const Props& p, const char* name) {
        const Prop* q = p.get(name);
        if (!q || (q->flags & DRM_MODE_PROP_IMMUTABLE)) return;
        saved_.push_back(SavedProp{kind, obj, name, q->id, q->value});
    };
    // Re-read the current values (values from construction time may be stale)
    const Props crtc = load_props(crtc_id_, DRM_MODE_OBJECT_CRTC);
    for (const char* n : {"MODE_ID", "ACTIVE", "GAMMA_LUT", "CTM"}) save("crtc", crtc_id_, crtc, n);
    const Props conn = load_props(conn_id_, DRM_MODE_OBJECT_CONNECTOR);
    for (const char* n : {"CRTC_ID", "Broadcast RGB"}) save("connector", conn_id_, conn, n);
    static const char* const kPlaneProps[] = {"FB_ID",  "CRTC_ID", "SRC_X",    "SRC_Y",            "SRC_W",
                                              "SRC_H",  "CRTC_X",  "CRTC_Y",   "CRTC_W",           "CRTC_H",
                                              "zpos",   "rotation", "alpha",   "pixel blend mode", "COLOR_ENCODING",
                                              "COLOR_RANGE"};
    for (const Plane& p : planes_) {
        if (!(p.possible_crtcs & (1u << crtc_idx_))) continue;
        const Props pp = load_props(p.id, DRM_MODE_OBJECT_PLANE);
        const Prop* c = pp.get("CRTC_ID");
        const bool ours = &p == vplane_ || &p == oplane_;
        // Save only planes attached to our CRTC (the modeset turns them off) and the two planes we will use (the modeset overwrites them)
        if (!ours && (c == nullptr || c->value != crtc_id_)) continue;
        for (const char* n : kPlaneProps) save("plane", p.id, pp, n);
    }
    state_captured_ = true;
}

std::string KmsScanout::saved_state_text() const {
    std::string out;
    for (const SavedProp& s : saved_)
        out += s.kind + "[" + std::to_string(s.obj) + "]." + s.name + "=" + std::to_string(s.value) + "\n";
    return out;
}

int KmsScanout::commit_restore() {
    if (!state_captured_ || saved_.empty()) return -1;
    AtomicReq req;
    if (!req.p) return -1;
    for (const SavedProp& s : saved_)
        if (drmModeAtomicAddProperty(req.p, s.obj, s.prop_id, s.value) < 0) return -1;
    return atomic_commit(fd_, req.p, DRM_MODE_ATOMIC_ALLOW_MODESET, nullptr);
}

int KmsScanout::commit_planes_off(bool crtc_off) {
    AtomicReq req;
    if (!req.p) return -1;
    bool ok = add(req.p, vplane_->id, vplane_->props, "FB_ID", 0, true) &&
              add(req.p, vplane_->id, vplane_->props, "CRTC_ID", 0, true) &&
              add(req.p, oplane_->id, oplane_->props, "FB_ID", 0, true) &&
              add(req.p, oplane_->id, oplane_->props, "CRTC_ID", 0, true);
    if (crtc_off)
        ok = ok && add(req.p, crtc_id_, crtc_props_, "ACTIVE", 0, true) &&
             add(req.p, crtc_id_, crtc_props_, "MODE_ID", 0, true) && add(req.p, conn_id_, conn_props_, "CRTC_ID", 0, true);
    if (!ok) return -1;   // distinguish request construction failure from ioctl failure (B6)
    return atomic_commit(fd_, req.p, DRM_MODE_ATOMIC_ALLOW_MODESET, nullptr);
}

// ── Atomic request ────────────────────────────────────────────────────────────────────────────
bool KmsScanout::add(void* req, std::uint32_t obj, const Props& p, const char* name, std::uint64_t value, bool required) {
    const std::uint32_t id = p.id(name);
    if (id == 0) {
        if (required) last_error_ = std::string("property ") + name + " missing (obj " + std::to_string(obj) + ")";
        return !required;
    }
    if (drmModeAtomicAddProperty(req_of(req), obj, id, value) < 0) {
        last_error_ = std::string("drmModeAtomicAddProperty(") + name + ") failed";
        return false;
    }
    return true;
}

std::uint64_t KmsScanout::color_encoding_value() const {
    const Prop* p = vplane_->props.get("COLOR_ENCODING");
    if (!p) return 0;
    auto it = p->enums.find("ITU-R BT.709 YCbCr");
    return it != p->enums.end() ? it->second : p->value;
}

std::uint64_t KmsScanout::color_range_value() const {
    const Prop* p = vplane_->props.get("COLOR_RANGE");
    if (!p) return 0;
    // In KMS the requested value goes to the plane as is (no labwc inversion). auto = limited = decoder output (HEVC TV range).
    const char* want = opt_.color_range == ColorRangeRequest::Full ? "YCbCr full range" : "YCbCr limited range";
    auto it = p->enums.find(want);
    return it != p->enums.end() ? it->second : p->value;
}

const char* KmsScanout::color_range_text() const {
    return opt_.color_range == ColorRangeRequest::Full ? "full" : "limited";
}

bool KmsScanout::build_request(void* req, std::uint32_t video_fb, int vw, int vh, int overlay_idx, bool modeset) {
    const Props& vp = vplane_->props;
    const Props& op = oplane_->props;
    if (modeset) {
        if (!add(req, conn_id_, conn_props_, "CRTC_ID", crtc_id_, true)) return false;
        if (!add(req, crtc_id_, crtc_props_, "MODE_ID", mode_blob_, true)) return false;
        if (!add(req, crtc_id_, crtc_props_, "ACTIVE", 1, true)) return false;
        for (const ModesetExtra& x : modeset_extras_) {
            const Props& p = std::string(x.kind) == "crtc" ? crtc_props_ : conn_props_;
            if (!add(req, x.obj, p, x.name.c_str(), x.value, false)) return false;
        }
        // Turn off the other planes currently attached to this CRTC (primary·cursor etc. used by labwc/fbcon). Re-read the actual
        // placement at commit time — planes used by another CRTC (second HDMI) are not touched even if possible_crtcs overlaps.
        for (const Plane& p : planes_) {
            if (&p == vplane_ || &p == oplane_ || !(p.possible_crtcs & (1u << crtc_idx_))) continue;
            std::uint32_t cur = 0;
            if (drmModePlanePtr pl = drmModeGetPlane(fd_, p.id)) {
                cur = pl->crtc_id;
                drmModeFreePlane(pl);
            }
            if (cur != crtc_id_) continue;
            if (!add(req, p.id, p.props, "FB_ID", 0, true) || !add(req, p.id, p.props, "CRTC_ID", 0, true)) return false;
        }
    }
    // Video plane: set everything on every commit (including COLOR_* — reverts it even if another master changed it in between)
    const std::uint32_t vo = vplane_->id;
    if (!add(req, vo, vp, "FB_ID", video_fb, true) || !add(req, vo, vp, "CRTC_ID", crtc_id_, true) ||
        !add(req, vo, vp, "SRC_X", 0, true) || !add(req, vo, vp, "SRC_Y", 0, true) ||
        !add(req, vo, vp, "SRC_W", static_cast<std::uint64_t>(vw) << 16, true) ||
        !add(req, vo, vp, "SRC_H", static_cast<std::uint64_t>(vh) << 16, true) ||
        !add(req, vo, vp, "CRTC_X", 0, true) || !add(req, vo, vp, "CRTC_Y", 0, true) ||
        !add(req, vo, vp, "CRTC_W", static_cast<std::uint64_t>(mode_w_), true) ||
        !add(req, vo, vp, "CRTC_H", static_cast<std::uint64_t>(mode_h_), true))
        return false;
    if (!add(req, vo, vp, "COLOR_ENCODING", color_encoding_value(), false) ||
        !add(req, vo, vp, "COLOR_RANGE", color_range_value(), false))
        return false;
    if (video_zpos_settable_ && !add(req, vo, vp, "zpos", video_zpos_, false)) return false;
    // Box plane
    const std::uint32_t oo = oplane_->id;
    if (overlay_idx >= 0) {
        const OverlayFb& b = overlay_[static_cast<std::size_t>(overlay_idx)];
        if (!add(req, oo, op, "FB_ID", b.fb, true) || !add(req, oo, op, "CRTC_ID", crtc_id_, true) ||
            !add(req, oo, op, "SRC_X", 0, true) || !add(req, oo, op, "SRC_Y", 0, true) ||
            !add(req, oo, op, "SRC_W", static_cast<std::uint64_t>(b.w) << 16, true) ||
            !add(req, oo, op, "SRC_H", static_cast<std::uint64_t>(b.h) << 16, true) ||
            !add(req, oo, op, "CRTC_X", 0, true) || !add(req, oo, op, "CRTC_Y", 0, true) ||
            !add(req, oo, op, "CRTC_W", static_cast<std::uint64_t>(mode_w_), true) ||
            !add(req, oo, op, "CRTC_H", static_cast<std::uint64_t>(mode_h_), true))
            return false;
        if (overlay_zpos_settable_ && !add(req, oo, op, "zpos", overlay_zpos_, false)) return false;
        if (const Prop* bm = op.get("pixel blend mode")) {
            auto it = bm->enums.find("Pre-multiplied");
            if (it != bm->enums.end() && !add(req, oo, op, "pixel blend mode", it->second, false)) return false;
        }
        if (!add(req, oo, op, "alpha", 0xffff, false)) return false;
    } else if (modeset) {
        // If the first commit has no boxes, turn the box plane off (so the previous master's contents do not remain)
        if (!add(req, oo, op, "FB_ID", 0, true) || !add(req, oo, op, "CRTC_ID", 0, true)) return false;
    }
    // If overlay_idx < 0 and not a modeset, box plane properties are not set → previous boxes stay
    return true;
}

// ── Commit ────────────────────────────────────────────────────────────────────────────────────
bool KmsScanout::commit(const DrmFrame& video, std::uint64_t token, int overlay_idx) {
    if (shut_ || fd_ < 0 || !fatal_.empty()) return false;
    if (master_lost_) {   // the warning was issued only once in on_master_lost. After that, drop silently (no ioctl)
        ++lost_drops_;
        return false;
    }
    if (!master_) {
        ++commit_fail_;
        fail("commit rejected: no DRM master of " + opt_.card + " (current master: " +
             find_master_holders(opt_.dri_debug_dir) + ")");
        return false;
    }
    if (overlay_idx >= static_cast<int>(overlay_.size())) return false;
    check_flip_watchdog();
    if (in_flight_.valid) {
        // The previous flip has not arrived yet. A non-blocking commit would get EBUSY — reject without an ioctl (PlaneDisplay counts it as a drop).
        ++busy_;
        return false;
    }
    const std::uint32_t vfb = video_fb_for(video);
    if (vfb == 0) {
        ++commit_fail_;
        fail("video FB failed (token " + std::to_string(token) + "): " + last_error_);
        return false;
    }
    const bool modeset = !modeset_done_;
    if (modeset && !state_captured_) {
        capture_state();   // state right before the screen is first changed (shutdown restores it)
        if (opt_.verbose) std::printf("[kms] saved previous KMS state (%zu properties):\n%s", saved_.size(), saved_state_text().c_str());
    }
    AtomicReq req;
    if (!req.p || !build_request(req.p, vfb, video.width, video.height, overlay_idx, modeset)) {
        ++commit_fail_;
        fail("atomic request construction failed: " + last_error_);
        return false;
    }
    if (modeset) {
        // First commit: check with TEST_ONLY first (the screen stays unchanged even on failure).
        const int e = atomic_commit(fd_, req.p, DRM_MODE_ATOMIC_TEST_ONLY | DRM_MODE_ATOMIC_ALLOW_MODESET, nullptr);
        if (e != 0) {
            ++commit_fail_;
            if (e == EACCES) {
                on_master_lost("first modeset TEST_ONLY");
                return false;
            }
            ++modeset_test_failures_;
            fail("first modeset TEST_ONLY failed " + std::to_string(modeset_test_failures_) + "/" +
                 std::to_string(kMaxModesetTestFailures) + ": " + errno_text(e) + " (" + conn_name_ + " " + mode_name_ +
                 " crtc " + std::to_string(crtc_id_) + " video plane " + std::to_string(vplane_->id) + " boxes plane " +
                 std::to_string(oplane_->id) + ")");
            if (modeset_test_failures_ >= kMaxModesetTestFailures) {
                fatal_ = "kms: the first modeset was rejected by TEST_ONLY " + std::to_string(modeset_test_failures_) +
                         " times in a row (last: " + errno_text(e) + "). The driver does not accept this connector·mode·plane combination (" + conn_name_ + " " +
                         mode_name_ + ", crtc " + std::to_string(crtc_id_) + ", plane " + std::to_string(vplane_->id) +
                         "/" + std::to_string(oplane_->id) + ") — exiting";
                std::fprintf(stderr, "[kms] %s\n", fatal_.c_str());
            }
            return false;
        }
        modeset_test_failures_ = 0;
    }
    // PAGE_FLIP_EVENT is attached even to the blocking modeset → the event queues on the fd and is handled in the same flow in dispatch().
    // (B8/B10) This first commit blocks for tens of ms and its flip time is the modeset completion time — exclude the first N frames from measurements.
    const std::uint32_t flags = DRM_MODE_PAGE_FLIP_EVENT |
                                (modeset ? DRM_MODE_ATOMIC_ALLOW_MODESET : DRM_MODE_ATOMIC_NONBLOCK);
    const int e = atomic_commit(fd_, req.p, flags, reinterpret_cast<void*>(static_cast<std::uintptr_t>(token)));
    if (e != 0) {
        ++commit_fail_;
        if (e == EACCES) {
            on_master_lost(modeset ? "first modeset commit" : "non-blocking commit");
            return false;
        }
        if (e == EBUSY) ++busy_;
        fail(std::string(modeset ? "first modeset" : "non-blocking") + " atomic commit failed (token " + std::to_string(token) +
             "): " + errno_text(e));
        return false;
    }
    modeset_done_ = true;
    in_flight_ = Shown{true, token, overlay_idx, vfb};
    in_flight_since_ = SteadyClock::now();
    return true;
}

bool KmsScanout::dispatch(int timeout_ms) {
    if (fd_ < 0) return false;
    pollfd p{fd_, POLLIN, 0};
    const int pr = ::poll(&p, 1, std::max(timeout_ms, 0));
    if (pr < 0) {
        const int e = errno;
        return e == EINTR;
    }
    bool ok = true;
    if (pr > 0) {
        if (p.revents & (POLLERR | POLLHUP | POLLNVAL)) return false;
        drmEventContext ev{};
        ev.version = 3;   // page_flip_handler2 (includes crtc_id)
        ev.page_flip_handler2 = &KmsScanout::on_flip;
        KmsScanout* prev = t_dispatching;
        t_dispatching = this;
        ok = drmHandleEvent(fd_, &ev) == 0;
        t_dispatching = prev;
    }
    check_flip_watchdog();
    return ok;
}

void KmsScanout::on_flip(int /*fd*/, unsigned seq, unsigned sec, unsigned usec, unsigned /*crtc*/, void* data) {
    if (t_dispatching == nullptr) return;
    t_dispatching->handle_flip(static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(data)), true, sec, usec, seq);
}

void KmsScanout::check_flip_watchdog() {
    if (!in_flight_.valid) return;
    if (SteadyClock::now() - in_flight_since_ < std::chrono::milliseconds(kFlipTimeoutMs)) return;
    ++flip_timeouts_;
    if (flip_timeouts_ == 1)
        std::fprintf(stderr,
                     "[kms] warning: flip event did not arrive within %d ms (token %llu) — treating it as on screen and "
                     "continuing (no on_presented, counted in flip_timeouts; further identical warnings suppressed)\n",
                     kFlipTimeoutMs, static_cast<unsigned long long>(in_flight_.token));
    handle_flip(in_flight_.token, false, 0, 0, 0);
}

void KmsScanout::handle_flip(std::uint64_t token, bool have_time, unsigned sec, unsigned usec, unsigned seq) {
    // Accept only events for the commit we are waiting on. Late events of commits already promoted by the watchdog, shutdown commits, etc. are ignored.
    if (!in_flight_.valid || in_flight_.token != token) return;
    if (have_time) ++flips_;
    const Shown now = in_flight_;
    const Shown prev = on_screen_;
    const int prev_overlay = shown_overlay_;
    in_flight_ = Shown{};
    on_screen_ = now;
    if (now.overlay >= 0) shown_overlay_ = now.overlay;
    if (!listener_) return;
    // Order contract: frame_done → presented → release previous video → (if boxes changed) release previous boxes
    listener_->on_frame_done();
    if (have_time) {
        const std::int64_t t_ns =
            static_cast<std::int64_t>(sec) * 1000000000LL + static_cast<std::int64_t>(usec) * 1000LL;
        listener_->on_presented(now.token, t_ns, ts_monotonic_, true, seq, refresh_ns_);
    }
    if (prev.valid) listener_->on_video_released(prev.token);
    if (now.overlay >= 0 && prev_overlay >= 0 && prev_overlay != now.overlay) listener_->on_overlay_released(prev_overlay);
}

// ── Shutdown ──────────────────────────────────────────────────────────────────────────────────
void KmsScanout::shutdown(int timeout_ms) {
    if (shut_) return;
    shut_ = true;
    if (fd_ < 0) return;
    // 1. Wait a bounded time for an in-flight flip (that commit must finish so the shutdown commit does not get EBUSY).
    const auto deadline = SteadyClock::now() + std::chrono::milliseconds(std::max(timeout_ms, 0));
    while (in_flight_.valid && SteadyClock::now() < deadline)
        if (!dispatch(10)) break;
    // 2. Revert the screen: restore saved previous state → on failure turn off both planes → if that also fails, turn off the CRTC too.
    if (!modeset_done_) {
        shutdown_result_ = "not-needed (before modeset)";
    } else if (!master_) {
        shutdown_result_ = master_lost_ ? "skipped (master lost — screen is up to the new master)" : "skipped (no master)";
    } else {
        const int er = commit_restore();
        if (er == 0) {
            shutdown_result_ = "restore ok (" + std::to_string(saved_.size()) + " props)";
        } else {
            const int ep = commit_planes_off(false);
            if (ep == 0) {
                shutdown_result_ = "restore failed (" + errno_text(er) + ") -> planes-off ok";
            } else {
                const int ec = commit_planes_off(true);
                shutdown_result_ = "restore failed (" + errno_text(er) + ") -> planes-off failed (" + errno_text(ep) +
                                   ") -> crtc-off " + (ec == 0 ? std::string("ok") : "failed (" + errno_text(ec) + ")");
            }
        }
        std::printf("[kms] shutdown: %s\n", shutdown_result_.c_str());
        std::fflush(stdout);
    }
    // If the shutdown commit finished (blocking), the HVS no longer reads our buffers → report all remaining ones as released.
    // Even if a late flip event remains on the fd, it is ignored because in_flight_ was cleared.
    PlaneScanoutListener* l = listener_;
    const Shown a = on_screen_, b = in_flight_;
    const int ov_a = shown_overlay_, ov_b = in_flight_.valid ? in_flight_.overlay : -1;
    on_screen_ = in_flight_ = Shown{};
    shown_overlay_ = -1;
    if (l) {
        if (a.valid) l->on_video_released(a.token);
        if (b.valid) l->on_video_released(b.token);
        if (ov_a >= 0) l->on_overlay_released(ov_a);
        if (ov_b >= 0 && ov_b != ov_a) l->on_overlay_released(ov_b);
    }
    listener_ = nullptr;
    // 3. Release master → the next master (labwc on VT return) takes the screen again.
    if (master_) {
        drmDropMaster(fd_);
        master_ = false;
    }
}

void KmsScanout::clear_video_cache() {
    if (fd_ < 0) return;
    for (auto& [k, v] : video_) destroy_video_fb(v);   // RmFB on an FB being scanned out makes the kernel turn that plane off
    video_.clear();
}

std::vector<std::string> KmsScanout::report_lines() const {
    std::vector<std::string> out;
    out.push_back("kms: flips " + std::to_string(flips_) + ", flip_timeouts " + std::to_string(flip_timeouts_) +
                  ", commit failures " + std::to_string(commit_fail_) + ", busy " + std::to_string(busy_) +
                  ", master_lost " + (master_lost_ ? "yes (dropped " + std::to_string(lost_drops_) + ")" : std::string("no")));
    out.push_back("kms: shutdown " + shutdown_result_);
    if (!fatal_.empty()) out.push_back("kms: fatal " + fatal_);
    return out;
}

std::string KmsScanout::describe() const {
    char buf[384];
    const Prop* vz = vplane_ ? vplane_->props.get("zpos") : nullptr;
    std::snprintf(buf, sizeof(buf),
                  "kms %s %s %dx%d@%d crtc=%u video=plane[%u](%s NV12/SAND128 zpos %llu%s) "
                  "boxes=plane[%u](%s AB24 LINEAR zpos %llu%s) color=BT.709/%s%s master=%s ts=%s",
                  opt_.card.c_str(), conn_name_.c_str(), mode_w_, mode_h_, mode_hz_, crtc_id_, video_plane_id(),
                  vplane_ ? vplane_->type.c_str() : "-",
                  static_cast<unsigned long long>(video_zpos_settable_ ? video_zpos_ : (vz ? vz->value : 0)),
                  video_zpos_settable_ ? " set" : " immutable", overlay_plane_id(), oplane_ ? oplane_->type.c_str() : "-",
                  static_cast<unsigned long long>(overlay_zpos_), overlay_zpos_settable_ ? " set" : " immutable",
                  color_range_text(), opt_.color_range == ColorRangeRequest::Auto ? "(auto)" : "",
                  master_ ? "yes" : "no", ts_monotonic_ ? "MONOTONIC" : "other");
    std::string s = buf;
    s += " modeset:";
    if (modeset_extras_.empty()) s += " (no color properties)";
    for (const ModesetExtra& x : modeset_extras_) s += " " + x.text;
    return s;
}

} // namespace yvo::display
