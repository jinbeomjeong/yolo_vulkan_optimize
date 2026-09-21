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
    if (e < 0) return "요청 구성 실패(속성 없음 또는 할당 실패)";
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

// drmModeAtomicReq 의 RAII
struct AtomicReq {
    drmModeAtomicReqPtr p = drmModeAtomicAlloc();
    ~AtomicReq() {
        if (p) drmModeAtomicFree(p);
    }
    AtomicReq() = default;
    AtomicReq(const AtomicReq&) = delete;
    AtomicReq& operator=(const AtomicReq&) = delete;
};

// drmModeAtomicCommit 을 부르고 errno 를 **즉시** 돌려준다 (0 = 성공). 이후 호출(free 등)이 errno 를 덮기 전에 잡는다.
int atomic_commit(int fd, drmModeAtomicReqPtr req, std::uint32_t flags, void* user) {
    if (drmModeAtomicCommit(fd, req, flags, user) == 0) return 0;
    const int e = errno;
    return e != 0 ? e : EIO;
}

// drmHandleEvent 는 user_data 만 넘겨준다. user_data 에는 token 을 싣고, 인스턴스는 dispatch 동안 이 포인터로 찾는다.
thread_local KmsScanout* t_dispatching = nullptr;

bool mode_interlaced(const drmModeModeInfo& m) { return (m.flags & DRM_MODE_FLAG_INTERLACE) != 0; }

} // namespace

// ── 속성 표 ───────────────────────────────────────────────────────────────────────────────────
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
        throw std::runtime_error("kms: drmModeGetPlaneResources 실패: " + errno_text(e));
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
                    // SAND128 판정은 열 높이 파라미터를 뺀 modifier 로 한다(drm_state.cpp 와 같은 규칙)
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

// ── 생성 ──────────────────────────────────────────────────────────────────────────────────────
KmsScanout::KmsScanout(const KmsScanoutOptions& opt) : opt_(opt) {
    fd_ = ::open(opt_.card.c_str(), O_RDWR | O_CLOEXEC);
    if (fd_ < 0) {
        const int e = errno;
        throw std::runtime_error("kms: " + opt_.card + " 열기 실패: " + errno_text(e));
    }
    try {
        if (drmSetClientCap(fd_, DRM_CLIENT_CAP_UNIVERSAL_PLANES, 1) != 0) {
            const int e = errno;
            throw std::runtime_error("kms: DRM_CLIENT_CAP_UNIVERSAL_PLANES 실패: " + errno_text(e));
        }
        if (drmSetClientCap(fd_, DRM_CLIENT_CAP_ATOMIC, 1) != 0) {
            const int e = errno;
            throw std::runtime_error("kms: DRM_CLIENT_CAP_ATOMIC 실패(atomic 미지원): " + errno_text(e));
        }
        std::uint64_t cap = 0;
        ts_monotonic_ = drmGetCap(fd_, DRM_CAP_TIMESTAMP_MONOTONIC, &cap) == 0 && cap != 0;

        acquire_master_or_throw();
        select_output();
        load_planes();
        select_planes();

        crtc_props_ = load_props(crtc_id_, DRM_MODE_OBJECT_CRTC);
        conn_props_ = load_props(conn_id_, DRM_MODE_OBJECT_CONNECTOR);
        if (!conn_props_.id("CRTC_ID")) throw std::runtime_error("kms: 커넥터 속성 CRTC_ID 없음");
        for (const char* n : {"MODE_ID", "ACTIVE"})
            if (!crtc_props_.id(n)) throw std::runtime_error(std::string("kms: CRTC 속성 ") + n + " 없음");
        prepare_modeset_extras();
        // 모드 blob 은 master 없이 만들 수 있다(CREATEPROPBLOB 은 DRM_MASTER ioctl 이 아니다)
        if (drmModeCreatePropertyBlob(fd_, mode_raw_.data(), mode_raw_.size(), &mode_blob_) != 0) {
            const int e = errno;
            throw std::runtime_error("kms: drmModeCreatePropertyBlob(mode) 실패: " + errno_text(e));
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
        // command 는 20칸 오른쪽 정렬(%20s) — 공백이 들어간 comm 도 있으므로 폭으로 자른다
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
    return out.empty() ? "(없음)" : out;
}

void KmsScanout::acquire_master_or_throw() {
    // open 시점에 master 가 비어 있었다면 커널이 이미 이 파일을 master 로 만들었다.
    master_ = drmIsMaster(fd_) != 0;
    if (!opt_.acquire_master) {
        // 테스트 모드: 절대 master 를 쥐지 않는다. 자동으로 받았으면 즉시 놓는다(= 열기 전 상태 복원).
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
    if (holders.empty()) holders = "알 수 없음 (" + opt_.dri_debug_dir + "/clients 를 읽지 못했다 — root:sudo 0750)";
    throw KmsMasterError("kms: " + opt_.card + " 의 DRM master 를 얻지 못했다 (drmSetMaster: " + errno_text(e) +
                         "). 현재 master: " + holders +
                         ". 데스크톱 세션을 비활성화한 뒤 실행해야 한다: loginctl activate <비데스크톱 세션 id> "
                         "(sudo 불필요, 또는 sudo chvt N). lightdm 은 멈추지 않아도 된다. 실행 중에는 VT 를 전환하지 않는다");
}

void KmsScanout::select_output() {
    drmModeResPtr res = drmModeGetResources(fd_);
    if (!res) {
        const int e = errno;
        throw std::runtime_error("kms: drmModeGetResources 실패: " + errno_text(e));
    }
    // possible_crtcs 는 32비트 마스크다 — 인덱스 32 이상 CRTC 는 표현할 수 없으니 보지 않는다(B7).
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
            throw std::runtime_error("kms: 커넥터 " + nm + " 이 연결돼 있지 않다(모드 없음)");
        }
        // 모드: 인터레이스 제외(B3). 1920x1080@60 → preferred → 첫 프로그레시브 모드
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
            throw std::runtime_error("kms: 커넥터 " + nm + " 에 프로그레시브 모드가 없다(인터레이스만)");
        }
        conn_id_ = c->connector_id;
        conn_name_ = nm;
        mode_raw_.assign(reinterpret_cast<const std::uint8_t*>(pick),
                         reinterpret_cast<const std::uint8_t*>(pick) + sizeof(drmModeModeInfo));
        mode_w_ = pick->hdisplay;
        mode_h_ = pick->vdisplay;
        mode_hz_ = static_cast<int>(pick->vrefresh);
        mode_name_ = pick->name;
        // CRTC: 인코더가 지금 쓰는 CRTC → 사용 중이 아닌 possible CRTC → 첫 possible CRTC (B4)
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
                                 (opt_.connector.empty() ? std::string("연결된 HDMI-A 커넥터") : "커넥터 " + opt_.connector) +
                                 " 를 찾지 못했다 (있는 것: " + seen + ")");
    if (!crtc_id_ || crtc_idx_ < 0 || crtc_idx_ >= 32)
        throw std::runtime_error("kms: 커넥터 " + conn_name_ + " 에 쓸 CRTC 가 없다");
}

void KmsScanout::select_planes() {
    const std::uint32_t bit = 1u << crtc_idx_;
    // 다른 출력(CRTC)이 쓰는 평면은 뺏지 않는다: 지금 비어 있거나 우리 CRTC 에 붙은 평면 우선, 그다음 아무거나(B2).
    auto free_or_ours = [&](const Plane& p) { return p.cur_crtc == 0 || p.cur_crtc == crtc_id_; };
    // 영상: NV12 SAND128 인 Primary 우선, 없으면 Overlay
    for (int pass = 0; pass < 2 && !vplane_; ++pass)
        for (const char* want : {"Primary", "Overlay"}) {
            for (const Plane& p : planes_)
                if ((p.possible_crtcs & bit) && p.nv12_sand && p.type == want && (pass == 1 || free_or_ours(p))) {
                    vplane_ = &p;
                    break;
                }
            if (vplane_) break;
        }
    if (!vplane_) throw std::runtime_error("kms: CRTC " + std::to_string(crtc_id_) + " 에 NV12 SAND128 평면이 없다");
    for (int pass = 0; pass < 2 && !oplane_; ++pass)
        for (const Plane& p : planes_)
            if (&p != vplane_ && (p.possible_crtcs & bit) && p.ab24_lin && p.type == "Overlay" &&
                (pass == 1 || free_or_ours(p))) {
                oplane_ = &p;
                break;
            }
    if (!oplane_) throw std::runtime_error("kms: CRTC " + std::to_string(crtc_id_) + " 에 AB24 LINEAR Overlay 평면이 없다");
    for (const Plane* p : {vplane_, oplane_})
        for (const char* n : {"FB_ID", "CRTC_ID", "SRC_X", "SRC_Y", "SRC_W", "SRC_H", "CRTC_X", "CRTC_Y", "CRTC_W", "CRTC_H"})
            if (!p->props.id(n))
                throw std::runtime_error("kms: 평면 " + std::to_string(p->id) + " 에 속성 " + n + " 가 없다");

    // zpos: 박스 평면이 영상 위. Primary 의 zpos 는 보통 immutable 이다(현재값을 기준으로 삼는다).
    // mutable 이면 min 을 매 커밋 명시한다(B1 — 이전 master 가 남긴 값에 기대지 않는다).
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
                throw std::runtime_error("kms: 박스 평면 zpos(immutable " + std::to_string(overlay_zpos_) +
                                         ") 가 영상 평면 zpos " + std::to_string(vz) + " 보다 위가 아니다");
        } else {
            overlay_zpos_ = std::clamp<std::uint64_t>(vz + 1, z->min, z->max);
            overlay_zpos_settable_ = true;
            if (overlay_zpos_ <= vz)
                throw std::runtime_error("kms: 박스 평면 zpos 범위가 영상 평면 zpos 보다 위로 갈 수 없다");
        }
    }
}

// 모드셋 때 색 파이프라인을 알려진 기본값으로 둔다(B11). 이전 master 가 감마/CTM/Broadcast RGB 를 바꿔 뒀어도
// 우리 판정(BT.709 limited)이 그대로 화면에 나가게 한다. 속성이 있고 mutable 일 때만 넣는다. 복원은 capture_state 가 한다.
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
    for (const auto& [h, n] : handle_refs_) drmCloseBufferHandle(fd_, h);   // 남은 것이 있으면(버그) 누수 방지
    handle_refs_.clear();
    if (mode_blob_) drmModeDestroyPropertyBlob(fd_, mode_blob_);
    if (master_) drmDropMaster(fd_);
    ::close(fd_);
    fd_ = -1;
}

VkExtent2D KmsScanout::output_extent() const {
    return VkExtent2D{static_cast<std::uint32_t>(mode_w_), static_cast<std::uint32_t>(mode_h_)};
}

// ── GEM 핸들 ──────────────────────────────────────────────────────────────────────────────────
// drmPrimeFDToHandle 은 같은 dma-buf 에 대해 같은 핸들을 돌려주고 핸들은 참조 계수가 없다(close 한 번이면 사라진다).
// 그래서 우리가 센다 — Y/UV 가 한 객체이거나 두 FB 가 같은 dma-buf 를 가리켜도 너무 일찍 닫지 않는다.
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
    // 매 프레임 같은 오류가 나면 로그가 넘친다 — 처음 몇 번과 이후 100번마다만 찍는다.
    if (commit_fail_ <= 5 || commit_fail_ % 100 == 0) std::fprintf(stderr, "[kms] %s\n", msg.c_str());
}

void KmsScanout::on_master_lost(const std::string& where) {
    if (master_lost_) return;
    master_lost_ = true;
    master_ = false;
    last_error_ = "DRM master 를 잃었다 (" + where + ": EACCES)";
    std::fprintf(stderr,
                 "[kms] 경고: %s — VT 전환 등으로 다른 프로세스가 master 가 됐다. 이후 커밋은 모두 드롭한다"
                 " (현재 master: %s)\n",
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
        // fd 는 OverlayPlane 소유 그대로다. drmPrimeFDToHandle 은 fd 를 소비하지 않는다(GEM 이 dma-buf 참조를 따로 쥔다).
        if (!import_handle(b.fd, &o.handle, &e))
            throw std::runtime_error("kms: 박스 버퍼 #" + std::to_string(i) + " drmPrimeFDToHandle 실패: " + errno_text(e));
        std::uint32_t handles[4] = {o.handle, 0, 0, 0};
        std::uint32_t pitches[4] = {b.pitch, 0, 0, 0};
        std::uint32_t offsets[4] = {b.offset, 0, 0, 0};
        std::uint64_t mods[4] = {b.modifier, 0, 0, 0};
        if (drmModeAddFB2WithModifiers(fd_, b.width, b.height, b.fourcc, handles, pitches, offsets, mods, &o.fb,
                                       DRM_MODE_FB_MODIFIERS) != 0) {
            e = errno;
            release_handle(o.handle);
            throw std::runtime_error("kms: 박스 버퍼 #" + std::to_string(i) + " AddFB2(AB24 LINEAR " +
                                     std::to_string(b.width) + "x" + std::to_string(b.height) + " pitch " +
                                     std::to_string(b.pitch) + ") 실패: " + errno_text(e));
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
        last_error_ = "영상 프레임 fd 가 유효하지 않다 (fd=" + std::to_string(f.fd) + "/" + std::to_string(f.uv_fd) + ")";
        return 0;
    }
    const VideoKey key{static_cast<std::uint64_t>(sy.st_dev), static_cast<std::uint64_t>(sy.st_ino),
                       static_cast<std::uint64_t>(suv.st_dev), static_cast<std::uint64_t>(suv.st_ino)};
    auto it = video_.find(key);
    if (it != video_.end()) {
        VideoFb& v = it->second;
        if (v.w == f.width && v.h == f.height && v.modifier == f.modifier) return v.fb;
        // 같은 dma-buf 인데 기술자가 바뀌었다(해상도 변경). 화면·flip 대기 중인 FB 는 지우지 않는다(RmFB 가 평면을 끈다).
        if ((on_screen_.valid && on_screen_.video_fb == v.fb) || (in_flight_.valid && in_flight_.video_fb == v.fb)) {
            last_error_ = "기술자가 바뀐 영상 버퍼가 아직 스캔아웃 중이다";
            return 0;
        }
        destroy_video_fb(v);
        video_.erase(it);
    }
    if (f.drm_format != DRM_FORMAT_NV12) {
        char b[64];
        std::snprintf(b, sizeof(b), "지원하지 않는 영상 포맷 fourcc=0x%08x", f.drm_format);
        last_error_ = b;
        return 0;
    }
    VideoFb v;
    int e = 0;
    if (!import_handle(f.fd, &v.hy, &e)) {
        last_error_ = "drmPrimeFDToHandle(Y) 실패: " + errno_text(e);
        return 0;
    }
    if (!import_handle(f.uv_fd, &v.huv, &e)) {
        last_error_ = "drmPrimeFDToHandle(UV) 실패: " + errno_text(e);
        release_handle(v.hy);   // 프로브 결함 수정: Y 핸들을 흘리지 않는다
        return 0;
    }
    // SAND128: 각 plane 의 pitch 필드 = SAND 열 높이(Y 1088 / UV 544). 커널 vc4 HVS6 관례(frame_types.hpp DrmFrame).
    std::uint32_t handles[4] = {v.hy, v.huv, 0, 0};
    std::uint32_t pitches[4] = {static_cast<std::uint32_t>(f.y_pitch), static_cast<std::uint32_t>(f.uv_pitch), 0, 0};
    std::uint32_t offsets[4] = {static_cast<std::uint32_t>(f.y_offset), static_cast<std::uint32_t>(f.uv_offset), 0, 0};
    std::uint64_t mods[4] = {f.modifier, f.modifier, 0, 0};
    if (drmModeAddFB2WithModifiers(fd_, static_cast<std::uint32_t>(f.width), static_cast<std::uint32_t>(f.height),
                                   DRM_FORMAT_NV12, handles, pitches, offsets, mods, &v.fb, DRM_MODE_FB_MODIFIERS) != 0) {
        e = errno;   // snprintf 가 errno 를 덮기 전에 잡는다(B5)
        char b[160];
        std::snprintf(b, sizeof(b), "AddFB2(NV12 %dx%d mod=0x%016llx pitch(colh)=%zu/%zu) 실패: ", f.width, f.height,
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

// ── 상태 저장·복원 ────────────────────────────────────────────────────────────────────────────
void KmsScanout::capture_state() {
    saved_.clear();
    auto save = [&](const char* kind, std::uint32_t obj, const Props& p, const char* name) {
        const Prop* q = p.get(name);
        if (!q || (q->flags & DRM_MODE_PROP_IMMUTABLE)) return;
        saved_.push_back(SavedProp{kind, obj, name, q->id, q->value});
    };
    // 지금 값을 다시 읽는다(생성 시점 값은 낡았을 수 있다)
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
        // 우리 CRTC 에 붙은 평면(모드셋이 끈다)과 우리가 쓸 두 평면(모드셋이 덮어쓴다)만 저장한다
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
    if (!ok) return -1;   // 요청 구성 실패와 ioctl 실패를 구분한다(B6)
    return atomic_commit(fd_, req.p, DRM_MODE_ATOMIC_ALLOW_MODESET, nullptr);
}

// ── atomic 요청 ───────────────────────────────────────────────────────────────────────────────
bool KmsScanout::add(void* req, std::uint32_t obj, const Props& p, const char* name, std::uint64_t value, bool required) {
    const std::uint32_t id = p.id(name);
    if (id == 0) {
        if (required) last_error_ = std::string("속성 ") + name + " 없음 (obj " + std::to_string(obj) + ")";
        return !required;
    }
    if (drmModeAtomicAddProperty(req_of(req), obj, id, value) < 0) {
        last_error_ = std::string("drmModeAtomicAddProperty(") + name + ") 실패";
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
    // KMS 는 요청 값이 그대로 평면에 실린다(labwc 의 뒤집힘이 없다). auto = limited = 디코더 출력(HEVC TV range).
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
        // 지금 이 CRTC 에 붙어 있는 나머지 평면(labwc/fbcon 이 쓰던 primary·커서 등)을 끈다. 커밋 시점의 실제
        // 배치를 다시 읽는다 — 다른 CRTC(두 번째 HDMI)가 쓰는 평면은 possible_crtcs 가 겹쳐도 건드리지 않는다.
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
    // 영상 평면: 매 커밋 전부 넣는다(COLOR_* 포함 — 다른 master 가 중간에 바꿔도 되돌린다)
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
    // 박스 평면
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
        // 첫 커밋에 박스가 없으면 박스 평면을 꺼 둔다(이전 master 의 내용이 남지 않게)
        if (!add(req, oo, op, "FB_ID", 0, true) || !add(req, oo, op, "CRTC_ID", 0, true)) return false;
    }
    // overlay_idx < 0 이고 modeset 이 아니면 박스 평면 속성을 넣지 않는다 → 직전 박스 그대로
    return true;
}

// ── 커밋 ──────────────────────────────────────────────────────────────────────────────────────
bool KmsScanout::commit(const DrmFrame& video, std::uint64_t token, int overlay_idx) {
    if (shut_ || fd_ < 0 || !fatal_.empty()) return false;
    if (master_lost_) {   // 경고는 on_master_lost 에서 한 번만 냈다. 이후는 조용히 드롭(ioctl 없음)
        ++lost_drops_;
        return false;
    }
    if (!master_) {
        ++commit_fail_;
        fail("commit 거부: " + opt_.card + " 의 DRM master 가 없다 (현재 master: " +
             find_master_holders(opt_.dri_debug_dir) + ")");
        return false;
    }
    if (overlay_idx >= static_cast<int>(overlay_.size())) return false;
    check_flip_watchdog();
    if (in_flight_.valid) {
        // 이전 flip 이 아직 안 왔다. 비차단 커밋은 EBUSY 가 된다 — ioctl 없이 거부(PlaneDisplay 가 드롭으로 센다).
        ++busy_;
        return false;
    }
    const std::uint32_t vfb = video_fb_for(video);
    if (vfb == 0) {
        ++commit_fail_;
        fail("영상 FB 실패 (token " + std::to_string(token) + "): " + last_error_);
        return false;
    }
    const bool modeset = !modeset_done_;
    if (modeset && !state_captured_) {
        capture_state();   // 화면을 처음 바꾸기 직전의 상태 (shutdown 이 되돌린다)
        if (opt_.verbose) std::printf("[kms] 저장한 이전 KMS 상태 (%zu 속성):\n%s", saved_.size(), saved_state_text().c_str());
    }
    AtomicReq req;
    if (!req.p || !build_request(req.p, vfb, video.width, video.height, overlay_idx, modeset)) {
        ++commit_fail_;
        fail("atomic 요청 구성 실패: " + last_error_);
        return false;
    }
    if (modeset) {
        // 첫 커밋: TEST_ONLY 로 먼저 확인한다(실패해도 화면은 그대로다).
        const int e = atomic_commit(fd_, req.p, DRM_MODE_ATOMIC_TEST_ONLY | DRM_MODE_ATOMIC_ALLOW_MODESET, nullptr);
        if (e != 0) {
            ++commit_fail_;
            if (e == EACCES) {
                on_master_lost("첫 모드셋 TEST_ONLY");
                return false;
            }
            ++modeset_test_failures_;
            fail("첫 모드셋 TEST_ONLY 실패 " + std::to_string(modeset_test_failures_) + "/" +
                 std::to_string(kMaxModesetTestFailures) + ": " + errno_text(e) + " (" + conn_name_ + " " + mode_name_ +
                 " crtc " + std::to_string(crtc_id_) + " video plane " + std::to_string(vplane_->id) + " boxes plane " +
                 std::to_string(oplane_->id) + ")");
            if (modeset_test_failures_ >= kMaxModesetTestFailures) {
                fatal_ = "kms: 첫 모드셋이 TEST_ONLY 에서 " + std::to_string(modeset_test_failures_) +
                         "번 연속 거부됐다 (마지막: " + errno_text(e) + "). 이 커넥터·모드·평면 조합(" + conn_name_ + " " +
                         mode_name_ + ", crtc " + std::to_string(crtc_id_) + ", plane " + std::to_string(vplane_->id) +
                         "/" + std::to_string(oplane_->id) + ")을 드라이버가 받지 않는다 — 종료한다";
                std::fprintf(stderr, "[kms] %s\n", fatal_.c_str());
            }
            return false;
        }
        modeset_test_failures_ = 0;
    }
    // 블로킹 모드셋에도 PAGE_FLIP_EVENT 를 붙인다 → 이벤트가 fd 에 쌓이고 dispatch() 에서 같은 흐름으로 처리된다.
    // (B8/B10) 이 첫 커밋은 수십 ms 블로킹이고 그 flip 시각은 모드셋 완료 시각이다 — 측정에서 첫 N 프레임은 뺄 것.
    const std::uint32_t flags = DRM_MODE_PAGE_FLIP_EVENT |
                                (modeset ? DRM_MODE_ATOMIC_ALLOW_MODESET : DRM_MODE_ATOMIC_NONBLOCK);
    const int e = atomic_commit(fd_, req.p, flags, reinterpret_cast<void*>(static_cast<std::uintptr_t>(token)));
    if (e != 0) {
        ++commit_fail_;
        if (e == EACCES) {
            on_master_lost(modeset ? "첫 모드셋 커밋" : "비차단 커밋");
            return false;
        }
        if (e == EBUSY) ++busy_;
        fail(std::string(modeset ? "첫 모드셋" : "비차단") + " atomic commit 실패 (token " + std::to_string(token) +
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
        ev.version = 3;   // page_flip_handler2 (crtc_id 포함)
        ev.page_flip_handler2 = &KmsScanout::on_flip;
        KmsScanout* prev = t_dispatching;
        t_dispatching = this;
        ok = drmHandleEvent(fd_, &ev) == 0;
        t_dispatching = prev;
    }
    check_flip_watchdog();
    return ok;
}

void KmsScanout::on_flip(int /*fd*/, unsigned /*seq*/, unsigned sec, unsigned usec, unsigned /*crtc*/, void* data) {
    if (t_dispatching == nullptr) return;
    t_dispatching->handle_flip(static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(data)), true, sec, usec);
}

void KmsScanout::check_flip_watchdog() {
    if (!in_flight_.valid) return;
    if (SteadyClock::now() - in_flight_since_ < std::chrono::milliseconds(kFlipTimeoutMs)) return;
    ++flip_timeouts_;
    if (flip_timeouts_ == 1)
        std::fprintf(stderr,
                     "[kms] 경고: flip 이벤트가 %d ms 안에 오지 않았다 (token %llu) — 화면에 올라간 것으로 간주하고 "
                     "계속한다(on_presented 없음, flip_timeouts 에 센다; 이후 같은 경고 생략)\n",
                     kFlipTimeoutMs, static_cast<unsigned long long>(in_flight_.token));
    handle_flip(in_flight_.token, false, 0, 0);
}

void KmsScanout::handle_flip(std::uint64_t token, bool have_time, unsigned sec, unsigned usec) {
    // 우리가 기다리는 커밋의 이벤트만 받는다. 워치독이 이미 승격한 커밋의 늦은 이벤트, 종료 커밋 등은 무시한다.
    if (!in_flight_.valid || in_flight_.token != token) return;
    if (have_time) ++flips_;
    const Shown now = in_flight_;
    const Shown prev = on_screen_;
    const int prev_overlay = shown_overlay_;
    in_flight_ = Shown{};
    on_screen_ = now;
    if (now.overlay >= 0) shown_overlay_ = now.overlay;
    if (!listener_) return;
    // 순서 계약: frame_done → presented → 직전 영상 release → (박스가 바뀌었으면) 직전 박스 release
    listener_->on_frame_done();
    if (have_time) {
        const std::int64_t t_ns =
            static_cast<std::int64_t>(sec) * 1000000000LL + static_cast<std::int64_t>(usec) * 1000LL;
        listener_->on_presented(now.token, t_ns, ts_monotonic_, true);
    }
    if (prev.valid) listener_->on_video_released(prev.token);
    if (now.overlay >= 0 && prev_overlay >= 0 && prev_overlay != now.overlay) listener_->on_overlay_released(prev_overlay);
}

// ── 종료 ──────────────────────────────────────────────────────────────────────────────────────
void KmsScanout::shutdown(int timeout_ms) {
    if (shut_) return;
    shut_ = true;
    if (fd_ < 0) return;
    // 1. 날아가 있는 flip 을 유한 시간 기다린다(그 커밋이 끝나야 종료 커밋이 EBUSY 가 안 된다).
    const auto deadline = SteadyClock::now() + std::chrono::milliseconds(std::max(timeout_ms, 0));
    while (in_flight_.valid && SteadyClock::now() < deadline)
        if (!dispatch(10)) break;
    // 2. 화면 되돌리기: 저장한 이전 상태 복원 → 실패하면 평면 두 장 끄기 → 그래도 실패하면 CRTC 까지 끄기.
    if (!modeset_done_) {
        shutdown_result_ = "not-needed (모드셋 전)";
    } else if (!master_) {
        shutdown_result_ = master_lost_ ? "skipped (master 상실 — 화면은 새 master 몫)" : "skipped (master 없음)";
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
        std::printf("[kms] 종료: %s\n", shutdown_result_.c_str());
        std::fflush(stdout);
    }
    // 종료 커밋이 끝났으면(블로킹) HVS 는 더 이상 우리 버퍼를 읽지 않는다 → 남은 것을 전부 release 로 알린다.
    // 늦게 온 flip 이벤트가 fd 에 남아 있어도 in_flight_ 를 비웠으므로 무시된다.
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
    // 3. master 를 놓는다 → 다음 master(VT 복귀 시 labwc)가 화면을 다시 잡는다.
    if (master_) {
        drmDropMaster(fd_);
        master_ = false;
    }
}

void KmsScanout::clear_video_cache() {
    if (fd_ < 0) return;
    for (auto& [k, v] : video_) destroy_video_fb(v);   // 스캔아웃 중인 FB 를 RmFB 하면 커널이 그 평면을 끈다
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
    if (modeset_extras_.empty()) s += " (색 속성 없음)";
    for (const ModesetExtra& x : modeset_extras_) s += " " + x.text;
    return s;
}

} // namespace yvo::display
