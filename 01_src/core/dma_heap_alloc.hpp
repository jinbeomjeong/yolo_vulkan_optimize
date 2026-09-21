// dma-heap 버퍼 할당·매핑·CPU 캐시 동기화 (헤더 전용, 순수 POSIX)
//
// 계획서 부록 B.11의 일원화 대상이다. 같은 일을 세 번 구현했던 것을 여기 하나로 모은다:
//   · core/dmabuf_map.hpp  DmaBufReadMap   — 남의 fd를 읽기 전용 매핑 (할당 안 함). **그대로 남는다.**
//                                              내부 ioctl만 이 헤더의 dma_sync_* 로 위임한다.
//   · tools/hailo/p2_out_buffers.hpp HeapOutPool — DmaHeapPool 위의 얇은 어댑터로 축소됐다.
//   · (2026-09-21 흡수 완료) isp/dma_heap_buffer.hpp DmaHeapBuffers 는 제거하고 이 풀로 옮겼다.
//
// **Vulkan 심볼을 쓰지 않는다.** yvo_core는 INTERFACE 라이브러리라 이 헤더를 더해도 링크 그래프가
// 한 변도 늘지 않고, yvo_hailo가 yvo_vulkan에 의존하지 않는다는 링크 격리(계획서 §5.2)가 그대로 지켜진다.
// Vulkan import 글루(헤드룸 질의)는 01_src/vulkan/vk_storage_headroom.hpp 에 따로 있다.
//
// ── 세 구현이 각각 빠뜨렸던 것을 전부 채운다 ─────────────────────────────────────────────────
//   ① READ/WRITE/RW 양방향 sync  (DmaBufReadMap·HeapOutPool은 READ만, DmaHeapBuffers는 WRITE만)
//   ② ioctl 반환값 검사 (기본 strict. 두 구현이 반환값을 버렸다 — 부록 B.10 R5)
//   ③ 크기 정렬 + **시작 주소 정렬**. mmap은 페이지 정렬만 보장한다. HeapOutPool은 안 맞으면 throw했는데
//      그건 정상 동작에서 프로세스를 죽인다 → 주소 공간을 예약한 뒤 정렬 지점에 MAP_FIXED로 재매핑한다.
//      **dma-buf 안의 오프셋은 언제나 0이다** — HailoRT read_async에 오프셋 인자가 없다(계획서 §4.1).
//   ④ memset(zero_init)을 WRITE sync로 감싼다 — 부록 B.6(c)가 지적한 현행 결함이다.
//      감싸지 않으면 더러운 캐시 라인이 나중에 evict되어 **장치가 쓴 데이터를 덮어쓴다**.
//   ⑤ DMA_BUF_SET_NAME (디버깅용 이름)
//   ⑥ device_headroom (부록 B.12) — Vulkan import 대상 버퍼가 드라이버 여유분(V3DV: +256 B)만큼
//      잘리는 것을 막는다. 상수를 박지 않는다. 질의는 vk_storage_headroom.hpp가 한다.
//
// ── 3자 일관성 규약 (부록 B.10) ───────────────────────────────────────────────────────────────
//   R1 장치↔장치(Hailo DMA 쓰기 → V3D 읽기): sync ioctl 없음. Vulkan 배리어만.
//   R2 CPU 읽기: 완료 신호 **뒤에** START|READ → 읽기 → END|READ.
//   R3 CPU 쓰기: START|WRITE … END|WRITE. **초기화 memset도 예외가 아니다.**
//   R4 읽고 쓰면 READ|WRITE. 방향 비트를 아끼지 않는다.
//   R5 strict 모드에서 ioctl 실패는 예외다.
//   R6 비용: 1.2 MB당 START+END 합 약 0.057 ms. 고정비가 크므로 작은 버퍼에는 손해다.
// DMA_BUF_IOCTL_SYNC는 **CPU 캐시만** 다루고 순서를 만들지 않는다. Vulkan 배리어는 **V3D 캐시만**
// 다루고 CPU 캐시를 건드리지 않는다. 둘은 대체재가 아니다.
#pragma once

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <fcntl.h>
#include <linux/dma-buf.h>
#include <linux/dma-heap.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include "core/hailo_align.hpp"

namespace yvo {

// ── 동기화 방향 ───────────────────────────────────────────────────────────────────────────────
enum class DmaSyncDir { Read, Write, ReadWrite };

inline std::uint64_t dma_sync_dir_bits(DmaSyncDir d) {
    switch (d) {
        case DmaSyncDir::Read: return DMA_BUF_SYNC_READ;
        case DmaSyncDir::Write: return DMA_BUF_SYNC_WRITE;
        case DmaSyncDir::ReadWrite: return DMA_BUF_SYNC_RW;
    }
    return DMA_BUF_SYNC_RW;
}

inline const char* dma_sync_dir_name(DmaSyncDir d) {
    switch (d) {
        case DmaSyncDir::Read: return "READ";
        case DmaSyncDir::Write: return "WRITE";
        case DmaSyncDir::ReadWrite: return "READ|WRITE";
    }
    return "?";
}

namespace detail {

// DMA_BUF_IOCTL_SYNC 한 번. 성공이면 0, 실패면 errno를 돌려준다(반환값을 버리지 않는다 — R5).
inline int dma_sync_raw(int fd, std::uint64_t flags) {
    if (fd < 0) return EBADF;
    dma_buf_sync s{};
    s.flags = flags;
    return ::ioctl(fd, DMA_BUF_IOCTL_SYNC, &s) != 0 ? (errno != 0 ? errno : EIO) : 0;
}

inline void dma_sync_checked(int fd, std::uint64_t flags, DmaSyncDir dir, const char* phase, bool strict) {
    const int e = dma_sync_raw(fd, flags);
    if (e != 0 && strict)
        throw std::runtime_error(std::string("DMA_BUF_IOCTL_SYNC ") + phase + "|" + dma_sync_dir_name(dir) +
                                 " 실패 (fd " + std::to_string(fd) + "): " + std::strerror(e));
}

inline std::size_t align_up(std::size_t v, std::size_t a) { return a <= 1 ? v : (v + a - 1) / a * a; }

} // namespace detail

// CPU 접근 구간을 연다/닫는다. strict면 실패가 예외다(R5).
inline void dma_sync_begin(int fd, DmaSyncDir dir, bool strict = true) {
    detail::dma_sync_checked(fd, DMA_BUF_SYNC_START | dma_sync_dir_bits(dir), dir, "START", strict);
}
inline void dma_sync_end(int fd, DmaSyncDir dir, bool strict = true) {
    detail::dma_sync_checked(fd, DMA_BUF_SYNC_END | dma_sync_dir_bits(dir), dir, "END", strict);
}

// ── RAII CPU 접근 구간 ────────────────────────────────────────────────────────────────────────
// 생성 시 START, 소멸 시 END. **sync 구간을 장치 접근에 걸쳐 열어 두지 않는다**(R2).
// 소멸자는 예외를 던지지 않는다 — END를 직접 검사하려면 finish()를 명시 호출할 것.
class DmaCpuAccess {
public:
    DmaCpuAccess(int fd, DmaSyncDir dir, bool strict = true) : fd_(fd), dir_(dir), strict_(strict) {
        dma_sync_begin(fd_, dir_, strict_);
        open_ = true;
    }
    ~DmaCpuAccess() {
        try {
            finish();
        } catch (...) {
            // 소멸자에서 던지면 std::terminate다. 검사가 필요하면 finish()를 먼저 부른다.
        }
    }
    DmaCpuAccess(const DmaCpuAccess&) = delete;
    DmaCpuAccess& operator=(const DmaCpuAccess&) = delete;

    // END를 지금 보낸다. 두 번 불러도 안전하다. strict면 실패가 예외다.
    void finish() {
        if (!open_) return;
        open_ = false;
        dma_sync_end(fd_, dir_, strict_);
    }

    int fd() const { return fd_; }
    DmaSyncDir dir() const { return dir_; }

private:
    int fd_;
    DmaSyncDir dir_;
    bool strict_;
    bool open_ = false;
};

// ── 설정 ──────────────────────────────────────────────────────────────────────────────────────
struct DmaHeapConfig {
    // 앞에서부터 열어 보고 첫 성공을 쓴다. vidbuf_cached는 Pi 5에서 system으로의 **심볼릭 링크**다(실측)
    // — 둘은 같은 힙이다. 이름을 남겨 둔 것은 다른 기기에서 갈릴 수 있어서다.
    std::vector<std::string> heaps{"/dev/dma_heap/vidbuf_cached", "/dev/dma_heap/system"};
    // 할당 크기·시작 주소 정렬. 16384 = HailoRT read_async 요구(부록 A.4, vulkan/dmabuf_export.hpp kHailoAsyncAlign)
    std::size_t align = kHailoAsyncAlign;
    // 드라이버가 요청 크기 위에 더 요구하는 바이트 수(부록 B.12). Vulkan import 대상이면 > 0이어야 한다.
    // vk::storage_dmabuf_headroom()이 질의로 구한 값을 넣는다. **상수를 박지 않는다.**
    std::size_t device_headroom = 0;
    bool map = true;          // false면 ptr == nullptr (fd만 필요할 때)
    bool strict_sync = true;  // false면 ioctl 실패를 무시한다 (DmaBufReadMap 기존 동작 호환)
    bool zero_init = true;    // 0으로 채운다. **WRITE sync로 감싼다**(R3, B.6(c))
    std::string name;         // DMA_BUF_SET_NAME. 비면 설정하지 않는다
};

// ── 버퍼 하나 ─────────────────────────────────────────────────────────────────────────────────
// 크기가 둘인 이유(부록 A.13 크기 계약): HailoRT dma_map_dmabuf에는 **dmabuf_size**를,
// read_async에는 **requested_size**를 넘긴다. Vulkan import에도 dmabuf_size를 넘긴다.
struct DmaHeapBuf {
    int fd = -1;                    // dma-buf fd. 소유권은 DmaHeapPool에 있다(넘길 때는 dup_fd()).
    std::uint8_t* ptr = nullptr;    // CPU 매핑 시작 (align 정렬 보장). map=false면 nullptr
    std::size_t requested_size = 0; // 호출자가 allocate()에 넘긴 바이트 수 (텐서 크기)
    std::size_t dmabuf_size = 0;    // lseek(fd, 0, SEEK_END)로 실측한 dma-buf 객체 크기
};

// ── 풀 ────────────────────────────────────────────────────────────────────────────────────────
class DmaHeapPool {
public:
    explicit DmaHeapPool(const DmaHeapConfig& cfg = {}) : cfg_(cfg) {
        if (cfg_.align == 0) cfg_.align = 1;
        std::string errs;
        for (const std::string& h : cfg_.heaps) {
            heap_fd_ = ::open(h.c_str(), O_RDWR | O_CLOEXEC);
            if (heap_fd_ >= 0) {
                heap_ = h;
                break;
            }
            errs += h + "(" + std::strerror(errno) + ") ";
        }
        if (heap_fd_ < 0) throw std::runtime_error("DmaHeapPool: dma-heap을 열지 못했다: " + errs);
    }
    ~DmaHeapPool() {
        for (Entry& e : bufs_) {
            if (e.buf.ptr) ::munmap(e.buf.ptr, e.map_len);
            if (e.buf.fd >= 0) ::close(e.buf.fd);
        }
        if (heap_fd_ >= 0) ::close(heap_fd_);
    }
    DmaHeapPool(const DmaHeapPool&) = delete;
    DmaHeapPool& operator=(const DmaHeapPool&) = delete;

    // requested 바이트를 담을 버퍼를 하나 더 잡는다. 실제 할당은
    //   align_up(requested + device_headroom, align)
    // 이다(부록 B.12). 인덱스를 돌려준다. 실패하면 예외.
    int allocate(std::size_t requested) {
        if (requested == 0) throw std::invalid_argument("DmaHeapPool::allocate: requested == 0");
        const std::size_t want = detail::align_up(requested + cfg_.device_headroom, cfg_.align);

        dma_heap_allocation_data d{};
        d.len = want;
        d.fd_flags = O_RDWR | O_CLOEXEC;
        if (::ioctl(heap_fd_, DMA_HEAP_IOCTL_ALLOC, &d) != 0)
            throw std::runtime_error("DmaHeapPool: DMA_HEAP_IOCTL_ALLOC(" + std::to_string(want) +
                                     ") 실패: " + std::strerror(errno));

        Entry e;
        e.buf.fd = static_cast<int>(d.fd);
        e.buf.requested_size = requested;
        const off_t sz = ::lseek(e.buf.fd, 0, SEEK_END);
        e.buf.dmabuf_size = sz > 0 ? static_cast<std::size_t>(sz) : want;
        e.map_len = e.buf.dmabuf_size;

        try {
            if (!cfg_.name.empty()) {
                // 실패해도 치명적이지 않다(디버깅용 이름). 커널이 O_CLOEXEC dup을 막는 경우가 있다.
                ::ioctl(e.buf.fd, DMA_BUF_SET_NAME, cfg_.name.c_str());
            }
            if (cfg_.map) e.buf.ptr = map_aligned(e.buf.fd, e.map_len, cfg_.align);
            if (cfg_.map && cfg_.zero_init) {
                // R3 / B.6(c): 캐시 매핑을 더럽히기 전에 WRITE 구간을 연다.
                DmaCpuAccess acc(e.buf.fd, DmaSyncDir::Write, cfg_.strict_sync);
                std::memset(e.buf.ptr, 0, e.map_len);
                acc.finish();   // 실패를 소멸자에서 삼키지 않는다
            }
        } catch (...) {
            if (e.buf.ptr) ::munmap(e.buf.ptr, e.map_len);
            ::close(e.buf.fd);
            throw;
        }
        bufs_.push_back(e);
        return static_cast<int>(bufs_.size()) - 1;
    }

    std::size_t count() const { return bufs_.size(); }
    const std::string& heap() const { return heap_; }
    const DmaHeapConfig& config() const { return cfg_; }

    const DmaHeapBuf& at(int i) const { return bufs_.at(check(i)).buf; }

    // 호출자에게 넘길 새 fd (호출자가 close 한다). 원본 fd는 이 풀이 계속 들고 있다.
    int dup_fd(int i) const {
        const int nfd = ::dup(at(i).fd);
        if (nfd < 0) throw std::runtime_error(std::string("DmaHeapPool::dup_fd 실패: ") + std::strerror(errno));
        return nfd;
    }

    // 편의 래퍼 — 규약대로 감싼 CPU 접근 구간
    DmaCpuAccess cpu_access(int i, DmaSyncDir dir) const {
        return DmaCpuAccess(at(i).fd, dir, cfg_.strict_sync);
    }

private:
    struct Entry {
        DmaHeapBuf buf;
        std::size_t map_len = 0;   // munmap에 넘길 길이 (= dmabuf_size)
    };

    std::size_t check(int i) const {
        if (i < 0 || static_cast<std::size_t>(i) >= bufs_.size())
            throw std::out_of_range("DmaHeapPool: index " + std::to_string(i) + " 범위 밖");
        return static_cast<std::size_t>(i);
    }

    // align 정렬된 주소에 dma-buf를 매핑한다.
    //   1) PROT_NONE 무명 매핑으로 len+align 만큼 **주소 공간만 예약**
    //   2) 예약 안의 align 정렬 지점에 MAP_FIXED|MAP_SHARED로 dma-buf를 덮어쓴다
    //   3) 앞뒤 잉여를 munmap
    // dma-buf 안의 오프셋은 **언제나 0**이다 — 정렬은 가상 주소만 움직인다.
    // (이 기기는 페이지 크기 16384라 leading이 항상 0이지만, 4 KB 페이지 커널에서도 맞게 동작한다.)
    static std::uint8_t* map_aligned(int fd, std::size_t len, std::size_t align) {
        void* base = ::mmap(nullptr, len + align, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
        if (base == MAP_FAILED)
            throw std::runtime_error(std::string("DmaHeapPool: 주소 공간 예약(mmap PROT_NONE) 실패: ") +
                                     std::strerror(errno));
        const auto b = reinterpret_cast<std::uintptr_t>(base);
        const std::uintptr_t a = detail::align_up(b, align);
        const std::size_t lead = static_cast<std::size_t>(a - b);
        const std::size_t tail = align - lead;   // (len+align) - lead - len

        void* p = ::mmap(reinterpret_cast<void*>(a), len, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, fd, 0);
        if (p == MAP_FAILED) {
            const int err = errno;
            ::munmap(base, len + align);
            throw std::runtime_error(std::string("DmaHeapPool: mmap(MAP_FIXED) 실패: ") + std::strerror(err));
        }
        if (lead > 0) ::munmap(base, lead);
        if (tail > 0) ::munmap(reinterpret_cast<void*>(a + len), tail);
        return static_cast<std::uint8_t*>(p);
    }

    DmaHeapConfig cfg_;
    int heap_fd_ = -1;
    std::string heap_;
    std::vector<Entry> bufs_;
};

} // namespace yvo
