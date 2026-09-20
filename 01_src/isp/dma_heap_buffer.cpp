#include "isp/dma_heap_buffer.hpp"

#include <cerrno>
#include <cstring>
#include <stdexcept>

#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <linux/dma-buf.h>
#include <linux/dma-heap.h>

namespace yvo::isp {

namespace {

// vidbuf_cached는 Pi 5에서 system 힙 심볼릭 링크다(= CPU 캐시 매핑). 없으면 system으로 내려간다.
const char* const kHeaps[] = {"/dev/dma_heap/vidbuf_cached", "/dev/dma_heap/system"};

void sync_ioctl(int fd, std::uint64_t flags, const char* what) {
    dma_buf_sync s{};
    s.flags = flags;
    if (::ioctl(fd, DMA_BUF_IOCTL_SYNC, &s) != 0)
        throw std::runtime_error(std::string("DmaHeapBuffers: DMA_BUF_IOCTL_SYNC ") + what + " failed: " +
                                 std::strerror(errno));
}

} // namespace

DmaHeapBuffers::DmaHeapBuffers(const std::string& name, std::size_t size, int count) : size_(size) {
    if (size == 0 || count <= 0) throw std::invalid_argument("DmaHeapBuffers: size/count must be > 0");
    int heap_fd = -1;
    for (const char* h : kHeaps) {
        heap_fd = ::open(h, O_RDWR | O_CLOEXEC);
        if (heap_fd >= 0) {
            heap_ = h;
            break;
        }
    }
    if (heap_fd < 0)
        throw std::runtime_error(std::string("DmaHeapBuffers: cannot open any dma-heap: ") + std::strerror(errno));

    try {
        for (int i = 0; i < count; ++i) {
            dma_heap_allocation_data alloc{};
            alloc.len = size;
            alloc.fd_flags = O_CLOEXEC | O_RDWR;
            if (::ioctl(heap_fd, DMA_HEAP_IOCTL_ALLOC, &alloc) != 0)
                throw std::runtime_error(std::string("DmaHeapBuffers: DMA_HEAP_IOCTL_ALLOC failed: ") +
                                         std::strerror(errno));
            Buf b;
            b.fd = static_cast<int>(alloc.fd);
            ::ioctl(b.fd, DMA_BUF_SET_NAME, name.c_str()); // 실패해도 치명적이지 않다(디버깅용 이름)
            void* p = ::mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, b.fd, 0);
            if (p == MAP_FAILED) {
                ::close(b.fd);
                throw std::runtime_error(std::string("DmaHeapBuffers: mmap failed: ") + std::strerror(errno));
            }
            b.ptr = static_cast<std::uint8_t*>(p);
            bufs_.push_back(b);
        }
    } catch (...) {
        ::close(heap_fd);
        for (Buf& b : bufs_) {
            if (b.ptr) ::munmap(b.ptr, size_);
            if (b.fd >= 0) ::close(b.fd);
        }
        bufs_.clear();
        throw;
    }
    ::close(heap_fd);
}

DmaHeapBuffers::~DmaHeapBuffers() {
    for (Buf& b : bufs_) {
        if (b.ptr) ::munmap(b.ptr, size_);
        if (b.fd >= 0) ::close(b.fd);
    }
}

int DmaHeapBuffers::fd(int i) const {
    if (i < 0 || i >= count()) throw std::out_of_range("DmaHeapBuffers::fd: index out of range");
    return bufs_[static_cast<std::size_t>(i)].fd;
}

std::uint8_t* DmaHeapBuffers::ptr(int i) const {
    if (i < 0 || i >= count()) throw std::out_of_range("DmaHeapBuffers::ptr: index out of range");
    return bufs_[static_cast<std::size_t>(i)].ptr;
}

void DmaHeapBuffers::begin_cpu_write(int i) {
    sync_ioctl(fd(i), DMA_BUF_SYNC_START | DMA_BUF_SYNC_WRITE, "START|WRITE");
}

void DmaHeapBuffers::end_cpu_write(int i) {
    sync_ioctl(fd(i), DMA_BUF_SYNC_END | DMA_BUF_SYNC_WRITE, "END|WRITE");
}

} // namespace yvo::isp
