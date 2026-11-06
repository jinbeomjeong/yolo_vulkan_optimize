// DecodeCopy: yvo_pipeline --decode-copy. A comparison path that puts one CPU copy between the HW HEVC decoder and the GPU, to
// measure what the zero-copy hand-off (V3D importing the decoder's SAND128 dma-buf as a VkBuffer) saves. Not an operating path.
//
//   decoder frame (V4L2 DRM_PRIME dma-buf) → CPU reads it through an mmap (READ sync) → one memcpy per plane →
//   SandFramePool buffer (WRITE sync) → the decoder frame is released → GPU imports the pool buffer (rest of the pipeline unchanged)
//
// Source: the decoder's own dma-bufs, mapped read-only with whatever CPU mapping attributes the exporter gives. Reading them is
//   the cost being measured, so nothing here avoids it. Mappings are cached per dma-buf object (fstat st_dev/st_ino + size,
//   the key DmaBufImportCache uses): the decoder pool reuses a fixed set of buffers, and a real copy stage would not mmap per
//   frame. The first touch of a buffer (fstat + mmap) counts in total, not in memcpy or sync.
// Copy: each plane is copied whole — every SAND128 column including the rows that pad the column height — with one memcpy, so
//   the destination plane is byte-identical to the source plane and keeps its layout fields (column heights, pitch fields,
//   modifier). Only the fds, object sizes and offsets (0) change.
// Destination: SandFramePool (decode/sw_sand_frames.hpp, the pool of --decoder sw: system heap, cached CPU mapping). V3D does not
//   see CPU caches, so the copy runs inside a WRITE sync section of both planes (same reasoning as there).
// Lifetime: runs inside FfmpegDrmDecoder::next() (its frame filter). The decoder frame (AVFrame keepalive) is dropped as soon as the
//   copy is done, so the decoder pool only holds frames being decoded; downstream holds pool buffers instead. Pool size =
//   Options::extra_hw_frames + 4, the frames the HW pool lets the user hold (the same count --decoder sw uses). Pool frames hold the
//   pool (shared_from_this), so they may outlive this object. Exhaustion fails next() with an error instead of stalling.
// Thread: the thread that calls next() — the main thread, or the decode thread with --input-live. Either way the copy is in the
//   process CPU time. Not thread-safe: the stats are read after that thread is done (main.cpp joins the decode thread first).
#pragma once

#include <sys/mman.h>
#include <sys/stat.h>

#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "core/dma_heap_alloc.hpp"
#include "core/frame_types.hpp"
#include "decode/sw_sand_frames.hpp"

namespace yvo::pipeline {

class DecodeCopy {
public:
    // Throws std::invalid_argument for a bad size or count and std::runtime_error when the dma-heap fails.
    DecodeCopy(int width, int height, int buffers, std::vector<std::string> heaps = {})
        : width_(width), height_(height), pool_(std::make_shared<SandFramePool>(SandFramePool::Config{width, height, buffers, std::move(heaps)})) {}
    ~DecodeCopy() {
        for (const Mapping& m : maps_) ::munmap(const_cast<std::uint8_t*>(m.ptr), m.size);
    }
    DecodeCopy(const DecodeCopy&) = delete;
    DecodeCopy& operator=(const DecodeCopy&) = delete;

    // For FfmpegDrmDecoder::set_frame_filter. This object must outlive the decoder's use of it (until close()).
    std::function<bool(DrmFrame&, std::string&)> filter() {
        return [this](DrmFrame& f, std::string& why) { return copy(f, why); };
    }

    // Replaces f (a decoder frame) with a copy in a pool buffer; the decoder frame is released before returning.
    // false with the reason in why for a frame that does not fit the pool, an exhausted pool or a failed mmap / sync ioctl.
    bool copy(DrmFrame& f, std::string& why) {
        const auto t0 = Clock::now();
        if (f.width != width_ || f.height != height_ || f.sand_col_height != sand_y_col_height(height_) ||
            f.uv_sand_col_height != sand_uv_col_height(height_)) {
            why = "decode copy: frame " + std::to_string(f.width) + "x" + std::to_string(f.height) + " (column heights " +
                  std::to_string(f.sand_col_height) + "/" + std::to_string(f.uv_sand_col_height) + ") does not fit the pool " +
                  std::to_string(width_) + "x" + std::to_string(height_);
            return false;
        }
        const std::size_t y_bytes = sand_plane_bytes(width_, f.sand_col_height);
        const std::size_t uv_bytes = sand_plane_bytes(width_, f.uv_sand_col_height);
        if (f.y_offset > f.size || y_bytes > f.size - f.y_offset || f.uv_offset > f.uv_size || uv_bytes > f.uv_size - f.uv_offset) {
            why = "decode copy: a plane runs past the end of its dma-buf object";
            return false;
        }
        while (maps_.size() + 2 > kMaxMappings) {   // room for both planes, so neither lookup unmaps the other
            ::munmap(const_cast<std::uint8_t*>(maps_.front().ptr), maps_.front().size);
            maps_.erase(maps_.begin());
        }
        const std::uint8_t* src_y = map(f.fd, f.size, why);
        if (!src_y) return false;
        const std::uint8_t* src_uv = f.uv_fd == f.fd ? src_y : map(f.uv_fd, f.uv_size, why);
        if (!src_uv) return false;
        auto lease = pool_->acquire();
        if (!lease) {
            why = "decode copy: frame pool exhausted, all " + std::to_string(pool_->count()) +
                  " buffers are held downstream (raise Options::extra_hw_frames)";
            return false;
        }
        if (y_bytes > lease->y_size || uv_bytes > lease->uv_size) {
            why = "decode copy: pool buffer smaller than the decoder plane";
            return false;
        }
        Clock::time_point t1, t2, t3;
        const auto t_ready = Clock::now();
        try {
            DmaCpuAccess ry(f.fd, DmaSyncDir::Read);
            std::optional<DmaCpuAccess> ruv;
            if (f.uv_fd != f.fd) ruv.emplace(f.uv_fd, DmaSyncDir::Read);
            DmaCpuAccess wy(lease->y_fd, DmaSyncDir::Write), wuv(lease->uv_fd, DmaSyncDir::Write);
            t1 = Clock::now();
            std::memcpy(lease->y, src_y + f.y_offset, y_bytes);
            std::memcpy(lease->uv, src_uv + f.uv_offset, uv_bytes);
            t2 = Clock::now();
            wuv.finish();
            wy.finish();
            if (ruv) ruv->finish();
            ry.finish();
            t3 = Clock::now();
        } catch (const std::exception& e) {
            why = std::string("decode copy: ") + e.what();
            return false;
        }
        f.fd = lease->y_fd;
        f.size = lease->y_size;
        f.y_offset = 0;
        f.uv_fd = lease->uv_fd;
        f.uv_size = lease->uv_size;
        f.uv_offset = 0;
        f.keepalive = std::move(lease->keepalive);   // releases the decoder frame (AVFrame) to the decoder pool
        const auto t4 = Clock::now();
        memcpy_ms_.push_back(ms_of(t2 - t1));
        sync_ms_.push_back(ms_of((t1 - t_ready) + (t3 - t2)));
        total_ms_.push_back(ms_of(t4 - t0));
        return true;
    }

    // Per-frame samples in ms. total = memcpy + sync (START + END of the 3 or 4 sections) + mapping lookup, pool acquire and
    // the release of the decoder frame.
    const std::vector<double>& total_ms() const { return total_ms_; }
    const std::vector<double>& memcpy_ms() const { return memcpy_ms_; }
    const std::vector<double>& sync_ms() const { return sync_ms_; }
    int buffers() const { return pool_->count(); }
    int peak_in_use() const { return pool_->peak_in_use(); }
    std::size_t mappings() const { return maps_.size(); }   // decoder dma-buf objects mapped so far
    const std::string& heap() const { return pool_->heap(); }

private:
    using Clock = std::chrono::steady_clock;
    struct Mapping {
        dev_t dev;
        ino_t ino;
        std::size_t size;
        const std::uint8_t* ptr;
    };
    // A bound on the cache: the decoder pool is far smaller. Past it the oldest mapping goes (a new pool would be a new set of buffers).
    static constexpr std::size_t kMaxMappings = 64;

    static double ms_of(Clock::duration d) { return std::chrono::duration<double, std::milli>(d).count(); }

    const std::uint8_t* map(int fd, std::size_t size, std::string& why) {
        struct stat st{};
        if (fd < 0 || ::fstat(fd, &st) != 0) {
            why = "decode copy: fstat of decoder dma-buf fd " + std::to_string(fd) + " failed: " + std::strerror(errno);
            return nullptr;
        }
        for (const Mapping& m : maps_)
            if (m.dev == st.st_dev && m.ino == st.st_ino && m.size == size) return m.ptr;
        void* p = ::mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0);
        if (p == MAP_FAILED) {
            why = "decode copy: mmap of decoder dma-buf (" + std::to_string(size) + " B) failed: " + std::strerror(errno);
            return nullptr;
        }
        maps_.push_back({st.st_dev, st.st_ino, size, static_cast<const std::uint8_t*>(p)});
        return maps_.back().ptr;
    }

    const int width_, height_;
    std::shared_ptr<SandFramePool> pool_;
    std::vector<Mapping> maps_;
    std::vector<double> total_ms_, memcpy_ms_, sync_ms_;
};

} // namespace yvo::pipeline
