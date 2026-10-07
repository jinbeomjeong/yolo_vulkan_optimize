#pragma once
// NmsHarness: feeds a synthetic candidate DetBuf directly to `EfficientNms` (02_shaders/efficient_nms.comp) —
// **the single implementation for tests and probes**.
//
// Used by: 05_tests/test_postproc.cpp (EfficientNms.* synthetic gates),
//          01_src/tools/vulkan/postproc_probe.cpp (--probe nms, worst-case GPU cost with up to kMaxSelected candidates).
//
// Moved out of test_postproc.cpp on 2026-10-08 so the probe can time the same command buffer the tests verify,
// instead of a second copy. Behaviour for the tests is unchanged; the only addition is a timestamp pair around
// the NMS dispatch (`last_ms()`), written at the same stage as YoloPostproc's ts3/ts4 (`last_nms_ms()`).
//
// Command buffer (one per run(), re-recorded every time):
//   [reset 2 queries] → barrier HOST_WRITE → SHADER_STORAGE_READ (input) → ts0 (COMPUTE_SHADER)
//   → EfficientNms::record → ts1 (COMPUTE_SHADER) → barrier SHADER_STORAGE_WRITE → HOST_READ (output)
// The input buffer is a single HOST_VISIBLE|COHERENT one, so no flush/invalidate is needed.
//
// Header-only. Depends only on vulkan/ (STRUCTURE.md §2.2: testkit → core, ref, vulkan, display).
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include <vulkan/vulkan.h>

#include "ref/cpu_yolov8_decode.hpp"
#include "vulkan/efficient_nms.hpp"
#include "vulkan/vk_check.hpp"
#include "vulkan/vulkan_context.hpp"
#include "vulkan/yolo_postproc.hpp"

namespace yvo::fixture {

class NmsHarness {
public:
    static constexpr std::uint32_t kInputMax = 2048;

    explicit NmsHarness(yvo::vk::VulkanContext& ctx, const yvo::vk::EfficientNmsConfig& cfg = {}) : ctx_(ctx) {
        VkDevice dev = ctx_.device();
        size_ = sizeof(yvo::vk::YoloDetBufHeader) + kInputMax * sizeof(yvo::vk::YoloDetGpu);
        VkBufferCreateInfo bci{};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size = size_;
        bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        YVO_VK_CHECK(vkCreateBuffer(dev, &bci, nullptr, &buf_));
        VkMemoryRequirements req{};
        vkGetBufferMemoryRequirements(dev, buf_, &req);
        VkMemoryAllocateInfo mai{};
        mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mai.allocationSize = req.size;
        mai.memoryTypeIndex = ctx_.find_memory_type(
            req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        YVO_VK_CHECK(vkAllocateMemory(dev, &mai, nullptr, &mem_));
        YVO_VK_CHECK(vkBindBufferMemory(dev, buf_, mem_, 0));
        void* p = nullptr;
        YVO_VK_CHECK(vkMapMemory(dev, mem_, 0, VK_WHOLE_SIZE, 0, &p));
        host_ = static_cast<std::uint8_t*>(p);
        nms_ = std::make_unique<yvo::vk::EfficientNms>(ctx_, cfg, std::vector<VkBuffer>{buf_}, size_, kInputMax);

        VkCommandPoolCreateInfo cp{};
        cp.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        cp.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        cp.queueFamilyIndex = ctx_.queue_family();
        YVO_VK_CHECK(vkCreateCommandPool(dev, &cp, nullptr, &pool_));
        VkCommandBufferAllocateInfo cbi{};
        cbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cbi.commandPool = pool_;
        cbi.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbi.commandBufferCount = 1;
        YVO_VK_CHECK(vkAllocateCommandBuffers(dev, &cbi, &cmd_));
        VkFenceCreateInfo fi{};
        fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        YVO_VK_CHECK(vkCreateFence(dev, &fi, nullptr, &fence_));
        if (ctx_.timestamps_supported()) {
            VkQueryPoolCreateInfo qp{};
            qp.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
            qp.queryType = VK_QUERY_TYPE_TIMESTAMP;
            qp.queryCount = 2;
            YVO_VK_CHECK(vkCreateQueryPool(dev, &qp, nullptr, &query_pool_));
        }
    }
    ~NmsHarness() {
        VkDevice dev = ctx_.device();
        vkDeviceWaitIdle(dev);
        nms_.reset();
        if (query_pool_) vkDestroyQueryPool(dev, query_pool_, nullptr);
        vkDestroyFence(dev, fence_, nullptr);
        vkDestroyCommandPool(dev, pool_, nullptr);
        vkUnmapMemory(dev, mem_);
        vkDestroyBuffer(dev, buf_, nullptr);
        vkFreeMemory(dev, mem_, nullptr);
    }
    NmsHarness(const NmsHarness&) = delete;
    NmsHarness& operator=(const NmsHarness&) = delete;

    // count is the value to write in the header (default = in.size()). Only min(in.size(), kInputMax) dets are written.
    bool run(const std::vector<DetectionKeyed>& in, std::vector<DetectionKeyed>& out, std::uint32_t* reason,
             long count = -1, std::uint32_t overflow = 0) {
        auto* h = reinterpret_cast<yvo::vk::YoloDetBufHeader*>(host_);
        std::memset(h, 0, sizeof(*h));
        h->count = count >= 0 ? static_cast<std::uint32_t>(count) : static_cast<std::uint32_t>(in.size());
        h->overflow = overflow;
        auto* d = reinterpret_cast<yvo::vk::YoloDetGpu*>(host_ + sizeof(yvo::vk::YoloDetBufHeader));
        for (std::size_t i = 0; i < in.size() && i < kInputMax; ++i)
            d[i] = {in[i].x1, in[i].y1, in[i].x2, in[i].y2, in[i].score, static_cast<std::uint32_t>(in[i].class_id),
                    in[i].anchor_key, 1u};

        YVO_VK_CHECK(vkResetCommandBuffer(cmd_, 0));
        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        YVO_VK_CHECK(vkBeginCommandBuffer(cmd_, &bi));
        if (query_pool_) vkCmdResetQueryPool(cmd_, query_pool_, 0, 2);
        barrier(buf_, VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_WRITE_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
        if (query_pool_) vkCmdWriteTimestamp2(cmd_, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, query_pool_, 0);
        nms_->record(cmd_, 0);
        if (query_pool_) vkCmdWriteTimestamp2(cmd_, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, query_pool_, 1);
        barrier(nms_->output(0), VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_READ_BIT);
        YVO_VK_CHECK(vkEndCommandBuffer(cmd_));
        YVO_VK_CHECK(vkResetFences(ctx_.device(), 1, &fence_));
        VkCommandBufferSubmitInfo cbs{};
        cbs.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
        cbs.commandBuffer = cmd_;
        VkSubmitInfo2 si{};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
        si.commandBufferInfoCount = 1;
        si.pCommandBufferInfos = &cbs;
        YVO_VK_CHECK(vkQueueSubmit2(ctx_.queue(), 1, &si, fence_));
        YVO_VK_CHECK(vkWaitForFences(ctx_.device(), 1, &fence_, VK_TRUE, UINT64_MAX));
        read_timestamps();
        if (reason) *reason = nms_->fallback_reason(0);
        return nms_->read_results(0, out);
    }

    // NMS dispatch interval of the last run() in ms (ts1 - ts0). -1 without timestamp support.
    // Same definition as YoloPostproc::last_nms_ms(): on V3DV a timestamp is a CPU-queue job that waits for the
    // preceding GPU job, so the interval includes the CSD job submission of the dispatch.
    double last_ms() const { return last_ms_; }
    const yvo::vk::EfficientNms& nms() const { return *nms_; }

private:
    void read_timestamps() {
        last_ms_ = -1.0;
        if (!query_pool_) return;
        std::uint64_t ts[2]{};
        const VkResult r = vkGetQueryPoolResults(ctx_.device(), query_pool_, 0, 2, sizeof(ts), ts,
                                                 sizeof(std::uint64_t), VK_QUERY_RESULT_64_BIT);
        if (r == VK_NOT_READY) return;
        YVO_VK_CHECK(r);
        // Same valid-bit masking as yolo_postproc.cpp wait()
        const std::uint32_t bits = ctx_.timestamp_valid_bits();
        const std::uint64_t mask = bits >= 64 ? ~0ull : ((1ull << bits) - 1);
        last_ms_ = static_cast<double>(((ts[1] & mask) - (ts[0] & mask)) & mask) *
                   static_cast<double>(ctx_.timestamp_period()) * 1e-6;
    }

    void barrier(VkBuffer b, VkPipelineStageFlags2 ss, VkAccessFlags2 sa, VkPipelineStageFlags2 ds,
                 VkAccessFlags2 da) {
        VkBufferMemoryBarrier2 m{};
        m.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
        m.srcStageMask = ss;
        m.srcAccessMask = sa;
        m.dstStageMask = ds;
        m.dstAccessMask = da;
        m.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        m.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        m.buffer = b;
        m.offset = 0;
        m.size = VK_WHOLE_SIZE;
        VkDependencyInfo di{};
        di.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        di.bufferMemoryBarrierCount = 1;
        di.pBufferMemoryBarriers = &m;
        vkCmdPipelineBarrier2(cmd_, &di);
    }

    yvo::vk::VulkanContext& ctx_;
    VkDeviceSize size_ = 0;
    VkBuffer buf_ = VK_NULL_HANDLE;
    VkDeviceMemory mem_ = VK_NULL_HANDLE;
    std::uint8_t* host_ = nullptr;
    std::unique_ptr<yvo::vk::EfficientNms> nms_;
    VkCommandPool pool_ = VK_NULL_HANDLE;
    VkCommandBuffer cmd_ = VK_NULL_HANDLE;
    VkFence fence_ = VK_NULL_HANDLE;
    VkQueryPool query_pool_ = VK_NULL_HANDLE;
    double last_ms_ = -1.0;
};

}  // namespace yvo::fixture
