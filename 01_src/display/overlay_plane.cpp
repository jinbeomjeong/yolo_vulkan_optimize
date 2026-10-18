#include "display/overlay_plane.hpp"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <string>

#include "vulkan/dmabuf_export.hpp"
#include "vulkan/vk_check.hpp"
#include "vulkan/vulkan_context.hpp"

namespace yvo::display {

namespace {

constexpr std::uint32_t kFillQueries = 2;   // [0] before fill [1] after fill

std::uint32_t align_up_u32(std::uint32_t v, std::uint32_t a) { return a <= 1 ? v : (v + a - 1) / a * a; }

double ts_ms(yvo::vk::VulkanContext& ctx, std::uint64_t t0, std::uint64_t t1) {
    const std::uint32_t bits = ctx.timestamp_valid_bits();
    const std::uint64_t mask = bits >= 64 ? ~0ull : ((1ull << bits) - 1);
    const std::uint64_t d = ((t1 & mask) - (t0 & mask)) & mask;
    return static_cast<double>(d) * static_cast<double>(ctx.timestamp_period()) * 1e-6;
}

} // namespace

const char* clear_mode_name(OverlayClearMode m) {
    switch (m) {
    case OverlayClearMode::Auto: return "auto";
    case OverlayClearMode::Full: return "full";
    case OverlayClearMode::Erase: return "erase";
    }
    return "?";
}

OverlayPlane::OverlayPlane(yvo::vk::VulkanContext& ctx, const OverlayPlaneConfig& cfg) : ctx_(ctx), cfg_(cfg) {
    if (cfg_.ring < 1) throw std::invalid_argument("OverlayPlane: ring must be >= 1");
    if (cfg_.width <= 0 || cfg_.height <= 0) throw std::invalid_argument("OverlayPlane: bad size");
    if (cfg_.pitch_align == 0 || (cfg_.pitch_align & 3u) != 0)
        throw std::invalid_argument("OverlayPlane: pitch_align must be a positive multiple of 4");
    if (cfg_.timestamps && !ctx_.timestamps_supported()) cfg_.timestamps = false;
    // Auto = Erase: a full 1080p fill is 2.80 ms, over the 0.3 ms target (see header, OverlayPlane.ClearCost).
    clear_mode_ = cfg_.clear == OverlayClearMode::Full ? OverlayClearMode::Full : OverlayClearMode::Erase;
    foreign_qf_ = ctx_.external_queue_family();
    affine_ = make_overlay_affine(cfg_.src_w, cfg_.src_h, cfg_.width, cfg_.height);
    try {
        create();
        initial_clear();
    } catch (...) {
        destroy();
        throw;
    }
}

OverlayPlane::~OverlayPlane() { destroy(); }

void OverlayPlane::set_source_size(int w, int h) {
    if (w <= 0 || h <= 0) throw std::invalid_argument("OverlayPlane::set_source_size: bad size");
    cfg_.src_w = w;
    cfg_.src_h = h;
    affine_ = make_overlay_affine(w, h, cfg_.width, cfg_.height);
}

void OverlayPlane::create() {
    VkDevice dev = ctx_.device();
    const auto w = static_cast<std::uint32_t>(cfg_.width);
    const auto h = static_cast<std::uint32_t>(cfg_.height);
    const std::uint32_t pitch = align_up_u32(w * 4u, cfg_.pitch_align);
    pitch_px_ = pitch / 4u;
    const std::size_t size = static_cast<std::size_t>(pitch) * h;

    // Buffer: STORAGE (the shader writes) + TRANSFER_DST (vkCmdFillBuffer). No host mapping is used (the consumer uses the fd).
    yvo::vk::DmaBufExporterConfig ecfg;
    ecfg.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    ecfg.align = 4096;
    ecfg.map = false;
    exporter_ = std::make_unique<yvo::vk::DmaBufExporter>(ctx_, ecfg);

    OverlayConfig ocfg = cfg_.overlay;
    ocfg.num_slots = 2 * cfg_.ring;   // two slots per entry: draw / previous content
    ocfg.timestamps = cfg_.timestamps;
    ov_ = std::make_unique<OverlayRenderer>(ctx_, ocfg);

    VkCommandPoolCreateInfo cp{};
    cp.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cp.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cp.queueFamilyIndex = ctx_.queue_family();
    YVO_VK_CHECK(vkCreateCommandPool(dev, &cp, nullptr, &pool_));

    if (cfg_.timestamps) {
        VkQueryPoolCreateInfo qp{};
        qp.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        qp.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qp.queryCount = kFillQueries * static_cast<std::uint32_t>(cfg_.ring);
        YVO_VK_CHECK(vkCreateQueryPool(dev, &qp, nullptr, &fill_query_));
    }

    entries_.resize(static_cast<std::size_t>(cfg_.ring));
    for (std::size_t i = 0; i < entries_.size(); ++i) {
        Entry& e = entries_[i];
        const int bi = exporter_->allocate(size);
        const yvo::vk::ExportedDmaBuf& x = exporter_->at(bi);
        e.buffer = x.buffer;
        e.desc.fd = x.fd;
        e.desc.pitch = pitch;
        e.desc.width = w;
        e.desc.height = h;
        e.desc.size = size;
        e.desc.dmabuf_size = x.dmabuf_size;
        e.draw_slot = static_cast<int>(2 * i);

        VkCommandBufferAllocateInfo cai{};
        cai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cai.commandPool = pool_;
        cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cai.commandBufferCount = 1;
        YVO_VK_CHECK(vkAllocateCommandBuffers(dev, &cai, &e.cmd));
        VkFenceCreateInfo fi{};
        fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        YVO_VK_CHECK(vkCreateFence(dev, &fi, nullptr, &e.fence));
    }
}

// Once right after creation: fill every entry with (0,0,0,0) and release to FOREIGN (blocking, initialization only).
void OverlayPlane::initial_clear() {
    VkDevice dev = ctx_.device();
    for (Entry& e : entries_) {
        YVO_VK_CHECK(vkResetCommandBuffer(e.cmd, 0));
        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        YVO_VK_CHECK(vkBeginCommandBuffer(e.cmd, &bi));
        vkCmdFillBuffer(e.cmd, e.buffer, 0, e.desc.size, 0u);
        VkBufferMemoryBarrier2 rel{};
        rel.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
        rel.srcStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;
        rel.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        rel.dstStageMask = VK_PIPELINE_STAGE_2_NONE;
        rel.dstAccessMask = VK_ACCESS_2_NONE;
        rel.srcQueueFamilyIndex = ctx_.queue_family();
        rel.dstQueueFamilyIndex = foreign_qf_;
        rel.buffer = e.buffer;
        rel.offset = 0;
        rel.size = VK_WHOLE_SIZE;
        VkDependencyInfo di{};
        di.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        di.bufferMemoryBarrierCount = 1;
        di.pBufferMemoryBarriers = &rel;
        vkCmdPipelineBarrier2(e.cmd, &di);
        YVO_VK_CHECK(vkEndCommandBuffer(e.cmd));
        VkCommandBufferSubmitInfo cbi{};
        cbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
        cbi.commandBuffer = e.cmd;
        VkSubmitInfo2 si{};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
        si.commandBufferInfoCount = 1;
        si.pCommandBufferInfos = &cbi;
        YVO_VK_CHECK(vkQueueSubmit2(ctx_.queue(), 1, &si, e.fence));
        YVO_VK_CHECK(vkWaitForFences(dev, 1, &e.fence, VK_TRUE, 5'000'000'000ull));
        YVO_VK_CHECK(vkResetFences(dev, 1, &e.fence));
        e.released_foreign = true;
    }
}

void OverlayPlane::destroy() {
    VkDevice dev = ctx_.device();
    if (dev == VK_NULL_HANDLE) return;
    vkDeviceWaitIdle(dev); // only at shutdown
    for (Entry& e : entries_) {
        if (e.fence) vkDestroyFence(dev, e.fence, nullptr);
        e.fence = VK_NULL_HANDLE;
    }
    entries_.clear();   // command buffers are freed together with the pool
    if (fill_query_) vkDestroyQueryPool(dev, fill_query_, nullptr);
    fill_query_ = VK_NULL_HANDLE;
    if (pool_) vkDestroyCommandPool(dev, pool_, nullptr);
    pool_ = VK_NULL_HANDLE;
    ov_.reset();
    exporter_.reset();  // frees buffer, memory and fd (the consumer closes its own fd obtained via dup_fd separately)
}

OverlayPlane::Entry& OverlayPlane::entry(int idx) {
    if (idx < 0 || idx >= static_cast<int>(entries_.size())) throw std::out_of_range("OverlayPlane: bad index");
    return entries_[static_cast<std::size_t>(idx)];
}

const OverlayPlane::Entry& OverlayPlane::entry(int idx) const {
    if (idx < 0 || idx >= static_cast<int>(entries_.size())) throw std::out_of_range("OverlayPlane: bad index");
    return entries_[static_cast<std::size_t>(idx)];
}

const OverlayPlaneBuffer& OverlayPlane::buffer(int idx) const { return entry(idx).desc; }

int OverlayPlane::dup_fd(int idx) const {
    entry(idx); // range check
    return exporter_->dup_fd(idx);
}

bool OverlayPlane::held(int idx) const { return entry(idx).state == State::Held; }
bool OverlayPlane::claimed(int idx) const { return entry(idx).state == State::Claimed; }

int OverlayPlane::acquire() {
    int best = -1;
    for (int i = 0; i < static_cast<int>(entries_.size()); ++i) {
        Entry& e = entries_[static_cast<std::size_t>(i)];
        if (e.state != State::Free) continue;
        if (e.submitted && !ready(i)) continue;   // Free but the GPU is still using it (right after a drop) — skip
        if (best < 0 || e.seq < entries_[static_cast<std::size_t>(best)].seq) best = i;
    }
    if (best >= 0) {
        Entry& e = entries_[static_cast<std::size_t>(best)];
        e.state = State::Claimed;
        e.seq = ++seq_;
    }
    return best;
}

int OverlayPlane::draw(int idx, const std::vector<Detection>& dets, std::string_view hud) {
    Entry& e = entry(idx);
    if (e.state != State::Claimed) throw std::logic_error("OverlayPlane::draw: entry not claimed (acquire first)");
    if (e.submitted) throw std::logic_error("OverlayPlane::draw: entry already has a pending submit");

    VkDevice dev = ctx_.device();
    const int prev_slot = e.draw_slot;                         // slot holding the previous content
    const int slot = (e.draw_slot % 2 == 0) ? e.draw_slot + 1 : e.draw_slot - 1;
    const int n = ov_->set_frame(slot, dets, hud);
    if (ov_->overflow(slot)) ++overflows_;

    YVO_VK_CHECK(vkResetCommandBuffer(e.cmd, 0));
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    YVO_VK_CHECK(vkBeginCommandBuffer(e.cmd, &bi));

    const bool full = clear_mode_ == OverlayClearMode::Full;
    const bool do_erase = !full && e.has_prev;

    // ① External (scanout) → our queue ownership acquire. Buffer contents are preserved (Erase mode relies on this).
    //    The src scope on the acquire side is ignored, so NONE. dst is the following fill (transfer write) / shader write.
    if (e.released_foreign) {
        VkBufferMemoryBarrier2 acq{};
        acq.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
        acq.srcStageMask = VK_PIPELINE_STAGE_2_NONE;
        acq.srcAccessMask = VK_ACCESS_2_NONE;
        acq.dstStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        acq.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        acq.srcQueueFamilyIndex = foreign_qf_;
        acq.dstQueueFamilyIndex = ctx_.queue_family();
        acq.buffer = e.buffer;
        acq.offset = 0;
        acq.size = VK_WHOLE_SIZE;
        VkDependencyInfo di{};
        di.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        di.bufferMemoryBarrierCount = 1;
        di.pBufferMemoryBarriers = &acq;
        vkCmdPipelineBarrier2(e.cmd, &di);
    }

    const VkExtent2D ext{e.desc.width, e.desc.height};
    const VkDeviceSize range = static_cast<VkDeviceSize>(e.desc.size);
    e.pending_fill = false;
    e.pending_erase = false;
    if (full) {
        // ② Full: whole 0 → (WAW) → overlay
        const auto q = static_cast<std::uint32_t>(idx) * kFillQueries;
        if (fill_query_) {
            vkCmdResetQueryPool(e.cmd, fill_query_, q, kFillQueries);
            vkCmdWriteTimestamp2(e.cmd, VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, fill_query_, q);
        }
        vkCmdFillBuffer(e.cmd, e.buffer, 0, range, 0u);
        if (fill_query_) vkCmdWriteTimestamp2(e.cmd, VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, fill_query_, q + 1);
        VkBufferMemoryBarrier2 bb{};
        bb.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
        bb.srcStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;
        bb.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        bb.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        bb.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        bb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bb.buffer = e.buffer;
        bb.offset = 0;
        bb.size = VK_WHOLE_SIZE;
        VkDependencyInfo di{};
        di.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        di.bufferMemoryBarrierCount = 1;
        di.pBufferMemoryBarriers = &bb;
        vkCmdPipelineBarrier2(e.cmd, &di);
        e.pending_fill = true;
        ov_->record_buffer(slot, e.cmd, e.buffer, 0, range, ext, pitch_px_, affine_, false, false);
    } else {
        // ② Erase: redraw the previous content (prev_slot's DetBuf, previous affine) with (0,0,0,0)
        //    → global WAW barrier (record_buffer's barrier_prior_compute_write) → new overlay
        if (do_erase) {
            ov_->record_buffer(prev_slot, e.cmd, e.buffer, 0, range, ext, pitch_px_, e.prev_affine, false, true);
            e.pending_erase = true;
        }
        ov_->record_buffer(slot, e.cmd, e.buffer, 0, range, ext, pitch_px_, affine_, do_erase, false);
    }

    // ③ Our queue → external (scanout) release. The dst scope is ignored on release, so NONE.
    VkBufferMemoryBarrier2 rel{};
    rel.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
    rel.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    rel.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    rel.dstStageMask = VK_PIPELINE_STAGE_2_NONE;
    rel.dstAccessMask = VK_ACCESS_2_NONE;
    rel.srcQueueFamilyIndex = ctx_.queue_family();
    rel.dstQueueFamilyIndex = foreign_qf_;
    rel.buffer = e.buffer;
    rel.offset = 0;
    rel.size = VK_WHOLE_SIZE;
    VkDependencyInfo di{};
    di.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    di.bufferMemoryBarrierCount = 1;
    di.pBufferMemoryBarriers = &rel;
    vkCmdPipelineBarrier2(e.cmd, &di);

    YVO_VK_CHECK(vkEndCommandBuffer(e.cmd));
    YVO_VK_CHECK(vkResetFences(dev, 1, &e.fence));
    VkCommandBufferSubmitInfo cbi{};
    cbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
    cbi.commandBuffer = e.cmd;
    VkSubmitInfo2 si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
    si.commandBufferInfoCount = 1;
    si.pCommandBufferInfos = &cbi;
    YVO_VK_CHECK(vkQueueSubmit2(ctx_.queue(), 1, &si, e.fence));

    e.submitted = true;
    e.released_foreign = true;
    e.draw_slot = slot;
    e.has_prev = true;
    e.prev_affine = affine_;
    return n;
}

void OverlayPlane::collect(Entry& e, int idx) {
    e.submitted = false;
    last_clear_ms_ = -1.0;
    if (e.pending_erase) {
        const int prev_slot = (e.draw_slot % 2 == 0) ? e.draw_slot + 1 : e.draw_slot - 1;
        ov_->collect(prev_slot);
        last_clear_ms_ = ov_->last_shader_ms();
    }
    if (e.pending_fill && fill_query_) {
        std::array<std::uint64_t, kFillQueries> ts{};
        const VkResult r = vkGetQueryPoolResults(ctx_.device(), fill_query_,
                                                 static_cast<std::uint32_t>(idx) * kFillQueries, kFillQueries,
                                                 sizeof(ts), ts.data(), sizeof(std::uint64_t), VK_QUERY_RESULT_64_BIT);
        if (r == VK_SUCCESS) last_clear_ms_ = ts_ms(ctx_, ts[0], ts[1]);
        else if (r != VK_NOT_READY) YVO_VK_CHECK(r);
    }
    if (!e.pending_erase && !e.pending_fill) last_clear_ms_ = 0.0;   // first frame (nothing to erase)
    ov_->collect(e.draw_slot);
    last_overlay_ms_ = ov_->last_shader_ms();
    e.pending_erase = e.pending_fill = false;
}

bool OverlayPlane::ready(int idx) {
    Entry& e = entry(idx);
    if (!e.submitted) return true;
    const VkResult r = vkGetFenceStatus(ctx_.device(), e.fence);
    if (r == VK_NOT_READY) return false;
    YVO_VK_CHECK(r);
    collect(e, idx);
    return true;
}

void OverlayPlane::wait(int idx) {
    Entry& e = entry(idx);
    if (!e.submitted) return;
    YVO_VK_CHECK(vkWaitForFences(ctx_.device(), 1, &e.fence, VK_TRUE, 5'000'000'000ull));
    collect(e, idx);
}

void OverlayPlane::mark_held(int idx) {
    Entry& e = entry(idx);
    if (e.state != State::Claimed) throw std::logic_error("OverlayPlane::mark_held: entry not claimed");
    // The GPU write must be finished before handing over to the consumer (does not rely on implicit dma-buf fences).
    if (e.submitted && !ready(idx)) throw std::logic_error("OverlayPlane::mark_held: GPU work not finished");
    e.state = State::Held;
}

void OverlayPlane::release(int idx) {
    Entry& e = entry(idx);
    if (e.state == State::Free) throw std::logic_error("OverlayPlane::release: entry already free");
    e.state = State::Free;
}

} // namespace yvo::display
