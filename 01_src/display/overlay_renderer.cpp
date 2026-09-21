#include "display/overlay_renderer.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <stdexcept>
#include <string>

#include "overlay_boxes_spv.h" // 빌드 시 glslc 로 생성 (02_shaders/CMakeLists.txt)
#include "vulkan/vk_check.hpp"
#include "vulkan/vulkan_context.hpp"

namespace yvo::display {

namespace {

constexpr std::uint32_t kQueriesPerSlot = 2;     // [0] 디스패치 전 [1] 디스패치 후
constexpr std::size_t kGlyphWords = 95 * 4;      // kGlyph8x16
constexpr std::size_t kNameOffWords = 81;        // 클래스 80개 + 끝 오프셋

} // namespace

OverlayRenderer::OverlayRenderer(yvo::vk::VulkanContext& ctx, const OverlayConfig& cfg) : ctx_(ctx), cfg_(cfg) {
    if (cfg_.num_slots <= 0) throw std::invalid_argument("OverlayRenderer: num_slots must be > 0");
    if (cfg_.max_dets <= 0) throw std::invalid_argument("OverlayRenderer: max_dets must be > 0");
    if (cfg_.text_scale < 1) throw std::invalid_argument("OverlayRenderer: text_scale must be >= 1");
    if (cfg_.line_width < 1) throw std::invalid_argument("OverlayRenderer: line_width must be >= 1");
    const auto& lim = ctx_.properties().limits;
    if (cfg_.local_size_x == 0 || cfg_.local_size_x > lim.maxComputeWorkGroupInvocations ||
        cfg_.local_size_x > lim.maxComputeWorkGroupSize[0])
        throw std::invalid_argument("OverlayRenderer: local_size_x " + std::to_string(cfg_.local_size_x) +
                                    " exceeds device limits");
    // 워크그룹 = 검출 하나. 상한을 넘으면 DispatchIndirect 가 실패하므로 시작 시 거른다.
    if (static_cast<std::uint32_t>(cfg_.max_dets) > lim.maxComputeWorkGroupCount[0])
        throw std::invalid_argument("OverlayRenderer: max_dets > maxComputeWorkGroupCount[0]");
    if (cfg_.timestamps && !ctx_.timestamps_supported()) cfg_.timestamps = false;
    try {
        create_pipeline();
        create_font_buffer();
        create_slots();
    } catch (...) {
        destroy();
        throw;
    }
}

OverlayRenderer::~OverlayRenderer() { destroy(); }

OverlayStyle OverlayRenderer::style() const {
    OverlayStyle st;
    st.line_width = cfg_.line_width;
    st.text_scale = cfg_.text_scale;
    st.draw_text = cfg_.draw_text;
    return st;
}

void OverlayRenderer::create_pipeline() {
    VkDevice dev = ctx_.device();

    // 0: 출력 storage image, 1: DetBuf, 2: 폰트+이름 SSBO
    std::array<VkDescriptorSetLayoutBinding, 3> b{};
    for (std::uint32_t i = 0; i < 3; ++i) {
        b[i].binding = i;
        b[i].descriptorType = (i == 0) ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        b[i].descriptorCount = 1;
        b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dl{};
    dl.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dl.bindingCount = static_cast<std::uint32_t>(b.size());
    dl.pBindings = b.data();
    YVO_VK_CHECK(vkCreateDescriptorSetLayout(dev, &dl, nullptr, &set_layout_));

    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(OverlayPush)};
    VkPipelineLayoutCreateInfo pl{};
    pl.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pl.setLayoutCount = 1;
    pl.pSetLayouts = &set_layout_;
    pl.pushConstantRangeCount = 1;
    pl.pPushConstantRanges = &pcr;
    YVO_VK_CHECK(vkCreatePipelineLayout(dev, &pl, nullptr, &pipeline_layout_));

    VkShaderModuleCreateInfo smi{};
    smi.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smi.codeSize = sizeof(kOverlayBoxesSpv);
    smi.pCode = kOverlayBoxesSpv;
    VkShaderModule module = VK_NULL_HANDLE;
    YVO_VK_CHECK(vkCreateShaderModule(dev, &smi, nullptr, &module));

    // spec 0 = local_size_x, 1 = kDrawText (SPIR-V bool 은 32bit), 2 = kTextScale
    const std::array<std::uint32_t, 3> spec_data{cfg_.local_size_x, cfg_.draw_text ? VK_TRUE : VK_FALSE,
                                                 static_cast<std::uint32_t>(cfg_.text_scale)};
    const std::array<VkSpecializationMapEntry, 3> spec_map{{{0, 0, 4}, {1, 4, 4}, {2, 8, 4}}};
    VkSpecializationInfo spec{};
    spec.mapEntryCount = static_cast<std::uint32_t>(spec_map.size());
    spec.pMapEntries = spec_map.data();
    spec.dataSize = sizeof(spec_data);
    spec.pData = spec_data.data();

    VkComputePipelineCreateInfo cpi{};
    cpi.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpi.stage.module = module;
    cpi.stage.pName = "main";
    cpi.stage.pSpecializationInfo = &spec;
    cpi.layout = pipeline_layout_;
    const VkResult r = vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpi, nullptr, &pipeline_);
    vkDestroyShaderModule(dev, module, nullptr);
    YVO_VK_CHECK(r);

    const auto n = static_cast<std::uint32_t>(cfg_.num_slots);
    const std::array<VkDescriptorPoolSize, 2> ps{
        {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 * n}, {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, n}}};
    VkDescriptorPoolCreateInfo dp{};
    dp.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dp.maxSets = n;
    dp.poolSizeCount = static_cast<std::uint32_t>(ps.size());
    dp.pPoolSizes = ps.data();
    YVO_VK_CHECK(vkCreateDescriptorPool(dev, &dp, nullptr, &desc_pool_));

    if (cfg_.timestamps) {
        VkQueryPoolCreateInfo qp{};
        qp.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        qp.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qp.queryCount = kQueriesPerSlot * n;
        YVO_VK_CHECK(vkCreateQueryPool(dev, &qp, nullptr, &query_pool_));
    }
}

// 글리프 표 + COCO 이름 표를 한 버퍼에 담아 **시작 시 1회** 올린다(이후 읽기 전용 상수).
//   [0, 380)            kGlyph8x16
//   [380, 461)          name_off[81]  (바이트 오프셋, name_off[80] = 총 바이트 수)
//   [461, ...)          이름 ASCII 를 4바이트씩 패킹
void OverlayRenderer::create_font_buffer() {
    std::vector<std::uint32_t> words(kGlyphWords + kNameOffWords, 0u);
    std::memcpy(words.data(), kGlyph8x16, kGlyphWords * sizeof(std::uint32_t));

    std::vector<char> names;
    for (int c = 0; c < kNumCocoClasses; ++c) {
        words[kGlyphWords + static_cast<std::size_t>(c)] = static_cast<std::uint32_t>(names.size());
        const char* s = class_name(c);
        names.insert(names.end(), s, s + std::strlen(s));
    }
    words[kGlyphWords + kNumCocoClasses] = static_cast<std::uint32_t>(names.size());
    names.resize((names.size() + 3) / 4 * 4, 0); // 워드 패딩
    const std::size_t name_words = names.size() / 4;
    words.resize(words.size() + name_words, 0u);
    std::memcpy(words.data() + kGlyphWords + kNameOffWords, names.data(), names.size());

    font_size_ = static_cast<VkDeviceSize>(words.size() * sizeof(std::uint32_t));
    VkDevice dev = ctx_.device();
    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = font_size_;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    YVO_VK_CHECK(vkCreateBuffer(dev, &bci, nullptr, &font_buf_));
    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(dev, font_buf_, &req);
    const std::uint32_t type = ctx_.find_memory_type(
        req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (type == UINT32_MAX) throw std::runtime_error("OverlayRenderer: no HOST_VISIBLE|HOST_COHERENT memory type");
    VkMemoryAllocateInfo mai{};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = type;
    YVO_VK_CHECK(vkAllocateMemory(dev, &mai, nullptr, &font_mem_));
    YVO_VK_CHECK(vkBindBufferMemory(dev, font_buf_, font_mem_, 0));
    void* p = nullptr;
    YVO_VK_CHECK(vkMapMemory(dev, font_mem_, 0, VK_WHOLE_SIZE, 0, &p));
    std::memcpy(p, words.data(), static_cast<std::size_t>(font_size_));
    // UMA + HOST_COHERENT: flush 가 필요 없다. 이후 쓰지 않으므로 바로 unmap 한다.
    vkUnmapMemory(dev, font_mem_);
}

void OverlayRenderer::create_slots() {
    VkDevice dev = ctx_.device();
    det_size_ = static_cast<VkDeviceSize>(sizeof(OverlayDetBufHeader)) +
                static_cast<VkDeviceSize>(cfg_.max_dets) * sizeof(DetGpu);
    slots_.resize(static_cast<std::size_t>(cfg_.num_slots));
    for (Slot& s : slots_) {
        VkBufferCreateInfo bci{};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size = det_size_;
        // INDIRECT_BUFFER: vkCmdDispatchIndirect 가 맨 앞 uvec3 를 읽는다
        bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        YVO_VK_CHECK(vkCreateBuffer(dev, &bci, nullptr, &s.buf));
        VkMemoryRequirements req{};
        vkGetBufferMemoryRequirements(dev, s.buf, &req);
        const std::uint32_t type = ctx_.find_memory_type(
            req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (type == UINT32_MAX)
            throw std::runtime_error("OverlayRenderer: no HOST_VISIBLE|HOST_COHERENT memory type");
        VkMemoryAllocateInfo mai{};
        mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mai.allocationSize = req.size;
        mai.memoryTypeIndex = type;
        YVO_VK_CHECK(vkAllocateMemory(dev, &mai, nullptr, &s.mem));
        YVO_VK_CHECK(vkBindBufferMemory(dev, s.buf, s.mem, 0));
        void* p = nullptr;
        YVO_VK_CHECK(vkMapMemory(dev, s.mem, 0, VK_WHOLE_SIZE, 0, &p)); // 영구 매핑
        s.host = static_cast<std::uint8_t*>(p);
        std::memset(s.host, 0, static_cast<std::size_t>(det_size_));
        // 아직 검출이 없어도 DispatchIndirect 가 유효하도록 (1,1,1) 로 초기화한다
        OverlayDetBufHeader* h = reinterpret_cast<OverlayDetBufHeader*>(s.host);
        h->dispatch[0] = 1;
        h->dispatch[1] = 1;
        h->dispatch[2] = 1;

        VkDescriptorSetAllocateInfo dai{};
        dai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dai.descriptorPool = desc_pool_;
        dai.descriptorSetCount = 1;
        dai.pSetLayouts = &set_layout_;
        YVO_VK_CHECK(vkAllocateDescriptorSets(dev, &dai, &s.set));

        // binding 1(DetBuf), 2(폰트)는 슬롯 수명 동안 고정이다. binding 0 만 프레임마다 갱신한다.
        const std::array<VkDescriptorBufferInfo, 2> bi{{{s.buf, 0, det_size_}, {font_buf_, 0, font_size_}}};
        std::array<VkWriteDescriptorSet, 2> w{};
        for (std::uint32_t i = 0; i < 2; ++i) {
            w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[i].dstSet = s.set;
            w[i].dstBinding = i + 1;
            w[i].descriptorCount = 1;
            w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            w[i].pBufferInfo = &bi[i];
        }
        vkUpdateDescriptorSets(dev, 2, w.data(), 0, nullptr);
    }
}

void OverlayRenderer::destroy() {
    VkDevice dev = ctx_.device();
    if (dev == VK_NULL_HANDLE) return;
    vkDeviceWaitIdle(dev); // 종료 시에만
    for (Slot& s : slots_) {
        if (s.host) vkUnmapMemory(dev, s.mem);
        if (s.buf) vkDestroyBuffer(dev, s.buf, nullptr);
        if (s.mem) vkFreeMemory(dev, s.mem, nullptr);
        s.host = nullptr;
        s.buf = VK_NULL_HANDLE;
        s.mem = VK_NULL_HANDLE;
    }
    slots_.clear();
    if (font_buf_) vkDestroyBuffer(dev, font_buf_, nullptr);
    if (font_mem_) vkFreeMemory(dev, font_mem_, nullptr);
    if (query_pool_) vkDestroyQueryPool(dev, query_pool_, nullptr);
    if (desc_pool_) vkDestroyDescriptorPool(dev, desc_pool_, nullptr);
    if (pipeline_) vkDestroyPipeline(dev, pipeline_, nullptr);
    if (pipeline_layout_) vkDestroyPipelineLayout(dev, pipeline_layout_, nullptr);
    if (set_layout_) vkDestroyDescriptorSetLayout(dev, set_layout_, nullptr);
    font_buf_ = VK_NULL_HANDLE;
    font_mem_ = VK_NULL_HANDLE;
    query_pool_ = VK_NULL_HANDLE;
    desc_pool_ = VK_NULL_HANDLE;
    pipeline_ = VK_NULL_HANDLE;
    pipeline_layout_ = VK_NULL_HANDLE;
    set_layout_ = VK_NULL_HANDLE;
}

OverlayDetBufHeader* OverlayRenderer::header(int slot) {
    return reinterpret_cast<OverlayDetBufHeader*>(slots_.at(static_cast<std::size_t>(slot)).host);
}

DetGpu* OverlayRenderer::dets(int slot) {
    return reinterpret_cast<DetGpu*>(slots_.at(static_cast<std::size_t>(slot)).host +
                                     sizeof(OverlayDetBufHeader));
}

std::uint32_t OverlayRenderer::overflow(int slot) const {
    const Slot& s = slots_.at(static_cast<std::size_t>(slot));
    return reinterpret_cast<const OverlayDetBufHeader*>(s.host)->overflow;
}

int OverlayRenderer::set_frame(int slot, const std::vector<Detection>& in, std::string_view hud) {
    const std::size_t cap = static_cast<std::size_t>(cfg_.max_dets);
    const std::size_t m = std::min(in.size(), cap);
    DetGpu* d = dets(slot);
    for (std::size_t i = 0; i < m; ++i) {
        d[i].x1 = in[i].x1;
        d[i].y1 = in[i].y1;
        d[i].x2 = in[i].x2;
        d[i].y2 = in[i].y2;
        d[i].score = in[i].score;
        d[i].class_id = static_cast<std::uint32_t>(overlay_class_id(in[i].class_id));
        d[i].anchor_key = 0;
        d[i].keep = 1;
    }
    OverlayDetBufHeader* h = header(slot);
    h->dispatch[0] = static_cast<std::uint32_t>(std::max<std::size_t>(m, 1));
    h->dispatch[1] = 1;
    h->dispatch[2] = 1;
    h->count = static_cast<std::uint32_t>(m);
    h->overflow = in.size() > cap ? 1u : 0u;
    const std::size_t hn = std::min(hud.size(), kHudMaxBytes);
    h->text_bytes = static_cast<std::uint32_t>(hn);
    std::memset(h->text, 0, sizeof(h->text));
    if (hn) std::memcpy(h->text, hud.data(), hn);
    slots_[static_cast<std::size_t>(slot)].recorded = false;
    return static_cast<int>(m);
}

void OverlayRenderer::record(int slot, VkCommandBuffer cb, VkImageView dst_view, VkExtent2D dst_extent,
                             const OverlayAffine& affine, bool barrier_prior_compute_write) {
    if (cb == VK_NULL_HANDLE) throw std::invalid_argument("OverlayRenderer::record: null command buffer");
    Slot& s = slots_.at(static_cast<std::size_t>(slot));

    // binding 0 만 갱신한다. 이 슬롯을 쓰는 GPU 작업이 없을 때만 부르는 것은 호출자 책임이다
    // (DisplayConvert 의 슬롯 규약과 같다 — pending 슬롯에 begin() 할 수 없다).
    VkDescriptorImageInfo ii{};
    ii.imageView = dst_view;
    ii.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    VkWriteDescriptorSet w{};
    w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w.dstSet = s.set;
    w.dstBinding = 0;
    w.descriptorCount = 1;
    w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    w.pImageInfo = &ii;
    vkUpdateDescriptorSets(ctx_.device(), 1, &w, 0, nullptr);

    // ① 호스트가 쓴 DetBuf 를 가용화한다: HOST_WRITE → SHADER_READ | INDIRECT_COMMAND_READ
    VkBufferMemoryBarrier2 bb{};
    bb.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
    bb.srcStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
    bb.srcAccessMask = VK_ACCESS_2_HOST_WRITE_BIT;
    bb.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
    bb.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;
    bb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bb.buffer = s.buf;
    bb.offset = 0;
    bb.size = VK_WHOLE_SIZE;

    // ② 앞선 컴퓨트(display_convert)의 이미지 쓰기와의 WAW 해저드를 끊는다.
    // 레이아웃 전환이 없고(GENERAL 유지) record() 는 VkImage 가 아니라 뷰만 받으므로
    // **전역 메모리 배리어**를 쓴다(이미지 배리어의 상위집합이고, 같은 큐의 compute→compute
    // 의존성에서 V3D 비용도 같다). 오버레이는 이미지를 읽지 않으므로 dst 에 READ 를 넣지 않는다.
    VkMemoryBarrier2 mb{};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    mb.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    mb.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    mb.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    mb.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;

    VkDependencyInfo di{};
    di.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    di.bufferMemoryBarrierCount = 1;
    di.pBufferMemoryBarriers = &bb;
    if (barrier_prior_compute_write) {
        di.memoryBarrierCount = 1;
        di.pMemoryBarriers = &mb;
    }
    vkCmdPipelineBarrier2(cb, &di);

    OverlayPush pc{};
    pc.sx = affine.sx;
    pc.sy = affine.sy;
    pc.tx = affine.tx;
    pc.ty = affine.ty;
    pc.dst_w = static_cast<std::int32_t>(dst_extent.width);
    pc.dst_h = static_cast<std::int32_t>(dst_extent.height);
    pc.line_width = cfg_.line_width;
    pc.max_dets = static_cast<std::uint32_t>(cfg_.max_dets);

    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout_, 0, 1, &s.set, 0, nullptr);
    vkCmdPushConstants(cb, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

    const auto q = static_cast<std::uint32_t>(slot) * kQueriesPerSlot;
    if (query_pool_) {
        vkCmdResetQueryPool(cb, query_pool_, q, kQueriesPerSlot);
        // COMPUTE_SHADER 로 찍는다. 앞선 배리어 뒤이므로 변환 디스패치가 끝난 시점이 된다
        // (TOP_OF_PIPE 로 찍으면 변환이 끝나기 전에 resolve 되어 오버레이 구간을 과대 측정한다).
        vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, query_pool_, q);
    }
    vkCmdDispatchIndirect(cb, s.buf, 0);
    if (query_pool_) vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, query_pool_, q + 1);
    s.recorded = true;
}

void OverlayRenderer::collect(int slot) {
    Slot& s = slots_.at(static_cast<std::size_t>(slot));
    last_shader_ms_ = -1.0;
    if (!s.recorded || query_pool_ == VK_NULL_HANDLE) return;
    s.recorded = false;
    std::array<std::uint64_t, kQueriesPerSlot> ts{};
    const VkResult r = vkGetQueryPoolResults(ctx_.device(), query_pool_,
                                             static_cast<std::uint32_t>(slot) * kQueriesPerSlot, kQueriesPerSlot,
                                             sizeof(ts), ts.data(), sizeof(std::uint64_t), VK_QUERY_RESULT_64_BIT);
    if (r == VK_NOT_READY) return;
    YVO_VK_CHECK(r);
    // sand_letterbox.cpp:579-593 의 유효 비트 마스킹과 같은 계산
    const std::uint32_t bits = ctx_.timestamp_valid_bits();
    const std::uint64_t mask = bits >= 64 ? ~0ull : ((1ull << bits) - 1);
    const double period = static_cast<double>(ctx_.timestamp_period());
    const std::uint64_t d = ((ts[1] & mask) - (ts[0] & mask)) & mask;
    last_shader_ms_ = static_cast<double>(d) * period * 1e-6;
}

} // namespace yvo::display
