#include "vulkan/yolo_postproc.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>

#include "vulkan/dmabuf_export.hpp"
#include "vulkan/dmabuf_import.hpp"
#include "vulkan/vk_check.hpp"
#include "vulkan/vk_storage_headroom.hpp"
#include "vulkan/vulkan_context.hpp"
#include "yolo_decode_spv.h" // 빌드 시 glslc 로 생성 (02_shaders/CMakeLists.txt)

namespace yvo::vk {

namespace {

// [0] cmd 시작 [1] 디스패치 전 [2] 디스패치 후 [3] cmd 끝
constexpr std::uint32_t kQueriesPerSlot = 4;

} // namespace

YoloPostproc::YoloPostproc(VulkanContext& ctx, const YoloTensorLayout& layout, const YoloPostprocConfig& cfg)
    : ctx_(ctx), layout_(layout), cfg_(cfg) {
    validate();
    if (cfg_.timestamps && !ctx_.timestamps_supported()) cfg_.timestamps = false;
    try {
        if (cfg_.input_alloc == YoloInputAlloc::ExportDmaBuf) {
            DmaBufExporterConfig ec;
            ec.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;   // yolo_decode.comp 가 읽는다
            ec.align = kHailoAsyncAlign;                     // 부록 A.4: HailoRT read_async 는 16384 정렬
            ec.map = true;
            ec.verbose = cfg_.verbose;
            exporter_ = std::make_unique<DmaBufExporter>(ctx_, ec);
        } else if (cfg_.input_alloc == YoloInputAlloc::ImportHeap) {
            setup_heap();
        }
        create_pipelines();
        create_slots();
    } catch (...) {
        destroy();
        throw;
    }
}

YoloPostproc::~YoloPostproc() { destroy(); }

void YoloPostproc::validate() {
    if (cfg_.num_slots <= 0) throw std::invalid_argument("YoloPostproc: num_slots must be > 0");
    if (cfg_.max_dets <= 0) throw std::invalid_argument("YoloPostproc: max_dets must be > 0");
    if (cfg_.score_thr <= 0.0f) throw std::invalid_argument("YoloPostproc: score_thr must be > 0");
    if (layout_.reg_max != kDflRegMax)
        throw std::invalid_argument("YoloPostproc: reg_max must be " + std::to_string(kDflRegMax) +
                                    " (셰이더의 컴파일 타임 상수다)");
    if (layout_.num_classes <= 0) throw std::invalid_argument("YoloPostproc: num_classes must be > 0");

    const auto& lim = ctx_.properties().limits;
    if (cfg_.local_size_x == 0 || cfg_.local_size_y == 0 ||
        cfg_.local_size_x * cfg_.local_size_y > lim.maxComputeWorkGroupInvocations ||
        cfg_.local_size_x > lim.maxComputeWorkGroupSize[0] || cfg_.local_size_y > lim.maxComputeWorkGroupSize[1])
        throw std::invalid_argument("YoloPostproc: local size " + std::to_string(cfg_.local_size_x) + "x" +
                                    std::to_string(cfg_.local_size_y) + " exceeds device limits");

    for (int i = 0; i < kYoloNumScales; ++i) {
        const YoloScaleSpec& s = layout_.s[i];
        if (s.grid_h <= 0 || s.grid_w <= 0 || s.w_padded < s.grid_w || s.stride <= 0)
            throw std::invalid_argument("YoloPostproc: scale " + std::to_string(i) + " geometry invalid");
        if (s.grid_w % 4 != 0)
            throw std::invalid_argument("YoloPostproc: grid_w must be a multiple of 4 (워드 = 4앵커)");
        if (s.w_padded % 4 != 0)
            throw std::invalid_argument("YoloPostproc: w_padded must be a multiple of 4 (워드 정렬)");
        const std::size_t need_cls =
            static_cast<std::size_t>(s.grid_h) * static_cast<std::size_t>(layout_.num_classes) *
            static_cast<std::size_t>(s.w_padded);
        const std::size_t need_box = static_cast<std::size_t>(s.grid_h) *
                                     static_cast<std::size_t>(kDflSides * layout_.reg_max) *
                                     static_cast<std::size_t>(s.w_padded);
        if (s.cls_bytes < need_cls || s.box_bytes < need_box)
            throw std::invalid_argument("YoloPostproc: scale " + std::to_string(i) +
                                        " tensor size smaller than NHCW extent");
        if (s.cls_scale <= 0.0f || s.box_scale <= 0.0f)
            throw std::invalid_argument("YoloPostproc: qp_scale must be > 0");
    }
}

std::uint32_t YoloPostproc::q_thr_low(int scale_id) const {
    // 부록 A.5: q_thr_low = floor(thr / qp_scale). 클래스 텐서는 qp_zp = 0 이므로 zp 보정이 없다.
    // zp != 0 인 HEF 를 만나면 (thr/scale + zp) 로 옮겨야 하므로 여기서 한 번에 처리한다.
    const YoloScaleSpec& s = layout_.s[scale_id];
    const double q = static_cast<double>(cfg_.score_thr) / static_cast<double>(s.cls_scale) +
                     static_cast<double>(s.cls_zp);
    if (q <= 0.0) return 0;
    if (q >= 255.0) return 255;
    return static_cast<std::uint32_t>(std::floor(q));
}

void YoloPostproc::create_pipelines() {
    VkDevice dev = ctx_.device();

    // 0: 클래스 텐서, 1: box 텐서, 2: DetBuf — 한 디스패치가 보는 SSBO 는 이 3개뿐이다
    std::array<VkDescriptorSetLayoutBinding, 3> b{};
    for (std::uint32_t i = 0; i < 3; ++i) {
        b[i].binding = i;
        b[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        b[i].descriptorCount = 1;
        b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dl{};
    dl.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dl.bindingCount = static_cast<std::uint32_t>(b.size());
    dl.pBindings = b.data();
    YVO_VK_CHECK(vkCreateDescriptorSetLayout(dev, &dl, nullptr, &set_layout_));

    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(YoloDecodePush)};
    VkPipelineLayoutCreateInfo pl{};
    pl.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pl.setLayoutCount = 1;
    pl.pSetLayouts = &set_layout_;
    pl.pushConstantRangeCount = 1;
    pl.pPushConstantRanges = &pcr;
    YVO_VK_CHECK(vkCreatePipelineLayout(dev, &pl, nullptr, &pipeline_layout_));

    // 개발용: 환경변수 YVO_YOLO_DECODE_SPV 로 임베드 대신 외부 .spv 를 쓸 수 있다(셰이더 실험·튜닝).
    // sand_letterbox.cpp:84-96 과 같은 관례다.
    std::vector<std::uint32_t> ext_spv;
    if (const char* path = std::getenv("YVO_YOLO_DECODE_SPV"); path && *path) {
        std::ifstream in(path, std::ios::binary | std::ios::ate);
        if (!in) throw std::runtime_error(std::string("YoloPostproc: cannot open ") + path);
        const auto n = static_cast<std::size_t>(in.tellg());
        if (n == 0 || n % 4) throw std::runtime_error(std::string("YoloPostproc: bad SPIR-V size: ") + path);
        ext_spv.resize(n / 4);
        in.seekg(0);
        in.read(reinterpret_cast<char*>(ext_spv.data()), static_cast<std::streamsize>(n));
        std::fprintf(stderr, "[YoloPostproc] using external SPIR-V %s\n", path);
    }
    VkShaderModuleCreateInfo smi{};
    smi.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smi.codeSize = ext_spv.empty() ? sizeof(kYoloDecodeSpv) : ext_spv.size() * 4;
    smi.pCode = ext_spv.empty() ? kYoloDecodeSpv : ext_spv.data();
    VkShaderModule module = VK_NULL_HANDLE;
    YVO_VK_CHECK(vkCreateShaderModule(dev, &smi, nullptr, &module));

    // spec 0/1 = local_size_x/y, 2 = kChecksum (SPIR-V bool 은 32bit 로 넘긴다)
    std::array<std::uint32_t, 3> spec_data{cfg_.local_size_x, cfg_.local_size_y, VK_FALSE};
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
    VkResult r = vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpi, nullptr, &pipeline_);
    if (r == VK_SUCCESS && cfg_.checksum_pipeline) {
        spec_data[2] = VK_TRUE;
        r = vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpi, nullptr, &pipeline_sum_);
    }
    vkDestroyShaderModule(dev, module, nullptr);
    YVO_VK_CHECK(r);

    const auto n = static_cast<std::uint32_t>(cfg_.num_slots * kYoloNumScales);
    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 * n};
    VkDescriptorPoolCreateInfo dp{};
    dp.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dp.maxSets = n;
    dp.poolSizeCount = 1;
    dp.pPoolSizes = &ps;
    YVO_VK_CHECK(vkCreateDescriptorPool(dev, &dp, nullptr, &desc_pool_));

    VkCommandPoolCreateInfo cp{};
    cp.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cp.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cp.queueFamilyIndex = ctx_.queue_family();
    YVO_VK_CHECK(vkCreateCommandPool(dev, &cp, nullptr, &cmd_pool_));

    if (cfg_.timestamps) {
        VkQueryPoolCreateInfo qp{};
        qp.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        qp.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qp.queryCount = kQueriesPerSlot * static_cast<std::uint32_t>(cfg_.num_slots);
        YVO_VK_CHECK(vkCreateQueryPool(dev, &qp, nullptr, &query_pool_));
    }
}

// ImportHeap 준비 — 계획서 부록 B.12 절차를 그대로 따른다.
//   1) 6텐서 각 크기로 더미 VkBuffer 를 만들어 `vkGetBufferMemoryRequirements2()` 로 요구 크기를 묻고
//   2) headroom = req.size - requested 의 **최대값**을 `DmaHeapConfig::device_headroom` 에 넣는다
//      (풀 하나에 값 하나라 최대값을 쓴다. 과할당은 안전하다 — V3D 는 requires/prefers Dedicated = 0 이라
//       `VkMemoryDedicatedAllocateInfo` 를 붙이지 않고, 따라서 VUID-VkMemoryDedicatedAllocateInfo-buffer-02965
//       의 `allocationSize == req.size` 요구가 적용되지 않는다. validusage.json, Vulkan 1.4.309)
// **상수 256 을 박지 않는다.** 드라이버가 바뀌면 질의값도 바뀐다.
void YoloPostproc::setup_heap() {
    StorageHeadroomQuery hq(ctx_.device());
    std::array<std::size_t, kYoloNumTensors> sizes{};
    for (int i = 0; i < kYoloNumScales; ++i) {
        sizes[static_cast<std::size_t>(yolo_cls_tensor(i))] = layout_.s[i].cls_bytes;
        sizes[static_cast<std::size_t>(yolo_box_tensor(i))] = layout_.s[i].box_bytes;
    }
    heap_headroom_ = hq.max_headroom(sizes.begin(), sizes.end());

    yvo::DmaHeapConfig hc;
    hc.align = kHailoAsyncAlign;          // 16384 — HailoRT read_async 요구(부록 A.4)
    hc.device_headroom = heap_headroom_;
    hc.map = true;
    hc.strict_sync = true;                // R5: ioctl 실패는 예외다
    hc.zero_init = true;                  // R3: memset 을 WRITE sync 로 감싼 채 0 으로 채운다
    hc.name = "yvo_yolo_raw";
    heap_ = std::make_unique<yvo::DmaHeapPool>(hc);
    import_cache_ = std::make_unique<DmaBufImportCache>(
        ctx_, static_cast<std::size_t>(cfg_.num_slots) * kYoloNumTensors + 4);
    if (cfg_.verbose)
        std::fprintf(stderr, "[YoloPostproc] ImportHeap: heap=%s headroom=%zu B align=%zu heap_sync=%d\n",
                     heap_->heap().c_str(), heap_headroom_, static_cast<std::size_t>(kHailoAsyncAlign),
                     cfg_.heap_sync ? 1 : 0);
}

YoloPostproc::Tensor YoloPostproc::alloc_tensor(std::size_t bytes) {
    Tensor t;
    if (heap_) {
        // dma-heap 에서 align_up(bytes + headroom, 16384) 를 잡는다. zero_init 는 풀이 WRITE sync 로 감싼다.
        const int hi = heap_->allocate(bytes);
        const yvo::DmaHeapBuf hb = heap_->at(hi);   // POD 복사 — 다음 allocate() 가 벡터를 재할당해도 안전
        // min_range = bytes → 드라이버 여유 때문에 range 가 실데이터보다 작아지면 여기서 예외다(B.12 강제 검증)
        ImportedDmaBuf& e = import_cache_->get(hb.fd, hb.dmabuf_size, bytes);
        // 슬롯 수명 내내 살아 있어야 한다 → pin. (캐시 소멸자는 pin 과 무관하게 전부 해제한다.)
        import_cache_->pin(e);
        t.buf = e.buffer;
        t.mem = VK_NULL_HANDLE;   // import 캐시 소유
        t.host = hb.ptr;          // **힙 fd 의 mmap 포인터**다. vkMapMemory 가 아니다
        t.fd = hb.fd;             // 소유권은 DmaHeapPool 에 있다
        t.dmabuf_size = hb.dmabuf_size;
        t.range = e.range;
        t.from_heap = true;
        return t;
    }
    if (exporter_) {
        const int idx = exporter_->allocate(bytes);
        const ExportedDmaBuf& e = exporter_->at(idx);
        t.buf = e.buffer;
        t.mem = VK_NULL_HANDLE;   // exporter 소유
        t.host = e.aligned_host_ptr();
        t.fd = e.fd;
        t.dmabuf_size = e.dmabuf_size;
        t.range = VK_WHOLE_SIZE;
        std::memset(t.host, 0, bytes);
        return t;
    }
    VkDevice dev = ctx_.device();
    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = static_cast<VkDeviceSize>(bytes);
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    YVO_VK_CHECK(vkCreateBuffer(dev, &bci, nullptr, &t.buf));
    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(dev, t.buf, &req);
    const std::uint32_t type = ctx_.find_memory_type(
        req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (type == UINT32_MAX) throw std::runtime_error("YoloPostproc: no HOST_VISIBLE|HOST_COHERENT memory type");
    VkMemoryAllocateInfo mai{};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = type;
    YVO_VK_CHECK(vkAllocateMemory(dev, &mai, nullptr, &t.mem));
    YVO_VK_CHECK(vkBindBufferMemory(dev, t.buf, t.mem, 0));
    void* p = nullptr;
    YVO_VK_CHECK(vkMapMemory(dev, t.mem, 0, VK_WHOLE_SIZE, 0, &p));   // 영구 매핑
    t.host = static_cast<std::uint8_t*>(p);
    t.range = bci.size;
    std::memset(t.host, 0, bytes);
    return t;
}

void YoloPostproc::create_slots() {
    VkDevice dev = ctx_.device();
    det_size_ = static_cast<VkDeviceSize>(sizeof(YoloDetBufHeader)) +
                static_cast<VkDeviceSize>(cfg_.max_dets) * sizeof(YoloDetGpu);
    slots_.resize(static_cast<std::size_t>(cfg_.num_slots));

    for (Slot& s : slots_) {
        // ── 원시 출력 6개 ──
        for (int i = 0; i < kYoloNumScales; ++i) {
            s.t[yolo_cls_tensor(i)] = alloc_tensor(layout_.s[i].cls_bytes);
            s.t[yolo_box_tensor(i)] = alloc_tensor(layout_.s[i].box_bytes);
        }

        // ── DetBuf ──
        VkBufferCreateInfo bci{};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size = det_size_;
        // INDIRECT_BUFFER: 오버레이가 vkCmdDispatchIndirect 로 맨 앞 uvec3 를 읽는다(계획서 §5.4)
        bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        YVO_VK_CHECK(vkCreateBuffer(dev, &bci, nullptr, &s.det));
        VkMemoryRequirements req{};
        vkGetBufferMemoryRequirements(dev, s.det, &req);
        const std::uint32_t type = ctx_.find_memory_type(
            req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (type == UINT32_MAX) throw std::runtime_error("YoloPostproc: no HOST_VISIBLE|HOST_COHERENT memory type");
        VkMemoryAllocateInfo mai{};
        mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mai.allocationSize = req.size;
        mai.memoryTypeIndex = type;
        YVO_VK_CHECK(vkAllocateMemory(dev, &mai, nullptr, &s.det_mem));
        YVO_VK_CHECK(vkBindBufferMemory(dev, s.det, s.det_mem, 0));
        void* p = nullptr;
        YVO_VK_CHECK(vkMapMemory(dev, s.det_mem, 0, VK_WHOLE_SIZE, 0, &p));   // 영구 매핑
        s.det_host = static_cast<std::uint8_t*>(p);
        std::memset(s.det_host, 0, static_cast<std::size_t>(det_size_));

        // ── 디스크립터 셋: 스케일당 1개, **전부 정적**(초기화 때 한 번만 갱신) ──
        for (int i = 0; i < kYoloNumScales; ++i) {
            VkDescriptorSetAllocateInfo dai{};
            dai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            dai.descriptorPool = desc_pool_;
            dai.descriptorSetCount = 1;
            dai.pSetLayouts = &set_layout_;
            YVO_VK_CHECK(vkAllocateDescriptorSets(dev, &dai, &s.set[i]));

            const Tensor& tc = s.t[yolo_cls_tensor(i)];
            const Tensor& tb = s.t[yolo_box_tensor(i)];
            const std::array<VkDescriptorBufferInfo, 3> bi{
                {{tc.buf, 0, tc.range}, {tb.buf, 0, tb.range}, {s.det, 0, det_size_}}};
            std::array<VkWriteDescriptorSet, 3> w{};
            for (std::uint32_t k = 0; k < 3; ++k) {
                w[k].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                w[k].dstSet = s.set[i];
                w[k].dstBinding = k;
                w[k].descriptorCount = 1;
                w[k].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                w[k].pBufferInfo = &bi[k];
            }
            vkUpdateDescriptorSets(dev, 3, w.data(), 0, nullptr);
        }

        VkCommandBufferAllocateInfo cbi{};
        cbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cbi.commandPool = cmd_pool_;
        cbi.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbi.commandBufferCount = 1;
        YVO_VK_CHECK(vkAllocateCommandBuffers(dev, &cbi, &s.cmd));

        VkFenceCreateInfo fi{};
        fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        YVO_VK_CHECK(vkCreateFence(dev, &fi, nullptr, &s.fence));
    }
}

void YoloPostproc::record(Slot& s, int slot, const LetterboxParams& lb, int src_w, int src_h, bool checksum) {
    VkCommandBuffer cb = s.cmd;
    YVO_VK_CHECK(vkResetCommandBuffer(cb, 0));
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    YVO_VK_CHECK(vkBeginCommandBuffer(cb, &bi));

    const std::uint32_t qf = ctx_.queue_family();
    const std::uint32_t ext = ctx_.external_queue_family();
    const auto qslot = static_cast<std::uint32_t>(slot) * kQueriesPerSlot;
    if (query_pool_) {
        vkCmdResetQueryPool(cb, query_pool_, qslot, kQueriesPerSlot);
        vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, query_pool_, qslot);
    }

    // ── 진입 배리어 ───────────────────────────────────────────────────────────────────────────
    // ① DetBuf: 호스트가 방금 쓴 헤더(count/overflow/dispatch)를 셰이더에 가용화한다.
    //    vkQueueSubmit 자체가 호스트 쓰기에 대한 암묵적 의존성을 만들지만, 명시하는 편이
    //    synchronization validation 과 의도 전달에 모두 낫다(overlay_renderer.cpp 와 같은 관례).
    // ② 원시 출력 6개: input_sync 에 따라 FOREIGN_EXT acquire / HOST_WRITE 가용화 / 없음.
    std::array<VkBufferMemoryBarrier2, 1 + kYoloNumTensors> acq{};
    std::uint32_t nacq = 0;
    {
        VkBufferMemoryBarrier2& b = acq[nacq++];
        b.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
        b.srcStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
        b.srcAccessMask = VK_ACCESS_2_HOST_WRITE_BIT;
        b.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        b.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.buffer = s.det;
        b.offset = 0;
        b.size = VK_WHOLE_SIZE;
    }
    if (cfg_.input_sync != YoloInputSync::None) {
        for (int i = 0; i < kYoloNumTensors; ++i) {
            VkBufferMemoryBarrier2& b = acq[nacq++];
            b.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
            b.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            b.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
            b.buffer = s.t[i].buf;
            b.offset = 0;
            b.size = VK_WHOLE_SIZE;
            if (cfg_.input_sync == YoloInputSync::Foreign) {
                // 외부 메모리(Hailo PCIe DMA 가 쓴 것)의 내용을 읽으려면 FOREIGN_EXT 로부터 acquire 한다.
                // Vulkan 1.4.309 spec "Queue Family Ownership Transfer" / VK_EXT_queue_family_foreign,
                // VUID-VkBufferMemoryBarrier2-buffer-09095 (EXCLUSIVE + 두 인덱스가 다르면 src 는
                // EXTERNAL / FOREIGN_EXT / 유효 큐 패밀리여야 한다).
                b.srcStageMask = VK_PIPELINE_STAGE_2_NONE;
                b.srcAccessMask = VK_ACCESS_2_NONE;
                b.srcQueueFamilyIndex = ext;
                b.dstQueueFamilyIndex = qf;
            } else {
                b.srcStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
                b.srcAccessMask = VK_ACCESS_2_HOST_WRITE_BIT;
                b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            }
        }
    }
    VkDependencyInfo di{};
    di.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    di.bufferMemoryBarrierCount = nacq;
    di.pBufferMemoryBarriers = acq.data();
    vkCmdPipelineBarrier2(cb, &di);

    // ── 3 디스패치 (스케일별. 파이프라인은 하나, push constant 만 다르다. 사이에 배리어 없음) ──
    VkPipeline pipe = checksum ? pipeline_sum_ : pipeline_;
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    if (query_pool_) vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, query_pool_, qslot + 1);
    for (int i = 0; i < kYoloNumScales; ++i) {
        const YoloScaleSpec& sc = layout_.s[i];
        YoloDecodePush pc{};
        pc.grid_h = sc.grid_h;
        pc.grid_w = sc.grid_w;
        pc.w_words = sc.grid_w / 4;
        pc.w_words_pad = sc.w_padded / 4;
        pc.num_classes = layout_.num_classes;
        pc.scale_id = i;
        pc.q_thr_low = q_thr_low(i);
        pc.max_dets = static_cast<std::uint32_t>(cfg_.max_dets);
        pc.stride = static_cast<float>(sc.stride);
        pc.cls_scale = sc.cls_scale;
        pc.cls_zp = sc.cls_zp;
        pc.box_scale = sc.box_scale;
        pc.box_zp = sc.box_zp;
        pc.thr = cfg_.score_thr;
        pc.inv_scale = 1.0f / lb.scale;
        pc.pad_x = static_cast<float>(lb.pad_x);
        pc.pad_y = static_cast<float>(lb.pad_y);
        pc.max_x = static_cast<float>(src_w);
        pc.max_y = static_cast<float>(src_h);

        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout_, 0, 1, &s.set[i], 0, nullptr);
        vkCmdPushConstants(cb, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        const std::uint32_t gx =
            (static_cast<std::uint32_t>(pc.w_words) + cfg_.local_size_x - 1) / cfg_.local_size_x;
        const std::uint32_t gy =
            (static_cast<std::uint32_t>(pc.grid_h) + cfg_.local_size_y - 1) / cfg_.local_size_y;
        vkCmdDispatch(cb, gx, gy, 1);
    }
    if (query_pool_) vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, query_pool_, qslot + 2);

    // ── 종료 배리어 ───────────────────────────────────────────────────────────────────────────
    // ① 원시 출력 release (Foreign 일 때만). 읽기만 했으므로 가용화할 쓰기가 없다 → srcAccessMask NONE.
    //    release 연산에서는 dstStageMask/dstAccessMask 가 무시된다(sand_letterbox.cpp:401-424 와 같다).
    // ② DetBuf: 셰이더 쓰기(atomic 포함 — atomic 은 read+write 접근이다) → 호스트 읽기 가시성.
    std::array<VkBufferMemoryBarrier2, 1 + kYoloNumTensors> rel{};
    std::uint32_t nrel = 0;
    if (cfg_.input_sync == YoloInputSync::Foreign) {
        for (int i = 0; i < kYoloNumTensors; ++i) {
            VkBufferMemoryBarrier2& b = rel[nrel++];
            b.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
            b.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            b.srcAccessMask = VK_ACCESS_2_NONE;
            b.dstStageMask = VK_PIPELINE_STAGE_2_NONE;
            b.dstAccessMask = VK_ACCESS_2_NONE;
            b.srcQueueFamilyIndex = qf;
            b.dstQueueFamilyIndex = ext;
            b.buffer = s.t[i].buf;
            b.offset = 0;
            b.size = VK_WHOLE_SIZE;
        }
    }
    {
        VkBufferMemoryBarrier2& b = rel[nrel++];
        b.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
        b.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        b.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        b.dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
        b.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.buffer = s.det;
        b.offset = 0;
        b.size = VK_WHOLE_SIZE;
    }
    di.bufferMemoryBarrierCount = nrel;
    di.pBufferMemoryBarriers = rel.data();
    vkCmdPipelineBarrier2(cb, &di);

    if (query_pool_) vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, query_pool_, qslot + 3);
    YVO_VK_CHECK(vkEndCommandBuffer(cb));
}

void YoloPostproc::submit(int slot, const LetterboxParams& lb, int src_w, int src_h, bool checksum) {
    if (slot < 0 || slot >= num_slots()) throw std::out_of_range("YoloPostproc::submit: bad slot");
    Slot& s = slots_[static_cast<std::size_t>(slot)];
    if (s.pending) throw std::logic_error("YoloPostproc::submit: slot " + std::to_string(slot) + " still pending");
    if (checksum && pipeline_sum_ == VK_NULL_HANDLE)
        throw std::logic_error("YoloPostproc::submit: checksum_pipeline 이 꺼져 있다");
    if (lb.scale <= 0.0f) throw std::invalid_argument("YoloPostproc::submit: letterbox scale must be > 0");

    // 카운터 초기화는 **호스트 쓰기**다(계획서 §4.3 — vkCmdFillBuffer 와 TRANSFER_DST 불필요).
    // dispatch = (1,1,1) 로 두면 후보 0개여도 오버레이 워크그룹 0(HUD)이 돈다(§5.4 규약).
    auto* h = reinterpret_cast<YoloDetBufHeader*>(s.det_host);
    h->dispatch[0] = 1;
    h->dispatch[1] = 1;
    h->dispatch[2] = 1;
    h->count = 0;
    h->overflow = 0;
    h->text_bytes = 0;
    std::memset(h->text, 0, sizeof(h->text));

    record(s, slot, lb, src_w, src_h, checksum);

    YVO_VK_CHECK(vkResetFences(ctx_.device(), 1, &s.fence));
    VkCommandBufferSubmitInfo cbi{};
    cbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
    cbi.commandBuffer = s.cmd;
    VkSubmitInfo2 si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
    si.commandBufferInfoCount = 1;
    si.pCommandBufferInfos = &cbi;
    YVO_VK_CHECK(vkQueueSubmit2(ctx_.queue(), 1, &si, s.fence));
    s.pending = true;
}

bool YoloPostproc::ready(int slot) const {
    const Slot& s = slots_.at(static_cast<std::size_t>(slot));
    if (!s.pending) return true;
    const VkResult r = vkGetFenceStatus(ctx_.device(), s.fence);
    if (r == VK_NOT_READY) return false;
    YVO_VK_CHECK(r);
    return true;
}

void YoloPostproc::wait(int slot, std::uint64_t timeout_ns) {
    if (slot < 0 || slot >= num_slots()) throw std::out_of_range("YoloPostproc::wait: bad slot");
    Slot& s = slots_[static_cast<std::size_t>(slot)];
    if (!s.pending) throw std::logic_error("YoloPostproc::wait: slot " + std::to_string(slot) + " not pending");
    YVO_VK_CHECK(vkWaitForFences(ctx_.device(), 1, &s.fence, VK_TRUE, timeout_ns));
    s.pending = false;

    last_pass1_ms_ = -1.0;
    last_total_ms_ = -1.0;
    if (query_pool_ == VK_NULL_HANDLE) return;
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
    last_pass1_ms_ = static_cast<double>(((ts[2] & mask) - (ts[1] & mask)) & mask) * period * 1e-6;
    last_total_ms_ = static_cast<double>(((ts[3] & mask) - (ts[0] & mask)) & mask) * period * 1e-6;
}

const YoloDetBufHeader* YoloPostproc::header(int slot) const {
    return reinterpret_cast<const YoloDetBufHeader*>(slots_.at(static_cast<std::size_t>(slot)).det_host);
}

std::uint32_t YoloPostproc::raw_count(int slot) const { return header(slot)->count; }
std::uint32_t YoloPostproc::overflow(int slot) const { return header(slot)->overflow; }

void YoloPostproc::checksums(int slot, std::uint32_t out[kYoloNumTensors]) const {
    const YoloDetBufHeader* h = header(slot);
    for (int i = 0; i < kYoloNumTensors; ++i) out[i] = h->text[i];
}

std::vector<DetectionKeyed> YoloPostproc::read_candidates(int slot) const {
    const YoloDetBufHeader* h = header(slot);
    const std::size_t n = std::min<std::size_t>(h->count, static_cast<std::size_t>(cfg_.max_dets));
    const auto* d = reinterpret_cast<const YoloDetGpu*>(slots_.at(static_cast<std::size_t>(slot)).det_host +
                                                        sizeof(YoloDetBufHeader));
    std::vector<DetectionKeyed> out;
    out.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        DetectionKeyed k;
        k.class_id = static_cast<int>(d[i].class_id);
        k.score = d[i].score;
        k.x1 = d[i].x1;
        k.y1 = d[i].y1;
        k.x2 = d[i].x2;
        k.y2 = d[i].y2;
        k.anchor_key = d[i].anchor_key;
        out.push_back(k);
    }
    return out;
}

// 부록 B.10 R2/R3. 슬롯의 원시 출력 6개 전부에 START/END 를 건다.
// ImportHeap 이 아니거나 heap_sync = false 면 아무것도 하지 않는다.
void YoloPostproc::raw_cpu_sync_begin(int slot, yvo::DmaSyncDir dir) {
    if (!raw_heap_sync_active()) return;
    const Slot& s = slots_.at(static_cast<std::size_t>(slot));
    for (const Tensor& t : s.t) yvo::dma_sync_begin(t.fd, dir, /*strict=*/true);
}

void YoloPostproc::raw_cpu_sync_end(int slot, yvo::DmaSyncDir dir) {
    if (!raw_heap_sync_active()) return;
    const Slot& s = slots_.at(static_cast<std::size_t>(slot));
    for (const Tensor& t : s.t) yvo::dma_sync_end(t.fd, dir, /*strict=*/true);
}

int YoloPostproc::raw_dmabuf_fd(int slot, int tensor) const {
    if (tensor < 0 || tensor >= kYoloNumTensors) throw std::out_of_range("YoloPostproc: bad tensor index");
    const int fd = slots_.at(static_cast<std::size_t>(slot)).t[tensor].fd;
    if (fd < 0) throw std::logic_error("YoloPostproc: input_alloc 이 dma-buf 경로가 아니다 (ExportDmaBuf/ImportHeap)");
    return fd;
}

std::size_t YoloPostproc::raw_dmabuf_size(int slot, int tensor) const {
    if (tensor < 0 || tensor >= kYoloNumTensors) throw std::out_of_range("YoloPostproc: bad tensor index");
    return slots_.at(static_cast<std::size_t>(slot)).t[tensor].dmabuf_size;
}

std::size_t YoloPostproc::raw_size(int tensor) const {
    if (tensor < 0 || tensor >= kYoloNumTensors) throw std::out_of_range("YoloPostproc: bad tensor index");
    const YoloScaleSpec& s = layout_.s[tensor / 2];
    return (tensor % 2 == 0) ? s.cls_bytes : s.box_bytes;
}

std::uint8_t* YoloPostproc::raw_host_ptr(int slot, int tensor) const {
    if (tensor < 0 || tensor >= kYoloNumTensors) throw std::out_of_range("YoloPostproc: bad tensor index");
    return slots_.at(static_cast<std::size_t>(slot)).t[tensor].host;
}

VkBuffer YoloPostproc::det_buffer(int slot) const { return slots_.at(static_cast<std::size_t>(slot)).det; }

void YoloPostproc::destroy() {
    VkDevice dev = ctx_.device();
    if (dev == VK_NULL_HANDLE) return;
    vkDeviceWaitIdle(dev);   // 종료 시에만
    for (Slot& s : slots_) {
        if (s.fence) vkDestroyFence(dev, s.fence, nullptr);
        if (s.det_host) vkUnmapMemory(dev, s.det_mem);
        if (s.det) vkDestroyBuffer(dev, s.det, nullptr);
        if (s.det_mem) vkFreeMemory(dev, s.det_mem, nullptr);
        for (Tensor& t : s.t) {
            // ExportDmaBuf 모드의 버퍼·메모리·fd 는 DmaBufExporter 가 소유한다(여기서 풀지 않는다).
            if (t.mem != VK_NULL_HANDLE) {
                if (t.host) vkUnmapMemory(dev, t.mem);
                if (t.buf) vkDestroyBuffer(dev, t.buf, nullptr);
                vkFreeMemory(dev, t.mem, nullptr);
            }
            t = Tensor{};
        }
        s = Slot{};
    }
    slots_.clear();
    // 파괴 순서: import 캐시(VkBuffer/VkDeviceMemory + dup fd)를 먼저 비우고 그다음 힙 풀(mmap/fd)을 놓는다.
    import_cache_.reset();
    heap_.reset();
    exporter_.reset();
    if (query_pool_) vkDestroyQueryPool(dev, query_pool_, nullptr);
    if (cmd_pool_) vkDestroyCommandPool(dev, cmd_pool_, nullptr);   // 커맨드 버퍼도 함께 해제
    if (desc_pool_) vkDestroyDescriptorPool(dev, desc_pool_, nullptr);
    if (pipeline_sum_) vkDestroyPipeline(dev, pipeline_sum_, nullptr);
    if (pipeline_) vkDestroyPipeline(dev, pipeline_, nullptr);
    if (pipeline_layout_) vkDestroyPipelineLayout(dev, pipeline_layout_, nullptr);
    if (set_layout_) vkDestroyDescriptorSetLayout(dev, set_layout_, nullptr);
    query_pool_ = VK_NULL_HANDLE;
    cmd_pool_ = VK_NULL_HANDLE;
    desc_pool_ = VK_NULL_HANDLE;
    pipeline_sum_ = VK_NULL_HANDLE;
    pipeline_ = VK_NULL_HANDLE;
    pipeline_layout_ = VK_NULL_HANDLE;
    set_layout_ = VK_NULL_HANDLE;
}

} // namespace yvo::vk
