// yvosandletterbox: GStreamer element wrapping yvo::vk::SandLetterbox (research path, stage 2a)
//
//   sink: video/x-raw(memory:DMABuf), format=DMA_DRM, drm-format=NV12:0x0700000000000004   (only; no CPU fallback)
//   src : application/x-yvo-tensor, layout=nhcw, format=RGB8, width=640, height=640 (+ framerate if upstream has one)
//
// GstBaseTransform subclass, not in-place, not passthrough, QoS off.
// Upstream (decoder) side: propose_allocation adds GST_VIDEO_META_API_TYPE (v4l2slh265dec refuses DMA_DRM without it)
//   and a pool proposal with a small min ("decoder-pool-min").
// Output side: a private GstBufferPool subclass with exactly "slots" buffers. Buffer i wraps SandLetterbox slot i's
//   exported dma-buf fd (exported once in start(), owned by the element, wrapped with DONT_CLOSE, so the fd number per
//   slot is stable for the element's lifetime). prepare_output_buffer acquires from that pool and BLOCKS while every
//   slot is held downstream → ring-slot backpressure. The downstream ALLOCATION answer is ignored.
//   A slot is free for a new memory only when its previous GstMemory died (destroy-notify qdata, which also keeps the
//   Vulkan resources and fds alive); acquire/release accounting uses a slot tag on the GstBuffer itself. If downstream
//   keeps a slot memory beyond its buffer, the element waits 2 s and then errors — it never overwrites that slot.
// transform: adapter (drm_frame_from_buffer) → SandLetterbox::submit(frame, slot) → wait(slot) → tensor meta.
//   All Vulkan work runs on the streaming thread; the input buffer's keepalive ref is dropped at wait().
// Shutdown (stop()): pool flushed + deactivated, short drain (200 ms) → clear_import_cache (decoder fds) → the element
//   drops its share of the Vulkan resources. Resources (VulkanContext, SandLetterbox, slot fds) are co-owned by the pool,
//   which every outstanding slot buffer keeps alive: when all slots are back they are destroyed inside stop() in the
//   pipeline/main.cpp order (SandLetterbox → export fds → VulkanContext); otherwise when the last slot buffer is
//   released (appsink keeps its last pulled buffer until READY→NULL, i.e. after this element's stop()).
//   PAUSED→READY and FLUSH_START set the pool flushing first, so a streaming thread blocked on a slot is released.
//
// Properties
//   slots               (int, 2..32, default 16)      output ring size; set before READY→PAUSED
//   decoder-pool-min    (uint, 1..32, default 4)      min buffers proposed to the upstream pool
//   validation          (bool, default false)          VulkanContextOptions::enable_validation
//   frames              (uint64, read-only)            frames transformed since start()
//   gpu-ms-mean         (double, read-only)            mean GPU dispatch time
//   slots-in-use        (uint, read-only)              slot buffers acquired and not yet returned (per-buffer tag)
//   slots-detached      (uint, read-only)              slot memories alive without their slot buffer (moved/shared
//                                                      downstream); such a slot is not re-used until the memory dies
//   validation-errors   (uint64, read-only)            Vulkan debug-messenger error count (kept after stop())
//   validation-active   (bool, read-only)              validation layer on and counted (property/YVO_VK_VALIDATION/env)
#pragma once

typedef struct _GstPlugin GstPlugin;

namespace yvo::gst {

inline constexpr const char* kSandLetterboxElementName = "yvosandletterbox";

// Registers the element factory (plugin may be null: static registration). Returns false on failure.
bool register_sand_letterbox(GstPlugin* plugin);

} // namespace yvo::gst
