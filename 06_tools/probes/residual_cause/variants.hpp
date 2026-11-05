// One-factor variants of cpu_letterbox_nv12_bt709 (diagnostic only, not part of the build).
#pragma once
#include <cstddef>
#include <cstdint>
#include "core/frame_types.hpp"

namespace rc {
using yvo::LetterboxParams;
using u8 = std::uint8_t;

#define RC_ARGS const u8* y_plane, std::size_t y_pitch, const u8* uv_plane, std::size_t uv_pitch, int src_w, int src_h, \
                const LetterboxParams& lb, u8* dst

// Telescoping chain: ref -> local_params -> local_params_hoist -> nobranch_content (+ ref pad rows) -> split
void local_params(RC_ARGS);        // ref + lb fields copied to locals (no per-pixel reload)
void hoist_y(RC_ARGS);             // ref + y taps computed per row (scvtf of clamp limits stays per pixel)
void hoist_all(RC_ARGS);           // ref + y taps and clamp limits computed per row (lb reloads and branch stay)
void local_hoist(RC_ARGS);         // local_params + hoist_all (only the per-pixel pad test stays)
void rowsplit(RC_ARGS);            // ref with pad rows split off in source; per-pixel dx test stays
// Content rows only (rows [pad_y, pad_y + new_h)):
void nobranch_content(RC_ARGS);    // no pad test; compiler hoists freely
void nobranch_nohoist_content(RC_ARGS); // no pad test; y taps and clamp limits forced per pixel by asm barriers
void rowbuf_content(RC_ARGS);      // split arithmetic through one row of float scratch (no 2.76 MB buffers)
void split_nohoist_resize(const u8* y_plane, std::size_t y_pitch, const u8* uv_plane, std::size_t uv_pitch, int src_w,
                          int src_h, const LetterboxParams& lb, float* out_y, float* out_u, float* out_v);
// Separate TUs:
void ref_notrap(RC_ARGS);          // ref source copy built with -fno-trapping-math
void ref_copy0(RC_ARGS);           // ref source copy, same flags, function aligned to 64
void ref_copy1(RC_ARGS);           // ... + 1 leading nop
void ref_copy2(RC_ARGS);           // ... + 2 leading nops
void ref_copy3(RC_ARGS);           // ... + 3 leading nops
} // namespace rc
