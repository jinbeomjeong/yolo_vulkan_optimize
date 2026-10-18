// BT.709 limited range YUV → gamma-encoded RGB 0~255. Included by sand_letterbox.comp (#include "bt709.glsl").
// (The display path no longer converts YUV on the GPU: the HVS does it during plane scanout.)
//
// Reference implementation: 01_src/ref/cpu_letterbox.hpp formula 5) (cpu_letterbox_nv12_bt709).
//   Y' = (Y - 16) * 255/219,  U' = (U - 128) * 255/224,  V' = (V - 128) * 255/224
//   R = Y' + 1.5748 V',  G = Y' - 0.187324 U' - 0.468124 V',  B = Y' + 1.8556 U'
//   clamp to 0~255, then round (add 0.5 and truncate)
// Every intermediate is precise (no fused multiply-add contraction): test_preproc and test_display_convert require a
// max error of 0 against the CPU reference. Do not reorder the operations.
#ifndef YVO_BT709_GLSL
#define YVO_BT709_GLSL

uint to_u8(float v) {
    precise float c = clamp(v, 0.0, 255.0) + 0.5;
    return uint(c);
}

uvec3 bt709_to_rgb(float Y, float U, float V) {
    precise float yf = (Y - 16.0) * (255.0 / 219.0);
    precise float uf = (U - 128.0) * (255.0 / 224.0);
    precise float vf = (V - 128.0) * (255.0 / 224.0);
    precise float R = yf + 1.5748 * vf;
    precise float G = yf - 0.187324 * uf - 0.468124 * vf;
    precise float B = yf + 1.8556 * uf;
    return uvec3(to_u8(R), to_u8(G), to_u8(B));
}

#endif // YVO_BT709_GLSL
