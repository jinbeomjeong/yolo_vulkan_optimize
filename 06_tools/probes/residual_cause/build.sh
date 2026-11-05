#!/usr/bin/env bash
# Builds the scratch probe with the same compiler flags as yvo_preproc_bench (-g does not change code generation).
set -euo pipefail
ROOT=/home/pi/workspace/yolo_vulkan_optimize
D=$ROOT/11_logs/2026-11-06_residual_cause
S=$D/src
O=$D/bin
FL="-DYVO_DEFAULT_VIDEO=\"$ROOT/04_data/videos/video_h265.mp4\" -I$ROOT/01_src -I$S -isystem /usr/include/libdrm -O3 -DNDEBUG -std=c++20 -mcpu=cortex-a76 -Wall -Wextra -Wpedantic -Wshadow -Wnon-virtual-dtor -g"
mkdir -p "$O"
c++ $FL -c "$S/residual_probe.cpp" -o "$O/residual_probe.o"
c++ $FL -c "$S/variants.cpp" -o "$O/variants.o"
c++ $FL -fno-trapping-math -DRC_FN_NAME=ref_notrap -DRC_FN_ATTR='[[gnu::noinline]]' -DRC_ENTRY_PAD= -x c++ -c "$S/ref_copy.inc" -o "$O/ref_notrap.o"
for k in 0 1 2 3; do
  case $k in
    0) P= ;;
    1) P='asm volatile("nop");' ;;
    2) P='asm volatile("nop\nnop");' ;;
    3) P='asm volatile("nop\nnop\nnop");' ;;
  esac
  c++ $FL -falign-functions=64 "-DRC_FN_NAME=ref_copy$k" "-DRC_FN_ATTR=[[gnu::noinline]]" "-DRC_ENTRY_PAD=$P" -x c++ -c "$S/ref_copy.inc" -o "$O/ref_copy$k.o"
done
c++ -O3 "$O"/residual_probe.o "$O"/variants.o "$O"/ref_notrap.o "$O"/ref_copy[0-3].o -o "$O/residual_probe" \
  "$ROOT/build/01_src/decode/libyvo_decoder.a" -lavformat -lavcodec -lavutil -ldrm
echo built
