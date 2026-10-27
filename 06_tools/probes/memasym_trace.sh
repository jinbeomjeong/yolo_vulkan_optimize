#!/usr/bin/env bash
# Kernel-side evidence for `yvo_postproc_probe --probe memasym` (07_results/memory_asymmetry):
#   (a) the page protection the kernel puts into the PTEs of each CPU mapping
#       - V3D BO      : drm_gem_shmem_mmap -> fault -> vmf_insert_pfn_prot(vma, addr, pfn, prot = vma->vm_page_prot)
#       - dma-heap cma: cma_heap_mmap      -> fault -> vmf_insert_pfn_prot(...)
#       - dma-heap system: system_heap_mmap -> remap_pfn_range(vma, addr, pfn, size, prot = vma->vm_page_prot)
#     The prot argument is read straight from the register ($argN, no BTF on this kernel); arm64 AttrIndx = prot[4:2]
#     (MT_NORMAL 0, MT_NORMAL_TAGGED 1, MT_NORMAL_NC 2, MT_DEVICE_nGnRnE 3, MT_DEVICE_nGnRE 4 — asm/memory.h).
#   (b) which CPU cache maintenance runs around V3D submit/wait and around DMA_BUF_IOCTL_SYNC:
#       arch_sync_dma_for_{device,cpu} with size/dir (+ kernel stack), dma_buf_*_cpu_access, v3d_submit_csd_ioctl /
#       v3d_csd_job_run / v3d_cache_clean_job_run / v3d_{invalidate,clean}_caches / v3d_wait_bo_ioctl.
#       The dcache_*_poc assembly routines refuse kprobes on this kernel ("Failed to register probe event"), so the
#       operation is inferred from dir (arch/arm64/mm/dma-mapping.c, rpi-6.18.y): for_device BIDIRECTIONAL(0) ->
#       dcache_clean_inval_poc, TO_DEVICE(1) -> dcache_clean_poc, FROM_DEVICE(2) -> dcache_inval_poc;
#       for_cpu TO_DEVICE -> nothing, otherwise dcache_inval_poc.
#
# Usage (root is needed for tracefs; the probe itself runs as the invoking user):
#   sudo bash 06_tools/probes/memasym_trace.sh <out_dir> [extra probe args...]
#   default probe args: --parts attr,coherence --coh-trials 1 --rounds 1 --repeats 2 --cpu 3
# Output in <out_dir>: probe.txt (probe log with [mark] mono= lines), trace.txt.gz (tracefs, trace_clock mono),
#   kprobe_events.txt (the definitions used), then 06_tools/probes/memasym_trace_summary.py turns them into summary.txt.
# Everything is registered in a private tracefs instance and kprobe group and removed on exit (trap), whatever happens.
set -euo pipefail

[[ $EUID -eq 0 ]] || { echo "run as root (sudo bash $0 <out_dir>)" >&2; exit 1; }
RUN_USER="${SUDO_USER:-}"
[[ -n "$RUN_USER" && "$RUN_USER" != root ]] || { echo "run through sudo from the user account (SUDO_USER unset)" >&2; exit 1; }
OUT="${1:?usage: $0 <out_dir> [probe args...]}"; shift
PROJ="$(cd "$(dirname "$0")/../.." && pwd)"
BIN="$PROJ/build/bin/yvo_postproc_probe"
[[ -x "$BIN" ]] || { echo "not found: $BIN (build first)" >&2; exit 1; }
PROBE_ARGS=("$@")
[[ ${#PROBE_ARGS[@]} -gt 0 ]] || PROBE_ARGS=(--parts attr,coherence --coh-trials 1 --rounds 1 --repeats 2 --cpu 3)

T=/sys/kernel/tracing
INST="$T/instances/yvo_memasym"
GRP=yvo_memasym
[[ -d "$INST" ]] && { echo "tracefs instance already exists: $INST (a previous run did not clean up?)" >&2; exit 1; }
grep -q "^p:$GRP/" "$T/kprobe_events" && { echo "kprobe group $GRP already registered" >&2; exit 1; }

# name|definition (kprobe_events syntax without the "p:GRP/name" prefix)
PROBES=(
  "vmf_insert_pfn_prot|vmf_insert_pfn_prot addr=\$arg2:x64 pfn=\$arg3:x64 prot=\$arg4:x64"
  "remap_pfn_range|remap_pfn_range addr=\$arg2:x64 pfn=\$arg3:x64 size=\$arg4:x64 prot=\$arg5:x64"
  "drm_gem_shmem_mmap|drm_shmem_helper:drm_gem_shmem_mmap"
  "drm_gem_shmem_fault|drm_shmem_helper:drm_gem_shmem_fault"
  "cma_heap_vm_fault|cma_heap_vm_fault"
  "system_heap_mmap|system_heap_mmap"
  "arch_sync_dma_for_device|arch_sync_dma_for_device paddr=\$arg1:x64 size=\$arg2:x64 dir=\$arg3:s32"
  "arch_sync_dma_for_cpu|arch_sync_dma_for_cpu paddr=\$arg1:x64 size=\$arg2:x64 dir=\$arg3:s32"
  "dma_buf_begin_cpu_access|dma_buf_begin_cpu_access dir=\$arg2:s32"
  "dma_buf_end_cpu_access|dma_buf_end_cpu_access dir=\$arg2:s32"
  "dma_buf_map_attachment|dma_buf_map_attachment dir=\$arg2:s32"
  "v3d_submit_csd_ioctl|v3d:v3d_submit_csd_ioctl"
  "v3d_csd_job_run|v3d:v3d_csd_job_run"
  "v3d_cache_clean_job_run|v3d:v3d_cache_clean_job_run"
  "v3d_invalidate_caches|v3d:v3d_invalidate_caches"
  "v3d_clean_caches|v3d:v3d_clean_caches"
  "v3d_wait_bo_ioctl|v3d:v3d_wait_bo_ioctl"
)
STACK_EVENTS=(arch_sync_dma_for_device arch_sync_dma_for_cpu)

cleanup() {
  set +e
  if [[ -d "$INST" ]]; then
    echo 0 > "$INST/tracing_on"
    for e in "${STACK_EVENTS[@]}"; do
      [[ -f "$INST/events/$GRP/$e/trigger" ]] && echo '!stacktrace' > "$INST/events/$GRP/$e/trigger" 2>/dev/null
    done
    [[ -d "$INST/events/$GRP" ]] && echo 0 > "$INST/events/$GRP/enable"
    rmdir "$INST"
  fi
  for p in "${PROBES[@]}"; do
    n="${p%%|*}"
    grep -q "^p:$GRP/$n " "$T/kprobe_events" && echo "-:$GRP/$n" >> "$T/kprobe_events"
  done
  if grep -q "^p:$GRP/" "$T/kprobe_events"; then echo "cleanup: kprobes of $GRP still registered" >&2
  else echo "cleanup: tracefs instance and kprobe group $GRP removed"; fi
  [[ -d "$OUT" ]] && chown -R "$RUN_USER:$(id -gn "$RUN_USER")" "$OUT"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

mkdir -p "$OUT"
chown "$RUN_USER:$(id -gn "$RUN_USER")" "$OUT"   # the probe (run as the user) writes probe.txt here
: > "$OUT/kprobe_events.txt"
for p in "${PROBES[@]}"; do
  n="${p%%|*}"; def="${p#*|}"
  echo "p:$GRP/$n $def" >> "$T/kprobe_events"
  echo "p:$GRP/$n $def" >> "$OUT/kprobe_events.txt"
done
mkdir "$INST"
echo mono > "$INST/trace_clock"
echo 65536 > "$INST/buffer_size_kb"   # per CPU
echo 1 > "$INST/options/record-tgid" 2>/dev/null || true
for e in "${STACK_EVENTS[@]}"; do echo stacktrace > "$INST/events/$GRP/$e/trigger"; done
echo 1 > "$INST/events/$GRP/enable"
echo 1 > "$INST/tracing_on"
echo "tracing on: instance $INST, ${#PROBES[@]} kprobes (trace_clock mono)"

# The probe runs as the invoking user from the output folder (hailort.log lands there).
(cd "$OUT" && runuser -u "$RUN_USER" -- "$BIN" --probe memasym "${PROBE_ARGS[@]}" --out-dir "$OUT" --out-file probe.txt) \
  > "$OUT/probe_stdout.log" 2>&1 || echo "probe exited with $? (see probe_stdout.log)"

echo 0 > "$INST/tracing_on"
cat "$INST/trace" > "$OUT/trace.txt"
grep -h . "$INST/per_cpu"/cpu*/stats 2>/dev/null | grep -E "overrun|dropped" > "$OUT/trace_overrun.txt" || true
echo "trace: $(wc -l < "$OUT/trace.txt") lines -> $OUT/trace.txt"
python3 "$PROJ/06_tools/probes/memasym_trace_summary.py" "$OUT" > "$OUT/summary.txt" && echo "summary: $OUT/summary.txt"
gzip -9 -f "$OUT/trace.txt" && echo "trace compressed: $OUT/trace.txt.gz"
[[ -s "$OUT/hailort.log" ]] || rm -f "$OUT/hailort.log"   # 0-byte HailoRT log (NPU not used) carries nothing
