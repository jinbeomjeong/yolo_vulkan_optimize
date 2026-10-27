# CPU↔GPU 메모리 접근 비대칭 — 미확인 4항목 (2026-10-27)

V3D 메모리를 CPU가 읽는 비용은 [`gpu_postprocess/runs/stride_read`](../gpu_postprocess/runs/stride_read/summary.md)가 쟀다
(1바이트 load 12.82 ms, `memcpy` 0.784 ms, 1,228,800 B). 그 결론에는 확인하지 않은 전제가 넷 있었다.

1. GPU 쪽은 같은 메모리를 얼마나 빨리 읽고 쓰는가(할당처별).
2. V3D 매핑의 페이지 속성이 정말 비캐시(Normal-NC)인가 — 소스 근거만 있었다(`v3d_bo.c`의 `map_wc` → `pgprot_writecombine`).
3. V3D 방향의 캐시 일관성 기제 — [`dmaheap_unify`](../dmaheap_unify/summary.md) 5단계 V3에서 "미확정"으로 남았다.
4. "write-combine 매핑은 CPU 쓰기가 싸다"는 일반론([`08_learn/06_zerocopy_dmabuf.md`](../../08_learn/06_zerocopy_dmabuf.md) §6.3)이 이 보드에서 맞는가.

작업 승인: 2026-10-27(사용자). 이 문서는 측정 결과만 적고 다른 문서는 고치지 않았다.

## 1. 측정 조건

- 날짜: 2026-10-27 13:29~13:32(KST, 정식 15회), 커널 추적 13:32
- 하드웨어: Raspberry Pi Compute Module 5 Rev 1.0, kernel 6.18.50+rpt-rpi-2712, Mesa 26.2.2(`mesa-vulkan-drivers 26.2.2-1~bpo13+0~rpt1`)
  - V3D 1150 MHz(`v3d_freq=1150`), ARM 2400 MHz, governor ondemand, CPU 측정 스레드는 코어 3에 고정
  - NPU·표시 미사용. CMA 64 MiB(여유 54 MiB)
- 코드: 작업 트리(HEAD `fa0e2b5` + 미커밋 변경). 바이너리 md5 `a80774324838fa633d633fc194598e51`([`env_bench.txt`](env_bench.txt))
  - 프로브 `yvo_postproc_probe --probe memasym`(`01_src/tools/vulkan/postproc_probe.cpp` `run_memasym`)
  - 셰이더 `02_shaders/diag/mem_bw.comp`(읽기 합산·쓰기·복사 × uint/uvec4, 6변종)
  - 실행기 `06_tools/probes/memasym_run.sh`, 집계 `06_tools/probes/memasym_aggregate.py`,
    커널 추적 `06_tools/probes/memasym_trace.sh` + `memasym_trace_summary.py`
- 반복: 설정 5종 × 독립 실행 3회, 실행이 바깥 루프·설정이 안쪽 루프. 실행마다 셀당 30표본(3 rounds × 10 repeats), 할당처 순서를 표본마다 돌린다.
  표의 값은 **실행별 중앙값의 3회 중앙값**이고, 일관성 비율만 3회 × 5시행의 라인 표본을 모두 합쳤다.
- 냉각: 실행 전마다 SoC ≤ 70 °C(프로젝트 기준 ≤ 75 °C)와 `get_throttled` 현재 비트(0~3) = 0을 확인했다.
  실행 시작 68.1~70.3 °C, 종료 최고 73.6 °C. `throttled=0x80000`은 과거 기록 비트다(각 `raw/<설정>_run<N>/run_env.txt`).
- 결과 검증: 15회 모두 `[RESULT] memasym PASS`. GPU 읽기는 표본마다 워드 합(uint32)을, GPU 쓰기는 쓴 워드 전부를 CPU에서,
  CPU 쓰기는 다시 읽은 체크섬을 대조했다. 불일치 0, 워크그룹 수 불일치 0([`evidence/medians.txt`](evidence/medians.txt) 머리).
  검증 레이어를 켠 스모크 실행(전 항목)에서 errors 0 / warnings 0.
- 할당처
  - **v3d**: V3D 자기 메모리(메모리 타입 0x7 하나, `HOST_CACHED` 없음), `vkMapMemory` 상시 매핑
  - **heap**: 캐시 dma-heap `/dev/dma_heap/system`을 Vulkan에 import(장치 attachment 있음). CPU 접근은 `DMA_BUF_IOCTL_SYNC`로 감싼다
  - **cma**: `/dev/dma_heap/linux,cma`를 같은 방식으로 import
  - **malloc**: `aligned_alloc(16384)` — CPU 항목의 기준

## 2. 결과

### 2.1 GPU 읽기·쓰기 대역폭 (항목 1, GB/s)

`raw / net` 두 값이다. raw는 타임스탬프 두 개 사이에 디스패치 K개(K = ⌈48 MiB / 크기⌉: 1.2 MB는 41, 16 MiB는 3)를
배리어로 이어 붙인 구간이고, net은 1워크그룹짜리 빈 디스패치로 잰 **디스패치당 고정비 31.3~35.2 µs**를 뺀 값이다.
매핑 속성은 §2.2의 측정값이다.

| 크기 | 커널 (디스패치 형태) | v3d (Normal-NC) | heap (Normal WB) | cma (Normal WB) |
|---|---|---:|---:|---:|
| 1,228,800 B | 읽기 uint (256×3) | 3.934 / 4.379 | 3.933 / 4.377 | 3.927 / 4.368 |
| | 읽기 uvec4 (256×3) | 10.641 / 14.660 | 10.635 / 14.648 | 10.511 / 14.422 |
| | 쓰기 uint (256×3) | 5.520 / 6.436 | 5.523 / 6.441 | 4.534 / 5.134 |
| | 쓰기 uvec4 (128×2) | 11.386 / 16.032 | 11.394 / 16.068 | 8.220 / 10.395 |
| 16,777,216 B | 읽기 uint (256×3) | 4.294 / 4.329 | 4.292 / 4.328 | 4.109 / 4.143 |
| | 읽기 uvec4 (256×3) | **14.804** / 15.234 | 14.780 / 15.221 | 14.570 / 14.986 |
| | 쓰기 uint (256×3) | 3.966 / 3.996 | 3.966 / 3.996 | 3.388 / 3.411 |
| | 쓰기 uvec4 (128×2) | **14.485** / 14.940 | 14.497 / 14.953 | 13.857 / 14.274 |

- 3회 실행별 중앙값의 범위는 대부분 0.3% 안이다. 예외는 16 MiB 쓰기 uint의 128×2 형태(heap 3.599~3.795)이며 표에는 256×3 형태를 실었다.
- 디스패치 형태 = `local_size_x × 워크그룹 수`. 같은 버퍼에서도 형태에 따라 0.04~14.7 GB/s로 바뀐다.
  예비 스윕(표본 2개, v3d만, [`evidence/bw_shape_sweep.tsv`](evidence/bw_shape_sweep.tsv))에서 **워크그룹을 적게(1~4개) 크게(256) 잡은
  grid-stride 루프가 가장 빨랐고**, 읽기 최적 256×3과 쓰기 최적 128×2를 정식 설정 둘로 돌렸다. 두 설정의 전체 표는 `evidence/medians.txt` §1.
- 할당처 비: **v3d와 heap은 모든 셀에서 0.3% 안으로 같다.** cma는 읽기에서 1~4%, 쓰기에서 4~31% 느리다(원인은 가르지 않았다).

### 2.2 페이지 속성 (항목 2)

**(a) 커널이 PTE에 넣는 prot — tracefs kprobe** ([`evidence/trace/summary.txt`](evidence/trace/summary.txt) §1)

이 커널에는 BTF(`/sys/kernel/btf/vmlinux`)가 없다. 그래서 구조체 오프셋을 쓰지 않고, prot를 **값으로 받는 함수의 레지스터 인자**를 읽었다.
V3D BO와 cma는 fault 때 `vmf_insert_pfn(vma, addr, pfn)` → `vmf_insert_pfn_prot(..., prot = vma->vm_page_prot)`의 4번째 인자를,
system heap은 mmap 때 `remap_pfn_range(..., prot = vma->vm_page_prot)`의 5번째 인자를 읽었다(소스: raspberrypi/linux rpi-6.18.y
`drm_gem_shmem_helper.c`, `cma_heap.c`, `system_heap.c`). `insert_pfn`은 `pfn_pte(pfn, prot)`로 PTE를 만들므로 AttrIndx는 이 값 그대로 들어간다
(소스 근거. 최종 PTE를 페이지 테이블에서 읽지는 못했다 — `CONFIG_PTDUMP_DEBUGFS` 꺼짐).

| 매핑 | 경로 | 페이지 수 | prot | AttrIndx [4:2] |
|---|---|---:|---|---|
| v3d (1.2 MB 버퍼, 일관성 복사 대상 256 KiB) | `drm_gem_shmem_fault` → `vmf_insert_pfn_prot` | 75/75, 16/16 | `0x68000000000fcb` | **2 = `MT_NORMAL_NC`** |
| heap (system) | `system_heap_mmap` → `remap_pfn_range` | 75/75, 16/16 | `0x68000000000fc3` | 0 = `MT_NORMAL` |
| cma | `cma_heap_vm_fault` → `vmf_insert_pfn_prot` | 75/75, 16/16 | `0x68000000000fc3` | 0 = `MT_NORMAL` |

두 prot 값은 AttrIndx 비트만 다르다. 인덱스 뜻은 `arch/arm64/include/asm/memory.h`(`MT_NORMAL 0, MT_NORMAL_TAGGED 1, MT_NORMAL_NC 2, MT_DEVICE_nGnRnE 3, MT_DEVICE_nGnRE 4`).
MAIR_EL1의 실제 값은 EL0에서 읽을 수 없어서 (b)로 동작을 확인했다.

**(b) 동작 판별** (`raw/attr_run{1,2,3}`)

| 할당처 | 같은 64 B 라인 의존 load (ns/load) | 비정렬 `ldr x`(+1) · 라인 걸침 `ldr x`(+61) · `ldr w`(+2) · `dc zva` |
|---|---:|---|
| v3d | **121.36** | 모두 정상, 값 일치 |
| heap | 2.08 | 모두 정상, 값 일치 |
| cma | 2.08 | 모두 정상, 값 일치 |
| malloc | 2.08 | 모두 정상, 값 일치 |
| (기준) malloc 16 MiB 무작위 체인 | 108.86 | — |

- 의존 load 체인 `i = p[i]`(`p[0] = 0`, 200,000회)이라 매번 같은 라인을 읽는다. 캐시되면 L1 수준(2.08 ns ≈ 5사이클), 안 되면 DRAM 수준이다.
  v3d는 무작위 DRAM 체인(108.9 ns)과 같은 자릿수다 → **CPU 캐시를 쓰지 않는다.**
- Device 메모리라면 비정렬 접근과 `dc zva`가 정렬 폴트(SIGBUS)를 낸다. 넷 다 정상이다 → **Normal 메모리다.**
- 둘을 합치면 V3D 매핑은 **Normal Non-cacheable**(Linux의 "writecombine")이고, (a)의 AttrIndx 2와 맞는다.

### 2.3 V3D 방향 캐시 일관성 (항목 3)

캐시 dma-heap 버퍼에서 `DMA_BUF_IOCTL_SYNC`를 **생략한 셀(off)** 과 지킨 셀(on)을 비교했다. Vulkan 쪽은 두 셀 모두 올바르다
(import 버퍼의 queue family acquire/release, 디스패치 사이 배리어). 64 B 라인마다 새 값 / 옛 값(stale) / 섞임으로 분류했다.
대조(sync를 지켜 쓴 A를 GPU가 그대로 읽는가)는 전 시행 불일치 0이다.

| 방향 | 크기 | heap off: stale | cma off: stale | on (heap·cma) |
|---|---|---:|---:|---:|
| CPU 쓰기 → GPU 읽기, 평범한 `memcpy` (`cpu2gpu_plain`) | 256 KiB | 0.0699 | 0.0781 | 0 |
| | 8 MiB | 0.0025 | 0.0018 | 0 |
| CPU가 먼저 읽어 라인을 상주시킨 뒤 `memcpy` (`cpu2gpu_resident`) | 256 KiB | **0.9635** | **0.9665** | 0 |
| | 8 MiB | 0.1594 | 0.1517 | 0 |
| CPU가 읽어 캐시를 채움 → GPU 쓰기 → CPU 읽기 (`gpu2cpu`) | 256 KiB | **0.9739** | **0.9728** | 0 |
| | 8 MiB | 0.1566 | 0.1422 | 0 |

- 라인 표본: 256 KiB는 셀당 61,440개(4,096 라인 × 5시행 × 3회), 8 MiB는 1,966,080개. 섞인 라인은 전 셀 0이다.
- 원문: `raw/coh_run{1,2,3}/stdout.log`, 합계는 `evidence/medians.txt` §3.

**커널 추적** (`evidence/trace/`, attr + 256 KiB 일관성 1시행, 시스템 전역 kprobe 17개, trace_clock mono로 프로브의 `[mark]`와 맞춤, overrun 0)

- **V3D 제출·대기 구간에는 CPU 캐시 유지 호출이 0건이다.** 일관성 12개 셀의 GPU 구간(`gpu-copy`/`gpu-write-B` → `gpu-done`) 어디에도
  `arch_sync_dma_for_{device,cpu}`가 없다. 있는 것은 `v3d_submit_csd_ioctl`·`v3d_wait_bo_ioctl`과 kworker의 `v3d_csd_job_run` →
  `v3d_invalidate_caches`, `v3d_cache_clean_job_run` → `v3d_clean_caches`다(GPU 쪽 L2T·슬라이스 캐시 무효화·정리).
- **CPU 캐시 유지는 sync ioctl에서만, V3D attachment를 통해 일어난다.** 스택:
  `dma_buf_ioctl` → `dma_buf_{begin,end}_cpu_access` → `{system,cma}_heap_dma_buf_{begin,end}_cpu_access` → `__dma_sync_sg_for_{cpu,device}` → `arch_sync_dma_for_{cpu,device}`.
  system heap은 페이지마다(272 KiB = 17회), cma는 연속 영역 1회다.
- 방향별 동작(인자 `dir`과 `arch/arm64/mm/dma-mapping.c` rpi-6.18.y로 판정): START|WRITE는 아무것도 하지 않고, END|WRITE는 `dcache_clean_poc`,
  START|READ와 END|READ는 둘 다 `dcache_inval_poc`다(이 커널의 `arch_sync_dma_for_device`는 FROM_DEVICE를 inval로 처리한다).
  `dcache_*_poc` 어셈블리 함수는 kprobe 등록이 거부돼("Failed to register probe event") 호출자에서 판정했다.
- V3D 노드(`/proc/device-tree/axi/v3d@2000000`)에 `dma-coherent`가 없고 IOMMU 그룹도 없다.

### 2.4 CPU 쓰기 비용 (항목 4, 1,228,800 B, ms)

측정 구간은 `dsb sy`로 닫는다(저장 완료·write-combine 버퍼 배출). heap 행은 START|WRITE + END|WRITE(= `dcache_clean_poc`)를 포함한다.

| 할당처 | `memset`(0 아닌 값) | `memset`(0) | `memcpy`(malloc 원본) | 1바이트 순차 저장 | sync 몫 |
|---|---:|---:|---:|---:|---:|
| **v3d** | **0.098** (12.54 GB/s) | 0.098 | 0.203 (6.05) | **0.514** (2.39) | — |
| heap (sync 포함) | 0.157 (7.83) | 0.157 | 0.251 (4.90) | 0.533 (2.31) | 0.021~0.022 |
| malloc | 0.119 (10.33) | 0.121 | 0.206 (5.97) | 0.512 (2.40) | — |

- 1바이트 저장 루프는 `strb w2, [x0], #1` 스칼라 루프로 컴파일됐다(objdump, 벡터 명령 0).
- 3회 범위: v3d `memcpy` 0.196~0.259, v3d 1바이트 0.512~0.580(각 1회가 튐), 나머지 3% 안.
- malloc 행은 낙관적 기준이다(목적지가 표본 사이에 L2/L3에 남는다).

### 2.5 비대칭 요약 (같은 V3D 메모리, 1,228,800 B)

| 접근 | 시간 | 대역폭 | 출처 |
|---|---:|---:|---|
| CPU 1바이트 읽기 | 12.820 ms | 0.096 GB/s | stride_read |
| CPU `memcpy` 읽기 | 0.784 ms | 1.567 GB/s | stride_read |
| CPU 1바이트 쓰기 | 0.514 ms | 2.39 GB/s | 이 실험 |
| CPU `memcpy` 쓰기 | 0.203 ms | 6.05 GB/s | 이 실험 |
| CPU `memset` | 0.098 ms | 12.54 GB/s | 이 실험 |
| GPU uvec4 읽기 (net) | — | 14.66 GB/s | 이 실험 |
| GPU uvec4 쓰기 (net) | — | 16.03 GB/s | 이 실험 |

- CPU 쓰기 / 읽기: 1바이트 **24.9배**, `memcpy` 3.9배 쓰기가 빠르다.
- 같은 메모리를 GPU는 CPU `memcpy`보다 약 9배(net 14.66 / 1.567), CPU 바이트 읽기보다 약 150배 빨리 읽는다.

## 3. 결론

관찰과 추정을 나눠 적는다.

1. **V3D 매핑은 Normal Non-cacheable이다(관찰).** 커널이 넣는 prot의 AttrIndx가 2(`MT_NORMAL_NC`)이고, 동작도 맞다 —
   같은 라인 반복 읽기가 DRAM 수준(121 ns)이고 비정렬 접근이 정상이다. heap·cma 매핑은 `MT_NORMAL`(캐시)이다.
   stride_read 결론 3의 "(**해석**, 매핑 PTE 속성은 보지 않았다)"는 이제 측정으로 뒷받침된다.
2. **GPU 대역폭은 CPU 쪽 매핑 속성과 무관하다(관찰).** v3d(NC)와 heap(WB)은 모든 셀에서 0.3% 안으로 같다. 페이지 속성은 CPU PTE에만 있고
   V3D는 자기 MMU로 같은 DRAM을 읽는다. dmaheap_unify의 실행 시간 비 heap/plain 0.9994와 맞고, 이번에는 대역폭으로 확인했다.
   GPU 상한은 uvec4 읽기 약 14.8 GB/s, 쓰기 약 14.5 GB/s(16 MiB)다. uint 접근은 읽기 4.3 GB/s로 넓은 load가 3.4배 빠르다.
3. **비대칭은 CPU 읽기 하나에 몰려 있다(관찰).** 같은 V3D 메모리에서 CPU 쓰기는 캐시 메모리만큼 싸고(`memset` 0.098 ms로 malloc 0.119보다 빠르다,
   1바이트 쓰기 0.514 ms로 malloc과 같다), CPU 읽기만 비싸다(1바이트 12.8 ms, 쓰기의 25배). §6.3의 "write-combine: 읽기 느림, 쓰기 빠름"이
   이 보드에서 성립한다. 1바이트 쓰기까지 싼 것은 Normal-NC가 쓰기 모으기(gathering)를 허용해서라고 본다(**추정**, 버스 트랜잭션은 보지 않았다).
4. **V3D는 CPU 캐시를 보지 않는다(관찰).** sync를 생략하면 캐시에 상주한 라인은 양방향 모두 옛 값이 보인다(256 KiB에서 96~97%).
   V3D 제출·대기 경로에는 CPU 캐시 유지가 없고, 일관성은 **`DMA_BUF_IOCTL_SYNC`가 V3D attachment를 통해 실행하는 `dcache_clean_poc`/`dcache_inval_poc`에서만** 나온다.
   sync를 지키면 모든 셀에서 0이다. dmaheap_unify가 미확정으로 남긴 두 가설 중 "V3D가 스누핑한다"는 기각되고, sync가 V3D 방향에 **필요함이 실증됐다.**
5. **V3에서 실패가 재현되지 않은 이유(추정).** 이번 `cpu2gpu_plain`처럼 버퍼 전체를 평범하게 덮어쓰면 옛 값이 7%(256 KiB)·0.2%(8 MiB)만 남는다.
   A76의 쓰기 스트리밍(연속 전체 라인 쓰기를 캐시에 할당하지 않음)으로 대부분이 메모리로 바로 가기 때문으로 본다(PMU로 확인하지 않았다).
   V3의 off 셀은 여기에 더해 Hailo 드라이버의 프레임별 `dcache_clean_poc`(V3에서 perf로 확인)를 거쳤다. 두 효과가 겹쳐 오염이 보이지 않았을 것이다.

## 4. 주의

- GPU 대역폭은 디스패치 형태에 크게 좌우된다. 표는 이 셰이더(grid-stride, 워크그룹 내 공유 메모리 합산)와 두 형태에서 얻은 최댓값이지
  V3D의 이론 상한이 아니다. 형태 스윕은 표본 2개짜리 예비 측정이다.
- 디스패치당 고정비 31~35 µs는 1워크그룹 디스패치로 잰 값이다. 추적에서 제출마다 `v3d_csd_job_run`과 `v3d_cache_clean_job_run`이 하나씩 보이므로
  디스패치마다 CSD 잡과 캐시 정리 잡이 하나씩 붙는 비용으로 본다(**추정**, 한 커맨드 버퍼 안의 디스패치 K개가 잡 K개가 되는지는 추적하지 않았다).
- cma가 GPU 쓰기에서 느린 원인(물리 배치, 연속 영역)은 가르지 않았다. CPU 쪽(같은 라인 지연)은 heap과 같다.
- 8 MiB `resident`/`gpu2cpu`의 stale 15~16%(약 1.3 MB)는 L2 512 KiB + L3 2 MiB보다 작다. L3 교체 정책이나 다른 코어의 사용 때문일 수 있다(확인하지 않았다).
- 커널 추적은 1회(256 KiB, 1시행)다. 같은 경로가 8 MiB에서도 같은지는 정식 15회의 결과(on 셀 전부 0)로만 간접 확인했다.
- 실행 디렉터리의 `hailort.log`는 0 B(NPU 미사용)라 지웠다. 측정 콘솔 전문은 `11_logs/2026-10-27_memasym_run_console.log`(git 밖)에 있다.
- 측정 위임: 프로젝트 `vulkan-expert`가 세션에 로드되지 않아 전역 `cpp-expert`가 `vulkan-expert.md` 규약을 따라 구현·측정했다. 본문은 그 보고를 메인 에이전트가 원자료와 대조해 저장했다.
- 결정: 없음 — 이 결과로 코드 경로·기본값을 바꾸지 않았다. `08_learn`·논문 반영은 사용자 확인 대기 (메인 에이전트 기록, 2026-10-27).
