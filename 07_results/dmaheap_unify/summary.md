# 캐시 매핑 dma-heap 검토 — 1~5단계 결과

- 측정일: 2026-09-20
- 환경: Raspberry Pi CM5 (BCM2712, Cortex-A76 ×4, **PAGESIZE 16384**), **V3D 7.1.10.2 / V3DV Mesa 26.2.2**,
  Hailo-8 **PCIe 8.0 GT/s x1 (Gen3)**, HailoRT 4.24.0, kernel 6.18.50, g++ 14.2 Release
- 계획: `~/.claude/plans/jaunty-dazzling-quasar.md` **부록 B**. 1단계 상세는 [REPORT.md](REPORT.md)
- 목적: 사용자가 정한 두 가지 — ① **설계 제약 제거**("GPU가 쓴 버퍼를 CPU로 읽지 마라"를 근거로 닫는다)
  ② **메모리 관리 일원화**. **"오늘 더 빠르게"가 아니다**(§결론 참조)

## 결과

| 단계 | 무엇을 물었나 | 실측 | 판정 |
|---|---|---|---|
| **1 (P0)** | V3DV가 dma-heap fd를 import 하는가 | `memoryTypeBits = 0x1` 6/6, `vkAllocateMemory` SUCCESS 6/6, 왕복 일치 6/6. `requires/prefersDedicatedAllocation = 0` 6/6 | **조건부 합격** (`range = size−256` 6/6) |
| **2** | 일원화 | `common/dma_heap_alloc.hpp`(헤더 전용·순수 POSIX) 신규. `DmaBufReadMap` 위임, `HeapOutPool` 어댑터화. **링크 그래프 변화 0변** | **완료** |
| **3** | −256 B 함정 | 헤드룸 질의 **256 B**(크기 무관 상수). `conv71_178` 409,600 → **425,984 B**, range **425,728 ≥ 409,600**. **6/6 `range ≥ hw`** | **S4 통과** |
| **4** | **GPU가 캐시된 heap 버퍼를 읽어도 빠른가** | **heap/plain = 0.9994** (3라운드 인터리브, 300프레임/라운드). 독립 4회 전부 밴드 안 | **밴드 A: 손해 없음** |
| **4 (G5)** | 정확성 | plain·heap 둘 다 후보 115/115, 점수 비트 불일치 **0**, 좌표 최대오차 **0.000122 px**. **두 경로 후보 집합이 바이트 단위 동일** | **S2 해당 없음** |
| **5 (V1)** | 3자 체크섬 대조 | 120건(20프레임 × 6텐서) **불일치 0** | **PASS** |
| **5 (V2)** | 스테일 4셀 | sync on 4셀 전부 정확(A 25/25, B 11/11), sum_mismatch 0 | **PASS** |
| **5 (V3)** | **sync가 실제로 필요한가** | 3변종 전부 **실패 재현 못 함**. 단 perf로 **원인을 특정**했다(아래) | **on PASS, 필요성 미실증** |

### 4단계 상세 — B.2의 결정적 미지수가 닫혔다

| 라운드 | plain median | heap median | heap/plain |
|---|---:|---:|---:|
| 0 | 0.5623 ms | 0.5630 ms | 1.0013 |
| 1 | 0.5623 | 0.5618 | 0.9992 |
| 2 | 0.5620 | 0.5616 | 0.9994 |

독립 4회: **0.9974**(NPU 경합 중, 미채택) · **0.9988** · **0.9994** · **1.0003**(조정자 실행).
→ **GPU가 캐시된 dma-heap 버퍼를 읽어도 V3D 자기 메모리만큼 빠르다.**
절대값이 아니라 **같은 실행 안의 인터리브 비율**로 판정했고, 경합 중 값과 clean window 값이 거의 같아
**비율이 경합에 둔감하다는 설계 판단이 확인됐다.**

### CPU 읽기 — 3배 빠르다

`cpu_read_1.22MB`(같은 실행·인터리브 3라운드 × 10회, `DMA_BUF_IOCTL_SYNC` 포함):

| alloc | median | p90 | GB/s |
|---|---:|---:|---:|
| malloc (**낙관적 기준** ↓) | 0.189 ms | 0.271 | 6.47 |
| **heap** (캐시 매핑) | **0.262 ms** | 0.380 | 4.65 |
| plain (V3D `HOST_COHERENT`) | 0.789 ms | 1.074 | 1.55 |

- **heap/plain = 0.333배.** 판정에 쓰는 것은 이 둘의 비교이고 같은 실행·같은 방법이다
- ⚠ **malloc 행은 낙관적이다** — src·dst 둘 다 malloc이고 같은 버퍼를 반복 읽어 L2/SLC에 남는다
  (그래서 P2의 0.244 ms보다 빠른 0.189가 나온다)
- **이전 기록의 불일치가 해소됐다**: `gpu_postprocess`의 P2가 heap 0.396 ms, 2단계가 0.292 ms였는데
  원인은 **측정 조건**이다 — P2는 HailoRT 세션이 살아 있는 상태(NPU 경합)에서 쟀다.
  유휴 상태의 실제 값은 **0.26~0.32 ms** 구간이다

### 5단계 V3 상세 — 실패를 재현하지 못했고, 그 이유를 찾았다

계획서 시나리오(전체 `0xA5` + `END|WRITE` 생략)가 재현되지 않아 **변종 둘을 추가**했다:

| 변종 | 방법 | off 셀 diff | 판정 |
|---|---|---:|---|
| ① full-dirty | 전체 `memset(0xA5)`, `END｜WRITE` 생략 | **0** | 재현 실패 |
| ② line-dirty | 창 4096 B에서 **64 B 라인마다 1바이트만** 쓴다(부분 라인 → dirty 강제) | **0** | 재현 실패 |
| ③ **stale-read** | 프레임 B로 창을 3회 읽어 캐시를 채운 뒤 DMA가 **다른** 프레임 A를 쓴다 → `START｜READ` 없이 읽는다 | **0** (A·B가 실제로 5,372 B 다름) | 재현 실패 |

**원인을 `perf record -g`로 특정했다** (`evidence/stage5_v3_perf_cachemaint.txt`):
```
system_heap_map_dma_buf   → dma_map_sgtable → __dma_map_sg_attrs → __pi_dcache_clean_inval_poc
system_heap_unmap_dma_buf → dma_unmap_sg_attrs → dma_direct_unmap_sg → __pi_dcache_inval_poc
hailo_vdma_buffer_map     → __dma_sync_single_for_device → __pi_dcache_clean_poc
```
→ **Hailo 드라이버가 장치 쪽에서 스트리밍 DMA API로 캐시 관리를 한다.** 디바이스 트리에
`dma-coherent`가 0건인 것과 모순되지 않는다 — 비일관 장치이므로 드라이버가 해 주는 것이다.
**Hailo DMA 방향에서는 CPU 쪽 sync가 중복**이고, 그래서 세 변종이 전부 실패했다.

**그런데 이것이 sync를 없앨 근거는 못 된다**:
1. **HailoRT 4.24.0 + `hailo_pci` 구현에 의존하는 관찰**이지 스펙도 계약도 아니다.
   바뀌면 조용히 깨지고 증상은 "간헐적 오검출"이라 추적이 거의 불가능하다
2. **V3D 방향은 이 증거가 덮지 못한다.** V3D는 import 시점에 attach를 한 번 하고 프레임마다
   다시 map하지 않으므로 위의 프레임별 캐시 관리가 없다. 그런데도 GPU 바이트 합이 항상 맞았고 — **기제 미확정**
3. 앞으로 heap 버퍼를 **장치가 아닌 소비자**(녹화 스테이징, 트래킹)가 쓰면 이 보호가 사라진다

**→ sync는 그대로 유지한다.** 실험이 "필요하다"를 반증한 것이 아니라 **반증하지 못한** 것이다.

## 회귀 검증 (전부 통과, 조정자 독립 재확인)

| 게이트 | 결과 |
|---|---|
| 빌드 경고·에러 | **0** (`--clean-first`) |
| ctest | **39/39** (126초) |
| `test_preproc` | **max 오차 0**, pad_mismatch 0, NHCW mismatch 0 |
| `test_detections` | 3경로 전부 매칭 **1.0000**, preproc bit-identical 100/100 |
| **P6** GPU median | **2.128 / 2.127 / 2.129 ms** (게이트 [2.074, 2.162]) |
| **P8** 1:1 shader median | **4.244 ms**, 비트 일치 max 오차 0, dropped 0 |
| 검증 레이어(`--sync-validation` 포함) | **errors 0** (gpuab / fixture heap / live heap / stale 4셀 / v3 2셀 / heapimport) |
| ASan/UBSan (`build-asan/`) | 리포트·누수 **0** |
| fd 누수 | **0** (gpuab 14→14, live 7→7, heapimport 14→14) |
| `hailort.log` | error **0**, 온도·클럭 warning **0** |
| **기본 경로 (S5)** | `input_alloc` 기본 **`Plain`** 불변. heap은 `--alloc heap`으로만 진입 → **되돌림이 플래그 한 개 값** |

측정 조건: 칩 59.7~62.7 °C(**75 °C 미달, 냉각 대기 0초**), SoC 63.1~70.8 °C, ARM 2.40 GHz.
`throttled = 0xe0000`은 비트 17·18·19의 **과거 발생 래치**이고 현재 상태 비트(0~3)는 전부 0 →
**측정 중 스로틀링 없음**(세션 시작 시 `0x80000`에서 늘었다 — 다른 세션의 연속 NPU 벤치 흔적).
모든 프로브 **직전·직후** NPU 점유·경합 프로세스를 확인했다. 경합 구간의 잠정치는 **채택하지 않고**
clean window에서 다시 쟀다.

## 결론

**1. 목적 ①(설계 제약 제거) 달성.** "GPU가 쓴 버퍼를 CPU로 읽지 마라"가 이제 근거로 닫혔다:
- V3DV는 dma-heap fd를 **import한다**(전례 0건이던 질문)
- **GPU 읽기 손해가 없다**(밴드 A, 0.9994)
- **CPU 읽기가 3배 빠르다**(0.333배)
- 정확성은 영향 없다(후보 집합 바이트 일치)
- 일관성 규약(B.10 R1~R6)이 글과 코드로 존재하고, V1·V2가 실행 경로를 확인했다

**2. 목적 ②(일원화) 달성.** 같은 일을 하던 세 구현이 하나로 모였다. 그 과정에서
**실재하던 잠재 결함 둘이 사라졌다**: `HeapOutPool`의 sync 없는 `memset`(더러운 라인이 나중에 evict되면
장치가 쓴 데이터를 덮어쓴다 — 비결정적이라 디버깅 거의 불가), ioctl 반환값을 버리던 두 곳.

**3. ⚠ 기존 버퍼는 하나도 옮기지 않았다. 의도된 것이다.**
현재 핫패스에 CPU가 V3D 메모리를 크게 읽는 곳이 **없다**(그렇게 만든 것이 이 프로젝트의 성과다).
**오늘의 직접 성능 이득은 0이다.** 51.6 FPS / 검출 9457 / P6 2.12 ms / P8 4.23 ms에 정확히 맞춰진
게이트를 이득 0인 변경으로 흔드는 것은 나쁜 거래다.
**단, 밴드 A가 나왔으므로 "나중에 싸게 할 수 있는 일"이 됐다** — 전환이 필요해지면 GPU 손해를 걱정하지 않아도 된다.

**4. 이득이 생기는 조건**(아직 존재하지 않는 버퍼):
- **녹화 스테이징** — Pi 5에 HW 인코더가 없어 CPU x264가 1080p RGBA 8.29 MB/frame을 읽어야 한다.
  V3D면 **5.35 ms/frame**(1.55 GB/s), heap이면 **1.78 ms/frame**(4.65 GB/s) → **차 3.57 ms/frame**
- 트래킹·다중 스트림에서 후보가 1.6 KB → 수백 KB로 커질 때
- 입력 해상도·스트림 수 증가로 1.22 MB 상수가 배가 될 때

## 주의와 남은 일

- **⚠ `--alloc heap` 경로가 ctest에 없다.** 계획서 (d')(gtest 하나로 살려 두기)를 **하지 않았다**
  → **안 쓰는 코드는 조용히 깨진다.** 다음에 이 경로를 쓸 사람은 **먼저 `--probe gpuab`·`--probe fixture --alloc heap`을
  돌려 살아 있는지 확인해야 한다.** 편입은 `05_tests/`에 픽스처 모드 테스트 하나(NPU 불필요, 약 1분)면 된다
- **V3D 읽기 방향의 캐시 일관성 기제가 미확정이다.** Hailo 쪽은 드라이버가 한다는 것을 확인했지만
  V3D는 프레임마다 map하지 않는데도 오염이 관측되지 않았다. 축출이 이미 끝났을 가능성과
  V3D가 스누핑할 가능성을 가르지 못했다
- **결정적 실험을 하지 않았다**: ftrace로 `arch_sync_dma_for_cpu|for_device`를 프레임 단위로 세어
  **전송마다** 부르는지 **매핑 때 한 번만** 부르는지 가르는 것. **tracepoint가 root를 요구한다**
- `--probe v3`의 `off` 셀과 `--probe heapimport --no-headroom`은 **FAIL이 정상**이다(오염·축소 재현이 목적).
  CI에 넣으려면 "실패해야 통과"로 뒤집어야 한다
- **`--probe heapimport`의 기본 출력 파일명이 `p0_heapimport.txt`라 1단계 증거를 덮는다.**
  옵션 없이 돌리면 또 덮는다. `--out-file`을 쓸 것
- **`01_src/isp/dma_heap_buffer.*` 흡수는 소유 세션 조율 대기다.** 그 파일에 결함 둘이 남아 있다:
  **16384 정렬을 확인하지 않는데 그 버퍼가 HailoRT `set_dma_buffer`로 간다**(부록 A.4 요구),
  **`begin/end_cpu_read`가 없다**(D2H 용도로 그대로 쓸 수 없다). `dma_heap_alloc.hpp` 머리말에 기록해 두었다

## 파일

- `REPORT.md` — 1단계 P0 관문 상세
- `evidence/p0_heapimport.txt` — P0 6텐서 전수
- `evidence/stage2_unify.txt`, `stage2_p2_heap_result.txt` — 일원화, `--probe p2 --alloc heap` 30/30 일치
- `evidence/stage3_headroom.txt`, `stage3_headroom_probe.txt`, `stage3_headroom_negative_control.txt`
  — 헤드룸 적용 전후, **음성 대조**(`--no-headroom`이 `conv71_178`에서만 예외)
- `evidence/stage4_gpu_ab.txt` — A/B 인터리브 + G5 + cpu_read 3자
- `evidence/stage5_v1.txt`, `stage5_v2.txt`, `stage5_v3.txt`, **`stage5_v3_perf_cachemaint.txt`**(perf 근거)
- `verification/stage45_regression.txt`, `verify_summary.txt`, `asan_*`
- `env_bench.txt` — 단계별 환경 스냅샷
