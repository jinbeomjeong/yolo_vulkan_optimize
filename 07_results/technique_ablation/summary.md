# 확정 파이프라인에서 기법을 하나씩 뺀 비교 (2026-11-06)

> **대체됨(2026-11-09):** 아래 표는 `c97c274` 이전 바이너리(ef24741c)의 값이다. 9설정 전체를 현재 바이너리(dc447c5)로 다시 잰 값은 [`runs/rerun_dc447c5`](runs/rerun_dc447c5/summary.md)이고, 논문 §6.4·§7.1과 그림 12·13·15는 그 회차를 쓴다 (사용자 결정, 2026-11-09)

확정 구성(`base`)에서 기법 하나를 뺀 구성의 응용 CPU를 같은 조건에서 쟀다. 논문 그림 12·13의 입력이다.

## 측정 조건

- 날짜: 2026-11-06 02:01~02:24(KST). 벤치 `06_tools/bench/technique_ablation_bench.sh`, 3회 인터리브, 실행 전 냉각. 환경은 [`env_bench.txt`](env_bench.txt).
- 하드웨어: Raspberry Pi Compute Module 5, Hailo-8, PCIe Gen3 x1, V3D 1150 MHz, ARM 2.4 GHz.
- **CPU 거버너 performance 고정**(측정 동안만, trap으로 `ondemand` 원복).
- 모델 `yolov11m.hef`, HailoRT 4.24.0. 입력 `video_h265.mp4`(1080p30, 1,816프레임).
- 공통: `--mode dmabuf --decoder hw`, 디코더 1스레드, 배치 1/슬롯 1, 입력 페이싱 source, 입력 큐 1, 표시 on(전체화면 Vulkan 합성), 기록 on.
- 코드: 커밋 `4593529`, 바이너리 md5 `c8515629c17b1714f54c02c4021f9c76`(측정 뒤 같음).

| 설정 | 확정 구성에서 달라진 것 |
|---|---|
| `base` | 없음 (NHCW dma-buf, GPU 후처리, GPU NMS) |
| `post_host` | `--postproc host`: 런타임이 CPU에서 후처리 |
| `layout_nhwc` | `--layout nhwc --postproc host`: 런타임이 NHWC→NHCW 변환까지 CPU에서 |
| `nms_cpu` | `--nms cpu`: 억제만 CPU |
| `input_copy` | `--input-copy`: GPU 출력을 CPU가 캐시 dma-heap으로 복사한 뒤 NPU로 |

`layout_nhwc`는 후처리도 호스트다. GPU 후처리는 NHCW 입력 전용이어서 레이아웃만 바꿀 수 없다.

## 결과 (3회 중앙값, 괄호는 최소~최대)

| 설정 | `fps` | 입력 버림 | `cpu_ms_per_frame` (처리 기준) | 도착 기준 CPU ms | `npu_ms_mean` | `e2e_arr` 평균 ms | 도착→화면 평균 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| `base` | 24.30 | 18.9% | **1.534** (1.452~1.600) | 1.243 | 32.69 | 53.77 | 70.26 |
| `post_host` | 24.13 | 19.5% | **3.058** (3.046~3.190) | 2.462 | 33.66 | 53.65 | 70.27 |
| `layout_nhwc` | **18.68** | **37.7%** | **16.433** (16.416~16.618) | 10.244 | 45.79 | 66.19 | 82.79 |
| `nms_cpu` | 24.33 | 18.9% | **1.436** (1.426~1.449) | 1.165 | 32.79 | 53.59 | 70.22 |
| `input_copy` | 24.16 | 19.4% | **2.278** (2.262~2.296) | 1.836 | 32.17 | 53.82 | 70.41 |

| 기법을 뺐을 때 늘어나는 CPU (처리 기준, 중앙값끼리 차) | 비교 | ms/프레임 |
|---|---|---:|
| NHWC→NHCW 변환 | `layout_nhwc` − `post_host` | **+13.375** |
| GPU 후처리 전체(디코드 + NMS) | `post_host` − `base` | **+1.524** |
| GPU→NPU dma-buf | `input_copy` − `base` | **+0.744** |
| GPU NMS | `nms_cpu` − `base` | **−0.098** |

`input_copy`의 복사 구간은 memcpy 0.822 + sync 0.027 = 0.848 ms/프레임이다(`input_copy_ms`).

## 결론

- **가장 큰 항목은 NHWC→NHCW 변환이다(+13.4 ms).** 슬롯 1에서는 이 변환이 NPU 제출 경로에 들어가 처리량까지 깎는다(24.30 → 18.68 fps, 버림 37.7%, 도착 → 결과 +12 ms).
- **GPU 후처리의 CPU 절감은 이 운용점에서 약 1.5 ms다.** 이전 값(3.247 → 0.938, 표시 off, asap)보다 작다. 조건이 달라 직접 비교하지 않는다.
- **GPU→NPU dma-buf는 약 0.74~0.85 ms를 없앤다.** 합성이 아니라 실제 경로의 값이다. 복사해도 FPS·지연은 거의 같아서 이득은 CPU다.
- **GPU NMS는 CPU를 줄이지 않고 약 0.10 ms 늘렸다.** 범위가 겹치지 않는다(1.426~1.449 대 1.452~1.600). 원인은 가르지 않았다. GPU 쪽에 CSD 잡이 하나 더 있고, 호스트의 잡 제출·대기 비용으로 보이나 확인하지 않았다. 옮긴 이유가 구조였다는 기존 서술(§7.1)과 맞는다.
- 검출: `--input-copy`는 GPU·호스트 후처리 모두 복사 없는 실행과 `detections.jsonl`이 바이트 동일하다(300프레임, asap, [`evidence/bit_exact_input_copy.txt`](evidence/bit_exact_input_copy.txt)).

결정: 그림 12의 정의를 "확정 파이프라인에서 그 기법 하나만 뺐을 때 늘어나는 CPU 시간"으로 한다 (사용자 결정, 2026-11-06)
결정: NHWC→NHCW 변환은 호스트 후처리 조건에서의 차이로 측정한다 (사용자 결정, 2026-11-06)
결정: GPU→NPU 구간을 실제 복사 경로로 측정한다 (사용자 결정, 2026-11-06)

## 신뢰성

- `hailort.log` warning·error 15회 모두 0건, 종료 코드 0, `ok` 전부 1.
- SoC 5초 기록([`evidence/soc_watch.csv`](evidence/soc_watch.csv), 뒤의 `pipeline_live_input` 측정까지 포함) 62.6~76.3 °C, 실행 전 최고 69.7 °C. `get_throttled` 내내 `0xe0000`(현재 비트 0). Hailo 실행 전 최고 65.0 °C.

## 주의·남은 일

- 값은 처리 프레임 기준이다. `layout_nhwc`는 처리 프레임이 적어(1,132 대 1,472) 도착 기준으로는 7.78 ms 차이지만, 변환은 처리한 프레임에만 일어나므로 처리 기준을 쓴다.
- `layout_nhwc − post_host`는 호스트 후처리 조건의 값이고, 단일 기법 제거가 아니다. 또 NPU 대기 양상이 달라 순수 변환 외의 차이가 섞일 수 있다.
- `base`의 실행 간 폭이 0.148 ms로 넓다. 이보다 작은 차이에는 큰 의미를 두지 않는다. `nms_cpu` 차이는 범위가 겹치지 않을 뿐 폭과 비슷한 크기다.
- 표시·기록 on 값이다. 표시 off인 기존 값과 섞지 않는다.
- 측정 중 진행 확인과 읽기 전용 검토 에이전트가 돌았다. 벤치의 코어는 고정하지 않았다.
- `input_copy`의 NPU 지연이 `base`보다 0.5 ms 짧다(32.17 대 32.69). 캐시 힙 입력이 원인으로 보이나 확인하지 않았다.
  - → 정정(2026-11-07): 캐시 힙 효과가 아니라 `npu_ms` 측정 구간 시작점 효과다. 처리량·지연 이득은 없다([`runs/npu_latency_window`](runs/npu_latency_window/summary.md)).

## 정정 (2026-11-06, 메인 에이전트)

- 위 결정의 "그 기법 하나만 뺐을 때" 정의는 그림 12의 8·10·11번에만 성립한다. 그림 12의 정의를 "각 기법의 CPU 절감을 가장 가까운 측정으로 모은 값"으로 고치고, 감소율은 실측 구성이 있는 1·7·8·10번에만 적는다.

결정: 그림 12의 정의를 완화하고 7번 색을 분리하며 감소율은 1·7·8·10번에만 적는다 (사용자 결정, 2026-11-06)
결정: 그림 13은 기존·NHCW 직접 입력·제안의 3행으로 그린다 (사용자 결정, 2026-11-06)
결정: 헤드라인 58.4·70.2 ms는 유지하고 파일 입력 조건의 단서를 붙인다 (사용자 결정, 2026-11-06)

## 하위 회차 (2026-11-07 색인)

결정: 기법 제거 요약에 하위 회차 목록을 둔다 (사용자 결정, 2026-11-07)
결정: 위 "주의" 절의 `input_copy` NPU 지연 추정에는 원문을 두고 정정 참조만 덧붙인다 (사용자 결정, 2026-11-07)

| 회차 | 내용 |
|---|---|
| [`runs/dmabuf_ablation`](runs/dmabuf_ablation/summary.md) | dma-buf 공유만 구간별로 CPU 복사로 바꾼 비교 (2026-11-06) |
| [`runs/display_onoff`](runs/display_onoff/summary.md) | 표시를 끄면 도착 기준 지연이 길어지는 현상 (2026-11-06) |
| [`runs/display_onoff_aspm_off`](runs/display_onoff_aspm_off/summary.md) | 위 현상의 진단 회차, ASPM L1 끔 (2026-11-06) |
| [`runs/o2_perf_profile`](runs/o2_perf_profile/summary.md) | 확정 구성의 남은 응용 CPU를 perf로 분해 (2026-11-06) |
| [`runs/cpu_trim`](runs/cpu_trim/summary.md) | 변경 3건(사전 매핑 크기·후처리 CB 사전 기록·`--gpu-timestamps`) 전후 비교 (2026-11-06) |
| [`runs/npu_latency_window`](runs/npu_latency_window/summary.md) | `input_copy`의 NPU 지연이 짧게 나오는 이유 — 측정 구간 시작점 효과 (2026-11-07) |
| [`runs/o2_mot20_02`](runs/o2_mot20_02/summary.md) · [`o2_mot20_05`](runs/o2_mot20_05/summary.md) | 확정 구성을 혼잡 장면 MOT20 두 영상으로 재측정 (2026-11-07) |
| [`runs/display_npu_window`](runs/display_npu_window/summary.md) | 표시 on이 NPU 지연·도착 기준 지연을 줄이는 이유 — 장치 주기는 같고, 제출이 NPU 출력 뒤 꼬리와 겹침 (2026-11-09) |
| [`runs/submit_delay`](runs/submit_delay/summary.md) | 표시 off에서 제출을 `--submit-delay-ms`로 늦춘 스윕 — D 4 ms까지 처리량 그대로 도착 기준 지연 −4.6 ms (2026-11-09) |
| [`runs/rerun_dc447c5`](runs/rerun_dc447c5/summary.md) | 9설정 전체를 현재 바이너리(dc447c5)로 재측정 — `base` 1.534 → 1.348 ms, NHWC +13.92·후처리 +1.76·GPU→NPU +0.88 ms (2026-11-09) |
