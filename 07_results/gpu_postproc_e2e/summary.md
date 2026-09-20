# 단계 3b — `yvo_pipeline --postproc gpu` (GPU 후처리 e2e 통합)

호스트 NMS(HailoRT 가 CPU 에서 하는 고정비)를 **GPU pass 1(`yolo_decode.comp`) + CPU greedy 억제**로
바꾼 뒤의 측정. 기준 경로는 `--postproc host`(HEF 내장 NMS)다.

## 환경
`env_bench.txt` 참고. PCIe **Gen3**(8.0 GT/s x1), kernel 6.18.50+rpt-rpi-2712, HailoRT FW 4.24.0,
`vcgencmd get_throttled = 0xe0000`(과거 비트만, 하위 4비트 0).
측정 규약: `.claude/skills/measure` — 설정당 3회, 반복 바깥 루프, 칩 75 °C 초과 시 65 °C 까지 냉각.

## 처리량·CPU

> ### ⛔ 아래 gpu 행의 FPS·지연은 **결함 상태의 값이다. 인용하지 마라.**
> 측정 당시 `HailoRawRunner` 에 백프레셔 이중 계수 결함이 있어 NPU 동시 비행이 슬롯의 절반으로
> 묶여 있었다(§"해소됨" 참조). **수정 후의 값은 아래 표를 쓴다.**
> CPU 절감과 정확도(§정확도)는 이 결함의 영향을 받지 않으므로 그대로 유효하다.

### 수정 후 — 유효한 값 (`07_results/raw_runner_backpressure/`, 900프레임 3회 중앙값, host·gpu 인터리브)

| 설정 | FPS | host 대비 | CPU ms/frame | 절감 | NPU latency mean | 검출 |
|---|---:|---:|---:|---:|---:|---:|
| host b2/s4 | 34.97 | — | 2.921 | — | 111.33 | 3369 |
| **gpu b2/s4** | **34.96** | **−0.03 %** | **1.122** | **−1.80** | 110.31 | 3369 |
| host b8/s16 | 51.51 | — | 3.135 | — | 301.24 | 3369 |
| **gpu b8/s16** | **51.48** | **−0.06 %** | **1.152** | **−1.98** | 297.16 | 3369 |

**GPU 후처리는 처리량을 사실상 깎지 않으면서 CPU 를 1.8~2.0 ms/frame 줄인다.**

### 결함 상태의 원본 기록 (보존용 — 인용 금지)

| 설정 | FPS | CPU ms/frame | NPU latency mean | e2e mean | 검출 |
|---|---|---|---|---|---|
| host b2/s4 | 34.98 | 3.338 | 111.49 | 142.14 | 3369 |
| gpu  b2/s4 | ~~24.48 (-30.0%)~~ | 1.282 (-2.056) | ~~81.62~~ | ~~183.30~~ | 3369 |
| host b8/s16 | 51.55 | 3.140 | 301.43 | 325.24 | 3369 |
| gpu  b8/s16 | ~~49.66 (-3.7%)~~ | 1.161 (-1.979) | ~~159.68~~ | ~~283.33~~ | 3369 |

GPU pass 1: b2/s4 mean 0.635 / p90 0.726 ms, b8/s16 mean 1.176 / p90 4.042 ms.
CPU 억제: 2.2~2.7 µs/frame. 후보 수 mean 16.81 / max 45 (상한 4096, overflow 0).

## 정확도 (전체 1816프레임, b8/s16)

`06_tools/python/compare_detections.py` — `verification/accuracy_host_vs_gpu.csv`

| 지표 | 값 | 게이트 |
|---|---|---|
| 검출 총수 (host / gpu) | 9457 / 9457 (+0.00 %) | G3 9457±1% **PASS** |
| match @ IoU 0.5 | 1.0000 | — |
| match @ IoU 0.7 | 1.0000 | — |
| match @ IoU 0.9 | **0.9990** | G1 ≥0.95 / G2 ≥0.99 **PASS** |

클래스별 개수도 14개 클래스 전부 동일하다.

## CPU 사용처 (perf, 400프레임)

| | host | gpu |
|---|---|---|
| `libhailort` `0x30ffxx` (호스트 NMS) | 5개 심볼, **합계 45.10 %** | **0건** |
| Event count (approx.) | 4,193,831,232 | 2,120,991,113 (**-49.4 %**) |

## ✅ 해소됨 — 처리량 저하의 원인은 후처리가 아니라 러너 백프레셔였다

> **2026-09-20 수정 완료.** 아래 분석이 맞았고, `HailoRawRunner::submit()` 의 수동
> `increase_ongoing_callbacks()` / `decrease_ongoing_callbacks()` 쌍을 **제거**해 해소했다
> (`infer_async()` 가 내부에서 요청당 1을 이미 센다 — 실측 확정).
> 결과: gpu `b2/s4` 24.48 → **34.96 FPS**, `b8/s16` 49.66 → **51.48 FPS** (host 대비 0.1 % 이내).
> 규명 과정·실측값·디스어셈블 근거는
> [`07_results/raw_runner_backpressure/summary.md`](../raw_runner_backpressure/summary.md) 에 있다.

### 당시 분석 (그대로 맞았다)

원시 출력 러너의 백프레셔가 NPU 동시 비행 프레임을 **`num_slots/2`** 로 묶는다.
`01_src/hailo/hailo_raw_runner.cpp:271` 이 `wait_for_ongoing_callbacks_count_under(num_slots_)` 를
쓰는데, 직전의 `ng_->increase_ongoing_callbacks()` 가 HailoRT `infer_async` 내부 증가와 **이중 계수**된다.

실측 in-flight(= FPS x npu_latency_mean): b2/s4 **2.00**, b2/s8 **3.96**, b8/s16 **7.85** — 전부 정확히 슬롯의 절반.

임계값만 `2 * num_slots_` 로 바꿔 재빌드한 **임시 실험**(300프레임, 이후 원복함):

| 설정 | 현재 | 임계값 2x | host 기준 |
|---|---|---|---|
| gpu b2/s4 | 24.41 | **34.78** | 35.00 |
| gpu b8/s16 | 49.24 | **50.93** | 51.55 |

즉 후처리 이관 자체의 처리량 비용은 **-1.2 % ~ -0.6 %** 수준이고, 관측된 -30 % / -3.7 % 는
러너의 백프레셔 버그다. → **수정 완료**(위 참조). `hailo_raw_probe.cpp` 의 `RawSession` 도 같이 고쳤다.
남은 같은 패턴: `01_src/vulkan/tools/postproc_probe.cpp:1309-1313`(임계값 1 의 직렬 제출이라 처리량 영향 없음).

## 파일
- `results_raw.csv` 실행별, `results.csv` 설정별 중앙값
- `raw/<id>_run<N>/{stdout.log,hailort.log}`
- `dump_host/`, `dump_gpu/` 전체 1816프레임 detections.jsonl
- `verification/` 검증 레이어·ASan·perf 로그, 정확도 CSV
