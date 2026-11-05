# 전처리 단계 분리와 NPU 출력 복사 비용 (2026-11-06)

CPU 기준 구현의 "리사이즈 + 색 변환 + 패딩" 8.85 ms를 나눠 재고, GPU 후처리가 dma-buf로 받지 않을 때의 CPU 복사 비용을 같은 규약으로 쟀다. 논문 그림 "기법별 CPU 절감"의 입력이다.

## 측정 조건

- 날짜: 2026-11-06 00:07 시작(KST). 실행기 `CONFIG_SET=stage_split RUNS=3 EXPECT_GOVERNOR=performance bash 06_tools/probes/preproc_cmp_run.sh`, 집계 `06_tools/probes/preproc_cmp_aggregate.py`. 환경은 [`env_bench.txt`](env_bench.txt).
- 하드웨어: Raspberry Pi Compute Module 5, V3D 1150 MHz, ARM 2.4 GHz. **CPU 거버너 performance 고정**(측정 동안만, trap으로 `ondemand` 원복). NPU 미사용, 표시 미사용.
- 입력: `video_h265.mp4`, 디코더 스레드 1, 워밍업 60 + 600프레임, 1스레드 `taskset -c 3`. 값은 실행별 중앙값(CPU는 평균)의 3회 중앙값.
- 부모 실험 [`preproc_cpu_vs_gpu`](../../summary.md)와 같은 규약이다. 도구는 `yvo_preproc_bench --impl cpuref-split|rawout-copy|rawout-import`.

## 결과 ([`evidence/medians.txt`](evidence/medians.txt))

| 설정 | wall 중앙값 ms (범위) | CPU ms/f (범위) |
|---|---:|---:|
| `cpuref_t1` | 10.562 (10.493~10.577) | 10.589 (10.518~10.632) |
| `cpuref_split_t1` | 9.217 (9.203~9.236) | 9.250 (9.239~9.265) |
| `gpu` | 1.799 (1.799~1.808) | 0.076 (0.076~0.081) |

| 레터박스 분해 (ms) | 값 | 출처 |
|---|---:|---|
| 전체(리사이즈 + 색 변환 + 패딩) | 8.846 | `cpuref_t1` |
| 리사이즈 | 5.064 | `cpuref_split_t1` |
| YUV→RGB 직접 | 2.072 | `cpuref_split_t1` |
| 패딩 | 0.191 | `cpuref_split_t1` |
| 나눠 잰 합 | 7.327 | |
| **잔차(전체 − 합)** | **1.526** | 원인 미확인 |
| 대조: YUV→RGB 차분(전체 − 리사이즈 − 패딩) | 3.595 | 쓰지 않음 |

같은 `cpuref_t1`의 나머지 단계: map 0.123 · SAND→선형 1.144 · 전치 0.422 · sync 0.004 ms.

| NPU 출력 1,221,120 B 전달 (ms) | 전달 구간 CPU | memcpy | sync | 사이클 CPU(GPU 패스 포함) |
|---|---:|---:|---:|---:|
| `rawout_copy_t1` | **0.279** | 0.225 | 0.053 | 0.432 |
| `rawout_import_t1` | **0.0004** | 0 | 0 | 0.144 |

사이클 CPU 차이는 실행별로 짝지어 0.289 ms다. 전달 구간 값과 맞는다.

## 결론

- 전처리 CPU의 가장 큰 항목은 리사이즈(5.06 ms)이고, 패딩(0.19 ms)은 작다. GPU 셰이더는 이 전체를 CPU 0.076 ms로 대체한다.
- 한 루프로 돌리면 나눠 돌릴 때보다 약 1.5 ms가 더 든다. 스모크와 비공식 실행에서도 같은 방향이어서 잡음은 아닌 것으로 본다. **원인은 규명하지 못했다**(레지스터 압력, 픽셀별 패딩 분기, 분기 예측은 확인하지 않은 후보다).
- NPU 출력을 CPU로 복사하는 비용은 약 0.28 ms로, 전처리 어느 큰 항목보다 작다.

결정: 3~7번 기준은 CPU 기준 구현으로 하고, 단계를 분리 측정한다 (사용자 결정, 2026-11-05)
결정: 논문 그림의 YUV→RGB는 직접 측정값으로 쓰고 잔차를 따로 표시한다 (사용자 결정, 2026-11-05)

## 신뢰성

- 정확도: 분리 출력은 ref와 660/660프레임 바이트 동일([`evidence/accuracy/`](evidence/accuracy/)). 복사 경로는 `copy_diff_bytes=0`, copy와 import의 GPU 결과 해시와 검출 2146개가 같다.
- 검증 레이어 200프레임: `gpu`·`rawout-copy`·`rawout-import` 모두 errors 0, warnings 2([`verification/`](verification/)).
- `get_throttled`는 `0xe0000`(현재 비트 0, 상위는 이력). SoC는 실행 시작 전 65.9 °C, 시간 측정 뒤 68.6 °C.
- 재현 확인: `cpuref_t1` 10.562는 부모 실험의 10.595, `gpu` CPU 0.076은 0.081과 근접한다.

## 주의·남은 일

- **출력이 같아도 시간까지 같지는 않다.** 분리 경로는 샘플링과 색 변환 사이의 float 중간값을 2.76 MB/프레임 쓰고 읽는데, ref에는 이 비용이 없다. 그럼에도 합이 더 작다.
- `cpuref_split_t1`의 SAND→선형은 `cpuref_t1`보다 약 0.16 ms 크다(1.309 대 1.144). 그래서 SAND→선형·map·전치는 `cpuref_t1` 행을 쓴다.
- 전치는 소스가 스칼라지만 `-O3`에서 자동 벡터화된다(`ld3`). "스칼라 기준"이라고 쓸 때 이 조건을 적는다.
- 패딩 0.191 ms는 ref 방식의 값이다. 같은 바이트 수의 `memset`은 부모 실험의 `sws_*` 행에서 0.058 ms다.
- **NPU 출력 복사는 합성 측정이다.** 파이프라인에 복사 경로는 없다. NPU를 쓰지 않아 소스 힙 버퍼를 V3D에 import해 attachment를 만들었으므로, HailoRT attachment로 직접 잰 sync 값이 아니다. 대조군의 입력 할당도 운용 기본값(`ExportDmaBuf`)이 아니라 힙 import다.
- 이 회차 전 2026-11-05 23:55에 거버너 `ondemand`로 실수 실행이 있었고, 중단 뒤 git 밖 `11_logs/2026-11-05_stage_split/accidental_ondemand_run/`으로 옮겼다. 수치로 쓰지 않는다. `RUNS`·`DRYRUN` 사용 실수였고 다른 측정과 겹치지 않았다.
- ASan은 이 도구 변경에 돌리지 않았다.
