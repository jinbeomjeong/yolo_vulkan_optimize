# 제안 방법 적용 전·후 비교 영상과 실시간 측정 (2026-11-05)

독자가 적용 전·후를 눈으로 비교하도록 만든 영상 2개와 그 근거 측정이다.
영상은 실행 기록으로 **합성**했다. 화면을 직접 녹화하지 않은 까닭은 이 보드에 H.264 하드웨어 인코더가 없어(디코더 `rpi-hevc-dec`만 있음), 실행 중 녹화하면 인코딩이 CPU를 써서 비교할 수치를 오염시키기 때문이다.

| 영상 | 내용 | 기록 |
|---|---|---|
| [`samples/before.mp4`](samples/before.mp4) | 발표자료 3장 흐름: 소프트웨어 디코드 → swscale → NPU(HailoRT 전치·NMS) → CPU 표시 | `raw/before_run3` |
| [`samples/after.mp4`](samples/after.mp4) | 발표자료 5장 구조: HW 디코더 → GPU NHCW → NPU → GPU 후처리·NMS → GPU 표시 | `raw/after_run3` |

영상은 CPU가 3회 중앙값인 실행으로 만들었고, 1920×1080·30 fps·60.6초, 각 약 29 MB(H.264 CRF 20)다. `*.mp4`는 git으로 추적하지 않으므로 아래 명령으로 다시 만든다.

```bash
/home/pi/miniconda3/envs/hailo-yolo/bin/python 06_tools/python/demo_compose.py \
  07_results/demo_before_after/raw/<before|after>_run3 04_data/videos/video_h265.mp4 \
  --label <before|after> --out 07_results/demo_before_after/samples/<before|after>.mp4
```

## 측정 조건

- 날짜: 2026-11-05. 벤치 `06_tools/bench/demo_compare_bench.sh`, 3회 인터리브, 실행 전 냉각. 환경은 [`env_bench.txt`](env_bench.txt).
- 하드웨어: Raspberry Pi Compute Module 5, Hailo-8, PCIe Gen3 x1, 부트로더 2026-10-30, V3D 1150 MHz, ARM 2.4 GHz.
- **CPU 거버너 performance 고정**(측정 동안만, trap으로 `ondemand` 원복).
- 모델 `yolov11m.hef`, HailoRT 4.24.0. 입력 `video_h265.mp4`(1080p30, 1,816프레임).
- 공통 운용: 배치 1, 동시 처리 1, 입력 페이싱 source, 입력 대기열 1, 표시 on(전체화면 HDMI-A-1).
- 적용 전 `yvo_cpu_baseline`: 소프트웨어 HEVC 1스레드, `fast_bilinear+full_chroma_int` swscale(4스레드), NHWC 입력, HailoRT NMS, CPU 박스 그리기, `wl_shm` 창.
- 적용 후 `yvo_pipeline` 기본값 + Vulkan 합성 표시.
- 기록을 켠 상태(`--frame-log`, `--dump-dir`)의 값이다. 코드는 `fa2e714` + 미커밋 도구로, 커밋은 이 측정 뒤에 했다.

## 결과 (3회 중앙값, 괄호는 최소~최대)

| 설정 | `fps` | `cpu_ms_per_frame` | 코어 사용률 | `labwc_ms_per_frame` | 처리/도착 | 입력 버림 |
|---|---:|---:|---:|---:|---:|---:|
| 적용 전 | 24.38 (24.38~24.38) | **47.862** (47.790~47.892) | 117% | 3.954 | 1479/1816 | 18.6% |
| 적용 후 | 24.32 (24.31~24.32) | **1.492** (1.459~1.494) | 4% | 0.401 | 1473/1816 | 18.9% |

CPU 시간은 프로세스 자신의 값이고, 컴포지터 labwc는 별도 열이다. 커널 스레드와 시스템 전체 CPU는 이 표에 없다(`raw/*/cpu_samples.csv`).

적용 전 CPU 구성(run3, 스레드별 on-CPU, ms/처리 프레임): 디코드 스레드 35.409 · 메인과 HailoRT·swscale 스레드 합 11.029 · 표시 스레드 1.424.

| 단계 시간(실행 평균, OSD 고정 표, ms) | 적용 전 | 적용 후 |
|---|---:|---:|
| 디코드 | 28.9 (CPU) | 2.8 (호출 시간, HW) |
| 전처리 | 3.0 (swscale, CPU) | 2.8 (GPU 구간) |
| NPU | 37.8 (HailoRT 전치·NMS 포함) | 32.7 |
| 후처리 | 위 NPU에 포함 | 0.8 (GPU, NMS 포함) |
| 표시 준비 | 3.1 (CPU) | 3.7 (GPU) |

단계 시간의 정의가 두 경로에서 다르므로(출처는 `06_tools/python/demo_compose.py`의 `stage_rows_*`) 행끼리 빼거나 합쳐 비교하지 않는다.

## 결론

- 응용 CPU가 프레임당 47.862 ms에서 1.492 ms로 **약 32분의 1**이 됐다. 처리 FPS는 같다. 두 경로 모두 NPU 주기에 묶이기 때문이다.
- 그래서 두 영상의 끊김은 비슷하고, 눈에 보이는 차이는 OSD의 CPU 값이다.
- **적용 전은 소프트웨어 디코드 1스레드가 영상 일부에서 30 fps를 따라가지 못한다.** 디코드 지각(`decode_late`)이 평균 81.5 · p90 314.6 · 최대 502.1 ms이고(적용 후 3.0 · 5.1 · 21.5), 그만큼 적용 전 영상의 표시가 늦다. 같은 시각의 두 영상을 나란히 보면 프레임이 다를 수 있다. 입력 지연은 OSD에 넣지 않았다(사용자 결정).

| 도착 기준 지연(참고, OSD에 없음, ms) | 평균 | 중앙값 | p90 | 최대 |
|---|---:|---:|---:|---:|
| 적용 전 | 135.91 | 87.71 | 367.92 | 560.14 |
| 적용 후 | 53.75 | 53.63 | 66.70 | 72.82 |

이 값의 끝점은 화면이 아니라 결과 처리 시점이고, 적용 전 지연의 상당 부분은 1스레드 디코드 지각이다. 자동 스레드로 재면 줄어들 수 있으나 재지 않았다.

결정: 녹화는 실행 후 합성으로 한다 (사용자 결정, 2026-11-05)
결정: 적용 전은 발표자료 3장 그림 그대로의 경로, swscale `fast_bilinear+full_chroma_int`, 디코더 1스레드로 한다 (사용자 결정, 2026-11-05)
결정: OSD는 FPS·응용 CPU(실시간)와 단계별 시간(실행 평균 고정)만 넣고, 지연과 버림은 넣지 않으며, 적용 후 표에 표시 행을 넣는다 (사용자 결정, 2026-11-05)

## 신뢰성

- `hailort.log` error·warning은 6회 모두 0건이다.
- SoC는 5초 간격 기록([`evidence/soc_watch.csv`](evidence/soc_watch.csv))에서 62.6~75.2 °C, `get_throttled`는 내내 `0xe0000`(현재 비트 0, 상위는 이번 부팅의 이력), ARM 클럭 2.4 GHz 유지. Hailo 칩은 실행 전후 최고 72.6 °C.
- 합성 자기 검사: OSD 평균과 `stdout.log` 차이는 FPS 0.0%·0.1%, CPU ms 0.1%·1.5%. 적용 후의 코어 사용률 경고(3.57 대 4)는 `stdout.log`의 정수 반올림 때문이다. 표시 프레임은 before 1477/1477, after 1472/1473(1개는 30 fps 틱 사이에서 유실).

## 주의·남은 일

- **17.6 ms와 직접 비교하지 않는다.** 발표자료의 17.6 ms는 HW 디코더·GPU 전처리를 쓴 `--mode ptr --layout nhwc --postproc host`, GPU 960 MHz·거버너 다름의 값이다. 이 실험의 47.862 ms는 디코드까지 CPU인 다른 경로다.
- **1.492 ms는 처리한 프레임 기준이고 기록 비용을 포함한다.** 입력의 약 19%는 버려진다. 발표자료의 "약 1 ms"는 표시·기록 조건이 달라 다른 값이다.
- 전처리가 달라 검출이 같지 않다. run3 공통 1,190프레임에서 검출 수 −1.57%(적용 후 기준), IoU 0.5 일치 0.763([`evidence/detections_before_vs_after_run3.txt`](evidence/detections_before_vs_after_run3.txt)). 박스 차이는 오프로딩 효과가 아니다. 낮은 점수의 경계 검출이 대부분이라는 판단은 300프레임 스모크의 점수별 일치율에 근거한다(점수 0.7 이상 100%, 기존 ptr 경로 대비, 전체 길이로는 재지 않았다).
- swscale 플래그가 `preproc_cpu_vs_gpu`(`bilinear`)와 다르다. `bilinear`는 스모크에서 검출이 18.5% 줄어 영상에 맞지 않았다.
- `present_ms`의 의미가 다르다(적용 전 shm 커밋, 적용 후 화면 도달). 영상의 표시 시각은 이 값을 쓰므로, 적용 전 영상의 실제 화면 도달은 더 늦을 수 있다.
- 적용 전 표시도 labwc 합성에 V3D를 쓴다. 도구 자체는 GPU를 열지 않는다.
- 측정 중 `btop` 모니터가 떠 있었다(점검 때 CPU 0.2%).
- 발표자료 3장의 그림(전부 CPU)과 17.6 ms의 경로가 다른 문제는 남아 있다.
