# 표시를 켜면 NPU 지연과 도착 기준 지연이 짧아지는 이유 — NPU 출력 뒤 꼬리 시간과의 겹침

[`display_onoff`](../display_onoff/summary.md)(2026-11-06)는 확정 구성(O2)에서 표시를 켜면 NPU 지연(`npu_ms`)이 32.7 대 38.3 ms, 도착 기준 지연이 53.7 대 58.4 ms로 **짧아지는** 현상을 보고했습니다. ASPM은 원인이 아니었고, 원인은 규명하지 못했습니다.
- 그 뒤 [`npu_latency_window`](../npu_latency_window/summary.md)(2026-11-07)가 slots 1에서는 `npu_ms` = 장치 주기 − (콜백 → 다음 제출)라는 측정 구간 효과를 찾았습니다.
- 이 회차는 그 효과가 표시 on/off 차이도 설명하는지 직접 잽니다. 결정: 논문 1152행은 표시 on/off에서 X·G를 직접 잰 뒤 반영한다 (사용자 결정, 2026-11-07).

**결론부터 적습니다.**
- 표시 on과 off의 장치 주기 X + G는 40.92 ms로 같습니다. NPU가 빨라진 것이 아닙니다.
- 표시 on에서는 NPU 콜백 뒤 다음 GPU 제출까지 호스트가 약 5 ms를 씁니다(G 4.94 대 0.57 ms). 그만큼 X(GPU 제출 → NPU 완료)가 짧아집니다(35.99 대 40.36 ms).
- X가 짧아지는 이유는 NPU의 출력 뒤 꼬리 시간입니다. 배치 1에서 장치는 마지막 출력을 낸 뒤에도 약 8.35 ms 더 일합니다([`hef_context_profile`](../../../hef_context_profile/summary.md) 표의 `tail_ms`, m B=1. 2026-11-09 [`model_zoo_small`](../../../hef_context_profile/runs/model_zoo_small/summary.md) 회차에서도 같은 값). 표시 off에서는 다음 프레임이 0.57 ms 만에 제출돼 이 꼬리를 장치 안에서 기다립니다. 표시 on에서는 그 사이 호스트가 표시 일을 해서, 제출된 프레임이 덜 기다립니다.
- 도착 → GPU 제출 시간은 두 조건이 거의 같습니다(on 16.87~17.02, off 17.16~17.27 ms로 off가 약 0.3 ms 김). 그래서 도착 기준 지연 차이 4.7 ms는 X가 줄어든 4.4 ms에 이 0.3 ms를 더한 값입니다.

## 1. 측정 조건

| 항목 | 값 |
|---|---|
| 날짜 | 2026-11-09 19:25~19:35 |
| 보드 | Raspberry Pi CM5, Hailo-8 PCIe 8.0 GT/s x1(Gen3), V3D 1150 MHz, 커널 6.18.50+rpt-rpi-2712, labwc 0.20.2 |
| 바이너리 | 3d78e59 빌드(md5 af63c101). `npu_latency_window`·`o2_mot20_*`와 같은 바이너리입니다. 실행 시점 HEAD는 02f7535이고, 그 사이 `01_src` 변경은 없습니다 |
| 드라이버 | `ONLY='^(base|display_off)$' RUNS=3 TIMEOUT_S=180 BIN_PIPELINE=<위 바이너리> bash 06_tools/bench/technique_ablation_bench.sh`. 벤치가 실행마다 `--frame-log frame_log.csv`를 붙입니다 |
| 인자 | `base`: 확정 구성(O2) `--mode dmabuf --decoder hw --batch 1 --slots 1 --input-pace source --input-queue 1 --layout nhcw --postproc gpu --nms gpu --display on --display-backend wayland --display-fullscreen --display-sync matched --display-pace source`, 기록 on, `YVO_DEC_THREADS=1`. `display_off`: 같은 인자에 `--display off` |
| 입력 | `04_data/videos/video_h265.mp4` 전체(1080p30, 도착 1,816프레임) |
| 반복 | 3회. 회차가 바깥 루프, 설정이 안쪽 루프입니다. 매 실행 전 Hailo ≤ 65 °C로 냉각했습니다. 표시 실행 전마다 HDMI-A-1 Enabled와 labwc DRM master를 확인했습니다 |
| 거버너 | 측정 동안만 performance로 두고, trap으로 ondemand로 되돌렸습니다 |
| 신뢰성 | `hailort.log` warning·error 0, 6회 모두. 표시 on은 presented = processed, 표시 쪽 dropped 0. 표시 off 행의 `all_ok` 0과 `note=no_result_csv5`는 표시 off에서 화면 결과 줄(RESULT_CSV5)이 나오지 않아 붙은 것이고, `ok`·`exit_code`는 정상(1·0)입니다 |

환경은 [`env_bench.txt`](env_bench.txt)에, 실행별 출력과 `frame_log.csv`는 `raw/<설정>_run<N>/`에 있습니다. 실행 기록은 [`evidence/run_log.txt`](evidence/run_log.txt)입니다.

## 2. 표 — 3회 중앙값 (최소~최대)

### 2.1 벤치 요약

출처: [`results.csv`](results.csv), 실행별 값은 [`results_raw.csv`](results_raw.csv)

| 항목 | 표시 on (`base`) | 표시 off | 차이 |
|---|---|---|---|
| `fps` | 24.33 (24.30~24.33) | 24.42 (24.42~24.42) | −0.09 |
| 입력 버림 | 18.8% | 18.6% | |
| `npu_ms_mean` | 32.77 (32.72~32.80) | 38.35 (38.34~38.36) | −5.58 |
| `e2e_arr` 중앙값 ms | 53.52 (53.43~53.64) | 58.19 (58.10~58.20) | −4.67 |
| `e2e_arr` p90 ms | 66.48 (66.46~66.57) | 71.86 (71.83~71.88) | −5.38 |
| 도착 → 화면 평균 ms | 70.14 (69.86~70.15) | — | |
| `cpu_ms_per_frame` | 1.331 (1.319~1.446) | 0.721 (0.717~0.732) | |

`display_onoff`(2026-11-06)의 도착 → 결과 평균 53.71 대 58.41 ms(이번 `e2e_arr` 평균 53.61 대 58.33 ms)와 NPU 지연 32.72 대 38.25 ms가 재현됩니다.

### 2.2 X·G — 프레임 기록

출처: [`evidence/xg_analysis.txt`](evidence/xg_analysis.txt)([`evidence/xg.py`](evidence/xg.py)). 처리된 프레임 중 seq ≥ 30이고, 실행마다 1,442~1,449쌍입니다.
- X = 그 프레임의 GPU 제출 → NPU 완료
- G = 이전 처리 프레임의 NPU 완료 → 이 프레임의 GPU 제출

| 설정 | X 중앙값 ms | G 중앙값 ms | G 평균 ms | X+G 중앙값 ms | `npu_ms_mean` |
|---|---|---|---|---|---|
| 표시 on | 35.99 (35.96~35.99) | 4.94 (4.93~4.97) | 5.47 | **40.92** | 32.77 |
| 표시 off | 40.36 (40.35~40.36) | 0.57 (0.57~0.57) | 0.73 | **40.92** | 38.35 |

G 구간별 X 중앙값(3회 합산)입니다.

| G 구간 | 0~1 ms | 1~2 | 2~4 | 4~8 | 8 이상 |
|---|---|---|---|---|---|
| 표시 on | — | — | — | 35.98 (n 3941) | 32.86 (n 389) |
| 표시 off | 40.36 (n 4034) | 39.41 (n 90) | 38.02 (n 171) | 36.61 (n 52) | — |

두 설정의 같은 G 구간(4~8 ms)에서 X가 35.98 대 36.61 ms로 가깝습니다. X를 정하는 것은 표시 여부가 아니라 G입니다.

### 2.3 도착 기준 지연의 분해

출처: [`evidence/stage_split.txt`](evidence/stage_split.txt)([`evidence/stages.py`](evidence/stages.py)). 실행별 중앙값, ms입니다.

| 구간 | 표시 on | 표시 off |
|---|---|---|
| 도착 → 디코드 끝 | 3.64~3.68 | 3.59~3.60 |
| 디코드 끝 → GPU 제출 | 14.29~14.41 | 14.27~14.57 |
| **도착 → GPU 제출** | **16.87~17.02** | **17.16~17.27** |
| **GPU 제출 → NPU 완료 (X)** | **35.96~35.99** | **40.35~40.36** |
| NPU 완료 → 결과 | 0.57~0.58 | 0.56 |
| 도착 → 결과 | 53.43~53.64 | 58.09~58.19 |

## 3. 결론

**1. 표시가 NPU를 빠르게 하는 것이 아닙니다.** 장치 주기 X + G가 두 조건에서 40.92 ms로 같습니다. `display_onoff`가 후보로 적은 "표시 활동이 NPU 입출력 경로를 빠르게 한다"는 설명은 맞지 않습니다.

**2. `npu_ms`의 차이(−5.6 ms)는 측정 구간 효과입니다.** slots 1에서 NPU 완료 시각은 장치 주기가 정합니다. 콜백 뒤 호스트가 다음 제출 전에 쓰는 시간 G가 길수록 다음 프레임의 X와 `npu_ms`가 짧게 잡힙니다. [`npu_latency_window`](../npu_latency_window/summary.md)와 같은 원리이고, 거기서는 G가 ≤ 1 ms 범위에서만 달랐습니다.

**3. 도착 기준 지연의 차이(−4.7 ms)는 실제 차이이고, NPU 출력 뒤 꼬리와의 겹침에서 옵니다.**
- 장치는 프레임의 출력을 내보낸 뒤에도 약 8 ms 더 일합니다(m B=1의 `tail_ms` 8.35 ms, 컨텍스트 dynamic_3의 config 재적재로 추정 — [`hef_context_profile`](../../../hef_context_profile/summary.md)).
- 표시 off에서는 다음 프레임이 콜백 0.57 ms 뒤에 제출돼, 이 꼬리가 끝날 때까지 장치 안에서 기다립니다. 그래서 X가 40.4 ms입니다.
- 표시 on에서는 호스트가 약 5 ms 동안 다른 일을 한 뒤 제출합니다. 꼬리의 상당 부분이 이미 지나서 X가 36.0 ms입니다.
- leaky 입력 큐(깊이 1)는 GPU 제출 시점에 가장 최근 프레임을 넘깁니다. 도착 간격(33 ms)과 장치 주기(41 ms)가 맞물리지 않아, 제출을 조금 늦춰도 프레임이 도착 후 기다린 시간은 거의 같습니다(도착 → 제출 약 17 ms, off가 0.3 ms 김). 그래서 X가 줄어든 만큼 도착 기준 지연이 줄어듭니다.
- 처리량은 표시 off가 0.09 FPS 높습니다. G의 긴 꼬리(표시 on에서 G ≥ 8 ms가 389프레임)가 가끔 장치 주기를 늘리는 것으로 보입니다(확인하지 않음).

## 4. 주의·남은 일

- **G 동안 호스트가 하는 일은 가르지 않았습니다.** 표시 쪽 처리(표시 변환 제출, 프레임 콜백 대기 등)로 추정합니다.
- **꼬리 8.35 ms는 `hailortcli` 단독 측정값입니다.** 파이프라인 안에서 장치 꼬리를 직접 재지 않았습니다. 표시 off의 X − 표시 on의 X(4.4 ms)가 G 차이(4.4 ms)와 같은 것은 이 해석과 맞습니다.
- **표시 off에서 제출을 일부러 늦추면 도착 기준 지연이 같은 원리로 줄 수 있습니다.** 이번 측정으로 보면 처리량은 거의 그대로입니다. 구현과 측정은 하지 않았고, 운용점 변경이라 사용자 결정 사항입니다.
- GPU 전처리 시간 `gpu_ms_mean`(표시 on 0.161 대 off 1.758 ms)은 [`o2_mot20_02`](../o2_mot20_02/summary.md) §4에 적은 표시 on 계측 현상(값이 비정상적으로 작게 나옴)이라 인용하지 않습니다.
- 논문 1152행은 아직 고치지 않았습니다(사용자 확인 대상).
