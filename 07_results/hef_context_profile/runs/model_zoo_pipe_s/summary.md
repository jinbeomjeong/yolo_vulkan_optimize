# Model Zoo yolov11n·s를 파이프라인에 넣은 결과 — 확정 구성(O2), 표시 off

[`model_zoo_small`](../model_zoo_small/summary.md)(2026-11-09)의 후속 회차입니다. 그 회차는 장치 측정만 하고 "파이프라인 투입은 확인하지 않았다"고 적었습니다. 이번에는 `yvo_pipeline --hef`만 바꿔 같은 운용점에서 세 모델을 재고, 검출을 m과 대조했습니다. 결정: n·s 둘 다 파이프라인으로 확인한다. 기본 모델은 바꾸지 않는다 (사용자 결정, 2026-11-09).
- yolov11s 결과는 이 폴더에 있습니다. n은 [`../model_zoo_pipe_n/`](../model_zoo_pipe_n/summary.md), 대조용 m은 [`../model_zoo_pipe_m/`](../model_zoo_pipe_m/summary.md)에 있습니다.
- 세 모델의 비교는 이 문서 한 곳에서 합니다.

**결론부터 적습니다.**
- 코드를 바꾸지 않고 `--hef`만 바꿔도 n·s가 정상으로 돕니다. 출력 상자의 좌표도 맞습니다.
- n·s는 30 fps 입력을 하나도 버리지 않습니다. 도착 기준 지연 중앙값은 n 13.8, s 21.2, m 58.2 ms입니다.
- 검출은 m과 꽤 다릅니다. 같은 프레임에서 IoU 0.5로 m과 짝지어지는 비율이 s 0.53, n 0.38이고, 사람을 s 45%, n 80% 덜 잡습니다. m끼리는 1.0000입니다.

## 1. 측정 조건

| 항목 | 값 |
|---|---|
| 날짜 | 2026-11-09 21:27~21:38 |
| 보드 | Raspberry Pi CM5, Hailo-8 PCIe 8.0 GT/s x1(Gen3), V3D 1150 MHz, 커널 6.18.50+rpt-rpi-2712, HailoRT 4.24.0 |
| 바이너리 | dc447c5 빌드(md5 070a8687). 저장소 밖에 고정한 사본입니다. `--submit-delay-ms`는 주지 않았습니다(기본 0) |
| 모델 | Model Zoo v2.19.0 hailo8 `03_models/yolov11{n,s,m}.hef`(sha256 앞 16자리 9b6fb370b660b331 / 5a9dbb513d2e3143 / d5716578f0d18b01, 출처는 `model_zoo_small`) |
| 드라이버 | 모델마다 `HEF=03_models/yolov11<m>.hef ONLY='^display_off$' RUNS=3 bash 06_tools/bench/technique_ablation_bench.sh`, 거버너 감싸개 [`evidence/run_model_zoo_pipe.sh`](evidence/run_model_zoo_pipe.sh), 실행 기록 [`evidence/run_log.txt`](evidence/run_log.txt) |
| 인자 | 확정 구성 O2, 표시 off: `--mode dmabuf --decoder hw --batch 1 --slots 1 --input-pace source --input-queue 1 --layout nhcw --postproc gpu --nms gpu --display off`, 기록 on(`detections.jsonl`), `YVO_DEC_THREADS=1` |
| 입력 | `04_data/videos/video_h265.mp4` 전체(1080p30, 도착 1,816프레임) |
| 반복 | 모델마다 3회. 매 실행 전 Hailo ≤ 65 °C로 냉각했습니다. 측정 순서는 n → s → m입니다 |
| 거버너 | 측정 동안만 performance로 두고, trap으로 ondemand로 되돌렸습니다 |
| 신뢰성 | 9회 모두 rc 0, `hailort.log` warning·error 0 |

**어느 HEF로 돈 실행인지:** 벤치의 `env_bench.txt`에는 HEF 경로가 남지 않았습니다(아래 §4). 그래도 두 가지로 확인됩니다.
- HEF 적재 시간(`hailort.log`, 1회차): n 20.7, s 29.0, m 42.8 ms입니다.
- NPU 지연: n 7.85, s 15.22, m 38.34 ms입니다. `model_zoo_small`의 장치 지연(B=1 n 7.75, s 15.20, m 32.51 ms)과 순서가 맞고, m 차이는 slots 1의 측정 구간 효과입니다([`npu_latency_window`](../../../technique_ablation/runs/npu_latency_window/summary.md)).

## 2. 표

### 2.1 처리량·지연·CPU — 3회 중앙값 (최소~최대)

출처: 각 폴더의 `results_raw.csv`([n](../model_zoo_pipe_n/results_raw.csv), [s](results_raw.csv), [m](../model_zoo_pipe_m/results_raw.csv))

| 항목 | yolov11n | yolov11s | yolov11m (기본) |
|---|---|---|---|
| `fps` | 30.00 (30.00~30.00) | 30.00 (30.00~30.00) | 24.42 (24.41~24.42) |
| 입력 버림 | 0% | 0% | 18.6% |
| `e2e_arr` 중앙값 ms | **13.76** (13.75~13.86) | **21.15** (21.14~21.18) | 58.20 (58.19~58.24) |
| `e2e_arr` p90 ms | 14.71 | 22.10 | 71.81 |
| `npu_ms_mean` | 7.85 (7.84~7.86) | 15.22 (15.22~15.23) | 38.34 (38.33~38.36) |
| `cpu_ms_per_frame` | 0.671 (0.657~0.747) | 0.630 (0.626~0.671) | 0.728 (0.726~0.771) |
| 검출 / 실행 | 5,943 (3회 같음) | 8,901 (3회 같음) | 7,698 (7,696~7,711) |

- n·s는 처리 프레임 수가 m보다 23% 많으므로(1,816 대 1,479), 검출 총수는 그대로 비교하지 않습니다.
- CPU는 처리 프레임당 값입니다. n·s는 큐 대기가 없어 폴링이 적고, m과 조건이 다릅니다.

### 2.2 검출 대조 — m 기준

출처: [`evidence/compare_vs_m_run1.txt`](evidence/compare_vs_m_run1.txt)(s), [`../model_zoo_pipe_n/evidence/compare_vs_m_run1.txt`](../model_zoo_pipe_n/evidence/compare_vs_m_run1.txt)(n). `06_tools/python/compare_detections.py`로 1회차끼리 대조했습니다. 두 실행이 모두 처리한 프레임만 비교합니다(1,479프레임).

| 대조 | 검출 수(m / 대상) | IoU 0.5 일치율 | IoU 0.7 | IoU 0.9 | 짝 지은 상자의 좌표 오차 p50 / p90 px(IoU 0.5) |
|---|---|---|---|---|---|
| m 1회차 대 m 2회차(기준) | 5,943 / 5,943(공통 1,144프레임) | 1.0000 | 1.0000 | 1.0000 | — |
| m 대 s | 7,698 / 7,247(−5.9%) | 0.5348 | 0.4848 | 0.2634 | 4.4 / 12.1 |
| m 대 n | 7,698 / 4,841(−37.1%) | 0.3757 | 0.3296 | 0.1608 | 7.4 / 16.0 |

일치율 = 짝지어진 수 ÷ max(m 수, 대상 수)입니다. 클래스별 차이(m 대비)는 다음과 같습니다.

| 클래스 | m | s | n |
|---|---|---|---|
| car | 4,691 | 4,326(−7.8%) | 3,205(−31.7%) |
| traffic light | 1,347 | 1,750(+29.9%) | 1,019(−24.4%) |
| person | 1,201 | 656(−45.4%) | 242(−79.9%) |
| truck | 188 | 344(+83.0%) | 205(+9.0%) |
| bicycle | 124 | 78(−37.1%) | 23(−81.5%) |

## 3. 결론

**1. 파이프라인은 HEF만 바꿔 n·s를 받아들입니다.** 세 HEF의 입출력 구성이 같고, NPU 실행기(`01_src/hailo/hailo_raw_runner.cpp`)가 출력 텐서를 이름이 아니라 형상으로 분류하며 양자화 값을 HEF에서 읽기 때문입니다. s의 첫 프레임 출력 그림([`raw/display_off_run1/frame00000.png`](raw/display_off_run1/frame00000.png))에서도 상자 두 개(car 0.89 등)가 실제 차 위치에 맞는 것을 눈으로 확인했습니다.

**2. 처리량과 지연은 크게 좋아집니다.**
- n·s는 장치 주기(9.5·19.2 ms)가 입력 간격(33.3 ms)보다 짧아, 30 fps 입력을 모두 처리합니다. 입력 큐에서 거의 기다리지 않습니다.
- 도착 기준 지연은 m 58.2 ms에서 s 21.2(−64%), n 13.8 ms(−76%)로 줄어듭니다.

**3. 대가는 검출입니다.** m과 같은 상자를 내는 비율이 s 약 절반, n 약 3분의 1입니다. 특히 사람과 자전거를 크게 덜 잡습니다. Model Zoo 표의 하드웨어 mAP(m 49.9, s 45.1, n 37.5)와 같은 방향입니다. 다만 이 대조는 m을 기준으로 삼은 것이지 정답(라벨)과 비교한 정확도가 아닙니다.

## 4. 주의·남은 일

- **기본 모델은 바꾸지 않았습니다.** 모델 교체는 사용자 결정 사항입니다.
- **정확도는 재지 않았습니다.** 이 영상에는 정답 라벨이 없습니다. m 대조는 "m과 얼마나 같은가"입니다.
- **표시 on은 재지 않았습니다.** n·s는 30 fps를 다 처리하므로 표시 박자와 지연이 m과 다르게 나올 수 있습니다.
- **벤치 기록의 빈틈:** `06_tools/bench/technique_ablation_bench.sh`의 `env_bench.txt`에 `HEF` 경로가 남지 않습니다. 이번 회차는 감싸개 스크립트와 적재 시간·NPU 지연으로 모델을 확인했습니다. 벤치가 HEF 경로와 md5를 적게 고치는 것이 좋습니다(벤치 담당 수정 사항).
