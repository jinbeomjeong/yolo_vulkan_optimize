# Model Zoo yolov11n·s HEF의 고정비와 처리량 — 기본 모델 m과 비교 (후속 5번)

후속 5번(사용자 승인 2026-11-06, 착수 결정 2026-11-09)의 회차입니다. 같은 Model Zoo의 작은 모델 두 개를 기본 모델 `yolov11m.hef`와 같은 방법으로 쟀습니다. 측정 방법은 상위 [`hef_context_profile`](../../summary.md)과 같습니다.
- **분석만 했습니다.** 기본 모델은 m 그대로이고, n·s는 `yvo_pipeline`에 넣어 보지 않았습니다. 모델 교체는 사용자 결정입니다.

**결론부터 적습니다.**
- 세 HEF는 입출력 구성이 같고 컨텍스트 수만 다릅니다(n·s 2개, m 4개).
- 배치 하나의 NPU 시간은 고정비 + 프레임당 몫으로 나뉩니다. 고정비는 n 5.7, s 11.5, m 24.6 ms이고, 프레임당 몫은 n 3.9, s 7.8, m 16.2 ms입니다.
- 배치 1에서 고정비 비중은 세 모델 모두 약 60%로 같습니다.
- 확정 구성(O2, 배치 1)에서 s를 쓰면 NPU 주기가 40.9 → 19.2 ms로 줄어 52 FPS가 됩니다. 30 fps 입력을 버리지 않고 처리할 수 있는 처리량입니다. 대가는 Model Zoo 표의 하드웨어 mAP 49.9 → 45.1입니다(이 보드에서 재지 않음).

## 1. 측정 조건

| 항목 | 값 |
|---|---|
| 날짜 | 2026-11-09 17:23~17:43 |
| 하드웨어 | Raspberry Pi CM5 + Hailo-8, PCIe 8.0 GT/s x1(Gen3) |
| 버전 | HailoRT·FW·`hailo_pci` 4.24.0, `hailortcli` 4.24.0 |
| 모델 | Hailo Model Zoo **v2.19.0** `Compiled/hailo8/yolov11{n,s,m}.hef`. sha256은 [`env_bench.txt`](env_bench.txt)에 있습니다. 기존 `03_models/yolov11m.hef`는 v2.19.0·v2.20.0의 m과 바이트까지 같습니다. v2.19.0은 DFC 3.34.0·HailoRT 4.24.0 조합입니다(Model Zoo `README.rst` 배지) |
| 정적 구성 | [`evidence/parse_hef_*.txt`](evidence/parse_hef_yolov11n.txt). 세 HEF 모두 입력 UINT8 NHCW 640×640×3(장치 스트림)과 raw 출력 6개(80×80·40×40·20×24 × {80, 64})입니다. 다른 것은 컨텍스트 수뿐입니다 |
| 입력 | `hailortcli`가 만든 입력입니다. 영상·전처리·후처리·표시는 없습니다 |
| 명령 | 실행 폴더를 CWD로 삼아 다음 두 가지를 돌렸습니다. ① `hailortcli run2 -t 3 -m raw_async measure-fw-actions --output-path fwactions.json set-net <hef> --batch-size B`(B = 1·2·8). ② `hailortcli benchmark --batch-size B -t 10 --csv benchmark.csv <hef>`(B = 1·8) |
| 반복 | 3회. 반복이 바깥 루프, 모델 n → s → m, 그 안에 B입니다. 드라이버는 [`evidence/run_zoo_fw_actions.sh`](evidence/run_zoo_fw_actions.sh), 실행 기록은 [`evidence/run_log.txt`](evidence/run_log.txt)입니다 |
| 냉각 | 실행마다 게이트를 두었습니다. Hailo 칩이 75 °C를 넘거나 직전 로그에 경고가 있으면 65 °C 이하로 식히고, SoC는 75 °C 이하에서 시작합니다. 실행 직전 칩 온도는 56.0~74.9 °C였습니다. s의 2·3회차 fw-actions 실행은 n 바로 뒤라 74.5~74.9 °C에서 시작했지만, 값이 1회차 fw-actions(69.1~69.8 °C 시작)와 같아 영향은 보이지 않습니다. 같은 회차 s의 benchmark는 64.7~69.7 °C에서 시작했습니다 |
| 거버너 | ondemand. NPU 시간과 FPS만 보는 측정이라 고정하지 않았습니다(CPU ms는 보지 않음) |
| 신뢰성 | 45회 모두 rc 0, `hailort.log` warning 0·error 0 |
| 분석 | `ANALYZE_ONLY=1`로 드라이버를 돌렸습니다. 모델마다 기존 [`context_table.py`](../../evidence/context_table.py)와 [`fw_actions_breakdown.py`](../../../../06_tools/python/fw_actions_breakdown.py)를 고치지 않고 실행했고, 출력 전문은 [`evidence/analysis.txt`](evidence/analysis.txt)에 있습니다 |

## 2. 표

### 2.1 NPU 처리량과 지연 — 3회 중앙값 (최소~최대)

출처: [`results.csv`](results.csv), 실행별 값은 [`results_raw_fwactions.csv`](results_raw_fwactions.csv)와 [`results_raw_benchmark.csv`](results_raw_benchmark.csv)

| 모델 | 컨텍스트 | FPS B=1 | FPS B=2 | FPS B=8 | 장치 지연 B=1 ms | 장치 지연 B=8 ms |
|---|---|---|---|---|---|---|
| yolov11n | 2 | 104.89 (104.89~104.89) | 147.51 (147.29~147.51) | 216.55 (216.54~216.83) | 7.75 (7.75~7.75) | 22.14 (22.13~22.14) |
| yolov11s | 2 | 51.95 (51.63~51.95) | 73.62 (73.60~73.62) | 106.89 (106.89~106.89) | 15.20 (15.20~15.21) | 47.95 (47.95~47.96) |
| yolov11m | 4 | 24.31 (24.31~24.33) | 34.63 (34.63~34.66) | 50.61 (50.61~50.61) | 32.51 (32.51~32.52) | 126.26 (126.26~126.27) |

- FPS는 `run2` stdout 값입니다. `benchmark`의 `hw_only` FPS도 같습니다(B=1 104.76·51.97·24.31, B=8 217.47·107.95·51.03).
- 장치 지연은 `benchmark`의 `hw_latency`입니다.

### 2.2 배치 하나의 NPU 시간 분해 — `span(B) = a + b·B`, ms

출처: [`evidence/analysis.txt`](evidence/analysis.txt)의 `context_table.py` 출력. B별 값은 3회 중앙값이고, 3회 범위의 폭은 0.04 ms 이하입니다.

| 모델 | 배치 구간 B=1 | B=2 | B=8 | 고정 a | 프레임당 b | B=1에서 a 비중 | 컨텍스트별 a |
|---|---|---|---|---|---|---|---|
| yolov11n | 9.51 | 13.53 | 36.70 | **5.69** | **3.88** | 60% | dyn0 3.10 · dyn1 2.59 |
| yolov11s | 19.23 | 27.02 | 73.66 | **11.46** | **7.78** | 60% | dyn0 3.86 · dyn1 7.60 |
| yolov11m | 40.92 | 57.09 | 154.61 | **24.63** | **16.25** | 60% | dyn0 4.40 · dyn1 7.25 · dyn2 2.95 · dyn3 10.03 |

- m은 2026-10-12 상위 회차(40.92 / 57.06 / 154.62 ms, a 24.79, b 16.23)를 재현합니다.
- 활성화 때 한 번 드는 preliminary 컨텍스트는 n 3.33, s 9.24, m 11.06 ms이고, 주기에는 들어가지 않습니다.

### 2.3 정확도 — Model Zoo 공식 표 (이 보드에서 재지 않음)

출처: Hailo Model Zoo v2.19.0 `docs/public_models/HAILO8/HAILO8_object_detection.rst` 사본 [`evidence/model_zoo_v2.19.0_HAILO8_object_detection.rst`](evidence/model_zoo_v2.19.0_HAILO8_object_detection.rst)

| 모델 | float mAP | Hardware mAP | Params (M) | OPS (G) | 표의 FPS B=1 / B=8 |
|---|---|---|---|---|---|
| yolov11n | 39.0 | 37.5 | 2.6 | 6.55 | 185 / 541 |
| yolov11s | 46.3 | 45.1 | 9.4 | 21.6 | 111 / 303 |
| yolov11m | 51.1 | 49.9 | 20.1 | 68.1 | 50.2 / 102 |

- 연산량(OPS)은 n 6.55 → s 21.6 → m 68.1 G로 약 3배씩 늘지만, 이 보드의 프레임당 NPU 몫 b는 약 2배씩 늘었습니다(§2.2).
- **표의 FPS는 이 보드 값과 비교하지 않습니다.** 표가 어떤 호스트와 PCIe 링크에서 잰 값인지 적혀 있지 않습니다. 이 보드(Gen3 x1)에서는 B=1에서 n 105 / s 52 / m 24 FPS, B=8에서 217 / 108 / 51 FPS입니다. m의 B=1(24.3 대 50.2)부터 차이가 큽니다.

## 3. 결론

**1. 작은 모델일수록 고정비와 프레임당 몫이 함께 줄어, 비중은 그대로입니다.**
- 고정비 a는 m 24.6 → s 11.5 → n 5.7 ms이고, 프레임당 몫 b는 16.2 → 7.8 → 3.9 ms입니다. 둘 다 모델이 한 단계 작아질 때마다 약 절반이 됩니다.
- 그래서 배치 1에서 고정비 비중은 세 모델 모두 60%입니다. 컨텍스트가 4개에서 2개로 줄어도 비중은 낮아지지 않았습니다.
- 상위 회차는 m의 고정비 40%가 dynamic_3에 몰려 있어 "컨텍스트가 적은 HEF"를 줄일 후보로 보았습니다. 컨텍스트 수만으로 그 몫이 사라지지는 않습니다. 같은 모델을 컨텍스트 수만 바꿔 컴파일한 비교가 아니므로, 고정비가 컨텍스트 수와 모델 크기 중 어디에서 오는지는 이 측정으로 가를 수 없습니다.

**2. 확정 구성(O2, 배치 1)에 넣으면 NPU 주기가 m의 약 1/2(s)·1/4(n)이 됩니다.**
- O2에서 처리량은 NPU 주기(m 40.9 ms ≈ 24.3 FPS)가 정하고, 30 fps 입력의 18.9%를 버립니다([`technique_ablation/runs/o2_mot20_02`](../../../technique_ablation/runs/o2_mot20_02/summary.md) 표).
- s는 B=1에서 52 FPS(주기 19.2 ms), n은 105 FPS(9.5 ms)입니다. 둘 다 30 fps 입력을 장치 처리량 기준으로는 버리지 않고 받을 수 있습니다.
- NPU 지연은 m 32.5 → s 15.2 → n 7.8 ms(장치 지연, B=1)입니다. 파이프라인 전체 지연이 얼마나 주는지는 재지 않았습니다.

**3. 대가는 정확도입니다.** Model Zoo 표의 하드웨어 mAP는 m 49.9, s 45.1, n 37.5입니다. 이 프로젝트 영상에서 검출이 얼마나 달라지는지는 재지 않았습니다.

## 4. 주의·남은 일

- **파이프라인 투입은 확인하지 않았습니다.** 입출력 형상과 버퍼 크기가 m과 같아, 코드 구조상으로는 `yvo_pipeline --hef`만 바꾸면 됩니다(hailo-expert 코드 판독: `01_src/hailo/hailo_raw_runner.cpp:75-136`이 텐서를 이름이 아니라 형상으로 분류하고, 양자화 값은 HEF에서 읽습니다). 검출 결과가 맞는지, 셰이더나 테스트에 m 고유 상수가 있는지는 보지 않았습니다.
- **정확도는 이 보드에서 재지 않았습니다.** §2.3은 Model Zoo가 발표한 COCO 값입니다.
- **파이프라인 전체 지연·CPU는 재지 않았습니다.** 장치 시간만 쟀습니다.
- `fps_json`(fw-actions JSON 안의 FPS)은 m B=8에서 47.95와 50.62가 섞여 나옵니다(상위 회차와 같은 현상). FPS는 stdout 값을 씁니다.
- 결정: 기본 모델은 바꾸지 않는다. 모델·HEF 교체는 사용자 결정이며 이번 범위 밖이다 (사용자 결정, 2026-11-06).
