# 표시 통합 e2e 재측정 — V3D 1150 MHz · 현재 기본값 (2026-10-16)

모 실험 [`../../summary.md`](../../summary.md)(2026-09-20, V3D 960 MHz)와 같은 벤치·설정·집계로 다시 쟀다.
목적은 [`09_diagrams/pipeline_overview.graphml`](../../../../09_diagrams/pipeline_overview.graphml)의 GPU 구간 수치를 한 조건으로 맞추는 것이다.
지금 그 그림은 `[960 MHz]` 노드 4개와 `[1150 MHz]` NMS 노드가 섞여 있다.

## 1. 측정 조건

- 날짜: 2026-10-16 01:12~01:26 (KST)
- 하드웨어: Raspberry Pi 5, Hailo-8, PCIe **Gen3 x1**(8.0 GT/s)
  - V3D **1150 MHz**: 부트로더 2026-09-25 기본값이다. 측정 전후 `vcgencmd get_config v3d_freq` = 1150, `measure_clock v3d` = 1150.0 MHz
  - ARM 2.4 GHz
  - 표시는 HDMI-A-1 1920×1080@60(labwc)
- 소프트웨어
  - kernel 6.18.50+rpt-rpi-2712, HailoRT 4.24.0, mesa-vulkan-drivers 26.2.2, libavcodec61 7.1.5 rpt2, labwc 0.20.2
  - 전부 [`env_bench.txt`](env_bench.txt)에 있다.
- 코드: **HEAD `445bbfe` + 미커밋 9파일**(`01_src/pipeline/*`, `hailo_runner.*`, `test_pipeline_logic.cpp`)
  - 출처는 [`verification/source.txt`](verification/source.txt)에 있다(diff stat과 sha256).
  - 이 빌드로 verify 1~4단계를 측정 직전에 통과했다(§5).
  - 바이너리 md5 `3dda5b0f…`는 측정 전후로 같다.
- 명령: `RUNS=3 FRAMES=900 OUT_DIR=07_results/display_e2e/runs/v3d1150_20261016 bash 06_tools/bench/display_bench.sh`
  - 설정 4개와 M5a는 모 실험과 같다.
  - 실행 순서는 run 바깥, 설정 안쪽이다.
- 모 실험 대비 바뀐 조건은 **현재 기본값 전부**다.
  - V3D 960 → 1150 MHz
  - FFmpeg 프레임 스레드 자동(5) → 1
  - `--nms` CPU → GPU(`efficient_nms.comp`)
  - `yolo_decode.comp` argmax 지름길(2026-10-11)
  - 그 밖의 리팩터
  - 따라서 **아래 차이를 클럭 효과만으로 읽지 않는다**(§3).
- 모델·입력: `03_models/yolov11m.hef`, `04_data/videos/video_h265.mp4` 앞 900프레임
- 신뢰성
  - 12회 모두 `hailort.log`의 warning 0, error 0이다.
  - 검출은 모든 실행에서 3369개, 표시 드롭은 0, slot timeout은 0이다.
  - Hailo 칩이 75 °C를 넘으면 실행 전에 냉각했다(85~110 s, 5회). 최고 온도는 81.1 °C로, 클럭 저하가 시작되는 구간(약 104 °C)보다 낮다.
  - `throttled=0xe0000`은 하위 4비트가 0이므로 현재 스로틀링이 아니다.

## 2. 결과

### 2.1 GPU 구간 (`disp_asap`, 중앙값) — 모 실험 §3.1과 같은 집계

집계 방법은 실행마다 stdout의 구간 `median`(p90)을 읽고 3회의 중앙값을 낸 것이다.
같은 방법으로 모 실험 raw를 다시 집계하면 2.088 / 0.626 / 4.153 / 0.128이 그대로 재현된다.

| 구간 | 모 실험 (960 MHz) | **이번 (1150 MHz)** | 변화 | 이번 p90 | 실행별 median |
|---|---:|---:|---:|---:|---|
| 전처리 `sand_letterbox.comp` | 2.088 | **1.812** | −13.2% | 1.885 | 1.812 · 1.810 · 1.813 |
| 후처리 `yolo_decode.comp` (pass 1) | 0.626 | **0.400** | −36.1% | 0.470 | 0.407 · 0.399 · 0.400 |
| NMS `efficient_nms.comp` | — (CPU NMS) | **0.038** | — | 0.056 | 0.0385 · 0.0382 · 0.0375 |
| 표시 변환 `display_convert.comp` | 4.153 | **3.662** | −11.8% | 3.788 | 3.663 · 3.659 · 3.662 |
| 오버레이 `overlay_boxes.comp` | 0.128 | **0.124** | −3.1% | 0.143 | 0.126 · 0.124 · 0.123 |
| **합** | 6.995 (4구간) | **6.036 (5구간)** | −13.7% | 6.342 | |

주기 28.6 ms 대비 합은 median **21.1%**, p90 22.2%다. 모 실험은 24.5% / 25.4%였다.

같은 회차의 다른 설정도 참고로 적는다(중앙값, ms). 설정마다 구간 시간이 조금씩 다르므로 `disp_asap`과 섞지 않는다.

| 설정 | 전처리 | 후처리 | NMS | 표시 변환 | 오버레이 |
|---|---:|---:|---:|---:|---:|
| `disp_off` | 1.827 | 0.453 | 0.049 | — | — |
| `disp_source` | 1.790 | 0.447 | 0.047 | 3.605 | 0.120 |

### 2.2 처리량·CPU·glass-to-glass (중앙값, 3회)

| 설정 | fps | cpu_ms_per_frame | glass median (ms) | glass p90 (ms) | 검출 | 드롭 |
|---|---:|---:|---:|---:|---:|---:|
| `reg_host_off` | 34.96 | 3.014 | — | — | 3369 | — |
| `disp_off` | 34.99 | 0.947 | — | — | 3369 | — |
| `disp_asap` | 34.87 | 1.626 | 127.91 | 136.83 | 3369 | 0 |
| `disp_source` | 29.91 | 1.832 | 100.10 | 116.04 | 3369 | 0 |

- 모 실험(960 MHz·자동 5스레드·CPU NMS)은 `disp_asap` 34.87 fps · CPU 1.842 · glass 130.75, `disp_source` 29.91 fps · CPU 2.179 · glass 101.20 / p90 118.36이었다.
- glass는 실행마다 stdout의 `glass-to-glass` median·p90을 3회 중앙값으로 낸 것이다. `results.csv`의 `e2e_glass_mean_median`(평균의 중앙값)과는 다른 값이다.

### 2.3 M5a (NPU 쉼, `yvo_display_probe --probe convert`, 300프레임 1회)

- 1080p 1:1 변환: shader median 3.685 ms, p90 3.939 ms
- 화면 위 실측: 변환 3.646 ms, 오버레이 0.193 ms(합성 박스 5개)
- 900p는 6.479 ms, 720p는 4.485 ms다. 해상도를 낮추면 오히려 비싸지는 순서는 그대로다.
- 원문은 [`raw/m5a_run1/m5a_baseline.txt`](raw/m5a_run1/m5a_baseline.txt)에 있다.

## 3. 결론

- **그림의 GPU 구간을 한 조건(1150 MHz·현재 기본값)으로 맞출 값이 생겼다.** 같은 실행의 `disp_asap`에서 다섯 구간을 모두 얻었다.
- **클럭 비율만으로 설명되는 구간은 일부뿐이다.**
  - 클럭 비율(960/1150)만큼 빨라지면 −16.5%다.
  - 전처리 −13.2%와 표시 변환 −11.8%는 그보다 작다. 2026-09-29의 전처리 −14%와 비슷한 크기다. 메모리 대역 몫은 클럭을 따라 줄지 않는 것으로 본다(**추정**).
  - 후처리 −36%는 클럭만으로 설명되지 않는다. 2026-10-11 argmax 지름길의 효과(GPU 후처리 −32~−35%, [`gpu_nms_e2e/runs/argmax_shortcut_mot20_05`](../../../gpu_nms_e2e/runs/argmax_shortcut_mot20_05/summary.md))가 함께 들어 있다.
  - 오버레이는 검출 수에 비례하는 작은 패스라 차이가 작다.
- **NMS 값은 측정 조건에 따라 다르다.** `disp_asap` 0.038 ms는 표시 off(`disp_off` 0.049, `gpu_nms_e2e` b2s4 0.0486)보다 작다. 그림에 넣을 때 출처 조건을 함께 적는다.
- **처리량과 검출은 그대로다.** FPS는 NPU·페이싱이 정하고, 검출 3369개는 모 실험과 같다. CPU ms/frame과 glass가 줄어든 것은 디코더 1스레드와 GPU NMS 몫이 섞인 결과다. 클럭 효과로 주장하지 않는다.

## 4. 주의·남은 일

- 이 회차는 **미커밋 코드가 포함된 빌드**다. 그 코드가 커밋되면 커밋 해시를 이 절에 덧붙인다(파일 내용은 고치지 않는다는 원칙에 따라 추가만 한다).
  - (2026-10-16 추가) 그 코드는 커밋 **`2f2d3d0`**(메인 스레드 이벤트 대기 · 평면 matched 슬롯 하한 · HailoRunner 종료 안전화)으로 들어갔다.
    `git diff 2f2d3d0~1 2f2d3d0 -- 01_src 05_tests | sha256sum`이 [`verification/source.txt`](verification/source.txt)의 `29253f8e…`와 같다.
    따라서 이 회차의 코드는 `2f2d3d0`이다(그 사이 커밋 `9313259`는 문서·결과만 바꿨다).
- V3D 클럭은 `env_bench.txt`에 자동으로 남지 않는다(벤치가 기록하지 않음). 값은 §1에 적은 대로 측정 전후 `vcgencmd`로 확인했다.
- `results.csv`의 `gpu_ms_mean_median`은 배치 모드에서 0이 섞인 평균이라 구간 대표값으로 쓰지 않는다(모 실험도 같음).
- 결정: 다이어그램 GPU 구간을 이 회차 `disp_asap` 값으로 갱신하기 위해 재측정했다 (사용자 결정, 2026-10-16).

## 5. 검증 (`verification/`)

| 단계 | 결과 | 로그 |
|---|---|---|
| 빌드 | 34단계, 경고·에러 0줄 | `build.log` |
| ctest | 184/184 통과 (261 s) | `ctest.log` |
| 검증 레이어 | `vk validation errors 0`(warnings 2 = 로더 강제 활성 알림), 200프레임 검출 606, hailort warning/error 0 | `validation_dmabuf_nhcw_200f.log`, `hailort_validation.log` |
| ASan/UBSan | `build-asan`(`YVO_SANITIZE=ON`) 경고 0. GPU·host 후처리 각 100프레임과 `test_pipeline_logic` 75개에서 ASan·LSan·UBSan 보고 0 | `asan_*.log`, `build_asan.log`, `hailort_asan.log` |
