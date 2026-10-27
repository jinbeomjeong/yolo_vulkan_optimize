# 회차 vblank_phase_20261027 — 평면 경로 늦은 프레임의 원인과 커밋 게이트 여유 스윕

[상위 실험](../../summary.md) §5·주의 3번의 "source 페이싱 늦은 프레임 약 26%(vblank 위상 추정, 미검증)"을 검증한 회차다.
코드 `ee445a3`에서 프레임별 커밋 시각(`commit_ms`)과 presentation vblank 순번(`present_seq`)을 frame-log에 남기고,
커밋 게이트(`1000/fps − margin`, 기본 margin 8 ms)의 여유를 진단 옵션 `--display-commit-margin-ms`로 바꿔 가며 쟀다.

## 1. 측정 조건

| 항목 | 값 |
|---|---|
| 날짜 | 2026-10-27 22:33~ |
| 하드웨어 | Raspberry Pi 5 + Hailo-8, PCIe **Gen3 x1**, V3D 1150 MHz(부트로더 2026-09-25), HDMI-A-1 1920×1080 60 Hz |
| 소프트웨어 | 커널 6.18.50+rpt-rpi-2712, HailoRT 4.24.0, mesa 26.2.2, labwc 0.20.2, git `ee445a3`(bin md5 `fa30bfb4…`) |
| 모델·입력 | `03_models/yolov11m.hef`, `04_data/videos/video_h265.mp4`(1080p30) |
| 설정 | b2 / slots 6, GPU 후처리·GPU NMS, `--display-sync matched --display-pace source`, `--display-warmup 30` |
| 경로 | `vk_source`(Vulkan 합성, 대조) · `wlp_source_m{8,4,0,12}`(wayland-planes, 커밋 게이트 여유 x ms → 게이트 33.33 − x ms). kms 미실행 |
| 반복 | 900프레임 × 3회, 반복 바깥 루프·설정 안쪽 루프, 매 실행 냉각(`COOL_EVERY=1`) |
| 명령 | `OUT_DIR=<이 폴더> ONLY='source' FRAME_LOG=1 COMMIT_MARGINS="8 4 0 12" SKIP_KMS=1 SLOTS=6 bash 06_tools/bench/display_planes_bench.sh` |
| 신뢰성 | 15회 모두 `hailort.log` warning 0, 벤치 검사 3/3 통과(color_ok·zero-copy·underrun 0) |

환경 스냅샷은 [`env_bench.txt`](env_bench.txt)(시작 때 SoC 68.1 °C, Hailo 59.0 °C, `throttled=0xe0000`은 과거 플래그만).
원자료는 [`results.csv`](results.csv)·[`results_raw.csv`](results_raw.csv)·`raw/<설정>_run<N>/`(`stdout.log`·`hailort.log`·`frame_log.csv`).

## 2. 표

### 2.1 설정별 중앙값(3회 범위)

| 설정 | 게이트 ms | late frames / 간격 | late 비율 | 표시 버림 | decode-to-screen 평균 ms | d2s p90 ms | 표시 간격 중앙 / p90 ms | 앱 CPU ms/f |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `vk_source`(대조) | — | —(판정 없음) | — | 0 | 99.23 [99.11–100.56] | 116.55 | 17.23 / 67.09 | 2.343 |
| `wlp_source_m12` | 21.33 | 308 [289–320] / 868 | **35.5%** | 1 [0–1] | **96.78** [96.64–97.21] | 110.87 | 33.33 / 50.64 | 1.949 |
| `wlp_source_m8`(기본) | 25.33 | 216 [187–222] / 868 | **24.9%** | 1 | 100.86 [99.93–100.92] | 114.60 | 33.33 / 50.00 | 1.939 |
| `wlp_source_m4` | 29.33 | 129 [71–132] / 867 | **14.9%** | 2 [1–2] | 104.23 [104.15–105.25] | 116.30 | 33.33 / 50.00 | 1.924 |
| `wlp_source_m0` | 33.33 | 18 [18–20] / 859 | **2.1%** | 10 [9–11] | **125.40** [125.32–125.93] | 132.36 | 33.33 / 33.34 | 1.923 |

late 비율 = late frames ÷ 판정한 간격 수(warmup 30 뒤). 간격 수는 [`evidence/phase_analysis.txt`](evidence/phase_analysis.txt)의 `gaps`.

### 2.2 늦은 간격의 원인 분류 — [`evidence/phase_analysis.py`](evidence/phase_analysis.py) → [`evidence/phase_analysis.txt`](evidence/phase_analysis.txt)

늦은 간격(vblank 순번 차 > 2)마다 그 프레임이 **놓친 vblank 전에 준비됐는지**(`complete_ms` + 정상 프레임의 커밋 → 화면 중앙값 < 기대 표시 시각)를 본다.
준비됐는데 늦었으면 `gate`(게이트가 붙잡음), 준비가 늦었으면 `ready`(위쪽 단계 도착이 늦음)로 센다.

| 설정 | late: gate | late: ready | 게이트 대기(`commit_ms − complete_ms`) 중앙 / p90 ms | 커밋 위상 분포(vblank 6등분) |
|---|---:|---:|---:|---|
| `m12` | 0~1 | 289~319 | 3.1~3.8 / 21.9~22.4 | 고름 |
| `m8` | 0 | 187~222 | 10.1 / 26.3 | 고름 |
| `m4` | 0 | 71~132 | 17.9 / 30.5~30.7 | 고름 |
| `m0` | 16~19 | 1~2 | 39.8~40.7 / 57.0~57.3 | 고름 |

## 3. 결론

1. **기본값(m8)의 늦은 프레임 24.9%는 이전 기록(약 26%)을 재현한다.** 그리고 그 늦은 간격은 **모두 `ready`**(187~222건 중 gate 0)다 —
   프레임이 놓친 vblank 전에 아직 후처리를 끝내지 못했다. 게이트가 준비된 프레임을 붙잡아서 생기는 것이 아니다.
2. **"앱 페이싱이 vblank 위상에 묶여 있지 않은 탓"이라는 기존 추정은 원인으로 맞지 않는다.** 커밋 위상은 모든 설정에서 vblank 안에 고르게 퍼져 있는데,
   늦은 비율은 2.1%~35.5%로 크게 달라진다. 늦음을 정하는 것은 위상이 아니라 **표시할 프레임이 제때 준비돼 있는가**이고, 그것을 정하는 손잡이가 게이트 여유다.
3. 기제(해석): batch 2라 검출은 약 66.7 ms마다 짝으로 끝난다. 게이트가 짧으면(m8·m12) 프레임이 끝나자마자 나가므로 표시 박자가 **NPU 짝 완료 박자를 그대로 물려받아**
   짝 사이의 긴 틈에서 한 vblank 밀린다. 게이트가 source 간격과 같으면(m0) 프레임이 대기열에 쌓여(게이트 대기 중앙 40 ms) 다음 vblank에 늘 내보낼 프레임이 있어
   간격이 33.3 ms로 고르다(p90 33.34).
4. **대가는 지연이다.** m0은 늦은 프레임을 2.1%로 줄이지만 디코드 → 화면 평균이 +24.5 ms(100.86 → 125.40)이고 표시 버림이 1 → 10이다.
   m12는 지연을 −4.1 ms 줄이지만 늦은 프레임이 35.5%다. **고른 박자와 낮은 지연은 이 손잡이로 맞바뀐다** — 기본값 8 ms는 그 중간이다.
5. Vulkan 합성(`vk_source`)은 지연이 m8과 범위가 겹치고(99.23 [99.11–100.56] 대 100.86 [99.93–100.92]) 표시 간격이 17/67 ms로 교대한다(상위 실험 결론과 같다). 평면 경로 m0의 박자가 가장 고르다.

## 4. 주의·남은 일

- `late frames`는 vblank 순번 차 > 2 판정이다(`plane_display.cpp`의 `late_gap_threshold`). 표시 버림이 낀 간격은 허용된다. m0의 버림 10은 대기열이 넘친 몫으로 본다(추정).
- 원인 분류의 "준비됐는가"는 정상 프레임의 커밋 → 화면 중앙값(7.9~10.5 ms)을 문턱으로 쓴 근사다. `show()` 시각은 기록하지 않아 `complete_ms`로 대신했다.
- 한 영상·한 운용점(b2/s6)만 쟀다. b1(O2)에는 적용하지 않았다(b1 평면 경로는 slots에서 지연과 처리량을 맞바꾼다, `input_pacing/runs/display_on`).
- 지연 m4~m0 사이(예: 2 ms)와 "짝의 첫 프레임만 붙잡는" 게이트 같은 대안은 재지 않았다.
- `commit_log_`는 frame-log를 켜지 않아도 커밋마다 쌓인다(약 24 B/프레임, 시간당 약 2.6 MB). 측정에는 무관하지만 장시간 실행 전에 정리할 일이다.

## 5. 결정

- 결정: 기본 게이트 여유(8 ms)는 이 회차로 바꾸지 않는다 — 박자와 지연의 교환이라 선택은 사용자에게 올린다 (메인 에이전트 기록, 2026-10-27).
