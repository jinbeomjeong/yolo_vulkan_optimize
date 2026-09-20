# HailoRawRunner 백프레셔 결함 수정 — 측정

## 1. 측정 조건

| 항목 | 값 |
|---|---|
| 일시 | 2026-09-20 20:40~20:52 KST |
| 하드웨어 | Raspberry Pi CM5 (BCM2712), Hailo-8, V3D 7.1 |
| **PCIe 링크** | **8.0 GT/s x1 (Gen3)** — sysfs `current_link_speed` |
| 펌웨어 / HailoRT | 4.24.0 (`hailortcli fw-control identify`) |
| 모델 / 입력 | `03_models/yolov11m.hef` / `04_data/videos/video_h265.mp4` (1080p30) |
| 프레임 수 / 반복 | 900 / 설정당 3회 (반복 바깥 루프, host·gpu 인터리브) |
| 경로 | `--mode dmabuf --layout nhcw`, 표시 off |
| 냉각 | 칩 > 75 °C 면 65 °C 까지 대기. 총 대기 335 s |
| 신뢰성 | 12회 전부 `hailort.log` error 0 / 온도·클럭 warning 0, `ok=1` |
| 스크립트 | 이 폴더의 `bench.sh` (구조는 `09_scripts/op_point_bench.sh` 복사) |

## 2. 수정 내용

`01_src/hailo/hailo_raw_runner.cpp` 의 submit 경로에서 **수동 `increase_ongoing_callbacks()` /
`decrease_ongoing_callbacks()` 쌍을 제거**했다(임계값은 `num_slots_` 그대로).
`infer_async()` 가 요청당 1 을 내부에서 이미 세기 때문에 수동 쌍은 이중 계수였고,
그 탓에 in-flight 상한이 `num_slots/2` 로 깎여 있었다.
근거 실측은 `evidence/ongoing_counter_sweep.txt`(프레임당 증가량 1 vs 2).
`01_src/hailo/tools/hailo_raw_probe.cpp` 의 `RawSession` 도 같은 패턴이라 함께 고쳤다.

## 3. 처리량 (900프레임 × 3회 중앙값)

| 설정 | FPS (min~max) | host 대비 | CPU ms/f | NPU 지연 mean | e2e mean | **in-flight** | 검출 |
|---|---:|---:|---:|---:|---:|---:|---:|
| host b2/s4 | 34.97 (34.95~34.97) | — | 2.921 | 111.33 | 142.05 | 3.89 | 3369 |
| **gpu b2/s4** | **34.96** (34.95~34.97) | **−0.03 %** | 1.122 | 110.31 | 142.58 | **3.86** | 3369 |
| host b8/s16 | 51.51 (51.48~51.52) | — | 3.135 | 301.24 | 325.11 | 15.52 | 3369 |
| **gpu b8/s16** | **51.48** (51.47~51.49) | **−0.06 %** | 1.152 | 297.16 | 327.24 | **15.29** | 3369 |

in-flight = FPS × npu_ms_mean / 1000.

수정 전(`07_results/gpu_postproc_e2e/`) 대비:

| 설정 | 수정 전 FPS | 수정 후 FPS | 수정 전 in-flight | 수정 후 in-flight |
|---|---:|---:|---:|---:|
| gpu b2/s4 | 24.48 | **34.96** (+42.8 %) | 2.00 (= slots/2) | **3.86 (≈ slots)** |
| gpu b8/s16 | 49.66 | **51.48** (+3.7 %) | 7.85 (= slots/2) | **15.29 (≈ slots)** |

## 4. 결론

- `--postproc gpu` 의 처리량 손실은 후처리가 아니라 러너 백프레셔의 이중 계수였다.
  수정 후 두 운용점 모두 host 대비 **0.1 % 이내**이고, `b2/s4` 가 **34.96 FPS 로 30 FPS 기준선을 넘는다.**
- CPU 비용 이득(호스트 NMS 제거)은 그대로다: b2/s4 2.921 → 1.122 ms/frame, b8/s16 3.135 → 1.152.
- 검출 수는 네 설정 전부 3369 로 동일하다(정확도 회귀 없음).
- 백프레셔는 살아 있다. in-flight 가 슬롯 수에서 멈추고 더 쌓이지 않는다
  (`evidence/ongoing_counter_sweep.txt` 의 slots 8 / threshold 2 → peak 2 도 같은 증거).

## 5. 주의·남은 일

- `min_queue`(스트림 `get_async_max_queue_size` 최솟값)는 이 수정과 무관하다. batch 2 에서 여전히 8 이라
  `--slots 16 --batch 2` 는 시작 시 거부된다. 다만 batch 2 는 `s4` 에서 이미 ~35 FPS 천장에 닿아
  `s8` 로 올려도 FPS 는 34.97 그대로이고 NPU 지연만 110 → 224 ms 로 늘어난다(실측). 운용점은 `b2/s4` 유지.
- `01_src/vulkan/tools/postproc_probe.cpp:1309-1313` 에 같은 수동 쌍이 남아 있다. 임계값이 1(직렬 제출)이라
  처리량에 영향은 없지만, 그 파일 소유 세션이 같은 방식으로 정리하는 편이 좋다.
