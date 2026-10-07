# GPU NMS 정식 A/B + 현재 기본값 기준선 (실험 A)

계획: [`display_vulkan_hdmi/notes/formal_measurement_plan.md`](../display_vulkan_hdmi/notes/formal_measurement_plan.md) §4.
벤치: `06_tools/bench/nms_bench.sh`(`MODE=perf`, `MODE=verify`), 프로브: `yvo_postproc_probe --probe nms`.

## 1. 측정 조건

- 날짜: 2026-10-08 01:48~02:09 (KST)
- 하드웨어: Raspberry Pi 5, Hailo-8, PCIe **Gen3 x1**(8.0 GT/s), V3D **1150 MHz**(부트로더 2026-09-25), CPU 2.4 GHz
- 소프트웨어: kernel 6.18.50+rpt-rpi-2712, HailoRT 4.24.0, mesa-vulkan-drivers 26.2.2, libavcodec61 7.1.5 rpt2, git `e5f3b9c`
- 모델 `03_models/yolov11m.hef`, 입력 `04_data/videos/video_h265.mp4`(1080p30 HEVC, 1816프레임)
- 경로: **현재 기본값** — `--mode dmabuf --layout nhcw --postproc gpu --postproc-alloc export --gpu-depth 3 --display off`,
  디코더 프레임 스레드 1(앱 기본, `YVO_DEC_THREADS` 제거). 설정별로 `--batch/--slots`와 `--nms cpu|gpu`만 다르다.
- 반복: 900프레임 × **3회**, run 바깥·설정 안쪽 인터리브, **매 실행 전 냉각**(Hailo ≤ 65 °C, 대기 30~75 s — 첫 실행은 시작 시 이미 60.9 °C라 0 s)
- 신뢰성: 15회 실행 모두 rc 0, `hailort.log` 경고·에러 **0건**(온도·클럭 경고 0). 환경 전후 스냅샷은 `env_bench.txt`

## 2. A1 — 파이프라인 A/B (표시 off)

설정당 3회 중앙값, 괄호는 최소~최대. 원시값 `results_raw.csv`, 집계 `results.csv`.

| 설정 | fps | cpu_ms_per_frame | e2e_ms_mean | e2e_ms_p90 | npu_ms_mean | gpu_ms_mean (전처리) | slot_hold_mean | 검출 |
|---|---|---|---|---|---|---|---|---|
| `b2s4_cpu` | 34.99 (34.99~34.99) | 0.896 (0.861~0.932) | 139.57 | 166.67 | 106.45 | 1.808 | 3.939 | 3369 |
| `b2s4_gpu` | 34.99 (34.98~34.99) | 0.908 (0.862~0.980) | 139.60 | 166.72 | 106.41 | 1.810 | 3.974 | 3369 |
| `b8s16_cpu` | 51.50 (51.49~51.50) | 0.857 (0.823~0.900) | 322.19 | 432.35 | 293.32 | 1.821 | 7.660 | 3369 |
| `b8s16_gpu` | 51.50 (51.48~51.51) | 0.869 (0.865~0.953) | 321.87 | 432.10 | 293.10 | 1.828 | 7.545 | 3369 |

억제(NMS) 비용:

| 설정 | CPU NMS (µs/frame, 평균 / 중앙) | GPU NMS (ms/frame, 중앙 / p90 / max) | fallback | V3D CSD 잡/frame | V3D CSD ms/frame |
|---|---|---|---|---|---|
| `b2s4_cpu` | 2.14 / 1.95 | — | — | 4.000 | 2.227 |
| `b2s4_gpu` | — | 0.0486 / 0.0578 / 0.0774 | 0 | 5.000 | 2.253 |
| `b8s16_cpu` | 1.99 / 1.85 | — | — | 4.000 | 2.243 |
| `b8s16_gpu` | — | 0.0500 / 0.0595 / 0.0785 | 0 | 5.000 | 2.270 |

- GPU NMS ms는 `vkCmdWriteTimestamp2` 쌍(파이프라인이 보고하는 `gpu nms ms`)이다. V3DV의 timestamp는 CPU 큐 잡이라 CSD 제출 지연이 들어간다.
- V3D CSD 값은 `/sys/class/drm/renderD128/device/gpu_stats`의 실행 전후 차를 프레임 수로 나눈 **시스템 전체** 값이다.
  NMS는 프레임당 CSD 잡을 정확히 1개 더하고(4 → 5), CSD 시간은 +0.025~0.027 ms/frame 늘었다.

## 3. A2 — 정확도 (전체 1816프레임, b2s4)

`evidence/nms_verify_summary.txt`, `evidence/compare_cpu_vs_gpu.txt`, 덤프는 `samples/{cpu,gpu,verify}_b2s4/`.

| 실행 | 검출 | `nms_verify_mismatch` | fallback |
|---|---|---|---|
| `--nms verify` | 9457 | **0** | 0 |
| `--nms cpu` | 9457 | — | — |
| `--nms gpu` | 9457 | — | 0 |

- `detections.jsonl` 정렬 비교: cpu ↔ gpu **동일**, cpu ↔ verify 동일.
- `compare_detections.py` cpu 대 gpu: 매칭 @ IoU 0.5 / 0.7 / 0.9 모두 **9457 / 9457 = 1.0000**.
- 판정 `[A2] PASS`. verify 모드는 GPU·CPU를 둘 다 돌리므로 성능값으로 쓰지 않는다.

## 4. A3 — NMS 단독 GPU 비용 (NPU 없음, `04_data/raw_tensors/` 픽스처)

`yvo_postproc_probe --probe nms` 3회, 연속 스트림 300 submit(워밍업 30), 표시 off, NPU 유휴(SoC 온도는 직전 `env/env_verify.txt` after 스냅샷 기준 70.8 °C).
원문: `evidence/a3_probe_nms_run{1,2,3}.txt`, `evidence/a3_probe_nms_run{1,2,3}_stdout.log`.

| 구간 | 중앙값 ms (3회) | p90 ms |
|---|---|---|
| 픽스처 스트림 `nms_ms`(후보 평균 23) | 0.0538 / 0.0538 / 0.0538 | 0.060 |
| 같은 실행 `pass1_ms`(디코드) | 0.500 / 0.499 / 0.499 | 0.555~0.556 |
| 합성 all-survive n=16 | 0.068 | 0.07 |
| 합성 n=64 | 0.158 | 0.16 |
| 합성 n=256 | 0.538~0.551 | 0.55 |
| 합성 n=1024 (용량 최악) | **3.03~3.05** | 3.06 |

합성 입력은 모든 n에서 CPU 결과와 비트 일치(`match ok`). 비용은 후보 수에 거의 선형이다.

## 5. 결론

- **GPU NMS는 회귀가 없다.** 두 운용점 모두 FPS·검출이 같고, CPU 차이(+0.012 ms/f)는 두 곳 모두 실행 간 범위가 겹친다.
  e2e 평균은 b2s4 겹침, b8s16 은 gpu 쪽이 0.32 ms 낮다(범위 비중첩 — 회귀 방향이 아니다). CPU 쪽 억제 비용이 원래 프레임당 약 2 µs라, `getrusage` CPU로는
  절감이 보이지 않는다 — 계획서 예상대로 **"CPU 절감 증명"이 아니라 "회귀 없음 + GPU 비용 0.05 ms"** 로 판정한다.
- **정확도는 바이트 동일**이다(9457 = 9457, verify 불일치 0, fallback 0). 기본값 `--nms gpu`를 유지할 근거가 된다.
- **현재 기본값의 정식 기준선(표시 off)**: b8s16 **51.50 fps, CPU 0.869 ms/frame**, e2e 평균 321.9 ms /
  b2s4 **34.99 fps, CPU 0.908 ms/frame**, e2e 평균 139.6 ms. 이전 권고 수치(960 MHz·자동 5스레드·CPU NMS의 b8 CPU 3.3~3.5 ms/f)와
  조건이 다르므로 섞지 않는다.
- **b2s4의 34.99 fps는 운용점 `b2/s4`의 파이프라인 처리량이다** — 파이프라인 처리량은 batch가 아니라 slots가 정한다
  ([`op_point/summary.md`](../op_point/summary.md) §4.2: 같은 조건 b2/s4 34.970). `hailortcli`의 `T(B) = 24.8 + 16.2·B`는 이 값에 쓰지 않는다(§4.3). 표시와 무관하다.
- NMS의 V3D 몫은 프레임당 CSD 잡 1개·약 0.025~0.05 ms로, GPU 4패스 합(수 ms) 대비 작다. 다만 후보가 용량(1024)까지 차면
  3 ms가 되므로 혼잡한 장면에서는 GPU 예산에 영향을 줄 수 있다(픽스처 5프레임의 후보는 평균 23·최대 30 수준).

## 6. 주의·남은 일

- **표시 on 기준선은 이 실험에 없다.** 표시 경로(Vulkan 합성 / wayland-planes / kms)는 실험 B에서 같은 조건으로 잰다.
- `--nms cpu` 실행에서 CSV2의 `gpu_nms_mean`/`gpu_nms_p90`은 nan이 아니라 0.0000으로 찍힌다(앱 출력 형식). stdout에서 뽑은 열은 nan이다.
- `perf stat -e task-clock` 보조 지표는 넣지 않았다. 대신 V3D CSD 잡·시간의 실행 전후 차를 넣었다.
- 매 실행 전 냉각으로 실행 직전 Hailo 온도는 64.6~65.0 °C(첫 실행만 60.9 °C)였고, 실행 직후 71~80 °C까지 올랐다. 연속 운용 시의 상한 유지 여부는
  이 실험의 범위 밖이다(열린 항목 "연속 운용 발열").
