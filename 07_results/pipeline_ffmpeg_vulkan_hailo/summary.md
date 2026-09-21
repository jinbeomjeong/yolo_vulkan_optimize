# ffmpeg → Vulkan → HailoRT 파이프라인 요약

자세한 구현과 실험 내용은 [REPORT.md](REPORT.md)에 있다.

- 측정: 2026-09-20, Raspberry Pi CM5 + Hailo-8 (PCIe Gen3 x1), `video_h265.mp4` 1816프레임, 설정당 3회 중앙값
- 처리량: **51.63 FPS** (dmabuf + NHCW, batch 8 / slots 16). NPU 상한 51.7 FPS의 99.9%
- CPU: **3.5 ms/frame**. 입력 레이아웃을 NHCW로 맞춰 17.6 ms에서 5배 줄였다
- 정확도: GPU 전처리가 CPU 기준과 비트 단위로 일치(NHWC·NHCW 모두). **27회 실행 검출 합계가 모두 9457개로 동일**
- 디코드 + GPU 전처리 CPU 비용: 약 0.4 ms/frame (GPU 셰이더 2.13 ms, 레이아웃과 무관)

## 설정별 결과

| 설정 | 레이아웃 | FPS | CPU ms/frame | E2E 평균 (ms) |
|---|---|---:|---:|---:|
| ptr b1/s2 | NHWC | 24.42 | 17.59 | 122.6 |
| dmabuf b1/s2 | NHWC | 24.41 | 17.39 | 122.6 |
| **dmabuf b1/s2** | **NHCW** | 24.42 | **3.29** | 122.5 |
| ptr b4/s8 | NHWC | 44.49 | 16.74 | 200.0 |
| dmabuf b4/s8 | NHWC | 44.47 | 16.76 | 200.0 |
| **dmabuf b4/s8** | **NHCW** | 44.52 | **3.17** | 199.9 |
| ptr b8/s16 | NHWC | 51.54 | 17.58 | 326.4 |
| dmabuf b8/s16 | NHWC | 51.55 | 17.61 | 326.4 |
| **dmabuf b8/s16** | **NHCW** | **51.63** | **3.52** | 326.3 |

## 결론

- **병목은 NPU다.** batch별 FPS가 `hailortcli benchmark`와 같고, 디코드와 GPU는 NPU를 막지 않는다.
  그래서 NHCW로 바꿔도 FPS는 거의 그대로다(51.55 → 51.63). **얻은 것은 CPU 코어 0.9개 → 0.16개의 여유다.**
- **NHWC일 때 14 ms를 먹던 것은 전치 연산이 아니라 CPU가 V3D 메모리를 읽는 비용이었다.**
  같은 전치를 캐시된 malloc 버퍼에서 하면 약 1.2 ms다. V3D의 `HOST_COHERENT` 매핑을 stride 3으로 읽어
  10배 이상 비싸졌다. → **GPU가 쓴 버퍼를 CPU로 훑는 설계는 겉보기보다 약 10배 비싸다**(후처리 설계 시 주의).
- **남은 CPU 3.5 ms의 정체는 호스트 NMS 후처리로 확정됐다**(2.2~3.2 ms/frame, 검출 개수와 무관한 고정 비용).
  원시 출력으로 전환하면 사라지고 **PCIe 트래픽도 늘지 않는다** — 원시 텐서가 이미 매 프레임 건너오고 있다.
  다만 병목이 PCIe/NPU라 **처리량은 오르지 않는다**. 자세한 내용: [`../nms_host_cost/summary.md`](../nms_host_cost/summary.md)
- **zero-copy는 dmabuf + NHCW에서만 성립한다.** V3D GEM mmap 포인터는 Hailo 드라이버가 DMA 매핑하지 못하므로
  `--mode ptr --layout nhcw`는 거부된다.

## 주의와 남은 일

- ~~원시 출력 전환 시 `InferModel`이 postprocess를 건너뛸 수 있는지 미확인이다.~~
  **해결(2026-09-20): 불가능하다.** 출력 vstream이 `yolov8_nms_postprocess` 하나뿐이고 postprocess를 끄는
  setter가 없다. `set_format_order(NHCW)` 후 `configure()`가 `HAILO_INVALID_ARGUMENT`로 실패하며
  `nms_post_process.cpp:42`가 원인을 찍는다. 원시 출력은 `VDevice::configure()` →
  `ConfiguredNetworkGroup::infer_async()` 또는 `get_output_streams()+read_async()` 저수준 경로로만 얻는다.
  **두 경로 모두 우리 코드로 동작 확인됐다**(`07_results/gpu_postprocess/evidence/`, ratio 1.0000).
- §4.3의 "V3D 메모리 읽기가 비싸다"는 해석에 **부분 확정 근거가 추가됐다(2026-09-20)**.
  `yvo_preproc_bench`가 V3D 출력 버퍼를 **순차 memcpy**로 1,228,800 B 읽는 시간을 재는데 **0.85~0.87 ms
  (약 1.41 GB/s)** 다. §4.3의 stride-3 읽기는 같은 양에 약 14 ms(약 0.088 GB/s)였다.
  → **16배 차이는 V3D 읽기 대역폭 자체가 아니라 접근 패턴에서 나온다.** 순차로 읽으면 1.2 MB에 0.87 ms로
  감당 가능하다. 다만 "캐시된 malloc 대비 몇 배인가"는 여전히 직접 비교가 필요하다(완전 확정은 아니다).
- 지연 최적 설정(b8/s8, b4/s4)은 미측정이다. 슬롯 수가 곧 큐 깊이라 지연이 중요하면 줄여야 한다.
- 연속 운용 시 칩이 90 °C대까지 오른다. CPU 부하가 1/5로 줄어도 칩 온도는 그대로다(발열원은 NPU다).
- **정정(2026-09-20):** `yvo_hailo_selftest`의 `[stream]` FPS는 측정 창에 파이프라인 드레인이 들어가
  약 4.8% 과소평가된다(기본 300프레임/16슬롯). REPORT.md §4.1 표의 "NPU만 50.0 FPS"는 보정하면 약 52.5이고
  benchmark 상한 51.7과 정합한다. **`yvo_pipeline`의 51.63 FPS는 wall-clock 기반이라 영향이 없다.**
  자세한 내용은 [REPORT.md](REPORT.md)의 "정정 1".
- 보류 과제: Gen3에서 칩을 식힌 상태의 overall latency 재측정 (`07_results/npu_benchmark/runs/gen3/summary.md`).
