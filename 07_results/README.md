# 측정 결과 색인

각 폴더는 `.claude/skills/measure` 규약을 따른다(설정당 3회·중앙값, 냉각 게이트,
`hailort.log` 경고로 신뢰성 판정, `summary.md` 필수).
**수치를 인용하기 전에 해당 `summary.md` 의 "주의·남은 일" 절을 먼저 읽어라.**

---

## 지금 권고하는 파이프라인

```
디코드(HW HEVC, DRM_PRIME dma-buf)
  → GPU 전처리 sand_letterbox.comp (NHCW 직접 출력, zero-copy)
  → NPU 원시 출력 6텐서 (infer_async, dma-buf H2D/D2H)
  → GPU 후처리 yolo_decode.comp + CPU greedy 억제
  → GPU 오버레이 overlay_boxes.comp + display_convert.comp
  → HDMI (Wayland 스왑체인)
```

```bash
XDG_RUNTIME_DIR=/run/user/1000 WAYLAND_DISPLAY=wayland-0 \
build/01_src/pipeline/yvo_pipeline \
  --video 04_data/videos/video_h265.mp4 --hef 03_models/yolov11m.hef \
  --mode dmabuf --layout nhcw --postproc gpu \
  --batch 2 --slots 4 \
  --display on --display-fullscreen --display-sync matched --display-pace source
```

위 커맨드(`--display-pace source`)의 실측:
**29.91 FPS(소스 30 fps 추종) · glass-to-glass median 101.20 ms(p90 118.36) · CPU 2.18 ms/frame ·
프레임 드롭 0 · 검출 3369(변화 없음)**.
`--display-pace asap` 으로 바꾸면 **34.87 FPS · glass 130.75 ms · CPU 1.84 ms/frame** 이다.
근거는 [`display_e2e/summary.md`](display_e2e/summary.md) §3·§3.2(운용점 확정은 [`op_point/summary.md`](op_point/summary.md) §8).
⚠ 두 페이싱은 **다른 실행**이다. FPS·CPU·glass 를 섞어 인용하지 마라.

운용점은 용도에 따라 고른다 — 최대 처리량은 `b8/s16`(51.5 FPS, 지연 325 ms),
최소 지연은 `b1/s1`(81.6 ms, 단 24.4 FPS 로 30 fps 를 못 따라간다).

---

## 이 하드웨어에서 설계를 지배하는 사실 (전부 실측)

1. **zero-copy 는 `dmabuf + NHCW` 에서만 성립한다.** V3D GEM mmap 포인터는 Hailo 가 DMA 매핑하지 못한다.
2. **GPU 가 쓴 버퍼를 CPU 로 훑으면 약 10배 비싸다.** V3D 메모리 타입이 `HOST_COHERENT` 하나뿐이고
   `HOST_CACHED` 가 없다(1.2 MB 기준 순차 **약 1.41 GB/s**, stride-3 은 0.088 GB/s).
   **반대 방향(CPU→GPU 쓰기)은 싸다.** 정본은
   [`pipeline_ffmpeg_vulkan_hailo/summary.md`](pipeline_ffmpeg_vulkan_hailo/summary.md)
   (0.85~0.87 ms / 1,228,800 B). 이 색인의 이전 판은 1.48 GB/s 라 적었다.
3. **처리량 상한은 PCIe Gen3 x1 이 정한다.** `T(B) = 24.8 + 16.2·B ms`, 점근 상한 61.6 FPS, 실측 51.7.
4. **파이프라인에서 처리량·지연을 지배하는 것은 batch 가 아니라 slots 다.** 지연 ≈ `slots × 프레임 주기`.
5. **표시 해상도를 낮추면 오히려 비싸진다**(1:1 2.05 ns/px vs 스케일 5.31 ns/px). 1920×1080 1:1 이 기본.
6. **원시 출력에 `InferModel` 은 쓸 수 없다** — `VDevice::configure()` + `infer_async()` 저수준 경로 필수.

---

## 실험 목록 (파이프라인 순서)

| 폴더 | 무엇을 쟀나 | 핵심 결론 |
|---|---|---|
| [`npu_benchmark/`](npu_benchmark/summary.md) | NPU 단독 처리량·지연 (Gen2 vs Gen3, batch 1~32) | batch 8 에서 **51.71 FPS**. Gen2→Gen3 로 두 항이 정확히 반씩 줄어 **PCIe 가 병목**임이 확정. ⚠ overall latency 표는 [`op_point/latency_gen3_cooled/`](op_point/latency_gen3_cooled/summary.md) 가 대체 |
| [`decode_benchmark/`](decode_benchmark/summary.md) | 디코드 경로 비교 (SW/HW × H.264/H.265) | GStreamer `v4l2slh265dec` 는 HW 를 쓰고도 32.7 FPS(CPU 30.6 ms/f, ORC 복사). 같은 HW 를 ffmpeg CLI `-hwaccel drm` 로 쓰면 **332.8 FPS / CPU 2.0 ms/f**. (앱에 직접 넣은 libavcodec DRM_PRIME 은 **398.5 FPS / CPU 0.23~0.28 ms/f** — 이 값의 출처는 이 폴더가 아니라 [`pipeline_ffmpeg_vulkan_hailo/REPORT.md`](pipeline_ffmpeg_vulkan_hailo/REPORT.md) §4.1 이다) |
| [`isp_resize/`](isp_resize/summary.md) | PiSP BE(ISP) 전처리 vs GPU 셰이더 | ISP 가 wall 로 23% 빠르고 V3D 를 비운다. **단 검출 −2.33% 로 게이트 불합격 → 통합 보류.** 재검토 조건 3개 명시(그중 하나가 CSI 카메라) |
| [`pipeline_ffmpeg_vulkan_hailo/`](pipeline_ffmpeg_vulkan_hailo/REPORT.md) | 디코드 → GPU 전처리 → NPU 통합 | **51.63 FPS**(NPU 상한의 99.9%). **NHCW 직접 출력으로 CPU 17.6 → 3.5 ms/f.** 비싼 것은 전치가 아니라 **CPU 가 V3D 메모리를 읽는 것**이었다 |
| [`nms_host_cost/`](nms_host_cost/summary.md) | 호스트 NMS 비용 | `yolov8_nms_postprocess` 는 **온칩이 아니라 호스트 CPU** 실행. **2.2~3.2 ms/f 고정비**(검출 수와 무관). 원시 출력으로 바꿔도 **PCIe 트래픽은 안 늘어난다** |
| [`gpu_postprocess/`](gpu_postprocess/summary.md) | GPU 후처리 프로브 (P1/P2/P3) + **[F0] 관문** | 텐서 해석 4변수 전수 확정. pass1 **0.5625 ms**, CPU 억제 3~11 µs. **[F0]: 입력·출력 dma-buf 동시 사용 동작 확인**(`evidence/f0_in_dmabuf.txt`) |
| [`display_vulkan_hdmi/`](display_vulkan_hdmi/summary.md) | 표시 프로브 (P4~P10) | 컴퓨트가 **스왑체인에 직접 STORAGE 쓰기** 가능(blit 불필요). 1080p 1:1 **4.25 ms**. ⚠ 창 모드는 **조용히 2.6배 느려진다** |
| [`dmaheap_unify/`](dmaheap_unify/summary.md) | dma-heap → V3DV import 관문 (별도 트랙) | V3DV 가 dma-heap fd 를 import 한다. **단 `range = size − 256`** 함정. GPU 읽기 성능(B.2)과 3자 일관성은 **미검증** |

### 이번 통합 작업 (2026-09-20)

| 폴더 | 무엇을 쟀나 | 핵심 결론 |
|---|---|---|
| [`op_point/`](op_point/summary.md) ⭐ | **운용점 결정 (O1~O4)** + 냉각 Gen3 지연 재측정 + 지연 스윕 | **이 폴더가 최종 결론이다(§8).** slots 가 지배 변수임을 정량 확인. `b4/s8` vs `b8/s8` 동률로 계획서 F1 가설 기각 |
| [`gpu_postproc_e2e/`](gpu_postproc_e2e/summary.md) | `--postproc gpu` 통합 정확도·CPU | **매칭 @ IoU 0.9 = 0.9990, 검출 9457 로 host 와 완전 동일.** perf 에서 호스트 NMS 45.10% → **0건**. ⚠ 이 폴더의 FPS·지연은 **결함 상태 값이라 인용 금지**(문서 안에 표시) |
| [`raw_runner_backpressure/`](raw_runner_backpressure/summary.md) | 러너 백프레셔 이중 계수 결함 규명·수정 | 수동 `increase_ongoing_callbacks()` 가 `infer_async` 내부 계수와 겹쳐 **in-flight 가 슬롯의 절반**. 제거 후 gpu `b2/s4` 24.48 → **34.96 FPS**(host 대비 −0.03%) |
| [`display_e2e/`](display_e2e/summary.md) | 표시 통합 e2e (M5a 기준선 + M4) | **glass-to-glass 101 ms**(계획서 추정 217 의 절반). GPU 4패스 합 **6.995 ms = 주기의 24.5%**. 표시 켜도 **검출 3369 불변**, 드롭 0 |

---

## 읽는 순서

- **"지금 뭘 쓰면 되나"** → [`op_point/summary.md`](op_point/summary.md) §8
- **"왜 이 설계인가"** → `pipeline_ffmpeg_vulkan_hailo/REPORT.md` §4.3 (메모리 제약) → `nms_host_cost/` (후처리 비용)
- **"표시가 어떻게 동작하나"** → `display_vulkan_hdmi/` (프로브) → `display_e2e/` (통합)
- **"정확도를 어떻게 보장하나"** → `ctest` 44개. 특히 `test_postproc`(픽스처 4 + live 1),
  `test_detections`, `test_preproc`, `test_overlay`. 판정 규칙은 `05_tests/detection_match.hpp` 하나로 공용화돼 있고
  `06_tools/python/compare_detections.py` 가 같은 규칙을 구현한다

## ⛔ 인용 금지 구간

**이 절이 정본이다.** 아래 두 구간은 이 프로젝트의 어떤 문서·논문·보고에서도 인용하지 않는다.

1. **백프레셔 결함 수정 이전의 GPU 후처리 경로 처리량·지연.**
   근거: [`raw_runner_backpressure/summary.md`](raw_runner_backpressure/summary.md).
   [`gpu_postproc_e2e/`](gpu_postproc_e2e/summary.md) 의 FPS·지연 값이 여기 해당한다
   (같은 폴더의 **정확도·perf 결과는 유효하다**).
   결함을 *서술하는* 맥락에서 "수정 전 24.48 → 수정 후 34.96" 처럼 쓰는 것은 인용이 아니라 설명이므로 허용한다.
2. **`--display-pace asap` + 표시 on 조건의 GPU timestamp `mean`.**
   V3D 가 연속 제출된 커맨드 버퍼를 병합해 한쪽 구간이 0에 가깝게 기록된다(**원인 미확정**).
   이 조건에서는 **중앙값만** 쓴다. 근거: [`display_e2e/summary.md`](display_e2e/summary.md).

> 이 규칙의 정본은 원래 `10_paper/paper_ko.md` 부록 A.4 였다. 논문은 저장소 밖으로 나가거나
> 개정될 문서이므로 2026-09-21 에 이리로 옮겼다. 논문 A.4 는 이제 이 절을 참조한다.

---

## 아직 열려 있는 것

**이 표가 정본이다.** 루트 `README.md` 와 `08_learn/10_why_this_design.md` 는 여기를 링크한다.

| 항목 | 왜 아직인가 |
|---|---|
| **연속 운용 발열 안정 FPS** ★ 가장 큰 구멍 | 측정이 전부 900프레임 ≈ 26초짜리다. 칩이 90 °C대로 가면 클럭이 400→250 MHz 로 떨어지고 `b2/s4` 의 여유(+16%)가 사라질 수 있다 |
| **카메라(CSI) 입력 경로** | 디코더가 `01_src/pipeline/ffmpeg_drm_decoder.cpp:166` 에서 **HEVC 파일만** 받는다. 카메라로 가면 ISP 경로 재검토 조건도 함께 열린다 |
| `--display-sync latest` 정식 측정 | 150프레임 확인 실행만 했다(glass median 28.2 ms) |
| dma-heap B.2 / 3자 일관성 | 별도 트랙. `--postproc-alloc heap` 은 옵션만 있고 **기본값은 `export`** 다 |
| 완전 GPU NMS | 하이브리드(GPU 디코드 + CPU 억제)가 3~11 µs 라 서두를 이유가 없다는 판정 |
| 다중 스트림 | **측정 불필요** — 1080p30 2개 = 60 FPS 인데 NPU 상한이 51.7 이다 |
| 워크그룹 최적값의 **이유** | `yolo_decode` 가 60 invocation(10×6)에서 256보다 빠르고 `display_convert` 가 3×85 에서 16×16보다 1.83배 빠른 이유를 설명하지 못했다. **측정치만이 근거인 가설**이다 |
| V3D 쪽 캐시 일관성 **기제** | Hailo 쪽은 드라이버가 관리함을 perf 로 확인했다. V3D 는 프레임마다 매핑하지 않는데도 오염이 관측되지 않는다 — 축출 완료인지 스누핑인지 **가르지 못했다**(결정적 실험에 root 필요) |
| `pace=asap` 의 GPU timestamp 평균 왜곡 | 커맨드 버퍼 병합으로 **추정하나 미확인**. 해당 값은 위 "인용 금지 구간" 2번이다 |
