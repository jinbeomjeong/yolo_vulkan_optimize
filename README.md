# yolo_vulkan_optimize — 실행 안내

Raspberry Pi CM5 + Hailo-8 NPU 에서 YOLOv11m 추론을 돌리되, 파이프라인의 **CPU 비용을
Vulkan(V3D GPU)으로 밀어낸** 프로젝트다. 디코드·전처리·추론·후처리·가시화가 **하나의 실행 파일**
(`yvo_pipeline`)로 통합돼 있고 전 구간 CPU 복사가 0회다.

실측(1080p30 HEVC, PCIe Gen3 x1, 운용점 `b2/s4 --postproc gpu`, 표시 on, 900프레임 ×3회 중앙값):

| `--display-pace` | FPS | glass-to-glass | CPU ms/frame | 드롭 |
|---|---:|---:|---:|---:|
| **`source`** (기본·최종 운용) | **29.91** (소스 30 fps 추종) | **101 ms** | **2.18** | 0 |
| `asap` | 34.87 | 131 ms | 1.84 | 0 |

검출 수는 표시 off/on, 두 페이싱 전부 **3369 로 동일**하다.
근거: [`07_results/display_e2e/summary.md`](07_results/display_e2e/summary.md) §3·§3.2.

---

## 1. 파이프라인

```
04_data/videos/video_h265.mp4  (HEVC 1080p30)
      │
      ▼  libavcodec DRM_PRIME                      [HW 디코더]   decode call 0.107 ms/f
   DrmFrame: Y/UV dma-buf (SAND128 NV12)  ── 복사 0
      │
      ├───────────────────────────────────────────┐
      ▼  sand_letterbox.comp        [V3D GPU] 2.088 ms  ▼ (표시용 원본 유지)
   NHCW 640×640×3  ── export dma-buf                    │
      │                                                 │
      ▼  infer_async (H2D dma-buf)  [Hailo-8] 69.7 ms   │
   원시 6텐서 1,221,120 B  ── D2H dma-buf               │
      │                                                 │
      ▼  yolo_decode.comp           [V3D GPU] 0.626 ms  │
   후보 ~16개                                           │
      ▼  cpu_nms_greedy_by_class    [CPU]     2.67 µs   │
   Detection (원본 1920×1080 좌표)                       │
      │                                                 │
      ▼─────────────────────────────────────────────────┘
   display_convert.comp + overlay_boxes.comp  [V3D GPU] 4.153 + 0.128 ms
      ▼  Wayland 스왑체인 present (FIFO)
   HDMI-A-1 1920×1080@60
```

dma-buf 세 그룹(디코더 SAND / 전처리 NHCW / 원시 텐서)이 전부 zero-copy 로 오간다.

> 구간 시간은 전부 **중앙값(median)** 이다. GPU 4구간(2.088 / 0.626 / 4.153 / 0.128 ms, 합 6.995 ms)은
> [`display_e2e/summary.md`](07_results/display_e2e/summary.md) §3.1(`disp_asap`), 디코드 호출·NPU·CPU NMS 는
> 같은 측정의 `raw/disp_source_run1/stdout.log` 다. `display_convert` 의 **4.226 ms 는 NPU 유휴 기준선(M5a)**
> 이지 실측이 아니다 — 혼동하지 마라. 같은 값을 쓰는 그림이 [`12_docs/pipeline_overview.graphml`](12_docs/pipeline_overview.graphml) 에 있다.

---

## 2. 빌드

```bash
cmake -S . -B build -G Ninja && cmake --build build
```

빌드 경고는 **0줄**이어야 한다. 셰이더는 `glslc -Werror` + `spirv-val` 을 자동으로 통과한다.

의존: HailoRT 4.24.0, Vulkan(V3DV), libavcodec, wayland-client + wayland-protocols, OpenCV, GTest.
표시 경로는 wayland 가 없으면 조용히 빠지고 나머지는 그대로 빌드된다.

---

## 3. 실행

### ⚠ 기본값은 권장 경로가 아니다

인자 없이 실행하면 `--mode ptr --layout nhwc --postproc host --display off` 로 돈다.
이것은 **가장 느린 경로**(CPU 17.6 ms/frame)다. 회귀 격리를 위해 기본값을 바꾸지 않았다.
**아래 플래그를 반드시 주어야 한다.**

### 3.1 실시간 HDMI 표시 (권장)

```bash
XDG_RUNTIME_DIR=/run/user/1000 WAYLAND_DISPLAY=wayland-0 \
build/01_src/pipeline/yvo_pipeline \
  --video 04_data/videos/video_h265.mp4 --hef 03_models/yolov11m.hef \
  --mode dmabuf --layout nhcw --postproc gpu \
  --batch 2 --slots 4 \
  --display on --display-fullscreen --display-sync matched --display-pace source
```

### 3.2 헤드리스 — 검출만 파일로

```bash
build/01_src/pipeline/yvo_pipeline \
  --video 04_data/videos/video_h265.mp4 --hef 03_models/yolov11m.hef \
  --mode dmabuf --layout nhcw --postproc gpu \
  --batch 8 --slots 16 \
  --dump-dir /tmp/out --dump-every 300
```

`/tmp/out/detections.jsonl` 에 프레임당 한 줄, 300프레임마다 박스 PNG 가 떨어진다.

### 3.3 운용점 선택

| 목적 | 인자 | 결과 |
|---|---|---|
| **실시간 표시** (권장) | `--batch 2 --slots 4` | 표시 off 34.95 FPS / 표시 on `pace source` **29.91 FPS, glass 101 ms** |
| 최대 처리량 | `--batch 8 --slots 16` | 51.5 FPS, e2e 325 ms |
| 최소 지연 | `--batch 1 --slots 1` | e2e 81.6 ms, 단 **24.4 FPS 로 30 fps 를 못 따라간다** |
| 영상만 저지연 표시 | 위에 `--display-sync latest` 추가 | 화면 glass 28 ms, 박스만 뒤처진다 |

**처리량·지연을 지배하는 것은 batch 가 아니라 slots 다.** 지연 ≈ `slots × 프레임 주기` 이고,
같은 slots 에서 batch 차이는 0.25% 뿐이다. 다만 **batch 1 은 천장이 24.4 FPS** 라
(배치당 고정비 24.8 ms) 30 fps 소스를 따라가려면 batch ≥ 2 가 필요하다.

### 3.4 실행 중단

**`Ctrl+C` 를 누르면 정상 종료한다.** `SIGINT`/`SIGTERM` 을 받으면 진행 중인 프레임을 정리하고
**통계 요약을 출력한 뒤** 끝난다. 다른 터미널에서는 `pkill -INT yvo_pipeline`.

```
[interrupted] SIGINT 수신 — 136 프레임에서 중단 (아래 통계는 부분 실행 값이다, ok=0)
frames          136 (decoded 136, errors 0)
wall            4.612 s, 29.49 FPS
...
```

| 항목 | 값 |
|---|---|
| 시그널 → 종료 | **0.2~0.5 초** |
| 종료 코드 | **130**(SIGINT) / **143**(SIGTERM) |
| `RESULT_CSV`·`RESULT_CSV2` 의 `ok` | **0** — 중단된 실행이 벤치 집계에 유효값으로 섞이지 않는다 |
| NPU·fd·디코더 버퍼 | 정상 종료 경로를 그대로 타므로 전부 해제 (ASan 누수 0 확인) |

**요약 머리의 `[interrupted]` 를 확인해라.** 부분 실행 값을 전체 실행 값으로 오인하면 안 된다.

- **두 번 연속 누르면 즉시 강제 종료**한다(`_exit(130)`, 요약 없음). 정상 종료가 지연될 때의 탈출구다.
- 전체화면에서는 **창 닫기·키보드로 종료할 수 없다.** 키 입력을 처리하지 않는다
  (`wayland_window.hpp` — xkbcommon 의존성을 의도적으로 뺐다). 창 모드에서는 닫기 버튼이 동작하지만
  창 모드 자체가 표시 변환을 2.6배 느리게 하므로 측정에는 쓰지 마라.
- 통계가 필요하고 중단 시점을 미리 정할 수 있으면 **`--frames N`** 이 더 깔끔하다.
  입력 영상 끝에서는 자동으로 정상 종료한다(1,816프레임, 약 60초).

---

## 4. 주요 옵션

| 옵션 | 기본 | 설명 |
|---|---|---|
| `--mode ptr\|dmabuf` | `ptr` | **`dmabuf` 필수.** V3D 포인터는 Hailo 가 DMA 매핑하지 못한다 |
| `--layout nhwc\|nhcw` | `nhwc` | **`nhcw` 필수.** HailoRT 의 CPU 입력 변환(약 14 ms/f)을 없앤다. dmabuf 전용 |
| `--postproc host\|gpu` | `host` | `gpu` 는 원시 출력 + `yolo_decode.comp` + CPU 억제. **`dmabuf`+`nhcw` 전용** |
| `--batch N` / `--slots N` | 8 / `max(4,batch*2)` | 운용점. §3.3 참조 |
| `--gpu-depth N` | 3 | GPU 동시 프레임 상한. **지연에 영향 없음**(실측 1 vs 3 동일) |
| `--display off\|on` | `off` | HDMI 실시간 표시 |
| `--display-fullscreen` | 꺼짐 | **반드시 켠다.** 창 모드는 에러·경고 없이 표시 변환이 **2.6배**(4.25 → 10.9 ms) 느려진다 |
| `--display-sync matched\|latest` | `matched` | `matched` 는 프레임과 그 검출을 정확히 맞춘다. `latest` 는 화면을 즉시 띄우고 박스만 뒤처진다 |
| `--display-pace source\|asap` | 표시 on 이면 `source` | 소스 fps 로 페이싱. `asap` 이면 큐가 꽉 차 glass 101 → 131 ms |
| `--postproc-alloc export\|heap` | `export` | **`heap` 은 쓰지 마라** — 3자 일관성(PCIe DMA ↔ CPU 캐시 ↔ V3D)이 미검증이다 |
| `--score T` | HEF 값 0.2 | 검출 점수 임계값 |
| `--frames N` / `--quiet` / `--csv-label L` | | 측정용 |

전체 목록은 `yvo_pipeline --help`.

---

## 5. 출력 읽는 법

실행이 끝나면 통계가 찍힌다. 확인할 것:

| 줄 | 정상 | 의미 |
|---|---|---|
| `held frames max` | **≤ 10** | 동시에 붙잡은 디코더 프레임. 약 13을 넘으면 **디코더가 에러 없이 멈춘다** |
| `slot timeout` | **0** | 슬롯 대기 타임아웃. 0이 아니면 슬롯 설정이 과하다 |
| `display dropped` | 0 | 표시를 버린 프레임 |
| `vk validation errors` | **0** | `warnings 1` 은 알려진 무해 항목(장치 스캔) |
| `detections` | 입력이 같으면 동일해야 함 | 경로를 바꿔도 검출이 변하면 회귀다 |

⚠ **`--display-pace asap` + 표시 on 에서는 GPU timestamp 의 `mean` 을 믿지 마라.**
median 을 써라(V3D 가 커맨드 버퍼를 합치면서 한쪽 구간이 0에 가깝게 잡힌다 — 원인 미확정).

---

## 6. 검증

```bash
ctest --test-dir build        # 44/44 통과해야 한다
```

| 테스트 | NPU | 무엇을 보는가 |
|---|---|---|
| `PostprocFixture.*` (4개) | 불필요 | GPU 후처리 vs CPU 기준: 후보 수 동일, 점수 **비트 일치**, 좌표 < 0.25 px, 10회 결정성 |
| `test_postproc_live` | 필요 | host 경로 vs GPU 경로 검출 매칭 (≥ 0.95) |
| `test_detections` | 필요 | CPU/GPU 전처리 경로 검출 일치 |
| `test_preproc` | 불필요 | GPU 전처리가 CPU 기준과 **비트 일치**(max 오차 0) |
| `test_overlay` | 불필요 | 오버레이 셰이더 vs CPU 기준 비트 일치 |

두 경로의 검출을 직접 비교하려면:

```bash
python 08_python/compare_detections.py ref/detections.jsonl test/detections.jsonl --csv out.csv
```

---

## 7. 측정

```bash
bash 09_scripts/op_point_bench.sh    # 운용점 스윕 (배치·슬롯)
bash 09_scripts/display_bench.sh     # 표시 e2e (M5a 기준선 + M4)
bash 09_scripts/pipeline_bench.sh    # 기존 9설정 회귀 기준선
bash 09_scripts/decode_bench.sh      # 디코드 경로 비교
```

공통 환경변수: `RUNS`(기본 3), `FRAMES`, `ONLY`(설정 id grep), `APPEND=1`, `BIN`, `OUT_DIR`,
`COOL_C`/`HOT_C`/`COOL_MAX_S`.

측정 규약은 `.claude/skills/measure` 에 있다. 요점만:
**설정당 3회 중앙값, 반복이 바깥 루프, 칩 75 °C 초과 시 65 °C 까지 냉각,
`hailort.log` 에 온도·클럭 경고가 있으면 그 실행의 성능 수치는 신뢰 불가.**

⚠ **측정 중에 빌드·테스트를 돌리지 마라.** CPU 시간 측정이 조용히 오염된다.
`.claude/hooks/npu_guard.sh` 가 NPU 충돌을 막아 준다.

---

## 8. 진단 도구

| 도구 | 용도 |
|---|---|
| `yvo_drm_dump` | 디코더 dma-buf descriptor 덤프, de-SAND 바이트 비교, 디코드 성능 |
| `yvo_preproc_bench` | GPU 전처리 단독 성능·워크그룹 스윕 |
| `yvo_postproc_probe` | GPU 후처리 프로브 (`--probe fixture` 는 **NPU 불필요**, 수 초) |
| `yvo_hailo_raw_probe` | 원시 출력 경로 프로브 (`--probe f0` 이 입출력 dma-buf 동시 사용 관문) |
| `yvo_hailo_selftest` | NPU 러너 단독 검증 |
| `yvo_display_probe` | 표시 경로 프로브 (surface/present/convert/overlay) |
| `yvo_display_pipeline_probe` | `DisplayPipeline` 단독 검증 (`yvo_pipeline` 미사용) |
| `yvo_isp_*` | PiSP BE(ISP) 전처리 — **통합 보류 상태** |

화면에 실제로 나왔는지는 사람 눈 없이 확인한다:

```bash
grim -o HDMI-A-1 shot.png
```

---

## 9. 문제 해결

| 증상 | 원인·조치 |
|---|---|
| `--postproc gpu` 가 시작 시 거부됨 | `--mode dmabuf --layout nhcw` 를 같이 줘야 한다 |
| `--slots` 가 거부됨 | 원시 경로의 큐 상한(`min_queue`)을 넘었다. 메시지에 상한이 찍힌다 |
| 표시가 조용히 2.6배 느림 | 창 모드다. `--display-fullscreen` 을 켜라 |
| 디코더가 멈춤(에러 없음) | 프레임을 13개 넘게 붙잡았다. `slots + gpu_depth ≤ 10` 을 지켜라 |
| `HAILO_OUT_OF_PHYSICAL_DEVICES` | 다른 프로세스가 NPU 를 쓰고 있다. Hailo-8 은 보드에 하나뿐이다 |
| 성능이 들쭉날쭉 | 칩 온도를 봐라. 90 °C대에서 클럭이 400→250 MHz 로 떨어진다 |
| FPS 가 예상의 절반 | 과거에 러너 백프레셔 이중 계수 결함이 있었다(해소됨). `in-flight = FPS × npu_latency` 가 `slots` 에 가까운지 봐라 |

---

## 10. 더 읽을 것

- **[`13_learn/`](13_learn/00_README.md)** — 배경지식이 없다면 **여기부터**. NPU·Vulkan·dma-buf·YOLO 텐서를
  맨 밑바닥부터 설명하고, 각 개념이 이 프로젝트의 어느 결정으로 이어졌는지 연결한 자습 교재 12장
- **[`07_results/README.md`](07_results/README.md)** — 12개 실험의 색인과 결론. **수치를 인용하기 전에 여기부터**
- [`07_results/op_point/summary.md`](07_results/op_point/summary.md) §8 — 운용점 확정 근거와 최종 결론
- [`07_results/display_e2e/summary.md`](07_results/display_e2e/summary.md) — 표시 통합 측정
- [`07_results/pipeline_ffmpeg_vulkan_hailo/REPORT.md`](07_results/pipeline_ffmpeg_vulkan_hailo/REPORT.md) §4.3 — 왜 이 설계인가(메모리 제약)
- `CLAUDE.md` — 폴더 규약, 개발 환경, 서브에이전트 라우팅

### 아직 열려 있는 것

- **연속 운용 발열** — 측정이 전부 26초짜리다. 몇 시간 도는 운용에서 `b2/s4` 의 30 fps 여유(+16%)가
  클럭 강등을 버티는지 미검증
- **카메라(CSI) 입력** — 디코더가 HEVC 파일만 받는다(`01_src/pipeline/ffmpeg_drm_decoder.cpp:166`).
  카메라로 가면 ISP 전처리 경로 재검토 조건도 함께 열린다
- **다중 스트림은 불가능하다** — 1080p30 두 개 = 60 FPS 인데 NPU 상한이 51.7 이다.
  확보한 CPU 여유는 이 한계를 풀지 못한다
