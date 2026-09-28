# yolo_vulkan_optimize — 실행 안내

Raspberry Pi CM5 + Hailo-8 NPU 에서 YOLOv11m 추론을 돌리되, 파이프라인의 **CPU 비용을
Vulkan(V3D GPU)으로 밀어낸** 프로젝트다. 디코드·전처리·추론·후처리·가시화가 **하나의 실행 파일**
(`yvo_pipeline`)로 통합돼 있고 전 구간 CPU 복사가 0회다.

실측(1080p30 HEVC, PCIe Gen3 x1, 운용점 `b2/s4 --postproc gpu`, 표시 on, 900프레임 ×3회 중앙값):

| `--display-pace` | FPS | glass-to-glass | CPU ms/frame | 드롭 |
|---|---:|---:|---:|---:|
| **`source`** (기본·최종 운용) | **29.91** (소스 30 fps 추종) | **101.20 ms** | **2.18** | 0 |
| `asap` | 34.87 | 130.75 ms | 1.84 | 0 |

검출 수는 표시 off/on, 두 페이싱 전부 **3369 로 동일**하다.
근거: [`07_results/display_e2e/summary.md`](07_results/display_e2e/summary.md) §3·§3.2.
이 수치는 억제를 **CPU**(`cpu_nms_greedy_by_class`, 지금의 `--nms cpu`)로 하던 때의 측정이다.
GPU NMS(`--nms gpu`, 기본)로 바꾼 뒤의 재측정은 아직 없다.

---

## 1. 파이프라인

```
04_data/videos/video_h265.mp4  (HEVC 1080p30)
      │
      ▼  libavcodec DRM_PRIME                      [HW 디코더]
   DrmFrame: Y/UV dma-buf (SAND128 NV12)  ── 복사 0
      │
      ├───────────────────────────────────────────┐
      ▼  sand_letterbox.comp              [V3D GPU]  ▼ (표시용 원본 유지)
   NHCW 640×640×3  ── export dma-buf                 │
      │                                              │
      ▼  infer_async (H2D dma-buf)        [Hailo-8]  │
   원시 6텐서 1,221,120 B  ── D2H dma-buf            │
      │                                              │
      ▼  yolo_decode.comp                 [V3D GPU]  │
   후보 ~16개  (같은 커맨드 버퍼)                    │
      ▼  efficient_nms.comp               [V3D GPU]  │
   Detection (원본 1920×1080 좌표)                    │
      │                                              │
      ▼──────────────────────────────────────────────┘
   display_convert.comp + overlay_boxes.comp  [V3D GPU]
      ▼  Wayland 스왑체인 present (FIFO)
   HDMI-A-1 1920×1080@60
```

dma-buf 세 그룹(디코더 SAND / 전처리 NHCW / 원시 텐서)이 전부 zero-copy 로 오간다.

> **표시에는 대안 경로가 하나 더 있다** — `--display-backend wayland-planes` / `kms`(§3.5).
> 디코더 SAND 를 `display_convert` 없이 HVS 영상 평면에 올리고, 박스는 GPU 가 AB24 dma-buf(네 번째 그룹)에 그려
> 그 위 평면에 올린다. YUV→RGB 와 블렌딩은 HVS 가 한다.

> **구간별 지연 수치는 여기 적지 않는다.** 같은 값이 세 곳에 복제돼 있다가 한 번 어긋난 적이 있어
> (`09_diagrams/README.md` 의 정렬 기록), 2026-09-21 에 이 도식에서는 **흐름만** 남겼다.
> 구간 median 은 [`display_e2e/summary.md`](07_results/display_e2e/summary.md) §3.1 이 정본이고,
> 같은 값을 붙인 그림이 [`09_diagrams/pipeline_overview.graphml`](09_diagrams/pipeline_overview.graphml) 에 있다.

---

## 2. 빌드

```bash
cmake -S . -B build -G Ninja && cmake --build build
```

빌드 경고는 **0줄**이어야 한다. 셰이더는 `glslc -Werror` + `spirv-val` 을 자동으로 통과한다.

의존: HailoRT 4.24.0, Vulkan(V3DV), libavcodec, libdrm, wayland-client + wayland-protocols, OpenCV, GTest.
표시 경로(`yvo_display`, kms 백엔드 포함)는 wayland 가 없으면 조용히 빠지고 나머지는 그대로 빌드된다.
wayland-protocols 에 staging `color-representation` XML 이 없으면 그 요청만 빠진다(`wayland-planes` 의 색 range 지정 불가).

---

## 3. 실행

### ⚠ 기본값은 권장 경로가 아니다

인자 없이 실행하면 `--mode ptr --layout nhwc --postproc host --display off` 로 돈다.
이것은 **가장 느린 경로**(CPU 17.6 ms/frame)다. 회귀 격리를 위해 기본값을 바꾸지 않았다.
**아래 플래그를 반드시 주어야 한다.**

### 3.1 실시간 HDMI 표시 (권장)

```bash
XDG_RUNTIME_DIR=/run/user/1000 WAYLAND_DISPLAY=wayland-0 \
build/bin/yvo_pipeline \
  --video 04_data/videos/video_h265.mp4 --hef 03_models/yolov11m.hef \
  --mode dmabuf --layout nhcw --postproc gpu \
  --batch 2 --slots 4 \
  --display on --display-fullscreen --display-sync matched --display-pace source
```

### 3.2 헤드리스 — 검출만 파일로

```bash
build/bin/yvo_pipeline \
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
`--display on`이면 **표시 창에서 `Esc`·`q`·`Ctrl+C`**를 눌러도 같은 경로로 끝난다
(전체 화면 창이 키보드 포커스를 가져가 터미널의 `Ctrl+C`가 닿지 않기 때문이다).
`timeout -s INT`는 프로세스 그룹에도 신호를 보내 두 번째 시그널(즉시 `_exit`)이 되므로 요약이 나오지 않는다.

```
[interrupted] SIGINT received — stopped at frame 136 (the stats below are partial-run values, ok=0)
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

> ⚠ **예외 — 스왑체인 표시가 멈춘 경우.** 컴포지터가 frame 콜백을 주지 않으면(labwc 가 HDMI 를 잃고 headless 출력만 쥔 경우)
> Vulkan 경로는 `vkQueuePresentKHR` 안에서 무기한 기다리고 **SIGTERM 도 먹지 않는다**(`kill -9` 필요). §9 참조.
> HVS 평면 백엔드(§3.5)는 frame 콜백을 50 ms 까지만 기다려 이 문제가 없다.

- **두 번 연속 누르면 즉시 강제 종료**한다(`_exit(130)`, 요약 없음). 정상 종료가 지연될 때의 탈출구다.
- 표시 창의 종료 키는 **물리 키 기준**이다(`wayland_window.cpp` — xkbcommon 없이 evdev 키코드로 판정한다).
  창 모드의 닫기 버튼은 중단이 아니라 정상 종료로 처리되지만, 창 모드 자체가 표시 변환을 2.6배 느리게 하므로
  측정에는 쓰지 마라.
- 통계가 필요하고 중단 시점을 미리 정할 수 있으면 **`--frames N`** 이 더 깔끔하다.
  입력 영상 끝에서는 자동으로 정상 종료한다(1,816프레임, 약 60초).

### 3.5 HVS 평면 표시 (`wayland-planes` / `kms`) — 구현·검증됨, 정식 측정 전

V3D 가 표시 변환을 하지 않는 두 번째 표시 방식이다. 디코더 SAND 버퍼를 **디스플레이 컨트롤러(HVS)의 영상 평면**에 그대로 올리고,
박스는 GPU 가 AB24 버퍼에 그려 **그 위 평면**에 올린다. YUV→RGB 와 블렌딩은 HVS 가 스캔아웃 중에 한다.
두 백엔드 모두 실제 HDMI 에서 평면 배치와 색(BT.709 limited)을 debugfs 로 확인했지만, **기존 경로와 비교한 정식 측정은 아직 없다**
(설명: [`08_learn/07_video_display.md`](08_learn/07_video_display.md) §7.12).

```bash
# wayland-planes — labwc 가 두 평면을 배치한다 (전체화면 강제)
XDG_RUNTIME_DIR=/run/user/1000 WAYLAND_DISPLAY=wayland-0 \
build/bin/yvo_pipeline \
  --video 04_data/videos/video_h265.mp4 --mode dmabuf --layout nhcw --postproc gpu \
  --batch 2 --slots 4 --display on --display-backend wayland-planes

# kms — KMS atomic 으로 직접 커밋한다. DRM master 가 필요하다
loginctl list-sessions                 # 데스크톱이 아닌 tty 세션 id 를 찾는다
loginctl activate <tty 세션 id>         # (또는 sudo chvt 3) 모니터가 콘솔로 바뀌고 labwc 가 master 를 놓는다
build/bin/yvo_pipeline \
  --video 04_data/videos/video_h265.mp4 --mode dmabuf --layout nhcw --postproc gpu \
  --batch 2 --slots 4 --display on --display-backend kms
loginctl activate <데스크톱 세션 id>    # (또는 sudo chvt 7) 데스크톱으로 돌아간다
```

- **kms 실행 중에는 VT 를 바꾸지 마라.** labwc 가 master 를 받지 못해 데스크톱이 검게 남을 수 있다. 끝내는 것은 Ctrl+C / SIGTERM 으로.
  master 를 못 얻으면 누가 쥐고 있는지 알려 주고 초기화 전에 **종료 코드 2** 로 끝난다. lightdm 은 멈출 필요가 없다.
- kms 는 종료할 때 이전 KMS 상태 복원을 시도하지만 지금은 `EINVAL` 로 실패하고 평면 끄기로 폴백한다(콘솔이 검게 남는다). 데스크톱으로 돌아가면 복구된다.
- `wayland-planes` 의 색: labwc 가 color-representation 의 range 를 뒤집어 넘기므로 기본값(`auto`)은 full 을 요청해 limited 를 얻는다.
  매 실행 요약의 `scanout ... color_ok 1` 을 확인해라.
- `--display-pace source` 에서는 batch 2 의 몰림을 펴려고 커밋 간격을 "소스 간격 − 8 ms" 이상으로 벌린다
  ([`notes/wayland_planes_pacing_gate.md`](07_results/display_vulkan_hdmi/notes/wayland_planes_pacing_gate.md)).

---

## 4. 주요 옵션

| 옵션 | 기본 | 설명 |
|---|---|---|
| `--mode ptr\|dmabuf` | `ptr` | **`dmabuf` 필수.** V3D 포인터는 Hailo 가 DMA 매핑하지 못한다 |
| `--layout nhwc\|nhcw` | `nhwc` | **`nhcw` 필수.** HailoRT 의 CPU 입력 변환(약 14 ms/f)을 없앤다. dmabuf 전용 |
| `--postproc host\|gpu` | `host` | `gpu` 는 원시 출력 + `yolo_decode.comp` + `--nms` 억제. **`dmabuf`+`nhcw` 전용** |
| `--nms gpu\|cpu\|verify` | `gpu` | `--postproc gpu` 의 억제 위치. `gpu` 는 `efficient_nms.comp`(CPU 기준과 비트 일치, 보장 못 하는 프레임만 자동 CPU 처리), `cpu` 는 `cpu_nms_greedy_by_class`, `verify` 는 매 프레임 둘을 비트 비교 |
| `--batch N` / `--slots N` | 8 / `max(4,batch*2)` | 운용점. §3.3 참조 |
| `--gpu-depth N` | 3 | GPU 동시 프레임 상한. **지연에 영향 없음**(실측 1 vs 3 동일) |
| `--display off\|on` | `off` | HDMI 실시간 표시 |
| `--display-backend auto\|wayland\|offscreen\|wayland-planes\|kms` | `auto` | `auto` = Wayland 창이 열리면 `wayland`, 아니면 `offscreen`. `wayland` = Vulkan 합성 + 스왑체인(strict — 실패하면 종료), `offscreen` = 화면 없이 변환만. `wayland-planes`/`kms` = HVS 평면 스캔아웃(§3.5, strict) |
| `--display-color-range auto\|full\|limited` | `auto` | 평면 백엔드가 요청할 COLOR_RANGE. `auto` = wayland-planes 는 full(labwc 뒤집힘 우회), kms 는 limited |
| `--display-connector NAME` | 첫 연결 HDMI-A | `kms` 의 출력 커넥터(다른 백엔드에서는 경고 후 무시) |
| `--display-fullscreen` | 꺼짐 | **반드시 켠다.** 창 모드는 에러·경고 없이 표시 변환이 **2.6배**(4.22 → 10.92 ms) 느려진다. 평면 백엔드는 자동으로 전체화면이다 |
| `--display-sync matched\|latest` | `matched` | `matched` 는 프레임과 그 검출을 정확히 맞춘다. `latest` 는 화면을 즉시 띄우고 박스만 뒤처진다 |
| `--display-pace source\|asap` | 표시 on 이면 `source` | 소스 fps 로 페이싱. `asap` 이면 큐가 꽉 차 glass 101.20 → 130.75 ms (둘 다 median) |
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
| `nms fallback` | score 0.2 에서 **0** | GPU NMS 가 비트 일치를 보장 못 해 CPU 로 처리한 프레임(용량 초과 / IoU 가 0.7 ± 4e-6 안) |
| `nms verify mismatch` | **0** | `--nms verify` 에서 GPU·CPU 결과가 비트 단위로 다른 프레임 |
| `scanout zero_copy, color_ok, hvs_underrun delta` | **1, 1, 0** | 평면 백엔드 전용. debugfs 로 본 실제 평면 배치(영상 NV12/SAND128 + 박스 AB24 위)·색(BT.709 limited)·HVS 언더런. 읽지 못하면 `unverified` |
| `scanout overlay skipped, frame-callback timeouts, busy reuse` | 0 근처 | 박스 링이 비어 박스 갱신을 건너뜀 / frame 콜백 50 ms 초과 / 전송층이 거부한 커밋 |
| `kms: flips, flip_timeouts, master_lost` · `kms: shutdown ...` | flips = presented, 0, `no` | kms 전용. flip 이벤트 1 s 워치독, 실행 중 master 상실, 종료 시 복원 결과 |

⚠ **`--display-pace asap` + 표시 on 에서는 GPU timestamp 의 `mean` 을 믿지 마라.**
median 을 써라(V3D 가 커맨드 버퍼를 합치면서 한쪽 구간이 0에 가깝게 잡힌다 — 원인 미확정).
인용 금지 구간의 정본은 [`07_results/README.md`](07_results/README.md#-인용-금지-구간) 다.

---

## 6. 검증

```bash
ctest --test-dir build        # 78/78 통과해야 한다
```

| 테스트 | NPU | 무엇을 보는가 |
|---|---|---|
| `PostprocFixture.*` (5개) | 불필요 | GPU 후처리 vs CPU 기준: 후보 수 동일, 점수 **비트 일치**, 좌표 < 0.25 px, 10회 결정성, GPU NMS 결과 **비트 일치** |
| `EfficientNms.*` (7개) | 불필요 | 합성 입력: 동점, 클래스 정원 100 초과, 클래스 간 겹침, 용량 초과·IoU 보호대 → CPU 처리 신호, 결정성 |
| `test_postproc_live` | 필요 | host 경로 vs GPU 경로 검출 매칭 (≥ 0.95) |
| `test_detections` | 필요 | CPU/GPU 전처리 경로 검출 일치 |
| `test_preproc` | 불필요 | GPU 전처리가 CPU 기준과 **비트 일치**(max 오차 0) |
| `test_overlay` | 불필요 | 오버레이 셰이더 vs CPU 기준 비트 일치. 박스 평면 dma-buf(`OverlayPlane.*`, 잔상 0), debugfs 파서(`DrmState.*`), KMS 탐색·AddFB2(`KmsScanout.*` — card1 사용, master 는 가져가지 않는다) |
| `test_display_convert` | 불필요 | 표시 변환 셰이더(1:1) vs CPU 기준 **비트 일치**. BT.709 계수 세 번째 사본의 게이트 |
| `test_isp_preproc` | 불필요(ISP 필요) | ISP 전처리 vs CPU 기준. **비트 일치가 아니다**(리샘플 커널이 다르다) — 패딩만 오차 0, 내용은 MAE ≤ 3.0 |

두 경로의 검출을 직접 비교하려면:

```bash
python 06_tools/python/compare_detections.py ref/detections.jsonl test/detections.jsonl --csv out.csv
```

---

## 7. 측정

```bash
bash 06_tools/bench/op_point_bench.sh    # 운용점 스윕 (배치·슬롯)
bash 06_tools/bench/display_bench.sh     # 표시 e2e (M5a 기준선 + M4). CSV 59열 — 이전(48·53열) CSV 와 APPEND 불가
bash 06_tools/bench/pipeline_bench.sh    # 기존 9설정 회귀 기준선
bash 06_tools/bench/decode_bench.sh      # 디코드 경로 비교
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
| `yvo_perf_counters` | V3D 성능 카운터 열거(93개, 1 pass 최대 32개)와 쿼리 동작 확인 (`--csv`, NPU 불필요) |
| `yvo_hailo_raw_probe` | 원시 출력 경로 프로브 (`--probe f0` 이 입출력 dma-buf 동시 사용 관문) |
| `yvo_hailo_selftest` | NPU 러너 단독 검증 |
| `yvo_display_probe` | 표시 경로 프로브 (surface/present/convert/overlay) |
| `yvo_display_pipeline_probe` | `DisplayPipeline` 단독 검증 (`yvo_pipeline` 미사용) |
| `yvo_isp_*` | PiSP BE(ISP) 전처리 — **통합 보류 상태** |
| `06_tools/probes/hvs_probe/` | (빌드 안 함) HVS 두 평면 스캔아웃 KMS/Wayland 프로브 소스. [`notes/kms_vs_wayland.md`](07_results/display_vulkan_hdmi/notes/kms_vs_wayland.md) 의 근거 |

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
| 표시 on 인데 멈춤, SIGTERM 도 안 먹음 | labwc 가 HDMI 를 잃었다(`wlr-randr` 에 `Headless output` 만 보임). 활성 VT 가 데스크톱이 아니거나 Xwayland 가 DRM master 를 가져간 경우다. `kill -9` 로 끝내고, `cat /sys/kernel/debug/dri/1/clients` 에서 master 를 확인한 뒤 Xwayland 를 끝내고 VT 를 한 번 토글(`loginctl activate` 로 tty ↔ 데스크톱)한다 |
| `--display-backend kms` 가 종료 코드 2 | DRM master 를 못 얻었다(데스크톱이 활성). §3.5 대로 비데스크톱 VT 를 활성화한 뒤 실행 |
| 평면 백엔드인데 `color_ok 0` | 컴포지터 COLOR_RANGE 처리가 바뀌었을 수 있다. `--display-color-range` 를 바꿔 보고 debugfs `range=` 를 확인 |

---

## 10. 더 읽을 것

- **[`08_learn/`](08_learn/00_README.md)** — 배경지식이 없다면 **여기부터**. NPU·Vulkan·dma-buf·YOLO 텐서를
  맨 밑바닥부터 설명하고, 각 개념이 이 프로젝트의 어느 결정으로 이어졌는지 연결한 자습 교재 12장
- **[`07_results/README.md`](07_results/README.md)** — 12개 실험의 색인과 결론. **수치를 인용하기 전에 여기부터**
- [`07_results/op_point/summary.md`](07_results/op_point/summary.md) §8 — 운용점 확정 근거와 최종 결론
- [`07_results/display_e2e/summary.md`](07_results/display_e2e/summary.md) — 표시 통합 측정
- [`07_results/pipeline_ffmpeg_vulkan_hailo/REPORT.md`](07_results/pipeline_ffmpeg_vulkan_hailo/REPORT.md) §4.3 — 왜 이 설계인가(메모리 제약)
- [`09_diagrams/`](09_diagrams/README.md) — yEd 로 여는 파이프라인 다이어그램 2장(분기·스레드 경계·dma-buf 소유권)
- [`10_paper/paper_ko.md`](10_paper/paper_ko.md) — 논문 형식의 전체 서술
- `CLAUDE.md` — 폴더 규약, 개발 환경, 서브에이전트 라우팅

### 아직 열려 있는 것

목록의 정본은 **[`07_results/README.md`](07_results/README.md#아직-열려-있는-것)** 다(13항목).
여기서는 운용에 바로 영향을 주는 셋만 적는다.

- **연속 운용 발열** — 측정이 전부 26초짜리다. 몇 시간 도는 운용에서 `b2/s4` 의 30 fps 여유(+16%)가
  클럭 강등을 버티는지 미검증
- **카메라(CSI) 입력** — 디코더가 HEVC 파일만 받는다(`01_src/decode/ffmpeg_drm_decoder.cpp:166`).
  카메라로 가면 ISP 전처리 경로 재검토 조건도 함께 열린다
- **다중 스트림은 불가능하다** — 1080p30 두 개 = 60 FPS 인데 NPU 상한이 51.7 이다.
  확보한 CPU 여유는 이 한계를 풀지 못한다
