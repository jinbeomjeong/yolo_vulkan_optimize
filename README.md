# yolo_vulkan_optimize — 실행 안내

Raspberry Pi CM5 + Hailo-8 NPU 에서 YOLOv11m 추론을 돌리되, 파이프라인의 **CPU 비용을
Vulkan(V3D GPU)으로 밀어낸** 프로젝트다. 디코드·전처리·추론·후처리·가시화가 **하나의 실행 파일**
(`yvo_pipeline`)로 통합돼 있고 전 구간 CPU 복사가 0회다.

실측(1080p30 HEVC, PCIe Gen3 x1, 운용점 `b2/s4 --postproc gpu`, 표시 on, 900프레임 ×5회 중앙값,
2026-10-16 · **V3D 1150 MHz · 디코더 1스레드 · `--nms gpu`** · 코드 `2f2d3d0` · **CPU 거버너 performance 고정**, 5회 중앙값):

| `--display-pace` | FPS | glass-to-glass | 파이프라인 CPU ms/frame (앱 + 컴포지터 + 커널 스레드) | 드롭 |
|---|---:|---:|---:|---:|
| **`source`** (기본·최종 운용) | **29.90** (소스 30 fps 추종) | **평균 97.87 ms** (5회 97.5~100.6, median 89.86) | **2.122** (1.456 + 0.389 + 0.289) | 0 |
| `asap` | 34.89 | 평균 128.74 ms | 2.044 (1.356 + 0.378 + 0.316) | 0 |

검출 수는 표시 off/on, 두 페이싱 전부 **3369 로 동일**하다. 비교 기준(호스트 NMS, 표시 off)은 2.444 ms/frame(앱 2.378)이라 `asap` 은 약 −16%(같은 회차 짝 −14~−24%)다. 표시를 끄면 0.933(−62%)이다.
파이프라인 CPU 합은 실행별 합의 5회 중앙값이라 괄호 안 성분별 중앙값의 합과 조금 다르다. CPU 는 앱의 측정 창에 맞춰 잰 시스템 전체 CPU 에서 무관한 백그라운드를 뺀 값이고, idle 중 hardirq 는 커널 설정상 잡히지 않는다(하한).
근거: [`07_results/display_e2e/runs/v3d1150_quiet_perfgov_20261016/summary.md`](07_results/display_e2e/runs/v3d1150_quiet_perfgov_20261016/summary.md) §2.
같은 날 같은 코드의 ondemand 거버너 회차: [`v3d1150_syscpu_20261016`](07_results/display_e2e/runs/v3d1150_syscpu_20261016/summary.md)(3.189 → 2.505, −21%), [`v3d1150_labwc_20261016`](07_results/display_e2e/runs/v3d1150_labwc_20261016/summary.md)(앱 + 컴포지터), [`v3d1150_20261016`](07_results/display_e2e/runs/v3d1150_20261016/summary.md)(앱만) — 거버너가 달라 CPU 는 직접 비교하지 않는다.
> **이전 측정(2026-09-20): V3D 960 MHz · FFmpeg 디코더 자동 5스레드 · 억제 CPU**(지금의 `--nms cpu`)에서는
> `source` 29.91 FPS · glass 101.20 ms · CPU 2.18, `asap` 34.87 · 130.75 ms · 1.84 였다([`display_e2e/summary.md`](07_results/display_e2e/summary.md) §3·§3.2).
> 그 사이 클럭·디코더 스레드·억제 위치·메인 스레드 대기 코드가 함께 바뀌어 차이를 한 요인으로 돌리지 않는다.
> `source` glass 의 차이(median 101.20 대 89.86)는 두 회차의 실행별 범위가 겹쳐 판정하지 않는다. 옛 조건을 재현하려면 `--nms cpu` 와 `YVO_DEC_THREADS=0` 을 준다
> (V3D 클럭은 펌웨어가 정한다).
> - V3D 클럭이 바뀌어도 FPS·CPU 는 그대로이고 GPU 전처리만 약 14% 빨라졌다([`bootloader_20260925`](07_results/pipeline_ffmpeg_vulkan_hailo/runs/bootloader_20260925/summary.md)).
> - 현재 기본값의 **표시 off 정식 기준선**(2026-10-08): b8s16 51.50 fps · CPU 0.869 ms/frame, GPU NMS 는 CPU NMS 대비 회귀 없음·검출 바이트 동일([`gpu_nms_e2e/`](07_results/gpu_nms_e2e/summary.md)). 표시 on 정식 측정은 [`display_planes_e2e/`](07_results/display_planes_e2e/summary.md)에 있다 — Vulkan 합성 소스 페이싱 b2/s4 CPU 2.059 ms/frame, 평면 경로는 CPU −0.44 ms/f·V3D −72%·지연 같음(`--slots 6` 필요). 위 표와는 회차·코드(메인 스레드 대기 변경 이전)가 달라 섞지 않는다.
> - 혼잡 장면(MOT20, 후보 약 11배)에서도 FPS 그대로·검출 바이트 동일·CPU로 넘긴 프레임 0이다. 디코딩 후보 버퍼 4096 넘침은 이 HEF로는 닿지 않아(양자화 바닥에서도 최대 1925) 고치지 않고 남겼다([`nms_crowd_stress/`](07_results/nms_crowd_stress/summary.md)).

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
> 구간 median 은 [`display_e2e/runs/v3d1150_quiet_perfgov_20261016`](07_results/display_e2e/runs/v3d1150_quiet_perfgov_20261016/summary.md) §2.3(`raw/disp_source_run*/stdout.log` 5회 중앙값)(소스 페이싱, V3D 1150 MHz · 현재 기본값, 논문 §6.7 표 24)이 정본이고
> (960 MHz 당시 값은 [`display_e2e/summary.md`](07_results/display_e2e/summary.md) §3.1), 같은 값을 붙인 그림이 [`09_diagrams/pipeline_overview.graphml`](09_diagrams/pipeline_overview.graphml) 에 있다.

---

## 2. 빌드

```bash
cmake -S . -B build -G Ninja && cmake --build build
```

빌드 경고는 **0줄**이어야 한다(`-Wall -Wextra -Wpedantic -Wshadow -Wnon-virtual-dtor`). 셰이더는 `glslc -Werror` + `spirv-val` 을 자동으로 통과한다.
ASan + UBSan 빌드는 별도 폴더에 `-DYVO_SANITIZE=ON` 을 준다(`cmake -S . -B build-asan -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DYVO_SANITIZE=ON`).

필수: HailoRT 4.24.0, Vulkan(V3DV), libavcodec(RPi 빌드 7.1.5 rpt2 — v4l2-request·SAND), libdrm. 테스트를 빌드하면(`YVO_BUILD_TESTS`, 기본 ON) GTest.
선택: OpenCV — 없으면 경고만 내고 OpenCV 를 쓰는 타깃(PNG 덤프 등)이 빠진다. wayland-client + wayland-protocols. libpisp — 없으면 `yvo_isp` 전체가 빠진다. GStreamer ≥ 1.24(`-base`·`-video`·`-allocators`·`-app`) — 없으면 연구 경로 `yvo_gstelem` 과 `tools/gst/` 도구만 빠진다.
표시 경로(`yvo_display`, kms 백엔드 포함)는 wayland 가 없으면 조용히 빠지고 나머지는 그대로 빌드된다.
wayland-protocols 에 staging `color-representation` XML 이 없으면 그 요청만 빠진다(`wayland-planes` 의 색 range 지정 불가).

---

## 3. 실행

### 기본값 = 측정된 최적 경로 (2026-10-08 변경)

인자 없이 실행하면 `--mode dmabuf --layout nhcw --postproc gpu --batch 8 --display off` 로 돈다(300프레임 1회 확인: CPU 0.904 ms/frame, 검출 동일).
2026-10-08 전에는 **가장 느린 경로** `--mode ptr --layout nhwc --postproc host`(CPU 17.6 ms/frame, 자동 스레드 조건 — 1스레드 16.97)가 기본이었다. 옛 측정을 재현하려면 그 플래그를 명시한다(벤치 스크립트는 명시한다). `--mode ptr` 만 주면 layout·postproc 은 호환 값(nhwc·host)으로 내려간다.
디코더 프레임 스레드는 2026-09-29 에 자동(5) → **1** 로 바꿨다(CPU −14%·e2e −8.8 ms, FPS·검출 동일,
[`dec_threads/`](07_results/dec_threads/summary.md)). 옛 측정과 비교할 때는 `YVO_DEC_THREADS=0`(자동)을 준다.
**표시는 기본 off 다 — 아래 3.1 의 플래그를 준다.**

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
| **실시간 표시** (권장) | `--batch 2 --slots 4` | 표시 off 34.98 FPS / 표시 on `pace source` **29.90 FPS, glass 평균 98 ms(median 90 ms)**(2026-10-16, 거버너 performance, [`v3d1150_quiet_perfgov_20261016`](07_results/display_e2e/runs/v3d1150_quiet_perfgov_20261016/summary.md)). 입력 페이싱(표시 off)이면 도착 기준 평균 80.7 · p90 100 ms, 버림 0([`code_2f2d3d0`](07_results/input_pacing/runs/code_2f2d3d0/summary.md)). 40분 연속 운용에서도 칩이 데워진 평형 상태로 34.25~34.75 FPS 를 유지한다([`thermal_soak/`](07_results/thermal_soak/summary.md)) |
| 최대 처리량 (**단시간만**) | `--batch 8 --slots 16` | 51.5 FPS, e2e 325 ms. ⚠ **추가 냉각 없이는 약 5분 만에 Hailo 칩이 red zone(약 120 °C)에 닿아 칩이 추론을 멈춘다** — 51.5 FPS 는 칩이 식어 있는 처음 약 2분의 값이다([`thermal_soak/`](07_results/thermal_soak/summary.md)) |
| **최소 지연** (O2, 2026-10-12 확정) | `--batch 1 --slots 1 --input-pace source --input-queue 1` | **도착 기준 평균 58.4 · p90 71.7 · max 75.0 ms**(표시 off). 화면까지는 `--display on --display-fullscreen`(Vulkan 합성)으로 **도착 → 화면 평균 70.2 · p90 78.6 ms**, 버림 19.1%(2026-10-16 · 코드 `2f2d3d0` 재측정, [`code_2f2d3d0`](07_results/input_pacing/runs/code_2f2d3d0/summary.md) · [`code_2f2d3d0_display_on`](07_results/input_pacing/runs/code_2f2d3d0_display_on/summary.md) — HVS 평면 백엔드는 slots 1 에서 교착하므로 쓰지 않는다). 처리 24.4 FPS 라 30 fps 입력의 **18.6% 를 버린다**(가장 오래된 프레임부터, 연속 버림 최대 1). 페이싱 없는 `--batch 1 --slots 1` 은 디코드 끝 기준 79.2 ms 지만 그중 약 42 ms 가 파일을 앞서 읽어 생긴 슬롯 대기다([`input_pacing/`](07_results/input_pacing/summary.md)) |
| 영상만 저지연 표시 | 위에 `--display-sync latest` 추가 | 화면 glass 28 ms, 박스만 뒤처진다 |

> 최소 지연 행과 페이싱 값은 **현재 기본값**(V3D 1150 MHz·디코더 1스레드·GPU 후처리·GPU 억제, 2026-10-12)이고, 나머지 지연은
> 디코더 자동 5스레드·V3D 960 MHz 조건 값이다([`op_point/`](07_results/op_point/summary.md) §8). 지금 기본값에서는 b8/s16 e2e 318.6 ms,
> 표시 off b2/s4 GPU 후처리 34.99 FPS 로 조금씩 짧고 FPS 는 같다([`dec_threads/`](07_results/dec_threads/summary.md)).
> 페이싱 없는 b1/s1 은 79.19 ms 이고 gpu-depth 1 과 3 이 같다([`latency_b1s1_20261012`](07_results/op_point/runs/latency_b1s1_20261012/summary.md)).
>
> **지연의 기점을 섞지 않는다.** 입력 페이싱(`--input-pace source`)의 `e2e_arr` 는 프레임이 **도착한 시각**(카메라라면 찍힌 시각)부터 결과까지,
> 페이싱 + 표시 on 의 `arrival-to-present`(a2p)는 **도착한 시각부터 화면에 닿을 때까지**, 페이싱 없는 e2e 는 **디코드가 끝난 시각**부터,
> 표시의 glass-to-glass 는 **슬롯을 얻은 시각**부터 잰다.
> 페이싱하면 30 fps 를 따라가는 b2/s4 는 평균 80.7 ms(b2/s3 80.5 ms, 둘이 같다)·p90 100 ms·버림 0 이다(2026-10-16 · 코드 `2f2d3d0`. 그 전 코드에서는 81.3 · 81.6 · p90 104 ms).
> 입력 큐 크기는 1 이 맞다 — 2·3 은 b2 에서 효과가 없고(큐 최대 깊이 늘 1) b1 에서 한 단계마다 +33 ms 다([`input_pacing/`](07_results/input_pacing/summary.md)).

**지연은 slots 가, 처리량의 천장은 batch 가 정한다.** 지연 ≈ `slots × 프레임 주기` 이고,
FPS = min(배치 천장 `B / T(B)`, 슬롯 제약)이며 천장은 **slots = 2 × batch** 에서 닿는다
([`grid_gpu_postproc`](07_results/op_point/runs/grid_gpu_postproc/summary.md)). 그보다 큰 slots 는 지연만 늘린다.
**batch 1 은 천장이 24.4 FPS** 라(배치당 고정비 24.8 ms) 30 fps 소스를 다 처리하려면 batch ≥ 2 가 필요하다.
입력 페이싱 + 큐 1 이면 batch 1 도 실시간을 유지한다 — 처리하지 못한 약 19% 를 버리고 가장 최근 프레임을 넣는다(최소 지연 운용점).

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

### 3.5 HVS 평면 표시 (`wayland-planes` / `kms`) — 구현·검증·정식 측정됨

V3D 가 표시 변환을 하지 않는 두 번째 표시 방식이다. 디코더 SAND 버퍼를 **디스플레이 컨트롤러(HVS)의 영상 평면**에 그대로 올리고,
박스는 GPU 가 AB24 버퍼에 그려 **그 위 평면**에 올린다. YUV→RGB 와 블렌딩은 HVS 가 스캔아웃 중에 한다.
두 백엔드 모두 실제 HDMI 에서 평면 배치와 색(BT.709 limited)을 debugfs 로 확인했다. 정식 측정(2026-10-08,
[`07_results/display_planes_e2e/`](07_results/display_planes_e2e/summary.md))에서 Vulkan 합성 대비 **CPU −0.44 ms/f(−20%)·V3D −72%** 이고,
디코드 → 화면 지연은 세 경로가 같다(평균 약 101~102 ms). **평면 경로는 버퍼를 최대 5장 쥐므로 `--slots 6` 이 필요하다** —
슬롯 4에서는 디코드 후 빈 슬롯을 평균 28~30 ms 기다린다. wayland-planes 와 kms 는 모든 지표가 겹친다
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
| `--mode ptr\|dmabuf` | `dmabuf` | **`dmabuf` 필수.** V3D 포인터는 Hailo 가 DMA 매핑하지 못한다 |
| `--layout nhwc\|nhcw` | `nhcw` | **`nhcw` 필수.** HailoRT 의 CPU 입력 변환(약 14 ms/f)을 없앤다. dmabuf 전용 |
| `--postproc host\|gpu` | `gpu` | `gpu` 는 원시 출력 + `yolo_decode.comp` + `--nms` 억제. **`dmabuf`+`nhcw` 전용** |
| `--nms gpu\|cpu\|verify` | `gpu` | `--postproc gpu` 의 억제 위치. `gpu` 는 `efficient_nms.comp`(CPU 기준과 비트 일치, 보장 못 하는 프레임만 자동 CPU 처리), `cpu` 는 `cpu_nms_greedy_by_class`, `verify` 는 매 프레임 둘을 비트 비교 |
| `--batch N` / `--slots N` | 8 / `max(4,batch*2)` | 운용점. §3.3 참조 |
| `--gpu-depth N` | 3 | GPU 동시 프레임 상한. **지연에 영향 없음**(실측 1 vs 3 동일) |
| `--input-pace asap\|source` | `asap` | 입력 도착 방식. `asap` = 슬롯이 비는 대로 디코드(지금까지의 방식). `source` = 프레임이 `t0 + k/fps` 에 도착한다(카메라 흉내). 요약에 도착 시각부터 잰 `e2e_arr`·버림·`RESULT_CSV4` 가, 표시 on 이면 도착 → 화면 `arrival-to-present`·`RESULT_CSV5` 가 붙는다(표시 쪽 프레임 게이트 `--display-pace` 는 입력이 이미 페이싱되므로 쓰지 않는다) |
| `--input-queue N` | 1 | `--input-pace source` 전용 leaky 입력 큐 크기 1~3. 넘치면 가장 오래된 프레임을 버린다(GStreamer `queue leaky=downstream`). **1 이 맞다** — 2~3 은 효과 없거나 지연만 는다([`input_pacing/`](07_results/input_pacing/summary.md)). 보유 예산 `gpu_depth + N + 1 ≤ 10`(표시 on 이면 표시 쪽 보유 상한에 N 을 더한다) |
| `--frame-log PATH` | 끔 | 프레임마다 도착·디코드 끝·제출·NPU 완료·후처리 완료·완료 시각(실행 시작 기준 ms)을 CSV 로 남긴다. 실행 끝에 한 번만 쓰고(루프 안 파일 I/O 없음), 경로가 잘못되면 장치를 열기 전에 종료 코드 2. `dropped` 1 = 입력 큐에서 버림, 2 = 중단 시 큐에 남음. 표시 on 이면 10번째 열 `present_ms`(화면에 닿은 시각)가 붙는다. 분석은 `06_tools/python/frame_log_stats.py` |
| `--display off\|on` | `off` | HDMI 실시간 표시 |
| `--display-backend auto\|wayland\|offscreen\|wayland-planes\|kms` | `auto` | `auto` = Wayland 창이 열리면 `wayland`, 아니면 `offscreen`. `wayland` = Vulkan 합성 + 스왑체인(strict — 실패하면 종료), `offscreen` = 화면 없이 변환만. `wayland-planes`/`kms` = HVS 평면 스캔아웃(§3.5, strict) |
| `--display-color-range auto\|full\|limited` | `auto` | 평면 백엔드가 요청할 COLOR_RANGE. `auto` = wayland-planes 는 full(labwc 뒤집힘 우회), kms 는 limited |
| `--display-connector NAME` | 첫 연결 HDMI-A | `kms` 의 출력 커넥터(다른 백엔드에서는 경고 후 무시) |
| `--display-fullscreen` | 꺼짐 | **반드시 켠다.** 창 모드는 에러·경고 없이 표시 변환이 **2.6배**(4.22 → 10.92 ms, V3D 960 MHz 측정) 느려진다. 평면 백엔드는 자동으로 전체화면이다 |
| `--display-sync matched\|latest` | `matched` | `matched` 는 프레임과 그 검출을 정확히 맞춘다. `latest` 는 화면을 즉시 띄우고 박스만 뒤처진다 |
| `--display-pace source\|asap` | 표시 on 이면 `source` | 소스 fps 로 페이싱. `asap` 이면 큐가 꽉 차 glass 평균 97.87 → 128.74 ms (median 89.86 → 128.04, 1150 MHz · 1스레드 · GPU NMS · CPU 2.4 GHz 고정) |
| `--postproc-alloc export\|heap` | `export` | **`heap` 은 쓰지 마라** — 3자 일관성(PCIe DMA ↔ CPU 캐시 ↔ V3D)이 미검증이다 |
| `--score T` | HEF 값 0.2 | 검출 점수 임계값 |
| `--frames N` / `--quiet` / `--csv-label L` | | 측정용 |

전체 목록은 `yvo_pipeline --help`.

환경변수(인자가 아니다):

| 변수 | 기본 | 설명 |
|---|---|---|
| `YVO_DEC_THREADS` | 1 | FFmpeg HEVC 디코더 프레임 스레드 수. `0` = FFmpeg 자동(코어 수, 2026-09-29 이전 기본). 1스레드가 CPU·지연 모두 낫다([`dec_threads/`](07_results/dec_threads/summary.md)) |
| `YVO_PERF_CTL` / `YVO_PERF_ACK` | 없음 | `perf ... -D -1 --control fifo:<ctl>,<ack>` 의 FIFO 경로. 주면 `cpu_ms_per_frame` 과 **같은 구간만** 프로파일한다(`01_src/toolkit/tool_util.hpp` 의 `perf_window`). perf 없이 CTL 만 주면 멈춘다 |

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
ctest --test-dir build        # 184/184 통과해야 한다
```

| 테스트 | NPU | 무엇을 보는가 |
|---|---|---|
| `PostprocFixture.*` (6개) | 불필요 | GPU 후처리 vs CPU 기준: 후보 수 동일, 점수 **비트 일치**, 좌표 < 0.25 px, 10회 결정성, GPU NMS 결과 **비트 일치**, heap 입력 경로(`YoloInputAlloc::ImportHeap`)가 Plain과 **비트 일치** |
| `EfficientNms.*` (7개) | 불필요 | 합성 입력: 동점, 클래스 정원 100 초과, 클래스 간 겹침, 용량 초과·IoU 보호대 → CPU 처리 신호, 결정성 |
| `test_postproc_live` | 필요 | host 경로 vs GPU 경로 검출 매칭 (≥ 0.95) |
| `test_detections` | 필요 | CPU/GPU 전처리 경로 검출 일치 |
| `test_preproc` | 불필요 | GPU 전처리가 CPU 기준과 **비트 일치**(max 오차 0). `SandRefVsFfmpeg.*`(GPU 불필요): 기준 de-SAND(`ref/sand.hpp`)가 rpi-ffmpeg `av_hwframe_transfer_data`(NV12)와 **바이트 일치**(프레임 0/100/500/900/1815, 보이는 1920×1080) |
| `test_overlay` | 불필요 | 오버레이 셰이더 vs CPU 기준 비트 일치. 박스 평면 dma-buf(`OverlayPlane.*`, 잔상 0), debugfs 파서(`DrmState.*`), KMS 탐색·AddFB2(`KmsScanout.*` — card1 사용, master 는 가져가지 않는다) |
| `test_display_convert` | 불필요 | 표시 변환 셰이더(1:1) vs CPU 기준 **비트 일치**. 표시 경로 BT.709(`02_shaders/bt709.glsl` 공유 include)의 게이트 |
| `test_pipeline_logic` | 불필요 | (하드웨어 없음) `SlotExchange` 슬롯 반납·중복 신호·다중 스레드, `validate_slot_budget` 보유 상한, RESULT_CSV·RESULT_CSV4·`--frame-log` 서식 바이트 일치, 엄격한 숫자·크기 옵션 파싱, 입력 페이싱 순수 로직(`LeakyQueue`·`ArrivalSchedule`), 입력 큐를 더한 보유 예산, RESULT_CSV5 |
| `test_display_options` | 불필요 | (하드웨어 없음) 표시 옵션 문자열 → enum 파서, 평면 백엔드 늦은 프레임 허용 간격(`LateGapThreshold.*` — 입력 큐에서 버린 원본 프레임을 늦음으로 세지 않는다) |
| `test_isp_preproc` | 불필요(ISP 필요) | ISP 전처리 vs CPU 기준. **비트 일치가 아니다**(리샘플 커널이 다르다) — 패딩만 오차 0, 내용은 MAE ≤ 3.0 |
| `test_gst_adapter` | 불필요 | (선택, GStreamer 연구 경로) 어댑터 `DrmFrame`·`yvosandletterbox` 출력이 FFmpeg 경로 NHCW와 **바이트 일치**(프레임 0/100/1815), 텐서 메타·슬롯 fd 고정, sink caps 는 SAND DMA_DRM 만, 슬롯 메모리를 다른 버퍼로 옮겨도 덮어쓰지 않고 메모리 교체로 슬롯이 새지 않음(`GstElementSlots.*`) |
| `test_gst_detections` | 필요 | (선택, GStreamer 연구 경로) `yvohailosink` 검출이 `yvo_pipeline --mode dmabuf --layout nhcw --postproc host` 추론 경로와 **프레임마다 동일**(앞 120프레임, 배치 4, `YVO_GST_DET_FRAMES`·`YVO_GST_DET_BATCH`) |

NPU 가 필요한 3개(`test_detections`·`test_postproc_live`·`test_gst_detections`)에는 `npu` 라벨이 있어 `ctest --test-dir build -LE npu` 로 뺄 수 있다.

두 경로의 검출을 직접 비교하려면:

```bash
python 06_tools/python/compare_detections.py ref/detections.jsonl test/detections.jsonl --csv out.csv
```

---

## 7. 측정

```bash
bash 06_tools/bench/op_point_bench.sh    # 운용점 스윕 (배치·슬롯)
bash 06_tools/bench/display_bench.sh     # 표시 e2e (M5a 기준선 + M4). CSV 70열(labwc CPU 2열 + 앱 CPU 창에 맞춘 시스템 전체 CPU sys_* 9열, `IDLE_S` 기본 10 s 기준선) — 이전(48·53·59·61열) CSV 와 APPEND 불가
bash 06_tools/bench/pipeline_bench.sh    # 기존 9설정 회귀 기준선
bash 06_tools/bench/decode_bench.sh      # 디코드 경로 비교
bash 06_tools/bench/dec_threads_bench.sh # 디코더 프레임 스레드 수 스윕 + 자동 대 1스레드 회귀 A/B
bash 06_tools/bench/gst_pipeline_bench.sh # 앱 대 GStreamer 연구 경로(yvo_gst_pipeline)
bash 06_tools/bench/input_pace_bench.sh  # 입력 페이싱 + leaky 큐 N=1·2·3 (b1s1·b2s3·b2s4 × asap 대조, 도착 900프레임). DISPLAY_RUN=1 = 표시 on(HDMI 필요)
```

벤치 스크립트는 모두 12개다. 나머지(`backpressure`·`isp`·`isp_pipeline`·`display_planes`·`nms`·`input_pace`)까지 포함한 목록은
[`08_learn/09_code_map.md`](08_learn/09_code_map.md) §9.7.

공통 환경변수: `RUNS`(기본 3), `FRAMES`, `ONLY`(설정 id grep), `APPEND=1`, `BIN`, `OUT_DIR`,
`COOL_C`/`HOT_C`/`COOL_MAX_S`. 호출자의 `YVO_DEC_THREADS` 는 앱에 그대로 전달된다(`dec_threads_bench.sh` 만 설정별로 정한다) —
2026-09-29 이전 CSV 와 CPU·e2e 를 비교하려면 `YVO_DEC_THREADS=0` 을 준다.

측정 규약은 `.claude/skills/measure` 에 있다. 요점만:
**설정당 3회 중앙값, 반복이 바깥 루프, 칩 75 °C 초과 시 65 °C 까지 냉각,
`hailort.log` 에 온도·클럭 경고가 있으면 그 실행의 성능 수치는 신뢰 불가.**

⚠ **측정 중에 빌드·테스트를 돌리지 마라.** CPU 시간 측정이 조용히 오염된다.
`.claude/hooks/npu_guard.sh` 가 NPU 충돌을 막고, 벤치가 도는 동안(`build/.bench_running` 락) 빌드·테스트도 막는다.
`yvo_gst_pipeline`(`--sink hailo` 는 NPU, `fake`/`hold` 는 부하)과 `test_gst_detections` 도 인식한다.
벤치 도중 `Ctrl+C` 를 누르면 남은 설정으로 넘어가지 않고 락을 지운 뒤 종료 코드 130(SIGTERM 은 143)으로 멈춘다.

---

## 8. 진단 도구

| 도구 | 용도 |
|---|---|
| `yvo_drm_dump` | 디코더 dma-buf descriptor 덤프, de-SAND 바이트 비교, 디코드 성능 |
| `yvo_preproc_bench` | GPU 전처리 단독 성능·워크그룹 스윕 |
| `yvo_postproc_probe` | GPU 후처리 프로브 (`--probe fixture` 는 **NPU 불필요**, 수 초. `--probe stages` 는 디코딩 단계별 GPU 시간 — `--variants` 로 진단 변형 선택. `--probe groups` 는 프리필터 통과 수 분포(CPU 전용), `--probe sweep` 은 local size 스윕) |
| `yvo_perf_counters` | V3D 성능 카운터 열거(93개, 1 pass 최대 32개)와 쿼리 동작 확인 (`--csv`, NPU 불필요) |
| `yvo_hailo_raw_probe` | 원시 출력 경로 프로브 (`--probe f0` 이 입출력 dma-buf 동시 사용 관문) |
| `yvo_hailo_selftest` | NPU 러너 단독 검증 |
| `yvo_display_probe` | 표시 경로 프로브 (surface/present/convert/overlay) |
| `yvo_display_pipeline_probe` | `DisplayPipeline` 단독 검증 (`yvo_pipeline` 미사용) |
| `yvo_isp_*` | PiSP BE(ISP) 전처리 — **통합 보류 상태** |
| `yvo_gst_sand_probe` | (선택) GStreamer `v4l2slh265dec` DMABuf → `DrmFrame` → GPU 전처리(NHCW)가 FFmpeg 경로와 바이트 일치하는지 가르는 관문 |
| `yvo_gst_pipeline` | (선택) GStreamer 연구 경로(B안): `v4l2slh265dec` → `yvosandletterbox`(정적 등록 요소, Vulkan NHCW) → `yvohailosink`(기본, NPU 추론·host NMS — `yvo_pipeline --mode dmabuf --layout nhcw --postproc host`와 같은 `RESULT_CSV`·`--dump-dir`) 또는 fakesink/hold. 슬롯 역압·정지 감시 |
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
| 성능이 들쭉날쭉 | 칩 온도를 봐라. 약 104 °C(orange zone)부터 클럭이 400→350→300 MHz… 로 단계적으로 떨어지고, 약 120 °C(red zone)에서 칩이 추론을 멈춘다([`thermal_soak/`](07_results/thermal_soak/summary.md)) |
| FPS 가 예상의 절반 | 과거에 러너 백프레셔 이중 계수 결함이 있었다(해소됨). `in-flight = FPS × npu_latency` 가 `slots` 에 가까운지 봐라 |
| 표시 on 인데 멈춤, SIGTERM 도 안 먹음 | labwc 가 HDMI 를 잃었다(`wlr-randr` 에 `Headless output` 만 보임). 활성 VT 가 데스크톱이 아니거나 Xwayland 가 DRM master 를 가져간 경우다. `kill -9` 로 끝내고, `cat /sys/kernel/debug/dri/1/clients` 에서 master 를 확인한 뒤 Xwayland 를 끝내고 VT 를 한 번 토글(`loginctl activate` 로 tty ↔ 데스크톱)한다 |
| 숫자 옵션에 `<prog>: invalid value for <opt>: '<text>'` 로 종료 코드 2 | 숫자가 아니거나, 뒤에 문자가 붙었거나(`10x`), 정수 자리에 소수(`4.5`)거나, 범위 밖이다. 2026-10-11 전에는 `--frames abc` 가 조용히 0(= 전체 영상)이 됐다 |
| `--display-backend kms` 가 종료 코드 2 | DRM master 를 못 얻었다(데스크톱이 활성). §3.5 대로 비데스크톱 VT 를 활성화한 뒤 실행 |
| 평면 백엔드인데 `color_ok 0` | 컴포지터 COLOR_RANGE 처리가 바뀌었을 수 있다. `--display-color-range` 를 바꿔 보고 debugfs `range=` 를 확인 |

---

## 10. 더 읽을 것

- **[`08_learn/`](08_learn/00_README.md)** — 배경지식이 없다면 **여기부터**. NPU·Vulkan·dma-buf·YOLO 텐서를
  맨 밑바닥부터 설명하고, 각 개념이 이 프로젝트의 어느 결정으로 이어졌는지 연결한 자습 교재 13장
- **[`07_results/README.md`](07_results/README.md)** — 25개 실험의 색인과 결론. **수치를 인용하기 전에 여기부터**
- [`07_results/op_point/summary.md`](07_results/op_point/summary.md) §8 — 운용점 확정 근거와 최종 결론
- [`07_results/display_e2e/summary.md`](07_results/display_e2e/summary.md) — 표시 통합 측정
- [`07_results/pipeline_ffmpeg_vulkan_hailo/REPORT.md`](07_results/pipeline_ffmpeg_vulkan_hailo/REPORT.md) §4.3 — 왜 이 설계인가(메모리 제약)
- [`09_diagrams/`](09_diagrams/README.md) — yEd 로 여는 파이프라인 다이어그램 3장(원본 2장 — 분기·스레드 경계·dma-buf 소유권 — + 논문판 1장)
- [`10_paper/paper_ko.md`](10_paper/paper_ko.md) — 논문 형식의 전체 서술
- `CLAUDE.md` — 폴더 규약, 개발 환경, 서브에이전트 라우팅

### 아직 열려 있는 것

목록의 정본은 **[`07_results/README.md`](07_results/README.md#아직-열려-있는-것)** 다(22항목).
여기서는 운용에 바로 영향을 주는 넷만 적는다.

- **연속 운용 발열** (일부 닫힘) — `b2/s4` 는 40분 지속 가능(평형 34.25~34.75 FPS), `b8/s16` 은 추가 냉각 없이
  약 5분에 칩 보호 정지. 중간 운용점·냉각 장치·40분 초과는 미검증
- **카메라(CSI) 입력** — 디코더가 HEVC 파일만 받는다(`01_src/decode/ffmpeg_drm_decoder.cpp:166`).
  프레임이 소스 시각에 도착하는 상황은 입력 페이싱으로 모사해 쟀다([`input_pacing/`](07_results/input_pacing/summary.md)) —
  실제 센서의 촬영 시각·ISP 지연은 미측정이다. 카메라로 가면 ISP 전처리 경로 재검토 조건도 함께 열린다
- **최소 지연 운용점의 표시는 Vulkan 합성으로만** — HVS 평면 백엔드는 slots 1 에서 교착하고(기존 제약), slots 2 는 처리가 vblank 3개(50 ms)에 잠긴다. kms 와 표시 on 입력 큐 2 이상은 미측정([`code_2f2d3d0_display_on`](07_results/input_pacing/runs/code_2f2d3d0_display_on/summary.md))
- **다중 스트림은 불가능하다** — 1080p30 두 개 = 60 FPS 인데 NPU 상한이 51.7 이다.
  확보한 CPU 여유는 이 한계를 풀지 못한다
