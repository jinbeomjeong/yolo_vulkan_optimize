# 09_diagrams — 파이프라인 다이어그램

최종 확정 파이프라인을 yEd 로 열고 편집할 수 있는 `.graphml` 2장으로 옮긴 것이다. 논문용으로 줄인 판(`pipeline_paper.graphml`) 1장이 따로 있다.
`README.md` §1 의 ASCII 도식과 같은 내용이되 분기·재합류가 그래프로 정확히 표현돼 있다.

| 파일 | 내용 | 노드 / 엣지 |
|---|---|---:|
| [`pipeline_overview.graphml`](pipeline_overview.graphml) | 개요 1장. 소스 → 디코드 → GPU 전처리 → NPU → GPU 후처리 → GPU NMS → 오버레이 → present → HDMI 와 **표시 분기·재합류**, 구간별 실측 지연, **대안 표시 경로(HVS 평면 스캔아웃)** | 29 / 19 |
| [`pipeline_detail.graphml`](pipeline_detail.graphml) | 상세 1장. **스레드 경계**(메인 / HailoRT 콜백), **dma-buf 소유권 3그룹**(+ 평면 백엔드의 ④ AB24 박스 dma-buf), 슬롯 수명, 제약·전제·함정·종료 순서, 대안 표시 경로(`PlaneDisplay`), 미연결 경로 | 39 / 22 |
| [`pipeline_paper.graphml`](pipeline_paper.graphml) | **논문판** 1장(폭 17.5 cm 그림용 단순화판, `10_paper/paper_ko.md` §4.1 의 ASCII 그림을 대체). HEVC 파일 → HW 디코더 → GPU 전처리 → NPU → GPU 후처리(디코드 + NMS) → Vulkan 합성 → present → HDMI, **대안 표시 경로(HVS 평면, 청색 점선)**, dma-buf 전달을 간선 라벨로, CPU 의 몫(제어만, 복사 0)을 띠 노드로. 번호 ①~⑦ 은 §4.1 그대로. **구간 시간 수치는 넣지 않았다**(조건에 따라 달라짐 — 수치는 위 두 장과 07_results). 내보내기: `yed-auto export`(아래 §내보내기) → `10_paper/figures/fig03_pipeline.svg` / `.png` | 20 / 12 |

세 장 모두 **좌표를 직접 지정**해 두었으므로 yEd 에서 레이아웃을 다시 돌릴 필요가 없다.
모든 노드의 `description`(`d5`)에 **소스 파일 절대경로·심볼·수치 출처**가 들어 있다 —
yEd 에서 노드를 선택하고 Properties 창(F6)을 열면 보인다.

## 여는 법

```bash
cd /home/pi/yed
DISPLAY=:0 java $(tr -d '\r' < /home/pi/yed/yed.vmoptions | tr '\n' ' ') -jar yed.jar \
  /home/pi/workspace/yolo_vulkan_optimize/09_diagrams/pipeline_overview.graphml
```

`java -jar` 는 `yed.vmoptions` 를 **자동으로 읽지 않는다.** 위처럼 직접 넘기지 않으면
`InaccessibleObjectException` / `IllegalAccessError` 가 난다.

⚠ **`yed.vmoptions` 는 CRLF 파일이다.** `tr -d '\r'` 를 빼면 인자가 `--add-opens\r` 이 되어
`Unrecognized option: --add-opens` / `Could not create the Java Virtual Machine` 으로 죽는다.

Swing 에는 Wayland 백엔드가 없으므로 `WAYLAND_DISPLAY` 가 아니라 `DISPLAY=:0`(XWayland) 이다.
창이 뜨고도 내용이 안 그려져 보이면 창을 한 번 클릭해 포커스를 준다(XWayland + Swing 의 첫 repaint 문제).

위 명령은 **사용자 화면**에 창을 띄운다 — 사람이 보고 편집할 때 쓴다(`CLAUDE.md` §사용자 확인: 사용자 환경 점유).

## 내보내기·렌더 확인 (가상 화면, 사람 불필요)

yEd 자체에는 명령행 내보내기가 없다. 대신 전역 런처 `~/.claude/tools/yed-auto/yed-auto` 가 yEd 를 비공개 Xvfb 화면에서 띄워
File ▸ Export 를 대신 눌러 준다. 사용자 화면을 건드리지 않고, 원본 파일과 `~/.yEd` 설정을 바꾸지 않는다(임시 사본으로 연다).

```bash
~/.claude/tools/yed-auto/yed-auto check  09_diagrams/pipeline_paper.graphml          # 열림·노드/간선 수·로드 오류
~/.claude/tools/yed-auto/yed-auto export 09_diagrams/pipeline_paper.graphml <out>.svg  # .png .pdf 도 같다
~/.claude/tools/yed-auto/yed-auto render 09_diagrams/pipeline_paper.graphml <out>.png  # 화면 캡처(눈으로 확인용)
```

- 각 형식의 옵션 대화상자는 기본값으로 통과한다. 비트맵 너비는 `--width PX`, 배율은 `--scale S`.
- 측정 중(`build/.bench_running`)에는 `npu_guard` 훅이 막는다 — JVM·Xvfb 가 몇 초간 CPU 를 쓴다.
- 결정: yEd 는 가상 화면 런처로 자동화한다(Xvfb 설치, Java 런처) (사용자 결정, 2026-10-16).

## 색 규약 (세 장 공통)

| 색 | 뜻 |
|---|---|
| `#99CC00` | V3D GPU — Vulkan 컴퓨트 셰이더 |
| `#FFCC00` | Hailo-8 NPU — HailoRT `infer_async` / 콜백 스레드 |
| `#E8EEF7` | CPU — 메인 스레드 |
| `#B39DDB` | HW 디코더 — libavcodec DRM_PRIME |
| `#4DB6E2` | 표시 — Wayland 스왑체인 present · HVS 평면 스캔아웃 |
| `#FF0000` | 소스 · 싱크 |
| `#FFFFCC` | 데이터 — dma-buf 버퍼 (복사 0) |
| 회색 + 점선 테두리 | 미연결 경로 (코드에는 있으나 파이프라인에 연결돼 있지 않음) |
| 청색 `#1E6FA8` 점선 간선 + 청색 굵은 테두리 | **대안 경로** — 연결돼 있고 옵션으로 고른다(`--display-backend wayland-planes` / `kms`). 기본 경로(검정 실선)와 나란히 그린다 |

각 파일 안에도 범례 노드(`n_leg_t`, `n_leg0`~)가 들어 있다. 대안 경로 범례는 overview `n_leg7`, detail `n_leg9`.

## 수치의 출처

**지연 수치의 정본은 `README.md` §1 의 ASCII 가 아니라 아래 측정 문서다.**
둘이 다르면 측정 문서를 따랐고, 그 사실을 해당 노드의 `description` 에 적어 두었다.

**측정 조건 (2026-09-30 표기 추가).** 아래 표에서 ⓒ 표시한 수치는 모두 `display_e2e`(2026-09-20), 즉 2026-09-28 이전 조건의 값이다 —
**V3D 960 MHz**(부트로더 2025-12-08) · **FFmpeg 프레임 스레드 자동(5)** · **CPU NMS**. 지금 기본값은 셋 다 다르다.

- V3D 1150 MHz — 부트로더 2026-09-25. FPS·CPU 는 그대로이고 GPU 전처리가 b1 2.11→1.81 ms, b8 평균 약 2.6→2.0 ms 로 줄었다
  ([`bootloader_20260925/summary.md`](../07_results/pipeline_ffmpeg_vulkan_hailo/runs/bootloader_20260925/summary.md)).
- 앱 디코더 프레임 스레드 기본 1 — `01_src/pipeline/main.cpp:196`, 환경변수 `YVO_DEC_THREADS`(0 = FFmpeg 자동, 읽는 곳 `main.cpp:348`).
  표시 on disp_source 에서 CPU 2.073→1.778 ms/f, glass 평균 99.48→99.52 ms, 앱 e2e p90 +1.9 ms
  ([`dec_threads/summary.md`](../07_results/dec_threads/summary.md)).
- NMS 기본 `--nms gpu` — **정식 측정 2026-10-08**([`gpu_nms_e2e/summary.md`](../07_results/gpu_nms_e2e/summary.md), 1150 MHz · 디코더 1스레드 · 표시 off, 설정당 3회 중앙값).
  GPU NMS median b2s4 0.0486 / p90 0.0578 ms, b8s16 0.0500 / 0.0595 ms, fallback 0, 검출 9457 이 CPU NMS 와 바이트 동일,
  V3D CSD 잡 프레임당 4 → 5. CPU NMS 가 원래 약 2 µs/f 라 판정은 "CPU 절감"이 아니라 "회귀 없음 + GPU 비용 약 0.05 ms"다.
  같은 실험이 현재 기본값의 표시 off 기준선이다 — b8s16 51.50 fps · CPU 0.869 ms/f, b2s4 34.99 fps · CPU 0.908 ms/f.
- 대안 표시 경로(HVS 평면) — **정식 측정 2026-10-08**([`display_planes_e2e/summary.md`](../07_results/display_planes_e2e/summary.md) b2/s4,
  [`runs/slots6/summary.md`](../07_results/display_planes_e2e/runs/slots6/summary.md) b2/s6; 1150 MHz · 디코더 1스레드 · GPU NMS · 표시 on source 페이싱).
  Vulkan 합성 대비 CPU −0.44~0.49 ms/f, V3D 10.2 → 2.9 ms/f(TFU 3.7 ms 소멸), wayland-planes ≈ kms.
  평면 경로는 프레임을 release 까지 최대 5장 쥐어(held_frames_max 5) s4 에서는 디코드된 프레임이 슬롯을 28~30 ms 기다리고,
  **s6 에서 0** 이 된다. s6 의 디코드 → 화면 평균은 vk 101.4 / wlp 102.3 / kms 101.6 ms 로 **지연 이득은 없다.**
  source 페이싱에서 약 25~27% 의 프레임이 한 vblank 늦게 나간다(열린 항목, 원인 미검증).

2026-09-30 표기 때는 재측정 전이라 다이어그램 수치를 바꾸지 않았고, 라벨에 `[960 MHz]`·`[측정 조건: FFmpeg 자동 5스레드]`, `n_note_perf` 에 측정 조건 3줄을 붙였다.
**2026-10-08 정식 측정(GPU NMS · 평면 표시)은 옛 수치를 대체하지 않고 현재 조건 값으로 더했다**(아래 표의 ⓝ = V3D 1150 MHz · 디코더 1스레드 · `--nms gpu`).
라벨에는 `[1150 MHz]`·`[s6]`·`b2/s6` 같은 조건 표기를 붙였고(detail `n0::n8`·`n4` 는 폭이 좁아 조건을 `d5` 와 `n_src_note` 로 넘겼다), 조건·출처 전문은 각 노드 `d5` 에 있다. 두 summary 의 주의 절대로 옛 기준선(display_e2e)과 섞지 않는다.
앱 대 GStreamer 연구 경로 비교는 [`gst_pipeline/summary.md`](../07_results/gst_pipeline/summary.md) 에 있다(다이어그램에는 출처로만 넣었다).

| 항목 | 출처 |
|---|---|
| GPU 구간별 median `[1150 MHz]` (전처리 1.812 / 후처리 0.400 / 변환 3.662 / 오버레이 0.124 ms, NMS 포함 5패스 합 6.036 ms = 21.1%) — 2026-10-16 재측정, 현재 기본값(1스레드·GPU NMS), `disp_asap` | [`07_results/display_e2e/runs/v3d1150_20261016/summary.md`](../07_results/display_e2e/runs/v3d1150_20261016/summary.md) §2.1 |
| (이전 값) ⓒ GPU 4패스 구간별 median (전처리 2.088 / 후처리 0.626 / 변환 4.153 / 오버레이 0.128 ms, 합 6.995 ms) — V3D 960 MHz. 노드 `d5` 에 "이전 값"으로만 남는다 | [`07_results/display_e2e/summary.md`](../07_results/display_e2e/summary.md) §3.1 |
| `n_note_perf` glass-to-glass 100.10 ms(median) · p90 116.04 ms, FPS · CPU ms/f · 검출 3369 · dropped 0 · held_frames_max · GPU 5패스 합 — 2026-10-16 회차 `[1150 MHz]` | [`07_results/display_e2e/runs/v3d1150_20261016/summary.md`](../07_results/display_e2e/runs/v3d1150_20261016/summary.md) §2.1·§2.2 |
| (이전 값) ⓒ glass-to-glass 101.20 ms(median) · p90 118.36 ms, FPS · CPU ms/f — V3D 960 MHz | [`07_results/display_e2e/summary.md`](../07_results/display_e2e/summary.md) §3·§3.2 |
| ⓝ GPU NMS median b2s4 0.0486 / p90 0.0578 ms · b8s16 0.0500 / 0.0595 ms · fallback 0 · 검출 9457 바이트 동일 · CSD 잡 4 → 5 · 표시 off 기준선 b2s4 34.99 fps / CPU 0.908 ms/f, b8s16 51.50 / 0.869 | [`07_results/gpu_nms_e2e/summary.md`](../07_results/gpu_nms_e2e/summary.md) §2·§3·§5 |
| ⓝ 평면 표시 b2/s4 — CPU vk 2.059 → wlp 1.569 / kms 1.576 ms/f · V3D 10.19 → 2.89 / 2.86 ms/f(TFU 3.72) · 평면 slot wait 28~30 ms · 박스 그리기+지우기 평균 0.336 / 0.330 ms · vk_source convert 평균 3.234 / overlay 0.111 ms | [`07_results/display_planes_e2e/summary.md`](../07_results/display_planes_e2e/summary.md) §2·§3 |
| ⓝ 평면 표시 b2/s6 — slot wait 0 · held_frames_max 5 · CPU 2.189 → 1.747 / 1.759 ms/f · 디코드 → 화면 평균 vk 101.43 / wlp 102.25 / kms 101.58 ms · 늦은 프레임 약 25~27% | [`07_results/display_planes_e2e/runs/slots6/summary.md`](../07_results/display_planes_e2e/runs/slots6/summary.md) §2·§3 |
| ⓒ decode call 0.107 ms(FFmpeg 자동 5스레드) · NPU median 69.7 / mean 73.9 ms · present 간격 16.955 ms · cpu nms 2.667 µs (CPU NMS 당시 측정 — 지금 기본은 GPU NMS, description 에만 남김) | [`07_results/display_e2e/raw/disp_source_run1/stdout.log`](../07_results/display_e2e/raw/disp_source_run1/stdout.log) |
| 운용점 `b2/s4 --postproc gpu` 와 최종 실행 커맨드 | [`07_results/op_point/summary.md`](../07_results/op_point/summary.md) §8 |
| 스레드·슬롯 규약, 종료 순서 | `01_src/pipeline/main.cpp:1-54` |
| `slots + gpu_depth <= 10` 강제 (Vulkan 합성 경로) | `01_src/pipeline/slot_exchange.cpp:64-69` (호출 `main.cpp:391-402`) |
| 평면 백엔드 보유 상한 `slots + 1 (+5 latest) <= 10`. 평면 경로는 release 까지 최대 5장을 쥐어 b2/s4 에서 슬롯 대기 28~30 ms — **표시 on 평면 경로는 `--slots 6` 필요**(ⓝ [`runs/slots6`](../07_results/display_planes_e2e/runs/slots6/summary.md) 결론 4) | `01_src/pipeline/slot_exchange.cpp:33-54` (호출 `main.cpp:391-402`) |
| `--postproc gpu` ⇒ `--mode dmabuf --layout nhcw` 강제 | `01_src/pipeline/main.cpp:487-492` |
| `--nms gpu\|cpu\|verify`(기본 gpu)는 `--postproc gpu` 전용 | `01_src/pipeline/main.cpp:32-34`, `:493-496` |
| 스레드 규율(VkQueue 1개 → Vulkan 은 메인 스레드만) | `01_src/pipeline/infer_stage.hpp:9-17` |
| HailoRT 콜백 람다 / CPU NMS fallback 호출 | `01_src/pipeline/gpu_postproc_stage.cpp:154` / `:203` (억제 분기 `finish()` `:191-228`) |
| GPU NMS (`efficient_nms.comp`, yolo_decode 뒤 같은 커맨드 버퍼) | `01_src/vulkan/yolo_postproc.hpp:10`, `01_src/vulkan/efficient_nms.hpp` |
| 평면 스캔아웃(대안 표시 경로) | `01_src/display/plane_display.hpp:1-42`, `overlay_plane.hpp`, `wayland_plane_scanout.hpp:1-30`, `kms_scanout.hpp:1-57` |
| `DrmFrame` · `PreprocOutput` · `Detection` 계약 | `01_src/core/frame_types.hpp` |

### `README.md` §1 과의 관계 (2026-09-20 정렬 완료)

최상위 `README.md` §1 과 7행 요약이 이 다이어그램과 같은 값으로 갱신됐다. 둘 사이에 남은 차이는 없다.
정렬 전에는 아래가 어긋나 있었다(기록용):

| 항목 | 갱신 전 `README.md` | 정본 |
|---|---|---|
| `yolo_decode.comp` | 0.53 ms | **0.626 ms** (`display_e2e/summary.md` §3.1) |
| `display_convert.comp` | 4.22 ms | **4.153 ms** (4.22 는 NPU 유휴 기준선 M5a 4.226) |
| `sand_letterbox.comp` | 2.10 ms | 2.088 ms |
| `overlay_boxes.comp` | 0.12 ms | 0.128 ms |
| `cpu_nms_greedy_by_class` | 2.4 µs | 2.667 µs |
| 디코드 | "CPU 0.10 ms/f" | decode call median **0.107 ms** (CPU 시간이 아니다) |
| 요약 1줄 | 29.9 FPS · glass 101 ms · CPU 1.84 (두 실행 혼합) | `disp_source` 29.91 / 101.20 / **2.179**, `disp_asap` 34.87 / 130.75 / 1.842 |

## 수치가 바뀌면 손봐야 할 곳

재측정으로 값이 바뀌면 아래 노드의 라벨(`y:NodeLabel` 텍스트)과 `d5` description 을 고친다.
id 로 찾으면 된다 (`grep -n 'id="n_pre"' pipeline_overview.graphml`).

**측정 조건이 바뀌어 재측정하면** 수치와 함께 아래 조건 표기도 고친다(ⓝ 이 아닌 기존 표기는 2026-09-28 이전 조건이다).

- **디코더 스레드 수** — overview `n_dec` 라벨 4행 `[측정 조건: FFmpeg 자동 5스레드]`, detail `n0::n0` 라벨 3행
  (`FFmpeg 프레임 스레드 기본 1 (YVO_DEC_THREADS) · 측정 당시 자동 5`), 두 노드와 detail `n0` 의 `d5`.
  기본값의 정본은 `01_src/pipeline/main.cpp:196`, 환경변수를 읽는 곳은 `:348` 이다.
- **V3D 클럭** — overview `n_pre`·`n_post`·`n_conv`·`n_ovl` 라벨의 `[1150 MHz]`와 `d5`, `n_note_perf` 의 GPU 합 줄, detail `n0::n5`·`n0::n7` 의 `d5`.
  2026-10-16 에 GPU 구간을 현재 기본값으로 다시 재서([`display_e2e/runs/v3d1150_20261016`](../07_results/display_e2e/runs/v3d1150_20261016/summary.md)) 모든 셰이더 노드가 `[1150 MHz]` 가 됐다.
  `[MHz]` 꼬리표는 **측정 당시 V3D 클럭**이다 — V3D 클럭은 GPU 전체에 하나뿐이고(부트로더 펌웨어 기본값) 셰이더마다 다르지 않다.
- **공통 요약** — overview `n_note_perf` 라벨의 측정 조건 3줄과 `d5`, detail `n_src_note` 의 `측정 조건 :` 줄과 출처 3줄.
- **ⓝ 현재 조건 값(2026-10-08)** — 라벨: overview `n_nms` 3행·`n_note_perf` 마지막 줄·`n_note_planes` 6~8행·`n_ovlp` 4행·`n_hvs` 5행,
  detail `n0::n5` 3행·`n0::n8` 6행·`n4` 3행·`n_src_note` 의 `측정 조건 :` 줄과 마지막 줄. `d5` 만 더한 곳: overview `n_conv`·`n_ovl`·`n_ab24`, detail `n0::n7`·`n3`.

**`pipeline_overview.graphml`**

| 노드 id | 고쳐야 할 값 |
|---|---|
| `n_dec` | decode call median |
| `n_pre` | 전처리 `sand_letterbox.comp` median |
| `n_npu` | NPU median / mean, batch·slots |
| `n_post` | 후처리 `yolo_decode.comp` median |
| `n_nms` | GPU NMS(`efficient_nms.comp`) 수치 — 라벨 3행 ⓝ median / p90(b2s4, `[1150 MHz]`), `d5` 에 b8s16·fallback·정확도. description 의 CPU NMS 2.667 µs 는 display_e2e 당시 값 |
| `n_det` | 프레임당 검출 수, 총 검출 수 |
| `n_conv` | 표시 변환 `display_convert.comp` median (`d5` 에 ⓝ vk_source 평균 3.234 ms · TFU 3.72 ms/f) |
| `n_ovl` | 오버레이 `overlay_boxes.comp` median (`d5` 에 ⓝ vk_source 평균 0.111 ms) |
| `n_pres` | present 간격 median |
| `n_note_perf` | FPS · CPU · glass-to-glass · GPU 5패스 합(`[1150 MHz]`, `display_e2e/runs/v3d1150_20261016`) (**요약 수치는 전부 여기**). 마지막 줄은 같은 회차의 표시 off b2s4 (`gpu_nms_e2e` 기준선은 `d5`) |
| `n_note_cmd` | 최종 실행 커맨드 (`op_point/summary.md` §8.3 과 한 글자도 다르면 안 된다) |
| `n_note_planes` | 대안 표시 경로 요약·상태와 ⓝ 정식 측정 요약(6~8행: CPU · V3D · 지연 · `--slots 6` 필요). b2/s4·b2/s6 전문과 늦은 프레임은 `d5` |
| `n_ovlp` | `overlay_boxes_buf.spv`(overlay_boxes.comp 의 storage-buffer 변형) + 지우기의 V3D 평균 — ⓝ wlp 0.336 / kms 0.330 ms(b2/s4). Vulkan 합성의 `overlay_boxes.comp` 값과 이름만 같다 |
| `n_ab24` | 박스 평면 링 크기(OverlayPlane `ring`=3)·포맷 |
| `n_hvs` | HVS 평면 번호(영상 83 / 박스 107 은 debugfs 실측, 코드가 고정하지 않음)·표시 지연 — 라벨 5행 ⓝ 디코드 → 화면 평균(b2/s6), `d5` 에 p90·늦은 프레임 |

**`pipeline_detail.graphml`**

| 노드 id | 고쳐야 할 값 |
|---|---|
| `n0::n0` | 디코더 프레임 스레드 수 (라벨 3행: 기본값 · 측정 당시 값) |
| `n0::n5` | GPU 후처리 / NMS 수치 (라벨 3행 ⓝ GPU NMS median / p90 b2s4 `[1150 MHz]`. `d5` 의 `gpu post ms`·CPU NMS 2.667 µs 는 CPU NMS 당시) |
| `n0::n7` | 표시 변환·오버레이 median, 표시 CPU 증가분 (`d5` 에 ⓝ vk_source 평균 convert 3.234 / overlay 0.111 ms · TFU 3.72 ms/f) |
| `n3` | 슬롯 수, slot timeout / slot wait (`d5` 에 ⓝ 평면 백엔드 slot wait — s4 28~30 ms, s6 0) |
| `n4` | `slots + gpu_depth` 상한과 실측 `held_frames_max`, 라벨 3행 ⓝ 평면 표시 on 은 `--slots 6` 필요 |
| `n9` | 호스트 NMS 경로 CPU ms/f |
| `n2::n0` ~ `n2::n2` | dma-buf 크기·정렬·주소식 |
| `n2::n3` | ④ AB24 박스 dma-buf(평면 백엔드만) — 링 수·포맷·FOREIGN 소유권 전환 |
| `n0::n8` | 대안 표시 경로 `PlaneDisplay`(wayland-planes / kms) — 라벨 6행 ⓝ `s6 필요 · V3D −72% · CPU −20%`(b2/s6), 전문은 `d5` |
| `n_src_note` | 출처 목록 자체 (문서 절 번호가 바뀌면 여기). 마지막 줄이 ⓝ 출처 |

## 검증

```bash
xmllint --noout 09_diagrams/*.graphml                                  # well-formed
xmllint --noout --schema ~/.cache/graphml-xsd/ygraphml.xsd 09_diagrams/*.graphml
```

두 번째 명령에서 `pipeline_detail.graphml` 은 두 종류의 에러를 낸다.
**둘 다 yEd 3.25.1 산출물과 공개된 1.1 XSD 사이의 버전 차이이고, 이 파일의 구조 오류가 아니다.**
yEd 가 자기 jar 안에 넣어 배포하는 그룹 예제
(`com/yworks/yfx/resource/graph/examples/hierarchy/hierarchy1.graphmlz`)를 같은 스키마로 검증하면
**똑같은 두 에러가 똑같이 난다.**

1. `y:State`, attribute `closedHeight` / `closedWidth` is not allowed
   — 그룹 노드당 4건. yEd 가 쓰는 속성인데 공개 1.1 `ygraphics.xsd` 에 없다.
2. `More than one match found for key-sequence [...] of keyref edge_source_ref`
   — 중첩 그래프(그룹)가 있을 때 `graphml-structure.xsd` 의 identity constraint 가
   상위·하위 `graph` 양쪽에서 노드를 세는 탓이다. 그룹이 없는 `pipeline_overview.graphml` 은 **깨끗하게 통과한다.**

스키마 대신 쓰는 구조 점검(참조 무결성, id 규약, `data/@key` 선언, 사각형 겹침)은
`xml.etree.ElementTree` 만으로 확인했다. 스크립트는 저장소에 두지 않았다(일회용).
