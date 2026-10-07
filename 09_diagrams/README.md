# 09_diagrams — 파이프라인 다이어그램

최종 확정 파이프라인을 yEd 로 열고 편집할 수 있는 `.graphml` 2장으로 옮긴 것이다.
`README.md` §1 의 ASCII 도식과 같은 내용이되 분기·재합류가 그래프로 정확히 표현돼 있다.

| 파일 | 내용 | 노드 / 엣지 |
|---|---|---:|
| [`pipeline_overview.graphml`](pipeline_overview.graphml) | 개요 1장. 소스 → 디코드 → GPU 전처리 → NPU → GPU 후처리 → GPU NMS → 오버레이 → present → HDMI 와 **표시 분기·재합류**, 구간별 실측 지연, **대안 표시 경로(HVS 평면 스캔아웃)** | 29 / 19 |
| [`pipeline_detail.graphml`](pipeline_detail.graphml) | 상세 1장. **스레드 경계**(메인 / HailoRT 콜백), **dma-buf 소유권 3그룹**(+ 평면 백엔드의 ④ AB24 박스 dma-buf), 슬롯 수명, 제약·전제·함정·종료 순서, 대안 표시 경로(`PlaneDisplay`), 미연결 경로 | 39 / 22 |

두 장 모두 **좌표를 직접 지정**해 두었으므로 yEd 에서 레이아웃을 다시 돌릴 필요가 없다.
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

SVG/PNG/PDF 내보내기는 yEd GUI 에서 사람이 한다(File ▸ Export). yEd 에는 헤드리스 변환 모드가 없다.

## 색 규약 (두 장 공통)

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
- 앱 디코더 프레임 스레드 기본 1 — `01_src/pipeline/main.cpp:633`, 환경변수 `YVO_DEC_THREADS`(0 = FFmpeg 자동).
  표시 on disp_source 에서 CPU 2.073→1.778 ms/f, glass 평균 99.48→99.52 ms, 앱 e2e p90 +1.9 ms
  ([`dec_threads/summary.md`](../07_results/dec_threads/summary.md)).
- NMS 기본 `--nms gpu` — GPU NMS 수치는 정식 측정 전이다.

다이어그램 수치는 재측정 전이라 바꾸지 않았고, 라벨에 `[960 MHz]`·`[측정 조건: FFmpeg 자동 5스레드]`, `n_note_perf` 에 측정 조건 3줄을 붙였다.
앱 대 GStreamer 연구 경로 비교는 [`gst_pipeline/summary.md`](../07_results/gst_pipeline/summary.md) 에 있다(다이어그램에는 출처로만 넣었다).

| 항목 | 출처 |
|---|---|
| ⓒ GPU 4패스 구간별 median (전처리 2.088 / 후처리 0.626 / 변환 4.153 / 오버레이 0.128 ms, 합 6.995 ms) — V3D 960 MHz | [`07_results/display_e2e/summary.md`](../07_results/display_e2e/summary.md) §3.1 |
| ⓒ glass-to-glass 101.20 ms(median) · p90 118.36 ms | 같은 문서 §3.2 |
| ⓒ FPS · CPU ms/f · 검출 3369 · dropped 0 · held_frames_max | 같은 문서 §3 표 |
| ⓒ decode call 0.107 ms(FFmpeg 자동 5스레드) · NPU median 69.7 / mean 73.9 ms · present 간격 16.955 ms · cpu nms 2.667 µs (CPU NMS 당시 측정 — 지금 기본은 GPU NMS, description 에만 남김) | [`07_results/display_e2e/raw/disp_source_run1/stdout.log`](../07_results/display_e2e/raw/disp_source_run1/stdout.log) |
| 운용점 `b2/s4 --postproc gpu` 와 최종 실행 커맨드 | [`07_results/op_point/summary.md`](../07_results/op_point/summary.md) §8 |
| 스레드·슬롯 규약, 종료 순서 | `01_src/pipeline/main.cpp:1-49` |
| `slots + gpu_depth <= 10` 강제 (Vulkan 합성 경로) | `01_src/pipeline/main.cpp:354` |
| 평면 백엔드 보유 상한 `slots + 1 (+5 latest) <= 10` | `01_src/pipeline/main.cpp:340-353` |
| `--postproc gpu` ⇒ `--mode dmabuf --layout nhcw` 강제 | `01_src/pipeline/main.cpp:499-504` |
| `--nms gpu\|cpu\|verify`(기본 gpu)는 `--postproc gpu` 전용 | `01_src/pipeline/main.cpp:32-34`, `:505-508` |
| 스레드 규율(VkQueue 1개 → Vulkan 은 메인 스레드만) | `01_src/pipeline/infer_stage.hpp:9-17` |
| HailoRT 콜백 람다 / CPU NMS fallback 호출 | `01_src/pipeline/gpu_postproc_stage.cpp:154` / `:202` (억제 분기 `finish()` `:190-227`) |
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

**측정 조건이 바뀌어 재측정하면** 수치와 함께 아래 조건 표기도 고친다(지금 표기는 2026-09-28 이전 조건이다).

- **디코더 스레드 수** — overview `n_dec` 라벨 4행 `[측정 조건: FFmpeg 자동 5스레드]`, detail `n0::n0` 라벨 3행
  (`FFmpeg 프레임 스레드 기본 1 (YVO_DEC_THREADS) · 측정 당시 자동 5`), 두 노드와 detail `n0` 의 `d5`.
  기본값의 정본은 `01_src/pipeline/main.cpp:633` 다.
- **V3D 클럭** — overview `n_pre`·`n_post`·`n_conv`·`n_ovl` 라벨의 `[960 MHz]`와 `d5`, detail `n0::n5`·`n0::n7` 의 `d5`.
- **공통 요약** — overview `n_note_perf` 라벨의 측정 조건 3줄과 `d5`, detail `n_src_note` 의 `측정 조건 :` 줄과 출처 3줄.

**`pipeline_overview.graphml`**

| 노드 id | 고쳐야 할 값 |
|---|---|
| `n_dec` | decode call median |
| `n_pre` | 전처리 `sand_letterbox.comp` median |
| `n_npu` | NPU median / mean, batch·slots |
| `n_post` | 후처리 `yolo_decode.comp` median |
| `n_nms` | GPU NMS(`efficient_nms.comp`) 수치 — **아직 없음**(정식 측정 전). description 의 CPU NMS 2.667 µs 는 display_e2e 당시 값 |
| `n_det` | 프레임당 검출 수, 총 검출 수 |
| `n_conv` | 표시 변환 `display_convert.comp` median |
| `n_ovl` | 오버레이 `overlay_boxes.comp` median |
| `n_pres` | present 간격 median |
| `n_note_perf` | FPS · CPU · glass-to-glass · GPU 4패스 합 (**요약 수치는 전부 여기**) |
| `n_note_cmd` | 최종 실행 커맨드 (`op_point/summary.md` §8.3 과 한 글자도 다르면 안 된다) |
| `n_note_planes` | 대안 표시 경로 요약·상태. 지금은 "구현·실표시 확인, 정식 측정 전"이라 **수치를 넣지 않았다** — 측정되면 여기와 `n_ovlp`·`n_hvs` 에 넣는다 |
| `n_ovlp` | `overlay_boxes_buf.spv`(overlay_boxes.comp 의 storage-buffer 변형) median — 미측정 |
| `n_ab24` | 박스 평면 링 크기(OverlayPlane `ring`=3)·포맷 |
| `n_hvs` | HVS 평면 번호(영상 83 / 박스 107 은 debugfs 실측, 코드가 고정하지 않음)·표시 지연 — 미측정 |

**`pipeline_detail.graphml`**

| 노드 id | 고쳐야 할 값 |
|---|---|
| `n0::n0` | 디코더 프레임 스레드 수 (라벨 3행: 기본값 · 측정 당시 값) |
| `n0::n5` | GPU 후처리 / NMS 수치 (GPU NMS 는 정식 측정 전, 기존 값은 CPU NMS 당시) |
| `n0::n7` | 표시 변환·오버레이 median, 표시 CPU 증가분 |
| `n3` | 슬롯 수, slot timeout / slot wait |
| `n4` | `slots + gpu_depth` 상한과 실측 `held_frames_max` |
| `n9` | 호스트 NMS 경로 CPU ms/f |
| `n2::n0` ~ `n2::n2` | dma-buf 크기·정렬·주소식 |
| `n2::n3` | ④ AB24 박스 dma-buf(평면 백엔드만) — 링 수·포맷·FOREIGN 소유권 전환 |
| `n0::n8` | 대안 표시 경로 `PlaneDisplay`(wayland-planes / kms) — 수치 없음(정식 측정 전) |
| `n_src_note` | 출처 목록 자체 (문서 절 번호가 바뀌면 여기) |

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
