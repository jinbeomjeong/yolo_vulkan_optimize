# 09_diagrams — 파이프라인 다이어그램

최종 확정 파이프라인을 yEd 로 열고 편집할 수 있는 `.graphml` 2장으로 옮긴 것이다.
`README.md` §1 의 ASCII 도식과 같은 내용이되 분기·재합류가 그래프로 정확히 표현돼 있다.

| 파일 | 내용 | 노드 / 엣지 |
|---|---|---:|
| [`pipeline_overview.graphml`](pipeline_overview.graphml) | 개요 1장. 소스 → 디코드 → GPU 전처리 → NPU → GPU 후처리 → NMS → 오버레이 → present → HDMI 와 **표시 분기·재합류**, 구간별 실측 지연 | 24 / 14 |
| [`pipeline_detail.graphml`](pipeline_detail.graphml) | 상세 1장. **스레드 경계**(메인 / HailoRT 콜백), **dma-buf 소유권 3그룹**, 슬롯 수명, 제약·전제·함정·종료 순서, 미연결 경로 | 36 / 18 |

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
| `#4DB6E2` | 표시 · present — Wayland 스왑체인 |
| `#FF0000` | 소스 · 싱크 |
| `#FFFFCC` | 데이터 — dma-buf 버퍼 (복사 0) |
| 회색 + 점선 테두리 | 미연결 경로 (코드에는 있으나 파이프라인에 연결돼 있지 않음) |

각 파일 안에도 범례 노드(`n_leg_t`, `n_leg0`~)가 들어 있다.

## 수치의 출처

**지연 수치의 정본은 `README.md` §1 의 ASCII 가 아니라 아래 측정 문서다.**
둘이 다르면 측정 문서를 따랐고, 그 사실을 해당 노드의 `description` 에 적어 두었다.

| 항목 | 출처 |
|---|---|
| GPU 4패스 구간별 median (전처리 2.088 / 후처리 0.626 / 변환 4.153 / 오버레이 0.128 ms, 합 6.995 ms) | [`07_results/display_e2e/summary.md`](../07_results/display_e2e/summary.md) §3.1 |
| glass-to-glass 101.20 ms(median) · p90 118.36 ms | 같은 문서 §3.2 |
| FPS · CPU ms/f · 검출 3369 · dropped 0 · held_frames_max | 같은 문서 §3 표 |
| decode call 0.107 ms · NPU median 69.7 / mean 73.9 ms · present 간격 16.955 ms · cpu nms 2.667 µs | [`07_results/display_e2e/raw/disp_source_run1/stdout.log`](../07_results/display_e2e/raw/disp_source_run1/stdout.log) |
| 운용점 `b2/s4 --postproc gpu` 와 최종 실행 커맨드 | [`07_results/op_point/summary.md`](../07_results/op_point/summary.md) §8 |
| 스레드·슬롯 규약, 종료 순서 | `01_src/pipeline/main.cpp:1-34` |
| `slots + gpu_depth <= 10` 강제 | `01_src/pipeline/main.cpp:276` |
| `--postproc gpu` ⇒ `--mode dmabuf --layout nhcw` 강제 | `01_src/pipeline/main.cpp:413-418` |
| 스레드 규율(VkQueue 1개 → Vulkan 은 메인 스레드만) | `01_src/pipeline/infer_stage.hpp:9-17` |
| HailoRT 콜백 람다 / NMS 호출 | `01_src/pipeline/gpu_postproc_stage.cpp:136` / `:182` |
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

**`pipeline_overview.graphml`**

| 노드 id | 고쳐야 할 값 |
|---|---|
| `n_dec` | decode call median |
| `n_pre` | 전처리 `sand_letterbox.comp` median |
| `n_npu` | NPU median / mean, batch·slots |
| `n_post` | 후처리 `yolo_decode.comp` median |
| `n_nms` | CPU NMS µs, 후보 수 |
| `n_det` | 프레임당 검출 수, 총 검출 수 |
| `n_conv` | 표시 변환 `display_convert.comp` median |
| `n_ovl` | 오버레이 `overlay_boxes.comp` median |
| `n_pres` | present 간격 median |
| `n_note_perf` | FPS · CPU · glass-to-glass · GPU 4패스 합 (**요약 수치는 전부 여기**) |
| `n_note_cmd` | 최종 실행 커맨드 (`op_point/summary.md` §8.3 과 한 글자도 다르면 안 된다) |

**`pipeline_detail.graphml`**

| 노드 id | 고쳐야 할 값 |
|---|---|
| `n0::n5` | GPU 후처리 / CPU NMS 수치 |
| `n0::n7` | 표시 변환·오버레이 median, 표시 CPU 증가분 |
| `n3` | 슬롯 수, slot timeout / slot wait |
| `n4` | `slots + gpu_depth` 상한과 실측 `held_frames_max` |
| `n9` | 호스트 NMS 경로 CPU ms/f |
| `n2::n0` ~ `n2::n2` | dma-buf 크기·정렬·주소식 |
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
