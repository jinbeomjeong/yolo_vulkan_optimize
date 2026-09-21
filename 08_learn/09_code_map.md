# 09. 코드 지도 — 무엇을 어떤 순서로 읽을 것인가

> **선수 지식**: [01_big_picture.md](01_big_picture.md)
> (개념을 다 모르는 채로 읽어도 된다. "어디에 무엇이 있는지"만 잡는 챕터다.)
> **다 읽으면 읽을 수 있는 것**: [`CLAUDE.md`](../CLAUDE.md) 폴더 규약,
> [`README.md`](../README.md) §2·§8, 그리고 `01_src/` 아래 아무 파일이나 열었을 때
> "이게 어디쯤 있는 코드인지" 감이 잡힌다.

---

## 9.1 폴더 규약

> **배치 규칙의 정본은 [`STRUCTURE.md`](../STRUCTURE.md)다.** 이 장은 "어디에 무엇이 있는지" 읽는 법을 설명하는 교재이고,
> 새 파일을 어디에 둘지·무엇이 허용되는지는 STRUCTURE.md 와 검사기 `06_tools/python/check_structure.py` 가 정한다.

최상위 폴더 이름 앞의 번호는 **네 대역**으로 묶인다.
`build/`, `build-asan/`, `cmake/`, `.claude/` 처럼 번호가 없는 것도 있다.

| 대역 | 폴더 | 무엇이 있나 |
|---|---|---|
| **코드** | [`01_src/`](../01_src/) | C++ 소스 전부. 하위 폴더는 §9.2 |
| | [`02_shaders/`](../02_shaders/) | GLSL 컴퓨트 셰이더 — 본 파이프라인 5개 + `diag/` 진단 전용 2개 (`.comp`). `overlay_boxes.comp` 는 버퍼 변형까지 SPIR-V 2개로 빌드 |
| **자산** | `03_models/` | `yolov11m.hef` — 컴파일된 YOLO 모델 (33 MB, git에 없음) |
| | `04_data/` | 입력 영상(지금은 `video_h265.mp4` 하나)과 테스트용 원시 텐서 픽스처 (git에 없음) |
| **검증·측정** | [`05_tests/`](../05_tests/) | GoogleTest 실행 파일 6개(`test_postproc_live` 를 따로 세면 7), ctest 항목 78개 |
| | [`06_tools/`](../06_tools/) | 빌드 밖 도구 — `bench/`(측정 셸 7개), `python/`(5개), `probes/`(일회성 소스, `hvs_probe/` 포함) |
| | [`07_results/`](../07_results/) | 실험별 측정 원시 출력과 `summary.md` |
| **문서** | [`08_learn/`](00_README.md) | 지금 읽고 있는 이 자습 교재 12장 |
| | [`09_diagrams/`](../09_diagrams/) | yEd로 여는 파이프라인 다이어그램 2장 |
| | [`10_paper/`](../10_paper/) | 논문 초고 `paper_ko.md` |
| **로그** | `11_logs/` | 07_results에 기록하지 않는 디버깅 실행의 런타임 로그 (git에 없음) |

> 2026-09-21에 폴더를 재배치했다. 이전에는 `08_python`·`09_scripts`가 나뉘어 있었고
> (파이썬 파일 2개가 서로 다른 폴더에 흩어져 있었다), `06_bench/`가 `.gitkeep` 하나뿐인
> 빈 폴더인데도 CMake의 `find_package(benchmark REQUIRED)`가 configure 하드 의존이었다.
> 문서 3폴더(`11_paper`·`12_docs`·`13_learn`)는 번호가 시간 순으로 붙어 있었다.
> 2026-09-22에는 대역 안쪽을 재배치했다(`common/` → `core/`·`ref/`·`toolkit/`, 도구를 `01_src/tools/`로, `07_results` 회차를 `runs/`로).
> 옛 경로 → 새 경로 대조표는 [`STRUCTURE.md`](../STRUCTURE.md) §6.

모델·영상·로그는 `.gitignore`에 들어 있어서 저장소에 없다.

---

## 9.2 `01_src/` 하위 폴더의 역할

```
01_src/
├── core/       헤더만. 공용 계약 타입·OS 글루 — 다른 모든 폴더가 여기에 의존 (include 루트는 여기서만 지정)
├── ref/        헤더만. CPU 기준 구현 — 수식의 단일 진실 원천 (셰이더는 이것의 번역본)
├── toolkit/    헤더만. 도구·앱 전용 헬퍼(CLI·이미지 저장·통계) — 라이브러리는 include 하지 않는다
├── testkit/    헤더만. 테스트·프로브가 공유하는 검증 하네스(픽스처 파서, 오프스크린 타깃, 비트 일치 판정)
├── decode/     HW HEVC 디코더 (FFmpeg DRM_PRIME → DrmFrame)
├── hailo/      NPU (HailoRT)                    ┐ 이 둘은 서로
├── vulkan/     GPU (Vulkan 컴퓨트)              ┘ 링크하지 않는다
├── display/    화면 표시 (Vulkan 합성·스왑체인 + HVS 평면 스캔아웃: Wayland linux-dmabuf / KMS atomic)   — vulkan과 별도 라이브러리
├── isp/        ISP 전처리 — 구현됐지만 미사용
├── pipeline/   이 전부를 묶는 실행 파일 main.cpp
└── tools/      빌드되는 진단·측정 도구 10개 ({hailo,vulkan,display,isp,decode}/ — 측정 대상 도메인 기준)
```

### 왜 일부러 갈라 뒀나

`hailo/`와 `vulkan/`은 **서로를 링크하지 않는다.** 두 폴더는 오직 **fd(파일 디스크립터)와 크기**만
주고받는다. 이렇게 하면 한쪽 라이브러리의 문제가 다른 쪽으로 번지지 않는다.

`display/`도 마찬가지 이유로 `vulkan/`과 별도 라이브러리(`yvo_display`)다.
GPU 전처리를 테스트하는 `test_preproc`을 빌드할 때 Wayland 라이브러리가 딸려 오지 않게 하기 위해서다.

대가는 **같은 구조체를 두 번 정의해야 하는 것**이다. 예를 들어 검출 결과 버퍼의 배치는
`01_src/vulkan/yolo_postproc.hpp`와 `01_src/display/overlay_renderer.hpp` 양쪽에 따로 정의돼 있고,
`static_assert`로 **바이트 오프셋까지 같은지 컴파일 시점에 강제**한다.
한쪽만 고치면 빌드가 실패한다.

> 이런 설계 습관은 [08_method.md](08_method.md)에서 따로 다룬다.

---

## 9.3 읽는 순서

### 1단계 — 계약을 먼저 본다 (30분)

**[`01_src/core/frame_types.hpp`](../01_src/core/frame_types.hpp)**

파이프라인 단계 사이를 오가는 **자료형 네 개**가 여기 다 있다.

| 타입 | 무엇인가 |
|---|---|
| `DrmFrame` | 디코더가 내놓은 프레임. Y/UV dma-buf fd, 크기, stride |
| `LetterboxParams` | 1920×1080을 640×640에 어떻게 넣을지(배율, 패딩 위치) |
| `PreprocOutput` | GPU 전처리 결과. dma-buf fd와 레이아웃 |
| `Detection` | 최종 검출 하나. 원본 좌표 기준 박스 + 클래스 + 점수 |

이 네 개를 이해하면 파이프라인의 데이터 흐름이 거의 다 보인다.

### 2단계 — main.cpp 머리 주석을 읽는다 (1시간)

**[`01_src/pipeline/main.cpp`](../01_src/pipeline/main.cpp) 1~49행**

코드가 아니라 **주석 49줄**이다. 여기에 이 프로젝트의 운영 규칙이 전부 들어 있다:

- 스레드 구조 (메인 스레드 / HailoRT 콜백 스레드)
- **슬롯 규약** — 슬롯을 언제 반납하는가. 반납 조건이 **둘**이라는 것
- 모드(`ptr` vs `dmabuf`), 표시(`off`/`on`), 후처리(`host`/`gpu`)
- 종료 순서 — 왜 이 순서여야 하는지
- 시그널 처리

> ⚠ **여기 적힌 함정이 실제 함정이다.** 예를 들어 "디코더는 약 13프레임 이상 붙잡으면
> 에러 없이 조용히 멈춘다"는 문장 때문에 `slots + gpu_depth <= 10` 검사가
> [`main.cpp:350`](../01_src/pipeline/main.cpp) 에 들어 있다(HVS 평면 백엔드용 보유 검사는 바로 위 :336-349).

### 3단계 — 수식의 원문을 본다 (2시간)

이 프로젝트에는 **"수식의 단일 진실 원천(single source of truth)"** 규칙이 있다.
GPU 셰이더는 전부 CPU 기준 구현의 **번역본**이고, 원문은 `01_src/ref/` 헤더의 상단 주석이다.

| 원문 (읽어야 할 것) | 번역본 (GPU) |
|---|---|
| [`ref/cpu_letterbox.hpp`](../01_src/ref/cpu_letterbox.hpp) 주석 1)~5) | [`sand_letterbox.comp`](../02_shaders/sand_letterbox.comp) |
| [`ref/cpu_yolov8_decode.hpp`](../01_src/ref/cpu_yolov8_decode.hpp) 주석 1)~7) | [`yolo_decode.comp`](../02_shaders/yolo_decode.comp) |
| 같은 헤더의 `cpu_nms_greedy_by_class()`·`iou_xyxy()` | [`efficient_nms.comp`](../02_shaders/efficient_nms.comp) (구조는 [11장](11_tensorrt_efficient_nms.md)) |
| [`ref/sand.hpp`](../01_src/ref/sand.hpp) | 위 두 셰이더의 주소 계산 |
| [`ref/overlay_reference.hpp`](../01_src/ref/overlay_reference.hpp) 규약 R0~R8 | [`overlay_boxes.comp`](../02_shaders/overlay_boxes.comp) |

**셰이더를 먼저 읽지 마라.** 셰이더는 최적화가 잔뜩 들어가 있어서 수식이 안 보인다.
헤더의 주석을 읽고 나서 셰이더를 보면 "아, 이 수식을 이렇게 비튼 거구나"가 보인다.

### 4단계 — 단계별 구현 (필요한 것만)

| 관심사 | 읽을 파일 |
|---|---|
| 디코드 | [`decode/ffmpeg_drm_decoder.cpp`](../01_src/decode/ffmpeg_drm_decoder.cpp) |
| GPU 전처리 | [`vulkan/sand_letterbox.cpp`](../01_src/vulkan/sand_letterbox.cpp) |
| dma-buf 주고받기 | [`vulkan/dmabuf_import.cpp`](../01_src/vulkan/dmabuf_import.cpp), [`dmabuf_export.cpp`](../01_src/vulkan/dmabuf_export.cpp) |
| NPU (기준 경로) | [`hailo/hailo_runner.cpp`](../01_src/hailo/hailo_runner.cpp) |
| NPU (실제 쓰는 경로) | [`hailo/hailo_raw_runner.cpp`](../01_src/hailo/hailo_raw_runner.cpp) |
| GPU 후처리 | [`vulkan/yolo_postproc.cpp`](../01_src/vulkan/yolo_postproc.cpp), [`vulkan/efficient_nms.cpp`](../01_src/vulkan/efficient_nms.cpp), [`pipeline/gpu_postproc_stage.cpp`](../01_src/pipeline/gpu_postproc_stage.cpp) |
| 화면 표시 (Vulkan 합성) | [`display/display_pipeline.cpp`](../01_src/display/display_pipeline.cpp), [`vk_swapchain.cpp`](../01_src/display/vk_swapchain.cpp) |
| 화면 표시 (HVS 평면) | [`display/plane_display.cpp`](../01_src/display/plane_display.cpp), [`wayland_plane_scanout.cpp`](../01_src/display/wayland_plane_scanout.cpp), [`kms_scanout.cpp`](../01_src/display/kms_scanout.cpp), [`overlay_plane.cpp`](../01_src/display/overlay_plane.cpp), [`drm_state.cpp`](../01_src/display/drm_state.cpp) — 둘 다 [`display_sink.hpp`](../01_src/display/display_sink.hpp) 인터페이스 뒤에 있다 |

---

## 9.4 셰이더 — 본 5개 + 진단 2개

전부 `02_shaders/` 에 있고 확장자는 `.comp`(컴퓨트 셰이더)다. 진단 전용 2개는 `02_shaders/diag/` 에 따로 두고
(`yvo_shaders_diag`), 도구만 링크한다. 본 5개는 `yvo_shaders` 다.
`yvo_shaders` 의 SPIR-V 산출물은 6개다 — `overlay_boxes.comp` 를 `YVO_OVERLAY_BUFFER` 로 한 번 더 컴파일한
`overlay_boxes_buf.spv`(HVS 박스 평면용, storage buffer 에 그린다)가 더 있다. 문법은 [05장](05_vulkan_compute.md) 빌드 파이프라인.

| 셰이더 | 하는 일 | 파이프라인에서 |
|---|---|---|
| [`sand_letterbox.comp`](../02_shaders/sand_letterbox.comp) | SAND128 NV12 → letterbox RGB → NHWC 또는 NHCW | ② 전처리 |
| [`yolo_decode.comp`](../02_shaders/yolo_decode.comp) | 원시 텐서 → 후보 박스 압축 | ④ 후처리 |
| [`efficient_nms.comp`](../02_shaders/efficient_nms.comp) | 후보 정렬 + 클래스별 greedy 억제 (TRT EfficientNMS 구조) | ④ 후처리 |
| [`display_convert.comp`](../02_shaders/display_convert.comp) | SAND128 NV12 → 화면용 RGBA | ⑥ 표시 변환 |
| [`overlay_boxes.comp`](../02_shaders/overlay_boxes.comp) | 박스·라벨·HUD 그리기 (변형 `overlay_boxes_buf`: AB24 dma-buf 버퍼에) | ⑦ 오버레이 |
| [`diag/display_probe.comp`](../02_shaders/diag/display_probe.comp) | 고정 테스트 패턴 | 진단 전용 |
| [`diag/heap_sum.comp`](../02_shaders/diag/heap_sum.comp) | 바이트 합 리덕션 | 진단 전용 |

앞의 5개가 실제 파이프라인에서 돌고, 뒤의 2개는 **"이 경로가 정말 동작하는가"를 가르는 관문**용이다.

### 셰이더 읽을 때 알아 둘 것

- 파일 맨 위 주석에 **기준 구현이 어느 헤더인지**와 **왜 이렇게 최적화했는지**가 측정 수치와 함께 적혀 있다.
  예: `sand_letterbox.comp`의 "열 LUT로 뺐다 (16x4 워크그룹 기준 5.07 → 4.64 ms)"
- 워크그룹 크기는 셰이더에 박혀 있지 않고 **호스트가 정한다**(specialization constant).
  같은 셰이더로 여러 크기를 재 보고 고르기 위해서다
- 빌드할 때 `glslc -Werror` → `spirv-val` 검증 → SPIR-V를 C++ 헤더로 변환까지 자동으로 돈다.
  그래서 실행 파일이 `.spv` 파일을 따로 읽지 않는다 (규칙은 [`cmake/YvoShaders.cmake`](../cmake/YvoShaders.cmake)의 `yvo_add_shaders`)

---

## 9.5 빌드

```bash
cmake -S . -B build -G Ninja && cmake --build build
```

**빌드 경고는 0줄이어야 한다.** 이건 이 프로젝트의 규칙이다
([`.claude/skills/verify`](../.claude/skills/verify/SKILL.md)).

### 만들어지는 라이브러리

| 이름 | 내용 | 특징 |
|---|---|---|
| `yvo_core` | 헤더만 (INTERFACE) | 공용 타입 + OS 글루. include 루트 `01_src/` 를 지정하는 유일한 타깃 |
| `yvo_ref` | 헤더만 (INTERFACE) | CPU 기준 구현 |
| `yvo_toolkit` | 헤더만 (INTERFACE) | 도구·앱 전용 헬퍼. 라이브러리는 링크하지 않는다 |
| `yvo_testkit_postproc` / `yvo_testkit_display` | 헤더만 (INTERFACE) | 테스트·프로브 공용 검증 하네스 |
| `yvo_warnings` | 경고 플래그만 | `-Wall -Wextra -Wpedantic` |
| `yvo_shaders` | 컴파일된 셰이더 헤더 (본 5개 + 버퍼 변형 1 = 6개) | 링크하면 include 경로가 붙는다 |
| `yvo_shaders_diag` | 진단 셰이더 헤더 (`diag/` 2개) | 도구만 링크한다 |
| `yvo_hailo` | NPU | **`yvo_vulkan`에 의존하지 않는다** |
| `yvo_vulkan` | GPU | |
| `yvo_display` | 화면 표시 (Vulkan 합성 + HVS 평면: wayland-planes / kms, libdrm 링크) | Wayland 없으면 조용히 빠진다(kms 백엔드도 함께) |
| `yvo_isp` | ISP | `libpisp` 없으면 통째로 빠진다 |
| `yvo_decoder` | 디코드 | libavcodec + libdrm |
| `yvo_stages` | 후처리 글루 | OpenCV에 의존하지 않는다(테스트가 링크할 수 있게) |
| `yvo_assets` | 컴파일 정의만 (INTERFACE) | `YVO_REPO_ROOT`·`YVO_DEFAULT_HEF`·`YVO_DEFAULT_VIDEO`·`YVO_FIXTURE_DIR` 절대 경로. 도구·앱·테스트만 링크 → 어느 디렉터리에서 실행해도 기본 경로가 맞다 |

### 실행 파일

**본체는 하나다**: `build/bin/yvo_pipeline`

나머지는 전부 **진단 도구**다(소스는 [`01_src/tools/`](../01_src/tools/), 실행파일은 모두 `build/bin/`).
각 도구는 "이 단계만 따로 돌려 본다"는 목적이 있다.

| 도구 | 무엇을 따로 보나 | NPU 필요? |
|---|---|---|
| `yvo_drm_dump` | 디코더가 내놓는 버퍼의 실제 모양 | 아니오 |
| `yvo_preproc_bench` | GPU 전처리만, 워크그룹 크기 스윕 | 아니오 |
| `yvo_postproc_probe` | GPU 후처리. `--probe fixture`는 저장된 텐서로 검증 | `fixture`는 불필요 |
| `yvo_hailo_selftest` | NPU 러너만 | 예 |
| `yvo_hailo_raw_probe` | 원시 출력 경로, dma-buf 동시 사용 관문 | 예 |
| `yvo_display_probe` | 화면 경로만 | 아니오 |
| `yvo_display_pipeline_probe` | 표시 상태기계만 | 아니오 |
| `yvo_isp_*` | ISP 경로 (보류 상태) | 아니오 |

> **이 도구들이 프로젝트의 방법론이다.** 새 기능을 바로 본체에 넣지 않고,
> 먼저 프로브로 "이게 되는가"를 가른 다음 통과한 것만 승격한다.
> 자세한 것은 [08_method.md](08_method.md).

---

## 9.6 테스트

```bash
ctest --test-dir build
```

| 테스트 | NPU | 합격 기준 |
|---|---|---|
| `test_preproc` | 불필요 | GPU 전처리가 CPU 기준과 **NHCW에서 바이트 완전 일치** |
| `PostprocFixture.*` (5개) | **불필요** | 후보 수 동일, 점수 **비트 일치**, 좌표 오차 < 0.25 px, 10회 결정성, GPU NMS 결과 **비트 일치**(score 0.2·0.01) |
| `EfficientNms.*` (7개) | **불필요** | 합성 DetBuf: 동점, 클래스 정원, 클래스 간 겹침, 용량 초과·IoU 보호대 신호, 0/1개, 결정성 |
| `test_postproc_live` | 필요 | host 경로 vs GPU 경로 검출 매칭 ≥ 0.95 |
| `test_detections` | 필요 | 네 경로(CPU/GPU-ptr/GPU-dmabuf/NHCW) 검출 일치 |
| `test_overlay` | 불필요 | 오버레이 GPU vs CPU 기준 **오차 0**. `OverlayPlane.*`·`Clear/OverlayPlaneMode.*`(박스 평면 dma-buf 비트 일치·잔상 0), `DrmState.*`(debugfs 파서), `KmsScanout.*`(card1 탐색·AddFB2 — DRM master 를 가져가지 않고, 아무도 master 가 아니면 skip) |
| `test_display_convert` | 불필요 | 표시 변환(1:1)이 CPU 기준과 **오차 0**. BT.709 계수의 세 번째 사본을 지킨다 |
| `test_isp_preproc` | 불필요(ISP 필요) | 패딩은 오차 0, 내용은 MAE ≤ 3.0 |

주목할 점 둘:

1. **후처리 테스트를 일부러 두 개로 쪼갰다.** `PostprocFixture.*`는 저장해 둔 텐서 파일
   (`04_data/raw_tensors/`)만 쓰므로 **NPU 없이 수 초 만에** 돈다.
   다른 사람이 NPU를 쓰고 있어도 셰이더 회귀는 잡힌다.
2. **합격 기준이 "비슷하면 통과"가 아니라 "완전히 같아야 통과"** 인 곳이 여럿이다.
   GPU 최적화가 결과를 바꾸지 않았음을 증명하려면 이 수준이 필요하다.

검출 결과를 직접 비교하는 스크립트도 있다:
[`06_tools/python/compare_detections.py`](../06_tools/python/compare_detections.py).
C++ 테스트의 판정 규칙([`01_src/ref/detection_match.hpp`](../01_src/ref/detection_match.hpp))과
**일부러 같은 알고리즘**으로 구현돼 있다.

---

## 9.7 측정 스크립트

[`06_tools/bench/`](../06_tools/bench/) 의 셸 스크립트 7개가 `07_results/` 의 결과를 만들어 낸다.

| 스크립트 | 무엇을 재나 |
|---|---|
| `op_point_bench.sh` | **운용점 스윕** (batch·slots 조합) — 최종 결론의 근거 |
| `display_bench.sh` | 화면 표시 포함 종단간 |
| `pipeline_bench.sh` | 회귀 기준선 (고정 9설정) |
| `decode_bench.sh` | 디코드 경로 비교 |
| `backpressure_bench.sh` | 백프레셔 결함 전후 비교 |
| `isp_bench.sh` / `isp_pipeline_bench.sh` | ISP 경로 |

공통 규약(설정당 3회 중앙값, 냉각 게이트, 환경 스냅샷)은 [08_method.md](08_method.md)에서 다룬다.

> `pipeline_bench.sh`의 설정 형식은 **바꾸면 안 된다.** 다른 스크립트들이
> 이걸 대조군 기준으로 쓴다.

일곱 스크립트 모두 시작할 때 `build/.bench_running` 에 자기 PID를 쓰고 끝날 때 지운다.
[`.claude/hooks/npu_guard.sh`](../.claude/hooks/npu_guard.sh) 가 이 파일을 보고 **측정 중에는
빌드·테스트 명령을 막는다** — 측정 중 다른 작업이 돌면 CPU 시간이 에러 없이 오염되기 때문이다.

### 그 밖의 도구

| 폴더 | 내용 |
|---|---|
| [`06_tools/python/`](../06_tools/python/) | `compare_detections.py`(검출 비교), `gen_glyph_font_8x16.py`(폰트 헤더 생성 — 빌드에 관여하지 않는 재현 기록), `check_links.py`(문서 링크·`file:line`·`--stale` 옛 경로), `check_structure.py`(폴더 배치 규칙), `relocate.py`(매핑표 기반 이동) |
| [`06_tools/probes/`](../06_tools/probes/) | 일회성 프로브 소스. 빌드 대상이 아니며, 어떤 실험에서 쓴 것인지는 해당 `summary.md`에 적혀 있다. `hvs_probe/` 는 외부 저장소 헤더 기준이라 이 저장소에서 그대로는 빌드되지 않는다 |

---

## 9.8 실행 — 기본값이 권장 경로가 **아니다**

```bash
# 이렇게 돌리면 가장 느린 경로로 돈다
build/bin/yvo_pipeline --video ... --hef ...
```

인자 없이 돌리면 `--mode ptr --layout nhwc --postproc host --display off` 로 간다.
이건 **의도된 것**이다 — 기본값을 바꾸면 과거 측정과의 비교가 깨지기 때문에 일부러 두었다.

실제 권장 실행 커맨드는 [`07_results/README.md`](../07_results/README.md) 맨 위와
[`README.md`](../README.md) §3.1 에 있다. 핵심 세 옵션:

- `--mode dmabuf` — zero-copy 필수
- `--layout nhcw` — CPU 변환 제거 필수
- `--postproc gpu` — GPU 후처리. 위 둘이 켜져 있어야만 쓸 수 있다
- (표시할 거면) `--display on --display-fullscreen` — **전체화면이 아니면 조용히 2.6배 느려진다**

---

## 9.9 자가 점검

**Q1. GPU 셰이더의 계산이 맞는지 어떻게 확인하는가?**

`01_src/ref/`의 CPU 기준 구현과 비교한다. 셰이더는 그 수식의 번역본이고,
`ctest`가 두 결과를 대조한다. 전처리 NHCW와 오버레이는 **바이트 완전 일치**가 합격 기준이다.

**Q2. `hailo/`와 `vulkan/`이 서로 링크하지 않으면 데이터는 어떻게 주고받는가?**

**fd(파일 디스크립터)와 크기 숫자**만 주고받는다. 실제 메모리는 dma-buf로 공유한다.
그래서 두 라이브러리가 서로의 타입을 전혀 몰라도 된다. (→ [06_zerocopy_dmabuf.md](06_zerocopy_dmabuf.md))

**Q3. 셰이더를 고쳤는데 빌드가 안 된다. 왜?**

빌드 과정에 `glslc -Werror`(경고도 에러로 취급)와 `spirv-val`(SPIR-V 검증)이 들어 있다.
둘 중 하나라도 걸리면 빌드가 실패한다. 규칙은 `cmake/YvoShaders.cmake`의 `yvo_add_shaders`에 있다.

**Q4. NPU가 다른 사람에게 잡혀 있을 때도 돌릴 수 있는 테스트가 있는가?**

있다. `PostprocFixture.*`(5개), `EfficientNms.*`(7개), `test_preproc`, `test_overlay` 는 NPU가 필요 없다(`test_overlay` 의 `KmsScanout.*` 는 card1 과 HW 디코더를 쓴다).
후처리 테스트를 일부러 두 개로 쪼갠 이유가 이것이다.

**Q5. 처음 코드를 읽는다면 어디부터 봐야 하는가?**

`01_src/core/frame_types.hpp`(자료형 4개) → `01_src/pipeline/main.cpp` 1~49행(운영 규칙 주석)
→ `01_src/ref/cpu_yolov8_decode.hpp` 상단 주석(수식 원문) 순서다.
**셰이더를 먼저 읽으면 안 된다** — 최적화 때문에 수식이 안 보인다.

---

**다음** → 개념을 채우려면 [02_hardware.md](02_hardware.md) 부터 순서대로,
"왜 이렇게 만들었나"가 궁금하면 [10_why_this_design.md](10_why_this_design.md).
