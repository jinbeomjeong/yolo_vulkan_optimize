# 저장소 구조 규칙 (정본)

이 문서가 **폴더 배치 규칙의 정본**이다. 다른 문서(`CLAUDE.md`, `08_learn/09_code_map.md`,
`07_results/README.md`, `.claude/skills/measure/SKILL.md`)는 요약하고 여기로 링크만 한다.

- 사람용 정본: 이 문서
- 기계용 정본: [`06_tools/structure_rules.toml`](06_tools/structure_rules.toml)
- 검사기: `python3 06_tools/python/check_structure.py` — 위반이 있으면 exit 1
- 새 파일 차단: Claude Code Write 훅 `.claude/hooks/placement_guard.sh`
  (새 파일만 본다. Bash로 만든 파일은 작업이 끝날 때 검사기로 확인한다)

두 정본이 어긋나면 검사기가 `D1`로 잡는다. 규칙을 바꿀 때는 **이 문서와 TOML을 같은 커밋에서** 고친다.

---

## 1. 최상위

| 항목 | git | 내용 |
|---|---|---|
| `01_src` | 추적 | 빌드되는 C++ 소스 전부 (라이브러리·앱·도구) |
| `02_shaders` | 추적 | GLSL 컴퓨트 셰이더. `.comp`는 여기에만 둔다 |
| `03_models` | 무시 | HEF 모델 |
| `04_data` | 무시 | 입력 영상(`videos/`), NPU 원시 출력 픽스처(`raw_tensors/`) |
| `05_tests` | 추적 | GoogleTest 소스 (`test_*.cpp`) |
| `06_tools` | 추적 | **빌드에 들어가지 않는 것**: `bench/`(측정 셸), `python/`, `probes/`, `structure_rules.toml` |
| `07_results` | 추적 | 측정 결과. **수치의 정본** |
| `08_learn` | 추적 | 자습 교재 |
| `09_diagrams` | 추적 | yEd `.graphml` |
| `10_paper` | 추적 | 논문 초고. 그림은 `figures/`(§2.7) |
| `11_logs` | 무시 | 07_results에 기록하지 않는 실행의 런타임 로그 |
| `cmake` | 추적 | CMake 모듈 |
| `.claude` | 추적 | 에이전트·스킬·훅 설정 |
| `build` | 무시 | 빌드 산출물(`build*/` 전부) |
| `build-asan` | 무시 | 새니타이저 빌드 |
| `.git` | 무시 | |
| `CLAUDE.md` | 추적 | 에이전트 지침 |
| `README.md` | 추적 | 저장소 소개 |
| `STRUCTURE.md` | 추적 | 이 문서 |
| `CMakeLists.txt` | 추적 | 최상위 빌드 |
| `environment.yml` | 추적 | conda 환경 정의 |
| `.gitignore` | 추적 | |
| `.mcp.json` | 추적 | 프로젝트 MCP 서버 |

- 이 표에 없는 최상위 항목은 만들지 않는다. 필요하면 이 표와 TOML을 먼저 고친다.
- 루트에 `hailort.log`·`perf.data`·`core`가 생기면 잔재다. 옮기거나 지운다(§3.5).

## 2. 코드 대역

### 2.1 `01_src/` 하위 폴더

| 폴더 | 타깃 | 담는 것 | 넣지 않는 것 |
|---|---|---|---|
| `core/` | `yvo_core` (INTERFACE) | 단계 간 계약 타입, 라벨, OS 글루(dma-heap·dma-buf), HailoRT 정렬 상수 | Vulkan·HailoRT 의존 코드 |
| `ref/` | `yvo_ref` (INTERFACE) | **CPU 기준 구현**: 셰이더 수식의 원문과 판정 규칙(letterbox, SAND, YOLO 디코드, 검출 매칭, 오버레이, 글리프) | 성능용 코드 |
| `toolkit/` | `yvo_toolkit` (INTERFACE) | 도구 전용 헬퍼(인자 파싱, 이미지 저장, 통계) | 라이브러리가 include하는 것 |
| `testkit/` | `yvo_testkit_*` (INTERFACE) | 테스트와 프로브가 **함께** 쓰는 하네스·판정 로직 | 테스트 하나만 쓰는 헬퍼(그 테스트 파일에 둔다) |
| `decode/` | `yvo_decoder` | FFmpeg + DRM PRIME 디코더 | |
| `hailo/` | `yvo_hailo` | HailoRT 래퍼 | Vulkan 의존 |
| `vulkan/` | `yvo_vulkan` | Vulkan 컨텍스트, dma-buf import/export, 전처리·후처리 호스트 코드 | HailoRT 의존 |
| `display/` | `yvo_display` | 표시: Vulkan 합성(Wayland 창, 스왑체인, 표시 변환, 오버레이)과 HVS 평면 스캔아웃(`PlaneDisplay`, Wayland linux-dmabuf / KMS atomic, 박스 평면, debugfs 판정) | |
| `isp/` | `yvo_isp` | PiSP BE 전처리 (파이프라인 미연결, 보류) | |
| `pipeline/` | `yvo_stages`, `yvo_pipeline` | 단계 조립과 앱 본체 `main.cpp` | |
| `gstelem/` | `yvo_gstelem` (선택: GStreamer ≥1.24 dev가 없으면 빠진다) | 연구용 GStreamer 요소(도구에 정적 등록, .so 없음): SAND DMABuf → `DrmFrame` 어댑터, `yvosandletterbox`, `yvohailosink` | `yvo_pipeline` 본체 경로 코드 |
| `tools/<도메인>/` | `yvo_*` 실행파일 | **빌드되는** 프로브·벤치·자가시험. 파일당 `main()` 1개 | 라이브러리 코드 |

- `01_src/` 바로 아래에는 이 표의 폴더만 둔다. 이름은 `/usr/include`의 폴더명과 겹치면 안 된다(`hailo`·`vulkan`은 허용 예외, 파일명은 겹치지 않게 한다).
- 여러 도메인을 링크하는 도구는 **측정·검증하는 대상** 도메인의 `tools/` 폴더에 둔다.
- `main()`은 `01_src/tools/**`와 `01_src/pipeline/main.cpp`에만 있다.
- 실행파일은 모두 `build/bin/`에 나온다.

### 2.2 include 계층

따옴표 include의 첫 경로 성분으로 판정한다. `../` include는 금지.

| 폴더 | include해도 되는 폴더 |
|---|---|
| `core` | (없음) |
| `ref` | core |
| `toolkit` | core, ref |
| `decode` | core |
| `hailo` | core, ref — **vulkan 금지** |
| `vulkan` | core, ref — **hailo 금지** |
| `display` | core, ref, vulkan |
| `isp` | core, ref |
| `testkit` | core, ref, vulkan, display, testkit |
| `pipeline` | core, ref, hailo, vulkan, display, decode |
| `gstelem` | core, ref, hailo, vulkan |
| `tools` | 전부 |
| `05_tests` | tools를 뺀 전부 |

라이브러리 폴더(`core ref decode hailo vulkan display isp pipeline gstelem`)는 `toolkit`·`testkit`·`tools`를 include하지 않는다.

### 2.3 셰이더

- `02_shaders/*.comp`: 라이브러리가 쓰는 본 셰이더 → `yvo_shaders`
- `02_shaders/diag/*.comp`: 진단 전용 → `yvo_shaders_diag`. `01_src/tools/**`만 링크한다.
- 모든 `.comp`는 자기 폴더의 `CMakeLists.txt`(`02_shaders/`, `02_shaders/diag/`)에 등록돼 있어야 한다. 셰이더 머리 주석이 `ref/`의 원문 헤더를 가리킨다.
- 같은 소스를 매크로만 바꿔 한 번 더 빌드할 때는 새 `.comp`를 복사하지 않고 `yvo_add_shaders()` 항목을 `<src>.comp@<out_name>@<DEF1>[,<DEF2>...]`로 쓴다
  (예: `overlay_boxes.comp@overlay_boxes_buf@YVO_OVERLAY_BUFFER` → `overlay_boxes_buf.spv`). 문법 검증은 `cmake/YvoShaders.cmake`.

### 2.4 CMake

- 공통 옵션(`-mcpu`, 경고), OpenCV, 자산 경로는 `cmake/YvoTargets.cmake`에서만 정한다. 하위 CMakeLists에 `-mcpu`나 `${OpenCV_LIBS}`를 직접 쓰지 않는다.
- 필수 폴더는 일반 `add_subdirectory`로 붙인다. 폴더 존재 여부로 분기하지 않는다(`if(EXISTS <소스 폴더>)` 금지). 선택 의존은 `if(TARGET …)`로 판단한다.
- 추적되는 모든 `.cpp`는 어떤 CMakeLists에 등록돼 있어야 한다.

### 2.5 자산 경로

- 소스에 `"03_models/…"`, `"04_data/…"`, `"07_results/…"` 같은 CWD 기준 리터럴을 쓰지 않는다.
  도구·앱·테스트는 `yvo_assets`가 주는 `YVO_DEFAULT_HEF`, `YVO_DEFAULT_VIDEO`, `YVO_FIXTURE_DIR`, `YVO_REPO_ROOT`를 쓴다.
- 라이브러리는 자산 위치를 모른다(`yvo_assets`를 링크하지 않는다). 경로는 호출자가 넘긴다.
- 도구의 기본 출력은 `build/out/<도구>/`다. `07_results`에 쓸 때는 `--out-dir`로 명시한다 — 보관된 결과를 실수로 덮어쓰지 않기 위해서다.

### 2.6 `06_tools/`

| 폴더 | 담는 것 |
|---|---|
| `bench/` | 측정 셸 `*_bench.sh`와 공통부 `lib/common.sh`. 바이너리는 `$BIN_DIR`, 파이썬은 `$PY_HAILO`(둘 다 `lib/common.sh`에서만 정의) |
| `python/` | 빌드와 무관한 파이썬 도구(`check_links.py`, `check_structure.py`, `relocate.py` 등). `paper/`는 논문 그림·docx 생성 스크립트(§2.7), `tests/`는 그 단위 테스트(`unittest`) |
| `probes/` | 빌드하지 않는 일회성 C/C++ 프로브 |

벤치 스크립트를 추가·삭제하면 `.claude/hooks/npu_guard.py`가 그 목록을 glob으로 찾는지, 그리고 하드코딩된 폴백 `_BENCH_FALLBACK`(있을 때)이 `bench/` 최상위 `.sh`와 같은지 검사기(C10)로 확인한다. C10은 그 파일 안의 **리터럴 목록만** 본다 — 동적으로 계산되는 이름은 자동으로 맞는 것으로 넘긴다.

### 2.7 `10_paper/`

```
10_paper/
  paper_ko.md            논문 초고
  figures/
    README.md            그림 등록부: 번호 · 파일 · 생성 스크립트 · 원본 경로 · 측정 조건 · 논문 절
    figNN_<slug>.svg     본문용(벡터)
    figNN_<slug>.png     투고·미리보기용 300 dpi
  template/
    reference.docx       docx 서식 견본(글꼴·여백·스타일). make_reference_docx.py 의 산출물 — 추적. 손으로 고치지 않는다
  build/                 docx 산출물(paper_ko.docx 등). git 밖(.gitignore) — 언제든 다시 만든다
06_tools/python/paper/
  paper_style.py         공용 스타일(글꼴·크기·색)
  figNN_<slug>.py        그림 하나당 스크립트 하나 — 07_results 만 읽고 10_paper/figures 에 쓴다
  build_figures.py       전체 재생성
  build_docx.py          paper_ko.md + template/reference.docx → build/paper_ko.docx (conda python-dev 에서 실행)
  make_reference_docx.py 서식 견본 생성기 → template/reference.docx (서식을 바꿀 때는 이 스크립트를 고치고 다시 만든다)
```

- 그림 파일은 **스크립트 실행으로만** 만든다(손으로 고치지 않는다). 다이어그램은 예외로, `09_diagrams/*.graphml`을 yEd GUI로 내보낸 것이다.
- 담당: 그림 스크립트(`figNN_*`, `paper_style.py`, `build_figures.py`)는 python-expert, 실행·논문 삽입·커밋은 메인 에이전트(`10_paper`는 사용자 확인 대상). 절차는 `.claude/skills/paper-figures/`.
- 수치의 출처는 `07_results` 정본뿐이다. 등록부의 행마다 생성 스크립트와 원본 경로를 적는다. 검사기 D3(`check_structure.py`)가 그림 파일 ↔ 등록부 행과 경로 존재를 대조한다.
- docx 는 `build_docx.py` 실행으로만 만든다(산출물을 손으로 고치지 않는다 — 내용은 원고에서, 서식은 `make_reference_docx.py`에서 고친다). 그림은 `figures/` 의 PNG 를 넣는다.
  docx 스크립트 작성은 docx-expert, 실행·견본 재생성·커밋은 메인 에이전트다.
  docx 스크립트의 실행 환경(정의 파일 없음 — 이 줄이 정본): conda `python-dev`(Python 3.13, conda-forge `--override-channels -c conda-forge`): python-docx 1.2.0 · lxml 6.1.3 · pandoc 3.12.1 · defusedxml 0.7.1 · mammoth 1.13.0 · docxcompose 2.2.0 · docxtpl 0.20.2 · pillow 12.3.0 · cairosvg 2.9.1 · nodejs 26.10.0, pip docx2python 3.7.1, npm(env 안 전역) docx 9.11.0. 렌더 검사는 시스템 LibreOffice 25.2·poppler-utils·fonts-nanum·fonts-noto-cjk.
  결정: python-dev 환경은 정의 파일 없이 이 문서에 패키지·버전만 적는다 (사용자 결정, 2026-10-14) 결정: docx 견본은 `10_paper/template/`, 산출물은 `10_paper/build/`(추적 안 함), 스크립트는 `06_tools/python/paper/` (사용자 결정, 2026-10-14)

## 3. `07_results/` 레이아웃

### 3.1 실험 폴더

```
07_results/
  README.md            색인 = 실험 등록부 (폴더 ↔ 행 1:1, 유형·상태 열)
  duplicate_png.md     지운 중복 산출물의 md5 대조표
  <실험>/
    summary.md         필수 — 조건 → 표 → 결론 → 주의
    env_bench.txt      필수 (§5 예외만 면제)
    REPORT.md          선택 — 긴 보고서
    notes/*.md         선택 — 부수 메모
    env/               선택 — 측정 전후 추가 스냅샷
    results.csv  results_raw.csv  results_raw_*.csv     스윕형 필수(분할 원본 *_part<N> 허용)
    raw/<설정>_run<N>/   스윕형 필수 — 실행 1회 = 폴더 1개. stdout.log(전문) | run.log(발췌) | hailort.log
    raw/_session/      세션 부산물: hailort.log(온도 프로브), bench_stdout.log(콘솔 전문)
    evidence/          프로브형 필수, 스윕형 선택
    verification/      선택
    samples/           선택 — 측정과 별도인 1회 실행 산출물(PNG, detections.jsonl, 덤프)
    runs/<회차>/       선택 — 같은 실험의 회차·변종. 내부는 위 레이아웃(summary.md 선택). 깊이 1
```

- **실험 하나 = 색인 표의 한 행.** 새 실험 폴더를 만들면 같은 작업에서 색인 행을 추가한다.
- **스윕형**: 설정 × 반복을 돌려 CSV로 집계한다. **프로브형**: 관문 통과 여부나 원인을 확인한다.

### 3.2 evidence/ 와 verification/ 의 경계 — 목적으로 나눈다

- `evidence/`: **결론의 근거.** 관문 통과, 원인 특정, 정확도 판정. perf든 hailortcli든 도구는 상관없다.
- `verification/`: **변경이 아무것도 깨지 않았다는 게이트 로그.** verify 스킬 단계(빌드·ctest·검증 레이어·ASan·perf·hailort.log)와 그 게이트 표에 오른 회귀 확인.

판정 질문: *이 파일이 없으면 summary의 **결론**이 근거를 잃는가(→ evidence), **변경의 무해성**이 근거를 잃는가(→ verification)?*

### 3.3 내용 불변 원칙

- `07_results` 안의 로그·CSV·txt·jsonl은 **내용을 고치지 않는다.** 그 안에 적힌 옛 경로도 실행 당시의 역사다(§6으로 해독한다).
- 규약은 **소급해서 내용을 만들지 않는다.** 없는 `env_bench.txt`를 나중에 재구성하는 것은 날조다.
- **위치·이름만 바꾸는 이동은 허용한다.** 조건은 셋이다.
  1. 바이트가 그대로다(git rename R100, blob 해시 동일).
  2. 파일 ↔ 회차 대응이 경로·파일명에서 **기계적으로** 정해진다. 해석이 필요하면 옮기지 않고 §5 예외로 둔다.
  3. 새 이름이 내용을 다르게 말하지 않는다(발췌 로그를 `stdout.log`라고 부르지 않는다).

  이동은 `06_tools/python/relocate.py`와 §6 매핑표로 하고, **이동 커밋과 참조 갱신 커밋을 나눈다.**

### 3.4 중복 산출물 삭제

md5로 바이트 동일을 확인 → `duplicate_png.md` 대조표에 기록 → 삭제. 정본 1부는 반드시 남긴다.
**이 절차는 바이트 동일 사본에만 쓴다.** 그 밖의 삭제(유일본·코드·문서)는 사용자 확인을 받는다(`CLAUDE.md` §사용자 확인).
md5가 하나라도 다르면 건드리지 않는다 — 그 차이 자체가 결론일 수 있다.
대조표의 "남긴 파일" 열은 이동을 따라 갱신하고, "지운 파일" 열은 삭제 당시 기록이라 고치지 않는다.

### 3.5 hailort.log

HailoRT는 **실행한 디렉터리(CWD)**에 `hailort.log`를 쓴다.
- 07_results에 기록하는 측정·검증 실행: 그 실행 폴더(`raw/<설정>_run<N>/`), 세션 부산물은 `raw/_session/`,
  검증 실행은 `verification/hailort_<무엇>.log`. git으로 추적한다(0바이트 로그는 추적하지 않아도 된다).
- 그 밖의 디버깅 실행: `11_logs/<YYYY-MM-DD>_<무엇>.log`로 옮긴다(git 밖). 한 작업의 실행 로그가 여러 개면
  `11_logs/<YYYY-MM-DD>_<무엇>/` 디렉터리를 CWD로 삼아 그 안에 `hailort.log`와 stdout 로그를 함께 둬도 된다.
- 저장소 루트의 `hailort.log`는 잔재다.

## 4. 폴더별 담당

`CLAUDE.md`의 "폴더 → 담당" 표가 라우팅 정본이다. 아래는 **요약이라 행이 빠져 있고**(`06_tools/probes`, `11_logs`,
`build*`, 저장소 밖 `~/.claude/**`), 비고 열도 없다. **두 표가 어긋나면 `CLAUDE.md`가 이긴다.**
검사기 D2가 이 표의 경로와 담당이 `CLAUDE.md` 표와 맞는지 대조한다.

| 경로 | 담당 |
|---|---|
| `01_src/{hailo,tools/hailo}`, `03_models`, `04_data/raw_tensors` | hailo-expert |
| `01_src/{vulkan,display,testkit,tools/vulkan,tools/display}`, `02_shaders` | vulkan-expert |
| `01_src/{core,toolkit,decode,isp,tools/isp,tools/decode}`, `**/CMakeLists.txt`, `cmake/` | cpp-expert |
| `01_src/ref` | 그 원문을 쓰는 셰이더·경로의 전문가 (바꾸면 verify ctest 필수) |
| `01_src/pipeline` | 바뀌는 단계의 전문가 |
| `01_src/{gstelem,tools/gst}` | gstreamer-expert (SandLetterbox 사용은 vulkan-expert, HailoRunner 바인딩은 hailo-expert 검토) |
| `05_tests` | 테스트 대상 도메인의 전문가 + verify 스킬 |
| `06_tools/bench` | 실행은 measure 스킬, 수정은 측정 대상 도메인의 전문가 |
| `06_tools/python` | python-expert (하위 예외: `paper/` docx 스크립트는 docx-expert, HEF·fw-actions 분석은 hailo-expert) |
| `07_results` | 쓰기는 measure·verify 스킬, 이동은 relocate.py, 정합 점검은 integrity-auditor |
| `08_learn`, `README.md` | 메인 에이전트 (기계적 수정은 integrity-auditor) |
| `09_diagrams` | graphml-expert (포맷), 수치는 도메인 전문가 |
| `10_paper`, `.claude`, `CLAUDE.md`, `STRUCTURE.md`, `06_tools/structure_rules.toml` | 메인 에이전트(사용자 확인). 논문 논증 검토는 paper-reviewer(읽기 전용) |

## 5. 예외 정책

- 예외는 `06_tools/structure_rules.toml`의 `[[exception]]`에만 둔다. `path`, `waive`(면제할 검사), `reason`(필수)을 적는다.
- 쓰이지 않는 예외는 검사기가 경고한다. 원인이 사라지면 예외를 지운다.
- 규칙이 틀렸다고 판단되면 우회하지 말고, 사용자 확인 후 이 문서와 TOML을 고친다.
  **확인을 받고 고치는 것은 메인 에이전트다** — 서브에이전트는 "규칙표 갱신 필요"라고 보고만 한다(`CLAUDE.md` §사용자 확인).
- 일시 우회: Write 훅은 `YVO_PLACEMENT_OFF=1`로 세션을 시작하면 꺼진다. 검사기에는 우회가 없다.

## 6. 경로 이력

옛 경로 → 새 경로 대조표. 원시 로그·옛 커밋에 남은 경로를 읽을 때 쓰고, `relocate.py`의 입력이다.

### 코드 대역 (C1–C8, 2026-09-22, 적용 완료)

이 표는 해독용이다(머리행이 `이전 위치`라 `relocate.py`가 읽지 않는다).

| 이전 위치 | 현재 위치 | 커밋 |
|---|---|---|
| `build/01_src/<도메인>/yvo_*`, `build/01_src/<도메인>/tools/yvo_*` | `build/bin/yvo_*` | C1 |
| `01_src/pipeline/ffmpeg_drm_decoder.*` | `01_src/decode/ffmpeg_drm_decoder.*` | C3 |
| `01_src/common/{frame_types,coco_labels,dma_heap_alloc,dmabuf_map,hailo_align}.hpp` | `01_src/core/` | C4 |
| `01_src/common/{cpu_letterbox,sand,cpu_yolov8_decode,detection_match,glyph_font_8x16}.hpp` | `01_src/ref/` | C4 |
| `01_src/display/overlay_reference.hpp` | `01_src/ref/overlay_reference.hpp` | C4 |
| `01_src/common/{tool_util,tool_image,stats}.hpp` | `01_src/toolkit/` | C4 |
| 타깃 `yvo_common` | `yvo_core` | C4 |
| `01_src/display/{offscreen_target,display_convert_check}.hpp`, `01_src/vulkan/postproc_fixture.hpp` | `01_src/testkit/` | C5 |
| `01_src/<도메인>/tools/*.cpp` | `01_src/tools/<도메인>/*.cpp` | C6 |
| `01_src/pipeline/tools/drm_dump.cpp` | `01_src/tools/decode/drm_dump.cpp` | C6 |
| `02_shaders/{display_probe,heap_sum}.comp` | `02_shaders/diag/` | C7 |
| 도구 기본 출력 `07_results/<실험>/…` | `build/out/<도구>/…` | C8 |

### 07_results (R1, 2026-09-22)

| 옛 경로 | 새 경로 | 규칙 | 이유 |
|---|---|---|---|
| `07_results/decode_benchmark/raw/(?P<cfg>[^/]+)_run(?P<n>\d+)\.log` | `07_results/decode_benchmark/raw/\g<cfg>_run\g<n>/run.log` | re | raw 평면 로그 → 실행 폴더 (84개) |
| `07_results/nms_host_cost/raw/(?P<cfg>[^/]+)_(?P<n>\d+)\.log` | `07_results/nms_host_cost/raw/\g<cfg>_run\g<n>/stdout.log` | re | raw 평면 로그 → 실행 폴더, `_N` → `_runN` (6개) |
| `07_results/(?P<exp>[^/]+)/raw/hailort\.log` | `07_results/\g<exp>/raw/_session/hailort.log` | re | 세션 hailort.log 는 raw/_session/ (6개) |
| `07_results/isp_resize/pipeline/raw/hailort.log` | `07_results/isp_resize/runs/pipeline/raw/_session/hailort.log` | file | 회차 폴더의 세션 로그 |
| `07_results/isp_resize/pipeline_rerun/raw/hailort.log` | `07_results/isp_resize/runs/pipeline_rerun/raw/_session/hailort.log` | file | 회차 폴더의 세션 로그 |
| `07_results/op_point/latency_gen3_cooled/raw/hailort.log` | `07_results/op_point/runs/latency_gen3_cooled/raw/_session/hailort.log` | file | 회차 폴더의 세션 로그 |
| `07_results/op_point/latency_sweep/raw/hailort.log` | `07_results/op_point/runs/latency_sweep/raw/_session/hailort.log` | file | 회차 폴더의 세션 로그 |
| `07_results/isp_resize/pipeline/` | `07_results/isp_resize/runs/pipeline/` | dir | 회차 폴더는 runs/ |
| `07_results/isp_resize/pipeline_rerun/` | `07_results/isp_resize/runs/pipeline_rerun/` | dir | 회차 폴더는 runs/ |
| `07_results/isp_resize/detections/compare_isp_A_nhwc_b8.csv` | `07_results/isp_resize/evidence/compare_isp_A_nhwc_b8.csv` | file | 비교 결과는 결론의 근거 → evidence/ |
| `07_results/isp_resize/detections/compare_isp_C_nhcw_b8.csv` | `07_results/isp_resize/evidence/compare_isp_C_nhcw_b8.csv` | file | 〃 |
| `07_results/isp_resize/detections/compare_stdout.log` | `07_results/isp_resize/evidence/compare_stdout.log` | file | 〃 |
| `07_results/isp_resize/detections/strong_match_diagnostic.txt` | `07_results/isp_resize/evidence/strong_match_diagnostic.txt` | file | 〃 (비교 진단) |
| `07_results/isp_resize/detections/` | `07_results/isp_resize/samples/detections/` | dir | 설정별 검출 덤프는 표본 |
| `07_results/op_point/latency_gen3_cooled/` | `07_results/op_point/runs/latency_gen3_cooled/` | dir | 회차 폴더는 runs/ |
| `07_results/op_point/latency_sweep/` | `07_results/op_point/runs/latency_sweep/` | dir | 회차 폴더는 runs/ |
| `07_results/npu_benchmark/gen2/` | `07_results/npu_benchmark/runs/gen2/` | dir | 회차 폴더는 runs/ (내부는 레거시 예외) |
| `07_results/npu_benchmark/gen3/cooled/` | `07_results/npu_benchmark/runs/gen3_cooled/` | dir | runs/ 는 깊이 1 → 평탄화 |
| `07_results/npu_benchmark/gen3/` | `07_results/npu_benchmark/runs/gen3/` | dir | 회차 폴더는 runs/ |
| `07_results/pipeline_ffmpeg_vulkan_hailo/preproc/` | `07_results/pipeline_ffmpeg_vulkan_hailo/samples/` | dir | 전처리 결과 이미지는 표본 |
| `07_results/pipeline_ffmpeg_vulkan_hailo/selftest/` | `07_results/pipeline_ffmpeg_vulkan_hailo/evidence/` | dir | selftest 는 관문 통과 증거 |
| `07_results/gpu_postproc_e2e/dump_gpu/` | `07_results/gpu_postproc_e2e/samples/` | dir | 검출 덤프는 표본 |
| `07_results/display_e2e/m5a_baseline.txt` | `07_results/display_e2e/raw/m5a_run1/m5a_baseline.txt` | file | M5a 단일 실행의 stdout 발췌 |
| `07_results/display_e2e/raw/m5a/` | `07_results/display_e2e/raw/m5a_run1/` | dir | 실행 폴더 이름 규약 `<설정>_run<N>` |
| `07_results/display_vulkan_hdmi/verification/p07_overlay_timing_idle.txt` | `07_results/display_vulkan_hdmi/evidence/p07_overlay_timing_idle.txt` | file | P7 결론의 근거(게이트 로그가 아님) |
| `07_results/raw_runner_backpressure/verification/bench_stdout.log` | `07_results/raw_runner_backpressure/raw/_session/bench_stdout.log` | file | 벤치 세션 stdout 은 raw/_session/ |
