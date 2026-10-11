# yolo_vulkan_optimize

Raspberry Pi 5 + Hailo-8 NPU로 YOLO(`03_models/yolov11m.hef`) 추론 파이프라인을 만들고 최적화하는 프로젝트.

## 목적
1. **CPU 오프로딩** — NPU(Hailo-8), GPU(V3D, Vulkan), 하드웨어 블록(HEVC 디코더, ISP(PiSP BE), HVS 표시 평면)을
   최대한 써서 CPU가 하던 일을 넘긴다.
2. **파이프라인 단계 간 메모리 복사 최소화** — 단계 사이 버퍼는 dma-buf 공유(zero-copy)로 넘기고,
   CPU 복사와 레이아웃 변환을 없앤다.
3. **YOLO 모델의 NPU 최적화** — HEF(컨텍스트 구성, 입출력 형식, 후처리 위치)를 Hailo-8에 맞춰 처리량·지연을 개선한다.
   재컴파일(DFC)은 x86 전용이라 Pi에서는 분석·측정까지만 한다. 모델이나 HEF를 바꾸는 일은 사용자가 결정한다.

## 언어
- **코드 주석·출력 문구는 영어**다(`01_src/`, `02_shaders/`, `05_tests/`, `06_tools/`, CMake, `.claude/hooks/`).
  printf·`--help`·예외·gtest 메시지·셸 echo·docstring 포함. 식별자도 영어.
- **문서(`.md`)와 사용자 응답은 한국어**다(`08_learn/`, `07_results/**/summary.md`, README, 논문 등).
- 예외: `06_tools/python/check_structure.py`·`relocate.py` 안에서 한국어 문서의 표 머리·제목을 찾는 파싱 문자열은 한국어로 둔다.
- 코드 줄 번호를 문서가 `file:line`으로 많이 인용하므로, 주석을 고칠 때 괜히 줄을 늘리거나 줄이지 않는다.

## 폴더 구조
**배치 규칙의 정본은 [`STRUCTURE.md`](STRUCTURE.md)다**(기계용은 `06_tools/structure_rules.toml`). 여기는 요약만 둔다.
파일 단위의 자세한 지도(학습용)는 [`08_learn/09_code_map.md`](08_learn/09_code_map.md).
최상위 번호는 네 대역이다(2026-09-21). 번호 없는 것: `cmake/`, `.claude/`, `build*/`. 대역 안쪽은 2026-09-22에 재배치했다.

| 대역 | 폴더 |
|---|---|
| 코드 | `01_src/` — 라이브러리 `{core,ref,toolkit,testkit,decode,hailo,vulkan,display,isp,pipeline,gstelem}/`, 빌드되는 도구 `tools/<도메인>/`. `02_shaders/` (진단용은 `diag/`) |
| 자산 | `03_models/` (HEF), `04_data/{videos,raw_tensors}/` |
| 검증·측정 | `05_tests/` (GoogleTest 실행 파일 8개 + live 1, ctest 94), `06_tools/{bench,python,probes}/`, `07_results/<실험>/` |
| 문서 | `08_learn/` (자습 교재), `09_diagrams/` (yEd `.graphml`), `10_paper/` (논문 초고) |
| 로그 | `11_logs/` 07_results에 기록하지 않는 실행의 런타임 로그(git 밖) |

- `01_src/ref/`는 **CPU 기준 구현**(셰이더 수식의 원문·판정 규칙), `toolkit/`은 도구 전용 헬퍼(라이브러리 include 금지),
  `testkit/`은 테스트·프로브 공용 하네스다. include 계층은 `STRUCTURE.md` §2.2 — hailo↔vulkan은 서로 include·링크하지 않는다.
- `01_src/display/`는 표시 경로다. Vulkan 합성(`DisplayPipeline`: display_convert + 스왑체인)과 HVS 평면 스캔아웃
  (`PlaneDisplay`: `--display-backend wayland-planes`는 Wayland linux-dmabuf, `kms`는 KMS atomic)이 `DisplaySink` 인터페이스 뒤에 있다.
  회귀 격리를 위해 `yvo_vulkan`과 별도 라이브러리(`yvo_display`)다. kms 실표시는 DRM master가 필요하다 — lightdm을 멈추지 말고
  `loginctl activate <tty 세션>`으로 데스크톱을 비활성화한 뒤 실행하고, 실행 중에는 VT를 바꾸지 않는다(`08_learn/07_video_display.md` §7.12).
  화면이 사용자 눈앞에서 바뀌므로 **kms 실행은 사용자가 자리에 있을 때만 한다** — §사용자 확인. 기본은 `SKIP_KMS=1`이다.
- `01_src/isp/`(`yvo_isp`)는 PiSP BE 전처리 대체 경로. **구현돼 있으나 파이프라인에 연결돼 있지 않다**(검출 −2.33%로 보류).
- 실행파일은 모두 `build/bin/`에 나온다. 도구의 기본 자산 경로는 `yvo_assets` 매크로(절대 경로)라 CWD와 무관하고,
  기본 출력은 `build/out/<도구>/`다 — `07_results`에 쓸 때는 `--out-dir`을 명시한다.
- `04_data/videos/`의 기본 입력은 `video_h265.mp4`(1080p30, 약 1분)다. 혼잡 장면용 `mot20_{02,05}_h265.mp4`(MOT20, 1080p25 900프레임, 출처 `07_results/nms_crowd_stress/evidence/inputs.txt`)도 있다.
  `urban_street.mp4`·`video_h264.mp4`는 `decode_benchmark` 당시 입력이며 저장소에 없다.
  `04_data/raw_tensors/`는 `PostprocFixture.*` 테스트가 쓰는 NPU 원시 출력 픽스처(5프레임 × 6텐서).
- `06_tools/`는 **빌드에 들어가지 않는 것**만 둔다. 언어가 아니라 그 기준으로 나눈다 — [`06_tools/README.md`](06_tools/README.md).
- `07_results/<실험>/` 측정 원시 출력과 `summary.md`. **수치의 정본이다.** 색인(= 실험 등록부)은 [`07_results/README.md`](07_results/README.md).
  결과 파일의 내용은 고치지 않는다. 위치를 바꿀 때는 `relocate.py` + `STRUCTURE.md` §6 매핑표.
- 모델·영상·빌드·perf 덤프는 `.gitignore`에 들어 있다. 단 `07_results/**/hailort*.log`는 신뢰성 판정 근거라 추적한다
  (0바이트 로그는 정보가 없어 예외 — 추적하지 않아도 된다).

## 새 파일 배치 규칙
- 새 파일은 [`STRUCTURE.md`](STRUCTURE.md)의 위치 규칙을 따른다. **Write 훅 `placement_guard`**가 새 파일 경로를
  규칙표로 판정해 어긋나면 막고 올바른 위치를 알려 준다(기존 파일 수정, 저장소 밖, `build*/`·`11_logs/`는 보지 않는다).
- 훅은 Bash로 만든 파일(벤치 출력, `cp`, `git mv`)을 보지 못한다. 파일을 만들거나 옮기는 작업을 끝낼 때
  `python3 06_tools/python/check_structure.py`와 `python3 06_tools/python/check_links.py`를 돌려 둘 다 0건인지 확인한다
  (0건의 대상은 `errors`와 `broken`이다. `--stale`의 `guessed`·`manual check`는 오탐이 흔한 **경고**이고 종료 코드에 반영되지 않는다 — 사람이 읽고 판정한다).
- 요점: 루트에 새 파일을 만들지 않는다 · 빌드되는 도구는 `01_src/tools/<도메인>/` · 셰이더는 `02_shaders/` ·
  측정 결과는 `07_results/<실험>/`(실행 로그는 `raw/<설정>_run<N>/`, 회차는 `runs/<회차>/`) · 새 실험 폴더는 색인 행과 함께.
- 규칙이 틀렸다고 판단되면 우회하지 말고 사용자에게 확인한 뒤 `STRUCTURE.md`와 `06_tools/structure_rules.toml`을
  **같은 커밋에서** 고친다(검사기 D1이 둘의 불일치를 잡는다). 서브에이전트는 "규칙표 갱신 필요"라고 보고만 한다 — §사용자 확인.

## 구조 변경 절차 (폴더 이동·이름 변경·대규모 재배치)
1. 바꾸기 전: `integrity-auditor`로 기준선(`check_structure` errors 0 · `check_links` broken 0)을 잡는다.
2. 규칙표를 먼저 고친다 — `STRUCTURE.md`(§6 경로 이력 포함)와 `structure_rules.toml`. 메인 에이전트가 사용자 확인 후 한다.
3. 이동은 `06_tools/python/relocate.py --map STRUCTURE.md --section <절> --apply-moves`로 하고,
   **이동 커밋(R100만)과 참조 갱신 커밋(`--apply-refs` + 수동 목록)을 나눈다.**
4. 바꾼 뒤: `check_structure.py`, `check_links.py`, `check_links.py --stale STRUCTURE.md`(판정 대상은 `old paths remaining`
   0건이다 — `guessed`·`manual check` 경고는 사람이 읽고 넘긴다), 코드가 바뀌었으면 `verify` 스킬 1~2단계.
5. `integrity-auditor`로 사후 점검한다.

## 사용자 확인 (정본)

아래 일은 **사용자에게 먼저 확인받는다.** 우회하지 않고, 답이 오기 전에 저지르지 않는다.
개별 문맥에도 같은 규칙이 적혀 있지만(§새 파일 배치 규칙, §구조 변경 절차, §폴더 → 담당, 에이전트 정의),
**충돌하면 이 절이 이긴다.**

| 확인이 필요한 일 | 비고 |
|---|---|
| 배치 **규칙표** 수정 — `STRUCTURE.md` + `06_tools/structure_rules.toml`을 같은 커밋에서 | 서브에이전트는 "규칙표 갱신 필요"라고 보고만 한다 |
| `CLAUDE.md`·`STRUCTURE.md`·`.claude/**`·`~/.claude/**`·`10_paper/` 수정 | 아래 **주체** 규칙 |
| **삭제** — 결과·코드·문서. **바이트 동일 사본만 예외**(`STRUCTURE.md` §3.4 절차를 그대로 따를 때) | 유일본은 언제나 확인. "정리해줘"는 후보만 찾는 요청으로 읽는다 |
| 모델·**HEF 교체** | 재컴파일(DFC)은 x86 전용이라 Pi에서는 분석·측정까지만 한다 |
| **시스템 변경** — `sudo`, sysctl, `/boot/firmware/config.txt`, 펌웨어·드라이버, 패키지 설치(apt·pip), 재부팅 | 에이전트가 직접 하지 않는다. 변경안·기대 효과·되돌리는 법을 보고한다 |
| 보관된 결과 폴더 **덮어쓰기**(`FORCE=1`) | `guard_out_dir`가 막는 것을 푸는 일이다 |
| **사용자 환경 점유** — kms 세션 전환(`loginctl activate`), yEd GUI 창, 장시간 측정 | 사용자가 자리에 있을 때만. 측정은 `npu_guard` 락으로 다른 작업을 막는다 |
| **설계 이탈** — 승인된 계획·설계와 다르게 가야 할 때 | 고치기 전에 보고한다. 이탈한 채로 끝내고 사후 통보하지 않는다 |

**주체.** `CLAUDE.md`·`STRUCTURE.md`·`.claude/**`·`~/.claude/**`·`10_paper/`·`06_tools/structure_rules.toml`은
**승인된 계획에 그 수정이 적혀 있어도 메인 에이전트가 직접 고친다.** 서브에이전트에 위임하지 않는다 —
줄번호·개수 같은 기계적 수정도 예외가 아니다. 서브에이전트는 읽고 "이렇게 고쳐야 한다"고 보고만 한다.
`integrity-auditor`는 `Edit` 도구를 쥐고 있어 **도구로는 가능하고 이 규칙으로만 막힌다.**

**보고 형식.** 서브에이전트가 확인이 필요하다고 판단하면 보고의 `## 남은 이슈`에 이 라벨로 적는다.
본문에 섞어 적지 않는다 — 메인 에이전트가 그 줄을 찾아 사용자에게 올린다.

```
사용자 확인 필요: <무엇을 바꾸려는가>
  왜:          <확인이 필요한 이유>
  선택지:      <2~3개와 각각의 대가>
  되돌리는 법: <롤백 방법. 불가능하면 "불가">
  기본 권고:   <답이 없으면 무엇을 하지 않을 것인가>
```

**기록.** 확인받은 결정은 **주체와 날짜**를 함께 적는다 — `결정: … (사용자 결정, YYYY-MM-DD)`.
"보류한다", "수정하지 않는다" 같은 결정도 같은 형식으로 남긴다. 누가 정했는지 없는 기록은
나중에 근거로 쓸 수 없다. 앞으로 쓰는 문서에만 적용한다(`07_results`의 기존 결과 파일 내용은 고치지 않는다).


## 개발 환경
- C++: g++ 14.2, CMake 3.31 + Ninja. 빌드는 `cmake -S . -B build -G Ninja && cmake --build build`, 산출물은 `build/`.
- CMake 의존 (실제 `find_package`·`pkg_check_modules` 호출 기준. 추가하기 전에 `CMakeLists.txt`를 확인할 것):
  - **필수**: `HailoRT`(→ `HailoRT::libhailort`), `Vulkan`(→ `Vulkan::Vulkan`),
    `PkgConfig` → `libavformat`·`libavcodec`·`libavutil`·`libdrm` (`01_src/decode/CMakeLists.txt`)
  - **선택**: `OpenCV`(**REQUIRED 아님** — 없으면 경고만 내고 OpenCV를 쓰는 타깃이 빠진다),
    `wayland-client`+`wayland-protocols`(없으면 표시 경로가 kms 백엔드까지 조용히 빠진다. staging `color-representation` XML은 있으면 쓴다), `libpisp`(없으면 `yvo_isp` 전체가 빠진다),
    `gstreamer-1.0`·`-base`·`-video`·`-allocators`·`-app` ≥1.24(없으면 `yvo_gstelem`과 `tools/gst/` 도구가 빠진다)
  - **`YVO_BUILD_TESTS`일 때**: `GTest`(→ `GTest::gtest_main`)
  - `pybind11`·`nlohmann_json`은 **쓰지 않는다.** `find_package` 호출이 없다.
    nlohmann_json은 conda prefix에만 있어 다른 시스템 헤더를 가릴 위험이 있어 의도적으로 뺐다
    (`05_tests/test_postproc.cpp`, `01_src/tools/vulkan/postproc_probe.cpp` 주석).
  - Google Benchmark 의존은 2026-09-21에 제거했다(소비자가 0개인 configure 하드 의존이었다).
- 셰이더: `glslc`, `glslangValidator`, `spirv-val`/`spirv-opt`, Khronos 검증 레이어.
- 테스트·측정: GoogleTest(CPU 기준 구현 대비 GPU/NPU 경로 결과 비교), `perf`, `valgrind`, `gdb`.
  GPU 구간 시간은 Vulkan timestamp query로 잰다. 측정 자동화는 `06_tools/bench/*.sh`.
  **벤치가 도는 동안 `build/.bench_running` 락이 생기고 `npu_guard` 훅이 빌드·테스트를 막는다** —
  측정 중 다른 작업이 돌면 CPU 시간이 에러 없이 오염되기 때문이다.
- `kernel.perf_event_paranoid=1`(`/etc/sysctl.d/99-perf.conf`로 영구 설정): 일반 사용자도 perf로 커널·드라이버 구간까지
  프로파일링할 수 있다(tracepoint·raw 이벤트 등 일부는 여전히 root 필요). 에이전트는 이 설정을 바꾸지 않는다.
- Python: conda 환경 `hailo-yolo`(Python 3.13, `/home/pi/miniconda3/envs/hailo-yolo/bin/python`), 정의는 `environment.yml`.
  `hailo_platform` 4.24.0 설치 완료(`~/Downloads/hailort-4.24.0-cp313-cp313-linux_aarch64.whl`, 시스템 `libhailort.so.4.24.0`에 링크).
  Python에서 InferModel을 쓸 때는 VDevice 스케줄러를 `ROUND_ROBIN`으로 켜야 한다(기본값 NONE이면 `HAILO_STREAM_NOT_ACTIVATED`).
- 설치 금지: apt `python3-hailort`, `hailo-all`. Python 바인딩이 4.23이라 런타임 4.24.0과 맞지 않는다.
- `hailort.log`는 HailoRT 런타임이 **실행한 디렉터리(CWD)**에 쓰는 로그다. 에러를 분석할 때 참고한다. 보관 위치는 하나로 정한다
  (`STRUCTURE.md` §3.5): **07_results에 기록하는 측정·검증 실행**은 그 실행 폴더(`raw/<설정>_run<N>/`, 세션 부산물은
  `raw/_session/`, 검증은 `verification/hailort_<무엇>.log`)에 두고 git으로 추적한다. **그 밖의 디버깅 실행**은
  `11_logs/<YYYY-MM-DD>_<무엇>.log`로 옮긴다. 저장소 루트의 `hailort.log`는 잔재다.
- MCP 서버 (프로젝트 `.mcp.json`):
  - `gstreamer` (`/home/pi/workspace/gstreamer-mcp`) → 도구 `mcp__gstreamer__*`.
    **단 앱 본체(`yvo_pipeline`)는 GStreamer를 쓰지 않는다** — `decode_bench.sh`의 디코드 비교에서 외부 프로세스로 호출하고,
    `01_src/gstelem/`(선택)은 GStreamer 디코드 → Vulkan 전처리 → Hailo 를 비교하는 **연구 경로**다(도구 `tools/gst/`에 정적 등록).
  - `vulkan-docs` (`/home/pi/workspace/mcp-Vulkan`) → 도구 `mcp__vulkan-docs__*`.
    VUID 원문·구조체는 로컬 레지스트리 `/usr/share/vulkan/registry/{validusage.json,vk.xml}`가 더 정확하고,
    MCP는 서술적 설명에 쓴다.

## 폴더 → 담당 (라우팅 정본)
작업 대상 파일의 위치로 먼저 담당을 고르고, 아래 "서브에이전트 라우팅"의 기술 영역 규칙으로 보정한다.
한 작업이 여러 행에 걸치면 **가장 많이 바뀌는 쪽** 담당에 맡기고 나머지는 검토를 붙인다.

| 경로 | 담당 | 비고 |
|---|---|---|
| `01_src/hailo/`, `01_src/tools/hailo/`, `03_models/`, `04_data/raw_tensors/` | `hailo-expert` | 픽스처는 `yvo_hailo_raw_probe` 산출물. **HEF 교체는 사용자 결정**(§목적 3) |
| `01_src/vulkan/`, `01_src/display/`, `01_src/testkit/`, `01_src/tools/{vulkan,display}/`, `02_shaders/` | `vulkan-expert` | `display/kms_scanout.*`(libdrm atomic)을 크게 바꾸면 `cpp-expert` 검토 |
| `01_src/core/`, `01_src/decode/`, `01_src/isp/`, `01_src/tools/{isp,decode}/`, `CMakeLists.txt`, `cmake/` | `cpp-expert` | `core/`의 dma-heap·dma-buf 글루를 바꾸면 양쪽 전문가 검토 |
| `01_src/ref/` | 그 원문을 쓰는 셰이더·경로의 전문가 | **바꾸면 `verify` ctest 필수**(비트 일치 테스트의 기준이다) |
| `01_src/toolkit/` | `cpp-expert` | 라이브러리가 include하면 안 된다 |
| `01_src/pipeline/` | 바뀌는 단계의 전문가 | 디코더 연결은 `cpp-expert`, `gpu_postproc_stage`는 `vulkan-expert`, `infer_stage`는 `hailo-expert` |
| `01_src/gstelem/`, `01_src/tools/gst/` | `gstreamer-expert` | `SandLetterbox` 사용·dma-buf 수명은 `vulkan-expert`, `HailoRunner` 바인딩은 `hailo-expert`, CMake 선택 의존은 `cpp-expert` 검토 |
| `05_tests/` | 테스트 대상 도메인의 전문가 + `verify` 스킬 | |
| `06_tools/bench/` | 실행은 `measure` 스킬, 수정은 측정 대상 도메인의 전문가 | 추가·삭제 시 `npu_guard`가 자동으로 찾는지 확인(검사기 C10) |
| `06_tools/python/` | `python-expert` | |
| `06_tools/probes/` | 소속 도메인의 전문가 | 빌드하지 않는다 |
| `07_results/` | 쓰기는 `measure`(측정)·`verify`(`verification/`) 스킬, 이동은 `relocate.py`, 정합 점검은 `integrity-auditor` | 결과 파일 내용은 고치지 않는다. 삭제·`FORCE=1` 덮어쓰기는 §사용자 확인(바이트 동일 사본은 `STRUCTURE.md` §3.4 절차로 예외) |
| `08_learn/`, `README.md` | 메인 에이전트 | 기계적 수정(경로·줄번호·개수)은 `integrity-auditor` |
| `09_diagrams/` | `graphml-expert`(포맷) | 수치 변경은 도메인 전문가 |
| `10_paper/`, `.claude/`, `~/.claude/agents/*`, `~/.claude/settings.json`, `CLAUDE.md`, `STRUCTURE.md`, `06_tools/structure_rules.toml` | 메인 에이전트(사용자 확인) | **§사용자 확인**이 정본이다. 승인된 계획이어도 위임하지 않는다. `.claude/agents`·훅 변경은 **다음 세션부터** 반영된다. `~/.claude/**`는 저장소 밖이라 검사기·`placement_guard` 훅이 보지 않는다 |
| `11_logs/` | 누구나 | git 밖 |
| `build*/` | 쓰지 않는다 | 산출물 |

## 서브에이전트 라우팅 (프로젝트 우선)
전역 지침보다 이 규칙이 우선한다.
- Vulkan/GPU 관련 작업(Vulkan 호스트 코드, GLSL/SPIR-V 셰이더, 검증 레이어 에러, GPU 전/후처리,
  Hailo↔GPU 버퍼 공유, GPU 성능 측정) → `vulkan-expert` (전역 `cpp-expert`의 Vulkan 항목 대신)
- Hailo NPU 관련 작업(HailoRT C++/Python API, HEF 분석, 장치·드라이버·펌웨어·PCIe, NPU 처리량·지연 측정,
  hailonet GStreamer, HailoRT 쪽 DMA 버퍼) → `hailo-expert` (전역 `cpp-expert`/`python-expert` 대신)
- Vulkan·Hailo와 무관한 C/C++(빌드 설정, CPU/NEON 경로) → `cpp-expert`
- Vulkan·Hailo와 무관한 파이썬(벤치마크 집계 스크립트, 시각화) → `python-expert`
- 한 작업이 전문 영역과 일반 C++에 걸치면 핵심 쪽 전문 에이전트(`vulkan-expert`/`hailo-expert`)에 맡긴다.
- GStreamer 파이프라인 전반(구성·caps 협상·성능·appsink 연동) → `gstreamer-expert`(전역).
  단, hailonet/synchailonet의 HEF·HailoRT 설정 문제 → `hailo-expert`, vulkan* 요소의 셰이더·호스트 코드 → `vulkan-expert`.
  hailonet이 들어간 파이프라인 작업은 파이프라인 문자열·연결은 `gstreamer-expert`, NPU 설정은 `hailo-expert`에 나눠 맡긴다.
- Hailo↔Vulkan 버퍼 공유(zero-copy)처럼 양쪽에 걸치면: 먼저 인터페이스(dma-buf fd, 크기, 포맷, 정렬,
  동기화 시점)를 정하고 HailoRT 쪽은 `hailo-expert`, Vulkan import 쪽은 `vulkan-expert`에 나눠 위임한다.
- `09_diagrams/*.graphml`(yEd 파이프라인 다이어그램 2장)의 **파일 포맷·스키마·yEd 방언·검증**이 핵심인 작업
  → `graphml-expert`(전역). 관리 규약은 `09_diagrams/README.md`에 있다.
  단 다이어그램에 적힌 **수치를 고치는 일**은 측정 문서를 근거로 하는 작업이라 해당 영역 전문 에이전트에 맡긴다.
  "파이프라인 그래프", "노드", "추론 그래프"처럼 **단어만 겹치는** 작업은 `graphml-expert`로 보내지 않는다.
- 문서·코드 무결성 점검(문서의 링크·`file:line`·심볼·개수·수치가 코드와 `07_results` 정본에 맞는지, 죽은 코드,
  미인용·중복 결과 파일, `CLAUDE.md`·`.claude` 설정 서술의 정합) → `integrity-auditor`(프로젝트, Sonnet 5.5 · effort high).
  폴더 재배치·대규모 리팩터·코드 삭제 뒤, 측정 결과를 문서에 반영한 뒤, 큰 변경을 커밋하기 전에 **먼저** 부른다.
  - 이 에이전트는 **기계적 문서 수정**(줄번호·개수·옮겨진 경로)만 직접 한다. 보고된 코드 문제는 위 규칙대로
    `vulkan-expert`/`hailo-expert`/`cpp-expert`/`python-expert`에, 빌드·테스트 확인은 `verify` 스킬로 넘긴다.
  - 기능 구현·버그 수정·성능 측정은 이 에이전트에 보내지 않는다. "정리해줘"처럼 삭제까지 원하는 요청도
    먼저 이 에이전트로 후보를 찾고, 삭제는 사용자 확인 후 담당 에이전트가 한다(§사용자 확인 —
    바이트 동일 사본만 `STRUCTURE.md` §3.4 절차로 예외다).
- 위임 프롬프트에 대상 파일, 목표, 제약, 검증 방법(셰이더 컴파일·검증 레이어·측정 조건)을 포함한다.
  **새 파일을 만드는 작업이면 둘 위치를 `STRUCTURE.md` 기준으로 프롬프트에 적고**, 끝에 `check_structure.py` 결과를 보고하게 한다.
  전역 에이전트(`cpp`·`python`·`gstreamer`·`graphml-expert`)는 다른 프로젝트도 쓰므로 정의 파일에 이 프로젝트 규칙을 적지 않는다 —
  배치 규칙·결과 규약·§사용자 확인의 보고 형식은 **위임 프롬프트에 적어 준다.**
