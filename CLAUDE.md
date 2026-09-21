# yolo_vulkan_optimize

Raspberry Pi 5 + Hailo-8 NPU로 YOLO(`03_models/yolov11m.hef`) 추론을 수행하며, Vulkan(Pi 5 VideoCore VII GPU)을 활용해 전/후처리 등 파이프라인 가속화를 연구하는 프로젝트.

## 폴더 구조
최상위 폴더 번호는 **네 대역**이다(2026-09-21 재배치). 번호 없는 것: `cmake/`, `.claude/`, `build/`, `build-asan/`.
**상세한 폴더·파일 지도는 [`08_learn/09_code_map.md`](08_learn/09_code_map.md)가 정본이다.** 여기는 요약만 둔다.

| 대역 | 폴더 |
|---|---|
| 코드 | `01_src/{common,hailo,vulkan,display,isp,pipeline}/`, `02_shaders/` (GLSL, SPIR-V 산출물은 `build/`) |
| 자산 | `03_models/` (HEF), `04_data/{videos,raw_tensors}/` |
| 검증·측정 | `05_tests/` (GoogleTest 7종), `06_tools/{bench,python,probes}/`, `07_results/<실험>/` |
| 문서 | `08_learn/` (자습 교재), `09_diagrams/` (yEd `.graphml`), `10_paper/` (논문 초고) |
| 로그 | `11_logs/` 보관용 런타임 로그 |

- `01_src/display/`는 Wayland(xdg-shell) 창 + 스왑체인 표시 경로. 회귀 격리를 위해 `yvo_vulkan`과 별도 라이브러리(`yvo_display`)다.
- `01_src/isp/`(`yvo_isp`)는 PiSP BE 전처리 대체 경로. **구현돼 있으나 파이프라인에 연결돼 있지 않다**(검출 −2.33%로 보류).
- `04_data/videos/` 입력 영상: `urban_street.mp4`(720p30 h264, 75분), `video_h264.mp4`·`video_h265.mp4`(1080p30, 약 1분).
  `04_data/raw_tensors/`는 `PostprocFixture.*` 테스트가 쓰는 NPU 원시 출력 픽스처(5프레임 × 6텐서).
- `06_tools/`는 **빌드에 들어가지 않는 것**만 둔다. 언어가 아니라 그 기준으로 나눈다 — [`06_tools/README.md`](06_tools/README.md).
- `07_results/<실험>/` 측정 원시 출력과 `summary.md`. **수치의 정본이다.** 색인은 [`07_results/README.md`](07_results/README.md).
- 모델·영상·빌드·perf 덤프는 `.gitignore`에 들어 있다. 단 `07_results/**/hailort*.log`는 신뢰성 판정 근거라 추적한다
  (0바이트 로그는 정보가 없어 예외 — 추적하지 않아도 된다).

## 개발 환경
- C++: g++ 14.2, CMake 3.31 + Ninja. 빌드는 `cmake -S . -B build -G Ninja && cmake --build build`, 산출물은 `build/`.
- CMake 의존 (실제 `find_package`·`pkg_check_modules` 호출 기준. 추가하기 전에 `CMakeLists.txt`를 확인할 것):
  - **필수**: `HailoRT`(→ `HailoRT::libhailort`), `Vulkan`(→ `Vulkan::Vulkan`),
    `PkgConfig` → `libavformat`·`libavcodec`·`libavutil`·`libdrm` (`01_src/pipeline/CMakeLists.txt`)
  - **선택**: `OpenCV`(**REQUIRED 아님** — 없으면 경고만 내고 OpenCV를 쓰는 타깃이 빠진다),
    `wayland-client`+`wayland-protocols`(없으면 표시 경로가 조용히 빠진다), `libpisp`(없으면 `yvo_isp` 전체가 빠진다)
  - **`YVO_BUILD_TESTS`일 때**: `GTest`(→ `GTest::gtest_main`)
  - `pybind11`·`nlohmann_json`은 **쓰지 않는다.** `find_package` 호출이 없다.
    nlohmann_json은 conda prefix에만 있어 다른 시스템 헤더를 가릴 위험이 있어 의도적으로 뺐다
    (`05_tests/test_postproc.cpp`, `01_src/vulkan/tools/postproc_probe.cpp` 주석).
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
- `hailort.log`는 HailoRT 런타임이 실행한 디렉터리에 쓰는 로그다. 에러를 분석할 때 참고하고, 보관할 로그는 `11_logs/`에 날짜를 붙여 옮긴다.
- MCP 서버 (프로젝트 `.mcp.json`):
  - `gstreamer` (`/home/pi/workspace/gstreamer-mcp`) → 도구 `mcp__gstreamer__*`.
    **단 앱 본체는 GStreamer를 쓰지 않는다** — `decode_bench.sh`의 디코드 비교에서만 외부 프로세스로 호출한다.
  - `vulkan-docs` (`/home/pi/workspace/mcp-Vulkan`) → 도구 `mcp__vulkan-docs__*`.
    VUID 원문·구조체는 로컬 레지스트리 `/usr/share/vulkan/registry/{validusage.json,vk.xml}`가 더 정확하고,
    MCP는 서술적 설명에 쓴다.

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
  미인용·중복 결과 파일, `CLAUDE.md`·`.claude` 설정 서술의 정합) → `integrity-auditor`(프로젝트, sonnet).
  폴더 재배치·대규모 리팩터·코드 삭제 뒤, 측정 결과를 문서에 반영한 뒤, 큰 변경을 커밋하기 전에 **먼저** 부른다.
  - 이 에이전트는 **기계적 문서 수정**(줄번호·개수·옮겨진 경로)만 직접 한다. 보고된 코드 문제는 위 규칙대로
    `vulkan-expert`/`hailo-expert`/`cpp-expert`/`python-expert`에, 빌드·테스트 확인은 `verify` 스킬로 넘긴다.
  - 기능 구현·버그 수정·성능 측정은 이 에이전트에 보내지 않는다. "정리해줘"처럼 삭제까지 원하는 요청도
    먼저 이 에이전트로 후보를 찾고, 삭제는 사용자 확인 후 담당 에이전트가 한다.
- 위임 프롬프트에 대상 파일, 목표, 제약, 검증 방법(셰이더 컴파일·검증 레이어·측정 조건)을 포함한다.
