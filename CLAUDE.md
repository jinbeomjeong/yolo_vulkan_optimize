# yolo_vulkan_optimize

Raspberry Pi 5 + Hailo-8 NPU로 YOLO(`03_models/yolov11m.hef`) 추론을 수행하며, Vulkan(Pi 5 VideoCore VII GPU)을 활용해 전/후처리 등 파이프라인 가속화를 연구하는 프로젝트.

## 폴더 구조
최상위 폴더 번호는 중요도 순이다(코드 → 모델·데이터 → 검증 → 결과 → 보조 도구 → 로그). `build/`, `.claude/`는 번호 없이 둔다.
- `03_models/` HEF 모델 (`yolov11m.hef`)
- `04_data/videos/` 입력 영상: `urban_street.mp4`(720p30 h264, 75분), `video_h264.mp4`·`video_h265.mp4`(1080p30, 약 1분)
- `01_src/{common,hailo,vulkan,display,pipeline}/` C++ 소스, `02_shaders/` GLSL 컴퓨트 셰이더(SPIR-V 산출물은 `build/`)
  `01_src/display/`는 Wayland(xdg-shell) 창 + 스왑체인 표시 경로. 회귀 격리를 위해 `yvo_vulkan`과 별도 라이브러리(`yvo_display`)다.
- `05_tests/` GoogleTest, `06_bench/` Google Benchmark. 각 폴더에 `CMakeLists.txt`가 생기면 최상위 `CMakeLists.txt`가 자동으로 포함한다.
- `06_tools/python/` hailo-yolo 환경 스크립트(집계·시각화·Python 추론), `06_tools/bench/` 측정 자동화 셸 스크립트
- `07_results/<실험>/` 측정 원시 출력과 `summary.md`. 예: `07_results/npu_benchmark/{gen2,gen3}/`
- `11_logs/` 보관용 런타임 로그
- 모델·영상·로그는 `.gitignore`에 들어 있다.

## 개발 환경
- C++: g++ 14.2, CMake 3.31 + Ninja. 빌드는 `cmake -S . -B build -G Ninja && cmake --build build`, 산출물은 `build/`.
- CMake 패키지(`find_package`로 탐지 확인됨): `HailoRT`(→ `HailoRT::libhailort`), `Vulkan`(→ `Vulkan::Vulkan`),
  `OpenCV`, `GTest`(→ `GTest::gtest_main`), `benchmark`(→ `benchmark::benchmark`), `pybind11`,
  `nlohmann_json`(→ `nlohmann_json::nlohmann_json`).
- 셰이더: `glslc`, `glslangValidator`, `spirv-val`/`spirv-opt`, Khronos 검증 레이어.
- 테스트·측정: GoogleTest(CPU 기준 구현 대비 GPU/NPU 경로 결과 비교), Google Benchmark(CPU 마이크로 벤치),
  `perf`, `valgrind`, `gdb`. GPU 구간 시간은 Vulkan timestamp query로 잰다.
- `kernel.perf_event_paranoid=1`(`/etc/sysctl.d/99-perf.conf`로 영구 설정): 일반 사용자도 perf로 커널·드라이버 구간까지
  프로파일링할 수 있다(tracepoint·raw 이벤트 등 일부는 여전히 root 필요). 에이전트는 이 설정을 바꾸지 않는다.
- Python: conda 환경 `hailo-yolo`(Python 3.13, `/home/pi/miniconda3/envs/hailo-yolo/bin/python`), 정의는 `environment.yml`.
  `hailo_platform` 4.24.0 설치 완료(`~/Downloads/hailort-4.24.0-cp313-cp313-linux_aarch64.whl`, 시스템 `libhailort.so.4.24.0`에 링크).
  Python에서 InferModel을 쓸 때는 VDevice 스케줄러를 `ROUND_ROBIN`으로 켜야 한다(기본값 NONE이면 `HAILO_STREAM_NOT_ACTIVATED`).
- 설치 금지: apt `python3-hailort`, `hailo-all`. Python 바인딩이 4.23이라 런타임 4.24.0과 맞지 않는다.
- `hailort.log`는 HailoRT 런타임이 실행한 디렉터리에 쓰는 로그다. 에러를 분석할 때 참고하고, 보관할 로그는 `11_logs/`에 날짜를 붙여 옮긴다.
- GStreamer MCP: `gstreamer` 서버(`/home/pi/workspace/gstreamer-mcp`, 전역 `~/.claude.json`과 프로젝트 `.mcp.json`에 등록). 도구는 `mcp__gstreamer__*`.

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
- 이 프로젝트에는 `.graphml` 파일이 없다. "파이프라인 그래프", "노드", "추론 그래프"처럼 단어만 겹치는 작업은
  전역 `graphml-expert`로 보내지 않는다(해당 영역의 전문 에이전트로 보낸다). 실제 `.graphml` 파일을 다루는
  작업이 생기면 그때만 `graphml-expert`에 위임한다.
- 위임 프롬프트에 대상 파일, 목표, 제약, 검증 방법(셰이더 컴파일·검증 레이어·측정 조건)을 포함한다.
