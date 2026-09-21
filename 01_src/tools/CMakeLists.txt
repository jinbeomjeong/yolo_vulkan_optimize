# 빌드되는 진단·측정 도구 10개를 한 곳에서 등록한다(2026-09-22, 전에는 도메인 폴더마다 tools/ 가 흩어져 있었다).
# 폴더는 **측정 대상 도메인** 기준이다(여러 도메인에 걸치면 무엇을 재는 도구인지로 정한다). 파일명은 옛 이름 그대로다.
# 실행파일은 모두 build/bin/ 에 나온다. 라이브러리 타깃이 없으면(예: libpisp·wayland 부재) REQUIRES 로 조용히 건너뛴다.
#
# 링크 수준 격리(계획서 §5.2): **yvo_hailo·yvo_vulkan 라이브러리는 서로 링크하지 않는다.** 두 쪽을 함께 쓰는 것은
# 도구(여기)와 yvo_stages 뿐이다.

# ── hailo ──────────────────────────────────────────────────────────────────────
# 자체 검증 도구: CPU letterbox(NV12 BT.709) → HailoRunner → 검출 출력·PNG·처리량
yvo_add_tool(yvo_hailo_selftest hailo/hailo_selftest.cpp
  REQUIRES yvo_opencv
  LIBS yvo_hailo yvo_opencv yvo_toolkit)

# 원시 출력 6스트림 프로브 (계획서 §6.2 P1 + P2).
#   --probe p1: 호스트 malloc 버퍼만 쓴다(GPU 미사용)
#   --probe p2: 출력을 dma-buf 로 받는다 → --alloc vulkan 이 yvo_vulkan(VulkanContext, DmaBufExporter)을 쓴다
# **yvo_hailo 라이브러리 자체는 yvo_vulkan 에 의존하지 않는다** — 도구만 둘 다 링크한다.
yvo_add_tool(yvo_hailo_raw_probe hailo/hailo_raw_probe.cpp
  LIBS yvo_hailo yvo_vulkan yvo_toolkit)

# ── vulkan ─────────────────────────────────────────────────────────────────────
# 성능·검증 도구: 전체 디코드 + GPU 전처리
yvo_add_tool(yvo_preproc_bench vulkan/preproc_bench.cpp
  LIBS yvo_vulkan yvo_decoder yvo_toolkit
  OPTIONAL yvo_opencv=YVO_HAVE_OPENCV)

# 후처리 프로브 (계획서 §6.2 P3).
#   --probe fixture|sweep : NPU 불필요 (04_data/raw_tensors 픽스처)
#   --probe live|stale    : NPU 독점. HailoRT 가 있을 때만 켜진다.
# yvo_hailo 라이브러리에 링크해도 **yvo_vulkan 자체의 의존성은 늘지 않는다**
# (계획서 §5.2 링크 수준 격리 — test_preproc / yvo_preproc_bench 의 링크 그래프가 바뀌지 않는다).
# yvo_shaders_diag: --probe heapimport 가 쓰는 heap_sum.comp 임베드 헤더 (부록 B 1단계)
yvo_add_tool(yvo_postproc_probe vulkan/postproc_probe.cpp
  LIBS yvo_vulkan yvo_shaders_diag yvo_toolkit yvo_testkit_postproc
  OPTIONAL yvo_hailo=YVO_HAVE_HAILO)

# ── display ────────────────────────────────────────────────────────────────────
# 프로브: 계획서 §6.3 (P4 surface / P5 present / P6 context-regression / P8 convert).
# P8 은 디코더 SAND dma-buf 를 입력으로 쓴다. yvo_decoder 는 항상 있으므로 YVO_HAVE_DECODER 도 항상 1 이다
# (소스의 #ifdef 는 문서의 file:line 을 지키려고 남겨 두었다).
yvo_add_tool(yvo_display_probe display/display_probe.cpp
  REQUIRES yvo_display
  LIBS yvo_display yvo_shaders_diag yvo_decoder yvo_toolkit yvo_testkit_display
  DEFINES YVO_HAVE_DECODER=1)

# DisplayPipeline 단독 프로브 (마일스톤 7). yvo_pipeline 을 건드리지 않고 표시 경로만 재현한다:
# display_probe 의 run_convert_display() 수치를 DisplayPipeline 경유로 다시 내는 것이 목적이다.
yvo_add_tool(yvo_display_pipeline_probe display/display_pipeline_probe.cpp
  REQUIRES yvo_display
  LIBS yvo_display yvo_decoder yvo_toolkit)

# ── isp ────────────────────────────────────────────────────────────────────────
# 스파이크 도구: 설정 덤프 / 1프레임 변환 / 패딩 검사
yvo_add_tool(yvo_isp_probe isp/isp_probe.cpp
  REQUIRES yvo_isp
  LIBS yvo_isp yvo_decoder
  OPTIONAL yvo_opencv=YVO_HAVE_OPENCV)

# 성능 측정: 전체 디코드 + ISP 전처리 (+ NHCW 전치). yvo_preproc_bench와 플래그 이름을 맞췄다.
yvo_add_tool(yvo_isp_bench isp/isp_bench.cpp
  REQUIRES yvo_isp
  LIBS yvo_isp yvo_decoder yvo_toolkit
  OPTIONAL yvo_opencv=YVO_HAVE_OPENCV)

# 전체 파이프라인(ISP 전처리 + NPU 추론). yvo_pipeline과 같은 플래그·RESULT_CSV 형식을 쓴다.
# yvo_pipeline(01_src/pipeline/main.cpp)은 건드리지 않는다 — 이건 별도 도구다.
yvo_add_tool(yvo_isp_pipeline isp/isp_pipeline.cpp
  REQUIRES yvo_isp yvo_hailo yvo_opencv
  LIBS yvo_isp yvo_hailo yvo_decoder yvo_opencv yvo_toolkit)

# ── decode ─────────────────────────────────────────────────────────────────────
# 확인 도구: descriptor 덤프, de-SAND 비교, 디코드 성능
yvo_add_tool(yvo_drm_dump decode/drm_dump.cpp
  LIBS yvo_decoder yvo_toolkit)
