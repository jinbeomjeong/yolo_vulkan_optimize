# 06_tools — 빌드 밖 자동화·도구

CMake 빌드에 들어가지 않는 것들이다. 빌드 대상 코드는 [`01_src/`](../01_src/) 에 있다.

2026-09-21 폴더 재배치에서 만들었다. 이전에는 `08_python/`(파이썬)과 `09_scripts/`(셸)로
나뉘어 있었는데, 파이썬 파일 2개가 서로 다른 폴더에 흩어져 경계가 무너진 상태였다.
실제 구분선은 "언어"가 아니라 **"빌드에 들어가는가"** 다.

## `bench/` — 측정 자동화 셸 7개

| 스크립트 | 무엇을 재나 | 출력 |
|---|---|---|
| `op_point_bench.sh` | 운용점 스윕 (batch·slots) — 최종 결론의 근거 | `07_results/op_point/` |
| `display_bench.sh` | 화면 표시 포함 종단간 | `07_results/display_e2e/` |
| `pipeline_bench.sh` | 회귀 기준선 (고정 9설정) | `07_results/pipeline_ffmpeg_vulkan_hailo/` |
| `decode_bench.sh` | 디코드 경로 비교 | `07_results/decode_benchmark/` |
| `backpressure_bench.sh` | 백프레셔 결함 전후 비교 | `07_results/raw_runner_backpressure/` |
| `isp_bench.sh` | ISP 전처리 단독 | `07_results/isp_resize/` |
| `isp_pipeline_bench.sh` | ISP 포함 전체 파이프라인 | `07_results/isp_resize/pipeline/` |

공통 환경변수: `RUNS`(기본 3), `FRAMES`, `ONLY`(설정 id grep), `APPEND=1`, `BIN`, `OUT_DIR`,
`COOL_C`/`HOT_C`/`COOL_MAX_S`. 측정 규약은 [`.claude/skills/measure`](../.claude/skills/measure/SKILL.md).

> ⚠ `pipeline_bench.sh` 의 `configs` 형식(`id|mode|layout|batch|slots` 5필드)은 **바꾸지 마라.**
> 다른 스크립트들이 이걸 대조군 기준으로 쓴다.

### 벤치 락

일곱 스크립트 모두 시작할 때 `build/.bench_running` 에 자기 PID를 쓰고 `trap` 으로 지운다.
[`.claude/hooks/npu_guard.sh`](../.claude/hooks/npu_guard.sh) 가 이 파일을 보고 **측정 중에는
빌드·테스트 명령을 막는다.** 측정 중에 다른 작업이 돌면 CPU 시간 측정이 **에러 없이 조용히
오염되기** 때문이다. 훅이 PID 생존 여부를 보므로 비정상 종료로 남은 락은 자동 무효가 된다.

## `python/` — 집계·생성

| 파일 | 하는 일 | 실행 환경 |
|---|---|---|
| `check_links.py` | 저장소 모든 `.md` 의 마크다운 링크와 `파일:행` 범위가 실재하는지 검사. 깨진 링크가 있으면 종료 코드 1 | 시스템 python3 |
| `compare_detections.py` | 두 `detections.jsonl` 비교(IoU 매칭·좌표 오차). 판정 규칙이 [`01_src/common/detection_match.hpp`](../01_src/common/detection_match.hpp) 와 **의도적으로 동일**하다 | conda `hailo-yolo` |
| `gen_glyph_font_8x16.py` | `01_src/common/glyph_font_8x16.hpp` 생성. **빌드에 관여하지 않는다** — 산출물은 커밋된 헤더이고 이 파일은 그 상수를 어떻게 얻었는지의 재현 기록이다 | 시스템 python3 |

## `probes/` — 일회성 프로브 소스

빌드 대상이 아니다. 조사할 때 한 번 컴파일해 쓰고 버린 것들인데, 결론의 근거라서 남겨 두었다.
결과 폴더 안에 소스를 두지 않는다는 규칙에 따라 여기로 모았다.

| 파일 | 어디서 썼나 |
|---|---|
| `dmabuf_modifier_dump.c` | 컴포지터가 광고하는 dma-buf format/modifier 덤프 — [`07_results/display_vulkan_hdmi/`](../07_results/display_vulkan_hdmi/summary.md) |
| `ongoing_probe.cpp` | 백프레셔 계수 관찰 — [`07_results/raw_runner_backpressure/`](../07_results/raw_runner_backpressure/summary.md) |
