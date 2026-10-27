# 06_tools — 빌드 밖 자동화·도구

CMake 빌드에 들어가지 않는 것들이다. 빌드 대상 코드는 [`01_src/`](../01_src/) 에 있다
(빌드되는 진단·측정 도구 13개는 [`01_src/tools/`](../01_src/tools/), 실행파일은 `build/bin/`).

2026-09-21 폴더 재배치에서 만들었다. 이전에는 `08_python/`(파이썬)과 `09_scripts/`(셸)로
나뉘어 있었는데, 파이썬 파일 2개가 서로 다른 폴더에 흩어져 경계가 무너진 상태였다.
실제 구분선은 "언어"가 아니라 **"빌드에 들어가는가"** 다.

## `bench/` — 측정 자동화 셸 12개

| 스크립트 | 무엇을 재나 | 출력 |
|---|---|---|
| `op_point_bench.sh` | 운용점 스윕 (batch·slots) — 최종 결론의 근거 | `07_results/op_point/` |
| `display_bench.sh` | 화면 표시 포함 종단간. `results_raw.csv` 70열(RESULT_CSV 22 + RESULT_CSV2 뒤 33 + 스크립트 4 + labwc CPU 2 + 시스템 전체 CPU sys_* 9, 실행마다 `labwc_stat_{pre,post}.txt`·`syscpu{,_idle}.{json,csv}`; `IDLE_S` 기본 10 s, 0=건너뜀) — 48·53·59·61열 옛 CSV 와 `APPEND=1` 불가 | `07_results/display_e2e/` `IRQ_TRACE=1`(2026-10-16): 샘플러 대신 `sudo perf record`(irq·softirq tracepoint)를 앱 CPU 창에 맞춰 띄우고 `irq_time_window.py` 로 미계수 인터럽트 시간을 `results_raw_irq.csv` 에 남긴다(sudo 는 `SUDO_ASKPASS` 있으면 `-A`, 없으면 `-n`; 이 모드의 CPU ms 는 tracepoint 오버헤드가 섞여 헤드라인에 쓰지 않는다). `DRYRUN=1` = 명령만 출력 |
| `pipeline_bench.sh` | 회귀 기준선 (고정 9설정) | `07_results/pipeline_ffmpeg_vulkan_hailo/` |
| `gst_pipeline_bench.sh` | 앱(`yvo_pipeline` dmabuf/NHCW) 대 GStreamer 연구 경로(`yvo_gst_pipeline --sink hailo`), batch 1/4/8 × 같은 slots. `results_raw.csv`·`results.csv` 앞 26열은 `pipeline_bench.sh` 와 같고 뒤에 도구 열이 붙는다(gst `decode_ms_mean` = nan, NaN 안전 중앙값). 바이너리는 `APP_BIN`·`GST_BIN`, `DETS=1` 검출 비교(`evidence/compare_b<B>`), `DRYRUN=1` 명령줄만 출력 | `07_results/gst_pipeline/` |
| `dec_threads_bench.sh` | 앱의 FFmpeg HEVC 프레임 스레드 수(`YVO_DEC_THREADS`) — nhcw b8/s16 스레드 스윕(1·2·3·4·auto) + 모드별 auto 대 1 회귀 A/B(ptr·nhwc·b1·b4·gpu 후처리·지연·표시). 설정 형식 `id|threads|frames|args`(auto = 변수 제거). `results_raw.csv` 65열(스크립트 10 + RESULT_CSV 22 + RESULT_CSV2 33), NaN 안전 중앙값. 기본 매 실행 전 냉각(`COOL_EVERY=1`), 표시 실행은 `wlr-randr` HDMI-A-1 확인 + `timeout -s KILL 90`, `SKIP_DISPLAY=1`, `DRYRUN=1` | `07_results/dec_threads/` |
| `display_planes_bench.sh` | HVS 평면 표시 정식 측정(실험 B): Vulkan 합성(`--display-backend wayland`) / `wayland-planes` / `kms` × `source`·`asap`, b2/s4(`SLOTS=N`으로 모든 설정의 `--slots` 변경, 기본 4), 900f×3 인터리브, 모든 설정 `--display-warmup 30`. `results_raw.csv` 102열(스크립트 11 + RESULT_CSV 22 + RESULT_CSV2 33 + RESULT_CSV3 16(decode-to-screen `d2s_*` 4 포함) + slot wait 2·kms·labwc CPU·V3D 엔진별 20. 2026-10-08 보관본은 96열). 실행 전 HDMI-A-1·labwc master 확인, kms 는 `loginctl activate` 로 tty ↔ 데스크톱 전환(sudo 없음, 세션 id 는 런타임 탐색). `SKIP_KMS=1`, `SESSIONS_ONLY=1`, `DRYRUN=1`, `CHECK_ONLY=1` | `07_results/display_planes_e2e/` |
| `nms_bench.sh` | GPU NMS 정식 A/B(실험 A, 표시 off) — `MODE=perf`: `b2s4`·`b8s16` × `--nms cpu|gpu` 900f×3 인터리브, `results_raw.csv` 77열(스크립트 10 + RESULT_CSV 22 + RESULT_CSV2 33 + stdout·V3D 12: gpu nms median/max, fallback 원인, cpu NMS µs, slot hold, e2e median, V3D CSD 잡·ms/frame). `MODE=verify`: b2s4 전체 영상 `--nms verify`·cpu·gpu 덤프 대조(`[A2] PASS/FAIL`). ⚠ verify 는 CPU·GPU 억제끼리만 비교해(같은 `yolo_decode.comp` 후보를 받는다) **디코딩 쪽 회귀를 못 잡는다** — `BASELINE=<보관된 verify 폴더>` 로 세 `detections.jsonl` 을 기준선과 바이트 대조한다(영상·HEF·실효 프레임이 다르면 NPU 를 쓰기 전에 FAIL). `VERDICT_ONLY=1 OUT_DIR=<기존 verify 폴더>` 는 NPU 없이 판정만 다시 한다. IoU 매칭(`compare_detections.py`)은 점수만 바뀐 것을 못 보므로 판정은 바이트 비교다. 모든 플래그 명시, `DRYRUN=1`·`CHECK_ONLY=1` | `07_results/gpu_nms_e2e/` |
| `decode_bench.sh` | 디코드 경로 비교 | `07_results/decode_benchmark/` |
| `backpressure_bench.sh` | 백프레셔 결함 전후 비교 | `07_results/raw_runner_backpressure/` |
| `input_pace_bench.sh` | 입력 페이싱 + leaky 입력 큐 — b1s1·b2s3·b2s4 × {asap, `--input-pace source --input-queue` 1·2·3}, 도착 900프레임 × 3, 표시 off. `results_raw.csv` 47열(RESULT_CSV 22 + RESULT_CSV4 21 + 스크립트 4, asap 행의 CSV4 는 nan). 페이싱 행의 FPS·`cpu_ms_per_frame` 은 asap 과 비교 금지. `DRYRUN=1`·`CHECK_ONLY=1`. `FRAME_LOG=1` 이면 실행마다 `raw/<설정>_run<N>/frame_log.csv`(`--frame-log`, 분석은 `frame_log_stats.py`). **`DISPLAY_RUN=1`** = 페이싱 + 표시 on(큐 1): `dvk_b1s1`·`dwlp_b1s2`·`dwlp_b1s3`·`dvk_b2s4`·`dwlp_b2s6`(Vulkan 합성·wayland-planes, kms 없음), 실행마다 HDMI-A-1·labwc master 확인 + `timeout -s KILL $TIMEOUT_S`(90), `WARMUP`(30), raw 87열(+ RESULT_CSV5 6 + 표시 필드 31 + 스크립트 7), 기본 출력 `runs/display_on` | `07_results/input_pacing/` |
| `isp_bench.sh` | ISP 전처리 단독 | `07_results/isp_resize/` |
| `isp_pipeline_bench.sh` | ISP 포함 전체 파이프라인 | `07_results/isp_resize/runs/pipeline/` |

공통 환경변수: `RUNS`(기본 3), `FRAMES`, `ONLY`(설정 id grep), `APPEND=1`, `BIN`, `OUT_DIR`,
`COOL_C`/`HOT_C`/`COOL_MAX_S`, `BUILD_DIR`(기본 `build`)·`BIN_DIR`(기본 `$BUILD_DIR/bin`)·`PY_HAILO`
(셋 다 [`lib/common.sh`](bench/lib/common.sh) 한 곳에서 정의), `CHECK_ONLY=1`(바이너리 존재만 확인하고 락·냉각 전에 종료).
`lib/common.sh` 에는 벤치들이 각자 복사해 쓰던 헬퍼도 모여 있다(2026-10-12) — 냉각 `cool_before_run`, 환경 `env_git_head`·`env_bin`·`gpu_stats_snap`, CSV 행 `nan_list`·`arg_val`·`result_csv12`(옮길 때 CSV 바이트 불변을 보관 raw 로 대조했다). 측정 규약은 [`.claude/skills/measure`](../.claude/skills/measure/SKILL.md).

`OUT_DIR` 이 이미 있고 비어 있지 않으면 **`FORCE=1` 없이는 에러로 멈춘다**(`guard_out_dir`, 보관 결과 보호 — 기본 `OUT_DIR` 은 모두 보관본이 있으니 새 측정은 새 `OUT_DIR` 을 주거나 `APPEND=1` 이라도 `FORCE=1` 을 함께 준다).
출력 레이아웃은 [`STRUCTURE.md`](../STRUCTURE.md) §3: 실행 1회 = `raw/<설정>_run<N>/`(`stdout.log` 전문 | `run.log` 발췌 | `hailort.log`), 온도 프로브의 세션 `hailort.log` 는 `raw/_session/`, 1회 덤프는 `samples/`, 회차는 `runs/<회차>/`.

> ⚠ `pipeline_bench.sh` 의 `configs` 형식(`id|mode|layout|batch|slots` 5필드)은 **바꾸지 마라.**
> 다른 스크립트들이 이걸 대조군 기준으로 쓴다.

### 벤치 락

12개 스크립트 모두 시작할 때 `build/.bench_running` 에 자기 PID를 쓰고 `trap` 으로 지운다.
[`.claude/hooks/npu_guard.sh`](../.claude/hooks/npu_guard.sh) 가 이 파일을 보고 **측정 중에는
빌드·테스트 명령을 막는다.** 측정 중에 다른 작업이 돌면 CPU 시간 측정이 **에러 없이 조용히
오염되기** 때문이다. 훅이 PID 생존 여부를 보므로 비정상 종료로 남은 락은 자동 무효가 된다.

## `python/` — 집계·생성

| 파일 | 하는 일 | 실행 환경 |
|---|---|---|
| `check_links.py` | 저장소 모든 `.md` 의 마크다운 링크와 `파일:행` 범위가 실재하는지 검사. 깨진 링크가 있으면 종료 코드 1. `--stale STRUCTURE.md` 는 옮겨진 옛 경로가 남았는지 본다 — 종료 코드는 **실제로 남은 옛 경로만** 반영하고, `guessed`·`manual check` 줄은 사람이 읽고 판단하는 경고다(오탐이 흔하다). `--lines` 는 살아 있는 문서(`07_results` 제외 — 결과 문서의 인용은 측정 당시 코드를 가리키는 역사 기록이다)의 `파일:행` 인용이 같은 줄의 백틱 식별자를 ±5줄 안에서 가리키는지 본다. **경고 전용**(종료 코드 늘 0). 같은 줄에 식별자가 있는 인용만 내용까지 검증되고, 1~4줄의 작은 밀림은 ±5 창 안이라 놓친다. `.graphml`(`09_diagrams/`)의 설명문·노드·엣지 라벨 인용도 문장 단위로 보고 별도 요약 줄 `--- graphml: …` 로 낸다(위치는 `<파일>:<XML 줄>`) | 시스템 python3 |
| `check_structure.py` | 폴더 배치 규칙 검사(최상위·코드 계층·include DAG·07_results 레이아웃). 규칙표는 [`structure_rules.toml`](structure_rules.toml), 정본 문서는 [`STRUCTURE.md`](../STRUCTURE.md). 위반이 있으면 종료 코드 1. Write 훅 `placement_guard` 가 이 파일의 `classify_new_path()` 를 쓴다 | 시스템 python3 |
| `relocate.py` | `STRUCTURE.md` §6 매핑표로 `git mv`(`--apply-moves`)와 참조 재작성(`--apply-refs`)을 한다. 먼저 `--dry-run` | 시스템 python3 |
| `compare_detections.py` | 두 `detections.jsonl` 비교(IoU 매칭·좌표 오차). 판정 규칙이 [`01_src/ref/detection_match.hpp`](../01_src/ref/detection_match.hpp) 와 **의도적으로 동일**하다 | conda `hailo-yolo` |
| `gen_glyph_font_8x16.py` | `01_src/ref/glyph_font_8x16.hpp` 생성. **빌드에 관여하지 않는다** — 산출물은 커밋된 헤더이고 이 파일은 그 상수를 어떻게 얻었는지의 재현 기록이다 | 시스템 python3 |
| `hef_bytes.py` | HEF protobuf(hef.proto v4.24.0)를 직접 디코드해 컨텍스트별 config(CCW) 바이트와 edge layer 프레임당 전송 크기를 나열한다. 출력은 표준 출력. [`candidate_reduction_survey`](../07_results/candidate_reduction_survey/summary.md) 의 `evidence/hef_proto_bytes.txt` 생성기 | 시스템 python3 |
| `hef_byte_model.py` | HEF 바이트 회계(`hef_bytes` 사용) + fw-actions 컨텍스트 시간·T(B)에 PCIe 전송 모델 적합. 입력 `<evidence 폴더(fwactions_probe_b{1,8}.json)> <hef>`, 출력은 표준 출력 또는 `--out` — **입력 폴더 안은 거부한다**(보관된 결과를 덮어쓰지 않게). `evidence/pcie_byte_model_fit.txt` §1~9 생성기. §5 T(B)·§8 Gen2 측정값과 제목의 HEF 이름이 하드코딩돼 있어 **다른 HEF에 쓰면 그 절은 맞지 않는다** | 시스템 python3 |
| `frame_log_stats.py` | `yvo_pipeline --frame-log` CSV(프레임별 도착·디코드 끝·제출·NPU 완료·후처리 완료·완료 시각) 분석 — 지연(도착 기준·디코드 기준), 구간 분해, 완료 간격 분포·군집, NPU 완료 시각으로 추정한 배치 구성과 짝 위치별 지연, 상위 지연 프레임(시작·끝 표시). 10열 로그(표시 on)면 `present_ms` 로 도착 → 화면(a2p)·디코드 → 화면·표시 구간·present 간격(vsync 배수, `--vsync-ms`)도 낸다. 여러 회차를 파일별·합산으로, `--top`·`--startup`·`--csv`·`--json`. 사용례 [`input_pacing/runs/frame_log`](../07_results/input_pacing/runs/frame_log/summary.md) | 시스템 python3 |
| `sys_cpu_window.py` | 앱의 CPU 측정 창(`YVO_PERF_CTL`/`YVO_PERF_ACK` FIFO, `tool_util.hpp` `perf_window`)에 맞춰 시스템 전체 CPU 를 잰다 — `/proc/stat`(tick)·`/proc/schedstat`(ns)·`/proc/interrupts`·전 프로세스 `/proc/<pid>/stat` 스냅샷 차분, 분류(app·labwc·커널 스레드·기타). `--duration S` 는 기준선. `display_bench.sh` 가 실행마다 부른다. 이 커널은 `IRQ_TIME_ACCOUNTING` 이 꺼져 idle 중 hardirq 는 잡히지 않는다 | 시스템 python3 |
| `irq_time_window.py` | `display_bench.sh` `IRQ_TRACE=1` 가 앱 CPU 창에 맞춰 `sudo perf record`(irq·softirq tracepoint)로 남긴 `perf script` 텍스트를 읽어 hardirq/softirq **self time** 을 CPU 별로 짝지어 집계한다. 끊긴 태스크가 pid 0(idle)인 몫 = tick 계수(`IRQ_TIME_ACCOUNTING` 꺼짐)에서 빠지는 **미계수 몫**(`uncounted_ms_per_frame`). JSON + CSV 한 줄, 표준 라이브러리만(`/usr/bin/python3`) |
| `fw_actions_breakdown.py` | `hailortcli run2 … measure-fw-actions` JSON 을 컨텍스트별 구간·액션 종류별 시간·전환 간격으로 나누고 배치 크기에 대해 T(B) = a + b·B 를 적합(기준 24.8 + 16.2·B). 액션 이름 해석은 추정. JSON 의 `fps` 는 중간 표본이라 주기는 stdout fps 로. 사용례 [`hef_context_profile`](../07_results/hef_context_profile/summary.md) | 시스템 python3 |
| `hef_context_layers.py` | 다중 컨텍스트 HEF 의 컨텍스트(`dynamic_0..3`)가 계산하는 ONNX 블록(`/model.N/...`)을 대응시킨다. HEF 파일(과 선택적으로 `measure-fw-actions` JSON)만 읽고 장치는 열지 않는다. 출력 `context_layers.json`·`.txt`. [`hef_context_profile`](../07_results/hef_context_profile/summary.md) 의 `evidence/context_layers.*` 생성기 | 시스템 python3 (stdlib만) |
| `paper/` | 논문(`10_paper/`) 산출물 생성 스크립트 11개(배치 규칙 [`STRUCTURE.md`](../STRUCTURE.md) §2.7): `paper_style.py`(그림 공용 스타일)·`fig01`·`fig02`·`fig04`·`fig05`·`fig07`·`fig08`·`fig09`(그림 하나당 스크립트 하나, `07_results` 를 읽고 `10_paper/figures/` 에 씀)·`build_figures.py`(그림 전체 재생성)·`build_docx.py`(`paper_ko.md` + `10_paper/template/reference.docx` → `10_paper/build/paper_ko.docx`)·`make_reference_docx.py`(서식 견본 생성기 → `template/reference.docx`). 그림은 [`paper-figures`](../.claude/skills/paper-figures/SKILL.md) 스킬 절차를 따른다 | 그림 스크립트: conda `hailo-yolo`(matplotlib). `build_docx.py`·`make_reference_docx.py`: conda `python-dev`(패키지·버전 목록은 [`STRUCTURE.md`](../STRUCTURE.md) §2.7) |
| `tests/` | 위 도구와 `check_links`·`check_structure`·`compare_detections`·`paper/` 의 stdlib 단위 테스트 8개 파일 141개 — `python3 -m unittest discover -s 06_tools/python/tests`(1초 미만, 하드웨어 불필요). 시스템 python3 로는 124개가 돌고 17개는 건너뛴다: `test_paper_style.py`(9개, matplotlib 필요)는 `hailo-yolo` 에서, `test_build_docx.py`(8개, python-docx·lxml·pandoc ≥ 3 필요)는 `python-dev` 에서만 돈다 | 시스템 python3 (`test_paper_style.py` 는 `hailo-yolo`, `test_build_docx.py` 는 `python-dev`) |

## `probes/` — 일회성 프로브 소스

빌드 대상이 아니다. 조사할 때 한 번 컴파일해 쓰고 버린 것들인데, 결론의 근거라서 남겨 두었다.
결과 폴더 안에 소스를 두지 않는다는 규칙에 따라 여기로 모았다.

| 파일 | 어디서 썼나 |
|---|---|
| `dmabuf_modifier_dump.c` | 컴포지터가 지원 목록으로 알려 주는 dma-buf format/modifier 덤프 — [`07_results/display_vulkan_hdmi/`](../07_results/display_vulkan_hdmi/summary.md) |
| `ongoing_probe.cpp` | 백프레셔 계수 관찰 — [`07_results/raw_runner_backpressure/`](../07_results/raw_runner_backpressure/summary.md) |
| `memasym_run.sh`, `memasym_aggregate.py` | `yvo_postproc_probe --probe memasym` 정식 실행(설정 5종 × 3회, 벤치 락 사용)과 집계 — `07_results/memory_asymmetry/` ([`evidence/medians.txt`](../07_results/memory_asymmetry/evidence/medians.txt)). `bench/`가 아니라 여기 둔 이유: 새 `bench/*.sh`는 `npu_guard` 목록(C10)도 바꿔야 한다 |
| `memasym_trace.sh`, `memasym_trace_summary.py` | 위 프로브를 tracefs kprobe 아래에서 돌려 PTE prot(AttrIndx)와 sync·V3D 제출 경로의 CPU 캐시 유지를 기록(root 필요, 인스턴스·kprobe는 trap으로 해제) — [`evidence/trace/summary.txt`](../07_results/memory_asymmetry/evidence/trace/summary.txt) |
| `hvs_probe/` | KMS 직접·Wayland 전체화면에서 NV12 SAND 영상 평면 + AB24 박스 평면을 HVS가 블렌딩하는지 비교 — [`kms_vs_wayland.md`](../07_results/display_vulkan_hdmi/notes/kms_vs_wayland.md). 원래 `yvo_pipeline_port` 저장소(git 밖)에서 그 저장소 헤더(`common/`, `display/overlay_reference.hpp`, `pipeline/ffmpeg_drm_decoder`)에 맞춰 빌드했다. 메모의 `tools/hvs_probe/` 가 이 폴더다. 원시 증거는 [`07_results/display_vulkan_hdmi/evidence/hvs_review/`](../07_results/display_vulkan_hdmi/evidence/hvs_review/README.md)(검증 로그는 같은 실험의 `verification/hvs_review/`) |
