# 논문 그림 등록부

`10_paper/paper_ko.md`에 들어가는 그림의 목록이다. 규칙의 정본은 `STRUCTURE.md` §2.7이고, 절차는 `.claude/skills/paper-figures/SKILL.md`에 있다.

- 그림 파일(`figNN_<slug>.svg|png`)은 `06_tools/python/paper/`의 스크립트로만 만든다. 다만 다이어그램은 yEd 내보내기(가상 화면 런처 `yed-auto export`, `09_diagrams/README.md` §내보내기)로 만든다.
- 수치는 `07_results` 정본에서만 가져온다. 행마다 생성 스크립트와 원본 경로를 적고, 원본 경로는 저장소 루트 기준으로 쓴다.
- 조건 약어:
  - **구**: 960 MHz · 디코더 자동 스레드 · CPU NMS
  - **신**: 1150 MHz · 디코더 1스레드 · GPU NMS
- 결정: `10_paper/figures` 구조를 도입한다 (사용자 결정, 2026-10-13).

| 번호 | 파일 | 생성 스크립트 | 원본 | 측정 조건 | 논문 절 |
|---|---|---|---|---|---|
| 1 | `fig01_npu_batch_fit.{svg,png}` | `06_tools/python/paper/fig01_npu_batch_fit.py` | `07_results/op_point/runs/latency_gen3_cooled/benchmark_raw.csv`, `07_results/npu_benchmark/runs/{gen2,gen3}/benchmark_b{1,2,4,8,16}.csv`, `07_results/npu_benchmark/runs/gen3_cooled/benchmark_b16.csv` | `hailortcli benchmark` 랜덤 입력, HailoRT 4.24.0. (a) Gen3 x1 냉각 세션 3회 (b) Gen2·Gen3 x1 각 1회. 파이프라인 아님 | §3.3 제약 2 |
| 2 | `fig02_npu_context_cost.{svg,png}` | `06_tools/python/paper/fig02_npu_context_cost.py` | `07_results/hef_context_profile/evidence/context_layers.json`, `07_results/hef_context_profile/evidence/breakdown.json` | (a) HEF 메타데이터 분석(장치 미사용, `06_tools/python/hef_context_layers.py`), fw-actions `raw/b1_run1`로 교차 확인 (b) Gen3 x1, `hailortcli run2 measure-fw-actions`, B = 1·2·8 × 3회. JSON fps 미사용 | §3.3 제약 2 |
| 4 | `fig04_op_point_grid.{svg,png}` | `06_tools/python/paper/fig04_op_point_grid.py` | `07_results/op_point/runs/grid_gpu_postproc/results.csv` | 신(디코더 스레드는 빌드 시각 근거), 표시 off, Gen3 x1, 900f × 3 | §6.6.1 (4) |
| 5 | `fig05_input_queue.{svg,png}` | `06_tools/python/paper/fig05_input_queue.py` | `07_results/input_pacing/results_raw.csv` | 신, 표시 off, `--input-pace source`, 900 도착 × 3 | §6.6.3 (2) |
| 7 | `fig07_arrival_to_present.{svg,png}` | `06_tools/python/paper/fig07_arrival_to_present.py` | `07_results/input_pacing/runs/display_on/raw/dvk_{b1s1,b2s4}_run*/frame_log.csv`, `07_results/input_pacing/runs/display_on/results_raw.csv` | 신, 표시 on(Vulkan 합성) 1080p60, 페이싱 큐 1, 900 도착 × 3 | §6.6.3 (6) |
| 8 | `fig08_display_paths.{svg,png}` | `06_tools/python/paper/fig08_display_paths.py` | `07_results/display_planes_e2e/runs/slots6/results.csv` | 신, 표시 on 1080p60, b2/s6, `--display-pace source`, 900f × 3 | §6.7.1 (2) |
| 9 | `fig09_thermal_soak.{svg,png}` | `06_tools/python/paper/fig09_thermal_soak.py` | `07_results/thermal_soak/evidence/soak_{b2s4,b8s16}.csv`, `07_results/thermal_soak/evidence/soak_{b2s4,b8s16}_events.txt` | 신, 표시 off, MOT20-05 반복, 추가 냉각 없음, 40분 × 1회 | §7.4 (1) |

### 예정 (번호만 비워 둠)

- 그림 3 `fig03_pipeline` — §4.1 블록도. 원본 `09_diagrams/pipeline_paper.graphml`을 `yed-auto export`로 내보낸다(사용자 동석 불필요).
- 그림 6 `fig06_o2_breakdown` — §6.6.3 (5) O2 지연 분해. 깨움 코드 측정 회차가 나온 뒤 `input_pacing/runs/frame_log`과 전후 비교로 그린다.
