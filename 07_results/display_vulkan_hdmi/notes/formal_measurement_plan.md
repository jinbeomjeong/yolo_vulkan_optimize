# 실험 계획: GPU NMS · HVS 평면 표시 정식 측정

작성 2026-09-29. **계획만 있고 아직 실행하지 않았다.** 이 문서의 수치는 계획의 근거로 쓴 기존 값이며 새 측정이 아니다.
[`07_results/README.md`](../../README.md) "주의·남은 일"의 "GPU NMS·HVS 평면 표시 정식 측정" 행을 닫기 위한 계획이다.

## 1. 배경

- GPU NMS(`efficient_nms.comp`, `--nms gpu`, 2026-09-22)와 HVS 평면 스캔아웃 표시(`--display-backend wayland-planes|kms`)는
  구현·검증(비트 일치, debugfs 실표시 판정)만 되어 있다.
- 현재 공개 수치(glass 101.20 ms, CPU 1.842 ms/f 등, [`display_e2e/summary.md`](../../display_e2e/summary.md))는 모두
  **CPU 억제 + Vulkan 합성 경로**의 값이다. 루트 `README.md`도 "GPU NMS로 바꾼 뒤의 재측정은 아직 없다"고 적는다.
- 목표: measure 스킬 규약(설정당 3회·중앙값, 냉각 게이트, `hailort.log` 경고 판정)으로 두 기능을 기존 경로와 같은 조건에서 비교한다.

## 2. 선행 조건

1. EEPROM 전후 재측정(`pipeline_ffmpeg_vulkan_hailo/runs/bootloader_*`)과 그 커밋이 끝난 뒤 시작한다.
   부트로더 적용 **후** 환경에서 재야 새 기준선과 비교할 수 있다.
2. 벤치가 도는 동안(`build/.bench_running`)에는 계측 보강 코드를 빌드하지 않는다(`npu_guard`).
3. 실험 B의 kms 설정은 데스크톱 세션을 비활성화하므로 사용자가 자리에 있을 때 돌린다.

## 3. 이미 있는 계측 (조사 결과)

| 항목 | 위치 | 비고 |
|---|---|---|
| `--nms gpu\|cpu\|verify` | `01_src/pipeline/main.cpp:266-272`, 기본 gpu `:176` | `--postproc gpu`에서만 허용(`:508-509`) |
| decode·NMS 분리 타임스탬프 | `01_src/vulkan/yolo_postproc.cpp:24-26`, `:625-627` | pass1 = ts2−ts1, nms = ts4−ts3. 배리어 구간(ts2→ts3)은 어느 값에도 안 들어간다 |
| 종료 요약 | `main.cpp:1078-1144` | `gpu nms ms mean/median/p90/max`, `nms fallback N frames (capacity X, iou guard Y)`, glass·present 간격 |
| `RESULT_CSV2` | `main.cpp:1177-1188` | `nms,gpu_nms_mean,gpu_nms_p90,nms_fallback,nms_verify_mismatch` + 평면 6필드. **median·fallback 원인별 값은 stdout에만** 있다 |
| CPU fallback | `01_src/pipeline/gpu_postproc_stage.cpp:209-228` | 용량(후보 > 1024)·IoU 보호대로 그 프레임만 CPU 억제 |
| 평면 표시 시각 | `wayland_plane_scanout.cpp:351-367`, `kms_scanout.cpp:876-889` | wp_presentation / page-flip 하드웨어 시각 |
| 벤치 템플릿 | `06_tools/bench/display_bench.sh` | CSV+CSV2 59열 전체 수집, 설정은 `id\|args`. **`--nms`·`--display-backend` 설정은 없다** |
| V3D 부하 | `/sys/class/drm/renderD128/device/gpu_stats`, `/proc/<pid>/fdinfo`의 `drm-engine-*` | 파서 선례 `06_tools/probes/hvs_probe/probe_common.hpp:57-100` |

비공식 선행 값(인용 금지, 계획 판단용):
- GPU NMS median 0.053 ms, fallback 0, 검출 3369 — 다른 트리에서 1회 실행(`../evidence/hvs_review/baseline/disp_off_stdout.log`).
- 평면 경로 300프레임 1회: wayland-planes glass 98.25 ms / CPU 1.738, kms 99.47 ms / 1.702 (`11_logs/2026-09-22_*`, git 밖).

## 4. 실험 A — GPU NMS

- **폴더**: `07_results/gpu_nms_e2e/` (스윕형, 색인 행 함께 추가)
- **성격**: 없어지는 CPU 억제 비용은 3~11 µs/frame([`gpu_postprocess/summary.md`](../../gpu_postprocess/summary.md))으로,
  getrusage CPU의 실행 간 흔들림(~0.1 ms)보다 작다. **"CPU 절감 증명"이 아니라 "회귀 없음 + GPU 비용" 판정**으로 설계한다.

### A1. 파이프라인 A/B
- 조건: 표시 off, `--mode dmabuf --layout nhcw --postproc gpu`, 900프레임 × 3회, 인터리브(run 바깥·설정 안쪽).
- 설정: `b2s4_cpu`, `b2s4_gpu`, `b8s16_cpu`, `b8s16_gpu` (`--nms cpu|gpu`).
- 지표: FPS, CPU ms/f, e2e, slot hold, 검출 수, `gpu nms` median/p90, fallback 원인별 건수.
- 보조: `perf stat -e task-clock`, 또는 `perf record`로 `cpu_nms*` 심볼 비중.
- 판정: FPS·e2e 범위 겹침(회귀 없음), 검출 수 cpu = gpu, fallback 0 또는 원인 설명.

### A2. 정확도
- `--nms verify` 1,816프레임 → `nms_verify_mismatch 0`. `compare_detections.py`로 cpu/gpu 검출 대조.
- verify 모드는 GPU·CPU를 둘 다 돌리므로 **성능값으로 쓰지 않는다.**

### A3. NMS 단독 GPU 비용 (코드 수정 필요)
- `01_src/tools/vulkan/postproc_probe.cpp`의 P3-⑥ 연속 스트림 루프(`:431-445`)에 `enable_nms` + `last_nms_ms()` 수집을 넣는다.
  현재 이 프로브에는 NMS가 없다. → 구현(2026-10-08): fixture 출력의 `cmdbuf_ms` 의미를 바꾸지 않으려고 P3-⑥ 루프에 넣지 않고 같은 루프의 별도 모드 `--probe nms`로 만들었다(최악 경우는 `01_src/testkit/nms_harness.hpp`로 옮긴 `NmsHarness` 공유).
- `04_data/raw_tensors/` 픽스처로 NPU 없이 median을 잰다. 최악 경우는 `05_tests/test_postproc.cpp`의 `NmsHarness`로 후보 1024개.
- 단발 submit/wait 값(`EfficientNmsBitExact`의 `gpu nms max`)은 DVFS로 부풀려지므로 쓰지 않는다.
- 담당 `vulkan-expert`, 끝나면 verify 스킬.

## 5. 실험 B — HVS 평면 표시

- **폴더**: `07_results/display_planes_e2e/` (스윕형, 색인 행 함께 추가) — 2026-10-08 실행 완료([`summary.md`](../../display_planes_e2e/summary.md)). b2/s4 에서 평면 경로가 슬롯 부족(slot wait 28~30 ms)이라 지연 비교는 slots ≥ 6 재측정으로 넘겼다
- **설정** (모두 b2/s4, `--postproc gpu --nms gpu`, 추론 on, `--display-fullscreen --display-sync matched`, 900프레임 × 3회):

| id | 백엔드 | pace |
|---|---|---|
| `vk_source` / `vk_asap` | Vulkan 합성(기본) | source / asap |
| `wlp_source` / `wlp_asap` | `wayland-planes` | source / asap |
| `kms_source` / `kms_asap` | `kms` | source / asap |

- **지표**: FPS, 프로세스 CPU ms/f, **labwc CPU**(`/proc/<pid>/stat` tick 전후), **V3D 총량·프로세스별**(gpu_stats / fdinfo),
  glass median/p90(stdout), presented/dropped, `hvs_underrun_delta`, `scanout_zero_copy`·`zero_copy_feedback`, overlay_ms.

### 해석할 때 주의할 점
1. **glass 끝점이 다르다.** 평면 경로는 하드웨어 표시 시각, Vulkan 경로는 `vkWaitForPresentKHR`를 폴링해 CPU가 알아챈 시각
   (`display_pipeline.cpp:336-363`)이라 Vulkan 쪽이 폴링 주기만큼 불리하다. summary에 명시한다.
2. **Vulkan 경로의 숨은 V3D TFU 비용.** 표시 on에서 TFU 약 3.4 ms/f가 생기는데 `convert_ms`에 안 잡힌다
   (`../evidence/hvs_review/baseline/disp_on_source_v3d_delta.txt`, 원인은 Mesa WSI 복사로 추정·미확인).
   V3D 비교는 반드시 gpu_stats·fdinfo로 한다.
3. **overlay_ms는 이름만 같다.** 평면 경로는 박스 그리기 + 지우기, Vulkan 경로는 `overlay_boxes.comp`만. convert_ms는 평면 경로에서 0.
4. **버퍼 보유 시간 차이.** 평면 경로는 release까지 버퍼를 1~2 vsync 더 쥔다(`plane_display.hpp:10-18`).

### 필요한 계측 보강 (코드 수정, 담당 `vulkan-expert`, kms 부분은 `cpp-expert` 검토) — 2026-10-08 완료: `--display-warmup N`, 요약의 `discarded`·`late frames`, `RESULT_CSV3`
- glass 통계에서 **앞 N프레임 제외** 옵션. kms 첫 커밋이 블로킹 모드셋이라서다(`kms_scanout.hpp:33-36`).
  `.claude/skills/measure/SKILL.md:27`은 "display_bench가 뺀다"고 적지만 스크립트에 그 로직이 없다 → 둘을 맞춘다. → 완료: 앱 `--display-warmup N`, `display_planes_bench.sh`는 WARMUP=30, display_bench.sh 는 0.
- `discarded_`를 ScanoutReport·요약·CSV에 노출(`plane_display.hpp:213`).
- 평면 경로에서 page-flip `seq` 간격 > 2로 늦은 프레임 집계(선례 `06_tools/probes/hvs_probe/hvs_kms_probe.cpp:872-881`).

### 스크립트
- `display_bench.sh`를 바탕으로 `06_tools/bench/display_planes_bench.sh`(A용은 `nms_bench.sh`)를 만든다. → 완료(2026-10-08, 96열, `SKIP_KMS`·`SESSIONS_ONLY`·`DRYRUN`).
  `display_bench.sh`의 설정은 바꾸지 않는다(기존 결과의 재현 기준).
- 실행마다 전후 gpu_stats, labwc·yvo_pipeline fdinfo/stat, 중간 debugfs `state`를 `raw/<설정>_run<N>/`에 저장
  (`../evidence/hvs_review/baseline/run_baseline.sh` 방식을 이식).
- **kms 세션 전환 절차**(`08_learn/07_video_display.md` §7.12):
  1. `loginctl activate <tty 세션>` → kms 설정 실행(실행 중 VT 전환 금지) → 정상 종료 확인
  2. `loginctl activate <데스크톱 세션>`
  3. 다음 Vulkan/wayland 설정 전에 `wlr-randr`에 HDMI-A-1이 있는지 확인, 없으면 중단
     (labwc가 headless면 Vulkan 경로가 `vkQueuePresentKHR`에서 무기한 멈춘다)
- 새 스크립트를 `npu_guard`가 찾는지 검사기 C10으로 확인한다.

## 6. 순서

1. 선행 조건(§2) 충족
2. 계측 보강 + 스크립트 2개(전문 에이전트) → verify 스킬(빌드 경고 0, ctest, 검증 레이어)
3. 실험 A → 4. 실험 B
5. 각 폴더 `summary.md`, 색인 행, 루트 `README.md`·논문 수치 반영 → `integrity-auditor`

## 7. 합격 기준

- 모든 실행 `hailort.log` 경고 0, 냉각 대기 기록, `check_structure.py`·`check_links.py` 0건.
- A: 검출 수 cpu = gpu, verify mismatch 0, FPS 회귀 없음.
- B: 매 실행 debugfs로 평면 스캔아웃 확인, `hvs_underrun_delta` 0, `plane_color_ok` 참.
