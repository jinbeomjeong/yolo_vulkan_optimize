# 현재 HEAD 표시 off 재측정 — 2026-10-11 리팩터 회귀 확인

`gpu_nms_e2e` 실험의 회차다. 2026-10-08 정식 기준선(이 실험의 본 폴더) 뒤에 들어간 변경 —
argmax 지름길(`c0e0810`)과 2026-10-11 개선 작업 4건(`a713233` 수명 결함, `3a7aa97` 테스트·`SlotExchange` 이동,
`bf0256f` LUT·BT.709·파이프라인 생성 중복 제거, `0b5a583` `run()` 분할·CLI 파서·RESULT_CSV 단일화) —
이 **처리량·지연·CPU·검출을 바꾸지 않았는지** 같은 스크립트·같은 명령으로 다시 쟀다.

## 1. 측정 조건

- 날짜 2026-10-12. git `85a0d9a`(바이너리 md5 `5d179d98…`, [`env_bench.txt`](env_bench.txt))
- Raspberry Pi 5 / BCM2712 + Hailo-8, PCIe **Gen3 x1**(8.0 GT/s), V3D **1150 MHz**(부트로더 2026-09-25), ARM 2.4 GHz,
  HailoRT 4.24.0, 디코더 1스레드(앱 기본값), 표시 off
- 입력 `04_data/videos/video_h265.mp4`(1080p30), HEF `03_models/yolov11m.hef`
- `06_tools/bench/nms_bench.sh` 를 수정 없이 `ONLY='_gpu$'`·`OUT_DIR=` 만 주고 실행:
  - `MODE=perf`: `b2s4_gpu`·`b8s16_gpu`(현재 기본값 `--nms gpu`), 900프레임 × 3회, 반복 바깥·설정 안쪽
  - `MODE=verify BASELINE=07_results/gpu_nms_e2e`: 전체 영상 1816프레임 cpu·gpu·verify 각 1회
- 냉각: 매 실행 전 Hailo 칩 65 °C 게이트. 실행 직전 최대 65.0 °C, 냉각 대기 합 perf 340초·verify 320초
- 신뢰성: 6회 모두 `hailort.log` 온도·클럭 경고 0, 에러 0(`unreliable_runs` 0)
- 콘솔 로그: `11_logs/2026-10-12_head_remeasure.log`(git 밖)

## 2. 성능 — 기준선(2026-10-08, 본 폴더 [`results.csv`](../../results.csv))과 비교

중앙값, 괄호는 3회 최소~최대.

| 설정 | 항목 | 기준선 2026-10-08 | 이번 2026-10-12 |
|---|---|---:|---:|
| b2s4_gpu | fps | 34.99 | **34.99** (34.99~34.99) |
| | cpu_ms_per_frame | 0.908 (0.862~0.980) | **0.915** (0.873~1.099) |
| | e2e_ms_mean | 139.60 | 139.54 |
| | e2e_ms_p90 | 166.72 | 166.75 |
| | npu_ms_mean | 106.41 | 106.40 |
| | gpu_ms_mean (전처리) | 1.810 | 1.807 |
| | gpu_post_mean (ms) | 0.5542 | **0.4685** (0.4682~0.4818) |
| | gpu_nms 중앙 (ms) | 0.0486 | 0.0485 |
| | v3d_csd_ms_per_frame | 2.2525 | 2.1626 |
| | 검출 | 3369 | 3369 |
| b8s16_gpu | fps | 51.50 | **51.51** (51.50~51.52) |
| | cpu_ms_per_frame | 0.869 (0.865~0.953) | **0.909** (0.889~0.951) |
| | e2e_ms_mean | 321.87 | 321.75 |
| | e2e_ms_p90 | 432.10 | 432.19 |
| | npu_ms_mean | 293.10 | 293.16 |
| | gpu_ms_mean (전처리) | 1.828 | 1.829 |
| | gpu_post_mean (ms) | 0.5897 | **0.5043** (0.5040~0.5051) |
| | gpu_nms 중앙 (ms) | 0.0500 | 0.0501 |
| | v3d_csd_ms_per_frame | 2.2703 | 2.1871 |
| | 검출 | 3369 | 3369 |

전 열은 [`results.csv`](results.csv), 실행별은 [`results_raw.csv`](results_raw.csv)·`raw/<설정>_run<N>/`.

## 3. 검출 — 기준선과 바이트 대조

`MODE=verify` 결과 `[A2] PASS`([`evidence/nms_verify_summary.txt`](evidence/nms_verify_summary.txt)).

| 실행 | 기준선 대비(정렬 후 바이트) |
|---|---|
| `cpu_b2s4` (1816프레임) | **identical** |
| `gpu_b2s4` | **identical** |
| `verify_b2s4` | **identical**, `nms_verify_mismatch` 0 |

검출 9457개, cpu 대 gpu IoU 0.9 매칭 1.0000. 기준선이 지름길 이전 기록이므로 이 대조는 지름길과 리팩터를 함께 덮는다
(verify 단독 PASS 는 디코딩 쪽 회귀를 못 잡는다 — [`argmax_shortcut_mot20_05`](../argmax_shortcut_mot20_05/summary.md) §1).

## 4. 결론

- **회귀 없음.** FPS·e2e 평균/p90·NPU·GPU 전처리·GPU NMS·검출이 기준선과 같다.
  CPU 중앙값은 b2s4 +0.007, b8s16 +0.040 ms/f 이지만 **두 곳 모두 3회 범위가 기준선 범위와 겹친다**(getrusage 잡음 약 0.1 ms) — 회귀로 보지 않는다.
- **GPU 후처리는 −15.5%(b2s4 0.554 → 0.469 ms)·−14.5%(b8s16 0.590 → 0.504 ms), V3D CSD −0.08~0.09 ms/f.**
  argmax 지름길의 효과가 기본 영상에서도 보인다(mot20_05 회차의 −32~−35% 보다 작은 것은 후보가 적어 ⑤ emit 비중이 작기 때문).
  FPS 로는 주장하지 않는다 — 처리량은 NPU(PCIe)가 정한다.
- 따라서 README·색인의 **표시 off 현재 기본값 기준선(b8s16 51.50 fps · CPU 0.869 ms/f, b2s4 34.99 · 0.908)은 그대로 유효하다.**

## 5. 주의·남은 일

- 측정 도중 드러난 벤치 결함: `OUT_DIR` 을 **상대 경로**로 주면 verify 실행이 실행 폴더로 `cd` 한 뒤 같은 상대 경로를
  `--dump-dir` 로 넘겨 `samples/<id>/07_results/…/samples/<id>/` 에 중첩 저장했고, 첫 판정은 `missing` 으로 FAIL 했다.
  측정값 자체는 정상이라 파일을 `samples/<id>/` 로 옮기고 `VERDICT_ONLY=1` 로 다시 판정했다(위 PASS, NPU 재사용 없음).
  첫 판정 요약은 덮어썼다. 11개 벤치 스크립트 모두 `OUT_DIR` 을 `realpath -m` 으로 절대화해 고쳤다(같은 커밋).
- 이번 회차는 `--nms gpu` 두 설정만 쟀다(`--nms cpu` 대조군은 2026-10-08 본 폴더에 있다). b1s1 지연, 표시 on 은 재지 않았다 —
  표시 on 헤드라인(29.91 FPS · glass 101.20 ms · CPU 2.18 ms/f)은 여전히 960 MHz·자동 5스레드·CPU NMS 조건 값이다.
- 결정: 표시 on 재측정은 이번 범위에서 제외 (사용자 결정, 2026-10-12).
