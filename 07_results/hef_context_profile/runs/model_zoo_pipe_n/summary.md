# Model Zoo 모델 파이프라인 회차 — yolov11n

[`model_zoo_small`](../model_zoo_small/summary.md) 후속 회차의 yolov11n 부분입니다. **세 모델(n·s·m)의 비교와 결론은 [`../model_zoo_pipe_s/summary.md`](../model_zoo_pipe_s/summary.md)에 있습니다.** 조건(바이너리 dc447c5 빌드 md5 070a8687, 확정 구성 O2 표시 off, 거버너 performance, 냉각, 3회 반복)도 그 문서 §1과 같습니다.

| 항목 | 3회 중앙값 (최소~최대) |
|---|---|
| 날짜 | 2026-11-09 |
| HEF | `03_models/yolov11n.hef`(sha256 앞 16자리 9b6fb370b660b331) |
| `fps` | 30.00 (30.00~30.00) |
| 입력 버림 | 0% |
| `e2e_arr` 중앙값 ms | 13.76 (13.75~13.86) |
| `npu_ms_mean` | 7.85 (7.84~7.86) |
| `cpu_ms_per_frame` | 0.671 (0.657~0.747) |
| 검출 / 실행 | 5,943 (3회 같음) |
| m 대조(1회차, IoU 0.5 일치율) | 0.3757 — [`evidence/compare_vs_m_run1.txt`](evidence/compare_vs_m_run1.txt) |
| `hailort.log` warning·error | 0 |

출처는 [`results.csv`](results.csv), [`results_raw.csv`](results_raw.csv), [`env_bench.txt`](env_bench.txt)입니다. 벤치의 `env_bench.txt`에는 HEF 경로가 남지 않습니다. 모델 확인 방법은 비교 문서 §1에 적었습니다.
