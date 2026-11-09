# Model Zoo 모델 파이프라인 회차 — yolov11m (기본 모델, 대조용)

[`model_zoo_small`](../model_zoo_small/summary.md) 후속 회차의 yolov11m (기본 모델, 대조용) 부분입니다. **세 모델(n·s·m)의 비교와 결론은 [`../model_zoo_pipe_s/summary.md`](../model_zoo_pipe_s/summary.md)에 있습니다.** 조건(바이너리 dc447c5 빌드 md5 070a8687, 확정 구성 O2 표시 off, 거버너 performance, 냉각, 3회 반복)도 그 문서 §1과 같습니다.

| 항목 | 3회 중앙값 (최소~최대) |
|---|---|
| 날짜 | 2026-11-09 |
| HEF | `03_models/yolov11m.hef`(sha256 앞 16자리 d5716578f0d18b01) |
| `fps` | 24.42 (24.41~24.42) |
| 입력 버림 | 18.6% |
| `e2e_arr` 중앙값 ms | 58.20 (58.19~58.24) |
| `npu_ms_mean` | 38.34 (38.33~38.36) |
| `cpu_ms_per_frame` | 0.728 (0.726~0.771) |
| 검출 / 실행 | 7,698 (7,696~7,711) |
| 1회차 대 2회차 검출 대조 | 공통 1,144프레임에서 IoU 0.5·0.7·0.9 모두 1.0000 |
| `hailort.log` warning·error | 0 |

출처는 [`results.csv`](results.csv), [`results_raw.csv`](results_raw.csv), [`env_bench.txt`](env_bench.txt)입니다. 벤치의 `env_bench.txt`에는 HEF 경로가 남지 않습니다. 모델 확인 방법은 비교 문서 §1에 적었습니다.
