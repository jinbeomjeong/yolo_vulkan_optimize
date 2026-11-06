# 확정 구성(O2) — MOT20-05

후속 3번 회차의 MOT20-05 부분입니다. **두 영상의 비교와 결론은 [`../o2_mot20_02/summary.md`](../o2_mot20_02/summary.md)에 있습니다.** 조건(바이너리 af63c101, 인자, 거버너 performance, 냉각, 3회 반복)도 그 문서 §1과 같습니다.

| 항목 | 3회 중앙값 (최소~최대) |
|---|---|
| 날짜 | 2026-11-07 01:16~01:20 |
| 입력 | `04_data/videos/mot20_05_h265.mp4` (1080p 25 fps, 900프레임) |
| `fps` | 24.30 (24.28~24.30) |
| `cpu_ms_per_frame` | 1.388 (1.384~1.400) |
| 입력 버림 | 2.7% |
| `e2e_arr` 중앙값 ms | 57.31 (57.05~57.47) |
| 도착→화면 평균 ms | 73.87 (73.86~74.09) |
| 검출 | 23,427개(26.7 / 처리 프레임) |
| `hailort.log` warning·error | 0 |

출처는 [`results.csv`](results.csv), [`results_raw.csv`](results_raw.csv), [`env_bench.txt`](env_bench.txt)입니다.
