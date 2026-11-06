# `input_copy`의 NPU 지연이 짧게 나오는 이유 — 측정 구간 시작점 효과

후속 2번(2026-11-06 승인) 회차입니다. [`technique_ablation`](../../summary.md) 표에서 `input_copy`의 `npu_ms_mean`이 `base`보다 0.5 ms 짧았습니다(32.17 대 32.69). 당시에는 "캐시 힙 입력이 원인으로 보이나 확인하지 않았다"고 적어 두었는데, 이 회차에서 그 원인을 가립니다.

**결론부터 적습니다.**
- 원인은 입력 메모리 종류가 아닙니다. `npu_ms`라는 열이 **어디서부터 재기 시작하는지**가 원인입니다.
- `--input-copy`의 memcpy(약 0.85 ms)는 측정 구간 앞으로 옮겨질 뿐입니다. 그래서 FPS, 도착 기준 지연, GPU 제출 → NPU 완료 시간은 두 설정이 같습니다.

## 1. 측정 조건

| 항목 | 값 |
|---|---|
| 날짜 | 2026-11-07 00:13~01:11 |
| 보드 | Raspberry Pi CM5, Hailo-8 PCIe 8.0 GT/s x1(Gen3), V3D 1150 MHz, 커널 6.18.50+rpt-rpi-2712 |
| HailoRT·HEF | 4.24.0 · `03_models/yolov11m.hef` |
| 입력 | `04_data/videos/video_h265.mp4` 영상 전체(도착 1,816 / 처리 약 1,479프레임) |
| 공통 인자 | `--mode dmabuf --decoder hw --batch 1 --slots 1 --input-pace source --input-queue 1 --layout nhcw --postproc gpu --nms gpu --display off` |
| 설정 | `base`, `input_copy`(`--input-copy`, CPU가 V3D 출력을 캐시 dma-heap `vidbuf_cached`로 복사) |
| 바이너리 | new = HEAD 3d78e59 빌드(md5 af63c101). old = c97c274 이전(md5 ef24741c, 사전 매핑을 lseek 크기로 해서 매핑 캐시가 빗나감) |
| 반복 | 회차마다 new base → new ic → old base → old ic 순서로 돌리고, 이를 3회 반복 |
| 거버너 | 측정 동안만 performance로 두고, trap으로 ondemand로 되돌림 |
| 냉각 | 매 실행 전 Hailo ≤ 65 °C, SoC ≤ 75 °C |
| 신뢰성 | `hailort.log` warning·error 0, 12회 모두 |

환경은 [`env_bench.txt`](env_bench.txt)에, 실행별 출력은 `raw/<바이너리>_<설정>_run<N>/`에 있습니다. 드라이버 스크립트는 [`evidence/run_npu_lat.sh`](evidence/run_npu_lat.sh)와 [`evidence/run_gap.sh`](evidence/run_gap.sh)입니다.

표시를 끈 조건이라 기존 회차의 표시 on 값(32.6 ms)보다 NPU 지연이 크게 나옵니다. 이 차이도 아래 §3과 같은 원리로 설명됩니다.

## 2. 표 — 3회 중앙값 (최소~최대)

출처: [`results.csv`](results.csv), 실행별 값은 [`results_raw.csv`](results_raw.csv)

| 바이너리·설정 | `fps` | `npu_ms_mean` | `npu_ms_median` | `e2e_arr` 중앙값 ms | `cpu_ms_per_frame` |
|---|---|---|---|---|---|
| new `base` | 24.43 | 38.341 (38.339~38.353) | 38.527 | 58.189 (58.167~58.193) | 0.711 (0.703~0.714) |
| new `input_copy` | 24.43 | 37.474 (37.470~37.477) | 37.662 | 58.227 (58.171~58.290) | 1.584 (1.580~1.598) |
| old `base` | 24.43 | 38.291 (38.279~38.293) | 38.474 | 58.236 (58.135~58.244) | 0.800 (0.797~0.814) |
| old `input_copy` | 24.43 | 37.396 (37.169~37.405) | 37.606 | 58.233 (58.155~58.260) | 1.685 (1.679~1.920) |

- `input_copy`에서 `base`를 뺀 NPU 지연은 new에서 −0.867 ms, old에서 −0.895 ms입니다. 차이는 재현되고, 표시를 끈 조건에서는 오히려 더 큽니다.
- FPS와 도착 기준 e2e는 네 행이 같습니다. 차이는 0.05 ms 이하입니다.
- `input copy:` 줄에 따르면 복사 비용은 프레임당 평균 0.876 ms입니다(memcpy 0.852 + 캐시 동기화 0.023).

## 3. 가설별 판정

**(b) 프레임마다 반복된 dma-buf 매핑 — 기각**
- 변경 전 바이너리에서는 V3D 입력 BO만 매핑 캐시에서 빗나갔습니다. 크기는 1,245,184 B이고, heap 입력은 1,228,800 B입니다([`evidence/bufinfo_*.txt`](evidence/bufinfo_base.txt)).
- strace로 센 매핑 ioctl은 처리 244프레임에서 old `base`가 프레임당 7회, old `input_copy`가 6회입니다. new 바이너리는 두 설정 모두 0회입니다.
- 한 번의 비용은 map 7 µs, unmap 6 µs입니다. 출처는 [`evidence/strace_map_counts.txt`](evidence/strace_map_counts.txt)이며, strace 원문(2.4~3.6 MB)은 보관하지 않았습니다.
- 그런데 new와 old 바이너리 모두에서 `input_copy` − `base` 차이가 똑같이 남습니다. 따라서 매핑은 원인이 아닙니다.

**(a) NPU가 읽는 메모리 종류(heap 대 V3D BO)가 H2D DMA를 바꾼다 — 기각**
- 이 가설이 맞다면 장치 주기가 짧아져 FPS가 올라야 합니다.
- 그러나 FPS가 소수 둘째 자리까지 같습니다. 아래 X도 40.36 대 40.37로 같습니다.

**(c) 측정 구간 시작점 효과 — 채택**
- `npu_ms`는 `HailoRawRunner::submit`의 `infer_async` 바로 앞(`01_src/hailo/hailo_raw_runner.cpp:297`)에서 시작해, 마지막 출력 콜백(`:322`)에서 끝납니다.
- `--input-copy`의 memcpy는 GPU fence 뒤, NPU submit 앞에서 돕니다(`01_src/pipeline/main.cpp:1183`). 즉 이 구간 밖입니다.
- 프레임 기록(`--frame-log`, 900프레임 중 seq ≥ 30인 703쌍)에서 두 양을 계산했습니다. 출처는 [`evidence/gap_analysis.txt`](evidence/gap_analysis.txt)이고, 원본은 `evidence/gap_src_*_frame_log.csv`입니다.
  - X = 그 프레임의 GPU 제출 → NPU 완료
  - G = 이전 프레임의 NPU 완료 → 이 프레임의 GPU 제출

| 설정 | X 중앙값 ms | G 중앙값 ms | X+G 중앙값 ms |
|---|---|---|---|
| `base` | 40.36 | 0.56 | 40.92 |
| `input_copy` | 40.37 | 0.56 | 40.92 |

- X+G = 40.92 ms는 1/FPS(40.93 ms)와 같습니다. [`op_point`](../../../op_point/summary.md)의 T(1) ≈ 41 ms와도 맞습니다.
- 즉 slots 1에서는 NPU 장치 주기가 완료 시각을 정합니다. 콜백 뒤 호스트가 쓰는 시간(GPU 전처리, 복사)은 그만큼 다음 프레임의 `npu_ms`에서 빠집니다.
- 이는 `base`의 G 구간별로도 확인됩니다. G가 클수록 X가 줄어듭니다.

| G 구간 | X 중앙값 ms |
|---|---|
| 0~1 ms | 40.36 |
| 1~2 ms | 39.23 |
| 2~4 ms | 38.13 |
| 4~8 ms | 36.66 |

- 콜백 뒤 장치가 실제로 무엇을 하는지는 보지 않았습니다. 프레임 경계의 컨텍스트 전환이나 config 재적재일 것으로 추정합니다.

## 4. 결론

- `input_copy`의 "NPU 지연 −0.5~0.9 ms"는 이득이 아닙니다. 처리량과 도착 기준 지연은 같고, 응용 CPU만 프레임당 약 0.87 ms 늘어납니다.
- GPU → NPU 버퍼를 dma-heap으로 바꿔도(V3D가 heap을 import하는 방식) NPU 주기가 같으므로 처리량·지연 이득은 기대하지 않습니다.
- **해석 규칙:** 이 파이프라인(slots 1)의 `npu_ms`는 "장치 주기 − (콜백 → 다음 제출 간격)"입니다. 따라서 앞단 호스트 일이 늘면 짧아 보입니다.
  - 앞단 일이 다른 설정끼리(표시 on/off, 복사 옵션) NPU를 비교할 때는 `npu_ms`가 아니라 FPS나 X(GPU 제출 → NPU 완료)를 씁니다.
- 덤으로 확인된 것: 표시 off에서 사전 매핑 크기 맞춤(c97c274)의 앱 CPU 효과는 old 0.800 → new 0.711 ms/f입니다. 표시 on 확정 구성 비교는 [`cpu_trim`](../cpu_trim/summary.md)에 있습니다.

## 5. 주의·남은 일

- 3단계의 asap 비교 2건은 `--input-queue`가 source 전용이라 실행되지 않았습니다(rc=2). 결론에는 영향이 없어 다시 돌리지 않았습니다.
- 표시를 켠 배치 1에서 NPU 지연이 짧아지는 현상(35.2 → 31.1 ms, `08_learn/08_method.md` 미해결 표)도 같은 원리로 설명될 가능성이 큽니다. 다만 표시 on으로 X와 G를 직접 재지는 않았습니다.
- 콜백 뒤 NPU의 실제 작업을 확정하려면 `measure-fw-actions`로 프레임 경계를 보면 됩니다. 필요할 때만 합니다.
- 기존 문서에서 `npu_ms`를 앞단 일이 다른 설정끼리 비교한 서술은 integrity-auditor로 점검합니다.
