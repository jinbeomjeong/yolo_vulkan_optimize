# dma-buf 공유만 구간별로 뺀 비교 (2026-11-06)

세 CPU 절감 기법(① 하드웨어 디코더, ② GPU의 NHCW 직접 생성, ③ 결과 해석의 GPU 처리)은 단계 사이를 dma-buf 공유로 잇는다.
이 회차는 기법은 그대로 두고 **각 구간의 dma-buf 공유만 CPU 복사 하나로 바꿔**, dma-buf 자체가 줄인 CPU를 구간별로 잰다.
부모 실험은 [`technique_ablation`](../../summary.md)이다.

## 측정 조건

- 날짜: 2026-11-06 10:07~10:35(KST). 벤치 `06_tools/bench/technique_ablation_bench.sh`, `ONLY='^(base|decode_copy|input_copy|output_copy|all_copy)$' RUNS=3`, 3회 인터리브, 매 실행 전 냉각(Hailo ≤ 65 °C). 환경은 [`env_bench.txt`](env_bench.txt).
- Raspberry Pi CM5, Hailo-8 PCIe Gen3 x1, V3D 1150 MHz, HailoRT 4.24.0, YOLO11m, `video_h265.mp4`(1080p30, 1816프레임).
- **CPU 거버너 performance 고정**(측정 동안만, trap으로 ondemand 원복 확인).
- 바이너리 md5 `ef24741c7e5ca9457253f73e03aa73ca`. 측정 시점에 `--decode-copy`·`--output-copy`는 **커밋 전 작업 트리**의 코드였다(`env_bench.txt`의 `git HEAD: db83a90 (tracked changes)`).
- 공통: 확정 구성(배치 1/슬롯 1, 입력 페이싱 source, 입력 큐 1, 표시 on — Vulkan 합성 전체 화면, 기록 on, 디코더 1스레드).

| 설정 | dma-buf 대신 CPU 복사를 넣은 구간 | 복사 방식 |
|---|---|---|
| `base` | 없음 | — |
| `decode_copy` | ① 디코더 → GPU | CPU가 디코더 출력 dma-buf를 mmap으로 읽어(READ sync) cached dma-heap SAND 버퍼로 plane당 memcpy 1회, GPU는 그 버퍼를 읽는다 |
| `input_copy` | ② GPU → NPU | GPU fence 뒤 CPU가 V3D 출력을 cached dma-heap 버퍼로 memcpy, NPU는 그 버퍼를 읽는다 |
| `output_copy` | ③ NPU → GPU | NPU가 원시 출력 6텐서를 cached dma-heap 버퍼에 쓰고, CPU가 GPU 후처리 입력 버퍼로 memcpy |
| `all_copy` | ①②③ 모두 | 위 셋을 함께 |

300프레임 기능 확인에서 다섯 설정의 `detections.jsonl`은 바이트 동일했다(asap, 표시 off, 정식 측정 아님).

## 결과 (3회 중앙값, 괄호는 최소~최대)

| 설정 | `fps` | 응용 CPU ms/처리 프레임 | base 대비 | 도착→결과 평균 ms | 도착→화면 평균 ms | 처리 프레임 |
|---|---:|---:|---:|---:|---:|---:|
| `base` | 24.31 | **1.437** (1.422~1.489) | — | 53.69 | 70.94 | 1472 |
| `decode_copy` | 24.25 | **2.610** (2.535~2.644) | +1.17 | 53.33 | 69.59 | 1469 |
| `input_copy` | 24.12 | **2.300** (2.273~2.519) | +0.86 | 54.12 | 70.64 | 1461 |
| `output_copy` | 24.26 | **1.763** (1.738~1.764) | +0.33 | 53.79 | 71.01 | 1469 |
| `all_copy` | 23.96 | **3.802** (3.770~3.951) | +2.37 | 53.43 | 71.27 | 1452 |

| 복사 단독 시간 (각 실행 stdout의 통계 줄, 3회 평균값) | ms/복사 | 복사 횟수 |
|---|---:|---|
| ① 디코더 → GPU | 0.87~0.94 | 도착 프레임마다(1816) |
| ② GPU → NPU | 0.87~1.10 | 처리 프레임마다(약 1460) |
| ③ NPU → GPU | 0.29 | 처리 프레임마다(약 1469) |

## 결론

- **dma-buf 공유 자체가 줄이는 CPU는 구간당 0.3~1.2 ms/처리 프레임이다.** 셋을 모두 빼도 +2.37 ms다.
- 구간별 증가는 복사 단독 시간과 맞는다. ①은 도착 프레임마다 복사하므로 처리 프레임 기준으로 약 0.91 × 1816/1469 ≈ 1.13 ms이고, 측정 증가 1.17 ms와 같다. 세 구간의 증가 합(1.17 + 0.86 + 0.33 = 2.36)은 `all_copy`의 증가(2.37)와 같아 **서로 더해진다**.
- 상대 크기: 적용 전 대비 적용 후의 감소(47.86 → 1.49 ms, [`demo_before_after`](../../../demo_before_after/summary.md))에 비하면 약 5%다. 반면 확정 구성의 남은 CPU(1.44 ms)에 비하면 **약 2.6배(+165%)**가 된다. 즉 큰 절감은 일을 하드웨어로 넘기고 형식을 맞춘 데서 나오고, dma-buf는 마지막 남은 복사를 없애 CPU를 데이터 경로에서 완전히 뺀다.
- 처리 FPS는 거의 같다(최대 −0.35 fps, `all_copy`). 처리 프레임이 줄어든 만큼 입력 큐 버림이 약간 늘었다(18.9% → 20.1%). 도착→결과와 도착→화면 지연 평균은 1 ms 안에서 같다.
- 복사는 넓은 단위 memcpy라 싸다. 비캐시 GPU 메모리를 CPU가 바이트 단위로 읽는 경우(NHWC 전치, 13.38 ms)와는 다른 비용이다.

## 신뢰성

- `hailort.log` warning·error 15회 모두 0건, 종료 코드 0, `all_ok` 전부 1. `get_throttled` 현재 비트 0(`0xe0000`은 과거 기록 비트).
- 시작·끝 SoC 68.6 → 70.8 °C, Hailo는 실행마다 65 °C 이하로 냉각한 뒤 시작했다.
- `input_copy` 3회 중 1회(1.103 ms/복사, CPU 2.519)가 다른 2회(0.88)보다 높다. `all_copy`의 3회차 입력 복사(1.114)도 같다. 중앙값에는 영향이 없다.

## 주의·남은 일

- 검출 수(`dets`)는 처리 프레임 수가 달라 설정마다 다르다. 기능 동일성은 위의 asap 300프레임 바이트 비교로 확인했다.
- `--decode-copy`·`--output-copy` 코드는 측정 뒤 커밋해야 한다(측정 바이너리 md5는 위).
- `--output-copy`의 셰이더 입력 배리어는 기존 `Foreign` acquire 그대로다(검증 레이어 errors 0). Vulkan 쪽 검토는 남았다.
- 벤치 CSV는 `input copy` 통계만 열로 집계한다. `decode copy`·`output copy` 통계는 `raw/<설정>_run<N>/stdout.log`에만 있다.
