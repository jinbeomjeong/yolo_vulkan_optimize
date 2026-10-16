# 표시 통합 e2e — 미계수 인터럽트 처리 시간 측정 (2026-10-16)

헤드라인 회차 [`../v3d1150_quiet_perfgov_20261016`](../v3d1150_quiet_perfgov_20261016/summary.md)(Q)의 계수 한계를 직접 잰 회차다.

배경은 이렇다. 이 커널은 `IRQ_TIME_ACCOUNTING`이 꺼져 있어, CPU가 유휴일 때 처리된 하드웨어 인터럽트(hardirq)와 softirq 시간이 프로세스 CPU 계수에서 빠진다.

논문 8차 심사 M1은 이 몫이 Q의 감소폭(2.444 → 2.044 ms/f, 0.400 ms/f)을 뒤집을 수 있는지 물었다. 민감도 계산상 회당 약 7 µs가 넘으면 뒤집히는 조건이었다.

결정: M1을 인터럽트 처리 시간 측정으로 보완한다 (사용자 결정, 2026-10-16).

## 1. 측정 조건

- 날짜: 2026-10-16 15:59~16:12 (KST)
- 하드웨어·코드·설정: Q와 같다.
  - Raspberry Pi Compute Module 5 Rev 1.0, Hailo-8, PCIe Gen3 x1, V3D 1150 MHz
  - 바이너리 md5 `3dda5b0f…`(커밋 `2f2d3d0`)
  - b2/s4, 900프레임
  - **CPU 거버너 performance**(측정 동안만, 끝난 뒤 ondemand로 복원)
- 벤치: 커밋 `45eb03e`의 `display_bench.sh` `IRQ_TRACE=1` 모드
  - 명령: `IRQ_TRACE=1 RUNS=3 FRAMES=900 OUT_DIR=07_results/display_e2e/runs/v3d1150_irqtime_20261016 bash 06_tools/bench/display_bench.sh`
  - 콘솔 전문(거버너 설정·원복 줄 포함)은 [`raw/_session/bench_stdout.log`](raw/_session/bench_stdout.log)에 있다.
- **기록 방법**
  - 실행마다 `sudo perf record -a -e irq:irq_handler_entry,irq:irq_handler_exit,irq:softirq_entry,irq:softirq_exit -D -1 --control fifo:…`를 띄운다.
  - 앱이 측정 창 시작·끝에 FIFO로 enable/disable을 보내므로, 기록 창이 앱 CPU 창과 같다.
  - `perf script` 텍스트(`raw/<run>/irq_script.txt.gz`)를 [`06_tools/python/irq_time_window.py`](../../../../06_tools/python/irq_time_window.py)로 집계했다.
    - CPU별로 entry·exit를 스택으로 짝짓는다.
    - 중첩은 self time으로 뺀다.
    - **끊긴 태스크가 pid 0(idle)인 몫을 미계수 몫**(`uncounted_ms_per_frame`)으로 분류한다.
  - 실행별 원값: [`results_raw_irq.csv`](results_raw_irq.csv), `raw/<run>/irq.json`
- 신뢰성
  - 13개 `hailort.log`(12 실행 + 세션) 모두 warning 0, error 0
  - 검출 3369(12회), 표시 드롭 0
  - 파싱 실패 0줄
  - 짝 안 맞는 이벤트는 12회 합계 1건
  - perf 이벤트 손실 기록 없음
- **주의**: 이 회차의 CPU ms(`results*.csv`)에는 tracepoint 오버헤드가 섞인다. 헤드라인에 쓰지 않는다. 인터럽트 시간만 쓴다.

## 2. 결과 (실행별 값의 3회 중앙값, ms/프레임)

| 설정 | hardirq 횟수/f | hardirq 시간 (idle 몫) | softirq 시간 (idle 몫) | hardirq 회당 평균 (µs) | **미계수 몫(idle hardirq + softirq)** [3회 범위] |
|---|---:|---:|---:|---:|---:|
| `reg_host_off` (기준) | 72.4 | 0.182 (0.172) | 0.040 (0.039) | 2.53 | **0.211** [0.210–0.218] |
| `disp_off` | 91.1 | 0.188 (0.183) | 0.041 (0.039) | 2.05 | 0.222 [0.221–0.225] |
| `disp_asap` (표시, 최대속도) | 126.0 | 0.269 (0.259) | 0.072 (0.070) | 2.13 | **0.328** [0.327–0.329] |
| `disp_source` (표시, 소스 30 fps) | 174.8 | 0.332 (0.321) | 0.080 (0.078) | 1.90 | 0.399 [0.396–0.404] |

- hardirq 시간의 95~97%가 CPU 유휴 중에 처리됐다. 이 회차의 부하에서는 인터럽트 대부분이 노는 코어를 깨운다.
- 프레임당 hardirq 횟수(72 / 91 / 126 / 175)는 Q의 `/proc/interrupts` 증가분(71 / 91 / 126 / 175)과 같다.
- 기준 → 최대속도 + 표시의 미계수 몫 증가 원천(`disp_asap` run2 대 `reg_host_off` run2)은 다음과 같다. 소계 기준이다.
  - IPI(함수 호출): 25.7 → 58.9회/f, idle 0.037 → 0.070 ms/f
  - arch_timer: 35.0 → 43.4회/f, 0.064 → 0.085 ms/f
  - v3d_core0: 1.0 → 7.0회/f, 0.003 → 0.014 ms/f
  - vc4 crtc: 0 → 1.5회/f, 0 → 0.011 ms/f(회당 중앙값 7 µs)
  - hvs·v3d_hub: 각 1~2회/f, 합 약 0.006 ms/f
  - Hailo MSI는 7.0회/f로 같다.
  - softirq 증가의 대부분은 RCU(16.0회/f)와 SCHED(12.8회/f)다.

## 3. 결론

- **미계수 몫의 증가(기준 → 최대속도 + 표시)는 프레임당 0.118 ms다.** Q의 감소폭 0.400 ms/f보다 작으므로 감소의 부호는 유지된다.
  - 미계수 몫을 양쪽에 더하면 2.444 + 0.211 = 2.655 → 2.044 + 0.328 = 2.372 ms/f다. 이때 **감소율은 약 −11%**(Q의 계수 범위 −16.4%)다.
  - 두 회차는 별개 실행이라 짝지은 범위는 내지 않는다. 미계수 몫의 3회 범위가 좁아(±2%) 중앙값으로 더했다.
- hardirq 회당 평균은 1.9~2.5 µs다. 8차 심사가 문제 삼은 7 µs보다 3배쯤 짧다.
- 소스 페이싱(`disp_source`)의 미계수 몫은 0.399 ms/f로 가장 크다. 같은 방식으로 더하면 2.122 + 0.399 = 2.521 ms/f이고, 기준(2.655) 대비 −5%다.
- 남은 비계수: softirq를 ksoftirqd가 아니라 IRQ 끝에서 처리한 몫은 위 값에 포함된다. 다만 idle에서 깨어나는 비용(C-state 복귀)과 tracepoint 자체 비용은 따로 가르지 않았다.
  - tracepoint 비용은 핸들러 시간을 늘리는 쪽이라, 위 미계수 몫은 **상한 쪽으로 기운다**.

## 4. 주의

- 기록은 root `perf`(tracepoint)로 했다. 시스템 설정(`perf_event_paranoid`)은 바꾸지 않았다.
- 이 회차의 CPU ms와 처리량은 tracepoint 오버헤드가 섞여 Q와 비교하지 않는다.
- 미계수 몫을 Q의 계수값에 더한 것은 회차가 다른 값의 합이다. 같은 실행에서 함께 재려면 샘플러와 perf가 같은 창 FIFO를 나눠 써야 하는데, 지금 벤치는 둘 중 하나만 띄운다.
- 벤치가 처음 쓴 파일 이름은 `irq_results_raw.csv`였다. 구조 규칙(`results_raw_*.csv`)에 맞춰 커밋 전에 `results_raw_irq.csv`로 바꾸었다. 내용은 그대로이고, 벤치의 파일명 상수도 같이 고쳤다.
