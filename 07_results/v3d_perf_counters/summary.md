# V3D 성능 카운터(VK_KHR_performance_query) — 동작 특성 프로브 (E1~E3)

- 측정일: 2026-09-29
- 환경: Raspberry Pi 5 (BCM2712), V3D 7.1.10.2 / V3DV **Mesa 26.2.2**, kernel 6.18.50+rpt-rpi-2712, 레지스트리 Vulkan 1.4.309.
  V3D 클럭 960 MHz 고정(측정 중 `measure_clock v3d` 74표본 전부 960.0 MHz), SoC 64~70 °C, `throttled=0x0`.
  데스크톱(labwc, HDMI-A-1 연결) 켜 둔 상태. **NPU 미사용**(`/dev/hailo0` 사용자 0, `hailort_service` 데몬만 상주). 상세 `env_bench.txt`
- 도구: `build/bin/yvo_perf_counters` (`01_src/tools/vulkan/perf_counters.cpp`) `--exp granularity|contention|overhead`, `--load <sec>`
- 작업: `02_shaders/diag/heap_sum.comp` 4 MiB 바이트 합 1 dispatch(256 WG × 64) ≈ 4.9 ms. 매 반복 GPU 합 == CPU 합 확인(불일치 **0**)
- 방식: 연속 submit→fence 스트림, 설정마다 워밍업 30회 뒤 E1·E2 100회 / E3 200회. 중앙값·p90. 검증 레이어 **끈 채** 측정(게이트 실행은 따로)
- 카운터: 1 pass 21개(compute 15 + 그래픽 대조 6, **이름으로 런타임 열거에서 찾음**). 열거 원본 `evidence/counters.{txt,csv}`
- 커맨드 버퍼: `[ts0 TOP][perf begin]{[compute→compute 배리어][dispatch]}×k[perf end][compute→host 배리어][ts1 BOTTOM]`.
  V3DV 카운터는 93개 모두 scope COMMAND 라 begin/end 를 첫·끝 명령에 둘 의무가 없다(VUID-vkCmdBeginQuery-queryPool-03224 비해당)
- 목적: 성능 카운터의 미확인 3항목 — ① 무엇 단위로 모이는가 ② 다른 V3D 사용자가 섞이는가 ③ 쿼리 자체 비용

## 결과

### E1 분해 단위 — 한 CB 에 dispatch k개 vs CB k개 (각 1 dispatch)

| 설정 | gpu_ns 중앙값 (p90) | 배수 | compute-active-cycles | QPU-active | core-memory-reads | 실효 클럭 cycle-count/gpu_ns | 커널 잡/iter (csd·cpu) |
|---|---|---|---|---|---|---|---|
| onecb_k1 | 4 916 550 (4 942 344) | 1.000 | 4 686 183 | **2 110 720** | 16 396 | 960.6 MHz | 1 · 2 |
| kcb_k1 | 4 914 734 (4 946 824) | 1.000 | 4 684 522 | 2 110 720 | 16 396 | 960.6 | 1 · 2 |
| onecb_k2 | 9 822 302 (9 950 392) | 1.998 | 9 366 270 (×1.999) | **4 221 440 (×2.000)** | 32 793 (×2.000) | 960.3 | 2 · 2 |
| kcb_k2 | 9 823 922 (9 972 344) | 1.998 | 9 369 408 | 4 221 440 | 32 793 | 960.7 | 2 · 4 |
| onecb_k4 | 19 663 928 (20 153 218) | 4.000 | 18 736 617 (×3.998) | **8 442 880 (×4.000)** | 65 584 (×4.000) | 960.3 | 4 · 2 |
| kcb_k4 | 19 638 826 (20 111 551) | 3.994 | 18 732 843 | 8 442 880 | 65 584 | 960.7 | 4 · 8 |
| onecb_k8 | 39 328 346 (39 791 870) | 7.999 | 37 469 909 (×7.996) | **16 885 760 (×8.000)** | 131 172 (×8.000) | 960.2 | 8 · 2 |
| kcb_k8 | 39 290 325 (39 894 538) | 7.991 | 37 466 396 | 16 885 760 | 131 172 | 960.7 | 8 · 16 |

- kcb 값은 k개 쿼리의 합. onecb_k 와 kcb_k 가 모든 카운터에서 같다.
- 커널 잡(`gpu_stats` 차분): **dispatch 1개 = CSD 잡 1개**, 쿼리가 있어도 잡이 늘지 않는다. timestamp 는 **CPU 큐 잡**으로 돈다(쓰기 1회 = cpu 잡 1개).
- 실효 클럭 960.2~960.7 MHz = `measure_clock v3d` 960.0 MHz. compute-active/gpu_ns 는 953 MHz(구간의 99.3%가 CSD 활성).

### E2 혼입 — 고정 작업(4 MiB × 1) 100회, 조건별

| 조건 | 다른 클라이언트 bin/render 잡 (100회 동안) | QPU-active = 2 110 720 인 회수 | QPU-active 중앙값 (min) | compute-active 중앙값 (p90) | 그래픽 카운터 >0 회수 | CLE-render 중앙값 | gpu_ns 중앙값 (p90) |
|---|---|---|---|---|---|---|---|
| A 유휴 | 4 / 4 | **97** (3회 미달) | 2 110 720 (316 745) | 4 684 660 (4 692 009) | 0 | 0 | 4 922 382 (4 938 390) |
| B vkcube(MAILBOX) | **3 190 / 3 189** (+tfu 3 190) | **0** (100회 전부 미달) | **88 700** (244) | **163 686** (245 357) | **83** | **107 338** | 10 195 230 (10 412 973) |
| C 두 번째 프로세스 `--load 60` | 5 / 5 (+상대 csd 101) | **96** (4회 미달) | 2 110 720 (483 014) | 4 683 744 (4 688 499) | 0 | 0 | **9 750 670** (9 771 743) |
| A2 유휴(재측정) | 0 / 0 | **100** | 2 110 720 (2 110 720) | 4 684 229 (4 688 583) | 0 | 0 | 4 913 798 (4 920 547) |

- 결정적인 카운터(QPU-active, core-memory-reads, TMU-active)는 방해가 없으면 **반복마다 비트 단위로 같다**(A2: 표준편차 0).
- 미달 회수가 조건마다 **다른 클라이언트의 bin/render 잡 수를 따라간다**(A 4→3, C 5→4, A2 0→0, B 3190→100/100). E1 에서도 쿼리 창이 길수록(k↑) 데스크톱 잡이 끼어 미달이 늘었다(2→15회).
- B: vkcube 의 렌더 작업이 우리 쿼리에 **들어오고**(CLE-render·TLB·FEP >0, 83회), 동시에 우리 compute 값은 **정상의 ~4%로 잘린다**.
- C: 상대 프로세스의 CSD 잡은 **우리 값에 섞이지 않았다**(중앙값 동일, 그래픽 0). 대신 **timestamp 구간이 2배**(9.75 ms) — CSD 큐에서 상대 잡이 사이에 끼고, timestamp 가 CPU 잡이라 그 대기를 포함한다.

### E3 오버헤드 — 같은 작업 200회

| 설정 | gpu_ns 중앙값 (p90) | host submit→fence µs 중앙값 (p90) | `vkGetQueryPoolResults` perf µs | ts µs | 커널 잡/iter |
|---|---|---|---|---|---|
| (i) timestamp 만 | 4 912 317 (4 923 856) | 4 930.8 (4 944.4) | — | 2.9 | csd 1 · cpu 2 · cache_clean 1 |
| (ii) + 카운터 32개(1 pass) | 4 919 632 (4 931 776) | 4 950.3 (4 980.7) | 5.4 | 3.7 | 동일 |
| (iii) + 카운터 4개 | 4 919 993 (4 932 614) | 4 950.6 (4 974.5) | 2.9 | 3.9 | 동일 |
| (i) 반복(표류 대조) | 4 918 076 (4 943 257) | 4 946.3 (4 978.0) | — | 4.2 | 동일 |
| 전체 93개(3 pass) 1회 | — | **14 796.0 (제출 3회)** | 7.3 | — | — |

판정 규칙: (ii)−(i) 가 (i)의 잡음(p90−중앙값)보다 작으면 무시 가능.

| 지표 | (ii)−(i) | (iii)−(i) | (i반복)−(i) | 잡음(i) | 규칙 판정 |
|---|---|---|---|---|---|
| gpu_ns | +7.3 µs (+0.15%) | +7.7 µs (+0.16%) | +5.8 µs | 11.5 µs | **무시 가능** |
| host submit→fence | +19.5 µs (+0.40%) | +19.9 µs (+0.40%) | **+15.5 µs** | 13.6 µs | 규칙상 초과. 단 **같은 설정 반복만으로 +15.5 µs 표류** |

(i)과 (i반복)의 평균(4 938.6 µs)을 기준으로 하면 (ii) +11.8 µs, (iii) +12.1 µs 로 잡음 13.6 µs 안이다.

## 결론

1. **분해 단위 — 확정: 한 쿼리는 그 안의 모든 잡을 합산한다.** dispatch k개가 든 CB 하나의 값은 정확히 k배이고(QPU-active 2 110 720 × k,
   core-memory-reads 16 396 × k), CB k개로 나눠 각각 쿼리한 합과 같다. V3DV 는 dispatch 마다 CSD 잡을 하나 만들고,
   커널 perfmon 은 잡이 시작될 때 붙고 멈출 때 값을 **누적**한다(아래 근거). 따라서 **디스패치 하나만 보려면 쿼리 하나로 그 디스패치만 감싸야 한다.**
   부산물: cycle-count 는 V3D 코어 클럭을 세며 timestamp 구간과 0.1% 안에서 맞는다(960 MHz).
2. **혼입 — 확정: 다른 V3D 사용자의 bin/render 잡이 겹치면 값이 오염된다. 방향은 둘이다.**
   - **절단**: 다른 클라이언트의 잡이 시작되면 커널이 우리 perfmon 을 멈춘다. 우리 CSD 잡의 남은 구간은 세지 않는다(B 에서 정상의 ~4%).
   - **유입**: 우리 perfmon 이 켜져 있는 동안 다른 큐에서 이미 돌던 렌더 작업이 GPU 전역 카운터에 들어온다(B 에서 CLE-render·TLB >0).
   - 같은 CSD 큐의 다른 프로세스 잡은 직렬화되므로 값에 섞이지 않는다(C). 대신 **timestamp 구간이 늘어난다**.
   - 데스크톱 유휴에서도 labwc 의 간헐적 렌더 잡 때문에 **100회 중 0~3회**가 잘린다. 그래서 **평균은 쓰지 말고 중앙값을 쓰며, 결정적 카운터(QPU-active 등)가 최빈값과 다른 표본은 버린다.**
3. **쿼리 오버헤드 — GPU 쪽은 확정(무시 가능), 호스트 쪽은 불확실(≤ 약 20 µs/submit, 0.4%).**
   - 성능 쿼리를 붙여도 커널 잡 수는 같고, GPU 구간 차이(+7 µs)는 잡음(11.5 µs) 안이다. 카운터 수(4개 대 32개)의 차이도 없다.
   - 호스트 submit→fence 는 +19.5 µs 로 규칙상 잡음을 넘는다. 그러나 같은 설정의 반복이 +15.5 µs 표류해서 분리되지 않는다.
   - 결과 읽기는 32개에 5.4 µs, 4개에 2.9 µs 다.
   - 3 pass 는 **작업 전체를 3번 제출**한다(3 × 4.93 ms = 14.8 ms). pass 추가에 따른 별도 비용은 보이지 않지만 GPU 시간이 pass 수만큼 곱해진다.

### 커널 근거 (웹 참조 — 로컬 레지스트리·MCP 코퍼스에 커널 드라이버 소스가 없음)
- `raspberrypi/linux` rpi-6.18.y `drivers/gpu/drm/v3d/v3d_sched.c` `v3d_switch_perfmon()`: **bin/render/csd 잡이 시작될 때만** 호출된다.
  잡의 perfmon 이 활성 perfmon 과 다르면(NULL 포함) 활성 것을 `v3d_perfmon_stop(..., capture=true)` 한다. 잡이 끝날 때는 멈추지 않는다. TFU 잡은 호출하지 않는다.
  <https://raw.githubusercontent.com/raspberrypi/linux/rpi-6.18.y/drivers/gpu/drm/v3d/v3d_sched.c>
- 같은 트리 rpi-6.12.y `v3d_perfmon.c` `v3d_perfmon_stop()`: `perfmon->values[i] += V3D_CORE_READ(...)` 로 **누적**한다(6.18 판은 확인하지 않았고, 동작은 E1 측정으로 확인).
  <https://raw.githubusercontent.com/raspberrypi/linux/rpi-6.12.y/drivers/gpu/drm/v3d/v3d_perfmon.c>
- 이 코드는 E1(합산)·E2(절단·유입, 같은 큐 잡은 무혼입)의 관측과 모두 맞는다. 단 V3DV 쪽 구현(perfmon 을 쿼리·pass 마다 어떻게 만드는지)은 읽지 않았다.

## 주의와 남은 일

- **모든 수치는 heap_sum 한 작업(메모리 읽기 위주, 4.9 ms)의 값이다.** 짧은 디스패치(<0.1 ms)에서의 쿼리 비율 비용은 재지 않았다.
  파이프라인 셰이더(sand_letterbox 등)는 0.5~4 ms 라 같은 범위지만 확인은 안 했다.
- E2 B 는 vkcube MAILBOX(약 3 200 fps)라는 극단적 부하다. 실제 표시 경로(스왑체인 합성·HVS 평면)가 얼마나 자르는지는 따로 재야 한다.
  **실측 파이프라인에서 카운터를 쓰려면 표시 off 가 기본 조건이다.**
- TFU 잡은 perfmon 전환을 일으키지 않는다(커널 코드). TFU 작업 자체가 GPU 전역 카운터에 얼마나 들어오는지는 가르지 못했다.
- E3 호스트 비용은 설정 순서(i→ii→iii→i)에 따른 표류와 분리되지 않았다. 확정하려면 설정을 교차 배치(ABAB)하고 독립 실행 3회로 재야 한다.
- 검증 레이어 관찰: 3 pass 제출에서 timestamp 를 함께 쓰면, pass 마다 호스트 리셋을 해도 pass ≥ 1 에서
  `VUID-vkCmdWriteTimestamp-None-00830` 이 보고되고 레이어 안에서 크래시가 났다(레이어 없이 실행하면 정상). 레이어 오탐으로 **추정**하나 미확인이다.
  그래서 3 pass 1회 측정에는 timestamp 를 빼서 우회했다.
- 1차 실행은 `--exp-csv` 가 값을 `%.6g` 로 써서 16 885 760 같은 값이 잘리는 결함이 있었다. 폐기하고 수정한 도구로 전부 다시 쟀다(이 폴더가 2차).
  1차와 2차의 경향은 같았다. 1차에서만 보인 onecb_k8 8.24배는 2차에서 재현되지 않았다(7.999).
- 한 번의 실행 안에서 반복한 표본이다(measure 스킬의 "독립 실행 3회" 공식 측정 아님). 결론은 배수·동일성·유무 판정이라 이 표본으로 충분하다고 봤다.

## 파일

- `env_bench.txt` — 커널, Mesa, 데스크톱, NPU, 단계별 온도·throttled·클럭
- `evidence/counters.txt`, `evidence/counters.csv` — 카운터 93개 열거(이름·설명·UUID)와 pass 분석
- `evidence/e1_granularity_stdout.txt`, `e1_granularity_samples.csv`(긴 형식 원시 표본), `e1_v3d_clock.txt`(E1 중 0.25 s 간격 `measure_clock v3d`)
- `evidence/e2_<조건>_stdout.txt`, `e2_<조건>_samples.csv`, `e2_<조건>_gpu_stats_{before,after}.txt` — 조건 A_idle, B_vkcube, C_load, A2_idle
- `evidence/e2_B_vkcube_app.txt`(vkcube 출력, V3D 선택 확인), `evidence/e2_C_load_proc.txt`(부하 프로세스 출력)
- `evidence/e3_overhead_stdout.txt`, `e3_overhead_samples.csv`
- `verification/validation_{enum,e1_granularity,e2_contention,e3_overhead,load}.txt` — 검증 레이어 + 동기화 검증 게이트. VUID·SYNC 0건
