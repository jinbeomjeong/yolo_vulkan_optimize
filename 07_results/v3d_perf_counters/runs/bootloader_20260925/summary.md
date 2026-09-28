# V3D 성능 카운터 동작 특성 (E1~E3) 재측정: 부트로더 2026-09-25, V3D 1150 MHz

상위 실험 [`v3d_perf_counters`](../../summary.md)(2026-09-29 02:13, 부트로더 2025-12-08, V3D **960 MHz**)의 **회차**다.
프로토콜, 도구, 카운터 세트, 반복 수는 모두 같게 두었고 펌웨어만 바뀌었다. 부모의 표를 같은 형식으로 다시 채운 뒤 960 대 1150을 나란히 놓는다.

## 조건

- **측정일**: 2026-09-29 03:02~03:04. 재부팅(02:51) 후 11분 뒤이며 E3까지 31초 걸렸다. 상세 기록은 `env_bench.txt`에 있다.
- **바뀐 것(펌웨어)**:
  - 부트로더 `2026/09/25 15:59:09`(1accd665)
  - `v3d_freq=1150`, `v3d_freq_min=500`: config.txt에 v3d 설정이 없어서 펌웨어 기본값이 적용됐다.
  - `over_voltage_avs` 0x1f270 → 0x1e780
  - 적용 경위: [`firmware.txt`](../../../pipeline_ffmpeg_vulkan_hailo/runs/bootloader_20260925/env/firmware.txt)
- **같은 것**:
  - kernel 6.18.50+rpt-rpi-2712, V3DV Mesa 26.2.2, V3D 7.1.10.2
  - 도구 `build/bin/yvo_perf_counters`: 커밋 45c3135 소스이며 `ninja -n` 기준으로 최신이다. 다시 빌드하지 않았다.
  - 카운터 열거 결과(`evidence/counters.csv`)가 부모와 **바이트 단위로 같다**(93개, 1 pass에 21개).
- **V3D 클럭**:
  - E1 동안 0.25 s 간격 `measure_clock v3d` 68표본이 **전부 1150.008~1150.018 MHz**였다(`evidence/e1_v3d_clock.txt`).
  - 단계별 스냅샷 6개도 모두 1150 MHz였다. `v3d_freq_min=500`인데도 측정 중에는 1150에 고정돼 있었다.
- **온도와 상태**: SoC 64.8~70.8 °C, `throttled=0x0`. 데스크톱(labwc, HDMI-A-1)은 켜 둔 상태였다.
- **NPU 미사용**: `/dev/hailo0` 사용자 0, `hailort_service` 데몬만 상주했다.
- **작업과 방식** (부모와 동일):
  - 작업: heap_sum 4 MiB 1 dispatch(256 WG × 64).
  - 연속 submit→fence 스트림으로 돌렸다. 워밍업 30회 뒤 E1·E2는 100회, E3는 200회를 쟀다.
  - GPU/CPU 합계 불일치는 전 실행에서 **0**이었다.
- **측정 실행**: 검증 레이어를 **끈 채** 쟀다.
- **게이트 실행**: `verification/`에 따로 남겼다.
  - 켠 레이어: `VK_LOADER_LAYERS_ENABLE=*validation VK_KHRONOS_VALIDATION_VALIDATE_SYNC=true VK_KHRONOS_VALIDATION_REPORT_FLAGS=error,warn,info`
  - 동기화 검증이 실제로 켜졌는지는 각 파일의 레이어 출력 `Current Enables: VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT`로 확인했다.
  - 설정 키 `validate_sync`는 설치된 `VkLayer_khronos_validation.json`에서 확인했다.

### 실행 명령 (저장소 루트, `E=runs/bootloader_20260925/evidence`)
```
build/bin/yvo_perf_counters --csv $E/counters.csv                                   > $E/counters.txt
build/bin/yvo_perf_counters --exp granularity --label E1 --iters 100 --warmup 30 --exp-csv $E/e1_granularity_samples.csv > $E/e1_granularity_stdout.txt
  (동시에: while :; do echo "$(date +%s.%N) $(vcgencmd measure_clock v3d)"; sleep 0.25; done > $E/e1_v3d_clock.txt)
# E2 조건마다: cat /sys/class/drm/renderD128/device/gpu_stats > e2_<c>_gpu_stats_before.txt; 아래 실행; ... > _after.txt
build/bin/yvo_perf_counters --exp contention --label <c> --iters 100 --warmup 30 --exp-csv $E/e2_<c>_samples.csv > $E/e2_<c>_stdout.txt
  c = A_idle → B_vkcube → C_load → A2_idle
  B: XDG_RUNTIME_DIR=/run/user/1000 WAYLAND_DISPLAY=wayland-0 vkcube --wsi wayland --present_mode 1   (MAILBOX, 2 s 뒤 측정)
  C: build/bin/yvo_perf_counters --load 60   (2 s 뒤 측정, 끝나면 kill)
build/bin/yvo_perf_counters --exp overhead --label E3 --iters 200 --warmup 30 --exp-csv $E/e3_overhead_samples.csv > $E/e3_overhead_stdout.txt
# 게이트(레이어 ON): --csv /dev/null | --exp granularity --label val | --exp contention --label val | --exp overhead --label val | --load 5
```

## 결과

### E1 분해 단위: 한 CB에 dispatch k개 vs CB k개(각 1 dispatch)

| 설정 | gpu_ns 중앙값 (p90) | 배수 | compute-active-cycles | QPU-active | core-memory-reads | 실효 클럭 cycle-count/gpu_ns | 커널 잡/iter (csd·cpu) |
|---|---|---|---|---|---|---|---|
| onecb_k1 | 4 347 367.5 (4 362 973.7) | 1.000 | 4 958 716 | **2 110 720** | 16 396 | 1150.8 MHz | 1 · 2 |
| kcb_k1 | 4 346 682 (4 354 675.4) | 1.000 | 4 958 773.5 | 2 110 720 | 16 396 | 1150.9 | 1 · 2 |
| onecb_k2 | 8 686 957.5 (8 732 246.7) | 1.998 | 9 915 565 (×2.000) | **4 221 440 (×2.000)** | 32 792 (×2.000) | 1150.5 | 2 · 2 |
| kcb_k2 | 8 991 356 (9 060 786.4) | 2.068 | 10 223 975 | 4 221 440 | 32 792 | 1151.3 | 2 · 4 |
| onecb_k4 | 18 017 127 (18 208 495.2) | 4.144 | 20 547 388 (×4.144) | **8 442 880 (×4.000)** | 65 584 (×4.000) | 1150.5 | 4 · 2 |
| kcb_k4 | 18 014 980 (18 260 797.5) | 4.144 | 20 554 300 | 8 442 880 | 65 584 | 1151.4 | 4 · 8 |
| onecb_k8 | 36 915 161.5 (37 965 969.1) | 8.491 | 42 028 940 (×8.476) | **16 885 760 (×8.000)** | 131 170 (×8.000) | 1150.3 | 8 · 2 |
| kcb_k8 | 36 152 711 (37 229 189.1) | 8.316 | 40 703 282 | 16 885 760 | 131 170 | 1150.9 | 8 · 16 |

- 결정적 카운터 QPU-active, core-memory-reads, TMU-active-cycles(294 912 × k)는 **정확히 k배**이고, onecb_k와 kcb_k가 같다.
- **gpu_ns와 compute-active-cycles의 배수가 4.144~8.491로 k보다 크다.** 이는 분해 단위 때문이 아니라, 아래 "계단"(heap_sum 한 번의 시간이 실행 도중 계단식으로 바뀌는 현상) 때문이다.
  - 설정을 순서대로 도는 사이에 계단이 바뀌었다. kcb_k2는 30회째 전후로 4.35 → 4.50 ms/dispatch로 올라갔다.
  - **같은 계단끼리 dispatch당 값을 비교하면 onecb와 kcb가 같다.**

  | 계단 | onecb_k4 | kcb_k4 | onecb_k8 | kcb_k8 |
  |---|---|---|---|---|
  | 4.50 ms대 | 4 503.8 µs (n=91) | 4 503.4 µs (n=89) | — | — |
  | 4.45 ms대 | — | — | 4 457.9 µs (n=28) | 4 454.7 µs (n=44) |
  | 4.62 ms대 | — | — | 4 617.6 µs (n=55) | 4 619.0 µs (n=16) |

- 커널 잡(`gpu_stats` 차분)은 부모와 같다. **dispatch 1개 = CSD 잡 1개**이고, timestamp는 CPU 큐 잡이다(쓰기 1회 = cpu 잡 1개). tfu는 0이다.
- 실효 클럭(반복별 비의 중앙값)은 1150.3~1151.4 MHz로 `measure_clock v3d` 1150.0 MHz와 0.12% 안에서 맞는다. compute-active/gpu_ns는 1140.3~1141.6 MHz로, 구간의 99.1%가 CSD 활성이다.
- 절단(QPU-active < 2 110 720·k) 회수(onecb/kcb): k1 0/0, k2 1/1, k4 3/2, k8 **15/35**. 부모는 2/3, 3/5, 7/9, 13/14였다. k8 동안 데스크톱 bin/render 잡은 onecb 18개, kcb 40개였다.

### E2 혼입: 고정 작업(4 MiB × 1) 100회, 조건별

| 조건 | 다른 클라이언트 bin/render 잡 (100회 동안) | QPU-active = 2 110 720 인 회수 | QPU-active 중앙값 (min) | compute-active 중앙값 (p90) | 그래픽 카운터 >0 회수 | CLE-render 중앙값 | gpu_ns 중앙값 (p90) |
|---|---|---|---|---|---|---|---|
| A 유휴 | 5 / 5 | **95** (5회 미달) | 2 110 720 (310 430) | 5 089 005 (5 119 912.6) | 0 | 0 | 4 461 645.5 (4 539 353.3) |
| B vkcube(MAILBOX) | **2 888 / 2 889** (+tfu 2 888) | **0** (100회 전부 미달) | **29 164.5** (152) | **120 564** (307 363.5) | **67** | **32 808.5** | 9 446 263.5 (9 837 419.2) |
| C 두 번째 프로세스 `--load 60` | 10 / 10 (+상대 csd 101) | **94** (5회 미달, **1회 초과**) | 2 110 720 (26 719) | 5 090 335 (5 139 856.9) | 1 | 0 | **8 844 589** (9 276 912.5) |
| A2 유휴(재측정) | 5 / 5 | **95** (5회 미달) | 2 110 720 (222 306) | 5 088 771 (5 106 562) | 0 | 0 | 4 461 887 (4 601 988.3) |

- 방해가 없는 표본(QPU-active가 최빈값이고 그래픽 카운터가 0)에서 결정적 카운터는 **비트 단위로 같다**. 표본 수는 A 95, C 94, A2 95다.
  - QPU-active 2 110 720, TMU-active 294 912
  - core-memory-reads는 A_idle에서 16 397, C·A2에서 16 396이었다. **실행(프로세스)마다 상수이고 1 차이가 난다.** 부모도 C에서 16 397이었다.
- 미달 회수가 다른 클라이언트의 bin/render 잡 수를 따라간다(A 5→5, A2 5→5, C 10→5+초과 1, B 2 888→100/100).
- C의 **초과 1회**(iter 78)는 같은 큐의 상대 프로세스 잡이 섞인 것이 아니다. 그 표본은 다음과 같았다.
  - QPU-active 2 445 707(+334 987), core-memory-reads 33 170, **CLE-render 868 891 > 0**
  - 상대 heap_sum이 통째로 섞였다면 QPU-active가 +2 110 720 늘고 그래픽 카운터는 0이어야 한다. 실제 값은 그렇지 않다.
  - 따라서 **이미 돌던 데스크톱 렌더 잡의 유입**으로 본다(**추정**: 같은 표본에 그래픽 카운터가 있다는 것이 근거다).
- C의 gpu_ns는 A의 **1.98배**다. 부모도 1.98배(9.75/4.92 ms)였다.

### E3 오버헤드: 같은 작업 200회

| 설정 | gpu_ns 중앙값 (p90) | host submit→fence µs 중앙값 (p90) | `vkGetQueryPoolResults` perf µs | ts µs | 커널 잡/iter |
|---|---|---|---|---|---|
| (i) timestamp만 | 4 460 082 (4 551 037.2) | 4 482.1 (4 578.4) | — | 3.4 | csd 1 · cpu 2 · cache_clean 1 |
| (ii) + 카운터 32개(1 pass) | 4 467 878 (4 871 293.4) | 4 497.5 (4 931.0) | 4.7 | 3.1 | 동일 |
| (iii) + 카운터 4개 | 4 465 683 (4 606 311.3) | 4 493.5 (4 643.8) | 2.4 | 2.9 | 동일 |
| (i) 반복(표류 대조) | 4 622 350 (4 687 835.5) | 4 655.8 (4 731.8) | — | 5.4 | 동일 |
| 전체 93개(3 pass) 1회 | — | **13 822.0 (제출 3회)** | 7.8 | — | — |

판정 규칙: (ii)−(i)가 (i)의 잡음(p90−중앙값)보다 작으면 무시할 수 있다(도구 출력).

| 지표 | (ii)−(i) | (iii)−(i) | (i반복)−(i) | 잡음(i) | 규칙 판정 |
|---|---|---|---|---|---|
| gpu_ns | +7.8 µs (+0.17%) | +5.6 µs (+0.13%) | **+162.3 µs (+3.64%)** | 91.0 µs | (ii)·(iii) 무시 가능 |
| host submit→fence | +15.3 µs (+0.34%) | +11.4 µs (+0.25%) | **+173.7 µs (+3.88%)** | 96.3 µs | (ii)·(iii) 무시 가능 |

- 이번에는 잡음과 (i반복)의 표류가 모두 **계단** 때문에 커졌다. 부모의 잡음은 11.5 µs와 13.6 µs였다.
  - (i)의 200회 중 179회가 4.40~4.55 ms 계단에 있었고, (i반복)은 200회 **전부** 4.62 ms대 계단에 있었다.
- 같은 계단(4.40~4.55 ms)의 표본끼리 비교하면 다음과 같다.

  | 설정 (n) | gpu 중앙값 (µs) | (i) 대비 | host 중앙값 (µs) | (i) 대비 |
  |---|---|---|---|---|
  | (i) (179) | 4 459.1 | — | 4 480.7 | — |
  | (ii) (144) | 4 463.6 | **+4.5** | 4 488.7 | **+8.0** |
  | (iii) (151) | 4 462.2 | **+3.1** | 4 485.5 | **+4.8** |

  이 수치들은 `e3_overhead_samples.csv`를 따로 집계해 얻었다.

### 960 MHz(부모) 대 1150 MHz(이번), 주요 값

| 항목 | 960 MHz | 1150 MHz | 비 (960/1150 또는 1150/960) |
|---|---|---|---|
| `measure_clock v3d` (E1 중) | 960.0 MHz 74표본 | 1150.0 MHz 68표본 | 1.198 |
| 실효 클럭 cycle-count/gpu_ns (onecb_k1) | 960.6 MHz | 1150.8 MHz | 1.198 |
| heap_sum gpu_ns, E1 onecb_k1 (가장 빠른 계단) | 4 916 549.5 ns | 4 347 367.5 ns | **1.131배 빨라짐** |
| heap_sum gpu_ns, E2 A 유휴 | 4 922 382 ns | 4 461 645.5 ns | 1.103배 |
| compute-active-cycles (onecb_k1) | 4 686 183 | 4 958 716 | +5.8% |
| QPU-active | 2 110 720 | 2 110 720 | 동일(클럭 무관) |
| core-memory-reads | 16 396 | 16 396 | 동일 |
| TMU-active-cycles | 294 912 | 294 912 | 동일 |
| QPU-wait-bubble (onecb_k1) | 2 542 358 | 2 814 767.5 | +10.7% |
| L2T-stalls-read-data (onecb_k1) | 1 779 078.5 | 2 071 703 | +16.4% (ns로 환산하면 1 852 → 1 800 µs) |
| compute-active − L2T-stalls-read-data | 2 907 104.5 사이클 | 2 887 013 사이클 | −0.7% |
| E1 결정적 카운터 k배 | 정확 | 정확 | — |
| E2 A·A2 미달 회수 / 100 | 3 · 0 | 5 · 5 | — |
| E2 B QPU-active 중앙값 / CLE-render 중앙값 | 88 700 / 107 338 | 29 164.5 / 32 808.5 | — |
| E2 C gpu_ns / A gpu_ns | 1.98 | 1.98 | — |
| E3 (ii)−(i) gpu_ns | +7.3 µs | +7.8 µs (같은 계단 +4.5 µs) | — |
| E3 (ii)−(i) host | +19.5 µs | +15.3 µs (같은 계단 +8.0 µs) | — |
| E3 perf 결과 읽기 32개 / 4개 | 5.4 / 2.9 µs | 4.7 / 2.4 µs | — |
| E3 3 pass 1회 | 14 796.0 µs | 13 822.0 µs | 1.070 |

### 보조 진단: 계단 (`evidence/diag_clock_steps_*`, 부모 프로토콜 밖에서 추가)

E1·E3에서 heap_sum 한 번의 시간이 계단 사이를 오갔다. 계단은 4.35·4.45·4.50·4.62 ms였다. 원인을 가르려고 E2와 같은 작업을 600회 돌리면서 core·v3d·arm 클럭을 약 0.13 s 간격으로 기록했다.
- 실행 도중(약 240회째) **4 347.7 → 4 460.7 µs(+2.60%)**로 한 계단 올라갔다. 그동안 클럭은 변하지 않았다.
  - core 910 MHz
  - v3d 1150 MHz
  - arm 2.4 GHz(22표본)
- 두 계단의 카운터 중앙값을 비교했다(각 계단에서 방해 없는 표본 208회 대 292회).
  - 변하지 않은 카운터: QPU-active, TMU-active, L2T-hit·miss, core-memory-reads, QPU-stalls
  - 늘어난 카운터: **L2T-stalls-read-data +129 935(+6.3%)**, QPU-wait-bubble +129 474, compute-active-cycles +129 447
  - 즉 **늘어난 사이클은 전부 메모리 읽기 대기**다.
- 부모(960 MHz)에서는 이런 계단이 보이지 않았다. E1 배수는 7.991~8.000으로 정확했고, E3 (i반복)−(i)도 +5.8 µs에 그쳤다.

## 결론

1. **분해 단위: 유지된다. 한 쿼리는 그 안의 모든 잡을 합산한다.**
   - QPU-active, core-memory-reads, TMU-active는 1150 MHz에서도 정확히 k배이고, CB k개로 나눠 잰 합과 같다. dispatch 1개는 CSD 잡 1개다.
   - gpu_ns와 compute-active의 배수가 어긋난 것(최대 8.491)은 아래 계단 때문이다. 같은 계단끼리 비교하면 onecb와 kcb의 dispatch당 값이 같다(4 503.8 대 4 503.4 µs 등).
   - cycle-count는 여전히 V3D 코어 클럭을 센다(1150.8 MHz, timestamp 구간과 0.1% 안에서 맞음).
2. **혼입: 유지된다. 절단과 유입이 모두 재현됐다.**
   - B(vkcube)에서 100회 전부 잘렸다. QPU-active 중앙값은 정상의 **1.4%**(29 164.5)였고, 67회에 그래픽 카운터 >0(유입)이 나왔다.
   - 같은 CSD 큐의 다른 프로세스 잡(상대 csd 101개)은 섞이지 않았다. C의 초과 1회는 데스크톱 렌더 유입으로 본다(**추정**).
   - 대신 timestamp 구간이 1.98배로 늘었다(부모와 같음).
   - 유휴 데스크톱에서 잘린 회수는 이번에 **100회 중 5회**(A·A2)였다. 부모 결론의 "0~3회"보다 많다. 원인은 데스크톱 잡 수(5개)다.
   - 규칙(표시 off, 중앙값, 결정적 카운터 최빈값 필터)은 그대로 유효하다.
3. **쿼리 오버헤드: GPU 쪽은 유지된다(무시 가능). 호스트 쪽은 여전히 분리되지 않는다.**
   - 커널 잡 수는 같고, GPU 구간 차이는 +7.8 µs다. 같은 계단끼리 비교하면 +4.5 µs다.
   - 호스트 쪽은 +15.3 µs로 이번에는 규칙상 잡음 안에 든다. 그러나 잡음 자체가 계단 때문에 7배 커졌기 때문이다(91 µs).
   - 같은 계단끼리의 호스트 차이는 +8.0 µs(0.18%)로 부모의 ≤ 약 20 µs 범위 안이다.
   - 3 pass는 여전히 작업 전체를 3번 제출한다(13.8 ms ≈ 3 × 4.6 ms).
4. **클럭 상승 효과: heap_sum은 1.198배 클럭에서 1.10~1.13배만 빨라졌다. 연산에 묶인 작업이 아니라 메모리 대기가 큰 몫을 차지한다.**
   - QPU-active, TMU-active, core-memory-reads는 클럭과 무관하게 같다.
   - `compute-active − L2T-stalls-read-data` 사이클은 거의 그대로다(2.907M → 2.887M). 반면 L2T-stalls-read-data를 시간으로 환산하면 거의 그대로다(약 1.85 → 1.80 ms).
   - 해석(**추정**, 카운터 의미 미검증): 연산 부분은 클럭을 따라 줄고, 메모리 대기(ns 단위)는 클럭과 무관하다. 1150 MHz에서 구간의 약 41%(1.80/4.35 ms)가 메모리 읽기 대기다.
5. **새 현상: 1150 MHz에서 heap_sum 한 번의 시간이 +2.6~+6.3% 계단으로 바뀐다. 원인은 확인하지 못했다.**
   - 늘어난 사이클은 전부 메모리 읽기 대기(L2T-stalls-read-data)다. 같은 순간 vcgencmd로 보이는 core·v3d·arm 클럭은 변하지 않았다.
   - DRAM이나 인터커넥트 쪽 상태 변화로 **추정**한다. 새 펌웨어와의 인과는 확인하지 않았다(부모 960 MHz 회차에는 없었다는 것만 확인).
   - **새 펌웨어에서 잰 GPU 구간에는 이 정도의 계단 잡음이 섞일 수 있다.** 파이프라인 재측정과 비교할 때 고려해야 한다.

## 주의와 남은 일

- **AXI-read-wait-cycles는 대역폭이나 대기 판단에 쓰지 않는다.** k≥2(부모) 또는 k≥4(이번)에서 중앙값이 구간 사이클 수를 크게 넘었다. 예를 들어 이번 onecb_k4는 50 839 537.5로 구간 사이클 20.7M을 넘는다. 표본마다 흔들리기도 한다. 그래서 위 해석에는 L2T-stalls-read-data만 썼다.
- 계단의 원인은 가르지 못했다. 다음 항목은 재지 않았다.
  - SDRAM 클럭이나 상태: `measure_clock sdram`이 0을 돌려준다.
  - CPU 쪽 메모리 트래픽
  - 온도: 64.8~70.8 °C 범위였다.
  - 계단의 지속 시간과 빈도
- 계단 때문에 E1의 gpu_ns 배수와 E3의 호스트 비교는 설정 순서에 좌우된다. 설정을 교차 배치(ABAB)하고 독립 실행 3회로 재야 확정된다.
- 한 번의 실행 안에서 반복한 표본이다. measure 스킬의 "독립 실행 3회" 공식 측정이 아니다. 부모와 같은 조건이다.
- 게이트는 5개 파일 모두 VUID 0, SYNC-HAZARD 0, Validation Error·Warning 0, 종료 코드 0이었다(동기화 검증 활성 확인).
  - 3 pass 제출은 부모와 같이 timestamp를 빼고 돌려서 레이어 크래시(`VUID-vkCmdWriteTimestamp-None-00830` 오탐 추정)를 피했다.
  - 이 크래시가 이번 레이어에서도 나는지는 다시 보지 않았다.
- `env_bench.txt`의 "wf-panel 2"는 `pgrep -fc` 집계 값이다. 측정 후 `pgrep -a wf-panel`로는 1개였다.
- E2 B의 vkcube 부하는 부모(3 190 프레임)보다 조금 적었다(2 888 프레임, 약 0.95 s).

## 파일

- `env_bench.txt`: 커널, Mesa, 부트로더, `v3d_freq`·`v3d_freq_min`·`over_voltage_avs`, 단계별 온도·throttled·클럭
- `evidence/counters.{txt,csv}`: 카운터 93개 열거. csv는 부모와 바이트 단위로 같다.
- `evidence/e1_granularity_{stdout.txt,samples.csv}`, `evidence/e1_v3d_clock.txt`
- `evidence/e2_<조건>_{stdout.txt,samples.csv,gpu_stats_before.txt,gpu_stats_after.txt}`: 조건은 A_idle, B_vkcube, C_load, A2_idle
- `evidence/e2_B_vkcube_app.txt`: 첫 줄에 실행 명령, 이어서 V3D 선택 출력
- `evidence/e2_C_load_proc.txt`: 부하 프로세스 출력(측정이 끝난 뒤 kill해서 요약 줄이 없다)
- `evidence/e3_overhead_{stdout.txt,samples.csv}`
- `evidence/diag_clock_steps_{stdout.txt,samples.csv,clocks.txt}`: 보조 진단. 계단과 클럭의 관계를 봤다.
- `verification/validation_{enum,e1_granularity,e2_contention,e3_overhead,load}.txt`: 레이어와 동기화 검증 게이트. 첫 줄에 env와 명령이 있다.
