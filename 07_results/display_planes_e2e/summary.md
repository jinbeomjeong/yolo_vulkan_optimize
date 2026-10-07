# HVS 평면 표시 정식 측정 (실험 B)

계획: [`display_vulkan_hdmi/notes/formal_measurement_plan.md`](../display_vulkan_hdmi/notes/formal_measurement_plan.md) §5.
벤치: `06_tools/bench/display_planes_bench.sh`(기본값 그대로 실행). 콘솔 전문은 `raw/_session/bench_stdout.log`, 세션 전환 기록은 `evidence/sessions.txt`.

## 1. 측정 조건

- 날짜: 2026-10-08 05:19~05:41 (KST)
- 하드웨어: Raspberry Pi 5, Hailo-8, PCIe **Gen3 x1**(8.0 GT/s), V3D **1150 MHz**(부트로더 2026-09-25), CPU 2.4 GHz,
  HDMI-A-1 1920×1080 **60 Hz**
- 소프트웨어: kernel 6.18.50+rpt-rpi-2712, HailoRT 4.24.0, mesa-vulkan-drivers 26.2.2, libavcodec61 7.1.5 rpt2,
  labwc 0.20.2, libwlroots-0.20 0.20.2+rpt4(색 범위 반전 재판정 완료 조건 — [`notes/kms_vs_wayland.md`](../display_vulkan_hdmi/notes/kms_vs_wayland.md) §4.1), git `aa8bbf0`
- 모델 `03_models/yolov11m.hef`, 입력 `04_data/videos/video_h265.mp4`(1080p30 HEVC)
- 공통 경로: `--mode dmabuf --layout nhcw --postproc gpu --postproc-alloc export --nms gpu --gpu-depth 3 --batch 2 --slots 4
  --display on --display-fullscreen --display-sync matched --display-warmup 30`, 디코더 프레임 스레드 1(앱 기본)

| id | 백엔드 | pace |
|---|---|---|
| `vk_source` / `vk_asap` | `wayland`(Vulkan 합성: display_convert + 스왑체인) | source / asap |
| `wlp_source` / `wlp_asap` | `wayland-planes`(linux-dmabuf, `--display-color-range auto`) | source / asap |
| `kms_source` / `kms_asap` | `kms`(atomic, `--display-connector HDMI-A-1`, 데스크톱 세션 비활성) | source / asap |

- 반복: 900프레임 × **3회**, run 바깥·설정 안쪽 인터리브. 회차마다 kms 두 설정 앞뒤로 `loginctl activate` 전환 1번씩(3/3 정상 복귀)
- 냉각: **매 실행 전** Hailo ≤ 65 °C(대기 설정별 합 90~135 s). 실행 직전 최고 64.9 °C(첫 실행 57.5 °C)
- 신뢰성: 18회 모두 rc 0, 검사 통과, `hailort.log` 경고·에러 **0건**. `throttled=0xe0000`은 과거 기록 비트이고 현재 비트(하위 4비트)는 0이다
- glass·present 간격·늦은 프레임은 앞 30 프레젠테이션을 뺀 값이다(`--display-warmup 30`, kms 첫 커밋의 블로킹 모드셋 때문). e2e·slot wait는 전체 프레임 값이다

## 2. 결과 (설정당 3회 중앙값, 괄호는 최소~최대)

원시값 `results_raw.csv`(96열), 집계 `results.csv`.

### 2.1 처리량·CPU·지연

| 설정 | fps | cpu_ms_per_frame | glass_ms 중앙 / p90 | slot_wait_ms 평균 | e2e_ms_mean (디코드→NPU 완료) | present_interval_ms 중앙 / p90 |
|---|---|---|---|---|---|---|
| `vk_source` | 29.91 (29.87~29.92) | **2.059** (2.016~2.172) | 94.37 / 123.66 | **0.0** (0~0) | 80.01 | 33.08 / 66.96 |
| `wlp_source` | 29.23 (29.17~29.27) | **1.569** (1.520~1.603) | 99.93 / 116.47 | **28.7** (28.0~29.7) | 108.48 | 33.33 / 50.00 |
| `kms_source` | 29.33 (29.31~29.40) | **1.576** (1.526~1.629) | 99.85 / 116.38 | **27.8** (27.7~30.0) | 107.21 | 33.33 / 50.00 |
| `vk_asap` | 34.88 (34.87~34.89) | 1.832 (1.747~1.884) | 131.99 / 140.62 | 24.7 (24.6~24.7) | 130.44 | 18.10 / 50.45 |
| `wlp_asap` | 30.16 (30.12~30.19) | 1.546 (1.528~1.651) | 99.43 / 100.55 | 30.1 (30.1~30.1) | 111.76 | 16.67 / 66.67 |
| `kms_asap` | 30.26 (30.21~30.34) | 1.600 (1.589~1.603) | 99.48 / 100.48 | 30.0 (29.9~30.0) | 112.14 | 16.67 / 66.67 |

slot wait 은 실행별 stdout 의 `slot wait ms mean`(메인 스레드가 빈 슬롯을 기다린 시간)이다.

### 2.2 V3D·컴포지터 부하 (ms/frame, 실행 전후 차를 완료 프레임 수로 나눔)

| 설정 | V3D 합 (시스템) | CSD | TFU | yvo_pipeline V3D (fdinfo) | labwc CPU | labwc V3D | disp_convert_ms | disp_overlay_ms |
|---|---|---|---|---|---|---|---|---|
| `vk_source` | **10.19** | 6.17 | **3.72** | 10.13 | 0.533 | 0.022 | 3.234 | 0.111 |
| `wlp_source` | **2.89** | 2.52 | 0.014 | 2.86 | 0.589 | 0.024 | 0 | 0.336 |
| `kms_source` | **2.86** | 2.51 | 0.009 | 2.78 | 0.000 | 0.000 | 0 | 0.330 |
| `vk_asap` | 11.93 | 6.94 | 4.61 | 11.64 | 0.522 | 0.022 | 1.923 | 0.117 |
| `wlp_asap` | 2.90 | 2.53 | 0.014 | 2.86 | 0.567 | 0.024 | 0 | 0.336 |
| `kms_asap` | 2.88 | 2.53 | 0.009 | 2.88 | 0.000 | 0.000 | 0 | 0.334 |

### 2.3 평면 경로 검증 (wlp·kms 12회 전부)

- `scanout zero_copy 1`, `color_ok 1`, `hvs_underrun delta 0`, presentation ZERO_COPY 898~900/899~900, discarded 0
- kms: `flip_timeouts 0`, `commit failures 0`, `master_lost no`. 매 회차 데스크톱 복귀 후 labwc master y·HDMI-A-1 Enabled
- 중간 debugfs(`raw/<id>_run<N>/dri_state_mid.txt`): 영상 `plane[83]` NV12, crtc 1920×1080
- presented 899~900, dropped 0~1, 검출 **3369**(18회 모두 같음)
- 늦은 프레임(vblank seq 간격 > 2): source 223~231 / 868~869(**약 26%**), asap 296~305(약 35%), 최대 간격 4~6, chain breaks 0

## 3. 결론

1. **CPU: 평면 경로가 Vulkan 합성보다 프레임당 약 0.49 ms(24%) 적다**(source 2.059 → 1.569 / 1.576, 실행 간 범위 비중첩).
   컴포지터까지 더하면 vk 2.59, wlp 2.16, kms 1.58 ms/f다. wayland-planes 의 labwc CPU(0.589)는 Vulkan 합성(0.533)과 거의 같다.
   kms 의 labwc 0은 데스크톱 세션이 비활성이라 labwc가 합성을 하지 않아서다(labwc를 없앤 값이 아니다).
2. **V3D: 10.19 → 2.89 ms/f(−72%).** 사라지는 것은 display_convert(CSD 약 3.2 ms 몫)와 **TFU 3.72 ms**다. TFU는 앱 계측(`convert_ms`)에 잡히지 않던 비용이다.
   측정 fps 기준 V3D 점유가 약 30% → 8%로(30 fps로 환산하면 30.6% → 8.7%) 내려간다. 남는 2.5 ms 는 전처리·후처리 CSD다.
3. **wayland-planes 와 kms 는 같다.** CPU·V3D·glass·늦은 프레임이 모두 실행 간 범위 안에서 겹친다(glass 중앙 99.93 대 99.85).
   비정식 1회 측정의 차이(wlp 98.25 / kms 99.47 ms)는 재현되지 않았다. 데스크톱을 끄는 kms 를 쓸 성능상 이유는 없다.
4. **⚠ 지연은 이 운용점(b2/s4)에서 평면 경로가 오히려 길다.** glass 의 시작점은 디코드 반환이 아니라 **슬롯을 얻은 뒤**
   (`01_src/pipeline/main.cpp:895-897` `retain`, e2e 는 `:856` 디코드 반환)다. 평면 경로는 버퍼를 화면에서 내려갈 때(release)까지
   쥐므로(대기 2 + 커밋 후 미반납 3 = 최대 5장, `01_src/display/plane_display.hpp:17`) 슬롯 4개로는 모자라
   디코드된 프레임이 **평균 28~30 ms 슬롯을 기다린다**(vk_source 0). 이 대기는 glass 에 안 들어간다.
   - 디코드 반환 → 화면 평균 = slot wait + glass 평균: **vk_source 약 101.7 ms, wlp_source 약 130.5 ms, kms_source 약 129.4 ms**.
   - e2e(디코드 → NPU 완료)도 같은 방향이다: 80.0 대 108.5 / 107.2 ms.
   - source 설정의 fps 가 30 에 못 미치는 것(29.23~29.33, vk 29.91)도 같은 원인으로 본다.
   - 확인(비정식, 각 1회 450프레임, 인용 금지, `11_logs/2026-10-08_slotprobe_wlp_source_b2s{4,6,8}.log`): wlp_source 를
     `--slots 6`·`8` 로 돌리면 slot wait 이 **0**, e2e 중앙 107.2 → 73.7(s6)·73.8(s8) ms(vk_source 와 같은 수준), fps 29.33 → 29.80(s6)·29.81(s8) 이 된다.
   - 따라서 평면 경로의 **지연 비교는 이 실험으로 결론 내리지 않는다.** 평면 경로는 slots ≥ 6 에서 다시 재야 한다(§5).
5. **표시 박자(source)**: 평면 경로는 present 간격 중앙 33.33 / p90 50.0 ms, 프레임의 약 26%가 한 vblank 늦게 나간다(glass p90 = 중앙 + 16.7 ms).
   wlp·kms 가 같고 slots 6·8 확인 실행에서도 그대로라(late 109·111 / 418, s4 확인 실행은 112) 컴포지터나 슬롯이 아니라 **앱 페이싱이 vblank 위상에 묶여 있지 않은 탓**으로 추정한다(미검증).
   Vulkan 합성은 p90 66.96 ms 로 꼬리가 더 길다(glass p90 123.66).
6. **asap**: 평면 경로는 커밋 간격 게이트가 source 에서만 걸리므로 batch 2 의 두 장이 연달아 나간다(present 16.67 / 66.67 ms, 늦은 프레임 35%)
   — `plane_display.cpp:178-180` 주석이 예측한 그대로다. Vulkan 합성 asap 은 표시가 파이프라인을 묶지 않아 34.88 fps(운용점 b2/s4 처리량)로 돌고
   그만큼 큐가 쌓여 glass 132 ms 가 된다. 표시 용도로는 어느 쪽도 쓰지 않는다.

## 4. 해석할 때 주의할 점

- **glass 끝점이 다르다.** 평면 경로는 하드웨어 표시 시각(wp_presentation / page-flip), Vulkan 경로는 `vkWaitForPresentKHR`를 폴링해 알아챈 시각이다.
  Vulkan 쪽 glass 중앙(94.37)이 평면(99.9)보다 작은 것을 그대로 지연 차로 읽지 않는다. 그리고 위 결론 4의 슬롯 대기가 이보다 훨씬 크다.
- **glass 시작점이 슬롯 획득 뒤다.** slot wait 이 0이 아닌 설정(vk_asap 포함)은 glass 가 "디코드 → 화면"보다 작게 나온다. stdout 의 `decode → reaches screen` 표기는 이 경우 정확하지 않다.
- `disp_overlay_ms`는 이름만 같다(평면: 박스 그리기 + 지우기, Vulkan: `overlay_boxes.comp`).
- `yvo_ticks_*`(results.csv)는 프로세스 전체 수명(초기화 포함)의 /proc 값이다. 측정 구간 CPU 는 `cpu_ms_per_frame`(getrusage)을 쓴다.
- V3D 합·CSD·TFU 는 `gpu_stats` 시스템 전체 값이다(다른 클라이언트 포함). 프로세스별은 fdinfo 열.
- 옛 기준선 [`display_e2e/`](../display_e2e/summary.md)(V3D 960 MHz·자동 5스레드·CPU NMS)와는 조건이 달라 섞지 않는다.

## 5. 남은 일

1. **평면 경로 slots 재측정(권장 다음 단계)**: `vk_source`·`wlp_source`·`kms_source` 를 b2/s6(필요하면 s8)로 같은 프로토콜 900f×3.
   슬롯 대기가 사라진 상태에서 디코드 → 화면 지연·CPU 를 다시 비교한다. 지금 스크립트는 b2/s4 고정이라 slots 를 바꾸는 옵션이 필요하다.
2. **glass 시작점 정의**: 디코드 반환 시각에서 재는 값을 함께 내거나 시작점을 옮긴다(코드 수정, 표시 담당).
3. **source 페이싱의 vblank 위상**: 약 26% 늦은 프레임의 원인 확인(프레임별 커밋 시각 대 vblank 시각 기록)과 vblank 기준 일정 검토.
4. 표시 on 상태의 연속 운용 발열은 이 실험 범위 밖이다(실행마다 냉각).
