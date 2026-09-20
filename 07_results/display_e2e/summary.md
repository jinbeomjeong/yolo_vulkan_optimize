# 표시 통합 e2e — 디코드 → GPU 전처리 → NPU → GPU 후처리 → 오버레이 → HDMI

`01_src/display/` 의 표시 경로를 `DisplayPipeline` 으로 캡슐화하고 `yvo_pipeline` 의 슬롯 수명에 통합한 뒤의 측정.
이 측정으로 **파이프라인 전 구간이 하나의 실행 파일로 완성됐다.**

## 1. 측정 조건

| 항목 | 값 |
|---|---|
| 일시 | 2026-09-20 21:40~21:54 KST |
| 하드웨어 | Raspberry Pi CM5 (BCM2712), Hailo-8, V3D 7.1 (V3DV Mesa 26.2.2) |
| **PCIe 링크** | **8.0 GT/s x1 (Gen3)** — sysfs `current_link_speed` |
| 표시 | HDMI-A-1 1920×1080@60, labwc(Wayland), **`--display-fullscreen` 고정** |
| 모델 / 입력 | `03_models/yolov11m.hef` / `04_data/videos/video_h265.mp4` (HEVC 1080p30) |
| 운용점 | **`--batch 2 --slots 4`** (`07_results/op_point/summary.md` 에서 확정) |
| 프레임 수 / 반복 | 900 / 설정당 **3회** |
| 냉각 | 칩 > 75 °C 면 65 °C 이하까지. 합계 **315 s** |
| 바이너리 | 측정 중 재빌드 없음 (md5 전후 동일 `0c0ef2d3…`) |
| **신뢰성** | **12회 전부 `hailort.log` 온도·클럭 경고 0, 에러 0, `ok=1`** |
| 스크립트 | `09_scripts/display_bench.sh` (신규) |

> ⚠ **측정은 반드시 전체화면으로 한다.** 창 모드에서는 labwc 가 높이 1044 를 주면 `kUniformScale`
> 경로로 넘어가 변환이 **10.87 ms(2.6배)** 가 되는데 **에러도 경고도 없다.**

## 2. M5a — NPU 유휴 GPU 기준선 (M4 직전, 같은 열 상태)

경합 대조군이다. 이게 없으면 M4 의 GPU 시간이 NPU 경합 탓인지 코드 탓인지 구분할 수 없다.

| 구간 | median | p90 | n |
|---|---:|---:|---:|
| `display_convert.comp` (1920×1080 **1:1**) | **4.226 ms** | 4.296 | 299 |
| `overlay_boxes.comp` (합성 검출 5개) | **0.196 ms** | 0.212 | 299 |
| present 간격 (FIFO) | 16.662 ms | 16.737 | 294 |

`presented=300, display_dropped=0, recreates=1`

## 3. M4 — 표시 e2e (900프레임 ×3회 중앙값)

| 설정 | 표시 | pace | **FPS** | CPU ms/f | **검출** | NPU mean | e2e mean | **glass mean** | presented | dropped | held_max | timeout |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `reg_host_off` (오늘의 배포 경로) | off | – | 34.960 | 3.314 | **3369** | 111.3 | 142.1 | – | – | – | 3 | 0 |
| `disp_off` (postproc gpu) | off | – | **34.950** | **1.115** | **3369** | 107.8 | 142.5 | – | – | – | 3 | 0 |
| `disp_asap` | **on** | asap | **34.870** | 1.842 | **3369** | 95.9 | 132.9 | **131.42** | 900 | **0** | **4** | **0** |
| `disp_source` | **on** | source | 29.910 | 2.179 | **3369** | 73.9 | 80.8 | **101.13** | 900 | **0** | **4** | **0** |

**검출 수가 4설정 전부 3369 로 같다. 표시를 켜도 검출이 한 건도 변하지 않는다.**

### 3.1 GPU 4구간 (`disp_asap`, 중앙값)

| 구간 | 기준선 | 실측 | 증가 | p90 |
|---|---:|---:|---:|---:|
| 전처리 `sand_letterbox.comp` | 2.136 (`disp_off`) | **2.088** | **−2.2%** | 2.205 |
| 후처리 `yolo_decode.comp` | 0.627 (`disp_off`) | **0.626** | **−0.2%** | 0.704 |
| 표시 변환 `display_convert.comp` | 4.226 (M5a) | **4.153** | **−1.7%** | 4.207 |
| 오버레이 `overlay_boxes.comp` | 0.196 (M5a, 검출 5개) | **0.128** | **−34.7%** | 0.154 |
| **합** | — | **6.995 ms** | — | **7.270 ms** |

주기 28.6 ms 대비 합 median **24.5%**, p90 **25.4%**. **계획서 예산(7.1 ms / 25%)과 사실상 일치한다.**
오버레이가 기준선보다 싼 것은 실제 검출이 평균 3.74개로 합성 5개보다 적기 때문이다.
**NPU 동시 실행에 의한 GPU 저하는 관측되지 않았다**(네 구간 전부 기준선보다 같거나 빠르다).

### 3.2 glass-to-glass (디코드 반환 → `vkWaitForPresentKHR` 반환)

| 설정 | mean | median | p90 |
|---|---:|---:|---:|
| **`disp_source`** (지연 판정 축) | **101.13** | **101.20** | **118.36** |
| `disp_asap` | 131.42 | 130.75 | 140.56 |
| `--display-sync latest` (150프레임 확인 실행) | – | **28.2** | – |

**계획서 추정 217 ms 대비 실측 101 ms — 절반 이하다.** 같은 실행의 NPU e2e median 이 75.3 ms 이므로
**표시 경로가 더하는 지연은 약 26 ms**(GPU 변환 4.2 + present 1~2 vsync)다.

## 4. 게이트 판정

| 게이트 | 기준 | 실측 | |
|---|---|---|---|
| 회귀 (`--display off --postproc host`) | FPS 34.97 ±2%, 검출 3369 | 34.960 / 3369 | ✅ |
| 표시 off↔on FPS (asap) | ±2% | **−0.23%** | ✅ |
| 표시 off↔on 검출 | 동일 | **3369 = 3369** | ✅ |
| GPU 구간별 median 증가 | ≤ 10% | 최대 **−0.2%** (전부 감소) | ✅ |
| GPU 합 median / p90 | ≤ 9.0 / ≤ 17.2 ms | **6.995 / 7.270** | ✅ |
| glass-to-glass p90 | ≤ 180 ms | **118.36** | ✅ |
| `held_frames_max` | ≤ 10 | **4** (여유 6) | ✅ |
| `slot_timeout_count` / `display_dropped` | 0 | **0 / 0** (12회 전부) | ✅ |
| 표시 off↔on CPU (asap) | ≤ +0.5 ms | **+0.727 ms** | ❌ **미달** |

### CPU 미달 — 줄일 곳이 없음을 프로파일로 확인

`perf record -g` 로 시작 구간을 빼고 정상 구간만 비교했다. 표시를 켜면 늘어나는 것은 **syscall**
(`el0_svc` 2.42% → 4.90%)이고 그 `ioctl` 의 호출자는 **`VulkanSwapchain::present` 단 하나**다.
표시 경로에 **프레임당 힙 할당은 없다**(malloc 호출자에 표시 심볼 없음).
즉 +0.727 ms 는 "프레임마다 커맨드 버퍼를 하나 더 기록하고 present 한다"는 일 자체의 값이고,
폴링 낭비나 불필요한 복사 같은 **회수 가능한 몫이 아니다.**

**다만 절대값으로는 오히려 싸졌다.** 오늘의 배포 경로 `reg_host_off` 3.314 ms/f →
표시를 켠 `disp_asap` **1.842 ms/f**, **−1.47 ms**. **표시를 켜고도 현행보다 CPU를 덜 쓴다.**

## 5. 구현 중 잡은 버그 둘 (둘 다 교착)

1. **슬롯 대기 루프에 `flush_results()` 누락** — `matched` 모드에서 `show()` 를 거는 유일한 지점이
   `flush_results()` 인데 pump 루프에 빠져 있었다. `compute_done` 만 서고 `display_done` 이 영원히
   오지 않아 프레임 4에서 5초 타임아웃으로 죽었다.
2. **`acquire()` 가 리셋한 펜스를 `finish()` 가 무한 대기** — 백엔드 `acquire` 는 슬롯을 내주기 전
   그 슬롯의 submit 펜스를 `vkResetFences` 하는데 `DisplayConvert::finish` 는 **같은 펜스**를
   `vkWaitForFences(UINT64_MAX)` 로 기다린다. `show()` 에서 **acquire 보다 먼저** 끝난 슬롯을
   논블로킹으로 회수하도록 바꿔 해소했다. 프로브에서는 우연히 안 터졌지만 스왑체인 경로도 원리상 같은 위험이었다.

## 6. 검증

| 항목 | 결과 |
|---|---|
| 빌드 | 클린 빌드 경고·에러 **0줄** |
| ctest | **44/44** |
| Vulkan 검증 레이어 (sync validation 포함) | `--display-fullscreen` / `--display-blit` / `--display-sync latest` / **실행 중 리사이즈** 각각 **errors 0 warnings 0** |
| ASan/UBSan (별도 `build-asan/`) | 4구성 리포트 **0건**, 누수 0 |
| fd 누수 | 900프레임 실행 중 4회 샘플링 → **105개로 완전히 일정**(dma-buf 81 포함) |
| `hailort.log` | 12회 전부 error 0, 온도·클럭 warning 0 |
| **화면 도달** | `grim -o HDMI-A-1` → `samples/hdmi_boxes.png` 에 박스·라벨과 HUD `FPS 30.2 DET 3 NPU 69ms` 확인 |
| 링크 격리 | `ldd build/05_tests/test_preproc | grep -c wayland` = **0** |
| CSV 호환 | `RESULT_CSV` **22필드 그대로**, 신규 14지표는 `RESULT_CSV2` 뒤에만 |

## 7. 주의 · 남은 일

- ⚠ **`pace=asap` + 표시 on 에서 GPU timestamp 의 mean 을 인용하지 마라.** mean 이 median 보다 크게
  낮다(전처리 mean 1.102 / median 2.088, 변환 mean 2.505 / median 4.153). `pace=source` 나
  오프스크린에서는 mean ≈ median 으로 정상이다. 제출이 연달아 붙을 때 V3D 가 두 커맨드 버퍼를 한 job 으로
  합치면서 한쪽 timestamp 구간이 0 에 가깝게 잡히는 것으로 **추정**하나 확인하지 못했다.
  **이 문서의 모든 판정은 중앙값으로 했다.**
- ⚠ **CPU 비교 시 클럭을 같이 봐라.** `disp_source` 의 CPU(2.179)가 `disp_asap`(1.842)보다 큰데,
  일을 덜 하는데 ms 가 크다. 페이싱으로 프로세스가 놀아 거버너가 클럭을 내린 것으로 보인다
  (이 프로젝트가 NHWC 2.40 GHz vs NHCW 1.78 GHz 에서 이미 겪은 함정이다).
- `--display-sync latest` 는 150프레임 확인 실행만 했다. 900프레임 3회 정식 측정은 안 했다.
- 창 모드(비 전체화면) 운용은 측정하지 않았다. 쓰려면 `kUniformScale` 2.6배 비용부터 재야 한다.
- **연속 운용 발열은 미검증이다.** 이 측정은 900프레임(약 26초)이다. 카메라처럼 몇 시간 도는 운용에서
  칩이 90 °C대로 올라 클럭이 400→250 MHz 로 떨어지면 `b2/s4` 의 30 FPS 여유(+16%)가 사라질 수 있다.
- 오버레이 기준선 표기 주의: 계획서가 인용한 **0.151 ms** 는 `display_vulkan_hdmi` 의 **오프스크린 P7** 값이고,
  스왑체인 경로에서는 프로브 자체가 0.191~0.196 ms 를 낸다. 회귀가 아니라 기준선이 다르다.

## 8. 파일

- `results_raw.csv` (실행별), `results.csv` (설정별 중앙값), `env_bench.txt`, `m5a_baseline.txt`
- `raw/<설정>_run<N>/{stdout.log,hailort.log}`
- `verification/` — 검증 레이어·ASan·perf 로그
- `samples/hdmi_boxes.png` — `grim` 화면 캡처
- 재현: `bash 09_scripts/display_bench.sh`
