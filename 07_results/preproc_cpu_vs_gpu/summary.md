# 전처리 CPU 구현 대 GPU 셰이더 — 같은 조건 비교 (2026-10-27)

1080p HEVC 디코드 프레임(DRM_PRIME dma-buf, SAND128 NV12) → 640×640 RGB 레터박스(패딩 114) → NPU 입력 NHCW uint8.
이 변환을 CPU 구현들과 GPU 셰이더 `02_shaders/sand_letterbox.comp`가 **같은 입력 프레임에서 같은 출력 형식으로** 만들게 하고 비교했다.
작업 승인: 2026-10-27(사용자).

## 1. 측정 조건

- 날짜: 2026-10-27 14:30~14:49(KST). 하드웨어: Raspberry Pi Compute Module 5 Rev 1.0, kernel 6.18.50+rpt-rpi-2712, Mesa 26.2.2, FFmpeg(rpi) 7.1.5 rpt2(libswscale 8.3.100)
  - V3D 1150 MHz(`v3d_freq=1150`), ARM 2400 MHz. **거버너 performance**(측정 동안만 고정, 끝나고 ondemand로 원복. 직전 [`memory_asymmetry`](../memory_asymmetry/summary.md)는 ondemand)
  - NPU 미사용. 표시는 쓰지 않았다(데스크톱 labwc는 떠 있었고 유휴). PCIe Gen3 x1(기록용)
- 코드: HEAD `cd6c8bd` + 미커밋 변경. 측정 바이너리 md5 `0f46cad4fabaebe2d01d1d0f660f50e1`([`env_bench.txt`](env_bench.txt))
  - 도구 `yvo_preproc_bench --impl …`(`01_src/tools/vulkan/preproc_bench.cpp` `run_impl`), CPU 헬퍼 `01_src/toolkit/{cpu_preproc,sws_preproc}.hpp`
  - 실행기 `06_tools/probes/preproc_cmp_run.sh`, 집계 `06_tools/probes/preproc_cmp_aggregate.py`
- 입력: `04_data/videos/video_h265.mp4`. 디코더 프레임 스레드를 껐다(`threads=1`). 그래서 측정 구간 안에서 디코더 스레드가 돌지 않는다.
  실행마다 워밍업 60프레임 뒤 **600프레임**을 쟀다. 디코드 시간은 측정 구간 밖이다.
- 입력 dma-buf: Y 2,097,152 B·UV 1,048,576 B, 둘 다 exporter `system`(캐시 dma-heap), SAND 열 높이 1088/544.
  CPU 구현은 매 프레임 `DMA_BUF_SYNC START|READ … END|READ`로 감싸 읽는다(map 단계에 포함). 매핑은 inode 키로 캐시한다(`cached`). 대조로 `perframe`도 쟀다.
- 출력: CPU는 **캐시 dma-heap**(`/dev/dma_heap/vidbuf_cached` → system, 실제 NPU 입력과 같은 종류) 4개 링에 `START|WRITE … END|WRITE`로 쓴다(sync 단계). 대조로 malloc(aligned 16384)도 쟀다.
  GPU는 `yvo_pipeline --mode dmabuf`와 같은 V3D 메모리 export 버퍼(8슬롯)를 쓴다.
- 반복: 설정 13종 × 독립 실행 3회. 실행이 바깥 루프, 설정이 안쪽 루프다. 1스레드 설정과 gpu는 `taskset -c 3`, 4스레드 설정은 `taskset -c 0-3`.
- 지표(프레임마다): wall = steady_clock(전처리 전체), CPU = `CLOCK_PROCESS_CPUTIME_ID` 차(전 스레드 합), user/sys = `getrusage` 차.
  표의 값은 **실행별 중앙값(CPU는 평균)의 3회 중앙값**이다. 범위는 실행별 값의 최소~최대.
- 냉각: 실행 전마다 SoC ≤ 70 °C(프로젝트 기준 ≤ 75 °C)와 `get_throttled` 현재 비트 0을 확인했다. 실행 시작 67.5~71.4 °C, 종료 70.3~77.4 °C.
  39회 모두 실행 뒤 `throttled=0x80000`(과거 비트만), 냉각 대기 합계 725 s.
- 정확도: 별도 패스(시간 측정 없음)에서 처음 660프레임을 **매 프레임** CPU-ref 1스레드 출력과 바이트 대조했다([`evidence/accuracy/`](evidence/accuracy/)).
  "콘텐츠"는 640×360×3 그림 영역, "전체"는 640×640×3(패딩 포함)이다.

### 구현

| 이름 | 내용 |
|---|---|
| CPU-ref | `01_src/ref/` 기준 구현 그대로: `desand_nv12` → `cpu_letterbox_nv12_bt709`(스칼라, 패딩 포함) → `rgb_packed_to_nhcw`. 4스레드는 de-SAND를 행, 레터박스를 콘텐츠 행 기준 균등, 전치를 행으로 나눈다(ref 함수를 포인터만 옮겨 호출 → 1스레드와 바이트 동일). `mem_threads`는 메모리 단계(de-SAND·전치)의 스레드 수 |
| CPU-swscale rgb24 | rpi-ffmpeg `av_rpi_sand_to_planar_y8/_c8`(SAND → YUV420P, NEON) → libswscale `bilinear` YUV420P→RGB24 640×360(BT.709 limited→full, 크로마 left) → 패딩 memset → NEON 전치(`vld3q_u8`). 크로마는 swscale 기본값대로 가로 절반 해상도 |
| CPU-swscale gbrp | 같은 평면화 → swscale가 GBRP 평면을 **NHCW 텐서의 R/G/B 행 구간에 바로** 쓴다(linesize 1920) → 전치 없음. swscale가 전체 크로마 보간으로 바꾼다 |
| CPU-swscale fast_bilinear | rgb24와 같고 `sws_flags=fast_bilinear` |
| 4스레드(swscale) | swscale 슬라이스 스레드(`threads=4`, `sws_scale_frame` 경로). 평면화·전치는 1스레드(`mem_threads 1`, §3 참고) |
| GPU | `sand_letterbox.comp`(local 32×8) 동기 submit + wait(fence). device 시간은 timestamp |

## 2. 결과

### 2.1 구현별 wall·CPU·정확도 ([`results.csv`](results.csv), [`evidence/medians.txt`](evidence/medians.txt))

| 설정 | 코어 | wall 중앙값 ms (범위) | wall p90 ms | CPU ms/f (범위) | user / sys | CPU-ref 대비 (콘텐츠) |
|---|---|---:|---:|---:|---:|---|
| `cpuref_t1` | 3 | 10.595 (10.569–10.616) | 10.745 | 10.613 (10.605–10.676) | 10.394 / 0.286 | 기준 |
| `cpuref_t4` (mem 1) | 0–3 | 4.132 (4.130–4.163) | 4.316 | 10.782 (10.765–10.806) | 10.520 / 0.251 | 바이트 동일 660/660 |
| `cpuref_t4_mem4` | 0–3 | 4.553 (4.546–4.573) | 4.700 | 16.938 (16.925–16.989) | 16.548 / 0.393 | 바이트 동일 660/660 |
| `sws_rgb24_t1` | 3 | 5.462 (5.461–5.540) | 5.593 | 5.501 (5.500–5.551) | 5.256 / 0.248 | PSNR 38.37 dB · 평균 2.354 · 최대 144 · 동일 4.52% |
| `sws_rgb24_t4` | 0–3 | 2.621 (2.580–2.624) | 2.730 | 5.835 (5.802–5.836) | 5.548 / 0.275 | t1과 바이트 동일(해시) |
| `sws_gbrp_t1` | 3 | 6.846 (6.837–6.846) | 6.956 | 6.871 (6.864–6.880) | 6.646 / 0.237 | PSNR 42.17 dB · 평균 0.723 · 최대 137 · 동일 60.12% |
| `sws_gbrp_t4` | 0–3 | 2.768 (2.733–2.775) | 2.852 | 7.169 (7.140–7.189) | 6.911 / 0.283 | t1과 바이트 동일 |
| `sws_fastbil_t1` | 3 | 3.537 (3.519–3.564) | 3.721 | 3.584 (3.573–3.615) | 3.366 / 0.254 | PSNR 33.50 dB · 평균 4.118 · 최대 199 · 동일 9.08% |
| `sws_fastbil_t4` | 0–3 | 2.109 (2.105–2.116) | 2.193 | 3.926 (3.919–3.927) | 3.582 / 0.344 | t1과 바이트 동일 |
| **`gpu`** | 3 | **1.810** (1.810–1.812) | **1.822** | **0.081** (0.079–0.082) | 0.049 / 0.034 | 656/660 프레임 동일, 차이 총 **4바이트**(G 채널 ±1) |
| 대조 `cpuref_t1_permap` | 3 | 10.622 (10.617–10.658) | 10.818 | 10.667 | 10.376 / 0.291 | — |
| 대조 `cpuref_t1_malloc` | 3 | 10.556 (10.535–10.567) | 10.672 | 10.573 | 10.312 / 0.276 | — |
| 대조 `sws_rgb24_t1_malloc` | 3 | 5.487 (5.467–5.502) | 5.623 | 5.526 | 5.255 / 0.275 | — |

- swscale 평면화 결과(YUV420P)는 매 프레임 ref de-SAND와 바이트가 같다(차이 0바이트, `flatten_vs_ref_desand_diff_bytes`).
- GPU의 4바이트: 프레임 228·433·502·597에서 각 1바이트, 모두 G 채널(55↔56, 213↔214)([`evidence/gpu_mismatch_detail.log`](evidence/gpu_mismatch_detail.log)).
  ctest `test_preproc`가 보는 5프레임(0·100·500·900·1815)에는 없다.
- 표본 그림(프레임 100): [`samples/`](samples/). cpuref t1·t4와 gpu는 md5가 같다.

### 2.2 단계 분해 (중앙값 ms, 3회 중앙값)

| 설정 | map | 평면화 | 리사이즈+색변환 | 패딩 | 전치 | sync(출력) |
|---|---:|---:|---:|---:|---:|---:|
| `cpuref_t1` | 0.116 | 1.169 | 8.856 (패딩 포함) | — | 0.433 | 0.004 |
| `cpuref_t1_permap` | 0.164 | 1.149 | 8.854 | — | 0.430 | 0.004 |
| `cpuref_t1_malloc` | 0.116 | 1.140 | 8.839 | — | 0.435 | 0.000 |
| `cpuref_t4` | 0.116 | 1.203 | 2.287 | — | 0.492 | 0.005 |
| `cpuref_t4_mem4` | 0.132 | 1.596 | 2.294 | — | 0.523 | 0.005 |
| `sws_rgb24_t1` | 0.126 | 1.060 | 4.031 | 0.058 | 0.187 | 0.003 |
| `sws_rgb24_t1_malloc` | 0.127 | 1.055 | 4.053 | 0.058 | 0.186 | 0.000 |
| `sws_rgb24_t4` | 0.127 | 1.110 | 1.106 | 0.059 | 0.213 | 0.004 |
| `sws_gbrp_t1` | 0.124 | 0.969 | 5.684 | 0.058 | — | 0.004 |
| `sws_gbrp_t4` | 0.128 | 1.074 | 1.499 | 0.059 | — | 0.005 |
| `sws_fastbil_t1` | 0.126 | 1.053 | 2.115 | 0.059 | 0.182 | 0.003 |
| `sws_fastbil_t4` | 0.127 | 1.065 | 0.626 | 0.059 | 0.223 | 0.004 |
| `gpu` | submit 0.067 | wait 1.743 | device(timestamp) 1.757 | | | |

- map = 디코더 dma-buf 매핑 + `DMA_BUF_SYNC` READ start/end(3 MB 캐시 무효화). 평면화 = SAND → 선형 평면.

## 3. 결론

**관찰**
1. **스칼라 CPU "약 10 ms"는 그대로 재현된다** — 10.60 ms. [`REPORT.md`](../pipeline_ffmpeg_vulkan_hailo/REPORT.md) §4.1의 10.0 ms(de-SAND 1.36 + 레터박스 8.59)에 NHCW 전치 0.43 ms가 더해진 구성이다.
   프레임마다 mmap한 대조(옛 측정 방식)의 map+평면화가 1.31 ms로 옛 값 1.36에 가깝다. 레터박스는 8.86 ms(+3%, 거버너·빌드가 다르다).
2. **최적화한 CPU는 스칼라의 1/2~1/3이다.** swscale bilinear 1스레드 5.46 ms, fast_bilinear 1스레드 3.54 ms(CPU도 같은 양).
   GStreamer `videoconvertscale`의 "약 7~8 ms/f"([`decode_benchmark`](../decode_benchmark/summary.md), S2−S1 간접값)보다 낮다.
   다만 그쪽은 ondemand·I420 입력·변환 4스레드 조건이어서 직접 비교는 아니다.
3. **4스레드는 wall만 줄이고 CPU 총량은 줄이지 않는다.** 리사이즈 단계는 3.4~3.9배 확장된다(레터박스 8.86→2.29, swscale 4.03→1.11, gbrp 5.68→1.50, fast 2.12→0.63).
   반면 CPU ms/f는 5.50→5.84(rgb24), 3.58→3.93(fast)으로 오히려 는다.
4. **메모리 단계는 확장되지 않는다.** 평면화(약 1.0~1.2 ms) + map(0.12) + 전치(0.18~0.43)가 스레드와 무관한 바닥이다.
   넷으로 나누면 평면화 1.17→1.60 ms로 더 느려지고 CPU가 +6.2 ms/f(`cpuref_t4_mem4` 대 `cpuref_t4`) 는다. 그래서 4스레드 정식 설정은 메모리 단계를 1스레드로 두었다.
5. **CPU 최선 대 GPU.** GPU wall 1.810 ms(device 1.757)는 CPU 최속 `sws_fastbil_t4` 2.109 ms보다 14% 빠르다.
   정확도가 가장 가까운 swscale(`sws_gbrp_t4`, PSNR 42.2 dB) 2.768 ms보다는 35% 빠르다.
   프로세스 CPU는 **0.081 ms/f** — CPU 구현의 1/44(fast t4 3.93)~1/131(ref t1 10.61)이고, 정확도가 비슷한 1스레드 swscale 대비 1/68(rgb24)·1/85(gbrp)다.
   30 fps 기준 CPU 점유는 GPU 0.24%, swscale 1스레드 16.5~20.6%(한 코어 기준)다.
6. **출력 할당처는 CPU 비용을 바꾸지 않는다.** 캐시 dma-heap과 malloc의 차이는 ±0.04 ms 안이고, 출력 sync(END|WRITE의 dcache clean 포함)는 0.004 ms다.
   입력 쪽 READ sync(캐시 무효화)는 매 프레임 0.12 ms다. 프레임마다 mmap하면 0.048 ms가 더 든다.
7. **정확도.** CPU-ref 4스레드는 1스레드와 바이트 동일하고, swscale 1·4스레드도 서로 바이트 동일하다.
   swscale는 필터 정의(축소 때 지지 폭이 넓어지는 bilinear, 가로 크로마 절반)가 ref와 달라서 PSNR 33.5~42.2 dB다.
   GPU는 ref와 **비트 일치가 아니다** — 660프레임(811,008,000 B) 중 4바이트가 ±1이다.
8. GPU 전처리 시간은 1150 MHz 파이프라인 b1 값([`bootloader_20260925`](../pipeline_ffmpeg_vulkan_hailo/runs/bootloader_20260925/summary.md) 1.81 ms)과 같다.
   옛 2.13 ms(960 MHz)·[`isp_resize`](../isp_resize/summary.md) 표 1의 wall 2.081 / device 2.129 ms보다 빠르다(클럭 차이로 추정).

**추정**
- 메모리 단계가 확장되지 않는 원인은 DRAM 대역폭 상한으로 본다. 진단 마이크로벤치(정식 측정 아님, 보관 안 함)에서 malloc 소스의 de-SAND도 4스레드가 1스레드보다 느렸다(1.42 대 1.02 ms).
- GPU의 G 채널 ±1은 부동소수 반올림 경계(FMA 축약·연산 순서 차이)로 추정한다. 확인하지 않았다.

## 4. 주의·남은 일
- 이 비교는 **동기 1프레임** 기준이다. 실제 파이프라인에서 GPU는 depth 2로 겹쳐 돌아 wall이 가려진다. CPU 4스레드는 다른 단계 스레드와 코어를 다툰다. 둘 다 재지 않았다.
- 디코더가 같은 프로세스에서 HW 디코드를 앞서 진행할 수 있다(메모리 대역폭 공유). 모든 설정에 똑같이 해당한다.
- swscale 4스레드 설정의 `mem_threads 1` 선택은 스모크(1회, 보관 안 함)에 근거한다. cpuref는 정식 데이터(§2.1 `cpuref_t4` 대 `cpuref_t4_mem4`)가 있다.
- rpi-ffmpeg `av_rpi_sand_to_planar_y8/_c8`은 `_x == 0`(NEON) 경로에서 `y` 인자를 무시한다(합성 버퍼로 확인, libavutil 59.39.100 rpt2). 그래서 행 띠는 소스 포인터를 `r0 × 128` B 옮기고 `y = 0`으로 넘긴다(`01_src/toolkit/sws_preproc.hpp`).
- 측정 뒤 도구에 정확도 패스의 불일치 출력과 PNG 이름 태그만 추가했다(시간 측정 경로는 같음). 현재 바이너리 md5 `2247f4a0a3861f71d55e90f3488eafe6`.
  [`evidence/gpu_mismatch_detail.log`](evidence/gpu_mismatch_detail.log)는 그 사이의 중간 바이너리(정식 바이너리 + 불일치 출력만, md5 `318c977ed97a8e25c519c820897ee77c`, 로그 머리에 기록)로 만들었다.
- 처음 생성 때 `samples/`의 PNG 이름이 sws 플래그를 구분하지 못해 fast_bilinear 그림이 bilinear rgb24 그림을 덮어썼다. 그 두 장과 gbrp 두 장은 내용에 맞는 이름으로 옮겼고, bilinear rgb24 두 장은 고친 바이너리로 다시 만들었다.
- 검증: 검증 레이어를 켠 gpu 설정 200프레임에서 errors 0 / warnings 2(로더 메시지)([`verification/validation_impl_gpu_200f.log`](verification/validation_impl_gpu_200f.log)).
- 측정 위임: 프로젝트 `vulkan-expert`가 세션에 로드되지 않아 전역 `cpp-expert`가 `vulkan-expert.md` 규약을 따라 구현·측정했다. 본문은 그 보고를 메인 에이전트가 원자료와 대조해 저장했다(불일치 로그의 바이너리 md5 한 곳을 바로잡음).
- 결정: GPU 비트 일치 서술은 문서에서 "검사 프레임 5장 기준, 전 프레임 대조 시 660프레임 중 4바이트 ±1"로 고치고, 셰이더·`01_src/ref/`는 고치지 않는다 (사용자 결정, 2026-10-27). 반영: `README.md`, `08_learn/08_method.md`, `08_learn/09_code_map.md`.

## 5. 정정 (2026-10-27, 메인 에이전트 — 위 본문은 고치지 않는다)

- §3 결론 5의 "1/44(fast t4 3.93)"는 근거 표기가 틀렸다. CPU 구현 중 CPU ms/f가 가장 작은 것은 `sws_fastbil_t1` **3.584**이고, 3.584 / 0.081 ≈ **44**다. `sws_fastbil_t4`(3.926)로 계산하면 약 48이다. 범위 "1/44~1/131"은 맞다.
