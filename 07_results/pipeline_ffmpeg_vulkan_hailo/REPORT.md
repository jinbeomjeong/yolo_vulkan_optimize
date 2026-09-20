# ffmpeg(HW HEVC) → Vulkan → HailoRT 파이프라인: 구현과 실험 보고서

- 작성일: 2026-09-20 (같은 날 NHCW 전처리 출력 추가 후 갱신)
- 대상: `04_data/videos/video_h265.mp4` (HEVC Main 1920x1080@30, 1816프레임), 모델 `03_models/yolov11m.hef`
- 환경: Raspberry Pi CM5 (BCM2712, Cortex-A76 4코어), kernel 6.18.50, V3D 7.1 (Mesa V3DV), Hailo-8 (PCIe Gen3 x1), HailoRT 4.24.0, libavcodec 61.19, g++ 14.2 Release

## 1. 요약

| 항목 | 결과 |
|---|---|
| 처리량 | **51.63 FPS** (dmabuf + NHCW, batch 8 / slots 16). NPU 상한(`hailortcli benchmark` 51.7 FPS)의 99.9% |
| 정확도 | GPU 전처리가 CPU 기준 구현과 **비트 단위로 일치**(NHWC·NHCW 모두, 5프레임, 최대 오차 0). 검출도 CPU 경로와 **100% 일치**(100프레임, 203/203). 전체 1816프레임 27회 실행의 검출 합계가 모두 9457개로 같다 |
| 디코드 + GPU 전처리 CPU 비용 | 약 **0.4 ms/frame** (디코드 0.25 + GPU 제출 0.1~0.15). GPU 셰이더 시간 2.13 ms (레이아웃과 무관) |
| 전체 CPU 비용 | **3.5 ms/frame** (NHCW). NHWC일 때는 17.6 ms였고, 차이 약 14 ms는 HailoRT의 입력 레이아웃 변환 단계에서 났다. NMS가 아니다 |
| 14 ms의 정체 | 전치 연산 자체가 아니라 **CPU가 V3D 메모리를 바이트 단위로 읽는 비용**이다. 같은 전치를 일반 malloc 버퍼에서 하면 약 1.2 ms다 (§4.3) |
| 입력 레이아웃 | **NHCW + dmabuf가 기본 경로**다. NHCW는 dma-buf로만 넘길 수 있다(ptr 모드는 DMA 매핑 불가) |
| 남은 CPU의 정체 | **호스트 NMS 후처리로 확정**됐다(추정 아님). 2.2~3.2 ms/frame이고 검출 개수와 무관한 고정 비용이다 (§4.5) |
| 검증 | 빌드 경고 0, ctest 11/11, Vulkan 검증 레이어 에러 0, ASan/UBSan 에러·누수 0, hailort.log error/warning 0 |

## 2. 구현

### 2.1 구조

```
video_h265.mp4
  │ libavformat demux
  ▼
FfmpegDrmDecoder ── rpi-hevc-dec(V4L2 stateless) HW 디코드, AV_PIX_FMT_DRM_PRIME
  │ DrmFrame: Y dma-buf fd + UV dma-buf fd (SAND128 NV12, 1920x1088)
  ▼
SandLetterbox (Vulkan) ── dma-buf를 VkBuffer로 import(복사 없음)
  │ 컴퓨트 셰이더 1회: de-SAND + BT.709 limited YUV→RGB + bilinear 축소 + letterbox(패딩 114)
  │ PreprocOutput: 640x640x3 RGB888 (슬롯 버퍼, host 포인터 또는 export한 dma-buf fd)
  ▼
HailoRunner (HailoRT) ── InferModel run_async, 슬롯별 bindings
  │ 출력 yolov8_nms_postprocess (NMS BY CLASS) 파싱, letterbox 역변환
  ▼
Detection (원본 1920x1080 좌표) → JSON Lines / PNG
```

### 2.2 파일

| 경로 | 역할 | 담당 에이전트 |
|---|---|---|
| `01_src/common/frame_types.hpp` | 단계 간 인터페이스(DrmFrame, PreprocOutput, LetterboxParams, Detection, make_letterbox) | 메인 → cpp-expert 보완 |
| `01_src/common/sand.hpp` | CPU de-SAND 기준 구현 | cpp-expert |
| `01_src/common/cpu_letterbox.hpp` | CPU letterbox 기준 구현(GPU 셰이더와 같은 수식) | hailo-expert |
| `01_src/common/dmabuf_map.hpp` | dma-buf CPU 읽기 매핑(DMA_BUF_IOCTL_SYNC) | vulkan-expert |
| `01_src/pipeline/ffmpeg_drm_decoder.{hpp,cpp}` | libavcodec HW 디코더 → DrmFrame | cpp-expert |
| `01_src/pipeline/tools/drm_dump.cpp` → `yvo_drm_dump` | descriptor 덤프, de-SAND 바이트 비교, 디코드 성능 | cpp-expert |
| `01_src/vulkan/vulkan_context.{hpp,cpp}` | V3D 장치 선택, 확장, 검증 메시지 카운터 | vulkan-expert |
| `01_src/vulkan/dmabuf_import.{hpp,cpp}` | dma-buf → VkBuffer import와 캐시 | vulkan-expert |
| `01_src/vulkan/sand_letterbox.{hpp,cpp}` | 전처리기(submit/wait, 슬롯, timestamp, dma-buf export) | vulkan-expert |
| `02_shaders/sand_letterbox.comp` | 전처리 컴퓨트 셰이더 | vulkan-expert |
| `01_src/vulkan/tools/preproc_bench.cpp` → `yvo_preproc_bench` | 디코드 + 전처리 성능 | vulkan-expert |
| `01_src/hailo/hailo_runner.{hpp,cpp}` | HailoRT 비동기 러너, NMS 파싱 | hailo-expert |
| `01_src/hailo/tools/hailo_selftest.cpp` → `yvo_hailo_selftest` | CPU 전처리 기반 러너 검증과 처리량 | hailo-expert |
| `01_src/pipeline/main.cpp` → `yvo_pipeline` | 통합 앱 | cpp-expert |
| `05_tests/test_preproc.cpp` | GPU 전처리 vs CPU 기준 (GTest) | vulkan-expert |
| `05_tests/test_detections.cpp` | CPU / GPU ptr / GPU dmabuf / GPU dmabuf NHCW 검출 비교 (GTest) | cpp-expert |
| `CMakeLists.txt`, `cmake/yvo_embed_spirv.cmake` | `yvo_add_shaders()`: glslc → spirv-val → SPIR-V 헤더 임베드 | vulkan-expert |
| `06_tools/bench/pipeline_bench.sh` | 성능 측정(냉각 대기, CSV 집계) | cpp-expert |

소스와 스크립트를 합쳐 약 5,200줄이다.

### 2.3 디코더 (libavcodec DRM_PRIME)

- `hw_device_ctx`(DRM)를 쓰고, `get_format`에서 DRM_PRIME만 받는다. SW로 넘어가지 않고 실패한다.
- `thread_count=0`(자동)이다. 1스레드일 때 266 FPS, 자동일 때 398 FPS다.
- `extra_hw_frames=8`로 디코더 버퍼 풀이 18개다. **동시에 약 13프레임 넘게 들고 있으면 디코더가 에러 없이 멈춘다.** 그래서 앱은 GPU 대기 중인 프레임만 붙잡는다(최대 약 4개).
- 확인한 SAND 레이아웃(Pi 5 rpi-hevc-dec):
  - `nb_objects=2`. **Y와 UV가 서로 다른 dma-buf 객체다**(Y 2,097,152 B, UV 1,048,576 B).
  - modifier는 `0x0700000000000004`(SAND128, 파라미터 0)다. **pitch 필드에는 바이트 pitch가 아니라 열 높이가 들어 있다**(Y 1088, UV 544).
  - 주소 계산: `Y(x,y) = Y[(x/128)*128*1088 + y*128 + x%128]`, `UV(x,cy) = UV[(x/128)*128*544 + cy*128 + x%128]`, 열은 15개다.
  - CPU de-SAND 결과가 `av_hwframe_transfer_data`와 프레임 0/100/900/1815에서 **바이트 단위로 일치**했다.
- `close()`에서 `avcodec_flush_buffers()`를 호출한다. 스트림 중간에 종료할 때 생기던 1448 B 누수를 고쳤다.

### 2.4 Vulkan 전처리

- **import:** `vkGetMemoryFdPropertiesKHR`로 memoryTypeBits를 구하고, `VkExternalMemoryBufferCreateInfo(DMA_BUF)`와 `dup(fd)`를 `VkImportMemoryFdInfoKHR`로 넘긴다. 메모리 타입은 0(`DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT`)이다.
  - V3DV 제약: 스토리지 버퍼 요구 크기가 실제보다 256 B 크다. 그래서 VkBuffer 범위를 256 B 줄인다.
  - 캐시: 키는 fd 번호이고, `fstat`(st_dev, st_ino)와 크기로 검증한다. 대기 중인 항목은 pin하고, 차면 LRU로 제거한다. import는 버퍼 18개 × plane 2개 = 36회뿐이고 나머지는 전부 적중한다.
  - 디코더를 닫기 전에 `clear_import_cache()`를 호출해야 디코더 버퍼가 풀린다.
- **소유권 배리어:** 제출 앞에 `FOREIGN_EXT → 컴퓨트 큐` acquire, 뒤에 release를 건다. 출력은 `SHADER_WRITE → HOST_READ` 배리어를 건 뒤 fence로 기다린다.
- **셰이더:** invocation 하나가 4픽셀을 계산해 32비트 워드 3개로 쓴다.
  - 열 방향 값(SAND 워드 주소, 바이트 시프트, 가중치, 패딩 여부)은 호스트가 계산한 LUT(20 KB)에 넣었다. CPU 기준과 같은 float 연산 순서를 써서 비트 단위로 일치시켰다.
  - 워크그룹은 32x8이다(specialization constant).
- **최적화 과정:**

| 단계 | GPU 시간 |
|---|---:|
| 초기(16x4) | 6.28 ms |
| 4픽셀 루프 전개 | 6.12 ms |
| 가중치 0 샘플 생략 | 5.07 ms |
| 열 LUT | 4.64 ms |
| lane별 분기 제거 | 3.65 ms |
| 워크그룹 32x8 | **2.13 ms** |

  V3D에서는 lane별 분기와 32비트 정수 곱셈(umul24 에뮬레이션)이 비싸다.
- **출력 레이아웃:** 스펙 상수 `kOutNhcw`(constant_id 2)로 한 셰이더가 두 레이아웃을 모두 쓴다. 픽셀 값 계산은 공유하고 **쓰기 주소와 패킹만** 갈라진다.
  - NHWC(packed RGB888): 연속 3워드. `w0 = c0.r | c0.g<<8 | c0.b<<16 | c1.r<<24` 식으로 픽셀 경계를 가로지른다.
  - NHCW: 채널마다 4픽셀이 한 워드에 깔끔히 떨어진다(`c0.r | c1.r<<8 | c2.r<<16 | c3.r<<24`). 주소는 `dst_w/4`(160워드) 간격 3곳이다.
  - **GPU 시간은 두 레이아웃이 같다**(중앙값 2.123 vs 2.124~2.128 ms, 런 간 편차 수준). 쓰기가 흩어져도 V3D에서 손해가 없었다. 셰이더 시간의 대부분이 샘플 주소 계산과 로드라 쓰기 패턴에 둔감하다.
- **letterbox 규칙:**
  - `make_letterbox`(1920x1080 → 640x360 콘텐츠, pad_y=140), 패딩 114
  - half-pixel 샘플링(`sx=(dx-pad_x+0.5)/scale-0.5`)
  - 크로마 위치 left, BT.709 limited 계수

### 2.5 HailoRT 러너

- VDevice 스케줄러는 `ROUND_ROBIN`이다. 비동기 InferModel이 스케줄러를 전제로 하기 때문이다.
- `submit(PreprocOutput, cb)`는 `slot`을 러너 슬롯으로 그대로 쓴다.
  - `dmabuf_fd >= 0`이면 `set_dma_buffer`, 아니면 `set_buffer(MemoryView)`로 연결한다.
  - `wait_for_async_ready`로 백프레셔를 건 뒤 `run_async`를 호출한다.
  - 콜백 시점이 곧 슬롯 반납 시점이다.
- dmabuf 모드에서는 시작할 때 슬롯별로 `export_dmabuf_fd` → `map_dmabuf(fd, 1228800)`를 한 번만 한다.
- **입력 format order:** `HailoRunnerConfig::input_layout`이 `Nhcw`면 `configure()` 전에 `set_format_order(HAILO_FORMAT_ORDER_NHCW)`를 건다. 그러면 HailoRT의 CPU 입력 변환이 사라진다.
  - `submit()`이 `PreprocOutput::layout`과 러너 설정이 다르면 예외를 던진다.
  - NHCW인데 dma-buf가 아니면 거부한다. 다만 **일반 malloc 포인터는 NHCW에서도 DMA 매핑에 성공한다**(실측). 매핑에 실패하는 것은 V3D GEM mmap 포인터뿐이다. 러너는 포인터 출처를 알 수 없으므로 `allow_host_ptr_nhcw`(기본 false)를 호출자가 명시적으로 켜게 했다. `yvo_hailo_selftest`만 이를 쓴다.
- **NMS BY CLASS 파싱:** 클래스 0~79 순서로 `[float count][count × {y_min,x_min,y_max,x_max,score}]`가 이어진다. 좌표는 0~1 정규화 값이다. 원본 좌표 = (정규화 × 640 − pad) / scale이고, 원본 크기로 클램프한다.

### 2.6 통합 앱 `yvo_pipeline`

- 옵션: `--video --hef --frames --mode ptr|dmabuf --layout nhwc|nhcw --batch --slots --gpu-depth --score --dump-dir --dump-every --quiet --csv-label`
- `--layout`은 전처리기와 러너에 함께 걸린다. **`--mode ptr --layout nhcw`는 시작 시점에 거부**한다(V3D 포인터는 DMA 매핑 불가). PNG 덤프는 NHCW일 때 `nhcw_to_rgb_packed()`로 되돌려 그린다.
- 메인 스레드 순서:
  1. 디코드한다.
  2. 빈 슬롯을 확보한다.
  3. GPU에 submit한다.
  4. 대기가 3개를 넘거나 fence가 끝난 슬롯이 있으면 `wait` 뒤 HailoRT에 submit한다.
- HailoRT 콜백 스레드는 결과를 큐에 넣고 슬롯을 반납한다(mutex + condvar).
- 결과는 pts 순서로 JSONL/PNG에 기록한다.
- 종료 순서: drain → `clear_import_cache` → 디코더 close → 러너 해제(unmap) → fd close

## 3. 검증

### 3.1 전처리 정확도 (`test_preproc`)
- 비교 기준: `sand.hpp` de-SAND(dma-buf mmap) → `cpu_letterbox_nv12_bt709`
- 대상 프레임: 0, 100, 500, 900, 1815

| 레이아웃 | 프레임 | max 오차 | mean 오차 | 패딩 불일치 |
|---|---|---:|---:|---:|
| NHWC | 0 / 100 / 500 / 900 / 1815 | 0 | 0.00000 | 0 |
| NHCW | 0 / 100 / 500 / 900 / 1815 | 0 | 0.00000 | 0 |

NHCW 기준값은 `cpu_letterbox_nv12_bt709()` 결과를 `rgb_packed_to_nhcw()`로 전치해 만들었다.
색 변환 기준 구현은 한 벌뿐이라 비트 일치 검증의 의미가 유지된다. 버퍼 전체 1,228,800 B가 일치했다.

### 3.2 검출 일치 (`test_detections`, 100프레임, batch 1)

| 비교 | CPU 검출 | GPU 검출 | 매칭(같은 클래스, IoU≥0.9) | 비율 |
|---|---:|---:|---:|---:|
| CPU vs GPU ptr (NHWC) | 203 | 203 | 203 | 1.0000 |
| CPU vs GPU dmabuf (NHWC) | 203 | 203 | 203 | 1.0000 |
| CPU vs GPU dmabuf (NHCW) | 203 | 203 | 203 | 1.0000 |

전체 1816프레임을 **27회**(9설정 × 3회) 실행한 결과도 검출 합계가 모두 9457개로 같았다.
레이아웃을 바꿔도 NPU에 들어가는 값이 달라지지 않는다는 뜻이다.

### 3.3 기타
- 빌드: `cmake --build build` 경고 0. 셰이더는 glslc `-Werror`와 spirv-val을 통과한다.
- `ctest --test-dir build`: 11/11 통과
- Vulkan 검증 레이어(`VK_LOADER_LAYERS_ENABLE='*validation'`): 전처리 벤치 1816프레임, 동기화 검증 300프레임, yvo_pipeline 200프레임 × 2모드 모두 에러 0
- ASan/UBSan: 디코더, 전처리, yvo_pipeline(100프레임 × 2모드), test_detections 모두 에러·누수 0
- fd 누수: 디코더 도구가 시작 전후 fd 수를 세고, 모든 실행에서 같았다.
- 시각 확인: `samples/frame00000.png` 등 7장, `selftest/frame0_detections.png`, `preproc/letterbox_frame100.png`
  (`selftest/b4/`·`b8/` 의 같은 이름 PNG 와 `preproc/letterbox_frame100_nhcw.png` 은 위 파일들과 **바이트 일치**였다 — batch 와 레이아웃이 결과를 바꾸지 않는다는 뜻이다. 2026-09-21 에 정본만 남겼고 md5 는 `../duplicate_png.md` 에 있다)

## 4. 실험 결과

### 4.1 단계별 성능
- 디코드와 전처리 수치는 `yvo_drm_dump`와 `yvo_preproc_bench`로, NPU는 `yvo_hailo_selftest`로 쟀다.

| 단계 | FPS | CPU ms/frame | 비고 |
|---|---:|---:|---|
| HW 디코드만 (libavcodec DRM_PRIME) | 398.5 | 0.23~0.28 | ffmpeg CLI는 333 FPS, 2.0 ms |
| 디코드 + GPU 전처리 | 394~395 | 0.34~0.43 | GPU 셰이더 2.13 ms (p90 2.15) |
| NPU만 (CPU 전처리 입력, batch 8 / slots 16) | 50.0 ⚠️ | – | benchmark 상한 51.7 FPS. **이 값은 과소평가다 — 정정 1 참조** |
| 참고: CPU letterbox 스칼라 | – | 약 10.0 | de-SAND 1.36 + letterbox 8.59 |
| 참고: ffmpeg 파이프 + CPU 변환 (GStreamer) | 109.2 | 23.6 | `07_results/decode_benchmark` |
| 참고: GStreamer v4l2slh265dec + 변환 | 30.3 | 39.4 | 같은 폴더 |

#### 정정 1 (2026-09-20 추가): `yvo_hailo_selftest`의 `[stream]` FPS는 과소평가다

위 표의 **"NPU만 50.0 FPS"** 는 측정 창 artifact 때문에 낮게 나온 값이다. 실제 결핍이 아니다.

`01_src/hailo/tools/hailo_selftest.cpp`의 측정 창은
`t_bench0`(warmup 프레임을 **submit** 한 시각) → `t_bench1`(마지막 프레임 **완료** 시각)이다.
마지막 submit 이후 파이프라인에 남은 `slots`개가 빠지는 시간이 창에 포함되므로

```
fps_보고 ≈ fps_참값 × bench_frames / (bench_frames − 1 + slots)
```

기본값 `bench_frames=300`, `slots=16`(batch 8이면 `max(4, 2*8)`)이면 **약 4.8% 과소평가**다.
보정하면 50.0 → 약 52.5 FPS이고, `hailortcli benchmark` 상한 51.7 FPS와 정합한다.
**즉 표의 "50.0 vs 상한 51.7" 격차는 NPU 경로의 문제가 아니라 이 도구의 계산 방식 때문이었다.**

- 영향 범위는 **`yvo_hailo_selftest`의 `[stream]` FPS 한 줄뿐**이다.
  `yvo_pipeline`(§4.2 표)은 전체 실행의 wall-clock으로 재므로 영향이 없다 — 그래서 51.63 FPS가
  benchmark 51.71과 맞아떨어진다. 지연(`latency_ms`)과 검출 수도 영향받지 않는다.
- 같은 구조를 피한 예: `07_results/gpu_postprocess/`의 `yvo_hailo_raw_probe`는 이 창 문제를
  발견해 고친 뒤 재서 **51.90 FPS**(200프레임 × 3회 중앙값)를 얻었다. 고치기 전 같은 실행이 48.57이었다.
- 도구 자체의 수정은 이 보고서 범위 밖이다. 재측정이 필요하면 별도 작업으로 잡는다.

### 4.2 전체 파이프라인 (1816프레임, 3회 중앙값)

| 설정 | 레이아웃 | FPS | CPU ms/frame | GPU ms 평균/p90 | NPU 지연 평균 (ms) | E2E 평균/p90 (ms) |
|---|---|---:|---:|---:|---:|---:|
| ptr b1/s2 | NHWC | 24.42 | 17.59 | 2.12 / 2.14 | 79.6 | 122.6 / 122.7 |
| dmabuf b1/s2 | NHWC | 24.41 | 17.39 | 2.12 / 2.14 | 79.6 | 122.6 / 122.7 |
| **dmabuf b1/s2** | **NHCW** | 24.42 | **3.29** | 2.11 / 2.13 | 79.6 | 122.5 / 122.8 |
| ptr b4/s8 | NHWC | 44.49 | 16.74 | 3.03 / 5.60 | 173.2 | 200.0 / 262.7 |
| dmabuf b4/s8 | NHWC | 44.47 | 16.76 | 3.03 / 5.61 | 173.2 | 200.0 / 262.8 |
| **dmabuf b4/s8** | **NHCW** | 44.52 | **3.17** | 3.02 / 5.65 | 173.4 | 199.9 / 261.0 |
| ptr b8/s16 | NHWC | 51.54 | 17.58 | 2.68 / 5.56 | 302.2 | 326.4 / 446.2 |
| dmabuf b8/s16 | NHWC | 51.55 | 17.61 | 2.70 / 5.56 | 302.2 | 326.4 / 446.3 |
| **dmabuf b8/s16** | **NHCW** | **51.63** | **3.52** | 2.67 / 5.57 | 303.1 | 326.3 / 443.4 |

- 27회 실행 모두 검출 9457개, `hailort.log` error·warning 0이다.
- 반복 사이 FPS 편차는 ±0.04 이내다. NHCW의 CPU는 3.08~3.66 ms 사이에서 흔들린다(측정 중 칩 온도 차이).
- batch별 FPS가 `hailortcli benchmark`(24.3 / 44.5 / 51.7)와 같다. **병목은 NPU이고, 디코드와 GPU는 NPU를 막지 않는다.**
- **NHCW로 바꿔도 FPS는 거의 그대로다**(51.55 → 51.63). 이미 NPU 상한에 붙어 있었기 때문이다.
  얻은 것은 처리량이 아니라 **CPU 코어 0.9개 → 0.16개**의 여유다. 다중 스트림이나 후단 처리를 올릴 자리가 생겼다.
- NPU 지연과 E2E(디코드 반환부터 콜백까지)는 HailoRT 큐 대기를 포함한다. 슬롯 수가 곧 큐 깊이다. 지연이 중요하면 슬롯을 줄여야 한다(예: b8/s8, 미측정).

### 4.3 CPU 부하 분석: 전치 연산인가, GPU 메모리 읽기인가

**결론: NHWC일 때의 CPU 17.6 ms/frame 중 약 14 ms는 HailoRT의 입력 변환 단계에서 났다. 다만 비싼 것은
전치 연산 자체가 아니라, 그 단계에서 CPU가 V3D 메모리를 바이트 단위로 읽는 것이다.** NHCW로 바꾸면
HailoRT가 버퍼를 DMA로 직접 가져가므로 CPU가 그 메모리를 건드리지 않고, 비용이 통째로 사라진다.

**perf** (yvo_pipeline 300프레임, batch 8, 프로세스 전체 샘플)

| 위치 | ptr (NHWC) | dmabuf (NHWC) | dmabuf (NHCW) |
|---|---:|---:|---:|
| `libhailort` 0x1c10cc 부근 (입력 transform 루프) | **78.6%** | **77.7%** | **0%** (샘플 0건) |
| `libhailort` 0x30ffac 부근 (**호스트 NMS 후처리**, §4.5에서 확정) | 8.3% | 8.2% | 35.5% |
| `ld-linux` 심볼 조회 (시작 시 동적 링크) | 5.7% | 5.8% | 25.8% |
| 커널 (DMA 캐시 관리, ioctl 등) | 1% 안팎 | 1% 안팎 | 수 % |
| **총 사이클 (Event count)** | – | **13.1 G** | **3.0 G** |

비율뿐 아니라 **총 사이클이 4.3배 줄었다**(13.1 G → 3.0 G). NHCW 열에서 남은 항목의 비중이 커 보이는 것은
전체가 작아졌기 때문이다. 시작 시 동적 링크 비용은 프레임 수와 무관한 고정 비용이다.
원자료는 `verification/perf_{ptr,dmabuf}_top.txt`와 `verification/perf_nhcw_b8_top.txt`에 있다.

**근거**

1. **gdb:** 0x1c10cc 루프에 중단점을 걸면 콜스택이 `hailort::InputTransformContext::transform → transform_inner`다.
   레지스터 값은 채널 3, 폭 640이다. **1바이트씩 stride 3으로 읽어** 연속으로 쓰는 NHWC→NHCW transpose다.
2. **NHWC에서 ptr과 dmabuf가 같다(17.6 ms):** 두 모드 모두 이 변환을 거친다. 입력을 어떻게 넘기든 HailoRT가
   CPU로 재배열하며 복사하므로, dma-buf로 넘기는 것만으로는 아무것도 절약되지 않았다.
3. **NHCW 실측:** 셰이더가 NHCW로 직접 쓰고 러너에 `set_format_order(HAILO_FORMAT_ORDER_NHCW)`를 걸면
   b8/s16에서 **51.63 FPS, CPU 3.52 ms/frame**이다(1816프레임 3회 중앙값). 검출은 NHWC와 완전히 같다.
   17.6 − 3.5 ≈ **14.1 ms/frame**이 이 단계에서 나던 비용이다.
4. **그 14 ms는 전치 연산 값이 아니다.** `yvo_hailo_selftest`는 CPU 전처리 결과를 **일반 malloc 버퍼**로 넘긴다.
   같은 모델·같은 batch에서 NHWC와 NHCW의 CPU 차이를 재니 **약 1.1~1.3 ms/frame**에 그쳤다
   (batch 8, 408회 추론, `user+sys` 3회 비교. 디코드·CPU letterbox가 양쪽에 공통이라 상쇄된다. 거친 측정이다).
   - 같은 전치가 파이프라인에서는 약 14 ms, 캐시된 malloc 버퍼에서는 약 1.2 ms다. **10배 이상 차이난다.**
   - 차이는 읽는 메모리의 종류다. 전처리 출력은 V3D가 준 메모리이고 메모리 타입이
     `DEVICE_LOCAL | HOST_VISIBLE | **HOST_COHERENT**`다(§2.4). ARM에서 이런 매핑은 대개 캐시 불가이거나
     write-combined다. 쓰기는 버퍼링되어 빠르지만 **읽기는 매번 메모리까지 간다.**
     stride 3 바이트 읽기는 캐시 라인을 전혀 못 살리는 최악의 패턴이라 그대로 노출된다.
   - 즉 NHCW의 이득은 "전치를 없앤 것"보다 **"GPU가 쓴 메모리를 CPU가 읽는 일을 없앤 것"** 이 본질이다.
5. **일반화(다음 작업에 중요):** 이 파이프라인에서 **GPU가 쓴 버퍼를 CPU로 훑는 모든 작업은 겉보기보다 약 10배 비싸다.**
   후처리를 CPU로 옮기는 설계는 이 점을 먼저 확인해야 한다. 반대로 GPU나 DMA가 그 메모리를 소비하면 비용이 없다.
6. **ptr 모드는 NHCW에서 쓸 수 없다:** V3D GEM mmap 포인터는 Hailo 드라이버가 DMA 매핑하지 못한다
   (`Ioctl HAILO_VDMA_BUFFER_MAP failed due to invalid address`). NHWC에서 ptr이 동작하던 이유는 변환 단계의
   CPU 복사 덕분이었다. 따라서 **zero-copy는 dmabuf + NHCW에서만 성립한다.**
   - 다만 실패하는 것은 V3D 포인터이지 호스트 포인터 일반이 아니다. 일반 malloc(페이지 정렬) 버퍼는
     NHCW에서도 DMA 매핑에 성공한다(`yvo_hailo_selftest`로 확인).

**비용 분해 (b8/s16, 프레임당)**

전체 1816프레임 프로파일에서 심볼별로 나눈 값이다(시작 비용이 희석돼 있다).
`07_results/nms_host_cost/evidence/perf_{nhwc,nhcw}_1816_top.txt`에서 뽑았다.

| 항목 | NHWC | NHCW |
|---|---:|---:|
| HailoRT 입력 변환 (`0x1c10xx`, 대부분 V3D 메모리 읽기) | 14.57 (82.8%) | **0** |
| HailoRT 출력 처리 = 호스트 NMS (`0x30ffxx`) | 1.50 (8.5%) | 2.22 (58.2%) |
| 시작 시 동적 링크 (프레임 수와 무관한 고정 비용) | 0.18 | 0.27 |
| 나머지 (디코드 + GPU 제출 + 앱 + 드라이버) | 1.35 | 1.33 |
| **합계 (해당 실행의 실측)** | **17.59** | **3.82** |

"나머지"가 1.35 vs 1.33으로 거의 같은 것이 분해가 제대로 됐다는 방증이다.

**ms 값을 두 열 사이에서 직접 비교하면 안 된다.** 두 실행의 CPU 클럭이 달랐다(NHWC 2.40 GHz,
NHCW 1.78 GHz). NHCW는 프로세스가 코어의 20%만 써서 거버너가 클럭을 내린다.
**사이클로 보면 출력 처리가 3.60 vs 3.96 Mcycles/frame으로 같은 일**이고, ms 차이는 클럭 탓이다.

남은 "출력 처리"는 §4.5에서 추정이 아니라 실측으로 호스트 NMS임을 확정했다.

### 4.4 온도
- 측정 사이에 Hailo 칩이 75 °C를 넘으면 65 °C까지 식혔다. 17회 대기했고, 합계 1230초(회당 50~95초)였다.
- 실행이 끝날 때 칩 온도는 b1 약 76.5 °C, b4 약 81 °C, b8 약 82 °C였다.
- 냉각 없이 연속 실행하면(dmabuf b8, 1816프레임) 81.4 → **92.6 °C**까지 올랐다. 이번에는 클럭 저하 경고가 없었고 FPS는 51.9를 유지했다.
- SoC는 68~74 °C였다. `get_throttled=0x80000`은 과거 기록 비트이고, 측정 중 스로틀링은 없었다.
- 전체 측정은 약 37분(실행 902초 + 냉각 1230초)이었다.
- NHCW 3설정 × 3회 추가 측정(2026-09-20 09:15~09:33)에서도 칩은 실행 후 75~86 °C였고, 냉각 대기는 합계 440초였다.
  9회 모두 `hailort.log`에 온도·클럭 경고가 없었다. **CPU 부하가 1/5로 줄어도 칩 온도는 그대로다**(발열원은 NPU다).

### 4.5 남은 CPU의 정체: 호스트 NMS 후처리 (확정)

별도 실험 `07_results/nms_host_cost/`에서 `libhailort` 0x30ffac 계열이 **호스트 NMS 후처리임을 확정**했다.
이전 판의 "NMS로 추정"을 대체한다.

**확정 근거**

| 근거 | 내용 |
|---|---|
| HEF 사양 | `hailortcli parse-hef --parse-streams` 기준 출력이 `yolov8_nms_postprocess`, `Op YOLOV8 Post-Process`, 80 클래스. HailoRT가 **호스트 CPU**에서 실행한다 |
| gdb | HailoRT가 `std::thread`로 띄운 내부 워커 스레드에서 돈다(메인 스레드도, 우리 코드도 아님). 진입 시 `x1 = 0x50 = 80`으로 **클래스 개수와 일치** |
| 레이아웃 무관 | NHWC 3.60, NHCW 3.96 Mcycles/frame으로 거의 같다. 입력 경로와 무관한 **출력 쪽 작업**이다 |
| 우리 코드 아님 | `parse_nms_by_class` + letterbox 역변환은 프로파일 0.1% 아래로 잡히지 않는다 |

**검출 개수와 무관한 고정 비용이다.** score threshold를 올려 검출을 2090개 → 0개로 없애도 CPU가 변하지 않았다.

| score | 검출 | CPU ms/frame |
|---|---:|---:|
| 0.2 | 2090 | 3.256 |
| 0.9 | 2 | 3.366 |
| 0.99 | 0 | 3.363 |

억제(suppression)가 아니라 **원시 텐서 전체를 훑는 작업**이 비용이다. 앵커 8,400개(80²+40²+20²) ×
(클래스 80 + DFL 64) ≈ 120만 개 값을 매 프레임 dequantize하며 스캔해야 하고, 몇 개가 살아남든 동일하다.

**비용 크기.** 두 가지 방법으로 쟀고 결과가 다르다.

| 방법 | 값 | 포함 범위 |
|---|---:|---|
| 파이프라인 perf의 `0x30ffxx` 심볼 집계 | 2.22 ms/frame | NMS 연산만 |
| `hailortcli run2`의 `full_async` − `raw_async` | 3.15 ms/frame | NMS 연산 + 출력 변환(UINT8→FLOAT32) + 160 KB 출력 버퍼 처리 |

여기에 CPU 클럭 차이도 섞인다(둘 다 코어 점유율이 낮아 거버너가 클럭을 내린다).
**"2.2~3.2 ms/frame"이 정직한 범위**다. 프레임 예산 19.34 ms의 약 11~16%이고, 코어 4개 기준 약 3~4%다.

**원시 출력으로 바꿔도 PCIe 트래픽은 늘지 않는다.** 장치는 이미 원시 텐서 6개(합계 1,221,120 B)를
매 프레임 내보내고 있고, `yolov8_nms_postprocess`는 그것을 호스트에서 160,320 B로 줄이는 VStream 연산이다.
실측에서도 `raw_async`가 `full_async`보다 FPS가 **0.78% 높았다**(51.62 vs 51.22, 범위 겹치지 않음).

자세한 내용과 원자료는 [`07_results/nms_host_cost/summary.md`](../nms_host_cost/summary.md)에 있다.

## 5. 결론
1. **목표를 달성했다.** HW HEVC 디코드 → Vulkan 전처리 → HailoRT 파이프라인이 NPU 상한(51.7 FPS)의 99.9%인 **51.63 FPS**로 돈다. 정확도는 CPU 기준 경로와 완전히 같다.
2. **디코드와 전처리의 CPU 비용은 거의 사라졌다.** CPU 경로 23.6~39.4 ms/frame이 약 0.4 ms/frame으로 줄었다. GPU 셰이더는 2.13 ms로 NPU 주기(19 ms)보다 훨씬 짧다.
3. **입력 레이아웃을 NHCW로 맞춰 CPU를 17.6 → 3.5 ms/frame으로 줄였다(5배).** 모델의 장치 입력 형식이 NHCW라서,
   NHWC로 넘기면 HailoRT가 CPU로 재배열한다. 셰이더가 처음부터 NHCW로 쓰면 그 단계가 통째로 없어진다.
   **처리량은 이미 NPU 상한이라 거의 그대로이고, 얻은 것은 CPU 코어 0.9개 → 0.16개의 여유다.**
4. **비싼 것은 전치가 아니라 GPU 메모리를 CPU로 읽는 것이었다(§4.3).** 같은 전치를 캐시된 malloc 버퍼에서 하면
   약 1.2 ms다. V3D의 `HOST_COHERENT` 매핑을 stride 3으로 읽어서 10배 이상 비싸졌다.
   → **이 파이프라인에서 GPU가 쓴 버퍼를 CPU로 훑는 설계는 겉보기보다 약 10배 비싸다.** 후처리 설계 때 먼저 따진다.
5. **남은 CPU는 호스트 NMS 후처리로 확정됐다(§4.5).** 2.2~3.2 ms/frame이고 검출 개수와 무관한 고정 비용이다.
   원시 출력으로 전환하면 이 비용이 사라지며, **PCIe 트래픽은 늘지 않는다**(원시 텐서가 이미 건너오고 있다).
   `raw_async` 실측에서 FPS가 오히려 0.78% 높았다.
6. **다음 단계:**
   - 원시 출력 + GPU NMS는 **타당하지만 처리량을 올리지 못한다.** 병목이 PCIe/NPU이기 때문이다.
     사는 것은 CPU 코어 약 0.2개다. 후단 처리(다중 스트림, 트래킹)를 올릴 계획이 있을 때 착수한다.
     구현 난이도는 `InferModel`이 postprocess를 건너뛸 수 있는지에 달려 있다(안 되면 raw stream API로 러너 재작성).
   - 확보한 CPU 여유로 다중 스트림을 시도한다. 병목이 NPU이므로 스트림을 늘려도 합계 FPS는 51.7이 상한이다.
   - 지연이 중요한 설정(b8/s8, b4/s4)을 측정한다.

## 6. 남은 이슈
- ~~0x30ffac 부근 함수를 확정한다~~ → **해소.** 호스트 NMS 후처리로 확정했다(§4.5, `07_results/nms_host_cost/`).
- 원시 출력 전환 시 `InferModel`이 postprocess를 건너뛸 수 있는지 미확인이다. 불가하면
  `ConfiguredNetworkGroup`의 raw stream API로 러너를 상당 부분 다시 써야 한다.
- §4.3의 "V3D 메모리 읽기가 비싸다"는 해석은 malloc 버퍼와의 비교(거친 측정)에 근거한다.
  V3D 버퍼를 CPU로 순차 읽기 vs stride 3 읽기로 대역폭을 직접 재면 확정할 수 있다.
- 지연을 최적화하는 설정(b8/s8, b4/s4 등)은 아직 측정하지 않았다.
- 연속 운용 시 Hailo 칩이 90 °C대까지 오른다. 방열 대책이 필요하다.
- GPU를 단발로 실행하면 DVFS 때문에 셰이더 시간이 5.5 ms까지 늘어난다. 연속 스트림에서는 2.1 ms다.
- 별도 보류 과제: Gen3에서 칩을 식힌 상태로 overall latency를 다시 재야 한다(`07_results/npu_benchmark/gen3/summary.md`).

## 7. 재현 방법
```bash
cmake -S . -B build -G Ninja && cmake --build build
ctest --test-dir build
# 권장 경로: dmabuf + NHCW
build/01_src/pipeline/yvo_pipeline --video 04_data/videos/video_h265.mp4 \
    --mode dmabuf --layout nhcw --batch 8 --slots 16 --dump-dir /tmp/out --dump-every 300
bash 06_tools/bench/pipeline_bench.sh                       # 전체 9설정(약 55분, 냉각 포함)
ONLY='nhcw' APPEND=1 bash 06_tools/bench/pipeline_bench.sh  # NHCW만 재서 기존 결과에 추가
RUNS=1 FRAMES=300 bash 06_tools/bench/pipeline_bench.sh     # 짧게
```

## 8. 결과 파일 (`07_results/pipeline_ffmpeg_vulkan_hailo/`)
- `results.csv`(설정별 중앙값), `results_raw.csv`(실행별 값, 냉각 대기와 hailort.log 경고·에러 수 포함), `env_bench.txt`
- `raw/<설정>_run<N>/{stdout.log,hailort.log}`
- `samples/`: `detections.jsonl`, 박스를 그린 PNG 7장 (NHWC 경로로 생성. NHCW 경로의 PNG·JSONL과 바이트 단위로 같음을 확인했다)
- `verification/`: 검증 레이어 로그, ASan 로그, `perf_{ptr,dmabuf,nhcw_dmabuf}_top.txt`, 정식 NHCW 경로의 `perf_nhcw_b8_top.txt`
- `selftest/`: HailoRT 러너 단독 검증, `preproc/`: 전처리 샘플

관련 실험: [`07_results/nms_host_cost/`](../nms_host_cost/summary.md) — 호스트 NMS 비용 확정과 원시 출력 전환 타당성
