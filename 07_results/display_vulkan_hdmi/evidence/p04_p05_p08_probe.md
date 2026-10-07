# P4 / P5 / P8 프로브 결과 — 표시 트랙 2단계 ([4] VulkanSwapchain, [5] display_convert.comp)

측정일 2026-09-20. Raspberry Pi 5 / CM5, V3D 7.1.10.2 (Mesa 26.x, Vulkan 1.3),
labwc(Wayland) on HDMI-A-1 1920x1080@60, SSH 에서 `XDG_RUNTIME_DIR=/run/user/1000 WAYLAND_DISPLAY=wayland-0`.
칩 온도 66~69 °C (측정 중 `vcgencmd get_throttled` = 0x80000 = 과거 기록 비트만).
NPU 유휴(`ps -eo comm | grep yvo_` 0건) 상태에서 측정했다.

빌드: `build/bin/yvo_display_probe` (Release, `-mcpu=cortex-a76`).

---

## P4 `--probe surface` — 전부 통과

```
XDG_RUNTIME_DIR=/run/user/1000 WAYLAND_DISPLAY=wayland-0 \
  build/bin/yvo_display_probe --probe surface --display-fullscreen
```

| 기준 | 결과 | 실측값 |
|---|---|---|
| ① family 0 presentation support | **PASS** | `vkGetPhysicalDeviceWaylandPresentationSupportKHR` = VK_TRUE, `vkGetPhysicalDeviceSurfaceSupportKHR` = VK_TRUE |
| ② `R8G8B8A8_UNORM` + `SRGB_NONLINEAR` | **PASS** | 포맷 14개 중 존재 (7개 × SRGB_NONLINEAR / OTHER) |
| ③ `supportedUsageFlags` 에 STORAGE | **PASS** | `0x8009f` — TRANSFER_SRC/DST, SAMPLED, **STORAGE**, COLOR_ATTACHMENT, INPUT_ATTACHMENT, FEEDBACK_LOOP |
| ④ FIFO present mode | **PASS** | MAILBOX / FIFO / IMMEDIATE (FIFO_RELAXED 없음) |
| ⑤ `vkGetPhysicalDeviceImageFormatProperties2(R8G8B8A8_UNORM, OPTIMAL, STORAGE\|COLOR_ATTACHMENT\|TRANSFER_DST)` | **PASS** | `VK_SUCCESS`, maxExtent 7680x7680, maxMipLevels 13, maxArrayLayers 2048 |

부수 실측:
- `currentExtent` = `0xFFFFFFFF` → **이미지 크기 = 창 크기를 우리가 맞춘다** (계획서 §5.6 확인)
- image extent min 1x1 / max 7680x7680
- 기본 caps `minImageCount` = **4**, `maxImageCount` = 0(무제한)
- **present mode 별 `minImageCount`** (`VK_EXT_surface_maintenance1` + `VK_KHR_get_surface_capabilities2`):
  MAILBOX **4** / FIFO **3** / IMMEDIATE **3** — 계획서 예측과 정확히 일치
- blit 폴백 가능: `R8G8B8A8_UNORM` optimal tiling 에 BLIT_SRC=1, BLIT_DST=1

**→ 컴퓨트가 스왑체인 이미지에 직접 쓰는 경로가 성립한다. blit 폴백은 불필요하지만 `--display-blit` 으로
항상 테스트 가능하게 남겼다.**

### ⚠ 계획서와 다르게 한 것: 이미지 개수는 기본 4를 쓴다 (FIFO 3 은 opt-in)

FIFO 에서 `minImageCount = 3` 으로 스왑체인을 만들면 **검증 레이어 error 가 난다**:

```
[vulkan ERROR] vkCreateSwapchainKHR(): pCreateInfo->minImageCount 3, which is outside the bounds
returned by vkGetPhysicalDeviceSurfaceCapabilitiesKHR() (i.e. minImageCount = 4, maxImageCount = 0).
VUID-VkSwapchainCreateInfoKHR-presentMode-02839
```

`VK_EXT_swapchain_maintenance1` 의 `VkSwapchainPresentModesCreateInfoEXT` 를 체인에 넣어도 레이어(1.4.309)가
이 VU 를 완화해 주지 않는다. 완화 규정을 담은 `VK_KHR_surface_maintenance1` 은 **이 드라이버에도 설치된
Vulkan 헤더(1.4.309)에도 없다**(드라이버는 `VK_EXT_surface_maintenance1` 과 `VK_KHR_swapchain_maintenance1`
만 지원 목록에 올린다).

"검증 레이어 errors 0" 이 이 프로젝트의 하드 게이트이므로 **기본값은 안전한 4**로 두고, 이미지 1장
(1080p RGBA 8.29 MB)을 아끼는 선택은 `SwapchainOptions::lean_images` / 프로브 `--lean-images` 로
명시적 opt-in 으로 남겼다. 모드별 질의 자체는 그대로 하고 P4 가 값을 덤프한다.

---

## P5 `--probe present` — 전부 통과 (★핵심 관문)

```
XDG_RUNTIME_DIR=/run/user/1000 WAYLAND_DISPLAY=wayland-0 \
  build/bin/yvo_display_probe --probe present --display-fullscreen \
    --frames 120 --hold-seconds 12 --sync-validation
```

| 기준 | 결과 | 실측값 |
|---|---|---|
| ① `vkCreateSwapchainKHR` VK_SUCCESS | **PASS** | 1920x1080, images=4, FIFO, compute → swapchain storage image |
| ② 검증 레이어 errors 0 (sync validation 포함) | **PASS** | errors **0**, warnings **0** |
| ③ 120 present 완료, 타임아웃 0 | **PASS** | presented 120, `vkWaitForPresentKHR` 타임아웃 **0** |
| ④ `present_wait` 간격 중앙값 16.7 ms ±2 | **PASS** | **median 16.661 ms**, p90 16.697, mean 16.659, min 16.419, max 16.964 (n=118) |
| ⑤ `grim -o HDMI-A-1` 픽셀 검사 | **PASS** | 아래 표. **모든 샘플 최대 오차 0** |

### ⑤ grim 픽셀 검사 (HOLD 구간에서 캡처, `--hold-seconds 12`)

캡처 파일: `07_results/display_vulkan_hdmi/samples/p05_present_pattern.png` (1920x1080)

| 좌표 | 기대 RGB | 실측 RGB | 오차 |
|---|---|---|---|
| center (960,540) | (255,0,128) | (255,0,128) | 0 |
| TL (16,16) | (255,0,0) | (255,0,0) | 0 |
| TR (1904,16) | (0,255,0) | (0,255,0) | 0 |
| BL (16,1064) | (0,0,255) | (0,0,255) | 0 |
| BR (1904,1064) | (255,255,255) | (255,255,255) | 0 |
| background (400,200) | (32,32,32) | (32,32,32) | 0 |

영역 크기는 **기대와 1픽셀씩 다르다(조정자 정정, 2026-09-20)**: 저장된 `samples/p05_present_pattern.png`를
다시 세어 보면 RGB(255,0,128) 픽셀이 **119,301개 = 399×300 − 299** 즉 **399×299**다(400×300이 아니다).
bbox 는 x 761~1159, y 391~689 다. 꼭지점 마커도 4,096(64²) / 4,032(64×63) / 4,032 / 3,969(63²)로
같은 차이를 보인다. → 셰이더 블록 경계가 **포함/배제 off-by-one** 이다(예: `x > cx-200 && x < cx+200`).

**P5 판정에는 영향이 없다**: 색이 오차 0 으로 정확하고 위치도 중앙이며, 컴퓨트가 스왑체인에 직접 쓴 것이
화면에 도달했다는 결론은 그대로다. `display_probe.comp` 는 테스트 전용이라 파이프라인에 쓰이지 않는다.
다만 **`[6] overlay_boxes.comp` 는 박스 테두리를 그리고 `test_overlay` 가 CPU 기준과 비트 일치를
요구하므로, 경계 규약(테두리 픽셀 포함 범위, 선 두께가 안/밖으로 커지는지)을 GPU 와 CPU 가 정확히
같게 정의해야 한다.** 이 off-by-one 이 그 단계에서 재발하면 비트 일치 테스트가 바로 잡아낸다.

**→ `R8G8B8A8_UNORM` storage image 에 `byte/255.0` 으로 쓴 값이 화면까지 한 비트도 변하지 않고 도달한다.**
계획서 §5.6 의 "UNORM 이면 아무 변환 없이 통과한다"가 실측으로 확인됐다.

### 검증 매트릭스 (모두 `--sync-validation`, 전부 errors 0 / warnings 0)

| 구성 | present | recreate | release | present_wait 중앙값 | errors |
|---|---:|---:|---:|---:|---:|
| fullscreen, storage 직접 쓰기 | 120 | 1 | 1 | 16.661 ms | **0** |
| fullscreen, `--display-blit` | 120 | 1 | 1 | 16.643 ms | **0** |
| 창 1280x720, storage, `--resize-every 20`(fullscreen 토글) | 120 | **5** | 1 | 16.653 ms | **0** |
| 창 1280x720, blit, `--resize-every 20` | 120 | **5** | 1 | 16.644 ms | **0** |

- `recreate` 는 `VkSwapchainPresentFenceInfoEXT` 로 구 스왑체인 폐기 시점을 판정한 횟수다
  (`vkDeviceWaitIdle` 을 쓰지 않는다). `--resize-every 20` 은 실행 중 fullscreen 을 토글해
  **컴포지터가 실제로 새 크기를 configure** 하게 만든다(창 모드에서 labwc 는 1920x1044 를 준다).
- `release` 는 일부러 획득만 하고 안 그린 프레임을 `vkReleaseSwapchainImagesEXT` 로 되돌린 횟수다
  (프로브가 매 실행 1회 강제한다 → 이 경로도 항상 검증 레이어를 지난다).

### 검증 레이어가 잡아낸 실제 버그 2개 (수정 완료)

1. **blit 폴백의 WRITE_AFTER_READ 해저드.** 자체 이미지의 `UNDEFINED → GENERAL` 전환이
   *같은 이미지 인덱스를 썼던 이전 프레임의 `vkCmdBlitImage` 읽기* 와 충돌했다.
   원인: `acquire()` 의 펜스 확인은 **sync 슬롯** 단위인데 이미지 인덱스는 프레젠테이션 엔진이 고르므로
   둘이 대응하지 않는다. storage 모드는 acquire 세마포어가 프레젠테이션 엔진과의 체인을 만들어 주지만,
   자체 이미지는 프레젠테이션 엔진이 건드리지 않으므로 그 체인이 없다.
   수정: blit 모드에서 전환 배리어의 `srcStageMask` 에 `BLIT|COPY` 를 넣는다(첫 동기화 범위가
   submission order 로 앞선 모든 커맨드를 포함하므로 이전 제출의 blit 읽기까지 덮인다.
   원인이 읽기라 실행 의존성만 필요해 `srcAccessMask` 는 NONE).
2. **`vkReleaseSwapchainImagesEXT` 를 너무 일찍 불렀다.**
   `VUID-...-pImageIndices-07786` 은 "이미지의 모든 사용이 **실행을 완료**해야 한다"고 요구하는데,
   acquire 세마포어를 소비하는 빈 제출이 아직 진행 중이었다
   (`One or more of the images in this swapchain is still in use`).
   수정: 호스트를 블로킹하지 않기 위해 **펜스가 신호되면 반납하도록 미룬다**(`flush_pending_releases()`).

---

## P8 `--probe convert` — 전부 통과 (★원본 해상도 성립 여부)

```
XDG_RUNTIME_DIR=/run/user/1000 WAYLAND_DISPLAY=wayland-0 \
  build/bin/yvo_display_probe --probe convert --frames 300 \
    --display-fullscreen --sync-validation
```

| 기준 | 결과 | 실측값 |
|---|---|---|
| ① 1080p 중앙값 ≤ 7.0 ms **그리고** p90 ≤ 14 ms | **PASS** | median **4.249 ms**, p90 **4.601 ms** |
| ② CPU 기준과 **비트 일치** | **PASS** | max 오차 **0**, 다른 픽셀 **0 / 2,073,600**, alpha≠255 **0개** |

### ① 오프스크린 연속 스트림 300프레임 (present 페이싱 없음, local=3x85)

present 페이싱이 GPU 를 놀려 DVFS 로 시간이 부풀지 않게 **표시 없이 back-to-back** 으로 쟀다
(2.13 ms 전처리 수치와 같은 측정 조건).

| 출력 크기 | 경로 | shader median | p90 | mean | max | total median |
|---|---|---:|---:|---:|---:|---:|
| **1920x1080** | **1:1 (기본)** | **4.249** | **4.601** | 4.378 | 8.227 | 4.260 |
| 1600x900 | uniform scale (bilinear) | 7.647 | 7.667 | 7.656 | 8.116 | 7.658 |
| 1280x720 | uniform scale (bilinear) | 5.276 | 5.300 | 5.293 | 5.827 | 5.288 |

`total` = 커맨드 버퍼 전체(입력 dma-buf 소유권 배리어 + 레이아웃 전환 포함). shader 와 0.01 ms 차이뿐이다.

### 실제 스왑체인 경로 (1920x1080 fullscreen, FIFO, 300프레임)

| 경로 | shader median | p90 | total median | present_wait 간격 | presented | display_dropped |
|---|---:|---:|---:|---:|---:|---:|
| compute → 스왑체인 storage image | **4.213** | 4.250 | 4.232 | 16.662 ms | 300 | **0** |
| `--display-blit` 폴백 | 4.221 | 4.338 | **5.459** | 16.661 ms | 300 | **0** |

**blit 폴백 비용 = total 5.459 − 4.232 = 약 1.23 ms** (1080p RGBA `vkCmdBlitImage`).
계획서는 720p 에서 +3.6 ms 를 예상했는데 1080p 에서 1.23 ms 다 — 폴백이 예상보다 싸다.
그래도 STORAGE 직접 쓰기가 가능하므로 기본은 직접 쓰기다.

### ② 픽셀 정확도 — CPU 기준과 비트 일치

기준 구현: `01_src/ref/sand.hpp` `desand_nv12()` + `01_src/ref/cpu_letterbox.hpp`
`cpu_letterbox_nv12_bt709()` (scale = 1.0, pad = 0). 프레임 100, 1920x1080.

```
frame 100: max 오차 = 0, 다른 픽셀 0 / 2073600, alpha!=255 0
비트 일치: 예 (max 오차 0)
```

1:1 이라 보간이 없어 원리적으로 가능하다는 계획서의 근거가 맞았다. 단 **크로마는 세로 보간이 필요하고
(`chroma_location = left`) 홀수 열은 가로 가중치가 정확히 0.5** 라서 다음 두 가지를 정확히 맞춰야 비트
일치한다 — 둘 다 셰이더에 주석으로 박아 두었다:
- 보간 순서를 CPU 기준(`bilinear_u8`)과 같게 **가로 먼저, 세로 나중**으로 한다
  (세로 먼저 하면 부동소수점 결과가 달라진다)
- BT.709 수식과 `precise` 한정자를 `02_shaders/sand_letterbox.comp:94-101` 에서 **그대로** 옮긴다

### 워크그룹 크기 스윕 — 계획서 예산을 통과시킨 결정적 레버

`--probe convert --sweep` (1920x1080 1:1, 오프스크린 200프레임). **3회 인터리브 반복으로 DVFS 드리프트를
배제**했다(재현성 ±0.02 ms: 2x128 = 6.206 / 6.206 / 6.207, 3x85 = 4.235 / 4.246 / 4.246).

| local (x × y) | invocation | 픽셀 타일 | shader median (ms) |
|---|---:|---|---:|
| 8 × 8 | 64 | 32 × 8 | 15.01 |
| 4 × 16 | 64 | 16 × 16 | 13.12 |
| 16 × 8 | 128 | 64 × 8 | 10.07 |
| 8 × 16 | 128 | 32 × 16 | 9.94 |
| 128 × 2 | 256 | 512 × 2 | 8.09 |
| 64 × 4 | 256 | 256 × 4 | 8.05 |
| 32 × 8 | 256 | 128 × 8 | 7.80 |
| 8 × 32 | 256 | 32 × 32 | 7.78 |
| 3 × 48 | 144 | 12 × 48 | 7.74 |
| 16 × 16 | 256 | 64 × 16 | 7.74 |
| 2 × 64 | 128 | 8 × 64 | 7.59 |
| 3 × 64 | 192 | 12 × 64 | 6.70 |
| 1 × 256 | 256 | 4 × 256 | 6.78 |
| 5 × 36 | 180 | 20 × 36 | 6.63 |
| 4 × 64 | 256 | 16 × 64 | 6.62 |
| 6 × 42 | 252 | 24 × 42 | 6.15 |
| 7 × 36 | 252 | 28 × 36 | 6.14 |
| 2 × 128 | 256 | 8 × 128 | 6.21 |
| 5 × 51 | 255 | 20 × 51 | 5.84 |
| 3 × 72 | 216 | 12 × 72 | 5.59 |
| 2 × 108 | 216 | 8 × 108 | 5.28 |
| 3 × 84 | 252 | 12 × 84 | 4.55 |
| **3 × 85** | **255** | **12 × 85** | **4.23 ← 채택** |

(`local_size_x` 는 **4픽셀 묶음** 단위다. 즉 3 → 12 픽셀.)

두 가지가 같이 작용한다:
1. **invocation 이 클수록 좋다** — `sand_letterbox` 의 규율 그대로다(64 inv 는 13~15 ms, 256 inv 는 4~8 ms).
2. **같은 invocation 수 안에서는 세로로 긴 타일이 결정적으로 빠르다** — 64×16 px 7.74 → 12×85 px 4.23,
   **1.83배**. 읽기만 보면(SAND 열은 가로 128 B) 좁은 타일이 캐시 라인을 낭비하므로 불리해야 하는데도 더
   빠르다 → 병목은 **타일 배치(UIF)된 RGBA 이미지 쓰기** 쪽이고 세로로 긴 영역이 이미지 타일에 더 잘 맞는
   것으로 본다. **이 해석은 미확정 가설이고 수치만이 근거다.**

같은 최적값이 균일 스케일 경로에도 통한다 (1280x720, 오프스크린 200프레임):
16x16 **7.18** / 32x8 7.29 / 8x32 7.24 / 1x256 9.55 / 2x128 6.74 / 2x108 6.24 / 4x64 6.33 /
6x42 6.25 / 5x51 5.72 / 3x72 6.09 / 3x84 5.59 / **3x85 5.27**.

### 폴백 권고 — 계획서와 결론이 다르다

계획서는 "중앙값 7~10 ms 면 1600x900, 10 ms 초과면 1280x720" 을 폴백으로 적었다.
**실측은 그 방향이 통하지 않는다는 것을 보여준다**: 균일 스케일 경로는 4-tap bilinear + 열 LUT 때문에
픽셀당 비용이 1:1 경로의 약 2배다(1:1 2.05 ns/px vs scale 5.31 ns/px @1600x900).
따라서 **1600x900(7.65 ms)은 1920x1080 1:1(4.25 ms)보다 오히려 80% 느리고, 1280x720(5.28 ms)도 느리다.**

→ **해상도를 낮추는 것은 이 파이프라인에서 비용 절감 수단이 아니다.** 1920x1080 1:1 이 가장 싸다.
예산이 부족해지면 쓸 수 있는 레버는 ① `--display-every 2`(표시 15 fps, 평균 비용 절반)
② 계획서 §5.3 안 (III)(`zwp_linux_dmabuf` SAND 직접 스캔아웃, P10 이 프로토콜 가능성만 확인함)
③ `wp_viewporter` 로 작게 렌더하고 HVS 확대 — 단 ③은 1:1 경로를 못 쓰므로 위 수치대로 이득이 없다.
지금은 **어느 것도 필요하지 않다.**

### 예산 대조 (계획서 §5.3 표 갱신)

| 항목 | 계획서 추정 | 실측 |
|---|---:|---:|
| letterbox 전처리 | 2.13 | 2.13 (변경 없음) |
| **표시 변환 1080p** | **4~7** | **4.25** (median), 4.60 (p90) |
| 오버레이 | 0.3 | 미구현(다음 단계) |
| GPU 합 (평균) | 6.4~9.4 | **약 6.7** |

b4 NPU 주기 22.5 ms 의 약 30%. **계획서 추정 범위의 하단에 들어왔고 안 (I)은 성립한다.**

---

## 부가: `--probe export` — 후처리 트랙용 dma-buf 내보내기 헬퍼 확인

`01_src/vulkan/dmabuf_export.{hpp,cpp}` (`DmaBufExporter`). align = 16384 B (HailoRT `read_async` 요구).
`yolov11m.hef` 원시 6스트림 크기 + 640x640x3 전처리 텐서로 7개를 할당했다.

```
align = 16384 B
dedicated 할당 강제: 아니오
#       requested        alloc       dmabuf     fd host_ptr           aligned_ptr        align_offset
0          512000       540672       540672     10 0x7fff21768000     0x7fff21768000     0
1          409600       425984       425984     11 0x7fff21700000     0x7fff21700000     0
2          128000       147456       147456     12 0x7fff216dc000     0x7fff216dc000     0
3          102400       131072       131072     13 0x7fff216bc000     0x7fff216bc000     0
4           38400        65536        65536     14 0x7fff27aa0000     0x7fff27aa0000     0
5           30720        49152        49152     15 0x7fff29a84000     0x7fff29a84000     0
6         1228800      1245184      1245184     16 0x7fff2158c000     0x7fff2158c000     0
판정: PASS
```

확인된 사실 (후처리 트랙이 알아야 하는 것):

1. **`alloc_size == dmabuf_size` 이고 둘 다 16384 배수다.** 호스트 매핑 주소도 전부 16384 정렬이라
   `align_offset` 이 0 이다(여유 재할당 경로는 발동하지 않았다).
2. **`requested_size` 는 항상 그보다 작다** — 7개 전부. 예: 1,228,800 요청 → 1,245,184 실제.
   따라서 `dma_map_dmabuf(fd, size, D2H)` / `read_async(fd, size)` 에 넘길 size 후보가 **두 가지**다
   (① 텐서 크기 ② dma-buf 실측 크기). 헬퍼가 둘 다 노출하므로 `hailo-expert` 가 그대로 시험할 수 있다.
3. **V3D 는 dma-buf export 에 전용 할당을 요구하지 않는다**(`dmabuf_export_dedicated_only() == false`).
   이게 중요한 이유: `VkMemoryDedicatedAllocateInfo` 를 붙이면 spec 이 `allocationSize == VkMemoryRequirements::size`
   를 강제하고(`VUID-VkMemoryDedicatedAllocateInfo-buffer-02965`), V3DV 는 버퍼 크기 + 256 B 를 요구하므로
   **정렬을 맞출 수 없다**. 처음엔 `sand_letterbox.cpp` 처럼 전용 할당을 썼는데 이 프로브가 잡아냈다
   (`[SIZE FAIL]` 7/7 → 전용 할당을 조건부로 바꾼 뒤 PASS).
   드라이버가 전용 할당을 요구하는 환경이라면 `ExportedDmaBuf::dedicated == true` 로 알려 주고,
   그때는 `alloc_size` 가 정렬 배수가 아닐 수 있다.
4. 매핑은 실제로 읽고 쓰인다(프로브가 첫/끝 바이트를 왕복 검증한다). `dup_fd()` 로 넘길 fd 도 확인.

## 부가: `--lean-images` (FIFO 3장) 실측

```
[swapchain] created 1920x1080 images=3(min 3) mode=FIFO (compute -> swapchain storage image)
present_wait 간격 ms : median=16.659 p90=16.776 (n=58)  — 60 present, 타임아웃 0
vk validation errors 2 warnings 0 (layer on)
```
**런타임은 정상 동작한다**(60 present, 16.659 ms 간격). 문제는 검증 레이어 error 2건뿐이다
(생성 2회 × 1건). 그래서 기본값에서 빼고 opt-in 으로 남겼다 — 위 P4 절의 설명 참고.
