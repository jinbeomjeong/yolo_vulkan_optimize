# 07. 영상 포맷과 화면 표시 — 파일에서 모니터까지

> **선수 지식**: [01_big_picture.md](01_big_picture.md) (glass-to-glass·처리량·지연),
> [05_vulkan_compute.md](05_vulkan_compute.md) (컴퓨트 셰이더·spec constant·배리어),
> [06_zerocopy_dmabuf.md](06_zerocopy_dmabuf.md) (dma-buf·import)
> **다 읽으면 읽을 수 있는 것**: [`01_src/ref/sand.hpp`](../01_src/ref/sand.hpp),
> [`01_src/ref/cpu_letterbox.hpp`](../01_src/ref/cpu_letterbox.hpp) 상단 수식 주석,
> [`01_src/display/vk_swapchain.hpp`](../01_src/display/vk_swapchain.hpp),
> [`01_src/display/wayland_window.hpp`](../01_src/display/wayland_window.hpp),
> [`01_src/display/plane_display.hpp`](../01_src/display/plane_display.hpp) 상단 주석(§7.12),
> [`07_results/display_vulkan_hdmi/summary.md`](../07_results/display_vulkan_hdmi/summary.md)

---

## 7.1 압축 영상 — 컨테이너와 코덱은 다른 것이다

### mp4는 코덱이 아니다

`video_h265.mp4` 라는 파일 하나에 두 개의 서로 다른 개념이 들어 있다.

- **컨테이너(container)**: `.mp4`, `.mkv`, `.avi`.
  **"여러 트랙을 하나의 파일에 담는 상자"** 다. 영상 트랙, 음성 트랙, 자막, 그리고
  "이 영상은 몇 fps이고 몇 초짜리다" 같은 메타데이터가 들어 있다.
  상자를 여는 일을 **demux(디먹스)** 라고 한다.
- **코덱(codec)**: H.264, H.265(HEVC), AV1.
  **"영상을 어떻게 압축했는가"** 다. 상자 안의 영상 트랙이 어떤 규칙으로 압축돼 있는지를 말한다.

**같은 `.mp4` 안에 H.264가 들어 있을 수도, H.265가 들어 있을 수도 있다.**
이 프로젝트에 `video_h264.mp4` 와 `video_h265.mp4` 가 따로 있는 이유가 이것이다 —
컨테이너는 같고 코덱이 다르다. 그리고 §7.3에서 보겠지만 **이 차이가 성능을 10배 가른다.**

### 키프레임과 차분 프레임

1080p 한 장을 그대로 저장하면 약 3 MB다. 30 fps면 초당 90 MB다. 현실적이지 않다.

압축의 핵심 아이디어는 **"연속한 프레임은 거의 똑같다"** 는 것이다.
거리 영상에서 1/30초 사이에 바뀌는 것은 지나가는 차와 사람뿐이고 건물·도로·하늘은 그대로다.

```
   프레임 0    프레임 1    프레임 2    프레임 3   ...   프레임 60
   ┌──────┐   ┌──────┐   ┌──────┐   ┌──────┐         ┌──────┐
   │ 전체 │   │ 차이 │   │ 차이 │   │ 차이 │   ...   │ 전체 │
   │ 그림 │   │ 만   │   │ 만   │   │ 만   │         │ 그림 │
   └──────┘   └──────┘   └──────┘   └──────┘         └──────┘
    키프레임    ~수 KB     ~수 KB     ~수 KB          키프레임
    (I-frame)                                        (다시 시작)
    ~100 KB
```

- **키프레임(keyframe, I-frame)**: 혼자서 완전한 그림. 다른 프레임이 필요 없다.
- **차분 프레임(P/B-frame)**: "앞 프레임에서 이 블록을 이만큼 옮기고 이 값만큼 더해라".

이것이 두 가지 결과를 낳는다.

1. **아무 데서나 재생을 시작할 수 없다.** 키프레임부터 시작해야 한다.
   (동영상 탐색이 뚝뚝 끊기는 이유)
2. **디코드는 순서를 지켜야 한다.** 프레임 5를 풀려면 프레임 4가 필요하다.
   병렬화가 제한된다.

---

## 7.2 HW 디코더와 SW 디코더

압축을 푸는 것을 **디코드(decode)** 라고 한다. 두 가지 방법이 있다.

- **SW(소프트웨어) 디코더**: CPU가 코드를 돌려서 푼다. 어떤 코덱이든 라이브러리만 있으면 된다.
- **HW(하드웨어) 디코더**: **그 코덱을 푸는 전용 회로**가 칩 안에 있다. 훨씬 빠르고 전력도 적게 쓴다.
  대신 회로가 있는 코덱만 된다.

### ★ 이 보드는 HEVC만 하드웨어로 푼다

BCM2712(Pi 5 / CM5)에는 **HEVC(H.265) 디코더 회로만** 들어 있다. **H.264 회로는 없다.**
(이전 세대인 Pi 4에는 H.264 회로가 있었다.)

측정으로 보면 이렇다 ([`decode_benchmark/summary.md`](../07_results/decode_benchmark/summary.md), 1080p 중앙값):

| 디코더 | FPS | 프레임당 CPU 시간 |
|---|---:|---:|
| `avdec_h264` 자동 스레드 (SW) | 164.6 | 20.2 ms |
| `avdec_h265` 자동 스레드 (SW) | 109.1 | 31.8 ms |
| **ffmpeg `-hwaccel drm` HEVC (HW, CLI)** | **332.8** | **2.0 ms** |

HW 디코더는 1080p를 **약 11배속**으로 푼다. CPU는 거의 놀고 있다.
(2.0 ms도 대부분 CLI가 프레임을 CPU로 내려받는 비용이다. DRM_PRIME을 유지하면 0.55 ms,
[`runs/ffmpeg_outfmt`](../07_results/decode_benchmark/runs/ffmpeg_outfmt/summary.md))

GPU(V3D)로 푸는 길은 없다. V3DV에는 Vulkan Video 확장(`VK_KHR_video_*`)이 하나도 없고, 전용 엔진 없이
compute 셰이더로 푸는 FFmpeg Vulkan 코덱(FFv1·ProRes·DPX)도 V3D에서는 돌지 않는다(`shaderInt64` 등 기능 부족,
[`vulkan_compute_codecs`](../07_results/vulkan_compute_codecs/summary.md)). 그래서 **HEVC는 전용 HW 디코더, V3D는 전·후처리**라는 분업만 성립한다.

그래서 이 프로젝트는 **입력을 HEVC로 못 박았다.**
[`ffmpeg_drm_decoder.cpp`](../01_src/decode/ffmpeg_drm_decoder.cpp) 는 아예 코드로 거부한다:

```cpp
if (st->codecpar->codec_id != AV_CODEC_ID_HEVC) return fail("video stream is not HEVC");
```

> ⚠ 이것이 **아직 열려 있는 제약**이기도 하다. 카메라(CSI) 입력으로 가면 이 경로를
> 다시 봐야 한다 ([`07_results/README.md`](../07_results/README.md) "아직 열려 있는 것").

---

## 7.3 색을 나타내는 두 가지 방법 — RGB와 YUV

### RGB

화면의 픽셀 하나는 빨강·초록·파랑 세 개의 빛을 섞어서 만든다.
그래서 색을 **R, G, B 세 숫자**(각 0~255)로 표현하는 것이 가장 직관적이다.
모니터가 실제로 받는 것도 이 형식이다.

### 왜 영상은 YUV를 쓰는가

**사람 눈은 밝기 변화에는 민감하고 색 변화에는 둔하다.**
흐릿한 색 경계는 잘 못 알아채지만, 흐릿한 밝기 경계는 금방 알아챈다.

그래서 색을 이렇게 나눈다.

- **Y (luma, 루마)** — **밝기.** 흑백 사진에 해당한다.
- **U, V (chroma, 크로마)** — **색차.** "파랑 쪽으로 얼마나", "빨강 쪽으로 얼마나".
  (Cb, Cr이라고도 쓴다.)

나눠 놓으면 **크로마만 해상도를 낮춰도 사람 눈에 거의 안 보인다.**
이것이 **크로마 서브샘플링(chroma subsampling)** 이다.

### 4:2:0 크로마 서브샘플링

`4:2:0` 은 "루마 2×2 블록마다 크로마를 **한 쌍만** 둔다"는 뜻이다.

```
   루마 (Y) — 픽셀마다 하나         크로마 (U,V) — 2×2 마다 하나
   ┌───┬───┬───┬───┐               ┌───────┬───────┐
   │ Y │ Y │ Y │ Y │               │       │       │
   ├───┼───┼───┼───┤               │  U,V  │  U,V  │
   │ Y │ Y │ Y │ Y │               │       │       │
   ├───┼───┼───┼───┤               ├───────┼───────┤
   │ Y │ Y │ Y │ Y │               │       │       │
   ├───┼───┼───┼───┤               │  U,V  │  U,V  │
   │ Y │ Y │ Y │ Y │               │       │       │
   └───┴───┴───┴───┘               └───────┴───────┘
        16 바이트                        8 바이트
```

데이터가 **RGB 대비 절반**으로 줄어든다(픽셀당 3 B → 1.5 B).
1920×1080 한 프레임이 6.2 MB → **3.1 MB** 다.

### NV12 — 이 프로젝트가 다루는 배치

같은 4:2:0이라도 메모리에 어떻게 깔지는 여러 방식이 있다. **NV12** 는 이렇다.

```
   ┌─────────────────────────────┐
   │                             │
   │        Y 평면 (plane)        │   width × height 바이트
   │   Y Y Y Y Y Y Y Y Y ...     │   (밝기, 픽셀마다 1 B)
   │                             │
   ├─────────────────────────────┤
   │   U V U V U V U V ...       │   width × height/2 바이트
   │        UV 평면 (인터리브)      │   (U와 V가 번갈아 들어 있다)
   └─────────────────────────────┘
```

- **Y와 UV가 분리된 두 평면(plane)** 이다
- **UV 평면 안에서는 U와 V가 한 바이트씩 번갈아(interleave)** 있다
- UV 평면의 행 수는 Y의 절반이다

> 이 보드의 HW 디코더는 **Y와 UV를 아예 서로 다른 dma-buf 객체로** 준다
> (Y 2,097,152 B / UV 1,048,576 B). 그래서 셰이더도 SSBO를 두 개 받는다:
> `YBuf`, `UVBuf` ([`sand_letterbox.comp`](../02_shaders/sand_letterbox.comp)).

### limited range와 full range — 조용히 틀리는 함정

`Y` 는 0~255를 다 쓰지 않는다. **영상에서는 보통 16~235만 쓴다.**
(크로마는 16~240.) 이것을 **limited range**(또는 TV range, studio swing)라고 한다.
0~255를 다 쓰는 것은 **full range**(PC range)다.

왜 이런 게 있나? 아날로그 시대에 신호가 살짝 넘치는 것(오버슈트)을 담을 여유를 남긴 것이다.

**이걸 틀리면 어떻게 되나?** 프로그램이 죽지 않는다. 그냥 **화면이 뿌옇거나 대비가 과하게** 나온다.
"어? 색이 좀 이상한데?" 정도라서 눈치채기 어렵다. 그래서 반드시 명시해야 한다.

이 프로젝트는 **BT.709 limited range** 로 확정했고, 그 변환식이
[`cpu_letterbox.hpp`](../01_src/ref/cpu_letterbox.hpp) 상단 주석 5)에 적혀 있다:

```
   Y' = (Y - 16) × 255/219       ← 16~235 를 0~255 로 펴는 것
   U' = (U - 128) × 255/224      ← 128 이 "색 없음"의 중심
   V' = (V - 128) × 255/224

   R = Y' + 1.5748 V'
   G = Y' - 0.187324 U' - 0.468124 V'
   B = Y' + 1.8556 U'

   0~255 로 클램프 후 반올림(+0.5 후 절삭)
```

**BT.709** 는 HD 영상의 색 규격 이름이다(SD 시대에는 BT.601, 4K HDR은 BT.2020).
위의 계수 1.5748, 0.187324 … 가 BT.709에서 나온 값이다.

> ★ 이 프로젝트는 이 수식을 **두 셰이더에 똑같이** 넣었다 —
> `sand_letterbox.comp`(추론용)와 `display_convert.comp`(표시용). 주석이 이렇게 말한다:
> "**BT.709 수식은 02_shaders/sand_letterbox.comp:94-101 을 그대로 옮긴 것이다**(precise 포함) —
> 표시 색이 전처리 경로와 한 비트도 달라지면 안 된다."
> `precise` 한정자까지 맞춘 이유는 **부동소수점 연산 순서가 바뀌면 결과가 1 LSB 달라질 수 있어서**다.

### 감마와 sRGB — 왜 "그냥 통과"인가

마지막 개념 하나. 위 식이 내놓는 R, G, B는 **물리적인 빛의 양이 아니다.**

사람 눈은 어두운 쪽 변화에 민감하다. 그래서 디지털 영상은 값을 **비선형으로 인코딩**한다 —
값 128이 빛의 양 50%가 아니라 약 22%다. 이것을 **감마 인코딩(gamma encoding)** 이라고 하고,
PC 화면의 표준 인코딩이 **sRGB** 다.

중요한 결론은 이것이다: **YUV 영상의 값은 이미 감마 인코딩돼 있고, BT.709 변환식은
그 인코딩을 유지한 채 RGB로 바꾼다.** 즉 나온 값은 **이미 sRGB로 인코딩된 0~255** 다.
모니터에 그대로 보내면 맞는다. §7.7에서 이것이 포맷 선택의 근거가 된다.

---

## 7.4 SAND128 — Broadcom 전용 타일 배치

여기서 이 보드 특유의 사정이 나온다.

HW 디코더가 내놓는 NV12는 **평범한 NV12가 아니다.** 행이 연속으로 깔려 있지 않다.

### 보통의 NV12 (linear)

```
   메모리 주소 →
   [0행 전체 1920 B][1행 전체 1920 B][2행 전체 1920 B] ...
   주소 = y × stride + x
```

### SAND128

**이미지를 폭 128 바이트짜리 세로 열(column)로 쪼개고, 열 하나를 통째로 연속 배치한다.**

```
   화면 모습                       메모리 배치
   ┌────┬────┬────┬ ... ┬────┐
   │col0│col1│col2│     │col14│     [ col0 전체 ][ col1 전체 ][ col2 전체 ] ...
   │128B│128B│128B│     │128B │       ↑             ↑
   │    │    │    │     │     │       128 B × 1088행  = 139,264 B  (= col_stride)
   │    │    │    │     │     │
   │1088│    │    │     │     │     col0 안에서는:
   │ 행 │    │    │     │     │       [0행의 128 B][1행의 128 B][2행의 128 B] ...
   └────┴────┴────┴ ... ┴────┘
    1920 B = 15 열
```

주소 식은 이렇게 된다 ([`sand.hpp`](../01_src/ref/sand.hpp)):

```
   Y(x, y)   = y_base [ y_offset  + (x / 128) × y_col_stride  + y  × 128 + (x % 128) ]
                                     ↑열 번호   ↑열 하나 크기    ↑열 안의 행  ↑열 안의 열
   UV(x, cy) = uv_base[ uv_offset + (x / 128) × uv_col_stride + cy × 128 + (x % 128) ]
```

셰이더 안에서는 나눗셈·나머지를 시프트·마스크로 바꾼다(`x >> 7`, `x & 127`).

### 왜 이렇게 두는가

**디코더 하드웨어의 편의** 때문이다. 영상 코덱은 이미지를 **블록(보통 16×16이나 64×64)
단위로** 처리한다. 블록 하나를 쓸 때 linear 배치면 메모리의 16군데에 흩어져 쓰게 되는데,
세로로 긴 열 안에 두면 **훨씬 연속적**이다. 메모리 컨트롤러 입장에서 효율이 좋다.

대가는 **읽는 쪽이 복잡해진다**는 것이다. 이 프로젝트는 SAND 주소 계산을 셰이더가 직접 한다(§7.7).
(rpt3 시절 GStreamer가 1080p에서 CPU 복사로 빠진 것을 SAND 탓으로 봤지만, 실제 원인은 크롭 처리 버그였다 — §7.7.)

> 이 주소 규칙은 추측이 아니라 **실측으로 확인했다.** `yvo_drm_dump` 로 디코더 출력을 덤프하고
> CPU de-SAND 결과를 `av_hwframe_transfer_data`(ffmpeg의 표준 변환)와 비교해
> 프레임 0/100/900/1815에서 **바이트 단위로 일치**하는 것을 확인했다
> ([`REPORT.md §2.3`](../07_results/pipeline_ffmpeg_vulkan_hailo/REPORT.md)).
> modifier 값은 `0x0700000000000004` 이고, **`pitch` 필드에 바이트 pitch가 아니라 열 높이가
> 들어 있다**(Y 1088, UV 544)는 것도 그때 알아냈다.

---

## 7.5 letterbox — 비율을 유지하며 정사각형에 넣기

YOLO 모델은 **640×640 정사각형**만 받는다. 우리 영상은 1920×1080 (16:9)이다.

그냥 늘려서 640×640으로 만들면 그림이 찌그러진다 — 사람이 땅딸막해지고,
학습 때와 다른 모양이 되어 검출이 나빠진다.

그래서 **비율을 유지한 채 축소하고, 남는 곳을 회색으로 채운다.** 이것이 **letterbox** 다.
(영화를 TV에 틀 때 위아래에 검은 띠가 생기는 그 letterbox다.)

```
   1920×1080                        640×640
   ┌────────────────┐               ┌────────────────┐
   │                │               │▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒│ ← pad_y = 140 행, 값 114
   │                │   scale =     ├────────────────┤
   │   원본 영상     │   640/1920    │                │
   │                │   = 0.3333    │   640 × 360    │
   │                │      ──►      │                │
   │                │               ├────────────────┤
   └────────────────┘               │▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒│ ← 140 행
                                    └────────────────┘
```

- `scale = min(640/1920, 640/1080) = 0.3333`
- `new_w = 640, new_h = 360`
- `pad_x = 0, pad_y = (640-360)/2 = 140`
- **패딩 값은 114** — YOLO 계열의 관례값이다(중간 회색)

계산은 [`frame_types.hpp`](../01_src/core/frame_types.hpp) 의 `make_letterbox()` 에 있다.

### ★ half-pixel 규칙 — 왜 `+0.5 / -0.5` 가 붙는가

축소할 때 "출력 픽셀 (dx, dy)는 원본의 어느 위치에서 왔는가"를 계산해야 한다.
순진하게 하면 `sx = dx / scale` 인데, **이것이 틀렸다.**

이유는 **픽셀이 점이 아니라 사각형**이기 때문이다. 픽셀 `dx` 가 차지하는 영역은
`[dx, dx+1)` 이고, 그 **중심**은 `dx + 0.5` 다. 좌표를 다룰 때는 중심끼리 맞춰야 한다.

```
   출력 픽셀 dx 의 중심:  dx + 0.5
        │ ÷ scale 해서 원본 좌표계로
        ▼
   원본에서의 중심 위치:  (dx + 0.5) / scale
        │ 다시 "픽셀 번호"로 돌리려면 0.5 를 뺀다
        ▼
   sx = (dx + 0.5) / scale - 0.5
```

letterbox 여백까지 넣으면 [`cpu_letterbox.hpp`](../01_src/ref/cpu_letterbox.hpp) 주석 2)의 식이 된다:

```
   sx = (dx - pad_x + 0.5) / scale - 0.5
   sy = (dy - pad_y + 0.5) / scale - 0.5
```

이것은 `cv::resize` 의 `INTER_LINEAR` 와 **같은 규칙**이다. 맞춰 놓은 이유는
CPU 기준 구현과 GPU 셰이더가 **비트 단위로 일치**해야 하기 때문이다
(`test_preproc`: NHWC·NHCW 양쪽 5프레임 **max 오차 0**).

**0.5를 빼먹으면 어떻게 되나?** 이미지가 반 픽셀 밀린다. 눈으로는 안 보이지만
박스 좌표가 조금씩 틀어지고, 무엇보다 CPU 기준과의 비트 일치가 깨져서
**"GPU가 맞는지"를 검증할 수단이 사라진다.**

### 크로마 위치 — 또 하나의 반 픽셀

4:2:0에서 크로마 샘플 하나가 루마 2×2를 대표한다면, **그 샘플은 정확히 어디에 있는가?**
규격마다 다르다. HEVC의 기본이자 이 영상의 메타데이터(`chroma_location=left`)는
**"수평으로는 왼쪽 루마 열과 같은 위치, 수직으로는 두 루마 행의 가운데"** 다.

```
   Y₀₀  Y₀₁
     ╳          ← 크로마 샘플은 수평으로 Y₀₀ 과 같은 x,
   Y₁₀  Y₁₁       수직으로는 Y₀₀ 와 Y₁₀ 의 중간 y

   cx = sx / 2
   cy = (sy + 0.5) / 2 - 0.5
```

`cy` 에만 `+0.5` 가 더 붙는 것이 그 "수직 중간"이다.

---

## 7.6 화면에 내보내기 — Wayland와 스왑체인

### Wayland가 무엇인가

리눅스에서 창을 띄우는 시스템이다. 예전에는 X11(X Window System)이었고 지금은 Wayland로
넘어가는 중이다. 등장인물이 둘이다.

- **컴포지터(compositor)**: 화면 전체를 책임지는 서버. 모든 창을 합성해서 모니터에 내보낸다.
  이 보드에서 도는 것은 **labwc** 다.
- **클라이언트(client)**: 창을 띄우고 싶은 프로그램. 우리 프로그램이 여기 해당한다.

클라이언트는 컴포지터에게 프로토콜로 말을 건다. 핵심 프로토콜이 **xdg-shell** 이다 —
"창을 하나 만들어 줘", "제목은 이거야", "전체화면으로 해 줘" 같은 것을 다룬다.

[`wayland_window.hpp`](../01_src/display/wayland_window.hpp) 가 붙이는 것들:

| 프로토콜 객체 | 역할 |
|---|---|
| `wl_compositor` | surface(그림을 올릴 면)를 만든다 |
| `xdg_wm_base` | xdg-shell의 입구. ping/pong으로 살아 있음을 알린다 |
| `xdg_surface` | surface를 "창"으로 승격시킨다. `ack_configure` 로 크기 협상 |
| `xdg_toplevel` | 제목·app_id·전체화면·닫기 이벤트 |

여기에 `wl_seat` + `wl_keyboard` 도 붙인다 — **종료 키(Esc/q/Ctrl+C)만** 본다. 전체화면 창이 키보드 포커스를
가져가면 터미널의 Ctrl+C 가 프로그램에 닿지 않기 때문이다(키 해석은 evdev 코드만 보고 xkbcommon 은 쓰지 않는다).
HVS 평면 백엔드(§7.12)일 때만 `zwp_linux_dmabuf_v1`, `wl_subcompositor`, `wp_presentation`, `wp_viewporter`,
color-representation 을 추가로 붙인다.

**붙이지 않는 것** — `wl_shm`(공유 메모리 버퍼: 스왑체인이나 dma-buf 가 대신한다), EGL.

### VkSurfaceKHR와 스왑체인

Vulkan은 플랫폼 중립이라 "창"을 모른다. 그래서 다리가 둘 필요하다.

- **`VkSurfaceKHR`** — "이 Wayland surface가 Vulkan이 그릴 대상이다"라는 핸들
- **`VkSwapchainKHR`(스왑체인)** — **화면에 내보낼 이미지 여러 장을 돌려 쓰는 링 버퍼**

왜 여러 장인가? 한 장이면 **그리는 중인 그림이 화면에 나간다.** 반쯤 그려진 프레임이 보인다.
그래서 최소 두 장을 두고 하나는 화면에 나가 있고 하나는 그린다(더블 버퍼링).

```
   ┌──────── 스왑체인 (이미지 3~4장) ────────┐
   │   [img0]   [img1]   [img2]   [img3]    │
   └────────────────────────────────────────┘
        ▲                          │
        │ acquire                  │ present
        │ "빈 거 하나 줘"           ▼ "이거 화면에 내보내"
   ┌────┴──────────────────────────────────┐
   │  우리 프로그램: 그 이미지에 그림을 그린다  │
   └───────────────────────────────────────┘
```

### present mode — 언제 화면을 바꾸나

모니터는 정해진 박자로 화면을 새로 그린다. 60 Hz면 **16.7 ms마다** 한 번이다.
이 순간을 **vsync(vertical sync, 수직 동기)** 라고 한다.

우리가 새 프레임을 준비했을 때 vsync를 기다릴 것인가 말 것인가 — 이것이 **present mode** 다.

| mode | 동작 | 결과 |
|---|---|---|
| **FIFO** | vsync까지 **기다렸다가** 순서대로 내보낸다 | **tearing 없음.** 최대 60 fps로 제한됨. **항상 지원이 보장된다** |
| **MAILBOX** | 기다리되, 그동안 새 프레임이 오면 **대기 중인 것을 버리고 교체** | tearing 없음. 항상 최신. 이미지가 더 필요함 |
| **IMMEDIATE** | 기다리지 않고 **즉시** 내보낸다 | 지연 최소. **화면이 찢어진다(tearing)** |

**tearing(티어링)** 이란? 모니터가 화면 위쪽을 그리는 도중에 내용이 바뀌면,
**위쪽은 옛 프레임, 아래쪽은 새 프레임**이 한 화면에 나온다. 가로로 갈라진 선이 보인다.

이 프로젝트는 **FIFO** 를 쓴다. 프로브에서 present 간격이
**median 16.661 ms**(p90 16.697, n=118)로 나와 60 Hz에 정확히 맞는 것을 확인했다
([`display_vulkan_hdmi/summary.md`](../07_results/display_vulkan_hdmi/summary.md) P5).

### 한 프레임의 사이클

```
   ① vkAcquireNextImageKHR   "빈 이미지 하나 줘"  → 이미지 인덱스 + acquire 세마포어
        │
   ② 커맨드 버퍼 기록·제출     그 이미지에 컴퓨트 셰이더가 쓴다
        │                      (acquire 세마포어를 기다리고, render-finished 세마포어를 세운다)
        │
   ③ vkQueuePresentKHR       "다 그렸으니 화면에 내보내"
                              (render-finished 세마포어를 기다린다)
```

세마포어가 두 개 필요한 이유는 [05_vulkan_compute.md](05_vulkan_compute.md) §5.6 참고다 —
**GPU가 GPU를 기다리는** 관계라서 fence가 아니라 세마포어다.
그리고 `vkAcquireNextImageKHR`/`vkQueuePresentKHR` 에는 **timeline 세마포어를 쓸 수 없다**
(binary 세마포어만 된다).

---

## 7.7 이 프로젝트에서는 — 디코드 경로 선택

### ★ 같은 HW 디코더인데 10배 차이가 났다

이 프로젝트 초기의 가장 놀라운 측정이다
([`decode_benchmark/summary.md`](../07_results/decode_benchmark/summary.md)).

| 경로 | FPS | CPU ms/frame |
|---|---:|---:|
| GStreamer `v4l2slh265dec` (HW 디코더 사용, gst-plugins-bad rpt3) | **32.7** | **30.6** |
| ffmpeg `-hwaccel drm` (같은 HW 디코더, CLI는 CPU로 내려받음) | **332.8** | **2.0** |

> **2026-09-29 갱신.** 위 GStreamer 값은 rpt3 패키지 기준이다. 이 복사는 upstream 1.26.4에서 고쳐진 버그였고
> ([RPi-Distro/repo#406](https://github.com/RPi-Distro/repo/issues/406)), rpt4가 수정을 들여온 뒤 DMABuf 경로는
> **381.5 FPS / 0.24 ms**다([`runs/gst_rpt4`](../07_results/decode_benchmark/runs/gst_rpt4/summary.md)).
> ffmpeg 행의 2.0 ms도 대부분 CPU 내려받기이고, DRM_PRIME을 유지하면 386.9 FPS / 0.55 ms다
> ([`runs/ffmpeg_outfmt`](../07_results/decode_benchmark/runs/ffmpeg_outfmt/summary.md)). 아래 결론(병목은 복사)은 그대로다.

**같은 하드웨어 회로를 쓰는데 10배 차이였다.** 게다가 rpt3의 GStreamer 경로는 CPU 코어 하나가 100% 찼다.
HW 디코드를 쓰고도 CPU 비용(30.6 ms)이 SW H.265 디코드(31.8 ms)와 거의 같았다.
**하드웨어 가속의 의미가 통째로 사라진 것이다.**

**원인**: `perf` 로 프로파일하니 CPU 시간의 약 **90%가 ORC가 생성한 프레임 복사·변환 코드**였다.
당시에는 디코더 출력이 SAND128(§7.4)이라 GStreamer의 일반 경로가 그것을 linear NV12로 되돌리려고
CPU로 복사한다고 해석했다. `DMABuf` caps를 강제해도 복사가 남았기 때문이다.
**실제 원인은 SAND가 아니었다.** 1080p HEVC는 코딩 높이 1088에서 아래 8줄을 잘라 내는데(좌상단 (0,0)에서 시작하는 크롭 창),
v4l2codecs가 이 경우의 zero-copy를 처리하지 못하고 복사로 빠지는 upstream 버그가 있었다(1.26.4, MR !9315에서 수정).
rpt4는 이 수정을 들여왔고, 여전히 SAND128 DMABuf를 그대로 내보낸다([`runs/gst_rpt4`](../07_results/decode_benchmark/runs/gst_rpt4/summary.md)).

**결론**: 병목은 디코더가 아니라 **디코더 출력을 누가 어떻게 건드리느냐**였다.
[06_zerocopy_dmabuf.md](06_zerocopy_dmabuf.md) §6.7의 교훈과 정확히 같은 모양이다.

**해법(당시)**: GStreamer를 통하지 않고 **앱 안에 libavcodec DRM_PRIME 디코더를 직접 넣었다**
([`ffmpeg_drm_decoder.cpp`](../01_src/decode/ffmpeg_drm_decoder.cpp)).
디코더가 주는 dma-buf fd를 **그대로 Vulkan으로 import** 하고, SAND 주소 계산을 셰이더가 한다.
CPU는 그 메모리를 한 바이트도 읽지 않는다.

결과: **398.5 FPS / CPU 0.23~0.28 ms/frame** (디코드만 잰 값, FFmpeg 프레임 스레드 자동 5개,
[`REPORT.md §4.1`](../07_results/pipeline_ffmpeg_vulkan_hailo/REPORT.md)).
앱은 2026-09-29부터 **프레임 스레드 1개**로 디코드한다. 파이프라인이 NPU 병목(약 51.7 FPS)이라 디코드 처리량은 1스레드(약 266 FPS)로도 넘치고,
프레임 스레딩은 CPU(b8 −14%)와 지연(e2e −8.8 ms)만 늘렸기 때문이다([`dec_threads`](../07_results/dec_threads/summary.md), `YVO_DEC_THREADS=0`이면 자동).

### rpt4 이후 — GStreamer 연구 경로로 다시 비교했다

rpt4에서 복사가 사라졌으니 "GStreamer로도 같은 파이프라인이 되는가"를 직접 확인했다. 앱 본체는 그대로 두고,
연구 경로 `yvo_gst_pipeline`([`01_src/gstelem/`](../01_src/gstelem/))을 따로 만들었다.
`v4l2slh265dec`의 SAND128 DMABuf → 자체 요소 `yvosandletterbox`(앱과 같은 `SandLetterbox` 셰이더) → `yvohailosink`(같은 `HailoRunner`)다.

- **검출은 바이트 동일**(`detections.jsonl`, b1·b4·b8), **FPS도 같다**(NPU 상한 51.7).
- CPU는 GStreamer 쪽이 약 0.2 ms/frame(6%) 낮았다. perf와 인과 실험으로 보니 GStreamer의 디코드 코드가 가벼워서가 아니었다.
  **앱의 FFmpeg 프레임 스레드 비용**이었고, 앱을 1스레드로 돌리면 차이가 74% 줄었다
  ([`gst_pipeline`](../07_results/gst_pipeline/summary.md), [`runs/perf_dec_threads`](../07_results/gst_pipeline/runs/perf_dec_threads/summary.md)).
  이 결과가 위의 1스레드 기본값으로 이어졌다.
- 그래서 앱이 libavcodec을 쓰는 이유는 이제 성능이 아니다. GStreamer 경로는 **패키지 버전에 민감하다**(rpt3에서는 32.7 FPS였다)는 점이 남는다.
  비용을 정하는 것은 프레임워크가 아니라 **복사가 있느냐**와 **스레딩**이다.

---

## 7.8 이 프로젝트에서는 — 스왑체인에 컴퓨트가 직접 쓴다

보통의 설계라면 이렇게 한다: 컴퓨트 셰이더가 자기 이미지에 그린 다음 `vkCmdBlitImage` 로
스왑체인 이미지에 복사한다.

이 프로젝트는 **그 복사를 없앴다.** 컴퓨트 셰이더가 **스왑체인 이미지에 직접 쓴다.**

```
   디코더 SAND dma-buf
        │  (import)
        ▼
   display_convert.comp ──────► 스왑체인 이미지 [ rgba8 storage image ]
        │                              ▲
   overlay_boxes.comp ────────────────┘   같은 이미지에 이어서 덧그린다
        │
        ▼
   vkQueuePresentKHR  →  HDMI
```

가능한 이유: 이 드라이버가 스왑체인 이미지 usage에 **`STORAGE` 를 지원**한다(프로브 P4에서
`usage 0x8009f` 에 STORAGE 비트가 있음을 확인). 없으면 blit 폴백으로 내려가고,
그 폴백 비용도 재 뒀다(약 1.23 ms) — 그래서 `--display-blit` 로 강제할 수 있고 항상 테스트된다.

### ★ 포맷이 `R8G8B8A8_UNORM` 으로 고정된 세 가지 이유

[`vk_swapchain.hpp`](../01_src/display/vk_swapchain.hpp) 가 **"이유 셋이 같은 답을 가리킨다"** 고
적어 둔 대목이다.

**① Vulkan에서 `_SRGB` 포맷은 storage image가 될 수 없다.**
`VK_FORMAT_R8G8B8A8_SRGB` 는 읽을 때 자동으로 sRGB→선형 변환을, 쓸 때 반대를 해 주는 포맷인데,
storage image(셰이더가 직접 쓰는 이미지)로는 쓸 수 없다.

**② `shaderStorageImageWriteWithoutFormat = false` 다.**
이 기능이 있으면 GLSL에서 포맷 한정자 없이 `imageStore` 를 할 수 있다. 없으면
**한정자가 필수**인데, GLSL에는 `bgra8` 이라는 한정자가 **없다.** 쓸 수 있는 것은 `rgba8` 뿐이고,
그것이 곧 `VK_FORMAT_R8G8B8A8_UNORM` 이다.

**③ 셰이더가 내놓는 값이 이미 sRGB 인코딩이다.**
§7.3 끝에서 본 대로, BT.709 변환식의 출력은 이미 감마 인코딩된 0~255다.
`UNORM` 포맷은 **아무 변환도 하지 않고 통과**시킨다(`byte/255.0` 으로 쓰면 UNORM 양자화
`round(v*255)` 에서 정확히 원래 바이트로 되돌아온다).
만약 `_SRGB` 를 썼다면 드라이버가 **한 번 더** 변환해서 색이 틀어졌을 것이다.

즉 **①②는 "다른 선택지가 없다"이고, ③은 "그 선택지가 마침 옳다"** 이다.

검증도 했다: 컴퓨트가 쓴 값이 HDMI 화면까지 **한 비트도 변하지 않고** 도달하는 것을
`grim` 화면 캡처로 확인했다(P5: 픽셀 **오차 0**, P8: **max 오차 0 / 2,073,600 px**).

---

## 7.9 이 프로젝트에서는 — 표시가 디코드를 막으면 안 된다

### `acquire(timeout = 0)` — 절대 블로킹하지 않는다

`vkAcquireNextImageKHR` 에는 timeout 인자가 있다. 보통은 넉넉히 주고 기다린다.

**이 프로젝트는 0을 준다.** 빈 이미지가 없으면 `VK_NOT_READY` 를 받고,
**그 프레임의 표시를 그냥 버린다**(`display_dropped++`).

왜? [`vk_swapchain.hpp`](../01_src/display/vk_swapchain.hpp) 의 한 줄이다:

> "**디코드가 표시 때문에 막히면 안 되기 때문이다.**"

메인 스레드 하나가 디코드·GPU 제출·NPU 제출·표시를 전부 돌린다
([05_vulkan_compute.md](05_vulkan_compute.md) §5.2의 스레드 규율).
여기서 표시가 16.7 ms를 블로킹하면 **파이프라인 전체가 선다.**
표시는 "되면 좋은 것"이지 파이프라인의 주인이 아니다.

획득만 하고 그리지 않은 이미지는 `vkReleaseSwapchainImagesEXT`
(`VK_EXT_swapchain_maintenance1`)로 되돌린다. 확장이 없으면 다음 acquire가 재사용하도록
보관한다 — 이 하드웨어에서는 쓰이지 않는 안전망이다.

> 실제 측정에서 **`display_dropped = 0`** 이었다(900프레임 × 3회, 12회 전부)
> ([`display_e2e/summary.md`](../07_results/display_e2e/summary.md) §3). 버릴 일이 없었다는 뜻이다.
> 하지만 **버릴 수 있게 설계한 것**이 요점이다.

### `VK_KHR_present_id` / `VK_KHR_present_wait` — glass-to-glass를 추측하지 않는다

"화면에 언제 나갔는가"를 재는 것은 생각보다 어렵다. `vkQueuePresentKHR` 가 돌아온 시각은
**제출한 시각**이지 화면에 나간 시각이 아니다.

Vulkan 확장 두 개가 이것을 푼다.

| 확장 | 하는 일 |
|---|---|
| `VK_KHR_present_id` | present마다 **번호(presentId)** 를 붙일 수 있게 한다 |
| `VK_KHR_present_wait` | `vkWaitForPresentKHR(presentId)` — **그 번호의 프레임이 실제로 화면에 나갈 때까지** 기다린다 |

(vk.xml, Vulkan 1.4.309: `VK_KHR_present_wait` 는 `VK_KHR_swapchain + VK_KHR_present_id` 를 요구한다.)

그래서 이 프로젝트는 **glass-to-glass 지연을 실측한다** —
"디코더가 프레임을 돌려준 시각 → `vkWaitForPresentKHR` 가 돌아온 시각".

> 측정값: `--display-pace source` 에서 **median 101.20 ms** (mean 101.13, p90 118.36)
> ([`display_e2e/summary.md`](../07_results/display_e2e/summary.md) §3.2).
> **계획 단계의 추정치는 217 ms였다. 실측이 그 절반 이하였다.**
> 같은 실행의 NPU e2e median이 75.3 ms이므로 **표시 경로가 더하는 지연은 약 26 ms**
> (GPU 변환 4.2 ms + present 1~2 vsync)다.
>
> 이 값들은 V3D 960 MHz · FFmpeg 자동 5스레드 · CPU NMS 조건이다. 현재 기본값(1150 MHz · 1스레드 · GPU NMS)에서는
> 같은 source pace 설정의 glass **mean**이 99.5 ms였다(median은 재지 않음, [`dec_threads`](../07_results/dec_threads/summary.md) 표시 on).

Wayland에도 같은 목적의 `wp_presentation` 프로토콜이 있지만, **스왑체인 경로에서는 일부러 쓰지 않는다** —
"목적이 겹치는 두 수단을 이중으로 두지 않는다"는 판단이다
([`vk_swapchain.hpp`](../01_src/display/vk_swapchain.hpp)).
스왑체인이 없는 HVS 평면 백엔드(§7.12)는 반대로 `wp_presentation`(wayland-planes)이나
KMS flip 이벤트 시각(kms)으로 glass-to-glass 를 잰다.

> ⚠ **present 가 영원히 멈추는 경우** — FIFO 스왑체인은 컴포지터의 frame 콜백을 기다린다.
> labwc 가 HDMI 를 잃고 headless 출력만 쥐고 있으면(예: 활성 VT 가 데스크톱이 아니고 DRM master 를
> 다른 프로세스가 쥔 경우) 콜백이 오지 않고, `vkQueuePresentKHR` 가 드라이버(Mesa WSI) 안에서 무기한 기다린다.
> 우리 SIGINT/SIGTERM 핸들러는 플래그만 세우므로 **SIGTERM 으로도 끝나지 않는다**(SIGKILL 필요).
> 원인 설명은 Mesa 소스로 확인하지 않은 추정이다. 실행 전에 `wlr-randr` 에 HDMI 출력이 있는지 본다.
> HVS 평면 백엔드는 frame 콜백을 50 ms 까지만 기다리므로 이 함정에 걸리지 않는다.

---

## 7.10 ⚠ 두 개의 함정 — 에러 없이 느려지는 것들

이 절의 두 가지는 **프로그램이 정상으로 보이면서 성능만 무너지는** 경우다.
측정을 할 사람에게 가장 위험한 종류다. (이 절의 GPU ms는 모두 V3D 960 MHz 조건이다. 1150 MHz에서는 다시 재지 않았다.)

### 함정 ① 창 모드면 2.6배 느려진다

`display_convert.comp` 에는 경로가 두 개 있다(spec constant `kUniformScale`).

- **`kUniformScale = false` — 1:1 경로.** 출력 크기 = 원본 크기.
  루마 샘플이 **정확히 하나**고 보간이 없다. 4픽셀당 로드 5회로 끝난다.
- **`kUniformScale = true` — 균일 스케일 경로.** 4-tap bilinear + 열 LUT.
  `sand_letterbox.comp` 와 같은 구조다.

경로 선택은 [`display_pipeline.cpp`](../01_src/display/display_pipeline.cpp) 가
**출력 크기와 원본 크기를 비교해서 자동으로** 한다.

문제가 여기서 터진다. **labwc가 창 모드에서 높이를 1044로 준다**(1080 − 36, 타이틀바 몫).
1044 ≠ 1080이므로 자동으로 `kUniformScale` 경로로 넘어간다.

```
   전체화면 (1920×1080 = 원본)  →  1:1 경로      →   4.22 ms
   창 모드  (1920×1044 ≠ 원본)  →  스케일 경로   →  10.92 ms    ★ 2.6배
```

**에러도 경고도 없다.** 화면은 멀쩡히 나온다. 그냥 느리다.

> 그래서 **측정에는 `--display-fullscreen` 이 필수**다.
> [`README.md`](../README.md) §4가 "**반드시 켠다**"고 적어 두었고,
> [`display_e2e/summary.md`](../07_results/display_e2e/summary.md) §1도 경고를 반복한다.
> 근본 해법은 `wp_viewporter` 로 스왑체인을 1920×1080 고정으로 잡고 축소를
> **Pi 5의 HVS 하드웨어**에 맡기는 것이다. **스왑체인 경로에는 아직 구현하지 않았다.**
> `wp_viewporter` 는 HVS 평면 백엔드(§7.12)에서만 bind 한다. 그 백엔드는 변환 자체를 HVS 가 하므로 이 함정이 없다.

### 함정 ② 표시 해상도를 낮추면 **더 비싸진다**

"GPU가 부담되면 720p로 낮추자"가 상식이다. **여기서는 반대다.**

| 표시 크기 | 경로 | 시간 | 픽셀당 |
|---|---|---:|---:|
| **1920×1080** | 1:1 | **4.23 ms** | **2.04 ns/px** |
| 1280×720 | 스케일 | 5.27 ms | **5.72 ns/px** |
| 1600×900 | 스케일 | 7.64 ms | 5.31 ns/px |

> 픽셀당 값은 원문에 없고 `시간 ÷ 픽셀 수` 로 계산한 것이다:
> 4.229 ms ÷ 2,073,600 = 2.04 / 5.27 ms ÷ 921,600 = **5.72** / 7.64 ms ÷ 1,440,000 = 5.31.

([`display_vulkan_hdmi/summary.md`](../07_results/display_vulkan_hdmi/summary.md) 결론 4)

**픽셀 수는 720p가 1080p의 44%인데 시간은 더 든다.**
1:1 경로는 보간이 없어 픽셀당 2.04 ns인데, 스케일 경로는 4-tap bilinear라 **픽셀당 5.3~5.7 ns** 다.
720p 기준 **2.8배** 비싼 픽셀을 44% 개수만큼 처리하면 1.25배가 된다(4.23 → 5.27 ms).

**교훈**: "해상도를 낮추면 싸진다"는 **픽셀당 비용이 같을 때만** 참이다.
경로가 갈리면 성립하지 않는다.

---

## 7.11 이 프로젝트에서는 — 닭과 달걀 문제

마지막으로 구조 이야기 하나. 표시 경로를 만들려면 이 셋이 필요하다.

- `WaylandWindow` — Wayland 창
- `VulkanContext` — Vulkan 인스턴스·장치·큐
- `VulkanSwapchain` — surface와 스왑체인

순서를 정하려는데 순환이 생긴다.

```
   VulkanContext 를 만들려면
     → "present 가능한 큐 패밀리"를 골라야 하고
     → vkGetPhysicalDeviceSurfaceSupportKHR 는 VkSurfaceKHR 를 요구하고
     → VkSurfaceKHR 는 wl_surface 를 요구하고
     → wl_surface 는 WaylandWindow 를 요구하는데
     → ...  VkSurfaceKHR 를 만들려면 VkInstance 가 먼저 있어야 한다   ← 순환!
```

### 해법 — surface가 필요 없는 질의가 하나 있다

```
   vkGetPhysicalDeviceWaylandPresentationSupportKHR(pd, queueFamily, wl_display)
                                                                     ↑
                                          surface 가 아니라 wl_display 만 받는다 ★
```

그래서 순서를 이렇게 끊었다.

```
   ① WaylandWindow::open()          →  wl_display* 를 얻는다
   ② VulkanContext(wayland_display = 그 포인터)
         · 위 함수로 present 가능한 큐 패밀리를 고른다
         · ★ VkSurfaceKHR 를 영원히 모른다
   ③ VulkanSwapchain
         · ★ 여기서 처음으로 VkSurfaceKHR 를 만든다
```

### 그리고 의존성이 늘지 않게 막았다

이 분리에는 부수 효과가 하나 더 있다.

- `WaylandWindow` 는 **Vulkan 심볼을 하나도 쓰지 않는다.** `wl_display*`/`wl_surface*` 를
  `void*` 로만 내보낸다. "순수 플랫폼 글루"다.
- `VulkanContext` 는 `<wayland-client.h>` 도 `vulkan_wayland.h` 도 **include하지 않는다.**
  wl_display를 불투명 포인터(`void*`)로 받는다.

> "이 헤더가 `<wayland-client.h>` 나 `vulkan_wayland.h` 를 포함하지 않는 것도 같은 이유다
> (`yvo_vulkan` 의 의존성을 늘리지 않는다)."
> — [`vulkan_context.hpp`](../01_src/vulkan/vulkan_context.hpp)

그 결과 **Wayland가 없는 환경에서도 나머지가 그대로 빌드되고 돌아간다**
([`README.md`](../README.md) §2: "표시 경로는 wayland 가 없으면 조용히 빠지고 나머지는 그대로 빌드된다").
표시 코드가 별도 라이브러리(`yvo_display`)로 분리돼 있는 것도 같은 회귀 격리 목적이다.

그리고 **소멸 순서**도 이 의존의 역순이어야 한다 — 창이 컨텍스트보다 늦게 죽어야 하므로
[`main.cpp`](../01_src/pipeline/main.cpp) 는 `WaylandWindow` 를 `VulkanContext` **보다 먼저 선언**한다
(C++는 선언의 역순으로 파괴하므로).

> 이 순환은 **스왑체인 경로의 문제**다. HVS 평면 백엔드(§7.12)는 스왑체인이 없어 `VulkanContext` 를
> present 없이 만든다. `wayland-planes` 는 창만 먼저 열고, `kms` 는 창 없이 `KmsScanout`(DRM master 획득)을
> `VulkanContext`·NPU 보다 **먼저** 만든다. master 를 못 얻으면 비싼 초기화 전에 종료 코드 2로 끝내기 위해서다.

---

## 7.12 이 프로젝트에서는 — HVS 평면 스캔아웃 (`wayland-planes` / `kms`)

§7.8의 경로는 V3D가 NV12 SAND를 RGB로 바꾸고 박스를 그린 뒤 스왑체인으로 넘긴다. 그런데 Pi 5의 디스플레이
컨트롤러 **HVS**는 원래 **평면(plane)** 여러 장을 스캔아웃하면서 스케일·YUV→RGB 변환·알파 블렌딩을 하드웨어로 한다.
그래서 두 번째 표시 방식을 만들었다(2026-09-22). 기존 경로는 그대로 두고 `--display-backend` 로 고른다.

```
   기존 (wayland, 기본)   디코더 SAND ─▶ display_convert.comp(V3D) ─┐
                                     overlay_boxes.comp(V3D) ───┴▶ RGBA 스왑체인 ─▶ labwc ─▶ HVS
   HVS 평면 (wayland-planes / kms)
                          디코더 SAND ─────────────(변환 없음)──────▶ HVS 영상 평면 (plane 83, NV12 SAND128)
                          overlay_boxes_buf.spv(V3D) ─▶ AB24 dma-buf ▶ HVS 박스 평면 (plane 107, zpos 위)
                                                              HVS 가 YUV→RGB + 블렌딩 → HDMI
```

| 값 | 전송 | 필요한 것 |
|---|---|---|
| `wayland-planes` | 부모 wl_surface 에 NV12 SAND `wl_buffer`(linux-dmabuf), **sync subsurface** 에 박스 `wl_buffer`. 평면 배치는 labwc 가 한다 | labwc, **전체화면**(강제한다. 창 모드면 labwc 가 GPU로 합성해 버린다) |
| `kms` | KMS atomic 으로 두 평면을 직접 커밋한다 | card1 의 **DRM master**. 데스크톱이 떠 있으면 못 얻는다(아래) |

코드: [`plane_display.hpp`](../01_src/display/plane_display.hpp)(공통부 `PlaneDisplay` 와 전송 인터페이스 `PlaneScanout`),
[`wayland_plane_scanout.hpp`](../01_src/display/wayland_plane_scanout.hpp), [`kms_scanout.hpp`](../01_src/display/kms_scanout.hpp),
[`overlay_plane.hpp`](../01_src/display/overlay_plane.hpp)(박스 버퍼), [`drm_state.hpp`](../01_src/display/drm_state.hpp)(debugfs 판정).
`main.cpp` 는 두 경로를 [`display_sink.hpp`](../01_src/display/display_sink.hpp) 인터페이스 하나로만 본다.

### 박스 평면 — V3D는 linear 이미지에 쓰지 못한다

HVS가 읽을 박스 버퍼는 **LINEAR** 배치여야 한다(평면 `IN_FORMATS` 에 AB24 LINEAR). 그런데 이 장치는
`R8G8B8A8_UNORM` 의 `linearTilingFeatures` 에 **STORAGE_IMAGE 비트가 없다**(DRM modifier LINEAR도 같다).
컴퓨트가 LINEAR 이미지에 `imageStore` 할 수 없다는 뜻이다. 그래서 박스는 **storage buffer** 에 그린다 —
`overlay_boxes.comp` 를 `YVO_OVERLAY_BUFFER` 로 한 번 더 컴파일한 `overlay_boxes_buf.spv` 가 packed RGBA를 pitch 단위로 쓴다
(빌드 문법은 [05장](05_vulkan_compute.md) 빌드 파이프라인). 메모리 바이트 순서 R,G,B,A = DRM `ABGR8888`(AB24)이다.

- 버퍼는 3개짜리 링이다. 다시 쓰려면 **GPU 완료와 스캔아웃 release 를 둘 다** 받아야 한다.
- 셰이더는 박스 픽셀만 쓰므로 이전 프레임 박스를 지워야 한다. 8.3 MB 전체를 지우면 비싸서,
  **직전에 그린 검출을 같은 셰이더로 (0,0,0,0) 다시 그리는** 방식(erase)을 기본으로 한다.
- 외부 메모리라 매 프레임 끝에 우리 큐 → `VK_QUEUE_FAMILY_FOREIGN_EXT` 로 소유권을 넘기고, 재사용할 때 되받는다.

### 프레임을 더 오래 쥔다

기존 경로는 GPU 변환이 끝나면(submit 펜스) 디코더 프레임을 놓는다. 평면 경로에서는 **HVS가 디코더 버퍼 자체를
스캔아웃**하므로, 그 버퍼가 화면에서 내려갈 때까지(`wl_buffer.release` / 다음 flip) 놓을 수 없다.
`display_done` 이 1~2 vsync 늦어지고, 디코더 프레임 보유 계산도 이 백엔드용으로 따로 한다(`main.cpp` 보유 가드).

### 커밋 간격 게이트

`--batch 2` 는 검출을 두 장씩 한꺼번에 내놓는다. 준비되는 대로 커밋하면 present 간격이 16.7 ms / 66.7 ms 로
번갈아 나온다(judder). 그래서 `--display-pace source` 에서는 커밋 간격을 **소스 간격 − 8 ms** 이상으로 벌린다.
대가는 짝의 두 번째 프레임이 한 vblank 늦는 것이다. 지연이 중요하면 `--display-sync latest --display-pace asap`.
→ [`notes/wayland_planes_pacing_gate.md`](../07_results/display_vulkan_hdmi/notes/wayland_planes_pacing_gate.md)

### 색 범위 — 컴포지터가 뒤집는다 (조용히 틀리는 함정 하나 더)

평면에 NV12를 올리면 YUV→RGB 계수와 range 를 **평면 속성**(`COLOR_ENCODING`, `COLOR_RANGE`)이 정한다.
§7.3의 BT.709 limited 를 여기서도 맞춰야 한다.

- **kms**: 앱이 매 커밋 `BT.709` / `limited` 를 직접 넣는다.
- **wayland-planes**: color-representation 프로토콜로 요청하는데, labwc 0.20.1~0.20.2 + wlroots 0.20.2+rpt3~rpt4 조합은
  **range 를 뒤집어서** 커널에 넘긴다(limited 를 요청하면 full 로 나간다). 요청하지 않으면 BT.601 full 이 된다.
  그래서 기본값(`--display-color-range auto`)은 **full 을 요청해 limited 가 나오게** 하는 우회("fullhack")다.
  컴포지터가 업데이트되면 조용히 틀어질 수 있으므로 **debugfs 로 실제 평면 상태를 확인한다**(아래). 2026-10-07 업그레이드 뒤 재판정했고 반전은 그대로였다.
  → [`notes/kms_vs_wayland.md`](../07_results/display_vulkan_hdmi/notes/kms_vs_wayland.md) §4 · §4.1(재판정)

### debugfs 로 "정말 평면에 올라갔는가"를 판정한다

"zero-copy 로 스캔아웃된다"는 말은 로그만으로는 믿을 수 없다(창 모드처럼 컴포지터가 조용히 GPU 합성으로 바꿀 수 있다).
`PlaneDisplay` 는 시작 후와 종료 직전에 `/sys/kernel/debug/dri/1/state` 를 읽어 판정한다.

```
[scanout] debugfs start (presented 60): active=2 video=plane[83] NV12/SAND128 zpos=0
          enc='ITU-R BT.709 YCbCr' range='YCbCr limited range' | boxes=plane[107] AB24 zpos=1 | hvs_underrun=0
```

요약의 `scanout zero_copy 1, color_ok 1, hvs_underrun delta 0` 과 RESULT_CSV2 뒤쪽 6필드가 이 판정이다.
debugfs 를 읽지 못하면 `unverified` 로 남는다.

### kms 는 데스크톱을 끄지 않고 돌린다

KMS 커밋에는 card1 의 DRM master 가 필요하고, 평소에는 labwc 가 쥐고 있다. lightdm 을 멈추면 데스크톱 세션이
통째로 죽는다. 대신 **데스크톱이 아닌 VT 를 활성화**하면 logind 가 labwc 의 master 를 거둬 가고, 그때 card1 을
연 프로세스가 master 를 자동으로 받는다.

```bash
loginctl list-sessions                  # tty 세션 id 확인 (sudo 불필요)
loginctl activate <tty 세션 id>          # 또는 sudo chvt 3 — 모니터가 콘솔로 바뀐다
build/bin/yvo_pipeline ... --display on --display-backend kms
loginctl activate <데스크톱 세션 id>      # 또는 sudo chvt 7 — labwc 가 master 를 되찾는다
```

- **실행 중에 VT 를 전환하지 않는다.** labwc 가 master 를 못 받아 데스크톱이 검게 남을 수 있다.
  종료는 Ctrl+C / SIGTERM 으로 먼저 한다.
- master 를 못 얻으면 누가 쥐고 있는지를 알려 주고 NPU·GPU 초기화 전에 종료 코드 2로 끝난다.
- 종료 때 이전 KMS 상태 복원을 시도하지만 지금은 `EINVAL` 로 실패하고 평면 끄기로 폴백한다
  (labwc 의 FB 가 바뀐 것으로 추정). 데스크톱으로 돌아가면 labwc 가 화면을 새로 잡으므로 실사용 문제는 없다.
- Xwayland 가 비활성 VT 동안 card1 을 새로 열면 **Xwayland 가 master 를 가져가** labwc 가 headless 로 남는다.
  이때는 Xwayland 를 끝내고 VT 를 한 번 더 토글한다.
- 첫 커밋은 블로킹 모드셋이라 그 직후 프레임의 지연·간격은 측정에서 뺀다(`--display-warmup N` — 앞 N 표시분을 glass·present 간격·늦은 프레임 통계에서 제외. 기본 0, `display_planes_bench.sh` 는 모든 설정에 30). 평면 경로 요약의 `late frames` 는 vblank seq 간격이 정상 cadence 의 최대 간격 max(1, ceil(every × refresh/소스 fps − 0.01))을 넘은 수다(`RESULT_CSV3`, 플립 타임아웃 등으로 체인이 끊긴 횟수는 `seq_breaks`).

### 무엇이 확인됐고 무엇이 남았나

두 백엔드 모두 실제 HDMI 에서 영상 평면 `NV12/SAND128` + 박스 평면 `AB24`(영상 위), `BT.709 / limited`,
`hvs_underrun` 증가 0, V3D 표시 변환 0을 debugfs 로 확인했다. 검증 레이어 errors 0, ASan 0, `--nms verify` 불일치 0.
기존 경로와 비교한 정식 측정은 2026-10-08 에 했다([`display_planes_e2e/`](../07_results/display_planes_e2e/summary.md), 회차 `runs/slots6`).
Vulkan 합성 대비 CPU −0.44 ms/f·V3D −72%(TFU 3.7 ms 소멸), 디코드 → 화면 지연은 같다. 평면 경로는 디코더 버퍼를 다음 flip 까지 쥐어
최대 5장을 보유하므로 **`--slots 6`** 이 필요하다(슬롯 4에서는 디코드 후 28~30 ms 대기, glass-to-glass 에는 안 잡힌다 — 그래서 `decode-to-screen` 지표를 따로 둔다).
열린 항목은 [`07_results/README.md`](../07_results/README.md) 에 있다.

---

## 7.13 자가 점검

**Q1. 이 보드에서 H.264 영상을 쓰면 왜 느린가?**

BCM2712에는 **HEVC(H.265) 디코더 회로만** 있고 H.264 회로가 없다. H.264는 CPU가 소프트웨어로
풀어야 해서 1080p 기준 **20.2 ms/frame** 이 든다(HW HEVC는 ffmpeg CLI로 **2.0 ms/frame**인데 대부분 CPU 내려받기 비용이다. 복사 없는 앱 경로는 0.23~0.28 ms).
그래서 이 프로젝트의 디코더는 아예 HEVC가 아니면 거부한다
(`ffmpeg_drm_decoder.cpp`: `if (st->codecpar->codec_id != AV_CODEC_ID_HEVC) return fail(...)`).
근거: [`decode_benchmark/summary.md`](../07_results/decode_benchmark/summary.md).

**Q2. 같은 HW 디코더를 쓰는데 GStreamer 경로가 10배 느렸던 이유는?**

rpt3 패키지의 `v4l2slh265dec`가 **CPU 한 스레드로 프레임 전체를 복사**했다(perf에서 CPU 시간의 약 90%가 ORC 생성 복사 코드).
32.7 FPS / CPU 30.6 ms/frame이었고, **HW 디코드를 쓰고도 SW H.265(31.8 ms)와 CPU 비용이 같았다.**
당시에는 SAND128 배치 탓으로 봤지만, 원인은 1080p의 (0,0) 크롭 창에서 zero-copy가 깨지는 upstream 버그(1.26.4에서 수정)였다.
rpt4에서는 복사가 사라져 0.24 ms/frame이다.
당시의 해법은 GStreamer를 거치지 않고 **앱에 libavcodec DRM_PRIME 디코더를 직접 넣어** dma-buf fd를
Vulkan으로 import하고 SAND 주소 계산을 셰이더가 하게 한 것이다 → 398.5 FPS / 0.23~0.28 ms.
rpt4 이후 GStreamer 연구 경로도 검출·FPS가 앱과 같다(§7.7).

**Q3. 스왑체인 포맷이 왜 `R8G8B8A8_UNORM` 인가? `_SRGB` 를 쓰면 안 되는가?**

이유 셋이 같은 답을 가리킨다.
① Vulkan에서 **`_SRGB` 포맷은 storage image가 될 수 없다**(컴퓨트가 직접 못 쓴다).
② 이 장치는 `shaderStorageImageWriteWithoutFormat = false` 라 GLSL 포맷 한정자가 필수인데
**GLSL에 `bgra8` 이 없으므로 `rgba8` 뿐**이다.
③ BT.709 변환식이 내놓는 값은 **이미 감마 인코딩된 sRGB 0~255** 라 `UNORM` 이면 변환 없이 통과한다
(`_SRGB` 였다면 드라이버가 한 번 더 변환해 색이 틀어졌을 것이다).
근거: [`vk_swapchain.hpp`](../01_src/display/vk_swapchain.hpp), [`display_convert.comp`](../02_shaders/display_convert.comp) 머리 주석.

**Q4. 표시 해상도를 720p로 낮추면 GPU가 싸지는가?**

**아니다. 오히려 비싸진다.** 1920×1080은 출력이 원본과 같아 **1:1 경로**(보간 없음, 픽셀당 2.05 ns)로
**4.23 ms** 인데, 1280×720은 **균일 스케일 경로**(4-tap bilinear, 픽셀당 5.72 ns)로 **5.27 ms** 다.
픽셀 수가 44%로 줄어도 픽셀당 비용이 2.6배라 총합이 늘어난다.
"해상도를 낮추면 싸진다"는 **픽셀당 비용이 같을 때만** 참이다.
근거: [`display_vulkan_hdmi/summary.md`](../07_results/display_vulkan_hdmi/summary.md) 결론 4.

**Q5. 측정할 때 `--display-fullscreen` 이 왜 필수인가?**

창 모드에서 labwc가 창 높이를 **1044**(1080−36)로 주는데, 출력 크기가 원본과 달라지므로
`display_convert.comp` 가 자동으로 `kUniformScale` 경로로 넘어간다 →
변환이 **4.22 → 10.92 ms, 2.6배** 느려진다. **에러도 경고도 없고 화면은 멀쩡히 나온다.**
그래서 성능 수치가 조용히 오염된다.
근거: [`display_vulkan_hdmi/summary.md`](../07_results/display_vulkan_hdmi/summary.md) 결론 5,
[`display_e2e/summary.md`](../07_results/display_e2e/summary.md) §1.

**Q6. `acquire(timeout = 0)` 으로 하고 `VK_NOT_READY` 면 프레임을 버리는 이유는?**

메인 스레드 하나가 디코드·GPU 제출·NPU 제출·표시를 전부 돌리기 때문이다
(`VkQueue` 가 외부 동기화 객체라 Vulkan은 메인 스레드에서만 만진다 →
[05_vulkan_compute.md](05_vulkan_compute.md) §5.2). 여기서 표시가 vsync 한 박자(16.7 ms)를
블로킹하면 **파이프라인 전체가 선다.** **표시가 디코드를 막으면 안 된다**는 것이 설계 원칙이고,
표시는 "되면 좋은 것"이지 주인이 아니다. 실측에서는 `display_dropped = 0` 이었지만
**버릴 수 있게 설계한 것**이 요점이다.

**Q7. letterbox의 `+0.5 / -0.5` 는 왜 붙는가? 빼면 어떻게 되는가?**

픽셀은 점이 아니라 사각형이고 그 **중심**이 `dx + 0.5` 이기 때문이다. 축소할 때 중심끼리
맞춰야 해서 `sx = (dx - pad_x + 0.5)/scale - 0.5` 가 된다(`cv::resize` 의 `INTER_LINEAR` 와 같은 규칙).
빼먹으면 이미지가 반 픽셀 밀린다 — 눈으로는 안 보이지만 박스 좌표가 틀어지고,
무엇보다 **CPU 기준 구현과의 비트 일치가 깨져서 "GPU가 맞는지" 검증할 수단이 사라진다**
(`test_preproc` 은 NHWC·NHCW 양쪽에서 **max 오차 0** 을 요구한다).
근거: [`cpu_letterbox.hpp`](../01_src/ref/cpu_letterbox.hpp) 상단 수식 주석 2)~4).

**Q8. HVS 평면 백엔드에서 박스를 왜 이미지가 아니라 버퍼에 그리는가?**

HVS 박스 평면은 LINEAR 배치 버퍼를 읽는데, 이 장치의 `R8G8B8A8_UNORM` 은 **LINEAR 타일링에서 STORAGE_IMAGE 를 지원하지 않는다**.
컴퓨트가 LINEAR 이미지에 쓸 수 없으니 같은 셰이더를 storage-buffer 변형(`overlay_boxes_buf.spv`)으로 컴파일해
packed RGBA 를 pitch 단위로 쓴다. 근거: [`overlay_plane.hpp`](../01_src/display/overlay_plane.hpp) 머리 주석, 이 장 §7.12.

**Q9. wayland-planes 에서 limited range 를 원하는데 왜 full 을 요청하는가?**

labwc 0.20.1~0.20.2 + wlroots 0.20.2+rpt3~rpt4 조합이 color-representation 의 range 를 뒤집어 커널에 넘기기 때문이다. 요청 full → 평면 limited.
컴포지터 버그에 기댄 우회라서 debugfs 로 실제 `COLOR_RANGE` 를 매 실행 확인한다(`scanout color_ok`). 2026-10-07 재판정에서도 반전은 그대로였다.
근거: [`notes/kms_vs_wayland.md`](../07_results/display_vulkan_hdmi/notes/kms_vs_wayland.md) §4·§4.1.

---

**다음** → [08_method.md](08_method.md) (이 수치들을 어떻게 재고 어떻게 믿는가)
또는 코드 구조부터 보려면 [09_code_map.md](09_code_map.md),
설계 결정의 사슬을 따라가려면 [10_why_this_design.md](10_why_this_design.md).
