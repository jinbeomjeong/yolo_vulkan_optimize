# ffmpeg HW HEVC 디코드 결과를 GStreamer로 넘기는 방법

- 조사 일시: 2026-09-20
- 환경: Raspberry Pi CM5 (BCM2712), GStreamer 1.26.2, ffmpeg(Raspberry Pi 빌드, `-hwaccel drm`), 입력 `04_data/videos/video_h265.mp4`(HEVC Main 1080p30, 1816프레임)
- 배경: [summary.md](../summary.md)에서 같은 HEVC HW 디코더(`rpi-hevc-dec`)가 경로에 따라 크게 달랐다. ffmpeg는 333 FPS·2 ms/frame, GStreamer `v4l2slh265dec`는 32.7 FPS에 1코어가 포화됐다

## 1. gst-libav로는 안 된다
`avdec_h265`에는 hwaccel 관련 속성이 없다(`max-threads`, `direct-rendering`, `lowres` 등만 있다). GStreamer 안에서 ffmpeg의 HW 디코드를 켤 방법이 없다.

## 2. GStreamer HW 경로가 느린 원인: v4l2codecs 내부 CPU 복사

디코더가 내보낼 수 있는 포맷은 모두 1920x1088이다.
- `video/x-raw(memory:DMABuf), format=DMA_DRM, drm-format=NV12:0x0700000000000004`: Broadcom SAND128 modifier
- `video/x-raw, format=NV12_128C8`: 시스템 메모리 SAND
- 10bit 변형(P030, NV12_10LE32_128C8)

| 실험 | 결과 |
|---|---|
| DMA_DRM caps → `fakesink` | 협상 실패. `DMABuf caps negotiated without the mandatory support of VideoMeta` |
| DMA_DRM caps → `fakevideosink`(VideoMeta 지원) | 동작은 하지만 54.8초가 걸린다(≈33 FPS). user 54.5초로 1코어가 포화된다 |
| DMA_DRM caps → `vulkanupload` | link 실패. vulkanupload는 DMA_DRM을 받지 않는다 |

- perf 콜체인(DMABuf 출력 상태에서 측정): `gst_base_parse_push_frame` → `libgstcodecs`(output_picture) → `libgstv4l2codecs` → `libgstvideo`(프레임 복사) → ORC 코드
- CPU 시간의 약 90%가 이 ORC 복사였다
- 다운스트림이 DMABuf를 받아도 디코더 내부에서 프레임을 CPU로 복사한다. copy_frames 판단 조건 같은 정확한 원인은 v4l2codecs 소스를 확인해야 확정된다

## 3. 지금 바로 쓸 수 있는 방법: ffmpeg → 파이프 → GStreamer

```bash
ffmpeg -hide_banner -loglevel error -nostdin -hwaccel drm -i in.mp4 -an -fps_mode passthrough -f rawvideo -pix_fmt yuv420p - \
  | gst-launch-1.0 fdsrc fd=0 blocksize=3110400 \
      ! rawvideoparse width=1920 height=1080 format=i420 framerate=30/1 \
      ! videoconvertscale n-threads=4 add-borders=true \
      ! video/x-raw,format=RGB,width=640,height=640,pixel-aspect-ratio=1/1 ! ...
```

- ffmpeg가 HW 디코드한 SAND 프레임을 yuv420p로 내려받아 stdout으로 내보낸다
- GStreamer는 `fdsrc` + `rawvideoparse`로 프레임 경계를 복원한다
- `blocksize`는 한 프레임 크기(1920×1080×1.5)다
- `-fps_mode passthrough`가 없으면 ffmpeg가 CFR 출력을 맞추려고 프레임 1개를 복제한다(1817프레임, `dup=1`)
- 애플리케이션 안에서 쓸 때는 같은 방식으로 `appsrc`에 넣거나, `fdsrc`를 쓰는 서브프로세스로 구성할 수 있다

### 사전 측정 (1회, bash `time`, ffmpeg와 gst CPU 합산)

| 경로 | FPS | CPU ms/frame | 비고 |
|---|---:|---:|---|
| 파이프, 디코드만 | 147.7 | 11.6 | user 13.9 s, sys 7.1 s |
| 파이프 + 640x640 RGB 변환 | 112.8 | 23.0 | user 31.8 s, sys 10.1 s |
| GStreamer v4l2slh265dec 디코드만 / 변환 포함 | 32.7 / 30.3 | 30.6 / 39.4 | 3회 중앙값 |
| avdec_h264 1080p 디코드만 / 변환 포함 | 164.6 / 116.4 | 20.2 / 28.1 | 3회 중앙값 |
| 참고: ffmpeg 단독 → null | 332.8 | 2.0 | null muxer, 파이프·파일 출력 없음 |

### 정식 측정 (3회 중앙값, `h265_E_ffpipe_*` 조합, ffmpeg + gst CPU 합산)

| 경로 | FPS (최소~최대) | CPU (4코어 기준) | CPU ms/frame | max RSS |
|---|---:|---:|---:|---:|
| 파이프, 디코드만 (`h265_E_ffpipe_S1`) | 146.9 (145.6~147.1) | 42.5% | 11.7 | 246 MB |
| 파이프 + 640x640 RGB (`h265_E_ffpipe_S2_nt4`) | 109.2 (108.2~111.6) | 64.1% | 23.6 | 180 MB |

- 모든 실행에서 1816프레임을 처리했고, 편차는 1.1%와 3.1%다. 사전 측정값과의 차이는 ±5% 안이다
- CPU는 bash의 wait4 rusage로 잰다. 커널이 ffmpeg와 gst-launch의 사용량을 bash에 누적한다. max RSS는 세 프로세스 중 최대값이다
- 측정 중 SoC 온도는 73~78 °C였다. 기존 조합(냉각 기준 65 °C)과 온도 조건이 완전히 같지는 않다

### 장단점
- 장점
  - 추가 코드 없이 지금 쓸 수 있다
  - GStreamer HW 경로보다 CPU가 약 2.6배 싸고(디코드만 기준) 처리량은 약 4.5배다
  - SW H.264 디코드보다도 싸다
- 단점
  - 프레임당 3 MB를 파이프로 복사한다. sys 시간이 약 4 ms/frame이다
  - SAND→yuv420p 변환도 CPU로 한다
  - 프로세스가 두 개라 타임스탬프와 EOS 등 제어가 번거롭다. 오디오는 버려야 한다(`-an`)

## 4. zero-copy 경로 (다음 단계 후보)

필요한 조건은 갖춰져 있다.
- V3DV(Vulkan)가 `VK_EXT_external_memory_dma_buf`, `VK_EXT_image_drm_format_modifier`, `VK_KHR_external_memory_fd`를 지원한다
- libavcodec에 `hw_device_ctx`(DRM)를 설정하면 `AV_PIX_FMT_DRM_PRIME` 프레임을 받는다. `AVDRMFrameDescriptor`에 dma-buf fd, SAND128 modifier, plane별 offset·pitch가 들어 있다

구성안은 두 가지다.
- (a) **appsrc DMABuf**: fd를 `gst_dmabuf_allocator_alloc`으로 감싼다. caps는 `video/x-raw(memory:DMABuf),format=DMA_DRM,drm-format=NV12:0x0700000000000004`에 GstVideoMeta(offset/stride)를 붙여 GStreamer로 넘긴다. 다만 이 caps를 받는 기존 요소가 거의 없어(vulkanupload 불가) 결국 자체 요소나 appsink가 필요하다
- (b) **Vulkan 직접 import (권장)**: fd를 `VkBuffer`로 import하고, 컴퓨트 셰이더가 SAND128 타일을 직접 풀어 RGB letterbox 640x640 결과를 만든다. 이 결과를 Hailo 입력 버퍼로 넘긴다

V3DV가 SAND modifier를 `VkImage`로 직접 지원하는지는 확인하지 못했다. 버퍼로 import하면 이 문제를 피할 수 있다.

예상 효과: 디코드와 전처리의 CPU 비용이 약 2 ms/frame 수준까지 줄 수 있다(ffmpeg 단독 값 기준, GPU 셰이더 시간은 별도).

담당 분리: libavcodec 디코드와 fd 전달은 C++(`cpp-expert` 또는 `gstreamer-expert`), Vulkan import와 SAND 셰이더는 `vulkan-expert`에 맡긴다. 먼저 인터페이스(fd, 1920x1088, NV12 SAND128, plane offset/pitch, 동기화 시점)를 정한다.
