# P0.5 / P10 프로브 결과 (2026-09-20, 코드 변경 전)

환경: Raspberry Pi CM5, V3D 7.1.10.2 (Mesa 26.2.2), labwc(Wayland) on HDMI-A-1 1920x1080@60,
SSH 에서 `XDG_RUNTIME_DIR=/run/user/1000 WAYLAND_DISPLAY=wayland-0`.

## P0.5 — labwc 가 Wayland 클라이언트를 받는가: **통과**

```bash
gst-launch-1.0 -q videotestsrc pattern=smpte ! video/x-raw,width=640,height=480,framerate=30/1 ! waylandsink
grim -o HDMI-A-1 shot.png
```
- gst 로그 에러/경고 0줄, `grim` 캡처 성공 (1920x1080 PNG)
- SMPTE 6색이 모두 검출됨: yellow 29120 / cyan 33120 / green 29120 / magenta 33120 / red 29120 / blue 33080 픽셀
- **판정: HDMI 에 실제로 떴다.** `grim` 기반 객관적 화면 도달 판정 방식이 작동함도 함께 확인

## P10 — 디코더 SAND128 dma-buf 를 컴포지터에 직접 줄 수 있는가

### 1차 시도 (waylandsink): 실패 — 그러나 원인이 labwc 가 아니다

```bash
gst-launch-1.0 filesrc location=04_data/videos/video_h265.mp4 ! qtdemux ! h265parse \
  ! v4l2slh265dec ! waylandsink
```
디코더 출력 caps 는 정확히 우리가 쓰는 것과 같다:
```
video/x-raw(memory:DMABuf), format=DMA_DRM, 1920x1080, colorimetry=bt709,
drm-format=NV12:0x0700000000000004
```
그런데 sink 가 거부했다:
```
waylandsink gstwaylandsink.c:1062 error: buffer ... size 3110400 ... cannot have a wl_buffer
```

### 2차 확인 (직접 작성한 Wayland 클라이언트): **labwc 는 SAND128 을 지원한다**

`zwp_linux_dmabuf_v1` 을 version 3 으로 bind 해 format/modifier 표를 덤프했다
(소스: `dmabuf_modifier_dump.c`, `wayland-scanner` + `libwayland-client` 만 사용).

```
zwp_linux_dmabuf_v1 advertised (compositor version=4) -> bind as version 3
wl_subcompositor       advertised (version=1)
wp_presentation        advertised (version=2)
wp_viewporter          advertised (version=1)
  NV12  modifier 0x0700000000000006
  NV12  modifier 0x0000000000000000
  NV12  modifier 0x0700000000000004      <-- 우리 디코더의 modifier
  NV12  modifier 0x00ffffffffffffff
NV12 modifier 총 187개 이벤트 중
  SAND128(0x0700000000000004) 지원: 예
  LINEAR(0x0)             지원: 예
```

**결론: 1차 실패는 `waylandsink` 구현의 한계이고 labwc 의 제약이 아니다.**
즉 계획서 §5.3 의 안 (III)(SAND 를 wl_surface 에 직접 붙여 HVS 가 색변환) 은 **프로토콜 수준에서 가능하다.**

### 그래도 안 (III) 을 1순위로 올리지 않는 이유 (정직하게)

"modifier 를 받는다"와 "이득이 난다"는 다른 문제다. 확인되지 않은 것이 둘 남아 있다.

1. **박스 오버레이를 `wl_subsurface` 로 올리면 컴포지터가 direct scanout 을 포기할 수 있다.**
   wlroots 의 scanout 최적화는 대개 **단일 surface** 에만 적용된다. 영상 + 오버레이 두 layer 가 되면
   wlroots 가 1080p 를 GL 로 합성하게 되고, 그 비용(같은 V3D 에서 읽기+쓰기 약 3.6 ms 이상)이
   우리가 아낀 표시 변환 4~7 ms 를 상당 부분 되먹는다. **Pi 5 HVS 는 plane 이 56개라
   두 개를 하드웨어 plane 에 태울 여지가 있지만, 그 결정은 컴포지터가 한다.**
2. **프레임 수명이 더 길어진다.** 컴포지터가 `wl_buffer.release` 를 줄 때까지 디코더 버퍼를 잡고 있고,
   그 시점은 우리 present 펜스보다 늦을 수 있다. 계획서 §5.3 의 13프레임 제약이 더 빡빡해진다.

또한 안 (III) 은 Vulkan 스왑체인을 영상 layer 에 대해 우회하므로, P4/P5 로 검증하는 경로와 다른
구조가 된다. **따라서 안 (I)(present 시점 변환 + 스왑체인 직접 쓰기) 을 1순위로 유지하고,
안 (III) 은 "프로토콜 가능 확인됨, 이득 미검증" 상태로 단계 2 후보에 둔다.**
P8 이 표시 변환을 7 ms 넘게 재면 그때 안 (III) 의 실측(오버레이까지 포함한 wall-clock FPS)을 먼저 한다.

### 부수 확인: `wp_viewporter` 가 있다

`wp_viewport_set_source`/`set_destination` 으로 **확대를 HVS 가 한다.** 스왑체인의
`supportedPresentScaling = None` 제약을 우회하는 수단이고, P8 이 풀해상도 변환을 너무 비싸게
재면 **1280x720 으로 렌더하고 HVS 로 1080p 로 확대**하는 폴백이 가능하다.
이때도 박스 좌표 계산은 원본 1920x1080 기준으로 유지되므로 사용자 요구(원본 해상도 기준 박스)는 지켜진다.
