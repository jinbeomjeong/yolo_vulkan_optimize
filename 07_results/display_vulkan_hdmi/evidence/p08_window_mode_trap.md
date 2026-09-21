# P8 추가 발견: 창 모드에서 표시 변환이 2.6배 비싸진다 (조정자 독립 검증, 2026-09-20)

## 증상

`yvo_display_probe --probe convert` 를 **옵션 없이**(= 창 모드) 돌리면 온스크린 셰이더 시간이
오프스크린의 2.6배가 된다.

| 실행 | 스왑체인 extent | 경로 | on-screen shader median |
|---|---|---|---:|
| 오프스크린 300프레임 | – | 1:1 | **4.229 ms** |
| `--probe convert` (창 모드, 기본) | **1920x1044** | `kUniformScale` bilinear | **10.923 ms** |
| `--probe convert --display-fullscreen` | 1920x1080 | 1:1 | **4.224 ms** |

## 원인

labwc 가 창 모드에서 높이를 **1044** 로 준다(1080 − 36, 패널/장식 몫). `currentExtent = 0xFFFFFFFF`,
`supportedPresentScaling = None` 이라 **스왑체인 이미지 크기 = 창 크기**여야 하고,
1044 != 1080 이므로 `display_convert.comp` 가 spec constant `kUniformScale` 경로로 넘어간다.
그 경로는 4-tap bilinear + 열 LUT 를 쓰므로 픽셀당 비용이 1:1 의 약 2.6배다
(1:1 2.05 ns/px vs scale 5.31 ns/px — 표시 트랙 2단계 보고의 실측).

**조용히 느려지는 것이 문제다.** 에러도 경고도 없고, 로그의 `created 1920x1044` 한 줄만 단서다.

## 해법: `wp_viewporter` (이미 bind 되어 있다)

해상도를 낮추는 것은 **대책이 아니다** — 같은 이유로 1600x900(7.64 ms)과 1280x720(5.27 ms)이
1920x1080 1:1(4.23 ms)보다 느리다. 스케일링 경로 자체가 비싸다.

대신:
1. **항상 1920x1080 1:1 로 렌더한다**(4.23 ms, CPU 기준과 비트 일치).
2. `wp_viewport_set_destination(win_w, win_h)` 로 **축소를 Pi 5 HVS 하드웨어에 맡긴다.**
   `wp_viewporter` 는 labwc 가 광고하고(`version=1`) `WaylandWindow::create_viewport()` 로 이미 접근 가능하다.
   > 주(2026-09-21): 2026-09-21 리팩터에서 미사용 viewporter 바인딩(`create_viewport()` 등)을 삭제했다. 구현할 때 다시 붙인다([`08_learn/07_video_display.md:608`](../../../08_learn/07_video_display.md)).
3. 셰이더의 `kUniformScale` 경로는 **viewporter 가 없는 컴포지터를 위한 폴백으로만** 남긴다.

이러면 창 크기와 무관하게 빠른 비트 일치 경로가 유지되고, 스케일링 비용이 GPU 에서 HVS 로 옮겨간다.

## 반영할 곳

- `[7] DisplayPipeline` 구현 시: 스왑체인 extent 를 창 크기가 아니라 **콘텐츠 크기(1920x1080)** 로 잡고
  viewport destination 을 창 크기로 설정한다. viewporter 가 없으면 기존 동작(창 크기 스왑체인 + uniform scale)으로 폴백.
- 측정 실행은 `--display-fullscreen` 을 쓴다(계획서 §5.3 이 이미 권고. direct scanout 으로 컴포지팅 비용도 0).
- `yvo_pipeline --display` 가 창 모드로 뜰 때 extent != 1920x1080 이면 **경고 한 줄을 찍어라.**
  조용히 2.6배 느려지는 것을 막는다.

## 부수 확인

- 전체화면 300프레임: `present_wait` 간격 median **16.660 ms** = 60 Hz vsync 정확히 맞음,
  `presented=300 display_dropped=0 NOT_READY 0`, `recreates=1`(초기 configure)
- P6 회귀 게이트 재확인: `yvo_preproc_bench` GPU median **2.120 / 2.121 / 2.125 ms**
  (골든 2.116~2.120, 게이트 [2.074, 2.162]) — 표시 코드 추가에 회귀 없음
- ctest **11/11**

---

# P5 픽셀 검사 — 조정자 독립 재확인 (2026-09-20)

`samples/p05_present_pattern.png`(grim 캡처)를 직접 검사했다.

```
shape (1080, 1920, 3)
RGB(255,0,128) 오차 0 픽셀 : 119,301   bbox x 761~1159 (399px), y 391~689 (299px)
상위 색: (32,32,32) 1,935,536 / (255,0,128) 119,301 /
         (255,0,0) 4,096 / (0,0,255) 4,032 / (0,255,0) 4,032 / (255,255,255) 3,969
```

**P5 판정: 통과.** 컴퓨트 셰이더가 스왑체인 storage image 에 직접 쓴 패턴이 화면에 도달했고
**색이 오차 0 으로 정확**하며 위치도 중앙이다. 배경 `(32,32,32)`, 꼭지점 마커 4색 모두 제자리다.

**다만 2단계 보고의 "중앙 블록 정확히 120,000 px"는 실제와 다르다.** 캡처는 **119,301 px = 399x299**다
(400x300 이 아니다). 꼭지점 마커도 4,096(64²) / 4,032(64x63) / 4,032 / 3,969(63²)로 같은 차이를 보인다.
→ 셰이더의 블록 경계가 **포함/배제 off-by-one** 이다(예: `x > cx-200 && x < cx+200` 이면 399px).

영향: **없다.** `display_probe.comp` 는 P5 전용 테스트 패턴이고 파이프라인에 쓰이지 않는다.
다만 `[6] overlay_boxes.comp` 는 박스 테두리를 그리고 **`test_overlay` 가 CPU 기준과 비트 일치를
요구하므로 경계 규약(포함/배제)을 GPU 와 CPU 가 정확히 같게 정의해야 한다.**
이 off-by-one 이 그 단계에서 재발하면 비트 일치 테스트가 바로 잡아낸다 — 그것이 그 테스트의 목적이다.
