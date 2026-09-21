# HDMI 실시간 가시화 — Vulkan 표시 경로 프로브 결과

- 측정일: 2026-09-20
- 환경: Raspberry Pi CM5 (BCM2712, Cortex-A76 ×4), **V3D 7.1.10.2 / V3DV Mesa 26.2.2, Vulkan 1.3.354**,
  labwc(Wayland) on **HDMI-A-1 1920×1080@60**, kernel 6.18.50, g++ 14.2 Release
- 대상 영상: `04_data/videos/video_h265.mp4` (HEVC Main 1920×1080@30)
- 목적: 계획서(`/home/pi/.claude/plans/jaunty-dazzling-quasar.md`) §5의 표시 경로 설계에서
  **본구현 전에 검증해야 할 가정**을 닫는다. 사용자가 **원본 1920×1080 해상도 기준 박스 표현을 1순위**로 확정했다.
- 실행: `XDG_RUNTIME_DIR=/run/user/1000 WAYLAND_DISPLAY=wayland-0 build/bin/yvo_display_probe --probe ...`

## 결과

| 프로브 | 기준 | 실측 | 판정 |
|---|---|---|---|
| **P0.5** labwc 가 Wayland 클라이언트를 받는가 | 화면 도달 | SMPTE 6색 전부 `grim` 캡처에서 검출 | **통과** |
| **P4** surface·포맷·usage | 5기준 | presentation support `VK_TRUE` / `R8G8B8A8_UNORM`+`SRGB_NONLINEAR` 존재 / usage `0x8009f`에 **STORAGE** / MAILBOX·FIFO·IMMEDIATE / `ImageFormatProperties2(STORAGE)` `VK_SUCCESS` | **통과** |
| **P5** present 가 실제로 되는가 | 간격 16.7±2 ms, 검증 0, 화면 일치 | `present_wait` 간격 **median 16.661 ms** (p90 16.697, n=118), 120 present 타임아웃 0, 검증 레이어 **errors 0 / warnings 0**(sync 포함, 4구성), `grim` 픽셀 **오차 0** | **통과** |
| **P6** 기존 경로 회귀 없음 | GPU median [2.074, 2.162] ms | **2.120 / 2.121 / 2.125 ms** (골든 2.116~2.120), context diff **0** | **통과** |
| **P8** 원본 해상도가 예산에 드는가 | 1080p median ≤ 7.0 ms, CPU 기준과 비트 일치 | **4.229 ms** (p90 4.329), **max 오차 0 / 2,073,600 px**, alpha≠255 0개 | **통과** |
| **P10** SAND 직접 스캔아웃(선택) | 프로토콜 가능 여부 | labwc 가 `NV12 modifier 0x0700000000000004` **광고함**. 이득은 미검증 | 가능 확인 |
| **P7** 오버레이 박스·라벨 | 비트 일치, GPU ms, 검증 0 | 테두리 선두께 1/2/3 **max 오차 0**(변경 px 수까지 CPU와 동일), 글리프 ASCII 32~126 **max 오차 0**, 검출 5개 **0.151** / 40개 **0.555** ms, errors 0 | **통과** |

부수 실측: 모드별 `minImageCount` **MAILBOX 4 / FIFO 3 / IMMEDIATE 3**,
`currentExtent = 0xFFFFFFFF`, `supportedPresentScaling = None`,
blit 폴백 비용 **약 1.23 ms**(계획서 추정 3.6 ms보다 쌈),
labwc 광고 전역: `wp_viewporter=1`, `wp_presentation=1`, `wl_subcompositor=1`, `zwp_linux_dmabuf_v1` version 4.

## 결론

1. **사용자가 1순위로 정한 원본 1920×1080 해상도 박스 표현이 성립한다.**
   1:1 변환이 **4.23 ms**로 b4 주기 22.5 ms의 19%, 실시간 33.3 ms의 13%다. 계획서 추정 4~7 ms의 하단이다.
2. **컴퓨트 셰이더가 스왑체인 이미지에 직접 쓸 수 있다**(STORAGE usage 지원). blit 단계가 불필요하고,
   그 결과가 HDMI 화면까지 **한 비트도 변하지 않고** 도달한다(`R8G8B8A8_UNORM` + 값 통과).
3. **워크그룹 형태가 1.83배를 갈랐다.** 기본 후보 16×16(64×16 px 타일) 7.74 ms → **3×85(12×85 px) 4.23 ms**.
   같은 invocation 수(255~256) 안에서 **세로로 긴 타일이 결정적으로 빠르다.**
   읽기(SAND 열은 가로 128 B)만 보면 불리해야 하므로 병목은 타일 배치 RGBA 이미지 쓰기 쪽으로 보인다 —
   **이 해석은 미확정 가설이고 수치만이 근거다.**
4. **해상도를 낮추는 것은 비용 대책이 아니다(계획서와 반대 결론).**
   균일 스케일 경로가 4-tap bilinear를 쓰므로 픽셀당 2.04 ns → 5.3~5.7 ns로 뛴다
   (720p 5.72 / 900p 5.31 — `시간 ÷ 픽셀 수`):
   1920×1080 1:1 **4.23** ms < 1280×720 **5.27** ms < 1600×900 **7.64** ms.
5. **창 모드에서 조용히 2.6배 느려진다.** labwc 가 창 높이를 **1044**로 주면(1080−36)
   `kUniformScale` 경로로 넘어가 **10.92 ms**가 된다(전체화면 4.22 ms). 에러도 경고도 없다.
   → **`wp_viewporter`로 해결한다**: 스왑체인을 1920×1080 고정으로 잡고 축소를 HVS 하드웨어에 맡긴다.
6. **검증 레이어가 실제 버그 3개를 잡았다**: FIFO `minImageCount=3`의 VUID 위반,
   blit 폴백의 WRITE_AFTER_READ, 세마포어 소비 완료 전 `vkReleaseSwapchainImagesEXT` 호출.
   계획서가 "이 작업에서 가장 중요한 검증 단계"라고 한 판단이 맞았다.

7. **오버레이는 경계 규약을 먼저 글로 정해 비트 일치를 얻었다.** P5 에서 한 번 어긋났던 off-by-one
   (400x300 대신 399x299)이 재발하지 않았다. `01_src/ref/overlay_reference.hpp` 상단의 R0~R8 이 그 규약이고,
   GPU(`overlay_boxes.comp`)와 CPU 기준을 **그 글에 맞춰 각각** 썼다. R1(round half up + 닫힌 범위)이
   "399냐 400이냐"를 명시적으로 닫는다.
8. **오버레이를 `DisplayConvert` 와 같은 커맨드 버퍼에 이어 붙여도 변환이 느려지지 않는다.**
   변환 4.229 ms(기준선) → 4.234(검출 5) / 4.223(검출 80). 커맨드 버퍼 총합은 **4.46 / 5.28 ms** 로
   b4 주기 22.5 ms 의 20~23% 다. `presented=300 / dropped=0`.
   두 디스패치 사이 `SHADER_STORAGE_WRITE → SHADER_STORAGE_WRITE` 배리어는 **필요하다**(같은 픽셀 WAW).
9. **흩뿌린 쓰기의 처리량 바닥은 5.0~5.7 ns/px 다**(테두리만 그리는 가장 단순한 경우도 5.02).
   같은 GPU 의 순차 쓰기 최적치가 2.05 ns/px(`display_convert.comp`)이므로 그 차이가 오버레이 비용의 본질이다.
   `local_size_x` 는 계획서의 64 대신 **128**을 쓴다(스윕: 16→7.27 / 32→3.96 / 64→2.09 / **128→1.03** / 256→1.03 ms).

## 주의와 남은 일

- **P5 테스트 패턴이 400×300 대신 399×299를 그렸다**(경계 포함/배제 off-by-one). 판정에는 영향이 없지만
  (색 오차 0, 위치 정확) **`[6] overlay_boxes.comp`는 경계 규약을 CPU 기준과 정확히 같게 정의해야 한다** —
  `test_overlay`의 비트 일치 테스트가 그것을 잡는 것이 존재 이유다. 자세한 내용은 `evidence/p08_window_mode_trap.md`.
- **GPU 단독 벤치는 NPU 유휴 상태에서만 재야 한다.** NPU 동시 실행 시 2.14 → 2.20 ms(+3%), max 2.59 → 4.31 ms.
  `.claude/skills/measure/SKILL.md`에 이 항목이 없다. P6 게이트가 이탈하면 코드보다 경합·온도를 먼저 의심한다.
- **균일 스케일 경로는 비트 일치를 검증하지 않았다**(보간이 있어 원리적으로 어렵다). 1:1만 검증했다.
- **3×85 최적값은 해상도 의존적일 수 있다.** 1920×1080과 1280×720 양쪽에서 최적이었으나 그 외는 미측정.
  spec constant 이므로 `--probe convert --sweep --display-size WxH`로 다시 잰다.
- **MAILBOX/IMMEDIATE는 플래그 존재만 확인**했고 실제 present 는 FIFO 만 돌렸다.
- **glass-to-glass 지연은 아직 재지 않았다.** 프로브가 찍는 49 ms대 값은 프로브의 인플라이트 깊이 산물이다.
  실제 값은 `[8] main.cpp` 통합 후 P9(b4/s4)가 `present_wait`로 잰다.
- **`DmaBufExporter`의 "정렬 미스 시 재할당" 경로는 실행되지 않았다**(매핑이 항상 16 KB 정렬로 나옴) → 미테스트 경로.
- **P7 ③ 기준은 계획서 자체 모순이었다(내 실수, 부록 A.14).** §5.4 는 "검출 **40개** 최악에 0.7 ms"를
  근거로 썼는데 P7 표에는 "80개"라고 적었다. 80개는 DetBuf 상한이고 실제 평균은 5.21개다.
  올바른 기준으로 **0.555 ms < 0.7 ms 통과**이고, 80개(1.023 ms)조차 예산의 23%다.
  80개에서 0.7 ms 를 맞추려면 3.87 ns/px 가 필요한데 그건 순차 쓰기(2.05)보다 빨라야 한다는 뜻이라 **달성 불가능한 기준**이었다.
- **겹치는 박스의 교차 픽셀은 순서 의존이다**(R8, 허용 한계). 결정성 테스트가 실제로 이 위반을 잡아
  테스트 입력을 비겹침으로 한정했다. 필요하면 호스트에서 score 오름차순 정렬로 "높은 점수가 위"를 보장할 수 있다.
- **`grim` 화면 캡처를 오버레이 단계에서는 못 했다**(측정 중 데스크톱 창이 전면에 있었다).
  present 통계(`presented=300, dropped=0, 간격 16.659 ms`)로 대신했고 화면 도달 확인은 [7]/[8]로 넘긴다.
- 안 (III)(SAND 직접 스캔아웃)은 프로토콜 가능만 확인됐다. 오버레이를 subsurface 로 올리면 wlroots 가
  direct scanout 을 포기해 이득이 상쇄될 수 있어 **미검증 상태로 단계 2 후보**에 둔다.

## 파일

- `evidence/p04_p05_p08_probe.md` — P4/P5/P8 + export·lean-images 실측 전문, 워크그룹 스윕 23조합
- `evidence/p05_p10_probe.md` — P0.5/P10(조정자 실행), labwc modifier 표 덤프
- `evidence/p08_window_mode_trap.md` — 창 모드 2.6배 저하와 `wp_viewporter` 해법, P5 픽셀 재검사
- `evidence/p07_overlay.md` — 오버레이 P7 실측 전문, 경계 규약 R0~R8, 워크그룹 스윕
- `dmabuf_modifier_dump.c` — labwc 가 광고하는 dma-buf format/modifier 덤프 도구.
  2026-09-21 폴더 재배치에서 **`../../06_tools/probes/dmabuf_modifier_dump.c` 로 옮겼다**
  (결과 폴더에는 소스를 두지 않는다)
- `verification/README.md` — 6개 게이트 표와 로그 색인
- `verification/` — context 회귀(골든/present-on/full), preproc 벤치 전후, 검증 레이어 로그, ASan 로그, ctest, PNG 재현
- `samples/p05_present_pattern.png` — `grim` 캡처(픽셀 검사 근거)
