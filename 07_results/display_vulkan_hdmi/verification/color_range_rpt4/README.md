# wayland-planes 색 범위 재판정 — labwc 0.20.2 / wlroots 0.20.2+rpt4

2026-10-07. `wayland-planes` 기본값 `--display-color-range auto`가 기대는 **range 반전("fullhack")**이
컴포지터 업그레이드 뒤에도 유효한지 판정했다. 결론: **유효하다. 코드 변경 없음.**

계기는 2026-10-07 23:26 `apt upgrade`다 — `labwc` 0.20.1-1+rpt1 → 0.20.2-1~bpo13+1,
`libwlroots-0.20` 0.20.2-1+rpt3 → +rpt4(changelog 한 줄: *"Add patch to advertise
color-representation formats from gles2"*). [`../../notes/kms_vs_wayland.md`](../../notes/kms_vs_wayland.md) §4가
"컴포지터가 업데이트되면 조용히 틀어질 수 있다"고 경고해 둔 지점이다.

## 판정

| 요청 (`--display-color-range`) | 프로토콜로 보낸 값 | 커널 평면 실값 | `plane_color_ok` |
|---|---|---|---|
| `auto` (기본값) | BT.709 / **full** | BT.709 / **limited** | **1** |
| `limited` | BT.709 / limited | BT.709 / **full** | 0 |
| `full` | BT.709 / full | BT.709 / **limited** | 1 |

요청과 평면 실값이 양방향으로 어긋난다 → **반전이 그대로 살아 있다.** 기본값 `auto`는 여전히 옳다.
`color_ok`의 판정 기준은 `01_src/display/drm_state.hpp`의
`video_on_plane && encoding_bt709 && range_limited`이고, 근거는 `/sys/kernel/debug/dri/1/state`다.

## 달라진 것: 컴포지터가 알려 주는 색 매니저 수 2 → 1

rpt3 시절 labwc는 `wp_color_representation_manager_v1` 전역을 **2개** 알려 줬다
(`06_tools/probes/hvs_probe/hvs_wl_probe.cpp`의 측정 주석). rpt4에서는 **1개**다 —
세 회차 모두 `[managers advertised 1]`. 앱은 알려진 전역을 전부 바인드하고 BT.709를 지원하는 하나를
고르므로(`01_src/display/wayland_window.cpp` `select_color_manager()`) 동작에 영향은 없다.

지원 목록에 오른 `(coefficients, range)` 쌍은 7개다(`wayland_debug_color_auto.log`, 숫자는 프로토콜 enum 생값):

| 쌍 | 뜻 |
|---|---|
| (1, 1) | identity / full |
| (2, 2), (2, 1) | **bt709 / limited, bt709 / full** |
| (4, 2), (4, 1) | bt601 / limited, bt601 / full |
| (6, 2), (6, 1) | bt2020 / limited, bt2020 / full |

BT.709가 full·limited 둘 다 지원 목록에 있으므로 "요청 쌍이 목록에 없음 → 요청 생략"
(`01_src/display/wayland_plane_scanout.cpp`의 보호 분기) 고장 모드는 발생하지 않는다.
실제로 보낸 요청은 `set_coefficients_and_range(2, 1)` = bt709 / full이었다.

## 파일

| 파일 | 내용 |
|---|---|
| `env.txt` | 커널·패키지 버전·실행 컴포지터·git HEAD·바이너리 md5 |
| `stdout_{auto,limited,full}.log` | 3회 실행 전체 출력 (`[display] … color=` 줄, `[scanout] debugfs` 줄, `RESULT_CSV`) |
| `debugfs_plane_{auto,limited,full}.txt` | 실행 중 1초 간격으로 뜬 `dri/1/state` — 앱 자체 판정과 교차 확인용 |
| `debugfs_state_idle.txt` | 실행 전 상태. 데스크톱(labwc)이 `plane[83]`에 fb를 올려 둔 상태로 BT.601 / full이고, 나머지 평면은 fb 없음 |
| `wayland_debug_color_auto.log` | `WAYLAND_DEBUG=1`의 color-representation 트래픽만 추린 것 |
| `hailort.log` | 경고·에러 0줄 |

실행 바이너리는 주석 갱신 **전** 빌드(2026-09-29, md5는 `env.txt`)다. 그래서 `stdout_auto.log`의 `color=` 꼬리 문구가
`labwc 0.20.1 inverts it`으로 찍혀 있다. 지금 소스로 다시 빌드하면 `labwc 0.20.1~0.20.2 inverts it`으로 나온다 — 문구만 다르고 동작은 같다.

## 재현

```sh
cd 07_results/display_vulkan_hdmi/verification/color_range_rpt4   # hailort.log 를 여기에 남기려고
XDG_RUNTIME_DIR=/run/user/1000 WAYLAND_DISPLAY=wayland-0 \
  ../../../../build/bin/yvo_pipeline \
    --display on --display-backend wayland-planes --display-fullscreen \
    --mode dmabuf --layout nhcw --batch 2 --slots 4 --frames 300 \
    --display-color-range auto --csv-label cr_auto
```

함정 두 개:
- **`--frames`는 90 이상**이어야 한다. start 스냅샷이 프레젠트 60장 뒤에 잡히기 때문이다
  (`01_src/display/plane_display.hpp` `kSnapshotAfterPresented`).
- **`--quiet`를 주면 안 된다.** `[display] … color=` 줄이 사라져 컴포지터가 알려 준 지원 정보를 못 본다.

labwc 세션 안에서 돌기 때문에 VT 전환이나 lightdm 정지는 필요 없다. 실행 중 VT를 바꾸지 않는다.

## 성능은 보지 않았다

3회 모두 300프레임 1회 실행이라 **FPS·지연 수치는 인용하지 말 것**(`measure` 스킬의 냉각·반복 없음).
검출 수가 세 회차 모두 911개로 같은 것만 확인했다 — 색 범위는 표시 경로만 건드리고 추론에 닿지 않는다.
