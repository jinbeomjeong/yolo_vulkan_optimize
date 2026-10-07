# KMS 직접 경로 대 Wayland(labwc) 경로 — HVS 스캔아웃 비교

측정일: 2026-09-22 · 커널 6.18.50+rpt-rpi-2712 · Raspberry Pi 5 · HDMI 1920x1080@60
입력: `04_data/videos/video_h265.mp4` (1920x1080 HEVC, 30fps, V4L2 stateless HW 디코드 → NV12 SAND128)
프로브: `tools/hvs_probe/hvs_kms_probe.cpp`, `tools/hvs_probe/hvs_wl_probe.cpp` (300프레임, 앞 30프레임 제외, 270프레임 측정)

## 1. 요약

- **두 경로 모두 디코더 NV12 버퍼를 HVS 평면에 zero-copy로 올린다(전체화면 기준).** V3D(GPU) 사용률은 0%, 늦은 프레임도 0이다.
- **표시 지연은 KMS 직접 경로가 약 1 vblank(≈16ms) 짧다.** 17ms 대 33ms다. Wayland는 컴포지터 커밋이 한 단계 더 붙는다.
- **CPU는 KMS가 약간 가볍다.** Wayland는 labwc 몫으로 0.7~0.9 ms/프레임이 더 든다(한 코어의 2~3%).
- **색 정확도가 가장 큰 차이다.** KMS는 앱이 BT.709 limited를 직접 지정하고, Writeback 캡처로 49.5 dB PSNR까지 확인했다. Wayland는 기본값이 BT.601 full이고, 현재 labwc/wlroots 조합은 COLOR_RANGE를 뒤집어서 우회가 필요하다(§4).
- **Wayland 창 모드는 HVS 평면을 쓰지 못하고 GPU로 합성한다.** V3D 부하가 5.7~6.1 ms/프레임이고, 오버레이까지 켜면 늦은 프레임 67개가 나온다.

## 2. 용어

| 층 | 정체 | 역할 |
|---|---|---|
| DRM/KMS | 리눅스 커널 API (`/dev/dri/card1`, atomic commit) | "이 버퍼를 이 평면·위치에 올려라"를 요청하는 통로. 픽셀 처리는 하지 않는다 |
| vc4 드라이버 | 커널 드라이버 | KMS 평면 상태를 HVS display list로 변환한다 |
| HVS | VideoCore 디스플레이 전용 하드웨어 | 평면 스케일링, YUV→RGB, 알파 블렌딩을 스캔아웃 중에 처리한다. GPU(V3D)와 별개다 |
| V3D | GPU | 컴포지터가 GL로 창을 합성할 때 쓴다 |

"KMS 직접"과 "Wayland"는 모두 결국 KMS → HVS로 간다. 차이는 영상 버퍼가 HVS 평면에 바로 올라가는지, 그 전에 GPU 합성을 거치는지와 누가 평면 속성(색 등)을 정하는지다.

## 3. 성능 비교 (전체화면)

| 조건 | 경로 | 변형 | 늦은 프레임 | 간격 중앙값/p99 (ms) | 표시 지연 중앙값 (ms) | 오버레이 그리기 (ms) | 앱 CPU (ms/f) | labwc CPU (ms/f) | V3D |
|---|---|---|---|---|---|---|---|---|---|
| 1:1 영상만 | KMS | `kms_1to1_video_only` | 0 | 33.333 / 33.336 | **17.30** | – | 0.32 | – | 0% |
| | Wayland | `wl_fs_video_fullhack` | 0 | 33.333 / 33.983 | 33.55 | – | 0.38 | 0.78 | 0% |
| 1:1 + 오버레이 | KMS | `kms_1to1_overlay` | 0 | 33.333 / 33.337 | **17.21** | 0.080 | 0.41 | – | 0% |
| | Wayland | `wl_fs_overlay_dumb` | 0 | 33.333 / 33.339 | 33.46 | 0.129 | 0.56 | 0.74 | 0% |
| 720 스케일 영상만 | KMS | `kms_720_video_only` | 0 | 33.333 / 33.336 | **17.30** | – | 0.32 | – | 0% |
| | Wayland | `wl_fs_vp720_video` | 0 | 33.334 / 33.546 | 33.58 | – | 0.39 | 0.78 | 0% |
| 720 + 오버레이 | KMS | `kms_720_overlay` | 0 | 33.333 / 33.335 | **17.25** | 0.041 | 0.36 | – | 0% |
| | Wayland | `wl_fs_vp720_overlay` | 0 | 33.334 / 33.547 | 33.13 | 0.063 | 0.74 | 0.89 | 0% |

측정 기준:
- 표시 지연은 KMS에서 atomic commit부터 page-flip 이벤트까지, Wayland에서 `wl_surface.commit`부터 `wp_presentation` feedback까지다. 기준점이 달라서 정확한 차이보다 "약 1프레임 차이"로 읽어야 한다.
- KMS 쪽 FPS 요약값(29.89)은 벽시계 구간 차이 때문이다. flip 간격은 정확히 33.333ms(60Hz에서 2 vblank)다.
- 두 경로 모두 flip 간격 편차는 1ms 미만이다. 다만 p99는 KMS가 33.34ms로 고정이고, Wayland는 33.5~34.0ms로 조금 흔들린다.

평면 배치(debugfs `dri/1/state`, f150 기준):
- KMS: 평면 83(Primary) NV12 SAND128(`mod=0x0700000000000004`), 오버레이는 평면 107 AB24 zpos=1. 720 변형은 `crtc-pos=1280x720+320+180`으로 HVS가 하드웨어 스케일링한다.
- Wayland 전체화면: `alloc= labwc`로 같은 평면 83에 NV12가 올라가고 zero-copy는 298/299다. 커서 평면(64x64)이 하나 더 있다.
- `hvs_underrun`은 모든 변형에서 0이다.

## 4. 색 (평면 COLOR_ENCODING / COLOR_RANGE)

영상은 BT.709 limited range다.

| 경로 | 변형 | 요청 | 커널 평면 상태 | 판정 |
|---|---|---|---|---|
| KMS | 전 변형 | BT.709 / limited (앱이 직접 설정) | BT.709 / limited | 정상 |
| Wayland | `wl_fs_video_default` | color-representation 미사용 | **BT.601 / full** | 틀림 |
| Wayland | `wl_fs_video_lim` | BT.709 / limited | BT.709 / **full** | range 틀림 |
| Wayland | `wl_fs_video_fullhack`, `wl_fs_vp720_*`, `wl_fs_overlay_*` | BT.709 / **full** (우회) | BT.709 / limited | 정상 |

- labwc 0.20.1 + wlroots 0.20.2+rpt3 조합은 color-representation의 range를 뒤집어서 커널에 넘긴다. 설치된 wlroots `backend/drm/atomic.c`의 변환 함수 자체는 정상 매핑이다(`evidence/wlroots_color_range_excerpt.txt`). 따라서 뒤집힘은 그 앞단에서 일어나는 것으로 보이며, 원인 위치는 확인하지 않았다.
- `fullhack` 우회는 컴포지터 버그에 기대는 방식이다. labwc/wlroots가 업데이트되면 조용히 틀어질 수 있으므로, Wayland 경로를 쓰면 시작할 때 debugfs state로 range를 검증해야 한다.
- KMS 직접 경로는 Writeback 커넥터로 HVS 실제 출력을 캡처해 CPU 기준 변환(BT.709 limited)과 비교했다(`evidence/kms/psnr_writeback.txt`).
  - PSNR 49.53 dB, 채널별 평균 오차 R +0.57 / G −0.61 / B +0.84 (1 LSB 미만)
  - max|d| 25는 색 경계에서만 나타나는 것으로 보인다. 크로마 업샘플링 방식 차이로 추정하며 확인하지 않았다.

### 4.1 재판정 — labwc 0.20.2 / wlroots 0.20.2+rpt4 (2026-10-07, 추가 기록)

2026-10-07 `apt upgrade`로 `labwc` 0.20.1-1+rpt1 → 0.20.2-1~bpo13+1, `libwlroots-0.20` 0.20.2-1+rpt3 → +rpt4가 설치됐다. rpt4 changelog는 *"Add patch to advertise color-representation formats from gles2"* 한 줄이라, 위 §4의 반전 전제가 깨졌을 수 있어 재판정했다. **결론: 반전은 그대로다. `fullhack` 기본값은 여전히 옳고 코드는 바꾸지 않았다.**

| 요청 (`--display-color-range`) | 보낸 값 | 커널 평면 실값 | `plane_color_ok` |
|---|---|---|---|
| `auto` (기본값) | BT.709 / full | BT.709 / **limited** | 1 |
| `limited` | BT.709 / limited | BT.709 / **full** | 0 |
| `full` | BT.709 / full | BT.709 / **limited** | 1 |

- 판정 근거는 위와 같이 `/sys/kernel/debug/dri/1/state`의 평면 실값이다. `yvo_pipeline`(300프레임 × 3회, `wayland-planes`)의 자체 판정과 1초 간격 외부 덤프가 일치한다.
- **달라진 점 하나**: 컴포지터가 알려 주는 `wp_color_representation_manager_v1` 전역(색 매니저)이 **2개에서 1개로** 줄었다. 앱은 알려진 전역을 전부 바인드하고 BT.709를 지원하는 하나를 고르므로 동작에는 영향이 없다. 지원 목록에 오른 `(coefficients, range)` 쌍은 identity/full, bt709·bt601·bt2020 각각 full+limited로 7개이며, BT.709 full이 목록에 있으므로 "요청 쌍이 지원 목록에 없음 → 요청 생략" 고장 모드는 발생하지 않는다.
- 원시 기록: [`../verification/color_range_rpt4/`](../verification/color_range_rpt4/README.md). 성능 수치는 1회 실행이라 인용 대상이 아니다.
- §4의 권고("Wayland 경로를 쓰면 시작할 때 debugfs state로 range를 검증해야 한다")는 그대로 유효하다. 이번 업그레이드는 반전을 고치지 않았을 뿐이고, 다음 업그레이드에서 고쳐질 수 있다.


## 5. Wayland 창 모드 (참고)

| 변형 | 늦은 프레임 | 간격 p99 (ms) | V3D render (ms/f) | 앱 CPU | labwc CPU | 평면 |
|---|---|---|---|---|---|---|
| `wl_win720_video` | 0 | 33.994 | 5.69 | 0.55 | 1.30 | XR24 합성 버퍼 (zero-copy 0) |
| `wl_win720_overlay` | **67** | **50.002** | 6.13 | 0.90 | 1.59 | XR24 합성 버퍼 (zero-copy 0) |

창 모드에서는 labwc가 영상을 GPU로 합성해 RGB 한 장(XR24)으로 만든 뒤 스캔아웃한다. HVS 평면 직접 표시의 이점이 사라지고, 추론 파이프라인과 V3D를 나눠 쓰게 된다.

## 6. 결론과 권고

1. **전용 장비(키오스크, 한 화면 전체 표시)라면 KMS 직접 경로를 권장한다.** 지연이 1프레임 짧고, 색을 앱이 보장하고, 컴포지터 버전에 의존하지 않는다. 대신 DRM master가 필요하므로 lightdm/labwc와 함께 쓸 수 없다.
2. **데스크톱과 함께 써야 하면 Wayland 전체화면을 쓴다.** 처리량은 KMS와 같다. 단, 색은 `fullhack` 우회를 쓰고 시작 시 debugfs state로 검증해야 한다.
3. **Wayland 창 모드는 피한다.** GPU 합성이 생기고, 오버레이까지 켜면 프레임을 놓친다.

## 7. 한계

- 화면을 눈으로 확인한 기록(`evidence/kms/visual_notes.txt`)이 없다. 색 판정은 Writeback PSNR과 평면 속성에 근거한다.
- `kms_wb`(Writeback 캡처 실행)는 캡처 커밋 탓에 늦은 프레임 1개, CPU 2.9 ms/프레임이 나왔다. 성능 비교에서는 제외했다.
- 측정은 영상 표시만 했다. 추론(Hailo)을 같이 돌린 부하 조건은 포함하지 않는다.
- 각 변형은 1회씩만 실행했다.

## 8. 증거 파일

- KMS: `evidence/kms/` — `run_kms_probe.log`, `<variant>.log`, `<variant>_frames.csv`, `<variant>_state_f*.txt`, `psnr_writeback.txt`, `state_before.txt`/`state_after.txt`
- Wayland: `evidence/wayland/` — `wl_*.log`, `wl_*_frames.csv`, `wl_*_state_f*.txt`
- 색 관련: `evidence/wlroots_color_range_excerpt.txt`, `evidence/color/`
- 재현: `tools/hvs_probe/run_kms_probe.sh` (lightdm 정지 필요), `tools/hvs_probe/run_wl_probe.sh`
