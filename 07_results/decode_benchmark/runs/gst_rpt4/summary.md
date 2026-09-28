# 디코드 벤치마크 재측정 — gst-plugins-bad rpt4 (H.265 HW 조합만)

## 측정 조건
- 측정 일시: 2026-09-29 00:34~00:40 KST. 부트로더 EEPROM은 2025-12-08 버전(업데이트 적용 전)이다.
- 재측정 계기: 2026-09-29 `apt upgrade`로 `gstreamer1.0-plugins-bad`가 `1.26.2-3+rpt3+deb13u3`에서 `1.26.2-3+rpt4+deb13u3`으로 올라갔다.
  - 부모 측정([`../../summary.md`](../../summary.md), 2026-09-20)은 rpt3에서 했다. rpt3은 09-19 22:14에 설치됐다.
- rpt4 changelog의 v4l2codecs 항목:
  - copy_frames 로직을 decide_allocation으로 이동
  - has_videometa 멤버 제거
  - h265dec: 좌상단 오프셋이 0이면 크롭하지 않음
  - h265dec: DMABuf 출력에서 크롭이 필요하면 실패
- 환경:
  - Raspberry Pi CM5(BCM2712), governor ondemand, kernel 6.18.50, GStreamer 1.26.2
  - PCIe Gen3 x1(디코드에는 무관), `throttled=0x0`
  - 자세한 값은 [`env_bench.txt`](env_bench.txt)
- 명령: `ONLY='^h265_C' RUNS=3 OUT_DIR=… bash 06_tools/bench/decode_bench.sh`. 조합과 측정 방법은 부모 측정과 같다(wait4 rusage, 조합마다 3회, 중앙값).
- 입력: `04_data/videos/video_h265.mp4`(HEVC Main 1080p30, 1816프레임). 모든 실행에서 1816프레임을 끝까지 디코드했다.
- 대조군: `h265_Cref_ffmpeg_drm_S1`(ffmpeg CLI `-hwaccel drm -f null`). ffmpeg 패키지는 바뀌지 않았다.

## 결과 (중앙값, 3회)

CPU ms/frame = 중앙값 실행의 (user+sys) / 1816.

| 조합 | rpt3 FPS (부모) | **rpt4 FPS** | FPS 범위 | rpt4 CPU ms/frame | rpt3 CPU ms/frame (부모) |
|---|---|---|---|---|---|
| `h265_C_hw_S1` (`fakesink`, NV12_128C8 시스템 메모리) | 32.66 | **248.90** | 244.74~255.07 | 3.18 | 30.6 |
| `h265_C_hw_S1_fvsink` (`fakevideosink`, DMABuf) | 32.64 | **381.53** | 380.85~381.56 | 0.24 | 30.6 |
| `h265_C_hw_S2_nt1` (+ `videoconvertscale` 1스레드) | 26.09 | **40.58** | 40.17~40.69 | 24.6 | 38.3 |
| `h265_C_hw_S2_nt4` (+ `videoconvertscale` 4스레드) | 30.34 | **122.86** | 122.55~123.60 | 30.3 | 39.4 |
| `h265_Cref_ffmpeg_drm_S1` (대조군) | 332.79 | **334.44** | 333.99~335.32 | 2.0 | 2.0 |

출력 caps는 rpt3과 같다. `fakesink` 쪽은 `NV12_128C8`, `fakevideosink` 쪽은 `DMA_DRM NV12:0x0700000000000004`(SAND128)다.

## 결론
- **대조군은 그대로다**(332.8 → 334.4 FPS, +0.5%). 따라서 C 조합의 변화는 플러그인 업데이트 때문이다.
- **rpt3에서는 DMABuf 출력(fvsink)에서도 코어 하나가 100% 찼다.** caps가 DMABuf로 협상돼도 디코더 안에서 CPU 복사가 일어났다는 뜻이다.
  - 부모 측정과 [`../../notes/ffmpeg_to_gstreamer.md`](../../notes/ffmpeg_to_gstreamer.md)가 짚은 ORC 복사가 이것이다.
  - 1080p HEVC는 코딩 높이 1088에서 아래쪽 8줄만 잘라내므로, 좌상단 오프셋이 0인 경우에 해당한다. rpt4의 크롭 회피 변경과 맞아떨어진다.
  - 이 증상은 이미 알려진 upstream 버그이고, Raspberry Pi 패키지에 수정이 늦게 들어왔을 뿐이다. 근거는 아래 연혁이다. 원인을 코드 수준에서 직접 확인하지는 않았다.

  | 일자 | 사건 | 출처 |
  |---|---|---|
  | 2025-05-29 | GStreamer 1.26.2 릴리즈. Raspberry Pi OS trixie가 이 버전에 머문다 | [1.26 릴리즈 노트](https://gstreamer.freedesktop.org/releases/1.26/) |
  | 2025-07-16 | **1.26.4 릴리즈.** "v4l2codecs: h265dec: Fix zero-copy of cropped window located at position 0,0"([MR !9315](https://gitlab.freedesktop.org/gstreamer/gstreamer/-/merge_requests/9315))가 처음 들어간다 | 같은 릴리즈 노트 |
  | 2026-06-02 | [RPi-Distro/repo#406](https://github.com/RPi-Distro/repo/issues/406) 보고: "gstreamer1.0-plugins-bad 1.26.2 misses the v4l2codecs HEVC conformance-crop fix: 1080p HW decode falls back to a CPU copy" | GitHub 이슈 |
  | 2026-09-20~22 | RPi 엔지니어 계정(6by9, XECDesign)이 이슈에 댓글 | 같은 이슈 |
  | 2026-09-25 07:38 +0100 | rpt4 changelog 서명(Serge Schneider). v4l2codecs 항목 네 개 | 로컬 `/usr/share/doc/gstreamer1.0-plugins-bad/changelog.Debian.gz` |
  | 2026-09-25 08:36 (서버 시각) | rpt4가 archive.raspberrypi.com에 게시된다 | [pool 목록](http://archive.raspberrypi.com/debian/pool/main/g/gst-plugins-bad1.0/) |
  | 2026-09-29 00:17 KST | 이 장비에 rpt4 설치 | `/var/log/apt/history.log` |

  - 이슈 #406의 보고 환경은 이 측정과 같다. Pi 5/CM5, kernel 6.18.x, `v4l2slh265dec`(rpi-hevc-dec), 패키지는 rpt3+deb13u1이다.
  - 보고된 증상도 부모 측정(rpt3+deb13u3)과 같다. 1080p에서 zero-copy가 깨지고, 매 프레임을 시스템 메모리로 복사·디타일하며, CPU 코어 하나가 찬다. 이슈에 따르면 720p·1440p·2160p는 zero-copy가 유지된다.
  - 이슈 본문은 수정 버전을 "1.26.11"로 적었다. 하지만 릴리즈 노트 원문에서 이 항목이 처음 나오는 버전은 1.26.4다.
  - 서버 시각 08:36의 시간대는 확인하지 않았다(changelog의 +0100 기준이면 영국 서머타임).
  - 이 문서를 쓸 때(2026-09-29) 이슈가 닫혔는지는 확인하지 못했다.
- **rpt4에서는 DMABuf 경로가 381.5 FPS / CPU 0.24 ms/f다.** 앱의 libavcodec DRM_PRIME 경로(398.5 FPS / 0.23~0.28 ms/f, [`../../../pipeline_ffmpeg_vulkan_hailo/REPORT.md`](../../../pipeline_ffmpeg_vulkan_hailo/REPORT.md) §4.1)와 CPU 비용이 같은 수준이다. 측정 도구가 달라 FPS 4% 차이는 판정하지 않는다.
- 시스템 메모리로 받는 경로(S1 fakesink, 3.18 ms/f)와 CPU 변환 경로(S2, 24.6~30.3 ms/f)에는 여전히 CPU 비용이 남는다. 이 비용은 디코더가 아니라 디코드 뒤 처리에서 생긴다.

## 주의·남은 일
- 부모 측정의 GStreamer C 조합 결론("HW를 쓰고도 32.7 FPS, CPU 30.6 ms/f")은 **rpt3에서만 성립한다.** 부모 결과 파일은 rpt3 기록으로 유효하므로 고치지 않았다.
  - 이 결론을 인용하는 문서: `07_results/README.md` 색인, `08_learn/02_hardware.md`, `08_learn/07_video_display.md`, `../../../pipeline_ffmpeg_vulkan_hailo/REPORT.md`의 참고 행
  - 이 문서들의 갱신 여부는 별도로 정한다.
- E 조합(ffmpeg → pipe → GStreamer)과 H.264·SW 조합은 다시 재지 않았다. E는 v4l2codecs를 쓰지 않는다.
- 측정 중 SoC는 64~74 °C였다. 조합 사이마다 65 °C 이하로 식혔다.
