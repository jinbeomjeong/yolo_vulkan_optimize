# hvs_review 증거

[`../../notes/kms_vs_wayland.md`](../../notes/kms_vs_wayland.md)(KMS 직접 대 Wayland 전체화면 HVS 스캔아웃 비교)의 원시 증거다.

2026-09-22에 git 밖 `yvo_pipeline_port/docs/hvs_review/evidence/`에서 내용을 바꾸지 않고 옮겨 왔다.

## 경로 대응
메모 안의 경로는 원래 저장소 기준이다. 이 저장소에서는 다음 위치에 있다.

| 메모의 경로 | 이 저장소 |
|---|---|
| `evidence/kms/`, `evidence/wayland/`, `evidence/color/`, `evidence/baseline/`, `evidence/*.txt` | 이 폴더(`07_results/display_vulkan_hdmi/evidence/hvs_review/`) 아래 같은 이름 |
| `evidence/verification/asan_*.log` | [`../../verification/hvs_review/`](../../verification/hvs_review/)(검증 게이트 로그는 `verification/`에 둔다, `STRUCTURE.md` §3.2) |
| `tools/hvs_probe/*` | [`06_tools/probes/hvs_probe/`](../../../../06_tools/probes/hvs_probe/)(빌드하지 않는 프로브) |

## 참고
메모 §6은 "KMS는 lightdm/labwc와 함께 쓸 수 없다(lightdm 정지 필요)"고 쓴다. 이후 파이프라인의 `--display-backend kms`는
lightdm을 멈추지 않고 돌렸다. 방법은 `loginctl activate`로 비데스크톱 VT를 활성화해 DRM master를 비우는 것이다.
자세한 내용은 [`08_learn/07_video_display.md`](../../../../08_learn/07_video_display.md)의 HVS 평면 스캔아웃 절에 있다.
