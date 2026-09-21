# wayland-planes 커밋 간격 게이트 — batch 2 가 만드는 표시 cadence 흔들림

작성: 2026-09-22 · 대상: `yvo_pipeline --display on --display-backend wayland-planes` (HVS 두 평면 표시)
결정: **게이트를 유지한다**(사용자 결정, 2026-09-22).

> 이 메모의 수치는 개발 중 **각 1회** 실행한 디버깅 값이다(원시 로그는 git 밖 `11_logs/2026-09-22_wayland_planes/`).
> 정식 측정은 `measure` 스킬로 `07_results/`에 따로 기록한다. 여기서는 현상과 방향만 읽는다.

## 1. 현상

`--batch 2`에서는 NPU가 **두 프레임의 검출 결과를 한꺼번에** 돌려준다. 그래서 `--display-sync matched`(기본값)에서는
표시할 프레임도 두 장씩 동시에 준비된다. 화면은 60 Hz, 즉 16.7 ms마다 한 번 바뀐다(vblank).
준비되는 대로 바로 커밋하면 두 장이 연속한 vblank에 나가고, 다음 짝은 66.7 ms 뒤에 나간다.

```
게이트 없음:  [A][B]....[C][D]....      present 간격 16.7 / 66.7 ms 반복
게이트 있음:  [A]..[B]..[C]..[D]..      present 간격 33.3 ms (30 fps 원본 cadence)
```

30 fps 영상이 "빨리-멈춤-빨리"로 보인다(judder). 처리량(FPS)과 드롭 수에는 드러나지 않고
**present 간격 분포에서만** 보인다. 그래서 에러 없이 지나치기 쉽다.

## 2. 게이트

커밋과 커밋 사이 최소 간격을 **"소스 프레임 간격 − 8 ms"**로 둔다. 30 fps 기준 33.3 − 8 ≈ 25.3 ms다.
짝의 두 번째 프레임은 25 ms 이상 기다린다. 그래서 첫 프레임 다음 vblank가 아니라 그 다음 vblank(33.3 ms 뒤)에 나간다.
8 ms는 반 vblank 여유다. 이 여유가 있어서, 커밋이 조금 늦어도 한 vblank 전체를 놓치지 않는다.

- 코드: `01_src/display/plane_display.cpp`
  - 생성자의 `min_commit_interval_` 계산: 소스 간격 − 8 ms
  - 커밋 판단부의 `now - last_commit_ < min_commit_interval_` 검사
- 적용 범위: **`--display-pace source`(표시 on일 때 기본값)에서만** 걸린다. `asap`에서는 게이트가 없다.
- 종료(`drain`) 때는 남은 프레임을 평소 게이트대로 최대 200 ms 동안 내보내고, 그래도 남으면 게이트를 무시한다.

## 3. 대가 — 지연

짝의 두 번째 프레임은 한 vblank(16.7 ms)를 더 기다린다. 평균하면 프레임당 약 8 ms가 늘어난다.
같은 백엔드에서 게이트만 바꿔 1회씩 돌린 값(300프레임, 검증 레이어 on):

| 설정 | present 간격 median / p90 | glass-to-glass mean / median / p90 | presented / dropped |
|---|---|---|---|
| 게이트 없음 (`planes_val_300f_v2.log`) | 16.667 / 66.668 ms | 91.45 / 95.08 / 99.70 ms | 299 / 1 |
| 게이트 있음 (`planes_val_300f_v3.log`) | 33.333 / 50.001 ms | 101.69 / 99.71 / 116.61 ms | 299 / 1 |

게이트를 켜면 glass-to-glass median이 약 +4.6 ms, mean이 약 +10 ms 늘어난다. 전체(약 100 ms)에 견주면 작다.
반면 게이트가 없을 때의 16.7/66.7 ms 교대는 눈에 보인다. 그래서 게이트를 유지했다.

참고로 같은 조건의 기존 Vulkan 합성 경로(`--display-backend wayland --display-fullscreen`, 검증 레이어 off)는
present 간격 median 33.251 / p90 50.622 ms, glass-to-glass median 92.48 ms였다(`vulkan_fs_300f.log`).
wayland-planes(게이트 있음, 검증 레이어 off)는 median 98.25 ms다(`planes_noval_300f.log`).

## 4. 지연이 중요하면

`--display-sync latest --display-pace asap`을 쓴다. 게이트와 matched 대기가 모두 없어진다.
그 대신 검출은 "가장 최근 것"을 얹는다.

| 설정 | present 간격 median / p90 | glass-to-glass median | 비고 |
|---|---|---|---|
| latest + asap (`planes_latest_asap.log`) | 17.315 / 50.001 ms | **22.62 ms** | 34.67 FPS, dropped 0 |

영상 흐름을 보여 주는 용도는 source(기본값), 반응 속도가 중요한 용도는 latest + asap으로 쓴다.

## 5. 관련 사항과 남은 것

- 원인은 표시 쪽이 아니라 **추론 배치**다. `--batch 1`이면 짝이 생기지 않는다. 다만 NPU 처리량이 줄어든다(`07_results/npu_benchmark/`).
- 기존 Vulkan 경로는 따로 게이트를 두지 않았는데도 median 33 ms다. 그 경로의 페이싱(`pace_deadline` + FIFO 스왑체인)이 같은 역할을 하는지는 확인하지 않았다.
- pending(준비됐지만 아직 커밋하지 않은 프레임) 한도를 1로 두었을 때는 300프레임 중 26장이 드롭됐다(`planes_val_300f.log`). 그래서 2로 올렸다.
- 게이트의 8 ms 여유가 최적인지는 재지 않았다. 정식 측정 때 present 간격 p90과 glass-to-glass를 함께 본다.
