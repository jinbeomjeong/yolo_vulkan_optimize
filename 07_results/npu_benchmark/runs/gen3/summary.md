# yolov11m.hef NPU benchmark — PCIe Gen3 x1

Gen2 x1 기준선은 [../gen2/summary.md](../gen2/summary.md)에, 두 조건 비교 개요는 [../../summary.md](../../summary.md)에 있다.

- 측정 일시: 2026-09-20 00:24~00:44 KST
- 설정 변경: `/boot/firmware/config.txt`의 `[all]` 섹션에 `dtparam=pciex1_gen=3` 추가 후 재부팅 (원본 백업: `config.txt.bak-gen2`)
- 링크: 8.0 GT/s x1 (Gen3), 장치 최대 8.0 GT/s x4. 측정 내내 유지됐고 PCIe 에러는 없었다
- HailoRT CLI / FW: 4.24.0 / 4.24.0 (HAILO8)
- 조건: `hailortcli benchmark yolov11m.hef --batch-size N -t 15`, 랜덤 입력, power mode performance (Gen2와 같음)

## batch별 결과 (Gen2 대비)

| batch | Gen2 FPS | Gen3 FPS | 배율 | Gen2 HW 지연 (ms) | Gen3 HW 지연 (ms) | Gen3 프레임당 주기 (ms) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 12.71 | 24.34 | 1.92× | 61.69 | 32.51 | 41.1 |
| 2 | 18.09 | 34.93 | 1.93× | 88.82 | 46.19 | 28.6 |
| 4 | 22.62 | 44.52 | 1.97× | 142.55 | 73.32 | 22.5 |
| **8** | 26.07 | **51.71** | 1.98× | 247.61 | 126.33 | 19.3 |
| 16 | 22.39 | 43.72 | 1.95× | 576.66 | 299.52 | 22.9 |

- FPS는 streaming 값이다. hw_only 값과 차이가 0.2% 미만이다
- 프레임당 주기 = 1000 / FPS
- batch 16은 칩을 식힌 뒤 다시 잰 값이다 (`../gen3_cooled/benchmark_b16.*`). 첫 측정(43.60 FPS)은 클럭 저하 경고가 찍힌 상태였다
- batch 8 재현성: 식힌 뒤 두 번 더 재서 51.71, 51.64 FPS. 과열 상태에서 연달아 잰 반복값은 50.64 FPS였다
- batch 16의 하락은 Gen2와 같은 원인으로 보인다: `Desc page size value (1024) is not optimal for performance` 경고

## 처리 시간 선형 근사 (batch 1~8, 배치 주기 = batch / FPS)

| | 배치당 고정 비용 | 프레임당 비용 | 이론 상한 FPS |
|---|---:|---:|---:|
| Gen2 x1 | ~46 ms | ~32.6 ms | ~30 |
| Gen3 x1 | ~25 ms | ~16.2 ms | ~62 |

두 비용이 모두 거의 정확히 반으로 줄었다. 대역폭 2배가 그대로 성능 2배로 이어졌으므로 **PCIe 대역폭이 NPU 처리량의 병목**이라는 추정이 확인됐다.
배치당 고정 비용은 컨텍스트 4개의 가중치 재적재로, 프레임당 비용은 입력 전송과 컨텍스트 경계의 중간 activation 왕복으로 해석한다.

## 발열 (주의)

- Hailo 칩 온도: 30초 측정 한 번에 82~87 °C, 연속 측정 후 최고 92.7 °C. 식히면 약 20분 뒤 61 °C 안팎까지 내려간다 (`chip_temp_cooldown.txt`)
- 과열 구간(첫 묶음의 batch 16, batch 8 반복, `hailortcli run` 지연 측정)에서 `HEF was compiled assuming clock rate of 400 MHz, while the device clock rate is 250 MHz` 경고가 찍혔다. Gen2에서는 한 번도 나오지 않았다
- SoC `get_throttled`는 계속 0x0이었다. Pi가 아니라 Hailo 칩 쪽의 클럭 저하다
- 연속으로 돌리는 실제 파이프라인에서는 클럭 저하로 위 FPS가 유지되지 않을 수 있다. 방열판이나 팬을 달거나, 1~2분 이상 연속 부하로 안정 상태 FPS를 따로 확인해야 한다

## 전체 지연 (overall latency) — ✅ 재측정 완료 (2026-09-20)

> **이 절의 아래 Gen3 표는 대체됐다. 인용하지 마라.**
> 냉각 상태 재측정 결과는 [`07_results/op_point/runs/latency_gen3_cooled/summary.md`](../../../op_point/runs/latency_gen3_cooled/summary.md) 에 있다.
> 요약: b1 overall **40.894 ms**(과열값 42.78 대비 −4.4%), b8 overall **144.863 ms**(155.75 대비 −7.0%).
> 3회 중앙값, 15회 전부 온도·클럭 경고 0건. 아래 표는 과열 상태 기록으로 남겨 둔다.

| batch | Gen2 overall (ms) | Gen3 overall (ms, 신뢰 불가) | Gen3 HW (ms, 신뢰 불가) |
|---:|---:|---:|---:|
| 1 | 78.52 | 42.78 | 34.62 |
| 8 | 286.64 | 155.75 | 133.35 |

Gen3 값은 클럭이 250 MHz로 떨어진 상태에서 잰 것이다. benchmark의 HW 지연(b1 32.51, b8 126.33 ms)보다 크게 나왔다.
~~**TODO:** 칩을 약 60 °C까지 식힌 뒤 `hailortcli run <hef> --batch-size {1,8} --measure-latency --measure-overall-latency`로 다시 재고, clock rate 경고가 없는지 확인한다.~~ → **해소됨**(위 링크).

## 파일

- `device_info.txt` — identify, 링크 상태
- `benchmark_b{1,2,4,8,16}.*`, `benchmark_b8_repeat.*` — 첫 측정 묶음
- `run_b{1,8}_latency.txt` — 지연 측정 (과열 상태)
- `thermal_{before,after}_*.txt`, `chip_temp_cooldown.txt` — 온도 기록
- `../gen3_cooled/` — 식힌 뒤 다시 잰 batch 8(2회), batch 16
- `console.txt`, `hailort.log` — 콘솔 출력과 런타임 로그
