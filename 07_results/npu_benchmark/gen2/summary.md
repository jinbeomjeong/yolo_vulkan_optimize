# yolov11m.hef NPU benchmark — PCIe Gen2 x1

- 측정 일시: 2026-09-20 00:08~00:17 KST
- HailoRT CLI / FW: 4.24.0 / 4.24.0 (HAILO8)
- PCIe: 5.0 GT/s x1 (Gen2 x1), 장치 최대 8.0 GT/s x4
- 조건: `hailortcli benchmark yolov11m.hef --batch-size N -t 15`, 랜덤 입력, power mode performance

## HEF
- Multi Context 4개
- 입력: UINT8 NHWC 640x640x3
- 출력: FLOAT32 NMS BY CLASS (80 cls, max 100 box/cls, 160,320 B/frame), NMS(YOLOV8, score 0.2, IoU 0.7)는 호스트 CPU에서 수행
- NMS 전 raw UINT8 스트림 6개 (80x80 / 40x40 / 20x24 각각 box 64ch, class 80ch)

## batch별 결과

| batch | FPS hw_only | FPS streaming | HW 지연 (ms) | 프레임당 주기 (ms) |
|---:|---:|---:|---:|---:|
| 1 | 12.64 | 12.71 | 61.69 | 78.7 |
| 2 | 18.09 | 18.09 | 88.82 | 55.3 |
| 4 | 22.62 | 22.62 | 142.55 | 44.2 |
| **8** | **25.96** | **26.07** | **247.61** | 38.4 |
| 12 | 21.54 | 21.54 | 444.80 | - |
| 16 | 22.39 | 22.39 | 576.66 | - |
| 32 | 23.17 | 23.17 | 1046.29 | - |

- batch 8 재측정 값이 0.01 FPS 이내로 같음
- ultra_performance 모드: b1 12.64, b8 26.12 FPS (performance 모드와 차이 1% 미만)
- overall latency (`hailortcli run`): b1 78.52 ms, b8 286.64 ms

## 해석
- 최대 처리량: batch 8에서 약 26 FPS. 지연까지 고려한 절충점은 batch 4 (22.6 FPS, 142.5 ms)
- batch 12 이상: hailort.log에 desc page size 경고가 찍히고 FPS가 떨어짐
- hw_only와 streaming FPS가 같음: 호스트 NMS·데이터 변환은 처리량을 제한하지 않음
- 선형 근사: 배치당 고정 약 46 ms + 프레임당 약 32.6 ms. 배치를 키워도 상한은 약 30 FPS
- 병목: PCIe Gen2 x1. 컨텍스트 전환 때 가중치를 다시 적재하고 중간 activation이 호스트를 왕복함. Gen3 x1 실험에서 두 비용이 모두 반으로 줄어 확인됨 (아래 Gen3 x1 비교 참고)
- 온도 63.7~68.1 °C, 측정 중 스로틀링 없음. 전력 측정은 지원되지 않음 (HAILO_UNSUPPORTED_OPCODE)

## Gen3 x1 비교

PCIe Gen3 x1에서 다시 잰 결과는 [../gen3/summary.md](../gen3/summary.md)에 있다. batch 8 기준 51.7 FPS로 약 2배가 됐고, PCIe 병목 추정이 확인됐다.
