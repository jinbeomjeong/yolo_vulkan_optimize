# yolov11m.hef NPU benchmark 개요

`hailortcli benchmark`로 잰 순수 NPU 처리량이다. PCIe 링크 속도를 바꿔 두 번 측정했다.

- [runs/gen2/summary.md](runs/gen2/summary.md) — PCIe Gen2 x1 (`dtparam=pciex1_gen`을 설정하지 않았을 때의 값), 2026-09-20 00:08~00:17 KST
- [runs/gen3/summary.md](runs/gen3/summary.md) — PCIe Gen3 x1 (`dtparam=pciex1_gen=3`), 2026-09-20 00:24~00:44 KST

> 두 폴더 이름은 **그 측정 당시의 링크 세대**를 가리킨다. 지금 보드가 어느 세대로 도는지는
> 여기서 추론하지 말고 읽는다:
> ```bash
> cat /sys/bus/pci/devices/0001:01:00.0/current_link_speed   # 5.0 GT/s = Gen2, 8.0 GT/s = Gen3
> ```

공통 조건: HailoRT 4.24.0, 모델 `03_models/yolov11m.hef`(컨텍스트 4개, 입력 640x640x3 UINT8, 출력 NMS BY CLASS), `-t 15`, 랜덤 입력, power mode performance.

## batch별 FPS (streaming)

| batch | Gen2 x1 | Gen3 x1 | 배율 |
|---:|---:|---:|---:|
| 1 | 12.71 | 24.34 | 1.92× |
| 2 | 18.09 | 34.93 | 1.93× |
| 4 | 22.62 | 44.52 | 1.97× |
| **8** | 26.07 | **51.71** | 1.98× |
| 16 | 22.39 | 43.72 | 1.95× |

## 결론

- 최대 처리량은 두 조건 모두 batch 8이다. Gen3에서 **51.7 FPS**다
- 처리 시간을 "배치당 고정 + 프레임당" 비용으로 나누면 Gen2 ~46 + 32.6 ms가 Gen3 ~25 + 16.2 ms로 반이 됐다. **PCIe 대역폭이 병목**이다. Gen3 이론 상한은 약 62 FPS다
- batch 12 이상에서는 DMA desc page size 경고와 함께 FPS가 떨어진다
- hw_only와 streaming FPS가 같다. 호스트 NMS는 처리량을 제한하지 않는다

## 주의와 남은 일

- Gen3에서는 Hailo 칩이 최고 92.7 °C까지 올라 클럭이 400 MHz에서 250 MHz로 떨어졌다. 연속 부하에서 FPS를 유지하려면 방열 대책이 필요하다
- **TODO:** Gen3 overall latency는 과열 상태에서 잰 값이라 칩을 식힌 뒤 batch 1/8을 다시 재야 한다 (자세한 내용은 runs/gen3/summary.md)
