# 진단: ASPM L1을 끄고 표시 on/off 비교 (2026-11-06)

[`../display_onoff`](../display_onoff/summary.md)의 진단 회차다. 조건·결과·결론은 그 요약에 함께 적었다.

- 측정 동안만 Hailo-8 PCIe 링크의 ASPM L1을 껐고(`l1_aspm` 1 → 0, 실행 중 `LnkCtl: ASPM Disabled`), 끝난 뒤 1로 되돌린 것을 확인했다. 거버너는 측정 동안만 performance.
- 결과: 표시 on 도착→결과 53.66 ms·NPU 32.50 ms, 표시 off 58.40 ms·38.23 ms. ASPM 기본 회차와 같다. **ASPM L1은 원인이 아니다.**
- `hailort.log` warning·error 6회 모두 0건. 환경은 [`env_bench.txt`](env_bench.txt).

결정: ASPM L1을 측정 동안만 끄고 비교한다 (사용자 결정, 2026-11-06)
