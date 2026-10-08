# 후보 줄이기(점수·topK)로 PCIe 대역폭·NMS 부하를 줄일 수 있는가 — 조사

**제안.** 모델 원시 출력에 점수 임계값과 topK를 적용해 후보 자체를 줄이자는 것이다. 기대한 효과는 두 가지다.
- (1) NPU → 호스트 PCIe 전송량이 준다.
- (2) 그 뒤 NMS 부하가 준다.

이번 범위는 조사와 읽기 전용 측정이다. 코드와 HEF는 바꾸지 않았다.

**질문은 셋이다.**
- **A.** 지금 HEF는 PCIe로 무엇을 얼마나 옮기는가? 출력을 줄이면 처리량이 얼마나 오르는가?
- **B.** 후보를 칩(HEF) 안에서 줄일 수 있는가? PCIe를 줄이려면 칩 안에서 줄이는 수밖에 없다.
- **C.** 호스트 쪽 점수·topK는 NMS 부하를 얼마나 줄이는가?

## 1. 조건

- 2026-10-08, Raspberry Pi 5 + Hailo-8, PCIe **Gen3 x1**(8.0 GT/s, 최대 x4), HailoRT·FW·드라이버 4.24.0, `03_models/yolov11m.hef`(md5 `51566585…`). 환경은 [`env_bench.txt`](env_bench.txt)에 있다.
- **A**
  - 출처는 다섯 가지다.
    - `hailortcli parse-hef`([`evidence/parse_hef.txt`](evidence/parse_hef.txt))
    - HEF protobuf 직접 디코드([`evidence/hef_proto_bytes.txt`](evidence/hef_proto_bytes.txt), 디코더 전문 포함)
    - HailoRT info 로그의 버퍼 계획([`evidence/buffer_alloc_b8.txt`](evidence/buffer_alloc_b8.txt))
    - 숨은 명령 `hailortcli measure-nnc-performance`([`evidence/nnc_estimator_b1.txt`](evidence/nnc_estimator_b1.txt))
    - fw-actions 컨텍스트별 시간([`evidence/fwactions_descriptor_sums.txt`](evidence/fwactions_descriptor_sums.txt))
  - 바이트 표와 모델 적합은 [`evidence/pcie_byte_model_fit.txt`](evidence/pcie_byte_model_fit.txt)에 있다.
  - HailoRT 4.24 release 빌드는 debug 로그를 켤 수 없다(파일 sink가 info로 고정). 그래서 위 출처를 서로 맞춰 바이트를 구했다. 소스 근거는 [`evidence/hailort_src_refs.md`](evidence/hailort_src_refs.md)에 있다.
  - NPU는 각 3초씩 짧게 실행했다(13:50~13:51). 칩 온도는 55.8~58.7 °C였다. fw-actions JSON은 같은 날 09:14~09:16에 따로 실행한 것으로, 온도 기록은 없다. 모든 `evidence/hailort_*.log`에서 warning·error·온도 경고는 0이다.
- **B**: Model Zoo v2.19.1 설정, HailoRT v4.24.0 소스, Hailo 커뮤니티 답변을 조사했다. 재컴파일은 하지 않았다(DFC는 x86 전용). 근거와 인용은 [`evidence/onchip_options.md`](evidence/onchip_options.md), HEF의 NMS 설정은 [`evidence/parse_hef_nms.txt`](evidence/parse_hef_nms.txt)에 있다.
- **C**: 기존 측정 [`nms_crowd_stress`](../nms_crowd_stress/summary.md), [`postproc_stage_cost`](../postproc_stage_cost/summary.md), [`gpu_nms_e2e`](../gpu_nms_e2e/summary.md)를 정리했다. 새로 측정하지 않았다.

## 2. A — 지금 PCIe로 오가는 것

### 2.1 바이트 (HEF 디코드, HailoRT `get_context_transfer_size`와 같은 정의)

| 항목 | 방향 | 크기 | 단위 |
|---|---|---:|---|
| 경계 입력(640×640×3) | H2D | 1,228,800 | 프레임당 |
| **경계 출력(원시 6텐서)** | D2H | **1,221,120** | 프레임당 |
| 컨텍스트 사이 텐서 쓰기(inter-context) | D2H | 6,778,880 | 프레임당 |
| 컨텍스트 사이 텐서 읽기 | H2D | 7,024,640 | 프레임당 |
| DDR 버퍼(호스트 circular) 쓰기 / 읽기 | D2H / H2D | 1,105,920 / 1,105,920 | 프레임당 |
| 컨텍스트 config(가중치 포함) | H2D | **20,519,008** | **배치당 1회** |
| 합계(b1) | | 38,984,288 | = estimator 304,564 KB / 8과 일치 |
| 합계(b8) | | 168,241,248 | 프레임당 21.03 MB |

- 컨텍스트 4개의 중간 텐서는 모두 **호스트 RAM(pinned)을 거쳐 왕복한다**. 로그 기준 `Pinned=7,835,648 (b1) / 47,976,448 (b8)`이다.
- b8 프레임당 전송 21.03 MB 가운데 13.8 MB(66%)가 이 왕복이다. 경계 출력 1.22 MB는 프레임당 전체의 5.8%다(b1은 config 몫이 커서 3.1%).
- config 바이트는 세 출처로 교차 확인했다. fw-actions descriptor 수 × 512 B가 HEF config의 1.001~1.003배이고, descriptor 수는 b1과 b8에서 같다.

### 2.2 T(B) 대조 — 바이트로 시간이 설명된다

| 모델 | 배치당 바이트 | 실효 대역폭 | 예측 T(B) ms | 판정 |
|---|---|---|---|---|
| 실측(gen3 benchmark b1 + `op_point` b2/b4/b8) | — | — | **24.85 + 16.23·B** | 기준 |
| M1 전이중: 컨텍스트마다 max(H2D, D2H) | 20.52 + 12.84·B MB | 0.799 GB/s | 25.68 + 16.07·B | 맞음 |
| 부분 겹침: max + 0.32·min | — | 0.889 GB/s | 컨텍스트별 8점 rms 0.59 ms | 가장 잘 맞음 |
| M2 직렬: H2D + D2H | 20.52 + 18.47·B MB | 1.067 GB/s | — | **불가능**(Gen3 x1 한 방향 선로 0.985 GB/s 초과) |
| M3 H2D만 | — | 0.635 GB/s | — | 기각(rms 10.9 ms) |

- **배치 고정비 24.8 ms는 config 20.5 MB의 재적재와 정합한다.** 지금까지 추정이던 해석을 바이트로 뒷받침했다. 다만 대역폭 하나로 8점을 맞춘 모델이라, 고정비 예측은 모델에 따라 19~32 ms로 갈린다.
- **프레임당 16.2 ms는 컨텍스트마다 더 큰 방향의 바이트 12.84 MB와 정합한다.**
- Gen2 데이터에 같은 바이트를 맞추면 실효 대역폭이 0.405 GB/s(선로의 81%)다. Gen3 0.80 GB/s도 선로의 81%로 같다.
- 측정 기준: 단독 HW 처리량 `measure-nnc-performance` 24.45 / 51.76 FPS가 benchmark 24.34 / 51.71과 같다. 즉 상한은 호스트 SW가 아니라 HW(DMA)가 정한다.

### 2.3 경계 출력을 0으로 만들면

| 모델 | b1 | b8 |
|---|---:|---:|
| M1 전이중 | 0.0% | 0.0% |
| 부분 겹침(최적 적합) | 1.1% | **2.3%** |
| 직렬 합(물리적으로 불가능한 절대 천장) | 3.2% | 6.2% |

출력은 컨텍스트 2(921,600 B)와 3(299,520 B)에서 나간다. 두 컨텍스트는 **H2D가 지배**한다(H2D 4.59 MB 대 D2H 1.93 MB, 1.00 MB 대 0.30 MB). 출력은 비어 있는 D2H 방향을 타므로 임계 경로에 거의 없다.

## 3. B — 칩 안에서 후보 줄이기

| 방법 | 출력 D2H B/프레임 (지금 1,221,120) | 의미·정확도 변화 | 가능 여부 (근거는 `onchip_options.md`) |
|---|---:|---|---|
| 칩 NMS(`engine=nn_core`) | (N+80)×8, 검출 20개면 800 | 출력이 NMS 결과로 바뀐다 | **불가.** yolov8 메타(우리 yolov11m)는 cpu 엔진만 된다. Model Zoo yolov8 계열 alls 31개가 모두 `engine=cpu`이고, 커뮤니티 답변(C2)이 같다 |
| 점수 임계값·topK·gather(가변 길이 출력) | — | — | **불가.** 장치 스트림은 정적 프레임 크기다. 가변 길이를 낼 수 있는 것은 NMS 코어뿐이다 |
| 출력 양자화를 더 줄이기 | — | — | **불가.** 이미 UINT8이고, 공개 설정(Model Zoo `precision_mode`) 기준으로 Hailo-8 활성값은 8/16 bit뿐이다 |
| 클래스 max+argmax(80 → 2ch) | 559,680 (−54%) | 앵커당 top-1 클래스만 남는다. 지금은 임계값을 넘는 모든 클래스를 낸다 | 조건부. argmax 한도가 64 features라는 답변이 있어 80클래스는 분할 우회가 필요하다(미확인) |
| DFL을 칩에서(64 → 4ch, 16 bit 권장) | 746,240 (−39%) | 의미는 같다 | 가능성은 높지만 직접 선례가 없다. ONNX 수정 + 재컴파일이 필요하다 |
| 위 두 가지 조합 | 50,880(2ch argmax + 4ch UINT8) ~ 101,760(분할 argmax 4ch + 16 bit) (−92~96%) | top-1 클래스 + 박스 | 선례 없음 |

- 재컴파일에 필요한 것: x86 Ubuntu, DFC 3.34(HailoRT 4.24 짝), 원본 `yolo11m.onnx`. 권장 사항은 NVIDIA GPU(alls의 finetune 단계에 사실상 필요)와 보정 이미지 1024장 이상이다. 예상 1.5~3주다.
- `evidence/onchip_options.md`의 "처리량 효과의 상한" 절과 권장 3항(+8%, D2H의 15~20%)은 조사 A 이전의 추정이다. **2.3절이 그것을 대체한다.**
- 층이 늘어 컨텍스트가 4개에서 5개가 되면 config가 늘어 절감분이 상쇄될 위험이 있다.
- 칩 안에서 줄이더라도 효과는 2.3절의 상한 안에 갇힌다. **처리량 +0~2.3%(천장 6.2%)**다.

## 4. C — 호스트 쪽 점수·topK

- **점수 임계값은 이미 원시 데이터에 바로 적용되고 있다.**
  - GPU 디코딩 ①②(`02_shaders/yolo_decode.comp`)가 양자화된 바이트를 정수 임계값(`q_thr_low`)과 비교한다. 통과한 그룹만 박스를 계산한다.
  - 비용은 후보 수와 무관한 약 0.30 ms다([`postproc_stage_cost`](../postproc_stage_cost/summary.md)). 다만 PCIe를 건넌 **뒤**이므로 전송량은 줄지 않는다.
- **topK(NMS 앞 상위 K개)는 운용 임계값에서 할 일이 없다.**
  - 점수 0.2에서 후보는 기존 영상 평균 16.8·최대 45개, 가장 혼잡한 MOT20 최대 267개다. 그래서 K ≥ 300이면 결과도 비용도 같다.
  - 그때 NMS는 0.05~0.19 ms이고 CPU로 넘긴 프레임은 0이다.
  - 클래스당 100개 상한(HEF 설정, 억제 뒤 클래스별 top-K)도 혼잡 장면에서 한 번도 닿지 않았다(최대 36개).
- **topK가 의미 있는 곳은 점수를 낮춘 경우뿐이다(0.02 이하, 후보 1000~1900개).**
  - K = 1024로 자르면 GPU NMS 용량 안에 들어간다. 그러면 1024 초과 프레임을 CPU로 넘기지 않아도 되고, CPU 억제 0.8~1.5 ms/f가 사라진다.
  - 대신 CPU 기준 구현과 결과가 달라진다. K 밖의 후보가 최종 검출이 될 수 있기 때문이다.
  - 또 ⑤ emit 비용(후보에 비례, 0.003에서 1.06 ms)은 emit 뒤에 자르면 줄지 않는다.
  - 비교 기준: TRT EfficientNMS는 점수 거르기 뒤 상위 4096개로 자른다([`08_learn/11_tensorrt_efficient_nms.md`](../../08_learn/11_tensorrt_efficient_nms.md)). Ultralytics `non_max_suppression`(`ultralytics/utils/ops.py`)의 기본값은 `max_nms=30000`·`max_det=300`이다(저장소 안에서 검증하지 않은 외부 사실). 둘 다 이상 상황을 막는 안전장치이고 정상 부하를 줄이려는 것이 아니다.

## 5. 결론

1. **후보를 줄여 PCIe를 줄이는 방식으로는 처리량을 거의 올릴 수 없다.**
   - 경계 출력은 프레임당 전송의 약 6%다. 게다가 비어 있는 D2H 방향을 타므로, 0으로 만들어도 이득은 **최대 약 2.3%**(물리적 천장 6.2%)다.
   - 칩 안에서 할 수 있는 방법(DFL·argmax를 칩에서)도 재컴파일과 정확도 확인이 필요해 비용 대비 얻을 것이 없다.
   - 칩 NMS와 topK는 이 모델에서 불가능하다. **비권장.**
2. **NMS 부하도 이미 작다.** 점수 거르기는 이미 원시 데이터 단계에서 한다. topK는 점수 0.2에서 효과가 없다. **비권장.** 다만 낮은 임계값용 안전장치로는 둘 가치가 있다. 그 경우 결과가 바뀌므로 판단 기준이 따로 필요하다.
3. **처리량을 실제로 움직일 지렛대는 따로 있다.** 이번 회계로 처음 바이트 단위로 드러났다.
   - **배치마다 config 20.5 MB 재적재**(고정비 24.8 ms). 특히 컨텍스트 3이 8.15 MB다.
   - **컨텍스트 사이 중간 텐서의 호스트 왕복 13.8 MB/프레임**. 컨텍스트 0 → 2의 `conv21`(80×80×512, 3.28 MB)이 PCIe를 두 번 건너는 가장 큰 항목이다.
   - 둘 다 **컨텍스트 수와 분할**의 문제다. 컨텍스트가 적은 더 작은 모델이나 다른 분할의 HEF를 쓰는 쪽이 효과가 크다. 이는 보류해 둔 "HEF 컨텍스트 조사"의 2단계에 해당하며, 모델이 바뀌므로 사용자가 결정할 일이다.

## 6. 주의·남은 일

- 컨텍스트별 시간 모델은 3초 단발 fw-actions 1회(8점)에 맞췄다. 프로브형 근거이고 3회 중앙값이 아니다. 연산 시간이 전송 아래에 숨는다는 가정은 따로 재지 않았다. Model Zoo Gen3 x4 공개값(b8 102 FPS)을 보면 대역폭이 넓을 때는 연산 하한이 드러난다.
- 컨텍스트 사이 텐서 일부(`conv49`·`concat20`)는 패딩 포함 여부에 따라 1.6% 차이가 난다(위 표는 패딩 포함). 결론은 같다.
- PCIe MPS/MRRS는 root 권한이 없어 읽지 못했다. 81% 효율의 원인은 추정이다.
- B의 표에서 "미확인"으로 둔 것: argmax 64 features 한도가 DFC 3.34에서도 유효한지, 16폭 그룹 softmax가 컴파일되는지, `engine=auto`의 정의, top-1 클래스로 바꿨을 때 mAP가 얼마나 달라지는지.
- `evidence/fwactions_probe_b{1,8}.json`은 `11_logs/`(git 밖)의 바이트 동일 사본이다. 근거를 저장소 안에서 자립시키려고 넣었다.
- 조사 B 중에 `hailortcli fw-control identify`가 한 번 실행됐다(13:45, 장치 쿼리). 조사 A의 NPU 실행(13:51)보다 앞서 끝나 측정과 겹치지 않았다.
