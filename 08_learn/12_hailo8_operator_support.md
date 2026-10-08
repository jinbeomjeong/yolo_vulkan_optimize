# 12. Hailo-8 연산자 지원 — 무엇이 칩에서, 무엇이 호스트에서 도는가

> 작성 2026-10-08 (초안). 대상 칩은 **Hailo-8**이다. Hailo-8L·Hailo-10H는 §12.2.13에서 따로 구분한다.
> 장치 실행 없이 HEF 파일 읽기, 소스, 문서만으로 조사했다.
> 출처 약칭(UG, MZ, MZ-sk, RT, HEF, C1~C6)은 §12.7에 있다. **표의 모든 행에 출처를 단다.**
> 원문을 확인하지 못한 주장은 **미확인**, 근거를 해석한 것은 **(해석)**, 계산으로 낸 추정은 **(추정)**이라고 적는다.
>
> 선수: [03 YOLO와 텐서](03_yolo_tensor.md)(DFL·NMS·양자화), [04 NPU와 HailoRT](04_npu_hailort.md)(HEF·컨텍스트).
> 같은 성격의 외부 참고 문서: [11 TensorRT EfficientNMS](11_tensorrt_efficient_nms.md).
> 이 장의 앞선 조사: [`onchip_options.md`](../07_results/candidate_reduction_survey/evidence/onchip_options.md)
> (칩 안 후보 축소 조사. 이 장은 그 내용을 다시 쓰지 않고 링크한다).

---

## 12.0 이 장이 답하는 질문

- ONNX의 어떤 연산이 Hailo-8 칩(nn_core)에 올라가는가? 제약과 정밀도는?
- 칩에 못 올라가는 연산은 어디서 도는가? HailoRT 호스트 op인가, 사용자 코드인가?
- YOLO11의 블록(C3k2, SPPF, C2PSA, DFL, Upsample, Concat)은 각각 어디서 도는가?
- 그래서 프로젝트 목적 3 "YOLO 모델의 NPU 최적화"에서 무엇을 옮길 수 있고, 무엇은 못 옮기는가?

---

## 12.1 개요 — 세 층

```
 ONNX (PyTorch export)
   │
   ├─[파서 Parser]──── 잘라낸다: NMS, TopK, detection-output, DFL·디코드(관례)
   │                   (end_node_names 뒤는 HAR에 들어가지 않는다)
   ▼
 HN/HAR ── 최적화(양자화) ── 컴파일(할당·컨텍스트 분할) ──▶ HEF
   │
   ├─[칩 nn_core]───── conv, dw, pool, resize, concat, ew_add, matmul, softmax 분해 …
   │                   (UG §5.5의 레이어만. 정적 shape만)
   ▼
 [호스트 HailoRT net_flow op] ── YOLOV8/YOLOV5/YOLOX/SSD/IOU NMS, SOFTMAX, ARGMAX
   │                   (HEF 메타데이터의 op. 호스트 CPU에서 돈다)
   ▼
 애플리케이션 (이 프로젝트는 원시 6텐서를 받아 GPU로 후처리한다)
```

| 층 | 무엇이 정하는가 | 바꾸려면 | 출처 |
|---|---|---|---|
| 파서 | `translate_onnx_model(..., end_node_names=...)`. 후처리 앞의 conv에서 자른다 | ONNX 수정 + 재파싱(x86 DFC) | MZ-sk `hailo-parse/SKILL.md:88,146`, MZ `networks/yolov11m.yaml:26-34` |
| 칩 | DFC가 HN 레이어를 클러스터에 할당. 모델이 크면 컨텍스트로 나눈다 | alls 수정 + 재컴파일(x86 DFC) | UG §5.5, MZ `alls/generic/yolov11m.alls` |
| 호스트 op | alls의 `nms_postprocess(...)`, `logits_layer(...)`가 HEF에 op 메타데이터로 들어간다 | 재컴파일, 또는 원시 출력을 받아 직접 처리(이 프로젝트) | RT `net_flow/ops_metadata/op_metadata.hpp:30-39`, `hef.proto:290-332` |

### 12.1.1 "지원"의 세 가지 뜻

같은 "지원"이라도 층마다 뜻이 다르다. 셋 다 통과해야 칩에서 돈다.

1. **파서 지원**: ONNX op을 Hailo 레이어로 번역할 수 있다(UG §5.1 Table 2). ONNX opset은 UG v3.30 기준 8, 11~17이다(UG §5.1.3).
   - MZ-sk(`hailo-parse/SKILL.md:35`)는 "15~21"이라 적는다. 다만 이 스킬은 Hailo-10H/15용 DFC 5.x 기준이다(같은 파일 `:19`). DFC 3.34의 범위는 **미확인**.
2. **커널 지원**: 칩에 그 레이어의 구현이 있다(UG §5.5). UG는 두 수준을 구분한다.
   - "supported": 모든 정수 kernel·stride·dilation이 된다.
   - "optimized params"(Table 9·13·15): 그 조합만 빠르다. 나머지는 돌지만 느릴 수 있다(해석).
3. **할당 성공**: 자원(클러스터·메모리) 안에 들어간다. 큰 softmax처럼 커널은 있어도 할당이 안 되는 경우가 있다(MZ `docs/CHANGELOG.rst:13`, STARK의 400폭 softmax).

### 12.1.2 정밀도 모델

| 항목 | 값 | 출처 |
|---|---|---|
| 활성값(activation) | 8 bit 또는 16 bit | MZ alls 전체의 `precision_mode` 값이 `a8_w4`, `a8_w8`, `a16_w16` 세 가지뿐이다([`onchip_options.md`](../07_results/candidate_reduction_survey/evidence/onchip_options.md) Q4 (d)) |
| 가중치 | 4 / 8 / 16 bit. 4 bit는 가중치에만 쓴다 | 위와 같음. 큰 네트워크는 기본으로 가중치의 20%를 4 bit로 둔다(UG Changelog v3.20.0). `compression_level`이 비율을 정한다(MZ-sk `hailo-optimize/SKILL.md:62`) |
| 16 bit가 되는 레이어 | Activations, AvgPool, Concat, Const Input, Conv, Deconv, Depth to Space, DW Conv, EW Add/Sub\*, External Pad, Feature Shuffle, Feature Split, FC(출력도 16 bit여야 함), MaxPool, Normalization, Output, ReduceMax\*, ReduceSum\*, Resize\*, Reshape, Shortcut, Slice, Space to Depth | UG v3.30 p.90 "16-bit precision is supported on the following layers". \*표는 "Softmax chain의 일부가 아닐 때만" |
| 16 bit 목록에 없는 것 | Matmul, Softmax, Argmax, ReduceMean, LayerNorm 블록 | 위 목록에 없다. 이들의 16 bit 가능 여부는 **미확인** |
| 레이어 하나의 가중치 | ≤ 8 MB (모든 Conv) | UG §5.5.1 Note |

- 활성화 함수는 보통 **앞 레이어에 융합**된다. 융합이 안 될 때만 독립 레이어가 된다(UG §5.5.35).
  우리 HEF에서 SiLU가 따로 보이지 않는 이유다(§12.5.1).
- 이 HEF의 출력 6개는 모두 UINT8이다. 클래스 텐서는 `qp_zp=0`이라 sigmoid가 칩에서 이미 적용됐다
  ([`parse_hef_nms.txt`](../07_results/candidate_reduction_survey/evidence/parse_hef_nms.txt)).

---

## 12.2 칩(nn_core) 지원 연산자

- 기준: **DFC User Guide v3.30.0 §5.5 "Supported Layers"**(UG). v3.31.0도 같은 내용이다. 우리 런타임과 짝인 DFC 3.34의 원문은 **미확인**이다(공개 사본을 찾지 못했다).
- "Hailo-8 선례" 열은 실제로 Hailo-8용으로 컴파일된 근거다.
  - `HEF`: 우리 `03_models/yolov11m.hef` 안의 레이어 이름(§12.5.1).
  - `MZ h8`: Model Zoo `cfg/alls/hailo8/base/*.alls`. Hailo-8 할당 결과를 고정해 둔 스크립트라, 거기 나오는 레이어는 Hailo-8 칩에 배치된 것이다(해석).
- UG §5.5와 §5.1 표에는 **칩 세대별 구분이 없다**. 세대 차이는 §12.2.13에서 따로 다룬다.

### 12.2.1 Convolution 계열

| 연산 | Hailo-8 | 제약 | 정밀도 | 출처 |
|---|---|---|---|---|
| Conv2d (일반) | 지원 | kernel·stride·dilation 모든 정수. 패딩은 VALID, SAME(대칭), SAME_TENSORFLOW. 그 밖의 패딩은 External Pad 레이어가 된다. **최적화 조합**: 1×1 s1/s2, 3×3 s1/s2, 5×5·7×7 s1/s2, 6×6 s2, 비대칭 1×N·N×1 등(Table 9) | a8_w4/a8_w8/a16_w16 | UG §5.5.1 Table 9, §5.1.7 · HEF conv 95종 |
| Conv + elementwise add 융합 | 지원 | 1/3/5/7 조합 kernel, stride 1, dilation 1만(Table 10). **입력 텐서 2개만** 더할 수 있다 | 〃 | UG §5.5.1 Table 10 |
| Dilated conv | 지원 | 최적화 조합은 3×3에서 dilation 2/3/4/6/8/16, **stride 1만** | 〃 | UG §5.5.1 Table 9 |
| Depthwise | 지원 | 모든 정수. 최적화: 3×3 s1/s2(dil 2/4는 s1만), 5×5 s1/s2, 3×5·5×3, 9×9(Table 15). DW 1×1 + add는 텐서 2개만 | 〃 | UG §5.5.7 · HEF dw 7개 · MZ h8 `deeplab_v3_mobilenet_v2.alls:102` |
| Group conv | 지원 | 1×1/1, 1×1/2, 3×3/1, 7×7/2는 출력 채널 수 자유. 그 밖은 그룹당 출력 채널 OF%8=0 또는 OF<8 | 〃 | UG §5.5.8. Hailo-8 선례 **미확인** |
| Transposed conv (Deconv) | 제한 | kernel/rate 쌍이 정해져 있다: 16×16/8, 8×8/8, 8×8/4, 4×4/4, 4×4/2, 3×3/2, 2×2/2, 1×1/1. 패딩 SAME_TENSORFLOW만. Group·Depthwise deconv는 OF%8=0 또는 OF<8 | 〃 | UG §5.5.6, §5.5.9 · MZ `alls/generic/unet_mobilenet_v2.alls:3`(`deconv5`, `networks/unet_mobilenet_v2.yaml`은 hailo8 지원) |
| Conv1d | 지원 | Conv로 번역 | 〃 | UG §5.1 Table 2 |
| Conv3D | preview | rank-4 입출력 모델 안에서만. 3×3×3 등 두 조합 | 〃 | UG §5.5.1 Table 11 |
| Einsum | 일부 | group conv에 해당하는 식(`bmchw,bnmc->bmhwn`) 등 | 〃 | UG Table 2, Changelog v3.29.0 |

### 12.2.2 Dense, Pooling

| 연산 | Hailo-8 | 제약 | 정밀도 | 출처 |
|---|---|---|---|---|
| Dense / FC (`Gemm`, 상수 `MatMul`) | 지원 | Dense·Conv·MaxPool·GAP 뒤, 또는 첫 레이어일 때만. Conv 뒤면 벡터로 펴는데, 펼친 높이는 255행 이하 | 16 bit면 출력도 16 bit | UG §5.5.3 · MZ h8 `stdc1.alls:90`(`fc1`) |
| MaxPool | 지원 | 2×2 s1/s2, 1×2, 3×3 s1/s2, **5×5·9×9·13×13 s1**, 그 밖은 2 ≤ k ≤ 입력 크기. `ceil_mode=True`(ONNX·PyTorch) | 16 bit 가능 | UG §5.5.2 Table 12, Changelog v3.25.0 · HEF maxpool 3개 |
| AvgPool | 지원 | 모든 정수. 최적화: 2×2/2, 3×3/1·2, 5×5/1·2, h×W, Global | 16 bit 가능 | UG §5.5.4 Table 13 · MZ h8 `stdc1.alls:3`(`avgpool7`) |
| Global MaxPool | 지원 | ONNX 파서 지원 | — | UG Changelog v3.22.1·v3.23.0 |
| ReduceMean | 지원 | 공간 축이면 AvgPool, 채널 축이면 reduce mean으로 번역 | 16 bit 목록에 없음 | UG §5.5.25, Table 2 |

### 12.2.3 활성화 함수

UG §5.1.4 Table 3 전체를 옮긴다. 모두 "앞 레이어에 융합, 안 되면 독립 레이어"다(§5.5.35). 활성화는 16 bit 목록에 들어 있다.

| 활성화 | ONNX에서 인식하는 형태 | Hailo-8 | 출처 |
|---|---|---|---|
| ReLU, LeakyReLU, PReLU | `Relu`, `LeakyRelu`, `PRelu` | 지원 | UG Table 3 |
| ReLU6, Clip, Min/Max(상수) | `Clip`, `Min`, `Max` | 지원 | UG Table 3 |
| Sigmoid | `Sigmoid` | 지원 | UG Table 3 · HEF 출력 conv74/90/105(MZ `yolov11m.alls:2-4`) |
| **SiLU** | `Mul` + `Sigmoid` | 지원 | UG Table 3 · HEF(YOLO11 Conv 블록 전부, 융합되어 이름이 따로 없다) |
| Swish | TF `tf.nn.swish`만 | 지원 | UG Table 3 |
| HardSigmoid | `HardSigmoid` | 지원 | UG Table 3 |
| HardSwish | `Mul` + `HardSigmoid` | **preview** | UG Table 3 |
| GELU | `Mul` + `Erf` | **preview** | UG Table 3 |
| Mish | `Mul`, `Tanh`, `Softplus` | 지원 | UG Table 3 |
| Tanh, ELU, Softplus, Softsign | `Tanh`, `Elu`, `Softplus`, `Softsign` | 지원 | UG Table 3 |
| Exp, Log, Sqrt, Abs | `Exp`(ONNX 셀은 사본에서 비어 있음), `Log`, `Sqrt`, `Abs` | 지원 | UG Table 3 |
| Equal, Greater, Less | `Equal`, `Greater`, `Less` | 지원(Greater는 상수 비교만) | UG Table 3, Changelog v3.25.0 |
| Delta, Threshold, Linear | 내부 활성화 | 지원 | UG Table 3 |
| Sign, 단독 Erf | — | 표에 없음. **미확인** | UG Table 3에 행이 없다 |

### 12.2.4 정규화

| 연산 | Hailo-8 | 제약 | 정밀도 | 출처 |
|---|---|---|---|---|
| BatchNorm | 지원 | Conv·Dense에 접힌다(fold). 못 접으면 독립 레이어. **런타임 통계 계산은 불가** | — | UG §5.5.14 |
| 입력 정규화(mean/std) | 지원 | 첫 레이어로만 | 16 bit 가능 | UG §5.5.12 · HEF `normalization1`(MZ `yolov11m.alls:1`) |
| LayerNorm | 지원(분해) | "여러 Hailo 레이어 블록으로 번역". Hailo-8에서 low/high 정밀도 분할과 `square`·`reduce_mean`·`precision_change` 레이어로 쪼개진다 | 내부에 16 bit 혼합(해석) | UG Table 2, Changelog v3.30.0 · MZ h8 `deit_base.alls:119,1000` · MZ `generic/mivolo_d1.alls:5` |
| InstanceNorm | 지원(분해) | 블록으로 번역 | — | UG Table 2, Changelog v3.20.0 |
| L2Normalization, ReduceL2 | 지원(분해) | rank-4 레이어 뒤에서만 | — | UG §5.5.38, Table 2 |
| GroupNorm | **미확인** | UG 전체에 언급이 없다 | — | UG(없음) |

### 12.2.5 Elementwise

| 연산 | Hailo-8 | 제약 | 정밀도 | 출처 |
|---|---|---|---|---|
| Add / Sub | 지원 | bias는 Conv·Dense에 융합. 텐서끼리 더하기는 가능하면 Conv에 융합. 상수 스칼라 더하기 가능. shape 규칙은 Mul과 같다 | 16 bit 가능(softmax chain 밖) | UG §5.5.11 · HEF `ew_add` 16개 |
| Mul / Div | 지원 | ① 같은 shape ② `[N,H,W,1]` 방송 ③ `[N,1,1,F]` 방송 ④ `[N,1,1,1]` 방송. 이중 방송 `Hx1xC * 1xWxC`도 된다 | — | UG §5.5.10, Changelog v3.22.0 · MZ h8 `stdc1.alls:36`(`ew_mult3`) |
| 스칼라 곱 | 지원 | 상수 스칼라 | — | UG §5.5.13 |
| Elementwise Max / Min | 지원 | 같은 shape 두 텐서. 내부에서 Concat + ReduceMax로 구현 | — | UG §5.5.37, Table 2 |
| Square, Pow | 제한 | `x*x`, `Pow`는 지수 2 또는 0<a<1만 | — | UG §5.5.36 |

### 12.2.6 모양 바꾸기(shape)

| 연산 | Hailo-8 | 제약 | 정밀도 | 출처 |
|---|---|---|---|---|
| Concat | 지원 | 4차원 텐서, **채널(features) 축만**, **입력 최대 4개** | 16 bit 가능 | UG §5.5.5 · HEF `concat` 22개 |
| Split (Features Split) | 지원 | 채널 축을 연속 구간으로. **정적 분할만**(좌표가 데이터에 따라 바뀌면 안 된다) | 16 bit 가능 | UG §5.5.27 · HEF `conv_feature_splitter` 10개 |
| Slice | 지원 | H·W·F 축 연속 구간. **정적만** | 16 bit 가능 | UG §5.5.28 · MZ h8 `stereonet.alls:239` |
| Reshape | 제한 | ① Conv형 → Dense형(펴기) ② Dense형 → Conv형(F′%8=0) ③ `(N,H,1,F)` → `(N,H,W′,F′)` 세 경우만 | 16 bit 가능 | UG §5.5.29 · MZ h8 `vit_small_bn.alls:5` |
| Flatten | 지원 | Reshape ①에 해당 | — | UG §5.5.29 |
| Transpose | 제한 | W↔C(원문 "Width <-> Column"), H↔W와 H↔F는 양자화 크기 **1.5 MB 미만**만. 전체 버퍼링이 필요해 "pipeline stop"이 생기고 지연이 는다 | — | UG §5.5.34 |
| Depth to Space | 지원 | DCR·CRD, 블록 1×2·2×1·2×2(Table 16), 입력 채널 % (B_W·B_H) = 0 | 16 bit 가능 | UG §5.5.16. Hailo-8 선례 **미확인** |
| Space to Depth | 지원 | classic(M×N 블록), Focus(2×2, YOLOv5·YOLOP의 그것) | 16 bit 가능 | UG §5.5.17 · MZ h8 `vit_small_bn.alls:7`, `deit_base.alls:109` |
| Feature shuffle | 지원 | F%G=0. TF의 reshape→transpose→reshape 패턴을 자동 인식 | 16 bit 가능 | UG §5.5.26, §5.1(TF 패턴) |
| External Pad | 지원 | 세 패딩 방식 밖의 0 패딩 | 16 bit 가능 | UG §5.5.30 · MZ h8 `stereonet.alls:333` |
| Const input | 지원 | 지원 레이어 절은 없고 16 bit 목록에만 나온다. 범위 **미확인** | 16 bit 가능 | UG p.90 · MZ h8 `stereonet.alls:12` |
| Tile, Squeeze/Unsqueeze | **미확인** | Tile은 TF 표의 Concat 행에만 나온다. Squeeze는 Conv3D 설명에만 | — | UG Table 1, §5.5.1 |

### 12.2.7 Resize

| 연산 | Hailo-8 | 제약 | 정밀도 | 출처 |
|---|---|---|---|---|
| Resize nearest / bilinear | 지원 | 행·열 배율이 달라도 된다. 배율은 실수, 새 크기는 정수(행은 ≤ 4096). `align_corners`/`half_pixel` 조합 셋 중 하나. Nearest는 축소도 된다 | 16 bit 가능(softmax chain 밖) | UG §5.5.15, Changelog v3.22.0 · HEF `resize` 2개 · MZ h8 `deeplab_v3_mobilenet_v2.alls:13` |
| 방송형 resize | 지원(NN만) | `(N,H,1,F)` → `(N,rH,W,F)`, `(N,H,W,1)` → `(N,H,W,F)` | — | UG §5.5.15 |

### 12.2.8 Softmax, Reduce, Argmax

| 연산 | Hailo-8 | 제약 | 정밀도 | 출처 |
|---|---|---|---|---|
| Softmax | 지원(조건부) | ① Dense형 `(N,F)` 뒤 ② 열이 1개인 텐서(행마다) ③ 일반 텐서에서 **픽셀마다 채널 축으로**. ③은 다른 레이어로 쪼개 구현한다 | 16 bit 목록에 없음. 그 사슬 안의 ReduceMax·ReduceSum·Add는 16 bit 불가 | UG §5.5.18, p.90 Note · HEF `reduce_max_softmax1`·`ew_sub_softmax1`·`reduce_sum_softmax1`·`ew_mult_softmax1` · MZ h8 `efficientnet_m.alls:155` |
| Softmax 폭 한도 | **제한** | 400폭 softmax는 Hailo-8·8L에 매핑되지 않았다. 정확한 한도는 **미확인** | — | MZ `docs/CHANGELOG.rst:13`(STARK) |
| LogSoftmax | 지원(분해) | 다른 레이어로 쪼갠다 | — | UG §5.5.19 |
| ReduceMax | 제한 | **채널 축만, `keepdims=True`**, 4차원 입력 | 16 bit 가능(softmax chain 밖) | UG §5.5.21, Table 2, C4 |
| ReduceMin | 제한 | 채널 축만(`-max(-x)`) | — | UG §5.5.24, Table 2 |
| ReduceSum | 지원 | 4차원이면 모든 축, 2차원이면 채널 축. H·W 동시 축은 v3.26부터 | 16 bit 가능(softmax chain 밖) | UG §5.5.22, Changelog v3.26.0 · MZ h8 `stereonet.alls:146` |
| ReduceSumSquare | 지원 | 채널 또는 공간 축 | — | UG §5.5.23 |
| ArgMax | **제한** | **네트워크의 마지막 레이어**일 때만, 앞 레이어 출력이 4차원, **채널 64개까지** | 16 bit 목록에 없음 | UG §5.5.20, C3 · MZ h8 `stdc1.alls:13`(`argmax1`, 19클래스) |

### 12.2.9 MatMul과 어텐션

| 연산 | Hailo-8 | 제약 | 정밀도 | 출처 |
|---|---|---|---|---|
| MatMul (데이터 × 데이터) | **제한** | "**Multi Head Attention 블록의 일부로만**" 지원. 한쪽이 상수면 Dense로 번역된다 | 16 bit 목록에 없음 | UG §5.5.31, Table 2 · HEF `matmul1`·`matmul2` · MZ h8 `deit_base.alls:1000`, `yolo26m_sem.alls:20` |
| Multi Head Attention | 지원 | 3차원 입력. `Softmax(QKᵀ/√d)V`. 블록 전체를 번역하며, 컴파일 때 head별로 쪼갤(defuse) 수 있다 | — | UG §5.5.32, Changelog v3.23.0(preview)·v3.25.0 |
| Window Attention | preview | — | — | UG Changelog v3.30.0 |
| Area attention (YOLOv12) | Hailo-8 선례 있음 | MZ `yolov12n`이 Hailo-8에 있다. 파서 에러 사례도 있다 | — | MZ `docs/public_models/HAILO8/HAILO8_object_detection.rst:554`, C4 |

### 12.2.10 순환·기타

| 연산 | Hailo-8 | 제약 | 출처 |
|---|---|---|---|
| RNN, LSTM(양방향 포함), GRU | 지원(펼침) | 시퀀스 길이만큼 블록을 복제한다. 길이 10(양방향 5) 초과면 성능이 떨어진다 | UG §5.5.33, Table 2 |
| OneHot | preview | delta 활성화 Conv로 구현, 마지막 축만 | UG Table 2, Changelog v3.25.0 |
| Dropout | 무시 | 추론에서 제거 | UG Table 2 |
| 입력 형식 변환 | 지원 | NV12/NV21/i420 → YUV, YUV → RGB/BGR, RGBX → RGB를 칩 첫 단에서 | UG Changelog v3.20.0·v3.22.0·v3.23.0 · MZ `generic/yolov5m_vehicles_nv12.alls:15` |
| 입력·출력 resize 삽입 | 지원 | alls `resize(...)` 명령 | UG Changelog v3.25.0 |

### 12.2.11 칩에서 안 되는 것 (요약, 자세히는 §12.4)

- 데이터에 따라 길이가 바뀌는 출력: TopK, 동적 Gather, NMS 결과 목록. 칩 스트림은 정적 프레임 크기다.
- 데이터에 따라 위치가 바뀌는 Slice/Split.
- MHA 밖의 데이터×데이터 MatMul, 1.5 MB를 넘는 H↔W transpose, 채널 축이 아닌 ReduceMax, 64채널을 넘는 ArgMax.

### 12.2.12 세 가지 "지원" 수준 정리

| 수준 | 예 | 의미 |
|---|---|---|
| 최적화 조합 | 3×3 s1 conv, 5×5 s1 maxpool | 표에 올라 있다. 빠르다 |
| 지원 | 11×11 conv, 4×4 avgpool | 돈다. 성능은 보장하지 않는다(해석) |
| preview | GELU, HardSwish, Conv3D, OneHot | 동작은 하지만 성숙도가 낮다 |

### 12.2.13 Hailo-8 · Hailo-8L · Hailo-10H 차이

| 항목 | Hailo-8 (이 프로젝트) | Hailo-8L | Hailo-10H | 출처 |
|---|---|---|---|---|
| 컴파일러 계열 | DFC 3.x. MZ v2.19.1 = DFC 3.34 + HailoRT 4.24 | Hailo-8과 같음 | DFC 5.x 계열. 다만 DFC 3.28~3.30도 `hailo10h` 대상을 가졌다 | MZ `README.rst:73`, MZ-sk `hailo-parse/SKILL.md:19,25`, UG Changelog v3.28.0, C6(DFC 5.3.0) |
| 런타임 | HailoRT 4.x (설치본 4.24.0) | 같음 | HailoRT 5.x | C6(HailoRT 5.3.0) |
| UG의 레이어 표 | 칩별 구분 없음 | 〃 | 〃 | UG §5.5, §5.1 |
| 자원 한계로 갈리는 모델 | 400폭 softmax 불가(8·8L). `yolo26s_seg`, `mivolo_d1` 등은 "Hailo-8 only"(8L 불가) | 더 작은 칩이라 자원 실패가 먼저 온다(해석) | 일부 트랜스포머가 처음엔 "Hailo-15/10H only"였다가 뒤에 Hailo-8로도 나왔다(PoolFormer, DaViT) | MZ `README.rst:80-83`, `docs/CHANGELOG.rst:12-13,25,46,90,122` |
| 다음 컨텍스트 config 미리 가져오기 | **없다** | 없다 | 있다(hailo1x) | RT `core_op/resource_manager/resource_manager_builder.cpp:993` |

- 연산자 커널 자체가 세대마다 다른지는 공개 자료로 **미확인**이다. 지금 보이는 차이는 자원과 컴파일러 성숙도에서 온다(해석).

---

## 12.3 호스트(HailoRT) 후처리 op

- HEF 안의 op 메타데이터를 HailoRT가 읽어 **호스트 CPU에서** 실행한다. 칩과 무관하다.
- 종류: RT `net_flow/ops_metadata/op_metadata.hpp:30-39`의 `OperationType` = YOLOX, YOLOV5, YOLOV8, YOLOV5SEG, SSD, SOFTMAX, ARGMAX, IOU.
- HEF 쪽 정의: `hef.proto:290-321`(`ProtoHEFNmsOp`, oneof `nms_op`), `:323-331`(`ProtoHEFLogitsOp`, softmax/argmax).

| op (alls) | HailoRT 구현 | 하는 일 | 입력 조건 | 출처 |
|---|---|---|---|---|
| **YOLOV8** (`nms_postprocess(meta_arch=yolov8, engine=cpu)`) | `net_flow/ops/yolov8_post_process.hpp:24` | DFL(16빈 softmax + 기댓값) → dist2bbox → 점수 거르기 → 클래스별 NMS. **sigmoid는 하지 않는다**(칩에서 이미 함) | 원시 reg 64ch + cls 80ch × 3스케일 | `yolov8_post_process.hpp:45-63`(get_bbox), `:142-144`(점수 거르기) · **우리 HEF가 쓰는 op** |
| YOLOV8 bbox만 | `yolov8_bbox_only_post_process.hpp:25` | 디코드만, NMS 없음 | 〃 | MZ `alls/generic/yolov8s_bbox_decoding_only.alls:2` |
| YOLOV5 / YOLOV5 bbox만 | `yolov5_post_process.hpp:26`, `yolov5_bbox_only_post_process.hpp:27` | 앵커 기반 디코드 + NMS | — | RT 같은 파일 |
| YOLOV5SEG | `yolov5_seg_post_process.hpp:23` | + 마스크 | — | RT 같은 파일 |
| YOLOX | `yolox_post_process.hpp:22` | objectness 포함 디코드 + NMS | — | RT 같은 파일. MZ `yolov6n.alls:1`도 yolox 메타를 쓴다 |
| SSD | `ssd_post_process.hpp:23` | 앵커 디코드 + NMS | — | RT 같은 파일 |
| IOU | `hef.proto:317`(`iou_op`) | IoU NMS만 | — | RT |
| SOFTMAX (`logits_layer(..., softmax, ...)`) | `softmax_post_process.hpp:34` | 채널 축 softmax | **NHWC·NC, float32 → float32만** | `softmax_post_process.cpp:59-92`, MZ `generic/efficientnet_lite0.alls:6` |
| ARGMAX (`logits_layer(..., argmax, ...)`) | `argmax_post_process.hpp:37` | 채널 축 argmax | NHCW·NHWC·NC, uint8/uint16 입력 | `argmax_post_process.cpp:48-125` |

### 12.3.1 칩 NMS(`engine=nn_core`)는 호스트 op이 아니다

- 칩 안의 후처리 클러스터에서 도는 NMS다. 출력이 가변 길이 NMS 형식이 된다([`onchip_options.md`](../07_results/candidate_reduction_survey/evidence/onchip_options.md) Q3).
- UG §5.1.5 목록:
  - 칩(neural core): SSD/EfficientDet(디코드·점수·IoU), CenterNet(디코드·점수), YOLOv5(디코드·점수, YOLOv7 호환).
  - CPU: YOLOv5, SSD, YOLOX, **YOLOv8**(NanoDet 호환), YOLOv5SEG.
- **"NMS on neural core is only supported in models that are compiled to single context"**(UG §5.1.5 Note). 멀티 컨텍스트면 동작이 정의되지 않는다.
- MZ v2.19.1에는 `yolov6` + `nn_core` 선례도 있다(`alls/generic/yolov6n_0.2.1_nms_core.alls:1`). UG v3.30 목록보다 넓다. **yolov8 + nn_core는 0건**이다.

### 12.3.2 알려진 함정 — DFL softmax에 max 빼기가 없다

- RT `softmax_post_process.cpp:39-55`는 `exp(x)`를 **최댓값을 빼지 않고** 계산한다. 검사는 `assert(!hasNaN())`뿐이라 release 빌드에선 빠진다.
- 역양자화한 로짓이 ln(FLT_MAX) ≈ 88.7을 넘으면 `inf/inf = NaN` 박스가 나온다(C6, Hailo-10H/HailoRT 5.3.0에서 보고. v4.24.0 소스도 같은 코드다).
- **우리 HEF는 안전하다(계산).** 박스 텐서의 역양자화 최댓값은 `(255 − zp) × scale`이다.
  - P3 `conv71` 19.13, P4 `conv87` 14.45, P5 `conv102` 13.02. 88.7보다 훨씬 작다([`parse_hef_nms.txt`](../07_results/candidate_reduction_survey/evidence/parse_hef_nms.txt)의 `qp_zp`·`qp_scale`).
- 우리 GPU 디코더는 정수 영역에서 max를 먼저 빼고 exp한다([`yolo_decode.comp`](../02_shaders/yolo_decode.comp) 41행 주석).
- 재컴파일로 박스 출력을 16 bit로 바꾸거나 보정 범위가 넓어지면, HailoRT op 경로(`hailo_runner`)에서만 다시 확인해야 한다.

---

## 12.4 칩에서 안 되는 연산과 우회법

| 연산 | 왜 안 되나 | 우회 | 출처 |
|---|---|---|---|
| ONNX `NonMaxSuppression`, TopK, detection-output | 가변 길이·데이터 의존. 파서가 거기까지 가지 못한다 | 앞의 conv/sigmoid에서 자르고 alls `nms_postprocess(...)`로 호스트 op를 붙인다 | MZ-sk `hailo-parse/SKILL.md:88,146`, UG §5.1.5 |
| 동적 Gather·Slice | 정적 분할만 지원 | 호스트에서 | UG §5.5.27-28, [`onchip_options.md`](../07_results/candidate_reduction_survey/evidence/onchip_options.md) Q4 (c) |
| **DFL**(Reshape → Transpose → Softmax → arange Conv) | 커널은 있지만(softmax 경우 ③), 64ch를 4×16으로 나누는 reshape·transpose가 UG의 reshape 세 경우에 맞지 않는다(해석). 파서가 끝 노드를 DFL 안쪽으로 잘못 권하기도 한다 | 관례: DFL 앞의 conv에서 자르고 호스트(HailoRT YOLOV8 op 또는 우리 GPU)에서 한다. 칩으로 옮기려면 ONNX를 손봐 Split(16×4) + 채널 softmax + 1×1 conv로 바꾼다(**선례 없음, 미확인**) | C5(Hailo 직원 Omer: "softmax 뒤 conv는 사실상 ReduceSum"), MZ `networks/yolov11m.yaml:26-34`, [`onchip_options.md`](../07_results/candidate_reduction_survey/evidence/onchip_options.md) Q4 (b) |
| dist2bbox(앵커 격자, stride 곱) | 앵커 생성이 Shape·Range 같은 그래프 연산이다(해석) | 호스트 | MZ `networks/yolov11m.yaml:26-34`(끝 노드가 head conv 6개) |
| 넓은 softmax(예: 400폭) | Hailo-8에 매핑되지 않음 | 호스트에서 | MZ `docs/CHANGELOG.rst:13` |
| 80클래스 ArgMax | 64채널 한도 | 40+40으로 나눠 두 번 + 호스트 비교(컴파일 가능 여부 **미확인**) | UG §5.5.20, C3, [`onchip_options.md`](../07_results/candidate_reduction_survey/evidence/onchip_options.md) Q4 (a2') |
| 채널 축이 아닌 ReduceMax | 채널 축·keepdims만 | transpose로 축을 옮긴다(1.5 MB 제한) | UG §5.5.21, §5.5.34 |
| MHA 밖의 데이터×데이터 MatMul | MHA 블록의 일부로만 | 패턴을 MHA 꼴로 맞춘다. 상수 MatMul은 Dense가 된다 | UG §5.5.31, Table 2 |
| 입력 5개 이상의 Concat | 최대 4입력 | 파서·컴파일러가 자동으로 나누는지는 **미확인** | UG §5.5.5 |
| 8 bit로 미리 양자화된 TFLite | 파서 미지원 | 32/16 bit 모델을 넣는다 | UG §5.1.2 |
| 런타임 BatchNorm 통계 | 미지원 | 추론 모드로 내보낸다(`TrainingMode.PRESERVE`/`EVAL`) | UG §5.5.14, MZ-sk `hailo-parse/SKILL.md:33` |

**일반 우회 규칙.** 지원되지 않는 연산 바로 앞에서 `end_node_names`로 자른다. 잘린 뒷부분은 호스트(CPU/GPU)에서 돌린다. 앞부분만 HAR로 만든다(MZ-sk `hailo-parse/SKILL.md:100`).

---

## 12.5 YOLO 블록별 매핑

### 12.5.1 이 HEF 안에 실제로 있는 칩 레이어

`parse-hef`는 경계 스트림만 보여 준다. 그런데 HEF 최상위 `bytes mapping` 필드(RT `hef.proto:8`)에 컴파일러의 할당 기록이 들어 있다. HailoRT는 이 필드를 해석하지 않는다. 그 안의 문자열에서 칩에 배치된 레이어 이름을 뽑을 수 있다.

```bash
# 파일 읽기만 (장치를 열지 않는다)
strings -n 4 03_models/yolov11m.hef | grep -oE "is not supported in [A-Za-z0-9_]+" \
  | awk '{print $5}' | sort -u
```

- 고유 이름은 647개다. 그중 341개는 `input_to_cluster_*`/`output_from_cluster_*`이고, 나머지도 `concat_f_from_*`, `ws_from_*`, `dw1_defuse_*`, `matmul2_reshape` 같은 할당 부산물이 섞여 있다.
- 아래 표는 이름이 **`종류+번호`로 시작하는** 것의 고유 `종류+번호` 개수다(`conv12_d0`·`conv12`는 `conv12` 하나로 센다). 647개를 전부 나눈 것이 아니라 아래 14개 접두만 센 것이다.

```bash
strings -n 4 03_models/yolov11m.hef | grep -oE "is not supported in [A-Za-z0-9_]+" | awk '{print $5}' | sort -u \
 | grep -oE '^(normalization|conv_feature_splitter|conv|dw|ew_add|concat|maxpool|resize|matmul|reduce_max_softmax|ew_sub_softmax|reduce_sum_softmax|ew_mult_softmax|activation)[0-9]+' \
 | sort -u | sed -E 's/[0-9]+$//' | sort | uniq -c
```
- 문자열 "X is not supported in Y"가 정확히 무엇을 기록한 것인지는 **미확인**이다. 레이어 이름 목록으로만 쓴다.

| 레이어 종류 | 개수 | YOLO11m에서의 자리 (해석) |
|---|---:|---|
| `normalization` | 1 | 입력 `/255` 정규화(MZ `yolov11m.alls:1`) |
| `conv` | 95 | 모든 Conv(+BN 접힘 +SiLU 융합). 출력 conv71/74/87/90/102/105 포함 |
| `dw` | 7 | Detect cv3의 DWConv 3×3 × 3스케일 × 2 = 6, C2PSA의 위치 인코딩(pe) DWConv 1 |
| `conv_feature_splitter` | 10 | C3k2·C2PSA의 `chunk`/`split` |
| `ew_add` | 16 | Bottleneck·PSABlock의 잔차 더하기 |
| `concat` | 22 | C3k2 내부 concat, SPPF concat, head concat |
| `maxpool` | 3 | SPPF의 5×5 s1 MaxPool × 3 |
| `resize` | 2 | head의 Upsample(nearest ×2) × 2 |
| `matmul` | 2 | C2PSA 어텐션의 QKᵀ, attn·V |
| `reduce_max_softmax`, `ew_sub_softmax`, `reduce_sum_softmax`, `ew_mult_softmax` | 각 1 | C2PSA softmax를 max → 빼기(exp 융합으로 보임) → 합 → 역수 곱으로 분해(UG §5.5.18 ③) |
| `activation` | 1 | 융합되지 못한 독립 활성화. 종류 **미확인** |

이 밖의 HEF 사실은 이미 정리돼 있다.
- 컨텍스트 4개, 입력 UINT8 640×640×3(장치 NHCW), 원시 출력 6개 UINT8 NHCW([`parse_hef.txt`](../07_results/candidate_reduction_survey/evidence/parse_hef.txt)).
- 클래스 텐서 `qp_zp=0` → 칩 안 sigmoid. 박스 텐서는 DFL 로짓 그대로다([`parse_hef_nms.txt`](../07_results/candidate_reduction_survey/evidence/parse_hef_nms.txt)).
- NMS는 HEF 메타의 호스트 op `YOLOV8`(score 0.2, IoU 0.7, 클래스당 100)이다. 값은 [`parse_hef_nms.txt`](../07_results/candidate_reduction_survey/evidence/parse_hef_nms.txt), 호스트에서 돈다는 설명은 [04장](04_npu_hailort.md) §4.6 (4).

### 12.5.2 YOLO11 (우리 `yolov11m`)

| 블록 | 구성 연산 | 도는 곳 | 근거 |
|---|---|---|---|
| Conv | Conv2d(3×3 s2 / 1×1) + BN + SiLU | **칩** | UG §5.5.1 Table 9, §5.5.14, Table 3 · HEF `conv` |
| C3k2 | 1×1 Conv → chunk(2) → Bottleneck(C3k) ×n(3×3 Conv ×2 + add) → Concat → 1×1 Conv | **칩** | UG §5.5.27, §5.5.11, §5.5.5 · HEF `conv_feature_splitter`, `ew_add`, `concat` |
| SPPF | 1×1 Conv → MaxPool 5×5 s1 ×3 → Concat(**입력 4개 = 한도**) → 1×1 Conv | **칩** | UG §5.5.2 Table 12(5×5 s1이 최적화 조합), §5.5.5 · HEF `maxpool` 3개 |
| C2PSA | 1×1 Conv → split → PSABlock[Attention(qkv 1×1 Conv → reshape/split → QKᵀ·scale → softmax → ·V → + pe DWConv 3×3 → proj Conv) + FFN + 잔차] → Concat → 1×1 Conv | **칩** | UG §5.5.31-32, §5.5.18 ③ · HEF `matmul` 2개, softmax 분해 4개, `dw` · MZ `alls/generic/yolov11m_seg.alls:8`(`shortcut(matmul1, [reduce_max_softmax1, ...])`) |
| Upsample | nearest ×2 | **칩** | UG §5.5.15 · HEF `resize` 2개 |
| Concat (neck) | 채널 축 | **칩** | UG §5.5.5 · HEF `concat` |
| Detect 박스 가지(cv2) | 3×3 Conv ×2 → 1×1 Conv(64ch) | **칩** | HEF 출력 conv71/87/102 |
| Detect 클래스 가지(cv3) | DWConv 3×3 + 1×1 Conv ×2 → 1×1 Conv(80ch) | **칩** | HEF `dw`, 출력 conv74/90/105 |
| 클래스 sigmoid | Sigmoid | **칩**(alls로 출력 conv에 붙임) | MZ `yolov11m.alls:2-4`, HEF `qp_zp=0` |
| DFL | reshape(4×16) → softmax(16) → arange 1×1 Conv | **호스트**(파서가 자름) | MZ `networks/yolov11m.yaml:26-34`, C5 |
| dist2bbox, stride 곱 | 앵커 격자 빼기·더하기·곱 | **호스트** | 〃 |
| 점수 거르기 + NMS | 클래스별 greedy NMS | **호스트**(HailoRT YOLOV8 op, 이 프로젝트는 GPU) | MZ `yolov11m.alls:5`, RT `yolov8_post_process.hpp:142-144` |

### 12.5.3 YOLOv8 · YOLOv12 · YOLO26

| 모델 | 블록 차이 | 칩/호스트 분할 | 근거 |
|---|---|---|---|
| YOLOv8 | C2f(chunk + Bottleneck×n + Concat). 어텐션 없음. Concat 입력이 2+n개라 큰 n은 4입력 한도와 만난다(처리 방식 **미확인**) | YOLO11과 같다(DFL 이후 호스트) | MZ `base/yolov8.yaml`, UG §5.5.5 |
| YOLOv12 | Area attention(A2C2f) | Hailo-8에서 돈다(`yolov12n`, b1 46.2 / b8 150 FPS). 파서 에러 사례가 있다 | MZ `HAILO8_object_detection.rst:554`, C4 |
| **YOLO26** | DFL 없음(`reg_max=1`, 박스 4ch), one2one head, NMS-free. 백본 블록 수준 차이는 **미확인** | 칩: head conv까지. **호스트: sigmoid와 top-K**(HailoRT op 없음, `hpp: false`). MZ는 출력 6개를 a16으로 둔다 | MZ `base/yolo26.yaml:3-8,29`, `alls/generic/yolo26m.alls:8-9`, `core/postprocessing/detection/yolo.py:171-176,209` |

- 공개 처리량(MZ, Gen3 x4, i5-9400 호스트): yolov11m b1 50.2 / b8 102 FPS, yolo26m b1 48.7 / b8 95.2 FPS
  (MZ `HAILO8_object_detection.rst:350,486`). 우리 Gen3 x1 조건과 직접 비교하지 않는다.

---

## 12.6 이 프로젝트에 주는 함의 (목적 3: NPU 모델 최적화)

| 후보 | 칩에서 되는가 | 기대 효과 | 판단 | 근거 |
|---|---|---|---|---|
| **NMS를 칩으로** | **안 된다.** yolov8 메타는 CPU 목록에만 있고, 칩 NMS는 단일 컨텍스트 모델에서만 된다. 우리 HEF는 컨텍스트 4개다 | — | 호스트(GPU NMS) 유지 | UG §5.1.5(목록과 Note), C2, [`onchip_options.md`](../07_results/candidate_reduction_survey/evidence/onchip_options.md) Q2 |
| **DFL을 칩으로** | 커널은 있다(채널 softmax ③ + Split + 1×1 Conv). ONNX 수정이 필요하고 선례가 없다 | D2H 출력 −39~42%. 그러나 출력은 비어 있는 D2H를 타서 처리량 이득은 **최대 약 2.3%** | 우선순위 낮음 | UG §5.5.18, §5.5.27, [`summary.md`](../07_results/candidate_reduction_survey/summary.md) §2.3 |
| **ArgMax(top-1 클래스)를 칩으로** | 80클래스는 64 한도를 넘는다. "마지막 레이어" 조건도 있다 | 위와 같은 2.3% 천장. 앵커당 클래스가 하나만 남는다(점수 0.2에서는 손실이 작고, 0.02에서 9~15%, 0.003에서 22~25%로 커진다) | 비권장 | UG §5.5.20, C3, [`topk_offline`](../07_results/topk_offline/summary.md) |
| **TopK·점수 거르기를 칩으로** | 안 된다(정적 shape) | — | 불가 | §12.4 |
| **입력 NV12 변환을 칩으로** | 된다(`input_conversion(nv12_to_rgb)`) | 입력 H2D가 1.23 → 0.61 MB/프레임. 그러나 입력이 실리는 컨텍스트 0은 D2H(4.71 MB)가 지배한다. 그래서 전이중 모델 기준 처리량 이득은 ≈ 0이다 **(추정)**. GPU letterbox는 그대로 필요하다 | 이득 작음, 미측정 | UG Changelog v3.22.0, MZ `generic/yolov5m_vehicles_nv12.alls:15`, [`pcie_byte_model_fit.txt`](../07_results/candidate_reduction_survey/evidence/pcie_byte_model_fit.txt) §1 |
| **YOLO26으로 바꾸기** | 칩에서 돈다 | NMS가 사라진다. 대신 출력이 16 bit라 D2H가 늘고, 호스트 top-K가 남는다. 공개 FPS도 yolov11m보다 낮다 | 이득 불확실 | §12.5.3, [`onchip_options.md`](../07_results/candidate_reduction_survey/evidence/onchip_options.md) Q4 참고 행 |
| **C2PSA 어텐션** | 이미 칩에서 돈다 | 호스트로 새는 연산이 없다 | 손댈 필요 없음 | §12.5.1 |

**재컴파일의 전제.**
- Pi에서는 할 수 없다. x86 Ubuntu + DFC 3.34(HailoRT 4.24 짝) + 원본 `yolo11m.onnx`가 필요하다. 보정 이미지 1024장 이상, finetune용 NVIDIA GPU를 권장한다.
- ONNX 내보내기는 `TrainingMode.PRESERVE`(또는 `EVAL`), `do_constant_folding=False`, `dynamo=False`로 한다(MZ-sk `hailo-parse/SKILL.md:29-36`).
- 자세한 조건과 일정 추정은 [`onchip_options.md`](../07_results/candidate_reduction_survey/evidence/onchip_options.md) Q5에 있다.
- 처리량의 지렛대는 연산자 위치가 아니다. config 재적재(배치당 20.5 MB)와 컨텍스트 사이 왕복(b8 프레임당 13.8 MB)이다([`summary.md`](../07_results/candidate_reduction_survey/summary.md) §2.1).
  컨텍스트 수를 줄이는 방향(더 작은 모델, 압축 수준 조정)이 연산자를 옮기는 것보다 큰 변수다(해석).

---

## 12.7 출처

| 약칭 | 대상 | 고정점·날짜 |
|---|---|---|
| UG | Hailo Dataflow Compiler User Guide **v3.30.0**(2025-01). §5.5 "Supported Layers" pp.128~139, §5.1 요약표 pp.60~71, Changelog §2, 16-bit 목록 p.90. 원 배포처는 로그인이 필요한 Hailo 개발자 존이다. 제3자가 올린 공개 사본으로 읽었다: https://www.scribd.com/document/850794798/hailo-dataflow-compiler-v3-30-0-user-guide (텍스트는 embed 보기 `https://www.scribd.com/embeds/850794798/content`). v3.31.0 사본(https://www.scribd.com/document/906243588/Hailo-Dataflow-Compiler-v3-31-0-User-Guide)도 §5.5·§5.1 내용이 같았다 | **DFC 3.34 원문은 미확인.** PDF 텍스트 층이라 합자(fi/fl)가 빠져 있어 그것만 되살려 읽었다 |
| MZ | `github.com/hailo-ai/hailo_model_zoo` | 태그 `v2.19.1`, 커밋 `b5aaad19f7353b96fce06905e34effa28a64d03a` |
| MZ-sk | 같은 저장소 `add-dfc-claude-skills` 브랜치 `.claude/skills/` | 커밋 `ded936fbb0e1035193bb7f030ad5225fb0942982` |
| RT | `github.com/hailo-ai/hailort` (경로는 `hailort/libhailort/` 기준) | 태그 `v4.24.0`, 커밋 `63adffec12da88f1e7cfdc07f2e07681671e83e8` |
| HEF | `03_models/yolov11m.hef` | sha256 `d5716578…`. `parse-hef`·`hailo_platform.HEF`·`strings`로 읽기만 했다 |
| C1~C4 | Hailo 커뮤니티 글 | [`onchip_options.md`](../07_results/candidate_reduction_survey/evidence/onchip_options.md) "출처 표기" 표 |
| C5 | https://community.hailo.ai/t/yolov9-tiny-compilation-fails/2130 | Omer(Hailo 직원 아바타), 2024-08-01. YOLOv8형 head의 끝 노드는 head conv 6개이고 DFL softmax 뒤 conv는 "사실상 ReduceSum"이라는 답 |
| C6 | https://community.hailo.ai/t/yolov8-bbox-decoding-dfl-softmax-overflows-in-float32-no-max-subtraction-producing-nan-boxes/19815 | Simon_Hagerlind, 2026-10-02. HailoRT 5.3.0/Hailo-10H에서 YOLOV8 op의 DFL softmax 오버플로 보고 |

### 12.7.1 미확인 항목 모음

- DFC 3.34의 Supported Layers 원문(3.30/3.31과 다른지).
- DFC 3.34의 ONNX opset 범위(UG 3.30은 8, 11~17).
- Softmax 폭의 정확한 한도(400은 불가, DFL의 16은 미확인).
- 80클래스 ArgMax를 40+40으로 나눠 컴파일할 수 있는지.
- DFL을 칩에 올린 컴파일 선례.
- Concat 입력이 5개 이상일 때의 처리.
- GroupNorm, Sign, 단독 Erf, Tile, Squeeze의 칩 지원.
- Group conv, Depth to Space의 Hailo-8 선례.
- Matmul·Softmax·Argmax의 16 bit 가능 여부.
- HEF `mapping` 문자열 "X is not supported in Y"의 정확한 의미, 독립 `activation1`의 종류.
- Hailo-8과 Hailo-10H의 연산자 커널 차이.

---

## 12.8 자가 점검

**Q1.** 우리 HEF의 NMS를 칩으로 옮길 수 없는 이유 두 가지는?

> yolov8 메타 아키텍처는 칩 NMS 목록에 없다(UG §5.1.5, C2). 그리고 칩 NMS는 단일 컨텍스트 모델에서만 지원되는데, 우리 HEF는 컨텍스트 4개다.

**Q2.** C2PSA의 softmax는 칩에서 도는가? 그 근거는?

> 돈다. HEF `mapping` 안에 `matmul1`·`matmul2`와 `reduce_max_softmax1` 등 softmax 분해 레이어 4개가 있다. UG §5.5.18 ③(픽셀마다 채널 축 softmax를 다른 레이어로 쪼개 구현)과 맞는다.

**Q3.** HailoRT YOLOV8 op의 DFL softmax 오버플로가 우리 HEF에서 생기지 않는 이유는?

> 박스 텐서의 역양자화 최댓값이 19.13(P3)으로 ln(FLT_MAX) ≈ 88.7보다 훨씬 작다. 또 실사용 경로는 HailoRT op이 아니라 max를 먼저 빼는 GPU 디코더다.

**Q4.** ArgMax를 head 끝에 넣어 출력을 줄이려 할 때 걸리는 UG 제약 두 가지는?

> 네트워크의 마지막 레이어여야 하고, 채널이 64개 이하여야 한다. COCO는 80클래스라 그대로는 안 된다.

**Q5.** DFL을 칩으로 옮겨도 처리량이 거의 오르지 않는 이유는?

> 줄어드는 것은 D2H 출력인데, 출력이 나가는 컨텍스트 2·3은 H2D가 지배한다. 그래서 임계 경로에 거의 없고, 이득 천장이 약 2.3%다([`summary.md`](../07_results/candidate_reduction_survey/summary.md) §2.3).
