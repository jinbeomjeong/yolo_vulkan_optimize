# 칩 안 후보 축소 조사 (조사 B: 문헌·소스, NPU 미사용)

- 작성: 2026-10-08, hailo-expert. 장치 측정 없음(HEF 파일 읽기와 소스·문서 조사만).
- 질문: Hailo-8에서 YOLOv11m 출력 후보를 HEF 안에서 줄여, 장치→호스트(D2H) PCIe 출력량
  (현재 1,221,120 B/프레임)과 호스트 NMS 부하를 줄일 수 있는가.
- 함께 보는 파일: [`parse_hef_nms.txt`](parse_hef_nms.txt) (이 HEF의 NMS 메타·원시 스트림 양자화 값).

## 출처 표기

| 약칭 | 대상 | 고정점 |
|---|---|---|
| MZ | `github.com/hailo-ai/hailo_model_zoo` | 태그 `v2.19.1`, 커밋 `b5aaad19f7353b96fce06905e34effa28a64d03a` (DFC v3.34.0 + HailoRT v4.24.0, `README.rst:73`) |
| MZ-skills | 같은 저장소 `add-dfc-claude-skills` 브랜치 | 커밋 `ded936fbb0e1035193bb7f030ad5225fb0942982` |
| RT | `github.com/hailo-ai/hailort` | 태그 `v4.24.0`, 커밋 `63adffec12da88f1e7cfdc07f2e07681671e83e8` (설치된 런타임과 같은 버전) |
| DFC UG | Hailo Dataflow Compiler User Guide §5.1.5 "NMS Post Processing" | 로그인 필요라 **원문은 미확인**. 절 번호는 커뮤니티 글 C1의 인용 |
| C1 | https://community.hailo.ai/t/understanding-nms-and-nms-postprocess/6875 | pierrem, 2024-12-04 |
| C2 | https://community.hailo.ai/t/nms-postprocess/11125 | Marina_Vilela_Bento, 2025-02-10 (직원 표기는 페이지에서 확인 못 함) |
| C3 | https://community.hailo.ai/t/argmax-16bit-output/14220 | Nadav(Hailo 팀), 2025-05-14/15 |
| C4 | https://community.hailo.ai/t/error-parsing-area-attention-module/13158 | DFC 파서 에러 메시지, 2025-03-19 |

## 0. 바이트 계산의 기준

원시 출력 6개(UINT8, NHCW). P5는 장치 쪽 폭이 24로 패딩되어 있어 앵커 수는 6400 + 1600 + 480 = **8,480**
(논리 앵커는 8,400). 아래 모든 바이트는 장치가 실제로 내보내는 패딩 포함 값이다.

| 텐서 | shape (H×W×C) | B/프레임 |
|---|---|---:|
| 클래스 P3/P4/P5 (`conv74/90/105`) | 80×80×80, 40×40×80, 20×24×80 | 678,400 |
| 박스 DFL P3/P4/P5 (`conv71/87/102`) | 80×80×64, 40×40×64, 20×24×64 | 542,720 |
| 합계 | | **1,221,120** |

## Q1. 현재 HEF의 NMS 설정

결론: NMS는 **HEF 메타데이터의 호스트 op**이고, 장치는 원시 6텐서만 내보낸다. 칩 안 NMS는 없다.

- `hailortcli parse-hef --parse-streams --parse-vstreams` (→ `parse_hef_nms.txt`):
  - Stream infos(장치 출력) = 원시 6개 UINT8 NHCW. VStream infos = `yolov8_nms_postprocess` FLOAT32 NMS BY CLASS 1개.
  - `Op YOLOV8`, score 0.200, IoU 0.70, classes 80, max bboxes/class 100, 640×640.
  - vstream → stream 매핑이 원시 6개 전부다(`get_stream_names_from_vstream_name`).
- 클래스 텐서 3개는 `qp_zp=0`, `qp_scale≈0.0035~0.0038` → sigmoid가 **이미 칩 안**에서 적용됐다.
  근거: MZ `hailo_model_zoo/cfg/alls/generic/yolov11m.alls:2-4` `change_output_activation(conv74|conv90|conv105, sigmoid)`.
  박스 텐서는 `qp_zp=74~88`인 DFL 로짓 그대로다.
- 같은 alls `:5` `nms_postprocess("../../postprocess_config/yolov11m_nms_config.json", meta_arch=yolov8, engine=cpu)`.
- 호스트 op의 점수 거르기: RT `hailort/libhailort/src/net_flow/ops/yolov8_post_process.hpp:137-144` —
  모든 앵커 × 80 클래스를 역양자화한 뒤 `class_confidence >= nms_score_th`로 거른다. 즉 거르기는 PCIe를 건넌 **뒤** 호스트에서 일어난다.
- 프로젝트 코드: `01_src/hailo/hailo_runner.cpp:112-125`(기준 경로, NMS BY CLASS를 받고 `set_nms_score_threshold`로 호스트 op 임계값만 바꿈),
  `01_src/hailo/hailo_raw_runner.hpp:1-5`(실사용 경로, 원시 6텐서를 dma-buf로 받아 GPU 후처리). 어느 쪽이든 PCIe 양은 1,221,120 B로 같다
  (`08_learn/04_npu_hailort.md` §(5), `07_results/nms_host_cost/summary.md`).

## Q2. `nms_postprocess`의 engine과 메타 아키텍처

결론: **YOLOv8/YOLOv11(meta_arch=yolov8)은 칩 NMS(engine=nn_core)를 지원하지 않는다.** Hailo-8 칩 NMS 선례는 SSD, YOLOv5, YOLOv6뿐이다.

- C1(pierrem): "for some of the most common models, Hailo offers the possibility to execute the NMS algorithm as part of the
  inference API, either on the host CPU (engine=cpu) or on a dedicated post-processing cluster available on the Hailo-8 (engine=nn_core)."
- C2: "The nms_postprocess is not supported on the neural core for yolov8. By using engine=cpu, it will still be added to the hef,
  but performed on the host cpu." (질문자는 bbox 디코딩·점수 거르기를 칩에서 하고 싶다고 물었다)
- `engine=auto`: DFC UG 원문을 보지 못해 **미확인**. 검색 요약에 "nn_core, cpu or auto"라는 문구가 보이지만 1차 출처를 찾지 못했다.
- MZ v2.19.1 전체 alls의 `nms_postprocess` 집계(`grep -rhoE "nms_postprocess\([^)]*\)" hailo_model_zoo/cfg/alls`):
  - `meta_arch=yolov8, engine=cpu` 31건, `yolov8 ... bbox_decoding_only=True` 1건(역시 cpu). **yolov8 + nn_core는 0건.**
  - nn_core 선례: `ssd`(ssd_mobilenet_v1/v2), `yolov5`(`yolov5xs_wo_spp_nms_core.alls:3`), `yolov6`(`yolov6n_0.2.1_nms_core.alls:1`).
- 칩 NMS의 처리량 비용(MZ `docs/public_models/HAILO8/HAILO8_object_detection.rst`, Gen3 x4, i5-9400 호스트):

  | 모델 | FPS b1 / b8 | 같은 모델 칩 NMS판 |
  |---|---|---|
  | yolov6n_0.2.1 (`:843-847`) | 1256 / 1256 | `_nms_core` (`:860-864`) **237 / 237** |
  | yolov5xs_wo_spp (`:792-796`) | 1139 / 1139 | `_nms_core` (`:809-813`) **100 / -** (alls에 `performance_param(fps=100)`) |

  칩 NMS는 별도 자원을 쓰므로 처리량 비용이 크다. 다만 두 사례 모두 단일 컨텍스트 소형 모델이라 yolov11m에 그대로 옮겨 적용할 수는 없다.
- yolov11m NMS 설정: MZ `hailo_model_zoo/cfg/postprocess_config/yolov11m_nms_config.json`
  `nms_scores_th 0.2`, `nms_iou_th 0.7`, `max_proposals_per_class 100`, `classes 80`, `regression_length 16`,
  `background_removal false`, decoder 3개(stride 8/16/32, reg conv71/87/102, cls conv74/90/105). 우리 HEF의 parse-hef 값과 같다.
  `networks/yolov11m.yaml:4-6` `device_pre_post_layers.nms: true`, `hpp: true`.
- 가설(**미확인**): DFL을 칩에서 끝내면(Q4-b) 헤드가 "ltrb 4ch + sigmoid 클래스" 꼴이 되어 yolov6 메타 아키텍처
  (`nms_core`, `fused_bbox_decoder_*`)와 구조가 비슷해진다. yolov6용 칩 NMS를 yolov11에 붙일 수 있는지는 공개 자료로 확인하지 못했다.

## Q3. 칩 NMS가 가능하다면 출력 크기

결론: 가능하다면 D2H는 **0.6~65 KB**로 줄어 1.22 MB 대비 95~99.9% 감소한다. 다만 Q2에 따라 yolov8 메타로는 불가능하다.

- 장치 bbox 형식: 8 B(uint64). RT `libhailort/src/net_flow/ops/nms_post_process.hpp:147-153` — 좌표 4개 × 12 bit + 점수 16 bit.
  클래스 구분자 `NMS_DELIMITER = 0xFFFFFFFFFFFFFFFF` (`stream_common/nms_stream.hpp:23`).
- Hailo-8의 읽기 모드 두 가지(RT `stream_common/nms_stream.cpp:9-26`, `:160-198`):
  - bbox 모드: 구분자가 나올 때까지 8 B씩 읽는다 → 프레임당 `(검출 수 N + 80) × 8 B`. 가변 길이다.
    N=0이면 640 B, N=20이면 800 B, N=100이면 1,440 B, 최악(80×100)이면 64,640 B.
    대신 8 B마다 `read_impl`을 한 번 부른다 → 작은 전송이 수백 번 일어나 전송당 오버헤드가 생길 위험이 있다(소스에서 추론, 측정 안 함).
  - burst 모드(H8_PER_CLASS): 클래스마다 burst 하나 → `80 × burst_size × 8 B`. burst_size ≥ 101이면 ≈ 64,640 B로 고정된다.
- 상한 공식: RT `libhailort/include/hailo/hailort_common.hpp:396-404` `get_nms_hw_frame_size` =
  `chunks × classes × (2 + bbox_size × max(burst, max_bboxes_per_class)) + bbox_size × burst`.
- 호스트 쪽 NMS BY CLASS float 형식은 `80 × 4 + N × 20 B`, 최대 160,320 B(parse-hef 값과 같다). 이 크기는 PCIe 전송량이 아니다.

## Q4. NMS가 아닌 칩 안 연산으로 출력 줄이기

| 방법 | D2H B/프레임 (현재 1,221,120) | 감소 | 의미 변화 | Hailo-8 지원 근거 |
|---|---:|---:|---|---|
| (a1) 클래스 reduce_max만 (1ch) | 551,200 | −669,920 | 클래스 id를 잃는다 → 단독으로는 못 쓴다 | C4: "Reduce max is only supported on the features axis, and with keepdim=True" → 채널축 max는 가능 |
| (a2) max + argmax (2ch) | 559,680 | −661,440 (−54.2%) | 앵커당 top-1 클래스만 남는다. 현재 디코더·HailoRT op는 임계값을 넘는 **모든 클래스**를 내보낸다(`yolov8_post_process.hpp:140-144`) | argmax 칩 선례: MZ `networks/fcn8_resnet_v1_18.yaml:10` 끝 노드 `ArgMax_82`(19클래스), `deeplab_v3_mobilenet_v2_wo_dilation.yaml:13,28`. **단 C3(Nadav): "the current implementation of Argmax supports up to 64 features"** → 80클래스는 한도를 넘는다 |
| (a2') 40+40으로 나눈 max+argmax (4ch) | 576,640 | −644,480 (−52.8%) | 위와 같고, 두 쌍 중 큰 쪽 고르기는 호스트에서 한다 | 64 한도를 피하는 우회. 컴파일 가능 여부는 **미확인** |
| (b) DFL softmax+적분을 칩에서 (64→4ch, UINT8) | 712,320 | −508,800 (−41.7%) | 의미는 같다. 박스 정밀도가 UINT8 눈금(0~16 stride를 256단계)으로 떨어진다: P3 ≈0.5 px, P5 ≈2 px 단위 | softmax 칩 선례: MZ `networks/resnet_v1_50.yaml:15` 끝 노드 `predictions/Softmax`. YOLO11 백본 C2PSA 어텐션 softmax가 칩에서 돈다(`alls/generic/yolov11m_seg.alls:8` `reduce_max_softmax1`). 폭 한도: MZ `docs/CHANGELOG.rst` v2.19.1 STARK 항목 "the 400-wide softmax it needs cannot be mapped on Hailo-8" → DFL의 16폭은 한도 안일 가능성이 높다(**미확인**). 단 DFL은 64ch를 4그룹×16으로 나눈 softmax라서, ONNX를 손봐(박스 conv를 16ch 4개로 쪼개고 softmax 뒤에 arange 1×1 conv) 채널축 softmax로 바꿔야 한다 |
| (b') 같은 것, 출력 16 bit | 746,240 | −474,880 (−38.9%) | 정밀도 문제가 해소된다 | MZ `alls/generic/yolo26m.alls`가 출력층을 `precision_mode=a16_w16`으로 둔 선례가 있다 |
| (a2)+(b) | **50,880** | −1,170,240 (−95.8%) | top-1 클래스 + 박스 UINT8 | 위 두 근거를 합친 것. 조합 컴파일 선례는 없다 |
| (a2')+(b') | 101,760 | −1,119,360 (−91.7%) | top-1 클래스 + 박스 16 bit | 〃 |
| (c) top-K / gather / 동적 길이 | — | 불가 | — | 장치 스트림은 정적 shape(`hw_frame_size`)이고, 가변 길이 출력은 NMS 코어의 구분자 프로토콜뿐이다(Q3). MZ-skills `.claude/skills/hailo-parse/SKILL.md:146` "Tries to parse through NMS / TopK / detection-output → Cut at the conv/sigmoid outputs". 칩 안 임계값 선례인 `centernet_resnet_v1_18_postprocess`(`networks/...yaml:4-5,21` max_finder + threshold ReLU)도 출력은 128×128×80 밀집 그대로다 → 0을 채울 뿐 바이트는 줄지 않는다 |
| (d) 출력 양자화 더 줄이기 | — | 0 | — | 이미 UINT8이다. Hailo-8 활성값 정밀도는 8/16 bit뿐이다(MZ alls 전체 `precision_mode` 값이 `a8_w4`, `a8_w8`, `a16_w16` 세 가지뿐. 4 bit는 가중치에만 쓴다) |
| 참고: YOLO26m(NMS-free, reg_max=1) | ≈1,424,640 (추정) | +203,520 | NMS가 필요 없다. 박스 4ch | MZ `base/yolo26.yaml` 출력 80×80×80 …, 80×80×4 …. `alls/generic/yolo26m.alls`가 출력 6개를 a16으로 둔다 → 클래스가 2배가 되어 PCIe는 오히려 는다(20×24 패딩 가정) |

- 클래스 텐서는 매우 희소하다: 0.2 임계값이 UINT8 q≈52~58에 해당한다(`07_results/gpu_postprocess/summary.md:46`). 그래도 정적 shape이라서 희소성을 PCIe 절감으로 바꿀 칩 안 수단은 없다.
- (a)·(b)를 하면 우리 GPU 디코더(`yolo_decode.comp`)와 `01_src/ref/` 기준 구현, `PostprocFixture` 픽스처도 새 텐서 형식에 맞춰 다시 만들어야 한다.

## Q5. 현실성

- 재컴파일 환경: x86_64 Ubuntu 22.04/24.04, Python 3.10~3.12, DFC **v3.34.0**(MZ v2.19.x, `README.rst:73`에 "Dataflow Compiler v3.34.0 and HailoRT v4.24.0"
  → 우리 런타임 4.24.0과 짝이 맞는다). NVIDIA GPU 권장(MZ `docs/GETTING_STARTED.rst:9-18`. 이 파일은 DFC 3.33/HailoRT 4.23으로 적혀 있어 README와 어긋난다).
  DFC는 Hailo 개발자 존 계정이 있어야 받을 수 있다.
- 원본: MZ `networks/yolov11m.yaml:11,13` `yolo11m.onnx`(S3 `.../yolov11m/2024-10-02/yolo11m.zip`). 끝 노드는 `:15-22`의 head conv 6개다.
  (a)·(b)를 하려면 ONNX를 손봐 끝 노드를 바꿔야 한다.
- 보정 데이터: MZ `docs/OPTIMIZATION.rst:49` "recommended to run this step on a GPU machine with dataset size of at least 1024 images".
  yolov11m.alls `:6-7`은 calibration batch 2, `post_quantization_optimization(finetune, ...)`를 쓴다 → finetune은 GPU가 사실상 필수다.
- 우리 HEF의 출처 버전(어느 MZ/DFC로 컴파일했는지)은 **미확인**이다(sha256 `d5716578…`). 재컴파일하면 정확도 기준선이 바뀌므로, 같은 alls로 만든 대조 HEF를 함께 만들어 비교해야 한다.
- 정확도 위험: (a2) top-1은 다중 클래스 emit을 잃는다(COCO mAP 영향은 **미측정**). (b)는 칩 안 8 bit softmax와 출력 양자화 오차가 박스에 그대로 들어간다 → 16 bit 출력을 권한다.
  추가 층이 컨텍스트 분할을 바꾸면(4 → 5) 가중치 재적재가 늘어 절감분을 상쇄할 위험이 있다(`resources_param` 0.9 사용률로 이미 꽉 찬 모델이다).
- 작업량(추정): 환경 구축 1~2일, ONNX 수술 + 파싱 1~2일, optimize(finetune)와 compile에 GPU로 반나절~하루, COCO 평가 1일,
  Pi 쪽 디코더·기준 구현·픽스처·테스트 교체 3~5일 → **총 1.5~3주**.

## Q6. 공개 벤치 (Model Zoo yolov11m)

- MZ `docs/public_models/HAILO8/HAILO8_object_detection.rst:486-494`: yolov11m float mAP 51.1, HW mAP 49.9, **FPS b1 50.2 / b8 102**, 640×640×3, 20.1M params, 68.1 GOPs.
  측정 조건(`:3-6`): 호스트 Intel i5-9400, PCIe **Gen3 x4**, 상온. 표 머리의 "Hailo Dataflow Compiler Version v2.19.1"은 MZ 버전 표기다.
- 측정 도구는 `hailortcli benchmark`다(MZ `docs/BENCHMARKS.rst:4-35`). 표가 hw_only 값인지 streaming 값인지는 **적혀 있지 않다(미확인)**.
  - hw_only는 원시 스트림만 쓰고 호스트 NMS op를 돌리지 않는다: RT `hailort/hailortcli/benchmark_command.cpp:122-129`, `run_command.cpp:265-266,879-890`.
  - streaming은 vstream(호스트 NMS 포함)을 쓴다(`benchmark_command.cpp:131-139`).
  - 우리 Pi에서는 hw_only와 streaming FPS가 같았다(`07_results/npu_benchmark/summary.md` 결론). 따라서 이 모델에서는 NMS 포함 여부가 FPS 비교를 바꾸지 않는다.

## 처리량 효과의 상한 (추정)

- PCIe는 전이중(full-duplex)이다. 출력은 D2H로 가고, 가중치(컨텍스트 전환)와 입력은 H2D로 간다. 출력을 줄여도 H2D 쪽 병목은 그대로다.
- 멀티 컨텍스트 HEF는 컨텍스트 사이 경계 텐서(`LayerType::INTER_CONTEXT`, RT `hef/layer_info.hpp:37`)를 호스트 메모리에 두므로, D2H에는 출력 말고도 이 경계 텐서가 실린다.
  동시에 진행 중인 조사 A의 로그(`hailort_run_b1.log`, `hailort_run_b8.log`, **정본 아님**)를 보면 내부 버퍼 Pinned가 b1 7.84 MB, b8 47.98 MB다
  → 출력 1.22 MB는 D2H의 일부(대략 15~20%, 추정)다.
- `npu_benchmark` Gen3 분해(배치 고정 ~25 ms + 프레임당 16.2 ms)에 D2H가 임계 경로라고 가정하고, Gen3 x1 실효 ~0.8 GB/s를 넣으면
  1.17 MB를 줄일 때 프레임당 약 1.5 ms가 준다 → b8 51.7 FPS가 **최대 약 56 FPS(+8%)**가 된다. 출력 DMA가 연산과 겹친다면 이득은 0에 가깝다.
  어느 쪽인지는 컨텍스트별 시간 분해(`hailortcli run2 ... measure-fw-actions`)로 마지막 컨텍스트의 D2H 비중을 봐야 알 수 있다.

## 권장

1. **칩 NMS(engine=nn_core)는 추진하지 않는다.** yolov8 메타는 지원되지 않고(C2, MZ 31/31 cpu), 지원되는 메타에서도 처리량 비용이 크다(yolov6n 1256 → 237 FPS).
2. PCIe 출력 축소를 정말 노린다면 1순위는 (b') **DFL을 칩에서 끝내고 16 bit로 출력**하는 것이다(−475 KB, 의미 불변).
   2순위는 (a2') 분할 max/argmax(−644 KB)인데, top-1으로 의미가 바뀌고 argmax 64 한도를 우회해야 한다.
3. 다만 재컴파일(x86 + DFC 3.34 + GPU, 1.5~3주)에 앞서, 먼저 **D2H 출력이 임계 경로에 있는지**부터 확인한다(조사 A의 fw-actions 분해).
   이득 상한이 +8% 수준이고, 호스트 NMS는 이미 GPU 경로로 옮겨 CPU 부담이 작다(`07_results/gpu_nms_e2e/`).
   확인되지 않으면 우선순위는 낮다.
