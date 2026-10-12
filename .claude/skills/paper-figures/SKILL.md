---
name: paper-figures
description: >
  논문(10_paper/paper_ko.md)에 넣을 그림·표·다이어그램을 만들거나 고치는 절차.
  07_results 정본에서만 수치를 가져오는 출처 규칙, 측정 조건(구/신·표시 on/off) 분리, 캡션 형식,
  10_paper/figures 배치·등록부, matplotlib 공용 스타일, 재생성·검증 순서, yEd 다이어그램 내보내기를 담는다.
  트리거: "논문 그림", "그래프 그려", "그림 만들어", "표 만들어", "도표", "캡션", "figure", "plot", "chart".
---

# 논문 그림·표 절차

그림은 표보다 더 오래, 더 멀리 인용된다. 출처나 조건이 빠진 그림은 틀린 결론을 조용히 퍼뜨린다.
배치 규칙의 정본은 [`STRUCTURE.md`](../../../STRUCTURE.md) §2.7이고, 등록부는 [`10_paper/figures/README.md`](../../../10_paper/figures/README.md)다.

## 0. 출처 규칙

- 수치는 `07_results/**`의 **CSV·JSON만** 읽는다(분석 스크립트가 만든 고정 형식 텍스트 — 예: `thermal_soak/evidence/soak_*_events.txt` 머리줄 — 는 정규식으로 읽고 교차 확인하면 허용). 스크립트에 `summary.md`의 숫자를 박아 넣지 않는다.
  원본 CSV가 없어 어쩔 수 없을 때만 상수 옆에 `# source: <path>:<line>`을 적고, 등록부 행에 "수기"라고 표시한다.
- 그리기 전에 원본 실험의 `summary.md`에서 **"주의·남은 일"**을 먼저 읽는다. "신뢰 불가"로 적힌 행은 뺀다.
- **인용 금지**(`07_results/README.md` "⛔ 인용 금지 구간")는 그림에도 쓰지 않는다.
  - 백프레셔 결함을 고치기 전의 `gpu_postproc_e2e/` FPS·지연. 결함을 설명하는 맥락의 "24.48 → 34.96"만 허용한다.
  - `--display-pace asap` + 표시 on 조건의 GPU timestamp **mean**. 이 조건에서는 중앙값만 쓴다.
  - `hef_context_profile`의 JSON `fps`. stdout 값을 쓴다.

## 1. 조건 규칙

- 측정 조건이 다른 값은 **한 축에 섞지 않는다.**
  - **구**: V3D 960 MHz · 디코더 자동 스레드 · CPU NMS
  - **신**: 1150 MHz · 디코더 1스레드 · GPU NMS
  - 이 밖에도 표시 on/off, PCIe 세대, 페이싱(asap/source)이 다르면 서로 다른 조건이다.
  - 섞어야 한다면 범례와 캡션에 조건을 나눠 적는다.
- 조건은 그 실험의 `env_bench.txt`와 `summary.md` §1에서 **읽는다.** 기억이나 다른 문서에서 옮겨 적지 않는다.
- 캡션 끝에는 `(측정: 날짜, 조건, 반복 n, 원본: 07_results/<경로>)`를 붙인다.

## 2. 위치·이름

- 스크립트: `06_tools/python/paper/figNN_<slug>.py` — 그림 하나에 스크립트 하나. 공용 스타일은 `paper_style.py`, 전체 재생성은 `build_figures.py`.
- 산출물: `10_paper/figures/figNN_<slug>.svg`(본문용)와 `.png`(300 dpi). Write 훅과 검사기가 다른 이름을 막는다.
- 번호는 논문에 나오는 순서대로 매긴다. 번호가 바뀌면 파일 이름, 등록부, 본문 참조를 **같은 커밋에서** 함께 고친다.
- 등록부 행: `| 번호 | 파일 | 생성 스크립트 | 원본 | 측정 조건 | 논문 절 |`

## 3. 스타일 (`paper_style.py`)

- matplotlib `Agg` 백엔드를 쓰고 conda 환경 `hailo-yolo`에서 돌린다. 시스템 python3에는 matplotlib이 없다.
- 글꼴은 NanumSquare를 쓰고 없으면 UnDotum으로 대신한다. `svg.fonttype='none'`, `pdf.fonttype=42`.
- 크기: 단일 열 8.5 cm, 두 열 17.5 cm. 글자는 8–9 pt.
- 색: 처리 장치 색은 [`09_diagrams/README.md`](../../../09_diagrams/README.md)의 색 규약(GPU `#99CC00`, NPU `#FFCC00`, CPU `#E8EEF7`, 디코더 `#B39DDB`, 표시 `#4DB6E2`)을 따른다. 그 밖의 계열은 색맹에 안전한 팔레트로 칠한다.
- 축 이름에는 단위를 적는다(`지연 (ms)`). 반복 측정은 **중앙값과 min–max 범위**로 그린다(measure 스킬의 집계와 같다).
- 그림 안 숫자 라벨은 `paper_style.fmt_half_up`(사사오입)으로 찍는다 — summary.md 의 반올림과 맞춘다(float `.1f` 는 102.25 → 102.2).
- 재현성을 위해 `svg.hashsalt`를 고정하고 메타데이터 날짜를 비운다(`save()`가 clip-path id 도 정규화한다). 같은 입력이면 SVG가 바이트 단위로 같아야 한다.

## 4. 절차

1. 후보와 원본 파일·열을 확정한다(§0·§1).
2. `python-expert`에 스크립트를 맡긴다. 전역 에이전트라 이 스킬을 읽지 않으므로, §0~§3을 위임 프롬프트에 적어 준다.
3. 렌더는 `/home/pi/miniconda3/envs/hailo-yolo/bin/python 06_tools/python/paper/build_figures.py`로 한다. 하드웨어는 쓰지 않는다. 실행은 메인 에이전트가 한다(`10_paper` 담당).
4. 메인 에이전트가 PNG를 Read 도구로 **직접 보고** 글자 겹침, 단위, 범례, 조건 표기를 확인한다.
5. `integrity-auditor`가 그림의 핵심 수치를 `summary.md`와 본문에 대조한다.
6. 사용자 확인을 받은 뒤 메인 에이전트가 본문에 `![그림 N. 제목](figures/figNN_<slug>.svg)`과 캡션을 넣는다.
7. `check_structure.py`와 `check_links.py`가 모두 0건인지 확인하고 커밋한다. 검사기 **D3**가 그림 파일 ↔ 등록부 행, 행의 생성 스크립트·원본 경로 존재를 대조한다.

## 5. 표

- 논문의 표는 본문 Markdown에 둔다. 표 위에 `**표 N.** 제목`을 달고, 표 아래에 `출처: 07_results/<경로>`를 한 줄 적는다.
- 수치를 고칠 때는 `summary.md`와 대조한다. 그림과 표가 같은 값을 보여 주면 두 곳이 같은 원본을 가리켜야 한다.

## 6. 다이어그램

- 원본은 `09_diagrams/*.graphml`이다. 논문용으로 줄인 판은 `09_diagrams/pipeline_paper.graphml`로 `graphml-expert`가 만든다. 원본 2장의 색 규약과 ⓒ/ⓝ 수치 표기를 따른다.
- yEd에는 헤드리스 내보내기가 없다. **사용자가 자리에 있을 때** yEd GUI(File ▸ Export)로 SVG/PNG를 내보내 `10_paper/figures/`에 둔다. 여는 방법은 `09_diagrams/README.md`에 있다.
- 다이어그램에 적힌 수치를 고치는 일은 그 측정 영역의 전문가가 맡는다.

## 7. 재측정 뒤

`build_figures.py`를 다시 돌리고, 바뀐 그림의 캡션 조건과 등록부를 갱신한다. 같은 그림에 이전 회차와 새 회차를 함께 그릴 때는 §1의 조건 분리를 따른다.
