// 발표 덱 생성 스크립트: NPU 기반 YOLO 추론에서 호스트 CPU 최소화 (13장, 16:9)
// 실행: NODE_PATH=<python-dev>/lib/node_modules <python-dev>/bin/node build_deck.js
// 내용 정본: ../content_spec.md (수치는 명세에 적힌 것만 쓴다)
// 입력·출력 경로는 이 파일 위치 기준으로 고정한다. 날짜·난수를 쓰지 않는다.
'use strict';
const path = require('path');
const fs = require('fs');
const pptxgen = require('pptxgenjs');
const React = require('react');
const ReactDOMServer = require('react-dom/server');
const sharp = require('sharp');
const lu = require('react-icons/lu');
const bs = require('react-icons/bs');

const HERE = __dirname;
const DECK = path.resolve(HERE, '..');
const FIG = path.join(DECK, '..', 'figures');  // 10_paper/figures (원본을 직접 참조)
const OUT = path.join(HERE, 'yolo_npu_cpu_offload_talk.pptx');

const SK = fs.readdirSync(path.join(process.env.HOME, '.claude/skills/synced'))
  .map((d) => path.join(process.env.HOME, '.claude/skills/synced', d, 'pptx/scripts'))
  .find((p) => fs.existsSync(path.join(p, 'apply_theme.js')));
if (!SK) throw new Error('pptx 스킬의 apply_theme.js를 찾지 못했다');
const { applyTheme } = require(path.join(SK, 'apply_theme.js'));
const JSZip = require(require.resolve('jszip', { paths: [require.resolve('pptxgenjs')] }));

// ---------------------------------------------------------------- 테마
const HEX = {
  dk1: '1E2A36', lt1: 'FFFFFF', dk2: '2B4157', lt2: 'EEF2F6',
  accent1: 'C8641E', // 유일한 강조색 (NPU·결과)
  accent2: '5A6E83', // CPU·보조 (회청색)
  accent3: '9AAABB', accent4: 'D6DEE6', accent5: '44586D', accent6: 'F0B27A',
  hlink: '2B6CB0', folHlink: '5A6E83',
};
const THEME = { name: 'Host CPU Offload', headFontFace: 'Arial', bodyFontFace: 'Arial', colors: HEX };
const EA_FONT = '맑은 고딕';
const LANG = 'ko-KR';
const FOOTER = 'NPU 기반 YOLO 추론에서 호스트 CPU 최소화';

const pres = new pptxgen();
pres.layout = 'LAYOUT_WIDE'; // 13.333 x 7.5 in (16:9)
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = 'NPU 기반 YOLO 추론에서 호스트 CPU 최소화';
pres.subject = '최저 지연 운용을 기준으로';
pres.author = '발표자 (작성 필요)';
const C = pres.SchemeColor;
const W = 13.333, M = 0.6, CW = W - 2 * M; // 폭, 여백, 본문 폭

// ---------------------------------------------------------------- 레이아웃
pres.defineSlideMaster({
  title: 'TITLE_DARK',
  background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: 'title', type: 'title', x: 0.8, y: 1.7, w: 11.7, h: 1.4, fontSize: 40, bold: true, color: C.background1, align: 'left', valign: 'bottom', margin: 0 }, text: '' } },
    { placeholder: { options: { name: 'subtitle', type: 'body', x: 0.8, y: 3.25, w: 11.7, h: 0.6, fontSize: 24, color: C.background2, align: 'left', valign: 'top', margin: 0 }, text: '' } },
    { placeholder: { options: { name: 'entitle', type: 'body', x: 0.8, y: 3.95, w: 11.7, h: 0.45, fontSize: 16, color: C.accent3, align: 'left', valign: 'top', margin: 0 }, text: '' } },
    { placeholder: { options: { name: 'presenter', type: 'body', x: 0.8, y: 5.6, w: 6.5, h: 0.5, fontSize: 18, color: C.background1, align: 'left', valign: 'middle', margin: 0 }, text: '' } },
  ],
});
pres.defineSlideMaster({
  title: 'CONTENT',
  background: { color: C.background1 },
  objects: [
    { placeholder: { options: { name: 'title', type: 'title', x: M, y: 0.35, w: CW, h: 0.85, fontSize: 28, bold: true, color: C.text1, align: 'left', valign: 'middle', margin: 0 }, text: '' } },
    { text: { text: FOOTER, options: { x: M, y: 7.0, w: 8, h: 0.3, fontSize: 10, color: C.accent2, margin: 0, lang: LANG } } },
  ],
  slideNumber: { x: W - M - 0.6, y: 7.0, w: 0.6, h: 0.3, fontSize: 10, color: C.accent2, align: 'right', margin: 0 },
});
pres.defineSlideMaster({
  title: 'CONCLUSION_DARK',
  background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: 'title', type: 'title', x: M, y: 0.35, w: CW, h: 0.85, fontSize: 28, bold: true, color: C.background1, align: 'left', valign: 'middle', margin: 0 }, text: '' } },
  ],
  slideNumber: { x: W - M - 0.6, y: 7.0, w: 0.6, h: 0.3, fontSize: 10, color: C.accent3, align: 'right', margin: 0 },
});

pres.defineSlideMaster({
  title: 'CLOSING_DARK',
  background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: 'title', type: 'title', x: 0.8, y: 1.1, w: 11.7, h: 1.2, fontSize: 48, bold: true, color: C.background1, align: 'left', valign: 'bottom', margin: 0 }, text: '' } },
    { placeholder: { options: { name: 'subtitle', type: 'body', x: 0.8, y: 2.45, w: 11.7, h: 0.6, fontSize: 26, color: C.background2, align: 'left', valign: 'top', margin: 0 }, text: '' } },
    { placeholder: { options: { name: 'presenter', type: 'body', x: 0.8, y: 3.4, w: 6.5, h: 0.5, fontSize: 18, color: C.background1, align: 'left', valign: 'middle', margin: 0 }, text: '' } },
  ],
});

// ---------------------------------------------------------------- 도우미
// 숫자와 단위 사이 공백을 줄바꿈 없는 공백으로 바꾼다 (예: "42 ms")
const nb = (t) => t.replace(/(\d) (ms|MHz)/g, '$1\u00A0$2');
function tx(slide, content, opts) {
  if (typeof content === 'string') content = nb(content);
  else content = content.map((r) => Object.assign({}, r, { text: nb(r.text) }));
  slide.addText(content, Object.assign({ isTextBox: true, lang: LANG, color: C.text1, fontSize: 16, margin: 0, valign: 'top' }, opts));
}
function bullets(items, base) {
  // items: 문자열 또는 run 배열. 각 항목은 한 문단.
  const runs = [];
  items.forEach((it, i) => {
    const last = i === items.length - 1;
    const parts = typeof it === 'string' ? [{ text: it }] : it;
    parts.forEach((p, j) => {
      const o = Object.assign({}, base || {}, p.options || {});
      if (j === 0) { o.bullet = (base && base.bullet) || { indent: 18 }; o.paraSpaceAfter = (base && base.paraSpaceAfter) || 8; }
      if (j === parts.length - 1 && !last) o.breakLine = true;
      runs.push({ text: p.text, options: o });
    });
  });
  return runs;
}
async function icon(Comp, hex, px = 256) {
  const svg = ReactDOMServer.renderToStaticMarkup(React.createElement(Comp, { size: px, color: '#' + hex }));
  const buf = await sharp(Buffer.from(svg)).resize(px, px).png().toBuffer();
  return 'image/png;base64,' + buf.toString('base64');
}
function card(slide, name, x, y, w, h, fill) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: fill || C.background2 }, line: { type: 'none' }, objectName: name });
}
function arrow(slide, name, x1, y1, x2, y2, color, dash) {
  slide.addShape(pres.shapes.LINE, {
    x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1), h: Math.abs(y2 - y1),
    flipV: y2 < y1, line: { color: color || HEX.accent3, width: 2, endArrowType: 'triangle', dashType: dash || 'solid' }, objectName: name,
  });
}
function footnote(slide, text, y, color) {
  tx(slide, text, { x: M, y, w: CW, h: 0.3, fontSize: 10, color: color || C.accent2, objectName: '각주' });
}
const BAR_CHART_BASE = () => ({
  barDir: 'col', lang: LANG,
  showTitle: true, titleFontSize: 14, titleColor: C.text1, titleFontFace: '+mn-lt',
  showValue: true, dataLabelPosition: 'outEnd', dataLabelFontSize: 16, dataLabelFontBold: true, dataLabelColor: C.text1, dataLabelFontFace: '+mn-lt',
  catAxisLabelFontSize: 14, catAxisLabelColor: C.text1, catAxisLabelFontFace: '+mn-lt', catAxisLineShow: true,
  valAxisLabelFontSize: 12, valAxisLabelColor: C.accent2, valAxisLabelFontFace: '+mn-lt', valAxisMinVal: 0, valAxisLineShow: false,
  valGridLine: { color: HEX.accent4, size: 0.75 }, catGridLine: { style: 'none' },
  showLegend: false, barGapWidthPct: 70,
  chartColors: [HEX.accent3, HEX.accent1], // 이전 = 회색, 이후 = 강조색
});

// ---------------------------------------------------------------- 슬라이드
async function build() {
  const I = {
    film: await icon(lu.LuFilm, HEX.lt1), gpu: await icon(bs.BsGpuCard, HEX.lt1), npu: await icon(lu.LuBrainCircuit, HEX.lt1),
    scan: await icon(lu.LuScanSearch, HEX.lt1), monitor: await icon(lu.LuMonitor, HEX.lt1), shuffle: await icon(lu.LuShuffle, HEX.lt1),
    filmD: await icon(lu.LuFilm, HEX.accent5), board: await icon(lu.LuCircuitBoard, HEX.accent5), chip: await icon(lu.LuMicrochip, HEX.accent5),
    model: await icon(lu.LuBoxes, HEX.accent5), timerL: await icon(lu.LuTimer, HEX.accent6), info: await icon(lu.LuInfo, HEX.accent5),
    send: await icon(lu.LuSend, HEX.accent1), key: await icon(lu.LuKeyRound, HEX.accent6), link: await icon(lu.LuArrowRightLeft, HEX.accent6),
    clock: await icon(lu.LuClock, HEX.accent6), x: await icon(lu.LuCircleX, HEX.accent2), layers: await icon(lu.LuLayers, HEX.lt1),
    cpuD: await icon(lu.LuCpu, HEX.lt1), poll: await icon(lu.LuRefreshCw, HEX.accent2), bell: await icon(lu.LuBellRing, HEX.accent1), check: await icon(lu.LuListChecks, HEX.lt1),
  };

  // 1. 제목 -------------------------------------------------------------
  pres.addSection({ title: '도입' });
  let s = pres.addSlide({ masterName: 'TITLE_DARK', sectionTitle: '도입' });
  s.addText('NPU 기반 YOLO 추론에서 호스트 CPU 최소화', { placeholder: 'title', lang: LANG });
  s.addText('최저 지연 운용을 기준으로', { placeholder: 'subtitle', lang: LANG });
  s.addText('Minimizing Host CPU Load in NPU-Based YOLO Inference', { placeholder: 'entitle', lang: 'en-US' });
  s.addText('발표자 (작성 필요)', { placeholder: 'presenter', lang: LANG });
  // 장식 대신 파이프라인 아이콘 4개 (디코드 → GPU → NPU → 표시)
  [I.film, I.gpu, I.npu, I.monitor].forEach((ic, i) => {
    const x = 8.55 + i * 1.05;
    s.addShape(pres.shapes.OVAL, { x, y: 5.45, w: 0.8, h: 0.8, fill: { color: i === 2 ? C.accent1 : C.accent5 }, line: { type: 'none' }, objectName: `흐름 아이콘 원 ${i + 1}` });
    s.addImage({ data: ic, x: x + 0.2, y: 5.65, w: 0.4, h: 0.4, altText: ['디코드', 'GPU', 'NPU', '표시'][i], objectName: `흐름 아이콘 ${i + 1}` });
  });
  s.addNotes('오늘은 NPU가 추론을 맡은 뒤에도 CPU에 남는 일을 어떻게 걷어냈는지, 지연이 가장 짧은 운용 방식을 기준으로 말씀드린다.');

  // 2. 목차 (신규) --------------------------------------------------------
  s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: '도입' });
  s.addText('목차', { placeholder: 'title', lang: LANG });
  const toc = [
    ['문제와 대상 시스템', '추론만 NPU로 옮겨도 CPU는 바쁘다 · 대상 시스템과 운용 기준 · 전체 구조', '3~5'],
    ['CPU 최소화 기술 ①~⑤', '① 하드웨어 디코더 · ② NHCW 직접 생성 · ③ 결과 해석 · ④ 최저 지연 운용 · ⑤ 지휘만', '6~10'],
    ['결과', '결과 요약', '11'],
    ['결론', 'CPU가 데이터를 읽지 않게 하는 것', '12'],
  ];
  toc.forEach(([head, sub, range], i) => {
    const y = 1.5 + i * 1.25;
    card(s, `목차 카드 ${i + 1}`, M, y, CW, 1.0, C.background2);
    s.addShape(pres.shapes.OVAL, { x: M + 0.3, y: y + 0.15, w: 0.7, h: 0.7, fill: { color: C.text2 }, line: { type: 'none' }, objectName: `목차 번호 원 ${i + 1}` });
    tx(s, String(i + 1), { x: M + 0.3, y: y + 0.15, w: 0.7, h: 0.7, fontSize: 22, bold: true, color: C.background1, align: 'center', valign: 'middle', objectName: `목차 번호 ${i + 1}` });
    tx(s, [
      { text: head, options: { fontSize: 22, bold: true, color: C.text1, breakLine: true } },
      { text: sub, options: { fontSize: 14, color: C.accent2 } },
    ], { x: M + 1.3, y: y + 0.08, w: CW - 3.3, h: 0.84, valign: 'middle', paraSpaceAfter: 2, objectName: `목차 항목 ${i + 1}` });
    tx(s, '슬라이드 ' + range, { x: W - M - 1.85, y: y + 0.08, w: 1.55, h: 0.84, fontSize: 14, bold: true, color: C.accent2, align: 'right', valign: 'middle', objectName: `목차 범위 ${i + 1}` });
  });
  s.addNotes('먼저 문제와 대상을 보고, 다섯 가지 기술을 차례로 설명한 뒤, 결과와 결론으로 마무리한다.');

  // 3. 문제 -------------------------------------------------------------
  pres.addSection({ title: '문제와 대상 시스템' });
  s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: '문제와 대상 시스템' });
  s.addText('문제 — 추론만 NPU로 옮겨도 CPU는 바쁘다', { placeholder: 'title', lang: LANG });
  tx(s, '한 프레임의 흐름', { x: M, y: 1.35, w: 4, h: 0.3, fontSize: 12, bold: true, color: C.accent2, objectName: '흐름 라벨' });
  const stages = [
    ['디코드', I.film, 'CPU'], ['크기·색·형식 변환', I.shuffle, 'CPU'], ['추론', I.npu, 'NPU'],
    ['결과 해석\n(박스 복원·NMS)', I.scan, 'CPU'], ['표시', I.monitor, 'CPU'],
  ];
  const bw = 2.0, gap = (CW - 5 * bw) / 4, by = 1.75, bh = 1.3;
  stages.forEach(([label, ic, who], i) => {
    const x = M + i * (bw + gap);
    const npu = who === 'NPU';
    card(s, `단계 ${i + 1}`, x, by, bw, bh, npu ? C.accent1 : C.accent2);
    s.addImage({ data: ic, x: x + bw / 2 - 0.2, y: by + 0.15, w: 0.4, h: 0.4, altText: label.replace('\n', ' '), objectName: `단계 아이콘 ${i + 1}` });
    tx(s, label, { x: x + 0.08, y: by + 0.6, w: bw - 0.16, h: 0.65, fontSize: 15, bold: true, color: C.background1, align: 'center', valign: 'middle', objectName: `단계 라벨 ${i + 1}` });
    tx(s, who + (npu ? ' 담당' : ' 담당'), { x, y: by + bh + 0.08, w: bw, h: 0.3, fontSize: 13, bold: true, color: npu ? C.accent1 : C.accent2, align: 'center', objectName: `담당 ${i + 1}` });
    if (i < 4) arrow(s, `화살표 ${i + 1}`, x + bw + 0.08, by + bh / 2, x + bw + gap - 0.08, by + bh / 2);
  });
  tx(s, bullets([
    '추론만 NPU가 맡으면 나머지는 CPU가 하며, 프레임마다 수 MB를 읽고 쓴다',
    '스마트폰·서버에서도 추론 밖 전·후처리와 데이터 이동이 종단간 성능을 좌우한다는 보고가 있다 [1][2]',
  ], { fontSize: 16, paraSpaceAfter: 14 }), { x: M, y: 3.85, w: 7.3, h: 2.75, valign: 'middle', objectName: '본문' });
  card(s, '출발점 카드', 8.4, 3.85, W - M - 8.4, 2.75, C.background2);
  tx(s, [
    { text: '출발점 · 배치 1 응용 CPU', options: { fontSize: 14, color: C.accent2, bold: true, breakLine: true } },
    { text: '17.6', options: { fontSize: 60, bold: true, color: C.accent1 } },
    { text: ' ms/프레임', options: { fontSize: 20, bold: true, color: C.accent1, breakLine: true } },
    { text: '한 코어의 40% 이상', options: { fontSize: 16, color: C.text1 } },
  ], { x: 8.7, y: 4.1, w: W - M - 9.0, h: 2.3, valign: 'middle', objectName: '출발점 수치' });
  s.addNotes('목표는 CPU 개입을 최소화하는 것이다. 단, 처리량이 아니라 최저 지연 운용에서 그렇게 한다.');

  // 4. 대상 시스템 --------------------------------------------------------
  s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: '문제와 대상 시스템' });
  s.addText('대상 시스템과 운용 기준', { placeholder: 'title', lang: LANG });
  const sys = [
    ['보드', 'Raspberry Pi Compute Module 5 [3]', I.board],
    ['NPU', 'Hailo-8 (PCIe)\n런타임 HailoRT 4.24.0 [4]', I.chip],
    ['모델', 'YOLO11m [5]', I.model],
    ['입력', 'HEVC 영상\n(SoC 하드웨어 디코더)', I.filmD],
  ];
  const cw3 = 3.6, ch3 = 2.4, g3 = 0.3;
  sys.forEach(([k, v, ic], i) => {
    const x = M + (i % 2) * (cw3 + g3), y = 1.5 + Math.floor(i / 2) * (ch3 + g3);
    card(s, `시스템 카드 ${k}`, x, y, cw3, ch3, C.background2);
    s.addShape(pres.shapes.OVAL, { x: x + 0.3, y: y + 0.3, w: 0.7, h: 0.7, fill: { color: C.background1 }, line: { type: 'none' }, objectName: `아이콘 원 ${k}` });
    s.addImage({ data: ic, x: x + 0.45, y: y + 0.45, w: 0.4, h: 0.4, altText: k, objectName: `아이콘 ${k}` });
    tx(s, k, { x: x + 1.2, y: y + 0.3, w: cw3 - 1.5, h: 0.7, fontSize: 16, bold: true, color: C.accent2, valign: 'middle', objectName: `라벨 ${k}` });
    tx(s, v, { x: x + 0.3, y: y + 1.15, w: cw3 - 0.5, h: 1.05, fontSize: 18, bold: true, color: C.text1, objectName: `값 ${k}` });
  });
  const px3 = M + 2 * cw3 + 2 * g3 + 0.1;
  card(s, '운용 기준 패널', px3, 1.5, W - M - px3, 2 * ch3 + g3, C.text2);
  s.addImage({ data: I.timerL, x: px3 + 0.4, y: 2.45, w: 0.6, h: 0.6, altText: '지연', objectName: '운용 기준 아이콘' });
  tx(s, [
    { text: '운용 기준', options: { fontSize: 16, bold: true, color: C.accent3, breakLine: true } },
    { text: '처리량(FPS)이 아니라', options: { fontSize: 20, color: C.background1, breakLine: true } },
    { text: '최저 지연', options: { fontSize: 44, bold: true, color: C.accent6 } },
  ], { x: px3 + 0.4, y: 3.25, w: W - M - px3 - 0.7, h: 2.2, paraSpaceAfter: 6, objectName: '운용 기준 문구' });
  s.addNotes('같은 계열 구성의 선행 사례가 있다 [6][7].');

  // 5. 전체 구조 ----------------------------------------------------------
  s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: '문제와 대상 시스템' });
  s.addText('전체 구조 — 데이터는 하드웨어끼리, CPU는 지휘만', { placeholder: 'title', lang: LANG });
  const fw4 = 11.2, fh4 = fw4 * 1223 / 2700;
  s.addImage({ path: path.join(FIG, 'fig10_brief_cpu_offload.png'), x: (W - fw4) / 2, y: 1.3, w: fw4, h: fh4,
    altText: '그림 1. CPU 최소화 기술의 전체 구조: 영상 입력 → 하드웨어 디코더 → GPU 전처리(NHCW) → NPU → GPU 후처리(NMS) → 화면, CPU는 점선으로 제어만', objectName: '그림 1 전체 구조' });
  tx(s, '굵은 실선 = dma-buf로 복사 없이 넘기는 데이터 · 가는 실선 = CPU를 거치는 작은 데이터 · 점선 = CPU 제어 · ①~⑤ = 다섯 기술',
    { x: (W - fw4) / 2, y: 1.3 + fh4 + 0.1, w: fw4, h: 0.35, fontSize: 12, color: C.accent2, objectName: '그림 캡션' });
  s.addNotes('다섯 기술 — ① 하드웨어 디코더와 제로 카피 ② NPU 형식을 GPU가 직접 생성 ③ 결과 해석도 GPU ④ 최저 지연 운용 ⑤ CPU는 지휘만.');

  // 6. ① 하드웨어 디코더 (2026-10-27 개정: 가로 막대 차트 2개) ------------
  pres.addSection({ title: 'CPU 최소화 기술' });
  s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'CPU 최소화 기술' });
  s.addText('① 하드웨어 디코더 + 제로 카피(dma-buf)', { placeholder: 'title', lang: LANG });
  tx(s, bullets([
    'HEVC 하드웨어 디코더로 풀고(FFmpeg 하드웨어 경로 [9]), 출력은 dma-buf로 GPU에 바로 넘긴다 [8]',
  ], { fontSize: 16 }), { x: M, y: 1.4, w: CW, h: 0.45, objectName: '본문' });
  // 범위값 0.23~0.28은 0.23까지 진한 구간 + 0.05(=0.28-0.23) 옅은 구간의 누적 막대로 그린다.
  // 값 라벨은 차트 자동 라벨을 끄고, 고정한 그림 영역(inner manualLayout) 기준으로 계산한 위치에 글상자로 단다.
  const LAY6 = { x: 0.40, y: 0.16, w: 0.50, h: 0.70 };
  const ch6 = 3.55, cy6 = 1.95, cw6 = (CW - 0.4) / 2;
  const hbar = (x, title, labels, series, colors, maxVal, major, fmt, valueLabels, name) => {
    s.addChart(pres.charts.BAR, series, Object.assign(BAR_CHART_BASE(), {
      x, y: cy6, w: cw6, h: ch6, barDir: 'bar', barGrouping: 'stacked', layout: Object.assign({}, LAY6),
      title, showValue: false, dataLabelPosition: 'ctr', chartColors: colors.slice(), barGapWidthPct: 60,
      valAxisMinVal: 0, valAxisMaxVal: maxVal, valAxisMajorUnit: major, valAxisLabelFormatCode: fmt, objectName: name,
    }));
    const ix = x + LAY6.x * cw6, iw = LAY6.w * cw6, iy = cy6 + LAY6.y * ch6, ih = LAY6.h * ch6, n = labels.length;
    valueLabels.forEach(([idx, end, text]) => {
      const yc = iy + ih * (1 - (idx + 0.5) / n);
      tx(s, text, { x: ix + iw * end / maxVal + 0.06, y: yc - 0.18, w: 1.3, h: 0.36, fontSize: 16, bold: true, color: C.text1, valign: 'middle', objectName: `값 라벨 ${text}` });
    });
  };
  // 가로 막대는 첫 범주가 아래에 그려지므로 위→아래 순서의 역순으로 넣는다.
  hbar(M, '(a) 소프트웨어 대 하드웨어 (ms/프레임)',
    ['하드웨어 + dma-buf\n(응용)', '소프트웨어 디코드\n(avdec_h265)'],
    [
      { name: '소프트웨어 디코드', labels: ['하드웨어 + dma-buf\n(응용)', '소프트웨어 디코드\n(avdec_h265)'], values: [0, 31.8] },
      { name: 'dma-buf 0.23까지', labels: ['하드웨어 + dma-buf\n(응용)', '소프트웨어 디코드\n(avdec_h265)'], values: [0.23, 0] },
      { name: 'dma-buf 0.23~0.28 범위', labels: ['하드웨어 + dma-buf\n(응용)', '소프트웨어 디코드\n(avdec_h265)'], values: [0.05, 0] },
    ],
    [HEX.accent3, HEX.accent1, HEX.accent6], 40, 10, '0',
    [[1, 31.8, '31.8'], [0, 0.28, '0.23~0.28']], '차트 (a) 소프트웨어 대 하드웨어 디코드');
  const L6b = ['dma-buf 유지\n(응용)', 'dma-buf 유지\n(명령행)', 'CPU로 내려받음\n(명령행)'];
  hbar(M + cw6 + 0.4, '(b) 하드웨어 출력 경로별 · 확대 (ms/프레임)', L6b,
    [
      { name: 'CPU로 내려받음', labels: L6b, values: [0, 0, 2.02] },
      { name: 'dma-buf 유지 (명령행)', labels: L6b, values: [0, 0.55, 0] },
      { name: 'dma-buf 0.23까지', labels: L6b, values: [0.23, 0, 0] },
      { name: 'dma-buf 0.23~0.28 범위', labels: L6b, values: [0.05, 0, 0] },
    ],
    [HEX.accent3, HEX.accent1, HEX.accent1, HEX.accent6], 2.5, 0.5, '0.0',
    [[2, 2.02, '2.02'], [1, 0.55, '0.55'], [0, 0.28, '0.23~0.28']], '차트 (b) 하드웨어 출력 경로별');
  s.addImage({ data: I.info, x: M, y: 5.67, w: 0.35, h: 0.35, altText: '한 줄 메시지', objectName: '메시지 아이콘' });
  tx(s, [
    { text: '같은 하드웨어 디코드라도 ' },
    { text: 'CPU로 내려받으면 약 3.6배', options: { bold: true, color: C.accent1 } },
    { text: ' (명령행 기준 2.02 대 0.55 ms) — 비용은 디코드가 아니라 복사' },
  ], { x: M + 0.5, y: 5.62, w: CW - 0.5, h: 0.45, fontSize: 15, valign: 'middle', objectName: '한 줄 메시지' });
  tx(s, bullets([
    '(a)의 소프트웨어 값은 GStreamer avdec_h265 측정(2026-09-20), 하드웨어 값은 ffmpeg 명령행·응용 측정(2026-09-29 이후)으로 측정 날짜·도구가 다르다. 모두 1080p HEVC 1,816프레임, 디코드만, 3회 중앙값',
    '소프트웨어 = avdec_h265 자동 스레드 · 명령행 = ffmpeg (내려받음: 출력 형식 미지정 → yuv420p, 유지: DRM_PRIME) · 응용 = libavcodec DRM_PRIME',
  ], { fontSize: 10, color: C.accent2, paraSpaceAfter: 2, bullet: { indent: 10 } }), { x: M, y: 6.17, w: CW, h: 0.7, objectName: '각주' });
  s.addNotes('소프트웨어 디코드는 프레임당 약 32 ms로 한 코어를 거의 다 쓴다. 하드웨어 디코더를 쓰고 출력을 dma-buf로 그대로 넘기면 0.23~0.28 ms다. 오른쪽은 같은 하드웨어 디코드에서 출력을 CPU로 내려받기만 해도 약 3.6배 비싸진다는 것 — CPU 비용의 원인이 복사라는 증거다.');

  // 7. ② NHCW를 GPU가 -----------------------------------------------------
  s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'CPU 최소화 기술' });
  s.addText('② NPU가 원하는 형식(NHCW)을 GPU가 바로 만든다', { placeholder: 'title', lang: LANG });
  const lw6 = 6.7;
  tx(s, bullets([
    'NPU는 칩 내부 배치 NHCW를 쓴다. NHWC로 넘기면 런타임이 CPU에서 변환한다',
    [{ text: '원인은 연산이 아니라 ' }, { text: '캐시되지 않는 GPU 메모리를 CPU가 바이트 단위로 읽는 것', options: { bold: true } }],
  ], { fontSize: 16, paraSpaceAfter: 10 }), { x: M, y: 1.45, w: lw6, h: 1.7, objectName: '본문 위' });
  tx(s, '같은 전치 루프 단독', { x: M + 0.3, y: 3.2, w: 4, h: 0.3, fontSize: 14, bold: true, color: C.accent2, objectName: '전치 루프 라벨' });
  const m6w = (lw6 - 0.3 - 0.3) / 2;
  [['일반 메모리', '0.95 ms', C.background2, C.text1], ['GPU 메모리', '12.7 ms', C.background2, C.accent1]].forEach(([k, v, f, col], i) => {
    const x = M + 0.3 + i * (m6w + 0.3);
    card(s, `전치 루프 ${k}`, x, 3.55, m6w, 0.95, f);
    tx(s, [{ text: k + '  ', options: { fontSize: 14, color: C.accent2, bold: true } }, { text: v, options: { fontSize: 26, bold: true, color: col } }],
      { x: x + 0.25, y: 3.55, w: m6w - 0.4, h: 0.95, valign: 'middle', objectName: `전치 루프 값 ${k}` });
  });
  tx(s, bullets([
    '해법: Vulkan 전처리 셰이더 [10]가 GPU(V3D [11])에서 처음부터 NHCW로 쓴다',
    [{ text: '결과: 배치 1 응용 CPU ' }, { text: '17.6 → 3.3 ms/프레임', options: { bold: true, color: C.accent1 } }],
  ], { fontSize: 16, paraSpaceAfter: 10 }), { x: M, y: 4.75, w: lw6, h: 1.5, objectName: '본문 아래' });
  s.addChart(pres.charts.BAR, [{ name: '응용 CPU', labels: ['출발점', '② GPU가 NHCW 생성'], values: [17.6, 3.3] }],
    Object.assign(BAR_CHART_BASE(), { x: 7.75, y: 1.4, w: W - M - 7.75, h: 4.95, title: '배치 1 응용 CPU (ms/프레임)', dataLabelFormatCode: '0.0', valAxisLabelFormatCode: '0', objectName: '차트 응용 CPU 17.6 대 3.3' }));
  footnote(s, '측정 조건: GPU 960 MHz · 디코더 자동 스레드 · 배치 1/슬롯 2', 6.5);
  s.addNotes('NPU는 칩 내부용 NHCW 배치를 쓰므로 NHWC로 넘기면 런타임이 CPU에서 형식을 바꾼다. 비용의 원인은 변환 연산이 아니라 CPU가 캐시되지 않는 GPU 메모리를 바이트 단위로 읽는 데 있었다. 그래서 Vulkan 전처리 셰이더가 처음부터 NHCW로 쓰게 했고, 배치 1 응용 CPU가 17.6에서 3.3 ms/프레임으로 줄었다.');

  // 8. ③ 결과 해석도 GPU --------------------------------------------------
  s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'CPU 최소화 기술' });
  s.addText('③ 결과 해석(박스 복원 + NMS)도 GPU로', { placeholder: 'title', lang: LANG });
  const cw7 = 5.0;
  s.addChart(pres.charts.BAR, [{ name: '응용 CPU', labels: ['결과 해석 CPU (직전)', '③ 결과 해석 GPU'], values: [3.25, 0.94] }],
    Object.assign(BAR_CHART_BASE(), { x: M, y: 1.4, w: cw7, h: 4.95, title: '배치 1/슬롯 1 응용 CPU (ms/프레임)', dataLabelFormatCode: '0.00', valAxisLabelFormatCode: '0.0', objectName: '차트 응용 CPU 3.25 대 0.94' }));
  const rx7 = M + cw7 + 0.5, rw7 = W - M - rx7;
  tx(s, bullets([
    'NPU 원시 출력도 dma-buf로 GPU에 넘기고, GPU 셰이더가 박스 복원과 NMS를 한다',
    'CPU는 최종 검출 몇 개만 받는다',
    [{ text: '결과: 응용 CPU ' }, { text: '3.25 → 0.94 ms/프레임', options: { bold: true, color: C.accent1 } }],
  ], { fontSize: 16, paraSpaceAfter: 10 }), { x: rx7, y: 1.45, w: rw7, h: 2.3, objectName: '본문' });
  const k7w = (rw7 - 0.3) / 2;
  card(s, '지연 카드', rx7, 3.55, k7w, 2.75, C.background2);
  tx(s, [
    { text: '지연은 그대로', options: { fontSize: 14, bold: true, color: C.accent2, breakLine: true } },
    { text: '79.1 대 79.2 ms', options: { fontSize: 24, bold: true, color: C.text1 } },
  ], { x: rx7 + 0.25, y: 3.8, w: k7w - 0.45, h: 2.3, paraSpaceAfter: 8, objectName: '지연 값' });
  card(s, '정확도 카드', rx7 + k7w + 0.3, 3.55, k7w, 2.75, C.background2);
  tx(s, [
    { text: '정확도', options: { fontSize: 14, bold: true, color: C.accent2, breakLine: true } },
    { text: 'CPU 기준 구현과 같은 결과', options: { fontSize: 15, bold: true, color: C.text1, breakLine: true } },
    { text: 'NPU 런타임 NMS와 IoU 0.9 매칭 0.999', options: { fontSize: 15, color: C.text1 } },
  ], { x: rx7 + k7w + 0.55, y: 3.8, w: k7w - 0.45, h: 2.3, paraSpaceAfter: 8, objectName: '정확도 값' });
  footnote(s, '측정 조건: GPU 1150 MHz · 디코더 1스레드 · 배치 1/슬롯 1 (7번 슬라이드와 조건이 다르다)', 6.5);
  s.addNotes('NPU 원시 출력도 dma-buf로 GPU에 넘겨 박스 복원과 NMS를 GPU 셰이더가 하고, CPU는 최종 검출 몇 개만 받는다. 응용 CPU는 3.25에서 0.94 ms/프레임으로 줄었고 지연은 그대로였다. 정확도는 CPU 기준 구현과 같고, NPU 런타임 NMS와는 IoU 0.9 매칭 0.999다. 이 측정은 7번 슬라이드와 조건이 다르다.');

  // 9. ④ 최저 지연 운용 ---------------------------------------------------
  s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'CPU 최소화 기술' });
  s.addText('④ 최저 지연 운용 — 한 번에 한 장, 줄 세우지 않는다', { placeholder: 'title', lang: LANG });
  // 개념 도식 (수치 축 없음): 소스(페이싱) → 대기열 1칸 → 슬롯 1·배치 1(NPU) → 결과, 대기열에서 버림 분기
  const nodes = [
    ['입력 페이싱', '카메라처럼 소스 시각에 도착', I.film, C.accent2],
    ['대기열 1칸', '줄 세우지 않는다', I.layers, C.accent2],
    ['배치 1 · 슬롯 1', 'NPU가 한 번에 한 장', I.npu, C.accent1],
    ['검출 결과', '', I.check, C.accent5],
  ];
  const nw = 2.55, ng = (CW - 4 * nw) / 3, ny = 1.4, nh = 1.15;
  nodes.forEach(([a, b, ic, f], i) => {
    const x = M + i * (nw + ng);
    card(s, `도식 노드 ${i + 1}`, x, ny, nw, nh, f);
    s.addImage({ data: ic, x: x + 0.2, y: ny + nh / 2 - 0.2, w: 0.4, h: 0.4, altText: a, objectName: `도식 아이콘 ${i + 1}` });
    tx(s, b ? [{ text: a, options: { fontSize: 16, bold: true, color: C.background1, breakLine: true } }, { text: b, options: { fontSize: 12, color: C.background1 } }]
      : [{ text: a, options: { fontSize: 16, bold: true, color: C.background1 } }],
    { x: x + 0.75, y: ny + 0.1, w: nw - 0.85, h: nh - 0.2, valign: 'middle', objectName: `도식 라벨 ${i + 1}` });
    if (i < 3) arrow(s, `도식 화살표 ${i + 1}`, x + nw + 0.06, ny + nh / 2, x + nw + ng - 0.06, ny + nh / 2, HEX.accent2);
  });
  const qx = M + nw + ng; // 대기열 노드 x
  arrow(s, '버림 화살표', qx + nw / 2, ny + nh + 0.04, qx + nw / 2, ny + nh + 0.42, HEX.accent2, 'dash');
  s.addImage({ data: I.x, x: qx + nw / 2 + 0.12, y: ny + nh + 0.12, w: 0.3, h: 0.3, altText: '버림', objectName: '버림 아이콘' });
  tx(s, '처리 못 한 프레임은 버림', { x: qx + nw / 2 + 0.5, y: ny + nh + 0.12, w: 3.5, h: 0.3, fontSize: 13, bold: true, color: C.accent2, valign: 'middle', objectName: '버림 라벨' });
  // 아래 왼쪽: 본문 + 핵심 수치
  const lw8 = 5.6;
  tx(s, bullets([
    '페이싱이 없으면 디코드 끝 기준 79 ms의 절반 넘게(약 42 ms, 추정)가 앞 프레임을 기다린 대기였다',
  ], { fontSize: 15 }), { x: M, y: 3.35, w: lw8, h: 1.0, objectName: '본문' });
  card(s, '결과 카드', M, 4.45, lw8, 2.25, C.background2);
  tx(s, [
    { text: '결과 · 도착 기준 평균', options: { fontSize: 14, bold: true, color: C.accent2, breakLine: true } },
    { text: '58.4 ms', options: { fontSize: 44, bold: true, color: C.accent1 } },
    { text: '   입력 18.6% 버림', options: { fontSize: 22, bold: true, color: C.accent1, breakLine: true } },
    { text: '그 안의 입력 대기열 대기 약 14.5 ms', options: { fontSize: 15, color: C.text1 } },
  ], { x: M + 0.3, y: 4.6, w: lw8 - 0.5, h: 1.95, valign: 'middle', paraSpaceAfter: 4, objectName: '결과 수치' });
  // 아래 오른쪽: 원고 그림
  const fx8 = M + lw8 + 0.4, fw8 = W - M - fx8, fh8 = fw8 * 708 / 2066;
  s.addImage({ path: path.join(FIG, 'fig05_input_queue.png'), x: fx8, y: 3.35, w: fw8, h: fh8,
    altText: '입력 대기열 크기 N에 따른 (a) 도착→결과 지연과 (b) 입력 버림률', objectName: '그림 입력 대기열' });
  tx(s, '대기열 크기 N이 1 늘 때마다 b1/s1 평균 지연 +33.1 ms (1/30 s), 버림률은 거의 같다',
    { x: fx8, y: 3.35 + fh8 + 0.1, w: fw8, h: 0.6, fontSize: 12, color: C.accent2, objectName: '그림 캡션' });
  s.addNotes('배치 1, 슬롯 1로 한 번에 한 프레임만 처리하고, 입력은 카메라처럼 소스 시각에 맞춰 도착시키며, 대기열은 한 칸만 두어 처리 못 한 프레임은 버린다. 페이싱이 없을 때는 디코드 끝 기준 79 ms의 절반 넘게가 앞 프레임을 기다린 대기였다. 이 방식으로 도착 기준 평균 58.4 ms를 얻었고, 그 대가로 입력의 18.6%를 버린다.');

  // 10. ⑤ CPU는 거의 지휘만 (신규 분리) ------------------------------------
  s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'CPU 최소화 기술' });
  s.addText('⑤ CPU는 거의 지휘만 한다', { placeholder: 'title', lang: LANG });
  // 개념 도식 (수치 축 없음): 5번 구조도와 같은 선 규칙 — 굵은 실선 = 데이터(dma-buf), 점선 = CPU 제어
  const dw = 7.3, bw10 = 1.45, bg10 = (dw - 4 * bw10) / 3, by10 = 1.5, bh10 = 0.95;
  const hw = [['하드웨어\n디코더', C.accent5], ['GPU\n전처리', C.accent5], ['NPU\n추론', C.accent1], ['GPU\n후처리', C.accent5]];
  const cpuX = M + dw / 2 - 1.35, cpuY = 3.45, cpuW = 2.7, cpuH = 0.85;
  hw.forEach(([label, fill], i) => {
    const x = M + i * (bw10 + bg10);
    card(s, `도식 하드웨어 ${i + 1}`, x, by10, bw10, bh10, fill);
    tx(s, label, { x, y: by10, w: bw10, h: bh10, fontSize: 15, bold: true, color: C.background1, align: 'center', valign: 'middle', objectName: `도식 하드웨어 라벨 ${i + 1}` });
    if (i < 3) s.addShape(pres.shapes.LINE, { x: x + bw10 + 0.05, y: by10 + bh10 / 2, w: bg10 - 0.1, h: 0, line: { color: HEX.dk1, width: 4, endArrowType: 'triangle' }, objectName: `데이터 화살표 ${i + 1}` });
    // CPU 상자 윗변에서 각 하드웨어로 가는 점선 제어 화살표
    const sx = cpuX + 0.45 + i * (cpuW - 0.9) / 3, bx = x + bw10 / 2, top = by10 + bh10 + 0.05;
    const ctl = { color: HEX.accent2, width: 1.5, dashType: 'dash' };
    if (bx < sx) s.addShape(pres.shapes.LINE, { x: bx, y: top, w: sx - bx, h: cpuY - top, line: Object.assign({ beginArrowType: 'triangle' }, ctl), objectName: `제어 화살표 ${i + 1}` });
    else s.addShape(pres.shapes.LINE, { x: sx, y: top, w: bx - sx, h: cpuY - top, flipV: true, line: Object.assign({ endArrowType: 'triangle' }, ctl), objectName: `제어 화살표 ${i + 1}` });
  });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: cpuX, y: cpuY, w: cpuW, h: cpuH, rectRadius: 0.08, fill: { color: C.background1 }, line: { color: HEX.accent2, width: 1.5 }, objectName: '도식 CPU' });
  tx(s, [{ text: 'CPU', options: { bold: true, fontSize: 16, breakLine: true } }, { text: '제출 · 이벤트 대기', options: { fontSize: 13, color: C.accent2 } }],
    { x: cpuX, y: cpuY, w: cpuW, h: cpuH, align: 'center', valign: 'middle', objectName: '도식 CPU 라벨' });
  // 범례 (5번 구조도와 같은 규칙)
  s.addShape(pres.shapes.LINE, { x: M, y: 4.67, w: 0.6, h: 0, line: { color: HEX.dk1, width: 4, endArrowType: 'triangle' }, objectName: '범례 데이터 선' });
  tx(s, '데이터 (dma-buf · 복사 없음)', { x: M + 0.75, y: 4.52, w: 3.0, h: 0.3, fontSize: 12, color: C.text1, valign: 'middle', objectName: '범례 데이터' });
  s.addShape(pres.shapes.LINE, { x: M + 3.9, y: 4.67, w: 0.6, h: 0, line: { color: HEX.accent2, width: 1.5, dashType: 'dash', endArrowType: 'triangle' }, objectName: '범례 제어 선' });
  tx(s, '제어 (CPU)', { x: M + 4.65, y: 4.52, w: 2.0, h: 0.3, fontSize: 12, color: C.text1, valign: 'middle', objectName: '범례 제어' });
  // 폴링 대신 이벤트 대기 (전/후 두 칸)
  const pw10 = (dw - 0.3) / 2;
  [['이전 · 폴링', '일정 간격으로 완료를 확인한다', I.poll, C.background2, C.accent2], ['지금 · 이벤트 대기', '완료 이벤트로 깨어나 기다린다', I.bell, C.background2, C.accent1]].forEach(([h, d, ic, f, col], i) => {
    const x = M + i * (pw10 + 0.3);
    card(s, `대기 방식 ${i + 1}`, x, 5.05, pw10, 1.5, f);
    s.addImage({ data: ic, x: x + 0.3, y: 5.5, w: 0.55, h: 0.55, altText: h, objectName: `대기 방식 아이콘 ${i + 1}` });
    tx(s, [{ text: h, options: { fontSize: 16, bold: true, color: col, breakLine: true } }, { text: d, options: { fontSize: 14, color: C.text1 } }],
      { x: x + 1.05, y: 5.15, w: pw10 - 1.25, h: 1.3, valign: 'middle', paraSpaceAfter: 4, objectName: `대기 방식 설명 ${i + 1}` });
  });
  // 오른쪽: 수치 카드 + 글머리
  const rx10 = M + dw + 0.45, rw10 = W - M - rx10;
  card(s, '⑤ 결과 카드', rx10, 1.5, rw10, 2.3, C.background2);
  tx(s, [
    { text: '①~⑤ 모두 적용 · 최저 지연 운용점', options: { fontSize: 14, bold: true, color: C.accent2, breakLine: true } },
    { text: '응용 CPU', options: { fontSize: 16, bold: true, color: C.text1, breakLine: true } },
    { text: '약 1.0', options: { fontSize: 48, bold: true, color: C.accent1 } },
    { text: ' ms/프레임', options: { fontSize: 20, bold: true, color: C.accent1, breakLine: true } },
    { text: '(처리한 프레임 기준)', options: { fontSize: 14, color: C.text1 } },
  ], { x: rx10 + 0.3, y: 1.6, w: rw10 - 0.5, h: 2.1, valign: 'middle', objectName: '⑤ 결과 수치' });
  tx(s, bullets([
    [{ text: 'CPU는 큰 데이터를 만지지 않고 ' }, { text: '작업 제출과 완료 확인만', options: { bold: true } }, { text: ' 한다' }],
    [{ text: '완료도 폴링 대신 ' }, { text: '완료 이벤트로 깨어나 기다린다', options: { bold: true } }],
    '감소의 대부분은 ②·③의 몫이다',
  ], { fontSize: 15, paraSpaceAfter: 10 }), { x: rx10, y: 4.05, w: rw10, h: 2.5, objectName: '본문' });
  s.addNotes('남은 CPU 일은 지휘뿐이다. 작업을 제출하고, 완료 이벤트가 오면 깨어나 다음 일을 넘긴다. 이렇게 해서 최저 지연 운용점에서 처리한 프레임당 약 1 ms가 남았고, 그 감소의 대부분은 앞의 ②·③ 덕분이다.');

  // 11. 결과 요약 ----------------------------------------------------------
  pres.addSection({ title: '결과' });
  s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: '결과' });
  s.addText('결과 요약', { placeholder: 'title', lang: LANG });
  const hdr = (t, al) => ({ text: t, options: { lang: LANG, bold: true, color: C.background1, fill: { color: C.text2 }, align: al } });
  const cell = (t, al, extra) => ({ text: typeof t === 'string' ? nb(t) : t.map((r) => Object.assign({}, r, { text: nb(r.text) })), options: Object.assign({ align: al, lang: LANG }, extra || {}) });
  const hl = { fill: { color: C.background2 } };
  const rows = [
    [hdr('구성', 'left'), hdr('응용 CPU (ms/프레임)', 'right'), hdr('지연', 'right')],
    [cell('출발점: 형식 변환·NMS를 CPU에서', 'left'), cell('17.6', 'right'), cell('–', 'right')],
    [cell('② NPU 형식(NHCW)을 GPU가 직접 생성', 'left'), cell('3.3', 'right'), cell('–', 'right')],
    [cell('③ 결과 해석(NMS 포함)을 GPU로', 'left'), cell('0.94 (직전 3.25)', 'right'), cell('79.2 ms (디코드 끝 기준)', 'right')],
    [cell('④⑤ 최저 지연 운용(입력 페이싱 + 대기열 1)', 'left', hl), cell('약 1.0', 'right', Object.assign({ bold: true }, hl)),
      cell([{ text: '58.4 ms (도착 기준, 입력 18.6% 버림)', options: { bold: true, breakLine: true } }, { text: '화면까지 70.2 ms' }], 'right', hl)],
  ];
  s.addTable(rows, {
    x: M, y: 1.5, w: CW, colW: [5.4, 2.8, CW - 8.2], fontSize: 14, color: C.text1, lang: LANG,
    border: { type: 'solid', pt: 0.75, color: HEX.accent4 }, margin: [0.06, 0.12, 0.06, 0.12], valign: 'middle',
    rowH: [0.55, 0.55, 0.55, 0.55, 0.85], objectName: '표 1 결과 요약',
  });
  tx(s, '검출 결과는 기준 경로와 같다 (1,816프레임에서 9,457개)', { x: M, y: 4.8, w: CW, h: 0.35, fontSize: 14, objectName: '검출 결과' });
  tx(s, bullets([
    '1~2행: GPU 960 MHz · 디코더 자동 스레드 · 배치 1/슬롯 2 (1,816프레임 3회 중앙값). 3~4행: GPU 1150 MHz · 디코더 1스레드 · 배치 1/슬롯 1 (900프레임 3회 중앙값)',
    '4행 CPU는 처리한 프레임 기준, 3·4행 지연은 기점이 달라 서로 빼지 않는다',
    '58.4 ms는 표시 없이 입력 18.6% 버림, 70.2 ms는 화면 표시를 켠 별도 실행(버림 19.1%)',
  ], { fontSize: 11, color: C.accent2, paraSpaceAfter: 3, bullet: { indent: 10 } }), { x: M, y: 5.35, w: CW, h: 1.2, objectName: '표 각주' });
  s.addNotes('표는 기술별 응용 CPU 시간과 최저 지연 운용점을 정리한 것이다. 위 두 행과 아래 두 행은 측정 조건이 다르고, 4행 CPU는 처리한 프레임 기준이며, 3·4행의 지연은 기점이 달라 서로 빼지 않는다. 검출 결과는 기준 경로와 같았다.');

  // 12. 결론 -------------------------------------------------------------
  pres.addSection({ title: '결론' });
  s = pres.addSlide({ masterName: 'CONCLUSION_DARK', sectionTitle: '결론' });
  s.addText('결론', { placeholder: 'title', lang: LANG });
  s.addImage({ data: I.key, x: M, y: 1.95, w: 0.6, h: 0.6, altText: '핵심', objectName: '핵심 아이콘' });
  tx(s, [
    { text: 'CPU를 줄이는 열쇠는 연산을 빠르게 하는 것이 아니라 ', options: { color: C.background1 } },
    { text: 'CPU가 데이터를 읽지 않게 하는 것', options: { bold: true, color: C.accent6 } },
  ], { x: M + 0.9, y: 1.6, w: CW - 0.9, h: 1.3, fontSize: 26, valign: 'middle', objectName: '핵심 메시지' });
  const pts11 = [
    [I.link, '디코더·GPU·NPU가 dma-buf로 직접 주고받고, 형식 생성과 결과 해석을 GPU에 맡기면 CPU에는 지휘만 남는다'],
    [I.clock, '최저 지연 운용("한 번에 한 장, 줄 세우지 않기")에서 응용 CPU 약 1 ms/프레임, 도착 기준 58.4 ms (입력 18.6% 버림)'],
  ];
  pts11.forEach(([ic, t], i) => {
    const y = 3.5 + i * 1.3;
    s.addImage({ data: ic, x: M + 0.08, y: y + 0.28, w: 0.45, h: 0.45, altText: '요점', objectName: `요점 아이콘 ${i + 1}` });
    tx(s, t, { x: M + 0.9, y, w: CW - 0.9, h: 1.0, fontSize: 18, color: C.background1, valign: 'middle', objectName: `요점 ${i + 1}` });
  });
  s.addNotes('정리하면, CPU를 줄이는 열쇠는 연산을 빠르게 하는 것이 아니라 CPU가 데이터를 읽지 않게 하는 것이다. 디코더·GPU·NPU가 dma-buf로 직접 주고받고 형식 생성과 결과 해석을 GPU에 맡기면 CPU에는 지휘만 남는다. 최저 지연 운용에서 응용 CPU는 약 1 ms/프레임, 도착 기준 58.4 ms였고, 그 대가로 입력의 18.6%를 버린다.');

  // 13. 맺음말 (신규) ------------------------------------------------------
  pres.addSection({ title: '맺음말' });
  s = pres.addSlide({ masterName: 'CLOSING_DARK', sectionTitle: '맺음말' });
  s.addText('감사합니다', { placeholder: 'title', lang: LANG });
  s.addText('질의응답 (Q&A)', { placeholder: 'subtitle', lang: LANG });
  s.addText('발표자 (작성 필요)', { placeholder: 'presenter', lang: LANG });
  // 1번 표지와 짝: 같은 파이프라인 아이콘 4개
  [I.film, I.gpu, I.npu, I.monitor].forEach((ic, i) => {
    const x = 8.55 + i * 1.05;
    s.addShape(pres.shapes.OVAL, { x, y: 3.25, w: 0.8, h: 0.8, fill: { color: i === 2 ? C.accent1 : C.accent5 }, line: { type: 'none' }, objectName: `흐름 아이콘 원 ${i + 1}` });
    s.addImage({ data: ic, x: x + 0.2, y: 3.45, w: 0.4, h: 0.4, altText: ['디코드', 'GPU', 'NPU', '표시'][i], objectName: `흐름 아이콘 ${i + 1}` });
  });
  tx(s, '참고문헌', { x: M, y: 4.85, w: 3, h: 0.3, fontSize: 11, bold: true, color: C.accent3, objectName: '참고문헌 머리' });
  const refs = [
    '[1] M. Buch 외, "AI Tax in Mobile SoCs," ISPASS 2021.',
    '[2] A. F. AbouElhamayed 외, "Beyond Inference," DAC 2024.',
    '[3] Raspberry Pi Ltd., "Raspberry Pi Compute Module 5 Datasheet," 2026.',
    '[4] Hailo Technologies, "HailoRT v4.24.0 User Guide."',
    '[5] G. Jocher and J. Qiu, "Ultralytics YOLO11," 2024.',
    '[6] B. Vergara 외, "A Hailo-Accelerated System for Precision Agriculture," CHILECON 2025.',
    '[7] 정필성, "실시간 객체탐지를 위한 저비용 엣지 AI 가속기," 한국컴퓨터정보학회논문지, 2026.',
    '[8] Linux Kernel Documentation, "Buffer Sharing and Synchronization (dma-buf)," Linux 6.18.',
    '[9] FFmpeg Developers, "FFmpeg," version 7.1.5.',
    '[10] Khronos Vulkan Working Group, "Vulkan 1.4.365 Specification," 2026.',
    '[11] Mesa 3D Graphics Library, "V3D," Mesa 26.2.2 문서.',
  ];
  const refRuns = (list) => list.map((t, i) => ({ text: t, options: { breakLine: i < list.length - 1 } }));
  const rw12 = (CW - 0.4) / 2;
  tx(s, refRuns(refs.slice(0, 6)), { x: M, y: 5.17, w: rw12, h: 1.6, fontSize: 10, color: C.accent4, paraSpaceAfter: 2, objectName: '참고문헌 1~6' });
  tx(s, refRuns(refs.slice(6)), { x: M + rw12 + 0.4, y: 5.17, w: rw12, h: 1.6, fontSize: 10, color: C.accent4, paraSpaceAfter: 2, objectName: '참고문헌 7~11' });
  s.addNotes('이상입니다. 질문 받겠습니다.');

  // ---------------------------------------------------------------- 저장·후처리
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  await postprocess(OUT);
  console.log('생성 완료: ' + OUT);
}

function guid(seed) {
  const h = require('crypto').createHash('md5').update('section:' + seed).digest('hex');
  return `{${h.slice(0, 8)}-${h.slice(8, 12)}-${h.slice(12, 16)}-${h.slice(16, 20)}-${h.slice(20, 32)}}`.toUpperCase();
}
// 내장 xlsx를 고정 날짜로 다시 묶는다 (core.xml 날짜와 zip 항목 날짜)
async function fixedZip(buf, date) {
  const z = await JSZip.loadAsync(buf);
  const o = new JSZip();
  for (const n of Object.keys(z.files).filter((k) => !z.files[k].dir).sort()) {
    let d = await z.file(n).async(n === 'docProps/core.xml' ? 'string' : 'nodebuffer');
    if (typeof d === 'string') d = d.replace(/(<dcterms:(created|modified)[^>]*>)[^<]*(<\/dcterms:\2>)/g, '$12026-10-27T00:00:00Z$3');
    o.file(n, d, { date, createFolders: false });
  }
  return o.generateAsync({ type: 'nodebuffer', compression: 'DEFLATE' });
}

// 한글(ea) 글꼴 지정과 재현성(고정 날짜) 후처리
async function postprocess(file) {
  const zip = await JSZip.loadAsync(fs.readFileSync(file));
  const FIXED = new Date(Date.UTC(2026, 9, 27, 0, 0, 0));
  const names = Object.keys(zip.files).filter((n) => !zip.files[n].dir).sort();
  const out = new JSZip();
  for (const n of names) {
    let data = await zip.file(n).async(n.endsWith('.xml') || n.endsWith('.rels') ? 'string' : 'nodebuffer');
    if (n.startsWith('ppt/embeddings/') && n.endsWith('.xlsx')) data = await fixedZip(data, FIXED); // 차트 내장 통합문서
    if (typeof data === 'string') {
      if (n === 'ppt/presentation.xml') {
        // 섹션 id(무작위 GUID)를 섹션 이름에서 유도한 고정 GUID로 바꾼다
        data = data.replace(/(<p14:section name="([^"]*)" id=")\{[^}]*\}/g, (m, head, nm) => head + guid(nm));
      }
      if (n === 'ppt/theme/theme1.xml') {
        // 테마 major/minor 글꼴의 ea를 맑은 고딕으로 명시
        data = data.replace(/(<a:(majorFont|minorFont)>[\s\S]*?)<a:ea typeface=""\/>/g, `$1<a:ea typeface="${EA_FONT}"/>`);
        if (!data.includes(`<a:ea typeface="${EA_FONT}"/>`)) throw new Error('테마 ea 글꼴 치환 실패');
      }
      if (n.startsWith('ppt/charts/') && n.endsWith('.xml')) {
        // 차트: latin 뒤에 ea가 없으면 테마 ea(+mn-ea)를 붙인다
        data = data.replace(/(<a:latin typeface="([^"]*)"\s*\/>)(?!\s*<a:ea)/g, (m, tag, face) => `${tag}<a:ea typeface="${face.startsWith('+mj') ? '+mj-ea' : '+mn-ea'}"/>`);
      }
      if (/^ppt\/(slides|slideLayouts)\/[^/]+\.xml$/.test(n)) {
        // pptxgenjs는 한 문단의 둘째 run부터도 <a:pPr>를 run 뒤에 쓴다(스키마 위반, 문단 속성이 덮어써짐).
        // 문단 속성은 첫 run 앞의 것만 남긴다.
        let prev;
        do { prev = data; data = data.replace(/(<\/a:r>)<a:pPr\b(?:[^>]*\/>|[^>]*>[\s\S]*?<\/a:pPr>)/g, '$1'); } while (data !== prev);
      }
      if (n === 'docProps/core.xml') {
        data = data.replace(/(<dcterms:(created|modified)[^>]*>)[^<]*(<\/dcterms:\2>)/g, '$12026-10-27T00:00:00Z$3');
      }
    }
    out.file(n, data, { date: FIXED, createFolders: false });
  }
  fs.writeFileSync(file, await out.generateAsync({ type: 'nodebuffer', compression: 'DEFLATE' }));
}

build().catch((e) => { console.error('실패: ' + (e && e.stack || e)); process.exit(1); });
