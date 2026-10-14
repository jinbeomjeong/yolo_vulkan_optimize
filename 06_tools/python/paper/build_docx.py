"""Convert the paper manuscript (10_paper/paper_ko.md) to Word using the style template.

    /home/pi/miniconda3/envs/python-dev/bin/python 06_tools/python/paper/build_docx.py
    ... build_docx.py --out /tmp/paper_ko.docx                 # test build somewhere else
    ... build_docx.py --md other.md --reference ref.docx --out other.docx

Pipeline: pandoc markdown -> JSON AST -> (edits below) -> pandoc docx with --reference-doc -> python-docx/lxml
post-processing -> zip with fixed entry times. The manuscript is never modified.

AST edits (in memory):
  - image paths ending in .svg are replaced by the same-name .png (the 300 dpi files in figures/); a missing PNG
    is an error. Relative paths resolve against the manuscript folder (pandoc --resource-path);
  - a leading level-1 heading becomes the document title, and a level-2 heading right after it the subtitle
    (Title / Subtitle styles), because the manuscript has no YAML metadata block.
Markdown reader: pandoc markdown without implicit_figures (each figure already has its own "**그림 N.** ..."
caption paragraph, so pandoc's caption from the alt text would duplicate it; the alt text stays as the image
description), without smart (text is kept as typed, no curly quotes or dashes) and without subscript ("~" is a
range sign in Korean text, e.g. "300~350 MHz").

Post-processing:
  - styles: theme font attributes are removed and every style without an East Asian font gets one
    (body fonts from docDefaults, code fonts from Verbatim Char, heading fonts from Heading 1);
  - figures: image-only paragraphs get the Figure style and are scaled down to the text width (aspect kept);
    the paragraph right after a figure that starts with "그림 N" / "Figure N" gets the Image Caption style,
    a paragraph right before a table that starts with "표 N" / "Table N" gets Table Caption;
  - tables: full text width with column widths from the cell text (pandoc's come from the dash counts of the
    separator row), header row repeated on each page and kept with the next row, rows not split, cell
    paragraphs use Table Heading / Table Text, the borders of the template's "Table" style are copied onto each
    table as direct formatting (LibreOffice ignores table-style borders), 6 pt before the paragraph after a table;
  - code blocks: when the code font is one CJK monospace font in every slot (NanumGothicCoding), characters of
    East Asian width "A" (box drawing, arrows, circled digits, '·', '×') are drawn at 50 % width so they take one
    cell, as in a terminal, and the manuscript's text diagrams keep the columns they have in the source;
  - symbols that only one font of the Latin/Korean pair has (①..⑳, ▶ ▼; U+2212 minus) get that font in every
    slot, so they are not drawn in a substitute font;
  - '---' rules: pandoc's VML rectangle becomes an empty paragraph of the Horizontal Rule style;
  - paragraphs under a "참고문헌" / "References" heading get the Bibliography style;
  - core properties: created/modified = --date (default 2026-01-01), last-modified-by empty.
The same input gives a byte-identical docx (pandoc runs with SOURCE_DATE_EPOCH, zip entries get fixed times).

At the end the output path and a summary (paragraphs, headings, tables, figures, captions, code blocks) are
printed, and the manuscript's heading/table/image counts are checked against the docx (exit 1 on mismatch).
Page count is not known without rendering. Runs in conda env python-dev (python-docx, lxml, pandoc >= 3).
"""
from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import unicodedata
from collections import Counter
from collections.abc import Callable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import docx  # noqa: E402
from docx.oxml import OxmlElement  # noqa: E402
from docx.oxml.ns import qn  # noqa: E402
from lxml import etree  # noqa: E402

import make_reference_docx as ref  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MD = REPO_ROOT / "10_paper" / "paper_ko.md"
DEFAULT_REFERENCE = REPO_ROOT / "10_paper" / "template" / "reference.docx"
DEFAULT_OUT = REPO_ROOT / "10_paper" / "build" / "paper_ko.docx"
DEFAULT_DATE = "2026-01-01"

READER = "markdown-implicit_figures-smart-subscript"
FIG_CAPTION_RE = re.compile(r"^\s*(그림|Figure|Fig\.)\s*\d+")
TABLE_CAPTION_RE = re.compile(r"^\s*(표|Table)\s*\d+")
# AST node types reported in the summary because docx renders them poorly or not at all
NOTABLE = ("CodeBlock", "BlockQuote", "HorizontalRule", "Note", "Math", "RawInline", "RawBlock", "LineBlock",
           "DefinitionList", "Div", "Span", "Cite", "LineBreak")
BODY_STYLES_IN_CELLS = {None, "Compact", "BodyText", "FirstParagraph"}

EMU_PER_TWIP = 635
MIN_COL_CELLS = 6                 # narrowest automatic column, in half-width cells
REFERENCE_HEADINGS = {"참고문헌", "references", "bibliography", "reference"}
AFTER_TABLE_TWIPS = 120           # space between a table and the paragraph after it (6 pt)
# Characters that only one of a Latin/Korean font pair has. LibreOffice picks the font slot of such symbols from
# the neighbouring text, so "①" after Latin text lands in Times New Roman (no glyph) and "−" after Hangul in
# NanumMyeongjo (no glyph). Word's choice also depends on language and hints; one font in every slot is safe.
EAST_ASIAN_ONLY = ((0x2460, 0x24FF), (0x25A0, 0x25FF))   # enclosed alphanumerics (①), geometric shapes (▶ ▼)
LATIN_ONLY = ((0x2212, 0x2212),)                          # minus sign
CODE_RUN_STYLES = {"VerbatimChar"}
NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "pic": "http://schemas.openxmlformats.org/drawingml/2006/picture",
}


class ManuscriptError(ref.ToolError):
    """Problem in the manuscript or its assets (e.g. a figure without its PNG)."""


# --------------------------------------------------------------------------- pandoc AST stage

def walk(node, fn) -> None:
    """Call fn(element) for every AST element (dict with key 't'), depth first."""
    if isinstance(node, dict):
        if "t" in node:
            fn(node)
        for v in node.values():
            walk(v, fn)
    elif isinstance(node, list):
        for v in node:
            walk(v, fn)


def read_ast(pandoc: Path, md: Path) -> dict:
    out = subprocess.run([str(pandoc), "-f", READER, "-t", "json", str(md)],
                         capture_output=True, check=True).stdout
    return json.loads(out)


def svg_to_png(ast: dict, base: Path) -> list[str]:
    """Point .svg images at the same-name .png. Returns the image paths after the change."""
    paths: list[str] = []
    missing: list[str] = []

    def fix(el: dict) -> None:
        if el["t"] != "Image":
            return
        target = el["c"][2]
        src = target[0]
        if src.lower().endswith(".svg"):
            png = src[:-4] + ".png"
            if not (base / png).is_file():
                missing.append(png)
            target[0] = src = png
        paths.append(src)

    walk(ast["blocks"], fix)
    if missing:
        raise ManuscriptError("PNG missing for SVG figure(s): " + ", ".join(missing) +
                              " (regenerate the figures, e.g. 06_tools/python/paper/build_figures.py)")
    return paths


def promote_title(ast: dict) -> list[str]:
    """Leading H1 -> meta title, an H2 right after it -> meta subtitle. Returns the promoted meta keys."""
    blocks, meta = ast["blocks"], ast["meta"]
    promoted: list[str] = []
    if "title" in meta or not blocks or blocks[0]["t"] != "Header" or blocks[0]["c"][0] != 1:
        return promoted
    meta["title"] = {"t": "MetaInlines", "c": blocks.pop(0)["c"][2]}
    promoted.append("title")
    if blocks and blocks[0]["t"] == "Header" and blocks[0]["c"][0] == 2 and "subtitle" not in meta:
        meta["subtitle"] = {"t": "MetaInlines", "c": blocks.pop(0)["c"][2]}
        promoted.append("subtitle")
    return promoted


def ast_counts(ast: dict) -> Counter:
    c: Counter = Counter()

    def count(el: dict) -> None:
        c[el["t"]] += 1
        if el["t"] == "Header":
            c[f"Header{el['c'][0]}"] += 1

    walk(ast["blocks"], count)
    return c


def write_docx(pandoc: Path, ast: dict, reference: Path, resource_dir: Path, epoch: int) -> bytes:
    env = dict(os.environ, SOURCE_DATE_EPOCH=str(epoch))
    with tempfile.TemporaryDirectory(prefix="build_docx_") as tmp:
        out = Path(tmp) / "pandoc.docx"
        subprocess.run([str(pandoc), "-f", "json", "-t", "docx", "--reference-doc", str(reference),
                        "--resource-path", str(resource_dir), "-o", str(out)],
                       input=json.dumps(ast).encode(), env=env, check=True)
        return out.read_bytes()


# --------------------------------------------------------------------------- docx post-processing

def _style_id(p: etree._Element) -> str | None:
    ps = p.find("w:pPr/w:pStyle", NS)
    return None if ps is None else ps.get(qn("w:val"))


def _set_style(p: etree._Element, style_id: str) -> None:
    ppr = p.find(qn("w:pPr"))
    if ppr is None:
        ppr = OxmlElement("w:pPr")
        p.insert(0, ppr)
    ps = ppr.find(qn("w:pStyle"))
    if ps is None:
        ps = OxmlElement("w:pStyle")
        ppr.insert(0, ps)
    ps.set(qn("w:val"), style_id)


def _text(p: etree._Element) -> str:
    return "".join(t.text or "" for t in p.iter(qn("w:t")))


def _is_figure(p: etree._Element) -> bool:
    return p.tag == qn("w:p") and p.find(".//w:drawing", NS) is not None and not _text(p).strip()


def fix_style_fonts(document) -> dict[str, int]:
    """Remove theme font attributes; give every style lacking an East Asian font the matching family's fonts."""
    root = document.styles.element
    stripped = ref.strip_theme_fonts(root)

    def fonts_of(el) -> dict[str, str] | None:
        rf = None if el is None else el.find(".//w:rFonts", NS)
        if rf is None or rf.get(qn("w:eastAsia")) is None:
            return None
        return {k: rf.get(qn(f"w:{k}")) for k in ("ascii", "hAnsi", "eastAsia", "cs") if rf.get(qn(f"w:{k}"))}

    def style_el(sid: str):
        return root.find(f"w:style[@w:styleId='{sid}']", NS)

    defaults = root.find("w:docDefaults/w:rPrDefault/w:rPr", NS)
    body = fonts_of(defaults)
    if body is None:                          # template without East Asian default: fall back to our fonts
        ref.set_rfonts(defaults, "body")
        body = fonts_of(defaults)
    families = {"body": body,
                "mono": fonts_of(style_el("VerbatimChar")) or body,
                "heading": fonts_of(style_el("Heading1")) or body}
    added = 0
    for s in root.iter(qn("w:style")):
        sid = s.get(qn("w:styleId"))
        if sid in ref.SKIP_IDS or s.get(qn("w:type")) == "numbering":
            continue
        rpr = s.find(qn("w:rPr"))
        rf = None if rpr is None else rpr.find(qn("w:rFonts"))
        if rf is not None and rf.get(qn("w:eastAsia")):
            continue
        if rpr is None:
            rpr = OxmlElement("w:rPr")
            nxt = next((c for c in s if c.tag in {qn("w:tblPr"), qn("w:trPr"), qn("w:tcPr"),
                                                  qn("w:tblStylePr")}), None)
            nxt.addprevious(rpr) if nxt is not None else s.append(rpr)
        if rf is None:
            rf = OxmlElement("w:rFonts")
            rpr.insert(1 if rpr.find(qn("w:rStyle")) is not None else 0, rf)
        for k, v in families[ref.family_of(sid)].items():
            if rf.get(qn(f"w:{k}")) is None or k == "eastAsia":
                rf.set(qn(f"w:{k}"), v)
        added += 1
    return {"theme_attrs_removed": stripped, "styles_given_east_asia": added}


def text_width_emu(document) -> int:
    sec = document.sections[-1]
    return int(sec.page_width - sec.left_margin - sec.right_margin)


def fix_figures(body: etree._Element, max_cx: int, has_style: set[str]) -> dict[str, int]:
    figures = scaled = captions = 0
    for p in body.iterchildren(qn("w:p")):
        if not _is_figure(p):
            continue
        figures += 1
        if "Figure" in has_style:
            _set_style(p, "Figure")
        for inline in p.iterfind(".//wp:inline", NS):
            ext = inline.find("wp:extent", NS)
            cx, cy = int(ext.get("cx")), int(ext.get("cy"))
            if cx > max_cx:
                ncx, ncy = max_cx, round(cy * max_cx / cx)
                ext.set("cx", str(ncx))
                ext.set("cy", str(ncy))
                for a_ext in inline.iterfind(".//pic:spPr/a:xfrm/a:ext", NS):
                    a_ext.set("cx", str(ncx))
                    a_ext.set("cy", str(ncy))
                scaled += 1
        nxt = p.getnext()
        if nxt is not None and nxt.tag == qn("w:p") and FIG_CAPTION_RE.match(_text(nxt)):
            if "ImageCaption" in has_style:
                _set_style(nxt, "ImageCaption")
            captions += 1
    return {"figures": figures, "figures_scaled": scaled, "figure_captions": captions}


def _table_style_borders(document) -> tuple[etree._Element | None, etree._Element | None]:
    st = document.styles.element.find("w:style[@w:styleId='Table']", NS)
    if st is None:
        return None, None
    tbl_borders = st.find("w:tblPr/w:tblBorders", NS)
    first_row = st.find("w:tblStylePr[@w:type='firstRow']/w:tcPr/w:tcBorders", NS)
    return tbl_borders, first_row


TBLPR_ORDER = ("tblStyle", "tblpPr", "tblOverlap", "bidiVisual", "tblStyleRowBandSize", "tblStyleColBandSize",
               "tblW", "jc", "tblCellSpacing", "tblInd", "tblBorders", "shd", "tblLayout", "tblCellMar", "tblLook",
               "tblCaption", "tblDescription")
TRPR_ORDER = ("cnfStyle", "divId", "gridBefore", "gridAfter", "wBefore", "wAfter", "cantSplit", "trHeight",
              "tblHeader", "tblCellSpacing", "jc", "hidden")
TCPR_ORDER = ("cnfStyle", "tcW", "gridSpan", "hMerge", "vMerge", "tcBorders", "shd", "noWrap", "tcMar",
              "textDirection", "tcFitText", "vAlign", "hideMark")


def _put(parent: etree._Element, child: etree._Element, order: tuple[str, ...]) -> None:
    """Insert or replace child in parent at its schema position."""
    name = etree.QName(child).localname
    old = parent.find(qn(f"w:{name}"))
    if old is not None:
        parent.replace(old, child)
        return
    idx = order.index(name)
    for sib in parent:
        if etree.QName(sib).localname in order[idx + 1:]:
            sib.addprevious(child)
            return
    parent.append(child)


def _flag(tag: str) -> etree._Element:
    return OxmlElement(f"w:{tag}")


def _display_width(text: str) -> int:
    """Width in half-width cells: Hangul and other wide characters count 2."""
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


def _auto_widths(tbl: etree._Element, ncols: int, avail: int) -> list[int] | None:
    """Column widths in half-width cells from the cell text, like HTML auto layout.

    pandoc derives widths from the dash counts of the Markdown separator row (or splits evenly), which leaves
    columns of long Korean text a few characters wide. Each column wants its longest paragraph; if the table
    does not fit, every column keeps at least its longest word (capped) and the rest is shared in proportion
    to what each column still wants.
    """
    want = [1] * ncols
    word = [1] * ncols
    for tr in tbl.findall(qn("w:tr")):
        tcs = tr.findall(qn("w:tc"))
        if len(tcs) != ncols:                         # merged cells: keep pandoc's grid
            return None
        for i, tc in enumerate(tcs):
            for p in tc.iter(qn("w:p")):
                text = _text(p)
                want[i] = max(want[i], _display_width(text))
                word[i] = max([word[i]] + [_display_width(w) for w in text.split()])
    if sum(want) <= avail:
        return want
    cap = max(MIN_COL_CELLS, avail // ncols)
    base = [min(want[i], max(MIN_COL_CELLS, min(word[i], cap))) for i in range(ncols)]
    rest = avail - sum(base)
    if rest <= 0:
        return base
    extra = [want[i] - base[i] for i in range(ncols)]
    total = sum(extra) or 1
    return [base[i] + rest * extra[i] / total for i in range(ncols)]


def _fit_grid(tbl: etree._Element, width_twips: int, cell_pt: float, cell_margin: int) -> None:
    """Set the column grid from the cell text and scale it to the text width; cells follow the grid."""
    cols = tbl.findall("w:tblGrid/w:gridCol", NS)
    if not cols:
        return
    unit = cell_pt * 10                               # twips per half-width cell (half an em)
    margin = 2 * cell_margin
    auto = _auto_widths(tbl, len(cols), int((width_twips - margin * len(cols)) / unit))
    widths = [int(c.get(qn("w:w"), "0")) for c in cols] if auto is None else [a * unit + margin for a in auto]
    total = sum(widths)
    if total <= 0:
        return
    scaled = [int(w * width_twips // total) for w in widths]
    scaled[-1] += width_twips - sum(scaled)          # rounding remainder to the last column
    for c, w in zip(cols, scaled):
        c.set(qn("w:w"), str(w))
    for tc in tbl.iter(qn("w:tc")):                  # stale cell widths would override the grid
        tcpr = tc.find(qn("w:tcPr"))
        if tcpr is not None and tcpr.find(qn("w:tcW")) is not None:
            tcpr.remove(tcpr.find(qn("w:tcW")))


def fix_tables(document, body: etree._Element, has_style: set[str]) -> dict[str, int]:
    tbl_borders, header_borders = _table_style_borders(document)
    width_twips = text_width_emu(document) // EMU_PER_TWIP
    cell_pt = _style_size_pt(document, "TableText", 9.0)
    mar = document.styles.element.find("w:style[@w:styleId='Table']/w:tblPr/w:tblCellMar/w:left", NS)
    cell_margin = int(mar.get(qn("w:w"))) if mar is not None else 108     # 108 twips: Word's default
    tables = captions = 0
    for tbl in body.iterchildren(qn("w:tbl")):
        tables += 1
        tblpr = tbl.find(qn("w:tblPr"))
        w = OxmlElement("w:tblW")
        w.set(qn("w:w"), str(width_twips))
        w.set(qn("w:type"), "dxa")
        _put(tblpr, w, TBLPR_ORDER)
        _fit_grid(tbl, width_twips, cell_pt, cell_margin)
        if tbl_borders is not None:
            _put(tblpr, etree.fromstring(etree.tostring(tbl_borders)), TBLPR_ORDER)
        rows = tbl.findall(qn("w:tr"))
        for i, tr in enumerate(rows):
            trpr = tr.find(qn("w:trPr"))
            if trpr is None:
                trpr = OxmlElement("w:trPr")
                tr.insert(1 if tr.find(qn("w:tblPrEx")) is not None else 0, trpr)
            _put(trpr, _flag("cantSplit"), TRPR_ORDER)
            header = i == 0 and len(rows) > 1
            if header:
                _put(trpr, _flag("tblHeader"), TRPR_ORDER)
            for tc in tr.findall(qn("w:tc")):
                if header and header_borders is not None:
                    tcpr = tc.find(qn("w:tcPr"))
                    if tcpr is None:
                        tcpr = OxmlElement("w:tcPr")
                        tc.insert(0, tcpr)
                    _put(tcpr, etree.fromstring(etree.tostring(header_borders)), TCPR_ORDER)
                for p in tc.iterchildren(qn("w:p")):
                    want = "TableHeading" if header else "TableText"
                    if _style_id(p) in BODY_STYLES_IN_CELLS and want in has_style:
                        _set_style(p, want)
                    if header:                     # keep the header with the first body row
                        ppr = p.find(qn("w:pPr"))
                        if ppr.find(qn("w:keepNext")) is None:
                            ppr.find(qn("w:pStyle")).addnext(_flag("keepNext"))
        nxt = tbl.getnext()
        if nxt is not None and nxt.tag == qn("w:p") and not (_style_id(nxt) or "").startswith(("Heading", "Table")):
            _space_before(nxt, AFTER_TABLE_TWIPS)
        prev = tbl.getprevious()
        if prev is not None and prev.tag == qn("w:p") and TABLE_CAPTION_RE.match(_text(prev)):
            if "TableCaption" in has_style:
                _set_style(prev, "TableCaption")
            captions += 1
    return {"tables": tables, "table_captions": captions}


def _is_narrowed(ch: str) -> bool:
    return ord(ch) > 0x7F and unicodedata.east_asian_width(ch) == "A"


PPR_ORDER = ("pStyle", "keepNext", "keepLines", "pageBreakBefore", "framePr", "widowControl", "numPr",
             "suppressLineNumbers", "pBdr", "shd", "tabs", "suppressAutoHyphens", "kinsoku", "wordWrap",
             "overflowPunct", "topLinePunct", "autoSpaceDE", "autoSpaceDN", "bidi", "adjustRightInd", "snapToGrid",
             "spacing", "ind", "contextualSpacing", "mirrorIndents", "suppressOverlap", "jc", "textDirection",
             "textAlignment", "textboxTightWrap", "outlineLvl", "divId", "cnfStyle", "rPr", "sectPr", "pPrChange")


def _space_before(p: etree._Element, twips: int) -> None:
    ppr = p.find(qn("w:pPr"))
    if ppr is None:
        ppr = OxmlElement("w:pPr")
        p.insert(0, ppr)
    sp = ppr.find(qn("w:spacing"))
    if sp is None:
        sp = OxmlElement("w:spacing")
        _put(ppr, sp, PPR_ORDER)
    sp.set(qn("w:before"), str(twips))


def _style_size_pt(document, style_id: str, default: float) -> float:
    root = document.styles.element
    sid: str | None = style_id
    while sid:
        st = root.find(f"w:style[@w:styleId='{sid}']", NS)
        if st is None:
            break
        sz = st.find("w:rPr/w:sz", NS)
        if sz is not None:
            return int(sz.get(qn("w:val"))) / 2
        based = st.find("w:basedOn", NS)
        sid = None if based is None else based.get(qn("w:val"))
    return default


def fix_rules(body: etree._Element, has_style: set[str]) -> dict[str, int]:
    """pandoc writes '---' as a VML rectangle (drawn as a heavy double bar by LibreOffice); use a bordered
    empty paragraph of the template's Horizontal Rule style instead."""
    n = 0
    if "HorizontalRule" not in has_style:
        return {"rules": 0}
    for p in body.iterchildren(qn("w:p")):
        rects = p.findall(".//v:rect", {"v": "urn:schemas-microsoft-com:vml"})
        if not rects or _text(p).strip() or not all(
                r.get("{urn:schemas-microsoft-com:office:office}hr") == "t" for r in rects):
            continue
        for child in list(p):
            p.remove(child)
        _set_style(p, "HorizontalRule")
        n += 1
    return {"rules": n}


def fix_bibliography(body: etree._Element, has_style: set[str]) -> dict[str, int]:
    """Body paragraphs under a "참고문헌" / "References" heading get the Bibliography style (left aligned,
    hanging indent; justified lines with long URLs would spread out)."""
    if "Bibliography" not in has_style:
        return {"bibliography_entries": 0}
    level = None
    n = 0
    for p in body.iterchildren(qn("w:p")):
        sid = _style_id(p) or ""
        if re.fullmatch(r"Heading\d", sid):
            lvl = int(sid[-1])
            if level is not None and lvl <= level:
                level = None
            if _text(p).strip().lower() in REFERENCE_HEADINGS:
                level = lvl
            continue
        if level is not None and sid in ("BodyText", "FirstParagraph", ""):
            _set_style(p, "Bibliography")
            n += 1
    return {"bibliography_entries": n}


def _resolved_fonts(document) -> Callable[[str | None], tuple[str, str]]:
    """Function: paragraph style id -> (latin font, East Asian font), following basedOn down to docDefaults."""
    root = document.styles.element
    defaults = root.find("w:docDefaults/w:rPrDefault/w:rPr/w:rFonts", NS)
    default_para = root.find("w:style[@w:type='paragraph'][@w:default='1']", NS)
    cache: dict[str | None, tuple[str, str]] = {}

    def resolve(sid: str | None) -> tuple[str, str]:
        if sid in cache:
            return cache[sid]
        latin = east = None
        cur = sid if sid else (default_para.get(qn("w:styleId")) if default_para is not None else None)
        seen = set()
        while cur and cur not in seen and (latin is None or east is None):
            seen.add(cur)
            st = root.find(f"w:style[@w:styleId='{cur}']", NS)
            if st is None:
                break
            rf = st.find("w:rPr/w:rFonts", NS)
            if rf is not None:
                latin = latin or rf.get(qn("w:ascii"))
                east = east or rf.get(qn("w:eastAsia"))
            based = st.find("w:basedOn", NS)
            cur = None if based is None else based.get(qn("w:val"))
        if defaults is not None:
            latin = latin or defaults.get(qn("w:ascii"))
            east = east or defaults.get(qn("w:eastAsia"))
        cache[sid] = (latin or "", east or "")
        return cache[sid]

    return resolve


def _symbol_class(ch: str) -> str | None:
    cp = ord(ch)
    if any(a <= cp <= b for a, b in EAST_ASIAN_ONLY):
        return "east"
    if any(a <= cp <= b for a, b in LATIN_ONLY):
        return "latin"
    return None


def _split_runs(p: etree._Element, classify, decorate) -> int:
    """Split each plain text run of p into maximal segments of equal classify(ch); decorate(rPr, cls) is
    called for every segment whose class is not None. Returns the number of decorated characters."""
    changed = 0
    for r in list(p.iter(qn("w:r"))):
        ts = r.findall(qn("w:t"))
        rpr = r.find(qn("w:rPr"))
        if len(ts) != 1 or len(r) != (2 if rpr is not None else 1):
            continue                      # only plain text runs (breaks, fields and drawings are untouched)
        text = ts[0].text or ""
        if not any(classify(ch) for ch in text):
            continue
        segs: list[list] = []
        for ch in text:
            c = classify(ch)
            if segs and segs[-1][0] == c:
                segs[-1][1] += ch
            else:
                segs.append([c, ch])
        for cls, seg in segs:
            nr = etree.fromstring(etree.tostring(r))
            nt = nr.find(qn("w:t"))
            nt.text = seg
            nt.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
            if cls is not None:
                nrpr = nr.find(qn("w:rPr"))
                if nrpr is None:
                    nrpr = OxmlElement("w:rPr")
                    nr.insert(0, nrpr)
                decorate(nrpr, cls)
                changed += len(seg)
            r.addprevious(nr)
        r.getparent().remove(r)
    return changed


def fix_symbol_fonts(document, body: etree._Element) -> dict[str, int]:
    """Give symbols that only the Korean (or only the Latin) font has that font in every slot."""
    resolve = _resolved_fonts(document)
    n = 0
    for p in body.iter(qn("w:p")):
        sid = _style_id(p)
        if sid == "SourceCode":
            continue
        latin, east = resolve(sid)

        def decorate(rpr: etree._Element, cls: str, latin=latin, east=east) -> None:
            rs = rpr.find(qn("w:rStyle"))
            if rs is not None and (rs.get(qn("w:val")) in CODE_RUN_STYLES or rs.get(qn("w:val")).endswith("Tok")):
                return                      # code font covers these symbols
            font = east if cls == "east" else latin
            rf = OxmlElement("w:rFonts")
            for k in ("ascii", "hAnsi", "eastAsia", "cs"):
                rf.set(qn(f"w:{k}"), font)
            _put_rpr(rpr, rf)

        n += _split_runs(p, _symbol_class, decorate)
    return {"symbol_chars_refonted": n}


def fix_code_blocks(document, body: etree._Element) -> dict[str, int]:
    """Split code runs so East-Asian-ambiguous characters are drawn at 50 % width (see module docstring)."""
    vc = document.styles.element.find("w:style[@w:styleId='VerbatimChar']/w:rPr/w:rFonts", NS)
    single_cjk_mono = vc is not None and vc.get(qn("w:ascii")) and vc.get(qn("w:ascii")) == vc.get(qn("w:eastAsia"))
    blocks = narrowed = 0
    for p in body.iter(qn("w:p")):
        if _style_id(p) != "SourceCode":
            continue
        blocks += 1
        if not single_cjk_mono:
            continue
        narrowed += _split_runs(p, lambda ch: "narrow" if _is_narrowed(ch) else None, _narrow)
    return {"code_blocks": blocks, "code_chars_narrowed": narrowed}


RPR_ORDER = ("rStyle", "rFonts", "b", "bCs", "i", "iCs", "caps", "smallCaps", "strike", "dstrike", "outline",
             "shadow", "emboss", "imprint", "noProof", "snapToGrid", "vanish", "webHidden", "color", "spacing", "w",
             "kern", "position", "sz", "szCs", "highlight", "u", "effect", "bdr", "shd", "fitText", "vertAlign",
             "rtl", "cs", "em", "lang", "eastAsianLayout", "specVanish", "oMath")


def _put_rpr(rpr: etree._Element, child: etree._Element) -> None:
    _put(rpr, child, RPR_ORDER)


def _narrow(rpr: etree._Element, _cls: str) -> None:
    wel = OxmlElement("w:w")
    wel.set(qn("w:val"), "50")
    _put_rpr(rpr, wel)


def docx_counts(document) -> Counter:
    body = document.element.body
    c: Counter = Counter()
    for p in body.iter(qn("w:p")):
        c["paragraphs"] += 1
        sid = _style_id(p)
        if sid and re.fullmatch(r"Heading\d", sid):
            c["headings"] += 1
            c[f"Header{sid[-1]}"] += 1
        if sid in ("Title", "Subtitle", "ImageCaption", "TableCaption", "SourceCode", "BlockText", "HorizontalRule",
                   "Bibliography"):
            c[sid] += 1
    c["tables"] = len(body.findall(".//w:tbl", NS))
    c["images"] = len(body.findall(".//w:drawing", NS))
    return c


def postprocess(data: bytes, date: dt.datetime, title: str) -> tuple[bytes, dict[str, int], Counter]:
    document = docx.Document(io.BytesIO(data))
    has_style = {s.get(qn("w:styleId")) for s in document.styles.element.iter(qn("w:style"))}
    body = document.element.body
    stats: dict[str, int] = {}
    stats |= fix_style_fonts(document)
    stats |= fix_figures(body, text_width_emu(document), has_style)
    stats |= fix_tables(document, body, has_style)
    stats |= fix_code_blocks(document, body)
    stats |= fix_rules(body, has_style)
    stats |= fix_bibliography(body, has_style)
    stats |= fix_symbol_fonts(document, body)          # last: needs the final paragraph styles
    cp = document.core_properties
    cp.title = title
    cp.last_modified_by = ""
    cp.revision = 1
    cp.created = cp.modified = date.replace(tzinfo=None)
    buf = io.BytesIO()
    document.save(buf)
    return ref.deterministic_zip(buf.getvalue()), stats, docx_counts(document)


# --------------------------------------------------------------------------- main

def _plain(inlines: list) -> str:
    out: list[str] = []

    def take(el: dict) -> None:
        if el["t"] == "Str":
            out.append(el["c"])
        elif el["t"] in ("Space", "SoftBreak"):
            out.append(" ")

    walk(inlines, take)
    return "".join(out)


def build(md: Path, reference: Path, out: Path, pandoc: Path, date: dt.datetime) -> tuple[dict, Counter, Counter]:
    for path, what in ((md, "manuscript"), (reference, "reference docx")):
        if not path.is_file():
            hint = " (generate it with 06_tools/python/paper/make_reference_docx.py --out ...)" \
                if what == "reference docx" else ""
            raise ManuscriptError(f"{what} not found: {path}{hint}")
    ast = read_ast(pandoc, md)
    svg_to_png(ast, md.parent)
    promoted = promote_title(ast)
    md_counts = ast_counts(ast)
    md_counts["promoted"] = len(promoted)
    title = _plain(ast["meta"]["title"]["c"]) if "title" in ast["meta"] else ""
    data = write_docx(pandoc, ast, reference, md.parent, int(date.timestamp()))
    data, stats, dx = postprocess(data, date, title)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    return stats, md_counts, dx


def check_counts(md: Counter, dx: Counter) -> list[str]:
    problems = []
    pairs = [("headings", sum(md[f"Header{i}"] for i in range(1, 7)), dx["headings"]),
             ("tables", md["Table"], dx["tables"]),
             ("images", md["Image"], dx["images"]),
             ("code blocks", md["CodeBlock"], dx["SourceCode"]),
             ("title+subtitle", md["promoted"], dx["Title"] + dx["Subtitle"])]
    for name, a, b in pairs:
        if a != b:
            problems.append(f"{name}: manuscript {a}, docx {b}")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Convert the paper Markdown to docx with the style template.")
    ap.add_argument("--md", type=Path, default=DEFAULT_MD, help=f"manuscript (default {DEFAULT_MD})")
    ap.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE,
                    help=f"style template docx (default {DEFAULT_REFERENCE})")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"output docx (default {DEFAULT_OUT})")
    ap.add_argument("--pandoc", default=None, help="pandoc executable (default: python-dev env pandoc, then PATH)")
    ap.add_argument("--date", default=DEFAULT_DATE,
                    help=f"document created/modified date, YYYY-MM-DD (default {DEFAULT_DATE}; fixed for "
                         "reproducible output)")
    a = ap.parse_args(argv)
    try:
        date = dt.datetime.strptime(a.date, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc)
    except ValueError:
        ap.error(f"--date must be YYYY-MM-DD, got {a.date!r}")
    try:
        pandoc = ref.find_pandoc(a.pandoc)
        out = a.out.resolve()
        stats, md, dx = build(a.md.resolve(), a.reference.resolve(), out, pandoc, date)
    except ref.ToolError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except subprocess.CalledProcessError as e:
        err = e.stderr.decode(errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
        print(f"error: {' '.join(map(str, e.cmd[:3]))} ... failed ({e.returncode}) {err}", file=sys.stderr)
        return 2

    print(f"wrote: {out}")
    print(f"pandoc: {pandoc}")
    print("page count: unknown without rendering (e.g. soffice --headless --convert-to pdf)")
    levels = " ".join(f"H{i}={dx[f'Header{i}']}" for i in range(1, 7) if dx[f"Header{i}"])
    print(f"docx: paragraphs={dx['paragraphs']} headings={dx['headings']} ({levels}) title={dx['Title']} "
          f"subtitle={dx['Subtitle']} tables={dx['tables']} images={dx['images']} "
          f"figure_captions={dx['ImageCaption']} table_captions={dx['TableCaption']} "
          f"code_blocks={dx['SourceCode']} block_quotes(paragraphs)={dx['BlockText']}")
    print("post-processing: " + " ".join(f"{k}={v}" for k, v in stats.items()))
    notable = " ".join(f"{t}={md[t]}" for t in NOTABLE if md[t])
    print(f"manuscript constructs worth a look in the docx: {notable or 'none'}")
    problems = check_counts(md, dx)
    for p in problems:
        print(f"error: count mismatch, {p}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
