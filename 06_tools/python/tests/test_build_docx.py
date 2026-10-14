"""Tests for 06_tools/python/paper/{build_docx,make_reference_docx}.py (stdlib unittest, no pytest).

Run: /home/pi/miniconda3/envs/python-dev/bin/python -m unittest discover -s 06_tools/python/tests \
         -p 'test_build_docx.py'
Needs python-docx, lxml and pandoc >= 3 (conda env python-dev); skipped elsewhere. A small Markdown file is
converted in a temporary folder; 10_paper is not read or written.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import struct
import sys
import tempfile
import unittest
import zipfile
import zlib
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1] / "paper"
HAVE_DOCX = importlib.util.find_spec("docx") is not None and importlib.util.find_spec("lxml") is not None

if HAVE_DOCX:
    sys.path.insert(0, str(PAPER))
    import build_docx as bd  # noqa: E402
    import make_reference_docx as mr  # noqa: E402
    from lxml import etree  # noqa: E402

    try:
        PANDOC = mr.find_pandoc()
    except mr.ToolError:
        PANDOC = None
else:
    PANDOC = None

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

MANUSCRIPT = """# 시험 논문 제목
## — 부제

**English title**

# 1. 서론

본문 문단이다. Latin ① 다음과 값 −2.3 사이. 범위 300~350 MHz 와 "따옴표" -- 그대로.

| 항목 | 아주 긴 설명이 들어가는 열 | 값 |
|---|---|---:|
| CPU | 프레임당 처리 시간을 줄였다 | 1.842 |
| GPU | 같음 | 0.5 |

![그림 1. 시험 그림](figures/fig01_test.svg)

**그림 1.** 시험 그림의 캡션.

```
free ──▶ 전처리 ─┐
                 ▼
```

---

# 참고문헌

[R1] A. Author, "Title," 2026. Available: https://example.org/very/long/url

# 부록 A. 끝

마지막 문단.
"""


def tiny_png(width: int, height: int) -> bytes:
    """Gray PNG written with the standard library only."""
    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    raw = b"".join(b"\x00" + b"\x80" * width for _ in range(height))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


@unittest.skipUnless(HAVE_DOCX and PANDOC is not None, "python-docx/lxml/pandoc>=3 missing (use conda env python-dev)")
class BuildDocxTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory(prefix="test_build_docx_")
        root = Path(cls.tmp.name)
        (root / "figures").mkdir()
        (root / "figures" / "fig01_test.png").write_bytes(tiny_png(3000, 1000))   # wider than the text
        cls.md = root / "paper.md"
        cls.md.write_text(MANUSCRIPT, encoding="utf-8")
        cls.reference = mr.build_reference(root / "reference.docx", PANDOC)
        cls.out = root / "out" / "paper.docx"
        cls.rc, cls.stdout = cls.run_build(cls.out)
        z = zipfile.ZipFile(cls.out)
        parser = etree.XMLParser(resolve_entities=False)
        cls.doc = etree.fromstring(z.read("word/document.xml"), parser)
        cls.styles = etree.fromstring(z.read("word/styles.xml"), parser)
        cls.zip_infos = z.infolist()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    @classmethod
    def run_build(cls, out: Path, md: Path | None = None) -> tuple[int, str]:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
            rc = bd.main(["--md", str(md or cls.md), "--reference", str(cls.reference), "--out", str(out)])
        return rc, buf.getvalue()

    def paras(self, style: str) -> list:
        return [p for p in self.doc.iter(W + "p")
                if p.find(f"{W}pPr/{W}pStyle[@{W}val='{style}']") is not None]

    def text(self, p) -> str:
        return "".join(t.text or "" for t in p.iter(W + "t"))

    def test_counts_and_summary(self) -> None:
        self.assertEqual(self.rc, 0, self.stdout)
        self.assertIn("headings=3", self.stdout)          # title and subtitle moved out of the headings
        self.assertEqual(len(self.paras("Title")), 1)
        self.assertEqual(len(self.paras("Subtitle")), 1)
        self.assertEqual(len(self.doc.findall(f".//{W}tbl")), 1)

    def test_every_style_has_east_asian_font_and_no_theme_fonts(self) -> None:
        for rf in self.styles.iter(W + "rFonts"):
            for a in ("asciiTheme", "hAnsiTheme", "eastAsiaTheme", "cstheme"):
                self.assertIsNone(rf.get(W + a))
        defaults = self.styles.find(f"{W}docDefaults/{W}rPrDefault/{W}rPr/{W}rFonts")
        self.assertEqual(defaults.get(W + "eastAsia"), mr.FONTS["body"]["east_asia"])
        for s in self.styles.iter(W + "style"):
            if s.get(W + "styleId") in mr.SKIP_IDS:
                continue
            rf = s.find(f"{W}rPr/{W}rFonts")
            self.assertIsNotNone(rf, s.get(W + "styleId"))
            self.assertTrue(rf.get(W + "eastAsia"), s.get(W + "styleId"))

    def test_figure_fits_and_caption_styled(self) -> None:
        figs = self.paras("Figure")
        self.assertEqual(len(figs), 1)
        ext = figs[0].find(".//{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}extent")
        text_width = (mr.PAGE_W_MM - 2 * mr.MARGIN_MM) * 36000
        self.assertLessEqual(int(ext.get("cx")), text_width + 36000)       # within 1 mm of the text width
        self.assertAlmostEqual(int(ext.get("cx")) / int(ext.get("cy")), 3.0, places=2)
        caps = self.paras("ImageCaption")
        self.assertEqual(len(caps), 1)
        self.assertTrue(self.text(caps[0]).startswith("그림 1."))

    def test_table_header_borders_and_widths(self) -> None:
        tbl = self.doc.find(f".//{W}tbl")
        first = tbl.find(f"{W}tr")
        self.assertIsNotNone(first.find(f"{W}trPr/{W}tblHeader"))
        self.assertIsNotNone(tbl.find(f"{W}tblPr/{W}tblBorders/{W}top"))
        styles = {p.find(f"{W}pPr/{W}pStyle").get(W + "val") for p in first.iter(W + "p")}
        self.assertEqual(styles, {"TableHeading"})
        cols = [int(c.get(W + "w")) for c in tbl.iter(W + "gridCol")]
        sect = self.doc.find(f".//{W}sectPr")
        mar = sect.find(f"{W}pgMar")
        text_width = int(sect.find(f"{W}pgSz").get(W + "w")) - int(mar.get(W + "left")) - int(mar.get(W + "right"))
        self.assertEqual(sum(cols), text_width)
        self.assertGreater(cols[1], cols[2])                 # the long-text column gets more room

    def test_text_kept_as_typed(self) -> None:
        body = " ".join(self.text(p) for p in self.doc.iter(W + "p"))
        self.assertIn("300~350 MHz", body)                  # no subscript
        self.assertIn('"따옴표" --', body)                   # no smart quotes or dashes

    def test_symbols_code_rules_and_references(self) -> None:
        refonted = [r for r in self.doc.iter(W + "r") if self.text(r) in ("①", "−")]
        self.assertEqual(len(refonted), 2)
        for r in refonted:
            rf = r.find(f"{W}rPr/{W}rFonts")
            self.assertEqual(rf.get(W + "ascii"), rf.get(W + "eastAsia"))
        narrowed = [r for r in self.paras("SourceCode")[0].iter(W + "r") if r.find(f"{W}rPr/{W}w") is not None]
        self.assertTrue(narrowed)
        self.assertTrue(all(ord(ch) > 0x7F for r in narrowed for ch in self.text(r)))
        self.assertEqual(len(self.paras("HorizontalRule")), 1)
        bib = self.paras("Bibliography")
        self.assertEqual([self.text(p)[:4] for p in bib], ["[R1]"])     # the appendix after it is not touched

    def test_reproducible_bytes_and_fixed_zip_times(self) -> None:
        again = Path(self.tmp.name) / "again.docx"
        rc, _ = self.run_build(again)
        self.assertEqual(rc, 0)
        self.assertEqual(again.read_bytes(), self.out.read_bytes())
        self.assertTrue(all(i.date_time == mr.ZIP_DATE for i in self.zip_infos))
        self.assertEqual(self.zip_infos[0].filename, "[Content_Types].xml")

    def test_missing_png_is_an_error(self) -> None:
        md = Path(self.tmp.name) / "missing.md"
        md.write_text("# T\n\n![x](figures/nope.svg)\n", encoding="utf-8")
        rc, _ = self.run_build(Path(self.tmp.name) / "missing.docx", md)
        self.assertEqual(rc, 2)
        self.assertFalse((Path(self.tmp.name) / "missing.docx").exists())


if __name__ == "__main__":
    unittest.main()
