"""Tests for 06_tools/python/check_links.py (stdlib unittest, no pytest).

Run: python3 -m unittest discover -s 06_tools/python/tests

Every test builds a small fake repo in a temp dir and points check_links.ROOT (and relocate's ROOT, REPO_ABS,
tracked() for --stale) at it. The live repo is not read.
"""
from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import check_links as cl  # noqa: E402
import relocate  # noqa: E402

# 20 lines; read_candidates() is on line 10, FooBar::emit on line 15
SOURCE_A = "\n".join(
    "void read_candidates() {}" if i == 10 else "struct FooBar { void emit(); };" if i == 15 else f"// line {i}"
    for i in range(1, 21)) + "\n"


class FakeRepo(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name).resolve()
        p = mock.patch.object(cl, "ROOT", self.root)
        p.start()
        self.addCleanup(p.stop)
        self.write("01_src/core/a.cpp", SOURCE_A)

    def write(self, rel: str, text: str) -> Path:
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        return p

    def run_main(self, *args: str) -> tuple[int, list[str]]:
        out = io.StringIO()
        with mock.patch.object(sys, "argv", ["check_links.py", *args]), contextlib.redirect_stdout(out):
            rc = cl.main()
        return rc, out.getvalue().splitlines()


class LinkModeTest(FakeRepo):
    def test_clean_links_exit_0(self) -> None:
        self.write("08_learn/a.md", "[src](../01_src/core/a.cpp) [l](../01_src/core/a.cpp:20) "
                                    "[web](https://x.org) [anchor](#top) [glob](../01_src/*.cpp)\n")
        rc, out = self.run_main()
        self.assertEqual(rc, 0)
        self.assertEqual(out[-1], "checked 2, broken 0")

    def test_missing_target_and_line_out_of_range(self) -> None:
        self.write("08_learn/a.md", "[gone](../01_src/core/gone.cpp)\n[far](../01_src/core/a.cpp:21-30)\n")
        self.write("build/skipped.md", "[x](nowhere.md)\n")   # build output is not scanned
        rc, out = self.run_main()
        self.assertEqual(rc, 1)
        self.assertIn("missing target   08_learn/a.md -> ../01_src/core/gone.cpp", out)
        self.assertIn("line out of range (file has 20 lines) 08_learn/a.md -> ../01_src/core/a.cpp:21-30", out)
        self.assertEqual(out[-1], "checked 2, broken 2")

    def test_usage_errors(self) -> None:
        self.assertEqual(self.run_main("--lines", "extra")[0], 2)
        self.assertEqual(self.run_main("--stale")[0], 2)


class LinesModeMarkdownTest(FakeRepo):
    def setUp(self) -> None:
        super().setUp()
        self.write("01_src/x/dup.hpp", "a\nb\nc\n")
        self.write("01_src/y/dup.hpp", "a\nb\nc\n")

    def test_each_verdict(self) -> None:
        self.write("08_learn/doc.md", textwrap.dedent("""\
            ok: `01_src/core/a.cpp:10` `read_candidates()`
            ok by suffix and method name: `core/a.cpp:14-16` `FooBar::emit`
            drift: `a.cpp:3` `read_candidates()`
            out of range: `01_src/core/a.cpp:99`
            manual, no identifier: `01_src/core/a.cpp:10`
            manual continuation: `01_src/core/a.cpp:3` and `:15` `missing_name()`
            ambiguous: `dup.hpp:2` `something`
            external: `hailo/hailort.hpp:120` `InferModel`
            """))
        self.write("07_results/exp/summary.md", "`01_src/core/a.cpp:99` is history, never checked\n")
        rc, out = self.run_main("--lines")
        self.assertEqual(rc, 0)   # warnings only
        self.assertIn("drift  08_learn/doc.md:3 -> 01_src/core/a.cpp:3  'read_candidates' not within ±5 lines", out)
        self.assertIn("drift  08_learn/doc.md:4 -> 01_src/core/a.cpp:99  line out of range (file has 20 lines)", out)
        self.assertTrue(any(s.startswith("ambiguous  08_learn/doc.md:7 -> dup.hpp:2  (2 candidates:") for s in out))
        # n = 2 ok + 2 drift + 1 ambiguous + 3 manual (no ident, continuation, miss beside continuation) + 1 external
        self.assertEqual(out[-2], "--- 9 citations checked, 2 drift (warning), 1 ambiguous, 3 manual, 1 external skipped")
        self.assertEqual(out[-1], "--- graphml: 0 citations checked, 0 drift (warning), 0 ambiguous, 0 manual, "
                                  "0 external skipped (0 files)")

    def test_link_text_and_target_count_once_and_ambiguous_resolved_by_full_path(self) -> None:
        self.write("README.md", "[`a.cpp:10`](01_src/core/a.cpp:10) `read_candidates()`\n"
                                "[`dup.hpp:2`](01_src/x/dup.hpp:2)\n")
        _, out = self.run_main("--lines")
        self.assertEqual(out[-2], "--- 4 citations checked, 0 drift (warning), 0 ambiguous, 1 manual, 0 external skipped")

    def test_identifiers(self) -> None:
        line = "`read()` `Class::method_name()` `x` `main.cpp` `a/b` `01_src/a.cpp:3` `--flag` `kConst`"
        self.assertEqual(sorted(cl.identifiers(line)), ["Class::method_name", "kConst", "method_name", "read"])


GRAPHML = """<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<graphml xmlns="http://graphml.graphdrawing.org/xmlns" xmlns:y="http://www.yworks.com/xml/graphml">
  <key attr.name="url" attr.type="string" for="node" id="d4"/>
  <key attr.name="description" attr.type="string" for="node" id="d5"/>
  <key for="node" id="d6" yfiles.type="nodegraphics"/>
  <key attr.name="description" attr.type="string" for="edge" id="d8"/>
  <graph edgedefault="directed" id="G">
    <node id="n0">
      <data key="d4"><![CDATA[@ROOT@/01_src/core/a.cpp:3 read_candidates()]]></data>
      <data key="d5">@ROOT@/01_src/core/a.cpp:10 read_candidates() is the entry &amp; more.
second line: 01_src/core/a.cpp:3-4 FooBar::emit is far away. Same sentence again, a.cpp:15 (:16) FooBar::emit.</data>
      <data key="d6">
        <y:ShapeNode>
          <y:NodeLabel modelName="internal">rule : 01_src/core/a.cpp:99<y:LabelModel/></y:NodeLabel>
        </y:ShapeNode>
      </data>
    </node>
    <edge id="e0" source="n0" target="n0">
      <data key="d8">a.cpp:2 and a.cpp:14 share_name; FooBar used near 15. hailo/hailort.hpp:5 Vulkan words only.</data>
    </edge>
  </graph>
</graphml>
"""


class LinesModeGraphmlTest(FakeRepo):
    def setUp(self) -> None:
        super().setUp()
        self.g = self.write("09_diagrams/p.graphml", GRAPHML.replace("@ROOT@", str(self.root)))

    def test_graphml_texts_lines_and_scope(self) -> None:
        texts = cl.graphml_texts(self.g)
        self.assertEqual([ln for ln, _ in texts], [10, 11, 14, 19])   # url data (d4, line 9) is not read
        self.assertTrue(texts[0][1].endswith("is the entry & more."))   # entities decoded
        self.assertEqual(texts[2][1], "rule : 01_src/core/a.cpp:99")      # NodeLabel text, not its children

    def test_plain_identifiers(self) -> None:
        got = cl.plain_identifiers("see /r/01_src/a.cpp:3 read_candidates() and FooBar::emit, VkQueue, Vulkan, "
                                   "main.cpp, flush(x), mutex, wl_buffer.release")
        self.assertEqual(got, ["read_candidates", "FooBar::emit", "emit", "VkQueue", "flush", "release",
                               "wl_buffer.release"])

    def test_cite_g_accepts_absolute_paths_and_cont_g_plain_continuations(self) -> None:
        line = f"{self.root}/01_src/core/a.cpp:3-4 x (main.cpp:7, :12) (평면 :33-46) 10:30 §5-3"
        self.assertEqual([m.group(1) for m in cl.CITE_G.finditer(line)], [f"{self.root}/01_src/core/a.cpp", "main.cpp"])
        self.assertEqual(cl.CONT_G.findall(line), [("12", ""), ("33", "46")])
        self.assertEqual([m.group(1) for m in cl.CITE.finditer(line)], ["main.cpp"])   # .md regex unchanged

    def test_lines_mode_on_graphml(self) -> None:
        self.write("08_learn/doc.md", "`01_src/core/a.cpp:10` `read_candidates()`\n")
        rc, out = self.run_main("--lines")
        self.assertEqual(rc, 0)
        drift = [s for s in out if s.startswith("drift")]
        self.assertEqual(drift, [
            # sentence 1 of line 11: emit is on line 15, outside 3-4 +-5
            "drift  09_diagrams/p.graphml:11 -> 01_src/core/a.cpp:3-4  'FooBar::emit' / 'emit' not within ±5 lines",
            "drift  09_diagrams/p.graphml:14 -> 01_src/core/a.cpp:99  line out of range (file has 20 lines)",
            # 'share_name' is in no citation of the sentence -> both are drift
            "drift  09_diagrams/p.graphml:19 -> 01_src/core/a.cpp:2  'share_name' not within ±5 lines",
            "drift  09_diagrams/p.graphml:19 -> 01_src/core/a.cpp:14  'share_name' not within ±5 lines",
        ])
        # md summary line is unaffected by the graphml scope
        self.assertEqual(out[-2], "--- 1 citations checked, 0 drift (warning), 0 ambiguous, 0 manual, 0 external skipped")
        # ok: line 10 (absolute path) and a.cpp:15 of line 11's sentence 2, whose (:16) continuation is manual
        self.assertEqual(out[-1], "--- graphml: 8 citations checked, 4 drift (warning), 0 ambiguous, 1 manual, "
                                  "1 external skipped (1 files)")

    def test_shared_identifier_in_a_sentence_makes_a_miss_manual(self) -> None:
        self.g.write_text(GRAPHML.replace("@ROOT@", str(self.root)).replace("share_name", "FooBar::emit"),
                          encoding="utf-8")
        _, out = self.run_main("--lines")
        self.assertNotIn("09_diagrams/p.graphml:19", "\n".join(s for s in out if s.startswith("drift")))
        # a.cpp:14 finds FooBar::emit, so the a.cpp:2 miss in the same sentence is manual
        self.assertEqual(out[-1], "--- graphml: 8 citations checked, 2 drift (warning), 0 ambiguous, 2 manual, "
                                  "1 external skipped (1 files)")

    def test_unreadable_graphml_is_reported_not_fatal(self) -> None:
        self.write("09_diagrams/broken.graphml", "<graphml><node>")
        rc, out = self.run_main("--lines")
        self.assertEqual(rc, 0)
        self.assertTrue(any(s.startswith("unreadable  09_diagrams/broken.graphml") for s in out))


class StaleModeTest(FakeRepo):
    MAP = textwrap.dedent("""\
        # map

        | 옛 경로 | 새 경로 | 규칙 | 이유 |
        |---|---|---|---|
        | `07_results/old_exp/` | `07_results/new_exp/` | dir | renamed |
        """)

    def setUp(self) -> None:
        super().setUp()
        self.write("STRUCTURE.md", self.MAP)
        self.write("07_results/new_exp/summary.md", "x\n")
        self.files = ["STRUCTURE.md", "07_results/new_exp/summary.md", "README.md"]
        for name, value in (("ROOT", self.root), ("REPO_ABS", str(self.root)),
                            ("tracked", lambda: list(self.files))):
            p = mock.patch.object(relocate, name, value)
            p.start()
            self.addCleanup(p.stop)

    def test_old_path_remaining_exits_1(self) -> None:
        self.write("README.md", "see 07_results/old_exp/summary.md\n")
        rc, out = self.run_main("--stale", str(self.root / "STRUCTURE.md"))
        self.assertEqual(rc, 1)
        self.assertIn("old path remains  README.md:1 -> 07_results/old_exp/summary.md  "
                      "(new: 07_results/new_exp/summary.md)", out)
        self.assertEqual(out[-1], "1 mapping rows, 1 old paths remaining, 0 guessed (warning), 0 manual checks (warning)")

    def test_guess_and_manual_are_warnings_only(self) -> None:
        self.write("README.md", "prose old_exp/summary.md and a glob 07_results/old_exp/*.csv\n")
        (self.root / "STRUCTURE.md").rename(self.root / "map_fixture.md")   # only the map file's table is excluded
        self.files[0] = "map_fixture.md"
        rc, out = self.run_main("--stale", "map_fixture.md")   # relative and not in the CWD: resolved against ROOT
        self.assertEqual(rc, 0)
        self.assertIn("old path guessed  README.md:1 -> old_exp/summary.md  "
                      "(looks like prose shorthand relative to 07_results/)", out)
        self.assertIn("manual check      README.md:1 -> 07_results/old_exp/*.csv  (glob/placeholder)", out)
        self.assertEqual(out[-1], "1 mapping rows, 0 old paths remaining, 1 guessed (warning), 1 manual checks (warning)")

    def test_map_errors(self) -> None:
        self.write("README.md", "clean\n")
        self.assertEqual(self.run_main("--stale", str(self.root / "missing.md"))[0], 2)
        self.write("empty.md", "# no table\n")
        self.assertEqual(self.run_main("--stale", str(self.root / "empty.md"))[0], 2)


if __name__ == "__main__":
    unittest.main()
