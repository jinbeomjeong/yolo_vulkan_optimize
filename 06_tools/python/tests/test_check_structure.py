"""Tests for 06_tools/python/check_structure.py (stdlib unittest, no pytest).

Run: python3 -m unittest discover -s 06_tools/python/tests

Most tests use a small fixture rule table and a throwaway git repo in a temp dir.
The live repo is only read (the real rule table + the built-in SELFTEST_CASES), never written.
"""
from __future__ import annotations

import contextlib
import io
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import check_structure as cs  # noqa: E402

# A trimmed copy of the live structure_rules.toml layout (same keys, fewer entries).
FIXTURE_TOML = r'''
[top_level]
tracked = ["01_src", "02_shaders", "05_tests", "06_tools", "07_results", "08_learn",
           "README.md", "STRUCTURE.md", "CMakeLists.txt"]
untracked = ["03_models", "11_logs", "build", ".git"]
untracked_globs = ["build*"]

[junk]
root_logs = ["hailort.log"]
dump_globs = ["*.data", "perf.data*", "core"]

[code]
src_root = "01_src"
allowed_dirs = ["core", "ref", "hailo", "vulkan", "pipeline", "tools"]
root_files = ["CMakeLists.txt"]
library_dirs = ["core", "ref", "hailo", "vulkan", "pipeline"]
library_exclude = ["01_src/pipeline/main.cpp"]
library_forbidden_subdirs = ["tools"]
main_allowed = ["01_src/tools/**/*.cpp", "01_src/pipeline/main.cpp"]
system_include_roots = []
system_name_ok = []
source_exts = [".cpp", ".hpp", ".h"]
tests_dir = "05_tests"

[code.suggest]
common = "01_src/core/"

[code.include_deps]
core = []
ref = ["core"]
hailo = ["core", "ref"]
vulkan = ["core", "ref"]
pipeline = ["core", "ref", "hailo", "vulkan"]
tools = ["*"]
"05_tests" = ["core", "ref", "hailo", "vulkan", "pipeline"]

[code.include_unprefixed]
globs = ["*_spv.h"]

[cmake]
forbidden_outside_cmake_dir = ["-mcpu"]
forbidden_anywhere = ["yvo_add_subdirectory_if_exists"]

[bench]
dir = "06_tools/bench"
forbidden = ["build/01_src"]
py_hailo_owner = "06_tools/bench/lib/common.sh"
npu_guard = ".claude/hooks/npu_guard.py"

[docs]
structure_md = "STRUCTURE.md"

[[placement]]
prefix = "02_shaders/"
allow = ["*.comp", "diag/*.comp", "CMakeLists.txt"]
hint = "shaders"

[[placement]]
prefix = "06_tools/"
allow = ["README.md"]
hint = "06_tools/{bench,python}/"

[[placement]]
prefix = "06_tools/bench/"
allow = ["*.sh", "lib/*.sh"]
hint = "bench shells"

[[placement]]
prefix = "06_tools/python/"
allow = ["**/*.py"]
hint = "python only"

[placement_root_suggest]
".py" = "06_tools/python/"

[results]
root = "07_results"
index = "07_results/README.md"
root_files = ["README.md"]
required = ["summary.md", "env_bench.txt"]
run_required = ["env_bench.txt"]
files = ["summary.md", "env_bench.txt", "results.csv", "results_raw_*.csv"]
dirs = ["raw", "evidence", "notes", "runs"]
run_dir_re = '^[A-Za-z0-9][A-Za-z0-9_.-]*_run\d+$'
session_dir = "_session"
session_files = ["hailort.log", "bench_stdout.log"]
notes_globs = ["*.md", "**/*.md"]
evidence_warn_globs = ["asan_*"]

[results.type_required]
"sweep" = ["results.csv", "raw/"]
"probe" = ["evidence/"]

[results.suggest_dirs]
"dump_*" = "samples/"

[[exception]]
path = "07_results/legacy"
waive = ["layout"]
reason = "fixture: old flat layout"
'''


class RuleFixture(unittest.TestCase):
    """Points check_structure at a temp dir (ROOT) and the fixture rule table (RULES_PATH)."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name).resolve()
        rules = self.root / "06_tools" / "structure_rules.toml"
        rules.parent.mkdir(parents=True)
        rules.write_text(FIXTURE_TOML, encoding="utf-8")
        for target, value in (("ROOT", self.root), ("RULES_PATH", rules)):
            p = mock.patch.object(cs, target, value)
            p.start()
            self.addCleanup(p.stop)
        self._clear_caches()
        self.addCleanup(self._clear_caches)
        self.addCleanup(self._tmp.cleanup)

    @staticmethod
    def _clear_caches() -> None:
        cs.load_rules.cache_clear()
        cs.tracked_files.cache_clear()
        cs.untracked_files.cache_clear()

    def write(self, rel: str, text: str = "") -> None:
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")


class GlobTest(unittest.TestCase):
    def test_star_is_one_component(self) -> None:
        self.assertTrue(cs.glob_match("a.py", "*.py"))
        self.assertFalse(cs.glob_match("sub/a.py", "*.py"))

    def test_double_star_slash_is_zero_or_more_components(self) -> None:
        for p in ("a.py", "x/a.py", "x/y/a.py"):
            self.assertTrue(cs.glob_match(p, "**/*.py"), p)
        self.assertFalse(cs.glob_match("x/a.txt", "**/*.py"))

    def test_trailing_double_star_and_question_mark(self) -> None:
        self.assertTrue(cs.glob_match("assets/a/b.png", "assets/**"))
        self.assertTrue(cs.glob_match("a1", "a?"))
        self.assertFalse(cs.glob_match("a/", "a?"))

    def test_regex_metacharacters_are_literal(self) -> None:
        self.assertTrue(cs.glob_match("a+b.(x)", "a+b.(x)"))
        self.assertFalse(cs.glob_match("aab", "a.b"))


class RuleTableLoadTest(RuleFixture):
    def test_load_rules_reads_fixture_and_caches(self) -> None:
        rules = cs.load_rules()
        self.assertEqual(rules["results"]["type_required"]["sweep"], ["results.csv", "raw/"])
        self.assertEqual(rules["placement"][0]["prefix"], "02_shaders/")
        self.assertIs(cs.load_rules(), rules)   # lru_cache: one parse per process


class ClassifyNewPathTest(RuleFixture):
    def rid(self, path: str) -> str | None:
        r = cs.classify_new_path(path)
        return r[0] if r else None

    def test_root_files(self) -> None:
        self.assertIsNone(self.rid("README.md"))
        rule, _, suggest = cs.classify_new_path("tool.py")
        self.assertEqual(rule, "T1")
        self.assertEqual(suggest, "06_tools/python/")   # placement_root_suggest by extension
        self.assertEqual(self.rid("hailort.log"), "T2")
        self.assertEqual(self.rid("99_misc/x.md"), "T1")

    def test_untracked_bands_and_outside_paths_pass(self) -> None:
        for p in ("build/x.o", "build-asan/y", "11_logs/a.log", "03_models/m.hef", "/tmp/elsewhere.txt",
                  "../outside.txt", "", "."):
            self.assertIsNone(self.rid(p), p)

    def test_absolute_path_inside_root_is_normalized(self) -> None:
        self.assertEqual(self.rid(str(self.root / "stray.txt")), "T1")
        self.assertIsNone(self.rid(str(self.root / "06_tools/python/ok.py")))
        self.assertEqual(self.rid("06_tools/python/../../stray.txt"), "T1")

    def test_shader_and_dump_rules(self) -> None:
        rule, _, suggest = cs.classify_new_path("06_tools/python/x.comp")
        self.assertEqual((rule, suggest), ("C3", "02_shaders/x.comp"))
        self.assertIsNone(self.rid("02_shaders/diag/new.comp"))
        self.assertEqual(self.rid("08_learn/perf.data.old"), "T2")

    def test_src_rules(self) -> None:
        self.assertIsNone(self.rid("01_src/CMakeLists.txt"))
        self.assertIsNone(self.rid("01_src/vulkan/foo.cpp"))
        self.assertEqual(self.rid("01_src/stray.cpp"), "C1")
        rule, _, suggest = cs.classify_new_path("01_src/common/x.hpp")
        self.assertEqual((rule, suggest), ("C1", "01_src/core/"))   # code.suggest
        rule, _, suggest = cs.classify_new_path("01_src/vulkan/tools/probe.cpp")
        self.assertEqual((rule, suggest), ("C1", "01_src/tools/vulkan/probe.cpp"))

    def test_placement_longest_prefix_wins(self) -> None:
        self.assertIsNone(self.rid("06_tools/README.md"))
        self.assertIsNone(self.rid("06_tools/python/tests/test_new.py"))
        self.assertEqual(self.rid("06_tools/python/data.csv"), "T1")
        self.assertEqual(self.rid("06_tools/bench/README.md"), "C9")   # bench prefix reports C9
        self.assertIsNone(self.rid("06_tools/bench/lib/extra.sh"))
        self.assertEqual(self.rid("06_tools/stray.txt"), "T1")
        self.assertIsNone(self.rid("05_tests/anything.txt"))            # no placement row -> not judged

    def test_results_layout(self) -> None:
        cases = {
            "07_results/README.md": None,
            "07_results/stray.md": "R2",
            "07_results/x/summary.md": None,
            "07_results/x/results_raw_part2.csv": None,
            "07_results/x/foo.txt": "R4",
            "07_results/x/detections/a.jsonl": "R4",
            "07_results/x/raw/c_run1/stdout.log": None,
            "07_results/x/raw/a.log": "R6",
            "07_results/x/raw/badname/stdout.log": "R6",
            "07_results/x/raw/c_run1/sub/x.log": "R6",
            "07_results/x/raw/_session/hailort.log": None,
            "07_results/x/raw/_session/other.log": "R7",
            "07_results/x/runs/r1/env_bench.txt": None,
            "07_results/x/runs/stray.txt": "R8",
            "07_results/x/runs/r1/runs/r2/env_bench.txt": "R8",
            "07_results/x/notes/memo.md": None,
            "07_results/x/notes/a.txt": "R9",
        }
        for path, want in cases.items():
            self.assertEqual(self.rid(path), want, path)

    def test_raw_file_suggestions(self) -> None:
        self.assertEqual(cs.classify_new_path("07_results/x/raw/b8_run2.log")[2], "07_results/x/raw/b8_run2/run.log")
        self.assertEqual(cs.classify_new_path("07_results/x/raw/b8_3.log")[2], "07_results/x/raw/b8_run3/stdout.log")
        self.assertEqual(cs.classify_new_path("07_results/x/raw/hailort.log")[2], "07_results/x/raw/_session/hailort.log")
        self.assertEqual(cs.classify_new_path("07_results/x/dump_a/f.bin")[2], "07_results/x/samples/")

    def test_exception_waives_by_token(self) -> None:
        self.assertIsNone(self.rid("07_results/legacy/flat.txt"))       # R4 'layout' waived
        self.assertEqual(self.rid("07_results/legacy/notes/a.txt"), "R9")  # 'notes' token is not waived


class LiveRuleTableTest(unittest.TestCase):
    """Read-only checks against the real 06_tools/structure_rules.toml."""

    def setUp(self) -> None:
        cs.load_rules.cache_clear()
        self.addCleanup(cs.load_rules.cache_clear)

    def test_builtin_selftest_passes(self) -> None:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = cs.selftest()
        self.assertEqual(rc, 0, out.getvalue())

    def test_these_tests_are_placed_where_the_rules_allow(self) -> None:
        here = Path(__file__).resolve().relative_to(cs.ROOT).as_posix()
        self.assertIsNone(cs.classify_new_path(here))


class MarkdownTableParsingTest(unittest.TestCase):
    STRUCTURE = textwrap.dedent("""\
        # Title

        ## 1. 최상위 대역

        | 항목 | 추적 | 내용 |
        |---|---|---|
        | `01_src` | 추적 | code |
        | `build/`, `build-asan/` | 밖 | output |

        text after

        ## 4. 폴더별 담당

        | 경로 | 담당 |
        |:--|--:|
        | `01_src/{core,decode}/` | `cpp-expert` |
        | `05_tests/` | 도메인 전문가 + `verify` 스킬 |
        | `07_results/` | 쓰기는 `measure`(측정)·`verify`(`verification/`) |
        | `09_diagrams/` | (포맷은 hailo-expert) `graphml-expert` |

        | other | table |
        |---|---|
        | `not_read` | x |
        """)

    def test_first_table_col0(self) -> None:
        self.assertEqual(cs._first_table_col0(self.STRUCTURE, "최상위"), {"01_src", "build", "build-asan"})
        self.assertIsNone(cs._first_table_col0(self.STRUCTURE, "없는 제목"))

    def test_first_table_rows_line_numbers(self) -> None:
        rows = cs._first_table_rows(self.STRUCTURE, "폴더별 담당")
        self.assertEqual([no for no, _ in rows], [16, 17, 18, 19])
        self.assertEqual(rows[0][1], ["`01_src/{core,decode}/`", "`cpp-expert`"])

    def test_expand_braces(self) -> None:
        self.assertEqual(cs._expand_braces("a/{b,c}/d"), ["a/b/d", "a/c/d"])
        self.assertEqual(cs._expand_braces("{x,y}/{1,2}"), ["x/1", "x/2", "y/1", "y/2"])
        self.assertEqual(cs._expand_braces("plain"), ["plain"])

    def test_owner_token(self) -> None:
        self.assertEqual(cs._owner_token("`cpp-expert`"), "cpp-expert")
        self.assertEqual(cs._owner_token("도메인 전문가 + `verify` 스킬"), "전문가")      # earliest wins
        self.assertEqual(cs._owner_token("`verification/` 만"), None)                  # not 'verify'
        self.assertEqual(cs._owner_token("(포맷은 hailo-expert) `graphml-expert`"), "graphml-expert")  # () ignored
        self.assertEqual(cs._owner_token("쓰기는 `measure`(측정)·`verify`(`verification/`)"), "measure")

    def test_owner_table(self) -> None:
        table = cs._owner_table(self.STRUCTURE, "폴더별 담당")
        self.assertEqual(table["01_src/core"], (16, "cpp-expert"))
        self.assertEqual(table["01_src/decode"], (16, "cpp-expert"))
        self.assertEqual(table["09_diagrams"], (19, "graphml-expert"))
        self.assertNotIn("not_read", table)


class SourceScannerTest(unittest.TestCase):
    def test_scan_cpp_blanks_comments_and_strings_keeping_lines(self) -> None:
        src = 'int a; // "07_x" no\n/* b\n c */ const char* p = "03_models/m.hef";\nauto r = R"(raw "q")";\n'
        code, lits = cs.scan_cpp(src)
        self.assertEqual(code.count("\n"), src.count("\n"))
        self.assertNotIn("07_x", code)
        self.assertNotIn("03_models", code)
        self.assertIn("const char* p =", code)
        self.assertEqual(lits, [(3, "03_models/m.hef"), (4, 'raw "q"')])

    def test_strip_cmake_comments_keeps_quoted_hash(self) -> None:
        text = 'set(A "x#y") # comment -mcpu\n# whole line\nadd(B)'
        self.assertEqual(cs.strip_cmake_comments(text), 'set(A "x#y") \n\nadd(B)')


class CheckerOnFixtureRepoTest(RuleFixture):
    """Runs individual Checker passes on a throwaway git repo (files untracked is enough: all_files = tracked + untracked)."""

    def setUp(self) -> None:
        super().setUp()
        subprocess.run(["git", "init", "-q", str(self.root)], check=True)

    def run_pass(self, name: str) -> list[tuple[str, str]]:
        checker = cs.Checker(cs.load_rules())
        getattr(checker, name)()
        return sorted((v.rule, v.path) for v in checker.out)

    def test_results_index_required_files_and_types(self) -> None:
        self.write("07_results/README.md", textwrap.dedent("""\
            | 실험 | 유형 | 설명 |
            |---|---|---|
            | [good](good/summary.md) | `sweep` | ok |
            | [probe](probe/summary.md) | probe | missing evidence/ |
            | [ghost](ghost/summary.md) | sweep | no folder |
            | [odd](odd/summary.md) | weird | unknown type |
            """))
        for f in ("summary.md", "env_bench.txt", "results.csv", "raw/a_run1/stdout.log", "notes/x.txt",
                  "evidence/asan_1.log"):
            self.write(f"07_results/good/{f}")
        for f in ("summary.md", "env_bench.txt"):
            self.write(f"07_results/probe/{f}")
            self.write(f"07_results/odd/{f}")
        self.write("07_results/orphan/summary.md")
        self.write("07_results/good/runs/r1/summary.md")
        got = self.run_pass("check_results")
        self.assertEqual(got, sorted([
            ("R1", "07_results/orphan/"),           # folder without an index row
            ("R1", "07_results/ghost/"),            # index row without a folder
            ("R3", "07_results/orphan/"),           # env_bench.txt missing
            ("R3", "07_results/good/runs/r1/"),     # run_required env_bench.txt
            ("R5", "07_results/probe/"),            # type probe needs evidence/
            ("R5", "07_results/odd/"),              # unknown type
            ("R9", "07_results/good/notes/x.txt"),  # notes/ holds only .md
            ("R9", "07_results/good/evidence/asan_1.log"),  # warning: belongs in verification/
        ]))

    def test_code_layers_main_literals_and_cmake(self) -> None:
        self.write("01_src/CMakeLists.txt", "add_library(x core/a.cpp hailo/h.cpp) # stray.cpp\nset(F -mcpu=x)\n")
        self.write("01_src/core/a.cpp", '#include "core/a.hpp"\nconst char* p = "07_results/x";\n')
        self.write("01_src/core/a.hpp", "")
        self.write("01_src/hailo/h.cpp", '#include "vulkan/v.hpp"\n#include "../core/a.hpp"\n#include "gen_spv.h"\n'
                                         '#include "loose.hpp"\nint main() { return 0; }\n')
        self.write("01_src/vulkan/unregistered.cpp", "")
        got = self.run_pass("check_code")
        self.assertEqual(got, sorted([
            ("C4", "01_src/hailo/h.cpp:1"),               # hailo -> vulkan is not in include_deps
            ("C4", "01_src/hailo/h.cpp:2"),               # relative include
            ("C4", "01_src/hailo/h.cpp:4"),               # unprefixed include (gen_spv.h is allowed)
            ("C5", "01_src/hailo/h.cpp:5"),               # main() outside tools/
            ("C6", "01_src/vulkan/unregistered.cpp"),     # only mentioned in a CMake comment elsewhere
            ("C7", "01_src/core/a.cpp:2"),                # CWD-relative path literal
            ("C8", "01_src/CMakeLists.txt:2"),            # -mcpu outside cmake/
        ]))

    def test_docs_top_level_table_vs_toml(self) -> None:
        self.write("STRUCTURE.md", "## 1. 최상위\n\n| 항목 | 뜻 |\n|---|---|\n"
                                   "| `01_src` `02_shaders` `05_tests` `06_tools` `07_results` | a |\n"
                                   "| `README.md` `STRUCTURE.md` `CMakeLists.txt` | b |\n"
                                   "| `03_models` `11_logs` `build/` `.git` `extra` | c |\n")
        got = self.run_pass("check_docs")
        self.assertEqual(got, [("D1", "STRUCTURE.md"), ("D1", "STRUCTURE.md")])
        checker = cs.Checker(cs.load_rules())
        checker.check_docs()
        whys = sorted(v.why for v in checker.out)
        self.assertIn("'08_learn' missing from the table (present in TOML top_level)", whys)
        self.assertIn("'extra' in the table is not in TOML top_level", whys)


if __name__ == "__main__":
    unittest.main()
