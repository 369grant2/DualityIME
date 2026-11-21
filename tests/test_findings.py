"""tools/findings.py: an open review finding stops the release's full check until fixed or left by the user."""
import contextlib
import io
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import findings  # noqa: E402


class FindingsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.saved = findings.LEDGER, findings.BACKLOG, findings.APPROVALS, findings.LOCK
        findings.LEDGER = os.path.join(self.tmp, "findings.json")
        findings.LOCK = os.path.join(self.tmp, "findings.lock")
        findings.APPROVALS = os.path.join(self.tmp, "approvals.json")
        findings.BACKLOG = os.path.join(self.tmp, "review-backlog.md")
        with open(findings.BACKLOG, "w", encoding="utf-8") as f:
            f.write("# 審查備忘\n\n- 2026-11-18 an older line（使用者選不修）\n")

    def tearDown(self):
        findings.LEDGER, findings.BACKLOG, findings.APPROVALS, findings.LOCK = self.saved
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_cmd(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = findings.main(list(argv))
        return code, out.getvalue() + err.getvalue()

    def backlog(self):
        with open(findings.BACKLOG, encoding="utf-8") as f:
            return f.read()

    def test_an_open_finding_stops_the_gate_until_fixed(self):
        self.assertEqual(self.run_cmd("gate")[0], 0)
        self.run_cmd("add", "--source", "code-reviewer", "--where", "tools/x.py:1", "--what", "a hole")
        code, out = self.run_cmd("gate")
        self.assertEqual(code, 1)
        self.assertIn("a hole", out)
        self.run_cmd("fixed", "1", "--how", "closed it; test_x")
        self.assertEqual(self.run_cmd("gate")[0], 0)
        self.assertNotIn("a hole", self.backlog())  # (fixed: nothing for the backlog)

    def test_skip_needs_the_users_answer_unless_they_run_it_themselves(self):
        # (the user's hook records an AskUserQuestion answer "同意不修" to a question naming "findings #<id>")
        self.run_cmd("add", "--source", "code-reviewer", "--where", "tools/x.py:1", "--what", "flags too much")
        with self.assertRaises(SystemExit) as e:
            self.run_cmd("skip", "1")
        self.assertIn("has not agreed", str(e.exception))
        self.assertNotIn("flags too much", self.backlog())
        with open(findings.APPROVALS, "w", encoding="utf-8") as f:
            f.write('{"1": {"question": "findings #1 ...", "answer": "同意不修", "time": "2000-01-01T00:00:00"}}')
        with self.assertRaises(SystemExit):  # (an answer older than the finding: another #1's)
            self.run_cmd("skip", "1")
        with open(findings.APPROVALS, "w", encoding="utf-8") as f:
            f.write('{"1": {"question": "findings #1 ...", "answer": "同意不修", "time": "2999-01-01T00:00:00"}}')
        self.assertEqual(self.run_cmd("skip", "1")[0], 0)
        self.assertIn("flags too much", self.backlog())

    def test_an_unreadable_list_stops_the_gate(self):
        # (/code-review high 2026-11-21: read as empty, the gate passed with the findings lost)
        with open(findings.LEDGER, "w", encoding="utf-8") as f:
            f.write("[{broken")
        with self.assertRaises(SystemExit) as e:
            self.run_cmd("gate")
        self.assertNotEqual(e.exception.code, 0)

    def test_a_finding_left_by_the_user_goes_into_the_backlog_with_the_agreement(self):
        # (2026-11-21: a finding went into the backlog as "not fixed" on Claude's word alone)
        self.run_cmd("add", "--source", "code-reviewer", "--where", "tools/x.py:1", "--what", "flags too much")
        self.run_cmd("skip", "--direct", "1", "--why", "rather too many than too few")
        self.assertEqual(self.run_cmd("gate")[0], 0)
        last = self.backlog().strip().split("\n")[-1]
        self.assertTrue(last.startswith("- 20"))
        self.assertIn("flags too much", last)
        self.assertIn("同意不修", last)
        self.assertIn("2026-11-18 an older line", self.backlog())

    def test_direct_is_never_guessed_from_a_shorter_word(self):
        # (code-reviewer 2026-11-21: `skip 1 --d` was taken for --direct, which guard_bash knows only in full)
        self.run_cmd("add", "--source", "x", "--where", "y", "--what", "z")
        for short in ("--d", "--dir"):
            with self.assertRaises(SystemExit):
                self.run_cmd("skip", "1", short)
        self.assertNotIn("z（", self.backlog())

    def test_findings_added_at_once_are_all_kept(self):
        # (code-reviewer 2026-11-21: two writers at once lost half the findings)
        import threading
        threads = [threading.Thread(target=lambda k=k: self.run_cmd("add", "--source", "x", "--where", "y",
                                                                    "--what", f"f{k}")) for k in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        items = findings.load()
        self.assertEqual(len(items), 20)
        self.assertEqual(len({it["id"] for it in items}), 20)

    def test_a_backlog_line_taken_out_once_fixed(self):
        self.run_cmd("unlist", "2026-11-18 an older")
        self.assertNotIn("an older line", self.backlog())
        with self.assertRaises(SystemExit):  # (exactly one line, or nothing)
            self.run_cmd("unlist", "nothing like it")


if __name__ == "__main__":
    unittest.main()
