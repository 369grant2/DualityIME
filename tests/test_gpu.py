"""tools/lm/gpu.py and tools/lm/gpu_run.py: a long GPU job neither spills into system memory unseen nor stalls
unreported (/collab:correct 2026-11-14: teacher_train.py and distill.py each stalled half an hour unnoticed)."""
import importlib.util
import os
import subprocess
import sys
import shutil
import tempfile
import time
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LM = os.path.join(ROOT, "tools", "lm")


def load(name):
    # (by its path: tools/lm on sys.path would shadow modules of other tests - train, common...)
    spec = importlib.util.spec_from_file_location(f"lm_{name}", os.path.join(LM, f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gpu_run = load("gpu_run")

HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "torch is not installed")
class TidyTest(unittest.TestCase):
    def tidy_with(self, reserved, total=100, peak=0):
        gpu = load("gpu")
        import torch
        props = mock.Mock(total_memory=total)
        with mock.patch.object(torch.cuda, "is_available", return_value=True), \
                mock.patch.object(torch.cuda, "get_device_properties", return_value=props), \
                mock.patch.object(torch.cuda, "memory_reserved", return_value=reserved),                 mock.patch.object(torch.cuda, "max_memory_allocated", return_value=peak), \
                mock.patch.object(torch.cuda, "empty_cache") as empty:
            emptied = gpu.tidy()
        return emptied, empty.call_count

    def test_cache_emptied_past_the_limit(self):
        self.assertEqual(self.tidy_with(90), (True, 1))

    def test_cache_kept_under_the_limit(self):
        self.assertEqual(self.tidy_with(80), (False, 0))

    def test_cache_kept_when_the_job_itself_uses_it(self):
        # (a big model whose step peaks at 88 of 100: emptying every step would only slow it - /code-review 2026-11-15;
        # tidy() runs between steps, when the step's own memory is already freed: the peak tells, not what is allocated)
        self.assertEqual(self.tidy_with(90, peak=88), (False, 0))

    def test_no_gpu_nothing_done(self):
        gpu = load("gpu")
        import torch
        with mock.patch.object(torch.cuda, "is_available", return_value=False), \
                mock.patch.object(torch.cuda, "empty_cache") as empty:
            self.assertFalse(gpu.tidy())
        self.assertEqual(empty.call_count, 0)

    def test_every_gpu_script_tidies(self):
        # (a new loop copied from any of them keeps it; tools/lm/gpu.py says why)
        for path in ("train.py", "distill/teacher_train.py", "distill/teacher_rerank.py", "distill/distill.py",
                     "distill/ft_teacher.py"):
            with open(os.path.join(LM, path), encoding="utf-8") as f:
                text = f.read()
            self.assertRegex(text, r"from gpu import (note, )?tidy", path)
            self.assertIn("tidy()", text, path)

    def test_distill_tidies_its_evaluation_too(self):
        # (the evaluation sets carry contexts of every length as well: code-reviewer 2026-11-14)
        with open(os.path.join(LM, "distill", "distill.py"), encoding="utf-8") as f:
            self.assertGreaterEqual(f.read().count("tidy()"), 2)


class WatchTest(unittest.TestCase):
    """gpu_run.watch: one line, on the first of the job's end or a log that stopped growing."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.log = os.path.join(self.tmp.name, "job.log")
        with open(self.log, "w", encoding="utf-8") as f:
            f.write("step 500\n")

    def watch(self, stall=60, tick=10, on_sleep=None):
        lines, now = [], [0.0]

        def sleep(_):
            now[0] += tick
            if on_sleep:
                on_sleep(now[0])
        with mock.patch.object(gpu_run, "gpu_state", return_value="100 %, 23851 MiB, 24564 MiB, 134 W"):
            code = gpu_run.watch(self.log, stall, out=lines.append, clock=lambda: now[0], sleep=sleep)
        return code, lines

    def exit_with(self, code):
        with open(self.log + ".exit", "w", encoding="utf-8") as f:
            f.write(str(code))

    def test_done(self):
        self.exit_with(0)
        code, lines = self.watch()
        self.assertEqual(code, 0)
        self.assertTrue(lines[0].startswith("DONE"), lines)

    def test_failed(self):
        self.exit_with(3)
        code, lines = self.watch()
        self.assertEqual(code, 1)
        self.assertTrue(lines[0].startswith("FAILED (exit 3)"), lines)

    def test_stall_reported_with_the_gpu(self):
        code, lines = self.watch(stall=60)
        self.assertEqual(code, 2)  # (not 0: a chain must not take a stalled job for a finished one)
        self.assertTrue(lines[0].startswith("STALL"), lines)
        self.assertIn("23851 MiB", lines[0])
        self.assertIn("--attach", lines[0])

    def test_a_growing_log_is_no_stall(self):
        def grow(now):
            if now <= 300:
                with open(self.log, "a", encoding="utf-8") as f:
                    f.write("step\n")
            else:
                self.exit_with(0)
        code, lines = self.watch(stall=60, on_sleep=grow)
        self.assertTrue(lines[0].startswith("DONE"), lines)


class RunTest(unittest.TestCase):
    """gpu_run.py started for real on small jobs (polls every 0.2 s)."""

    def run_job(self, code, stall=30, command=None):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        log = os.path.join(tmp.name, "job.log")
        env = dict(os.environ, GPU_RUN_POLL="0.2")
        command = command or [sys.executable, "-c", code]
        started = time.time()
        out = subprocess.run([sys.executable, os.path.join(LM, "gpu_run.py"), "--log", log, "--stall", str(stall),
                              "--", *command], capture_output=True, encoding="utf-8", errors="replace", env=env,
                             timeout=60)  # (gpu_run prints UTF-8: Windows' own error text is Chinese here)
        self.seconds = time.time() - started
        # (after a STALL the job still runs, writing the log: wait for its end before the folder goes)
        for _ in range(300):
            if os.path.exists(log + ".exit"):
                break
            time.sleep(0.1)
        with open(log, encoding="utf-8", errors="replace") as f:
            return out, f.read()

    def test_output_goes_to_the_log_and_done(self):
        out, log = self.run_job("print('step 1')")
        self.assertIn("DONE", out.stdout)
        self.assertIn("step 1", log)

    def test_failure(self):
        out, log = self.run_job("import sys; print('boom'); sys.exit(2)")
        self.assertIn("FAILED (exit 2)", out.stdout)
        self.assertEqual(out.returncode, 1)

    def test_stall_while_the_job_runs(self):
        out, log = self.run_job("import time; print('started', flush=True); time.sleep(6); print('ended')", stall=1)
        self.assertIn("STALL", out.stdout)
        self.assertLess(self.seconds, 5)  # (reported while the job sleeps, not at its end)
        self.assertIn("ended", log)  # (the job went on after the watcher's STALL)

    def test_a_command_that_cannot_start_fails(self):
        out, log = self.run_job(None, command=["no_such_program_dualityime", "--x"])
        self.assertIn("FAILED (exit 127)", out.stdout)
        self.assertIn("Traceback", log)

    @unittest.skipUnless(shutil.which("sh"), "no sh on PATH")
    def test_a_failed_first_step_of_a_chain_fails(self):
        py = sys.executable.replace("\\", "/")
        out, _ = self.run_job(None, command=["sh", "-c", f"'{py}' -c 'import sys; sys.exit(3)' && '{py}' -c 'print(2)'"])
        self.assertIn("FAILED (exit 3)", out.stdout)


class SplitTest(unittest.TestCase):
    def test_the_commands_own_arguments_stay_its_own(self):
        own, command = gpu_run.split(["--log", "a.log", "--", "py", "ft_teacher.py", "--log", "200", "--run", "x"])
        self.assertEqual(own, ["--log", "a.log"])
        self.assertEqual(command, ["py", "ft_teacher.py", "--log", "200", "--run", "x"])

    def test_without_its_own_log_nothing_starts(self):
        with self.assertRaises(SystemExit):
            gpu_run.main(["--", "py", "ft_teacher.py", "--log", "200"])


if __name__ == "__main__":
    unittest.main()
