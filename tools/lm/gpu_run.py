"""Start a long GPU job and watch it: the one way to start teacher scoring, distillation or training. It ends - so a
background run reports - on the first of:

    DONE    the job ended with exit code 0
    FAILED  the job ended with another exit code (127: the command could not start, its error is in the log)
    STALL   the log has not grown for --stall seconds (the job keeps running; the GPU's memory and power are shown:
            nearly full memory at low power = spilling into system memory, see tools/lm/gpu.py). Exit code 2, so a
            chain (a && b) does not go on as if the job had finished

    py -3 tools/lm/gpu_run.py --log <file> [--stall 600] -- <command ...>
    py -3 tools/lm/gpu_run.py --attach <file> [--stall 600]     (after a STALL that was none: watch the same job again)

Only the arguments before `--` are gpu_run's. The command's output (stdout and stderr, unbuffered) goes to the log;
a chain goes as one shell command joined with && (`-- sh -c "a && b"`: with `;` a failed first step would still end
DONE), so the log grows whatever step runs. <log>.exit gets the exit code when the job ends. After a STALL whose job
still runs, watch it with --attach, never start another job on the same log (both would write it).
2026-11-14: teacher_train.py and then distill.py stalled, each unnoticed for half an hour (a watch on the epoch lines
only, that expired).
"""
import os
import subprocess
import sys
import time
import traceback

POLL = float(os.environ.get("GPU_RUN_POLL", 15))  # (seconds between looks; tests make it short)
CANNOT_START = 127


def gpu_state():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=30).stdout.strip()
        return out or "nvidia-smi printed nothing"
    except (OSError, subprocess.SubprocessError) as error:
        return f"nvidia-smi failed: {error}"


def last_line(log):
    try:
        with open(log, "rb") as f:
            f.seek(max(0, os.path.getsize(log) - 2000))
            text = f.read().decode("utf-8", errors="replace").replace("\r", "\n")
    except OSError:
        return ""
    lines = [line for line in text.split("\n") if line.strip()]
    return lines[-1] if lines else ""


def run(log, command):
    """(--run, started by start()) the job itself: its output into the log, then its exit code into <log>.exit -
    also when the command cannot start (a wrong name, no sh on PATH): its error goes into the log, the code is 127."""
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    with open(log, "ab") as out:
        try:
            code = subprocess.call(command, stdout=out, stderr=subprocess.STDOUT, env=env)
        except Exception:  # noqa: BLE001 - whatever it was, the watcher must hear the job is over
            out.write(traceback.format_exc().encode("utf-8", errors="replace"))
            code = CANNOT_START
    with open(log + ".exit", "w", encoding="utf-8") as f:
        f.write(str(code))
    return code


def start(log, command):
    """The job in a process of its own, so it outlives this watcher after a STALL; it inherits none of the watcher's
    handles, so whatever reads the watcher's output gets its end with the watcher's, not the job's."""
    if os.path.exists(log + ".exit"):  # (an earlier job's: set aside, or the watch would end at once)
        os.replace(log + ".exit", log + ".exit.old")
    open(log, "wb").close()
    flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    subprocess.Popen([sys.executable, os.path.abspath(__file__), "--run", log, "--", *command], creationflags=flags,
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def watch(log, stall, out=print, clock=time.time, sleep=time.sleep):
    """Wait for the job writing `log`; one line saying how it ended (or stalled). Returns 0 (DONE), 1 (FAILED) or
    2 (STALL)."""
    seen, changed = -1, clock()
    while True:
        sleep(POLL)
        if os.path.exists(log + ".exit"):
            with open(log + ".exit", encoding="utf-8") as f:
                code = f.read().strip()
            if code == "0":
                out(f"DONE: {log}; last: {last_line(log)}")
                return 0
            out(f"FAILED (exit {code}): {log}; last: {last_line(log)}")
            return 1
        size = os.path.getsize(log) if os.path.exists(log) else 0
        if size != seen:
            seen, changed = size, clock()
        elif clock() - changed > stall:
            out(f"STALL: {log} unchanged for {clock() - changed:.0f}s; last: {last_line(log)}; GPU: {gpu_state()}"
                f" (watch again: py -3 tools/lm/gpu_run.py --attach {log})")
            return 2


def split(argv):
    """gpu_run's own arguments (before --) and the command (after it): a command's own --log or --run stays its own."""
    if "--" in argv:
        k = argv.index("--")
        return argv[:k], argv[k + 1:]
    return argv, []


def arg(own, name, default=None):
    return own[own.index(name) + 1] if name in own and own.index(name) + 1 < len(own) else default


def main(argv=None):
    own, command = split(sys.argv[1:] if argv is None else argv)
    if "--run" in own:
        return run(arg(own, "--run"), command)
    stall = float(arg(own, "--stall", 600))
    if "--attach" in own:
        return watch(arg(own, "--attach"), stall)
    log = arg(own, "--log")
    if not log or not command:
        sys.exit(__doc__)
    log = os.path.abspath(log)
    start(log, command)
    print(f"started; log {log}", flush=True)
    return watch(log, stall)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # (here, not in main(): a test calling main() keeps its own stdout)
    sys.exit(main())
