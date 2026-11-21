"""PostToolUse hook on AskUserQuestion (the user's, installed by apply_findings_dialog.py): a question naming
"findings #<id>" answered "同意不修" by the user is recorded in .git/dualityime-findings-approvals.json, where
tools/findings.py skip looks for it - the user agreeing from any device (the phone included), in a way Claude cannot
write for them (the answers come from Claude Code, not from Claude; .git is off limits to Claude's tools).
The last payload is kept in .git/dualityime-last-askuser.json (to see what Claude Code sent)."""
import datetime
import json
import os
import re
import sys

ROOT = os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd()
APPROVALS = os.path.join(ROOT, ".git", "dualityime-findings-approvals.json")
LAST = os.path.join(ROOT, ".git", "dualityime-last-askuser.json")
AGREE = "同意不修"


def agreed_of(data):
    """{finding id: (question, answer)} the user agreed to leave: from the tool's result only (Claude Code's
    tool_response: its questions and the answers the user gave - never text Claude wrote, code-reviewer 2026-11-21),
    a question naming exactly one "findings #<id>", answered exactly with that question's own option "同意不修" (a
    typed "Other" answer does not count)."""
    resp = data.get("tool_response")
    if not isinstance(resp, dict) or not isinstance(resp.get("answers"), dict):
        return {}
    questions = {q.get("question"): q for q in resp.get("questions") or [] if isinstance(q, dict)}
    out = {}
    for question, answer in resp["answers"].items():
        q = questions.get(question)
        if q is None or answer != AGREE:
            continue
        if AGREE not in [o.get("label") for o in q.get("options") or [] if isinstance(o, dict)]:
            continue
        ids = re.findall(r"findings\s*#(\d+)", question)
        if len(set(ids)) == 1:
            out[ids[0]] = (question, answer)
    return out


def main():
    try:  # (bytes, as UTF-8: Claude Code sends UTF-8, and the console's code page would garble Chinese - 2026-11-21)
        data = json.loads(sys.stdin.buffer.read().decode("utf-8", "replace"))
    except ValueError:
        return 0
    if data.get("tool_name") != "AskUserQuestion":
        return 0
    try:
        with open(LAST, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
    except OSError:
        pass
    got = {fid: {"question": q, "answer": a, "time": datetime.datetime.now().isoformat()}
           for fid, (q, a) in agreed_of(data).items()}
    if not got:
        return 0
    try:
        with open(APPROVALS, encoding="utf-8") as f:
            have = json.load(f)
    except (OSError, ValueError):
        have = {}
    have.update(got)
    with open(APPROVALS, "w", encoding="utf-8") as f:
        json.dump(have, f, ensure_ascii=False, indent=1)
    sys.stdout.buffer.write(("findings: the user agreed to leave " + ", ".join(f"#{k}" for k in sorted(got, key=int))
                             + " (tools/findings.py skip <id> now writes its backlog line)\n").encode("utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
