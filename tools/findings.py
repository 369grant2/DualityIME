"""Review findings, each recorded until it is fixed or the user agrees to leave it: nothing found by a review is left
unfixed on Claude's word alone (the user, 2026-11-21: "以後所有不修的都要完整列出來給我看 我同意才能放行" - a
code-reviewer finding went into docs/review-backlog.md as "not fixed" without asking).

    py -3 tools/findings.py add --source <code-reviewer|code-review-high|...> --where <file:line> --what "<the defect>"
    py -3 tools/findings.py fixed <id> --how "<the fix and its evidence>"
    py -3 tools/findings.py list [--all]       the open ones (all: every one)
    py -3 tools/findings.py gate               exit 1 listing the open ones (check.py, before a release's full check)
    py -3 tools/findings.py skip <id> [<id> ...] [--why "<reason>"]    leaves it unfixed and writes its line into
                                               docs/review-backlog.md - only after the user agreed: a question naming
                                               "findings #<id>" answered "同意不修" (AskUserQuestion: the phone too),
                                               which the user's hook .claude/hooks/findings_approval.py records in
                                               .git/dualityime-findings-approvals.json
    py -3 tools/findings.py skip --direct <id> ...   THE USER ONLY (guard_bash blocks Claude): the same from their own
                                               terminal, no question needed
    py -3 tools/findings.py unlist "<start of a backlog line>"         a backlog line taken out (fixed later, or moved
                                               into docs/status.md for the next version)

The findings are this repo's: .git/dualityime-findings.json (not in git). docs/review-backlog.md is written only
through skip and unlist (guard_paths keeps the editing tools off it).
"""
import argparse
import datetime
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEDGER = os.path.join(ROOT, ".git", "dualityime-findings.json")
APPROVALS = os.path.join(ROOT, ".git", "dualityime-findings-approvals.json")  # (the user's hook writes it)
BACKLOG = os.path.join(ROOT, "docs", "review-backlog.md")
sys.stdout.reconfigure(encoding="utf-8")


LOCK = os.path.join(ROOT, ".git", "dualityime-findings.lock")


class locked:
    """The read-change-write of the list, one process at a time (two reviewers adding at once lost findings -
    code-reviewer 2026-11-21); a lock older than 60 s is a dead writer's and is taken over."""

    def __enter__(self):
        for _ in range(400):
            try:
                os.close(os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
                return self
            except FileExistsError:
                try:
                    if time.time() - os.path.getmtime(LOCK) > 60:
                        os.remove(LOCK)
                        continue
                except OSError:
                    pass
                time.sleep(0.05)
        sys.exit(f"the findings list stays locked ({LOCK}): another findings.py is still writing")

    def __exit__(self, *exc):
        try:
            os.remove(LOCK)
        except OSError:
            pass


def load():
    """The list; none yet: []. An unreadable one stops everything (as empty, the gate would pass with findings lost and
    ids would start again at 1 - /code-review high 2026-11-21)."""
    if not os.path.exists(LEDGER):
        return []
    try:
        with open(LEDGER, encoding="utf-8") as f:
            items = json.load(f)
        if not isinstance(items, list):
            raise ValueError("not a list")
        return items
    except (OSError, ValueError) as e:
        sys.exit(f"the findings list {LEDGER} cannot be read ({e}): nothing done; tell the user")


def save(items):
    tmp = LEDGER + f".{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=1)
    for k in range(40):  # (Windows: a reader holding the file for a moment)
        try:
            os.replace(tmp, LEDGER)
            return
        except PermissionError:
            if k == 39:
                raise
            time.sleep(0.05)


def find(items, fid):
    for it in items:
        if it["id"] == fid:
            return it
    sys.exit(f"no finding {fid} (py -3 tools/findings.py list --all)")


def show(it):
    extra = {"fixed": f" - fixed: {it.get('how', '')}", "skipped": f" - the user agreed to leave it: {it.get('why', '')}"}
    return f"#{it['id']} [{it['state']}] {it['source']} {it['where']}: {it['what']}{extra.get(it['state'], '')}"


def main(argv):
    ap = argparse.ArgumentParser(description="Review findings until fixed or left with the user's agreement.",
                                 allow_abbrev=False)
    # (no abbreviations anywhere: `skip 1 --d` was taken for --direct, which guard_bash knows only in full -
    # code-reviewer 2026-11-21)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add", allow_abbrev=False)
    a.add_argument("--source", required=True)
    a.add_argument("--where", required=True)
    a.add_argument("--what", required=True)
    f = sub.add_parser("fixed", allow_abbrev=False)
    f.add_argument("id", type=int)
    f.add_argument("--how", required=True)
    s = sub.add_parser("skip", allow_abbrev=False)
    s.add_argument("ids", type=int, nargs="+")
    s.add_argument("--why", default="")
    s.add_argument("--direct", action="store_true")
    ls = sub.add_parser("list", allow_abbrev=False)
    ls.add_argument("--all", action="store_true")
    sub.add_parser("gate", allow_abbrev=False)
    u = sub.add_parser("unlist", allow_abbrev=False)
    u.add_argument("start")
    args = ap.parse_args(argv)
    with locked():
        return run(args)


def run(args):
    items = load()
    today = datetime.date.today().isoformat()
    if args.cmd == "add":
        it = {"id": max([x["id"] for x in items], default=0) + 1, "source": args.source, "where": args.where,
              "what": args.what, "state": "open", "added": today,
              "added_at": datetime.datetime.now().isoformat()}
        items.append(it)
        save(items)
        print(show(it))
    elif args.cmd == "fixed":
        it = find(items, args.id)
        it.update(state="fixed", how=args.how, closed=today)
        save(items)
        print(show(it))
    elif args.cmd == "skip":
        try:
            with open(APPROVALS, encoding="utf-8") as fh:
                agreed = json.load(fh)
        except (OSError, ValueError):
            agreed = {}
        if not args.direct:
            # (an answer counts for the finding asked about: given after it was added - an older answer to another
            # finding of the same number does not - /code-review high 2026-11-21)
            missing = [fid for fid in args.ids
                       if str(fid) not in agreed
                       or agreed[str(fid)].get("time", "") <= find(items, fid).get("added_at", "")]
            if missing:
                sys.exit("the user has not agreed to leave " + ", ".join(f"#{m}" for m in missing) + ": ask them "
                         "(AskUserQuestion, one question per finding naming \"findings #<id>\", options \"同意不修\" "
                         "and \"要修\"), or give them `! py -3 tools/findings.py skip --direct <id>` to run")
        lines = []
        for fid in args.ids:
            it = find(items, fid)
            if it["state"] != "open":
                sys.exit(f"#{fid} is {it['state']}, not open")
            if not args.direct:
                it["agreed"] = agreed[str(fid)]
            it.update(state="skipped", why=args.why, closed=today)
            lines.append(f"- {today} {it['what']}（{it['where']}；{it['source']}）"
                         + (f"：{args.why}" if args.why else "") + f"（使用者 {today} 同意不修）")
        with open(BACKLOG, encoding="utf-8") as fh:
            text = fh.read()
        with open(BACKLOG, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text.rstrip("\n") + "\n" + "\n".join(lines) + "\n")
        save(items)
        for line in lines:
            print("backlog:", line)
    elif args.cmd == "unlist":
        with open(BACKLOG, encoding="utf-8") as fh:
            rows = fh.read().split("\n")
        hit = [i for i, r in enumerate(rows) if r.startswith("- ") and r[2:].startswith(args.start.lstrip("- "))]
        if len(hit) != 1:
            sys.exit(f"{len(hit)} backlog lines start with that (exactly one, please)")
        print("taken out:", rows[hit[0]])
        del rows[hit[0]]
        with open(BACKLOG, "w", encoding="utf-8", newline="\n") as fh:
            fh.write("\n".join(rows))
    else:
        shown = [it for it in items if args.cmd == "list" and getattr(args, "all", False) or it["state"] == "open"]
        for it in shown:
            print(show(it))
        if args.cmd == "gate" and shown:
            print(f"{len(shown)} review finding(s) open: fix each (`tools/findings.py fixed <id> --how ...`) or list "
                  "it for the user, who leaves it with `tools/findings.py skip <id>` (CLAUDE.md Workflow 3)",
                  file=sys.stderr)
            return 1
        if not shown:
            print("no open findings" if args.cmd != "list" or not getattr(args, "all", False) else "no findings")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
