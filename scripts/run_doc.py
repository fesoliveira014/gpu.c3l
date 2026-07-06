#!/usr/bin/env python3
"""Execute a markdown walkthrough.

Fenced blocks with marked info strings drive the run; everything else is
display-only prose:

    ```sh run          -- executed with bash -e, in order
    ```c3 file=path    -- written to path (any language tag works)

Usage: run_doc.py <doc.md> [--workspace DIR] [--dry-run]
"""

import argparse
import pathlib
import re
import subprocess
import sys

FENCE = re.compile(r"^```(\w+)((?:\s+\S+)*)\s*$")


def parse_blocks(text):
    blocks = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        m = FENCE.match(lines[i])
        if not m:
            i += 1
            continue
        info = m.group(2).split()
        body = []
        i += 1
        while i < len(lines) and lines[i].rstrip() != "```":
            body.append(lines[i])
            i += 1
        i += 1
        if "run" in info:
            blocks.append(("run", None, "\n".join(body)))
        else:
            for tok in info:
                if tok.startswith("file="):
                    blocks.append(("file", tok[5:], "\n".join(body)))
    return blocks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("doc")
    ap.add_argument("--workspace", default=".")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    ws = pathlib.Path(args.workspace).resolve()
    ws.mkdir(parents=True, exist_ok=True)
    blocks = parse_blocks(pathlib.Path(args.doc).read_text())
    if not blocks:
        print("run_doc: no marked blocks found", file=sys.stderr)
        return 1

    for kind, path, body in blocks:
        if kind == "file":
            target = ws / path
            if args.dry_run:
                print(f"-- would write {target} ({len(body)} chars)")
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(body + "\n")
            print(f"-- wrote {target}")
        else:
            if args.dry_run:
                print(f"-- would run:\n{body}\n")
                continue
            print(f"-- run:\n{body}")
            result = subprocess.run(["bash", "-ec", body], cwd=ws)
            if result.returncode != 0:
                print(f"run_doc: step failed with {result.returncode}", file=sys.stderr)
                return result.returncode
    print("run_doc: walkthrough complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
