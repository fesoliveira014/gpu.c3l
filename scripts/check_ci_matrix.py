"""Verify the blocking CI target matrix against the project and documentation."""

import json
import os
import sys
from pathlib import Path


MATRIX_HEADING = "The blocking headless matrix is shared by Linux and Windows:"


def read_project_targets(path):
    source = "".join(
        line for line in path.read_text(encoding="utf-8").splitlines(keepends=True)
        if not line.lstrip().startswith("//")
    )
    project = json.loads(source)
    return [
        name for name, target in project["targets"].items()
        if target.get("type") == "test"
    ]


def read_documented_targets(path):
    source = path.read_text(encoding="utf-8")
    heading = source.index(MATRIX_HEADING)
    block_start = source.index("```text", heading) + len("```text")
    block_end = source.index("```", block_start)
    return source[block_start:block_end].split()


def report_mismatch(label, actual, expected):
    print(f"{label} matrix mismatch", file=sys.stderr)
    print(f"  actual:   {' '.join(actual)}", file=sys.stderr)
    print(f"  expected: {' '.join(expected)}", file=sys.stderr)


def main(root=None):
    root = root or Path(__file__).resolve().parents[1]
    configured = os.environ.get("HEADLESS_TEST_TARGETS", "").split()
    project = read_project_targets(root / "test" / "project.json")
    documented = read_documented_targets(root / "docs" / "testing.md")

    valid = True
    if sorted(configured) != sorted(project):
        report_mismatch("workflow", configured, project)
        valid = False
    if documented != configured:
        report_mismatch("documentation", documented, configured)
        valid = False
    if not valid:
        return 1

    print(f"CI matrix matches {len(project)} test targets and documentation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
