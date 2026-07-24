#!/usr/bin/env python3
"""Verify C3 feature-exclusive command return signatures."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SPIKE_PROJECT = ROOT / "test" / "command_profile_spike"
CPU_PROJECT = ROOT / "test" / "cpu"
PASS_TARGETS = (
    (SPIKE_PROJECT, "fast_pass"),
    (SPIKE_PROJECT, "checked_pass"),
    (CPU_PROJECT, "command_profile_surface_fast"),
    (CPU_PROJECT, "command_profile_surface_checked"),
)
FAIL_TARGETS = {
    (SPIKE_PROJECT, "fast_rethrow"):
        "No optional to rethrow before '!'",
    (SPIKE_PROJECT, "checked_unhandled"):
        "which is an optional and must be handled",
    (CPU_PROJECT, "command_profile_fast_rethrow_fail"):
        "No optional to rethrow before '!'",
    (CPU_PROJECT, "command_profile_checked_unhandled_fail"):
        "which is an optional and must be handled",
}


def build(
    project: Path,
    target: str,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["c3c", "build", target, "--path", str(project)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def main() -> int:
    failures = []
    for project, target in PASS_TARGETS:
        result = build(project, target)
        if result.returncode != 0:
            failures.append(f"{target} did not compile")
    for (project, target), diagnostic in FAIL_TARGETS.items():
        result = build(project, target)
        output = result.stdout + result.stderr
        if result.returncode == 0:
            failures.append(f"{target} unexpectedly compiled")
        elif diagnostic not in output:
            failures.append(f"{target} failed without its expected diagnostic")

    if failures:
        print("command profile signature spike failures:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    print("command profile feature-exclusive signatures compile as expected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
