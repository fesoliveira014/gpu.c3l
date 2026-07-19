#!/usr/bin/env python3
"""Build and run the fixed gpu.c3l performance baseline suite."""

import argparse
import datetime
import os
import pathlib
import platform
import re
import subprocess
import sys


BENCHMARK_TARGETS = (
    "arena_allocation_bench",
    "resource_create_bench",
    "descriptor_churn_bench",
    "upload_throughput_bench",
    "command_record_bench",
    "frame_signal_bench",
    "pipeline_cache_bench",
    "async_overlap_bench",
)

BENCHMARK_METHODS = {
    "arena_allocation_bench": ("frame=100000; persistent=4096", "ns/allocation, ns/free"),
    "resource_create_bench": ("300/worker; workers=1,2,4", "ns/op"),
    "descriptor_churn_bench": ("320/worker; workers=1,2,4", "ns/descriptor"),
    "upload_throughput_bench": (
        "warmup=1; payload_iterations=4096:2048,262144:512,4194304:32; workers=1,2,4",
        "uploads/s",
    ),
    "command_record_bench": ("20000/phase/repetition; repetitions=5", "ns/record"),
    "frame_signal_bench": ("2000/phase", "ns/end_frame, submits/frame"),
    "pipeline_cache_bench": ("cold=200; duplicate=200000", "ns/create"),
    "async_overlap_bench": ("calibration=2; measured=5", "ms"),
}

C3_BUILD_FLAGS = ("-O1",)


CONTEXT_FIELDS = ("adapter:", "driver:", "validation:", "queues:")
# A unit token alone is not enough: the startup header already declares
# units=..., so the gate demands a number carrying the unit.
MEASURED_VALUE = re.compile(
    r"\d[\d,.]*\s?(?:ns/(?:allocation|free|op|descriptor|record|end_frame|create)|ms)\b"
    r"|uploads_per_sec=\d[\d,.]*\b"
)
UPLOAD_MEASUREMENT = re.compile(
    r"\bworkers=\d+\s+payload_bytes=\d+\s+iterations=\d+\s+"
    r"uploads_per_sec=\d[\d,.]*\b"
)


def require_context_fields(output):
    for field in CONTEXT_FIELDS:
        if field not in output:
            raise ValueError(f"benchmark context is missing {field[:-1]}")


def require_measurement(output, target):
    if not re.search(r"\biterations?=\S+", output):
        raise ValueError(f"{target} is missing an iteration count")
    if "uploads_per_sec=" in output and not re.search(r"\bunits=uploads/s\b", output):
        raise ValueError(f"{target} is missing uploads/s units")
    if "uploads_per_sec=" in output and not UPLOAD_MEASUREMENT.search(output):
        raise ValueError(f"{target} is missing upload measurement fields")
    if not MEASURED_VALUE.search(output):
        raise ValueError(f"{target} is missing a measured value")


def run(command, cwd, env=None):
    result = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(f"{' '.join(command)} failed:\n{result.stdout}")
    return result.stdout.rstrip()


def executable(root, target):
    suffix = ".exe" if os.name == "nt" else ""
    return root / "test" / "build" / f"{target}{suffix}"


def report_section(title, output):
    return f"## {title}\n\n```text\n{output}\n```\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("test/build/benchmark-report.md"),
    )
    args = parser.parse_args()

    root = pathlib.Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["GPU_C3L_BENCH_VALIDATION"] = "0"
    env["VK_LOADER_LAYERS_DISABLE"] = "~implicit~"

    targets = ("benchmark_info",) + BENCHMARK_TARGETS
    run((sys.executable, "scripts/build_shaders.py"), root, env)
    for target in targets:
        run(("c3c",) + C3_BUILD_FLAGS + ("build", target, "--path", "test"), root, env)

    context = run((str(executable(root, "benchmark_info")),), root, env)
    require_context_fields(context)

    compiler = run(("c3c", "--version"), root, env).splitlines()[0]
    lines = [
        "# gpu.c3l benchmark report",
        "",
        f"- timestamp_utc={datetime.datetime.now(datetime.timezone.utc).isoformat()}",
        f"- host={platform.platform()}",
        f"- compiler={compiler}",
        f"- optimization={' '.join(C3_BUILD_FLAGS)}",
        "- validation=disabled",
        "- repetitions=one fixed suite invocation; target-internal repetitions are listed below",
        "",
        report_section("Context", context),
    ]

    for target in BENCHMARK_TARGETS:
        iterations, units = BENCHMARK_METHODS[target]
        output = run((str(executable(root, target)),), root, env)
        require_measurement(output, target)
        annotated = f"iterations={iterations}\nunits={units}\n{output}"
        lines.append(report_section(target, annotated))

    output_path = args.output if args.output.is_absolute() else root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
