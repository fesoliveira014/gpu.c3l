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
    "allocation_bench",
    "resource_create_bench",
    "descriptor_churn_bench",
    "upload_throughput_bench",
    "command_record_bench",
    "lifecycle_bench",
    "pipeline_cache_bench",
    "async_overlap_bench",
)

BENCHMARK_METHODS = {
    "allocation_bench": ("4000/phase", "ns/allocation, ns/free"),
    "resource_create_bench": ("300/worker; workers=1,2,4", "ns/op"),
    "descriptor_churn_bench": ("320/worker; workers=1,2,4", "ns/descriptor, ns/op"),
    "upload_throughput_bench": (
        "warmup=1; payload_iterations=4096:2048,262144:512,4194304:32; workers=1,2,4",
        "uploads/s",
    ),
    "command_record_bench": (
        "direct=20000/phase/repetition; generated=1000 prewarm+1000/repetition; repetitions=5",
        "ns/record",
    ),
    "lifecycle_bench": (
        "submit=256x5; poll=100000x5; destroy=300x5",
        "ns/submit, ns/poll, ns/destroy",
    ),
    "pipeline_cache_bench": ("cold=200; duplicate=200000; batch=64x2000", "ns/create"),
    "async_overlap_bench": ("calibration=2; measured=5", "ms"),
}

C3_BUILD_FLAGS = ("-O1",)


CONTEXT_FIELDS = ("adapter:", "driver:", "validation:", "queues:")
# A unit token alone is not enough: the startup header already declares
# units=..., so the gate demands a number carrying the unit.
MEASURED_VALUE = re.compile(
    r"\d[\d,.]*\s?(?:ns/(?:allocation|free|op|descriptor|record|create|submit|poll|destroy)|ms)\b"
    r"|uploads_per_sec=\d[\d,.]*\b"
)
UPLOAD_MEASUREMENT = re.compile(
    r"\bworkers=\d+\s+payload_bytes=\d+\s+iterations=\d+\s+"
    r"uploads_per_sec=\d[\d,.]*\b"
)
ALLOCATION_NUMBER = r"[0-9]+(?:\.[0-9]+)?"
ALLOCATION_PHASES = (
    (
        "cpu_write allocate",
        re.compile(
            rf"^cpu_write allocate: iterations=4000 size=64 align=16 "
            rf"{ALLOCATION_NUMBER} ns/allocation$",
            re.MULTILINE,
        ),
    ),
    (
        "cpu_write free",
        re.compile(
            rf"^cpu_write free: iterations=4000 size=64 align=16 "
            rf"{ALLOCATION_NUMBER} ns/free$",
            re.MULTILINE,
        ),
    ),
)
ALLOCATION_SCHEMA = re.compile(
    rf"\Acpu_write allocate: iterations=4000 size=64 align=16 "
    rf"{ALLOCATION_NUMBER} ns/allocation\r?\n"
    rf"cpu_write free: iterations=4000 size=64 align=16 "
    rf"{ALLOCATION_NUMBER} ns/free\Z"
)
LIFECYCLE_SCHEMA = re.compile(
    rf"\Asubmission: iterations=256 repetitions=5 median={ALLOCATION_NUMBER} ns/submit\r?\n"
    rf"completion poll: iterations=100000 repetitions=5 median={ALLOCATION_NUMBER} ns/poll\r?\n"
    rf"texture destroy: iterations=300 repetitions=5 median={ALLOCATION_NUMBER} ns/destroy\r?\n"
    r"invariants: point_allocations=0 destruction_waits=0 "
    r"deferred_releases=0\Z"
)
COMMAND_RECORD_INVARIANTS = re.compile(
    r"^invariants: registry_locks=0 recording_allocations=0 "
    r"draw_compilations=0 preprocess_allocations=0$",
    re.MULTILINE,
)

REGRESSION_THRESHOLDS = {
    "allocation_bench": (
        ("CPU_WRITE allocation", re.compile(r"cpu_write allocate:.* (?P<value>[0-9]+(?:\.[0-9]+)?) ns/allocation$"), 5_000.0, True),
        ("CPU_WRITE free", re.compile(r"cpu_write free:.* (?P<value>[0-9]+(?:\.[0-9]+)?) ns/free$"), 5_000.0, True),
    ),
    "command_record_bench": (
        ("barrier recording", re.compile(r"^barrier:.* median=(?P<value>[0-9]+(?:\.[0-9]+)?) ns/record$", re.MULTILINE), 2_000.0, True),
        ("hazard barrier recording", re.compile(r"^hazard barrier:.* median=(?P<value>[0-9]+(?:\.[0-9]+)?) ns/record$", re.MULTILINE), 2_000.0, True),
        ("indirect dispatch recording", re.compile(r"^indirect dispatch:.* median=(?P<value>[0-9]+(?:\.[0-9]+)?) ns/record$", re.MULTILINE), 3_000.0, True),
        ("generated dispatch recording", re.compile(r"^generated dispatch:.* median=(?P<value>[0-9]+(?:\.[0-9]+)?) ns/record$", re.MULTILINE), 20_000.0, False),
    ),
    "lifecycle_bench": (
        ("submission", re.compile(r"^submission:.* median=(?P<value>[0-9]+(?:\.[0-9]+)?) ns/submit$", re.MULTILINE), 100_000.0, True),
        ("completion poll", re.compile(r"^completion poll:.* median=(?P<value>[0-9]+(?:\.[0-9]+)?) ns/poll$", re.MULTILINE), 1_000.0, True),
        ("texture destruction", re.compile(r"^texture destroy:.* median=(?P<value>[0-9]+(?:\.[0-9]+)?) ns/destroy$", re.MULTILINE), 10_000.0, True),
    ),
    "pipeline_cache_bench": (
        ("cold pipeline creation", re.compile(r"^phase 1 .*: (?P<value>[0-9]+(?:\.[0-9]+)?) ns/create$", re.MULTILINE), 500_000.0, True),
        ("duplicate pipeline lookup", re.compile(r"^phase 2 .*: (?P<value>[0-9]+(?:\.[0-9]+)?) ns/create$", re.MULTILINE), 20_000.0, True),
        ("cached pipeline batch", re.compile(r"^phase 3 .*: (?P<value>[0-9]+(?:\.[0-9]+)?) ns/create$", re.MULTILINE), 20_000.0, True),
    ),
}

def require_context_fields(output):
    for field in CONTEXT_FIELDS:
        if field not in output:
            raise ValueError(f"benchmark context is missing {field[:-1]}")


def require_regression_thresholds(output, target):
    for label, pattern, maximum, required in REGRESSION_THRESHOLDS.get(target, ()):
        match = None
        for line in output.splitlines():
            match = pattern.search(line)
            if match is not None:
                break
        if match is None:
            if required:
                raise ValueError(f"{target} is missing {label} threshold measurement")
            continue
        value = float(match.group("value"))
        if value > maximum:
            raise ValueError(
                f"{target} {label} exceeded regression threshold: "
                f"{value:g} > {maximum:g}"
            )

def require_measurement(output, target, enforce_thresholds=True):
    if target == "allocation_bench":
        for phase, pattern in ALLOCATION_PHASES:
            if not pattern.search(output):
                raise ValueError(f"{target} is missing {phase} measurement")
        if not ALLOCATION_SCHEMA.fullmatch(output):
            raise ValueError(f"{target} output does not match the exact schema")
    if target == "lifecycle_bench" and not LIFECYCLE_SCHEMA.fullmatch(output):
        raise ValueError(f"{target} output does not match the exact schema")
    if target == "command_record_bench" and not COMMAND_RECORD_INVARIANTS.search(output):
        raise ValueError(f"{target} recording invariants are missing or nonzero")
    if enforce_thresholds:
        require_regression_thresholds(output, target)

    if not re.search(r"\biterations?=\S+", output):
        raise ValueError(f"{target} is missing an iteration count")
    is_upload = target == "upload_throughput_bench" or "uploads_per_sec=" in output
    if is_upload:
        if not re.search(r"\bunits=uploads/s\b", output):
            raise ValueError(f"{target} is missing uploads/s units")
        if not UPLOAD_MEASUREMENT.search(output):
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
    parser.add_argument(
        "--validation",
        action="store_true",
        help="enable debug validation and skip release-performance thresholds",
    )
    args = parser.parse_args()

    root = pathlib.Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["GPU_C3L_BENCH_VALIDATION"] = "1" if args.validation else "0"
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
        f"- validation={'enabled' if args.validation else 'disabled'}",
        "- repetitions=one fixed suite invocation; target-internal repetitions are listed below",
        "",
        report_section("Context", context),
    ]

    for target in BENCHMARK_TARGETS:
        iterations, units = BENCHMARK_METHODS[target]
        output = run((str(executable(root, target)),), root, env)
        require_measurement(output, target, enforce_thresholds=not args.validation)
        annotated = f"iterations={iterations}\nunits={units}\n{output}"
        lines.append(report_section(target, annotated))

    output_path = args.output if args.output.is_absolute() else root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
