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
    "submit_batch_bench",
    "pipeline_cache_bench",
    "async_overlap_bench",
)

BENCHMARK_METHODS = {
    "allocation_bench": ("4000/phase", "ns/allocation, ns/free"),
    "resource_create_bench": ("300/worker; workers=1,2,4", "ns/op"),
    "descriptor_churn_bench": (
        "320/worker; workers=1,2,4; sampler occupancy=8,64,1024,65536; ownership highwater=16,4096,65536",
        "ns/descriptor, ns/op, ns/destroy, ns/check; exact sampler probes and ownership work",
    ),
    "upload_throughput_bench": (
        "warmup=1; payload_iterations=4096:2048,262144:512,4194304:32; workers=1,2,4",
        "uploads/s",
    ),
    "command_record_bench": (
        "direct=20000/phase/repetition; generated=64 prewarm+64/repetition; repetitions=5; cold/warm work counters",
        "ns/record",
    ),
    "lifecycle_bench": (
        "submit=256x5; poll=100000x5; destroy=300x5",
        "ns/submit, ns/poll, ns/destroy",
    ),
    "submit_batch_bench": (
        "batch_sizes=1,8,32,128,1024; exact token visits",
        "ns/submit; exact work units",
    ),
    "pipeline_cache_bench": (
        "raster=200; duplicate=200000; batch=64x2000; identity=1024,65536,1048576",
        "ns/create, ns/state; exact identity work",
    ),
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
    r"invariants: point_allocations=0 destruction_queries=0 "
    r"destruction_completion_waits=0 cached_poll_queries=0 "
    r"retirement_locks=0\Z"
)
SUBMIT_BATCH_SIZES = (1, 8, 32, 128, 1024)
COMMAND_RECORD_INVARIANTS = re.compile(
    r"^invariants: registry_locks=0 recording_allocations=0 "
    r"draw_compilations=0 preprocess_allocations=0$",
    re.MULTILINE,
)
COMMAND_RECORD_RESOLUTION = re.compile(
    r"^resolution: recording_commands=[1-9][0-9]* "
    r"native_commands=[1-9][0-9]* device_registry=0 "
    r"retained_pins=0 lifecycle_vtable=0 command_table=0 "
    r"pipeline_table=0 pipeline_cache=0 policy=0$",
    re.MULTILINE,
)
COMMAND_RECORD_COLD_WORK = re.compile(
    r"^cold work: host_allocations=[0-9]+ "
    r"command_buffer_allocations=[0-9]+ command_buffer_frees=[0-9]+ "
    r"command_buffer_resets=[0-9]+ image_view_creations=[0-9]+ "
    r"vma_allocations=[0-9]+ generated_scratch_misses=[0-9]+$",
    re.MULTILINE,
)
COMMAND_RECORD_WARM_WORK = re.compile(
    r"^warm work: host_allocations=0 command_buffer_allocations=0 "
    r"command_buffer_frees=0 command_buffer_resets=[1-9][0-9]* "
    r"image_view_creations=0 vma_allocations=0 "
    r"generated_scratch_misses=0$",
    re.MULTILINE,
)
PIPELINE_CACHE_MATRIX = re.compile(
    r"^phase 1 \(raster matrix, requested=200 native=1 "
    r"cache_entries=1 aliases=200\): [0-9]+(?:\.[0-9]+)? ns/create$",
    re.MULTILINE,
)
PIPELINE_CACHE_RASTER_RECORDING = re.compile(
    r"^raster recording \(requested=200 native=1\): "
    r"[0-9]+(?:\.[0-9]+)? ns/state$",
    re.MULTILINE,
)
PIPELINE_IDENTITY_SIZES = (1_024, 65_536, 1_048_576)
SAMPLER_LOOKUP_OCCUPANCIES = (8, 64, 1_024, 65_536)
SAMPLER_LOOKUP_EVIDENCE = re.compile(
    r"^sampler lookup occupancy=(?P<occupancy>[0-9]+) "
    r"bucket_count=(?P<bucket_count>[0-9]+) "
    r"probes=(?P<probes>[0-9]+) elapsed_ns=(?P<elapsed_ns>[0-9]+)$"
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
        ("raster-matrix pipeline aliases", re.compile(r"^phase 1 .*: (?P<value>[0-9]+(?:\.[0-9]+)?) ns/create$", re.MULTILINE), 500_000.0, True),
        ("duplicate pipeline lookup", re.compile(r"^phase 2 .*: (?P<value>[0-9]+(?:\.[0-9]+)?) ns/create$", re.MULTILINE), 20_000.0, True),
        ("cached pipeline batch", re.compile(r"^phase 3 .*: (?P<value>[0-9]+(?:\.[0-9]+)?) ns/create$", re.MULTILINE), 20_000.0, True),
    ),
}

def require_context_fields(output):
    for field in CONTEXT_FIELDS:
        if field not in output:
            raise ValueError(f"benchmark context is missing {field[:-1]}")


def evaluate_regression_thresholds(output, target, enforce=False):
    advisories = []
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
            message = (
                f"{target} {label} exceeded regression threshold: "
                f"{value:g} > {maximum:g}"
            )
            if enforce:
                raise ValueError(message)
            advisories.append(message)
    return advisories

def require_sampler_lookup_evidence(output):
    lines = [
        line for line in output.splitlines()
        if line.startswith("sampler lookup ")
    ]
    if len(lines) != len(SAMPLER_LOOKUP_OCCUPANCIES):
        raise ValueError(
            "descriptor_churn_bench sampler lookup tiers are missing or duplicated"
        )
    for line, expected_occupancy in zip(lines, SAMPLER_LOOKUP_OCCUPANCIES):
        match = SAMPLER_LOOKUP_EVIDENCE.fullmatch(line)
        if match is None:
            raise ValueError(
                "descriptor_churn_bench sampler lookup evidence is malformed"
            )
        occupancy = int(match.group("occupancy"))
        bucket_count = int(match.group("bucket_count"))
        probes = int(match.group("probes"))
        if occupancy != expected_occupancy:
            raise ValueError(
                "descriptor_churn_bench sampler lookup occupancy mismatch: "
                f"{occupancy} != {expected_occupancy}"
            )
        if (
            bucket_count < occupancy * 2
            or bucket_count == 0
            or bucket_count & (bucket_count - 1)
        ):
            raise ValueError(
                "descriptor_churn_bench sampler bucket count must be a "
                "power of two at least twice occupancy"
            )
        if not 1 <= probes <= 8:
            raise ValueError(
                "descriptor_churn_bench sampler lookup probes must be in [1, 8]"
            )


def require_measurement(
    output,
    target,
    enforce_thresholds=False,
    evaluate_thresholds=True,
):
    if target == "allocation_bench":
        for phase, pattern in ALLOCATION_PHASES:
            if not pattern.search(output):
                raise ValueError(f"{target} is missing {phase} measurement")
        if not ALLOCATION_SCHEMA.fullmatch(output):
            raise ValueError(f"{target} output does not match the exact schema")
    if target == "lifecycle_bench" and not LIFECYCLE_SCHEMA.fullmatch(output):
        raise ValueError(f"{target} output does not match the exact schema")
    if target == "submit_batch_bench":
        for size in SUBMIT_BATCH_SIZES:
            pattern = re.compile(
                rf"^submit batch size={size}: {ALLOCATION_NUMBER} ns/submit "
                rf"token_visits={size} epoch_reset_cells=0$",
                re.MULTILINE,
            )
            if not pattern.search(output):
                raise ValueError(
                    f"{target} is missing exact work for batch size {size}"
                )
        if "submit batch leaks=0" not in output:
            raise ValueError(f"{target} reports live resources")
    if target == "descriptor_churn_bench":
        require_sampler_lookup_evidence(output)
    if target == "command_record_bench" and not COMMAND_RECORD_INVARIANTS.search(output):
        raise ValueError(f"{target} recording invariants are missing or nonzero")
    if target == "command_record_bench" and not COMMAND_RECORD_RESOLUTION.search(output):
        raise ValueError(
            f"{target} recording resolution evidence is missing or nonzero"
        )
    if target == "command_record_bench" and not COMMAND_RECORD_COLD_WORK.search(output):
        raise ValueError(f"{target} cold recording work evidence is missing")
    if target == "command_record_bench" and not COMMAND_RECORD_WARM_WORK.search(output):
        raise ValueError(f"{target} warm recording work is missing or nonzero")
    if target == "pipeline_cache_bench" and not PIPELINE_CACHE_MATRIX.search(output):
        raise ValueError(
            f"{target} raster matrix did not collapse to one native pipeline"
        )
    if (
        target == "pipeline_cache_bench"
        and not PIPELINE_CACHE_RASTER_RECORDING.search(output)
    ):
        raise ValueError(f"{target} raster recording evidence is missing")
    if target == "pipeline_cache_bench":
        for byte_count in PIPELINE_IDENTITY_SIZES:
            evidence = re.compile(
                rf"^identity size_bytes={byte_count} intern_probes=[0-9]+ "
                rf"intern_bytes_compared=[0-9]+ "
                rf"owned_bytes_cloned={byte_count} pipeline_key_probes=1 "
                rf"owned_bytes_freed={byte_count} elapsed_ns=[0-9]+$",
                re.MULTILINE,
            )
            if not evidence.search(output):
                raise ValueError(
                    f"{target} identity evidence for {byte_count} bytes is "
                    "missing, malformed, or nonzero after interning"
                )
    advisories = []
    if evaluate_thresholds:
        advisories = evaluate_regression_thresholds(
            output,
            target,
            enforce=enforce_thresholds,
        )

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
    return advisories


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
    parser.add_argument("--pinned-runner")
    parser.add_argument("--pinned-driver")
    parser.add_argument("--comparison-profile")
    args = parser.parse_args()
    pinned_fields = (
        args.pinned_runner,
        args.pinned_driver,
        args.comparison_profile,
    )
    if any(pinned_fields) and not all(pinned_fields):
        parser.error(
            "--pinned-runner, --pinned-driver, and --comparison-profile "
            "must be supplied together"
        )
    pinned = all(pinned_fields)
    if args.validation and pinned:
        parser.error(
            "--validation cannot be combined with pinned comparison fields"
        )

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
        f"- timing_mode={'blocking' if pinned else 'advisory'}",
        f"- pinned_runner={args.pinned_runner or 'none'}",
        f"- pinned_driver={args.pinned_driver or 'none'}",
        f"- comparison_profile={args.comparison_profile or 'none'}",
        "- repetitions=one fixed suite invocation; target-internal repetitions are listed below",
        "",
        report_section("Context", context),
    ]

    timing_advisories = []
    for target in BENCHMARK_TARGETS:
        iterations, units = BENCHMARK_METHODS[target]
        output = run((str(executable(root, target)),), root, env)
        timing_advisories.extend(require_measurement(
            output,
            target,
            enforce_thresholds=pinned,
            evaluate_thresholds=not args.validation,
        ))
        annotated = f"iterations={iterations}\nunits={units}\n{output}"
        lines.append(report_section(target, annotated))

    advisory_output = "none" if not timing_advisories else "\n".join(
        f"ADVISORY: {message}" for message in timing_advisories
    )
    lines.append(report_section("Timing advisories", advisory_output))

    output_path = args.output if args.output.is_absolute() else root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
