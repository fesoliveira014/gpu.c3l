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
    "command_wrapper_bench",
    "command_path_baseline_bench",
    "command_reference_bench",
    "command_record_bench",
    "lifecycle_bench",
    "submit_batch_bench",
    "pipeline_cache_bench",
    "async_overlap_bench",
    "completion_wait_scope_bench",
)

BENCHMARK_METHODS = {
    "allocation_bench": ("4000/phase", "ns/allocation, ns/free"),
    "resource_create_bench": ("300/worker; workers=1,2,4", "ns/op"),
    "descriptor_churn_bench": (
        "320/worker; workers=1,2,4; sampler occupancy=8,64,1024,65536; ownership highwater=16,4096,65536",
        "ns/descriptor, ns/op, ns/destroy, ns/check; bounded sampler probes and ownership work",
    ),
    "upload_throughput_bench": (
        "warmup=1; payload_iterations=4096:2048,262144:512,4194304:32; workers=1,2,4",
        "uploads/s",
    ),
    "command_wrapper_bench": (
        "operations=5; direct/public=20000x5; alternating order",
        "ns/op; advisory public/direct ratio; exact observation",
    ),
    "command_path_baseline_bench": (
        "operations=5; direct/public=20000x5; lifecycle=0,1,16,256x5",
        "ns/op; advisory ratio; exact work and equivalence",
    ),
    "command_reference_bench": (
        "unique=1,8,64,256,1024,4096; repeated=100000; mixed=4096; collisions=64; near_capacity=4095",
        "ns/reference advisory; exact semantic balance; bounded probe and mutex work",
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
        "batch_sizes=1,8,32,128,1024; exact authoritative work",
        "ns/submit, ns/list advisory; exact work and allocation outcomes",
    ),
    "pipeline_cache_bench": (
        "raster=200; duplicate=200000; batch=64x2000; identity=1024,65536,1048576",
        "ns/create, ns/state; exact identity work",
    ),
    "async_overlap_bench": ("calibration=2; measured=5", "ms"),
    "completion_wait_scope_bench": (
        "16 producer/indirect-consumer pairs per scope",
        "ms; advisory all/draw_arguments ratio",
    ),
}

C3_BUILD_FLAGS = ("-O1",)
EXPECTATION_VERSION = 2

BENCHMARK_PROJECTS = {
    "command_wrapper_bench": "test/cpu",
}


CONTEXT_FIELDS = ("adapter:", "driver:", "validation:", "queues:")
# A unit token alone is not enough: the startup header already declares
# units=..., so the gate demands a number carrying the unit.
MEASURED_VALUE = re.compile(
    r"\d[\d,.]*\s?(?:ns/(?:allocation|free|op|descriptor|record|create|submit|poll|destroy|reference)|ms)\b"
    r"|uploads_per_sec=\d[\d,.]*\b"
    r"|(?:direct|public)_median_ns=\d[\d,.]*\b"
)
UPLOAD_MEASUREMENT = re.compile(
    r"\bworkers=\d+\s+payload_bytes=\d+\s+iterations=\d+\s+"
    r"uploads_per_sec=\d[\d,.]*\b"
)
ALLOCATION_NUMBER = r"[0-9]+(?:\.[0-9]+)?"
SUBMIT_BATCH_LINE = re.compile(
    rf"^submit batch size=(?P<size>[0-9]+) "
    rf"ns/submit=(?P<ns_submit>{ALLOCATION_NUMBER}) "
    rf"ns/list=(?P<ns_list>{ALLOCATION_NUMBER}) "
    r"resolutions=(?P<resolutions>[0-9]+) "
    r"duplicate_visits=(?P<duplicate_visits>[0-9]+) "
    r"epoch_reset_cells=(?P<epoch_reset_cells>[0-9]+) "
    r"command_mutex=(?P<command_mutex>[0-9]+) "
    r"queue_submission_mutex=(?P<queue_submission_mutex>[0-9]+) "
    r"rollback_mutex=(?P<rollback_mutex>[0-9]+) "
    r"native_submissions=(?P<native_submissions>[0-9]+) "
    r"host_allocations=(?P<host_allocations>[0-9]+)$"
)
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
COMMAND_RECORD_EXPECTATION = re.compile(
    r"^expectation_version=(?P<version>[1-9][0-9]*)$",
    re.MULTILINE,
)
COMMAND_RECORD_TOKENS = re.compile(
    r"^command_tokens: representation=bounded "
    r"recording_token_bytes=(?P<recording_token_bytes>[1-9][0-9]*) "
    r"executable_token_bytes=(?P<executable_token_bytes>[1-9][0-9]*) "
    r"record_bytes=(?P<record_bytes>[1-9][0-9]*) "
    r"cell_bytes=(?P<cell_bytes>[1-9][0-9]*) "
    r"fixed_storage_bytes=(?P<fixed_storage_bytes>[1-9][0-9]*) "
    r"commands_per_list=1,16,256,4096$",
    re.MULTILINE,
)
COMMAND_RECORD_RESOLUTION = re.compile(
    r"^resolution: recording_commands=(?P<recording_commands>[1-9][0-9]*) "
    r"native_commands=(?P<native_commands>[1-9][0-9]*) "
    r"pipeline_binds=(?P<pipeline_binds>[0-9]+) "
    r"descriptor_set_binds=(?P<descriptor_set_binds>[0-9]+) "
    r"descriptor_buffer_binds=(?P<descriptor_buffer_binds>[0-9]+) "
    r"descriptor_buffer_offsets=(?P<descriptor_buffer_offsets>[0-9]+) "
    r"device_registry=(?P<device_registry>[0-9]+) "
    r"retained_pins=(?P<retained_pins>[0-9]+) "
    r"lifecycle_vtable=(?P<lifecycle_vtable>[0-9]+) "
    r"command_table=(?P<command_table>[0-9]+) "
    r"pipeline_table=(?P<pipeline_table>[0-9]+) "
    r"pipeline_cache=(?P<pipeline_cache>[0-9]+) "
    r"policy=(?P<policy>[0-9]+) "
    r"encoder_cells=(?P<encoder_cells>[0-9]+) "
    r"encoder_leases=(?P<encoder_leases>[0-9]+)$",
    re.MULTILINE,
)
COMMAND_RECORD_COLD_WORK = re.compile(
    r"^cold work: host_allocations=[1-9][0-9]* "
    r"command_pool_creations=1 command_buffer_allocations=1 "
    r"command_buffer_frees=[0-9]+ "
    r"command_buffer_resets=[0-9]+ image_view_creations=[0-9]+ "
    r"vma_allocations=[0-9]+ generated_scratch_misses=[0-9]+$",
    re.MULTILINE,
)
COMMAND_RECORD_WARM_WORK = re.compile(
    r"^warm work: host_allocations=(?P<host_allocations>[0-9]+) "
    r"command_pool_creations=0 command_buffer_allocations=0 "
    r"command_buffer_frees=0 command_buffer_resets=[1-9][0-9]* "
    r"image_view_creations=0 vma_allocations=0 "
    r"generated_scratch_misses=0$",
    re.MULTILINE,
)
COMMAND_RECORD_GENERATED = re.compile(
    r"^generated dispatch: iterations=64 repetitions=5 "
    rf"median={ALLOCATION_NUMBER} ns/record$",
    re.MULTILINE,
)
COMMAND_RECORD_GENERATED_UNSUPPORTED = "generated dispatch: unsupported"
COMMAND_RECORD_EXPECTED_DIRECT_COMMANDS = 300_000
COMMAND_RECORD_EXPECTED_DIRECT_NATIVE_COMMANDS = 400_000
COMMAND_RECORD_EXPECTED_GENERATED_COMMANDS = 320
COMMAND_POLICY_EVIDENCE = re.compile(
    r"^validation policy=(?P<policy>trusted|object_boundaries|full) "
    r"tracking=(?P<tracking>true|false) layers=(?P<layers>true|false) "
    r"semantic_checks=(?P<semantic_checks>[0-9]+) "
    r"tracking_calls=(?P<tracking_calls>[0-9]+) "
    r"reference_allocations=(?P<reference_allocations>[0-9]+) "
    r"reference_increments=(?P<reference_increments>[0-9]+) "
    r"reference_releases=(?P<reference_releases>[0-9]+)$"
)
COMMAND_POLICY_MODES = (
    ("trusted", False, False),
    ("object_boundaries", False, False),
    ("full", True, False),
    ("full", True, True),
)
COMMAND_PATH_OPERATIONS = (
    "dispatch",
    "draw",
    "barrier",
    "viewport",
    "copy_buffer",
)
COMMAND_PATH_NATIVE_CALLS = {
    "dispatch": 2,
    "draw": 2,
    "barrier": 1,
    "viewport": 1,
    "copy_buffer": 1,
}
COMMAND_WRAPPER_CHECKSUMS = {
    "dispatch": 600_000,
    "draw": 1_000_000,
    "barrier": 1_400_000,
    "viewport": 2_200_000,
    "copy_buffer": 2_600_000,
}
COMMAND_WRAPPER_OPERATION = re.compile(
    r"^command_path_cpu operation=(?P<operation>[a-z_]+) "
    r"iterations=(?P<iterations>[0-9]+) repetitions=(?P<repetitions>[0-9]+) "
    r"checksum=(?P<checksum>[0-9]+) "
    r"direct_min_ns=(?P<direct_min>[0-9]+(?:\.[0-9]+)?) "
    r"direct_median_ns=(?P<direct_median>[0-9]+(?:\.[0-9]+)?) "
    r"direct_max_ns=(?P<direct_max>[0-9]+(?:\.[0-9]+)?) "
    r"public_min_ns=(?P<public_min>[0-9]+(?:\.[0-9]+)?) "
    r"public_median_ns=(?P<public_median>[0-9]+(?:\.[0-9]+)?) "
    r"public_max_ns=(?P<public_max>[0-9]+(?:\.[0-9]+)?) "
    r"ratio=(?P<ratio>[0-9]+(?:\.[0-9]+)?)$"
)
COMMAND_WRAPPER_CHECK = re.compile(
    r"^command_path_cpu_check operations=(?P<operations>[0-9]+) "
    r"observed=(?P<observed>[0-9]+) expected=(?P<expected>[0-9]+) "
    r"status=pass$"
)
COMMAND_PATH_VK_POLICY = re.compile(
    r"^command_path_vk_policy validation=(?P<policy>trusted|object_boundaries|full) "
    r"tracking=(?P<tracking>true|false) layers=(?P<layers>true|false) "
    r"resolution_stats=false recording_work_stats=true$"
)
COMMAND_PATH_VK_OPERATION = re.compile(
    r"^command_path_vk operation=(?P<operation>[a-z_]+) "
    r"iterations=(?P<iterations>[0-9]+) repetitions=(?P<repetitions>[0-9]+) "
    r"native_calls_per_iteration=(?P<native_calls>[0-9]+) "
    r"direct_min_ns=(?P<direct_min>[0-9]+(?:\.[0-9]+)?) "
    r"direct_median_ns=(?P<direct_median>[0-9]+(?:\.[0-9]+)?) "
    r"direct_max_ns=(?P<direct_max>[0-9]+(?:\.[0-9]+)?) "
    r"public_min_ns=(?P<public_min>[0-9]+(?:\.[0-9]+)?) "
    r"public_median_ns=(?P<public_median>[0-9]+(?:\.[0-9]+)?) "
    r"public_max_ns=(?P<public_max>[0-9]+(?:\.[0-9]+)?) "
    r"ratio=(?P<ratio>[0-9]+(?:\.[0-9]+)?)$"
)
COMMAND_PATH_VK_WORK = re.compile(
    r"^command_path_vk_work operation=(?P<operation>[a-z_]+) loops=(?P<loops>[0-9]+) "
    r"host_allocations=(?P<host_allocations>[0-9]+) "
    r"command_pool_creations=(?P<command_pool_creations>[0-9]+) "
    r"command_buffer_allocations=(?P<command_buffer_allocations>[0-9]+) "
    r"command_buffer_frees=(?P<command_buffer_frees>[0-9]+) "
    r"command_buffer_resets=(?P<command_buffer_resets>[0-9]+) "
    r"image_view_creations=(?P<image_view_creations>[0-9]+) "
    r"vma_allocations=(?P<vma_allocations>[0-9]+) "
    r"registry_lock_acquisitions=(?P<registry_lock_acquisitions>[0-9]+) "
    r"shader_module_creations=(?P<shader_module_creations>[0-9]+) "
    r"pipeline_creations=(?P<pipeline_creations>[0-9]+) status=pass$"
)
COMMAND_PATH_VK_EQUIVALENCE = re.compile(
    r"^command_path_vk_equivalence operation=(?P<operation>[a-z_]+) "
    r"elements=(?P<elements>[0-9]+) expected_checksum=(?P<expected>[0-9]+) "
    r"direct_checksum=(?P<direct>[0-9]+) public_checksum=(?P<public>[0-9]+) "
    r"pairwise=true status=pass$"
)
COMMAND_PATH_EQUIVALENCE_ELEMENTS = 64
COMMAND_PATH_EQUIVALENCE_CHECKSUMS = {
    "dispatch": 2 * sum(range(1, COMMAND_PATH_EQUIVALENCE_ELEMENTS + 1)),
    "copy_buffer": sum(range(1, COMMAND_PATH_EQUIVALENCE_ELEMENTS + 1)),
}
COMMAND_PATH_LIFECYCLE_CASES = (0, 1, 16, 256)
COMMAND_PATH_VK_LIFECYCLE = re.compile(
    r"^command_path_vk_lifecycle commands=(?P<commands>[0-9]+) "
    r"repetitions=(?P<repetitions>[0-9]+) "
    r"min_ns=(?P<minimum>[0-9]+(?:\.[0-9]+)?) "
    r"median_ns=(?P<median>[0-9]+(?:\.[0-9]+)?) "
    r"max_ns=(?P<maximum>[0-9]+(?:\.[0-9]+)?) "
    r"paired_delta_median_ns=(?P<paired_delta>-?[0-9]+(?:\.[0-9]+)?) "
    r"incremental_ns_per_command=(?P<incremental>-?[0-9]+(?:\.[0-9]+)?)$"
)
COMMAND_PATH_VK_LIFECYCLE_WORK = re.compile(
    r"^command_path_vk_lifecycle_work commands=(?P<commands>[0-9]+) "
    r"samples=(?P<samples>[0-9]+) "
    r"host_allocations=(?P<host_allocations>[0-9]+) "
    r"command_pool_creations=(?P<command_pool_creations>[0-9]+) "
    r"command_buffer_allocations=(?P<command_buffer_allocations>[0-9]+) "
    r"command_buffer_frees=(?P<command_buffer_frees>[0-9]+) "
    r"command_buffer_resets=(?P<command_buffer_resets>[0-9]+) "
    r"image_view_creations=(?P<image_view_creations>[0-9]+) "
    r"vma_allocations=(?P<vma_allocations>[0-9]+) "
    r"registry_lock_acquisitions=(?P<registry_lock_acquisitions>[0-9]+) "
    r"shader_module_creations=(?P<shader_module_creations>[0-9]+) "
    r"pipeline_creations=(?P<pipeline_creations>[0-9]+) status=pass$"
)
COMMAND_REFERENCE_UNIQUE_SIZES = (1, 8, 64, 256, 1_024, 4_096)
COMMAND_REFERENCE_WORK = re.compile(
    r"^reference_index (?P<kind>unique|mixed|collisions|near_capacity)="
    r"(?P<count>[0-9]+)(?: additional=(?P<additional>[0-9]+) "
    r"capacity_fault=(?P<capacity_fault>true|false)| unique=(?P<unique>[0-9]+))? "
    r"lookups=(?P<lookups>[0-9]+) probes=(?P<probes>[0-9]+) "
    r"equality=(?P<equality>[0-9]+) "
    r"(?:duplicates=(?P<duplicates>[0-9]+) )?"
    r"publications=(?P<publications>[0-9]+) mutex=(?P<mutex>[0-9]+) "
    r"retains=(?P<retains>[0-9]+) releases=(?P<releases>[0-9]+) "
    r"host_allocations=(?P<host_allocations>[0-9]+) "
    r"probes_per_reference=(?P<probes_per_reference>[0-9]+(?:\.[0-9]+)?) "
    r"equality_per_reference=(?P<equality_per_reference>[0-9]+(?:\.[0-9]+)?) "
    r"ns/reference=(?P<timing>[0-9]+(?:\.[0-9]+)?)$"
)
COMMAND_REFERENCE_REPEATED = re.compile(
    r"^reference_index repeated=(?P<count>[0-9]+) "
    r"lookups=(?P<lookups>[0-9]+) probes=(?P<probes>[0-9]+) "
    r"equality=(?P<equality>[0-9]+) duplicates=(?P<duplicates>[0-9]+) "
    r"publications=(?P<publications>[0-9]+) mutex=(?P<mutex>[0-9]+) "
    r"retains=(?P<retains>[0-9]+) releases=(?P<releases>[0-9]+) "
    r"host_allocations=(?P<host_allocations>[0-9]+) "
    r"ns/reference=(?P<timing>[0-9]+(?:\.[0-9]+)?)$"
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
    r"probes=(?P<probes>[0-9]+) empty_bucket_miss_probes=0 "
    r"elapsed_ns=(?P<elapsed_ns>[0-9]+)$"
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


def require_command_wrapper_evidence(output):
    lines = output.splitlines()
    if len(lines) != len(COMMAND_PATH_OPERATIONS) + 1:
        raise ValueError(
            "command_wrapper_bench record count is missing or duplicated"
        )
    for line, expected_operation in zip(
        lines[:len(COMMAND_PATH_OPERATIONS)],
        COMMAND_PATH_OPERATIONS,
    ):
        match = COMMAND_WRAPPER_OPERATION.fullmatch(line)
        if match is None:
            raise ValueError("command_wrapper_bench operation record is malformed")
        if match.group("operation") != expected_operation:
            raise ValueError(
                "command_wrapper_bench operation order or identity mismatch"
            )
        if int(match.group("iterations")) != 20_000:
            raise ValueError("command_wrapper_bench iteration count mismatch")
        if int(match.group("repetitions")) != 5:
            raise ValueError("command_wrapper_bench repetition count mismatch")
        if int(match.group("checksum")) != COMMAND_WRAPPER_CHECKSUMS[expected_operation]:
            raise ValueError("command_wrapper_bench operation checksum mismatch")
        direct = tuple(
            float(match.group(field))
            for field in ("direct_min", "direct_median", "direct_max")
        )
        public = tuple(
            float(match.group(field))
            for field in ("public_min", "public_median", "public_max")
        )
        if not (0.0 < direct[0] <= direct[1] <= direct[2]):
            raise ValueError("command_wrapper_bench direct timing range is invalid")
        if not (0.0 < public[0] <= public[1] <= public[2]):
            raise ValueError("command_wrapper_bench public timing range is invalid")
        ratio = float(match.group("ratio"))
        calculated = public[1] / direct[1]
        if abs(ratio - calculated) > max(1e-6, calculated * 1e-6):
            raise ValueError("command_wrapper_bench ratio calculation mismatch")
    check = COMMAND_WRAPPER_CHECK.fullmatch(lines[-1])
    if check is None:
        raise ValueError("command_wrapper_bench observation check is malformed")
    if int(check.group("operations")) != len(COMMAND_PATH_OPERATIONS):
        raise ValueError("command_wrapper_bench operation check count mismatch")
    observed = int(check.group("observed"))
    expected = int(check.group("expected"))
    if observed != expected or expected != 7_800_000:
        raise ValueError("command_wrapper_bench observation mismatch")


def require_command_path_vulkan_evidence(output, expected=None):
    lines = output.splitlines()
    operation_count = len(COMMAND_PATH_OPERATIONS)
    lifecycle_count = len(COMMAND_PATH_LIFECYCLE_CASES)
    expected_records = 1 + operation_count * 2 + 2 + lifecycle_count * 2
    if len(lines) != expected_records:
        raise ValueError(
            "command_path_baseline_bench record count is missing or duplicated"
        )
    policy_match = COMMAND_PATH_VK_POLICY.fullmatch(lines[0])
    if policy_match is None:
        raise ValueError("command_path_baseline_bench policy record is malformed")
    actual = (
        policy_match.group("policy"),
        policy_match.group("tracking") == "true",
        policy_match.group("layers") == "true",
    )
    if expected is not None and actual != expected:
        raise ValueError(
            "command_path_baseline_bench policy mode mismatch: "
            f"{actual} != {expected}"
        )

    operation_lines = lines[1:1 + operation_count]
    work_lines = lines[1 + operation_count:1 + operation_count * 2]
    for line, work_line, expected_operation in zip(
        operation_lines,
        work_lines,
        COMMAND_PATH_OPERATIONS,
    ):
        match = COMMAND_PATH_VK_OPERATION.fullmatch(line)
        if match is None:
            raise ValueError(
                "command_path_baseline_bench operation record is malformed"
            )
        if match.group("operation") != expected_operation:
            raise ValueError(
                "command_path_baseline_bench operation order or identity mismatch"
            )
        if int(match.group("iterations")) != 20_000:
            raise ValueError(
                "command_path_baseline_bench operation iteration count mismatch"
            )
        if int(match.group("repetitions")) != 5:
            raise ValueError(
                "command_path_baseline_bench operation repetition count mismatch"
            )
        if int(match.group("native_calls")) != COMMAND_PATH_NATIVE_CALLS[expected_operation]:
            raise ValueError(
                "command_path_baseline_bench native work count mismatch"
            )
        direct = tuple(
            float(match.group(field))
            for field in ("direct_min", "direct_median", "direct_max")
        )
        public = tuple(
            float(match.group(field))
            for field in ("public_min", "public_median", "public_max")
        )
        if not (0.0 < direct[0] <= direct[1] <= direct[2]):
            raise ValueError(
                "command_path_baseline_bench direct timing range is invalid"
            )
        if not (0.0 < public[0] <= public[1] <= public[2]):
            raise ValueError(
                "command_path_baseline_bench public timing range is invalid"
            )
        ratio = float(match.group("ratio"))
        calculated = public[1] / direct[1]
        if abs(ratio - calculated) > max(1e-6, calculated * 1e-6):
            raise ValueError(
                "command_path_baseline_bench ratio calculation mismatch"
            )

        work = COMMAND_PATH_VK_WORK.fullmatch(work_line)
        if work is None:
            raise ValueError(
                "command_path_baseline_bench operation work record is malformed"
            )
        if work.group("operation") != expected_operation:
            raise ValueError(
                "command_path_baseline_bench work order or identity mismatch"
            )
        if int(work.group("loops")) != 10:
            raise ValueError(
                "command_path_baseline_bench operation work loop count mismatch"
            )
        work_fields = (
            "host_allocations",
            "command_pool_creations",
            "command_buffer_allocations",
            "command_buffer_frees",
            "command_buffer_resets",
            "image_view_creations",
            "vma_allocations",
            "registry_lock_acquisitions",
            "shader_module_creations",
            "pipeline_creations",
        )
        if any(int(work.group(field)) != 0 for field in work_fields):
            raise ValueError(
                "command_path_baseline_bench operation structural work is nonzero"
            )

    equivalence_start = 1 + operation_count * 2
    equivalence_lines = lines[equivalence_start:equivalence_start + 2]
    for line, expected_operation in zip(
        equivalence_lines,
        ("dispatch", "copy_buffer"),
    ):
        match = COMMAND_PATH_VK_EQUIVALENCE.fullmatch(line)
        if match is None:
            raise ValueError(
                "command_path_baseline_bench equivalence record is malformed"
            )
        if match.group("operation") != expected_operation:
            raise ValueError(
                "command_path_baseline_bench equivalence order or identity mismatch"
            )
        if int(match.group("elements")) != COMMAND_PATH_EQUIVALENCE_ELEMENTS:
            raise ValueError(
                "command_path_baseline_bench equivalence element count mismatch"
            )
        expected = int(match.group("expected"))
        direct = int(match.group("direct"))
        public = int(match.group("public"))
        exact = COMMAND_PATH_EQUIVALENCE_CHECKSUMS[expected_operation]
        if expected != exact or direct != exact or public != exact:
            raise ValueError(
                "command_path_baseline_bench equivalence checksum mismatch"
            )

    lifecycle_start = equivalence_start + 2
    lifecycle_lines = lines[lifecycle_start:lifecycle_start + lifecycle_count]
    lifecycle_work_lines = lines[lifecycle_start + lifecycle_count:]
    for line, work_line, expected_commands in zip(
        lifecycle_lines,
        lifecycle_work_lines,
        COMMAND_PATH_LIFECYCLE_CASES,
    ):
        match = COMMAND_PATH_VK_LIFECYCLE.fullmatch(line)
        if match is None:
            raise ValueError(
                "command_path_baseline_bench lifecycle record is malformed"
            )
        if int(match.group("commands")) != expected_commands:
            raise ValueError(
                "command_path_baseline_bench lifecycle order or command count mismatch"
            )
        if int(match.group("repetitions")) != 5:
            raise ValueError(
                "command_path_baseline_bench lifecycle repetition count mismatch"
            )
        timings = tuple(
            float(match.group(field))
            for field in ("minimum", "median", "maximum")
        )
        if not (0.0 < timings[0] <= timings[1] <= timings[2]):
            raise ValueError(
                "command_path_baseline_bench lifecycle timing range is invalid"
            )
        paired_delta = float(match.group("paired_delta"))
        incremental = float(match.group("incremental"))
        if expected_commands == 0:
            if paired_delta != 0.0 or incremental != 0.0:
                raise ValueError(
                    "command_path_baseline_bench zero lifecycle increment is nonzero"
                )
        else:
            calculated = paired_delta / expected_commands
            if abs(incremental - calculated) > max(1e-3, abs(calculated) * 1e-6):
                raise ValueError(
                    "command_path_baseline_bench lifecycle increment calculation mismatch"
                )

        work = COMMAND_PATH_VK_LIFECYCLE_WORK.fullmatch(work_line)
        if work is None:
            raise ValueError(
                "command_path_baseline_bench lifecycle work record is malformed"
            )
        if int(work.group("commands")) != expected_commands:
            raise ValueError(
                "command_path_baseline_bench lifecycle work command count mismatch"
            )
        if int(work.group("samples")) != 5:
            raise ValueError(
                "command_path_baseline_bench lifecycle work sample count mismatch"
            )
        zero_fields = (
            "host_allocations",
            "command_pool_creations",
            "command_buffer_allocations",
            "command_buffer_frees",
            "image_view_creations",
            "vma_allocations",
            "shader_module_creations",
            "pipeline_creations",
        )
        if any(int(work.group(field)) != 0 for field in zero_fields):
            raise ValueError(
                "command_path_baseline_bench lifecycle unrelated work is nonzero"
            )
        if int(work.group("command_buffer_resets")) != 5:
            raise ValueError(
                "command_path_baseline_bench lifecycle reset count mismatch"
            )
        if int(work.group("registry_lock_acquisitions")) != 0:
            raise ValueError(
                "command_path_baseline_bench lifecycle registry-lock count mismatch"
            )
    return actual


def require_command_path_policy_matrix(outputs):
    if len(outputs) != len(COMMAND_POLICY_MODES):
        raise ValueError(
            "command_path_baseline_bench four-mode policy matrix is incomplete"
        )
    for output, expected in zip(outputs, COMMAND_POLICY_MODES):
        require_command_path_vulkan_evidence(output, expected)


def require_submit_batch_evidence(output):
    lines = output.splitlines()
    if len(lines) != len(SUBMIT_BATCH_SIZES) + 2:
        raise ValueError(
            "submit_batch_bench record count is missing or duplicated"
        )
    if lines[0] != (
        "iterations=5 batch_sizes=1,8,32,128,1024 "
        "units=ns/submit,ns/list"
    ):
        raise ValueError("submit_batch_bench header is malformed")

    zero_fields = (
        "epoch_reset_cells",
        "rollback_mutex",
        "host_allocations",
    )
    for line, expected_size in zip(lines[1:-1], SUBMIT_BATCH_SIZES):
        match = SUBMIT_BATCH_LINE.fullmatch(line)
        if match is None:
            raise ValueError(
                f"submit_batch_bench batch size {expected_size} schema is malformed"
            )
        if int(match.group("size")) != expected_size:
            raise ValueError(
                f"submit_batch_bench batch size {expected_size} is missing or reordered"
            )
        if float(match.group("ns_submit")) <= 0.0 or float(
            match.group("ns_list")
        ) <= 0.0:
            raise ValueError(
                f"submit_batch_bench batch size {expected_size} timing is invalid"
            )
        for field in ("resolutions", "duplicate_visits"):
            if int(match.group(field)) != expected_size:
                raise ValueError(
                    f"submit_batch_bench batch size {expected_size} {field} mismatch"
                )
        for field in (
            "command_mutex",
            "queue_submission_mutex",
            "native_submissions",
        ):
            if int(match.group(field)) != 1:
                raise ValueError(
                    f"submit_batch_bench batch size {expected_size} {field} mismatch"
                )
        if any(int(match.group(field)) != 0 for field in zero_fields):
            raise ValueError(
                f"submit_batch_bench batch size {expected_size} forbidden work is nonzero"
            )
    if lines[-1] != "submit batch leaks=0":
        raise ValueError("submit_batch_bench reports live resources")


def require_command_reference_evidence(output):
    lines = output.splitlines()
    if len(lines) != 12:
        raise ValueError(
            "command_reference_bench record count is missing or duplicated"
        )
    expected_header = (
        "iterations=unique:1,8,64,256,1024,4096;mixed:4096;"
        "near_capacity:4095;repeated:100000;collisions:64 "
        "units=ns/reference"
    )
    if lines[0] != expected_header:
        raise ValueError("command_reference_bench iteration matrix is malformed")

    def require_common(
        match,
        expected_count,
        expected_unique,
        expected_lookups=None,
    ):
        if match is None:
            raise ValueError("command_reference_bench work record is malformed")
        values = {
            field: int(match.group(field))
            for field in (
                "count",
                "lookups",
                "probes",
                "equality",
                "publications",
                "mutex",
                "retains",
                "releases",
                "host_allocations",
            )
        }
        if values["count"] != expected_count:
            raise ValueError("command_reference_bench workload size mismatch")
        if expected_lookups is None:
            expected_lookups = expected_count
        if values["lookups"] != expected_lookups:
            raise ValueError("command_reference_bench lookup work mismatch")
        if values["host_allocations"] != 0:
            raise ValueError("command_reference_bench warm allocation is nonzero")
        for field in ("publications", "retains", "releases"):
            if values[field] != expected_unique:
                raise ValueError(
                    f"command_reference_bench {field} work mismatch"
                )
        if values["mutex"] > expected_unique:
            raise ValueError(
                "command_reference_bench mutex work bound exceeded"
            )
        if values["equality"] > values["probes"]:
            raise ValueError(
                "command_reference_bench equality work bound exceeded"
            )
        probes_per_reference = float(match.group("probes_per_reference"))
        equality_per_reference = float(match.group("equality_per_reference"))
        expected_probe_ratio = values["probes"] / expected_count
        expected_equality_ratio = values["equality"] / expected_count
        if abs(probes_per_reference - expected_probe_ratio) > 0.001:
            raise ValueError("command_reference_bench probe ratio mismatch")
        if abs(equality_per_reference - expected_equality_ratio) > 0.001:
            raise ValueError("command_reference_bench equality ratio mismatch")
        if float(match.group("timing")) <= 0.0:
            raise ValueError("command_reference_bench advisory timing is invalid")
        return values

    for line, expected_count in zip(
        lines[1:7],
        COMMAND_REFERENCE_UNIQUE_SIZES,
    ):
        match = COMMAND_REFERENCE_WORK.fullmatch(line)
        if match is None or match.group("kind") != "unique":
            raise ValueError("command_reference_bench unique record is malformed")
        values = require_common(match, expected_count, expected_count)
        if values["probes"] > expected_count * 8:
            raise ValueError("command_reference_bench unique probe bound exceeded")

    repeated = COMMAND_REFERENCE_REPEATED.fullmatch(lines[7])
    if repeated is None:
        raise ValueError("command_reference_bench repeated record is malformed")
    repeated_count = 100_000
    exact_repeated = {
        "count": repeated_count,
        "lookups": repeated_count + 1,
        "duplicates": repeated_count,
        "publications": 1,
        "retains": 1,
        "releases": 1,
        "host_allocations": 0,
    }
    for field, expected in exact_repeated.items():
        if int(repeated.group(field)) != expected:
            raise ValueError(
                f"command_reference_bench repeated {field} work mismatch"
            )
    if int(repeated.group("probes")) > repeated_count + 1:
        raise ValueError(
            "command_reference_bench repeated probe bound exceeded"
        )
    if int(repeated.group("equality")) > repeated_count:
        raise ValueError(
            "command_reference_bench repeated equality bound exceeded"
        )
    if int(repeated.group("mutex")) > 1:
        raise ValueError(
            "command_reference_bench repeated mutex bound exceeded"
        )
    if float(repeated.group("timing")) <= 0.0:
        raise ValueError("command_reference_bench advisory timing is invalid")

    mixed = COMMAND_REFERENCE_WORK.fullmatch(lines[8])
    if (mixed is None or mixed.group("kind") != "mixed"
            or int(mixed.group("unique") or 0) != 2_048
            or int(mixed.group("duplicates") or 0) != 2_048):
        raise ValueError("command_reference_bench mixed record is malformed")
    mixed_values = require_common(mixed, 4_096, 2_048)
    if mixed_values["probes"] > 4_096 * 8:
        raise ValueError("command_reference_bench mixed probe bound exceeded")

    collision = COMMAND_REFERENCE_WORK.fullmatch(lines[9])
    if collision is None or collision.group("kind") != "collisions":
        raise ValueError("command_reference_bench collision record is malformed")
    collision_values = require_common(collision, 64, 64)
    if collision_values["probes"] > 64 * 65 // 2:
        raise ValueError(
            "command_reference_bench collision probe bound exceeded"
        )

    near = COMMAND_REFERENCE_WORK.fullmatch(lines[10])
    if (near is None or near.group("kind") != "near_capacity"
            or int(near.group("additional") or 0) != 2
            or near.group("capacity_fault") != "true"):
        raise ValueError("command_reference_bench capacity record is malformed")
    near_values = require_common(
        near,
        4_095,
        4_095,
        expected_lookups=4_097,
    )
    if near_values["probes"] > 4_095 * 8:
        raise ValueError("command_reference_bench capacity probe bound exceeded")

    if lines[11] != "reference_index_check status=pass":
        raise ValueError("command_reference_bench structural check failed")


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
        if not 0 <= probes <= 8:
            raise ValueError(
                "descriptor_churn_bench sampler lookup probes must be in [0, 8]"
            )


def require_command_policy_evidence(output, expected=None):
    lines = [
        line for line in output.splitlines()
        if line.startswith("validation ")
    ]
    if len(lines) != 1:
        raise ValueError(
            "command_record_bench validation policy evidence is missing or duplicated"
        )
    match = COMMAND_POLICY_EVIDENCE.fullmatch(lines[0])
    if match is None:
        raise ValueError("command_record_bench validation policy evidence is malformed")
    policy = match.group("policy")
    tracking = match.group("tracking") == "true"
    layers = match.group("layers") == "true"
    actual = (policy, tracking, layers)
    if expected is not None and actual != expected:
        raise ValueError(
            "command_record_bench policy mode mismatch: "
            f"{actual} != {expected}"
        )
    counters = {
        field: int(match.group(field))
        for field in (
            "semantic_checks",
            "tracking_calls",
            "reference_allocations",
            "reference_increments",
            "reference_releases",
        )
    }
    if policy == "full":
        if counters["semantic_checks"] == 0:
            raise ValueError("command_record_bench full policy is missing semantic work")
    elif counters["semantic_checks"] != 0:
        raise ValueError(
            "command_record_bench trusted lowering performed forbidden semantic work"
        )
    reference_fields = (
        "tracking_calls",
        "reference_allocations",
        "reference_increments",
        "reference_releases",
    )
    if tracking:
        if (
            counters["tracking_calls"] == 0
            or counters["reference_increments"] == 0
            or counters["reference_releases"] == 0
        ):
            raise ValueError("command_record_bench tracking policy is missing work")
        if counters["reference_allocations"] > counters["reference_increments"]:
            raise ValueError(
                "command_record_bench reference allocation/increment mismatch"
            )
        if counters["reference_releases"] != counters["reference_increments"]:
            raise ValueError(
                "command_record_bench reference release/increment mismatch"
            )
        if counters["reference_allocations"] != 0:
            raise ValueError(
                "command_record_bench warm reference allocation is nonzero"
            )
    elif any(counters[field] != 0 for field in reference_fields):
        raise ValueError(
            "command_record_bench tracking-off policy performed forbidden reference work"
        )
    warm_lines = [
        line for line in output.splitlines()
        if line.startswith("warm work:")
    ]
    if len(warm_lines) == 1:
        warm = COMMAND_RECORD_WARM_WORK.fullmatch(warm_lines[0])
        if warm is not None and int(warm.group("host_allocations")) != 0:
            raise ValueError(
                "command_record_bench warm host allocation is nonzero"
            )
    return actual, counters


def require_command_policy_matrix(outputs):
    if len(outputs) != len(COMMAND_POLICY_MODES):
        raise ValueError("command_record_bench four-mode policy matrix is incomplete")
    for output, expected in zip(outputs, COMMAND_POLICY_MODES):
        require_command_policy_evidence(output, expected)
        require_command_record_outcomes(output)


def require_command_record_outcomes(output):
    expectation_records = tuple(COMMAND_RECORD_EXPECTATION.finditer(output))
    if len(expectation_records) != 1:
        raise ValueError(
            "semantic invariant: command_record_bench expectation version "
            "is missing or duplicated"
        )
    version = int(expectation_records[0].group("version"))
    if version != EXPECTATION_VERSION:
        raise ValueError(
            "semantic invariant: command_record_bench expectation version "
            f"{version} != {EXPECTATION_VERSION}"
        )
    token_records = tuple(COMMAND_RECORD_TOKENS.finditer(output))
    if len(token_records) != 1:
        raise ValueError(
            "semantic invariant: command_record_bench token/storage record "
            "is missing, duplicated, or malformed"
        )
    token_record = token_records[0]
    recording_token_bytes = int(token_record.group("recording_token_bytes"))
    executable_token_bytes = int(
        token_record.group("executable_token_bytes")
    )
    if recording_token_bytes != executable_token_bytes:
        raise ValueError(
            "semantic invariant: command recording/executable token sizes differ"
        )
    cell_bytes = int(token_record.group("cell_bytes"))
    record_bytes = int(token_record.group("record_bytes"))
    fixed_storage_bytes = int(token_record.group("fixed_storage_bytes"))
    if record_bytes > cell_bytes or fixed_storage_bytes != cell_bytes * 4096:
        raise ValueError(
            "semantic invariant: command fixed-storage evidence is inconsistent"
        )
    invariant_lines = [
        line for line in output.splitlines()
        if line.startswith("invariants:")
    ]
    if (
        len(invariant_lines) != 1
        or COMMAND_RECORD_INVARIANTS.fullmatch(invariant_lines[0]) is None
    ):
        raise ValueError(
            "forbidden work: command_record_bench recording invariants "
            "are missing, duplicated, malformed, or nonzero"
        )
    resolution_records = tuple(COMMAND_RECORD_RESOLUTION.finditer(output))
    if len(resolution_records) != 1:
        raise ValueError(
            "semantic invariant: command_record_bench resolution record "
            "is missing, duplicated, or malformed"
        )
    resolution = resolution_records[0]
    values = {
        field: int(resolution.group(field))
        for field in (
            "recording_commands",
            "native_commands",
            "pipeline_binds",
            "descriptor_set_binds",
            "descriptor_buffer_binds",
            "descriptor_buffer_offsets",
            "device_registry",
            "retained_pins",
            "lifecycle_vtable",
            "command_table",
            "pipeline_table",
            "pipeline_cache",
            "policy",
            "encoder_cells",
            "encoder_leases",
        )
    }
    forbidden = (
        "pipeline_binds",
        "descriptor_set_binds",
        "descriptor_buffer_binds",
        "descriptor_buffer_offsets",
        "device_registry",
        "retained_pins",
        "lifecycle_vtable",
        "pipeline_table",
        "pipeline_cache",
        "policy",
    )
    nonzero = [field for field in forbidden if values[field] != 0]
    if nonzero:
        raise ValueError(
            "forbidden work: command_record_bench warm path reported "
            + ", ".join(nonzero)
        )
    generated_records = tuple(COMMAND_RECORD_GENERATED.finditer(output))
    generated_unsupported = sum(
        line == COMMAND_RECORD_GENERATED_UNSUPPORTED
        for line in output.splitlines()
    )
    if len(generated_records) + generated_unsupported != 1:
        raise ValueError(
            "semantic invariant: command_record_bench generated-dispatch "
            "record is missing, duplicated, or malformed"
        )
    expected_generated = (
        COMMAND_RECORD_EXPECTED_GENERATED_COMMANDS
        if generated_records else 0
    )
    expected_recording_commands = (
        COMMAND_RECORD_EXPECTED_DIRECT_COMMANDS + expected_generated
    )
    expected_native_commands = (
        COMMAND_RECORD_EXPECTED_DIRECT_NATIVE_COMMANDS + expected_generated
    )
    recording_commands = values["recording_commands"]
    if recording_commands != expected_recording_commands:
        raise ValueError(
            "semantic invariant: command_record_bench recording command "
            f"count {recording_commands} != {expected_recording_commands}"
        )
    expected_command_table = recording_commands
    if values["command_table"] != expected_command_table:
        raise ValueError(
            "forbidden work: command_record_bench command-table work "
            f"{values['command_table']} != {expected_command_table}"
        )
    if values["encoder_cells"] != 0:
        raise ValueError(
            "forbidden work: command_record_bench encoder-cell work is nonzero"
        )
    if values["encoder_leases"] != 0:
        raise ValueError(
            "forbidden work: command_record_bench encoder-lease work is nonzero"
        )
    if values["native_commands"] != expected_native_commands:
        raise ValueError(
            "minimal native lowering: command_record_bench native command "
            f"count {values['native_commands']} != {expected_native_commands}"
        )
    cold_lines = [
        line for line in output.splitlines()
        if line.startswith("cold work:")
    ]
    if (
        len(cold_lines) != 1
        or COMMAND_RECORD_COLD_WORK.fullmatch(cold_lines[0]) is None
    ):
        raise ValueError(
            "semantic invariant: command_record_bench cold work record "
            "is missing, duplicated, or malformed"
        )
    warm_lines = [
        line for line in output.splitlines()
        if line.startswith("warm work:")
    ]
    if (
        len(warm_lines) != 1
        or COMMAND_RECORD_WARM_WORK.fullmatch(warm_lines[0]) is None
    ):
        raise ValueError(
            "forbidden work: command_record_bench warm work is missing, "
            "duplicated, malformed, or nonzero"
        )


def require_measurement(
    output,
    target,
    enforce_thresholds=False,
    evaluate_thresholds=True,
):
    if target == "command_wrapper_bench":
        require_command_wrapper_evidence(output)
    if target == "command_path_baseline_bench":
        require_command_path_vulkan_evidence(output)
    if target == "command_reference_bench":
        require_command_reference_evidence(output)
    if target == "allocation_bench":
        for phase, pattern in ALLOCATION_PHASES:
            if not pattern.search(output):
                raise ValueError(f"{target} is missing {phase} measurement")
        if not ALLOCATION_SCHEMA.fullmatch(output):
            raise ValueError(f"{target} output does not match the exact schema")
    if target == "lifecycle_bench" and not LIFECYCLE_SCHEMA.fullmatch(output):
        raise ValueError(f"{target} output does not match the exact schema")
    if target == "submit_batch_bench":
        require_submit_batch_evidence(output)
    if target == "descriptor_churn_bench":
        require_sampler_lookup_evidence(output)
    if target == "command_record_bench":
        require_command_record_outcomes(output)
        require_command_policy_evidence(output)
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
                rf"owned_bytes_cloned={byte_count} "
                rf"lookup_shader_intern_probes=0 "
                rf"lookup_shader_bytes_compared=0 "
                rf"lookup_owned_bytes_cloned=0 pipeline_key_probes=[01] "
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
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode:
        diagnostics = result.stdout + result.stderr
        raise RuntimeError(f"{' '.join(command)} failed:\n{diagnostics}")
    if result.stderr:
        print(result.stderr.rstrip(), file=sys.stderr)
    return result.stdout.rstrip()


def benchmark_build_targets():
    targets = ("benchmark_info",) + BENCHMARK_TARGETS
    return tuple(
        (target, BENCHMARK_PROJECTS.get(target, "test"))
        for target in targets
    )


def executable(root, target):
    suffix = ".exe" if os.name == "nt" else ""
    project = BENCHMARK_PROJECTS.get(target, "test")
    return root / project / "build" / f"{target}{suffix}"


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

    run((sys.executable, "scripts/build_shaders.py"), root, env)
    for target, project in benchmark_build_targets():
        run(("c3c",) + C3_BUILD_FLAGS + ("build", target, "--path", project), root, env)

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
        f"- expectation_version={EXPECTATION_VERSION}",
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
        if target == "command_record_bench":
            command_outputs = []
            for policy, tracking, layers in COMMAND_POLICY_MODES:
                command_env = env.copy()
                command_env["GPU_C3L_BENCH_CONTRACT"] = policy
                command_env["GPU_C3L_BENCH_TRACKING"] = str(tracking).lower()
                command_env["GPU_C3L_BENCH_LAYERS"] = str(layers).lower()
                output = run(
                    (str(executable(root, target)),),
                    root,
                    command_env,
                )
                command_outputs.append(output)
                trusted_release = (
                    policy,
                    tracking,
                    layers,
                ) == COMMAND_POLICY_MODES[0]
                timing_advisories.extend(require_measurement(
                    output,
                    target,
                    enforce_thresholds=pinned and trusted_release,
                    evaluate_thresholds=trusted_release,
                ))
                annotated = f"iterations={iterations}\nunits={units}\n{output}"
                title = (
                    f"{target} [{policy} tracking={str(tracking).lower()} "
                    f"layers={str(layers).lower()}]"
                )
                lines.append(report_section(title, annotated))
            require_command_policy_matrix(command_outputs)
            continue
        if target == "command_path_baseline_bench":
            command_path_outputs = []
            for policy, tracking, layers in COMMAND_POLICY_MODES:
                command_env = env.copy()
                command_env["GPU_C3L_BENCH_CONTRACT"] = policy
                command_env["GPU_C3L_BENCH_TRACKING"] = str(tracking).lower()
                command_env["GPU_C3L_BENCH_LAYERS"] = str(layers).lower()
                output = run(
                    (str(executable(root, target)),),
                    root,
                    command_env,
                )
                command_path_outputs.append(output)
                timing_advisories.extend(require_measurement(
                    output,
                    target,
                    enforce_thresholds=False,
                    evaluate_thresholds=False,
                ))
                annotated = f"iterations={iterations}\nunits={units}\n{output}"
                title = (
                    f"{target} [{policy} tracking={str(tracking).lower()} "
                    f"layers={str(layers).lower()}]"
                )
                lines.append(report_section(title, annotated))
            require_command_path_policy_matrix(command_path_outputs)
            continue
        output = run((str(executable(root, target)),), root, env)
        timing_advisories.extend(require_measurement(
            output,
            target,
            enforce_thresholds=pinned,
            evaluate_thresholds=not args.validation,
        ))
        annotated = f"iterations={iterations}\nunits={units}\n{output}"
        lines.append(report_section(target, annotated))

    advisory_lines = [
        f"ADVISORY: {message}" for message in timing_advisories
    ]
    advisory_output = "none" if not advisory_lines else "\n".join(advisory_lines)
    for line in advisory_lines:
        print(line, file=sys.stderr)
    lines.append(report_section("Timing advisories", advisory_output))

    output_path = args.output if args.output.is_absolute() else root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
