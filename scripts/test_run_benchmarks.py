#!/usr/bin/env python3

import importlib.util
import pathlib
import subprocess
import sys
import tempfile
import textwrap
import unittest


SCRIPT = pathlib.Path(__file__).with_name("run_benchmarks.py")
UPLOAD_BENCHMARK = SCRIPT.parents[1] / "test" / "src" / "upload_throughput_bench.c3"
BENCH_LIFETIME = SCRIPT.parents[1] / "test" / "src" / "bench_lifetime.c3"
ALLOCATION_OUTPUT = "\n".join(
    (
        "cpu_write allocate: iterations=4000 size=64 align=16 23.75 ns/allocation",
        "cpu_write free: iterations=4000 size=64 align=16 8.25 ns/free",
    )
)
LIFECYCLE_OUTPUT = "\n".join(
    (
        "submission: iterations=256 repetitions=5 median=8400.0 ns/submit",
        "completion poll: iterations=100000 repetitions=5 median=125.0 ns/poll",
        "texture destroy: iterations=300 repetitions=5 median=410.0 ns/destroy",
        (
            "invariants: point_allocations=0 destruction_queries=0 "
            "destruction_completion_waits=0 cached_poll_queries=0 "
            "retirement_locks=0"
        ),
    )
)
COMMAND_OUTPUT = "\n".join(
    (
        "expectation_version=3",
        (
            "command_tokens: representation=direct "
            "recording_token_bytes=16 executable_token_bytes=16 "
            "record_bytes=56 cell_bytes=456 fixed_storage_bytes=1867776 "
            "commands_per_list=1,16,256,4096"
        ),
        "barrier: iterations=20000 repetitions=5 median=130.0 ns/record",
        "hazard barrier: iterations=20000 repetitions=5 median=135.0 ns/record",
        "indirect dispatch: iterations=20000 repetitions=5 median=180.0 ns/record",
        "generated dispatch: iterations=64 repetitions=5 median=900.0 ns/record",
        (
            "invariants: registry_locks=0 recording_allocations=0 "
            "draw_compilations=0 preprocess_allocations=0"
        ),
        (
            "resolution: recording_commands=300320 native_commands=400320 "
            "pipeline_binds=0 descriptor_set_binds=0 "
            "descriptor_buffer_binds=0 descriptor_buffer_offsets=0 "
            "device_registry=0 "
            "retained_pins=0 command_table=0 "
            "pipeline_table=0 pipeline_cache=0 policy=0 "
            "encoder_cells=0 encoder_leases=0"
        ),
        (
            "validation policy=trusted tracking=false layers=false "
            "semantic_checks=0 tracking_calls=0 reference_allocations=0 "
            "reference_increments=0 reference_releases=0"
        ),
        (
            "cold work: host_allocations=5 command_pool_creations=1 "
            "command_buffer_allocations=1 "
            "command_buffer_frees=0 command_buffer_resets=3 "
            "image_view_creations=0 vma_allocations=64 "
            "generated_scratch_misses=0"
        ),
        (
            "warm work: host_allocations=0 command_pool_creations=0 "
            "command_buffer_allocations=0 "
            "command_buffer_frees=0 command_buffer_resets=20 "
            "image_view_creations=0 vma_allocations=0 "
            "generated_scratch_misses=0"
        ),
        "generated preprocess: reuse_events=320",
    )
)
OBJECT_BOUNDARIES_COMMAND_OUTPUT = COMMAND_OUTPUT.replace(
    "policy=trusted",
    "policy=object_boundaries",
)
FULL_COMMAND_OUTPUT = COMMAND_OUTPUT.replace(
    (
        "policy=trusted tracking=false layers=false semantic_checks=0 "
        "tracking_calls=0 reference_allocations=0 reference_increments=0 "
        "reference_releases=0"
    ),
    (
        "policy=full tracking=true layers=false semantic_checks=300330 "
        "tracking_calls=200335 reference_allocations=0 "
        "reference_increments=25 reference_releases=25"
    ),
)
FULL_LAYERS_COMMAND_OUTPUT = FULL_COMMAND_OUTPUT.replace(
    "layers=false",
    "layers=true",
)
COMMAND_POLICY_OUTPUTS = (
    COMMAND_OUTPUT,
    OBJECT_BOUNDARIES_COMMAND_OUTPUT,
    FULL_COMMAND_OUTPUT,
    FULL_LAYERS_COMMAND_OUTPUT,
)
COMMAND_REFERENCE_OUTPUT = "\n".join((
    "iterations=unique:1,8,64,256,1024,4096;mixed:4096;near_capacity:4095;repeated:100000;collisions:64 units=ns/reference",
    "reference_index unique=1 lookups=1 probes=1 equality=0 publications=1 mutex=1 retains=1 releases=1 host_allocations=0 probes_per_reference=1.000 equality_per_reference=0.000 ns/reference=50.000",
    "reference_index unique=8 lookups=8 probes=13 equality=5 publications=8 mutex=8 retains=8 releases=8 host_allocations=0 probes_per_reference=1.625 equality_per_reference=0.625 ns/reference=50.000",
    "reference_index unique=64 lookups=64 probes=90 equality=26 publications=64 mutex=64 retains=64 releases=64 host_allocations=0 probes_per_reference=1.406 equality_per_reference=0.406 ns/reference=50.000",
    "reference_index unique=256 lookups=256 probes=327 equality=71 publications=256 mutex=256 retains=256 releases=256 host_allocations=0 probes_per_reference=1.277 equality_per_reference=0.277 ns/reference=50.000",
    "reference_index unique=1024 lookups=1024 probes=1746 equality=722 publications=1024 mutex=1024 retains=1024 releases=1024 host_allocations=0 probes_per_reference=1.705 equality_per_reference=0.705 ns/reference=50.000",
    "reference_index unique=4096 lookups=4096 probes=6041 equality=1945 publications=4096 mutex=4096 retains=4096 releases=4096 host_allocations=0 probes_per_reference=1.475 equality_per_reference=0.475 ns/reference=50.000",
    "reference_index repeated=100000 lookups=100001 probes=100001 equality=100000 duplicates=100000 publications=1 mutex=1 retains=1 releases=1 host_allocations=0 ns/reference=25.000",
    "reference_index mixed=4096 unique=2048 lookups=4096 probes=6170 equality=4122 duplicates=2048 publications=2048 mutex=2048 retains=2048 releases=2048 host_allocations=0 probes_per_reference=1.506 equality_per_reference=1.006 ns/reference=40.000",
    "reference_index collisions=64 lookups=64 probes=2080 equality=2016 publications=64 mutex=64 retains=64 releases=64 host_allocations=0 probes_per_reference=32.500 equality_per_reference=31.500 ns/reference=220.000",
    "reference_index near_capacity=4095 additional=2 capacity_fault=true lookups=4097 probes=6042 equality=1945 publications=4095 mutex=4095 retains=4095 releases=4095 host_allocations=0 probes_per_reference=1.475 equality_per_reference=0.475 ns/reference=50.000",
    "reference_index_check status=pass",
))
PIPELINE_OUTPUT = "\n".join(
    (
        "iterations=raster=200;duplicate=200000;batch=64x2000 "
        "units=ns/create,ns/state",
        (
            "phase 1 (raster matrix, requested=200 native=1 "
            "cache_entries=1 aliases=200): 42621.0 ns/create"
        ),
        "raster recording (requested=200 native=1): 99.0 ns/state",
        "phase 2 (duplicate, 200000 at full alias set): 1388.1 ns/create",
        "phase 3 (cached batch, 64x2000): 2454.9 ns/create",
        (
            "identity size_bytes=1024 intern_probes=0 "
            "intern_bytes_compared=0 owned_bytes_cloned=1024 "
            "lookup_shader_intern_probes=0 lookup_shader_bytes_compared=0 "
            "lookup_owned_bytes_cloned=0 "
            "pipeline_key_probes=1 owned_bytes_freed=1024 elapsed_ns=1200"
        ),
        (
            "identity size_bytes=65536 intern_probes=0 "
            "intern_bytes_compared=0 owned_bytes_cloned=65536 "
            "lookup_shader_intern_probes=0 lookup_shader_bytes_compared=0 "
            "lookup_owned_bytes_cloned=0 "
            "pipeline_key_probes=1 owned_bytes_freed=65536 elapsed_ns=8400"
        ),
        (
            "identity size_bytes=1048576 intern_probes=0 "
            "intern_bytes_compared=0 owned_bytes_cloned=1048576 "
            "lookup_shader_intern_probes=0 lookup_shader_bytes_compared=0 "
            "lookup_owned_bytes_cloned=0 "
            "pipeline_key_probes=1 owned_bytes_freed=1048576 elapsed_ns=94000"
        ),
    )
)
SUBMIT_BATCH_OUTPUT = "\n".join(
    (
        "iterations=5 batch_sizes=1,8,32,128,1024 "
        "units=ns/submit,ns/list",
    ) + tuple(
        f"submit batch size={size} ns/submit=125.0 "
        f"ns/list={125.0 / size:.3f} "
        f"resolutions={size} duplicate_visits={size} "
        "epoch_reset_cells=0 command_mutex=1 "
        "queue_submission_mutex=1 "
        "rollback_mutex=0 native_submissions=1 host_allocations=0"
        for size in (1, 8, 32, 128, 1024)
    ) + ("submit batch leaks=0",)
)
DESCRIPTOR_CHURN_OUTPUT = "\n".join(
    (
        "iterations=320/worker units=ns/descriptor,ns/op",
        "phase sampler intern+publish hits workers=1: 120.0 ns/op, 1.00x scaling vs 1 thread",
        "sampler lookup occupancy=8 bucket_count=16 probes=1 empty_bucket_miss_probes=0 elapsed_ns=20",
        "sampler lookup occupancy=64 bucket_count=128 probes=1 empty_bucket_miss_probes=0 elapsed_ns=20",
        "sampler lookup occupancy=1024 bucket_count=2048 probes=1 empty_bucket_miss_probes=0 elapsed_ns=20",
        "sampler lookup occupancy=65536 bucket_count=131072 probes=1 empty_bucket_miss_probes=0 elapsed_ns=20",
    )
)
COMMAND_WRAPPER_OUTPUT = "\n".join(
    (
        "command_path_cpu operation=dispatch iterations=20000 repetitions=5 checksum=600000 direct_min_ns=1.000 direct_median_ns=2.000 direct_max_ns=3.000 public_min_ns=4.000 public_median_ns=5.000 public_max_ns=6.000 ratio=2.500000",
        "command_path_cpu operation=draw iterations=20000 repetitions=5 checksum=1000000 direct_min_ns=1.000 direct_median_ns=2.000 direct_max_ns=3.000 public_min_ns=4.000 public_median_ns=5.000 public_max_ns=6.000 ratio=2.500000",
        "command_path_cpu operation=barrier iterations=20000 repetitions=5 checksum=1400000 direct_min_ns=1.000 direct_median_ns=2.000 direct_max_ns=3.000 public_min_ns=4.000 public_median_ns=5.000 public_max_ns=6.000 ratio=2.500000",
        "command_path_cpu operation=viewport iterations=20000 repetitions=5 checksum=2200000 direct_min_ns=1.000 direct_median_ns=2.000 direct_max_ns=3.000 public_min_ns=4.000 public_median_ns=5.000 public_max_ns=6.000 ratio=2.500000",
        "command_path_cpu operation=copy_buffer iterations=20000 repetitions=5 checksum=2600000 direct_min_ns=1.000 direct_median_ns=2.000 direct_max_ns=3.000 public_min_ns=4.000 public_median_ns=5.000 public_max_ns=6.000 ratio=2.500000",
        "command_path_cpu_check operations=5 observed=7800000 expected=7800000 status=pass",
    )
)
COMMAND_PATH_VK_OUTPUT = "\n".join(
    (
        "command_path_vk_policy validation=trusted tracking=false layers=false resolution_stats=false recording_work_stats=true",
        *tuple(
            f"command_path_vk operation={operation} iterations=20000 repetitions=5 native_calls_per_iteration={native_calls} direct_min_ns=1.000 direct_median_ns=2.000 direct_max_ns=3.000 public_min_ns=4.000 public_median_ns=5.000 public_max_ns=6.000 ratio=2.500000"
            for operation, native_calls in (
                ("dispatch", 2),
                ("draw", 2),
                ("barrier", 1),
                ("viewport", 1),
                ("copy_buffer", 1),
            )
        ),
        *tuple(
            f"command_path_vk_work operation={operation} loops=10 host_allocations=0 command_pool_creations=0 command_buffer_allocations=0 command_buffer_frees=0 command_buffer_resets=0 image_view_creations=0 vma_allocations=0 registry_lock_acquisitions=0 shader_module_creations=0 pipeline_creations=0 status=pass"
            for operation in (
                "dispatch",
                "draw",
                "barrier",
                "viewport",
                "copy_buffer",
            )
        ),
        "command_path_vk_equivalence operation=dispatch elements=64 expected_checksum=4160 direct_checksum=4160 public_checksum=4160 pairwise=true status=pass",
        "command_path_vk_equivalence operation=copy_buffer elements=64 expected_checksum=2080 direct_checksum=2080 public_checksum=2080 pairwise=true status=pass",
        "command_path_vk_lifecycle commands=0 repetitions=5 min_ns=100.000 median_ns=110.000 max_ns=120.000 paired_delta_median_ns=0.000 incremental_ns_per_command=0.000",
        "command_path_vk_lifecycle commands=1 repetitions=5 min_ns=190.000 median_ns=210.000 max_ns=230.000 paired_delta_median_ns=100.000 incremental_ns_per_command=100.000",
        "command_path_vk_lifecycle commands=16 repetitions=5 min_ns=250.000 median_ns=270.000 max_ns=290.000 paired_delta_median_ns=160.000 incremental_ns_per_command=10.000",
        "command_path_vk_lifecycle commands=256 repetitions=5 min_ns=2600.000 median_ns=2670.000 max_ns=2700.000 paired_delta_median_ns=2560.000 incremental_ns_per_command=10.000",
        *tuple(
            f"command_path_vk_lifecycle_work commands={commands} samples=5 host_allocations=0 command_pool_creations=0 command_buffer_allocations=0 command_buffer_frees=0 command_buffer_resets=5 image_view_creations=0 vma_allocations=0 registry_lock_acquisitions=0 shader_module_creations=0 pipeline_creations=0 status=pass"
            for commands in (0, 1, 16, 256)
        ),
    )
)
COMMAND_PATH_VK_OUTPUTS = tuple(
    COMMAND_PATH_VK_OUTPUT.replace(
        "validation=trusted tracking=false layers=false",
        f"validation={policy} tracking={str(tracking).lower()} "
        f"layers={str(layers).lower()}",
        1,
    )
    for policy, tracking, layers in (
        ("trusted", False, False),
        ("object_boundaries", False, False),
        ("full", True, False),
        ("full", True, True),
    )
)



def load_runner():
    spec = importlib.util.spec_from_file_location("run_benchmarks", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BenchmarkRunnerTests(unittest.TestCase):
    def run_main_harness(
        self,
        benchmark_output: str,
        threshold: bool = False,
    ) -> tuple[subprocess.CompletedProcess[str], pathlib.Path]:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        report = pathlib.Path(directory.name) / "report.md"
        harness = textwrap.dedent(
            """
            import importlib.util
            import pathlib
            import re
            import sys

            script = pathlib.Path(sys.argv[1])
            report = pathlib.Path(sys.argv[2])
            benchmark_output = sys.argv[3]
            threshold = sys.argv[4] == "threshold"
            spec = importlib.util.spec_from_file_location("run_benchmarks", script)
            runner = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(runner)
            runner.BENCHMARK_TARGETS = ("probe_bench",)
            runner.BENCHMARK_METHODS = {
                "probe_bench": ("synthetic=1", "ns/op"),
            }
            runner.REGRESSION_THRESHOLDS = {}
            if threshold:
                runner.REGRESSION_THRESHOLDS = {
                    "probe_bench": ((
                        "probe",
                        re.compile(r"value=(?P<value>[0-9]+(?:\\.[0-9]+)?)"),
                        10.0,
                        True,
                    ),),
                }

            def fake_executable(root, target):
                return pathlib.Path(target)

            def fake_run(command, cwd, env=None):
                command = tuple(str(part) for part in command)
                if command[0] == sys.executable:
                    return ""
                if command[0] == "c3c":
                    if "--version" in command:
                        return "C3 Compiler Version: 0.8.0"
                    return ""
                target = pathlib.Path(command[0]).name
                if target == "benchmark_info":
                    return "\\n".join((
                        'adapter: name="probe" type=cpu',
                        'driver: name="probe" id=0 version=0',
                        'validation: enabled=false',
                        'queues: graphics=0:0 compute=0:0 transfer=0:0',
                    ))
                if target == "probe_bench":
                    return benchmark_output
                raise AssertionError(command)

            runner.executable = fake_executable
            runner.run = fake_run
            sys.argv = [str(script), "--output", str(report)]
            raise SystemExit(runner.main())
            """
        )
        result = subprocess.run(
            (
                sys.executable,
                "-B",
                "-c",
                harness,
                str(SCRIPT),
                str(report),
                benchmark_output,
                "threshold" if threshold else "plain",
            ),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return result, report

    def test_allocation_benchmark_uses_explicit_cpu_write_allocations(self):
        runner = load_runner()
        self.assertEqual(
            runner.BENCHMARK_METHODS["allocation_bench"],
            ("4000/phase", "ns/allocation, ns/free"),
        )

    def test_suite_order_covers_stabilization_baselines(self):
        runner = load_runner()
        self.assertEqual(
            runner.BENCHMARK_TARGETS,
            (
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
            ),
        )

        self.assertEqual(
            runner.BENCHMARK_METHODS["upload_throughput_bench"],
            ("warmup=1; payload_iterations=4096:2048,262144:512,4194304:32; workers=1,2,4", "uploads/s"),
        )
        self.assertIn("repetitions=5", runner.BENCHMARK_METHODS["command_record_bench"][0])
        self.assertEqual(
            runner.BENCHMARK_PROJECTS["command_wrapper_bench"],
            "test/cpu",
        )
        self.assertIn(
            "lifecycle=0,1,16,256x5",
            runner.BENCHMARK_METHODS["command_path_baseline_bench"][0],
        )
        self.assertIn(
            ("command_wrapper_bench", "test/cpu"),
            runner.benchmark_build_targets(),
        )
        self.assertIn(
            ("command_record_bench", "test"),
            runner.benchmark_build_targets(),
        )
        self.assertIn(
            ("command_path_baseline_bench", "test"),
            runner.benchmark_build_targets(),
        )
        self.assertIn(
            ("command_reference_bench", "test"),
            runner.benchmark_build_targets(),
        )
        self.assertIn(
            "generated=64 prewarm+64/repetition",
            runner.BENCHMARK_METHODS["command_record_bench"][0],
        )
        self.assertEqual(
            runner.BENCHMARK_METHODS["lifecycle_bench"],
            (
                "submit=256x5; poll=100000x5; destroy=300x5",
                "ns/submit, ns/poll, ns/destroy",
            ),
        )
        self.assertEqual(
            runner.BENCHMARK_METHODS["submit_batch_bench"],
            (
                "batch_sizes=1,8,32,128,1024; exact authoritative work",
                "ns/submit, ns/list advisory; exact work and allocation outcomes",
            ),
        )
        self.assertEqual(
            runner.BENCHMARK_METHODS["pipeline_cache_bench"],
            (
                "raster=200; duplicate=200000; batch=64x2000; identity=1024,65536,1048576",
                "ns/create, ns/state; exact identity work",
            ),
        )
        self.assertEqual(
            runner.BENCHMARK_METHODS["completion_wait_scope_bench"],
            (
                "16 producer/indirect-consumer pairs per scope",
                "ms; advisory all/draw_arguments ratio",
            ),
        )
        self.assertEqual(
            runner.BENCHMARK_METHODS["descriptor_churn_bench"],
            (
                "320/worker; workers=1,2,4; sampler occupancy=8,64,1024,65536; ownership highwater=16,4096,65536",
                "ns/descriptor, ns/op, ns/destroy, ns/check; bounded sampler probes and ownership work",
            ),
        )
        self.assertEqual(runner.C3_BUILD_FLAGS, ("-O1",))

    def test_upload_workers_are_persistent_and_warmed_before_timing(self):
        source = UPLOAD_BENCHMARK.read_text(encoding="utf-8")
        worker_create = source.index("threads[worker].create(&record_upload")
        warmup = source.index("run_upload_iteration(", worker_create)
        clock_start = source.index("Clock start", warmup)
        self.assertLess(worker_create, warmup)
        self.assertLess(warmup, clock_start)
        self.assertNotIn(".create(&record_upload", source[clock_start:])
        self.assertIn(
            "UPLOAD_BENCH_ITERATIONS = { 2048, 512, 32 }",
            source,
        )
        self.assertIn("uint measured_iterations", source)

    def test_command_wrapper_measurement_accepts_exact_schema(self):
        runner = load_runner()
        runner.require_measurement(COMMAND_WRAPPER_OUTPUT, "command_wrapper_bench")

    def test_command_wrapper_measurement_rejects_schema_mutations(self):
        runner = load_runner()
        operation_line = COMMAND_WRAPPER_OUTPUT.splitlines()[0]
        mutations = (
            (COMMAND_WRAPPER_OUTPUT.replace(operation_line + "\n", "", 1), "record count"),
            (COMMAND_WRAPPER_OUTPUT.replace("operation=draw", "operation=dispatch", 1), "identity"),
            (COMMAND_WRAPPER_OUTPUT.replace("iterations=20000", "iterations=19999", 1), "iteration"),
            (COMMAND_WRAPPER_OUTPUT.replace("repetitions=5", "repetitions=4", 1), "repetition"),
            (COMMAND_WRAPPER_OUTPUT.replace("checksum=600000", "checksum=599999", 1), "checksum"),
            (COMMAND_WRAPPER_OUTPUT.replace("direct_min_ns=1.000", "direct_min_ns=3.000", 1), "direct timing"),
            (COMMAND_WRAPPER_OUTPUT.replace("public_max_ns=6.000", "public_max_ns=4.000", 1), "public timing"),
            (COMMAND_WRAPPER_OUTPUT.replace("ratio=2.500000", "ratio=2.000000", 1), "ratio"),
            (COMMAND_WRAPPER_OUTPUT.replace("observed=7800000", "observed=7799999"), "observation"),
            (COMMAND_WRAPPER_OUTPUT.replace("status=pass", "status=fail"), "check"),
        )
        for output, error in mutations:
            with self.subTest(error=error):
                with self.assertRaisesRegex(ValueError, error):
                    runner.require_measurement(output, "command_wrapper_bench")

    def test_command_path_vulkan_measurement_accepts_exact_schema(self):
        runner = load_runner()
        for output in COMMAND_PATH_VK_OUTPUTS:
            runner.require_measurement(
                output,
                "command_path_baseline_bench",
            )

    def test_command_path_vulkan_policy_matrix_requires_every_mode(self):
        runner = load_runner()
        runner.require_command_path_policy_matrix(COMMAND_PATH_VK_OUTPUTS)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            runner.require_command_path_policy_matrix(COMMAND_PATH_VK_OUTPUTS[:-1])
        swapped = (
            COMMAND_PATH_VK_OUTPUTS[1],
            COMMAND_PATH_VK_OUTPUTS[0],
            *COMMAND_PATH_VK_OUTPUTS[2:],
        )
        with self.assertRaisesRegex(ValueError, "mode mismatch"):
            runner.require_command_path_policy_matrix(swapped)

    def test_command_reference_measurement_accepts_exact_schema(self):
        runner = load_runner()
        runner.require_measurement(
            COMMAND_REFERENCE_OUTPUT,
            "command_reference_bench",
        )

    def test_command_reference_accepts_lower_private_work(self):
        runner = load_runner()
        output = COMMAND_REFERENCE_OUTPUT.replace(
            (
                "reference_index unique=1 lookups=1 probes=1 equality=0 "
                "publications=1 mutex=1 retains=1 releases=1 "
                "host_allocations=0 probes_per_reference=1.000 "
                "equality_per_reference=0.000"
            ),
            (
                "reference_index unique=1 lookups=1 probes=0 equality=0 "
                "publications=1 mutex=0 retains=1 releases=1 "
                "host_allocations=0 probes_per_reference=0.000 "
                "equality_per_reference=0.000"
            ),
        )
        runner.require_measurement(output, "command_reference_bench")

    def test_command_reference_measurement_rejects_structural_mutations(self):
        runner = load_runner()
        mutations = (
            ("host_allocations=0", "host_allocations=1", "allocation"),
            ("retains=4096", "retains=4095", "retains"),
            ("releases=2048", "releases=2047", "releases"),
            ("duplicates=100000", "duplicates=99999", "duplicates"),
            (
                "probes=2080 equality=2016",
                "probes=2081 equality=2016",
                "probe ratio",
            ),
            ("capacity_fault=true", "capacity_fault=false", "capacity"),
            ("status=pass", "status=fail", "structural check"),
        )
        for old, new, error in mutations:
            with self.subTest(error=error):
                output = COMMAND_REFERENCE_OUTPUT.replace(old, new, 1)
                with self.assertRaisesRegex(ValueError, error):
                    runner.require_measurement(
                        output,
                        "command_reference_bench",
                    )

    def test_command_path_vulkan_measurement_rejects_schema_mutations(self):
        runner = load_runner()
        first_operation = COMMAND_PATH_VK_OUTPUT.splitlines()[1]
        mutations = (
            (
                COMMAND_PATH_VK_OUTPUT.replace(first_operation + "\n", "", 1),
                "record count",
            ),
            (
                COMMAND_PATH_VK_OUTPUT.replace(
                    "validation=trusted",
                    "validation=unknown",
                    1,
                ),
                "policy",
            ),
            (
                COMMAND_PATH_VK_OUTPUT.replace(
                    "operation=draw",
                    "operation=dispatch",
                    1,
                ),
                "identity",
            ),
            (
                COMMAND_PATH_VK_OUTPUT.replace("iterations=20000", "iterations=1", 1),
                "iteration",
            ),
            (
                COMMAND_PATH_VK_OUTPUT.replace("repetitions=5", "repetitions=4", 1),
                "repetition",
            ),
            (
                COMMAND_PATH_VK_OUTPUT.replace(
                    "native_calls_per_iteration=2",
                    "native_calls_per_iteration=1",
                    1,
                ),
                "native work",
            ),
            (
                COMMAND_PATH_VK_OUTPUT.replace("direct_min_ns=1.000", "direct_min_ns=3.000", 1),
                "direct timing",
            ),
            (
                COMMAND_PATH_VK_OUTPUT.replace("ratio=2.500000", "ratio=2.000000", 1),
                "ratio",
            ),
            (
                COMMAND_PATH_VK_OUTPUT.replace("public_max_ns=6.000", "public_max_ns=4.000", 1),
                "public timing",
            ),
            (
                COMMAND_PATH_VK_OUTPUT.replace("loops=10", "loops=9", 1),
                "loop count",
            ),
            (
                COMMAND_PATH_VK_OUTPUT.replace(
                    "command_path_vk_work operation=draw",
                    "command_path_vk_work operation=dispatch",
                    1,
                ),
                "work order",
            ),
            (
                COMMAND_PATH_VK_OUTPUT.replace("host_allocations=0", "host_allocations=1", 1),
                "structural work",
            ),
            (
                COMMAND_PATH_VK_OUTPUT.replace(
                    "shader_module_creations=0",
                    "shader_module_creations=1",
                    1,
                ),
                "structural work",
            ),
            (
                COMMAND_PATH_VK_OUTPUT.replace(
                    "pipeline_creations=0",
                    "pipeline_creations=1",
                    1,
                ),
                "structural work",
            ),
            (
                COMMAND_PATH_VK_OUTPUT.replace(
                    "pipeline_creations=0 status=pass",
                    "pipeline_creations=0 status=fail",
                    1,
                ),
                "work record",
            ),
            (
                COMMAND_PATH_VK_OUTPUT.replace(
                    "operation=copy_buffer elements=64",
                    "operation=copy_buffer elements=63",
                    1,
                ),
                "element count",
            ),
            (
                COMMAND_PATH_VK_OUTPUT.replace(
                    "operation=copy_buffer elements=64 expected_checksum=2080",
                    "operation=dispatch elements=64 expected_checksum=2080",
                    1,
                ),
                "equivalence order",
            ),
            (
                COMMAND_PATH_VK_OUTPUT.replace("direct_checksum=4160", "direct_checksum=4159", 1),
                "checksum",
            ),
            (
                COMMAND_PATH_VK_OUTPUT.replace(
                    "expected_checksum=4160 direct_checksum=4160 public_checksum=4160",
                    "expected_checksum=0 direct_checksum=0 public_checksum=0",
                    1,
                ),
                "checksum",
            ),
            (
                COMMAND_PATH_VK_OUTPUT.replace(
                    "expected_checksum=4160 direct_checksum=4160 public_checksum=4160",
                    "expected_checksum=4159 direct_checksum=4159 public_checksum=4159",
                    1,
                ),
                "checksum",
            ),
            (
                COMMAND_PATH_VK_OUTPUT.replace(
                    "expected_checksum=2080 direct_checksum=2080 public_checksum=2080",
                    "expected_checksum=2079 direct_checksum=2079 public_checksum=2079",
                    1,
                ),
                "checksum",
            ),
            (
                COMMAND_PATH_VK_OUTPUT.replace("pairwise=true", "pairwise=false", 1),
                "equivalence record",
            ),
            (
                COMMAND_PATH_VK_OUTPUT.replace(
                    "commands=16 repetitions=5",
                    "commands=15 repetitions=5",
                    1,
                ),
                "command count",
            ),
            (
                COMMAND_PATH_VK_OUTPUT.replace("min_ns=100.000", "min_ns=120.000", 1),
                "lifecycle timing",
            ),
            (
                COMMAND_PATH_VK_OUTPUT.replace(
                    "paired_delta_median_ns=0.000 incremental_ns_per_command=0.000",
                    "paired_delta_median_ns=1.000 incremental_ns_per_command=1.000",
                    1,
                ),
                "zero lifecycle increment",
            ),
            (
                COMMAND_PATH_VK_OUTPUT.replace(
                    "paired_delta_median_ns=160.000 incremental_ns_per_command=10.000",
                    "paired_delta_median_ns=160.000 incremental_ns_per_command=9.000",
                    1,
                ),
                "increment calculation",
            ),
            (
                COMMAND_PATH_VK_OUTPUT.replace(
                    "commands=0 samples=5",
                    "commands=0 samples=4",
                    1,
                ),
                "sample count",
            ),
            (
                COMMAND_PATH_VK_OUTPUT.replace(
                    "command_path_vk_lifecycle_work commands=16",
                    "command_path_vk_lifecycle_work commands=15",
                    1,
                ),
                "work command count",
            ),
            (
                COMMAND_PATH_VK_OUTPUT.replace(
                    "commands=0 samples=5 host_allocations=0",
                    "commands=0 samples=5 host_allocations=1",
                    1,
                ),
                "unrelated work",
            ),
            (
                COMMAND_PATH_VK_OUTPUT.replace(
                    "command_buffer_resets=5",
                    "command_buffer_resets=4",
                    1,
                ),
                "reset count",
            ),
            (
                COMMAND_PATH_VK_OUTPUT.replace(
                    "command_path_vk_lifecycle_work commands=0 samples=5 host_allocations=0 command_pool_creations=0 command_buffer_allocations=0 command_buffer_frees=0 command_buffer_resets=5 image_view_creations=0 vma_allocations=0 registry_lock_acquisitions=0",
                    "command_path_vk_lifecycle_work commands=0 samples=5 host_allocations=0 command_pool_creations=0 command_buffer_allocations=0 command_buffer_frees=0 command_buffer_resets=5 image_view_creations=0 vma_allocations=0 registry_lock_acquisitions=1",
                    1,
                ),
                "registry-lock count",
            ),
            (
                COMMAND_PATH_VK_OUTPUT.replace(
                    "command_path_vk_lifecycle_work commands=0 samples=5 host_allocations=0 command_pool_creations=0 command_buffer_allocations=0 command_buffer_frees=0 command_buffer_resets=5 image_view_creations=0 vma_allocations=0 registry_lock_acquisitions=0 shader_module_creations=0",
                    "command_path_vk_lifecycle_work commands=0 samples=5 host_allocations=0 command_pool_creations=0 command_buffer_allocations=0 command_buffer_frees=0 command_buffer_resets=5 image_view_creations=0 vma_allocations=0 registry_lock_acquisitions=0 shader_module_creations=1",
                    1,
                ),
                "unrelated work",
            ),
            (
                COMMAND_PATH_VK_OUTPUT.replace(
                    "command_path_vk_lifecycle_work commands=0 samples=5 host_allocations=0 command_pool_creations=0 command_buffer_allocations=0 command_buffer_frees=0 command_buffer_resets=5 image_view_creations=0 vma_allocations=0 registry_lock_acquisitions=0 shader_module_creations=0 pipeline_creations=0",
                    "command_path_vk_lifecycle_work commands=0 samples=5 host_allocations=0 command_pool_creations=0 command_buffer_allocations=0 command_buffer_frees=0 command_buffer_resets=5 image_view_creations=0 vma_allocations=0 registry_lock_acquisitions=0 shader_module_creations=0 pipeline_creations=1",
                    1,
                ),
                "unrelated work",
            ),
        )
        for output, error in mutations:
            with self.subTest(error=error):
                with self.assertRaisesRegex(ValueError, error):
                    runner.require_measurement(
                        output,
                        "command_path_baseline_bench",
                    )

    def test_validation_mode_reaches_every_benchmark_device(self):
        source = BENCH_LIFETIME.read_text(encoding="utf-8")
        self.assertIn('env::tget_var("GPU_C3L_BENCH_VALIDATION")', source)
        self.assertIn(
            "runtime_desc.contract_validation = gpu::ContractValidation.FULL",
            source,
        )
        self.assertIn("&& !explicit_policy", source)
        self.assertIn("gpu::create_runtime(&runtime_desc)", source)

    def test_context_requires_reproducibility_fields(self):
        runner = load_runner()
        output = "\n".join(
            (
                'adapter: name="GPU" type=discrete',
                'driver: name="Driver" id=1 version=2',
                'validation: enabled=false',
                'queues: graphics=0:0 compute=1:0 transfer=2:0',
            )
        )
        runner.require_context_fields(output)

        with self.assertRaisesRegex(ValueError, "driver"):
            runner.require_context_fields(output.replace("driver:", "missing:"))

    def test_measurements_require_iteration_count_and_units(self):
        runner = load_runner()
        runner.require_measurement("phase: iterations=100 12.5 ns/op", "target")
        runner.require_measurement(
            "iterations=payload-specific warmup=1 units=uploads/s\n"
            "workers=4 payload_bytes=262144 iterations=512 uploads_per_sec=12345",
            "target",
        )

        with self.assertRaisesRegex(ValueError, "iteration count"):
            runner.require_measurement("phase: 12.5 ns/op", "target")
        with self.assertRaisesRegex(ValueError, "measured value"):
            runner.require_measurement("phase: iterations=100 value=12.5", "target")
        with self.assertRaisesRegex(ValueError, "uploads/s units"):
            runner.require_measurement(
                "iterations=payload-specific\n"
                "workers=4 payload_bytes=262144 iterations=512 uploads_per_sec=12345",
                "upload_throughput_bench",
            )
        with self.assertRaisesRegex(ValueError, "upload measurement fields"):
            runner.require_measurement(
                "iterations=payload-specific units=uploads/s\n"
                "workers=4 payload_bytes=262144 uploads_per_sec=12345",
                "upload_throughput_bench",
            )

    def test_allocation_measurement_accepts_exact_schema(self):
        runner = load_runner()
        runner.require_measurement(ALLOCATION_OUTPUT, "allocation_bench")

    def test_lifecycle_measurement_accepts_exact_schema(self):
        runner = load_runner()
        runner.require_measurement(LIFECYCLE_OUTPUT, "lifecycle_bench")

    def test_submit_batch_measurement_requires_exact_linear_work(self):
        runner = load_runner()
        runner.require_measurement(SUBMIT_BATCH_OUTPUT, "submit_batch_bench")
        for size in (1, 8, 32, 128, 1024):
            with self.subTest(size=size):
                output = SUBMIT_BATCH_OUTPUT.replace(
                    f"resolutions={size}",
                    f"resolutions={size + 1}",
                    1,
                )
                with self.assertRaisesRegex(ValueError, f"batch size {size}"):
                    runner.require_measurement(output, "submit_batch_bench")

    def test_submit_batch_measurement_rejects_forbidden_work_or_leaks(self):
        runner = load_runner()
        for field in (
            "epoch_reset_cells",
            "rollback_mutex",
            "host_allocations",
        ):
            with self.subTest(field=field):
                output = SUBMIT_BATCH_OUTPUT.replace(
                    f"{field}=0",
                    f"{field}=1",
                    1,
                )
                with self.assertRaisesRegex(ValueError, "forbidden work"):
                    runner.require_measurement(output, "submit_batch_bench")
        with self.assertRaisesRegex(ValueError, "live resources"):
            runner.require_measurement(
                SUBMIT_BATCH_OUTPUT.replace(
                    "submit batch leaks=0",
                    "submit batch leaks=1",
                ),
                "submit_batch_bench",
            )

    def test_submit_batch_measurement_rejects_structural_mutations(self):
        runner = load_runner()
        first_size_line = SUBMIT_BATCH_OUTPUT.splitlines()[1]
        mutations = (
            (
                SUBMIT_BATCH_OUTPUT.replace(first_size_line + "\n", "", 1),
                "record count",
            ),
            (
                SUBMIT_BATCH_OUTPUT.replace(
                    first_size_line,
                    first_size_line + "\n" + first_size_line,
                    1,
                ),
                "record count",
            ),
            (
                SUBMIT_BATCH_OUTPUT.replace(
                    "iterations=5",
                    "iterations=4",
                    1,
                ),
                "header",
            ),
            (
                SUBMIT_BATCH_OUTPUT.replace(
                    "native_submissions=1",
                    "native_submissions=2",
                    1,
                ),
                "native_submissions",
            ),
            (
                SUBMIT_BATCH_OUTPUT.replace(
                    "host_allocations=0",
                    "host_allocations=0 unknown=0",
                    1,
                ),
                "schema",
            ),
        )
        for output, error in mutations:
            with self.subTest(error=error):
                with self.assertRaisesRegex(ValueError, error):
                    runner.require_measurement(output, "submit_batch_bench")

    def test_descriptor_churn_requires_every_sampler_lookup_tier(self):
        runner = load_runner()
        runner.require_measurement(
            DESCRIPTOR_CHURN_OUTPUT,
            "descriptor_churn_bench",
        )
        missing = DESCRIPTOR_CHURN_OUTPUT.replace(
            "sampler lookup occupancy=64 bucket_count=128 probes=1 empty_bucket_miss_probes=0 elapsed_ns=20\n",
            "",
        )
        with self.assertRaisesRegex(ValueError, "tiers"):
            runner.require_measurement(missing, "descriptor_churn_bench")

    def test_descriptor_churn_rejects_malformed_sampler_evidence(self):
        runner = load_runner()
        malformed = DESCRIPTOR_CHURN_OUTPUT.replace(
            "probes=1 empty_bucket_miss_probes=0 elapsed_ns=20",
            "probe_count=1 empty_bucket_miss_probes=0 elapsed_ns=20",
            1,
        )
        with self.assertRaisesRegex(ValueError, "malformed"):
            runner.require_measurement(malformed, "descriptor_churn_bench")

    def test_descriptor_churn_rejects_wrong_occupancy_or_bucket_count(self):
        runner = load_runner()
        mutations = (
            (
                DESCRIPTOR_CHURN_OUTPUT.replace("occupancy=1024", "occupancy=1023"),
                "occupancy mismatch",
            ),
            (
                DESCRIPTOR_CHURN_OUTPUT.replace("bucket_count=2048", "bucket_count=1024"),
                "bucket count",
            ),
            (
                DESCRIPTOR_CHURN_OUTPUT.replace("bucket_count=2048", "bucket_count=3072"),
                "bucket count",
            ),
        )
        for output, error in mutations:
            with self.subTest(error=error):
                with self.assertRaisesRegex(ValueError, error):
                    runner.require_measurement(output, "descriptor_churn_bench")

    def test_descriptor_churn_accepts_zero_private_probes(self):
        runner = load_runner()
        output = DESCRIPTOR_CHURN_OUTPUT.replace(
            "probes=1",
            "probes=0",
            1,
        )
        runner.require_measurement(output, "descriptor_churn_bench")

    def test_descriptor_churn_rejects_excessive_probes(self):
        runner = load_runner()
        output = DESCRIPTOR_CHURN_OUTPUT.replace(
            "probes=1",
            "probes=9",
            1,
        )
        with self.assertRaisesRegex(ValueError, "probes"):
            runner.require_measurement(output, "descriptor_churn_bench")

    def test_descriptor_churn_rejects_nonzero_empty_bucket_miss_work(self):
        runner = load_runner()
        output = DESCRIPTOR_CHURN_OUTPUT.replace(
            "empty_bucket_miss_probes=0",
            "empty_bucket_miss_probes=1",
            1,
        )
        with self.assertRaisesRegex(ValueError, "malformed"):
            runner.require_measurement(output, "descriptor_churn_bench")

    def test_lifecycle_measurement_rejects_nonzero_invariants(self):
        runner = load_runner()
        for field in (
            "point_allocations",
            "destruction_queries",
            "destruction_completion_waits",
            "cached_poll_queries",
            "retirement_locks",
        ):
            with self.subTest(field=field):
                output = LIFECYCLE_OUTPUT.replace(
                    f"{field}=0",
                    f"{field}=1",
                )
                with self.assertRaisesRegex(ValueError, "exact schema"):
                    runner.require_measurement(output, "lifecycle_bench")
    def test_command_measurement_requires_zero_hot_path_invariants(self):
        runner = load_runner()
        runner.require_measurement(COMMAND_OUTPUT, "command_record_bench")
        for field in (
            "registry_locks",
            "recording_allocations",
            "draw_compilations",
            "preprocess_allocations",
        ):
            with self.subTest(field=field):
                output = COMMAND_OUTPUT.replace(f"{field}=0", f"{field}=1")
                with self.assertRaisesRegex(ValueError, "recording invariants"):
                    runner.require_measurement(output, "command_record_bench")

    def test_command_measurement_requires_direct_representation(self):
        runner = load_runner()
        output = COMMAND_OUTPUT.replace(
            "representation=direct",
            "representation=" + "bounded",
        )
        with self.assertRaisesRegex(ValueError, "token/storage record"):
            runner.require_measurement(output, "command_record_bench")

    def test_command_measurement_rejects_forbidden_resolution_work(self):
        runner = load_runner()
        for field in (
            "pipeline_binds",
            "descriptor_set_binds",
            "descriptor_buffer_binds",
            "descriptor_buffer_offsets",
            "device_registry",
            "retained_pins",
            "command_table",
            "pipeline_table",
            "pipeline_cache",
            "policy",
        ):
            with self.subTest(field=field):
                output = COMMAND_OUTPUT.replace(f"{field}=0", f"{field}=1")
                with self.assertRaisesRegex(ValueError, "forbidden work"):
                    runner.require_measurement(output, "command_record_bench")

    def test_command_measurement_requires_zero_private_encoder_work(self):
        runner = load_runner()
        for field in ("encoder_cells", "encoder_leases"):
            with self.subTest(field=field):
                output = COMMAND_OUTPUT.replace(
                    f"{field}=0",
                    f"{field}=1",
                )
                with self.assertRaisesRegex(ValueError, "nonzero"):
                    runner.require_measurement(output, "command_record_bench")

    def test_command_measurement_requires_exact_semantic_and_native_counts(self):
        runner = load_runner()
        mutations = (
            (
                "recording_commands=300320",
                "recording_commands=1",
                "recording command count",
            ),
            (
                "native_commands=400320",
                "native_commands=400321",
                "native command count",
            ),
            (
                "native_commands=400320",
                "native_commands=999999",
                "native command count",
            ),
        )
        for before, after, error in mutations:
            with self.subTest(after=after):
                output = COMMAND_OUTPUT.replace(before, after)
                with self.assertRaisesRegex(ValueError, error):
                    runner.require_measurement(output, "command_record_bench")

    def test_command_measurement_requires_versioned_deterministic_records(self):
        runner = load_runner()
        resolution_line = next(
            line
            for line in COMMAND_OUTPUT.splitlines()
            if line.startswith("resolution:")
        )
        mutations = (
            (
                COMMAND_OUTPUT.replace("expectation_version=3\n", ""),
                "expectation version",
            ),
            (
                COMMAND_OUTPUT.replace(
                    "expectation_version=3",
                    "expectation_version=4",
                ),
                "expectation version",
            ),
            (
                COMMAND_OUTPUT.replace(
                    "resolution: recording_commands=300320",
                    "resolution: command_count=300320",
                ),
                "resolution record",
            ),
            (
                COMMAND_OUTPUT.replace(
                    resolution_line,
                    resolution_line + "\n" + resolution_line,
                ),
                "resolution record",
            ),
        )
        for output, error in mutations:
            with self.subTest(error=error):
                with self.assertRaisesRegex(ValueError, error):
                    runner.require_measurement(output, "command_record_bench")

    def test_command_measurement_rejects_duplicated_work_records(self):
        runner = load_runner()
        for prefix, error in (
            ("invariants:", "recording invariants"),
            ("cold work:", "cold work record"),
            ("warm work:", "warm work"),
        ):
            with self.subTest(prefix=prefix):
                line = next(
                    line for line in COMMAND_OUTPUT.splitlines()
                    if line.startswith(prefix)
                )
                output = COMMAND_OUTPUT.replace(line, line + "\n" + line)
                with self.assertRaisesRegex(ValueError, error):
                    runner.require_measurement(output, "command_record_bench")

    def test_command_measurement_requires_zero_warm_work(self):
        runner = load_runner()
        for field in (
            "host_allocations",
            "command_buffer_allocations",
            "command_buffer_frees",
            "image_view_creations",
            "vma_allocations",
            "generated_scratch_misses",
        ):
            with self.subTest(field=field):
                warm_line = next(
                    line for line in COMMAND_OUTPUT.splitlines()
                    if line.startswith("warm work:")
                )
                output = COMMAND_OUTPUT.replace(
                    warm_line,
                    warm_line.replace(f"{field}=0", f"{field}=1"),
                )
                error = (
                    "host allocation"
                    if field == "host_allocations"
                    else "warm work"
                )
                with self.assertRaisesRegex(ValueError, error):
                    runner.require_measurement(output, "command_record_bench")

    def test_command_policy_matrix_requires_every_mode_in_order(self):
        runner = load_runner()
        runner.require_command_policy_matrix(COMMAND_POLICY_OUTPUTS)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            runner.require_command_policy_matrix(COMMAND_POLICY_OUTPUTS[:-1])
        wrong_order = (
            COMMAND_POLICY_OUTPUTS[1],
            COMMAND_POLICY_OUTPUTS[0],
            *COMMAND_POLICY_OUTPUTS[2:],
        )
        with self.assertRaisesRegex(ValueError, "mode mismatch"):
            runner.require_command_policy_matrix(wrong_order)

    def test_command_policy_matrix_rejects_malformed_or_mismatched_policy(self):
        runner = load_runner()
        malformed = list(COMMAND_POLICY_OUTPUTS)
        malformed[0] = malformed[0].replace(
            "validation policy=trusted",
            "validation contract=trusted",
        )
        with self.assertRaisesRegex(ValueError, "malformed"):
            runner.require_command_policy_matrix(malformed)

        tracking_mismatch = list(COMMAND_POLICY_OUTPUTS)
        tracking_mismatch[1] = tracking_mismatch[1].replace(
            "tracking=false",
            "tracking=true",
        )
        with self.assertRaisesRegex(ValueError, "mode mismatch"):
            runner.require_command_policy_matrix(tracking_mismatch)

        layer_mismatch = list(COMMAND_POLICY_OUTPUTS)
        layer_mismatch[2] = layer_mismatch[2].replace(
            "layers=false",
            "layers=true",
        )
        with self.assertRaisesRegex(ValueError, "mode mismatch"):
            runner.require_command_policy_matrix(layer_mismatch)

    def test_command_policy_matrix_rejects_wrong_work_relationships(self):
        runner = load_runner()
        trusted_work = list(COMMAND_POLICY_OUTPUTS)
        trusted_work[0] = trusted_work[0].replace(
            "semantic_checks=0",
            "semantic_checks=1",
        )
        with self.assertRaisesRegex(ValueError, "forbidden semantic work"):
            runner.require_command_policy_matrix(trusted_work)

        missing_full_work = list(COMMAND_POLICY_OUTPUTS)
        missing_full_work[2] = missing_full_work[2].replace(
            "semantic_checks=300330",
            "semantic_checks=0",
        )
        with self.assertRaisesRegex(ValueError, "missing semantic work"):
            runner.require_command_policy_matrix(missing_full_work)

        reference_mismatch = list(COMMAND_POLICY_OUTPUTS)
        reference_mismatch[2] = reference_mismatch[2].replace(
            "reference_allocations=0",
            "reference_allocations=1",
        )
        reference_mismatch[2] = reference_mismatch[2].replace(
            "warm work: host_allocations=0",
            "warm work: host_allocations=1",
        )
        with self.assertRaisesRegex(ValueError, "reference allocation"):
            runner.require_command_policy_matrix(reference_mismatch)

        policy_reselection = list(COMMAND_POLICY_OUTPUTS)
        policy_reselection[3] = policy_reselection[3].replace(
            "pipeline_cache=0 policy=0",
            "pipeline_cache=0 policy=1",
        )
        with self.assertRaisesRegex(ValueError, "warm path reported policy"):
            runner.require_command_policy_matrix(policy_reselection)

    def test_allocation_measurement_rejects_extra_fields(self):
        runner = load_runner()
        output = ALLOCATION_OUTPUT.replace(" size=64", " status=ok size=64", 1)
        with self.assertRaises(ValueError):
            runner.require_measurement(output, "allocation_bench")

    def test_allocation_measurement_requires_cpu_write_allocate_phase(self):
        runner = load_runner()
        output = ALLOCATION_OUTPUT.splitlines()[1]
        with self.assertRaisesRegex(ValueError, "cpu_write allocate"):
            runner.require_measurement(output, "allocation_bench")

    def test_allocation_measurement_requires_cpu_write_free_phase(self):
        runner = load_runner()
        output = ALLOCATION_OUTPUT.splitlines()[0]
        with self.assertRaisesRegex(ValueError, "cpu_write free"):
            runner.require_measurement(output, "allocation_bench")

    def test_allocation_measurement_requires_exact_cpu_write_iteration_counts(self):
        runner = load_runner()
        for phase in ("allocate", "free"):
            with self.subTest(phase=phase):
                output = ALLOCATION_OUTPUT.replace(
                    f"cpu_write {phase}: iterations=4000",
                    f"cpu_write {phase}: iterations=3999",
                )
                with self.assertRaisesRegex(ValueError, f"cpu_write {phase}"):
                    runner.require_measurement(output, "allocation_bench")

    def test_allocation_measurement_requires_size_and_alignment(self):
        runner = load_runner()
        for phase in ("allocate", "free"):
            for field in ("size=64 ", "align=16 "):
                with self.subTest(phase=phase, field=field):
                    line = next(
                        line
                        for line in ALLOCATION_OUTPUT.splitlines()
                        if line.startswith(f"cpu_write {phase}:")
                    )
                    output = ALLOCATION_OUTPUT.replace(line, line.replace(field, ""))
                    with self.assertRaisesRegex(ValueError, f"cpu_write {phase}"):
                        runner.require_measurement(output, "allocation_bench")

    def test_allocation_measurement_rejects_intervening_text(self):
        runner = load_runner()
        output = ALLOCATION_OUTPUT.replace("\n", "\nbenchmark error\n")
        with self.assertRaises(ValueError):
            runner.require_measurement(output, "allocation_bench")

    def test_allocation_measurement_rejects_malformed_decimal_separators(self):
        runner = load_runner()
        for value in ("23,75", "23..75", "1,023.75"):
            with self.subTest(value=value):
                output = ALLOCATION_OUTPUT.replace("23.75", value)
                with self.assertRaises(ValueError):
                    runner.require_measurement(output, "allocation_bench")

    def test_unpinned_regression_thresholds_are_advisory(self):
        runner = load_runner()
        cases = (
            ("allocation_bench", ALLOCATION_OUTPUT.replace("23.75", "5001.0")),
            ("command_record_bench", COMMAND_OUTPUT.replace("130.0", "2001.0")),
            ("lifecycle_bench", LIFECYCLE_OUTPUT.replace("8400.0", "100001.0")),
            (
                "pipeline_cache_bench",
                PIPELINE_OUTPUT.replace("42621.0", "500001.0"),
            ),
        )
        for target, output in cases:
            with self.subTest(target=target):
                advisories = runner.require_measurement(output, target)
                self.assertEqual(len(advisories), 1)
                self.assertIn("regression threshold", advisories[0])

    def test_pinned_regression_thresholds_are_required(self):
        runner = load_runner()
        output = ALLOCATION_OUTPUT.replace("23.75", "5001.0")
        with self.assertRaisesRegex(ValueError, "regression threshold"):
            runner.require_measurement(
                output,
                "allocation_bench",
                enforce_thresholds=True,
            )

    def test_validation_rejects_pinned_comparison(self):
        result = subprocess.run(
            (
                sys.executable,
                str(SCRIPT),
                "--validation",
                "--pinned-runner",
                "runner",
                "--pinned-driver",
                "driver",
                "--comparison-profile",
                "profile",
            ),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn(
            "--validation cannot be combined with pinned comparison fields",
            result.stderr,
        )

    def test_validation_skips_release_threshold_evaluation(self):
        runner = load_runner()
        output = ALLOCATION_OUTPUT.replace("23.75", "5001.0")
        advisories = runner.require_measurement(
            output,
            "allocation_bench",
            evaluate_thresholds=False,
        )
        self.assertEqual(advisories, [])

    def test_pipeline_cache_requires_one_native_raster_pipeline(self):
        runner = load_runner()
        with self.assertRaisesRegex(ValueError, "one native pipeline"):
            runner.require_measurement(
                PIPELINE_OUTPUT.replace("native=1", "native=2"),
                "pipeline_cache_bench",
            )

    def test_pipeline_cache_requires_raster_recording_evidence(self):
        runner = load_runner()
        with self.assertRaisesRegex(ValueError, "recording evidence"):
            runner.require_measurement(
                PIPELINE_OUTPUT.replace("99.0 ns/state", "missing"),
                "pipeline_cache_bench",
            )

    def test_pipeline_cache_requires_every_identity_size(self):
        runner = load_runner()
        missing = next(
            line
            for line in PIPELINE_OUTPUT.splitlines()
            if "size_bytes=65536" in line
        )
        with self.assertRaisesRegex(ValueError, "65536 bytes"):
            runner.require_measurement(
                PIPELINE_OUTPUT.replace(missing, ""),
                "pipeline_cache_bench",
            )

    def test_pipeline_cache_rejects_malformed_identity_evidence(self):
        runner = load_runner()
        output = PIPELINE_OUTPUT.replace(
            "owned_bytes_cloned=1048576",
            "owned_bytes_cloned=1_048_576",
        )
        with self.assertRaisesRegex(ValueError, "1048576 bytes"):
            runner.require_measurement(output, "pipeline_cache_bench")

    def test_pipeline_cache_rejects_post_intern_shader_work(self):
        runner = load_runner()
        output = PIPELINE_OUTPUT.replace(
            "lookup_shader_bytes_compared=0",
            "lookup_shader_bytes_compared=1",
            1,
        )
        with self.assertRaisesRegex(ValueError, "1024 bytes"):
            runner.require_measurement(output, "pipeline_cache_bench")

    def test_pipeline_cache_accepts_removed_private_key_probe(self):
        runner = load_runner()
        output = PIPELINE_OUTPUT.replace(
            "pipeline_key_probes=1",
            "pipeline_key_probes=0",
        )
        runner.require_measurement(output, "pipeline_cache_bench")

    def test_pipeline_cache_rejects_key_probe_over_budget(self):
        runner = load_runner()
        output = PIPELINE_OUTPUT.replace(
            "pipeline_key_probes=1",
            "pipeline_key_probes=2",
            1,
        )
        with self.assertRaisesRegex(ValueError, "1024 bytes"):
            runner.require_measurement(output, "pipeline_cache_bench")

    def test_upload_target_rejects_generic_measurement(self):
        runner = load_runner()
        with self.assertRaisesRegex(ValueError, "uploads/s units"):
            runner.require_measurement(
                "iterations=1 units=ns/op 1 ns/op",
                "upload_throughput_bench",
            )

    def test_generic_target_rejects_malformed_upload_measurement(self):
        runner = load_runner()
        with self.assertRaisesRegex(ValueError, "uploads/s units"):
            runner.require_measurement(
                "iterations=1 uploads_per_sec=123",
                "generic_benchmark",
            )

    def test_startup_header_alone_is_not_a_measurement(self):
        runner = load_runner()
        with self.assertRaisesRegex(ValueError, "measured value"):
            runner.require_measurement("iterations=300/worker units=ns/op", "target")
        with self.assertRaisesRegex(ValueError, "measured value"):
            runner.require_measurement(
                "iterations=2000/phase units=ns/record\nbenchmark crashed",
                "target",
            )

    def test_main_validates_raw_output_before_annotation(self):
        result, report = self.run_main_harness("12.5 ns/op")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("iteration count", result.stderr)
        self.assertFalse(report.exists())

    def test_main_surfaces_unpinned_advisories_on_stderr(self):
        result, report = self.run_main_harness(
            "iterations=1 value=20.0 20.0 ns/op",
            threshold=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        advisory = (
            "ADVISORY: probe_bench probe exceeded regression threshold: "
            "20 > 10"
        )
        self.assertIn(advisory, result.stderr)
        self.assertIn(advisory, report.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
