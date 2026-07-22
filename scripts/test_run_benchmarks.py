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
        "barrier: iterations=20000 repetitions=5 median=130.0 ns/record",
        "hazard barrier: iterations=20000 repetitions=5 median=135.0 ns/record",
        "indirect dispatch: iterations=20000 repetitions=5 median=180.0 ns/record",
        "generated dispatch: iterations=64 repetitions=5 median=900.0 ns/record",
        (
            "invariants: registry_locks=0 recording_allocations=0 "
            "draw_compilations=0 preprocess_allocations=0"
        ),
        (
            "resolution: recording_commands=305000 native_commands=405000 "
            "device_registry=0 "
            "retained_pins=0 lifecycle_vtable=0 command_table=0 "
            "pipeline_table=0 pipeline_cache=0 policy=0"
        ),
        (
            "cold work: host_allocations=5 command_buffer_allocations=1 "
            "command_buffer_frees=0 command_buffer_resets=3 "
            "image_view_creations=0 vma_allocations=64 "
            "generated_scratch_misses=0"
        ),
        (
            "warm work: host_allocations=0 command_buffer_allocations=0 "
            "command_buffer_frees=0 command_buffer_resets=20 "
            "image_view_creations=0 vma_allocations=0 "
            "generated_scratch_misses=0"
        ),
        "generated preprocess: reuse_events=320",
    )
)
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
    ("iterations=batch_sizes=1,8,32,128,1024 units=ns/submit",) + tuple(
        f"submit batch size={size}: 125.0 ns/submit "
        f"token_visits={size} epoch_reset_cells=0"
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
                "command_record_bench",
                "lifecycle_bench",
                "submit_batch_bench",
                "pipeline_cache_bench",
                "async_overlap_bench",
            ),
        )

        self.assertEqual(
            runner.BENCHMARK_METHODS["upload_throughput_bench"],
            ("warmup=1; payload_iterations=4096:2048,262144:512,4194304:32; workers=1,2,4", "uploads/s"),
        )
        self.assertIn("repetitions=5", runner.BENCHMARK_METHODS["command_record_bench"][0])
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
                "batch_sizes=1,8,32,128,1024; exact token visits",
                "ns/submit; exact work units",
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
            runner.BENCHMARK_METHODS["descriptor_churn_bench"],
            (
                "320/worker; workers=1,2,4; sampler occupancy=8,64,1024,65536; ownership highwater=16,4096,65536",
                "ns/descriptor, ns/op, ns/destroy, ns/check; exact sampler probes and ownership work",
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

    def test_validation_mode_reaches_every_benchmark_device(self):
        source = BENCH_LIFETIME.read_text(encoding="utf-8")
        self.assertIn('env::tget_var("GPU_C3L_BENCH_VALIDATION")', source)
        self.assertIn(
            "runtime_desc.enable_validation = bench_validation_enabled()",
            source,
        )
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
                    f"token_visits={size}",
                    f"token_visits={size + 1}",
                    1,
                )
                with self.assertRaisesRegex(ValueError, f"batch size {size}"):
                    runner.require_measurement(output, "submit_batch_bench")

    def test_submit_batch_measurement_rejects_rollover_or_leaks(self):
        runner = load_runner()
        with self.assertRaisesRegex(ValueError, "batch size 1"):
            runner.require_measurement(
                SUBMIT_BATCH_OUTPUT.replace(
                    "epoch_reset_cells=0",
                    "epoch_reset_cells=1",
                    1,
                ),
                "submit_batch_bench",
            )
        with self.assertRaisesRegex(ValueError, "live resources"):
            runner.require_measurement(
                SUBMIT_BATCH_OUTPUT.replace(
                    "submit batch leaks=0",
                    "submit batch leaks=1",
                ),
                "submit_batch_bench",
            )

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

    def test_descriptor_churn_rejects_zero_or_excessive_probes(self):
        runner = load_runner()
        for probes in (0, 9):
            with self.subTest(probes=probes):
                output = DESCRIPTOR_CHURN_OUTPUT.replace(
                    "probes=1",
                    f"probes={probes}",
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

    def test_command_measurement_requires_zero_resolution_evidence(self):
        runner = load_runner()
        for field in (
            "device_registry",
            "retained_pins",
            "lifecycle_vtable",
            "command_table",
            "pipeline_table",
            "pipeline_cache",
            "policy",
        ):
            with self.subTest(field=field):
                output = COMMAND_OUTPUT.replace(f"{field}=0", f"{field}=1")
                with self.assertRaisesRegex(ValueError, "resolution evidence"):
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
                with self.assertRaisesRegex(ValueError, "warm recording work"):
                    runner.require_measurement(output, "command_record_bench")

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
