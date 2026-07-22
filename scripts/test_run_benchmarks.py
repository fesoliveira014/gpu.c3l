#!/usr/bin/env python3

import importlib.util
import pathlib
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
            "invariants: point_allocations=0 destruction_waits=0 "
            "deferred_releases=0 cached_poll_queries=0 retirement_locks=0"
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
    )
)



def load_runner():
    spec = importlib.util.spec_from_file_location("run_benchmarks", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BenchmarkRunnerTests(unittest.TestCase):
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
            runner.BENCHMARK_METHODS["pipeline_cache_bench"],
            (
                "raster=200; duplicate=200000; batch=64x2000",
                "ns/create, ns/state",
            ),
        )
        self.assertEqual(
            runner.BENCHMARK_METHODS["descriptor_churn_bench"],
            (
                "320/worker; workers=1,2,4; ownership highwater=16,4096,65536",
                "ns/descriptor, ns/op, ns/destroy, ns/check; exact work units",
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

    def test_lifecycle_measurement_rejects_nonzero_invariants(self):
        runner = load_runner()
        for field in (
            "point_allocations",
            "destruction_waits",
            "deferred_releases",
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

    def test_regression_thresholds_reject_order_of_magnitude_slowdowns(self):
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
                with self.assertRaisesRegex(ValueError, "regression threshold"):
                    runner.require_measurement(output, target)

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
        source = SCRIPT.read_text(encoding="utf-8")
        main = source[source.index("def main()"):]
        validation_call = (
            "require_measurement(output, target, "
            "enforce_thresholds=not args.validation)"
        )
        self.assertIn(validation_call, main)
        self.assertIn('"--validation"', main)
        self.assertIn('"1" if args.validation else "0"', main)
        validation = main.index(validation_call)
        annotation = main.index('annotated = f"iterations={iterations}')
        self.assertLess(validation, annotation)


if __name__ == "__main__":
    unittest.main()
