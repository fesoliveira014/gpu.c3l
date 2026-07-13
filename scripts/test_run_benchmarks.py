#!/usr/bin/env python3

import importlib.util
import pathlib
import unittest


SCRIPT = pathlib.Path(__file__).with_name("run_benchmarks.py")


def load_runner():
    spec = importlib.util.spec_from_file_location("run_benchmarks", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BenchmarkRunnerTests(unittest.TestCase):
    def test_suite_order_covers_stabilization_baselines(self):
        runner = load_runner()
        self.assertEqual(
            runner.BENCHMARK_TARGETS,
            (
                "arena_allocation_bench",
                "resource_create_bench",
                "descriptor_churn_bench",
                "upload_throughput_bench",
                "context_reset_bench",
                "command_record_bench",
                "frame_signal_bench",
                "pipeline_cache_bench",
                "async_overlap_bench",
            ),
        )

        self.assertIn("repetitions=5", runner.BENCHMARK_METHODS["command_record_bench"][0])
        self.assertEqual(runner.C3_BUILD_FLAGS, ("-O1",))

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

        with self.assertRaisesRegex(ValueError, "iteration count"):
            runner.require_measurement("phase: 12.5 ns/op", "target")
        with self.assertRaisesRegex(ValueError, "unit"):
            runner.require_measurement("phase: iterations=100 value=12.5", "target")


if __name__ == "__main__":
    unittest.main()
