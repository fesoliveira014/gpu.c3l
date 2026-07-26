from __future__ import annotations

import contextlib
import io
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import check_retired_api


class RetiredApiCheckTests(unittest.TestCase):
    def test_live_scan_rejects_retired_pipeline_batches(self) -> None:
        source = (
            "gpu::create_compute_pipelines(&device, compute, outputs)!;\n"
            "gpu::create_graphics_pipelines(&device, graphics, outputs)!;\n"
        )
        with tempfile.TemporaryDirectory() as temp_directory:
            root = Path(temp_directory)
            path = root / "test" / "src" / "consumer.c3"
            path.parent.mkdir(parents=True)
            path.write_text(source, encoding="utf-8")
            with mock.patch.object(check_retired_api, "ROOT", root):
                failures = check_retired_api.find_live_retired_usages((path,))
        self.assertEqual(len(failures), 2)
        self.assertTrue(any("create_compute_pipelines" in item for item in failures))
        self.assertTrue(any("create_graphics_pipelines" in item for item in failures))

    def test_live_scan_rejects_retired_shader_preparation_surface(self) -> None:
        source = (
            "gpu::ShaderCode code = {};\n"
            "gpu::ShaderStage role;\n"
            "(void)gpu::prepare_shader_code(&desc);\n"
            "spvreflect::ShaderStage reflected = entry.stage();\n"
        )
        with tempfile.TemporaryDirectory() as temp_directory:
            root = Path(temp_directory)
            path = root / "test" / "src" / "consumer.c3"
            path.parent.mkdir(parents=True)
            path.write_text(source, encoding="utf-8")
            with mock.patch.object(check_retired_api, "ROOT", root):
                failures = check_retired_api.find_live_retired_usages((path,))
        self.assertEqual(len(failures), 3)
        for symbol in ("ShaderCode", "ShaderStage", "prepare_shader_code"):
            with self.subTest(symbol=symbol):
                self.assertTrue(any(symbol in item for item in failures))

    def test_live_scan_rejects_retired_readme_threading_claim(self) -> None:
        source = (
            "- Tiered threading: automatic per-worker command pools, "
            "thread-safe allocation.\n"
        )
        with tempfile.TemporaryDirectory() as temp_directory:
            root = Path(temp_directory)
            path = root / "README.md"
            path.write_text(source, encoding="utf-8")
            with mock.patch.object(check_retired_api, "ROOT", root):
                failures = check_retired_api.find_live_retired_usages((path,))
        self.assertEqual(
            failures,
            ["README.md:1: automatic per-worker command pools"],
        )

    def test_live_scan_rejects_retired_allocator_calls_in_c3_code(self) -> None:
        source = (
            "fn void first(gpu::Queue graphics) { gpu::begin_commands(graphics)!; }\n"
            "fn void first(gpu::CommandAllocator* allocator) {}\n"
            "fn void second(gpu::Queue transfer) {\n"
            "    gpu::reserve_generated_scratch(transfer, &desc)!;\n"
            "    gpu::release_generated_scratch(transfer, pipeline, kind)!;\n"
            "}\n"
            "// gpu::begin_commands(graphics)!;\n"
            "ZString ignored = \"gpu::begin_commands(graphics)!;\";\n"
        )
        with tempfile.TemporaryDirectory() as temp_directory:
            root = Path(temp_directory)
            path = root / "test" / "src" / "consumer.c3"
            path.parent.mkdir(parents=True)
            path.write_text(source, encoding="utf-8")
            with mock.patch.object(check_retired_api, "ROOT", root):
                failures = check_retired_api.find_live_retired_usages((path,))
        self.assertEqual(len(failures), 3)
        self.assertTrue(any("begin_commands(Queue)" in item for item in failures))
        self.assertTrue(any(
            "reserve_generated_scratch(Queue, ...)" in item
            for item in failures
        ))
        self.assertTrue(any(
            "release_generated_scratch(Queue, ...)" in item
            for item in failures
        ))

    def test_live_scan_checks_c3_fences_without_flagging_migration_prose(self) -> None:
        source = (
            "The retired `begin_commands(queue)` spelling is discussed here.\n\n"
            "```c3\n"
            "gpu::Queue worker = gpu::get_queue(&device, kind)!;\n"
            "gpu::begin_commands(worker)!;\n"
            "```\n\n"
            "```text\n"
            "gpu::begin_commands(worker)!;\n"
            "```\n"
        )
        with tempfile.TemporaryDirectory() as temp_directory:
            root = Path(temp_directory)
            path = root / "docs" / "migration.md"
            path.parent.mkdir(parents=True)
            path.write_text(source, encoding="utf-8")
            with mock.patch.object(check_retired_api, "ROOT", root):
                failures = check_retired_api.find_live_retired_usages((path,))
        self.assertEqual(
            failures,
            ["docs/migration.md:5: begin_commands(Queue)"],
        )

    def test_live_scan_includes_indexed_planning_docs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_directory:
            root = Path(temp_directory)
            live = root / "docs" / "api.md"
            planning = root / "docs" / "specs" / "current.md"
            live.parent.mkdir(parents=True)
            planning.parent.mkdir(parents=True)
            live.write_text("SamplerIndex is public.\n", encoding="utf-8")
            planning.write_text("publish_sampler\n", encoding="utf-8")
            with mock.patch.object(check_retired_api, "ROOT", root):
                self.assertEqual(
                    check_retired_api.find_live_retired_usages((root / "docs",)),
                    ["docs/specs/current.md:1: publish_sampler"],
                )

    def test_live_scan_reports_named_and_struct_field_usage(self) -> None:
        source = (
            "gpu::TextureDesc desc = { .dimension = value };\n"
            "gpu::publish_sampler(device, sampler)!;\n"
        )
        with tempfile.TemporaryDirectory() as temp_directory:
            root = Path(temp_directory)
            path = root / "test" / "src" / "positive.c3"
            path.parent.mkdir(parents=True)
            path.write_text(source, encoding="utf-8")
            with mock.patch.object(check_retired_api, "ROOT", root):
                failures = check_retired_api.find_live_retired_usages((path,))
            self.assertEqual(len(failures), 2)
            self.assertTrue(any("TextureDesc.dimension/depth" in item for item in failures))
            self.assertTrue(any("publish_sampler" in item for item in failures))

    def test_live_scan_rejects_retired_device_request_protocol(self) -> None:
        source = (
            "gpu::DeviceRequest request = gpu::strict_device_request();\n"
            "gpu::request_presentation(&request, &surface);\n"
            "gpu::request_queues(&request, queues);\n"
            "gpu::DeviceRequestSupport support = "
            "gpu::supports_device_request(&adapter, &request)!;\n"
            "bool enabled = caps.strict_enabled;\n"
            "bool supported = info.strict_supported;\n"
        )
        expected_markers = {
            "DeviceRequest",
            "DeviceRequestSupport",
            "strict_device_request",
            "request_presentation",
            "request_queues",
            "supports_device_request",
            "DeviceCaps.strict_enabled",
            "AdapterInfo.strict_supported",
        }
        with tempfile.TemporaryDirectory() as temp_directory:
            root = Path(temp_directory)
            path = root / "test" / "src" / "retired_device_request.c3"
            path.parent.mkdir(parents=True)
            path.write_text(source, encoding="utf-8")
            with mock.patch.object(check_retired_api, "ROOT", root):
                failures = check_retired_api.find_live_retired_usages((path,))
        self.assertEqual(
            {failure.rsplit(": ", 1)[1] for failure in failures},
            expected_markers,
        )

    def test_live_scan_allows_device_request_migration_references_only(self) -> None:
        migration = (
            "### Breaking migration\n\n"
            "`strict_device_request()` is replaced by `DeviceDesc`.\n\n"
            "Do not restore publish_sampler.\n\n"
            "## Next section\n"
        )
        stale_spec = "`DeviceRequest` remains the canonical request.\n"
        with tempfile.TemporaryDirectory() as temp_directory:
            root = Path(temp_directory)
            api = root / "docs" / "api.md"
            spec = root / "docs" / "specs" / "current.md"
            spec.parent.mkdir(parents=True)
            api.write_text(migration, encoding="utf-8")
            spec.write_text(stale_spec, encoding="utf-8")
            with mock.patch.object(check_retired_api, "ROOT", root):
                failures = check_retired_api.find_live_retired_usages(
                    (root / "docs",),
                )
        self.assertEqual(
            failures,
            [
                "docs/api.md:5: publish_sampler",
                "docs/specs/current.md:1: DeviceRequest",
            ],
        )

    def test_live_scan_reports_retired_runtime_desc_initializers_only(self) -> None:
        source = (
            "gpu::RuntimeDesc desc = { .enable_validation = true };\n"
            "fn gpu::RuntimeDesc legacy() => { .enable_validation = true };\n"
            "gpu::RuntimeDesc tracked = { .track_resource_lifetimes = true };\n"
            "vk::VkInstanceDesc backend = { .enable_validation = true };\n"
        )
        with tempfile.TemporaryDirectory() as temp_directory:
            root = Path(temp_directory)
            path = root / "test" / "src" / "runtime_desc.c3"
            path.parent.mkdir(parents=True)
            path.write_text(source, encoding="utf-8")
            with mock.patch.object(check_retired_api, "ROOT", root):
                failures = check_retired_api.find_live_retired_usages((path,))
            self.assertEqual(len(failures), 3)
            self.assertEqual(
                sum("RuntimeDesc.enable_validation" in item for item in failures),
                2,
            )
            self.assertEqual(
                sum(
                    "RuntimeDesc.track_resource_lifetimes" in item
                    for item in failures
                ),
                1,
            )

    def test_accepts_exact_retired_pipeline_signature_diagnostic(self) -> None:
        source = (
            "gpu::cmd_dispatch(commands, pipeline, root, {})!;\n"
        )
        pipeline_column = source.index("pipeline") + 1
        output = (
            f"(retired_cmd_dispatch_pipeline.c3:1:{pipeline_column}) Error: "
            "It is not possible to cast 'PipelineHandle' to 'GpuAddress'.\n"
        )
        with tempfile.TemporaryDirectory() as temp_directory:
            project = Path(temp_directory)
            (project / "retired_cmd_dispatch_pipeline.c3").write_text(
                source,
                encoding="utf-8",
            )
            with mock.patch.object(
                check_retired_api,
                "PROJECT",
                project,
            ):
                self.assertTrue(check_retired_api.has_expected_diagnostic(
                    "retired_cmd_dispatch_pipeline",
                    "pipeline",
                    output,
                ))

    def test_accepts_exact_retired_allocator_signature_diagnostic(self) -> None:
        source = "gpu::begin_commands(queue)!;\n"
        queue_column = source.index("queue") + 1
        output = (
            f"(retired_begin_commands_queue.c3:1:{queue_column}) Error: "
            "It is not possible to cast 'Queue' to 'CommandAllocator*'.\n"
        )
        with tempfile.TemporaryDirectory() as temp_directory:
            project = Path(temp_directory)
            (project / "retired_begin_commands_queue.c3").write_text(
                source,
                encoding="utf-8",
            )
            with mock.patch.object(
                check_retired_api,
                "PROJECT",
                project,
            ):
                self.assertTrue(check_retired_api.has_expected_diagnostic(
                    "retired_begin_commands_queue",
                    "queue",
                    output,
                ))

    def test_rejects_invalid_member_diagnostic_for_unrelated_member(self) -> None:
        source = (
            "fn gpu::DeviceDesc use() => { .persistent_arena_size = 1, "
            ".other_missing = 1 };\n"
        )
        unrelated_column = source.index(".other_missing") + 1
        output = (
            f"(persistent_arena_size.c3:1:{unrelated_column}) Error: "
            "This is not a valid member of 'DeviceDesc'.\n"
        )
        result = subprocess.CompletedProcess(
            args=[],
            returncode=1,
            stdout=output,
            stderr="",
        )
        with tempfile.TemporaryDirectory() as temp_directory:
            project = Path(temp_directory)
            (project / "persistent_arena_size.c3").write_text(
                source,
                encoding="utf-8",
            )
            with mock.patch.object(
                check_retired_api,
                "FIXTURES",
                {"persistent_arena_size": "persistent_arena_size"},
            ), mock.patch.object(
                check_retired_api,
                "PROJECT",
                project,
            ), mock.patch.object(
                check_retired_api.subprocess,
                "run",
                return_value=result,
            ), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(check_retired_api.main(), 1)

    def test_rejects_expected_error_after_unrelated_error(self) -> None:
        output = (
            "(test/retired_api/alloc_persistent_span.c3:3:1) Error: "
            "unrelated failure\n"
            "(test/retired_api/alloc_persistent_span.c3:4:5) Error: "
            "'gpu::alloc_persistent_span' could not be found, "
            "did you spell it right?\n"
        )
        result = subprocess.CompletedProcess(
            args=[],
            returncode=1,
            stdout=output,
            stderr="",
        )
        with mock.patch.object(
            check_retired_api,
            "FIXTURES",
            {"alloc_persistent_span": "alloc_persistent_span"},
        ), mock.patch.object(
            check_retired_api.subprocess,
            "run",
            return_value=result,
        ), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(check_retired_api.main(), 1)

    def test_rejects_enum_diagnostic_for_unrelated_source_token(self) -> None:
        source = (
            "fn gpu::DebugResourceKind use() => "
            "gpu::DebugResourceKind.FRAME + unrelated;\n"
        )
        unrelated_column = source.index("unrelated") + 1
        output = (
            f"(debug_frame.c3:1:{unrelated_column}) Error: "
            "'DebugResourceKind' has no enumeration value 'FRAME'.\n"
        )
        result = subprocess.CompletedProcess(
            args=[],
            returncode=1,
            stdout=output,
            stderr="",
        )
        with tempfile.TemporaryDirectory() as temp_directory:
            project = Path(temp_directory)
            (project / "debug_frame.c3").write_text(
                source,
                encoding="utf-8",
            )
            with mock.patch.object(
                check_retired_api,
                "FIXTURES",
                {"debug_frame": "FRAME"},
            ), mock.patch.object(
                check_retired_api,
                "PROJECT",
                project,
            ), mock.patch.object(
                check_retired_api.subprocess,
                "run",
                return_value=result,
            ), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(check_retired_api.main(), 1)

    def test_rejects_unrelated_error_that_echoes_retired_symbol(self) -> None:
        output = (
            " 3: fn void? use_retired(gpu::Device* device) {\n"
            " 4:     gpu::alloc_persistent_span(device, null)!;\n"
            "(test/retired_api/alloc_persistent_span.c3:4:5) Error: "
            "'gpu::alloc_persistent_span' could not be found, "
            "did you spell it right?\n"
            "(test/retired_api/alloc_persistent_span.c3:5:1) Error: "
            "unrelated failure\n"
        )
        result = subprocess.CompletedProcess(
            args=[],
            returncode=1,
            stdout=output,
            stderr="",
        )
        with mock.patch.object(
            check_retired_api,
            "FIXTURES",
            {"alloc_persistent_span": "alloc_persistent_span"},
        ), mock.patch.object(
            check_retired_api.subprocess,
            "run",
            return_value=result,
        ), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(check_retired_api.main(), 1)


if __name__ == "__main__":
    unittest.main()
