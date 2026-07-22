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
    def test_live_scan_excludes_negative_and_historical_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_directory:
            root = Path(temp_directory)
            live = root / "docs" / "api.md"
            historical = root / "docs" / "specs" / "old.md"
            live.parent.mkdir(parents=True)
            historical.parent.mkdir(parents=True)
            live.write_text("SamplerIndex is public.\n", encoding="utf-8")
            historical.write_text("publish_sampler\n", encoding="utf-8")
            with mock.patch.object(check_retired_api, "ROOT", root):
                self.assertEqual(
                    check_retired_api.find_live_retired_usages((root / "docs",)),
                    [],
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
