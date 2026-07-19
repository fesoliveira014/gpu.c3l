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
