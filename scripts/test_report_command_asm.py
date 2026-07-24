from __future__ import annotations

from contextlib import redirect_stdout
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import report_command_asm


SCRIPT = Path(report_command_asm.__file__)
ASM = """
.type gpu.cmd_dispatch,@function
gpu.cmd_dispatch:
    movq (%rax), %rcx
    callq *%rcx
    callq vk_cmd_dispatch
    jne .Lretry
    mystery %rax
    retq
.size gpu.cmd_dispatch, .-gpu.cmd_dispatch

.type gpu.cmd_draw,@function
gpu.cmd_draw:
    movq %rax, (%rcx)
    retq
.size gpu.cmd_draw, .-gpu.cmd_draw

.type gpu.internal.vk.trusted_dispatch,@function
gpu.internal.vk.trusted_dispatch:
    callq vkCmdPushConstants
    callq vkCmdDispatch
    retq
.size gpu.internal.vk.trusted_dispatch, .-gpu.internal.vk.trusted_dispatch
"""


class CommandAssemblyReportTests(unittest.TestCase):
    def collect(self, source: str = ASM):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        (root / "commands.s").write_text(source, encoding="utf-8")
        return report_command_asm.collect(root)

    def test_reports_broad_observations(self) -> None:
        observations = self.collect()
        dispatch = observations["dispatch"]
        self.assertIsNotNone(dispatch)
        assert dispatch is not None
        self.assertEqual(dispatch.calls, 2)
        self.assertEqual(dispatch.indirect_calls, 1)
        self.assertEqual(dispatch.branches, 1)
        self.assertEqual(dispatch.loads, 1)
        self.assertEqual(dispatch.native_dispatch, 2)
        self.assertEqual(dispatch.unknown, 1)
        draw = observations["draw"]
        self.assertIsNotNone(draw)
        assert draw is not None
        self.assertEqual(draw.stores, 1)

    def test_direct_symbols_with_register_prefixes_are_not_indirect(self) -> None:
        direct_calls = """
.type gpu.cmd_dispatch,@function
gpu.cmd_dispatch:
    callq record_command
    callq retain_resource
    callq release_resource
    callq external_helper
    retq
.size gpu.cmd_dispatch, .-gpu.cmd_dispatch
"""
        observations = self.collect(direct_calls)
        dispatch = observations["dispatch"]
        self.assertIsNotNone(dispatch)
        assert dispatch is not None
        self.assertEqual(dispatch.calls, 4)
        self.assertEqual(dispatch.indirect_calls, 0)

    def test_register_call_operands_are_indirect(self) -> None:
        for operands in ("*%rcx", "%r8", "rax", "x12", "w3"):
            with self.subTest(operands=operands):
                self.assertTrue(
                    report_command_asm.is_indirect_call_operand(operands)
                )
        for symbol in (
            "record_command",
            "retain_resource",
            "release_resource",
            "external_helper",
            "write_command",
        ):
            with self.subTest(symbol=symbol):
                self.assertFalse(
                    report_command_asm.is_indirect_call_operand(symbol)
                )

    def test_compiler_version_normalizes_build_suffix(self) -> None:
        self.assertEqual(
            report_command_asm.parse_compiler_version(
                "C3 Compiler Version: 0.8.0_2\n"
            ),
            "0.8.0",
        )
        self.assertEqual(
            report_command_asm.parse_compiler_version(
                "C3 Compiler Version: 0.8.0\n"
            ),
            "0.8.0",
        )

    def test_cli_has_no_direct_token_switch(self) -> None:
        result = subprocess.run(
            (sys.executable, "-B", str(SCRIPT), "--help"),
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertNotIn("--direct-command-" + "tokens", result.stdout)

    def test_emit_builds_only_the_bounded_profile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            asm_dir = root / "asm"
            with patch.object(report_command_asm.subprocess, "run") as run:
                report_command_asm.emit_assembly(
                    root,
                    asm_dir,
                    "linux-x64",
                )
        command = run.call_args.args[0]
        self.assertEqual(
            command,
            (
                "c3c",
                "build",
                "command_path_baseline_bench",
                "--path",
                "test",
                "-O1",
                "--emit-asm",
                "--asm-out",
                str(asm_dir),
                "--target",
                "linux-x64",
            ),
        )
        self.assertNotIn("DIRECT_COMMAND_" + "TOKENS", command)

    def test_advisory_report_identifies_bounded_representation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            argv = [
                str(SCRIPT),
                "--asm-dir",
                directory,
            ]
            output = io.StringIO()
            with patch.object(sys, "argv", argv), redirect_stdout(output):
                self.assertEqual(report_command_asm.main(), 0)
        self.assertIn(
            "asm_expectation version=1 mode=advisory "
            "representation=bounded",
            output.getvalue(),
        )

    def test_unpinned_missing_symbols_and_unknown_forms_are_advisory(self) -> None:
        observations = self.collect()
        self.assertIsNone(observations["barrier"])
        dispatch = observations["dispatch"]
        assert dispatch is not None
        self.assertGreater(dispatch.unknown, 0)

    def test_pinned_limits_use_broad_counts(self) -> None:
        observations = self.collect()
        limits = {
            "dispatch": {
                "calls": 2,
                "indirect_calls": 1,
                "branches": 1,
            },
            "draw": {"stores": 1},
        }
        failures = report_command_asm.validate_limits(observations, limits)
        self.assertIn("barrier: representative symbol is missing", failures)
        self.assertIn("viewport: representative symbol is missing", failures)
        self.assertIn("copy_buffer: representative symbol is missing", failures)
        complete = {
            operation: observation
            for operation, observation in observations.items()
            if observation is not None
        }
        for operation in ("barrier", "viewport", "copy_buffer"):
            complete[operation] = observations["draw"]
        self.assertEqual(
            report_command_asm.validate_limits(complete, limits),
            [],
        )
        too_low = {"dispatch": {"calls": 1}}
        self.assertIn(
            "dispatch: calls 2 exceeds 1",
            report_command_asm.validate_limits(complete, too_low),
        )
        native_missing = {
            "dispatch": {"native_dispatch": {"minimum": 3, "maximum": 3}},
        }
        self.assertIn(
            "dispatch: native_dispatch 2 is below 3",
            report_command_asm.validate_limits(complete, native_missing),
        )

    def test_pinned_profile_identity_is_versioned_and_exact(self) -> None:
        profile = {
            "expectation_version": report_command_asm.EXPECTATION_VERSION,
            "identity": {
                "compiler": "0.8.0",
                "target": "linux-x64",
                "comparison_profile": "command-fastpath-o1-v1",
                "optimization": "O1",
            },
            "limits": {
                operation: {
                    field: {"maximum": 100}
                    for field in (
                        "instructions",
                        "calls",
                        "indirect_calls",
                        "atomics",
                        "branches",
                        "loads",
                        "stores",
                        "native_dispatch",
                    )
                }
                for operation in report_command_asm.OPERATIONS
            },
        }
        self.assertEqual(
            report_command_asm.validate_profile(
                profile,
                "0.8.0",
                "linux-x64",
                "command-fastpath-o1-v1",
            ),
            [],
        )
        profile["identity"]["target"] = "windows-x64"
        self.assertIn(
            "profile identity target 'windows-x64' != 'linux-x64'",
            report_command_asm.validate_profile(
                profile,
                "0.8.0",
                "linux-x64",
                "command-fastpath-o1-v1",
            ),
        )

    def test_extra_atomic_instruction_fails_pinned_limit(self) -> None:
        mutated = ASM.replace(
            "    callq *%rcx\n",
            "    lock cmpxchgq %rax, (%rcx)\n    callq *%rcx\n",
        )
        observations = self.collect(mutated)
        complete = {
            operation: observation
            for operation, observation in observations.items()
            if observation is not None
        }
        for operation in ("barrier", "viewport", "copy_buffer"):
            complete[operation] = observations["draw"]
        self.assertIn(
            "dispatch: atomics 1 exceeds 0",
            report_command_asm.validate_limits(
                complete,
                {"dispatch": {"atomics": {"maximum": 0}}},
            ),
        )


if __name__ == "__main__":
    unittest.main()
