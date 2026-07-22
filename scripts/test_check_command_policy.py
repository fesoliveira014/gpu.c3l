from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts import check_command_policy


TABLE_SOURCE = """
module gpu::vk;

gpu::CommandOps trusted_command_ops @private = {
    .copy = &trusted_copy,
};

gpu::CommandOps trusted_tracking_command_ops @private = {
    .copy = &trusted_tracking_copy,
};

gpu::CommandOps checked_command_ops @private = {
    .copy = &checked_copy,
};

gpu::CommandOps checked_tracking_command_ops @private = {
    .copy = &checked_tracking_copy,
};

fn void trusted_copy() {
    lower_copy();
}

fn void trusted_tracking_copy() {
    lower_copy();
    retain_copy();
}

fn void checked_copy() {
    validate_copy();
}

fn void checked_tracking_copy() {
    validate_copy();
    retain_copy();
}

fn void lower_copy() {}
fn void validate_copy() {}

fn void retain_copy() {
    track_command_reference();
    rollback_command_references();
}

fn void track_command_reference() {
    ensure_command_reference_capacity();
}

fn void ensure_command_reference_capacity() {
    note_command_reference_allocation();
}

fn void note_command_reference_allocation() {}
fn void rollback_command_references() {}
"""


class CommandPolicyCheckTests(unittest.TestCase):
    def write_source(
        self,
        root: Path,
        relative: str,
        source: str,
    ) -> None:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")

    def test_current_sources_satisfy_contract(self) -> None:
        self.assertEqual(check_command_policy.check(), [])

    def test_accepts_policy_free_tables_with_tracking_isolated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_source(root, "gpu/vk/device.c3", TABLE_SOURCE)
            self.assertEqual(check_command_policy.check(root), [])

    def test_rejects_direct_policy_read(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = TABLE_SOURCE.replace(
                "fn void trusted_copy() {\n    lower_copy();\n}",
                (
                    "fn void trusted_copy() {\n"
                    "    if (state.validation_policy.contract) lower_copy();\n"
                    "}"
                ),
            )
            self.write_source(root, "gpu/vk/device.c3", source)
            errors = check_command_policy.check(root)
            self.assertTrue(any("reads validation_policy" in error for error in errors))

    def test_rejects_renamed_tracking_helper(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = TABLE_SOURCE.replace(
                "    lower_copy();\n}\n\nfn void trusted_tracking_copy",
                (
                    "    lower_copy();\n"
                    "    quiet_retention();\n"
                    "}\n\n"
                    "fn void quiet_retention() {\n"
                    "    track_command_reference();\n"
                    "}\n\n"
                    "fn void trusted_tracking_copy"
                ),
            )
            self.write_source(root, "gpu/vk/device.c3", source)
            errors = check_command_policy.check(root)
            self.assertTrue(any(
                "trusted_command_ops reaches tracking work" in error
                for error in errors
            ))

    def test_rejects_cross_file_policy_read(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = TABLE_SOURCE.replace(
                "    lower_copy();\n}\n\nfn void trusted_tracking_copy",
                "    cross_file_helper();\n}\n\nfn void trusted_tracking_copy",
            )
            self.write_source(root, "gpu/vk/device.c3", source)
            self.write_source(
                root,
                "gpu/vk/nested/helper.c3",
                "fn void cross_file_helper() { (void)state.vulkan_layers; }\n",
            )
            errors = check_command_policy.check(root)
            self.assertTrue(any("reads vulkan_layers" in error for error in errors))

    def test_rejects_policy_read_in_any_overload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = TABLE_SOURCE.replace(
                "    lower_copy();\n}\n\nfn void trusted_tracking_copy",
                "    overloaded_helper();\n}\n\nfn void trusted_tracking_copy",
            )
            source += """
fn void overloaded_helper() {}
fn void overloaded_helper(uint value) {
    (void)value;
    (void)state.debug_names;
}
"""
            self.write_source(root, "gpu/vk/device.c3", source)
            errors = check_command_policy.check(root)
            self.assertTrue(any("reads debug_names" in error for error in errors))

    def test_rejects_superseded_command_function(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = TABLE_SOURCE + "\nfn void vk_cmd_copy_buffer() {}\n"
            self.write_source(root, "gpu/vk/device.c3", source)
            errors = check_command_policy.check(root)
            self.assertTrue(any(
                "superseded command functions remain declared: vk_cmd_copy_buffer"
                in error
                for error in errors
            ))


if __name__ == "__main__":
    unittest.main()
