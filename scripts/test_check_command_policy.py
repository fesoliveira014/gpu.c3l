from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts import check_command_policy


TABLE_SOURCE = """
module gpu::vk;

const gpu::CommandOps TRUSTED_COMMAND_OPS @private = {
    .copy = &trusted_copy,
};

const gpu::CommandOps TRUSTED_TRACKING_COMMAND_OPS @private = {
    .copy = &trusted_tracking_copy,
};

const gpu::CommandOps CHECKED_COMMAND_OPS @private = {
    .copy = &checked_copy,
};

const gpu::CommandOps CHECKED_TRACKING_COMMAND_OPS @private = {
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

fn void vk_create_command_allocator() {}
fn void vk_destroy_command_allocator() {}
fn void vk_reserve_generated_scratch() {}
fn void vk_release_generated_scratch() {}
fn void vk_begin_commands() {}
fn void vk_end_commands() {}
fn void vk_discard_commands() {}
fn void vk_submit() {}
fn void retire_observed_completion_and_drain_with_query() {}
fn void drain_completed_submitted_commands() {}
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
                "TRUSTED_COMMAND_OPS reaches tracking work" in error
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

    def test_rejects_uninventoried_command_table(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = TABLE_SOURCE + """
const gpu::CommandOps EXPERIMENTAL_COMMAND_OPS @private = {
    .copy = &trusted_copy,
};
"""
            self.write_source(root, "gpu/vk/device.c3", source)
            self.assertEqual(
                check_command_policy.check(root),
                ["unexpected command policy table: EXPERIMENTAL_COMMAND_OPS"],
            )

    def test_rejects_direct_temporary_pool_use(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = TABLE_SOURCE.replace(
                "fn void vk_begin_commands() {}",
                "fn void vk_begin_commands() { @pool() {} }",
            )
            self.write_source(root, "gpu/vk/device.c3", source)
            errors = check_command_policy.check(root)
            self.assertTrue(any(
                "command path reaches temporary pool" in error
                for error in errors
            ))

    def test_ignores_ambient_markers_in_comments_and_strings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = TABLE_SOURCE.replace(
                "fn void vk_begin_commands() {}",
                (
                    "fn void vk_begin_commands() {\n"
                    "    io::print(\"@pool() tlocal RecordingContextTable\");\n"
                    "    // mem::talloc_array must remain retired.\n"
                    "}"
                ),
            )
            self.write_source(root, "gpu/vk/device.c3", source)
            self.assertEqual(check_command_policy.check(root), [])

    def test_rejects_reachable_thread_local_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = TABLE_SOURCE.replace(
                "fn void vk_begin_commands() {}",
                (
                    "fn void vk_begin_commands() { hidden_tls(); }\n"
                    "fn void hidden_tls() { tlocal int command_cache; }"
                ),
            )
            self.write_source(root, "gpu/vk/device.c3", source)
            errors = check_command_policy.check(root)
            self.assertTrue(any(
                "command path reaches thread-local state" in error
                for error in errors
            ))

    def test_rejects_renamed_warm_allocation_helper(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = TABLE_SOURCE.replace(
                "fn void vk_begin_commands() {}",
                (
                    "fn void vk_begin_commands() { hidden_growth(); }\n"
                    "fn void hidden_growth() { mem::new_array(uint, 1); }"
                ),
            )
            self.write_source(root, "gpu/vk/device.c3", source)
            errors = check_command_policy.check(root)
            self.assertTrue(any(
                "warm command path reaches host allocation" in error
                for error in errors
            ))

    def test_rejects_capacity_sized_submit_stack_storage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = TABLE_SOURCE.replace(
                "fn void vk_submit() {}",
                (
                    "fn void vk_submit() { submit_with_scratch(); }\n"
                    "fn void submit_with_scratch() {\n"
                    "    ulong[MAX_SUBMIT_COMMAND_LISTS] records;\n"
                    "}"
                ),
            )
            self.write_source(root, "gpu/vk/device.c3", source)
            errors = check_command_policy.check(root)
            self.assertTrue(any(
                "warm command path reaches capacity-sized stack storage" in error
                for error in errors
            ))

    def test_rejects_cross_file_fallible_host_allocation_helper(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = TABLE_SOURCE.replace(
                "fn void vk_begin_commands() {}",
                "fn void vk_begin_commands() { cross_file_growth(); }",
            )
            self.write_source(root, "gpu/vk/device.c3", source)
            self.write_source(
                root,
                "gpu/vk/nested/helper.c3",
                "fn void cross_file_growth() { alloc::new_array_try(a, uint, 1); }\n",
            )
            errors = check_command_policy.check(root)
            self.assertTrue(any(
                "warm command path reaches host allocation" in error
                for error in errors
            ))

    def test_rejects_reachable_vma_allocator_method(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = TABLE_SOURCE.replace(
                "fn void vk_begin_commands() {}",
                (
                    "fn void vk_begin_commands() { hidden_vma_growth(); }\n"
                    "fn void hidden_vma_growth() {\n"
                    "    state.allocator.create_buffer_with_alignment();\n"
                    "}"
                ),
            )
            self.write_source(root, "gpu/vk/device.c3", source)
            errors = check_command_policy.check(root)
            self.assertTrue(any(
                "warm command path reaches VMA allocation" in error
                for error in errors
            ))

    def test_rejects_reachable_cold_native_allocator_helper(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = TABLE_SOURCE.replace(
                "fn void vk_begin_commands() {}",
                "fn void vk_begin_commands() { allocate_command_buffers_real(); }",
            )
            source += "\nfn void allocate_command_buffers_real() { ops.allocate(); }\n"
            self.write_source(root, "gpu/vk/device.c3", source)
            errors = check_command_policy.check(root)
            self.assertTrue(any(
                "warm command path reaches cold allocation helper" in error
                for error in errors
            ))

    def test_rejects_cross_file_ambient_context_helper(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = TABLE_SOURCE.replace(
                "fn void vk_submit() {}",
                "fn void vk_submit() { cross_file_context(); }",
            )
            self.write_source(root, "gpu/vk/device.c3", source)
            self.write_source(
                root,
                "gpu/vk/nested/helper.c3",
                (
                    "fn void cross_file_context() {\n"
                    "    RecordingContextTable contexts;\n"
                    "}\n"
                ),
            )
            errors = check_command_policy.check(root)
            self.assertTrue(any(
                "command path reaches recording context" in error
                for error in errors
            ))

    def test_rejects_allocation_in_any_reachable_overload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = TABLE_SOURCE.replace(
                "fn void vk_end_commands() {}",
                (
                    "fn void vk_end_commands() { overloaded_finish(); }\n"
                    "fn void overloaded_finish() {}\n"
                    "fn void overloaded_finish(uint count) {\n"
                    "    vk::allocate_command_buffers(device, info, buffers);\n"
                    "}"
                ),
            )
            self.write_source(root, "gpu/vk/device.c3", source)
            errors = check_command_policy.check(root)
            self.assertTrue(any(
                "warm command path reaches native command allocation" in error
                for error in errors
            ))


if __name__ == "__main__":
    unittest.main()
