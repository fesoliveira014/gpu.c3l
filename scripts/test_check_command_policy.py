from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts import check_command_policy


TABLE_SOURCE = """
module gpu::internal::vk;

const gpu::internal::CommandOps TRUSTED_COMMAND_OPS @private = {
    .copy = &trusted_copy,
};

const gpu::internal::CommandOps TRUSTED_TRACKING_COMMAND_OPS @private = {
    .copy = &trusted_tracking_copy,
};

const gpu::internal::CommandOps CHECKED_COMMAND_OPS @private = {
    .copy = &checked_copy,
};

const gpu::internal::CommandOps CHECKED_TRACKING_COMMAND_OPS @private = {
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
    retain_tracked_command_reference();
    publish_command_reference();
}

fn void resolve_tracked_texture_command_reference() {
    tracked_command_references.add();
    publish_command_reference();
}

fn bool indexed_resource_reference_matches() {
    return cell.owner == reference.owner
        && cell.index == reference.index
        && cell.generation == reference.generation;
}

fn void ensure_command_reference_capacity() {
    note_command_reference_allocation();
}

fn void note_command_reference_allocation() {}
fn void rollback_command_references() {
    reset_command_reference_index();
    for (uint i = 0; i < reference_count; i++) publish_command_reference();
}
fn void retain_tracked_command_reference() {}
fn void publish_command_reference() {}
fn void reset_command_reference_index() {}

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

ENCODER_PROOF_SOURCE = """
module gpu::internal;

fn void command_encoder() {
    note_command_encoder_cell_computation();
    if (opaque != encoder) return;
    note_command_encoder_lease_comparison();
    if (encoder.lease != make_command_lease(device, handle)) return;
}

fn void recording_encoder() {
    command_encoder();
}

fn void executable_encoder() {
    command_encoder();
}

fn void command_operation() {}
fn void executable_command_operation() {}
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
            self.write_source(root, "gpu/internal/vk/device.c3", TABLE_SOURCE)
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
            self.write_source(root, "gpu/internal/vk/device.c3", source)
            errors = check_command_policy.check(root)
            self.assertTrue(any("reads validation_policy" in error for error in errors))

    def test_rejects_trusted_encoder_capability_check(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = TABLE_SOURCE.replace(
                "fn void trusted_copy() {\n    lower_copy();\n}",
                (
                    "fn void trusted_copy() {\n"
                    "    if (commands == null) return;\n"
                    "    lower_copy();\n"
                    "}"
                ),
            )
            self.write_source(root, "gpu/internal/vk/device.c3", source)
            errors = check_command_policy.check(root)
            self.assertTrue(any(
                "trusted command path reaches encoder null check" in error
                for error in errors
            ))

    def test_rejects_duplicate_frontend_encoder_work(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_source(root, "gpu/internal/vk/device.c3", TABLE_SOURCE)
            source = ENCODER_PROOF_SOURCE.replace(
                "    if (opaque != encoder) return;\n",
                (
                    "    if (opaque != encoder) return;\n"
                    "    if (encoder.handle != handle) return;\n"
                ),
            )
            self.write_source(root, "gpu/internal/command.c3", source)
            errors = check_command_policy.check(root)
            self.assertTrue(any(
                "frontend command resolution performs stored handle comparison"
                in error
                for error in errors
            ))

    def test_rejects_frontend_device_loss_load(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_source(root, "gpu/internal/vk/device.c3", TABLE_SOURCE)
            source = ENCODER_PROOF_SOURCE.replace(
                "    note_command_encoder_lease_comparison();\n",
                (
                    "    if (encoder.device_lost.load(AtomicOrdering.ACQUIRE)) return;\n"
                    "    note_command_encoder_lease_comparison();\n"
                ),
            )
            self.write_source(root, "gpu/internal/command.c3", source)
            errors = check_command_policy.check(root)
            self.assertTrue(any(
                "frontend command resolution performs device-loss load"
                for error in errors
            ))

    def test_rejects_device_loss_load_in_recording_encoder(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_source(root, "gpu/internal/vk/device.c3", TABLE_SOURCE)
            source = ENCODER_PROOF_SOURCE.replace(
                "fn void recording_encoder() {\n    command_encoder();\n}",
                (
                    "fn void recording_encoder() {\n"
                    "    if (encoder.device_lost.load(AtomicOrdering.ACQUIRE)) return;\n"
                    "    command_encoder();\n"
                    "}"
                ),
            )
            self.write_source(root, "gpu/internal/command.c3", source)
            errors = check_command_policy.check(root)
            self.assertTrue(any(
                "frontend command resolution performs device-loss load"
                in error
                and error.endswith(":recording_encoder")
                for error in errors
            ))

    def test_rejects_missing_encoder_lease_note(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_source(root, "gpu/internal/vk/device.c3", TABLE_SOURCE)
            source = ENCODER_PROOF_SOURCE.replace(
                "    note_command_encoder_lease_comparison();\n",
                "",
            )
            self.write_source(root, "gpu/internal/command.c3", source)
            errors = check_command_policy.check(root)
            self.assertIn(
                "command_encoder must record exactly one note_command_encoder_lease_comparison",
                errors,
            )

    def test_rejects_duplicate_encoder_lease_note(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_source(root, "gpu/internal/vk/device.c3", TABLE_SOURCE)
            source = ENCODER_PROOF_SOURCE.replace(
                "    note_command_encoder_lease_comparison();\n",
                (
                    "    note_command_encoder_lease_comparison();\n"
                    "    note_command_encoder_lease_comparison();\n"
                ),
            )
            self.write_source(root, "gpu/internal/command.c3", source)
            errors = check_command_policy.check(root)
            self.assertIn(
                "command_encoder must record exactly one note_command_encoder_lease_comparison",
                errors,
            )

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
            self.write_source(root, "gpu/internal/vk/device.c3", source)
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
            self.write_source(root, "gpu/internal/vk/device.c3", source)
            self.write_source(
                root,
                "gpu/internal/vk/nested/helper.c3",
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
            self.write_source(root, "gpu/internal/vk/device.c3", source)
            errors = check_command_policy.check(root)
            self.assertTrue(any("reads debug_names" in error for error in errors))

    def test_rejects_superseded_command_function(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = TABLE_SOURCE + "\nfn void vk_cmd_copy_buffer() {}\n"
            self.write_source(root, "gpu/internal/vk/device.c3", source)
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
const gpu::internal::CommandOps EXPERIMENTAL_COMMAND_OPS @private = {
    .copy = &trusted_copy,
};
"""
            self.write_source(root, "gpu/internal/vk/device.c3", source)
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
            self.write_source(root, "gpu/internal/vk/device.c3", source)
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
            self.write_source(root, "gpu/internal/vk/device.c3", source)
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
            self.write_source(root, "gpu/internal/vk/device.c3", source)
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
            self.write_source(root, "gpu/internal/vk/device.c3", source)
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
            self.write_source(root, "gpu/internal/vk/device.c3", source)
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
            self.write_source(root, "gpu/internal/vk/device.c3", source)
            self.write_source(
                root,
                "gpu/internal/vk/nested/helper.c3",
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
            self.write_source(root, "gpu/internal/vk/device.c3", source)
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
            self.write_source(root, "gpu/internal/vk/device.c3", source)
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
            self.write_source(root, "gpu/internal/vk/device.c3", source)
            self.write_source(
                root,
                "gpu/internal/vk/nested/helper.c3",
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
            self.write_source(root, "gpu/internal/vk/device.c3", source)
            errors = check_command_policy.check(root)
            self.assertTrue(any(
                "warm command path reaches native command allocation" in error
                for error in errors
            ))

    def test_rejects_accumulated_reference_count_loop(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = TABLE_SOURCE.replace(
                "fn void track_command_reference() {\n"
                "    ensure_command_reference_capacity();\n"
                "    retain_tracked_command_reference();\n"
                "    publish_command_reference();\n"
                "}",
                (
                    "fn void track_command_reference() {\n"
                    "    for (uint i = 0; i < scratch.reference_count; i++) {\n"
                    "        inspect(scratch.references[i]);\n"
                    "    }\n"
                    "}"
                ),
            )
            self.write_source(root, "gpu/internal/vk/device.c3", source)
            errors = check_command_policy.check(root)
            self.assertTrue(any(
                "reference index hot path contains reference-count loop"
                in error
                for error in errors
            ))

    def test_rejects_accumulated_reference_slice_loop(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = TABLE_SOURCE + """
fn void preflight_command_references() {
    foreach (&reference : scratch.references[:scratch.reference_count]) {
        inspect(reference);
    }
}
"""
            self.write_source(root, "gpu/internal/vk/device.c3", source)
            errors = check_command_policy.check(root)
            self.assertTrue(any(
                "reference index hot path contains reference-slice loop"
                in error
                for error in errors
            ))

    def test_rejects_accumulated_scan_in_reachable_helper(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = TABLE_SOURCE.replace(
                "    ensure_command_reference_capacity();",
                "    scan_reference_helper();\n"
                "    ensure_command_reference_capacity();",
                1,
            ) + """
fn void scan_reference_helper() {
    for (uint i = 0; i < scratch.reference_count; i++) inspect(i);
}
"""
            self.write_source(root, "gpu/internal/vk/device.c3", source)
            errors = check_command_policy.check(root)
            self.assertTrue(any(
                "reference index hot path contains reference-count loop"
                in error
                and error.endswith(":scan_reference_helper")
                for error in errors
            ))

    def test_rejects_reference_publication_before_retain(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = TABLE_SOURCE.replace(
                "    retain_tracked_command_reference();\n"
                "    publish_command_reference();",
                "    publish_command_reference();\n"
                "    retain_tracked_command_reference();",
                1,
            )
            self.write_source(root, "gpu/internal/vk/device.c3", source)
            errors = check_command_policy.check(root)
            self.assertTrue(any(
                "reference index publication is not after retain"
                in error
                for error in errors
            ))

    def test_rejects_publication_hidden_in_pre_retain_helper(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = TABLE_SOURCE.replace(
                "    retain_tracked_command_reference();",
                "    publish_early();\n"
                "    retain_tracked_command_reference();",
                1,
            ) + "\nfn void publish_early() { publish_command_reference(); }\n"
            self.write_source(root, "gpu/internal/vk/device.c3", source)
            errors = check_command_policy.check(root)
            self.assertTrue(any(
                "reference index publication has unauthorized caller"
                in error
                and error.endswith(":publish_early")
                for error in errors
            ))

    def test_rejects_hash_only_reference_equality(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = TABLE_SOURCE.replace(
                "    return cell.owner == reference.owner\n"
                "        && cell.index == reference.index\n"
                "        && cell.generation == reference.generation;",
                "    return cell.hash == reference.hash;",
            )
            self.write_source(root, "gpu/internal/vk/device.c3", source)
            errors = check_command_policy.check(root)
            self.assertTrue(any(
                "reference index equality omits exact owner"
                in error
                for error in errors
            ))

    def test_rejects_unsafe_rollback_cell_deletion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = TABLE_SOURCE.replace(
                "    reset_command_reference_index();\n"
                "    for (uint i = 0; i < reference_count; i++) "
                "publish_command_reference();",
                "    reference_index.cells[index].epoch = 0;",
            )
            self.write_source(root, "gpu/internal/vk/device.c3", source)
            errors = check_command_policy.check(root)
            self.assertTrue(any(
                "reference rollback must reset and rebuild"
                in error
                for error in errors
            ))


if __name__ == "__main__":
    unittest.main()
