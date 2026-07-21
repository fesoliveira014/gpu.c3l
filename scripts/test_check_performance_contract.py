from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from scripts import check_performance_contract


REQUIRED_PATHS = (
    "gpu/command.c3",
    "gpu/device.c3",
    "gpu/vk/attachment_view.c3",
    "gpu/vk/command.c3",
    "gpu/vk/device.c3",
    "gpu/vk/command_state.c3",
    "gpu/vk/lifetime.c3",
    "gpu/vk/pipeline_cache.c3",
    "gpu/vk/queue.c3",
    "gpu/vk/render_pass.c3",
    "gpu/vk/sync.c3",
    "gpu/vk/texture.c3",
    "test/src/command_record_bench.c3",
    "test/src/lifecycle_bench.c3",
)


class PerformanceContractTests(unittest.TestCase):
    def copied_tree(self, destination: Path) -> None:
        for relative in REQUIRED_PATHS:
            source = check_performance_contract.ROOT / relative
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)

    def mutate(
        self,
        root: Path,
        relative: str,
        old: str,
        new: str,
    ) -> None:
        path = root / relative
        source = path.read_text(encoding="utf-8")
        self.assertIn(old, source)
        path.write_text(source.replace(old, new, 1), encoding="utf-8")

    def test_current_sources_satisfy_contract(self):
        self.assertEqual(check_performance_contract.check(), [])

    def test_registry_lock_on_public_recording_path_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.copied_tree(root)
            self.mutate(
                root,
                "gpu/command.c3",
                "fn void? cmd_barrier(CommandList* commands, Barrier* barrier) {",
                (
                    "fn void? cmd_barrier(CommandList* commands, Barrier* barrier) {\n"
                    "    lock_device_registry();"
                ),
            )
            errors = check_performance_contract.check(root)
            self.assertTrue(any("lock_device_registry(" in error for error in errors))

    def test_lifecycle_vtable_on_public_recording_path_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.copied_tree(root)
            self.mutate(
                root,
                "gpu/command.c3",
                "fn void? cmd_barrier(CommandList* commands, Barrier* barrier) {",
                (
                    "fn void? cmd_barrier(CommandList* commands, Barrier* barrier) {\n"
                    "    command_operation(commands)!.vtable.cmd_barrier("
                    "commands, barrier);"
                ),
            )
            errors = check_performance_contract.check(root)
            self.assertTrue(any("command_operation(" in error for error in errors))

    def test_public_recording_helper_relocation_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.copied_tree(root)
            self.mutate(
                root,
                "gpu/command.c3",
                "fn void? cmd_barrier(CommandList* commands, Barrier* barrier) {",
                (
                    "fn void forbidden_public_resolution(CommandList* commands) {\n"
                    "    (void)command_operation(commands);\n"
                    "}\n\n"
                    "fn void? cmd_barrier(CommandList* commands, Barrier* barrier) {\n"
                    "    forbidden_public_resolution(commands);"
                ),
            )
            errors = check_performance_contract.check(root)
            self.assertTrue(any("command_operation(" in error for error in errors))

    def test_backend_state_helper_relocation_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.copied_tree(root)
            self.mutate(
                root,
                "gpu/vk/render_pass.c3",
                "fn void? vk_cmd_draw_generated(",
                (
                    "fn void forbidden_recording_state_lookup(gpu::Device* device) {\n"
                    "    (void)gpu::device_backend_state_ptr(device);\n"
                    "}\n\n"
                    "fn void? vk_cmd_draw_generated("
                ),
            )
            self.mutate(
                root,
                "gpu/vk/render_pass.c3",
                "    CommandRecord* record = encoder_command(commands);",
                (
                    "    forbidden_recording_state_lookup(&commands.device);\n"
                    "    CommandRecord* record = encoder_command(commands);"
                ),
            )
            errors = check_performance_contract.check(root)
            self.assertTrue(
                any("device_backend_state_ptr(" in error for error in errors)
            )

    def test_post_bind_pipeline_helper_relocation_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.copied_tree(root)
            self.mutate(
                root,
                "gpu/vk/render_pass.c3",
                "fn void? vk_cmd_draw_generated(",
                (
                    "fn void forbidden_pipeline_lookup(VkDeviceState* state) {\n"
                    "    (void)state.pipelines.get({});\n"
                    "}\n\n"
                    "fn void? vk_cmd_draw_generated("
                ),
            )
            self.mutate(
                root,
                "gpu/vk/render_pass.c3",
                "    CommandRecord* record = encoder_command(commands);",
                (
                    "    forbidden_pipeline_lookup(encoder_device_state(commands));\n"
                    "    CommandRecord* record = encoder_command(commands);"
                ),
            )
            errors = check_performance_contract.check(root)
            self.assertTrue(
                any("post-bind pipeline resolution" in error for error in errors)
            )

    def test_compute_layout_cache_access_from_recording_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.copied_tree(root)
            self.mutate(
                root,
                "gpu/vk/command.c3",
                "fn void? vk_cmd_dispatch_generated(",
                (
                    "fn uint forbidden_layout_cache_read(VkDeviceState* state) {\n"
                    "    return state.compute_layout_cache.count;\n"
                    "}\n\n"
                    "fn void? vk_cmd_dispatch_generated("
                ),
            )
            errors = check_performance_contract.check(root)
            self.assertTrue(
                any(
                    "must not access compute-layout cache storage" in error
                    for error in errors
                )
            )

    def test_compute_layout_cache_access_from_other_recording_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.copied_tree(root)
            self.mutate(
                root,
                "gpu/vk/render_pass.c3",
                "fn void? vk_cmd_draw_generated(",
                (
                    "fn uint forbidden_layout_cache_helper(VkDeviceState* state) {\n"
                    "    return state.compute_layout_cache.count;\n"
                    "}\n\n"
                    "fn void? vk_cmd_draw_generated("
                ),
            )
            errors = check_performance_contract.check(root)
            self.assertTrue(
                any(
                    "gpu/vk/render_pass.c3 must not access compute-layout "
                    "cache storage" in error
                    for error in errors
                )
            )

    def test_completion_point_allocation_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.copied_tree(root)
            self.mutate(
                root,
                "gpu/vk/queue.c3",
                "return gpu::next_completion_point(queue, &completion.next_sequence);",
                (
                    "mem::new(gpu::CompletionPoint);\n"
                    "    return gpu::next_completion_point("
                    "queue, &completion.next_sequence);"
                ),
            )
            errors = check_performance_contract.check(root)
            self.assertTrue(any("reserve_queue_completion_locked" in error for error in errors))

    def test_hidden_destruction_wait_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.copied_tree(root)
            self.mutate(
                root,
                "gpu/vk/texture.c3",
                "fn void? vk_destroy_texture(gpu::Device* device, gpu::TextureHandle handle) @private {",
                (
                    "fn void? vk_destroy_texture("
                    "gpu::Device* device, gpu::TextureHandle handle) @private {\n"
                    "    wait_completion(device);"
                ),
            )
            errors = check_performance_contract.check(root)
            self.assertTrue(any("wait_completion(" in error for error in errors))

    def test_missing_completion_wait_instrumentation_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.copied_tree(root)
            self.mutate(
                root,
                "gpu/vk/sync.c3",
                "state.submit_stats.completion_wait_calls++;",
                "state.submit_stats.queue_submits++;",
            )
            errors = check_performance_contract.check(root)
            self.assertTrue(
                any(
                    "gpu/vk/sync.c3 is missing performance evidence token"
                    in error
                    for error in errors
                )
            )

    def test_missing_deferred_release_instrumentation_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.copied_tree(root)
            self.mutate(
                root,
                "gpu/vk/lifetime.c3",
                "fn CommandBufferStats command_buffer_stats(",
                "fn CommandBufferStats missing_stats(",
            )
            errors = check_performance_contract.check(root)
            self.assertTrue(
                any(
                    "gpu/vk/lifetime.c3 is missing performance evidence token"
                    in error
                    for error in errors
                )
            )

    def test_generated_native_execute_hidden_double_call_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.copied_tree(root)
            self.mutate(
                root,
                "gpu/vk/command.c3",
                "fn void? execute_generated_work(",
                (
                    "fn void execute_generated_commands_twice(\n"
                    "    VkDeviceState* state,\n"
                    "    vk::CommandBuffer command_buffer,\n"
                    "    vk::GeneratedCommandsInfoEXT* info,\n"
                    ") {\n"
                    "    state.device_dispatch.generated_work."
                    "cmd_execute_generated_commands(\n"
                    "        command_buffer,\n"
                    "        vk::FALSE,\n"
                    "        info,\n"
                    "    );\n"
                    "    state.device_dispatch.generated_work."
                    "cmd_execute_generated_commands(\n"
                    "        command_buffer,\n"
                    "        vk::FALSE,\n"
                    "        info,\n"
                    "    );\n"
                    "}\n\n"
                    "fn void? execute_generated_work("
                ),
            )
            self.mutate(
                root,
                "gpu/vk/command.c3",
                (
                    "    state.device_dispatch.generated_work."
                    "cmd_execute_generated_commands(\n"
                    "        record.command_buffer,\n"
                    "        vk::FALSE,\n"
                    "        &info,\n"
                    "    );"
                ),
                (
                    "    execute_generated_commands_twice(\n"
                    "        state,\n"
                    "        record.command_buffer,\n"
                    "        &info,\n"
                    "    );"
                ),
            )
            errors = check_performance_contract.check(root)
            self.assertTrue(
                any(
                    "must issue exactly one native call" in error
                    for error in errors
                )
            )

    def test_generated_native_execute_loop_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.copied_tree(root)
            self.mutate(
                root,
                "gpu/vk/command.c3",
                (
                    "    state.device_dispatch.generated_work."
                    "cmd_execute_generated_commands(\n"
                    "        record.command_buffer,\n"
                    "        vk::FALSE,\n"
                    "        &info,\n"
                    "    );"
                ),
                (
                    "    for (uint i = 0; i < 2; i++) {\n"
                    "    state.device_dispatch.generated_work."
                    "cmd_execute_generated_commands(\n"
                    "        record.command_buffer,\n"
                    "        vk::FALSE,\n"
                    "        &info,\n"
                    "    );\n"
                    "    }"
                ),
            )
            errors = check_performance_contract.check(root)
            self.assertTrue(
                any(
                    "ownership flow must match reviewed source" in error
                    for error in errors
                )
            )

    def test_unreviewed_generated_preprocess_function_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.copied_tree(root)
            unreviewed = root / "gpu/vk/unreviewed_generated_preprocess.c3"
            unreviewed.write_text(
                "module gpu::vk;\n\nfn void unreviewed_generated_preprocess_owner() {}\n",
                encoding="utf-8",
            )
            errors = check_performance_contract.check(root)
            self.assertTrue(
                any(
                    "ownership flow is unreviewed" in error
                    for error in errors
                )
            )

    def test_unreviewed_completion_release_caller_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.copied_tree(root)
            unreviewed = root / "gpu/vk/unreviewed_completion_release.c3"
            unreviewed.write_text(
                (
                    "module gpu::vk;\n\n"
                    "fn void unreviewed_release(VkDeviceState* state) {\n"
                    "    release_submitted_command_batch(state, 0);\n"
                    "}\n"
                ),
                encoding="utf-8",
            )
            errors = check_performance_contract.check(root)
            self.assertTrue(
                any(
                    "ownership flow is unreviewed" in error
                    for error in errors
                )
            )

    def test_comment_separated_completion_release_caller_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.copied_tree(root)
            unreviewed = root / "gpu/vk/unreviewed_completion_release.c3"
            unreviewed.write_text(
                (
                    "module gpu::vk;\n\n"
                    "fn void unreviewed_release /* reviewed? */ (VkDeviceState* state) {\n"
                    "    release_submitted_command_batch // completion gate\n"
                    "        (state, 0);\n"
                    "}\n"
                ),
                encoding="utf-8",
            )
            errors = check_performance_contract.check(root)
            self.assertTrue(
                any(
                    "ownership flow is unreviewed" in error
                    for error in errors
                )
            )

    def test_generated_early_completion_recycle_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.copied_tree(root)
            self.mutate(
                root,
                "gpu/vk/lifetime.c3",
                (
                    "        if (completed[queue_id]\n"
                    "            < gpu::completion_point_sequence(batch.completion)) {"
                ),
                "        if (false) {",
            )
            errors = check_performance_contract.check(root)
            self.assertTrue(
                any(
                    "ownership flow must match reviewed source" in error
                    for error in errors
                )
            )

    def test_generated_hot_acquisition_allocation_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.copied_tree(root)
            self.mutate(
                root,
                "gpu/vk/command.c3",
                (
                    "    uint max_count,\n"
                    ") {\n"
                    "    state.resource_mutex.lock()!!;"
                ),
                (
                    "    uint max_count,\n"
                    ") {\n"
                    "    alloc::new_array(state.host_allocator, char, 1);\n"
                    "    state.resource_mutex.lock()!!;"
                ),
            )
            errors = check_performance_contract.check(root)
            self.assertTrue(any("contains forbidden alloc::" in error for error in errors))

    def test_render_pass_image_view_creation_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.copied_tree(root)
            self.mutate(
                root,
                "gpu/vk/render_pass.c3",
                "    vk::RenderingAttachmentInfo depth_attachment;",
                (
                    "    vk::create_image_view(state.device, null, null, null);\n"
                    "    vk::RenderingAttachmentInfo depth_attachment;"
                ),
            )
            errors = check_performance_contract.check(root)
            self.assertTrue(any("create_image_view(" in error for error in errors))

    def test_generated_vma_allocation_counter_is_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.copied_tree(root)
            self.mutate(
                root,
                "gpu/vk/command.c3",
                "    note_recording_vma_allocation(state);",
                "    (void)state;",
            )
            errors = check_performance_contract.check(root)
            self.assertTrue(any("native work seam" in error for error in errors))

    def test_command_buffer_reset_counter_is_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.copied_tree(root)
            self.mutate(
                root,
                "gpu/vk/command.c3",
                "        note_command_buffer_reset(state);",
                "        (void)state;",
            )
            errors = check_performance_contract.check(root)
            self.assertTrue(any("native work seam" in error for error in errors))

    def test_attachment_image_view_counter_is_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.copied_tree(root)
            self.mutate(
                root,
                "gpu/vk/attachment_view.c3",
                "            (void)state.recording_image_view_creations.add(",
                "            (void)state.generated_scratch_misses.add(",
            )
            errors = check_performance_contract.check(root)
            self.assertTrue(any("native work seam" in error for error in errors))

    def test_generated_capacity_guard_is_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.copied_tree(root)
            self.mutate(
                root,
                "gpu/vk/command.c3",
                "max_count > reserved.reservation_max_commands",
                "max_count > reservation_max_commands",
            )
            errors = check_performance_contract.check(root)
            self.assertTrue(any("bounded reservation step" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
