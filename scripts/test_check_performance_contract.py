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
    "gpu/vk/pipeline_compute.c3",
    "gpu/vk/queue.c3",
    "gpu/vk/render_pass.c3",
    "gpu/vk/sampler.c3",
    "gpu/vk/shader.c3",
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

    def test_direct_sampler_table_scan_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.copied_tree(root)
            self.mutate(
                root,
                "gpu/vk/sampler.c3",
                "    ulong hash = sampler_key_hash(state, &key);",
                (
                    "    foreach (&cell : state.samplers.slots[:state.samplers.count]) {\n"
                    "        if (sampler_key_equal(&cell.key, &key)) return cell.index;\n"
                    "    }\n"
                    "    ulong hash = sampler_key_hash(state, &key);"
                ),
            )
            errors = check_performance_contract.check(root)
            self.assertTrue(any(
                "whole-table sampler scan" in error for error in errors
            ))

    def test_while_sampler_table_scan_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.copied_tree(root)
            self.mutate(
                root,
                "gpu/vk/sampler.c3",
                "    ulong hash = sampler_key_hash(state, &key);",
                (
                    "    uint scan_index;\n"
                    "    while (scan_index < state.samplers.count) {\n"
                    "        (void)state.samplers.slots[scan_index++];\n"
                    "    }\n"
                    "    ulong hash = sampler_key_hash(state, &key);"
                ),
            )
            errors = check_performance_contract.check(root)
            self.assertTrue(any(
                "whole-table sampler scan" in error for error in errors
            ))

    def test_reviewed_sampler_lookup_capacity_scan_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.copied_tree(root)
            self.mutate(
                root,
                "gpu/vk/sampler.c3",
                "    while (current != 0) {",
                (
                    "    for (uint index = 0; index < table.slots.len; index++) {\n"
                    "        (void)table.slots[index];\n"
                    "    }\n"
                    "    while (current != 0) {"
                ),
            )
            errors = check_performance_contract.check(root)
            self.assertTrue(any(
                "find_sampler_cell traverses the published sampler prefix"
                in error for error in errors
            ))

    def test_reviewed_sampler_lookup_chain_advance_is_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.copied_tree(root)
            self.mutate(
                root,
                "gpu/vk/sampler.c3",
                "        current = cell.next_in_bucket;",
                "        current = 0;",
            )
            errors = check_performance_contract.check(root)
            self.assertTrue(any(
                "find_sampler_cell must advance through bucket links" in error
                for error in errors
            ))

    def test_recursive_sampler_lookup_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.copied_tree(root)
            self.mutate(
                root,
                "gpu/vk/sampler.c3",
                "    uint current = table.bucket_heads[sampler_bucket(",
                (
                    "    if (table.slots.len == 0) {\n"
                    "        return find_sampler_cell(state, table, hash, key);\n"
                    "    }\n"
                    "    uint current = table.bucket_heads[sampler_bucket("
                ),
            )
            errors = check_performance_contract.check(root)
            self.assertTrue(any(
                "find_sampler_cell performs forbidden recursive sampler lookup"
                in error for error in errors
            ))

    def test_relocated_sampler_table_scan_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.copied_tree(root)
            self.mutate(
                root,
                "gpu/vk/texture.c3",
                "fn gpu::TextureHandle? TextureTable.alloc(&self, TextureSlot value) {",
                (
                    "fn void find_sampler_cell(SamplerCell[] slots, uint count) {\n"
                    "    foreach (&cell : slots[:count]) {}\n"
                    "}\n\n"
                    "fn gpu::TextureHandle? TextureTable.alloc(&self, TextureSlot value) {"
                ),
            )
            self.mutate(
                root,
                "gpu/vk/sampler.c3",
                (
                    "    defer state.resource_mutex.unlock();\n\n"
                    "    SamplerTable* table = &state.samplers;"
                ),
                (
                    "    defer state.resource_mutex.unlock();\n\n"
                    "    SamplerTable* table = &state.samplers;\n"
                    "    find_sampler_cell(table.slots, table.count);"
                ),
            )
            errors = check_performance_contract.check(root)
            self.assertTrue(any(
                "find_sampler_cell performs forbidden whole-table sampler scan"
                in error for error in errors
            ))

    def test_sampler_hash_table_shape_is_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.copied_tree(root)
            self.mutate(
                root,
                "gpu/vk/sampler.c3",
                "    uint[]        bucket_heads;",
                "    uint[]        renamed_heads;",
            )
            errors = check_performance_contract.check(root)
            self.assertTrue(any(
                "SamplerTable must contain fixed slots" in error
                for error in errors
            ))

    def test_non_strict_sampler_table_allocation_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.copied_tree(root)
            self.mutate(
                root,
                "gpu/vk/device.c3",
                (
                    "    if (state.strict_heap.enabled) {\n"
                    "        create_sampler_table(state)!;"
                ),
                (
                    "    create_sampler_table(state)!;\n"
                    "    if (state.strict_heap.enabled) {"
                ),
            )
            errors = check_performance_contract.check(root)
            self.assertTrue(any(
                "allocate the sampler table only for strict devices" in error
                for error in errors
            ))

    def test_texture_barrier_duplicate_authoritative_call_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.copied_tree(root)
            self.mutate(
                root,
                "gpu/vk/sync.c3",
                (
                    "LoweredTextureBarrier lowered = "
                    "validate_and_lower_texture_barrier("
                ),
                (
                    "LoweredTextureBarrier duplicate = "
                    "validate_and_lower_texture_barrier(\n"
                    "        state, record, barrier, "
                    '"cmd_texture_barrier",\n'
                    "    )!!;\n"
                    "    LoweredTextureBarrier lowered = "
                    "validate_and_lower_texture_barrier("
                ),
            )
            errors = check_performance_contract.check(root)
            self.assertTrue(any(
                "validate_and_lower_texture_barrier(" in error
                for error in errors
            ))

    def test_texture_barrier_missing_work_counter_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.copied_tree(root)
            self.mutate(
                root,
                "gpu/vk/sync.c3",
                "note_texture_barrier_range_resolution(state);",
                "note_texture_barrier_native_assembly(state);",
            )
            errors = check_performance_contract.check(root)
            self.assertTrue(any(
                "note_texture_barrier_range_resolution(state);" in error
                for error in errors
            ))

    def test_retired_texture_barrier_lowering_path_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.copied_tree(root)
            self.mutate(
                root,
                "gpu/vk/sync.c3",
                "struct TextureStateScope {",
                "struct TextureUseScope {",
            )
            errors = check_performance_contract.check(root)
            self.assertTrue(any(
                "retired texture-barrier path TextureUseScope" in error
                for error in errors
            ))

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

    def test_bound_pipeline_validation_cell_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.copied_tree(root)
            self.mutate(
                root,
                "gpu/vk/command_state.c3",
                "    gpu::PipelineHandle               handle;",
                (
                    "    gpu::PipelineHandle               handle;\n"
                    "    PipelineCell*                     validation_cell;"
                ),
            )
            errors = check_performance_contract.check(root)
            self.assertTrue(
                any(
                    "forbidden bound-pipeline revalidation token "
                    "validation_cell" in error
                    for error in errors
                )
            )

    def test_bound_pipeline_expected_generation_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.copied_tree(root)
            self.mutate(
                root,
                "gpu/vk/command_state.c3",
                "    uint                              cache_entry;",
                (
                    "    uint                              expected_generation;\n"
                    "    uint                              cache_entry;"
                ),
            )
            errors = check_performance_contract.check(root)
            self.assertTrue(
                any(
                    "forbidden bound-pipeline revalidation token "
                    "expected_generation" in error
                    for error in errors
                )
            )

    def test_bound_pipeline_identity_validator_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.copied_tree(root)
            self.mutate(
                root,
                "gpu/vk/command.c3",
                "fn BoundPipeline*? active_bound_pipeline(",
                (
                    "fn void validate_bound_pipeline_identity() {}\n\n"
                    "fn BoundPipeline*? active_bound_pipeline("
                ),
            )
            errors = check_performance_contract.check(root)
            self.assertTrue(
                any(
                    "forbidden bound-pipeline revalidation token "
                    "validate_bound_pipeline_identity" in error
                    for error in errors
                )
            )

    def test_renamed_bound_pipeline_cell_revalidation_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.copied_tree(root)
            self.mutate(
                root,
                "gpu/vk/command_state.c3",
                "    gpu::PipelineHandle               handle;",
                (
                    "    gpu::PipelineHandle               handle;\n"
                    "    PipelineCell*                     bound_cell;\n"
                    "    uint                              generation_snapshot;"
                ),
            )
            self.mutate(
                root,
                "gpu/vk/command.c3",
                "fn BoundPipeline*? active_bound_pipeline(",
                (
                    "fn void? ensure_bound_pipeline_live(\n"
                    "    BoundPipeline* bound,\n"
                    ") {\n"
                    "    if (!bound.bound_cell.used\n"
                    "        || bound.bound_cell.generation\n"
                    "            != bound.generation_snapshot) {\n"
                    "        return gpu::INVALID_HANDLE~;\n"
                    "    }\n"
                    "}\n\n"
                    "fn BoundPipeline*? active_bound_pipeline("
                ),
            )
            errors = check_performance_contract.check(root)
            self.assertTrue(
                any(
                    "BoundPipeline must contain only the reviewed native "
                    "snapshot fields" in error
                    for error in errors
                )
            )

    def test_pipeline_cell_moved_to_command_record_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.copied_tree(root)
            self.mutate(
                root,
                "gpu/vk/command_state.c3",
                "    CommandState                state;",
                (
                    "    CommandState                state;\n"
                    "    PipelineCell*               retained_pipeline_slot;"
                ),
            )
            errors = check_performance_contract.check(root)
            self.assertTrue(any(
                "CommandRecord must not retain a PipelineCell pointer" in error
                for error in errors
            ))

    def test_missing_bound_pipeline_reports_contract_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.copied_tree(root)
            self.mutate(
                root,
                "gpu/vk/command_state.c3",
                "struct BoundPipeline {",
                "struct RemovedBoundPipeline {",
            )
            errors = check_performance_contract.check(root)
            self.assertTrue(any(
                "missing struct BoundPipeline" in error
                for error in errors
            ))

    def test_retired_compute_layout_cache_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.copied_tree(root)
            self.mutate(
                root,
                "gpu/vk/command.c3",
                "fn void? vk_cmd_dispatch_generated(",
                (
                    "fn uint retired_layout_cache_read(VkDeviceState* state) {\n"
                    "    return state.compute_layout_cache.count;\n"
                    "}\n\n"
                    "fn void? vk_cmd_dispatch_generated("
                ),
            )
            errors = check_performance_contract.check(root)
            self.assertTrue(
                any(
                    "singleton compute layouts" in error
                    for error in errors
                )
            )

    def test_missing_singleton_compute_layout_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.copied_tree(root)
            self.mutate(
                root,
                "gpu/vk/device.c3",
                "vk::PipelineLayout         compute_layout;",
                "vk::PipelineLayout         compute_layout_removed;",
            )
            errors = check_performance_contract.check(root)
            self.assertTrue(
                any(
                    "exactly one compute pipeline layout" in error
                    for error in errors
                )
            )

    def test_compute_push_size_in_pipeline_key_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.copied_tree(root)
            self.mutate(
                root,
                "gpu/vk/pipeline_cache.c3",
                ".fragment_shader = SHADER_ID_INVALID,",
                (
                    ".fragment_shader = SHADER_ID_INVALID,\n"
                    "        .sample_count = (int)desc.push_constant_size,"
                ),
            )
            errors = check_performance_contract.check(root)
            self.assertTrue(
                any("fixed root layout" in error for error in errors)
            )

    def test_dynamic_raster_state_in_pipeline_key_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.copied_tree(root)
            self.mutate(
                root,
                "gpu/vk/pipeline_cache.c3",
                "key.sample_count           = (int)desc.sample_count;",
                (
                    "key.sample_count           = (int)desc.sample_count;\n"
                    "    key.depth_format = (int)desc.raster.cull_mode;"
                ),
            )
            errors = check_performance_contract.check(root)
            self.assertTrue(
                any("build_graphics_key" in error for error in errors)
            )

    def test_per_target_pipeline_key_shape_is_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.copied_tree(root)
            self.mutate(
                root,
                "gpu/vk/pipeline_cache.c3",
                "ColorTargetKey[gpu::MAX_COLOR_ATTACHMENTS] color_targets;",
                "int[gpu::MAX_COLOR_ATTACHMENTS] color_formats;",
            )
            errors = check_performance_contract.check(root)
            self.assertTrue(
                any("immutable key shape" in error for error in errors)
            )

    def test_pipeline_cache_entry_cannot_own_shader_code(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.copied_tree(root)
            self.mutate(
                root,
                "gpu/vk/pipeline_cache.c3",
                "PipelineKey                    key;",
                (
                    "PipelineKey                    key;\n"
                    "    gpu::ShaderCode                cached_shader;"
                ),
            )
            errors = check_performance_contract.check(root)
            self.assertTrue(any(
                "PipelineCacheEntry" in error
                for error in errors
            ))

    def test_pipeline_lookup_cannot_call_renamed_shader_comparator(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.copied_tree(root)
            self.mutate(
                root,
                "gpu/vk/pipeline_cache.c3",
                "PipelineCacheTable* cache = &state.pipeline_cache;",
                (
                    "PipelineCacheTable* cache = &state.pipeline_cache;\n"
                    "    compare_payload_equivalent(state);"
                ),
            )
            errors = check_performance_contract.check(root)
            self.assertTrue(any(
                "unexpected calls" in error
                for error in errors
            ))

    def test_pipeline_lookup_cannot_compare_spirv_directly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.copied_tree(root)
            self.mutate(
                root,
                "gpu/vk/pipeline_cache.c3",
                "PipelineCacheTable* cache = &state.pipeline_cache;",
                (
                    "PipelineCacheTable* cache = &state.pipeline_cache;\n"
                    "    mem::equals(state.shader_store.entries[0].spirv, "
                    "state.shader_store.entries[0].spirv);"
                ),
            )
            errors = check_performance_contract.check(root)
            self.assertTrue(any(
                "find_entry contains forbidden" in error
                for error in errors
            ))

    def test_shader_interning_exact_byte_check_is_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.copied_tree(root)
            self.mutate(
                root,
                "gpu/vk/shader.c3",
                "return mem::equals(entry.spirv, code.spirv);",
                "return true;",
            )
            errors = check_performance_contract.check(root)
            self.assertTrue(any(
                "collision verification" in error
                for error in errors
            ))

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
                "(void)state.submit_completion_wait_calls.add(1, AtomicOrdering.RELAXED);",
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
                    "    release_completed_submitted_commands_locked(state, queue_id, target);\n"
                    "    completion.retired_sequence.store(target, AtomicOrdering.RELEASE);"
                ),
                (
                    "    completion.retired_sequence.store(target, AtomicOrdering.RELEASE);\n"
                    "    release_completed_submitted_commands_locked(state, queue_id, target);"
                ),
            )
            errors = check_performance_contract.check(root)
            self.assertTrue(
                any(
                    "completion ordering" in error
                    or "ownership flow must match reviewed source" in error
                    for error in errors
                )
            )

    def test_completion_poll_fast_path_reordering_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.copied_tree(root)
            self.mutate(
                root,
                "gpu/vk/sync.c3",
                "    ulong current;\n    note_completion_counter_query(state);",
                (
                    "    note_completion_counter_query(state);\n"
                    "    ulong current;"
                ),
            )
            errors = check_performance_contract.check(root)
            self.assertTrue(any(
                "vk_poll_completion_with_query" in error
                for error in errors
            ))

    def test_retirement_published_prefix_cap_is_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.copied_tree(root)
            self.mutate(
                root,
                "gpu/vk/lifetime.c3",
                "        AtomicOrdering.ACQUIRE,",
                "        AtomicOrdering.RELAXED,",
            )
            errors = check_performance_contract.check(root)
            self.assertTrue(any(
                "retire_queue_through_locked" in error
                for error in errors
            ))

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
