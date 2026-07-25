module gpu::internal::vk @private;

import gpu::internal @public;
import gpu @public;
import vk;
import std::core::mem;
import std::core::env;
import std::hash::fnv64a;

alias CmdCopyBufferFn @private               = fn void? (CommandRecord*, gpu::BufferCopyDesc*);
alias CmdFillBufferFn @private               = fn void? (CommandRecord*, gpu::GpuSpan, uint);
alias CmdCopyBufferToTextureFn @private      = fn void? (CommandRecord*, gpu::BufferTextureCopyDesc*);
alias CmdCopyTextureToBufferFn @private      = fn void? (CommandRecord*, gpu::TextureBufferCopyDesc*);
alias CmdBarrierFn @private                  = fn void? (CommandRecord*, gpu::Barrier*);
alias CmdTextureBarrierFn @private           = fn void? (CommandRecord*, gpu::TextureBarrier*);
alias CmdBindPipelineFn @private             = fn void? (CommandRecord*, gpu::PipelineHandle);
alias CmdSetDepthStateFn @private            = fn void? (CommandRecord*, gpu::DepthState*);
alias CmdSetRasterStateFn @private           = fn void? (CommandRecord*, gpu::DynamicRasterState*);
alias CmdSetGraphicsStateFn @private         = fn void? (CommandRecord*, gpu::GraphicsState*);
alias CmdSetColorStateFn @private            = fn void? (CommandRecord*, gpu::ColorState*);
alias CmdDispatchFn @private                 = fn void? (CommandRecord*, gpu::GpuAddress, gpu::Vec3u);
alias CmdBeginRenderPassFn @private          = fn void? (CommandRecord*, gpu::RenderPassDesc*);
alias CmdBeginRenderPassWithStateFn @private = fn void? (CommandRecord*, gpu::RenderPassDesc*, gpu::GraphicsState*);
alias CmdSetViewportFn @private              = fn void? (CommandRecord*, gpu::Viewport*);
alias CmdSetScissorFn @private               = fn void? (CommandRecord*, gpu::ScissorRect*);
alias CmdEndRenderPassFn @private            = fn void? (CommandRecord*);
alias CmdDrawFn @private = fn void? (
    CommandRecord* record,
    gpu::GpuAddress vertex_root,
    gpu::GpuAddress fragment_root,
    uint vertex_count,
    uint instance_count,
);
alias CmdDrawIndexedFn @private = fn void? (
    CommandRecord* record,
    gpu::GpuAddress vertex_root,
    gpu::GpuAddress fragment_root,
    gpu::GpuSpan index_span,
    uint index_count,
    uint instance_count,
    gpu::IndexType index_type,
);
alias CmdDrawIndirectFn @private = fn void? (
    CommandRecord* record,
    gpu::GpuAddress vertex_root,
    gpu::GpuAddress fragment_root,
    gpu::GpuSpan args,
    uint draw_count,
);
alias CmdDrawIndexedIndirectFn @private = fn void? (
    CommandRecord* record,
    gpu::GpuAddress vertex_root,
    gpu::GpuAddress fragment_root,
    gpu::GpuSpan args,
    uint draw_count,
    gpu::GpuSpan index_span,
    gpu::IndexType index_type,
);
alias CmdDrawIndexedIndirectCountFn @private = fn void? (
    CommandRecord* record,
    gpu::GpuAddress vertex_root,
    gpu::GpuAddress fragment_root,
    gpu::GpuSpan args,
    gpu::GpuSpan count_span,
    uint max_draw_count,
    gpu::GpuSpan index_span,
    gpu::IndexType index_type,
);
alias CmdDispatchIndirectFn @private = fn void? (
    CommandRecord*,
    gpu::GpuAddress,
    gpu::GpuSpan,
);
alias CmdDrawGeneratedFn @private = fn void? (
    CommandRecord* record,
    gpu::GpuSpan records,
    gpu::GpuSpan count_span,
    uint max_draw_count,
);
alias CmdDrawIndexedGeneratedFn @private = fn void? (
    CommandRecord* record,
    gpu::GpuSpan records,
    gpu::GpuSpan count_span,
    uint max_draw_count,
    gpu::GpuSpan index_span,
    gpu::IndexType index_type,
);
alias CmdDispatchGeneratedFn @private = fn void? (
    CommandRecord* record,
    gpu::GpuSpan records,
    gpu::GpuSpan count_span,
    uint max_dispatch_count,
);
alias CmdBeginLabelFn @private = fn void? (
    CommandRecord*,
    ZString,
    float[4],
);
alias CmdEndLabelFn @private = fn void? (CommandRecord*);

const uint MAX_COMMAND_LISTS = gpu::internal::MAX_DEVICE_COMMANDS;
const uint COMMAND_FREE_NONE = 0;

struct RenderCompatibility {
    uint                                    color_count;
    gpu::Format[gpu::MAX_COLOR_ATTACHMENTS] color_formats;
    gpu::Format                             depth_format;
    gpu::SampleCount                        sample_count;
}

struct ColorCompatibility {
    uint                                    count;
    gpu::Format[gpu::MAX_COLOR_ATTACHMENTS] formats;
}

struct BoundPipeline {
    gpu::PipelineHandle               handle;
    vk::Pipeline                      pipeline;
    vk::PipelineLayout                layout;
    vk::IndirectCommandsLayoutEXT     generated_dispatch_layout;
    uint                              cache_entry;
    PipelineKind                      kind;
    GraphicsColorProfile              color_profile;
    RenderCompatibility               render;
}

struct NativePipelineBinding {
    vk::Pipeline                  pipeline;
    vk::PipelineLayout            layout;
    vk::IndirectCommandsLayoutEXT generated_dispatch_layout;
    uint                          cache_entry;
    PipelineKind                  kind;
    GraphicsColorProfile          color_profile;
    RenderCompatibility           render;
}

struct StrictHeapCommandBinding {
    bool descriptor_buffer_bound;
    bool graphics_offsets_set;
    bool compute_offsets_set;
    bool graphics_set_bound;
    bool compute_set_bound;
}

struct CommandRecord @public {
    CommandOps*                      ops;
    VkDeviceState*                   device_state;
    gpu::Device                      device;
    gpu::internal::CommandListHandle identity;
    gpu::internal::CommandState      state;
    vk::CommandBuffer                command_buffer;
    gpu::CommandAllocatorHandle      allocator;
    uint                             allocator_buffer;
    gpu::Queue                       queue;
    ulong                            submit_visit_epoch;
    gpu::CompletionPoint             submitted_completion;
    CommandRecord*                   next_submitted;
    RenderCompatibility              active_render;
    bool                             graphics_state_initialized;
    bool                             color_state_initialized;
    ColorCompatibility               color_state;
    BoundPipeline                    logical_pipeline;
    NativePipelineBinding            native_graphics_pipeline;
    NativePipelineBinding            native_compute_pipeline;
    StrictHeapCommandBinding         strict_heap_binding;
    CommandScratch*                  scratch;
}

struct CommandOps @public {
    CmdCopyBufferFn               copy_buffer;
    CmdFillBufferFn               fill_buffer;
    CmdCopyBufferToTextureFn      copy_buffer_to_texture;
    CmdCopyTextureToBufferFn      copy_texture_to_buffer;
    CmdBarrierFn                  barrier;
    CmdTextureBarrierFn           texture_barrier;
    CmdBindPipelineFn             bind_pipeline;
    CmdSetDepthStateFn            set_depth_state;
    CmdSetRasterStateFn           set_raster_state;
    CmdSetGraphicsStateFn         set_graphics_state;
    CmdSetColorStateFn            set_color_state;
    CmdDispatchFn                 dispatch;
    CmdBeginRenderPassFn          begin_render_pass;
    CmdBeginRenderPassWithStateFn begin_render_pass_with_state;
    CmdSetViewportFn              set_viewport;
    CmdSetScissorFn               set_scissor;
    CmdEndRenderPassFn            end_render_pass;
    CmdDrawFn                     draw;
    CmdDrawIndexedFn              draw_indexed;
    CmdDrawIndirectFn             draw_indirect;
    CmdDrawIndexedIndirectFn      draw_indexed_indirect;
    CmdDrawIndexedIndirectCountFn draw_indexed_indirect_count;
    CmdDispatchIndirectFn         dispatch_indirect;
    CmdDrawGeneratedFn            draw_generated;
    CmdDrawIndexedGeneratedFn     draw_indexed_generated;
    CmdDispatchGeneratedFn        dispatch_generated;
    CmdBeginLabelFn               begin_label;
    CmdEndLabelFn                 end_label;
}

struct CommandCell {
    CommandRecord record;
    uint          generation;
    uint          next_free;
    bool          used;
}

struct CommandTable {
    CommandCell[] slots;
    ulong         owner;
    uint          free_head;
    uint          next_new;
}

struct CommandReferenceProbe {
    uint cell_index;
    uint reference_index;
    bool found;
}

fn CommandRecord*? CommandTable.claim(
    &self,
    gpu::Device device,
    CommandOps* ops,
    VkDeviceState* device_state,
    CommandRecord value,
) {
    uint idx;
    bool found = false;
    while (self.free_head != COMMAND_FREE_NONE) {
        idx = self.free_head - 1;
        self.free_head = self.slots[idx].next_free;
        if (can_allocate_handle_generation(self.slots[idx].generation)) {
            found = true;
            break;
        }
        self.slots[idx].next_free = COMMAND_FREE_NONE;
    }
    if (!found) {
        if (self.next_new >= self.slots.len) return gpu::SLOT_TABLE_FULL~;
        idx = self.next_new++;
        self.slots[idx].generation = 0;
    }
    self.slots[idx].generation = next_handle_generation(self.slots[idx].generation)!;
    self.slots[idx].used = true;
    gpu::internal::CommandListHandle identity = {
        .owner = self.owner,
        .index = idx,
        .generation = self.slots[idx].generation,
    };
    self.slots[idx].record = value;
    self.slots[idx].record.ops = ops;
    self.slots[idx].record.device_state = device_state;
    self.slots[idx].record.device = device;
    self.slots[idx].record.identity = identity;
    self.slots[idx].record.state =
        gpu::internal::CommandState.INACTIVE;
    return &self.slots[idx].record;
}

fn CommandRecord*? CommandTable.get(
    &self,
    gpu::internal::CommandListHandle handle,
) {
    gpu::internal::note_command_table_lookup();
    if (handle.owner != self.owner) return gpu::INVALID_HANDLE~;
    uint idx = handle.index;
    if (idx >= self.slots.len || !self.slots[idx].used) return gpu::INVALID_HANDLE~;
    if (self.slots[idx].generation != handle.generation) return gpu::INVALID_HANDLE~;
    return &self.slots[idx].record;
}

fn void? CommandTable.free(&self, gpu::internal::CommandListHandle handle) {
    self.get(handle)!;
    uint idx = handle.index;
    self.slots[idx].record = {};
    self.slots[idx].used = false;
    if (!should_recycle_handle_generation(self.slots[idx].generation)) {
        self.slots[idx].generation = HANDLE_GENERATION_MAX;
        self.slots[idx].next_free = COMMAND_FREE_NONE;
        return;
    }
    self.slots[idx].generation = next_handle_generation(self.slots[idx].generation)!;
    self.slots[idx].next_free = self.free_head;
    self.free_head = idx + 1;
}

fn void? CommandTable.cancel_claim(
    &self,
    gpu::internal::CommandListHandle handle,
) {
    return self.free(handle);
}

fn CommandRecord*? resolve_command(
    gpu::CommandList* token,
) @if(env::TESTING) {
    CommandRecord* record =
        gpu::internal::recording_command(
            token,
            count_command: false,
        )!;
    return record;
}

fn void? ensure_command_reference_capacity(
    CommandRecord* record,
    uint additional,
) {
    if (additional
            <= record.scratch.references.len
                - record.scratch.reference_count) return;
    return gpu::COMMAND_ALLOCATOR_CAPACITY_EXCEEDED~;
}

fn ulong command_reference_hash(CommandResourceReference* reference) {
    CommandReferenceKey key = {
        .owner      = reference.owner,
        .index      = reference.index,
        .generation = reference.generation,
    };
    return fnv64a::hash(((char*)&key)[:CommandReferenceKey::size]);
}

fn bool indexed_resource_reference_matches(
    VkDeviceState* state,
    CommandReferenceIndexCell* cell,
    CommandResourceReference* reference,
) {
    note_command_reference_equality_check(state);
    return cell.owner == reference.owner
        && cell.index == reference.index
        && cell.generation == reference.generation;
}

fn CommandReferenceProbe? probe_command_reference(
    VkDeviceState* state,
    CommandScratch* scratch,
    CommandResourceReference* reference,
) {
    CommandReferenceIndex* reference_index = &scratch.reference_index;
    if (reference_index.cells.len == 0) {
        return gpu::COMMAND_ALLOCATOR_CAPACITY_EXCEEDED~;
    }
    if (reference_index.epoch == 0) reference_index.epoch = 1;
    note_command_reference_lookup(state);
    uint cell_count = (uint)reference_index.cells.len;
    ulong hash = command_reference_hash(reference);
    uint cell_index = (uint)(
        (hash ^ (hash >> 32))
        & (ulong)(cell_count - 1)
    );
    for (uint probe = 0; probe < cell_count; probe++) {
        note_command_reference_probe(state);
        CommandReferenceIndexCell* cell = &reference_index.cells[cell_index];
        if (cell.epoch != reference_index.epoch) {
            return {
                .cell_index = cell_index,
                .found      = false,
            };
        }
        if (indexed_resource_reference_matches(state, cell, reference)) {
            return {
                .cell_index      = cell_index,
                .reference_index = cell.reference_index,
                .found           = true,
            };
        }
        cell_index = (cell_index + 1) & (cell_count - 1);
    }
    return gpu::COMMAND_ALLOCATOR_CAPACITY_EXCEEDED~;
}

fn void publish_command_reference(
    VkDeviceState* state,
    CommandScratch* scratch,
    CommandResourceReference* reference,
    CommandReferenceProbe* probe,
    uint reference_index,
) {
    scratch.reference_index.cells[probe.cell_index] = {
        .owner           = reference.owner,
        .index           = reference.index,
        .generation      = reference.generation,
        .reference_index = reference_index,
        .epoch           = scratch.reference_index.epoch,
    };
    note_command_reference_publication(state);
}

fn void reset_command_reference_index(
    VkDeviceState* state,
    CommandScratch* scratch,
) {
    CommandReferenceIndex* reference_index = &scratch.reference_index;
    if (reference_index.cells.len == 0) return;
    if (reference_index.epoch == uint::max) {
        foreach (&cell : reference_index.cells) cell.epoch = 0;
        note_command_reference_rollover_clear(state, reference_index.cells.len);
        reference_index.epoch = 1;
        return;
    }
    reference_index.epoch++;
    if (reference_index.epoch == 0) reference_index.epoch = 1;
}

fn void? preflight_command_references(
    VkDeviceState* state,
    CommandRecord* record,
    CommandResourceReference[] candidates,
) {
    uint additional = 0;
    foreach (i, &candidate : candidates) {
        CommandReferenceProbe probe = probe_command_reference(
            state,
            record.scratch,
            candidate,
        )!;
        bool duplicate = probe.found;
        if (duplicate) continue;
        for (uint j = 0; j < i; j++) {
            if (!same_resource_reference(&candidates[j], candidate)) continue;
            duplicate = true;
            break;
        }
        if (!duplicate) additional++;
    }
    ensure_command_reference_capacity(record, additional)!;
}

fn void? track_command_reference(
    VkDeviceState* state,
    CommandRecord* record,
    ulong owner,
    uint index,
    uint generation,
) {
    CommandResourceReference reference = {
        .owner      = owner,
        .index      = index,
        .generation = generation,
    };
    CommandScratch* scratch = record.scratch;
    CommandReferenceProbe probe = probe_command_reference(
        state,
        scratch,
        &reference,
    )!;
    if (probe.found) {
        note_command_reference_duplicate(state);
        return;
    }
    ensure_command_reference_capacity(record, 1)!;
    state.resource_mutex.lock()!!;
    note_command_reference_mutex_acquisition(state);
    defer state.resource_mutex.unlock();
    retain_tracked_command_reference(state, reference)!;
    note_command_reference_retain(state);
    uint reference_index = scratch.reference_count++;
    scratch.references[reference_index] = reference;
    publish_command_reference(
        state:           state,
        scratch:         scratch,
        reference:       &reference,
        probe:           &probe,
        reference_index: reference_index,
    );
}

fn TextureSlot*? resolve_tracked_texture_command_reference(
    VkDeviceState* state,
    CommandRecord* record,
    gpu::TextureHandle handle,
) {
    CommandResourceReference reference = {
        .owner      = handle.owner,
        .index      = handle.index,
        .generation = handle.generation,
    };
    CommandScratch* scratch = record.scratch;
    CommandReferenceProbe probe = probe_command_reference(
        state,
        scratch,
        &reference,
    )!;
    if (probe.found) {
        note_command_reference_duplicate(state);
        return state.textures.get(handle);
    }

    ensure_command_reference_capacity(record, 1)!;
    state.resource_mutex.lock()!!;
    note_command_reference_mutex_acquisition(state);
    defer state.resource_mutex.unlock();
    TextureSlot* slot = state.textures.get(handle)!;
    reference.tracked_command_references = &slot.tracked_command_references;
    reference.tracked_command_references.add(1, AtomicOrdering.RELAXED);
    note_command_reference_retain(state);
    uint reference_index = scratch.reference_count++;
    scratch.references[reference_index] = reference;
    publish_command_reference(
        state:           state,
        scratch:         scratch,
        reference:       &reference,
        probe:           &probe,
        reference_index: reference_index,
    );
    return slot;
}

fn void clear_command_scratch_references(
    VkDeviceState* state,
    CommandScratch* scratch,
) {
    release_tracked_command_references(
        state,
        scratch.references,
        scratch.reference_count,
    );
    scratch.reference_count = 0;
    reset_command_reference_index(state, scratch);
}

fn void clear_command_references(
    VkDeviceState* state,
    CommandRecord* record,
) {
    clear_command_scratch_references(state, record.scratch);
}

fn void rollback_command_references(
    VkDeviceState* state,
    CommandRecord* record,
    uint checkpoint,
) {
    if (record.scratch.reference_count == checkpoint) return;
    while (record.scratch.reference_count > checkpoint) {
        record.scratch.reference_count--;
        release_tracked_command_reference(
            state,
            record.scratch.references[record.scratch.reference_count],
        );
    }
    reset_command_reference_index(state, record.scratch);
    for (uint i = 0; i < record.scratch.reference_count; i++) {
        CommandResourceReference* reference = &record.scratch.references[i];
        CommandReferenceProbe probe = probe_command_reference(
            state,
            record.scratch,
            reference,
        )!!;
        publish_command_reference(
            state:           state,
            scratch:         record.scratch,
            reference:       reference,
            probe:           &probe,
            reference_index: i,
        );
        note_command_reference_rollback_rebuild_publication(state);
    }
}

fn void release_command(
    VkDeviceState* state,
    gpu::internal::CommandListHandle handle,
) {
    if (try record = state.commands.get(handle)) {
        (void)state.commands.free(handle);
    }
}

fn void destroy_command_table(VkDeviceState* state) {
    if (state.commands.slots.ptr == null) return;
    foreach (&cell : state.commands.slots[:state.commands.next_new]) {
        if (cell.used) {
            clear_command_references(state, &cell.record);
        }
    }
    mem::free(state.commands.slots.ptr);
    state.commands = {};
}
