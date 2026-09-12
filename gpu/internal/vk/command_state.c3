module gpu::internal::vk @private;

import gpu::internal @public;
import gpu @public;
import vk;
import std::core::mem;
import std::core::env;

alias CmdCopyBufferFn @private               = fn void? (CommandRecord*, gpu::BufferCopyDesc*);
alias CmdFillBufferFn @private               = fn void? (CommandRecord*, gpu::GpuSpan, uint);
alias CmdCopyBufferToTextureFn @private      = fn void? (CommandRecord*, gpu::BufferTextureCopyDesc*);
alias CmdCopyTextureToBufferFn @private      = fn void? (CommandRecord*, gpu::TextureBufferCopyDesc*);
alias CmdBarrierFn @private                  = fn void? (CommandRecord*, gpu::Barrier*);
alias CmdTextureBarrierFn @private           = fn void? (CommandRecord*, gpu::TextureBarrier*);
alias CmdBindPipelineFn @private             = fn void? (CommandRecord*, gpu::PipelineHandle);
alias CmdSetGraphicsStateFn @private         = fn void? (CommandRecord*, gpu::GraphicsState*);
alias CmdDispatchFn @private                 = fn void? (CommandRecord*, gpu::GpuAddress, gpu::Vec3u);
alias CmdTraceRaysFn @private = fn void? (
    CommandRecord*,
    gpu::GpuAddress,
    gpu::RayTracingShaderBindingTable*,
    gpu::Vec3u,
);
alias CmdTraceRaysIndirectFn @private = fn void? (
    CommandRecord*,
    gpu::GpuAddress,
    gpu::RayTracingShaderBindingTable*,
    gpu::GpuSpan,
);
alias CmdTraceRaysIndirect2Fn @private = fn void? (
    CommandRecord*,
    gpu::GpuAddress,
    gpu::GpuSpan,
);
alias CmdSetRayTracingPipelineStackSizeFn @private = fn void? (
    CommandRecord*,
    ulong,
);
alias CmdBeginRenderPassFn @private          = fn void? (CommandRecord*, gpu::RenderPassDesc*);
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
alias CmdDrawMeshTasksFn @private = fn void? (
    CommandRecord* record,
    gpu::GpuAddress mesh_root,
    gpu::GpuAddress fragment_root,
    gpu::Vec3u groups,
);
alias CmdDrawMeshTasksIndirectFn @private = fn void? (
    CommandRecord* record,
    gpu::GpuAddress mesh_root,
    gpu::GpuAddress fragment_root,
    gpu::GpuSpan args,
    uint draw_count,
);
alias CmdDrawMeshTasksIndirectCountFn @private = fn void? (
    CommandRecord* record,
    gpu::GpuAddress mesh_root,
    gpu::GpuAddress fragment_root,
    gpu::GpuSpan args,
    gpu::GpuSpan count_span,
    uint max_draw_count,
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
alias CmdResetTimestampsFn @private = fn void? (
    CommandRecord*,
    gpu::TimestampPoolHandle,
    uint,
    uint,
);
alias CmdWriteTimestampFn @private = fn void? (
    CommandRecord*,
    gpu::TimestampPoolHandle,
    uint,
    gpu::StageMask,
);
alias CmdResolveTimestampsFn @private = fn void? (
    CommandRecord*,
    gpu::TimestampPoolHandle,
    uint,
    uint,
    gpu::GpuSpan,
);
alias CmdBuildAccelerationStructureFn @private = fn void? (
    CommandRecord*,
    gpu::AccelerationStructureBuildDesc*,
);
alias CmdUpdateAccelerationStructureFn @private = fn void? (
    CommandRecord*,
    gpu::AccelerationStructureBuildDesc*,
);
alias CmdAccelerationStructureIndirectFn @private = fn void? (
    CommandRecord*,
    gpu::AccelerationStructureBuildDesc*,
    gpu::GpuSpan,
);
alias CmdCloneAccelerationStructureFn @private = fn void? (
    CommandRecord*,
    gpu::AccelerationStructureHandle,
    gpu::AccelerationStructureHandle,
);

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
    RenderCompatibility               render;
    bool                              dynamic_ray_tracing_stack_size;
    bool                              mesh_task_stage;
    uint                              push_block_size;
    uint                              inline_root_offset;
}

struct NativePipelineBinding {
    vk::Pipeline pipeline;
    uint         cache_entry;
}

struct DescriptorHeapCommandBinding {
    bool graphics_set_bound;
    vk::PipelineLayout graphics_layout;
    bool compute_set_bound;
    bool ray_tracing_set_bound;
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
    gpu::PrimitiveTopology           raster_topology;
    bool                             raster_topology_emitted;
    bool                             color_state_initialized;
    ColorCompatibility               color_state;
    BoundPipeline                    logical_pipeline;
    NativePipelineBinding            native_graphics_pipeline;
    NativePipelineBinding            native_compute_pipeline;
    NativePipelineBinding            native_ray_tracing_pipeline;
    DescriptorHeapCommandBinding     descriptor_heap_binding;
    bool                             ray_tracing_stack_size_set;
    CommandScratch*                  scratch;
    uint                             pending_acceleration_structure_head;
    char[]                           inline_root;
    bool                             internal;
    gpu::TextureHandle[]             init_textures;
}

struct CommandOps @public {
    CmdCopyBufferFn               copy_buffer;
    CmdFillBufferFn               fill_buffer;
    CmdCopyBufferToTextureFn      copy_buffer_to_texture;
    CmdCopyTextureToBufferFn      copy_texture_to_buffer;
    CmdBarrierFn                  barrier;
    CmdTextureBarrierFn           texture_barrier;
    CmdBindPipelineFn             bind_pipeline;
    CmdSetGraphicsStateFn         set_graphics_state;
    CmdDispatchFn                 dispatch;
    CmdTraceRaysFn                trace_rays;
    CmdTraceRaysIndirectFn        trace_rays_indirect;
    CmdTraceRaysIndirect2Fn       trace_rays_indirect2;
    CmdSetRayTracingPipelineStackSizeFn set_ray_tracing_pipeline_stack_size;
    CmdBeginRenderPassFn          begin_render_pass;
    CmdSetViewportFn              set_viewport;
    CmdSetScissorFn               set_scissor;
    CmdEndRenderPassFn            end_render_pass;
    CmdDrawFn                     draw;
    CmdDrawIndexedFn              draw_indexed;
    CmdDrawIndirectFn             draw_indirect;
    CmdDrawIndexedIndirectFn      draw_indexed_indirect;
    CmdDrawIndexedIndirectCountFn draw_indexed_indirect_count;
    CmdDrawMeshTasksFn            draw_mesh_tasks;
    CmdDrawMeshTasksIndirectFn    draw_mesh_tasks_indirect;
    CmdDrawMeshTasksIndirectCountFn draw_mesh_tasks_indirect_count;
    CmdDispatchIndirectFn         dispatch_indirect;
    CmdDrawGeneratedFn            draw_generated;
    CmdDrawIndexedGeneratedFn     draw_indexed_generated;
    CmdDispatchGeneratedFn        dispatch_generated;
    CmdBeginLabelFn               begin_label;
    CmdEndLabelFn                 end_label;
    CmdResetTimestampsFn          reset_timestamps;
    CmdWriteTimestampFn           write_timestamp;
    CmdResolveTimestampsFn        resolve_timestamps;
    CmdBuildAccelerationStructureFn  build_acceleration_structure;
    CmdUpdateAccelerationStructureFn update_acceleration_structure;
    CmdAccelerationStructureIndirectFn build_acceleration_structure_indirect;
    CmdAccelerationStructureIndirectFn update_acceleration_structure_indirect;
    CmdCloneAccelerationStructureFn  clone_acceleration_structure;
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
) @feat(TESTING) {
    CommandRecord* record =
        gpu::internal::recording_command(token)!;
    return record;
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
            abandon_pending_acceleration_structure_builds(
                &cell.record,
                true,
            );
        }
    }
    mem::free(state.commands.slots.ptr);
    state.commands = {};
}
