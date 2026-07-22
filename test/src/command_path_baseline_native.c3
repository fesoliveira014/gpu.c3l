module gpu::vk @private;

import gpu @public;
import vk;

struct CommandPathNativeContext {
    CommandRecord*    record;
    vk::PipelineLayout pipeline_layout;
    bool              descriptor_buffer;
    bool              generated_work;
}

struct CommandPathNativeCopy {
    vk::Buffer     src_buffer;
    vk::Buffer     dst_buffer;
    vk::DeviceSize src_offset;
    vk::DeviceSize dst_offset;
    vk::DeviceSize size;
}

fn CommandPathNativeContext? command_path_native_context(
    gpu::CommandList* commands,
) {
    CommandRecord* record = resolve_command(commands)!;
    VkDeviceState* state = (VkDeviceState*)gpu::device_backend_state_ptr(
        &commands.device,
    );
    return {
        .record            = record,
        .pipeline_layout   = record.bound_pipeline.layout,
        .descriptor_buffer = state.strict_heap.implementation
            == VkHeapImplementation.DESCRIPTOR_BUFFER,
        .generated_work    = state.generated_work.enabled,
    };
}

fn CommandPathNativeCopy? command_path_native_copy(
    gpu::Device* device,
    gpu::GpuSpan src,
    gpu::GpuSpan dst,
) {
    VkDeviceState* state = (VkDeviceState*)gpu::device_backend_state_ptr(
        device,
    );
    if (state == null) return gpu::INVALID_HANDLE~;
    ResolvedSpan resolved_src = resolve_span(state, &src)!;
    ResolvedSpan resolved_dst = resolve_span(state, &dst)!;
    if (resolved_src.size != resolved_dst.size) return gpu::INVALID_ARGUMENT~;
    return {
        .src_buffer = resolved_src.slot.buffer,
        .dst_buffer = resolved_dst.slot.buffer,
        .src_offset = (vk::DeviceSize)resolved_src.offset,
        .dst_offset = (vk::DeviceSize)resolved_dst.offset,
        .size       = (vk::DeviceSize)resolved_src.size,
    };
}

fn void command_path_direct_dispatch(
    CommandPathNativeContext* context,
    gpu::GpuAddress root,
    gpu::Vec3u groups,
    ulong* native_calls,
) {
    gpu::RootPush push = { .root_gpu = root };
    vk::cmd_push_constants(
        command_buffer: context.record.command_buffer,
        layout:         context.pipeline_layout,
        stage_flags:    vk::ShaderStageFlagBits.COMPUTE_BIT,
        offset:         0,
        size:           gpu::RootPush::size,
        p_values:       &push,
    );
    vk::cmd_dispatch(
        command_buffer: context.record.command_buffer,
        group_count_x:  groups.x,
        group_count_y:  groups.y,
        group_count_z:  groups.z,
    );
    *native_calls += 2;
}

fn void command_path_direct_draw(
    CommandPathNativeContext* context,
    gpu::GpuAddress vertex_root,
    gpu::GpuAddress fragment_root,
    uint vertex_count,
    uint instance_count,
    ulong* native_calls,
) {
    gpu::GraphicsRootPush push = {
        .vertex_root_gpu   = vertex_root,
        .fragment_root_gpu = fragment_root,
    };
    uint stages = (uint)vk::ShaderStageFlagBits.VERTEX_BIT
        | (uint)vk::ShaderStageFlagBits.FRAGMENT_BIT;
    vk::cmd_push_constants(
        command_buffer: context.record.command_buffer,
        layout:         context.pipeline_layout,
        stage_flags:    (vk::ShaderStageFlags)stages,
        offset:         0,
        size:           gpu::GraphicsRootPush::size,
        p_values:       &push,
    );
    vk::cmd_draw(
        command_buffer: context.record.command_buffer,
        vertex_count:   vertex_count,
        instance_count: instance_count,
        first_vertex:   0,
        first_instance: 0,
    );
    *native_calls += 2;
}

fn void command_path_direct_barrier(
    CommandPathNativeContext* context,
    gpu::Barrier* barrier,
    ulong* native_calls,
) {
    vk::MemoryBarrier2 native_barrier = barrier_to_vk(
        barrier,
        context.descriptor_buffer,
        context.generated_work,
    );
    vk::DependencyInfo dependency =
        vk::dependency_info().set_memory_barriers((&native_barrier)[:1]);
    vk::cmd_pipeline_barrier2(context.record.command_buffer, &dependency);
    (*native_calls)++;
}

fn void command_path_direct_viewport(
    CommandPathNativeContext* context,
    gpu::Viewport* viewport,
    ulong* native_calls,
) {
    vk::Viewport native = {
        .x         = viewport.x,
        .y         = viewport.y,
        .width     = viewport.width,
        .height    = viewport.height,
        .min_depth = viewport.min_depth,
        .max_depth = viewport.max_depth,
    };
    vk::cmd_set_viewport(
        command_buffer: context.record.command_buffer,
        first_viewport: 0,
        viewport_count: 1,
        p_viewports:    &native,
    );
    (*native_calls)++;
}

fn void command_path_direct_copy_buffer(
    CommandPathNativeContext* context,
    CommandPathNativeCopy* copy,
    ulong* native_calls,
) {
    vk::BufferCopy region = {
        .src_offset = copy.src_offset,
        .dst_offset = copy.dst_offset,
        .size       = copy.size,
    };
    vk::cmd_copy_buffer(
        command_buffer: context.record.command_buffer,
        src_buffer:     copy.src_buffer,
        dst_buffer:     copy.dst_buffer,
        region_count:   1,
        p_regions:      &region,
    );
    (*native_calls)++;
}
