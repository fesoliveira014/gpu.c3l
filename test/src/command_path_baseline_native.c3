module gpu::internal::vk @private;

import gpu::internal @public;
import gpu @public;
import vk;

struct CommandPathNativeContext {
    CommandRecord*     record;
    vk::PipelineLayout pipeline_layout;
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
    CommandRecord* record = gpu::internal::recording_command(
        commands,
        count_command: false,
    )!;
    return {
        .record          = record,
        .pipeline_layout = record.logical_pipeline.layout,
    };
}

fn CommandPathNativeCopy? command_path_native_copy(
    gpu::Device* device,
    gpu::GpuSpan src,
    gpu::GpuSpan dst,
) {
    VkDeviceState* state =
        gpu::internal::device_slots[device.index].data.state;
    if (state == null) return gpu::INVALID_HANDLE~;
    ResolvedSpan resolved_src = resolve_span(state, &src)!;
    ResolvedSpan resolved_dst = resolve_span(state, &dst)!;
    if (resolved_src.size != resolved_dst.size) {
        return gpu::INVALID_ARGUMENT~;
    }
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
}

fn void command_path_direct_copy_buffer(
    CommandPathNativeContext* context,
    CommandPathNativeCopy* copy,
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
}
