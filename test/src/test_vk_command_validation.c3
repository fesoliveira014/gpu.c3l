module gpu_test;

import gpu::internal @public;
import gpu @public;
import gpu::internal::vk @public;
import vk;
import std::atomic::types;
import std::core::mem;
import std::io;

const char[*] VAL_VERT_SPIRV = $embed("shaders/offscreen.vert.spv");
const char[*] VAL_FRAG_SPIRV = $embed("shaders/offscreen.frag.spv");
const char[*] VAL_COMP_SPIRV = $embed("shaders/root_pointer.comp.spv");

const uint VAL_RT = 16;
const vk::QueueFamilyFlags TRANSFER_ONLY_QUEUE_FLAGS =
    (vk::QueueFamilyFlags)(uint)vk::QueueFlagBits.TRANSFER_BIT;

struct BarrierEmitCapture {
    uint calls;
    uint memory_barriers;
    uint image_barriers;
    vk::MemoryBarrier2 memory_barrier;
}

fn void capture_barrier_emit(
    void* user_data,
    vk::CommandBuffer command_buffer,
    vk::DependencyInfo* dependency,
) {
    (void)command_buffer;
    BarrierEmitCapture* capture = (BarrierEmitCapture*)user_data;
    capture.calls++;
    capture.memory_barriers = dependency.memory_barrier_count;
    capture.image_barriers = dependency.image_memory_barrier_count;
    if (dependency.memory_barrier_count == 1) {
        capture.memory_barrier = dependency.p_memory_barriers[0];
    }
}

fn gpu::Device make_val_device() {
    gpu::RuntimeDesc desc = {
        .enable_vulkan_validation = true,
        .application_name  = "gpu_command_validation_test",
    };
    return create_test_device(&desc)!!;
}

fn void texture_copies_use_transfer_optimal_layouts() @test {
    vk::TextureSlot texture = {
        .format = gpu::Format.RGBA8_UNORM,
    };

    assert(vk::texture_copy_layout(&texture, true)
        == vk::ImageLayout.TRANSFER_DST_OPTIMAL);
    assert(vk::texture_copy_layout(&texture, false)
        == vk::ImageLayout.TRANSFER_SRC_OPTIMAL);
}

fn void global_barrier_emits_no_texture_barrier() @test {
    gpu::Device device = make_val_device();
    defer destroy_test_device(&device)!!;
    gpu::Queue queue = gpu::get_queue(
        &device,
        gpu::QueueKind.COMPUTE,
    )!!;
    gpu::CommandAllocator allocator = gpu::create_command_allocator(
        &device,
        queue,
    )!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::CommandList commands = gpu::begin_commands(&allocator)!!;
    defer discard_test_commands(&commands)!!;

    BarrierEmitCapture capture;
    vk::BarrierEmitOps ops = {
        .user_data = &capture,
        .emit      = &capture_barrier_emit,
    };
    gpu::Barrier dependency = {
        .before = { .compute },
        .after  = { .compute },
    };
    vk::vk_cmd_barrier_with_ops(
        vk::resolve_command(&commands)!!,
        &dependency,
        &ops,
    )!!;

    assert(
        capture.calls == 1
            && capture.memory_barriers == 1
            && capture.image_barriers == 0,
        "the submitted dependency contains one memory barrier and no image barrier"
    );
}

fn void acceleration_structure_barrier_emits_exact_native_dependency() @test {
    gpu::Device device = make_val_device();
    defer destroy_test_device(&device)!!;
    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.COMPUTE)!!;
    gpu::CommandAllocator allocator =
        gpu::create_command_allocator(&device, queue)!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::CommandList commands = gpu::begin_commands(&allocator)!!;
    defer discard_test_commands(&commands)!!;
    vk::CommandRecord* record = vk::resolve_command(&commands)!!;
    bool previous_acceleration_structures = record.device_state.acceleration_structures_enabled;
    defer record.device_state.acceleration_structures_enabled = previous_acceleration_structures;
    bool previous_ray_queries = record.device_state.ray_queries_enabled;
    defer record.device_state.ray_queries_enabled = previous_ray_queries;

    BarrierEmitCapture capture;
    vk::BarrierEmitOps ops = {
        .user_data = &capture,
        .emit = &capture_barrier_emit,
    };
    gpu::Barrier dependency = {
        .before = { .acceleration_structure_build },
        .after = { .compute },
    };

    record.device_state.acceleration_structures_enabled = true;
    record.device_state.ray_queries_enabled = true;
    vk::vk_cmd_barrier_with_ops(record, &dependency, &ops)!!;

    ulong source_access =
        (ulong)vk::AccessFlagBits2
            .ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR
        | (ulong)vk::AccessFlagBits2
            .ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    ulong destination_access =
        (ulong)vk::AccessFlagBits2.ACCESS_2_SHADER_READ_BIT
        | (ulong)vk::AccessFlagBits2.ACCESS_2_SHADER_WRITE_BIT
        | (ulong)vk::AccessFlagBits2
            .ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    assert(capture.calls == 1 && capture.memory_barriers == 1
            && capture.image_barriers == 0
            && capture.memory_barrier.src_stage_mask
                == vk::PipelineStageFlagBits2
                    .STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR
            && (ulong)capture.memory_barrier.src_access_mask == source_access
            && capture.memory_barrier.dst_stage_mask
                == vk::PipelineStageFlagBits2.STAGE_2_COMPUTE_SHADER_BIT
            && (ulong)capture.memory_barrier.dst_access_mask
                == destination_access,
        "build-to-query records one exact caller-requested native barrier");
}

fn void pipeline_only_compute_to_indirect_barrier_omits_ray_query_access() @test {
    gpu::Device device = make_val_device();
    defer destroy_test_device(&device)!!;
    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.COMPUTE)!!;
    gpu::CommandAllocator allocator = gpu::create_command_allocator(
        &device,
        queue,
    )!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::CommandList commands = gpu::begin_commands(&allocator)!!;
    defer discard_test_commands(&commands)!!;
    vk::CommandRecord* record = vk::resolve_command(&commands)!!;
    record.device_state.acceleration_structures_enabled = true;
    record.device_state.ray_tracing_pipelines_enabled = true;
    record.device_state.ray_queries_enabled = false;

    BarrierEmitCapture capture = {};
    vk::BarrierEmitOps ops = {
        .user_data = &capture,
        .emit      = &capture_barrier_emit,
    };
    gpu::Barrier dependency = {
        .before = { .compute },
        .after  = { .indirect },
    };
    vk::vk_cmd_barrier_with_ops(record, &dependency, &ops)!!;

    ulong acceleration_read = (ulong)vk::AccessFlagBits2
        .ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    assert(capture.calls == 1,
        "pipeline-only compute production emits one barrier");
    assert(((ulong)capture.memory_barrier.src_access_mask
            & acceleration_read) == 0,
        "pipeline-only compute production omits ray-query access");
    assert(((ulong)capture.memory_barrier.dst_access_mask
            & (ulong)vk::AccessFlagBits2
                .ACCESS_2_INDIRECT_COMMAND_READ_BIT) != 0,
        "indirect consumption includes indirect-command read access");
}

fn void compute_to_index_input_barrier_lowers_index_read() @test {
    gpu::Device device = make_val_device();
    defer destroy_test_device(&device)!!;
    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator allocator = gpu::create_command_allocator(&device, queue)!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::CommandList commands = gpu::begin_commands(&allocator)!!;
    defer discard_test_commands(&commands)!!;
    vk::CommandRecord* record = vk::resolve_command(&commands)!!;

    BarrierEmitCapture capture = {};
    vk::BarrierEmitOps ops = { .user_data = &capture, .emit = &capture_barrier_emit };
    gpu::Barrier dependency = { .before = { .compute }, .after = { .index_input } };
    vk::vk_cmd_barrier_with_ops(record, &dependency, &ops)!!;

    assert(capture.calls == 1, "compute to index input emits one barrier");
    assert((ulong)capture.memory_barrier.src_stage_mask
            == (ulong)vk::PipelineStageFlagBits2.STAGE_2_COMPUTE_SHADER_BIT,
        "source stage is compute shader");
    assert((ulong)capture.memory_barrier.src_access_mask
            == (ulong)vk::AccessFlagBits2.ACCESS_2_SHADER_WRITE_BIT,
        "source access is shader write");
    assert((ulong)capture.memory_barrier.dst_stage_mask
            == (ulong)vk::PipelineStageFlagBits2.STAGE_2_INDEX_INPUT_BIT,
        "destination stage is index input");
    assert((ulong)capture.memory_barrier.dst_access_mask
            == (ulong)vk::AccessFlagBits2.ACCESS_2_INDEX_READ_BIT,
        "destination access is index read");
}

fn void pipeline_only_build_to_trace_barrier_keeps_acceleration_read() @test {
    gpu::Device device = make_val_device();
    defer destroy_test_device(&device)!!;
    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.COMPUTE)!!;
    gpu::CommandAllocator allocator = gpu::create_command_allocator(
        &device,
        queue,
    )!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::CommandList commands = gpu::begin_commands(&allocator)!!;
    defer discard_test_commands(&commands)!!;
    vk::CommandRecord* record = vk::resolve_command(&commands)!!;
    record.device_state.acceleration_structures_enabled = true;
    record.device_state.ray_tracing_pipelines_enabled = true;
    record.device_state.ray_queries_enabled = false;

    BarrierEmitCapture capture = {};
    vk::BarrierEmitOps ops = {
        .user_data = &capture,
        .emit      = &capture_barrier_emit,
    };
    gpu::Barrier dependency = {
        .before = { .acceleration_structure_build },
        .after  = { .ray_tracing },
    };
    vk::vk_cmd_barrier_with_ops(record, &dependency, &ops)!!;

    ulong acceleration_read = (ulong)vk::AccessFlagBits2
        .ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    assert(capture.calls == 1,
        "pipeline-only build-to-trace emits one barrier");
    assert(((ulong)capture.memory_barrier.dst_access_mask
            & acceleration_read) != 0,
        "pipeline-only ray tracing retains acceleration-structure read access");
}

<* Textured draw context, same shape as the indirect-draw tests: offscreen
   vertex/fragment code, one graphics pipeline, and a 1x1 white pattern feeding the
   fragment heap path. *>
struct ValDrawCtx {
    gpu::PipelineHandle graphics;
    gpu::TextureHandle  pattern;
    gpu::TextureView    tex_view;
    gpu::TextureIndex   tex_index;
    gpu::SamplerIndex   smp_index;
}

fn ValDrawCtx make_val_draw_ctx(gpu::Device* device) {
    ValDrawCtx ctx;
    gpu::ShaderDesc vert_sd = { .spirv = VAL_VERT_SPIRV[..], .entry_point = "main" };
    gpu::ShaderDesc vert = vert_sd;
    gpu::ShaderDesc frag_sd = { .spirv = VAL_FRAG_SPIRV[..], .entry_point = "main" };
    gpu::ShaderDesc frag = frag_sd;

    gpu::Format[1] pipeline_color_formats = {
        gpu::Format.RGBA8_UNORM,
    };
    gpu::GraphicsPipelineDesc pipe_desc = {
        .vertex_shader   = vert,
        .fragment_shader = frag,
        .color_formats   = pipeline_color_formats[..],
    };
    ctx.graphics = gpu::create_graphics_pipeline(device, &pipe_desc)!!;

    gpu::TextureDesc tex_desc = {
        .access    = { .graphics, .compute, .transfer },
        .width     = 1,
        .height    = 1,
        .format    = gpu::Format.RGBA8_UNORM,
        .usage     = { .sampled, .transfer_dst },
    };
    ctx.pattern = gpu::create_texture(device, &tex_desc)!!;
    ctx.tex_view = gpu::create_texture_view(device, ctx.pattern, null)!!;
    ctx.tex_index = ctx.tex_view.index;
    gpu::SamplerDesc sampler_desc = {
        .min_filter = gpu::Filter.NEAREST,
        .mag_filter = gpu::Filter.NEAREST,
        .address_u  = gpu::AddressMode.CLAMP_TO_EDGE,
        .address_v  = gpu::AddressMode.CLAMP_TO_EDGE,
        .address_w  = gpu::AddressMode.CLAMP_TO_EDGE,
    };
    ctx.smp_index = gpu::intern_sampler(device, &sampler_desc)!!;

    char[4] white = { (char)255, (char)255, (char)255, (char)255 };
    upload_test_texture_data(
        device:   device,
        dst:      ctx.pattern,
        mip:      0,
        data:     white[..],
        next_use: gpu::sampled_at({ .fragment_shader }),
    )!!;
    return ctx;
}

fn void destroy_val_draw_ctx(gpu::Device* device, ValDrawCtx* ctx) {
    gpu::destroy_texture_view(device, ctx.tex_view)!!;
    gpu::destroy_texture(device, ctx.pattern)!!;
    gpu::destroy_pipeline(device, ctx.graphics)!!;
}

fn gpu::TextureHandle make_val_target(gpu::Device* device) {
    gpu::TextureDesc desc = {
        .access    = { .graphics, .compute, .transfer },
        .width     = VAL_RT,
        .height    = VAL_RT,
        .format    = gpu::Format.RGBA8_UNORM,
        .usage     = { .color_attach, .transfer_src },
    };
    return gpu::create_texture(device, &desc)!!;
}

fn gpu::TextureHandle make_val_color_target(gpu::Device* device, uint width, uint height, uint mip_levels) {
    gpu::TextureDesc desc = {
        .access    = { .graphics, .compute, .transfer },
        .width      = width,
        .height     = height,
        .mip_levels = mip_levels,
        .format     = gpu::Format.RGBA8_UNORM,
        .usage      = { .color_attach, .transfer_src },
    };
    return gpu::create_texture(device, &desc)!!;
}

fn gpu::TextureHandle make_val_depth_target(gpu::Device* device, uint width, uint height) {
    gpu::TextureDesc desc = {
        .access    = { .graphics, .compute, .transfer },
        .width     = width,
        .height    = height,
        .format    = gpu::Format.D32_FLOAT,
        .usage     = { .depth_attach },
    };
    return gpu::create_texture(device, &desc)!!;
}

fn gpu::TextureHandle make_val_depth_target_mips(gpu::Device* device, uint width, uint height, uint mip_levels) {
    gpu::TextureDesc desc = {
        .access    = { .graphics, .compute, .transfer },
        .width      = width,
        .height     = height,
        .mip_levels = mip_levels,
        .format     = gpu::Format.D32_FLOAT,
        .usage      = { .depth_attach },
    };
    return gpu::create_texture(device, &desc)!!;
}

fn void record_val_depth_to_attach(gpu::CommandList* cmd, gpu::TextureHandle depth) {
    record_val_texture_transition(
        commands: cmd,
        texture:  depth,
        after:    {
            .layout = gpu::TextureLayout.DEPTH_ATTACHMENT,
            .stages = { .depth_output },
            .access = { .read, .write },
        },
    );
}

// Never actually dispatched, so the shader's ABI content is unused.
struct ValComputeCtx {
    gpu::PipelineHandle pipeline;
}

fn ValComputeCtx make_val_compute_ctx(gpu::Device* device) {
    ValComputeCtx ctx;
    gpu::ShaderDesc comp_sd = { .spirv = VAL_COMP_SPIRV[..], .entry_point = "main" };
    gpu::ShaderDesc shader = comp_sd;
    gpu::ComputePipelineDesc comp_desc = { .shader = shader };
    ctx.pipeline = gpu::create_compute_pipeline(device, &comp_desc)!!;
    return ctx;
}

fn void destroy_val_compute_ctx(gpu::Device* device, ValComputeCtx* ctx) {
    gpu::destroy_pipeline(device, ctx.pipeline)!!;
}

fn void record_val_texture_transition(
    gpu::CommandList* commands,
    gpu::TextureHandle texture,
    gpu::TextureState after,
) {
    gpu::TextureBarrier barrier = gpu::texture_transition(
        texture: texture,
        before:  { .layout = gpu::TextureLayout.UNDEFINED },
        after:   after,
    )!!;
    gpu::cmd_texture_barrier(commands, &barrier)!!;
}

fn gpu::GpuAddress val_span_address(gpu::Device* device, gpu::GpuSpan span) {
    return gpu::get_span_address(device, span)!!;
}

fn void write_val_triangle_verts(gpu::Device* device, gpu::GpuSpan* span) {
    char[] mapping = gpu::get_span_mapping(device, *span)!!;
    float* verts = (float*)mapping.ptr;
    float[12] vertex_data = {
        -0.8f, -0.8f, 0.1f, 0.1f,
         0.8f, -0.8f, 0.9f, 0.1f,
         0.0f,  0.8f, 0.5f, 0.9f,
    };
    for (uint i = 0; i < 12; i++) verts[i] = vertex_data[i];
}

fn void record_val_target_to_attach(gpu::CommandList* cmd, gpu::TextureHandle target) {
    gpu::TextureBarrier to_attach = {
        .texture = target,
        .before  = { .layout = gpu::TextureLayout.UNDEFINED },
        .after   = {
            .layout = gpu::TextureLayout.COLOR_ATTACHMENT,
            .stages = { .color_output },
            .access = { .read, .write },
        },
    };
    gpu::cmd_texture_barrier(cmd, &to_attach)!!;
}

fn void submit_requires_exact_recording_queue_token() @test {
    gpu::Device device = make_val_device();
    defer destroy_test_device(&device)!!;
    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator allocator = gpu::create_command_allocator(&device, queue)!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::CommandList commands = gpu::begin_commands(&allocator)!!;
    defer discard_test_commands(&commands)!!;
    gpu::ExecutableCommandList executable = gpu::end_commands(&commands)!!;
    defer discard_test_executable(&executable)!!;

    vk::VkDeviceState* state =
        gpu::internal::device_slots[device.index].data.state;
    vk::CommandRecord* record = executable.record;
    gpu::Queue recorded_queue = record.queue;
    record.queue.id++;

    gpu::ExecutableCommandList[1] lists = { executable };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    fault mismatched = @catch(gpu::submit(queue, &submit));
    assert(mismatched == gpu::INVALID_ARGUMENT,
        "submission requires the exact queue token captured while recording");
    assert(lists[0].is_valid(),
        "queue-token rejection preserves the executable token");

    record.queue = recorded_queue;
    gpu::CompletionPoint point = gpu::submit(queue, &submit)!!;
    gpu::wait_completion(point, 1_000_000_000)!!;
}

fn bool val_distinct_graphics_compute_families(gpu::Device* device) {
    vk::VkDeviceState* state = gpu::internal::device_slots[device.index].data.state;
    if (state.graphics_family != state.compute_family) return true;
    io::eprintfn("access-domain rejection not applicable: graphics and compute share family %d", state.graphics_family);
    return false;
}

fn void bound_pipeline_borrow_uses_command_snapshot_storage() @test {
    gpu::Device device = make_val_device();
    defer destroy_test_device(&device)!!;
    ValComputeCtx compute = make_val_compute_ctx(&device);
    defer destroy_val_compute_ctx(&device, &compute);
    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.COMPUTE)!!;
    gpu::CommandAllocator allocator =
        gpu::create_command_allocator(&device, queue)!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::CommandList commands = gpu::begin_commands(&allocator)!!;
    defer discard_test_commands(&commands)!!;
    vk::CommandRecord* record = vk::resolve_command(&commands)!!;

    assert(@catch(vk::borrow_bound_pipeline(record))
            == gpu::INVALID_RESOURCE_STATE,
        "borrowing with no pipeline bound reports INVALID_RESOURCE_STATE");

    gpu::cmd_bind_pipeline(&commands, compute.pipeline)!!;
    vk::BoundPipeline* borrowed = vk::borrow_bound_pipeline(record)!!;
    assert(borrowed == &record.logical_pipeline,
        "execution borrows the authoritative command snapshot");

}

fn void explicit_pipeline_binding_state_is_transactional() @test {
    gpu::Device device = make_val_device();
    defer destroy_test_device(&device)!!;
    ValComputeCtx compute = make_val_compute_ctx(&device);
    defer destroy_val_compute_ctx(&device, &compute);
    ValComputeCtx compute_alias = make_val_compute_ctx(&device);
    defer destroy_val_compute_ctx(&device, &compute_alias);
    ValDrawCtx draw = make_val_draw_ctx(&device);
    defer destroy_val_draw_ctx(&device, &draw);
    gpu::TextureHandle target = make_val_target(&device);
    defer gpu::destroy_texture(&device, target)!!;
    gpu::AttachmentViewHandle target_view = create_test_attachment_view(
        &device,
        target,
    )!!;
    defer destroy_test_attachment_view(&device, &target_view)!!;
    OwnedSpan root_owned = allocate_owned_span(
        device:       &device,
        size:         16,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       { .graphics, .compute },
        debug_name:   "explicit_pipeline_root",
    )!!;
    defer free_owned_span(&device, &root_owned)!!;
    gpu::GpuAddress root = val_span_address(&device, root_owned.span);

    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator allocator = gpu::create_command_allocator(&device, queue)!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::CommandList commands = gpu::begin_commands(&allocator)!!;
    defer discard_test_commands(&commands)!!;

    fault missing_pipeline = @catch(gpu::cmd_dispatch(
        commands: &commands,
        root:     root,
        groups:   { 1, 1, 1 },
    ));
    assert(missing_pipeline == gpu::INVALID_RESOURCE_STATE,
        "dispatch without an active pipeline faults command state");

    gpu::cmd_bind_pipeline(&commands, compute.pipeline)!!;
    gpu::cmd_bind_pipeline(&commands, compute_alias.pipeline)!!;

    vk::CommandRecord* record = vk::resolve_command(&commands)!!;
    gpu::SamplerDesc sampler_desc = {};
    (void)gpu::intern_sampler(&device, &sampler_desc)!!;
    gpu::cmd_bind_pipeline(&commands, compute_alias.pipeline)!!;

    gpu::Barrier global_barrier = {
        .before = { .compute = true },
        .after  = { .compute = true },
    };
    gpu::cmd_barrier(&commands, &global_barrier)!!;
    gpu::cmd_bind_pipeline(&commands, compute_alias.pipeline)!!;

    gpu::cmd_dispatch(
        commands: &commands,
        root:     (gpu::GpuAddress)0,
        groups:   { 0, 0, 0 },
    )!!;
    gpu::cmd_dispatch(
        commands: &commands,
        root:     root,
        groups:   { 0, 0, 0 },
    )!!;

    ValComputeCtx stale = make_val_compute_ctx(&device);
    gpu::PipelineHandle stale_handle = stale.pipeline;
    gpu::destroy_pipeline(&device, stale_handle)!!;
    vk::DescriptorHeapCommandBinding heap_binding_before =
        record.descriptor_heap_binding;
    fault stale_bind = @catch(gpu::cmd_bind_pipeline(&commands, stale_handle));
    assert(stale_bind == gpu::INVALID_HANDLE, "binding a stale pipeline faults INVALID_HANDLE");
    assert(record.descriptor_heap_binding.graphics_set_bound
            == heap_binding_before.graphics_set_bound
        && record.descriptor_heap_binding.compute_set_bound
            == heap_binding_before.compute_set_bound,
        "failed pipeline validation preserves descriptor heap state");
    gpu::cmd_dispatch(
        commands: &commands,
        root:     root,
        groups:   { 0, 0, 0 },
    )!!;

    record_val_target_to_attach(&commands, target);
    gpu::ColorTargetDesc[1] colors = {{
        .view     = target_view,
        .load_op  = gpu::LoadOp.CLEAR,
        .store_op = gpu::StoreOp.STORE,
    }};
    gpu::RenderPassDesc pass = {
        .colors = colors[..],
        .width  = VAL_RT,
        .height = VAL_RT,
    };
    gpu::GraphicsState graphics_state =
        gpu::render_geometry_state(pass.width, pass.height)!!;
    gpu::ColorTargetState[1] color_targets = {
        gpu::color_blend_disabled(),
    };
    graphics_state.color.targets = color_targets[..];
    gpu::cmd_bind_pipeline(&commands, draw.graphics)!!;
    gpu::cmd_begin_render_pass(&commands, &pass)!!;
    gpu::cmd_set_graphics_state(&commands, &graphics_state)!!;

    gpu::cmd_draw(
        commands:      &commands,
        vertex_root:   root,
        fragment_root: root,
        vertex_count:  0,
        instance_count: 0,
    )!!;
    gpu::cmd_draw(
        commands:      &commands,
        vertex_root:   (gpu::GpuAddress)0,
        fragment_root: root,
        vertex_count:  0,
        instance_count: 0,
    )!!;
    gpu::cmd_draw(
        commands:      &commands,
        vertex_root:   root,
        fragment_root: (gpu::GpuAddress)0,
        vertex_count:  0,
        instance_count: 0,
    )!!;

    gpu::cmd_draw(
        commands:      &commands,
        vertex_root:   root,
        fragment_root: root,
        vertex_count:  0,
        instance_count: 0,
    )!!;

    gpu::cmd_end_render_pass(&commands)!!;

    gpu::cmd_begin_render_pass(&commands, &pass)!!;
    gpu::cmd_bind_pipeline(&commands, draw.graphics)!!;
    gpu::cmd_set_graphics_state(&commands, &graphics_state)!!;
    gpu::cmd_draw(
        commands:      &commands,
        vertex_root:   root,
        fragment_root: root,
        vertex_count:  0,
        instance_count: 0,
    )!!;
    gpu::cmd_end_render_pass(&commands)!!;
}

fn void dispatch_uses_bind_time_pipeline_snapshot() @test {
    gpu::Device device = make_val_device();
    defer destroy_test_device(&device)!!;
    ValComputeCtx compute = make_val_compute_ctx(&device);
    defer destroy_val_compute_ctx(&device, &compute);
    ValDrawCtx draw = make_val_draw_ctx(&device);
    defer destroy_val_draw_ctx(&device, &draw);
    OwnedSpan root = allocate_owned_span(
        device:       &device,
        size:         16,
        memory_class: gpu::MemoryClass.GPU_PRIVATE,
        access:       { .compute },
        debug_name:   "bind_time_pipeline_root",
    )!!;
    defer free_owned_span(&device, &root)!!;
    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.COMPUTE)!!;
    gpu::CommandAllocator allocator = gpu::create_command_allocator(
        &device,
        queue,
    )!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::CommandList commands = gpu::begin_commands(&allocator)!!;
    defer discard_test_commands(&commands)!!;
    gpu::cmd_bind_pipeline(&commands, compute.pipeline)!!;

    vk::VkDeviceState* state =
        gpu::internal::device_slots[device.index].data.state;
    vk::PipelineCell* bound_cell =
        &state.pipelines.slots[compute.pipeline.index];
    vk::PipelineCell* alternate_cell =
        &state.pipelines.slots[draw.graphics.index];
    vk::Pipeline saved_pipeline = bound_cell.value.pipeline;
    vk::PipelineLayout saved_layout = bound_cell.value.layout;
    vk::IndirectCommandsLayoutEXT saved_generated_layout =
        bound_cell.value.generated_dispatch_layout;
    uint saved_entry = bound_cell.value.entry;
    uint saved_generation = bound_cell.generation;
    bool saved_used = bound_cell.used;
    assert(saved_layout != alternate_cell.value.layout,
        "the alternate pipeline provides a distinct backing layout");
    vk::CommandRecord* record = vk::resolve_command(&commands)!!;
    assert(record.logical_pipeline.layout == saved_layout,
        "binding snapshots the selected pipeline layout");

    {
        defer {
            bound_cell.value.pipeline = saved_pipeline;
            bound_cell.value.layout = saved_layout;
            bound_cell.value.generated_dispatch_layout =
                saved_generated_layout;
            bound_cell.value.entry = saved_entry;
            bound_cell.generation = saved_generation;
            bound_cell.used = saved_used;
        };
        bound_cell.value.pipeline = alternate_cell.value.pipeline;
        bound_cell.value.layout = alternate_cell.value.layout;
        bound_cell.value.generated_dispatch_layout =
            alternate_cell.value.generated_dispatch_layout;
        bound_cell.value.entry = alternate_cell.value.entry;
        bound_cell.generation = compute.pipeline.generation + 1;
        bound_cell.used = false;
        gpu::cmd_dispatch(
            commands: &commands,
            root:     val_span_address(&device, root.span),
            groups:   { 1, 1, 1 },
        )!!;
        assert(record.logical_pipeline.layout == saved_layout,
            "dispatch consumes the bind-time pipeline layout snapshot");
    }
}

fn void compute_pipeline_from_vertex_shader_faults() @test {
    gpu::Device device = make_val_device();
    defer destroy_test_device(&device)!!;

    gpu::ShaderDesc vert_sd = { .spirv = VAL_VERT_SPIRV[..], .entry_point = "main" };
    gpu::ShaderDesc vert_shader = vert_sd;

    gpu::ComputePipelineDesc desc = { .shader = vert_shader };
    fault wrong_role = @catch(gpu::create_compute_pipeline(&device, &desc));
    assert(wrong_role == gpu::SHADER_INVALID,
        "compute pipeline from a vertex shader faults SHADER_INVALID");
}

fn void graphics_pipeline_from_swapped_shader_stages_faults() @test {
    gpu::Device device = make_val_device();
    defer destroy_test_device(&device)!!;

    gpu::ShaderDesc vert_sd = { .spirv = VAL_VERT_SPIRV[..], .entry_point = "main" };
    gpu::ShaderDesc vert_shader = vert_sd;
    gpu::ShaderDesc frag_sd = { .spirv = VAL_FRAG_SPIRV[..], .entry_point = "main" };
    gpu::ShaderDesc frag_shader = frag_sd;

    gpu::Format[1] pipeline_color_formats = {
        gpu::Format.RGBA8_UNORM,
    };
    gpu::GraphicsPipelineDesc swapped = {
        .vertex_shader   = frag_shader,
        .fragment_shader = vert_shader,
        .color_formats   = pipeline_color_formats[..],
    };
    fault wrong_role = @catch(gpu::create_graphics_pipeline(&device, &swapped));
    assert(wrong_role == gpu::SHADER_INVALID,
        "graphics pipeline with swapped vertex and fragment shaders faults SHADER_INVALID");
}

fn void valid_copy_fill_and_draw_succeed() @test {
    gpu::Device device = make_val_device();
    defer destroy_test_device(&device)!!;
    ValDrawCtx ctx = make_val_draw_ctx(&device);
    defer destroy_val_draw_ctx(&device, &ctx);
    gpu::TextureHandle target = make_val_target(&device);
    defer gpu::destroy_texture(&device, target)!!;
    gpu::AttachmentViewHandle target_view = create_test_attachment_view(
        &device,
        target,
    )!!;
    defer destroy_test_attachment_view(&device, &target_view)!!;

    OwnedSpan upload = allocate_owned_span(
        device:       &device,
        size:         64,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "valid_copy_src",
    )!!;
    defer free_owned_span(&device, &upload)!!;
    OwnedSpan readback = allocate_owned_span(
        device:       &device,
        size:         64,
        memory_class: gpu::MemoryClass.CPU_READ,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "valid_copy_dst",
    )!!;
    defer free_owned_span(&device, &readback)!!;
    OwnedSpan fill_target = allocate_owned_span(
        device:       &device,
        size:         64,
        memory_class: gpu::MemoryClass.GPU_PRIVATE,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "valid_fill_dst",
    )!!;
    defer free_owned_span(&device, &fill_target)!!;
    OwnedSpan verts_owned = allocate_owned_span(
        device:       &device,
        size:         3 * 16,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       { .graphics },
        debug_name:   "valid_draw_vertices",
    )!!;
    defer free_owned_span(&device, &verts_owned)!!;
    gpu::GpuSpan verts_span = verts_owned.span;
    write_val_triangle_verts(&device, &verts_span);
    gpu::flush_mapped_span(&device, verts_span)!!;
    OwnedSpan frag_owned = allocate_owned_span(
        device:       &device,
        size:         FragRoot::size,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       { .graphics },
        debug_name:   "valid_draw_fragment_root",
    )!!;
    defer free_owned_span(&device, &frag_owned)!!;
    gpu::GpuSpan frag_span = frag_owned.span;
    char[] frag_mapping = gpu::get_span_mapping(&device, frag_span)!!;
    FragRoot* frag_root = (FragRoot*)frag_mapping.ptr;
    frag_root.texture_index = ctx.tex_index;
    frag_root.sampler_index = ctx.smp_index;
    gpu::flush_mapped_span(&device, frag_span)!!;

    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator allocator = gpu::create_command_allocator(&device, queue)!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::CommandList cmd = gpu::begin_commands(&allocator)!!;
    defer discard_test_commands(&cmd)!!;
    gpu::BufferCopyDesc copy = {
        .src = upload.span,
        .dst = readback.span,
    };
    gpu::cmd_copy_buffer(&cmd, &copy)!!;
    gpu::cmd_fill_buffer(&cmd, fill_target.span, 0)!!;

    record_val_target_to_attach(&cmd, target);
    gpu::ColorTargetDesc[1] colors = {{
        .view = target_view, .load_op = gpu::LoadOp.CLEAR, .store_op = gpu::StoreOp.STORE,
        .clear = { .rgba = { 8.0f / 255.0f, 16.0f / 255.0f, 32.0f / 255.0f, 1.0f } },
    }};
    gpu::RenderPassDesc pass = { .colors = colors[..], .width = VAL_RT, .height = VAL_RT };
    gpu::GraphicsState graphics_state =
        gpu::render_geometry_state(pass.width, pass.height)!!;
    gpu::ColorTargetState[1] color_targets = {
        gpu::color_blend_disabled(),
    };
    graphics_state.color.targets = color_targets[..];
    gpu::cmd_begin_render_pass(&cmd, &pass)!!;
    gpu::cmd_bind_pipeline(&cmd, ctx.graphics)!!;
    gpu::cmd_set_graphics_state(&cmd, &graphics_state)!!;
    gpu::cmd_draw(&cmd, val_span_address(&device, verts_span), val_span_address(&device, frag_span), 3, 1)!!;
    gpu::cmd_end_render_pass(&cmd)!!;

    gpu::ExecutableCommandList cmd_executable = gpu::end_commands(&cmd)!!;

    defer discard_test_executable(&cmd_executable)!!;
    gpu::ExecutableCommandList[1] lists = { cmd_executable };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::CompletionPoint completion = submit_test_queue(
        &device,
        gpu::QueueKind.GRAPHICS,
        &submit,
    )!!;
    gpu::wait_completion(completion)!!;
}

// Backing-range faults: subspan builds escaping spans without faulting;
// the fault lands at the consuming command.

fn void dispatch_indirect_args_span_beyond_buffer_faults() @test {
    gpu::Device device = make_val_device();
    defer destroy_test_device(&device)!!;
    ValComputeCtx ctx = make_val_compute_ctx(&device);
    defer destroy_val_compute_ctx(&device, &ctx);

    OwnedSpan args_buffer = allocate_owned_span(
        device:       &device,
        size:         16,
        memory_class: gpu::MemoryClass.GPU_PRIVATE,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "dispatch_indirect_args",
    )!!;
    defer free_owned_span(&device, &args_buffer)!!;
    gpu::GpuSpan full_span = args_buffer.span;
    gpu::GpuSpan escaped = full_span.unchecked_subspan(8, gpu::DispatchIndirectCommand::size);
    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator allocator = gpu::create_command_allocator(&device, queue)!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::CommandList cmd = gpu::begin_commands(&allocator)!!;
    defer discard_test_commands(&cmd)!!;
    bind_test_compute_pipeline(&cmd, ctx.pipeline);
    fault beyond = @catch(gpu::cmd_dispatch_indirect(
        commands: &cmd,
        root:     (gpu::GpuAddress)1,
        args:     escaped,
    ));
    assert(beyond == gpu::INVALID_ARGUMENT, "dispatch indirect args beyond the backing buffer faults INVALID_ARGUMENT");

    gpu::ExecutableCommandList cmd_executable = gpu::end_commands(&cmd)!!;

    defer discard_test_executable(&cmd_executable)!!;
}

fn void dispatch_indirect_args_span_boundary_exact_succeeds() @test {
    gpu::Device device = make_val_device();
    defer destroy_test_device(&device)!!;
    ValComputeCtx ctx = make_val_compute_ctx(&device);
    defer destroy_val_compute_ctx(&device, &ctx);

    OwnedSpan args_buffer = allocate_owned_span(
        device:       &device,
        size:         16,
        memory_class: gpu::MemoryClass.GPU_PRIVATE,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "dispatch_indirect_args",
    )!!;
    defer free_owned_span(&device, &args_buffer)!!;
    gpu::GpuSpan full_span = args_buffer.span;
    gpu::GpuSpan boundary = full_span.checked_subspan(4, gpu::DispatchIndirectCommand::size)!!;
    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator allocator = gpu::create_command_allocator(&device, queue)!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::CommandList cmd = gpu::begin_commands(&allocator)!!;
    defer discard_test_commands(&cmd)!!;
    bind_test_compute_pipeline(&cmd, ctx.pipeline);
    gpu::cmd_dispatch_indirect(
        commands: &cmd,
        root:     (gpu::GpuAddress)1,
        args:     boundary,
    )!!;

    gpu::ExecutableCommandList cmd_executable = gpu::end_commands(&cmd)!!;

    defer discard_test_executable(&cmd_executable)!!;
}

fn void draw_indexed_indirect_count_span_beyond_buffer_faults() @test {
    gpu::Device device = make_val_device();
    defer destroy_test_device(&device)!!;
    if (!gpu::get_device_caps(&device)!!.draw_indirect_count) return;
    ValDrawCtx ctx = make_val_draw_ctx(&device);
    defer destroy_val_draw_ctx(&device, &ctx);
    gpu::TextureHandle target = make_val_target(&device);
    defer gpu::destroy_texture(&device, target)!!;
    gpu::AttachmentViewHandle target_view = create_test_attachment_view(
        &device,
        target,
    )!!;
    defer destroy_test_attachment_view(&device, &target_view)!!;

    OwnedSpan args_buffer = allocate_owned_span(
        device:       &device,
        size:         gpu::DrawIndexedIndirectCommand::size * 2,
        memory_class: gpu::MemoryClass.GPU_PRIVATE,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "draw_indirect_args",
    )!!;
    defer free_owned_span(&device, &args_buffer)!!;
    gpu::GpuSpan args_span = args_buffer.span;

    OwnedSpan count_buffer = allocate_owned_span(
        device:       &device,
        size:         8,
        memory_class: gpu::MemoryClass.GPU_PRIVATE,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "draw_indirect_count",
    )!!;
    defer free_owned_span(&device, &count_buffer)!!;
    gpu::GpuSpan full_count_span = count_buffer.span;
    gpu::GpuSpan escaped_count = full_count_span.unchecked_subspan(8, 4);
    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator allocator = gpu::create_command_allocator(&device, queue)!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::CommandList cmd = gpu::begin_commands(&allocator)!!;
    defer discard_test_commands(&cmd)!!;
    record_val_target_to_attach(&cmd, target);
    gpu::ColorTargetDesc[1] colors = {{ .view = target_view, .load_op = gpu::LoadOp.CLEAR, .store_op = gpu::StoreOp.STORE }};
    gpu::RenderPassDesc pass = { .colors = colors[..], .width = VAL_RT, .height = VAL_RT };
    gpu::GraphicsState graphics_state =
        gpu::render_geometry_state(pass.width, pass.height)!!;
    gpu::ColorTargetState[1] color_targets = {
        gpu::color_blend_disabled(),
    };
    graphics_state.color.targets = color_targets[..];

    gpu::GpuSpan index_span_stub = {};
    gpu::cmd_begin_render_pass(&cmd, &pass)!!;
    gpu::cmd_bind_pipeline(&cmd, ctx.graphics)!!;
    gpu::cmd_set_graphics_state(&cmd, &graphics_state)!!;
    fault beyond = @catch(gpu::cmd_draw_indexed_indirect_count(
        commands:       &cmd,
        vertex_root:    (gpu::GpuAddress)1,
        fragment_root:  (gpu::GpuAddress)1,
        args:           args_span,
        count_span:     escaped_count,
        max_draw_count: 1,
        index_span:     index_span_stub,
    ));
    assert(beyond == gpu::INVALID_ARGUMENT, "indirect count span beyond the backing buffer faults INVALID_ARGUMENT");

    gpu::cmd_end_render_pass(&cmd)!!;
    gpu::ExecutableCommandList cmd_executable = gpu::end_commands(&cmd)!!;
    defer discard_test_executable(&cmd_executable)!!;
}

fn void indirect_draw_count_limits_follow_command_contract() @test {
    gpu::Device device = make_val_device();
    defer destroy_test_device(&device)!!;
    if (!gpu::get_device_caps(&device)!!.draw_indirect_count) return;
    ValDrawCtx ctx = make_val_draw_ctx(&device);
    defer destroy_val_draw_ctx(&device, &ctx);
    gpu::TextureHandle target = make_val_target(&device);
    defer gpu::destroy_texture(&device, target)!!;
    gpu::AttachmentViewHandle target_view = create_test_attachment_view(
        &device,
        target,
    )!!;
    defer destroy_test_attachment_view(&device, &target_view)!!;

    OwnedSpan args_buffer = allocate_owned_span(
        device:       &device,
        size:         gpu::DrawIndexedIndirectCommand::size * 2,
        memory_class: gpu::MemoryClass.GPU_PRIVATE,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "draw_count_args",
    )!!;
    defer free_owned_span(&device, &args_buffer)!!;
    gpu::GpuSpan args_span = args_buffer.span;

    OwnedSpan count_buffer = allocate_owned_span(
        device:       &device,
        size:         4,
        memory_class: gpu::MemoryClass.GPU_PRIVATE,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "draw_count_value",
    )!!;
    defer free_owned_span(&device, &count_buffer)!!;
    gpu::GpuSpan count_span = count_buffer.span;

    OwnedSpan index_buffer = allocate_owned_span(
        device:       &device,
        size:         4,
        memory_class: gpu::MemoryClass.GPU_PRIVATE,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "draw_count_indices",
    )!!;
    defer free_owned_span(&device, &index_buffer)!!;
    gpu::GpuSpan index_span = index_buffer.span;
    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator allocator = gpu::create_command_allocator(&device, queue)!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::CommandList cmd = gpu::begin_commands(&allocator)!!;
    defer discard_test_commands(&cmd)!!;
    record_val_target_to_attach(&cmd, target);
    gpu::ColorTargetDesc[1] colors = {{
        .view     = target_view,
        .load_op  = gpu::LoadOp.CLEAR,
        .store_op = gpu::StoreOp.STORE,
    }};
    gpu::RenderPassDesc pass = {
        .colors = colors[..],
        .width  = VAL_RT,
        .height = VAL_RT,
    };
    gpu::GraphicsState graphics_state =
        gpu::render_geometry_state(pass.width, pass.height)!!;
    gpu::ColorTargetState[1] color_targets = {
        gpu::color_blend_disabled(),
    };
    graphics_state.color.targets = color_targets[..];

    vk::VkDeviceState* state = gpu::internal::device_slots[device.index].data.state;
    uint original_limit = state.caps.max_draw_indirect_count;
    state.caps.max_draw_indirect_count = 1;
    defer state.caps.max_draw_indirect_count = original_limit;

    gpu::cmd_begin_render_pass(&cmd, &pass)!!;
    gpu::cmd_bind_pipeline(&cmd, ctx.graphics)!!;
    gpu::cmd_set_graphics_state(&cmd, &graphics_state)!!;
    gpu::cmd_draw_indirect(
        commands:      &cmd,
        vertex_root:   (gpu::GpuAddress)1,
        fragment_root: (gpu::GpuAddress)1,
        args:          args_span,
        draw_count:    1,
    )!!;

    gpu::cmd_bind_pipeline(&cmd, ctx.graphics)!!;
    gpu::cmd_draw_indexed_indirect_count(
        commands:       &cmd,
        vertex_root:    (gpu::GpuAddress)1,
        fragment_root:  (gpu::GpuAddress)1,
        args:           args_span,
        count_span:     count_span,
        max_draw_count: 2,
        index_span:     index_span,
    )!!;

    gpu::cmd_end_render_pass(&cmd)!!;
    gpu::ExecutableCommandList cmd_executable = gpu::end_commands(&cmd)!!;
    defer discard_test_executable(&cmd_executable)!!;
}

fn void generated_work_commands_require_capability() @test {
    gpu::Device device = make_val_device();
    defer destroy_test_device(&device)!!;
    ValComputeCtx compute = make_val_compute_ctx(&device);
    defer destroy_val_compute_ctx(&device, &compute);
    ValDrawCtx draw = make_val_draw_ctx(&device);
    defer destroy_val_draw_ctx(&device, &draw);
    gpu::TextureHandle target = make_val_target(&device);
    defer gpu::destroy_texture(&device, target)!!;
    gpu::AttachmentViewHandle target_view = create_test_attachment_view(
        &device,
        target,
    )!!;
    defer destroy_test_attachment_view(&device, &target_view)!!;

    OwnedSpan records = allocate_owned_span(
        device:       &device,
        size:         256,
        memory_class: gpu::MemoryClass.GPU_PRIVATE,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "generated_records",
    )!!;
    defer free_owned_span(&device, &records)!!;
    OwnedSpan count = allocate_owned_span(
        device:       &device,
        size:         16,
        memory_class: gpu::MemoryClass.GPU_PRIVATE,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "generated_count",
    )!!;
    defer free_owned_span(&device, &count)!!;
    OwnedSpan indices = allocate_owned_span(
        device:       &device,
        size:         16,
        memory_class: gpu::MemoryClass.GPU_PRIVATE,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "generated_indices",
    )!!;
    defer free_owned_span(&device, &indices)!!;

    vk::VkDeviceState* state = gpu::internal::device_slots[device.index].data.state;
    bool original_enabled = state.generated_work.enabled;
    state.generated_work.enabled = false;
    defer state.generated_work.enabled = original_enabled;

    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator allocator = gpu::create_command_allocator(&device, queue)!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::CommandList commands = gpu::begin_commands(&allocator)!!;
    defer discard_test_commands(&commands)!!;
    bind_test_compute_pipeline(&commands, compute.pipeline);
    fault dispatch_unsupported = @catch(gpu::cmd_dispatch_generated(
        commands:          &commands,
        records:           records.span,
        count_span:        count.span,
        max_dispatch_count: 1,
    ));
    assert(dispatch_unsupported == gpu::UNSUPPORTED_FEATURE,
        "generated dispatch requires the generated-work capability");

    record_val_target_to_attach(&commands, target);
    gpu::ColorTargetDesc[1] colors = {{
        .view     = target_view,
        .load_op  = gpu::LoadOp.CLEAR,
        .store_op = gpu::StoreOp.STORE,
    }};
    gpu::RenderPassDesc pass = {
        .colors = colors[..],
        .width  = VAL_RT,
        .height = VAL_RT,
    };
    gpu::GraphicsState graphics_state =
        gpu::render_geometry_state(pass.width, pass.height)!!;
    gpu::ColorTargetState[1] color_targets = {
        gpu::color_blend_disabled(),
    };
    graphics_state.color.targets = color_targets[..];
    gpu::cmd_begin_render_pass(&commands, &pass)!!;
    gpu::cmd_bind_pipeline(&commands, draw.graphics)!!;
    gpu::cmd_set_graphics_state(&commands, &graphics_state)!!;
    fault draw_unsupported = @catch(gpu::cmd_draw_generated(
        commands:      &commands,
        records:       records.span,
        count_span:    count.span,
        max_draw_count: 1,
    ));
    assert(draw_unsupported == gpu::UNSUPPORTED_FEATURE,
        "generated draw requires the generated-work capability");
    fault indexed_unsupported = @catch(gpu::cmd_draw_indexed_generated(
        commands:      &commands,
        records:       records.span,
        count_span:    count.span,
        max_draw_count: 1,
        index_span:    indices.span,
    ));
    assert(indexed_unsupported == gpu::UNSUPPORTED_FEATURE,
        "generated indexed draw requires the generated-work capability");

    state.generated_work.enabled = true;
    gpu::cmd_draw_generated(
        commands:      &commands,
        records:       {},
        count_span:    {},
        max_draw_count: 0,
    )!!;
    gpu::cmd_draw_indexed_generated(
        commands:      &commands,
        records:       {},
        count_span:    {},
        max_draw_count: 0,
        index_span:    {},
    )!!;
    gpu::cmd_end_render_pass(&commands)!!;

    gpu::ExecutableCommandList executable = gpu::end_commands(&commands)!!;
    defer discard_test_executable(&executable)!!;
}

fn void generated_dispatch_checks_bounds_and_reservations() @test {
    gpu::Device device = make_val_device();
    defer destroy_test_device(&device)!!;
    ValComputeCtx compute = make_val_compute_ctx(&device);
    defer destroy_val_compute_ctx(&device, &compute);

    OwnedSpan records = allocate_owned_span(
        device:       &device,
        size:         gpu::GeneratedDispatchRecord::size + 8,
        memory_class: gpu::MemoryClass.GPU_PRIVATE,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "generated_dispatch_records",
    )!!;
    defer free_owned_span(&device, &records)!!;
    OwnedSpan count = allocate_owned_span(
        device:       &device,
        size:         8,
        memory_class: gpu::MemoryClass.GPU_PRIVATE,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "generated_dispatch_count",
    )!!;
    defer free_owned_span(&device, &count)!!;

    vk::VkDeviceState* state = gpu::internal::device_slots[device.index].data.state;
    bool original_enabled = state.generated_work.enabled;
    uint original_limit = state.caps.max_generated_work_count;
    state.generated_work.enabled = true;
    state.caps.max_generated_work_count = 1;
    defer {
        state.generated_work.enabled = original_enabled;
        state.caps.max_generated_work_count = original_limit;
    }

    gpu::GpuSpan valid_records = records.span.checked_subspan(
        0,
        gpu::GeneratedDispatchRecord::size,
    )!!;
    gpu::GpuSpan misaligned_records = records.span.checked_subspan(
        4,
        gpu::GeneratedDispatchRecord::size,
    )!!;
    gpu::GpuSpan short_records = records.span.checked_subspan(
        0,
        gpu::GeneratedDispatchRecord::size - 1,
    )!!;
    gpu::GpuSpan valid_count = count.span.checked_subspan(0, 4)!!;
    gpu::GpuSpan misaligned_count = count.span.checked_subspan(2, 4)!!;
    gpu::GpuSpan short_count = count.span.checked_subspan(0, 2)!!;

    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator allocator = gpu::create_command_allocator(&device, queue)!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::CommandList commands = gpu::begin_commands(&allocator)!!;
    defer discard_test_commands(&commands)!!;
    bind_test_compute_pipeline(&commands, compute.pipeline);

    gpu::cmd_dispatch_generated(
        commands:          &commands,
        records:           {},
        count_span:        {},
        max_dispatch_count: 0,
    )!!;
    fault over_limit = @catch(gpu::cmd_dispatch_generated(
        commands:          &commands,
        records:           valid_records,
        count_span:        valid_count,
        max_dispatch_count: 2,
    ));
    assert(over_limit == gpu::INVALID_ARGUMENT,
        "generated dispatch rejects counts above the selected-device limit");
    fault record_size = @catch(gpu::cmd_dispatch_generated(
        commands:          &commands,
        records:           short_records,
        count_span:        valid_count,
        max_dispatch_count: 1,
    ));
    assert(record_size == gpu::INVALID_ARGUMENT,
        "generated record span must hold the declared maximum count");
    fault count_size = @catch(gpu::cmd_dispatch_generated(
        commands:          &commands,
        records:           valid_records,
        count_span:        short_count,
        max_dispatch_count: 1,
    ));
    assert(count_size == gpu::INVALID_ARGUMENT,
        "generated count span must hold one uint");

    gpu::ExecutableCommandList executable = gpu::end_commands(&commands)!!;
    defer discard_test_executable(&executable)!!;
}

fn void draw_indexed_index_span_beyond_buffer_faults() @test {
    gpu::Device device = make_val_device();
    defer destroy_test_device(&device)!!;
    ValDrawCtx ctx = make_val_draw_ctx(&device);
    defer destroy_val_draw_ctx(&device, &ctx);
    gpu::TextureHandle target = make_val_target(&device);
    defer gpu::destroy_texture(&device, target)!!;
    gpu::AttachmentViewHandle target_view = create_test_attachment_view(
        &device,
        target,
    )!!;
    defer destroy_test_attachment_view(&device, &target_view)!!;

    OwnedSpan index_buffer = allocate_owned_span(
        device:       &device,
        size:         16,
        memory_class: gpu::MemoryClass.GPU_PRIVATE,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "indexed_draw_data",
    )!!;
    defer free_owned_span(&device, &index_buffer)!!;
    gpu::GpuSpan full_index_span = index_buffer.span;
    gpu::GpuSpan escaped = full_index_span.unchecked_subspan(16, 4);
    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator allocator = gpu::create_command_allocator(&device, queue)!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::CommandList cmd = gpu::begin_commands(&allocator)!!;
    defer discard_test_commands(&cmd)!!;
    record_val_target_to_attach(&cmd, target);
    gpu::ColorTargetDesc[1] colors = {{ .view = target_view, .load_op = gpu::LoadOp.CLEAR, .store_op = gpu::StoreOp.STORE }};
    gpu::RenderPassDesc pass = { .colors = colors[..], .width = VAL_RT, .height = VAL_RT };
    gpu::GraphicsState graphics_state =
        gpu::render_geometry_state(pass.width, pass.height)!!;
    gpu::ColorTargetState[1] color_targets = {
        gpu::color_blend_disabled(),
    };
    graphics_state.color.targets = color_targets[..];
    gpu::cmd_begin_render_pass(&cmd, &pass)!!;
    gpu::cmd_bind_pipeline(&cmd, ctx.graphics)!!;
    gpu::cmd_set_graphics_state(&cmd, &graphics_state)!!;
    fault beyond = @catch(gpu::cmd_draw_indexed(
        commands:       &cmd,
        vertex_root:    (gpu::GpuAddress)1,
        fragment_root:  (gpu::GpuAddress)1,
        index_span:     escaped,
        index_count:    1,
        instance_count: 1,
    ));
    assert(beyond == gpu::INVALID_ARGUMENT, "index span beyond the backing buffer faults INVALID_ARGUMENT");

    gpu::cmd_end_render_pass(&cmd)!!;
    gpu::ExecutableCommandList cmd_executable = gpu::end_commands(&cmd)!!;
    defer discard_test_executable(&cmd_executable)!!;
}

fn void draw_indexed_index_span_boundary_exact_succeeds() @test {
    gpu::Device device = make_val_device();
    defer destroy_test_device(&device)!!;
    ValDrawCtx ctx = make_val_draw_ctx(&device);
    defer destroy_val_draw_ctx(&device, &ctx);
    gpu::TextureHandle target = make_val_target(&device);
    defer gpu::destroy_texture(&device, target)!!;
    gpu::AttachmentViewHandle target_view = create_test_attachment_view(
        &device,
        target,
    )!!;
    defer destroy_test_attachment_view(&device, &target_view)!!;

    OwnedSpan index_buffer = allocate_owned_span(
        device:       &device,
        size:         16,
        memory_class: gpu::MemoryClass.GPU_PRIVATE,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "indexed_draw_data",
    )!!;
    defer free_owned_span(&device, &index_buffer)!!;
    gpu::GpuSpan full_index_span = index_buffer.span;
    gpu::GpuSpan boundary = full_index_span.checked_subspan(12, 4)!!;
    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator allocator = gpu::create_command_allocator(&device, queue)!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::CommandList cmd = gpu::begin_commands(&allocator)!!;
    defer discard_test_commands(&cmd)!!;
    record_val_target_to_attach(&cmd, target);
    gpu::ColorTargetDesc[1] colors = {{ .view = target_view, .load_op = gpu::LoadOp.CLEAR, .store_op = gpu::StoreOp.STORE }};
    gpu::RenderPassDesc pass = { .colors = colors[..], .width = VAL_RT, .height = VAL_RT };
    gpu::GraphicsState graphics_state =
        gpu::render_geometry_state(pass.width, pass.height)!!;
    gpu::ColorTargetState[1] color_targets = {
        gpu::color_blend_disabled(),
    };
    graphics_state.color.targets = color_targets[..];
    gpu::cmd_begin_render_pass(&cmd, &pass)!!;
    gpu::cmd_bind_pipeline(&cmd, ctx.graphics)!!;
    gpu::cmd_set_graphics_state(&cmd, &graphics_state)!!;
    gpu::cmd_draw_indexed(
        commands:       &cmd,
        vertex_root:    (gpu::GpuAddress)1,
        fragment_root:  (gpu::GpuAddress)1,
        index_span:     boundary,
        index_count:    1,
        instance_count: 1,
    )!!;

    gpu::cmd_end_render_pass(&cmd)!!;
    gpu::ExecutableCommandList cmd_executable = gpu::end_commands(&cmd)!!;
    defer discard_test_executable(&cmd_executable)!!;
}

fn void copy_buffer_to_texture_span_beyond_buffer_faults() @test {
    gpu::Device device = make_val_device();
    defer destroy_test_device(&device)!!;

    gpu::TextureDesc tex_desc = {
        .access    = { .graphics, .compute, .transfer },
        .width     = 4,
        .height    = 4,
        .format    = gpu::Format.RGBA8_UNORM,
        .usage     = { .transfer_dst },
    };
    gpu::TextureHandle texture = gpu::create_texture(&device, &tex_desc)!!;
    defer gpu::destroy_texture(&device, texture)!!;

    OwnedSpan src_buffer = allocate_owned_span(
        device:       &device,
        size:         64,
        memory_class: gpu::MemoryClass.GPU_PRIVATE,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "texture_copy_src",
    )!!;
    defer free_owned_span(&device, &src_buffer)!!;
    gpu::GpuSpan full_span = src_buffer.span;
    gpu::GpuSpan escaped = full_span.unchecked_subspan(32, 64);
    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.TRANSFER)!!;
    gpu::CommandAllocator allocator = gpu::create_command_allocator(&device, queue)!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::CommandList cmd = gpu::begin_commands(&allocator)!!;
    defer discard_test_commands(&cmd)!!;
    record_val_texture_transition(
        commands: &cmd,
        texture:  texture,
        after:    {
            .layout = gpu::TextureLayout.TRANSFER_DESTINATION,
            .stages = { .transfer },
            .access = { .write },
        },
    );

    gpu::BufferTextureCopyDesc up = { .src = escaped, .texture = texture, .width = 4, .height = 4 };
    fault beyond = @catch(gpu::cmd_copy_buffer_to_texture(&cmd, &up));
    assert(beyond == gpu::INVALID_ARGUMENT, "buffer-to-texture copy span beyond the backing buffer faults INVALID_ARGUMENT");

    gpu::ExecutableCommandList cmd_executable = gpu::end_commands(&cmd)!!;

    defer discard_test_executable(&cmd_executable)!!;
}

fn void copy_texture_to_buffer_span_beyond_buffer_faults() @test {
    gpu::Device device = make_val_device();
    defer destroy_test_device(&device)!!;

    gpu::TextureDesc tex_desc = {
        .access    = { .graphics, .compute, .transfer },
        .width     = 4,
        .height    = 4,
        .format    = gpu::Format.RGBA8_UNORM,
        .usage     = { .transfer_src },
    };
    gpu::TextureHandle texture = gpu::create_texture(&device, &tex_desc)!!;
    defer gpu::destroy_texture(&device, texture)!!;

    OwnedSpan dst_buffer = allocate_owned_span(
        device:       &device,
        size:         64,
        memory_class: gpu::MemoryClass.GPU_PRIVATE,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "texture_copy_dst",
    )!!;
    defer free_owned_span(&device, &dst_buffer)!!;
    gpu::GpuSpan full_span = dst_buffer.span;
    gpu::GpuSpan escaped = full_span.unchecked_subspan(32, 64);
    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.TRANSFER)!!;
    gpu::CommandAllocator allocator = gpu::create_command_allocator(&device, queue)!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::CommandList cmd = gpu::begin_commands(&allocator)!!;
    defer discard_test_commands(&cmd)!!;
    record_val_texture_transition(
        commands: &cmd,
        texture:  texture,
        after:    {
            .layout = gpu::TextureLayout.TRANSFER_SOURCE,
            .stages = { .transfer },
            .access = { .read },
        },
    );

    gpu::TextureBufferCopyDesc down = { .texture = texture, .dst = escaped, .width = 4, .height = 4 };
    fault beyond = @catch(gpu::cmd_copy_texture_to_buffer(&cmd, &down));
    assert(beyond == gpu::INVALID_ARGUMENT, "texture-to-buffer copy span beyond the backing buffer faults INVALID_ARGUMENT");

    gpu::ExecutableCommandList cmd_executable = gpu::end_commands(&cmd)!!;

    defer discard_test_executable(&cmd_executable)!!;
}

fn void texture_copy_span_boundary_exact_succeeds() @test {
    gpu::Device device = make_val_device();
    defer destroy_test_device(&device)!!;

    gpu::TextureDesc tex_desc = {
        .access    = { .graphics, .compute, .transfer },
        .width     = 4,
        .height    = 4,
        .format    = gpu::Format.RGBA8_UNORM,
        .usage     = { .transfer_dst },
    };
    gpu::TextureHandle texture = gpu::create_texture(&device, &tex_desc)!!;
    defer gpu::destroy_texture(&device, texture)!!;

    OwnedSpan src_buffer = allocate_owned_span(
        device:       &device,
        size:         96,
        memory_class: gpu::MemoryClass.GPU_PRIVATE,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "texture_copy_boundary_src",
    )!!;
    defer free_owned_span(&device, &src_buffer)!!;
    gpu::GpuSpan full_span = src_buffer.span;
    gpu::GpuSpan boundary = full_span.checked_subspan(32, 64)!!;
    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.TRANSFER)!!;
    gpu::CommandAllocator allocator = gpu::create_command_allocator(&device, queue)!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::CommandList cmd = gpu::begin_commands(&allocator)!!;
    defer discard_test_commands(&cmd)!!;
    record_val_texture_transition(
        commands: &cmd,
        texture:  texture,
        after:    {
            .layout = gpu::TextureLayout.TRANSFER_DESTINATION,
            .stages = { .transfer },
            .access = { .write },
        },
    );

    gpu::BufferTextureCopyDesc up = { .src = boundary, .texture = texture, .width = 4, .height = 4 };
    gpu::cmd_copy_buffer_to_texture(&cmd, &up)!!;

    gpu::ExecutableCommandList cmd_executable = gpu::end_commands(&cmd)!!;

    defer discard_test_executable(&cmd_executable)!!;
}

fn void barrier_combined_stage_masks_succeed() @test {
    gpu::Device device = make_val_device();
    defer destroy_test_device(&device)!!;
    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator allocator = gpu::create_command_allocator(&device, queue)!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::CommandList commands = gpu::begin_commands(&allocator)!!;
    defer discard_test_commands(&commands)!!;
    gpu::Barrier barrier = {
        .before = {
            .compute = true,
            .color_output = true,
        },
        .after = {
            .vertex_shader = true,
            .fragment_shader = true,
        },
    };

    gpu::cmd_barrier(&commands, &barrier)!!;
    gpu::ExecutableCommandList executable = gpu::end_commands(&commands)!!;
    defer discard_test_executable(&executable)!!;
}

fn void barrier_hazard_matrix_submits() @test {
    gpu::Device device = make_val_device();
    defer destroy_test_device(&device)!!;
    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator allocator = gpu::create_command_allocator(&device, queue)!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::CommandList commands = gpu::begin_commands(&allocator)!!;
    defer discard_test_commands(&commands)!!;
    gpu::Barrier[] barriers = {
        {
            .before = { .host = true },
            .after = { .transfer = true },
        },
        {
            .before = { .transfer = true },
            .after = { .compute = true },
        },
        {
            .before = { .compute = true },
            .after = { .compute = true, .indirect = true },
        },
        {
            .before = { .compute = true },
            .after = { .compute = true, .depth_output = true },
        },
        {
            .before = {
                .color_output = true,
                .depth_output = true,
            },
            .after = { .fragment_shader = true },
        },
        {
            .before = { .color_output = true },
            .after = { .present = true },
        },
    };
    foreach (&barrier : barriers) {
        gpu::cmd_barrier(&commands, barrier)!!;
    }

    gpu::ExecutableCommandList executable = gpu::end_commands(&commands)!!;
    defer discard_test_executable(&executable)!!;
    gpu::ExecutableCommandList[1] lists = { executable };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::CompletionPoint completion = submit_test_queue(
        &device,
        gpu::QueueKind.GRAPHICS,
        &submit,
    )!!;
    gpu::wait_completion(completion)!!;
}

fn void zero_extent_pass_without_attachments_faults() @test {
    gpu::RuntimeDesc desc = {
        .enable_vulkan_validation = true,
        .application_name  = "gpu_render_area_test",
    };
    gpu::Device device = create_test_device(&desc)!!;
    defer destroy_test_device(&device)!!;
    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator allocator = gpu::create_command_allocator(&device, queue)!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::CommandList cmd = gpu::begin_commands(&allocator)!!;
    defer discard_test_commands(&cmd)!!;
    gpu::RenderPassDesc empty = {};
    assert(@catch(gpu::cmd_begin_render_pass(&cmd, &empty)) == gpu::INVALID_ARGUMENT,
        "a zero extent with no attachment faults before native recording");
    gpu::RenderPassDesc half_zero = { .width = 16 };
    assert(@catch(gpu::cmd_begin_render_pass(&cmd, &half_zero)) == gpu::INVALID_ARGUMENT,
        "a mixed extent faults before native recording");
}

fn void render_pass_color_count_limits_validate_before_lookup() @test {
    gpu::Device device = make_val_device();
    defer destroy_test_device(&device)!!;
    vk::VkDeviceState* state = gpu::internal::device_slots[device.index].data.state;
    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator allocator = gpu::create_command_allocator(&device, queue)!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::CommandList cmd = gpu::begin_commands(&allocator)!!;
    defer discard_test_commands(&cmd)!!;

    gpu::ColorTargetDesc[gpu::MAX_COLOR_ATTACHMENTS + 1] too_many;
    gpu::RenderPassDesc library_limit = { .colors = too_many[..], .width = 1, .height = 1 };
    assert(@catch(gpu::cmd_begin_render_pass(&cmd, &library_limit))
        == gpu::INVALID_ARGUMENT,
        "color count above MAX_COLOR_ATTACHMENTS faults before invalid handle lookup");
    assert(@catch(gpu::cmd_end_render_pass(&cmd)) == gpu::COMMAND_RECORDING_ERROR, "library-limit rejection records no pass");

    gpu::ExecutableCommandList cmd_executable = gpu::end_commands(&cmd)!!;

    defer discard_test_executable(&cmd_executable)!!;
}

fn void many_precreated_views_record_across_allocator_reuse() @test {
    gpu::Device device = make_val_device();
    defer destroy_test_device(&device)!!;
    const uint VIEW_COUNT = vk::INITIAL_CACHED_VIEW_CAPACITY + 1;
    gpu::TextureHandle texture = make_val_color_target(
        device:     &device,
        width:      512,
        height:     512,
        mip_levels: 10,
    );
    defer gpu::destroy_texture(&device, texture)!!;
    gpu::AttachmentViewHandle[VIEW_COUNT] views;
    for (uint mip = 0; mip < VIEW_COUNT; mip++) {
        views[mip] = create_test_attachment_view(
            &device,
            texture,
            mip,
        )!!;
    }
    gpu::Queue queue = gpu::get_queue(
        &device,
        gpu::QueueKind.GRAPHICS,
    )!!;
    gpu::CommandAllocator allocator = gpu::create_command_allocator(&device, queue)!!;
    defer gpu::destroy_command_allocator(&allocator)!!;

    for (uint iteration = 0; iteration < 2; iteration++) {
        gpu::CommandList commands = gpu::begin_commands(&allocator)!!;
        if (iteration == 0) record_val_target_to_attach(&commands, texture);
        foreach (view : views) {
            gpu::ColorTargetDesc[1] colors = { { .view = view } };
            gpu::RenderPassDesc pass = {
                .colors = colors[..],
                .width  = 1,
                .height = 1,
            };
            gpu::cmd_begin_render_pass(&commands, &pass)!!;
            gpu::cmd_end_render_pass(&commands)!!;
        }
        gpu::ExecutableCommandList executable = gpu::end_commands(&commands)!!;
        gpu::discard_executable_commands(&executable)!!;
    }

    foreach (&view : views) {
        destroy_test_attachment_view(&device, view)!!;
    }
}

fn void inline_root_length_is_validated() @test {
    gpu::Device device = make_val_device();
    defer destroy_test_device(&device)!!;
    ValComputeCtx ctx = make_val_compute_ctx(&device);
    defer destroy_val_compute_ctx(&device, &ctx);
    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.COMPUTE)!!;
    gpu::CommandAllocator allocator = gpu::create_command_allocator(&device, queue)!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::CommandList cmd = gpu::begin_commands(&allocator)!!;
    defer discard_test_commands(&cmd)!!;
    bind_test_compute_pipeline(&cmd, ctx.pipeline);
    char[gpu::INLINE_ROOT_CAPACITY + 4] bytes;
    fault unaligned = @catch(gpu::cmd_dispatch(
        commands:    &cmd,
        root:        (gpu::GpuAddress)0,
        groups:      { 1, 1, 1 },
        inline_root: bytes[:3],
    ));
    assert(unaligned == gpu::INVALID_ARGUMENT, "a payload length not a multiple of four faults INVALID_ARGUMENT");
    fault oversized = @catch(gpu::cmd_dispatch(
        commands:    &cmd,
        root:        (gpu::GpuAddress)0,
        groups:      { 1, 1, 1 },
        inline_root: bytes[..],
    ));
    assert(oversized == gpu::INVALID_ARGUMENT, "a payload past INLINE_ROOT_CAPACITY faults INVALID_ARGUMENT");
    gpu::cmd_dispatch(
        commands:    &cmd,
        root:        (gpu::GpuAddress)0,
        groups:      { 1, 1, 1 },
        inline_root: bytes[:gpu::INLINE_ROOT_CAPACITY],
    )!!;
    gpu::ExecutableCommandList executable = gpu::end_commands(&cmd)!!;
    defer discard_test_executable(&executable)!!;
}
