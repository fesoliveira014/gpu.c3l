module gpu_test;

import gpu;
import gpu::vk;
import vk;
import std::io;

// Headless (lavapipe): GPU-driven indirect execution and readback tickets.
// A compute pass builds draw/dispatch arguments through the root-pointer ABI;
// explicit barriers order the writes before indirect consumption; tickets
// read the GPU-written args back without blocking the recording frame.

const char[*] IND_VERT_SPIRV  = $embed("shaders/offscreen.vert.spv");
const char[*] IND_FRAG_SPIRV  = $embed("shaders/offscreen.frag.spv");
const char[*] IND_BUILD_SPIRV = $embed("shaders/build_indirect.comp.spv");
const char[*] IND_DOUBLE_SPIRV = $embed("shaders/root_pointer.comp.spv");

const uint IND_RT = 16;
const usz IND_RT_BYTES = IND_RT * IND_RT * 4;

struct IndDrawCtx {
    gpu::ShaderHandle   vert;
    gpu::ShaderHandle   frag;
    gpu::PipelineHandle pipeline;
    gpu::TextureHandle  pattern;
    gpu::TextureIndex   tex_index;
    gpu::SamplerIndex   smp_index;
}

fn gpu::Device make_ind_device(usz readback_arena_size = 0) {
    gpu::DeviceDesc desc = {
        .backend             = gpu::BackendKind.VULKAN,
        .enable_validation   = true,
        .frames_in_flight    = 2,
        .readback_arena_size = readback_arena_size,
        .application_name    = "gpu_indirect_test",
    };
    return gpu::create_device(&desc)!!;
}

<* Textured draw context: offscreen shaders, one graphics pipeline, and a 1x1
   white pattern feeding the fragment heap path. *>
fn IndDrawCtx make_ind_draw_ctx(gpu::Device* device) {
    IndDrawCtx ctx;
    gpu::ShaderDesc vert_sd = { .stage = gpu::ShaderStage.VERTEX, .spirv = IND_VERT_SPIRV[..], .entry_point = "main" };
    ctx.vert = gpu::create_shader(device, &vert_sd)!!;
    gpu::ShaderDesc frag_sd = { .stage = gpu::ShaderStage.FRAGMENT, .spirv = IND_FRAG_SPIRV[..], .entry_point = "main" };
    ctx.frag = gpu::create_shader(device, &frag_sd)!!;

    gpu::Format[1] formats = { gpu::Format.RGBA8_UNORM };
    gpu::GraphicsPipelineDesc pipe_desc = {
        .vertex_shader   = ctx.vert,
        .fragment_shader = ctx.frag,
        .topology        = gpu::PrimitiveTopology.TRIANGLES,
        .color_formats   = formats[..],
    };
    ctx.pipeline = gpu::create_graphics_pipeline(device, &pipe_desc)!!;

    gpu::TextureDesc tex_desc = {
        .dimension = gpu::TextureDimension.TEX_2D,
        .width     = 1,
        .height    = 1,
        .format    = gpu::Format.RGBA8_UNORM,
        .usage     = { .sampled, .transfer_dst },
    };
    ctx.pattern = gpu::create_texture(device, &tex_desc)!!;
    ctx.tex_index = gpu::create_texture_descriptor(device, ctx.pattern, null)!!;
    gpu::SamplerDesc sampler_desc = {
        .min_filter = gpu::Filter.NEAREST,
        .mag_filter = gpu::Filter.NEAREST,
        .address_u  = gpu::AddressMode.CLAMP_TO_EDGE,
        .address_v  = gpu::AddressMode.CLAMP_TO_EDGE,
        .address_w  = gpu::AddressMode.CLAMP_TO_EDGE,
    };
    ctx.smp_index = gpu::create_sampler(device, &sampler_desc)!!;

    char[4] white = { (char)255, (char)255, (char)255, (char)255 };
    gpu::upload_texture_data(
        device:      device,
        dst:         ctx.pattern,
        mip:         0,
        data:        white[..],
        next_stage:  gpu::Stage.FRAGMENT_SHADER,
        next_hazard: gpu::Hazard.SHADER_READ,
        next_layout: gpu::TextureLayout.SHADER_READ,
    )!!;
    return ctx;
}

fn void destroy_ind_draw_ctx(gpu::Device* device, IndDrawCtx* ctx) {
    gpu::destroy_sampler(device, ctx.smp_index)!!;
    gpu::destroy_texture_descriptor(device, ctx.tex_index)!!;
    gpu::destroy_texture(device, ctx.pattern)!!;
    gpu::destroy_pipeline(device, ctx.pipeline)!!;
    gpu::destroy_shader(device, ctx.frag)!!;
    gpu::destroy_shader(device, ctx.vert)!!;
}

fn gpu::TextureHandle make_ind_target(gpu::Device* device) {
    gpu::TextureDesc desc = {
        .dimension = gpu::TextureDimension.TEX_2D,
        .width     = IND_RT,
        .height    = IND_RT,
        .format    = gpu::Format.RGBA8_UNORM,
        .usage     = { .color_attach, .transfer_src },
    };
    return gpu::create_texture(device, &desc)!!;
}

fn void write_triangle_verts(gpu::GpuSpan* span) {
    float* verts = (float*)span.cpu;
    float[12] vertex_data = {
        -0.8f, -0.8f, 0.1f, 0.1f,
         0.8f, -0.8f, 0.9f, 0.1f,
         0.0f,  0.8f, 0.5f, 0.9f,
    };
    for (uint i = 0; i < 12; i++) verts[i] = vertex_data[i];
}

fn void record_target_to_attach(gpu::CommandList* cmd, gpu::TextureHandle target) {
    gpu::TextureBarrier to_attach = {
        .texture       = target,
        .before_stage  = gpu::Stage.HOST,
        .after_stage   = gpu::Stage.COLOR_ATTACHMENT,
        .before_hazard = gpu::Hazard.HOST_WRITE,
        .after_hazard  = gpu::Hazard.COLOR_WRITE,
        .old_layout    = gpu::TextureLayout.UNDEFINED,
        .new_layout    = gpu::TextureLayout.COLOR_ATTACHMENT,
    };
    gpu::cmd_texture_barrier(cmd, &to_attach)!!;
}

fn void record_target_readback(gpu::CommandList* cmd, gpu::TextureHandle target, gpu::GpuSpan* rb_span) {
    gpu::TextureBarrier to_src = {
        .texture       = target,
        .before_stage  = gpu::Stage.COLOR_ATTACHMENT,
        .after_stage   = gpu::Stage.TRANSFER,
        .before_hazard = gpu::Hazard.COLOR_WRITE,
        .after_hazard  = gpu::Hazard.TRANSFER_READ,
        .old_layout    = gpu::TextureLayout.COLOR_ATTACHMENT,
        .new_layout    = gpu::TextureLayout.TRANSFER_SRC,
    };
    gpu::cmd_texture_barrier(cmd, &to_src)!!;
    gpu::TextureBufferCopyDesc copy = { .texture = target, .dst = *rb_span };
    gpu::cmd_copy_texture_to_buffer(cmd, &copy)!!;
    gpu::BufferBarrier host_read = {
        .buffer = rb_span.buffer, .offset = rb_span.offset, .size = rb_span.size,
        .before_stage = gpu::Stage.TRANSFER, .after_stage = gpu::Stage.HOST,
        .before_hazard = gpu::Hazard.TRANSFER_WRITE, .after_hazard = gpu::Hazard.HOST_READ,
    };
    gpu::cmd_buffer_barrier(cmd, &host_read)!!;
}

fn char* center_pixel(gpu::GpuSpan* rb_span) {
    char* pixels = (char*)rb_span.cpu;
    return &pixels[(IND_RT / 2 * IND_RT + IND_RT / 2) * 4];
}

fn bool distinct_graphics_compute_families(gpu::Device* device, ZString test_name) {
    vk::VkDeviceState* state = (vk::VkDeviceState*)device.backend_state;
    if (state.graphics_family != state.compute_family) return true;
    io::eprintfn("%s not applicable: graphics and compute share family %d", test_name, state.graphics_family);
    return false;
}

fn gpu::PipelineHandle make_cross_family_compute_pipeline(
    gpu::Device* device,
    gpu::ShaderHandle* out_shader,
) {
    gpu::ShaderDesc shader_desc = {
        .stage       = gpu::ShaderStage.COMPUTE,
        .spirv       = IND_DOUBLE_SPIRV[..],
        .entry_point = "main",
    };
    *out_shader = gpu::create_shader(device, &shader_desc)!!;
    gpu::ComputePipelineDesc pipeline_desc = {
        .shader             = *out_shader,
        .push_constant_size = gpu::RootPush::size,
    };
    return gpu::create_compute_pipeline(device, &pipeline_desc)!!;
}

fn gpu::GpuSpan alloc_cross_family_indirect_args(gpu::Device* device) {
    gpu::GpuSpan span = gpu::alloc_frame_span(device, gpu::DispatchIndirectCommand::size, 16)!!;
    uint* words = (uint*)span.cpu;
    words[0] = 1;
    words[1] = 1;
    words[2] = 1;
    return span;
}

fn gpu::GpuSpan alloc_empty_compute_root(gpu::Device* device) {
    gpu::GpuSpan span = gpu::alloc_frame_span(device, ComputeRoot::size, 16)!!;
    ComputeRoot* root = (ComputeRoot*)span.cpu;
    *root = { .count = 0 };
    return span;
}

fn void frame_span_same_range_crosses_compute_and_graphics_families_when_available() @test {
    gpu::Device device = make_ind_device();
    defer gpu::destroy_device(&device)!!;
    if (!distinct_graphics_compute_families(&device, "same-range frame span")) return;

    gpu::ShaderHandle shader;
    gpu::PipelineHandle pipeline = make_cross_family_compute_pipeline(&device, &shader);
    defer gpu::destroy_shader(&device, shader)!!;
    defer gpu::destroy_pipeline(&device, pipeline)!!;
    gpu::SemaphoreDesc semaphore_desc = { .debug_name = "frame_arena_cross_family" };
    gpu::SemaphoreHandle semaphore = gpu::create_semaphore(&device, &semaphore_desc)!!;
    defer gpu::destroy_semaphore(&device, semaphore)!!;

    gpu::begin_frame(&device)!!;
    gpu::GpuSpan args = alloc_cross_family_indirect_args(&device);
    gpu::GpuSpan compute_root = alloc_empty_compute_root(&device);
    gpu::GpuSpan graphics_root = alloc_empty_compute_root(&device);

    gpu::CommandList compute = gpu::begin_commands(&device, gpu::QueueKind.COMPUTE)!!;
    gpu::cmd_dispatch_indirect(
        commands: &compute,
        pipeline: pipeline,
        root:     compute_root.gpu,
        args:     args,
    )!!;
    gpu::end_commands(&device, &compute)!!;
    gpu::CommandList[1] compute_lists = { compute };
    gpu::SemaphoreSignal[1] signals = {{
        .semaphore = semaphore,
        .value     = (gpu::SemaphoreValue)1,
        .stage     = gpu::Stage.INDIRECT_COMMAND,
    }};
    gpu::SubmitDesc compute_submit = {
        .command_lists = compute_lists[..],
        .signals       = signals[..],
    };
    gpu::submit(&device, &compute_submit)!!;

    gpu::CommandList graphics = gpu::begin_commands(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::cmd_dispatch_indirect(
        commands: &graphics,
        pipeline: pipeline,
        root:     graphics_root.gpu,
        args:     args,
    )!!;
    gpu::end_commands(&device, &graphics)!!;
    gpu::CommandList[1] graphics_lists = { graphics };
    gpu::SemaphoreWait[1] waits = {{
        .semaphore = semaphore,
        .value     = (gpu::SemaphoreValue)1,
        .stage     = gpu::Stage.INDIRECT_COMMAND,
    }};
    gpu::SubmitDesc graphics_submit = {
        .command_lists = graphics_lists[..],
        .waits         = waits[..],
    };
    gpu::submit(&device, &graphics_submit)!!;
    gpu::end_frame(&device)!!;
    gpu::wait_queue_idle(&device, gpu::QueueKind.COMPUTE)!!;
    gpu::wait_queue_idle(&device, gpu::QueueKind.GRAPHICS)!!;
}

fn void frame_span_reused_offset_switches_queue_family_when_available() @test {
    gpu::DeviceDesc desc = {
        .backend           = gpu::BackendKind.VULKAN,
        .enable_validation = true,
        .frames_in_flight  = 1,
        .application_name  = "frame_arena_reuse_cross_family",
    };
    gpu::Device device = gpu::create_device(&desc)!!;
    defer gpu::destroy_device(&device)!!;
    if (!distinct_graphics_compute_families(&device, "reused-offset frame span")) return;

    gpu::ShaderHandle shader;
    gpu::PipelineHandle pipeline = make_cross_family_compute_pipeline(&device, &shader);
    defer gpu::destroy_shader(&device, shader)!!;
    defer gpu::destroy_pipeline(&device, pipeline)!!;

    gpu::begin_frame(&device)!!;
    gpu::GpuSpan compute_args = alloc_cross_family_indirect_args(&device);
    assert(compute_args.offset == 0, "first slot use starts at offset zero");
    gpu::GpuSpan compute_root = alloc_empty_compute_root(&device);
    gpu::CommandList compute = gpu::begin_commands(&device, gpu::QueueKind.COMPUTE)!!;
    gpu::cmd_dispatch_indirect(
        commands: &compute,
        pipeline: pipeline,
        root:     compute_root.gpu,
        args:     compute_args,
    )!!;
    gpu::end_commands(&device, &compute)!!;
    gpu::CommandList[1] compute_lists = { compute };
    gpu::SubmitDesc compute_submit = { .command_lists = compute_lists[..] };
    gpu::submit(&device, &compute_submit)!!;
    gpu::end_frame(&device)!!;

    gpu::begin_frame(&device)!!;
    gpu::GpuSpan graphics_args = alloc_cross_family_indirect_args(&device);
    assert(graphics_args.buffer == compute_args.buffer, "retired slot reuses its backing buffer");
    assert(graphics_args.offset == 0, "retired slot reuses offset zero");
    gpu::GpuSpan graphics_root = alloc_empty_compute_root(&device);
    gpu::CommandList graphics = gpu::begin_commands(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::cmd_dispatch_indirect(
        commands: &graphics,
        pipeline: pipeline,
        root:     graphics_root.gpu,
        args:     graphics_args,
    )!!;
    gpu::end_commands(&device, &graphics)!!;
    gpu::CommandList[1] graphics_lists = { graphics };
    gpu::SubmitDesc graphics_submit = { .command_lists = graphics_lists[..] };
    gpu::submit(&device, &graphics_submit)!!;
    gpu::end_frame(&device)!!;
    gpu::wait_queue_idle(&device, gpu::QueueKind.GRAPHICS)!!;
}

fn void compute_generated_indexed_indirect_draw() @test {
    gpu::Device device = make_ind_device();
    defer gpu::destroy_device(&device)!!;
    IndDrawCtx ctx = make_ind_draw_ctx(&device);
    defer destroy_ind_draw_ctx(&device, &ctx);
    gpu::TextureHandle target = make_ind_target(&device);
    defer gpu::destroy_texture(&device, target)!!;

    gpu::ShaderDesc build_sd = { .stage = gpu::ShaderStage.COMPUTE, .spirv = IND_BUILD_SPIRV[..], .entry_point = "main" };
    gpu::ShaderHandle build_shader = gpu::create_shader(&device, &build_sd)!!;
    defer gpu::destroy_shader(&device, build_shader)!!;
    gpu::ComputePipelineDesc build_desc = { .shader = build_shader, .push_constant_size = gpu::RootPush::size };
    gpu::PipelineHandle build_pipeline = gpu::create_compute_pipeline(&device, &build_desc)!!;
    defer gpu::destroy_pipeline(&device, build_pipeline)!!;

    // Indexed args live in a DEVICE buffer: GPU-written, ticket is the only
    // way to observe them.
    gpu::BufferDesc args_desc = {
        .size        = 2 * gpu::DrawIndexedIndirectCommand::size,
        .usage       = { .storage, .addressable, .indirect, .transfer_src },
        .memory_kind = gpu::MemoryKind.DEVICE,
    };
    gpu::BufferHandle args_buffer = gpu::create_buffer(&device, &args_desc)!!;
    defer gpu::destroy_buffer(&device, args_buffer)!!;
    gpu::GpuSpan args_span = gpu::get_buffer_span(&device, args_buffer)!!;

    gpu::BufferDesc rb_desc = {
        .size = IND_RT_BYTES, .usage = { .transfer_dst }, .memory_kind = gpu::MemoryKind.READBACK,
    };
    gpu::BufferHandle readback = gpu::create_buffer(&device, &rb_desc)!!;
    defer gpu::destroy_buffer(&device, readback)!!;
    gpu::GpuSpan rb_span = gpu::get_buffer_span(&device, readback)!!;

    gpu::begin_frame(&device)!!;

    gpu::GpuSpan scratch = gpu::alloc_frame_span(&device, 64, 16)!!;
    gpu::GpuSpan verts_span = gpu::alloc_frame_span(&device, 3 * 16, 16)!!;
    write_triangle_verts(&verts_span);
    gpu::GpuSpan index_span = gpu::alloc_frame_span(&device, 3 * 4, 4)!!;
    uint* indices = (uint*)index_span.cpu;
    indices[0] = 0; indices[1] = 1; indices[2] = 2;
    gpu::GpuSpan frag_span = gpu::alloc_frame_span(&device, FragRoot::size, 16)!!;
    FragRoot* frag_root = (FragRoot*)frag_span.cpu;
    frag_root.texture_index = ctx.tex_index;
    frag_root.sampler_index = ctx.smp_index;

    gpu::GpuSpan root_span = gpu::alloc_frame_span(&device, BuildRoot::size, 16)!!;
    BuildRoot* root = (BuildRoot*)root_span.cpu;
    root.indexed_args_gpu = args_span.gpu;
    root.plain_args_gpu   = scratch.gpu;
    root.dispatch_args_gpu = (gpu::GpuAddress)((ulong)scratch.gpu + 16);
    root.count_gpu        = (gpu::GpuAddress)((ulong)scratch.gpu + 32);
    root.index_count      = 3;
    root.vertex_count     = 3;
    root.dispatch_x       = 1;
    root.count_value      = 1;

    gpu::CommandList cmd = gpu::begin_commands(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::cmd_dispatch(
        commands: &cmd,
        pipeline: build_pipeline,
        root:     root_span.gpu,
        groups:   { 1, 1, 1 },
    )!!;

    gpu::BufferBarrier args_ready = {
        .buffer        = args_buffer,
        .offset        = 0,
        .size          = args_desc.size,
        .before_stage  = gpu::Stage.COMPUTE_SHADER,
        .after_stage   = gpu::Stage.INDIRECT_COMMAND,
        .before_hazard = gpu::Hazard.SHADER_WRITE,
        .after_hazard  = gpu::Hazard.INDIRECT_READ,
    };
    gpu::cmd_buffer_barrier(&cmd, &args_ready)!!;

    // Ticket on the same GPU-written args: order the compute writes before
    // the ticket's transfer read.
    gpu::BufferBarrier args_to_copy = {
        .buffer        = args_buffer,
        .offset        = 0,
        .size          = args_desc.size,
        .before_stage  = gpu::Stage.COMPUTE_SHADER,
        .after_stage   = gpu::Stage.TRANSFER,
        .before_hazard = gpu::Hazard.SHADER_WRITE,
        .after_hazard  = gpu::Hazard.TRANSFER_READ,
    };
    gpu::cmd_buffer_barrier(&cmd, &args_to_copy)!!;
    gpu::ReadbackTicket ticket = gpu::cmd_readback_buffer(
        commands: &cmd,
        src:      args_buffer,
        offset:   0,
        size:     args_desc.size,
    )!!;

    record_target_to_attach(&cmd, target);
    gpu::ColorTargetDesc[1] colors = {{
        .texture = target, .load_op = gpu::LoadOp.CLEAR, .store_op = gpu::StoreOp.STORE,
        .clear = { .rgba = { 8.0f / 255.0f, 16.0f / 255.0f, 32.0f / 255.0f, 1.0f } },
    }};
    gpu::RenderPassDesc pass = { .colors = colors[..], .width = IND_RT, .height = IND_RT };
    gpu::cmd_begin_render_pass(&cmd, &pass)!!;
    gpu::cmd_draw_indexed_indirect(
        commands:      &cmd,
        pipeline:      ctx.pipeline,
        vertex_root:   verts_span.gpu,
        fragment_root: frag_span.gpu,
        args:          args_span,
        draw_count:    1,
        index_span:    index_span,
    )!!;
    gpu::cmd_end_render_pass(&cmd)!!;
    record_target_readback(&cmd, target, &rb_span);

    gpu::end_commands(&device, &cmd)!!;
    gpu::CommandList[1] lists = { cmd };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::submit(&device, &submit)!!;

    assert(!gpu::poll_readback(&device, &ticket), "ticket is not resolvable before the frame ends");
    fault early = @catch(gpu::resolve_readback(&device, &ticket, (char[])((char*)&ticket)[:1]));
    assert(early == gpu::INVALID_ARGUMENT || early == gpu::READBACK_NOT_READY, "early resolve is rejected");

    gpu::end_frame(&device)!!;
    gpu::wait_queue_idle(&device, gpu::QueueKind.GRAPHICS)!!;

    gpu::invalidate_buffer(&device, readback, 0, 0)!!;
    char* center = center_pixel(&rb_span);
    assert(center[0] == (char)255 && center[1] == (char)255 && center[2] == (char)255,
        "compute-generated indexed indirect draw renders the triangle");

    assert(gpu::poll_readback(&device, &ticket), "ticket resolvable after the recording frame ends");
    gpu::DrawIndexedIndirectCommand[2] got;
    gpu::resolve_readback(&device, &ticket, ((char*)&got)[:2 * gpu::DrawIndexedIndirectCommand::size])!!;
    for (uint i = 0; i < 2; i++) {
        assert(got[i].index_count == 3 && got[i].instance_count == 1, "GPU-written args read back intact");
        assert(got[i].first_index == 0 && got[i].vertex_offset == 0 && got[i].first_instance == 0,
            "GPU-written arg tail fields read back intact");
    }

    fault again = @catch(gpu::resolve_readback(&device, &ticket, ((char*)&got)[:2 * gpu::DrawIndexedIndirectCommand::size]));
    assert(again == gpu::INVALID_ARGUMENT, "double resolve faults INVALID_ARGUMENT");
    assert(!gpu::poll_readback(&device, &ticket), "poll on a resolved ticket is false");
}

fn void plain_indirect_draw_from_frame_span() @test {
    gpu::Device device = make_ind_device();
    defer gpu::destroy_device(&device)!!;
    IndDrawCtx ctx = make_ind_draw_ctx(&device);
    defer destroy_ind_draw_ctx(&device, &ctx);
    gpu::TextureHandle target = make_ind_target(&device);
    defer gpu::destroy_texture(&device, target)!!;

    gpu::BufferDesc rb_desc = {
        .size = IND_RT_BYTES, .usage = { .transfer_dst }, .memory_kind = gpu::MemoryKind.READBACK,
    };
    gpu::BufferHandle readback = gpu::create_buffer(&device, &rb_desc)!!;
    defer gpu::destroy_buffer(&device, readback)!!;
    gpu::GpuSpan rb_span = gpu::get_buffer_span(&device, readback)!!;

    gpu::begin_frame(&device)!!;
    gpu::GpuSpan verts_span = gpu::alloc_frame_span(&device, 3 * 16, 16)!!;
    write_triangle_verts(&verts_span);
    gpu::GpuSpan frag_span = gpu::alloc_frame_span(&device, FragRoot::size, 16)!!;
    FragRoot* frag_root = (FragRoot*)frag_span.cpu;
    frag_root.texture_index = ctx.tex_index;
    frag_root.sampler_index = ctx.smp_index;

    // Host-written args in a frame span: the arena carries indirect usage.
    gpu::GpuSpan args_span = gpu::alloc_frame_span(&device, gpu::DrawIndirectCommand::size, 16)!!;
    gpu::DrawIndirectCommand* args = (gpu::DrawIndirectCommand*)args_span.cpu;
    *args = { .vertex_count = 3, .instance_count = 1 };

    gpu::CommandList cmd = gpu::begin_commands(&device, gpu::QueueKind.GRAPHICS)!!;
    record_target_to_attach(&cmd, target);
    gpu::ColorTargetDesc[1] colors = {{
        .texture = target, .load_op = gpu::LoadOp.CLEAR, .store_op = gpu::StoreOp.STORE,
        .clear = { .rgba = { 8.0f / 255.0f, 16.0f / 255.0f, 32.0f / 255.0f, 1.0f } },
    }};
    gpu::RenderPassDesc pass = { .colors = colors[..], .width = IND_RT, .height = IND_RT };
    gpu::cmd_begin_render_pass(&cmd, &pass)!!;
    gpu::cmd_draw_indirect(
        commands:      &cmd,
        pipeline:      ctx.pipeline,
        vertex_root:   verts_span.gpu,
        fragment_root: frag_span.gpu,
        args:          args_span,
        draw_count:    1,
    )!!;
    gpu::cmd_end_render_pass(&cmd)!!;
    record_target_readback(&cmd, target, &rb_span);
    gpu::end_commands(&device, &cmd)!!;
    gpu::CommandList[1] lists = { cmd };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::submit(&device, &submit)!!;
    gpu::wait_queue_idle(&device, gpu::QueueKind.GRAPHICS)!!;

    gpu::invalidate_buffer(&device, readback, 0, 0)!!;
    char* center = center_pixel(&rb_span);
    assert(center[0] == (char)255, "non-indexed indirect draw renders");
    gpu::end_frame(&device)!!;
}

fn void indirect_dispatch_executes_gpu_counts() @test {
    gpu::Device device = make_ind_device();
    defer gpu::destroy_device(&device)!!;

    gpu::ShaderDesc build_sd = { .stage = gpu::ShaderStage.COMPUTE, .spirv = IND_BUILD_SPIRV[..], .entry_point = "main" };
    gpu::ShaderHandle build_shader = gpu::create_shader(&device, &build_sd)!!;
    defer gpu::destroy_shader(&device, build_shader)!!;
    gpu::ComputePipelineDesc build_desc = { .shader = build_shader, .push_constant_size = gpu::RootPush::size };
    gpu::PipelineHandle build_pipeline = gpu::create_compute_pipeline(&device, &build_desc)!!;
    defer gpu::destroy_pipeline(&device, build_pipeline)!!;

    gpu::ShaderDesc double_sd = { .stage = gpu::ShaderStage.COMPUTE, .spirv = IND_DOUBLE_SPIRV[..], .entry_point = "main" };
    gpu::ShaderHandle double_shader = gpu::create_shader(&device, &double_sd)!!;
    defer gpu::destroy_shader(&device, double_shader)!!;
    gpu::ComputePipelineDesc double_desc = { .shader = double_shader, .push_constant_size = gpu::RootPush::size };
    gpu::PipelineHandle double_pipeline = gpu::create_compute_pipeline(&device, &double_desc)!!;
    defer gpu::destroy_pipeline(&device, double_pipeline)!!;

    const uint COUNT = 100;
    gpu::begin_frame(&device)!!;

    gpu::GpuSpan io_span = gpu::alloc_frame_span(&device, COUNT * 4 * 2, 16)!!;
    float* values = (float*)io_span.cpu;
    for (uint i = 0; i < COUNT; i++) values[i] = (float)i;

    gpu::GpuSpan scratch = gpu::alloc_frame_span(&device, 64, 16)!!;
    gpu::GpuSpan dispatch_args = scratch.checked_subspan(16, gpu::DispatchIndirectCommand::size)!!;

    gpu::GpuSpan build_root_span = gpu::alloc_frame_span(&device, BuildRoot::size, 16)!!;
    BuildRoot* build_root = (BuildRoot*)build_root_span.cpu;
    build_root.indexed_args_gpu = scratch.gpu;
    build_root.plain_args_gpu   = scratch.gpu;
    build_root.dispatch_args_gpu = dispatch_args.gpu;
    build_root.count_gpu        = (gpu::GpuAddress)((ulong)scratch.gpu + 48);
    build_root.index_count      = 3;
    build_root.vertex_count     = 3;
    build_root.dispatch_x       = (COUNT + 63) / 64;
    build_root.count_value      = 1;

    gpu::GpuSpan double_root_span = gpu::alloc_frame_span(&device, ComputeRoot::size, 16)!!;
    ComputeRoot* double_root = (ComputeRoot*)double_root_span.cpu;
    double_root.input_gpu  = io_span.gpu;
    double_root.output_gpu = (gpu::GpuAddress)((ulong)io_span.gpu + COUNT * 4);
    double_root.count      = COUNT;

    gpu::CommandList cmd = gpu::begin_commands(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::cmd_dispatch(
        commands: &cmd,
        pipeline: build_pipeline,
        root:     build_root_span.gpu,
        groups:   { 1, 1, 1 },
    )!!;
    gpu::BufferBarrier args_ready = {
        .buffer        = dispatch_args.buffer,
        .offset        = dispatch_args.offset,
        .size          = dispatch_args.size,
        .before_stage  = gpu::Stage.COMPUTE_SHADER,
        .after_stage   = gpu::Stage.INDIRECT_COMMAND,
        .before_hazard = gpu::Hazard.SHADER_WRITE,
        .after_hazard  = gpu::Hazard.INDIRECT_READ,
    };
    gpu::cmd_buffer_barrier(&cmd, &args_ready)!!;
    gpu::cmd_dispatch_indirect(
        commands: &cmd,
        pipeline: double_pipeline,
        root:     double_root_span.gpu,
        args:     dispatch_args,
    )!!;
    gpu::BufferBarrier host_read = {
        .buffer = io_span.buffer, .offset = io_span.offset, .size = io_span.size,
        .before_stage = gpu::Stage.COMPUTE_SHADER, .after_stage = gpu::Stage.HOST,
        .before_hazard = gpu::Hazard.SHADER_WRITE, .after_hazard = gpu::Hazard.HOST_READ,
    };
    gpu::cmd_buffer_barrier(&cmd, &host_read)!!;
    gpu::end_commands(&device, &cmd)!!;
    gpu::CommandList[1] lists = { cmd };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::submit(&device, &submit)!!;
    gpu::wait_queue_idle(&device, gpu::QueueKind.GRAPHICS)!!;

    gpu::invalidate_buffer(&device, io_span.buffer, 0, 0)!!;
    float* doubled = &values[COUNT];
    for (uint i = 0; i < COUNT; i++) {
        assert(doubled[i] == (float)i * 2.0f, "indirect dispatch ran with GPU-written group counts");
    }
    gpu::end_frame(&device)!!;
}

fn void count_buffer_limits_draws() @test {
    gpu::Device device = make_ind_device();
    defer gpu::destroy_device(&device)!!;
    if (!device.caps.draw_indirect_count) return;

    IndDrawCtx ctx = make_ind_draw_ctx(&device);
    defer destroy_ind_draw_ctx(&device, &ctx);
    gpu::TextureHandle target_zero = make_ind_target(&device);
    defer gpu::destroy_texture(&device, target_zero)!!;
    gpu::TextureHandle target_one = make_ind_target(&device);
    defer gpu::destroy_texture(&device, target_one)!!;

    gpu::BufferDesc rb_desc = {
        .size = IND_RT_BYTES, .usage = { .transfer_dst }, .memory_kind = gpu::MemoryKind.READBACK,
    };
    gpu::BufferHandle rb_zero = gpu::create_buffer(&device, &rb_desc)!!;
    defer gpu::destroy_buffer(&device, rb_zero)!!;
    gpu::GpuSpan rb_zero_span = gpu::get_buffer_span(&device, rb_zero)!!;
    gpu::BufferHandle rb_one = gpu::create_buffer(&device, &rb_desc)!!;
    defer gpu::destroy_buffer(&device, rb_one)!!;
    gpu::GpuSpan rb_one_span = gpu::get_buffer_span(&device, rb_one)!!;

    gpu::begin_frame(&device)!!;
    gpu::GpuSpan verts_span = gpu::alloc_frame_span(&device, 3 * 16, 16)!!;
    write_triangle_verts(&verts_span);
    gpu::GpuSpan index_span = gpu::alloc_frame_span(&device, 3 * 4, 4)!!;
    uint* indices = (uint*)index_span.cpu;
    indices[0] = 0; indices[1] = 1; indices[2] = 2;
    gpu::GpuSpan frag_span = gpu::alloc_frame_span(&device, FragRoot::size, 16)!!;
    FragRoot* frag_root = (FragRoot*)frag_span.cpu;
    frag_root.texture_index = ctx.tex_index;
    frag_root.sampler_index = ctx.smp_index;

    gpu::GpuSpan args_span = gpu::alloc_frame_span(&device, gpu::DrawIndexedIndirectCommand::size, 16)!!;
    gpu::DrawIndexedIndirectCommand* args = (gpu::DrawIndexedIndirectCommand*)args_span.cpu;
    *args = { .index_count = 3, .instance_count = 1 };

    gpu::GpuSpan counts = gpu::alloc_frame_span(&device, 8, 4)!!;
    uint* count_values = (uint*)counts.cpu;
    count_values[0] = 0;
    count_values[1] = 1;
    gpu::GpuSpan count_zero = counts.checked_subspan(0, 4)!!;
    gpu::GpuSpan count_one = counts.checked_subspan(4, 4)!!;

    gpu::CommandList cmd = gpu::begin_commands(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::TextureHandle[2] targets = { target_zero, target_one };
    gpu::GpuSpan[2] count_spans = { count_zero, count_one };
    gpu::GpuSpan[2] rb_spans = { rb_zero_span, rb_one_span };
    for (uint i = 0; i < 2; i++) {
        record_target_to_attach(&cmd, targets[i]);
        gpu::ColorTargetDesc[1] colors = {{
            .texture = targets[i], .load_op = gpu::LoadOp.CLEAR, .store_op = gpu::StoreOp.STORE,
            .clear = { .rgba = { 8.0f / 255.0f, 16.0f / 255.0f, 32.0f / 255.0f, 1.0f } },
        }};
        gpu::RenderPassDesc pass = { .colors = colors[..], .width = IND_RT, .height = IND_RT };
        gpu::cmd_begin_render_pass(&cmd, &pass)!!;
        gpu::cmd_draw_indexed_indirect_count(
            commands:       &cmd,
            pipeline:       ctx.pipeline,
            vertex_root:    verts_span.gpu,
            fragment_root:  frag_span.gpu,
            args:           args_span,
            count_span:     count_spans[i],
            max_draw_count: 1,
            index_span:     index_span,
        )!!;
        gpu::cmd_end_render_pass(&cmd)!!;
        record_target_readback(&cmd, targets[i], &rb_spans[i]);
    }
    gpu::end_commands(&device, &cmd)!!;
    gpu::CommandList[1] lists = { cmd };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::submit(&device, &submit)!!;
    gpu::wait_queue_idle(&device, gpu::QueueKind.GRAPHICS)!!;

    gpu::invalidate_buffer(&device, rb_zero, 0, 0)!!;
    char* zero_center = center_pixel(&rb_zero_span);
    assert(zero_center[0] == (char)8 && zero_center[1] == (char)16,
        "GPU count of zero draws nothing");
    gpu::invalidate_buffer(&device, rb_one, 0, 0)!!;
    char* one_center = center_pixel(&rb_one_span);
    assert(one_center[0] == (char)255, "GPU count of one draws the triangle");
    gpu::end_frame(&device)!!;
}

fn void ticket_hybrid_and_short_dest() @test {
    // 4 KB readback arena: a few 1 KB tickets pin it full, forcing the
    // dedicated fallback.
    gpu::Device device = make_ind_device(readback_arena_size: 4096);
    defer gpu::destroy_device(&device)!!;

    const usz CHUNK = 1024;
    const uint TICKETS = 6;
    gpu::BufferDesc src_desc = {
        .size        = CHUNK,
        .usage       = { .transfer_src, .transfer_dst },
        .memory_kind = gpu::MemoryKind.STAGING,
    };
    gpu::BufferHandle src = gpu::create_buffer(&device, &src_desc)!!;
    defer gpu::destroy_buffer(&device, src)!!;
    gpu::GpuSpan src_span = gpu::get_buffer_span(&device, src)!!;
    char* src_bytes = (char*)src_span.cpu;
    for (usz i = 0; i < CHUNK; i++) src_bytes[i] = (char)(i % 251);
    gpu::flush_buffer(&device, src, 0, 0)!!;

    gpu::begin_frame(&device)!!;
    gpu::CommandList cmd = gpu::begin_commands(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::ReadbackTicket[TICKETS] tickets;
    uint dedicated_count = 0;
    for (uint i = 0; i < TICKETS; i++) {
        tickets[i] = gpu::cmd_readback_buffer(
            commands: &cmd,
            src:      src,
            offset:   0,
            size:     CHUNK,
        )!!;
        if (tickets[i].dedicated.is_valid()) dedicated_count++;
    }
    assert(dedicated_count > 0, "arena saturation falls back to dedicated tickets");
    assert(dedicated_count < TICKETS, "early tickets are arena-backed");
    gpu::end_commands(&device, &cmd)!!;
    gpu::CommandList[1] lists = { cmd };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::submit(&device, &submit)!!;
    gpu::end_frame(&device)!!;
    gpu::wait_queue_idle(&device, gpu::QueueKind.GRAPHICS)!!;

    char[CHUNK] dest;
    fault short_dest = @catch(gpu::resolve_readback(&device, &tickets[0], dest[:CHUNK - 1]));
    assert(short_dest == gpu::INVALID_ARGUMENT, "short dest faults INVALID_ARGUMENT");

    for (uint i = 0; i < TICKETS; i++) {
        assert(gpu::poll_readback(&device, &tickets[i]), "all tickets resolvable after the frame");
        gpu::resolve_readback(&device, &tickets[i], dest[..])!!;
        for (usz b = 0; b < CHUNK; b++) {
            assert(dest[b] == (char)(b % 251), "ticket bytes intact on both arena and dedicated paths");
        }
    }
}

fn void indirect_argument_validation() @test {
    gpu::Device device = make_ind_device();
    defer gpu::destroy_device(&device)!!;
    IndDrawCtx ctx = make_ind_draw_ctx(&device);
    defer destroy_ind_draw_ctx(&device, &ctx);
    gpu::TextureHandle target = make_ind_target(&device);
    defer gpu::destroy_texture(&device, target)!!;

    // Storage-only: no indirect usage.
    gpu::BufferDesc plain_desc = {
        .size        = 64,
        .usage       = { .storage, .addressable },
        .memory_kind = gpu::MemoryKind.DEVICE,
    };
    gpu::BufferHandle plain = gpu::create_buffer(&device, &plain_desc)!!;
    defer gpu::destroy_buffer(&device, plain)!!;
    gpu::GpuSpan plain_span = gpu::get_buffer_span(&device, plain)!!;

    gpu::begin_frame(&device)!!;
    gpu::GpuSpan verts_span = gpu::alloc_frame_span(&device, 3 * 16, 16)!!;
    write_triangle_verts(&verts_span);
    gpu::GpuSpan frag_span = gpu::alloc_frame_span(&device, 16, 16)!!;
    gpu::GpuSpan args_span = gpu::alloc_frame_span(&device, 64, 16)!!;

    gpu::CommandList cmd = gpu::begin_commands(&device, gpu::QueueKind.GRAPHICS)!!;
    record_target_to_attach(&cmd, target);
    gpu::ColorTargetDesc[1] colors = {{
        .texture = target, .load_op = gpu::LoadOp.CLEAR, .store_op = gpu::StoreOp.STORE,
    }};
    gpu::RenderPassDesc pass = { .colors = colors[..], .width = IND_RT, .height = IND_RT };
    gpu::cmd_begin_render_pass(&cmd, &pass)!!;

    fault no_usage = @catch(gpu::cmd_draw_indirect(
        commands:      &cmd,
        pipeline:      ctx.pipeline,
        vertex_root:   verts_span.gpu,
        fragment_root: frag_span.gpu,
        args:          plain_span,
        draw_count:    1,
    ));
    assert(no_usage == gpu::INVALID_ARGUMENT, "args without indirect usage fault INVALID_ARGUMENT");

    gpu::GpuSpan misaligned = args_span.checked_subspan(2, 32)!!;
    fault align_fault = @catch(gpu::cmd_draw_indirect(
        commands:      &cmd,
        pipeline:      ctx.pipeline,
        vertex_root:   verts_span.gpu,
        fragment_root: frag_span.gpu,
        args:          misaligned,
        draw_count:    1,
    ));
    assert(align_fault == gpu::INVALID_ARGUMENT, "misaligned args offset faults INVALID_ARGUMENT");

    fault overrun = @catch(gpu::cmd_draw_indirect(
        commands:      &cmd,
        pipeline:      ctx.pipeline,
        vertex_root:   verts_span.gpu,
        fragment_root: frag_span.gpu,
        args:          args_span,
        draw_count:    5,
    ));
    assert(overrun == gpu::INVALID_ARGUMENT, "draw_count past the span faults INVALID_ARGUMENT");

    gpu::cmd_end_render_pass(&cmd)!!;
    gpu::end_commands(&device, &cmd)!!;
    gpu::end_frame(&device)!!;
}
