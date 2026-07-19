module gpu_test;

import gpu @public;
import gpu::vk @public;
import vk;
import std::io;

// Headless (lavapipe): GPU-driven indirect execution. A compute pass builds
// arguments through the root-pointer ABI and explicit barriers order access.

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

fn gpu::Device make_ind_device() {
    gpu::DeviceDesc desc = {
        .backend           = gpu::BackendKind.VULKAN,
        .enable_validation = true,
        .application_name  = "gpu_indirect_test",
    };
    return gpu::create_device_from_desc(&desc)!!;
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
        .access    = { .graphics, .compute, .transfer },
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
    upload_test_texture_data(
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
        .access    = { .graphics, .compute, .transfer },
        .width     = IND_RT,
        .height    = IND_RT,
        .format    = gpu::Format.RGBA8_UNORM,
        .usage     = { .color_attach, .transfer_src },
    };
    return gpu::create_texture(device, &desc)!!;
}

fn void write_triangle_verts(gpu::Device* device, gpu::GpuSpan* span) {
    float* verts = (float*)test_span_mapping(device, *span).ptr;
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

fn void record_target_readback(
    gpu::CommandList* cmd,
    gpu::TextureHandle target,
    gpu::GpuSpan* rb_span,
) {
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
        .span = *rb_span,
        .before_stage = gpu::Stage.TRANSFER, .after_stage = gpu::Stage.HOST,
        .before_hazard = gpu::Hazard.TRANSFER_WRITE, .after_hazard = gpu::Hazard.HOST_READ,
    };
    gpu::cmd_buffer_barrier(cmd, &host_read)!!;
}

fn char* center_pixel(gpu::Device* device, gpu::GpuSpan* rb_span) {
    char* pixels = (char*)test_span_mapping(device, *rb_span).ptr;
    return &pixels[(IND_RT / 2 * IND_RT + IND_RT / 2) * 4];
}

fn bool distinct_graphics_compute_families(gpu::Device* device, ZString test_name) {
    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::device_backend_state(*device);
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

fn OwnedSpan alloc_cross_family_indirect_args(
    gpu::Device* device,
    gpu::QueueRoles access,
) {
    OwnedSpan owned = allocate_owned_span(
        device:       device,
        size:         gpu::DispatchIndirectCommand::size,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       access,
        debug_name:   "cross_family_indirect_args",
    )!!;
    uint* words = (uint*)test_span_mapping(device, owned.span).ptr;
    words[0] = 1;
    words[1] = 1;
    words[2] = 1;
    gpu::flush_mapped_span(device, owned.span)!!;
    return owned;
}

fn OwnedSpan alloc_empty_compute_root(
    gpu::Device* device,
    gpu::QueueRoles access,
) {
    OwnedSpan owned = allocate_owned_span(
        device:       device,
        size:         ComputeRoot::size,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       access,
        debug_name:   "cross_family_compute_root",
    )!!;
    ComputeRoot* root = (ComputeRoot*)test_span_mapping(device, owned.span).ptr;
    *root = { .count = 0 };
    gpu::flush_mapped_span(device, owned.span)!!;
    return owned;
}

fn void caller_owned_span_crosses_compute_and_graphics_families_when_available() @test {
    gpu::Device device = make_ind_device();
    defer destroy_test_device(&device)!!;
    if (!distinct_graphics_compute_families(&device, "same-range caller-owned span")) return;

    gpu::ShaderHandle shader;
    gpu::PipelineHandle pipeline = make_cross_family_compute_pipeline(&device, &shader);
    defer gpu::destroy_shader(&device, shader)!!;
    defer gpu::destroy_pipeline(&device, pipeline)!!;
    OwnedSpan args_owned = alloc_cross_family_indirect_args(
        &device,
        { .graphics, .compute },
    );
    defer free_owned_span(&device, &args_owned)!!;
    OwnedSpan compute_root_owned = alloc_empty_compute_root(&device, { .compute });
    defer free_owned_span(&device, &compute_root_owned)!!;
    OwnedSpan graphics_root_owned = alloc_empty_compute_root(&device, { .graphics });
    defer free_owned_span(&device, &graphics_root_owned)!!;

    gpu::CommandList compute = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.COMPUTE)!!)!!;
    defer discard_test_commands(&compute)!!;
    gpu::cmd_dispatch_indirect(
        commands: &compute,
        pipeline: pipeline,
        root:     test_span_address(&device, compute_root_owned.span),
        args:     args_owned.span,
    )!!;
    gpu::ExecutableCommandList compute_executable = gpu::end_commands(&compute)!!;
    defer discard_test_executable(&compute_executable)!!;
    gpu::ExecutableCommandList[1] compute_lists = { compute_executable };
    gpu::SubmitDesc compute_submit = { .command_lists = compute_lists[..] };
    gpu::CompletionPoint compute_completion = submit_test_queue(
        &device,
        gpu::QueueKind.COMPUTE,
        &compute_submit,
    )!!;

    gpu::CommandList graphics = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!)!!;
    defer discard_test_commands(&graphics)!!;
    gpu::cmd_dispatch_indirect(
        commands: &graphics,
        pipeline: pipeline,
        root:     test_span_address(&device, graphics_root_owned.span),
        args:     args_owned.span,
    )!!;
    gpu::ExecutableCommandList graphics_executable = gpu::end_commands(&graphics)!!;
    defer discard_test_executable(&graphics_executable)!!;
    gpu::ExecutableCommandList[1] graphics_lists = { graphics_executable };
    gpu::CompletionPoint[1] waits = { compute_completion };
    gpu::SubmitDesc graphics_submit = {
        .command_lists    = graphics_lists[..],
        .completion_waits = waits[..],
    };
    gpu::CompletionPoint graphics_completion = submit_test_queue(
        &device,
        gpu::QueueKind.GRAPHICS,
        &graphics_submit,
    )!!;
    gpu::wait_completion(graphics_completion, ulong::max)!!;
}

fn void caller_owned_span_reuses_across_queue_families_after_wait() @test {
    gpu::DeviceDesc desc = {
        .backend           = gpu::BackendKind.VULKAN,
        .enable_validation = true,
        .application_name  = "caller_owned_reuse_cross_family",
    };
    gpu::Device device = gpu::create_device_from_desc(&desc)!!;
    defer destroy_test_device(&device)!!;
    if (!distinct_graphics_compute_families(&device, "reused caller-owned span")) return;

    gpu::ShaderHandle shader;
    gpu::PipelineHandle pipeline = make_cross_family_compute_pipeline(&device, &shader);
    defer gpu::destroy_shader(&device, shader)!!;
    defer gpu::destroy_pipeline(&device, pipeline)!!;
    OwnedSpan args_owned = alloc_cross_family_indirect_args(
        &device,
        { .graphics, .compute },
    );
    defer free_owned_span(&device, &args_owned)!!;
    OwnedSpan compute_root_owned = alloc_empty_compute_root(&device, { .compute });
    defer free_owned_span(&device, &compute_root_owned)!!;
    OwnedSpan graphics_root_owned = alloc_empty_compute_root(&device, { .graphics });
    defer free_owned_span(&device, &graphics_root_owned)!!;

    gpu::CommandList compute = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.COMPUTE)!!)!!;
    defer discard_test_commands(&compute)!!;
    gpu::cmd_dispatch_indirect(
        commands: &compute,
        pipeline: pipeline,
        root:     test_span_address(&device, compute_root_owned.span),
        args:     args_owned.span,
    )!!;
    gpu::ExecutableCommandList compute_executable = gpu::end_commands(&compute)!!;
    defer discard_test_executable(&compute_executable)!!;
    gpu::ExecutableCommandList[1] compute_lists = { compute_executable };
    gpu::SubmitDesc compute_submit = { .command_lists = compute_lists[..] };
    gpu::CompletionPoint compute_completion = submit_test_queue(
        &device,
        gpu::QueueKind.COMPUTE,
        &compute_submit,
    )!!;
    gpu::wait_completion(compute_completion)!!;

    gpu::CommandList graphics = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!)!!;
    defer discard_test_commands(&graphics)!!;
    gpu::cmd_dispatch_indirect(
        commands: &graphics,
        pipeline: pipeline,
        root:     test_span_address(&device, graphics_root_owned.span),
        args:     args_owned.span,
    )!!;
    gpu::ExecutableCommandList graphics_executable = gpu::end_commands(&graphics)!!;
    defer discard_test_executable(&graphics_executable)!!;
    gpu::ExecutableCommandList[1] graphics_lists = { graphics_executable };
    gpu::SubmitDesc graphics_submit = { .command_lists = graphics_lists[..] };
    gpu::CompletionPoint graphics_completion = submit_test_queue(
        &device,
        gpu::QueueKind.GRAPHICS,
        &graphics_submit,
    )!!;
    gpu::wait_completion(graphics_completion)!!;
}
fn void compute_generated_indexed_indirect_draw() @test {
    gpu::Device device = make_ind_device();
    defer destroy_test_device(&device)!!;
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

    OwnedSpan args = allocate_owned_span(
        device:       &device,
        size:         2 * gpu::DrawIndexedIndirectCommand::size,
        memory_class: gpu::MemoryClass.GPU_PRIVATE,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "indexed_indirect_args",
    )!!;
    defer free_owned_span(&device, &args)!!;
    gpu::GpuSpan args_span = args.span;

    OwnedSpan args_readback = allocate_owned_span(
        device:       &device,
        size:         2 * gpu::DrawIndexedIndirectCommand::size,
        memory_class: gpu::MemoryClass.CPU_READ,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "indexed_args_readback",
    )!!;
    defer free_owned_span(&device, &args_readback)!!;

    OwnedSpan readback = allocate_owned_span(
        device:       &device,
        size:         IND_RT_BYTES,
        memory_class: gpu::MemoryClass.CPU_READ,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "indexed_indirect_readback",
    )!!;
    defer free_owned_span(&device, &readback)!!;
    gpu::GpuSpan rb_span = readback.span;
    OwnedSpan scratch_owned = allocate_owned_span(
        device:       &device,
        size:         64,
        memory_class: gpu::MemoryClass.GPU_PRIVATE,
        access:       { .graphics },
        debug_name:   "indexed_scratch",
    )!!;
    defer free_owned_span(&device, &scratch_owned)!!;
    gpu::GpuSpan scratch = scratch_owned.span;
    OwnedSpan verts_owned = allocate_owned_span(
        device:       &device,
        size:         3 * 16,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       { .graphics },
        debug_name:   "indexed_vertices",
    )!!;
    defer free_owned_span(&device, &verts_owned)!!;
    gpu::GpuSpan verts_span = verts_owned.span;
    write_triangle_verts(&device, &verts_span);
    gpu::flush_mapped_span(&device, verts_span)!!;
    OwnedSpan index_owned = allocate_owned_span(
        device:       &device,
        size:         3 * 4,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       { .graphics },
        debug_name:   "indexed_indices",
    )!!;
    defer free_owned_span(&device, &index_owned)!!;
    gpu::GpuSpan index_span = index_owned.span;
    uint* indices = (uint*)test_span_mapping(&device, index_span).ptr;
    indices[0] = 0; indices[1] = 1; indices[2] = 2;
    gpu::flush_mapped_span(&device, index_span)!!;
    OwnedSpan frag_owned = allocate_owned_span(
        device:       &device,
        size:         FragRoot::size,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       { .graphics },
        debug_name:   "indexed_fragment_root",
    )!!;
    defer free_owned_span(&device, &frag_owned)!!;
    gpu::GpuSpan frag_span = frag_owned.span;
    FragRoot* frag_root = (FragRoot*)test_span_mapping(&device, frag_span).ptr;
    frag_root.texture_index = ctx.tex_index;
    frag_root.sampler_index = ctx.smp_index;
    gpu::flush_mapped_span(&device, frag_span)!!;

    OwnedSpan root_owned = allocate_owned_span(
        device:       &device,
        size:         BuildRoot::size,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       { .graphics },
        debug_name:   "indexed_build_root",
    )!!;
    defer free_owned_span(&device, &root_owned)!!;
    gpu::GpuSpan root_span = root_owned.span;
    BuildRoot* root = (BuildRoot*)test_span_mapping(&device, root_span).ptr;
    root.indexed_args_gpu = test_span_address(&device, args_span);
    root.plain_args_gpu   = test_span_address(&device, scratch);
    root.dispatch_args_gpu = (gpu::GpuAddress)((ulong)test_span_address(&device, scratch) + 16);
    root.count_gpu        = (gpu::GpuAddress)((ulong)test_span_address(&device, scratch) + 32);
    root.index_count      = 3;
    root.vertex_count     = 3;
    root.dispatch_x       = 1;
    root.count_value      = 1;
    gpu::flush_mapped_span(&device, root_span)!!;

    gpu::CommandList cmd = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!)!!;
    defer discard_test_commands(&cmd)!!;
    gpu::cmd_dispatch(
        commands: &cmd,
        pipeline: build_pipeline,
        root:     test_span_address(&device, root_span),
        groups:   { 1, 1, 1 },
    )!!;

    gpu::BufferBarrier args_ready = {
        .span          = args_span,
        .before_stage  = gpu::Stage.COMPUTE_SHADER,
        .after_stage   = gpu::Stage.INDIRECT_COMMAND,
        .before_hazard = gpu::Hazard.SHADER_WRITE,
        .after_hazard  = gpu::Hazard.INDIRECT_READ,
    };
    gpu::cmd_buffer_barrier(&cmd, &args_ready)!!;

    // Read the GPU-written arguments through ordinary caller-owned memory.
    gpu::BufferBarrier args_to_copy = {
        .span          = args_span,
        .before_stage  = gpu::Stage.COMPUTE_SHADER,
        .after_stage   = gpu::Stage.TRANSFER,
        .before_hazard = gpu::Hazard.SHADER_WRITE,
        .after_hazard  = gpu::Hazard.TRANSFER_READ,
    };
    gpu::cmd_buffer_barrier(&cmd, &args_to_copy)!!;
    gpu::BufferCopyDesc args_copy = {
        .src = args_span,
        .dst = args_readback.span,
    };
    gpu::cmd_copy_buffer(&cmd, &args_copy)!!;
    gpu::BufferBarrier args_host_read = {
        .span          = args_readback.span,
        .before_stage  = gpu::Stage.TRANSFER,
        .after_stage   = gpu::Stage.HOST,
        .before_hazard = gpu::Hazard.TRANSFER_WRITE,
        .after_hazard  = gpu::Hazard.HOST_READ,
    };
    gpu::cmd_buffer_barrier(&cmd, &args_host_read)!!;

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
        vertex_root:   test_span_address(&device, verts_span),
        fragment_root: test_span_address(&device, frag_span),
        args:          args_span,
        draw_count:    1,
        index_span:    index_span,
    )!!;
    gpu::cmd_end_render_pass(&cmd)!!;
    record_target_readback(&cmd, target, &rb_span);

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
    gpu::invalidate_mapped_span(&device, readback.span)!!;
    char* center = center_pixel(&device, &rb_span);
    assert(center[0] == (char)255 && center[1] == (char)255 && center[2] == (char)255,
        "compute-generated indexed indirect draw renders the triangle");

    gpu::invalidate_mapped_span(&device, args_readback.span)!!;
    gpu::DrawIndexedIndirectCommand* got =
        (gpu::DrawIndexedIndirectCommand*)test_span_mapping(
            &device,
            args_readback.span,
        ).ptr;
    for (uint i = 0; i < 2; i++) {
        assert(got[i].index_count == 3 && got[i].instance_count == 1,
            "GPU-written args read back intact");
        assert(got[i].first_index == 0 && got[i].vertex_offset == 0
            && got[i].first_instance == 0,
            "GPU-written arg tail fields read back intact");
    }
}

fn void plain_indirect_draw_from_caller_owned_span() @test {
    gpu::Device device = make_ind_device();
    defer destroy_test_device(&device)!!;
    IndDrawCtx ctx = make_ind_draw_ctx(&device);
    defer destroy_ind_draw_ctx(&device, &ctx);
    gpu::TextureHandle target = make_ind_target(&device);
    defer gpu::destroy_texture(&device, target)!!;

    OwnedSpan readback = allocate_owned_span(
        device:       &device,
        size:         IND_RT_BYTES,
        memory_class: gpu::MemoryClass.CPU_READ,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "plain_indirect_readback",
    )!!;
    defer free_owned_span(&device, &readback)!!;
    gpu::GpuSpan rb_span = readback.span;
    OwnedSpan verts_owned = allocate_owned_span(
        device:       &device,
        size:         3 * 16,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       { .graphics },
        debug_name:   "plain_indirect_vertices",
    )!!;
    defer free_owned_span(&device, &verts_owned)!!;
    gpu::GpuSpan verts_span = verts_owned.span;
    write_triangle_verts(&device, &verts_span);
    gpu::flush_mapped_span(&device, verts_span)!!;
    OwnedSpan frag_owned = allocate_owned_span(
        device:       &device,
        size:         FragRoot::size,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       { .graphics },
        debug_name:   "plain_indirect_fragment_root",
    )!!;
    defer free_owned_span(&device, &frag_owned)!!;
    gpu::GpuSpan frag_span = frag_owned.span;
    FragRoot* frag_root = (FragRoot*)test_span_mapping(&device, frag_span).ptr;
    frag_root.texture_index = ctx.tex_index;
    frag_root.sampler_index = ctx.smp_index;
    gpu::flush_mapped_span(&device, frag_span)!!;

    // Caller-owned allocations include indirect usage.
    OwnedSpan args_owned = allocate_owned_span(
        device:       &device,
        size:         gpu::DrawIndirectCommand::size,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       { .graphics },
        debug_name:   "plain_indirect_args",
    )!!;
    defer free_owned_span(&device, &args_owned)!!;
    gpu::GpuSpan args_span = args_owned.span;
    gpu::DrawIndirectCommand* args = (gpu::DrawIndirectCommand*)test_span_mapping(&device, args_span).ptr;
    *args = { .vertex_count = 3, .instance_count = 1 };
    gpu::flush_mapped_span(&device, args_span)!!;

    gpu::CommandList cmd = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!)!!;
    defer discard_test_commands(&cmd)!!;
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
        vertex_root:   test_span_address(&device, verts_span),
        fragment_root: test_span_address(&device, frag_span),
        args:          args_span,
        draw_count:    1,
    )!!;
    gpu::cmd_end_render_pass(&cmd)!!;
    record_target_readback(&cmd, target, &rb_span);
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

    gpu::invalidate_mapped_span(&device, readback.span)!!;
    char* center = center_pixel(&device, &rb_span);
    assert(center[0] == (char)255, "non-indexed indirect draw renders");
}

fn void indirect_dispatch_executes_gpu_counts() @test {
    gpu::Device device = make_ind_device();
    defer destroy_test_device(&device)!!;

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
    OwnedSpan io_owned = allocate_owned_span(
        device:       &device,
        size:         COUNT * 4 * 2,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       { .graphics },
        debug_name:   "dispatch_io",
    )!!;
    defer free_owned_span(&device, &io_owned)!!;
    gpu::GpuSpan io_span = io_owned.span;
    float* values = (float*)test_span_mapping(&device, io_span).ptr;
    for (uint i = 0; i < COUNT; i++) values[i] = (float)i;

    OwnedSpan scratch_owned = allocate_owned_span(
        device:       &device,
        size:         64,
        memory_class: gpu::MemoryClass.GPU_PRIVATE,
        access:       { .graphics },
        debug_name:   "dispatch_scratch",
    )!!;
    defer free_owned_span(&device, &scratch_owned)!!;
    gpu::GpuSpan scratch = scratch_owned.span;
    gpu::GpuSpan dispatch_args = scratch.checked_subspan(16, gpu::DispatchIndirectCommand::size)!!;

    OwnedSpan build_root_owned = allocate_owned_span(
        device:       &device,
        size:         BuildRoot::size,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       { .graphics },
        debug_name:   "dispatch_build_root",
    )!!;
    defer free_owned_span(&device, &build_root_owned)!!;
    gpu::GpuSpan build_root_span = build_root_owned.span;
    BuildRoot* build_root = (BuildRoot*)test_span_mapping(&device, build_root_span).ptr;
    build_root.indexed_args_gpu = test_span_address(&device, scratch);
    build_root.plain_args_gpu   = test_span_address(&device, scratch);
    build_root.dispatch_args_gpu = test_span_address(&device, dispatch_args);
    build_root.count_gpu        = (gpu::GpuAddress)((ulong)test_span_address(&device, scratch) + 48);
    build_root.index_count      = 3;
    build_root.vertex_count     = 3;
    build_root.dispatch_x       = (COUNT + 63) / 64;
    build_root.count_value      = 1;
    gpu::flush_mapped_span(&device, build_root_span)!!;

    OwnedSpan double_root_owned = allocate_owned_span(
        device:       &device,
        size:         ComputeRoot::size,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       { .graphics },
        debug_name:   "dispatch_double_root",
    )!!;
    defer free_owned_span(&device, &double_root_owned)!!;
    gpu::GpuSpan double_root_span = double_root_owned.span;
    ComputeRoot* double_root = (ComputeRoot*)test_span_mapping(&device, double_root_span).ptr;
    double_root.input_gpu  = test_span_address(&device, io_span);
    double_root.output_gpu = (gpu::GpuAddress)((ulong)test_span_address(&device, io_span) + COUNT * 4);
    double_root.count      = COUNT;
    gpu::flush_mapped_span(&device, io_span)!!;
    gpu::flush_mapped_span(&device, double_root_span)!!;

    gpu::CommandList cmd = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!)!!;
    defer discard_test_commands(&cmd)!!;
    gpu::cmd_dispatch(
        commands: &cmd,
        pipeline: build_pipeline,
        root:     test_span_address(&device, build_root_span),
        groups:   { 1, 1, 1 },
    )!!;
    gpu::BufferBarrier args_ready = {
        .span          = dispatch_args,
        .before_stage  = gpu::Stage.COMPUTE_SHADER,
        .after_stage   = gpu::Stage.INDIRECT_COMMAND,
        .before_hazard = gpu::Hazard.SHADER_WRITE,
        .after_hazard  = gpu::Hazard.INDIRECT_READ,
    };
    gpu::cmd_buffer_barrier(&cmd, &args_ready)!!;
    gpu::cmd_dispatch_indirect(
        commands: &cmd,
        pipeline: double_pipeline,
        root:     test_span_address(&device, double_root_span),
        args:     dispatch_args,
    )!!;
    gpu::BufferBarrier host_read = {
        .span = io_span,
        .before_stage = gpu::Stage.COMPUTE_SHADER, .after_stage = gpu::Stage.HOST,
        .before_hazard = gpu::Hazard.SHADER_WRITE, .after_hazard = gpu::Hazard.HOST_READ,
    };
    gpu::cmd_buffer_barrier(&cmd, &host_read)!!;
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
    gpu::invalidate_mapped_span(&device, io_span)!!;

    float* doubled = &values[COUNT];
    for (uint i = 0; i < COUNT; i++) {
        assert(doubled[i] == (float)i * 2.0f, "indirect dispatch ran with GPU-written group counts");
    }
}

fn void count_buffer_limits_draws() @test {
    gpu::Device device = make_ind_device();
    defer destroy_test_device(&device)!!;
    if (!gpu::get_device_caps(&device)!!.draw_indirect_count) return;

    IndDrawCtx ctx = make_ind_draw_ctx(&device);
    defer destroy_ind_draw_ctx(&device, &ctx);
    gpu::TextureHandle target_zero = make_ind_target(&device);
    defer gpu::destroy_texture(&device, target_zero)!!;
    gpu::TextureHandle target_one = make_ind_target(&device);
    defer gpu::destroy_texture(&device, target_one)!!;

    OwnedSpan rb_zero = allocate_owned_span(
        device:       &device,
        size:         IND_RT_BYTES,
        memory_class: gpu::MemoryClass.CPU_READ,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "count_zero_readback",
    )!!;
    defer free_owned_span(&device, &rb_zero)!!;
    gpu::GpuSpan rb_zero_span = rb_zero.span;
    OwnedSpan rb_one = allocate_owned_span(
        device:       &device,
        size:         IND_RT_BYTES,
        memory_class: gpu::MemoryClass.CPU_READ,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "count_one_readback",
    )!!;
    defer free_owned_span(&device, &rb_one)!!;
    gpu::GpuSpan rb_one_span = rb_one.span;
    OwnedSpan verts_owned = allocate_owned_span(
        device:       &device,
        size:         3 * 16,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       { .graphics },
        debug_name:   "count_vertices",
    )!!;
    defer free_owned_span(&device, &verts_owned)!!;
    gpu::GpuSpan verts_span = verts_owned.span;
    write_triangle_verts(&device, &verts_span);
    gpu::flush_mapped_span(&device, verts_span)!!;
    OwnedSpan index_owned = allocate_owned_span(
        device:       &device,
        size:         3 * 4,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       { .graphics },
        debug_name:   "count_indices",
    )!!;
    defer free_owned_span(&device, &index_owned)!!;
    gpu::GpuSpan index_span = index_owned.span;
    uint* indices = (uint*)test_span_mapping(&device, index_span).ptr;
    indices[0] = 0; indices[1] = 1; indices[2] = 2;
    gpu::flush_mapped_span(&device, index_span)!!;
    OwnedSpan frag_owned = allocate_owned_span(
        device:       &device,
        size:         FragRoot::size,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       { .graphics },
        debug_name:   "count_fragment_root",
    )!!;
    defer free_owned_span(&device, &frag_owned)!!;
    gpu::GpuSpan frag_span = frag_owned.span;
    FragRoot* frag_root = (FragRoot*)test_span_mapping(&device, frag_span).ptr;
    frag_root.texture_index = ctx.tex_index;
    frag_root.sampler_index = ctx.smp_index;
    gpu::flush_mapped_span(&device, frag_span)!!;

    OwnedSpan args_owned = allocate_owned_span(
        device:       &device,
        size:         gpu::DrawIndexedIndirectCommand::size,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       { .graphics },
        debug_name:   "count_indirect_args",
    )!!;
    defer free_owned_span(&device, &args_owned)!!;
    gpu::GpuSpan args_span = args_owned.span;
    gpu::DrawIndexedIndirectCommand* args = (gpu::DrawIndexedIndirectCommand*)test_span_mapping(&device, args_span).ptr;
    *args = { .index_count = 3, .instance_count = 1 };
    gpu::flush_mapped_span(&device, args_span)!!;

    OwnedSpan counts_owned = allocate_owned_span(
        device:       &device,
        size:         8,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       { .graphics },
        debug_name:   "draw_counts",
    )!!;
    defer free_owned_span(&device, &counts_owned)!!;
    gpu::GpuSpan counts = counts_owned.span;
    uint* count_values = (uint*)test_span_mapping(&device, counts).ptr;
    count_values[0] = 0;
    count_values[1] = 1;
    gpu::flush_mapped_span(&device, counts)!!;
    gpu::GpuSpan count_zero = counts.checked_subspan(0, 4)!!;
    gpu::GpuSpan count_one = counts.checked_subspan(4, 4)!!;

    gpu::CommandList cmd = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!)!!;
    defer discard_test_commands(&cmd)!!;
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
            vertex_root:    test_span_address(&device, verts_span),
            fragment_root:  test_span_address(&device, frag_span),
            args:           args_span,
            count_span:     count_spans[i],
            max_draw_count: 1,
            index_span:     index_span,
        )!!;
        gpu::cmd_end_render_pass(&cmd)!!;
        record_target_readback(&cmd, targets[i], &rb_spans[i]);
    }
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

    gpu::invalidate_mapped_span(&device, rb_zero.span)!!;
    char* zero_center = center_pixel(&device, &rb_zero_span);
    assert(zero_center[0] == (char)8 && zero_center[1] == (char)16,
        "GPU count of zero draws nothing");
    gpu::invalidate_mapped_span(&device, rb_one.span)!!;
    char* one_center = center_pixel(&device, &rb_one_span);
    assert(one_center[0] == (char)255, "GPU count of one draws the triangle");
}

fn void discard_consumes_active_render_pass_recording() @test {
    gpu::Device device = make_ind_device();
    gpu::TextureHandle target = make_ind_target(&device);

    gpu::CommandList commands = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!)!!;
    record_target_to_attach(&commands, target);
    gpu::ColorTargetDesc[1] colors = {{
        .texture  = target,
        .load_op  = gpu::LoadOp.CLEAR,
        .store_op = gpu::StoreOp.STORE,
    }};
    gpu::RenderPassDesc pass = {
        .colors = colors[..],
        .width  = IND_RT,
        .height = IND_RT,
    };
    gpu::cmd_begin_render_pass(&commands, &pass)!!;

    gpu::discard_commands(&commands)!!;
    assert(!commands.handle.is_valid(),
        "discard consumes a recording with an active render pass");
    gpu::destroy_texture(&device, target)!!;
    gpu::destroy_device(&device)!!;
}

fn void indirect_argument_validation() @test {
    gpu::Device device = make_ind_device();
    defer destroy_test_device(&device)!!;
    IndDrawCtx ctx = make_ind_draw_ctx(&device);
    defer destroy_ind_draw_ctx(&device, &ctx);
    gpu::TextureHandle target = make_ind_target(&device);
    defer gpu::destroy_texture(&device, target)!!;

    vk::BufferDesc plain_desc = {
        .size        = 64,
        .usage       = { .storage, .addressable },
        .memory_kind = gpu::MemoryKind.DEVICE,
        .access      = { .graphics, .compute, .transfer },
    };
    vk::BufferHandle plain = vk::vk_create_buffer(&device, &plain_desc)!!;
    defer vk::vk_destroy_buffer(&device, plain)!!;
    gpu::GpuSpan plain_span = vk::vk_get_buffer_span(&device, plain)!!;
    OwnedSpan verts_owned = allocate_owned_span(
        device:       &device,
        size:         3 * 16,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       { .graphics },
        debug_name:   "validation_vertices",
    )!!;
    defer free_owned_span(&device, &verts_owned)!!;
    gpu::GpuSpan verts_span = verts_owned.span;
    write_triangle_verts(&device, &verts_span);
    gpu::flush_mapped_span(&device, verts_span)!!;
    OwnedSpan frag_owned = allocate_owned_span(
        device:       &device,
        size:         16,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       { .graphics },
        debug_name:   "validation_fragment_root",
    )!!;
    defer free_owned_span(&device, &frag_owned)!!;
    gpu::GpuSpan frag_span = frag_owned.span;
    OwnedSpan args_owned = allocate_owned_span(
        device:       &device,
        size:         64,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       { .graphics },
        debug_name:   "validation_indirect_args",
    )!!;
    defer free_owned_span(&device, &args_owned)!!;
    gpu::GpuSpan args_span = args_owned.span;

    gpu::CommandList cmd = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!)!!;
    defer discard_test_commands(&cmd)!!;
    record_target_to_attach(&cmd, target);
    gpu::ColorTargetDesc[1] colors = {{
        .texture = target, .load_op = gpu::LoadOp.CLEAR, .store_op = gpu::StoreOp.STORE,
    }};
    gpu::RenderPassDesc pass = { .colors = colors[..], .width = IND_RT, .height = IND_RT };
    gpu::cmd_begin_render_pass(&cmd, &pass)!!;

    fault no_usage = @catch(gpu::cmd_draw_indirect(
        commands:      &cmd,
        pipeline:      ctx.pipeline,
        vertex_root:   test_span_address(&device, verts_span),
        fragment_root: test_span_address(&device, frag_span),
        args:          plain_span,
        draw_count:    1,
    ));
    assert(no_usage == gpu::INVALID_ARGUMENT, "args without indirect usage fault INVALID_ARGUMENT");

    gpu::GpuSpan misaligned = args_span.checked_subspan(2, 32)!!;
    fault align_fault = @catch(gpu::cmd_draw_indirect(
        commands:      &cmd,
        pipeline:      ctx.pipeline,
        vertex_root:   test_span_address(&device, verts_span),
        fragment_root: test_span_address(&device, frag_span),
        args:          misaligned,
        draw_count:    1,
    ));
    assert(align_fault == gpu::INVALID_ARGUMENT, "misaligned args offset faults INVALID_ARGUMENT");

    fault overrun = @catch(gpu::cmd_draw_indirect(
        commands:      &cmd,
        pipeline:      ctx.pipeline,
        vertex_root:   test_span_address(&device, verts_span),
        fragment_root: test_span_address(&device, frag_span),
        args:          args_span,
        draw_count:    5,
    ));
    assert(overrun == gpu::INVALID_ARGUMENT, "draw_count past the span faults INVALID_ARGUMENT");

    gpu::cmd_end_render_pass(&cmd)!!;
    gpu::ExecutableCommandList cmd_executable = gpu::end_commands(&cmd)!!;
    defer discard_test_executable(&cmd_executable)!!;
}
