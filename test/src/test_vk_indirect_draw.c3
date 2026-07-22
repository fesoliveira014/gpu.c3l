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
const char[*] IND_GENERATED_WORK_SPIRV = $embed("shaders/generated_work.comp.spv");

const uint IND_RT = 16;
const usz IND_RT_BYTES = IND_RT * IND_RT * 4;

struct IndDrawCtx {
    gpu::PipelineHandle pipeline;
    gpu::TextureHandle  pattern;
    gpu::TextureView    tex_view;
    gpu::TextureIndex   tex_index;
    gpu::SamplerIndex   smp_index;
}

struct GeneratedRequirementsCapture {
    vk::PFN_vkGetGeneratedCommandsMemoryRequirementsEXT original;
    uint calls;
    uint[3] max_sequence_counts;
}

GeneratedRequirementsCapture* generated_requirements_capture;

fn void capture_generated_requirements(
    vk::Device device,
    vk::GeneratedCommandsMemoryRequirementsInfoEXT* info,
    vk::MemoryRequirements2* requirements,
) {
    GeneratedRequirementsCapture* capture = generated_requirements_capture;
    uint call = capture.calls++;
    if (call < capture.max_sequence_counts.len) {
        capture.max_sequence_counts[call] = info.max_sequence_count;
    }
    capture.original(device, info, requirements);
}

fn gpu::Device make_ind_device() {
    gpu::RuntimeDesc desc = {
        .backend           = gpu::BackendKind.VULKAN,
        .contract_validation      = gpu::ContractValidation.FULL,
        .track_resource_lifetimes = true,
        .enable_vulkan_validation = true,
        .application_name  = "gpu_indirect_test",
    };
    return create_test_device(&desc)!!;
}

fn gpu::Device make_generated_work_device() {
    gpu::RuntimeDesc desc = {
        .backend          = gpu::BackendKind.VULKAN,
        .application_name = "gpu_generated_work_test",
    };
    return create_test_device(&desc)!!;
}

<* Textured draw context: offscreen vertex/fragment code, one graphics pipeline, and a 1x1
   white pattern feeding the fragment heap path. *>
fn IndDrawCtx make_ind_draw_ctx(gpu::Device* device) {
    IndDrawCtx ctx;
    gpu::ShaderDesc vert_sd = { .stage = gpu::ShaderStage.VERTEX, .spirv = IND_VERT_SPIRV[..], .entry_point = "main" };
    gpu::ShaderCode vert = gpu::prepare_shader_code(&vert_sd)!!;
    gpu::ShaderDesc frag_sd = { .stage = gpu::ShaderStage.FRAGMENT, .spirv = IND_FRAG_SPIRV[..], .entry_point = "main" };
    gpu::ShaderCode frag = gpu::prepare_shader_code(&frag_sd)!!;

    gpu::ColorTargetState[1] pipeline_colors = {{
        .format     = gpu::Format.RGBA8_UNORM,
        .write_mask = gpu::COLOR_WRITE_ALL,
    }};
    gpu::GraphicsPipelineDesc pipe_desc = {
        .vertex_shader   = vert,
        .fragment_shader = frag,
        .colors          = pipeline_colors[..],
    };
    ctx.pipeline = gpu::create_graphics_pipeline(device, &pipe_desc)!!;

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

fn void destroy_ind_draw_ctx(gpu::Device* device, IndDrawCtx* ctx) {
    gpu::destroy_texture_view(device, ctx.tex_view)!!;
    gpu::destroy_texture(device, ctx.pattern)!!;
    gpu::destroy_pipeline(device, ctx.pipeline)!!;
}

fn gpu::TextureHandle make_ind_target(gpu::Device* device) {
    gpu::TextureDesc desc = {
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

fn void record_target_readback(
    gpu::CommandList* cmd,
    gpu::TextureHandle target,
    gpu::GpuSpan* rb_span,
) {
    gpu::TextureBarrier to_src = {
        .texture = target,
        .before  = {
            .layout = gpu::TextureLayout.COLOR_ATTACHMENT,
            .stages = { .color_output },
            .access = { .read, .write },
        },
        .after   = {
            .layout = gpu::TextureLayout.TRANSFER_SOURCE,
            .stages = { .transfer },
            .access = { .read },
        },
    };
    gpu::cmd_texture_barrier(cmd, &to_src)!!;
    gpu::TextureBufferCopyDesc copy = { .texture = target, .dst = *rb_span };
    gpu::cmd_copy_texture_to_buffer(cmd, &copy)!!;
    gpu::Barrier host_read = {
        .before = { .transfer = true },
        .after = { .host = true },
    };
    gpu::cmd_barrier(cmd, &host_read)!!;
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

fn gpu::PipelineHandle make_cross_family_compute_pipeline(gpu::Device* device) {
    gpu::ShaderDesc shader_desc = {
        .stage       = gpu::ShaderStage.COMPUTE,
        .spirv       = IND_DOUBLE_SPIRV[..],
        .entry_point = "main",
    };
    gpu::ShaderCode shader = gpu::prepare_shader_code(&shader_desc)!!;
    gpu::ComputePipelineDesc pipeline_desc = {
        .shader = shader,
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

    gpu::PipelineHandle pipeline = make_cross_family_compute_pipeline(&device);
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
    gpu::Queue compute_queue = gpu::get_queue(&device, gpu::QueueKind.COMPUTE)!!;
    gpu::CommandAllocator compute_allocator = gpu::create_command_allocator(&device, compute_queue)!!;
    defer gpu::destroy_command_allocator(&compute_allocator)!!;
    gpu::Queue graphics_queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator graphics_allocator = gpu::create_command_allocator(&device, graphics_queue)!!;
    defer gpu::destroy_command_allocator(&graphics_allocator)!!;

    gpu::CommandList compute = gpu::begin_commands(&compute_allocator)!!;
    defer discard_test_commands(&compute)!!;
    bind_test_compute_pipeline(&compute, pipeline);
    gpu::cmd_dispatch_indirect(
        commands: &compute,
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

    gpu::CommandList graphics = gpu::begin_commands(&graphics_allocator)!!;
    defer discard_test_commands(&graphics)!!;
    bind_test_compute_pipeline(&graphics, pipeline);
    gpu::cmd_dispatch_indirect(
        commands: &graphics,
        root:     test_span_address(&device, graphics_root_owned.span),
        args:     args_owned.span,
    )!!;
    gpu::ExecutableCommandList graphics_executable = gpu::end_commands(&graphics)!!;
    defer discard_test_executable(&graphics_executable)!!;
    gpu::ExecutableCommandList[1] graphics_lists = { graphics_executable };
    gpu::CompletionWait[1] waits = {{
        .point  = compute_completion,
        .before = { .all },
    }};
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
    gpu::RuntimeDesc desc = {
        .backend           = gpu::BackendKind.VULKAN,
        .contract_validation      = gpu::ContractValidation.FULL,
        .track_resource_lifetimes = true,
        .enable_vulkan_validation = true,
        .application_name  = "caller_owned_reuse_cross_family",
    };
    gpu::Device device = create_test_device(&desc)!!;
    defer destroy_test_device(&device)!!;
    if (!distinct_graphics_compute_families(&device, "reused caller-owned span")) return;

    gpu::PipelineHandle pipeline = make_cross_family_compute_pipeline(&device);
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
    gpu::Queue compute_queue = gpu::get_queue(&device, gpu::QueueKind.COMPUTE)!!;
    gpu::CommandAllocator compute_allocator = gpu::create_command_allocator(&device, compute_queue)!!;
    defer gpu::destroy_command_allocator(&compute_allocator)!!;
    gpu::Queue graphics_queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator graphics_allocator = gpu::create_command_allocator(&device, graphics_queue)!!;
    defer gpu::destroy_command_allocator(&graphics_allocator)!!;

    gpu::CommandList compute = gpu::begin_commands(&compute_allocator)!!;
    defer discard_test_commands(&compute)!!;
    bind_test_compute_pipeline(&compute, pipeline);
    gpu::cmd_dispatch_indirect(
        commands: &compute,
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

    gpu::CommandList graphics = gpu::begin_commands(&graphics_allocator)!!;
    defer discard_test_commands(&graphics)!!;
    bind_test_compute_pipeline(&graphics, pipeline);
    gpu::cmd_dispatch_indirect(
        commands: &graphics,
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
    gpu::AttachmentViewHandle target_view = create_test_attachment_view(
        &device,
        target,
    )!!;
    defer destroy_test_attachment_view(&device, &target_view)!!;

    gpu::ShaderDesc build_sd = { .stage = gpu::ShaderStage.COMPUTE, .spirv = IND_BUILD_SPIRV[..], .entry_point = "main" };
    gpu::ShaderCode build_shader = gpu::prepare_shader_code(&build_sd)!!;
    gpu::ComputePipelineDesc build_desc = { .shader = build_shader };
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

    gpu::Queue command_queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator command_allocator = gpu::create_command_allocator(&device, command_queue)!!;
    defer gpu::destroy_command_allocator(&command_allocator)!!;
    gpu::CommandList cmd = gpu::begin_commands(&command_allocator)!!;
    defer discard_test_commands(&cmd)!!;
    bind_test_compute_pipeline(&cmd, build_pipeline);
    gpu::cmd_dispatch(
        commands: &cmd,
        root:     test_span_address(&device, root_span),
        groups:   { 1, 1, 1 },
    )!!;

    gpu::Barrier args_ready = {
        .before = { .compute = true },
        .after = { .compute = true },
        .hazards = { .draw_arguments = true },
    };
    gpu::cmd_barrier(&cmd, &args_ready)!!;

    // Read the GPU-written arguments through ordinary caller-owned memory.
    gpu::Barrier args_to_copy = {
        .before = { .compute = true },
        .after = { .transfer = true },
    };
    gpu::cmd_barrier(&cmd, &args_to_copy)!!;
    gpu::BufferCopyDesc args_copy = {
        .src = args_span,
        .dst = args_readback.span,
    };
    gpu::cmd_copy_buffer(&cmd, &args_copy)!!;
    gpu::Barrier args_host_read = {
        .before = { .transfer = true },
        .after = { .host = true },
    };
    gpu::cmd_barrier(&cmd, &args_host_read)!!;

    record_target_to_attach(&cmd, target);
    gpu::ColorTargetDesc[1] colors = {{
        .view = target_view, .load_op = gpu::LoadOp.CLEAR, .store_op = gpu::StoreOp.STORE,
        .clear = { .rgba = { 8.0f / 255.0f, 16.0f / 255.0f, 32.0f / 255.0f, 1.0f } },
    }};
    gpu::RenderPassDesc pass = { .colors = colors[..], .width = IND_RT, .height = IND_RT };
    gpu::cmd_begin_render_pass(&cmd, &pass)!!;
    bind_test_graphics_pipeline(&cmd, ctx.pipeline);
    gpu::cmd_draw_indexed_indirect(
        commands:      &cmd,
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

fn void generated_work_executes_gpu_written_draws_and_dispatches() @test {
    gpu::Device device = make_generated_work_device();
    defer destroy_test_device(&device)!!;
    gpu::DeviceCaps caps = gpu::get_device_caps(&device)!!;
    if (!caps.generated_work) {
        io::printfn(
            "generated-work: NOT EXERCISED (generated work unavailable)",
        );
        return;
    }

    IndDrawCtx draw = make_ind_draw_ctx(&device);
    defer destroy_ind_draw_ctx(&device, &draw);
    gpu::TextureHandle plain_target = make_ind_target(&device);
    defer gpu::destroy_texture(&device, plain_target)!!;
    gpu::AttachmentViewHandle plain_view = create_test_attachment_view(
        &device,
        plain_target,
    )!!;
    defer destroy_test_attachment_view(&device, &plain_view)!!;
    gpu::TextureHandle indexed_target = make_ind_target(&device);
    defer gpu::destroy_texture(&device, indexed_target)!!;
    gpu::AttachmentViewHandle indexed_view = create_test_attachment_view(
        &device,
        indexed_target,
    )!!;
    defer destroy_test_attachment_view(&device, &indexed_view)!!;
    gpu::TextureHandle null_target = make_ind_target(&device);
    defer gpu::destroy_texture(&device, null_target)!!;
    gpu::AttachmentViewHandle null_view = create_test_attachment_view(
        &device,
        null_target,
    )!!;
    defer destroy_test_attachment_view(&device, &null_view)!!;

    gpu::ShaderDesc build_shader_desc = {
        .stage       = gpu::ShaderStage.COMPUTE,
        .spirv       = IND_GENERATED_WORK_SPIRV[..],
        .entry_point = "main",
    };
    gpu::ShaderCode build_shader = gpu::prepare_shader_code(&build_shader_desc)!!;
    gpu::ComputePipelineDesc build_pipeline_desc = {
        .shader = build_shader,
    };
    gpu::PipelineHandle build_pipeline = gpu::create_compute_pipeline(
        &device,
        &build_pipeline_desc,
    )!!;
    defer gpu::destroy_pipeline(&device, build_pipeline)!!;
    gpu::PipelineHandle dispatch_pipeline = make_cross_family_compute_pipeline(&device);
    defer gpu::destroy_pipeline(&device, dispatch_pipeline)!!;

    OwnedSpan draw_records = allocate_owned_span(
        device:       &device,
        size:         gpu::GeneratedDrawRecord::size,
        memory_class: gpu::MemoryClass.GPU_PRIVATE,
        access:       { .graphics },
        debug_name:   "generated_draw_records",
    )!!;
    defer free_owned_span(&device, &draw_records)!!;
    OwnedSpan indexed_records = allocate_owned_span(
        device:       &device,
        size:         gpu::GeneratedDrawIndexedRecord::size,
        memory_class: gpu::MemoryClass.GPU_PRIVATE,
        access:       { .graphics },
        debug_name:   "generated_indexed_records",
    )!!;
    defer free_owned_span(&device, &indexed_records)!!;
    OwnedSpan dispatch_records = allocate_owned_span(
        device:       &device,
        size:         gpu::GeneratedDispatchRecord::size,
        memory_class: gpu::MemoryClass.GPU_PRIVATE,
        access:       { .graphics },
        debug_name:   "generated_dispatch_records",
    )!!;
    defer free_owned_span(&device, &dispatch_records)!!;
    OwnedSpan count = allocate_owned_span(
        device:       &device,
        size:         4,
        memory_class: gpu::MemoryClass.GPU_PRIVATE,
        access:       { .graphics },
        debug_name:   "generated_work_count",
    )!!;
    defer free_owned_span(&device, &count)!!;
    OwnedSpan null_draw_records = allocate_owned_span(
        device:       &device,
        size:         gpu::GeneratedDrawRecord::size,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       { .graphics },
        debug_name:   "null_generated_draw_records",
    )!!;
    defer free_owned_span(&device, &null_draw_records)!!;
    OwnedSpan null_indexed_records = allocate_owned_span(
        device:       &device,
        size:         gpu::GeneratedDrawIndexedRecord::size,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       { .graphics },
        debug_name:   "null_generated_indexed_records",
    )!!;
    defer free_owned_span(&device, &null_indexed_records)!!;
    OwnedSpan null_dispatch_records = allocate_owned_span(
        device:       &device,
        size:         gpu::GeneratedDispatchRecord::size,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       { .graphics },
        debug_name:   "null_generated_dispatch_records",
    )!!;
    defer free_owned_span(&device, &null_dispatch_records)!!;
    OwnedSpan null_count = allocate_owned_span(
        device:       &device,
        size:         uint::size,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       { .graphics },
        debug_name:   "null_generated_count",
    )!!;
    defer free_owned_span(&device, &null_count)!!;
    gpu::GeneratedDrawRecord* null_draw =
        (gpu::GeneratedDrawRecord*)test_span_mapping(
            &device,
            null_draw_records.span,
        ).ptr;
    *null_draw = {
        .vertex_root_gpu   = (gpu::GpuAddress)0,
        .fragment_root_gpu = (gpu::GpuAddress)0,
        .arguments         = { .vertex_count = 3, .instance_count = 1 },
    };
    gpu::GeneratedDrawIndexedRecord* null_indexed =
        (gpu::GeneratedDrawIndexedRecord*)test_span_mapping(
            &device,
            null_indexed_records.span,
        ).ptr;
    *null_indexed = {
        .vertex_root_gpu   = (gpu::GpuAddress)0,
        .fragment_root_gpu = (gpu::GpuAddress)0,
        .arguments         = { .index_count = 3, .instance_count = 1 },
    };
    gpu::GeneratedDispatchRecord* null_dispatch =
        (gpu::GeneratedDispatchRecord*)test_span_mapping(
            &device,
            null_dispatch_records.span,
        ).ptr;
    *null_dispatch = {
        .root_gpu  = (gpu::GpuAddress)0,
        .arguments = { .x = 1, .y = 1, .z = 1 },
    };
    uint* null_count_value = (uint*)test_span_mapping(
        &device,
        null_count.span,
    ).ptr;
    *null_count_value = 1;
    gpu::flush_mapped_span(&device, null_draw_records.span)!!;
    gpu::flush_mapped_span(&device, null_indexed_records.span)!!;
    gpu::flush_mapped_span(&device, null_dispatch_records.span)!!;
    gpu::flush_mapped_span(&device, null_count.span)!!;

    OwnedSpan vertices = allocate_owned_span(
        device:       &device,
        size:         3 * 16,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       { .graphics },
        debug_name:   "generated_vertices",
    )!!;
    defer free_owned_span(&device, &vertices)!!;
    write_triangle_verts(&device, &vertices.span);
    gpu::flush_mapped_span(&device, vertices.span)!!;
    OwnedSpan indices = allocate_owned_span(
        device:       &device,
        size:         3 * uint::size,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       { .graphics },
        debug_name:   "generated_indices",
    )!!;
    defer free_owned_span(&device, &indices)!!;
    uint* index_values = (uint*)test_span_mapping(&device, indices.span).ptr;
    index_values[0] = 0;
    index_values[1] = 1;
    index_values[2] = 2;
    gpu::flush_mapped_span(&device, indices.span)!!;
    OwnedSpan fragment_root = allocate_owned_span(
        device:       &device,
        size:         FragRoot::size,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       { .graphics },
        debug_name:   "generated_fragment_root",
    )!!;
    defer free_owned_span(&device, &fragment_root)!!;
    FragRoot* fragment = (FragRoot*)test_span_mapping(&device, fragment_root.span).ptr;
    fragment.texture_index = draw.tex_index;
    fragment.sampler_index = draw.smp_index;
    gpu::flush_mapped_span(&device, fragment_root.span)!!;

    const uint VALUE_COUNT = 64;
    OwnedSpan io_values = allocate_owned_span(
        device:       &device,
        size:         VALUE_COUNT * float::size * 2,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       { .graphics },
        debug_name:   "generated_dispatch_values",
    )!!;
    defer free_owned_span(&device, &io_values)!!;
    float* values = (float*)test_span_mapping(&device, io_values.span).ptr;
    for (uint i = 0; i < VALUE_COUNT; i++) values[i] = (float)i;
    gpu::flush_mapped_span(&device, io_values.span)!!;
    OwnedSpan dispatch_root = allocate_owned_span(
        device:       &device,
        size:         ComputeRoot::size,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       { .graphics },
        debug_name:   "generated_dispatch_root",
    )!!;
    defer free_owned_span(&device, &dispatch_root)!!;
    ComputeRoot* compute_root = (ComputeRoot*)test_span_mapping(
        &device,
        dispatch_root.span,
    ).ptr;
    compute_root.input_gpu = test_span_address(&device, io_values.span);
    compute_root.output_gpu = (gpu::GpuAddress)(
        (ulong)test_span_address(&device, io_values.span)
        + VALUE_COUNT * float::size
    );
    compute_root.count = VALUE_COUNT;
    gpu::flush_mapped_span(&device, dispatch_root.span)!!;

    OwnedSpan build_root = allocate_owned_span(
        device:       &device,
        size:         GeneratedWorkBuildRoot::size,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       { .graphics },
        debug_name:   "generated_work_build_root",
    )!!;
    defer free_owned_span(&device, &build_root)!!;
    GeneratedWorkBuildRoot* root = (GeneratedWorkBuildRoot*)test_span_mapping(
        &device,
        build_root.span,
    ).ptr;
    root.draw_records_gpu = test_span_address(&device, draw_records.span);
    root.indexed_records_gpu = test_span_address(&device, indexed_records.span);
    root.dispatch_records_gpu = test_span_address(&device, dispatch_records.span);
    root.count_gpu = test_span_address(&device, count.span);
    root.vertex_root_gpu = test_span_address(&device, vertices.span);
    root.fragment_root_gpu = test_span_address(&device, fragment_root.span);
    root.dispatch_root_gpu = test_span_address(&device, dispatch_root.span);
    root.vertex_count = 3;
    root.index_count = 3;
    root.dispatch_x = 1;
    root.count_value = 1;
    gpu::flush_mapped_span(&device, build_root.span)!!;

    OwnedSpan plain_readback = allocate_owned_span(
        device:       &device,
        size:         IND_RT_BYTES,
        memory_class: gpu::MemoryClass.CPU_READ,
        access:       { .graphics },
        debug_name:   "generated_plain_readback",
    )!!;
    defer free_owned_span(&device, &plain_readback)!!;
    OwnedSpan indexed_readback = allocate_owned_span(
        device:       &device,
        size:         IND_RT_BYTES,
        memory_class: gpu::MemoryClass.CPU_READ,
        access:       { .graphics },
        debug_name:   "generated_indexed_readback",
    )!!;
    defer free_owned_span(&device, &indexed_readback)!!;

    gpu::Queue queue = gpu::get_queue(
        &device,
        gpu::QueueKind.GRAPHICS,
    )!!;
    gpu::GeneratedScratchDesc dispatch_scratch = {
        .pipeline                = dispatch_pipeline,
        .kind                    = gpu::GeneratedWorkKind.DISPATCH,
        .max_commands_per_list   = 2,
        .preprocess_buffer_count = 2,
    };
    gpu::GeneratedScratchDesc draw_scratch = {
        .pipeline                = draw.pipeline,
        .kind                    = gpu::GeneratedWorkKind.DRAW,
        .max_commands_per_list   = 2,
        .preprocess_buffer_count = 2,
    };
    gpu::GeneratedScratchDesc indexed_scratch = {
        .pipeline                = draw.pipeline,
        .kind                    = gpu::GeneratedWorkKind.DRAW_INDEXED,
        .max_commands_per_list   = 2,
        .preprocess_buffer_count = 2,
    };
    vk::VkDeviceState* state = (vk::VkDeviceState*)
        gpu::device_backend_state(device);
    vk::MemoryRequirements dispatch_requirements =
        vk::generated_scratch_requirements(state, &dispatch_scratch)!!;
    vk::MemoryRequirements draw_requirements =
        vk::generated_scratch_requirements(state, &draw_scratch)!!;
    vk::MemoryRequirements indexed_requirements =
        vk::generated_scratch_requirements(state, &indexed_scratch)!!;
    gpu::CommandAllocatorDesc allocator_desc = {
        .max_generated_preprocess_buffers_per_list = 7,
        .generated_preprocess_bytes = (usz)dispatch_requirements.size * 2
            + (usz)draw_requirements.size * 2
            + (usz)indexed_requirements.size * 2,
    };
    gpu::CommandAllocator allocator = gpu::create_command_allocator(
        &device,
        queue,
        &allocator_desc,
    )!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    GeneratedRequirementsCapture requirements_capture = {
        .original = state.device_dispatch.generated_work.get_memory_requirements,
    };
    generated_requirements_capture = &requirements_capture;
    state.device_dispatch.generated_work.get_memory_requirements =
        &capture_generated_requirements;
    defer {
        state.device_dispatch.generated_work.get_memory_requirements =
            requirements_capture.original;
        generated_requirements_capture = null;
    }
    gpu::reserve_generated_scratch(&allocator, &dispatch_scratch)!!;
    gpu::reserve_generated_scratch(&allocator, &draw_scratch)!!;
    gpu::reserve_generated_scratch(&allocator, &indexed_scratch)!!;
    assert(requirements_capture.calls == 3
            && requirements_capture.max_sequence_counts[0] == 2
            && requirements_capture.max_sequence_counts[1] == 2
            && requirements_capture.max_sequence_counts[2] == 2,
        "each reservation queries its declared maximum sequence count");
    gpu::CommandList commands = gpu::begin_commands(&allocator)!!;
    defer discard_test_commands(&commands)!!;
    bind_test_compute_pipeline(&commands, build_pipeline);
    gpu::cmd_dispatch(
        commands: &commands,
        root:     test_span_address(&device, build_root.span),
        groups:   { 1, 1, 1 },
    )!!;
    gpu::Barrier generated_ready = {
        .before  = { .compute = true },
        .after   = {
            .compute = true,
            .vertex_shader = true,
            .fragment_shader = true,
        },
        .hazards = { .draw_arguments = true },
    };
    gpu::cmd_barrier(&commands, &generated_ready)!!;

    bind_test_compute_pipeline(&commands, dispatch_pipeline);
    gpu::cmd_dispatch_generated(
        commands:           &commands,
        records:            null_dispatch_records.span,
        count_span:         null_count.span,
        max_dispatch_count: 1,
    )!!;

    record_target_to_attach(&commands, null_target);
    gpu::ColorTargetDesc[1] null_colors = {{
        .view     = null_view,
        .load_op  = gpu::LoadOp.CLEAR,
        .store_op = gpu::StoreOp.STORE,
    }};
    gpu::RenderPassDesc null_pass = {
        .colors = null_colors[..],
        .width  = IND_RT,
        .height = IND_RT,
    };
    gpu::cmd_begin_render_pass(&commands, &null_pass)!!;
    bind_test_graphics_pipeline(&commands, draw.pipeline);
    gpu::cmd_draw_generated(
        commands:      &commands,
        records:       null_draw_records.span,
        count_span:    null_count.span,
        max_draw_count: 1,
    )!!;
    gpu::cmd_draw_indexed_generated(
        commands:      &commands,
        records:       null_indexed_records.span,
        count_span:    null_count.span,
        max_draw_count: 1,
        index_span:    indices.span,
    )!!;
    gpu::cmd_end_render_pass(&commands)!!;

    bind_test_compute_pipeline(&commands, dispatch_pipeline);
    vk::CommandRecord* record = vk::resolve_command(&commands)!!;
    gpu::RecordingWorkCounters dispatch_work_before =
        vk::command_recording_work_counters(state);
    vk::CommandRecordingStats dispatch_recording_before =
        vk::command_recording_stats(state);
    uint dispatch_preprocess_count =
        record.scratch.generated_preprocess_count;
    gpu::reset_command_resolution_stats();
    gpu::cmd_dispatch_generated(
        commands:           &commands,
        records:            dispatch_records.span,
        count_span:         count.span,
        max_dispatch_count: 1,
    )!!;
    assert_warm_command_resolution(gpu::command_resolution_stats(), 1);
    gpu::RecordingWorkCounters dispatch_work_after =
        vk::command_recording_work_counters(state);
    vk::CommandRecordingStats dispatch_recording_after =
        vk::command_recording_stats(state);
    assert(dispatch_work_after.host_allocations
            == dispatch_work_before.host_allocations
        && dispatch_work_after.vma_allocations
            == dispatch_work_before.vma_allocations
        && dispatch_recording_after.preprocess_allocations
            == dispatch_recording_before.preprocess_allocations
        && dispatch_recording_after.recording_allocations
            == dispatch_recording_before.recording_allocations,
        "reserved generated dispatch performs no host, VMA, or preprocess allocation");
    assert(dispatch_work_after.generated_dispatch_emissions
            == dispatch_work_before.generated_dispatch_emissions + 1
        && dispatch_work_after.generated_draw_emissions
            == dispatch_work_before.generated_draw_emissions
        && dispatch_work_after.generated_indexed_draw_emissions
            == dispatch_work_before.generated_indexed_draw_emissions,
        "generated dispatch emits exactly one native dispatch-family command");
    assert(record.scratch.generated_preprocess_count
            == dispatch_preprocess_count + 1,
        "generated dispatch stays within its reserved preprocess capacity");

    gpu::RecordingWorkCounters dispatch_exhausted_before =
        vk::command_recording_work_counters(state);
    vk::CommandRecordingStats dispatch_exhausted_recording_before =
        vk::command_recording_stats(state);
    uint dispatch_exhausted_references = record.scratch.reference_count;
    uint dispatch_exhausted_preprocess =
        record.scratch.generated_preprocess_count;
    gpu::reset_command_resolution_stats();
    assert(@catch(gpu::cmd_dispatch_generated(
        commands:           &commands,
        records:            dispatch_records.span,
        count_span:         count.span,
        max_dispatch_count: 1,
    )) == gpu::GENERATED_SCRATCH_EXHAUSTED,
        "generated dispatch exhaustion preserves the exact public fault");
    gpu::RecordingWorkCounters dispatch_exhausted_after =
        vk::command_recording_work_counters(state);
    vk::CommandRecordingStats dispatch_exhausted_recording_after =
        vk::command_recording_stats(state);
    gpu::CommandResolutionStats dispatch_exhausted_resolution =
        gpu::command_resolution_stats();
    assert(dispatch_exhausted_after.generated_scratch_misses
            == dispatch_exhausted_before.generated_scratch_misses + 1
        && dispatch_exhausted_after.host_allocations
            == dispatch_exhausted_before.host_allocations
        && dispatch_exhausted_after.vma_allocations
            == dispatch_exhausted_before.vma_allocations
        && dispatch_exhausted_recording_after.preprocess_allocations
            == dispatch_exhausted_recording_before.preprocess_allocations
        && dispatch_exhausted_recording_after.recording_allocations
            == dispatch_exhausted_recording_before.recording_allocations,
        "generated dispatch exhaustion records one miss and no allocation");
    assert(dispatch_exhausted_after.generated_dispatch_emissions
            == dispatch_exhausted_before.generated_dispatch_emissions
        && dispatch_exhausted_after.generated_draw_emissions
            == dispatch_exhausted_before.generated_draw_emissions
        && dispatch_exhausted_after.generated_indexed_draw_emissions
            == dispatch_exhausted_before.generated_indexed_draw_emissions
        && dispatch_exhausted_resolution.native_commands == 0
        && record.scratch.reference_count == dispatch_exhausted_references
        && record.scratch.generated_preprocess_count
            == dispatch_exhausted_preprocess,
        "generated dispatch exhaustion emits and records no partial command");
    record_target_to_attach(&commands, plain_target);
    gpu::ColorTargetDesc[1] plain_colors = {{
        .view     = plain_view,
        .load_op  = gpu::LoadOp.CLEAR,
        .store_op = gpu::StoreOp.STORE,
    }};
    gpu::RenderPassDesc plain_pass = {
        .colors = plain_colors[..],
        .width  = IND_RT,
        .height = IND_RT,
    };
    gpu::cmd_begin_render_pass(&commands, &plain_pass)!!;
    bind_test_graphics_pipeline(&commands, draw.pipeline);
    gpu::RecordingWorkCounters draw_work_before =
        vk::command_recording_work_counters(state);
    vk::CommandRecordingStats draw_recording_before =
        vk::command_recording_stats(state);
    uint draw_preprocess_count = record.scratch.generated_preprocess_count;
    gpu::reset_command_resolution_stats();
    gpu::cmd_draw_generated(
        commands:      &commands,
        records:       draw_records.span,
        count_span:    count.span,
        max_draw_count: 1,
    )!!;
    assert_warm_command_resolution(gpu::command_resolution_stats(), 1);
    gpu::RecordingWorkCounters draw_work_after =
        vk::command_recording_work_counters(state);
    vk::CommandRecordingStats draw_recording_after =
        vk::command_recording_stats(state);
    assert(draw_work_after.host_allocations == draw_work_before.host_allocations
        && draw_work_after.vma_allocations == draw_work_before.vma_allocations
        && draw_recording_after.preprocess_allocations
            == draw_recording_before.preprocess_allocations
        && draw_recording_after.recording_allocations
            == draw_recording_before.recording_allocations,
        "reserved generated draw performs no host, VMA, or preprocess allocation");
    assert(draw_work_after.generated_draw_emissions
            == draw_work_before.generated_draw_emissions + 1
        && draw_work_after.generated_dispatch_emissions
            == draw_work_before.generated_dispatch_emissions
        && draw_work_after.generated_indexed_draw_emissions
            == draw_work_before.generated_indexed_draw_emissions
        && record.scratch.generated_preprocess_count
            == draw_preprocess_count + 1,
        "generated draw emits once within reserved preprocess capacity");

    gpu::RecordingWorkCounters draw_exhausted_before =
        vk::command_recording_work_counters(state);
    vk::CommandRecordingStats draw_exhausted_recording_before =
        vk::command_recording_stats(state);
    uint draw_exhausted_references = record.scratch.reference_count;
    uint draw_exhausted_preprocess =
        record.scratch.generated_preprocess_count;
    gpu::reset_command_resolution_stats();
    assert(@catch(gpu::cmd_draw_generated(
        commands:      &commands,
        records:       draw_records.span,
        count_span:    count.span,
        max_draw_count: 1,
    )) == gpu::GENERATED_SCRATCH_EXHAUSTED,
        "generated draw exhaustion preserves the exact public fault");
    gpu::RecordingWorkCounters draw_exhausted_after =
        vk::command_recording_work_counters(state);
    vk::CommandRecordingStats draw_exhausted_recording_after =
        vk::command_recording_stats(state);
    assert(draw_exhausted_after.generated_scratch_misses
            == draw_exhausted_before.generated_scratch_misses + 1
        && draw_exhausted_after.host_allocations
            == draw_exhausted_before.host_allocations
        && draw_exhausted_after.vma_allocations
            == draw_exhausted_before.vma_allocations
        && draw_exhausted_recording_after.preprocess_allocations
            == draw_exhausted_recording_before.preprocess_allocations
        && draw_exhausted_recording_after.recording_allocations
            == draw_exhausted_recording_before.recording_allocations
        && draw_exhausted_after.generated_draw_emissions
            == draw_exhausted_before.generated_draw_emissions
        && draw_exhausted_after.generated_dispatch_emissions
            == draw_exhausted_before.generated_dispatch_emissions
        && draw_exhausted_after.generated_indexed_draw_emissions
            == draw_exhausted_before.generated_indexed_draw_emissions
        && gpu::command_resolution_stats().native_commands == 0
        && record.scratch.reference_count == draw_exhausted_references
        && record.scratch.generated_preprocess_count
            == draw_exhausted_preprocess,
        "generated draw exhaustion performs no allocation, emission, or partial recording");
    gpu::cmd_end_render_pass(&commands)!!;
    record_target_readback(&commands, plain_target, &plain_readback.span);

    record_target_to_attach(&commands, indexed_target);
    gpu::ColorTargetDesc[1] indexed_colors = {{
        .view     = indexed_view,
        .load_op  = gpu::LoadOp.CLEAR,
        .store_op = gpu::StoreOp.STORE,
    }};
    gpu::RenderPassDesc indexed_pass = {
        .colors = indexed_colors[..],
        .width  = IND_RT,
        .height = IND_RT,
    };
    gpu::cmd_begin_render_pass(&commands, &indexed_pass)!!;
    bind_test_graphics_pipeline(&commands, draw.pipeline);
    gpu::RecordingWorkCounters indexed_work_before =
        vk::command_recording_work_counters(state);
    vk::CommandRecordingStats indexed_recording_before =
        vk::command_recording_stats(state);
    uint indexed_preprocess_count =
        record.scratch.generated_preprocess_count;
    gpu::reset_command_resolution_stats();
    gpu::cmd_draw_indexed_generated(
        commands:      &commands,
        records:       indexed_records.span,
        count_span:    count.span,
        max_draw_count: 1,
        index_span:    indices.span,
    )!!;
    assert_warm_command_resolution(gpu::command_resolution_stats(), 1);
    gpu::RecordingWorkCounters indexed_work_after =
        vk::command_recording_work_counters(state);
    vk::CommandRecordingStats indexed_recording_after =
        vk::command_recording_stats(state);
    assert(indexed_work_after.host_allocations
            == indexed_work_before.host_allocations
        && indexed_work_after.vma_allocations
            == indexed_work_before.vma_allocations
        && indexed_recording_after.preprocess_allocations
            == indexed_recording_before.preprocess_allocations
        && indexed_recording_after.recording_allocations
            == indexed_recording_before.recording_allocations,
        "reserved generated indexed draw performs no host, VMA, or preprocess allocation");
    assert(indexed_work_after.generated_indexed_draw_emissions
            == indexed_work_before.generated_indexed_draw_emissions + 1
        && indexed_work_after.generated_dispatch_emissions
            == indexed_work_before.generated_dispatch_emissions
        && indexed_work_after.generated_draw_emissions
            == indexed_work_before.generated_draw_emissions
        && record.scratch.generated_preprocess_count
            == indexed_preprocess_count + 1,
        "generated indexed draw emits once within reserved preprocess capacity");

    gpu::RecordingWorkCounters indexed_exhausted_before =
        vk::command_recording_work_counters(state);
    vk::CommandRecordingStats indexed_exhausted_recording_before =
        vk::command_recording_stats(state);
    uint indexed_exhausted_references = record.scratch.reference_count;
    uint indexed_exhausted_preprocess =
        record.scratch.generated_preprocess_count;
    gpu::reset_command_resolution_stats();
    assert(@catch(gpu::cmd_draw_indexed_generated(
        commands:      &commands,
        records:       indexed_records.span,
        count_span:    count.span,
        max_draw_count: 1,
        index_span:    indices.span,
    )) == gpu::GENERATED_SCRATCH_EXHAUSTED,
        "generated indexed-draw exhaustion preserves the exact public fault");
    gpu::RecordingWorkCounters indexed_exhausted_after =
        vk::command_recording_work_counters(state);
    vk::CommandRecordingStats indexed_exhausted_recording_after =
        vk::command_recording_stats(state);
    assert(indexed_exhausted_after.generated_scratch_misses
            == indexed_exhausted_before.generated_scratch_misses + 1
        && indexed_exhausted_after.host_allocations
            == indexed_exhausted_before.host_allocations
        && indexed_exhausted_after.vma_allocations
            == indexed_exhausted_before.vma_allocations
        && indexed_exhausted_recording_after.preprocess_allocations
            == indexed_exhausted_recording_before.preprocess_allocations
        && indexed_exhausted_recording_after.recording_allocations
            == indexed_exhausted_recording_before.recording_allocations
        && indexed_exhausted_after.generated_indexed_draw_emissions
            == indexed_exhausted_before.generated_indexed_draw_emissions
        && indexed_exhausted_after.generated_dispatch_emissions
            == indexed_exhausted_before.generated_dispatch_emissions
        && indexed_exhausted_after.generated_draw_emissions
            == indexed_exhausted_before.generated_draw_emissions
        && gpu::command_resolution_stats().native_commands == 0
        && record.scratch.reference_count == indexed_exhausted_references
        && record.scratch.generated_preprocess_count
            == indexed_exhausted_preprocess,
        "generated indexed-draw exhaustion performs no allocation, emission, or partial recording");
    gpu::cmd_end_render_pass(&commands)!!;
    record_target_readback(&commands, indexed_target, &indexed_readback.span);
    assert(requirements_capture.calls == 3,
        "smaller generated recordings perform no native requirements query");

    gpu::Barrier dispatch_host_read = {
        .before = { .compute = true },
        .after  = { .host = true },
    };
    gpu::cmd_barrier(&commands, &dispatch_host_read)!!;
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
    gpu::release_generated_scratch(
        &allocator,
        dispatch_pipeline,
        gpu::GeneratedWorkKind.DISPATCH,
    )!!;
    gpu::release_generated_scratch(
        &allocator,
        draw.pipeline,
        gpu::GeneratedWorkKind.DRAW,
    )!!;
    gpu::release_generated_scratch(
        &allocator,
        draw.pipeline,
        gpu::GeneratedWorkKind.DRAW_INDEXED,
    )!!;

    gpu::invalidate_mapped_span(&device, io_values.span)!!;
    float* doubled = &values[VALUE_COUNT];
    for (uint i = 0; i < VALUE_COUNT; i++) {
        assert(doubled[i] == (float)i * 2.0f,
            "generated dispatch uses its GPU-provided root and arguments");
    }
    gpu::invalidate_mapped_span(&device, plain_readback.span)!!;
    char* plain_center = center_pixel(&device, &plain_readback.span);
    assert(plain_center[0] == (char)255
            && plain_center[1] == (char)255
            && plain_center[2] == (char)255,
        "GPU-written generated draw renders the triangle");
    gpu::invalidate_mapped_span(&device, indexed_readback.span)!!;
    char* indexed_center = center_pixel(&device, &indexed_readback.span);
    assert(indexed_center[0] == (char)255
            && indexed_center[1] == (char)255
            && indexed_center[2] == (char)255,
        "GPU-written generated indexed draw renders the triangle");
    io::printfn("generated-work: EXERCISED");
}

fn void plain_indirect_draw_from_caller_owned_span() @test {
    gpu::Device device = make_ind_device();
    defer destroy_test_device(&device)!!;
    IndDrawCtx ctx = make_ind_draw_ctx(&device);
    defer destroy_ind_draw_ctx(&device, &ctx);
    gpu::TextureHandle target = make_ind_target(&device);
    defer gpu::destroy_texture(&device, target)!!;
    gpu::AttachmentViewHandle target_view = create_test_attachment_view(
        &device,
        target,
    )!!;
    defer destroy_test_attachment_view(&device, &target_view)!!;

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

    gpu::Queue command_queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator command_allocator = gpu::create_command_allocator(&device, command_queue)!!;
    defer gpu::destroy_command_allocator(&command_allocator)!!;
    gpu::CommandList cmd = gpu::begin_commands(&command_allocator)!!;
    defer discard_test_commands(&cmd)!!;
    record_target_to_attach(&cmd, target);
    gpu::ColorTargetDesc[1] colors = {{
        .view = target_view, .load_op = gpu::LoadOp.CLEAR, .store_op = gpu::StoreOp.STORE,
        .clear = { .rgba = { 8.0f / 255.0f, 16.0f / 255.0f, 32.0f / 255.0f, 1.0f } },
    }};
    gpu::RenderPassDesc pass = { .colors = colors[..], .width = IND_RT, .height = IND_RT };
    gpu::cmd_begin_render_pass(&cmd, &pass)!!;
    bind_test_graphics_pipeline(&cmd, ctx.pipeline);
    gpu::cmd_draw_indirect(
        commands:      &cmd,
        vertex_root:   (gpu::GpuAddress)0,
        fragment_root: (gpu::GpuAddress)0,
        args:          args_span,
        draw_count:    1,
    )!!;
    gpu::cmd_draw_indirect(
        commands:      &cmd,
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
    gpu::ShaderCode build_shader = gpu::prepare_shader_code(&build_sd)!!;
    gpu::ComputePipelineDesc build_desc = { .shader = build_shader };
    gpu::PipelineHandle build_pipeline = gpu::create_compute_pipeline(&device, &build_desc)!!;
    defer gpu::destroy_pipeline(&device, build_pipeline)!!;

    gpu::ShaderDesc double_sd = { .stage = gpu::ShaderStage.COMPUTE, .spirv = IND_DOUBLE_SPIRV[..], .entry_point = "main" };
    gpu::ShaderCode double_shader = gpu::prepare_shader_code(&double_sd)!!;
    gpu::ComputePipelineDesc double_desc = { .shader = double_shader };
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

    gpu::Queue command_queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator command_allocator = gpu::create_command_allocator(&device, command_queue)!!;
    defer gpu::destroy_command_allocator(&command_allocator)!!;
    gpu::CommandList cmd = gpu::begin_commands(&command_allocator)!!;
    defer discard_test_commands(&cmd)!!;
    bind_test_compute_pipeline(&cmd, build_pipeline);
    gpu::cmd_dispatch(
        commands: &cmd,
        root:     test_span_address(&device, build_root_span),
        groups:   { 1, 1, 1 },
    )!!;
    gpu::Barrier args_ready = {
        .before = { .compute = true },
        .after = { .compute = true },
        .hazards = { .draw_arguments = true },
    };
    gpu::cmd_barrier(&cmd, &args_ready)!!;
    bind_test_compute_pipeline(&cmd, double_pipeline);
    gpu::cmd_dispatch_indirect(
        commands: &cmd,
        root:     (gpu::GpuAddress)0,
        args:     dispatch_args,
    )!!;
    gpu::cmd_dispatch_indirect(
        commands: &cmd,
        root:     test_span_address(&device, double_root_span),
        args:     dispatch_args,
    )!!;
    gpu::Barrier host_read = {
        .before = { .compute = true },
        .after = { .host = true },
    };
    gpu::cmd_barrier(&cmd, &host_read)!!;
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
    gpu::AttachmentViewHandle zero_view = create_test_attachment_view(
        &device,
        target_zero,
    )!!;
    defer destroy_test_attachment_view(&device, &zero_view)!!;
    gpu::TextureHandle target_one = make_ind_target(&device);
    defer gpu::destroy_texture(&device, target_one)!!;
    gpu::AttachmentViewHandle one_view = create_test_attachment_view(
        &device,
        target_one,
    )!!;
    defer destroy_test_attachment_view(&device, &one_view)!!;

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

    gpu::Queue command_queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator command_allocator = gpu::create_command_allocator(&device, command_queue)!!;
    defer gpu::destroy_command_allocator(&command_allocator)!!;
    gpu::CommandList cmd = gpu::begin_commands(&command_allocator)!!;
    defer discard_test_commands(&cmd)!!;
    gpu::TextureHandle[2] targets = { target_zero, target_one };
    gpu::AttachmentViewHandle[2] target_views = { zero_view, one_view };
    gpu::GpuSpan[2] count_spans = { count_zero, count_one };
    gpu::GpuSpan[2] rb_spans = { rb_zero_span, rb_one_span };
    for (uint i = 0; i < 2; i++) {
        record_target_to_attach(&cmd, targets[i]);
        gpu::ColorTargetDesc[1] colors = {{
            .view = target_views[i], .load_op = gpu::LoadOp.CLEAR, .store_op = gpu::StoreOp.STORE,
            .clear = { .rgba = { 8.0f / 255.0f, 16.0f / 255.0f, 32.0f / 255.0f, 1.0f } },
        }};
        gpu::RenderPassDesc pass = { .colors = colors[..], .width = IND_RT, .height = IND_RT };
        gpu::cmd_begin_render_pass(&cmd, &pass)!!;
        bind_test_graphics_pipeline(&cmd, ctx.pipeline);
        gpu::cmd_draw_indexed_indirect_count(
            commands:       &cmd,
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
    gpu::AttachmentViewHandle target_view = create_test_attachment_view(
        &device,
        target,
    )!!;

    gpu::Queue command_queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator command_allocator = gpu::create_command_allocator(&device, command_queue)!!;
    gpu::CommandList commands = gpu::begin_commands(&command_allocator)!!;
    record_target_to_attach(&commands, target);
    gpu::ColorTargetDesc[1] colors = {{
        .view     = target_view,
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
    gpu::destroy_attachment_view(&device, target_view)!!;
    gpu::destroy_texture(&device, target)!!;
    gpu::destroy_command_allocator(&command_allocator)!!;
    destroy_test_device(&device)!!;
}

fn void indirect_argument_validation() @test {
    gpu::Device device = make_ind_device();
    defer destroy_test_device(&device)!!;
    IndDrawCtx ctx = make_ind_draw_ctx(&device);
    defer destroy_ind_draw_ctx(&device, &ctx);
    gpu::TextureHandle target = make_ind_target(&device);
    defer gpu::destroy_texture(&device, target)!!;
    gpu::AttachmentViewHandle target_view = create_test_attachment_view(
        &device,
        target,
    )!!;
    defer destroy_test_attachment_view(&device, &target_view)!!;

    vk::BufferDesc plain_desc = {
        .size        = 64,
        .usage       = { .storage, .addressable },
        .memory_class = gpu::MemoryClass.GPU_PRIVATE,
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

    gpu::Queue command_queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator command_allocator = gpu::create_command_allocator(&device, command_queue)!!;
    defer gpu::destroy_command_allocator(&command_allocator)!!;
    gpu::CommandList cmd = gpu::begin_commands(&command_allocator)!!;
    defer discard_test_commands(&cmd)!!;
    record_target_to_attach(&cmd, target);
    gpu::ColorTargetDesc[1] colors = {{
        .view = target_view, .load_op = gpu::LoadOp.CLEAR, .store_op = gpu::StoreOp.STORE,
    }};
    gpu::RenderPassDesc pass = { .colors = colors[..], .width = IND_RT, .height = IND_RT };
    gpu::cmd_begin_render_pass(&cmd, &pass)!!;

    bind_test_graphics_pipeline(&cmd, ctx.pipeline);
    fault no_usage = @catch(gpu::cmd_draw_indirect(
        commands:      &cmd,
        vertex_root:   test_span_address(&device, verts_span),
        fragment_root: test_span_address(&device, frag_span),
        args:          plain_span,
        draw_count:    1,
    ));
    assert(no_usage == gpu::INVALID_ARGUMENT, "args without indirect usage fault INVALID_ARGUMENT");

    gpu::GpuSpan misaligned = args_span.checked_subspan(2, 32)!!;
    bind_test_graphics_pipeline(&cmd, ctx.pipeline);
    fault align_fault = @catch(gpu::cmd_draw_indirect(
        commands:      &cmd,
        vertex_root:   test_span_address(&device, verts_span),
        fragment_root: test_span_address(&device, frag_span),
        args:          misaligned,
        draw_count:    1,
    ));
    assert(align_fault == gpu::INVALID_ARGUMENT, "misaligned args offset faults INVALID_ARGUMENT");

    bind_test_graphics_pipeline(&cmd, ctx.pipeline);
    fault overrun = @catch(gpu::cmd_draw_indirect(
        commands:      &cmd,
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
