module gpu_test;

import gpu::internal @public;
import gpu @public;
import gpu::internal::vk @public;
import vk;
import std::io;

const char[*] MESHLET_MESH_SPIRV = $embed("shaders/meshlet.mesh.spv");
const char[*] MESH_ARGS_SPIRV = $embed("shaders/mesh_args.comp.spv");

fn gpu::Device? make_mesh_device(VulkanValidationCapture* capture = null) {
    gpu::RuntimeDesc runtime_desc = {
        .contract_validation      = gpu::ContractValidation.FULL,
        .enable_vulkan_validation = true,
        .application_name         = "gpu_mesh_test",
        .debug_callback           = capture == null ? null : &capture_vulkan_validation_error,
        .debug_user_data          = capture,
    };
    gpu::DeviceDesc device_desc = { .enable_mesh_shaders = true };
    gpu::Runtime runtime = gpu::create_runtime(&runtime_desc)!;
    defer catch (void)gpu::destroy_runtime(&runtime);
    gpu::AdapterList adapters = gpu::enumerate_adapters(&runtime)!;
    for (uint i = 0; i < adapters.count; i++) {
        gpu::Adapter adapter = adapters.get(i)!;
        gpu::DeviceSupport support = gpu::supports_device_desc(&adapter, &device_desc)!;
        if (!support.supported) continue;
        gpu::Device? candidate = gpu::create_device(&adapter, &device_desc);
        if (catch err = candidate) {
            if (err == gpu::UNSUPPORTED_FEATURE) continue;
            return err~;
        }
        return candidate;
    }
    return gpu::UNSUPPORTED_FEATURE~;
}

fn gpu::PipelineHandle make_mesh_pipeline(gpu::Device* device, DepthCtx* ctx) {
    gpu::ShaderDesc mesh = { .spirv = MESHLET_MESH_SPIRV[..], .entry_point = "main" };
    gpu::Format[1] color_formats = { gpu::Format.RGBA8_UNORM };
    gpu::MeshPipelineDesc desc = {
        .mesh_shader     = mesh,
        .fragment_shader = ctx.frag,
        .color_formats   = color_formats[..],
        .debug_name      = "meshlet",
    };
    return gpu::create_mesh_pipeline(device, &desc)!!;
}

fn void mesh_shaders_gate_on_the_feature() @test {
    gpu::Device device = make_depth_device();
    defer destroy_test_device(&device)!!;
    gpu::DeviceCaps caps = gpu::get_device_caps(&device)!!;
    assert(!caps.mesh_shaders.enabled && !caps.mesh_shaders.task_shaders,
        "mesh caps are zero without the request");
    DepthCtx ctx = make_depth_ctx();
    gpu::ShaderDesc mesh = { .spirv = MESHLET_MESH_SPIRV[..], .entry_point = "main" };
    gpu::Format[1] color_formats = { gpu::Format.RGBA8_UNORM };
    gpu::MeshPipelineDesc desc = {
        .mesh_shader     = mesh,
        .fragment_shader = ctx.frag,
        .color_formats   = color_formats[..],
    };
    assert(@catch(gpu::create_mesh_pipeline(&device, &desc)) == gpu::UNSUPPORTED_FEATURE,
        "mesh pipelines need enable_mesh_shaders");

    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator allocator = gpu::create_command_allocator(&device, queue)!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::CommandList cmd = gpu::begin_commands(&allocator)!!;
    defer discard_test_commands(&cmd)!!;
    gpu::Barrier to_mesh = { .before = { .compute }, .after = { .mesh_shader } };
    assert(@catch(gpu::cmd_barrier(&cmd, &to_mesh)) == gpu::INVALID_ARGUMENT,
        "the mesh stage needs the feature");
}

fn void mesh_pipeline_draws_meshlet_triangle() @test {
    VulkanValidationCapture validation;
    gpu::Device? candidate = make_mesh_device(&validation);
    if (catch err = candidate) {
        assert(err == gpu::UNSUPPORTED_FEATURE, "device creation faults only for missing support");
        io::printfn("mesh shaders unsupported; skipping");
        return;
    }
    gpu::Device device = candidate;
    defer destroy_test_device(&device)!!;
    gpu::DeviceCaps caps = gpu::get_device_caps(&device)!!;
    assert(caps.mesh_shaders.enabled && caps.mesh_shaders.max_mesh_work_group_count.x > 0
        && caps.mesh_shaders.max_mesh_output_vertices >= 3,
        "mesh caps report the adapter limits");
    DepthCtx ctx = make_depth_ctx();
    gpu::PipelineHandle mesh_pipeline = make_mesh_pipeline(&device, &ctx);
    defer gpu::destroy_pipeline(&device, mesh_pipeline)!!;
    gpu::Format[1] color_formats = { gpu::Format.RGBA8_UNORM };
    gpu::GraphicsPipelineDesc graphics_desc = {
        .vertex_shader   = ctx.vert,
        .fragment_shader = ctx.frag,
        .color_formats   = color_formats[..],
    };
    gpu::PipelineHandle graphics_pipeline = gpu::create_graphics_pipeline(&device, &graphics_desc)!!;
    defer gpu::destroy_pipeline(&device, graphics_pipeline)!!;

    gpu::TextureHandle target = make_color_target(&device);
    defer gpu::destroy_texture(&device, target)!!;
    gpu::AttachmentViewHandle target_view = create_test_attachment_view(&device, target)!!;
    defer destroy_test_attachment_view(&device, &target_view)!!;
    OwnedSpan tri = write_tri(&device, 0.5f);
    defer free_owned_span(&device, &tri)!!;
    OwnedSpan color = write_color(&device, { 1.0f, 0.0f, 0.0f, 1.0f });
    defer free_owned_span(&device, &color)!!;
    OwnedSpan readback = make_readback(&device, DP_RT * DP_RT * 4);
    defer free_owned_span(&device, &readback)!!;

    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator allocator = gpu::create_command_allocator(&device, queue)!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::CommandList cmd = gpu::begin_commands(&allocator)!!;
    defer discard_test_commands(&cmd)!!;
    record_color_to_attach(&cmd, target);
    gpu::ColorTargetDesc[1] colors = {{
        .view     = target_view,
        .load_op  = gpu::LoadOp.CLEAR,
        .store_op = gpu::StoreOp.STORE,
        .clear    = { .rgba = { 0.0f, 0.0f, 0.0f, 1.0f } },
    }};
    gpu::RenderPassDesc pass = { .colors = colors[..], .width = DP_RT, .height = DP_RT };
    gpu::GraphicsState state = gpu::render_geometry_state(DP_RT, DP_RT)!!;
    gpu::ColorTargetState[1] color_on = {{ .write_mask = gpu::COLOR_WRITE_ALL }};
    state.color = { .targets = color_on[..] };
    gpu::cmd_begin_render_pass(&cmd, &pass)!!;
    gpu::cmd_bind_pipeline(&cmd, graphics_pipeline)!!;
    gpu::cmd_set_graphics_state(&cmd, &state)!!;
    assert(@catch(gpu::cmd_draw_mesh_tasks(
        commands:      &cmd,
        mesh_root:     test_span_address(&device, tri.span),
        fragment_root: test_span_address(&device, color.span),
        groups:        { 1, 1, 1 },
    )) == gpu::INVALID_ARGUMENT, "mesh draws need a mesh pipeline");
    gpu::cmd_bind_pipeline(&cmd, mesh_pipeline)!!;
    gpu::cmd_set_graphics_state(&cmd, &state)!!;
    assert(@catch(gpu::cmd_draw(
        commands:       &cmd,
        vertex_root:    test_span_address(&device, tri.span),
        fragment_root:  test_span_address(&device, color.span),
        vertex_count:   3,
        instance_count: 1,
    )) == gpu::INVALID_ARGUMENT, "vertex draws reject a mesh pipeline");
    gpu::Vec3u too_many = { caps.mesh_shaders.max_mesh_work_group_count.x + 1, 1, 1 };
    assert(@catch(gpu::cmd_draw_mesh_tasks(
        commands:      &cmd,
        mesh_root:     test_span_address(&device, tri.span),
        fragment_root: test_span_address(&device, color.span),
        groups:        too_many,
    )) == gpu::INVALID_ARGUMENT, "group counts past the caps fault");
    gpu::cmd_draw_mesh_tasks(
        commands:      &cmd,
        mesh_root:     (gpu::GpuAddress)0,
        fragment_root: test_span_address(&device, color.span),
        groups:        { 1, 1, 1 },
    )!!;
    gpu::cmd_draw_mesh_tasks(
        commands:      &cmd,
        mesh_root:     test_span_address(&device, tri.span),
        fragment_root: test_span_address(&device, color.span),
        groups:        { 1, 1, 1 },
    )!!;
    gpu::cmd_end_render_pass(&cmd)!!;
    gpu::GpuSpan rb_span = readback.span;
    record_color_readback(&cmd, target, &rb_span);
    submit_and_finish(&device, &cmd);
    gpu::invalidate_mapped_span(&device, readback.span)!!;
    char* pixels = (char*)test_span_mapping(&device, readback.span).ptr;
    char* center = &pixels[(DP_RT / 2 * DP_RT + DP_RT / 2) * 4];
    assert(center[0] == (char)255 && center[1] == (char)0, "the meshlet triangle covers the center in the fragment root color");
    assert(pixels[0] == (char)0, "the corner keeps the clear color");
    assert(validation.errors == 0, "the mesh draw is validation clean");
}

fn void mesh_indirect_draws_execute_gpu_written_arguments() @test {
    VulkanValidationCapture validation;
    gpu::Device? candidate = make_mesh_device(&validation);
    if (catch err = candidate) {
        assert(err == gpu::UNSUPPORTED_FEATURE, "device creation faults only for missing support");
        return;
    }
    gpu::Device device = candidate;
    defer destroy_test_device(&device)!!;
    gpu::DeviceCaps caps = gpu::get_device_caps(&device)!!;
    DepthCtx ctx = make_depth_ctx();
    gpu::PipelineHandle mesh_pipeline = make_mesh_pipeline(&device, &ctx);
    defer gpu::destroy_pipeline(&device, mesh_pipeline)!!;
    gpu::ShaderDesc build_shader = { .spirv = MESH_ARGS_SPIRV[..], .entry_point = "main" };
    gpu::ComputePipelineDesc build_desc = { .shader = build_shader };
    gpu::PipelineHandle build = gpu::create_compute_pipeline(&device, &build_desc)!!;
    defer gpu::destroy_pipeline(&device, build)!!;

    gpu::TextureHandle target = make_color_target(&device);
    defer gpu::destroy_texture(&device, target)!!;
    gpu::AttachmentViewHandle target_view = create_test_attachment_view(&device, target)!!;
    defer destroy_test_attachment_view(&device, &target_view)!!;
    OwnedSpan tri = write_tri(&device, 0.5f);
    defer free_owned_span(&device, &tri)!!;
    OwnedSpan color = write_color(&device, { 0.0f, 1.0f, 0.0f, 1.0f });
    defer free_owned_span(&device, &color)!!;
    OwnedSpan readback = make_readback(&device, DP_RT * DP_RT * 4);
    defer free_owned_span(&device, &readback)!!;
    OwnedSpan args = allocate_owned_span(
        device:       &device,
        size:         2 * gpu::DrawMeshTasksIndirectCommand::size,
        memory_class: gpu::MemoryClass.GPU_PRIVATE,
        access:       { .graphics, .compute },
        debug_name:   "mesh_args",
    )!!;
    defer free_owned_span(&device, &args)!!;
    OwnedSpan count = allocate_owned_span(
        device:       &device,
        size:         uint::size,
        memory_class: gpu::MemoryClass.GPU_PRIVATE,
        access:       { .graphics, .compute },
        debug_name:   "mesh_count",
    )!!;
    defer free_owned_span(&device, &count)!!;
    OwnedSpan root = allocate_owned_span(
        device:       &device,
        size:         MeshArgsRoot::size,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       { .compute },
        debug_name:   "mesh_args_root",
    )!!;
    defer free_owned_span(&device, &root)!!;
    *(MeshArgsRoot*)test_span_mapping(&device, root.span).ptr = {
        .args_gpu    = test_span_address(&device, args.span),
        .count_gpu   = test_span_address(&device, count.span),
        .count_value = 1,
    };
    gpu::flush_mapped_span(&device, root.span)!!;

    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator allocator = gpu::create_command_allocator(&device, queue)!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::CommandList cmd = gpu::begin_commands(&allocator)!!;
    defer discard_test_commands(&cmd)!!;
    bind_test_compute_pipeline(&cmd, build);
    gpu::cmd_dispatch(
        commands: &cmd,
        root:     test_span_address(&device, root.span),
        groups:   { 1, 1, 1 },
    )!!;
    gpu::Barrier args_ready = { .before = { .compute }, .after = { .indirect } };
    gpu::cmd_barrier(&cmd, &args_ready)!!;
    record_color_to_attach(&cmd, target);
    gpu::ColorTargetDesc[1] colors = {{
        .view     = target_view,
        .load_op  = gpu::LoadOp.CLEAR,
        .store_op = gpu::StoreOp.STORE,
        .clear    = { .rgba = { 0.0f, 0.0f, 0.0f, 1.0f } },
    }};
    gpu::RenderPassDesc pass = { .colors = colors[..], .width = DP_RT, .height = DP_RT };
    gpu::GraphicsState state = gpu::render_geometry_state(DP_RT, DP_RT)!!;
    gpu::ColorTargetState[1] color_on = {{ .write_mask = gpu::COLOR_WRITE_ALL }};
    state.color = { .targets = color_on[..] };
    gpu::cmd_begin_render_pass(&cmd, &pass)!!;
    gpu::cmd_bind_pipeline(&cmd, mesh_pipeline)!!;
    gpu::cmd_set_graphics_state(&cmd, &state)!!;
    gpu::cmd_draw_mesh_tasks_indirect(
        commands:      &cmd,
        mesh_root:     test_span_address(&device, tri.span),
        fragment_root: test_span_address(&device, color.span),
        args:          args.span,
        draw_count:    1,
    )!!;
    if (caps.draw_indirect_count) {
        gpu::cmd_draw_mesh_tasks_indirect_count(
            commands:       &cmd,
            mesh_root:      test_span_address(&device, tri.span),
            fragment_root:  test_span_address(&device, color.span),
            args:           args.span,
            count_span:     count.span,
            max_draw_count: 2,
        )!!;
    }
    gpu::cmd_end_render_pass(&cmd)!!;
    gpu::GpuSpan rb_span = readback.span;
    record_color_readback(&cmd, target, &rb_span);
    submit_and_finish(&device, &cmd);
    gpu::invalidate_mapped_span(&device, readback.span)!!;
    char* pixels = (char*)test_span_mapping(&device, readback.span).ptr;
    char* center = &pixels[(DP_RT / 2 * DP_RT + DP_RT / 2) * 4];
    assert(center[1] == (char)255 && center[0] == (char)0, "GPU-written mesh arguments draw the triangle");
    assert(validation.errors == 0, "indirect mesh draws are validation clean");
}
