module gpu_test;

import gpu::internal @public;
import gpu @public;
import gpu::internal::vk @public;
import vk;
import std::io;

const char[*] MESHLET_MESH_SPIRV = $embed("shaders/meshlet.mesh.spv");
const char[*] MESH_ARGS_SPIRV = $embed("shaders/mesh_args.comp.spv");
const char[*] SIMPLE_TASK_SPIRV = $embed("shaders/simple.task.spv");

struct MeshValidationCapture {
    uint      errors;
    char[512] first;
}

fn void capture_mesh_validation_error(gpu::DebugMessage* message, void* user_data) {
    if (!message.category.backend || !message.category.validation
        || message.severity != gpu::DebugMessageSeverity.ERROR) return;
    MeshValidationCapture* capture = (MeshValidationCapture*)user_data;
    if (capture.errors == 0 && message.backend_text != null) {
        String text = message.backend_text.str_view();
        usz len = text.len < capture.first.len - 1 ? text.len : capture.first.len - 1;
        capture.first[:len] = text[:len];
        capture.first[len] = 0;
    }
    capture.errors++;
}

fn void report_mesh_validation(MeshValidationCapture* capture, String what) {
    if (capture.errors == 0) return;
    io::eprintfn("%s: %d validation errors, first: %s", what, capture.errors, (ZString)&capture.first[0]);
}

fn vk::VkDeviceState* mesh_device_state(gpu::Device* device) {
    return gpu::internal::device_slots[device.index].data.state;
}

fn gpu::Device? make_mesh_device(
    MeshValidationCapture* capture = null,
    uint pipeline_capacity = 0,
) {
    gpu::RuntimeDesc runtime_desc = {
        .pipeline_capacity        = pipeline_capacity,
        .enable_vulkan_validation = true,
        .application_name         = "gpu_mesh_test",
        .debug_callback           = capture == null ? null : &capture_mesh_validation_error,
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

fn gpu::PipelineHandle make_mesh_pipeline(gpu::Device* device, DepthCtx* ctx, bool with_task = false) {
    gpu::ShaderDesc mesh = { .spirv = MESHLET_MESH_SPIRV[..], .entry_point = "main" };
    gpu::ShaderDesc task = { .spirv = SIMPLE_TASK_SPIRV[..], .entry_point = "main" };
    gpu::Format[1] color_formats = { gpu::Format.RGBA8_UNORM };
    gpu::MeshPipelineDesc desc = {
        .task_shader     = with_task ? task : (gpu::ShaderDesc){},
        .mesh_shader     = mesh,
        .fragment_shader = ctx.frag,
        .color_formats   = color_formats[..],
        .debug_name      = "meshlet",
    };
    return gpu::create_mesh_pipeline(device, &desc)!!;
}

fn gpu::PipelineHandle make_vertex_pipeline(gpu::Device* device, DepthCtx* ctx) {
    gpu::Format[1] color_formats = { gpu::Format.RGBA8_UNORM };
    gpu::GraphicsPipelineDesc desc = {
        .vertex_shader   = ctx.vert,
        .fragment_shader = ctx.frag,
        .color_formats   = color_formats[..],
    };
    return gpu::create_graphics_pipeline(device, &desc)!!;
}

fn gpu::RenderPassDesc mesh_pass(gpu::ColorTargetDesc[] colors) {
    return { .colors = colors, .width = DP_RT, .height = DP_RT };
}

<* Draws whose product exceeds the total limit while each axis fits, or
   zero when the adapter's total limit admits every per-axis combination. *>
fn gpu::Vec3u mesh_groups_over_total(gpu::MeshShaderCaps* caps) {
    gpu::Vec3u limit = caps.max_mesh_work_group_count;
    if ((ulong)limit.x * limit.y <= caps.max_mesh_work_group_total_count) return {};
    return { limit.x, limit.y, 1 };
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
}

fn void mesh_pipeline_draws_meshlet_triangle() @test {
    MeshValidationCapture validation;
    gpu::Device? candidate = make_mesh_device(&validation);
    if (catch err = candidate) {
        assert(err == gpu::UNSUPPORTED_FEATURE, "device creation faults only for missing support");
        io::printfn("mesh shaders unsupported; skipping");
        return;
    }
    gpu::Device device = candidate;
    defer destroy_test_device(&device)!!;
    validation.errors = 0;
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
    gpu::cmd_bind_pipeline(&cmd, mesh_pipeline)!!;
    gpu::cmd_set_graphics_state(&cmd, &state)!!;
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
    report_mesh_validation(&validation, "mesh draw");
    assert(validation.errors == 0, "the mesh draw is validation clean");
}

fn void mesh_indirect_draws_execute_gpu_written_arguments() @test {
    MeshValidationCapture validation;
    gpu::Device? candidate = make_mesh_device(&validation);
    if (catch err = candidate) {
        assert(err == gpu::UNSUPPORTED_FEATURE, "device creation faults only for missing support");
        return;
    }
    gpu::Device device = candidate;
    defer destroy_test_device(&device)!!;
    validation.errors = 0;
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
    report_mesh_validation(&validation, "indirect mesh draws");
    assert(validation.errors == 0, "indirect mesh draws are validation clean");
}

fn void mesh_and_vertex_pipelines_alternate_in_one_pass() @test {
    MeshValidationCapture validation;
    gpu::Device? candidate = make_mesh_device(&validation);
    if (catch err = candidate) {
        assert(err == gpu::UNSUPPORTED_FEATURE, "device creation faults only for missing support");
        io::printfn("mesh shaders unsupported; skipping");
        return;
    }
    gpu::Device device = candidate;
    defer destroy_test_device(&device)!!;
    validation.errors = 0;
    DepthCtx ctx = make_depth_ctx();
    gpu::PipelineHandle mesh_pipeline = make_mesh_pipeline(&device, &ctx);
    defer gpu::destroy_pipeline(&device, mesh_pipeline)!!;
    gpu::PipelineHandle vertex_pipeline = make_vertex_pipeline(&device, &ctx);
    defer gpu::destroy_pipeline(&device, vertex_pipeline)!!;
    gpu::TextureHandle target = make_color_target(&device);
    defer gpu::destroy_texture(&device, target)!!;
    gpu::AttachmentViewHandle target_view = create_test_attachment_view(&device, target)!!;
    defer destroy_test_attachment_view(&device, &target_view)!!;
    OwnedSpan tri = write_tri(&device, 0.5f);
    defer free_owned_span(&device, &tri)!!;
    OwnedSpan color = write_color(&device, { 1.0f, 0.0f, 0.0f, 1.0f });
    defer free_owned_span(&device, &color)!!;
    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator allocator = gpu::create_command_allocator(&device, queue)!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::CommandList cmd = gpu::begin_commands(&allocator)!!;
    defer discard_test_commands(&cmd)!!;
    record_color_to_attach(&cmd, target);
    gpu::ColorTargetDesc[1] colors = {{ .view = target_view, .load_op = gpu::LoadOp.CLEAR }};
    gpu::RenderPassDesc pass = mesh_pass(colors[..]);
    gpu::GraphicsState state = gpu::render_geometry_state(DP_RT, DP_RT)!!;
    gpu::ColorTargetState[1] color_on = {{ .write_mask = gpu::COLOR_WRITE_ALL }};
    state.color = { .targets = color_on[..] };
    gpu::GpuAddress tri_root = test_span_address(&device, tri.span);
    gpu::GpuAddress color_root = test_span_address(&device, color.span);
    vk::CommandRecord* record = vk::resolve_command(&cmd)!!;
    vk::VkDeviceState* device_state = mesh_device_state(&device);

    gpu::cmd_begin_render_pass(&cmd, &pass)!!;
    gpu::cmd_bind_pipeline(&cmd, vertex_pipeline)!!;
    gpu::cmd_set_graphics_state(&cmd, &state)!!;
    gpu::cmd_draw(
        commands:       &cmd,
        vertex_root:    tri_root,
        fragment_root:  color_root,
        vertex_count:   3,
        instance_count: 1,
    )!!;
    char[112] payload;
    gpu::cmd_draw(
        commands:       &cmd,
        vertex_root:    tri_root,
        fragment_root:  color_root,
        vertex_count:   3,
        instance_count: 1,
        inline_root:    payload[..],
    )!!;
    gpu::cmd_bind_pipeline(&cmd, mesh_pipeline)!!;
    assert(record.descriptor_heap_binding.graphics_layout == device_state.mesh_layout,
        "binding a mesh pipeline rebinds the heap with the mesh layout");
    gpu::cmd_set_graphics_state(&cmd, &state)!!;
    char[116] oversized;
    assert(@catch(gpu::cmd_draw_mesh_tasks(
        commands:      &cmd,
        mesh_root:     tri_root,
        fragment_root: color_root,
        groups:        { 1, 1, 1 },
        inline_root:   oversized[..],
    )) == gpu::INVALID_ARGUMENT, "a mesh payload past the 128-byte block faults");
    gpu::cmd_draw_mesh_tasks(
        commands:      &cmd,
        mesh_root:     tri_root,
        fragment_root: color_root,
        groups:        { 1, 1, 1 },
    )!!;
    assert(record.inline_root.len == 0, "a mesh draw without a payload clears the borrowed payload");
    gpu::cmd_draw_mesh_tasks(
        commands:      &cmd,
        mesh_root:     tri_root,
        fragment_root: color_root,
        groups:        { 1, 0, 1 },
    )!!;
    gpu::cmd_draw_mesh_tasks(
        commands:      &cmd,
        mesh_root:     tri_root,
        fragment_root: color_root,
        groups:        { 1, 1, 1 },
        inline_root:   payload[..],
    )!!;
    gpu::cmd_bind_pipeline(&cmd, vertex_pipeline)!!;
    assert(record.descriptor_heap_binding.graphics_layout == device_state.graphics_layout,
        "binding a vertex pipeline rebinds the heap with the graphics layout");
    gpu::cmd_draw(
        commands:       &cmd,
        vertex_root:    tri_root,
        fragment_root:  color_root,
        vertex_count:   3,
        instance_count: 1,
    )!!;
    gpu::cmd_end_render_pass(&cmd)!!;
    submit_and_finish(&device, &cmd);
    report_mesh_validation(&validation, "pipeline switch");
    assert(validation.errors == 0, "state set under a mesh pipeline carries to a later vertex draw");
}

fn void task_pipelines_release_every_shader_identity() @test {
    gpu::Device? candidate = make_mesh_device(pipeline_capacity: 1);
    if (catch err = candidate) {
        assert(err == gpu::UNSUPPORTED_FEATURE, "device creation faults only for missing support");
        io::printfn("mesh shaders unsupported; skipping");
        return;
    }
    gpu::Device device = candidate;
    defer destroy_test_device(&device)!!;
    if (!gpu::get_device_caps(&device)!!.mesh_shaders.task_shaders) {
        io::printfn("task shaders unsupported; skipping");
        return;
    }
    DepthCtx ctx = make_depth_ctx();
    vk::VkDeviceState* state = mesh_device_state(&device);
    gpu::PipelineHandle pipeline = make_mesh_pipeline(&device, &ctx, with_task: true);
    assert(vk::shader_store_live_count(state) == 3,
        "a task pipeline holds three shader identities in a capacity-one store");
    gpu::destroy_pipeline(&device, pipeline)!!;
    assert(vk::shader_store_live_count(state) == 0,
        "destroying a task pipeline releases the task identity too");
}
