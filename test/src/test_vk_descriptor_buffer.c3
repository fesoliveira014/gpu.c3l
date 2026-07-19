module gpu_test;

import gpu @public;
import gpu::vk @public;
import std::os::env;
import std::io;

// A private creation probe forces descriptor-buffer coverage without exposing
// backend selection. Shader-access E2E stays gated for affected Mesa versions.

const char[*] DB_WRITE_SPIRV  = $embed("shaders/heap_write.comp.spv");
const char[*] DB_SAMPLE_SPIRV = $embed("shaders/heap_sample.comp.spv");
const char[*] DB_VERT_SPIRV   = $embed("shaders/offscreen.vert.spv");
const char[*] DB_FRAG_SPIRV   = $embed("shaders/offscreen.frag.spv");

fn bool db_e2e_enabled() {
    if (try value = env::tget_var("GPU_C3L_RUN_DESCRIPTOR_BUFFER_E2E")) {
        return value == "1";
    }
    return false;
}

fn bool descriptor_buffer_creation_is_unavailable(fault err) {
    return err == gpu::UNSUPPORTED_FEATURE;
}

fn void descriptor_buffer_creation_skip_policy_is_exact() @test {
    assert(
        descriptor_buffer_creation_is_unavailable(gpu::UNSUPPORTED_FEATURE),
        "unavailable descriptor-buffer support may skip"
    );
    assert(
        !descriptor_buffer_creation_is_unavailable(gpu::BACKEND_ERROR),
        "backend failure must fail descriptor-buffer coverage"
    );
    assert(
        !descriptor_buffer_creation_is_unavailable(gpu::INVALID_ARGUMENT),
        "malformed creation must fail descriptor-buffer coverage"
    );
}

fn gpu::Device? create_descriptor_buffer_test_device(gpu::DeviceDesc* desc) {
    gpu::vk::DeviceCreationProbe probe = {
        .descriptor_buffer_override = true,
    };
    return gpu::vk::create_vk_device_from_desc_with_probe(desc, &probe);
}

fn void descriptor_buffer_device_and_heap() @test {
    gpu::DeviceDesc desc = {
        .backend               = gpu::BackendKind.VULKAN,
        .enable_validation     = true,
        .texture_heap_capacity = 8,
        .sampler_heap_capacity = 4,
        .application_name      = "gpu_descriptor_buffer_test",
    };

    gpu::Device? maybe_device = create_descriptor_buffer_test_device(&desc);
    if (catch err = maybe_device) {
        assert(
            descriptor_buffer_creation_is_unavailable(err),
            "descriptor-buffer device/heap creation failed unexpectedly: %s",
            err
        );
        io::eprintfn("descriptor_buffer_device_and_heap: no capable adapter, skipping");
        return;
    }
    gpu::Device device = maybe_device;
    defer destroy_test_device(&device)!!;
    gpu::vk::VkDeviceState* state =
        (gpu::vk::VkDeviceState*)gpu::device_backend_state(device);
    assert(
        gpu::vk::strict_heap_uses_descriptor_buffer(&state.strict_heap),
        "the private creation probe selects descriptor buffers"
    );

    gpu::TextureDesc tex_desc = {
        .dimension = gpu::TextureDimension.TEX_2D,
        .access    = { .graphics, .compute, .transfer },
        .width     = 4,
        .height    = 4,
        .format    = gpu::Format.RGBA8_UNORM,
        .usage     = { .sampled, .storage },
    };
    gpu::TextureHandle texture = gpu::create_texture(&device, &tex_desc)!!;
    defer gpu::destroy_texture(&device, texture)!!;

    gpu::TextureViewCreateDesc[2] batch = {
        { .texture = texture },
        { .texture = texture },
    };
    gpu::TextureView[2] views;
    gpu::create_texture_views(&device, batch[..], views[..])!!;
    foreach (view : views) {
        assert(view.index.is_valid(),
            "batch view index is written into the descriptor buffer");
        gpu::destroy_texture_view(&device, view)!!;
    }

    gpu::SamplerDesc sampler_desc = {};
    gpu::Sampler sampler_identity = gpu::intern_sampler(&device, &sampler_desc)!!;
    gpu::SamplerIndex sampler = gpu::publish_sampler(&device, sampler_identity)!!;
}

fn void descriptor_buffer_compute_to_graphics() @test {
    if (!db_e2e_enabled()) {
        io::eprintfn("descriptor_buffer_compute_to_graphics: skipped (set GPU_C3L_RUN_DESCRIPTOR_BUFFER_E2E=1; known older lavapipe descriptor-buffer bug)");
        return;
    }

    const uint TEX_SIZE = 16;
    const uint TARGET_SIZE = 32;
    const usz TARGET_BYTES = TARGET_SIZE * TARGET_SIZE * 4;
    const usz VERTEX_BYTES = 3 * 16;
    gpu::DeviceDesc device_desc = {
        .backend              = gpu::BackendKind.VULKAN,
        .enable_validation    = true,
        .application_name     = "gpu_descriptor_buffer_compute_to_graphics",
    };
    gpu::Device? maybe_device = create_descriptor_buffer_test_device(&device_desc);
    if (catch err = maybe_device) {
        assert(
            descriptor_buffer_creation_is_unavailable(err),
            "strict device creation failed unexpectedly: %s",
            err
        );
        io::eprintfn("descriptor_buffer_compute_to_graphics: no capable adapter, skipping");
        return;
    }
    gpu::Device device = maybe_device;
    defer destroy_test_device(&device)!!;
    gpu::vk::VkDeviceState* state =
        (gpu::vk::VkDeviceState*)gpu::device_backend_state(device);
    assert(
        gpu::vk::strict_heap_uses_descriptor_buffer(&state.strict_heap),
        "the private creation probe selects descriptor buffers"
    );
    io::eprintfn(
        "descriptor_buffer_compute_to_graphics: graphics_family=%d compute_family=%d topology=%s",
        state.graphics_family,
        state.compute_family,
        state.graphics_family == state.compute_family ? "same-family" : "distinct-family",
    );

    gpu::TextureDesc shared_desc = {
        .dimension = gpu::TextureDimension.TEX_2D,
        .access    = { .graphics, .compute, .transfer },
        .width     = TEX_SIZE,
        .height    = TEX_SIZE,
        .format    = gpu::Format.RGBA8_UNORM,
        .usage     = { .sampled, .storage },
    };
    gpu::TextureHandle shared_texture = gpu::create_texture(&device, &shared_desc)!!;
    defer gpu::destroy_texture(&device, shared_texture)!!;
    gpu::TextureView texture_index_view = gpu::create_texture_view(&device, shared_texture, null)!!;
    gpu::TextureIndex texture_index = texture_index_view.index;
    defer gpu::destroy_texture_view(&device, texture_index_view)!!;

    gpu::SamplerDesc sampler_desc = {
        .min_filter = gpu::Filter.NEAREST,
        .mag_filter = gpu::Filter.NEAREST,
        .address_u  = gpu::AddressMode.CLAMP_TO_EDGE,
        .address_v  = gpu::AddressMode.CLAMP_TO_EDGE,
        .address_w  = gpu::AddressMode.CLAMP_TO_EDGE,
    };
    gpu::Sampler sampler = gpu::intern_sampler(&device, &sampler_desc)!!;
    gpu::SamplerIndex sampler_index = gpu::publish_sampler(&device, sampler)!!;
    OwnedSpan compute_root_buffer = allocate_owned_span(
        device:       &device,
        size:         WriteRoot::size,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "descriptor_buffer_compute_root",
    )!!;
    defer free_owned_span(&device, &compute_root_buffer)!!;
    gpu::GpuSpan compute_root_span = compute_root_buffer.span;
    WriteRoot* compute_root = (WriteRoot*)test_span_mapping(&device, compute_root_span).ptr;
    *compute_root = {
        .texture_index = texture_index,
        .width         = TEX_SIZE,
        .height        = TEX_SIZE,
    };
    gpu::flush_mapped_span(&device, compute_root_buffer.span)!!;

    OwnedSpan compute_output = allocate_owned_span(
        device:       &device,
        size:         TEX_SIZE * TEX_SIZE * 16,
        memory_class: gpu::MemoryClass.CPU_READ,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "descriptor_buffer_compute_output",
    )!!;
    defer free_owned_span(&device, &compute_output)!!;
    gpu::GpuSpan compute_output_span = compute_output.span;

    OwnedSpan sample_root_buffer = allocate_owned_span(
        device:       &device,
        size:         SampleRoot::size,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "descriptor_buffer_sample_root",
    )!!;
    defer free_owned_span(&device, &sample_root_buffer)!!;
    gpu::GpuSpan sample_root_span = sample_root_buffer.span;
    SampleRoot* sample_root = (SampleRoot*)test_span_mapping(&device, sample_root_span).ptr;
    *sample_root = {
        .output_gpu    = test_span_address(&device, compute_output_span),
        .texture_index = texture_index,
        .sampler_index = sampler_index,
        .width         = TEX_SIZE,
        .height        = TEX_SIZE,
    };
    gpu::flush_mapped_span(&device, sample_root_buffer.span)!!;

    OwnedSpan graphics_root_buffer = allocate_owned_span(
        device:       &device,
        size:         VERTEX_BYTES + FragRoot::size,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "descriptor_buffer_graphics_root",
    )!!;
    defer free_owned_span(&device, &graphics_root_buffer)!!;
    gpu::GpuSpan graphics_root_span = graphics_root_buffer.span;
    float[12] vertex_data = {
        -0.8f, -0.8f, 0.1f, 0.1f,
         0.8f, -0.8f, 0.9f, 0.1f,
         0.0f,  0.8f, 0.5f, 0.9f,
    };
    float* vertices = (float*)test_span_mapping(&device, graphics_root_span).ptr;
    for (uint i = 0; i < vertex_data.len; i++) vertices[i] = vertex_data[i];
    FragRoot* fragment_root = (FragRoot*)((char*)test_span_mapping(&device, graphics_root_span).ptr + VERTEX_BYTES);
    *fragment_root = {
        .texture_index = texture_index,
        .sampler_index = sampler_index,
    };
    gpu::flush_mapped_span(&device, graphics_root_buffer.span)!!;

    gpu::TextureDesc target_desc = {
        .dimension = gpu::TextureDimension.TEX_2D,
        .access    = { .graphics, .compute, .transfer },
        .width     = TARGET_SIZE,
        .height    = TARGET_SIZE,
        .format    = gpu::Format.RGBA8_UNORM,
        .usage     = { .color_attach, .transfer_src },
    };
    gpu::TextureHandle target = gpu::create_texture(&device, &target_desc)!!;
    defer gpu::destroy_texture(&device, target)!!;
    OwnedSpan readback = allocate_owned_span(
        device:       &device,
        size:         TARGET_BYTES,
        memory_class: gpu::MemoryClass.CPU_READ,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "descriptor_buffer_readback",
    )!!;
    defer free_owned_span(&device, &readback)!!;

    gpu::ShaderDesc write_shader_desc = { .stage = gpu::ShaderStage.COMPUTE, .spirv = DB_WRITE_SPIRV[..], .entry_point = "main" };
    gpu::ShaderCode write_shader = gpu::prepare_shader_code(&write_shader_desc)!!;
    gpu::ComputePipelineDesc compute_pipeline_desc = { .shader = write_shader, .push_constant_size = gpu::RootPush::size };
    gpu::PipelineHandle compute_pipeline = gpu::create_compute_pipeline(&device, &compute_pipeline_desc)!!;
    gpu::ShaderDesc sample_shader_desc = {
        .stage       = gpu::ShaderStage.COMPUTE,
        .spirv       = DB_SAMPLE_SPIRV[..],
        .entry_point = "main",
    };
    gpu::ShaderCode sample_shader = gpu::prepare_shader_code(&sample_shader_desc)!!;
    gpu::ComputePipelineDesc sample_pipeline_desc = {
        .shader             = sample_shader,
        .push_constant_size = gpu::RootPush::size,
    };
    gpu::PipelineHandle sample_pipeline = gpu::create_compute_pipeline(&device, &sample_pipeline_desc)!!;
    defer gpu::destroy_pipeline(&device, sample_pipeline)!!;
    defer gpu::destroy_pipeline(&device, compute_pipeline)!!;

    gpu::ShaderDesc vertex_shader_desc = { .stage = gpu::ShaderStage.VERTEX, .spirv = DB_VERT_SPIRV[..], .entry_point = "main" };
    gpu::ShaderCode vertex_shader = gpu::prepare_shader_code(&vertex_shader_desc)!!;
    gpu::ShaderDesc fragment_shader_desc = { .stage = gpu::ShaderStage.FRAGMENT, .spirv = DB_FRAG_SPIRV[..], .entry_point = "main" };
    gpu::ShaderCode fragment_shader = gpu::prepare_shader_code(&fragment_shader_desc)!!;
    gpu::Format[1] color_formats = { gpu::Format.RGBA8_UNORM };
    gpu::GraphicsPipelineDesc graphics_pipeline_desc = {
        .vertex_shader   = vertex_shader,
        .fragment_shader = fragment_shader,
        .topology        = gpu::PrimitiveTopology.TRIANGLES,
        .color_formats   = color_formats[..],
    };
    gpu::PipelineHandle graphics_pipeline = gpu::create_graphics_pipeline(&device, &graphics_pipeline_desc)!!;
    defer gpu::destroy_pipeline(&device, graphics_pipeline)!!;
    gpu::CommandList compute_commands = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.COMPUTE)!!)!!;
    defer discard_test_commands(&compute_commands)!!;
    gpu::TextureBarrier to_general = {
        .texture       = shared_texture,
        .before_stage  = gpu::Stage.HOST,
        .after_stage   = gpu::Stage.COMPUTE_SHADER,
        .before_hazard = gpu::Hazard.HOST_WRITE,
        .after_hazard  = gpu::Hazard.SHADER_WRITE,
        .old_layout    = gpu::TextureLayout.UNDEFINED,
        .new_layout    = gpu::TextureLayout.GENERAL,
    };
    gpu::cmd_texture_barrier(&compute_commands, &to_general)!!;
    const uint GROUPS = (TEX_SIZE + 7) / 8;
    gpu::cmd_dispatch(
        commands: &compute_commands,
        pipeline: compute_pipeline,
        root:     test_span_address(&device, compute_root_span),
        groups:   { GROUPS, GROUPS, 1 },
    )!!;
    gpu::TextureBarrier write_to_sample = {
        .texture       = shared_texture,
        .before_stage  = gpu::Stage.COMPUTE_SHADER,
        .after_stage   = gpu::Stage.COMPUTE_SHADER,
        .before_hazard = gpu::Hazard.SHADER_WRITE,
        .after_hazard  = gpu::Hazard.SHADER_READ,
        .old_layout    = gpu::TextureLayout.GENERAL,
        .new_layout    = gpu::TextureLayout.GENERAL,
    };
    gpu::cmd_texture_barrier(&compute_commands, &write_to_sample)!!;
    gpu::cmd_dispatch(
        commands: &compute_commands,
        pipeline: sample_pipeline,
        root:     test_span_address(&device, sample_root_span),
        groups:   { GROUPS, GROUPS, 1 },
    )!!;
    gpu::BufferBarrier compute_output_to_host = {
        .span          = compute_output_span,
        .before_stage  = gpu::Stage.COMPUTE_SHADER,
        .after_stage   = gpu::Stage.HOST,
        .before_hazard = gpu::Hazard.SHADER_WRITE,
        .after_hazard  = gpu::Hazard.HOST_READ,
    };
    gpu::cmd_buffer_barrier(&compute_commands, &compute_output_to_host)!!;
    gpu::ExecutableCommandList compute_commands_executable = gpu::end_commands(&compute_commands)!!;
    defer discard_test_executable(&compute_commands_executable)!!;
    gpu::ExecutableCommandList[1] compute_lists = { compute_commands_executable };
    gpu::SubmitDesc compute_submit = { .command_lists = compute_lists[..] };
    gpu::CompletionPoint compute_completion = submit_test_queue(&device, gpu::QueueKind.COMPUTE, &compute_submit)!!;

    gpu::CommandList graphics_commands = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!)!!;
    defer discard_test_commands(&graphics_commands)!!;
    gpu::TextureBarrier target_to_attachment = {
        .texture       = target,
        .before_stage  = gpu::Stage.HOST,
        .after_stage   = gpu::Stage.COLOR_ATTACHMENT,
        .before_hazard = gpu::Hazard.HOST_WRITE,
        .after_hazard  = gpu::Hazard.COLOR_WRITE,
        .old_layout    = gpu::TextureLayout.UNDEFINED,
        .new_layout    = gpu::TextureLayout.COLOR_ATTACHMENT,
    };
    gpu::cmd_texture_barrier(&graphics_commands, &target_to_attachment)!!;
    gpu::ColorTargetDesc[1] colors = {{
        .texture  = target,
        .load_op  = gpu::LoadOp.CLEAR,
        .store_op = gpu::StoreOp.STORE,
        .clear    = { .rgba = { 8.0f / 255.0f, 16.0f / 255.0f, 32.0f / 255.0f, 1.0f } },
    }};
    gpu::RenderPassDesc pass = { .colors = colors[..], .width = TARGET_SIZE, .height = TARGET_SIZE };
    gpu::cmd_begin_render_pass(&graphics_commands, &pass)!!;
    gpu::GpuAddress fragment_root_gpu = (gpu::GpuAddress)((ulong)test_span_address(&device, graphics_root_span) + VERTEX_BYTES);
    gpu::cmd_draw(
        commands:       &graphics_commands,
        pipeline:       graphics_pipeline,
        vertex_root:    test_span_address(&device, graphics_root_span),
        fragment_root:  fragment_root_gpu,
        vertex_count:   3,
        instance_count: 1,
    )!!;
    gpu::cmd_end_render_pass(&graphics_commands)!!;
    gpu::TextureBarrier target_to_source = {
        .texture       = target,
        .before_stage  = gpu::Stage.COLOR_ATTACHMENT,
        .after_stage   = gpu::Stage.TRANSFER,
        .before_hazard = gpu::Hazard.COLOR_WRITE,
        .after_hazard  = gpu::Hazard.TRANSFER_READ,
        .old_layout    = gpu::TextureLayout.COLOR_ATTACHMENT,
        .new_layout    = gpu::TextureLayout.TRANSFER_SRC,
    };
    gpu::cmd_texture_barrier(&graphics_commands, &target_to_source)!!;
    gpu::GpuSpan readback_span = readback.span;
    gpu::TextureBufferCopyDesc copy = { .texture = target, .dst = readback_span };
    gpu::cmd_copy_texture_to_buffer(&graphics_commands, &copy)!!;
    gpu::BufferBarrier host_read = {
        .span = readback_span,
        .before_stage = gpu::Stage.TRANSFER, .after_stage = gpu::Stage.HOST,
        .before_hazard = gpu::Hazard.TRANSFER_WRITE, .after_hazard = gpu::Hazard.HOST_READ,
    };
    gpu::cmd_buffer_barrier(&graphics_commands, &host_read)!!;
    gpu::ExecutableCommandList graphics_commands_executable = gpu::end_commands(&graphics_commands)!!;
    defer discard_test_executable(&graphics_commands_executable)!!;
    gpu::ExecutableCommandList[1] graphics_lists = { graphics_commands_executable };
    gpu::CompletionPoint[1] graphics_waits = { compute_completion };
    gpu::SubmitDesc graphics_submit = {
        .command_lists    = graphics_lists[..],
        .completion_waits = graphics_waits[..],
    };
    gpu::CompletionPoint graphics_completion = submit_test_queue(&device, gpu::QueueKind.GRAPHICS, &graphics_submit)!!;
    gpu::wait_completion(graphics_completion, ulong::max)!!;
    gpu::invalidate_mapped_span(&device, readback.span)!!;
    gpu::invalidate_mapped_span(&device, compute_output.span)!!;

    float* compute_texels = (float*)test_span_mapping(&device, compute_output_span).ptr;
    for (uint y = 0; y < TEX_SIZE; y++) {
        for (uint x = 0; x < TEX_SIZE; x++) {
            float* texel = &compute_texels[(y * TEX_SIZE + x) * 4];
            assert(quantize_unorm8(texel[0]) == x, "compute sampled red matches");
            assert(quantize_unorm8(texel[1]) == y, "compute sampled green matches");
            assert(quantize_unorm8(texel[2]) == (x ^ y), "compute sampled blue matches");
        }
    }

    char* pixels = (char*)test_span_mapping(&device, readback_span).ptr;
    char* center = &pixels[(TARGET_SIZE / 2 * TARGET_SIZE + TARGET_SIZE / 2) * 4];
    assert(center[0] == (char)8, "center red comes from the compute-written descriptor");
    assert(center[1] == (char)8, "center green comes from the compute-written descriptor");
    assert(center[2] == (char)0, "center blue comes from the compute-written descriptor");
    assert(center[3] == (char)255, "center alpha remains opaque");
    assert(pixels[0] == (char)8 && pixels[1] == (char)16 && pixels[2] == (char)32 && pixels[3] == (char)255,
        "corner remains the clear color");
    if (state.graphics_family == state.compute_family) {
        io::eprintfn("descriptor_buffer_compute_to_graphics: PASS same-family; distinct-family=N/A (%d)", state.graphics_family);
    } else {
        io::eprintfn("descriptor_buffer_compute_to_graphics: PASS distinct-family graphics=%d compute=%d",
            state.graphics_family, state.compute_family);
    }
}

fn void descriptor_buffer_write_then_sample() @test {
    if (!db_e2e_enabled()) {
        io::eprintfn("descriptor_buffer_write_then_sample: skipped (set GPU_C3L_RUN_DESCRIPTOR_BUFFER_E2E=1; known Mesa 25.0.7 lavapipe descriptor-buffer bug)");
        return;
    }

    const uint TEX = 16;
    gpu::DeviceDesc device_desc = {
        .backend              = gpu::BackendKind.VULKAN,
        .enable_validation    = true,
        .application_name     = "gpu_descriptor_buffer_e2e",
    };
    gpu::Device? maybe_device = create_descriptor_buffer_test_device(&device_desc);
    if (catch err = maybe_device) {
        assert(
            descriptor_buffer_creation_is_unavailable(err),
            "descriptor-buffer shader E2E creation failed unexpectedly: %s",
            err
        );
        io::eprintfn("descriptor_buffer_write_then_sample: no capable adapter, skipping");
        return;
    }
    gpu::Device device = maybe_device;
    defer destroy_test_device(&device)!!;
    gpu::vk::VkDeviceState* state =
        (gpu::vk::VkDeviceState*)gpu::device_backend_state(device);
    assert(
        gpu::vk::strict_heap_uses_descriptor_buffer(&state.strict_heap),
        "the private creation probe selects descriptor buffers"
    );

    gpu::TextureDesc tex_desc = {
        .dimension = gpu::TextureDimension.TEX_2D,
        .access    = { .graphics, .compute, .transfer },
        .width     = TEX,
        .height    = TEX,
        .format    = gpu::Format.RGBA8_UNORM,
        .usage     = { .sampled, .storage },
    };
    gpu::TextureHandle texture = gpu::create_texture(&device, &tex_desc)!!;
    defer gpu::destroy_texture(&device, texture)!!;
    gpu::TextureView tex_index_view = gpu::create_texture_view(&device, texture, null)!!;
    gpu::TextureIndex tex_index = tex_index_view.index;
    defer gpu::destroy_texture_view(&device, tex_index_view)!!;

    gpu::SamplerDesc sampler_desc = {
        .min_filter = gpu::Filter.NEAREST,
        .mag_filter = gpu::Filter.NEAREST,
        .address_u  = gpu::AddressMode.CLAMP_TO_EDGE,
        .address_v  = gpu::AddressMode.CLAMP_TO_EDGE,
        .address_w  = gpu::AddressMode.CLAMP_TO_EDGE,
    };
    gpu::Sampler sampler = gpu::intern_sampler(&device, &sampler_desc)!!;
    gpu::SamplerIndex smp_index = gpu::publish_sampler(&device, sampler)!!;
    OwnedSpan output = allocate_owned_span(
        device:       &device,
        size:         TEX * TEX * 16,
        memory_class: gpu::MemoryClass.CPU_READ,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "descriptor_buffer_output",
    )!!;
    defer free_owned_span(&device, &output)!!;

    gpu::ShaderDesc write_sd = { .stage = gpu::ShaderStage.COMPUTE, .spirv = DB_WRITE_SPIRV[..], .entry_point = "main" };
    gpu::ShaderCode write_shader = gpu::prepare_shader_code(&write_sd)!!;
    gpu::ShaderDesc sample_sd = { .stage = gpu::ShaderStage.COMPUTE, .spirv = DB_SAMPLE_SPIRV[..], .entry_point = "main" };
    gpu::ShaderCode sample_shader = gpu::prepare_shader_code(&sample_sd)!!;

    gpu::ComputePipelineDesc write_pd = { .shader = write_shader, .push_constant_size = gpu::RootPush::size };
    gpu::PipelineHandle write_pipe = gpu::create_compute_pipeline(&device, &write_pd)!!;
    defer gpu::destroy_pipeline(&device, write_pipe)!!;
    gpu::ComputePipelineDesc sample_pd = { .shader = sample_shader, .push_constant_size = gpu::RootPush::size };
    gpu::PipelineHandle sample_pipe = gpu::create_compute_pipeline(&device, &sample_pd)!!;
    defer gpu::destroy_pipeline(&device, sample_pipe)!!;
    OwnedSpan write_root_owned = allocate_owned_span(
        device:       &device,
        size:         WriteRoot::size,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       { .compute },
        debug_name:   "descriptor_write_root",
    )!!;
    defer free_owned_span(&device, &write_root_owned)!!;
    gpu::GpuSpan write_root_span = write_root_owned.span;
    WriteRoot* write_root = (WriteRoot*)test_span_mapping(&device, write_root_span).ptr;
    write_root.texture_index = tex_index;
    write_root.width         = TEX;
    write_root.height        = TEX;

    OwnedSpan sample_root_owned = allocate_owned_span(
        device:       &device,
        size:         SampleRoot::size,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       { .compute },
        debug_name:   "descriptor_sample_root",
    )!!;
    defer free_owned_span(&device, &sample_root_owned)!!;
    gpu::GpuSpan sample_root_span = sample_root_owned.span;
    SampleRoot* sample_root = (SampleRoot*)test_span_mapping(&device, sample_root_span).ptr;
    sample_root.output_gpu    = test_span_address(&device, output.span);
    sample_root.texture_index = tex_index;
    sample_root.sampler_index = smp_index;
    sample_root.width         = TEX;
    sample_root.height        = TEX;
    gpu::flush_mapped_span(&device, write_root_span)!!;
    gpu::flush_mapped_span(&device, sample_root_span)!!;

    gpu::CommandList cmd = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.COMPUTE)!!)!!;
    defer discard_test_commands(&cmd)!!;
    gpu::TextureBarrier to_general = {
        .texture       = texture,
        .before_stage  = gpu::Stage.HOST,
        .after_stage   = gpu::Stage.COMPUTE_SHADER,
        .before_hazard = gpu::Hazard.HOST_WRITE,
        .after_hazard  = gpu::Hazard.SHADER_WRITE,
        .old_layout    = gpu::TextureLayout.UNDEFINED,
        .new_layout    = gpu::TextureLayout.GENERAL,
    };
    gpu::cmd_texture_barrier(&cmd, &to_general)!!;

    const uint GROUPS = (TEX + 7) / 8;
    gpu::cmd_dispatch(
        commands: &cmd,
        pipeline: write_pipe,
        root:     test_span_address(&device, write_root_span),
        groups:   { GROUPS, GROUPS, 1 },
    )!!;

    gpu::TextureBarrier write_to_sample = {
        .texture       = texture,
        .before_stage  = gpu::Stage.COMPUTE_SHADER,
        .after_stage   = gpu::Stage.COMPUTE_SHADER,
        .before_hazard = gpu::Hazard.SHADER_WRITE,
        .after_hazard  = gpu::Hazard.SHADER_READ,
        .old_layout    = gpu::TextureLayout.GENERAL,
        .new_layout    = gpu::TextureLayout.GENERAL,
    };
    gpu::cmd_texture_barrier(&cmd, &write_to_sample)!!;
    gpu::cmd_dispatch(
        commands: &cmd,
        pipeline: sample_pipe,
        root:     test_span_address(&device, sample_root_span),
        groups:   { GROUPS, GROUPS, 1 },
    )!!;

    gpu::BufferBarrier out_bar = {
        .span = output.span,
        .before_stage = gpu::Stage.COMPUTE_SHADER, .after_stage = gpu::Stage.HOST,
        .before_hazard = gpu::Hazard.SHADER_WRITE, .after_hazard = gpu::Hazard.HOST_READ,
    };
    gpu::cmd_buffer_barrier(&cmd, &out_bar)!!;
    gpu::ExecutableCommandList cmd_executable = gpu::end_commands(&cmd)!!;
    defer discard_test_executable(&cmd_executable)!!;

    gpu::ExecutableCommandList[1] lists = { cmd_executable };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::CompletionPoint completion = submit_test_queue(
        &device,
        gpu::QueueKind.COMPUTE,
        &submit,
    )!!;
    gpu::wait_completion(completion)!!;

    gpu::invalidate_mapped_span(&device, output.span)!!;
    gpu::GpuSpan out_span = output.span;
    float* texels = (float*)test_span_mapping(&device, out_span).ptr;
    for (uint y = 0; y < TEX; y++) {
        for (uint x = 0; x < TEX; x++) {
            float* texel = &texels[(y * TEX + x) * 4];
            assert(quantize_unorm8(texel[0]) == x, "red channel matches on the buffer path");
            assert(quantize_unorm8(texel[1]) == y, "green channel matches on the buffer path");
            assert(quantize_unorm8(texel[2]) == (x ^ y), "blue channel matches on the buffer path");
        }
    }
}
