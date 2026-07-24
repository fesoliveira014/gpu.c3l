module gpu_test;

import gpu @public;

// Headless (lavapipe): the bindless heap proof. Dispatch 1 writes a pattern
// into a storage image through its TextureIndex; a barrier orders the write
// against sampled reads; dispatch 2 samples the same index into a root-pointer
// buffer; readback must match. No upload machinery involved.

const char[*] HEAP_WRITE_SPIRV  = $embed("shaders/heap_write.comp.spv");
const char[*] HEAP_SAMPLE_SPIRV = $embed("shaders/heap_sample.comp.spv");
const char[*] HEAP_OFFSCREEN_VERT_SPIRV = $embed("shaders/offscreen.vert.spv");
const char[*] HEAP_OFFSCREEN_FRAG_SPIRV = $embed("shaders/offscreen.frag.spv");

const uint TEX_SIZE = 16;
const uint STORAGE_SAMPLE_X = 5;
const uint STORAGE_SAMPLE_Y = 3;
const uint STORAGE_GRAPHICS_TARGET_SIZE = 32;
const usz STORAGE_GRAPHICS_TARGET_BYTES =
    STORAGE_GRAPHICS_TARGET_SIZE * STORAGE_GRAPHICS_TARGET_SIZE * 4;

fn void bindless_texture_write_then_sample() @test {
    gpu::RuntimeDesc device_desc = {
        .backend           = gpu::BackendKind.VULKAN,
        .contract_validation      = gpu::ContractValidation.FULL,
        .track_resource_lifetimes = true,
        .enable_vulkan_validation = true,
        .application_name  = "gpu_texture_heap_test",
    };
    gpu::Device device = create_test_device(&device_desc)!!;
    defer destroy_test_device(&device)!!;

    gpu::TextureDesc tex_desc = {
        .access    = { .graphics, .compute, .transfer },
        .width      = TEX_SIZE,
        .height     = TEX_SIZE,
        .format     = gpu::Format.RGBA8_UNORM,
        .usage      = { .sampled, .storage },
        .debug_name = "heap_target",
    };
    gpu::TextureHandle texture = gpu::create_texture(&device, &tex_desc)!!;
    defer gpu::destroy_texture(&device, texture)!!;

    gpu::TextureView tex_index_view = gpu::create_texture_view(&device, texture, null)!!;
    gpu::TextureIndex tex_index = tex_index_view.index;

    gpu::SamplerDesc sampler_desc = {
        .min_filter = gpu::Filter.NEAREST,
        .mag_filter = gpu::Filter.NEAREST,
        .address_u  = gpu::AddressMode.CLAMP_TO_EDGE,
        .address_v  = gpu::AddressMode.CLAMP_TO_EDGE,
        .address_w  = gpu::AddressMode.CLAMP_TO_EDGE,
    };
    gpu::SamplerIndex smp_index = gpu::intern_sampler(&device, &sampler_desc)!!;
    OwnedSpan output = allocate_owned_span(
        device:       &device,
        size:         TEX_SIZE * TEX_SIZE * 16,
        memory_class: gpu::MemoryClass.CPU_READ,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "sample_output",
    )!!;
    defer free_owned_span(&device, &output)!!;

    gpu::ShaderDesc write_shader_desc = {
        .stage = gpu::ShaderStage.COMPUTE, .spirv = HEAP_WRITE_SPIRV[..], .entry_point = "main",
    };
    gpu::ShaderCode write_shader = gpu::prepare_shader_code(&write_shader_desc)!!;
    gpu::ShaderDesc sample_shader_desc = {
        .stage = gpu::ShaderStage.COMPUTE, .spirv = HEAP_SAMPLE_SPIRV[..], .entry_point = "main",
    };
    gpu::ShaderCode sample_shader = gpu::prepare_shader_code(&sample_shader_desc)!!;

    gpu::ComputePipelineDesc write_pipe_desc = { .shader = write_shader };
    gpu::PipelineHandle write_pipe = gpu::create_compute_pipeline(&device, &write_pipe_desc)!!;
    defer gpu::destroy_pipeline(&device, write_pipe)!!;
    gpu::ComputePipelineDesc sample_pipe_desc = { .shader = sample_shader };
    gpu::PipelineHandle sample_pipe = gpu::create_compute_pipeline(&device, &sample_pipe_desc)!!;
    defer gpu::destroy_pipeline(&device, sample_pipe)!!;
    OwnedSpan write_root_owned = allocate_owned_span(
        device:       &device,
        size:         WriteRoot::size,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       { .compute },
        debug_name:   "heap_write_root",
    )!!;
    defer free_owned_span(&device, &write_root_owned)!!;
    gpu::GpuSpan write_root_span = write_root_owned.span;
    WriteRoot* write_root = (WriteRoot*)test_span_mapping(&device, write_root_span).ptr;
    write_root.texture_index = tex_index;
    write_root.width         = TEX_SIZE;
    write_root.height        = TEX_SIZE;

    OwnedSpan sample_root_owned = allocate_owned_span(
        device:       &device,
        size:         SampleRoot::size,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       { .compute },
        debug_name:   "heap_sample_root",
    )!!;
    defer free_owned_span(&device, &sample_root_owned)!!;
    gpu::GpuSpan sample_root_span = sample_root_owned.span;
    SampleRoot* sample_root = (SampleRoot*)test_span_mapping(&device, sample_root_span).ptr;
    sample_root.output_gpu    = test_span_address(&device, output.span);
    sample_root.texture_index = tex_index;
    sample_root.sampler_index = smp_index;
    sample_root.width         = TEX_SIZE;
    sample_root.height        = TEX_SIZE;
    gpu::flush_mapped_span(&device, write_root_span)!!;
    gpu::flush_mapped_span(&device, sample_root_span)!!;

    gpu::Queue command_queue = gpu::get_queue(
        &device,
        gpu::QueueKind.COMPUTE,
    )!!;
    gpu::CommandAllocator command_allocator = gpu::create_command_allocator(
        &device,
        command_queue,
    )!!;
    defer gpu::destroy_command_allocator(&command_allocator)!!;
    gpu::CommandList cmd = gpu::begin_commands(&command_allocator)!!;
    defer discard_test_commands(&cmd)!!;

    gpu::TextureBarrier to_general = gpu::texture_transition(
        texture: texture,
        before:  { .layout = gpu::TextureLayout.UNDEFINED },
        after:   gpu::storage_at({ .compute }, { .write }),
    )!!;
    gpu::cmd_texture_barrier(&cmd, &to_general)!!;

    const uint GROUPS = (TEX_SIZE + 7) / 8;
    bind_test_compute_pipeline(&cmd, write_pipe);
    gpu::cmd_dispatch(
        commands: &cmd,
        root:     test_span_address(&device, write_root_span),
        groups:   { GROUPS, GROUPS, 1 },
    )!!;

    gpu::TextureBarrier write_to_sample = gpu::texture_transition(
        texture: texture,
        before:  gpu::storage_at({ .compute }, { .write }),
        after:   gpu::storage_at({ .compute }, { .read }),
    )!!;
    gpu::cmd_texture_barrier(&cmd, &write_to_sample)!!;

    bind_test_compute_pipeline(&cmd, sample_pipe);
    gpu::cmd_dispatch(
        commands: &cmd,
        root:     test_span_address(&device, sample_root_span),
        groups:   { GROUPS, GROUPS, 1 },
    )!!;

    gpu::Barrier out_bar = {
        .before = { .compute = true },
        .after = { .host = true },
    };
    gpu::cmd_barrier(&cmd, &out_bar)!!;

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

    assert_heap_pattern(&device, output.span, TEX_SIZE);

    gpu::destroy_texture_view(&device, tex_index_view)!!;
    const uint REUSED_TEX_SIZE = 8;
    gpu::TextureDesc reused_desc = tex_desc;
    reused_desc.width = REUSED_TEX_SIZE;
    reused_desc.height = REUSED_TEX_SIZE;
    reused_desc.debug_name = "heap_reused_target";
    gpu::TextureHandle reused_texture = gpu::create_texture(&device, &reused_desc)!!;
    defer gpu::destroy_texture(&device, reused_texture)!!;
    gpu::TextureView reused_view = gpu::create_texture_view(&device, reused_texture, null)!!;
    assert(reused_view.index == tex_index,
        "completed texture-view release recycles the same raw shader index");
    assert(reused_view.generation != tex_index_view.generation,
        "recycled texture-view ownership advances its CPU generation");

    write_root.texture_index = reused_view.index;
    write_root.width = REUSED_TEX_SIZE;
    write_root.height = REUSED_TEX_SIZE;
    sample_root.texture_index = reused_view.index;
    sample_root.width = REUSED_TEX_SIZE;
    sample_root.height = REUSED_TEX_SIZE;
    gpu::flush_mapped_span(&device, write_root_span)!!;
    gpu::flush_mapped_span(&device, sample_root_span)!!;

    gpu::CommandList reused_cmd = gpu::begin_commands(
        &command_allocator,
    )!!;
    defer discard_test_commands(&reused_cmd)!!;
    gpu::TextureBarrier reused_to_general = gpu::texture_transition(
        texture: reused_texture,
        before:  { .layout = gpu::TextureLayout.UNDEFINED },
        after:   gpu::storage_at({ .compute }, { .write }),
    )!!;
    gpu::cmd_texture_barrier(&reused_cmd, &reused_to_general)!!;
    const uint REUSED_GROUPS = (REUSED_TEX_SIZE + 7) / 8;
    bind_test_compute_pipeline(&reused_cmd, write_pipe);
    gpu::cmd_dispatch(
        commands: &reused_cmd,
        root:     test_span_address(&device, write_root_span),
        groups:   { REUSED_GROUPS, REUSED_GROUPS, 1 },
    )!!;
    gpu::TextureBarrier reused_write_to_sample = gpu::texture_transition(
        texture: reused_texture,
        before:  gpu::storage_at({ .compute }, { .write }),
        after:   gpu::storage_at({ .compute }, { .read }),
    )!!;
    gpu::cmd_texture_barrier(&reused_cmd, &reused_write_to_sample)!!;
    bind_test_compute_pipeline(&reused_cmd, sample_pipe);
    gpu::cmd_dispatch(
        commands: &reused_cmd,
        root:     test_span_address(&device, sample_root_span),
        groups:   { REUSED_GROUPS, REUSED_GROUPS, 1 },
    )!!;
    gpu::cmd_barrier(&reused_cmd, &out_bar)!!;

    gpu::ExecutableCommandList reused_executable = gpu::end_commands(&reused_cmd)!!;
    defer discard_test_executable(&reused_executable)!!;
    gpu::ExecutableCommandList[1] reused_lists = { reused_executable };
    gpu::SubmitDesc reused_submit = { .command_lists = reused_lists[..] };
    gpu::CompletionPoint reused_completion = submit_test_queue(
        &device,
        gpu::QueueKind.COMPUTE,
        &reused_submit,
    )!!;
    gpu::wait_completion(reused_completion)!!;
    assert_heap_pattern(&device, output.span, REUSED_TEX_SIZE);
    gpu::destroy_texture_view(&device, reused_view)!!;
}

fn void storage_write_then_fragment_sample() @test {
    gpu::RuntimeDesc device_desc = {
        .backend           = gpu::BackendKind.VULKAN,
        .contract_validation      = gpu::ContractValidation.FULL,
        .track_resource_lifetimes = true,
        .enable_vulkan_validation = true,
        .application_name  = "gpu_storage_fragment_test",
    };
    gpu::Device device = create_test_device(&device_desc)!!;
    defer destroy_test_device(&device)!!;

    gpu::TextureDesc sampled_desc = {
        .access     = { .graphics, .compute },
        .width      = TEX_SIZE,
        .height     = TEX_SIZE,
        .format     = gpu::Format.RGBA8_UNORM,
        .usage      = { .sampled, .storage },
        .debug_name = "storage_fragment_source",
    };
    gpu::TextureHandle sampled_texture = gpu::create_texture(&device, &sampled_desc)!!;
    defer gpu::destroy_texture(&device, sampled_texture)!!;
    gpu::TextureView sampled_view = gpu::create_texture_view(&device, sampled_texture, null)!!;
    defer gpu::destroy_texture_view(&device, sampled_view)!!;

    gpu::TextureDesc target_desc = {
        .access     = { .graphics, .transfer },
        .width      = STORAGE_GRAPHICS_TARGET_SIZE,
        .height     = STORAGE_GRAPHICS_TARGET_SIZE,
        .format     = gpu::Format.RGBA8_UNORM,
        .usage      = { .color_attach, .transfer_src },
        .debug_name = "storage_fragment_target",
    };
    gpu::TextureHandle target = gpu::create_texture(&device, &target_desc)!!;
    defer gpu::destroy_texture(&device, target)!!;
    gpu::AttachmentViewHandle target_view = create_test_attachment_view(&device, target)!!;
    defer destroy_test_attachment_view(&device, &target_view)!!;

    gpu::SamplerDesc sampler_desc = {
        .min_filter = gpu::Filter.NEAREST,
        .mag_filter = gpu::Filter.NEAREST,
        .address_u  = gpu::AddressMode.CLAMP_TO_EDGE,
        .address_v  = gpu::AddressMode.CLAMP_TO_EDGE,
        .address_w  = gpu::AddressMode.CLAMP_TO_EDGE,
    };
    gpu::SamplerIndex sampler_index = gpu::intern_sampler(&device, &sampler_desc)!!;

    gpu::ShaderDesc write_shader_desc = {
        .stage       = gpu::ShaderStage.COMPUTE,
        .spirv       = HEAP_WRITE_SPIRV[..],
        .entry_point = "main",
    };
    gpu::ShaderCode write_shader = gpu::prepare_shader_code(&write_shader_desc)!!;
    gpu::ComputePipelineDesc write_pipeline_desc = { .shader = write_shader };
    gpu::PipelineHandle write_pipeline =
        gpu::create_compute_pipeline(&device, &write_pipeline_desc)!!;
    defer gpu::destroy_pipeline(&device, write_pipeline)!!;

    gpu::ShaderDesc vertex_shader_desc = {
        .stage       = gpu::ShaderStage.VERTEX,
        .spirv       = HEAP_OFFSCREEN_VERT_SPIRV[..],
        .entry_point = "main",
    };
    gpu::ShaderCode vertex_shader = gpu::prepare_shader_code(&vertex_shader_desc)!!;
    gpu::ShaderDesc fragment_shader_desc = {
        .stage       = gpu::ShaderStage.FRAGMENT,
        .spirv       = HEAP_OFFSCREEN_FRAG_SPIRV[..],
        .entry_point = "main",
    };
    gpu::ShaderCode fragment_shader = gpu::prepare_shader_code(&fragment_shader_desc)!!;
    gpu::ColorTargetState[1] pipeline_colors = {{
        .format     = gpu::Format.RGBA8_UNORM,
        .write_mask = gpu::COLOR_WRITE_ALL,
    }};
    gpu::GraphicsPipelineDesc graphics_pipeline_desc = {
        .vertex_shader   = vertex_shader,
        .fragment_shader = fragment_shader,
        .colors          = pipeline_colors[..],
        .debug_name      = "storage_fragment_sample",
    };
    gpu::PipelineHandle graphics_pipeline =
        gpu::create_graphics_pipeline(&device, &graphics_pipeline_desc)!!;
    defer gpu::destroy_pipeline(&device, graphics_pipeline)!!;

    OwnedSpan write_root_owned = allocate_owned_span(
        device:       &device,
        size:         WriteRoot::size,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       { .compute },
        debug_name:   "storage_fragment_write_root",
    )!!;
    defer free_owned_span(&device, &write_root_owned)!!;
    gpu::GpuSpan write_root_span = write_root_owned.span;
    WriteRoot* write_root = (WriteRoot*)test_span_mapping(&device, write_root_span).ptr;
    write_root.texture_index = sampled_view.index;
    write_root.width = TEX_SIZE;
    write_root.height = TEX_SIZE;
    gpu::flush_mapped_span(&device, write_root_span)!!;

    OwnedSpan vertices_owned = allocate_owned_span(
        device:       &device,
        size:         3 * 16,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       { .graphics },
        debug_name:   "storage_fragment_vertices",
    )!!;
    defer free_owned_span(&device, &vertices_owned)!!;
    gpu::GpuSpan vertices_span = vertices_owned.span;
    float sample_u = ((float)STORAGE_SAMPLE_X + 0.5f) / (float)TEX_SIZE;
    float sample_v = ((float)STORAGE_SAMPLE_Y + 0.5f) / (float)TEX_SIZE;
    float[12] vertex_data = {
        -0.8f, -0.8f, sample_u, sample_v,
         0.8f, -0.8f, sample_u, sample_v,
         0.0f,  0.8f, sample_u, sample_v,
    };
    float* vertices = (float*)test_span_mapping(&device, vertices_span).ptr;
    for (uint i = 0; i < vertex_data.len; i++) vertices[i] = vertex_data[i];
    gpu::flush_mapped_span(&device, vertices_span)!!;

    OwnedSpan fragment_root_owned = allocate_owned_span(
        device:       &device,
        size:         FragRoot::size,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       { .graphics },
        debug_name:   "storage_fragment_root",
    )!!;
    defer free_owned_span(&device, &fragment_root_owned)!!;
    gpu::GpuSpan fragment_root_span = fragment_root_owned.span;
    FragRoot* fragment_root =
        (FragRoot*)test_span_mapping(&device, fragment_root_span).ptr;
    fragment_root.texture_index = sampled_view.index;
    fragment_root.sampler_index = sampler_index;
    gpu::flush_mapped_span(&device, fragment_root_span)!!;

    OwnedSpan readback = allocate_owned_span(
        device:       &device,
        size:         STORAGE_GRAPHICS_TARGET_BYTES,
        memory_class: gpu::MemoryClass.CPU_READ,
        access:       { .graphics, .transfer },
        debug_name:   "storage_fragment_readback",
    )!!;
    defer free_owned_span(&device, &readback)!!;

    gpu::Queue command_queue = gpu::get_queue(
        &device,
        gpu::QueueKind.GRAPHICS,
    )!!;
    gpu::CommandAllocator command_allocator = gpu::create_command_allocator(
        &device,
        command_queue,
    )!!;
    defer gpu::destroy_command_allocator(&command_allocator)!!;
    gpu::CommandList commands = gpu::begin_commands(&command_allocator)!!;
    defer discard_test_commands(&commands)!!;
    gpu::TextureBarrier to_storage = gpu::texture_transition(
        texture: sampled_texture,
        before:  { .layout = gpu::TextureLayout.UNDEFINED },
        after:   gpu::storage_at({ .compute }, { .write }),
    )!!;
    gpu::cmd_texture_barrier(&commands, &to_storage)!!;
    bind_test_compute_pipeline(&commands, write_pipeline);
    const uint GROUPS = (TEX_SIZE + HEAP_TILE - 1) / HEAP_TILE;
    gpu::cmd_dispatch(
        commands: &commands,
        root:     test_span_address(&device, write_root_span),
        groups:   { GROUPS, GROUPS, 1 },
    )!!;

    gpu::TextureBarrier write_to_fragment = gpu::texture_transition(
        texture: sampled_texture,
        before:  gpu::storage_at({ .compute }, { .write }),
        after:   gpu::sampled_at({ .fragment_shader }),
    )!!;
    gpu::cmd_texture_barrier(&commands, &write_to_fragment)!!;

    gpu::TextureBarrier target_to_attachment = gpu::texture_transition(
        texture: target,
        before:  { .layout = gpu::TextureLayout.UNDEFINED },
        after:   {
            .layout = gpu::TextureLayout.COLOR_ATTACHMENT,
            .stages = { .color_output },
            .access = { .read, .write },
        },
    )!!;
    gpu::cmd_texture_barrier(&commands, &target_to_attachment)!!;
    gpu::ColorTargetDesc[1] colors = {{
        .view     = target_view,
        .load_op  = gpu::LoadOp.CLEAR,
        .store_op = gpu::StoreOp.STORE,
        .clear    = { .rgba = { 0.0f, 0.0f, 0.0f, 1.0f } },
    }};
    gpu::RenderPassDesc pass = {
        .colors = colors[..],
        .width  = STORAGE_GRAPHICS_TARGET_SIZE,
        .height = STORAGE_GRAPHICS_TARGET_SIZE,
    };
    gpu::GraphicsState graphics =
        gpu::full_render_graphics_state(pass.width, pass.height)!!;
    gpu::cmd_begin_render_pass_with_state(&commands, &pass, &graphics)!!;
    bind_test_graphics_pipeline(&commands, graphics_pipeline);
    gpu::cmd_draw(
        commands:       &commands,
        vertex_root:    test_span_address(&device, vertices_span),
        fragment_root:  test_span_address(&device, fragment_root_span),
        vertex_count:   3,
        instance_count: 1,
    )!!;
    gpu::cmd_end_render_pass(&commands)!!;

    gpu::TextureBarrier target_to_transfer = gpu::texture_transition(
        texture: target,
        before:  {
            .layout = gpu::TextureLayout.COLOR_ATTACHMENT,
            .stages = { .color_output },
            .access = { .read, .write },
        },
        after:   {
            .layout = gpu::TextureLayout.TRANSFER_SOURCE,
            .stages = { .transfer },
            .access = { .read },
        },
    )!!;
    gpu::cmd_texture_barrier(&commands, &target_to_transfer)!!;
    gpu::GpuSpan readback_span = readback.span;
    gpu::TextureBufferCopyDesc copy = {
        .texture = target,
        .dst     = readback_span,
    };
    gpu::cmd_copy_texture_to_buffer(&commands, &copy)!!;
    gpu::Barrier host_read = {
        .before = { .transfer = true },
        .after  = { .host = true },
    };
    gpu::cmd_barrier(&commands, &host_read)!!;

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
    gpu::invalidate_mapped_span(&device, readback_span)!!;

    char* pixels = (char*)test_span_mapping(&device, readback_span).ptr;
    uint center_pixel =
        (STORAGE_GRAPHICS_TARGET_SIZE / 2 * STORAGE_GRAPHICS_TARGET_SIZE
            + STORAGE_GRAPHICS_TARGET_SIZE / 2) * 4;
    assert(pixels[center_pixel] == (char)STORAGE_SAMPLE_X,
        "fragment read sees the compute-written red channel");
    assert(pixels[center_pixel + 1] == (char)STORAGE_SAMPLE_Y,
        "fragment read sees the compute-written green channel");
    assert(pixels[center_pixel + 2] == (char)(STORAGE_SAMPLE_X ^ STORAGE_SAMPLE_Y),
        "fragment read sees the compute-written blue channel");
    assert(pixels[center_pixel + 3] == (char)255,
        "fragment read sees the compute-written alpha channel");
}

fn void assert_heap_pattern(
    gpu::Device* device,
    gpu::GpuSpan output,
    uint size,
) {
    gpu::invalidate_mapped_span(device, output)!!;
    float* texels = (float*)test_span_mapping(device, output).ptr;
    for (uint y = 0; y < size; y++) {
        for (uint x = 0; x < size; x++) {
            float* texel = &texels[(y * size + x) * 4];
            // GPU UNORM8 dequantization can differ from (float)n/255 by an ulp; compare bytes.
            assert(quantize_unorm8(texel[0]) == x, "red channel matches the written pattern");
            assert(quantize_unorm8(texel[1]) == y, "green channel matches the written pattern");
            assert(quantize_unorm8(texel[2]) == (x ^ y), "blue channel matches the written pattern");
            assert(quantize_unorm8(texel[3]) == 255, "alpha is 1");
        }
    }
}

fn uint quantize_unorm8(float value) {
    return (uint)(value * 255.0f + 0.5f);
}

fn void anisotropic_sampler_caps_honest() @test {
    gpu::RuntimeDesc desc = {
        .backend           = gpu::BackendKind.VULKAN,
        .contract_validation      = gpu::ContractValidation.FULL,
        .track_resource_lifetimes = true,
        .enable_vulkan_validation = true,
        .application_name  = "gpu_aniso_test",
    };
    gpu::Device device = create_test_device(&desc)!!;
    defer destroy_test_device(&device)!!;

    gpu::SamplerDesc aniso = {
        .min_filter = gpu::Filter.LINEAR,
        .mag_filter = gpu::Filter.LINEAR,
        .mip_filter = gpu::Filter.LINEAR,
        .address_u  = gpu::AddressMode.REPEAT,
        .address_v  = gpu::AddressMode.REPEAT,
        .address_w  = gpu::AddressMode.REPEAT,
        .max_lod    = 16.0f,
        .anisotropy_enable = true,
        .max_anisotropy    = 1024.0f,
    };
    gpu::DeviceCaps caps = gpu::get_device_caps(&device)!!;
    if (caps.max_sampler_anisotropy > 0) {
        aniso.max_anisotropy = caps.max_sampler_anisotropy;
        gpu::SamplerIndex smp = gpu::intern_sampler(&device, &aniso)!!;

        aniso.max_anisotropy = caps.max_sampler_anisotropy + 1.0f;
        fault over_limit = @catch(gpu::intern_sampler(&device, &aniso));
        assert(over_limit == gpu::INVALID_ARGUMENT,
            "anisotropy above the reported limit faults INVALID_ARGUMENT");
    } else {
        fault err = @catch(gpu::intern_sampler(&device, &aniso));
        assert(err == gpu::UNSUPPORTED_FEATURE, "expected UNSUPPORTED_FEATURE, got %s", err);
    }
}

fn void anisotropy_range_validated() @test {
    gpu::RuntimeDesc device_desc = {
        .backend           = gpu::BackendKind.VULKAN,
        .contract_validation      = gpu::ContractValidation.FULL,
        .track_resource_lifetimes = true,
        .enable_vulkan_validation = true,
        .application_name  = "gpu_aniso_range_test",
    };
    gpu::Device device = create_test_device(&device_desc)!!;
    defer destroy_test_device(&device)!!;
    gpu::DeviceCaps caps = gpu::get_device_caps(&device)!!;
    if (caps.max_sampler_anisotropy <= 0) return;

    float[3] bad = { 0.0f, -1.0f, 0.5f };
    foreach (value : bad) {
        gpu::SamplerDesc desc = {
            .min_filter        = gpu::Filter.LINEAR,
            .mag_filter        = gpu::Filter.LINEAR,
            .anisotropy_enable = true,
            .max_anisotropy    = value,
        };
        fault f = @catch(gpu::intern_sampler(&device, &desc));
        assert(f == gpu::INVALID_ARGUMENT, "max_anisotropy < 1 faults INVALID_ARGUMENT");
    }

    float[3] non_finite = { float::nan, float::inf, -float::inf };
    foreach (value : non_finite) {
        gpu::SamplerDesc desc = {
            .min_filter        = gpu::Filter.LINEAR,
            .mag_filter        = gpu::Filter.LINEAR,
            .anisotropy_enable = true,
            .max_anisotropy    = value,
        };
        fault f = @catch(gpu::intern_sampler(&device, &desc));
        assert(f == gpu::INVALID_ARGUMENT,
            "non-finite max_anisotropy faults INVALID_ARGUMENT");
    }

    gpu::SamplerDesc exact = {
        .min_filter        = gpu::Filter.LINEAR,
        .mag_filter        = gpu::Filter.LINEAR,
        .anisotropy_enable = true,
        .max_anisotropy    = caps.max_sampler_anisotropy,
    };
    assert(gpu::intern_sampler(&device, &exact)!!.is_valid(),
        "the exact anisotropy limit is accepted");

    gpu::SamplerDesc over_limit = exact;
    over_limit.max_anisotropy = caps.max_sampler_anisotropy + 1.0f;
    fault over_limit_fault = @catch(gpu::intern_sampler(&device, &over_limit));
    assert(over_limit_fault == gpu::INVALID_ARGUMENT,
        "max_anisotropy above the reported limit faults INVALID_ARGUMENT");
}
