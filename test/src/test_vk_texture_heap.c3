module gpu_test;

import gpu;

// Headless (lavapipe): the bindless heap proof. Dispatch 1 writes a pattern
// into a storage image through its TextureIndex; a barrier orders the write
// against sampled reads; dispatch 2 samples the same index into a root-pointer
// buffer; readback must match. No upload machinery involved.

const char[*] HEAP_WRITE_SPIRV  = $embed("shaders/heap_write.comp.spv");
const char[*] HEAP_SAMPLE_SPIRV = $embed("shaders/heap_sample.comp.spv");

const uint TEX_SIZE = 16;

fn void bindless_texture_write_then_sample() @test {
    gpu::DeviceDesc device_desc = {
        .backend              = gpu::BackendKind.VULKAN,
        .enable_validation    = true,
        .frames_in_flight     = 2,
        .descriptor_heap_mode = gpu::DescriptorHeapMode.DESCRIPTOR_INDEXING,
        .application_name     = "gpu_texture_heap_test",
    };
    gpu::Device device = gpu::create_device(&device_desc)!!;
    defer gpu::destroy_device(&device)!!;

    gpu::TextureDesc tex_desc = {
        .dimension  = gpu::TextureDimension.TEX_2D,
        .width      = TEX_SIZE,
        .height     = TEX_SIZE,
        .format     = gpu::Format.RGBA8_UNORM,
        .usage      = { .sampled, .storage },
        .debug_name = "heap_target",
    };
    gpu::TextureHandle texture = gpu::create_texture(&device, &tex_desc)!!;
    defer gpu::destroy_texture(&device, texture)!!;

    gpu::TextureIndex tex_index = gpu::create_texture_descriptor(&device, texture, null)!!;
    defer gpu::destroy_texture_descriptor(&device, tex_index)!!;

    gpu::SamplerDesc sampler_desc = {
        .min_filter = gpu::Filter.NEAREST,
        .mag_filter = gpu::Filter.NEAREST,
        .address_u  = gpu::AddressMode.CLAMP_TO_EDGE,
        .address_v  = gpu::AddressMode.CLAMP_TO_EDGE,
        .address_w  = gpu::AddressMode.CLAMP_TO_EDGE,
    };
    gpu::SamplerIndex smp_index = gpu::create_sampler(&device, &sampler_desc)!!;
    defer gpu::destroy_sampler(&device, smp_index)!!;

    gpu::BufferDesc out_desc = {
        .size        = TEX_SIZE * TEX_SIZE * 16,
        .usage       = { .storage, .addressable },
        .memory_kind = gpu::MemoryKind.READBACK,
        .debug_name  = "sample_output",
    };
    gpu::BufferHandle output = gpu::create_buffer(&device, &out_desc)!!;
    defer gpu::destroy_buffer(&device, output)!!;

    gpu::ShaderDesc write_shader_desc = {
        .stage = gpu::ShaderStage.COMPUTE, .spirv = HEAP_WRITE_SPIRV[..], .entry_point = "main",
    };
    gpu::ShaderHandle write_shader = gpu::create_shader(&device, &write_shader_desc)!!;
    defer gpu::destroy_shader(&device, write_shader)!!;
    gpu::ShaderDesc sample_shader_desc = {
        .stage = gpu::ShaderStage.COMPUTE, .spirv = HEAP_SAMPLE_SPIRV[..], .entry_point = "main",
    };
    gpu::ShaderHandle sample_shader = gpu::create_shader(&device, &sample_shader_desc)!!;
    defer gpu::destroy_shader(&device, sample_shader)!!;

    gpu::ComputePipelineDesc write_pipe_desc = { .shader = write_shader, .push_constant_size = gpu::RootPush::size };
    gpu::PipelineHandle write_pipe = gpu::create_compute_pipeline(&device, &write_pipe_desc)!!;
    defer gpu::destroy_pipeline(&device, write_pipe)!!;
    gpu::ComputePipelineDesc sample_pipe_desc = { .shader = sample_shader, .push_constant_size = gpu::RootPush::size };
    gpu::PipelineHandle sample_pipe = gpu::create_compute_pipeline(&device, &sample_pipe_desc)!!;
    defer gpu::destroy_pipeline(&device, sample_pipe)!!;

    gpu::FrameToken frame_token = gpu::begin_frame(&device)!!;

    gpu::GpuSpan write_root_span = gpu::alloc_frame_span(&frame_token, WriteRoot::size, 16)!!;
    WriteRoot* write_root = (WriteRoot*)write_root_span.cpu;
    write_root.texture_index = tex_index;
    write_root.width         = TEX_SIZE;
    write_root.height        = TEX_SIZE;

    gpu::GpuSpan sample_root_span = gpu::alloc_frame_span(&frame_token, SampleRoot::size, 16)!!;
    SampleRoot* sample_root = (SampleRoot*)sample_root_span.cpu;
    sample_root.output_gpu    = gpu::get_buffer_address(&device, output)!!;
    sample_root.texture_index = tex_index;
    sample_root.sampler_index = smp_index;
    sample_root.width         = TEX_SIZE;
    sample_root.height        = TEX_SIZE;

    gpu::CommandList cmd = gpu::begin_commands(&device, gpu::QueueKind.COMPUTE)!!;

    gpu::TextureBarrier to_general = gpu::texture_transition(
        texture: texture,
        before:  gpu::TextureUse.UNDEFINED,
        after:   gpu::TextureUse.STORAGE_COMPUTE,
    )!!;
    gpu::cmd_texture_barrier(&cmd, &to_general)!!;

    const uint GROUPS = (TEX_SIZE + 7) / 8;
    gpu::cmd_dispatch(
        commands: &cmd,
        pipeline: write_pipe,
        root:     write_root_span.gpu,
        groups:   { GROUPS, GROUPS, 1 },
    )!!;

    gpu::TextureBarrier write_to_sample = gpu::texture_transition(
        texture: texture,
        before:  gpu::TextureUse.STORAGE_COMPUTE,
        after:   gpu::TextureUse.STORAGE_COMPUTE,
    )!!;
    gpu::cmd_texture_barrier(&cmd, &write_to_sample)!!;

    gpu::cmd_dispatch(
        commands: &cmd,
        pipeline: sample_pipe,
        root:     sample_root_span.gpu,
        groups:   { GROUPS, GROUPS, 1 },
    )!!;

    gpu::BufferBarrier out_bar = {
        .buffer = output, .offset = 0, .size = TEX_SIZE * TEX_SIZE * 16,
        .before_stage = gpu::Stage.COMPUTE_SHADER, .after_stage = gpu::Stage.HOST,
        .before_hazard = gpu::Hazard.SHADER_WRITE, .after_hazard = gpu::Hazard.HOST_READ,
    };
    gpu::cmd_buffer_barrier(&cmd, &out_bar)!!;

    gpu::end_commands(&cmd)!!;
    gpu::CommandList[1] lists = { cmd };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::submit(&device, &submit)!!;
    gpu::wait_queue_idle(&device, gpu::QueueKind.COMPUTE)!!;

    gpu::GpuSpan out_span = gpu::get_buffer_span(&device, output)!!;
    float* texels = (float*)out_span.cpu;
    for (uint y = 0; y < TEX_SIZE; y++) {
        for (uint x = 0; x < TEX_SIZE; x++) {
            float* texel = &texels[(y * TEX_SIZE + x) * 4];
            // GPU UNORM8 dequantization can differ from (float)n/255 by an ulp; compare bytes.
            assert(quantize_unorm8(texel[0]) == x, "red channel matches the written pattern");
            assert(quantize_unorm8(texel[1]) == y, "green channel matches the written pattern");
            assert(quantize_unorm8(texel[2]) == (x ^ y), "blue channel matches the written pattern");
            assert(quantize_unorm8(texel[3]) == 255, "alpha is 1");
        }
    }

    gpu::end_frame(&frame_token)!!;
}

fn uint quantize_unorm8(float value) {
    return (uint)(value * 255.0f + 0.5f);
}

fn void anisotropic_sampler_caps_honest() @test {
    gpu::DeviceDesc desc = {
        .backend = gpu::BackendKind.VULKAN, .enable_validation = true,
        .frames_in_flight = 2, .application_name = "gpu_aniso_test",
    };
    gpu::Device device = gpu::create_device(&desc)!!;
    defer gpu::destroy_device(&device)!!;

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
    if (device.caps.max_sampler_anisotropy > 0) {
        // Creation succeeds (max_anisotropy clamps to the device limit).
        gpu::SamplerIndex smp = gpu::create_sampler(&device, &aniso)!!;
        gpu::destroy_sampler(&device, smp)!!;
    } else {
        fault err = @catch(gpu::create_sampler(&device, &aniso));
        assert(err == gpu::UNSUPPORTED_FEATURE, "expected UNSUPPORTED_FEATURE, got %s", err);
    }
}

fn void anisotropy_range_validated() @test {
    gpu::DeviceDesc device_desc = {
        .backend           = gpu::BackendKind.VULKAN,
        .enable_validation = true,
        .frames_in_flight  = 2,
        .application_name  = "gpu_aniso_range_test",
    };
    gpu::Device device = gpu::create_device(&device_desc)!!;
    defer gpu::destroy_device(&device)!!;
    if (device.caps.max_sampler_anisotropy <= 0) return;

    float[3] bad = { 0.0f, -1.0f, 0.5f };
    foreach (value : bad) {
        gpu::SamplerDesc desc = {
            .min_filter        = gpu::Filter.LINEAR,
            .mag_filter        = gpu::Filter.LINEAR,
            .anisotropy_enable = true,
            .max_anisotropy    = value,
        };
        fault f = @catch(gpu::create_sampler(&device, &desc));
        assert(f == gpu::INVALID_ARGUMENT, "max_anisotropy < 1 faults INVALID_ARGUMENT");
    }

    gpu::SamplerDesc huge = {
        .min_filter        = gpu::Filter.LINEAR,
        .mag_filter        = gpu::Filter.LINEAR,
        .anisotropy_enable = true,
        .max_anisotropy    = 1.0e6f,
    };
    gpu::SamplerIndex clamped = gpu::create_sampler(&device, &huge)!!;
    gpu::destroy_sampler(&device, clamped)!!;
}
