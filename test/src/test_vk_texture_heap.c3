module gpu_test;

import gpu::compat @public;

// Headless (lavapipe): the bindless heap proof. Dispatch 1 writes a pattern
// into a storage image through its TextureIndex; a barrier orders the write
// against sampled reads; dispatch 2 samples the same index into a root-pointer
// buffer; readback must match. No upload machinery involved.

const char[*] HEAP_WRITE_SPIRV  = $embed("shaders/heap_write.comp.spv");
const char[*] HEAP_SAMPLE_SPIRV = $embed("shaders/heap_sample.comp.spv");

const uint TEX_SIZE = 16;

fn void bindless_texture_write_then_sample() @test {
    gpu::compat::DeviceDesc device_desc = {
        .backend              = gpu::compat::BackendKind.VULKAN,
        .enable_validation    = true,
        .frames_in_flight     = 2,
        .descriptor_heap_mode = gpu::compat::DescriptorHeapMode.DESCRIPTOR_INDEXING,
        .application_name     = "gpu_texture_heap_test",
    };
    gpu::compat::Device device = gpu::compat::create_device(&device_desc)!!;
    defer gpu::compat::destroy_device(&device)!!;

    gpu::compat::TextureDesc tex_desc = {
        .dimension  = gpu::compat::TextureDimension.TEX_2D,
        .width      = TEX_SIZE,
        .height     = TEX_SIZE,
        .format     = gpu::compat::Format.RGBA8_UNORM,
        .usage      = { .sampled, .storage },
        .debug_name = "heap_target",
    };
    gpu::compat::TextureHandle texture = gpu::compat::create_texture(&device, &tex_desc)!!;
    defer gpu::compat::destroy_texture(&device, texture)!!;

    gpu::compat::TextureIndex tex_index = gpu::compat::create_texture_descriptor(&device, texture, null)!!;
    defer gpu::compat::destroy_texture_descriptor(&device, tex_index)!!;

    gpu::compat::SamplerDesc sampler_desc = {
        .min_filter = gpu::compat::Filter.NEAREST,
        .mag_filter = gpu::compat::Filter.NEAREST,
        .address_u  = gpu::compat::AddressMode.CLAMP_TO_EDGE,
        .address_v  = gpu::compat::AddressMode.CLAMP_TO_EDGE,
        .address_w  = gpu::compat::AddressMode.CLAMP_TO_EDGE,
    };
    gpu::compat::SamplerIndex smp_index = gpu::compat::create_sampler(&device, &sampler_desc)!!;
    defer gpu::compat::destroy_sampler(&device, smp_index)!!;

    gpu::compat::BufferDesc out_desc = {
        .size        = TEX_SIZE * TEX_SIZE * 16,
        .usage       = { .storage, .addressable },
        .memory_kind = gpu::compat::MemoryKind.READBACK,
        .debug_name  = "sample_output",
    };
    gpu::compat::BufferHandle output = gpu::compat::create_buffer(&device, &out_desc)!!;
    defer gpu::compat::destroy_buffer(&device, output)!!;

    gpu::compat::ShaderDesc write_shader_desc = {
        .stage = gpu::compat::ShaderStage.COMPUTE, .spirv = HEAP_WRITE_SPIRV[..], .entry_point = "main",
    };
    gpu::compat::ShaderHandle write_shader = gpu::compat::create_shader(&device, &write_shader_desc)!!;
    defer gpu::compat::destroy_shader(&device, write_shader)!!;
    gpu::compat::ShaderDesc sample_shader_desc = {
        .stage = gpu::compat::ShaderStage.COMPUTE, .spirv = HEAP_SAMPLE_SPIRV[..], .entry_point = "main",
    };
    gpu::compat::ShaderHandle sample_shader = gpu::compat::create_shader(&device, &sample_shader_desc)!!;
    defer gpu::compat::destroy_shader(&device, sample_shader)!!;

    gpu::compat::ComputePipelineDesc write_pipe_desc = { .shader = write_shader, .push_constant_size = gpu::compat::RootPush::size };
    gpu::compat::PipelineHandle write_pipe = gpu::compat::create_compute_pipeline(&device, &write_pipe_desc)!!;
    defer gpu::compat::destroy_pipeline(&device, write_pipe)!!;
    gpu::compat::ComputePipelineDesc sample_pipe_desc = { .shader = sample_shader, .push_constant_size = gpu::compat::RootPush::size };
    gpu::compat::PipelineHandle sample_pipe = gpu::compat::create_compute_pipeline(&device, &sample_pipe_desc)!!;
    defer gpu::compat::destroy_pipeline(&device, sample_pipe)!!;

    gpu::compat::FrameToken frame_token = gpu::compat::begin_frame(&device)!!;

    gpu::compat::GpuSpan write_root_span = gpu::compat::alloc_frame_span(&frame_token, WriteRoot::size, 16)!!;
    WriteRoot* write_root = (WriteRoot*)write_root_span.cpu;
    write_root.texture_index = tex_index;
    write_root.width         = TEX_SIZE;
    write_root.height        = TEX_SIZE;

    gpu::compat::GpuSpan sample_root_span = gpu::compat::alloc_frame_span(&frame_token, SampleRoot::size, 16)!!;
    SampleRoot* sample_root = (SampleRoot*)sample_root_span.cpu;
    sample_root.output_gpu    = gpu::compat::get_buffer_address(&device, output)!!;
    sample_root.texture_index = tex_index;
    sample_root.sampler_index = smp_index;
    sample_root.width         = TEX_SIZE;
    sample_root.height        = TEX_SIZE;

    gpu::compat::CommandList cmd = gpu::compat::begin_commands(&device, gpu::compat::QueueKind.COMPUTE)!!;

    gpu::compat::TextureBarrier to_general = gpu::compat::texture_transition(
        texture: texture,
        before:  gpu::compat::TextureUse.UNDEFINED,
        after:   gpu::compat::TextureUse.STORAGE_COMPUTE,
    )!!;
    gpu::compat::cmd_texture_barrier(&cmd, &to_general)!!;

    const uint GROUPS = (TEX_SIZE + 7) / 8;
    gpu::compat::cmd_dispatch(
        commands: &cmd,
        pipeline: write_pipe,
        root:     write_root_span.gpu,
        groups:   { GROUPS, GROUPS, 1 },
    )!!;

    gpu::compat::TextureBarrier write_to_sample = gpu::compat::texture_transition(
        texture: texture,
        before:  gpu::compat::TextureUse.STORAGE_COMPUTE,
        after:   gpu::compat::TextureUse.STORAGE_COMPUTE,
    )!!;
    gpu::compat::cmd_texture_barrier(&cmd, &write_to_sample)!!;

    gpu::compat::cmd_dispatch(
        commands: &cmd,
        pipeline: sample_pipe,
        root:     sample_root_span.gpu,
        groups:   { GROUPS, GROUPS, 1 },
    )!!;

    gpu::compat::BufferBarrier out_bar = {
        .buffer = output, .offset = 0, .size = TEX_SIZE * TEX_SIZE * 16,
        .before_stage = gpu::compat::Stage.COMPUTE_SHADER, .after_stage = gpu::compat::Stage.HOST,
        .before_hazard = gpu::compat::Hazard.SHADER_WRITE, .after_hazard = gpu::compat::Hazard.HOST_READ,
    };
    gpu::compat::cmd_buffer_barrier(&cmd, &out_bar)!!;

    gpu::compat::end_commands(&cmd)!!;
    gpu::compat::CommandList[1] lists = { cmd };
    gpu::compat::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::compat::submit(&device, &submit)!!;
    gpu::compat::wait_queue_idle(&device, gpu::compat::QueueKind.COMPUTE)!!;

    gpu::compat::GpuSpan out_span = gpu::compat::get_buffer_span(&device, output)!!;
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

    gpu::compat::end_frame(&frame_token)!!;
}

fn uint quantize_unorm8(float value) {
    return (uint)(value * 255.0f + 0.5f);
}

fn void anisotropic_sampler_caps_honest() @test {
    gpu::compat::DeviceDesc desc = {
        .backend = gpu::compat::BackendKind.VULKAN, .enable_validation = true,
        .frames_in_flight = 2, .application_name = "gpu_aniso_test",
    };
    gpu::compat::Device device = gpu::compat::create_device(&desc)!!;
    defer gpu::compat::destroy_device(&device)!!;

    gpu::compat::SamplerDesc aniso = {
        .min_filter = gpu::compat::Filter.LINEAR,
        .mag_filter = gpu::compat::Filter.LINEAR,
        .mip_filter = gpu::compat::Filter.LINEAR,
        .address_u  = gpu::compat::AddressMode.REPEAT,
        .address_v  = gpu::compat::AddressMode.REPEAT,
        .address_w  = gpu::compat::AddressMode.REPEAT,
        .max_lod    = 16.0f,
        .anisotropy_enable = true,
        .max_anisotropy    = 1024.0f,
    };
    if (gpu::compat::get_device_caps(&device)!!.max_sampler_anisotropy > 0) {
        // Creation succeeds (max_anisotropy clamps to the device limit).
        gpu::compat::SamplerIndex smp = gpu::compat::create_sampler(&device, &aniso)!!;
        gpu::compat::destroy_sampler(&device, smp)!!;
    } else {
        fault err = @catch(gpu::compat::create_sampler(&device, &aniso));
        assert(err == gpu::compat::UNSUPPORTED_FEATURE, "expected UNSUPPORTED_FEATURE, got %s", err);
    }
}

fn void anisotropy_range_validated() @test {
    gpu::compat::DeviceDesc device_desc = {
        .backend           = gpu::compat::BackendKind.VULKAN,
        .enable_validation = true,
        .frames_in_flight  = 2,
        .application_name  = "gpu_aniso_range_test",
    };
    gpu::compat::Device device = gpu::compat::create_device(&device_desc)!!;
    defer gpu::compat::destroy_device(&device)!!;
    if (gpu::compat::get_device_caps(&device)!!.max_sampler_anisotropy <= 0) return;

    float[3] bad = { 0.0f, -1.0f, 0.5f };
    foreach (value : bad) {
        gpu::compat::SamplerDesc desc = {
            .min_filter        = gpu::compat::Filter.LINEAR,
            .mag_filter        = gpu::compat::Filter.LINEAR,
            .anisotropy_enable = true,
            .max_anisotropy    = value,
        };
        fault f = @catch(gpu::compat::create_sampler(&device, &desc));
        assert(f == gpu::compat::INVALID_ARGUMENT, "max_anisotropy < 1 faults INVALID_ARGUMENT");
    }

    gpu::compat::SamplerDesc huge = {
        .min_filter        = gpu::compat::Filter.LINEAR,
        .mag_filter        = gpu::compat::Filter.LINEAR,
        .anisotropy_enable = true,
        .max_anisotropy    = 1.0e6f,
    };
    gpu::compat::SamplerIndex clamped = gpu::compat::create_sampler(&device, &huge)!!;
    gpu::compat::destroy_sampler(&device, clamped)!!;
}
