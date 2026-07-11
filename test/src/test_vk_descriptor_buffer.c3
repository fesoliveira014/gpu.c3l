module gpu_test;

import gpu;
import std::os::env;
import std::io;

// Forced-DESCRIPTOR_BUFFER coverage. The shader-access end-to-end is gated
// behind GPU_C3L_RUN_DESCRIPTOR_BUFFER_E2E=1: Mesa 25.0.7 lavapipe miscompiles
// descriptor-buffer image access (JIT segfault; fixed in later Mesa).

const char[*] DB_WRITE_SPIRV  = $embed("shaders/heap_write.comp.spv");
const char[*] DB_SAMPLE_SPIRV = $embed("shaders/heap_sample.comp.spv");

fn bool db_e2e_enabled() {
    if (try value = env::tget_var("GPU_C3L_RUN_DESCRIPTOR_BUFFER_E2E")) {
        return value == "1";
    }
    return false;
}

fn void descriptor_buffer_device_and_heap() @test {
    gpu::DeviceDesc desc = {
        .backend                     = gpu::BackendKind.VULKAN,
        .enable_validation           = true,
        .frames_in_flight            = 2,
        .descriptor_heap_mode        = gpu::DescriptorHeapMode.DESCRIPTOR_BUFFER,
        .texture_descriptor_capacity = 8,
        .sampler_descriptor_capacity = 4,
        .application_name            = "gpu_descriptor_buffer_test",
    };

    gpu::Device device;
    if (try dev = gpu::create_device(&desc)) {
        device = dev;
    } else {
        io::eprintfn("descriptor_buffer_device_and_heap: no capable adapter, skipping");
        return;
    }
    defer gpu::destroy_device(&device)!!;
    assert(device.caps.descriptor_buffer, "forced buffer mode implies adapter support");

    gpu::TextureDesc tex_desc = {
        .dimension = gpu::TextureDimension.TEX_2D,
        .width     = 4,
        .height    = 4,
        .format    = gpu::Format.RGBA8_UNORM,
        .usage     = { .sampled, .storage },
    };
    gpu::TextureHandle texture = gpu::create_texture(&device, &tex_desc)!!;
    defer gpu::destroy_texture(&device, texture)!!;

    gpu::TextureDescriptorDesc[2] batch = {
        { .texture = texture },
        { .texture = texture },
    };
    gpu::TextureIndex[2] indices;
    gpu::create_texture_descriptors(&device, batch[..], indices[..])!!;
    foreach (index : indices) {
        assert(index.is_valid(), "batch descriptor written into the descriptor buffer");
        gpu::destroy_texture_descriptor(&device, index)!!;
    }

    gpu::SamplerDesc sampler_desc = {};
    gpu::SamplerIndex sampler = gpu::create_sampler(&device, &sampler_desc)!!;
    gpu::destroy_sampler(&device, sampler)!!;
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
        .frames_in_flight     = 2,
        .descriptor_heap_mode = gpu::DescriptorHeapMode.DESCRIPTOR_BUFFER,
        .application_name     = "gpu_descriptor_buffer_e2e",
    };
    gpu::Device device;
    if (try dev = gpu::create_device(&device_desc)) {
        device = dev;
    } else {
        io::eprintfn("descriptor_buffer_write_then_sample: no capable adapter, skipping");
        return;
    }
    defer gpu::destroy_device(&device)!!;

    gpu::TextureDesc tex_desc = {
        .dimension = gpu::TextureDimension.TEX_2D,
        .width     = TEX,
        .height    = TEX,
        .format    = gpu::Format.RGBA8_UNORM,
        .usage     = { .sampled, .storage },
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
        .size        = TEX * TEX * 16,
        .usage       = { .storage, .addressable },
        .memory_kind = gpu::MemoryKind.READBACK,
    };
    gpu::BufferHandle output = gpu::create_buffer(&device, &out_desc)!!;
    defer gpu::destroy_buffer(&device, output)!!;

    gpu::ShaderDesc write_sd = { .stage = gpu::ShaderStage.COMPUTE, .spirv = DB_WRITE_SPIRV[..], .entry_point = "main" };
    gpu::ShaderHandle write_shader = gpu::create_shader(&device, &write_sd)!!;
    defer gpu::destroy_shader(&device, write_shader)!!;
    gpu::ShaderDesc sample_sd = { .stage = gpu::ShaderStage.COMPUTE, .spirv = DB_SAMPLE_SPIRV[..], .entry_point = "main" };
    gpu::ShaderHandle sample_shader = gpu::create_shader(&device, &sample_sd)!!;
    defer gpu::destroy_shader(&device, sample_shader)!!;

    gpu::ComputePipelineDesc write_pd = { .shader = write_shader, .push_constant_size = gpu::RootPush::size };
    gpu::PipelineHandle write_pipe = gpu::create_compute_pipeline(&device, &write_pd)!!;
    defer gpu::destroy_pipeline(&device, write_pipe)!!;
    gpu::ComputePipelineDesc sample_pd = { .shader = sample_shader, .push_constant_size = gpu::RootPush::size };
    gpu::PipelineHandle sample_pipe = gpu::create_compute_pipeline(&device, &sample_pd)!!;
    defer gpu::destroy_pipeline(&device, sample_pipe)!!;

    gpu::begin_frame(&device)!!;

    gpu::GpuSpan write_root_span = gpu::alloc_frame_span(&device, WriteRoot::size, 16)!!;
    WriteRoot* write_root = (WriteRoot*)write_root_span.cpu;
    write_root.texture_index = tex_index;
    write_root.width         = TEX;
    write_root.height        = TEX;

    gpu::GpuSpan sample_root_span = gpu::alloc_frame_span(&device, SampleRoot::size, 16)!!;
    SampleRoot* sample_root = (SampleRoot*)sample_root_span.cpu;
    sample_root.output_gpu    = gpu::get_buffer_address(&device, output)!!;
    sample_root.texture_index = tex_index;
    sample_root.sampler_index = smp_index;
    sample_root.width         = TEX;
    sample_root.height        = TEX;

    gpu::CommandList cmd = gpu::begin_commands(&device, gpu::QueueKind.COMPUTE)!!;
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
        root:     write_root_span.gpu,
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
        root:     sample_root_span.gpu,
        groups:   { GROUPS, GROUPS, 1 },
    )!!;

    gpu::BufferBarrier out_bar = {
        .buffer = output, .offset = 0, .size = TEX * TEX * 16,
        .before_stage = gpu::Stage.COMPUTE_SHADER, .after_stage = gpu::Stage.HOST,
        .before_hazard = gpu::Hazard.SHADER_WRITE, .after_hazard = gpu::Hazard.HOST_READ,
    };
    gpu::cmd_buffer_barrier(&cmd, &out_bar)!!;
    gpu::end_commands(&device, &cmd)!!;

    gpu::CommandList[1] lists = { cmd };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::submit(&device, &submit)!!;
    gpu::wait_queue_idle(&device, gpu::QueueKind.COMPUTE)!!;

    gpu::GpuSpan out_span = gpu::get_buffer_span(&device, output)!!;
    float* texels = (float*)out_span.cpu;
    for (uint y = 0; y < TEX; y++) {
        for (uint x = 0; x < TEX; x++) {
            float* texel = &texels[(y * TEX + x) * 4];
            assert(quantize_unorm8(texel[0]) == x, "red channel matches on the buffer path");
            assert(quantize_unorm8(texel[1]) == y, "green channel matches on the buffer path");
            assert(quantize_unorm8(texel[2]) == (x ^ y), "blue channel matches on the buffer path");
        }
    }
    gpu::end_frame(&device)!!;
}
