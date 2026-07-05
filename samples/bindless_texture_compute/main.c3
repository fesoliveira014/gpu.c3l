module bindless_texture_compute;

import gpu;
import std::io;

const char[*] HEAP_WRITE_SPIRV  = $embed("shaders/heap_write.comp.spv");
const char[*] HEAP_SAMPLE_SPIRV = $embed("shaders/heap_sample.comp.spv");

const uint TEX_SIZE = 16;

fn int main() {
    if (catch err = run()) {
        io::printfn("bindless_texture_compute: FAIL (%s)", err);
        return 1;
    }
    io::printn("bindless_texture_compute: PASS");
    return 0;
}

fn void? run() {
    gpu::DeviceDesc device_desc = {
        .backend           = gpu::BackendKind.VULKAN,
        .enable_validation = true,
        .frames_in_flight  = 2,
        .application_name  = "bindless_texture_compute",
    };
    gpu::Device device = gpu::create_device(&device_desc)!;
    defer (void)gpu::destroy_device(&device);

    gpu::TextureDesc tex_desc = {
        .dimension  = gpu::TextureDimension.TEX_2D,
        .width      = TEX_SIZE,
        .height     = TEX_SIZE,
        .format     = gpu::Format.RGBA8_UNORM,
        .usage      = { .sampled, .storage },
        .debug_name = "heap_target",
    };
    gpu::TextureHandle texture = gpu::create_texture(&device, &tex_desc)!;
    defer (void)gpu::destroy_texture(&device, texture);

    gpu::TextureIndex tex_index = gpu::create_texture_descriptor(&device, texture, null)!;
    defer (void)gpu::destroy_texture_descriptor(&device, tex_index);

    gpu::SamplerDesc sampler_desc = {
        .min_filter = gpu::Filter.NEAREST,
        .mag_filter = gpu::Filter.NEAREST,
        .address_u  = gpu::AddressMode.CLAMP_TO_EDGE,
        .address_v  = gpu::AddressMode.CLAMP_TO_EDGE,
        .address_w  = gpu::AddressMode.CLAMP_TO_EDGE,
    };
    gpu::SamplerIndex smp_index = gpu::create_sampler(&device, &sampler_desc)!;
    defer (void)gpu::destroy_sampler(&device, smp_index);

    gpu::BufferDesc out_desc = {
        .size        = TEX_SIZE * TEX_SIZE * 16,
        .usage       = { .storage, .addressable },
        .memory_kind = gpu::MemoryKind.READBACK,
        .debug_name  = "sample_output",
    };
    gpu::BufferHandle output = gpu::create_buffer(&device, &out_desc)!;
    defer (void)gpu::destroy_buffer(&device, output);

    gpu::ShaderDesc write_shader_desc = {
        .stage = gpu::ShaderStage.COMPUTE, .spirv = HEAP_WRITE_SPIRV[..], .entry_point = "main",
    };
    gpu::ShaderHandle write_shader = gpu::create_shader(&device, &write_shader_desc)!;
    defer (void)gpu::destroy_shader(&device, write_shader);
    gpu::ShaderDesc sample_shader_desc = {
        .stage = gpu::ShaderStage.COMPUTE, .spirv = HEAP_SAMPLE_SPIRV[..], .entry_point = "main",
    };
    gpu::ShaderHandle sample_shader = gpu::create_shader(&device, &sample_shader_desc)!;
    defer (void)gpu::destroy_shader(&device, sample_shader);

    gpu::ComputePipelineDesc write_pipe_desc = { .shader = write_shader, .push_constant_size = gpu::RootPush::size };
    gpu::PipelineHandle write_pipe = gpu::create_compute_pipeline(&device, &write_pipe_desc)!;
    defer (void)gpu::destroy_pipeline(&device, write_pipe);
    gpu::ComputePipelineDesc sample_pipe_desc = { .shader = sample_shader, .push_constant_size = gpu::RootPush::size };
    gpu::PipelineHandle sample_pipe = gpu::create_compute_pipeline(&device, &sample_pipe_desc)!;
    defer (void)gpu::destroy_pipeline(&device, sample_pipe);

    gpu::begin_frame(&device)!;

    gpu::GpuSpan write_root_span = gpu::alloc_frame_span(&device, WriteRoot::size, 16)!;
    WriteRoot* write_root = (WriteRoot*)write_root_span.cpu;
    write_root.texture_index = tex_index;
    write_root.width         = TEX_SIZE;
    write_root.height        = TEX_SIZE;

    gpu::GpuSpan sample_root_span = gpu::alloc_frame_span(&device, SampleRoot::size, 16)!;
    SampleRoot* sample_root = (SampleRoot*)sample_root_span.cpu;
    sample_root.output_gpu    = gpu::get_buffer_address(&device, output)!;
    sample_root.texture_index = tex_index;
    sample_root.sampler_index = smp_index;
    sample_root.width         = TEX_SIZE;
    sample_root.height        = TEX_SIZE;

    gpu::CommandList cmd = gpu::begin_commands(&device, gpu::QueueKind.COMPUTE)!;

    gpu::TextureBarrier to_general = {
        .texture       = texture,
        .before_stage  = gpu::Stage.HOST,
        .after_stage   = gpu::Stage.COMPUTE_SHADER,
        .before_hazard = gpu::Hazard.HOST_WRITE,
        .after_hazard  = gpu::Hazard.SHADER_WRITE,
        .old_layout    = gpu::TextureLayout.UNDEFINED,
        .new_layout    = gpu::TextureLayout.GENERAL,
    };
    gpu::cmd_texture_barrier(&cmd, &to_general)!;

    const uint GROUPS = (TEX_SIZE + HEAP_TILE - 1) / HEAP_TILE;
    gpu::cmd_dispatch(
        commands: &cmd,
        pipeline: write_pipe,
        root:     write_root_span.gpu,
        groups:   { GROUPS, GROUPS, 1 },
    )!;

    gpu::TextureBarrier write_to_sample = {
        .texture       = texture,
        .before_stage  = gpu::Stage.COMPUTE_SHADER,
        .after_stage   = gpu::Stage.COMPUTE_SHADER,
        .before_hazard = gpu::Hazard.SHADER_WRITE,
        .after_hazard  = gpu::Hazard.SHADER_READ,
        .old_layout    = gpu::TextureLayout.GENERAL,
        .new_layout    = gpu::TextureLayout.GENERAL,
    };
    gpu::cmd_texture_barrier(&cmd, &write_to_sample)!;

    gpu::cmd_dispatch(
        commands: &cmd,
        pipeline: sample_pipe,
        root:     sample_root_span.gpu,
        groups:   { GROUPS, GROUPS, 1 },
    )!;

    gpu::BufferBarrier out_bar = {
        .buffer = output, .offset = 0, .size = TEX_SIZE * TEX_SIZE * 16,
        .before_stage = gpu::Stage.COMPUTE_SHADER, .after_stage = gpu::Stage.HOST,
        .before_hazard = gpu::Hazard.SHADER_WRITE, .after_hazard = gpu::Hazard.HOST_READ,
    };
    gpu::cmd_buffer_barrier(&cmd, &out_bar)!;

    gpu::end_commands(&device, &cmd)!;
    gpu::CommandList[1] lists = { cmd };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::submit(&device, &submit)!;
    gpu::wait_queue_idle(&device, gpu::QueueKind.COMPUTE)!;

    gpu::GpuSpan out_span = gpu::get_buffer_span(&device, output)!;
    float* texels = (float*)out_span.cpu;
    for (uint y = 0; y < TEX_SIZE; y++) {
        for (uint x = 0; x < TEX_SIZE; x++) {
            float* texel = &texels[(y * TEX_SIZE + x) * 4];
            if (quantize_unorm8(texel[0]) != x
                || quantize_unorm8(texel[1]) != y
                || quantize_unorm8(texel[2]) != (x ^ y)
                || quantize_unorm8(texel[3]) != 255) {
                io::printfn("mismatch at (%d, %d): %f %f %f %f", x, y, texel[0], texel[1], texel[2], texel[3]);
                return gpu::INVALID_RESOURCE_STATE~;
            }
        }
    }

    gpu::end_frame(&device)!;
}

fn uint quantize_unorm8(float value) {
    return (uint)(value * 255.0f + 0.5f);
}
