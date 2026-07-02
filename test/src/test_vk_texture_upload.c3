module gpu_test;

import gpu;

// Headless (lavapipe): texture uploads and readbacks. Shares SampleRoot,
// quantize_unorm8, and the embedded heap shaders with test_vk_texture_heap.c3
// (same test target).

const uint UP_SIZE = 8;
const usz MIP0_BYTES = UP_SIZE * UP_SIZE * 4;
const usz MIP1_BYTES = (UP_SIZE / 2) * (UP_SIZE / 2) * 4;

fn gpu::Device make_upload_test_device() {
    gpu::DeviceDesc desc = {
        .backend           = gpu::BackendKind.VULKAN,
        .enable_validation = true,
        .frames_in_flight  = 2,
        .application_name  = "gpu_texture_upload_test",
    };
    return gpu::create_device(&desc)!!;
}

fn gpu::GpuSpan subspan(gpu::GpuSpan span, usz offset, usz size) {
    gpu::GpuSpan result = span;
    result.offset += offset;
    result.size = size;
    result.cpu = span.cpu == null ? null : (char*)span.cpu + offset;
    return result;
}

fn void sampled_texture_upload_and_mip_readback() @test {
    gpu::Device device = make_upload_test_device();
    defer gpu::destroy_device(&device)!!;

    // Sampled-only (no storage): descriptor expects SHADER_READ, proving the
    // usage-implies-layout convention end to end.
    gpu::TextureDesc tex_desc = {
        .dimension  = gpu::TextureDimension.TEX_2D,
        .width      = UP_SIZE,
        .height     = UP_SIZE,
        .mip_levels = 2,
        .format     = gpu::Format.RGBA8_UNORM,
        .usage      = { .sampled, .transfer_src, .transfer_dst },
        .debug_name = "upload_target",
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

    gpu::BufferDesc staging_desc = {
        .size        = MIP0_BYTES + MIP1_BYTES,
        .usage       = { .transfer_src },
        .memory_kind = gpu::MemoryKind.STAGING,
        .debug_name  = "upload_staging",
    };
    gpu::BufferHandle staging = gpu::create_buffer(&device, &staging_desc)!!;
    defer gpu::destroy_buffer(&device, staging)!!;

    gpu::GpuSpan staging_span = gpu::get_buffer_span(&device, staging)!!;
    char* bytes = (char*)staging_span.cpu;
    for (uint y = 0; y < UP_SIZE; y++) {
        for (uint x = 0; x < UP_SIZE; x++) {
            char* texel = &bytes[(y * UP_SIZE + x) * 4];
            texel[0] = (char)x;
            texel[1] = (char)y;
            texel[2] = (char)(x ^ y);
            texel[3] = (char)255;
        }
    }
    for (uint y = 0; y < UP_SIZE / 2; y++) {
        for (uint x = 0; x < UP_SIZE / 2; x++) {
            char* texel = &bytes[MIP0_BYTES + (usz)(y * (UP_SIZE / 2) + x) * 4];
            texel[0] = (char)(x + 100);
            texel[1] = (char)(y + 100);
            texel[2] = (char)7;
            texel[3] = (char)255;
        }
    }
    gpu::flush_buffer(&device, staging, 0, 0)!!;

    gpu::BufferDesc out_desc = {
        .size        = UP_SIZE * UP_SIZE * 16,
        .usage       = { .storage, .addressable },
        .memory_kind = gpu::MemoryKind.READBACK,
        .debug_name  = "sample_output",
    };
    gpu::BufferHandle sample_out = gpu::create_buffer(&device, &out_desc)!!;
    defer gpu::destroy_buffer(&device, sample_out)!!;

    gpu::BufferDesc rb_desc = {
        .size        = MIP1_BYTES,
        .usage       = { .transfer_dst },
        .memory_kind = gpu::MemoryKind.READBACK,
        .debug_name  = "mip1_readback",
    };
    gpu::BufferHandle mip1_readback = gpu::create_buffer(&device, &rb_desc)!!;
    defer gpu::destroy_buffer(&device, mip1_readback)!!;

    gpu::ShaderDesc sample_sd = {
        .stage = gpu::ShaderStage.COMPUTE, .spirv = HEAP_SAMPLE_SPIRV[..], .entry_point = "main",
    };
    gpu::ShaderHandle sample_shader = gpu::create_shader(&device, &sample_sd)!!;
    defer gpu::destroy_shader(&device, sample_shader)!!;
    gpu::ComputePipelineDesc sample_pd = { .shader = sample_shader, .push_constant_size = 8 };
    gpu::PipelineHandle sample_pipe = gpu::create_compute_pipeline(&device, &sample_pd)!!;
    defer gpu::destroy_pipeline(&device, sample_pipe)!!;

    gpu::begin_frame(&device)!!;

    gpu::GpuSpan sample_root_span = gpu::alloc_frame_span(&device, SampleRoot::size, 16)!!;
    SampleRoot* sample_root = (SampleRoot*)sample_root_span.cpu;
    sample_root.output_gpu    = gpu::get_buffer_address(&device, sample_out)!!;
    sample_root.texture_index = (uint)tex_index;
    sample_root.sampler_index = (uint)smp_index;
    sample_root.width         = UP_SIZE;
    sample_root.height        = UP_SIZE;

    gpu::CommandList cmd = gpu::begin_commands(&device, gpu::QueueKind.COMPUTE)!!;

    gpu::TextureBarrier to_dst = {
        .texture       = texture,
        .before_stage  = gpu::Stage.HOST,
        .after_stage   = gpu::Stage.TRANSFER,
        .before_hazard = gpu::Hazard.HOST_WRITE,
        .after_hazard  = gpu::Hazard.TRANSFER_WRITE,
        .old_layout    = gpu::TextureLayout.UNDEFINED,
        .new_layout    = gpu::TextureLayout.TRANSFER_DST,
    };
    gpu::cmd_texture_barrier(&cmd, &to_dst)!!;

    gpu::BufferTextureCopyDesc mip0_copy = {
        .src     = subspan(staging_span, 0, MIP0_BYTES),
        .texture = texture,
        .mip     = 0,
    };
    gpu::cmd_copy_buffer_to_texture(&cmd, &mip0_copy)!!;
    gpu::BufferTextureCopyDesc mip1_copy = {
        .src     = subspan(staging_span, MIP0_BYTES, MIP1_BYTES),
        .texture = texture,
        .mip     = 1,
    };
    gpu::cmd_copy_buffer_to_texture(&cmd, &mip1_copy)!!;

    gpu::TextureBarrier to_read = {
        .texture       = texture,
        .before_stage  = gpu::Stage.TRANSFER,
        .after_stage   = gpu::Stage.COMPUTE_SHADER,
        .before_hazard = gpu::Hazard.TRANSFER_WRITE,
        .after_hazard  = gpu::Hazard.SHADER_READ,
        .old_layout    = gpu::TextureLayout.TRANSFER_DST,
        .new_layout    = gpu::TextureLayout.SHADER_READ,
    };
    gpu::cmd_texture_barrier(&cmd, &to_read)!!;

    const uint GROUPS = (UP_SIZE + 7) / 8;
    gpu::cmd_dispatch(&cmd, sample_pipe, sample_root_span.gpu, { GROUPS, GROUPS, 1 })!!;

    gpu::TextureBarrier to_src = {
        .texture       = texture,
        .before_stage  = gpu::Stage.COMPUTE_SHADER,
        .after_stage   = gpu::Stage.TRANSFER,
        .before_hazard = gpu::Hazard.SHADER_READ,
        .after_hazard  = gpu::Hazard.TRANSFER_READ,
        .old_layout    = gpu::TextureLayout.SHADER_READ,
        .new_layout    = gpu::TextureLayout.TRANSFER_SRC,
    };
    gpu::cmd_texture_barrier(&cmd, &to_src)!!;

    gpu::GpuSpan rb_span = gpu::get_buffer_span(&device, mip1_readback)!!;
    gpu::TextureBufferCopyDesc mip1_out = {
        .texture = texture,
        .dst     = rb_span,
        .mip     = 1,
    };
    gpu::cmd_copy_texture_to_buffer(&cmd, &mip1_out)!!;

    gpu::BufferBarrier host_read = {
        .buffer = mip1_readback, .offset = 0, .size = MIP1_BYTES,
        .before_stage = gpu::Stage.TRANSFER, .after_stage = gpu::Stage.HOST,
        .before_hazard = gpu::Hazard.TRANSFER_WRITE, .after_hazard = gpu::Hazard.HOST_READ,
    };
    gpu::cmd_buffer_barrier(&cmd, &host_read)!!;

    gpu::end_commands(&device, &cmd)!!;
    gpu::CommandList[1] lists = { cmd };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::submit(&device, &submit)!!;
    gpu::wait_queue_idle(&device, gpu::QueueKind.COMPUTE)!!;

    gpu::GpuSpan out_span = gpu::get_buffer_span(&device, sample_out)!!;
    float* texels = (float*)out_span.cpu;
    for (uint y = 0; y < UP_SIZE; y++) {
        for (uint x = 0; x < UP_SIZE; x++) {
            float* texel = &texels[(y * UP_SIZE + x) * 4];
            assert(quantize_unorm8(texel[0]) == x, "sampled mip 0 red matches upload");
            assert(quantize_unorm8(texel[1]) == y, "sampled mip 0 green matches upload");
            assert(quantize_unorm8(texel[2]) == (x ^ y), "sampled mip 0 blue matches upload");
        }
    }

    gpu::invalidate_buffer(&device, mip1_readback, 0, 0)!!;
    char* mip1_bytes = (char*)rb_span.cpu;
    for (uint y = 0; y < UP_SIZE / 2; y++) {
        for (uint x = 0; x < UP_SIZE / 2; x++) {
            char* texel = &mip1_bytes[(y * (UP_SIZE / 2) + x) * 4];
            assert(texel[0] == (char)(x + 100), "read-back mip 1 red matches upload");
            assert(texel[1] == (char)(y + 100), "read-back mip 1 green matches upload");
            assert(texel[2] == (char)7, "read-back mip 1 blue matches upload");
        }
    }

    gpu::end_frame(&device)!!;
}

fn void storage_write_direct_readback() @test {
    gpu::Device device = make_upload_test_device();
    defer gpu::destroy_device(&device)!!;

    gpu::TextureDesc tex_desc = {
        .dimension = gpu::TextureDimension.TEX_2D,
        .width     = UP_SIZE,
        .height    = UP_SIZE,
        .format    = gpu::Format.RGBA8_UNORM,
        .usage     = { .storage, .transfer_src },
    };
    gpu::TextureHandle texture = gpu::create_texture(&device, &tex_desc)!!;
    defer gpu::destroy_texture(&device, texture)!!;
    gpu::TextureIndex tex_index = gpu::create_texture_descriptor(&device, texture, null)!!;
    defer gpu::destroy_texture_descriptor(&device, tex_index)!!;

    gpu::BufferDesc rb_desc = {
        .size        = MIP0_BYTES,
        .usage       = { .transfer_dst },
        .memory_kind = gpu::MemoryKind.READBACK,
    };
    gpu::BufferHandle readback = gpu::create_buffer(&device, &rb_desc)!!;
    defer gpu::destroy_buffer(&device, readback)!!;

    gpu::ShaderDesc write_sd = {
        .stage = gpu::ShaderStage.COMPUTE, .spirv = HEAP_WRITE_SPIRV[..], .entry_point = "main",
    };
    gpu::ShaderHandle write_shader = gpu::create_shader(&device, &write_sd)!!;
    defer gpu::destroy_shader(&device, write_shader)!!;
    gpu::ComputePipelineDesc write_pd = { .shader = write_shader, .push_constant_size = 8 };
    gpu::PipelineHandle write_pipe = gpu::create_compute_pipeline(&device, &write_pd)!!;
    defer gpu::destroy_pipeline(&device, write_pipe)!!;

    gpu::begin_frame(&device)!!;
    gpu::GpuSpan root_span = gpu::alloc_frame_span(&device, WriteRoot::size, 16)!!;
    WriteRoot* root = (WriteRoot*)root_span.cpu;
    root.texture_index = (uint)tex_index;
    root.width         = UP_SIZE;
    root.height        = UP_SIZE;

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

    const uint GROUPS = (UP_SIZE + 7) / 8;
    gpu::cmd_dispatch(&cmd, write_pipe, root_span.gpu, { GROUPS, GROUPS, 1 })!!;

    gpu::TextureBarrier to_src = {
        .texture       = texture,
        .before_stage  = gpu::Stage.COMPUTE_SHADER,
        .after_stage   = gpu::Stage.TRANSFER,
        .before_hazard = gpu::Hazard.SHADER_WRITE,
        .after_hazard  = gpu::Hazard.TRANSFER_READ,
        .old_layout    = gpu::TextureLayout.GENERAL,
        .new_layout    = gpu::TextureLayout.TRANSFER_SRC,
    };
    gpu::cmd_texture_barrier(&cmd, &to_src)!!;

    gpu::GpuSpan rb_span = gpu::get_buffer_span(&device, readback)!!;
    gpu::TextureBufferCopyDesc copy = { .texture = texture, .dst = rb_span };
    gpu::cmd_copy_texture_to_buffer(&cmd, &copy)!!;

    gpu::BufferBarrier host_read = {
        .buffer = readback, .offset = 0, .size = MIP0_BYTES,
        .before_stage = gpu::Stage.TRANSFER, .after_stage = gpu::Stage.HOST,
        .before_hazard = gpu::Hazard.TRANSFER_WRITE, .after_hazard = gpu::Hazard.HOST_READ,
    };
    gpu::cmd_buffer_barrier(&cmd, &host_read)!!;
    gpu::end_commands(&device, &cmd)!!;

    gpu::CommandList[1] lists = { cmd };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::submit(&device, &submit)!!;
    gpu::wait_queue_idle(&device, gpu::QueueKind.COMPUTE)!!;

    gpu::invalidate_buffer(&device, readback, 0, 0)!!;
    char* bytes = (char*)rb_span.cpu;
    for (uint y = 0; y < UP_SIZE; y++) {
        for (uint x = 0; x < UP_SIZE; x++) {
            char* texel = &bytes[(y * UP_SIZE + x) * 4];
            assert(texel[0] == (char)x, "storage-written red reads back");
            assert(texel[1] == (char)y, "storage-written green reads back");
            assert(texel[2] == (char)(x ^ y), "storage-written blue reads back");
            assert(texel[3] == (char)255, "storage-written alpha reads back");
        }
    }
    gpu::end_frame(&device)!!;
}

fn void copy_validation_faults() @test {
    gpu::Device device = make_upload_test_device();
    defer gpu::destroy_device(&device)!!;

    gpu::TextureDesc tex_desc = {
        .dimension = gpu::TextureDimension.TEX_2D,
        .width     = UP_SIZE,
        .height    = UP_SIZE,
        .format    = gpu::Format.RGBA8_UNORM,
        .usage     = { .sampled, .transfer_dst },
    };
    gpu::TextureHandle texture = gpu::create_texture(&device, &tex_desc)!!;
    defer gpu::destroy_texture(&device, texture)!!;

    gpu::BufferDesc staging_desc = {
        .size        = MIP0_BYTES,
        .usage       = { .transfer_src },
        .memory_kind = gpu::MemoryKind.STAGING,
    };
    gpu::BufferHandle staging = gpu::create_buffer(&device, &staging_desc)!!;
    defer gpu::destroy_buffer(&device, staging)!!;
    gpu::GpuSpan span = gpu::get_buffer_span(&device, staging)!!;

    gpu::begin_frame(&device)!!;
    gpu::CommandList cmd = gpu::begin_commands(&device, gpu::QueueKind.COMPUTE)!!;

    // Tracked layout is UNDEFINED, not TRANSFER_DST.
    gpu::BufferTextureCopyDesc wrong_layout = { .src = span, .texture = texture };
    bool layout_faulted = false;
    if (catch gpu::cmd_copy_buffer_to_texture(&cmd, &wrong_layout)) layout_faulted = true;
    assert(layout_faulted, "copy into a non-TRANSFER_DST layout must fault");

    gpu::TextureBarrier to_dst = {
        .texture       = texture,
        .before_stage  = gpu::Stage.HOST,
        .after_stage   = gpu::Stage.TRANSFER,
        .before_hazard = gpu::Hazard.HOST_WRITE,
        .after_hazard  = gpu::Hazard.TRANSFER_WRITE,
        .old_layout    = gpu::TextureLayout.UNDEFINED,
        .new_layout    = gpu::TextureLayout.TRANSFER_DST,
    };
    gpu::cmd_texture_barrier(&cmd, &to_dst)!!;

    gpu::BufferTextureCopyDesc past_extent = { .src = span, .texture = texture, .width = UP_SIZE + 1 };
    bool extent_faulted = false;
    if (catch gpu::cmd_copy_buffer_to_texture(&cmd, &past_extent)) extent_faulted = true;
    assert(extent_faulted, "region past the mip extent must fault");

    gpu::BufferTextureCopyDesc past_span = { .src = subspan(span, 0, MIP0_BYTES - 4), .texture = texture };
    bool span_faulted = false;
    if (catch gpu::cmd_copy_buffer_to_texture(&cmd, &past_span)) span_faulted = true;
    assert(span_faulted, "copy bytes past the span must fault");

    gpu::BufferTextureCopyDesc misaligned = { .src = subspan(span, 2, MIP0_BYTES - 2), .texture = texture };
    bool align_faulted = false;
    if (catch gpu::cmd_copy_buffer_to_texture(&cmd, &misaligned)) align_faulted = true;
    assert(align_faulted, "span offset misaligned to the texel size must fault");

    gpu::BufferTextureCopyDesc bad_mip = { .src = span, .texture = texture, .mip = 1 };
    bool mip_faulted = false;
    if (catch gpu::cmd_copy_buffer_to_texture(&cmd, &bad_mip)) mip_faulted = true;
    assert(mip_faulted, "copy to a missing mip must fault");

    // No transfer_src usage: reading the texture back must fault.
    gpu::TextureBufferCopyDesc no_src_usage = { .texture = texture, .dst = span };
    bool usage_faulted = false;
    if (catch gpu::cmd_copy_texture_to_buffer(&cmd, &no_src_usage)) usage_faulted = true;
    assert(usage_faulted, "texture without transfer_src cannot be read back");

    gpu::end_commands(&device, &cmd)!!;
    gpu::end_frame(&device)!!;
}

fn void flush_invalidate_stale_handle_faults() @test {
    gpu::Device device = make_upload_test_device();
    defer gpu::destroy_device(&device)!!;

    gpu::BufferDesc desc = {
        .size        = 64,
        .usage       = { .transfer_src },
        .memory_kind = gpu::MemoryKind.STAGING,
    };
    gpu::BufferHandle buffer = gpu::create_buffer(&device, &desc)!!;
    gpu::flush_buffer(&device, buffer, 0, 0)!!;
    gpu::invalidate_buffer(&device, buffer, 0, 64)!!;
    gpu::destroy_buffer(&device, buffer)!!;

    bool faulted = false;
    if (catch gpu::flush_buffer(&device, buffer, 0, 0)) faulted = true;
    assert(faulted, "flush on a destroyed buffer must fault INVALID_HANDLE");
}
