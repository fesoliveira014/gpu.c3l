module gpu_test;

import gpu::compat;

// Headless (lavapipe): texture uploads and readbacks. Shares SampleRoot,
// quantize_unorm8, and the embedded heap shaders with test_vk_texture_heap.c3
// (same test target).

const uint UP_SIZE = 8;
const usz MIP0_BYTES = UP_SIZE * UP_SIZE * 4;
const usz MIP1_BYTES = (UP_SIZE / 2) * (UP_SIZE / 2) * 4;

fn gpu::compat::Device make_upload_test_device() {
    gpu::compat::DeviceDesc desc = {
        .backend           = gpu::compat::BackendKind.VULKAN,
        .enable_validation = true,
        .frames_in_flight  = 2,
        .application_name  = "gpu_texture_upload_test",
    };
    return gpu::compat::create_device(&desc)!!;
}

fn void sampled_texture_upload_and_mip_readback() @test {
    gpu::compat::Device device = make_upload_test_device();
    defer gpu::compat::destroy_device(&device)!!;

    // Sampled-only (no storage): descriptor expects SHADER_READ, proving the
    // usage-implies-layout convention end to end.
    gpu::compat::TextureDesc tex_desc = {
        .dimension  = gpu::compat::TextureDimension.TEX_2D,
        .width      = UP_SIZE,
        .height     = UP_SIZE,
        .mip_levels = 2,
        .format     = gpu::compat::Format.RGBA8_UNORM,
        .usage      = { .sampled, .transfer_src, .transfer_dst },
        .debug_name = "upload_target",
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

    gpu::compat::BufferDesc staging_desc = {
        .size        = MIP0_BYTES + MIP1_BYTES,
        .usage       = { .transfer_src },
        .memory_kind = gpu::compat::MemoryKind.STAGING,
        .debug_name  = "upload_staging",
    };
    gpu::compat::BufferHandle staging = gpu::compat::create_buffer(&device, &staging_desc)!!;
    defer gpu::compat::destroy_buffer(&device, staging)!!;

    gpu::compat::GpuSpan staging_span = gpu::compat::get_buffer_span(&device, staging)!!;
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
    gpu::compat::flush_buffer(&device, staging, 0, 0)!!;

    gpu::compat::BufferDesc out_desc = {
        .size        = UP_SIZE * UP_SIZE * 16,
        .usage       = { .storage, .addressable },
        .memory_kind = gpu::compat::MemoryKind.READBACK,
        .debug_name  = "sample_output",
    };
    gpu::compat::BufferHandle sample_out = gpu::compat::create_buffer(&device, &out_desc)!!;
    defer gpu::compat::destroy_buffer(&device, sample_out)!!;

    gpu::compat::BufferDesc rb_desc = {
        .size        = MIP1_BYTES,
        .usage       = { .transfer_dst },
        .memory_kind = gpu::compat::MemoryKind.READBACK,
        .debug_name  = "mip1_readback",
    };
    gpu::compat::BufferHandle mip1_readback = gpu::compat::create_buffer(&device, &rb_desc)!!;
    defer gpu::compat::destroy_buffer(&device, mip1_readback)!!;

    gpu::compat::ShaderDesc sample_sd = {
        .stage = gpu::compat::ShaderStage.COMPUTE, .spirv = HEAP_SAMPLE_SPIRV[..], .entry_point = "main",
    };
    gpu::compat::ShaderHandle sample_shader = gpu::compat::create_shader(&device, &sample_sd)!!;
    defer gpu::compat::destroy_shader(&device, sample_shader)!!;
    gpu::compat::ComputePipelineDesc sample_pd = { .shader = sample_shader, .push_constant_size = gpu::compat::RootPush::size };
    gpu::compat::PipelineHandle sample_pipe = gpu::compat::create_compute_pipeline(&device, &sample_pd)!!;
    defer gpu::compat::destroy_pipeline(&device, sample_pipe)!!;

    gpu::compat::FrameToken frame_token = gpu::compat::begin_frame(&device)!!;

    gpu::compat::GpuSpan sample_root_span = gpu::compat::alloc_frame_span(&frame_token, SampleRoot::size, 16)!!;
    SampleRoot* sample_root = (SampleRoot*)sample_root_span.cpu;
    sample_root.output_gpu    = gpu::compat::get_buffer_address(&device, sample_out)!!;
    sample_root.texture_index = tex_index;
    sample_root.sampler_index = smp_index;
    sample_root.width         = UP_SIZE;
    sample_root.height        = UP_SIZE;

    gpu::compat::CommandList cmd = gpu::compat::begin_commands(&device, gpu::compat::QueueKind.COMPUTE)!!;

    gpu::compat::TextureBarrier to_dst = gpu::compat::texture_transition(
        texture: texture,
        before:  gpu::compat::TextureUse.UNDEFINED,
        after:   gpu::compat::TextureUse.TRANSFER_DESTINATION,
    )!!;
    gpu::compat::cmd_texture_barrier(&cmd, &to_dst)!!;

    gpu::compat::BufferTextureCopyDesc mip0_copy = {
        .src     = staging_span.checked_subspan(0, MIP0_BYTES)!!,
        .texture = texture,
        .mip     = 0,
    };
    gpu::compat::cmd_copy_buffer_to_texture(&cmd, &mip0_copy)!!;
    gpu::compat::BufferTextureCopyDesc mip1_copy = {
        .src     = staging_span.checked_subspan(MIP0_BYTES, MIP1_BYTES)!!,
        .texture = texture,
        .mip     = 1,
    };
    gpu::compat::cmd_copy_buffer_to_texture(&cmd, &mip1_copy)!!;

    gpu::compat::TextureBarrier to_read = gpu::compat::texture_transition(
        texture: texture,
        before:  gpu::compat::TextureUse.TRANSFER_DESTINATION,
        after:   gpu::compat::TextureUse.SAMPLED_COMPUTE,
    )!!;
    gpu::compat::cmd_texture_barrier(&cmd, &to_read)!!;

    const uint GROUPS = (UP_SIZE + 7) / 8;
    gpu::compat::cmd_dispatch(
        commands: &cmd,
        pipeline: sample_pipe,
        root:     sample_root_span.gpu,
        groups:   { GROUPS, GROUPS, 1 },
    )!!;

    gpu::compat::TextureBarrier to_src = {
        .texture       = texture,
        .before_stage  = gpu::compat::Stage.COMPUTE_SHADER,
        .after_stage   = gpu::compat::Stage.TRANSFER,
        .before_hazard = gpu::compat::Hazard.SHADER_READ,
        .after_hazard  = gpu::compat::Hazard.TRANSFER_READ,
        .old_layout    = gpu::compat::TextureLayout.SHADER_READ,
        .new_layout    = gpu::compat::TextureLayout.TRANSFER_SRC,
    };
    gpu::compat::cmd_texture_barrier(&cmd, &to_src)!!;

    gpu::compat::GpuSpan rb_span = gpu::compat::get_buffer_span(&device, mip1_readback)!!;
    gpu::compat::TextureBufferCopyDesc mip1_out = {
        .texture = texture,
        .dst     = rb_span,
        .mip     = 1,
    };
    gpu::compat::cmd_copy_texture_to_buffer(&cmd, &mip1_out)!!;

    gpu::compat::BufferBarrier host_read = {
        .buffer = mip1_readback, .offset = 0, .size = MIP1_BYTES,
        .before_stage = gpu::compat::Stage.TRANSFER, .after_stage = gpu::compat::Stage.HOST,
        .before_hazard = gpu::compat::Hazard.TRANSFER_WRITE, .after_hazard = gpu::compat::Hazard.HOST_READ,
    };
    gpu::compat::cmd_buffer_barrier(&cmd, &host_read)!!;

    gpu::compat::end_commands(&cmd)!!;
    gpu::compat::CommandList[1] lists = { cmd };
    gpu::compat::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::compat::submit(&device, &submit)!!;
    gpu::compat::wait_queue_idle(&device, gpu::compat::QueueKind.COMPUTE)!!;

    gpu::compat::GpuSpan out_span = gpu::compat::get_buffer_span(&device, sample_out)!!;
    float* texels = (float*)out_span.cpu;
    for (uint y = 0; y < UP_SIZE; y++) {
        for (uint x = 0; x < UP_SIZE; x++) {
            float* texel = &texels[(y * UP_SIZE + x) * 4];
            assert(quantize_unorm8(texel[0]) == x, "sampled mip 0 red matches upload");
            assert(quantize_unorm8(texel[1]) == y, "sampled mip 0 green matches upload");
            assert(quantize_unorm8(texel[2]) == (x ^ y), "sampled mip 0 blue matches upload");
        }
    }

    gpu::compat::invalidate_buffer(&device, mip1_readback, 0, 0)!!;
    char* mip1_bytes = (char*)rb_span.cpu;
    for (uint y = 0; y < UP_SIZE / 2; y++) {
        for (uint x = 0; x < UP_SIZE / 2; x++) {
            char* texel = &mip1_bytes[(y * (UP_SIZE / 2) + x) * 4];
            assert(texel[0] == (char)(x + 100), "read-back mip 1 red matches upload");
            assert(texel[1] == (char)(y + 100), "read-back mip 1 green matches upload");
            assert(texel[2] == (char)7, "read-back mip 1 blue matches upload");
        }
    }

    gpu::compat::end_frame(&frame_token)!!;
}

fn void storage_write_direct_readback() @test {
    gpu::compat::Device device = make_upload_test_device();
    defer gpu::compat::destroy_device(&device)!!;

    gpu::compat::TextureDesc tex_desc = {
        .dimension = gpu::compat::TextureDimension.TEX_2D,
        .width     = UP_SIZE,
        .height    = UP_SIZE,
        .format    = gpu::compat::Format.RGBA8_UNORM,
        .usage     = { .storage, .transfer_src },
    };
    gpu::compat::TextureHandle texture = gpu::compat::create_texture(&device, &tex_desc)!!;
    defer gpu::compat::destroy_texture(&device, texture)!!;
    gpu::compat::TextureIndex tex_index = gpu::compat::create_texture_descriptor(&device, texture, null)!!;
    defer gpu::compat::destroy_texture_descriptor(&device, tex_index)!!;

    gpu::compat::BufferDesc rb_desc = {
        .size        = MIP0_BYTES,
        .usage       = { .transfer_dst },
        .memory_kind = gpu::compat::MemoryKind.READBACK,
    };
    gpu::compat::BufferHandle readback = gpu::compat::create_buffer(&device, &rb_desc)!!;
    defer gpu::compat::destroy_buffer(&device, readback)!!;

    gpu::compat::ShaderDesc write_sd = {
        .stage = gpu::compat::ShaderStage.COMPUTE, .spirv = HEAP_WRITE_SPIRV[..], .entry_point = "main",
    };
    gpu::compat::ShaderHandle write_shader = gpu::compat::create_shader(&device, &write_sd)!!;
    defer gpu::compat::destroy_shader(&device, write_shader)!!;
    gpu::compat::ComputePipelineDesc write_pd = { .shader = write_shader, .push_constant_size = gpu::compat::RootPush::size };
    gpu::compat::PipelineHandle write_pipe = gpu::compat::create_compute_pipeline(&device, &write_pd)!!;
    defer gpu::compat::destroy_pipeline(&device, write_pipe)!!;

    gpu::compat::FrameToken frame_token = gpu::compat::begin_frame(&device)!!;
    gpu::compat::GpuSpan root_span = gpu::compat::alloc_frame_span(&frame_token, WriteRoot::size, 16)!!;
    WriteRoot* root = (WriteRoot*)root_span.cpu;
    root.texture_index = tex_index;
    root.width         = UP_SIZE;
    root.height        = UP_SIZE;

    gpu::compat::CommandList cmd = gpu::compat::begin_commands(&device, gpu::compat::QueueKind.COMPUTE)!!;
    gpu::compat::TextureBarrier to_general = {
        .texture       = texture,
        .before_stage  = gpu::compat::Stage.HOST,
        .after_stage   = gpu::compat::Stage.COMPUTE_SHADER,
        .before_hazard = gpu::compat::Hazard.HOST_WRITE,
        .after_hazard  = gpu::compat::Hazard.SHADER_WRITE,
        .old_layout    = gpu::compat::TextureLayout.UNDEFINED,
        .new_layout    = gpu::compat::TextureLayout.GENERAL,
    };
    gpu::compat::cmd_texture_barrier(&cmd, &to_general)!!;

    const uint GROUPS = (UP_SIZE + 7) / 8;
    gpu::compat::cmd_dispatch(
        commands: &cmd,
        pipeline: write_pipe,
        root:     root_span.gpu,
        groups:   { GROUPS, GROUPS, 1 },
    )!!;

    gpu::compat::TextureBarrier to_src = {
        .texture       = texture,
        .before_stage  = gpu::compat::Stage.COMPUTE_SHADER,
        .after_stage   = gpu::compat::Stage.TRANSFER,
        .before_hazard = gpu::compat::Hazard.SHADER_WRITE,
        .after_hazard  = gpu::compat::Hazard.TRANSFER_READ,
        .old_layout    = gpu::compat::TextureLayout.GENERAL,
        .new_layout    = gpu::compat::TextureLayout.TRANSFER_SRC,
    };
    gpu::compat::cmd_texture_barrier(&cmd, &to_src)!!;

    gpu::compat::GpuSpan rb_span = gpu::compat::get_buffer_span(&device, readback)!!;
    gpu::compat::TextureBufferCopyDesc copy = { .texture = texture, .dst = rb_span };
    gpu::compat::cmd_copy_texture_to_buffer(&cmd, &copy)!!;

    gpu::compat::BufferBarrier host_read = {
        .buffer = readback, .offset = 0, .size = MIP0_BYTES,
        .before_stage = gpu::compat::Stage.TRANSFER, .after_stage = gpu::compat::Stage.HOST,
        .before_hazard = gpu::compat::Hazard.TRANSFER_WRITE, .after_hazard = gpu::compat::Hazard.HOST_READ,
    };
    gpu::compat::cmd_buffer_barrier(&cmd, &host_read)!!;
    gpu::compat::end_commands(&cmd)!!;

    gpu::compat::CommandList[1] lists = { cmd };
    gpu::compat::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::compat::submit(&device, &submit)!!;
    gpu::compat::wait_queue_idle(&device, gpu::compat::QueueKind.COMPUTE)!!;

    gpu::compat::invalidate_buffer(&device, readback, 0, 0)!!;
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
    gpu::compat::end_frame(&frame_token)!!;
}

fn void copy_validation_faults() @test {
    gpu::compat::Device device = make_upload_test_device();
    defer gpu::compat::destroy_device(&device)!!;

    gpu::compat::TextureDesc tex_desc = {
        .dimension = gpu::compat::TextureDimension.TEX_2D,
        .width     = UP_SIZE,
        .height    = UP_SIZE,
        .format    = gpu::compat::Format.RGBA8_UNORM,
        .usage     = { .sampled, .transfer_dst },
    };
    gpu::compat::TextureHandle texture = gpu::compat::create_texture(&device, &tex_desc)!!;
    defer gpu::compat::destroy_texture(&device, texture)!!;

    gpu::compat::BufferDesc staging_desc = {
        .size        = MIP0_BYTES,
        .usage       = { .transfer_src },
        .memory_kind = gpu::compat::MemoryKind.STAGING,
    };
    gpu::compat::BufferHandle staging = gpu::compat::create_buffer(&device, &staging_desc)!!;
    defer gpu::compat::destroy_buffer(&device, staging)!!;
    gpu::compat::GpuSpan span = gpu::compat::get_buffer_span(&device, staging)!!;

    gpu::compat::FrameToken frame_token = gpu::compat::begin_frame(&device)!!;
    gpu::compat::CommandList cmd = gpu::compat::begin_commands(&device, gpu::compat::QueueKind.COMPUTE)!!;

    // Tracked layout is UNDEFINED, not TRANSFER_DST.
    gpu::compat::BufferTextureCopyDesc wrong_layout = { .src = span, .texture = texture };
    fault layout_fault = @catch(gpu::compat::cmd_copy_buffer_to_texture(&cmd, &wrong_layout));
    assert(layout_fault == gpu::compat::INVALID_RESOURCE_STATE, "copy into a non-TRANSFER_DST layout faults INVALID_RESOURCE_STATE");

    gpu::compat::TextureBarrier to_dst = {
        .texture       = texture,
        .before_stage  = gpu::compat::Stage.HOST,
        .after_stage   = gpu::compat::Stage.TRANSFER,
        .before_hazard = gpu::compat::Hazard.HOST_WRITE,
        .after_hazard  = gpu::compat::Hazard.TRANSFER_WRITE,
        .old_layout    = gpu::compat::TextureLayout.UNDEFINED,
        .new_layout    = gpu::compat::TextureLayout.TRANSFER_DST,
    };
    gpu::compat::cmd_texture_barrier(&cmd, &to_dst)!!;

    gpu::compat::BufferTextureCopyDesc past_extent = { .src = span, .texture = texture, .width = UP_SIZE + 1 };
    fault extent_fault = @catch(gpu::compat::cmd_copy_buffer_to_texture(&cmd, &past_extent));
    assert(extent_fault == gpu::compat::INVALID_ARGUMENT, "region past the mip extent faults INVALID_ARGUMENT");

    gpu::compat::BufferTextureCopyDesc past_span = { .src = span.checked_subspan(0, MIP0_BYTES - 4)!!, .texture = texture };
    fault span_fault = @catch(gpu::compat::cmd_copy_buffer_to_texture(&cmd, &past_span));
    assert(span_fault == gpu::compat::INVALID_ARGUMENT, "copy bytes past the span fault INVALID_ARGUMENT");

    gpu::compat::BufferTextureCopyDesc misaligned = { .src = span.checked_subspan(2, MIP0_BYTES - 2)!!, .texture = texture };
    fault align_fault = @catch(gpu::compat::cmd_copy_buffer_to_texture(&cmd, &misaligned));
    assert(align_fault == gpu::compat::INVALID_ARGUMENT, "misaligned span offset faults INVALID_ARGUMENT");

    gpu::compat::BufferTextureCopyDesc bad_mip = { .src = span, .texture = texture, .mip = 1 };
    fault mip_fault = @catch(gpu::compat::cmd_copy_buffer_to_texture(&cmd, &bad_mip));
    assert(mip_fault == gpu::compat::INVALID_ARGUMENT, "copy to a missing mip faults INVALID_ARGUMENT");

    // No transfer_src usage: reading the texture back must fault.
    gpu::compat::TextureBufferCopyDesc no_src_usage = { .texture = texture, .dst = span };
    fault usage_fault = @catch(gpu::compat::cmd_copy_texture_to_buffer(&cmd, &no_src_usage));
    assert(usage_fault == gpu::compat::INVALID_ARGUMENT, "texture without transfer_src faults INVALID_ARGUMENT on readback");

    gpu::compat::end_commands(&cmd)!!;
    gpu::compat::end_frame(&frame_token)!!;
}

fn void mip_subset_view_sampled() @test {
    gpu::compat::Device device = make_upload_test_device();
    defer gpu::compat::destroy_device(&device)!!;

    gpu::compat::TextureDesc tex_desc = {
        .dimension  = gpu::compat::TextureDimension.TEX_2D,
        .width      = UP_SIZE,
        .height     = UP_SIZE,
        .mip_levels = 2,
        .format     = gpu::compat::Format.RGBA8_UNORM,
        .usage      = { .sampled, .transfer_dst },
    };
    gpu::compat::TextureHandle texture = gpu::compat::create_texture(&device, &tex_desc)!!;
    defer gpu::compat::destroy_texture(&device, texture)!!;

    // Register a mip-1-only view: sampling it at lod 0 must return mip 1.
    gpu::compat::TextureViewDesc mip1_view = { .base_mip = 1, .mip_count = 1 };
    gpu::compat::TextureIndex tex_index = gpu::compat::create_texture_descriptor(&device, texture, &mip1_view)!!;
    defer gpu::compat::destroy_texture_descriptor(&device, tex_index)!!;

    gpu::compat::TextureViewDesc out_of_range = { .base_mip = 2, .mip_count = 1 };
    fault range_fault = @catch(gpu::compat::create_texture_descriptor(&device, texture, &out_of_range));
    assert(range_fault == gpu::compat::INVALID_ARGUMENT, "view past the mip chain faults INVALID_ARGUMENT");

    gpu::compat::SamplerDesc sampler_desc = {
        .min_filter = gpu::compat::Filter.NEAREST,
        .mag_filter = gpu::compat::Filter.NEAREST,
        .address_u  = gpu::compat::AddressMode.CLAMP_TO_EDGE,
        .address_v  = gpu::compat::AddressMode.CLAMP_TO_EDGE,
        .address_w  = gpu::compat::AddressMode.CLAMP_TO_EDGE,
    };
    gpu::compat::SamplerIndex smp_index = gpu::compat::create_sampler(&device, &sampler_desc)!!;
    defer gpu::compat::destroy_sampler(&device, smp_index)!!;

    gpu::compat::BufferDesc staging_desc = {
        .size = MIP0_BYTES + MIP1_BYTES, .usage = { .transfer_src }, .memory_kind = gpu::compat::MemoryKind.STAGING,
    };
    gpu::compat::BufferHandle staging = gpu::compat::create_buffer(&device, &staging_desc)!!;
    defer gpu::compat::destroy_buffer(&device, staging)!!;
    gpu::compat::GpuSpan staging_span = gpu::compat::get_buffer_span(&device, staging)!!;
    char* bytes = (char*)staging_span.cpu;
    for (uint i = 0; i < UP_SIZE * UP_SIZE; i++) {
        bytes[i * 4] = (char)10; bytes[i * 4 + 1] = (char)20; bytes[i * 4 + 2] = (char)30; bytes[i * 4 + 3] = (char)255;
    }
    for (uint i = 0; i < (UP_SIZE / 2) * (UP_SIZE / 2); i++) {
        char* texel = &bytes[MIP0_BYTES + (usz)i * 4];
        texel[0] = (char)200; texel[1] = (char)210; texel[2] = (char)220; texel[3] = (char)255;
    }
    gpu::compat::flush_buffer(&device, staging, 0, 0)!!;

    gpu::compat::BufferDesc out_desc = {
        .size        = (UP_SIZE / 2) * (UP_SIZE / 2) * 16,
        .usage       = { .storage, .addressable },
        .memory_kind = gpu::compat::MemoryKind.READBACK,
    };
    gpu::compat::BufferHandle sample_out = gpu::compat::create_buffer(&device, &out_desc)!!;
    defer gpu::compat::destroy_buffer(&device, sample_out)!!;

    gpu::compat::ShaderDesc sample_sd = {
        .stage = gpu::compat::ShaderStage.COMPUTE, .spirv = HEAP_SAMPLE_SPIRV[..], .entry_point = "main",
    };
    gpu::compat::ShaderHandle sample_shader = gpu::compat::create_shader(&device, &sample_sd)!!;
    defer gpu::compat::destroy_shader(&device, sample_shader)!!;
    gpu::compat::ComputePipelineDesc sample_pd = { .shader = sample_shader, .push_constant_size = gpu::compat::RootPush::size };
    gpu::compat::PipelineHandle sample_pipe = gpu::compat::create_compute_pipeline(&device, &sample_pd)!!;
    defer gpu::compat::destroy_pipeline(&device, sample_pipe)!!;

    gpu::compat::FrameToken frame_token = gpu::compat::begin_frame(&device)!!;
    gpu::compat::GpuSpan root_span = gpu::compat::alloc_frame_span(&frame_token, SampleRoot::size, 16)!!;
    SampleRoot* root = (SampleRoot*)root_span.cpu;
    root.output_gpu    = gpu::compat::get_buffer_address(&device, sample_out)!!;
    root.texture_index = tex_index;
    root.sampler_index = smp_index;
    root.width         = UP_SIZE / 2;
    root.height        = UP_SIZE / 2;

    gpu::compat::CommandList cmd = gpu::compat::begin_commands(&device, gpu::compat::QueueKind.COMPUTE)!!;
    gpu::compat::TextureBarrier to_dst = {
        .texture       = texture,
        .before_stage  = gpu::compat::Stage.HOST,
        .after_stage   = gpu::compat::Stage.TRANSFER,
        .before_hazard = gpu::compat::Hazard.HOST_WRITE,
        .after_hazard  = gpu::compat::Hazard.TRANSFER_WRITE,
        .old_layout    = gpu::compat::TextureLayout.UNDEFINED,
        .new_layout    = gpu::compat::TextureLayout.TRANSFER_DST,
    };
    gpu::compat::cmd_texture_barrier(&cmd, &to_dst)!!;
    gpu::compat::GpuSpan mip0_span = staging_span.checked_subspan(0, MIP0_BYTES)!!;
    gpu::compat::BufferTextureCopyDesc mip0_copy = { .src = mip0_span, .texture = texture, .mip = 0 };
    gpu::compat::cmd_copy_buffer_to_texture(&cmd, &mip0_copy)!!;
    gpu::compat::GpuSpan mip1_span = staging_span.checked_subspan(MIP0_BYTES, MIP1_BYTES)!!;
    gpu::compat::BufferTextureCopyDesc mip1_copy = { .src = mip1_span, .texture = texture, .mip = 1 };
    gpu::compat::cmd_copy_buffer_to_texture(&cmd, &mip1_copy)!!;
    gpu::compat::TextureBarrier to_read = {
        .texture       = texture,
        .before_stage  = gpu::compat::Stage.TRANSFER,
        .after_stage   = gpu::compat::Stage.COMPUTE_SHADER,
        .before_hazard = gpu::compat::Hazard.TRANSFER_WRITE,
        .after_hazard  = gpu::compat::Hazard.SHADER_READ,
        .old_layout    = gpu::compat::TextureLayout.TRANSFER_DST,
        .new_layout    = gpu::compat::TextureLayout.SHADER_READ,
    };
    gpu::compat::cmd_texture_barrier(&cmd, &to_read)!!;
    gpu::compat::cmd_dispatch(
        commands: &cmd,
        pipeline: sample_pipe,
        root:     root_span.gpu,
        groups:   { 1, 1, 1 },
    )!!;
    gpu::compat::BufferBarrier out_bar = {
        .buffer = sample_out, .offset = 0, .size = (UP_SIZE / 2) * (UP_SIZE / 2) * 16,
        .before_stage = gpu::compat::Stage.COMPUTE_SHADER, .after_stage = gpu::compat::Stage.HOST,
        .before_hazard = gpu::compat::Hazard.SHADER_WRITE, .after_hazard = gpu::compat::Hazard.HOST_READ,
    };
    gpu::compat::cmd_buffer_barrier(&cmd, &out_bar)!!;
    gpu::compat::end_commands(&cmd)!!;
    gpu::compat::CommandList[1] lists = { cmd };
    gpu::compat::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::compat::submit(&device, &submit)!!;
    gpu::compat::wait_queue_idle(&device, gpu::compat::QueueKind.COMPUTE)!!;

    gpu::compat::GpuSpan out_span = gpu::compat::get_buffer_span(&device, sample_out)!!;
    float* sampled = (float*)out_span.cpu;
    assert(quantize_unorm8(sampled[0]) == 200, "mip-subset view samples mip 1, not mip 0");
    assert(quantize_unorm8(sampled[1]) == 210, "mip-subset view green matches mip 1");
    gpu::compat::end_frame(&frame_token)!!;
}

fn void flush_invalidate_stale_handle_faults() @test {
    gpu::compat::Device device = make_upload_test_device();
    defer gpu::compat::destroy_device(&device)!!;

    gpu::compat::BufferDesc desc = {
        .size        = 64,
        .usage       = { .transfer_src },
        .memory_kind = gpu::compat::MemoryKind.STAGING,
    };
    gpu::compat::BufferHandle buffer = gpu::compat::create_buffer(&device, &desc)!!;
    gpu::compat::flush_buffer(&device, buffer, 0, 0)!!;
    gpu::compat::invalidate_buffer(&device, buffer, 0, 64)!!;
    gpu::compat::destroy_buffer(&device, buffer)!!;

    fault stale_fault = @catch(gpu::compat::flush_buffer(&device, buffer, 0, 0));
    assert(stale_fault == gpu::compat::INVALID_HANDLE, "flush on a destroyed buffer faults INVALID_HANDLE");
}
