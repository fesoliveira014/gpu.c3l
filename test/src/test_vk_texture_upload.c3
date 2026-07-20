module gpu_test;

import gpu;

const uint UP_SIZE = 8;
const usz MIP0_BYTES = UP_SIZE * UP_SIZE * 4;
const usz MIP1_BYTES = (UP_SIZE / 2) * (UP_SIZE / 2) * 4;

fn gpu::Device make_upload_test_device() {
    gpu::RuntimeDesc desc = {
        .backend           = gpu::BackendKind.VULKAN,
        .enable_validation = true,
        .application_name  = "gpu_texture_upload_test",
    };
    return create_test_device(&desc)!!;
}

fn void sampled_texture_upload_and_mip_readback() @test {
    gpu::Device device = make_upload_test_device();
    defer destroy_test_device(&device)!!;

    gpu::TextureDesc tex_desc = {
        .dimension  = gpu::TextureDimension.TEX_2D,
        .access    = { .graphics, .compute, .transfer },
        .width      = UP_SIZE,
        .height     = UP_SIZE,
        .mip_levels = 2,
        .format     = gpu::Format.RGBA8_UNORM,
        .usage      = { .sampled, .transfer_src, .transfer_dst },
        .debug_name = "upload_target",
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
    OwnedSpan staging = allocate_owned_span(
        device:       &device,
        size:         MIP0_BYTES + MIP1_BYTES,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "upload_staging",
    )!!;
    defer free_owned_span(&device, &staging)!!;
    gpu::GpuSpan staging_span = staging.span;
    char* bytes = (char*)test_span_mapping(&device, staging_span).ptr;
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
    gpu::flush_mapped_span(&device, staging.span)!!;

    OwnedSpan sample_out = allocate_owned_span(
        device:       &device,
        size:         UP_SIZE * UP_SIZE * 16,
        memory_class: gpu::MemoryClass.CPU_READ,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "sample_output",
    )!!;
    defer free_owned_span(&device, &sample_out)!!;
    OwnedSpan mip1_readback = allocate_owned_span(
        device:       &device,
        size:         MIP1_BYTES,
        memory_class: gpu::MemoryClass.CPU_READ,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "mip1_readback",
    )!!;
    defer free_owned_span(&device, &mip1_readback)!!;

    gpu::ShaderDesc sample_sd = {
        .stage = gpu::ShaderStage.COMPUTE, .spirv = HEAP_SAMPLE_SPIRV[..], .entry_point = "main",
    };
    gpu::ShaderCode sample_shader = gpu::prepare_shader_code(&sample_sd)!!;
    gpu::ComputePipelineDesc sample_pd = { .shader = sample_shader, .push_constant_size = gpu::RootPush::size };
    gpu::PipelineHandle sample_pipe = gpu::create_compute_pipeline(&device, &sample_pd)!!;
    defer gpu::destroy_pipeline(&device, sample_pipe)!!;
    OwnedSpan sample_root_owned = allocate_owned_span(
        device:       &device,
        size:         SampleRoot::size,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       { .compute },
        debug_name:   "upload_sample_root",
    )!!;
    defer free_owned_span(&device, &sample_root_owned)!!;
    gpu::GpuSpan sample_root_span = sample_root_owned.span;
    SampleRoot* sample_root = (SampleRoot*)test_span_mapping(&device, sample_root_span).ptr;
    sample_root.output_gpu    = test_span_address(&device, sample_out.span);
    sample_root.texture_index = tex_index;
    sample_root.sampler_index = smp_index;
    sample_root.width         = UP_SIZE;
    sample_root.height        = UP_SIZE;
    gpu::flush_mapped_span(&device, sample_root_span)!!;

    gpu::CommandList cmd = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.COMPUTE)!!)!!;
    defer discard_test_commands(&cmd)!!;

    gpu::TextureBarrier to_dst = gpu::texture_transition(
        texture: texture,
        before:  gpu::TextureUse.UNDEFINED,
        after:   gpu::TextureUse.TRANSFER_DESTINATION,
    )!!;
    gpu::cmd_texture_barrier(&cmd, &to_dst)!!;

    gpu::BufferTextureCopyDesc mip0_copy = {
        .src     = staging_span.checked_subspan(0, MIP0_BYTES)!!,
        .texture = texture,
        .mip     = 0,
    };
    gpu::cmd_copy_buffer_to_texture(&cmd, &mip0_copy)!!;
    gpu::BufferTextureCopyDesc mip1_copy = {
        .src     = staging_span.checked_subspan(MIP0_BYTES, MIP1_BYTES)!!,
        .texture = texture,
        .mip     = 1,
    };
    gpu::cmd_copy_buffer_to_texture(&cmd, &mip1_copy)!!;

    gpu::TextureBarrier to_read = gpu::texture_transition(
        texture: texture,
        before:  gpu::TextureUse.TRANSFER_DESTINATION,
        after:   gpu::TextureUse.SAMPLED_COMPUTE,
    )!!;
    gpu::cmd_texture_barrier(&cmd, &to_read)!!;

    const uint GROUPS = (UP_SIZE + 7) / 8;
    bind_test_compute_pipeline(&cmd, sample_pipe);
    gpu::cmd_dispatch(
        commands: &cmd,
        root:     test_span_address(&device, sample_root_span),
        groups:   { GROUPS, GROUPS, 1 },
    )!!;

    gpu::Barrier sample_host_read = {
        .before = { .compute = true },
        .after = { .host = true },
    };
    gpu::cmd_barrier(&cmd, &sample_host_read)!!;

    gpu::TextureBarrier to_src = {
        .texture = texture,
        .before  = gpu::TextureUse.SAMPLED_COMPUTE,
        .after   = gpu::TextureUse.TRANSFER_SOURCE,
    };
    gpu::cmd_texture_barrier(&cmd, &to_src)!!;

    gpu::GpuSpan rb_span = mip1_readback.span;
    gpu::TextureBufferCopyDesc mip1_out = {
        .texture = texture,
        .dst     = rb_span,
        .mip     = 1,
    };
    gpu::cmd_copy_texture_to_buffer(&cmd, &mip1_out)!!;

    gpu::Barrier host_read = {
        .before = { .transfer = true },
        .after = { .host = true },
    };
    gpu::cmd_barrier(&cmd, &host_read)!!;

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

    gpu::invalidate_mapped_span(&device, sample_out.span)!!;
    gpu::GpuSpan out_span = sample_out.span;
    float* texels = (float*)test_span_mapping(&device, out_span).ptr;
    for (uint y = 0; y < UP_SIZE; y++) {
        for (uint x = 0; x < UP_SIZE; x++) {
            float* texel = &texels[(y * UP_SIZE + x) * 4];
            assert(quantize_unorm8(texel[0]) == x, "sampled mip 0 red matches upload");
            assert(quantize_unorm8(texel[1]) == y, "sampled mip 0 green matches upload");
            assert(quantize_unorm8(texel[2]) == (x ^ y), "sampled mip 0 blue matches upload");
        }
    }

    gpu::invalidate_mapped_span(&device, mip1_readback.span)!!;
    char* mip1_bytes = (char*)test_span_mapping(&device, rb_span).ptr;
    for (uint y = 0; y < UP_SIZE / 2; y++) {
        for (uint x = 0; x < UP_SIZE / 2; x++) {
            char* texel = &mip1_bytes[(y * (UP_SIZE / 2) + x) * 4];
            assert(texel[0] == (char)(x + 100), "read-back mip 1 red matches upload");
            assert(texel[1] == (char)(y + 100), "read-back mip 1 green matches upload");
            assert(texel[2] == (char)7, "read-back mip 1 blue matches upload");
        }
    }
}

fn void storage_write_direct_readback() @test {
    gpu::Device device = make_upload_test_device();
    defer destroy_test_device(&device)!!;

    gpu::TextureDesc tex_desc = {
        .dimension = gpu::TextureDimension.TEX_2D,
        .access    = { .graphics, .compute, .transfer },
        .width     = UP_SIZE,
        .height    = UP_SIZE,
        .format    = gpu::Format.RGBA8_UNORM,
        .usage     = { .storage, .transfer_src },
    };
    gpu::TextureHandle texture = gpu::create_texture(&device, &tex_desc)!!;
    defer gpu::destroy_texture(&device, texture)!!;
    gpu::TextureView tex_index_view = gpu::create_texture_view(&device, texture, null)!!;
    gpu::TextureIndex tex_index = tex_index_view.index;
    defer gpu::destroy_texture_view(&device, tex_index_view)!!;

    OwnedSpan readback = allocate_owned_span(
        device:       &device,
        size:         MIP0_BYTES,
        memory_class: gpu::MemoryClass.CPU_READ,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "storage_readback",
    )!!;
    defer free_owned_span(&device, &readback)!!;

    gpu::ShaderDesc write_sd = {
        .stage = gpu::ShaderStage.COMPUTE, .spirv = HEAP_WRITE_SPIRV[..], .entry_point = "main",
    };
    gpu::ShaderCode write_shader = gpu::prepare_shader_code(&write_sd)!!;
    gpu::ComputePipelineDesc write_pd = { .shader = write_shader, .push_constant_size = gpu::RootPush::size };
    gpu::PipelineHandle write_pipe = gpu::create_compute_pipeline(&device, &write_pd)!!;
    defer gpu::destroy_pipeline(&device, write_pipe)!!;
    OwnedSpan root_owned = allocate_owned_span(
        device:       &device,
        size:         WriteRoot::size,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       { .compute },
        debug_name:   "storage_write_root",
    )!!;
    defer free_owned_span(&device, &root_owned)!!;
    gpu::GpuSpan root_span = root_owned.span;
    WriteRoot* root = (WriteRoot*)test_span_mapping(&device, root_span).ptr;
    root.texture_index = tex_index;
    root.width         = UP_SIZE;
    root.height        = UP_SIZE;
    gpu::flush_mapped_span(&device, root_span)!!;

    gpu::CommandList cmd = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.COMPUTE)!!)!!;
    defer discard_test_commands(&cmd)!!;
    gpu::TextureBarrier to_general = {
        .texture = texture,
        .before  = gpu::TextureUse.UNDEFINED,
        .after   = gpu::TextureUse.STORAGE_COMPUTE,
    };
    gpu::cmd_texture_barrier(&cmd, &to_general)!!;

    const uint GROUPS = (UP_SIZE + 7) / 8;
    bind_test_compute_pipeline(&cmd, write_pipe);
    gpu::cmd_dispatch(
        commands: &cmd,
        root:     test_span_address(&device, root_span),
        groups:   { GROUPS, GROUPS, 1 },
    )!!;

    gpu::TextureBarrier to_src = {
        .texture = texture,
        .before  = gpu::TextureUse.STORAGE_COMPUTE,
        .after   = gpu::TextureUse.TRANSFER_SOURCE,
    };
    gpu::cmd_texture_barrier(&cmd, &to_src)!!;

    gpu::GpuSpan rb_span = readback.span;
    gpu::TextureBufferCopyDesc copy = { .texture = texture, .dst = rb_span };
    gpu::cmd_copy_texture_to_buffer(&cmd, &copy)!!;

    gpu::Barrier host_read = {
        .before = { .transfer = true },
        .after = { .host = true },
    };
    gpu::cmd_barrier(&cmd, &host_read)!!;
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

    gpu::invalidate_mapped_span(&device, readback.span)!!;
    char* bytes = (char*)test_span_mapping(&device, rb_span).ptr;
    for (uint y = 0; y < UP_SIZE; y++) {
        for (uint x = 0; x < UP_SIZE; x++) {
            char* texel = &bytes[(y * UP_SIZE + x) * 4];
            assert(texel[0] == (char)x, "storage-written red reads back");
            assert(texel[1] == (char)y, "storage-written green reads back");
            assert(texel[2] == (char)(x ^ y), "storage-written blue reads back");
            assert(texel[3] == (char)255, "storage-written alpha reads back");
        }
    }
}

fn void copy_validation_faults() @test {
    gpu::Device device = make_upload_test_device();
    defer destroy_test_device(&device)!!;

    gpu::TextureDesc tex_desc = {
        .dimension = gpu::TextureDimension.TEX_2D,
        .access    = { .graphics, .compute, .transfer },
        .width     = UP_SIZE,
        .height    = UP_SIZE,
        .format    = gpu::Format.RGBA8_UNORM,
        .usage     = { .sampled, .transfer_dst },
    };
    gpu::TextureHandle texture = gpu::create_texture(&device, &tex_desc)!!;
    defer gpu::destroy_texture(&device, texture)!!;

    OwnedSpan staging = allocate_owned_span(
        device:       &device,
        size:         MIP0_BYTES,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "copy_validation_staging",
    )!!;
    defer free_owned_span(&device, &staging)!!;
    gpu::GpuSpan span = staging.span;
    gpu::CommandList cmd = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.COMPUTE)!!)!!;
    defer discard_test_commands(&cmd)!!;

    gpu::TextureBarrier to_dst = {
        .texture = texture,
        .before  = gpu::TextureUse.UNDEFINED,
        .after   = gpu::TextureUse.TRANSFER_DESTINATION,
    };
    gpu::cmd_texture_barrier(&cmd, &to_dst)!!;

    gpu::BufferTextureCopyDesc past_extent = { .src = span, .texture = texture, .width = UP_SIZE + 1 };
    fault extent_fault = @catch(gpu::cmd_copy_buffer_to_texture(&cmd, &past_extent));
    assert(extent_fault == gpu::INVALID_ARGUMENT, "region past the mip extent faults INVALID_ARGUMENT");

    gpu::BufferTextureCopyDesc past_span = { .src = span.checked_subspan(0, MIP0_BYTES - 4)!!, .texture = texture };
    fault span_fault = @catch(gpu::cmd_copy_buffer_to_texture(&cmd, &past_span));
    assert(span_fault == gpu::INVALID_ARGUMENT, "copy bytes past the span fault INVALID_ARGUMENT");

    gpu::BufferTextureCopyDesc misaligned = { .src = span.checked_subspan(2, MIP0_BYTES - 2)!!, .texture = texture };
    fault align_fault = @catch(gpu::cmd_copy_buffer_to_texture(&cmd, &misaligned));
    assert(align_fault == gpu::INVALID_ARGUMENT, "misaligned span offset faults INVALID_ARGUMENT");

    gpu::BufferTextureCopyDesc bad_mip = { .src = span, .texture = texture, .mip = 1 };
    fault mip_fault = @catch(gpu::cmd_copy_buffer_to_texture(&cmd, &bad_mip));
    assert(mip_fault == gpu::INVALID_ARGUMENT, "copy to a missing mip faults INVALID_ARGUMENT");

    gpu::TextureBufferCopyDesc no_src_usage = { .texture = texture, .dst = span };
    fault usage_fault = @catch(gpu::cmd_copy_texture_to_buffer(&cmd, &no_src_usage));
    assert(usage_fault == gpu::INVALID_ARGUMENT, "texture without transfer_src faults INVALID_ARGUMENT on readback");

    gpu::ExecutableCommandList cmd_executable = gpu::end_commands(&cmd)!!;

    defer discard_test_executable(&cmd_executable)!!;
}

fn void mip_subset_view_sampled() @test {
    gpu::Device device = make_upload_test_device();
    defer destroy_test_device(&device)!!;

    gpu::TextureDesc tex_desc = {
        .dimension  = gpu::TextureDimension.TEX_2D,
        .access    = { .graphics, .compute, .transfer },
        .width      = UP_SIZE,
        .height     = UP_SIZE,
        .mip_levels = 2,
        .format     = gpu::Format.RGBA8_UNORM,
        .usage      = { .sampled, .transfer_dst },
    };
    gpu::TextureHandle texture = gpu::create_texture(&device, &tex_desc)!!;
    defer gpu::destroy_texture(&device, texture)!!;

    // Register a mip-1-only view: sampling it at lod 0 must return mip 1.
    gpu::TextureViewDesc mip1_view = { .base_mip = 1, .mip_count = 1 };
    gpu::TextureView tex_index_view = gpu::create_texture_view(&device, texture, &mip1_view)!!;
    gpu::TextureIndex tex_index = tex_index_view.index;
    defer gpu::destroy_texture_view(&device, tex_index_view)!!;

    gpu::TextureViewDesc out_of_range = { .base_mip = 2, .mip_count = 1 };
    fault range_fault = @catch(gpu::create_texture_view(&device, texture, &out_of_range));
    assert(range_fault == gpu::INVALID_ARGUMENT, "view past the mip chain faults INVALID_ARGUMENT");

    gpu::SamplerDesc sampler_desc = {
        .min_filter = gpu::Filter.NEAREST,
        .mag_filter = gpu::Filter.NEAREST,
        .address_u  = gpu::AddressMode.CLAMP_TO_EDGE,
        .address_v  = gpu::AddressMode.CLAMP_TO_EDGE,
        .address_w  = gpu::AddressMode.CLAMP_TO_EDGE,
    };
    gpu::Sampler sampler = gpu::intern_sampler(&device, &sampler_desc)!!;
    gpu::SamplerIndex smp_index = gpu::publish_sampler(&device, sampler)!!;
    OwnedSpan staging = allocate_owned_span(
        device:       &device,
        size:         MIP0_BYTES + MIP1_BYTES,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "mip_subset_staging",
    )!!;
    defer free_owned_span(&device, &staging)!!;
    gpu::GpuSpan staging_span = staging.span;
    char* bytes = (char*)test_span_mapping(&device, staging_span).ptr;
    for (uint i = 0; i < UP_SIZE * UP_SIZE; i++) {
        bytes[i * 4] = (char)10; bytes[i * 4 + 1] = (char)20; bytes[i * 4 + 2] = (char)30; bytes[i * 4 + 3] = (char)255;
    }
    for (uint i = 0; i < (UP_SIZE / 2) * (UP_SIZE / 2); i++) {
        char* texel = &bytes[MIP0_BYTES + (usz)i * 4];
        texel[0] = (char)200; texel[1] = (char)210; texel[2] = (char)220; texel[3] = (char)255;
    }
    gpu::flush_mapped_span(&device, staging.span)!!;

    OwnedSpan sample_out = allocate_owned_span(
        device:       &device,
        size:         (UP_SIZE / 2) * (UP_SIZE / 2) * 16,
        memory_class: gpu::MemoryClass.CPU_READ,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "mip_subset_output",
    )!!;
    defer free_owned_span(&device, &sample_out)!!;

    gpu::ShaderDesc sample_sd = {
        .stage = gpu::ShaderStage.COMPUTE, .spirv = HEAP_SAMPLE_SPIRV[..], .entry_point = "main",
    };
    gpu::ShaderCode sample_shader = gpu::prepare_shader_code(&sample_sd)!!;
    gpu::ComputePipelineDesc sample_pd = { .shader = sample_shader, .push_constant_size = gpu::RootPush::size };
    gpu::PipelineHandle sample_pipe = gpu::create_compute_pipeline(&device, &sample_pd)!!;
    defer gpu::destroy_pipeline(&device, sample_pipe)!!;
    OwnedSpan root_owned = allocate_owned_span(
        device:       &device,
        size:         SampleRoot::size,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       { .compute },
        debug_name:   "mip_sample_root",
    )!!;
    defer free_owned_span(&device, &root_owned)!!;
    gpu::GpuSpan root_span = root_owned.span;
    SampleRoot* root = (SampleRoot*)test_span_mapping(&device, root_span).ptr;
    root.output_gpu    = test_span_address(&device, sample_out.span);
    root.texture_index = tex_index;
    root.sampler_index = smp_index;
    root.width         = UP_SIZE / 2;
    root.height        = UP_SIZE / 2;
    gpu::flush_mapped_span(&device, root_span)!!;

    gpu::CommandList cmd = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.COMPUTE)!!)!!;
    defer discard_test_commands(&cmd)!!;
    gpu::TextureBarrier to_dst = {
        .texture = texture,
        .before  = gpu::TextureUse.UNDEFINED,
        .after   = gpu::TextureUse.TRANSFER_DESTINATION,
    };
    gpu::cmd_texture_barrier(&cmd, &to_dst)!!;
    gpu::GpuSpan mip0_span = staging_span.checked_subspan(0, MIP0_BYTES)!!;
    gpu::BufferTextureCopyDesc mip0_copy = { .src = mip0_span, .texture = texture, .mip = 0 };
    gpu::cmd_copy_buffer_to_texture(&cmd, &mip0_copy)!!;
    gpu::GpuSpan mip1_span = staging_span.checked_subspan(MIP0_BYTES, MIP1_BYTES)!!;
    gpu::BufferTextureCopyDesc mip1_copy = { .src = mip1_span, .texture = texture, .mip = 1 };
    gpu::cmd_copy_buffer_to_texture(&cmd, &mip1_copy)!!;
    gpu::TextureBarrier to_read = {
        .texture = texture,
        .before  = gpu::TextureUse.TRANSFER_DESTINATION,
        .after   = gpu::TextureUse.SAMPLED_COMPUTE,
    };
    gpu::cmd_texture_barrier(&cmd, &to_read)!!;
    bind_test_compute_pipeline(&cmd, sample_pipe);
    gpu::cmd_dispatch(
        commands: &cmd,
        root:     test_span_address(&device, root_span),
        groups:   { 1, 1, 1 },
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

    gpu::invalidate_mapped_span(&device, sample_out.span)!!;
    gpu::GpuSpan out_span = sample_out.span;
    float* sampled = (float*)test_span_mapping(&device, out_span).ptr;
    assert(quantize_unorm8(sampled[0]) == 200, "mip-subset view samples mip 1, not mip 0");
    assert(quantize_unorm8(sampled[1]) == 210, "mip-subset view green matches mip 1");
}

fn void mapped_span_stale_owner_faults() @test {
    gpu::Device device = make_upload_test_device();
    defer destroy_test_device(&device)!!;

    OwnedSpan owned = allocate_owned_span(
        device:       &device,
        size:         64,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "stale_mapped_span",
    )!!;
    gpu::GpuSpan stale = owned.span;
    gpu::flush_mapped_span(&device, stale)!!;
    gpu::invalidate_mapped_span(&device, stale)!!;
    free_owned_span(&device, &owned)!!;

    fault stale_fault = @catch(gpu::flush_mapped_span(&device, stale));
    assert(stale_fault == gpu::INVALID_HANDLE, "flush on a freed allocation faults INVALID_HANDLE");
}
