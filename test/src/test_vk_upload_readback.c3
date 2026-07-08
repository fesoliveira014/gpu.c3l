module gpu_test;

import gpu;
import std::core::mem;

// Headless (lavapipe): upload/readback helpers over the staging and readback
// arenas. The compute shader from the heap tests consumes helper-uploaded data.

const char[*] UPLOAD_HEAP_SAMPLE_SPIRV = $embed("shaders/heap_sample.comp.spv");

const uint UR_TEX = 8;
const usz UR_TEX_BYTES = UR_TEX * UR_TEX * 4;

fn gpu::Device make_transfer_test_device(usz staging_size = 0) {
    gpu::DeviceDesc desc = {
        .backend            = gpu::BackendKind.VULKAN,
        .enable_validation  = true,
        .frames_in_flight   = 2,
        .staging_arena_size = staging_size,
        .application_name   = "gpu_upload_readback_test",
    };
    return gpu::create_device(&desc)!!;
}

fn void buffer_upload_readback_round_trip() @test {
    gpu::Device device = make_transfer_test_device();
    defer gpu::destroy_device(&device)!!;

    gpu::BufferDesc desc = {
        .size        = 4096,
        .usage       = { .storage, .transfer_src, .transfer_dst },
        .memory_kind = gpu::MemoryKind.DEVICE,
        .debug_name  = "device_buffer",
    };
    gpu::BufferHandle buffer = gpu::create_buffer(&device, &desc)!!;
    defer gpu::destroy_buffer(&device, buffer)!!;

    char[4096] data;
    for (uint i = 0; i < 4096; i++) data[i] = (char)(i * 7);
    gpu::upload_buffer_data(
        device:      &device,
        dst:         buffer,
        offset:      0,
        data:        data[..],
        next_stage:  gpu::Stage.TRANSFER,
        next_hazard: gpu::Hazard.TRANSFER_READ,
    )!!;

    char[4096] out;
    gpu::readback_buffer_data(
        device:      &device,
        src:         buffer,
        offset:      0,
        out_data:    out[..],
        from_stage:  gpu::Stage.TRANSFER,
        from_hazard: gpu::Hazard.TRANSFER_READ,
    )!!;
    for (uint i = 0; i < 4096; i++) {
        assert(out[i] == (char)(i * 7), "device-memory round trip preserves bytes");
    }
}

<* An unrelated blocking helper's completion must never retire a ticket
   whose command list was never submitted (gpu.c3l#80): before the helper
   timeline was split from the frame timeline, a blocking helper signaled
   the exact value an unsubmitted ticket recorded moments earlier had been
   tagged with, so poll/resolve reported it ready with stale bytes. *>
fn void cmd_readback_ticket_not_ready_before_submit() @test {
    gpu::Device device = make_transfer_test_device();
    defer gpu::destroy_device(&device)!!;

    // Settle the timeline on a known baseline before the deterministic part.
    gpu::begin_frame(&device)!!;
    gpu::end_frame(&device)!!;

    gpu::BufferDesc src_desc = {
        .size        = 256,
        .usage       = { .storage, .transfer_src, .transfer_dst },
        .memory_kind = gpu::MemoryKind.DEVICE,
        .debug_name  = "ticket_src",
    };
    gpu::BufferHandle src = gpu::create_buffer(&device, &src_desc)!!;
    defer gpu::destroy_buffer(&device, src)!!;

    gpu::CommandList cmd = gpu::begin_commands(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::ReadbackTicket ticket = gpu::cmd_readback_buffer(&cmd, src, 0, 256)!!;
    gpu::end_commands(&device, &cmd)!!;
    // Deliberately never gpu::submit(&device, ...): the ticket's copy never runs.

    gpu::BufferDesc unrelated_desc = {
        .size        = 64,
        .usage       = { .transfer_src, .transfer_dst },
        .memory_kind = gpu::MemoryKind.DEVICE,
        .debug_name  = "unrelated",
    };
    gpu::BufferHandle unrelated = gpu::create_buffer(&device, &unrelated_desc)!!;
    defer gpu::destroy_buffer(&device, unrelated)!!;

    char[64] payload;
    foreach (&b : payload) *b = (char)0xAB;
    gpu::upload_buffer_data(
        device:      &device,
        dst:         unrelated,
        offset:      0,
        data:        payload[..],
        next_stage:  gpu::Stage.TRANSFER,
        next_hazard: gpu::Hazard.TRANSFER_READ,
    )!!;

    assert(!gpu::poll_readback(&device, &ticket), "an unrelated blocking helper must not retire an unsubmitted ticket");

    char[256] dest;
    fault not_ready = @catch(gpu::resolve_readback(&device, &ticket, dest[..]));
    assert(not_ready == gpu::READBACK_NOT_READY, "resolve on an unsubmitted ticket faults READBACK_NOT_READY");
}

fn void oversized_upload_uses_dedicated_staging() @test {
    // Tiny arena: every upload larger than 1 KiB falls back to a dedicated buffer.
    gpu::Device device = make_transfer_test_device(staging_size: 1024);
    defer gpu::destroy_device(&device)!!;

    const usz BIG = 256 * 1024;
    gpu::BufferDesc desc = {
        .size        = BIG,
        .usage       = { .transfer_src, .transfer_dst },
        .memory_kind = gpu::MemoryKind.DEVICE,
    };
    gpu::BufferHandle buffer = gpu::create_buffer(&device, &desc)!!;
    defer gpu::destroy_buffer(&device, buffer)!!;

    @pool() {
        char[] data = mem::talloc_array(char, (sz)BIG);
        for (usz i = 0; i < BIG; i++) data[i] = (char)(i % 251);
        gpu::upload_buffer_data(
            device:      &device,
            dst:         buffer,
            offset:      0,
            data:        data,
            next_stage:  gpu::Stage.TRANSFER,
            next_hazard: gpu::Hazard.TRANSFER_READ,
        )!!;

        char[] out = mem::talloc_array(char, (sz)BIG);
        gpu::readback_buffer_data(
            device:      &device,
            src:         buffer,
            offset:      0,
            out_data:    out,
            from_stage:  gpu::Stage.TRANSFER,
            from_hazard: gpu::Hazard.TRANSFER_READ,
        )!!;
        for (usz i = 0; i < BIG; i += 4093) {
            assert(out[i] == (char)(i % 251), "larger-than-arena upload round-trips");
        }
    };
}

fn void staging_ring_reuses_across_frames() @test {
    // Arena fits ~4 allocations; 12 frames of uploads force wrap + recycle.
    gpu::Device device = make_transfer_test_device(staging_size: 4096);
    defer gpu::destroy_device(&device)!!;

    gpu::BufferDesc desc = {
        .size        = 1024,
        .usage       = { .storage, .transfer_dst },
        .memory_kind = gpu::MemoryKind.DEVICE,
    };
    gpu::BufferHandle buffer = gpu::create_buffer(&device, &desc)!!;
    defer gpu::destroy_buffer(&device, buffer)!!;

    char[1024] data;
    for (uint frame = 0; frame < 12; frame++) {
        for (uint i = 0; i < 1024; i++) data[i] = (char)(frame + i);
        gpu::begin_frame(&device)!!;
        gpu::CommandList cmd = gpu::begin_commands(&device, gpu::QueueKind.GRAPHICS)!!;
        gpu::cmd_upload_buffer(&cmd, buffer, 0, data[..])!!;
        gpu::BufferBarrier next_use = {
            .buffer = buffer, .offset = 0, .size = 1024,
            .before_stage = gpu::Stage.TRANSFER, .after_stage = gpu::Stage.COMPUTE_SHADER,
            .before_hazard = gpu::Hazard.TRANSFER_WRITE, .after_hazard = gpu::Hazard.SHADER_READ,
        };
        gpu::cmd_buffer_barrier(&cmd, &next_use)!!;
        gpu::end_commands(&device, &cmd)!!;
        gpu::CommandList[1] lists = { cmd };
        gpu::SubmitDesc submit = { .command_lists = lists[..] };
        gpu::submit(&device, &submit)!!;
        gpu::end_frame(&device)!!;
    }
    gpu::wait_queue_idle(&device, gpu::QueueKind.GRAPHICS)!!;
}

fn void out_of_frame_uploads_do_not_alias_staging() @test {
    // Two cmd-level uploads recorded outside any frame into one list: their
    // staging ranges must not alias (ranges hold until a timeline advance).
    gpu::Device device = make_transfer_test_device(staging_size: 4096);
    defer gpu::destroy_device(&device)!!;

    gpu::BufferDesc desc = {
        .size        = 1024,
        .usage       = { .transfer_src, .transfer_dst },
        .memory_kind = gpu::MemoryKind.DEVICE,
    };
    gpu::BufferHandle first = gpu::create_buffer(&device, &desc)!!;
    defer gpu::destroy_buffer(&device, first)!!;
    gpu::BufferHandle second = gpu::create_buffer(&device, &desc)!!;
    defer gpu::destroy_buffer(&device, second)!!;

    char[1024] a;
    char[1024] b;
    for (uint i = 0; i < 1024; i++) {
        a[i] = (char)11;
        b[i] = (char)77;
    }

    gpu::CommandList cmd = gpu::begin_commands(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::cmd_upload_buffer(&cmd, first, 0, a[..])!!;
    gpu::cmd_upload_buffer(&cmd, second, 0, b[..])!!;
    gpu::BufferBarrier next_use = {
        .buffer = first, .offset = 0, .size = 1024,
        .before_stage = gpu::Stage.TRANSFER, .after_stage = gpu::Stage.TRANSFER,
        .before_hazard = gpu::Hazard.TRANSFER_WRITE, .after_hazard = gpu::Hazard.TRANSFER_READ,
    };
    gpu::cmd_buffer_barrier(&cmd, &next_use)!!;
    gpu::end_commands(&device, &cmd)!!;
    gpu::CommandList[1] lists = { cmd };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::submit(&device, &submit)!!;
    gpu::wait_queue_idle(&device, gpu::QueueKind.GRAPHICS)!!;

    char[1024] out;
    gpu::readback_buffer_data(
        device:      &device,
        src:         first,
        offset:      0,
        out_data:    out[..],
        from_stage:  gpu::Stage.TRANSFER,
        from_hazard: gpu::Hazard.TRANSFER_READ,
    )!!;
    assert(out[0] == (char)11 && out[1023] == (char)11, "first upload's staging was not overwritten by the second");
}

fn void texture_upload_helper_sampled() @test {
    gpu::Device device = make_transfer_test_device();
    defer gpu::destroy_device(&device)!!;

    gpu::TextureDesc tex_desc = {
        .dimension = gpu::TextureDimension.TEX_2D,
        .width     = UR_TEX,
        .height    = UR_TEX,
        .format    = gpu::Format.RGBA8_UNORM,
        .usage     = { .sampled, .transfer_dst },
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

    char[UR_TEX_BYTES] texels;
    for (uint i = 0; i < UR_TEX * UR_TEX; i++) {
        texels[i * 4]     = (char)(i % 256);
        texels[i * 4 + 1] = (char)90;
        texels[i * 4 + 2] = (char)160;
        texels[i * 4 + 3] = (char)255;
    }
    gpu::upload_texture_data(
        device:      &device,
        dst:         texture,
        mip:         0,
        data:        texels[..],
        next_stage:  gpu::Stage.COMPUTE_SHADER,
        next_hazard: gpu::Hazard.SHADER_READ,
        next_layout: gpu::TextureLayout.SHADER_READ,
    )!!;

    // Re-uploading now that the texture is in SHADER_READ must fault.
    fault in_use = @catch(gpu::upload_texture_data(
        device:      &device,
        dst:         texture,
        mip:         0,
        data:        texels[..],
        next_stage:  gpu::Stage.COMPUTE_SHADER,
        next_hazard: gpu::Hazard.SHADER_READ,
        next_layout: gpu::TextureLayout.SHADER_READ,
    ));
    assert(in_use == gpu::INVALID_RESOURCE_STATE, "in-use texture upload faults INVALID_RESOURCE_STATE");

    // Sample it through the heap: the shader must see the uploaded texels.
    gpu::BufferDesc out_desc = {
        .size        = UR_TEX * UR_TEX * 16,
        .usage       = { .storage, .addressable },
        .memory_kind = gpu::MemoryKind.READBACK,
    };
    gpu::BufferHandle sample_out = gpu::create_buffer(&device, &out_desc)!!;
    defer gpu::destroy_buffer(&device, sample_out)!!;

    gpu::ShaderDesc sample_sd = {
        .stage = gpu::ShaderStage.COMPUTE, .spirv = UPLOAD_HEAP_SAMPLE_SPIRV[..], .entry_point = "main",
    };
    gpu::ShaderHandle shader = gpu::create_shader(&device, &sample_sd)!!;
    defer gpu::destroy_shader(&device, shader)!!;
    gpu::ComputePipelineDesc pd = { .shader = shader, .push_constant_size = gpu::RootPush::size };
    gpu::PipelineHandle pipeline = gpu::create_compute_pipeline(&device, &pd)!!;
    defer gpu::destroy_pipeline(&device, pipeline)!!;

    gpu::begin_frame(&device)!!;
    gpu::GpuSpan root_span = gpu::alloc_frame_span(&device, SampleRoot::size, 16)!!;
    SampleRoot* root = (SampleRoot*)root_span.cpu;
    root.output_gpu    = gpu::get_buffer_address(&device, sample_out)!!;
    root.texture_index = tex_index;
    root.sampler_index = smp_index;
    root.width         = UR_TEX;
    root.height        = UR_TEX;

    gpu::CommandList cmd = gpu::begin_commands(&device, gpu::QueueKind.COMPUTE)!!;
    gpu::cmd_dispatch(
        commands: &cmd,
        pipeline: pipeline,
        root:     root_span.gpu,
        groups:   { 1, 1, 1 },
    )!!;
    gpu::BufferBarrier out_bar = {
        .buffer = sample_out, .offset = 0, .size = UR_TEX * UR_TEX * 16,
        .before_stage = gpu::Stage.COMPUTE_SHADER, .after_stage = gpu::Stage.HOST,
        .before_hazard = gpu::Hazard.SHADER_WRITE, .after_hazard = gpu::Hazard.HOST_READ,
    };
    gpu::cmd_buffer_barrier(&cmd, &out_bar)!!;
    gpu::end_commands(&device, &cmd)!!;
    gpu::CommandList[1] lists = { cmd };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::submit(&device, &submit)!!;
    gpu::wait_queue_idle(&device, gpu::QueueKind.COMPUTE)!!;

    gpu::GpuSpan out_span = gpu::get_buffer_span(&device, sample_out)!!;
    float* sampled = (float*)out_span.cpu;
    assert(quantize_unorm8(sampled[1]) == 90, "sampled green matches helper upload");
    assert(quantize_unorm8(sampled[2]) == 160, "sampled blue matches helper upload");
    gpu::end_frame(&device)!!;
}

fn void texture_readback_helper_restores_layout() @test {
    gpu::Device device = make_transfer_test_device();
    defer gpu::destroy_device(&device)!!;

    gpu::TextureDesc tex_desc = {
        .dimension = gpu::TextureDimension.TEX_2D,
        .width     = UR_TEX,
        .height    = UR_TEX,
        .format    = gpu::Format.RGBA8_UNORM,
        .usage     = { .sampled, .transfer_src, .transfer_dst },
    };
    gpu::TextureHandle texture = gpu::create_texture(&device, &tex_desc)!!;
    defer gpu::destroy_texture(&device, texture)!!;

    char[UR_TEX_BYTES] texels;
    for (uint i = 0; i < UR_TEX_BYTES; i++) texels[i] = (char)(i % 253);
    gpu::upload_texture_data(
        device:      &device,
        dst:         texture,
        mip:         0,
        data:        texels[..],
        next_stage:  gpu::Stage.FRAGMENT_SHADER,
        next_hazard: gpu::Hazard.SHADER_READ,
        next_layout: gpu::TextureLayout.SHADER_READ,
    )!!;

    char[UR_TEX_BYTES] out;
    gpu::readback_texture_data(
        device:      &device,
        src:         texture,
        mip:         0,
        out_data:    out[..],
        from_stage:  gpu::Stage.FRAGMENT_SHADER,
        from_hazard: gpu::Hazard.SHADER_READ,
        from_layout: gpu::TextureLayout.SHADER_READ,
    )!!;
    for (uint i = 0; i < UR_TEX_BYTES; i++) {
        assert(out[i] == (char)(i % 253), "texture readback matches upload");
    }

    // Layout restored: a second readback with the same from trio must work.
    gpu::readback_texture_data(
        device:      &device,
        src:         texture,
        mip:         0,
        out_data:    out[..],
        from_stage:  gpu::Stage.FRAGMENT_SHADER,
        from_hazard: gpu::Hazard.SHADER_READ,
        from_layout: gpu::TextureLayout.SHADER_READ,
    )!!;
}

fn void upload_helper_validation_faults() @test {
    gpu::Device device = make_transfer_test_device();
    defer gpu::destroy_device(&device)!!;

    gpu::TextureDesc tex_desc = {
        .dimension = gpu::TextureDimension.TEX_2D,
        .width     = UR_TEX,
        .height    = UR_TEX,
        .format    = gpu::Format.RGBA8_UNORM,
        .usage     = { .sampled, .transfer_dst },
    };
    gpu::TextureHandle texture = gpu::create_texture(&device, &tex_desc)!!;
    defer gpu::destroy_texture(&device, texture)!!;

    char[16] short_data;
    fault len_fault = @catch(gpu::upload_texture_data(
        device:      &device,
        dst:         texture,
        mip:         0,
        data:        short_data[..],
        next_stage:  gpu::Stage.COMPUTE_SHADER,
        next_hazard: gpu::Hazard.SHADER_READ,
        next_layout: gpu::TextureLayout.SHADER_READ,
    ));
    assert(len_fault == gpu::INVALID_ARGUMENT, "mismatched data length faults INVALID_ARGUMENT");

    gpu::BufferDesc buf_desc = {
        .size = 64, .usage = { .transfer_dst }, .memory_kind = gpu::MemoryKind.DEVICE,
    };
    gpu::BufferHandle buffer = gpu::create_buffer(&device, &buf_desc)!!;
    defer gpu::destroy_buffer(&device, buffer)!!;

    char[128] too_big;
    fault bounds_fault = @catch(gpu::upload_buffer_data(
        device:      &device,
        dst:         buffer,
        offset:      0,
        data:        too_big[..],
        next_stage:  gpu::Stage.TRANSFER,
        next_hazard: gpu::Hazard.TRANSFER_READ,
    ));
    assert(bounds_fault == gpu::INVALID_ARGUMENT, "past-the-end upload faults INVALID_ARGUMENT");

    // cmd-level upload into a texture that is not TRANSFER_DST still faults.
    gpu::begin_frame(&device)!!;
    gpu::CommandList cmd = gpu::begin_commands(&device, gpu::QueueKind.GRAPHICS)!!;
    char[UR_TEX_BYTES] full;
    gpu::TextureUploadDesc upload = { .texture = texture, .mip = 0 };
    fault layout_fault = @catch(gpu::cmd_upload_texture(&cmd, &upload, full[..]));
    assert(layout_fault == gpu::INVALID_RESOURCE_STATE, "cmd upload into UNDEFINED layout faults INVALID_RESOURCE_STATE");
    gpu::end_commands(&device, &cmd)!!;
    gpu::end_frame(&device)!!;
}
