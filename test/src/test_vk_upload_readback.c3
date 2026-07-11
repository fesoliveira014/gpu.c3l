module gpu_test;

import gpu;
import gpu::vk;
import std::core::mem;
import std::io;

// Headless (lavapipe): upload/readback helpers over the staging and readback
// arenas. The compute shader from the heap tests consumes helper-uploaded data.

const char[*] UPLOAD_HEAP_SAMPLE_SPIRV = $embed("shaders/heap_sample.comp.spv");

const uint UR_TEX = 8;
const usz UR_TEX_BYTES = UR_TEX * UR_TEX * 4;
const usz STAGING_REUSE_BYTES = 64;

struct TransferDiagnosticCapture {
    uint                      calls;
    gpu::DebugMessageSeverity severity;
    gpu::DebugMessageCategory category;
    bool                      has_fault;
    fault                     public_fault;
    gpu::DebugResourceKind    resource_kind;
    bool                      has_public_identity;
    uint                      resource_index;
    uint                      resource_generation;
    char[64]                  operation;
    char[64]                  resource_name;
    char[64]                  rejected_field;
    char[128]                 invariant;
    char[128]                 backend_text;
    char[64]                  validation_id_name;
}

struct StagingReuseFixture {
    gpu::Device device;
    gpu::BufferHandle[2] buffers;
    gpu::GpuSpan[2] spans;
}

fn void copy_transfer_diagnostic_text(char* destination, usz capacity, ZString source) {
    usz index = 0;
    if (source != null) {
        while (index + 1 < capacity && source[index] != 0) {
            destination[index] = source[index];
            index++;
        }
    }
    destination[index] = 0;
}

fn void capture_transfer_diagnostic(gpu::DebugMessage* message, void* user_data) {
    TransferDiagnosticCapture* capture = (TransferDiagnosticCapture*)user_data;
    capture.calls++;
    capture.severity = message.severity;
    capture.category = message.category;
    capture.has_fault = message.has_fault;
    capture.public_fault = message.public_fault;
    capture.resource_kind = message.resource.kind;
    capture.has_public_identity = message.resource.has_public_identity;
    capture.resource_index = message.resource.index;
    capture.resource_generation = message.resource.generation;
    copy_transfer_diagnostic_text(&capture.operation[0], 64, message.operation);
    copy_transfer_diagnostic_text(&capture.resource_name[0], 64, message.resource.debug_name);
    copy_transfer_diagnostic_text(&capture.rejected_field[0], 64, message.rejected_field);
    copy_transfer_diagnostic_text(&capture.invariant[0], 128, message.invariant);
    copy_transfer_diagnostic_text(&capture.backend_text[0], 128, message.backend_text);
    copy_transfer_diagnostic_text(&capture.validation_id_name[0], 64, message.validation_id_name);
}

fn void assert_transfer_rejection(
    TransferDiagnosticCapture* capture,
    uint before,
    fault returned,
    ZString operation,
    ZString rejected_field,
    ZString invariant,
    gpu::DebugResourceKind resource_kind,
    bool has_public_identity,
    uint resource_index,
    uint resource_generation,
    ZString resource_name,
) {
    assert(capture.calls == before + 1, "transfer rejection emits exactly one diagnostic");
    assert(capture.severity == gpu::DebugMessageSeverity.ERROR, "transfer rejection severity is ERROR");
    assert(capture.category.public_contract, "transfer rejection category is public_contract");
    assert(((ZString)&capture.operation[0]).str_view() == operation.str_view(), "transfer operation is exact");
    assert(capture.has_fault && capture.public_fault == returned, "transfer diagnostic fault matches return");
    assert(capture.resource_kind == resource_kind, "transfer resource kind is exact");
    assert(capture.has_public_identity == has_public_identity, "transfer identity presence is exact");
    assert(capture.resource_index == resource_index, "transfer resource index is exact");
    assert(capture.resource_generation == resource_generation, "transfer resource generation is exact");
    assert(((ZString)&capture.rejected_field[0]).str_view() == rejected_field.str_view(), "transfer rejected field is exact");
    assert(((ZString)&capture.invariant[0]).str_view() == invariant.str_view(), "transfer invariant is exact");
    if (resource_name == null) {
        assert(capture.resource_name[0] == 0, "transfer diagnostic omits unavailable resource name");
    } else {
        assert(((ZString)&capture.resource_name[0]).str_view() == resource_name.str_view(), "transfer resource name is exact");
    }
}

fn gpu::Device make_transfer_test_device(
    usz staging_size = 0,
    usz readback_size = 0,
    TransferDiagnosticCapture* capture = null,
) {
    gpu::DeviceDesc desc = {
        .backend            = gpu::BackendKind.VULKAN,
        .enable_validation  = capture == null,
        .frames_in_flight   = 2,
        .staging_arena_size = staging_size,
        .readback_arena_size = readback_size,
        .application_name   = "gpu_upload_readback_test",
        .debug_callback     = capture == null ? null : &capture_transfer_diagnostic,
        .debug_user_data    = capture,
    };
    return gpu::create_device(&desc)!!;
}

fn StagingReuseFixture make_staging_reuse_fixture() {
    StagingReuseFixture fixture;
    fixture.device = make_transfer_test_device(staging_size: STAGING_REUSE_BYTES);
    gpu::BufferDesc desc = { .size = STAGING_REUSE_BYTES, .usage = { .transfer_dst }, .memory_kind = gpu::MemoryKind.READBACK };
    for (uint i = 0; i < 2; i++) {
        fixture.buffers[i] = gpu::create_buffer(&fixture.device, &desc)!!;
        fixture.spans[i] = gpu::get_buffer_span(&fixture.device, fixture.buffers[i])!!;
    }
    return fixture;
}

fn void destroy_staging_reuse_fixture(StagingReuseFixture* fixture) {
    gpu::destroy_buffer(&fixture.device, fixture.buffers[1])!!;
    gpu::destroy_buffer(&fixture.device, fixture.buffers[0])!!;
    gpu::destroy_device(&fixture.device)!!;
}

fn void fill_staging_payload(char[] payload, uint seed) {
    foreach (i, &value : payload) *value = (char)(seed + i * 17);
}

fn void verify_staging_payload(StagingReuseFixture* fixture, uint index, char[] expected) {
    gpu::invalidate_buffer(
        device: &fixture.device,
        buffer: fixture.buffers[index],
        offset: 0,
        size:   0,
    )!!;
    char* actual = (char*)fixture.spans[index].cpu;
    foreach (i, value : expected) assert(actual[i] == value, "reused staging range preserves payload bytes");
}

fn void blocking_staging_upload(StagingReuseFixture* fixture, char[] payload) {
    gpu::upload_buffer_data(
        device:      &fixture.device,
        dst:         fixture.buffers[0],
        offset:      0,
        data:        payload,
        next_stage:  gpu::Stage.HOST,
        next_hazard: gpu::Hazard.HOST_READ,
    )!!;
    verify_staging_payload(fixture, 0, payload);
}

<* Waits for frame retirement so the next upload can reuse physical offset zero. *>
fn void recorded_staging_upload(StagingReuseFixture* fixture, gpu::QueueKind queue, uint index, char[] payload) {
    gpu::FrameToken frame = gpu::begin_frame(&fixture.device)!!;
    gpu::CommandList commands = gpu::begin_commands(&fixture.device, queue)!!;
    gpu::cmd_upload_buffer(
        commands: &commands,
        dst:      fixture.buffers[index],
        offset:   0,
        data:     payload,
    )!!;
    gpu::BufferBarrier host_read = {
        .buffer = fixture.buffers[index], .offset = 0, .size = payload.len,
        .before_stage = gpu::Stage.TRANSFER, .after_stage = gpu::Stage.HOST,
        .before_hazard = gpu::Hazard.TRANSFER_WRITE, .after_hazard = gpu::Hazard.HOST_READ,
    };
    gpu::cmd_buffer_barrier(&commands, &host_read)!!;
    gpu::end_commands(&commands)!!;
    gpu::CommandList[1] lists = { commands };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::submit(&fixture.device, &submit)!!;
    gpu::wait_queue_idle(&fixture.device, queue)!!;
    gpu::end_frame(&frame)!!;
    gpu::wait_queue_idle(&fixture.device, gpu::QueueKind.GRAPHICS)!!;
    verify_staging_payload(fixture, index, payload);
}

fn void run_blocking_to_recorded_reuse(StagingReuseFixture* fixture, gpu::QueueKind queue, uint first_seed, uint second_seed) {
    char[STAGING_REUSE_BYTES] first_payload;
    char[STAGING_REUSE_BYTES] second_payload;
    fill_staging_payload(first_payload[..], first_seed);
    fill_staging_payload(second_payload[..], second_seed);
    gpu::vk::VkDeviceState* state = (gpu::vk::VkDeviceState*)fixture.device.backend_state;
    ulong fallbacks = gpu::vk::dedicated_fallback_count(state);
    blocking_staging_upload(fixture, first_payload[..]);
    recorded_staging_upload(
        fixture: fixture,
        queue:   queue,
        index:   1,
        payload: second_payload[..],
    );
    assert(gpu::vk::dedicated_fallback_count(state) == fallbacks, "cross-family reuse stays on the staging ring");
}

fn void staging_range_reuse_blocking_graphics_to_recorded_compute() @test {
    StagingReuseFixture fixture = make_staging_reuse_fixture();
    defer destroy_staging_reuse_fixture(&fixture);
    gpu::vk::VkDeviceState* state = (gpu::vk::VkDeviceState*)fixture.device.backend_state;
    if (state.graphics_family == state.compute_family) {
        io::eprintfn(
            "staging_range_reuse_blocking_graphics_to_recorded_compute: N/A graphics=%d compute=%d transfer=%d",
            state.graphics_family,
            state.compute_family,
            state.transfer_family,
        );
        return;
    }
    run_blocking_to_recorded_reuse(
        fixture:    &fixture,
        queue:      gpu::QueueKind.COMPUTE,
        first_seed: 11,
        second_seed: 97,
    );
    io::eprintfn(
        "staging_range_reuse_blocking_graphics_to_recorded_compute: PASS graphics=%d compute=%d transfer=%d",
        state.graphics_family,
        state.compute_family,
        state.transfer_family,
    );
}

fn void staging_range_reuse_blocking_graphics_to_recorded_transfer() @test {
    StagingReuseFixture fixture = make_staging_reuse_fixture();
    defer destroy_staging_reuse_fixture(&fixture);
    gpu::vk::VkDeviceState* state = (gpu::vk::VkDeviceState*)fixture.device.backend_state;
    if (state.graphics_family == state.transfer_family) {
        io::eprintfn(
            "staging_range_reuse_blocking_graphics_to_recorded_transfer: N/A graphics=%d compute=%d transfer=%d",
            state.graphics_family,
            state.compute_family,
            state.transfer_family,
        );
        return;
    }
    run_blocking_to_recorded_reuse(
        fixture:    &fixture,
        queue:      gpu::QueueKind.TRANSFER,
        first_seed: 23,
        second_seed: 131,
    );
    io::eprintfn(
        "staging_range_reuse_blocking_graphics_to_recorded_transfer: PASS graphics=%d compute=%d transfer=%d",
        state.graphics_family,
        state.compute_family,
        state.transfer_family,
    );
}

fn void staging_range_reuse_recorded_compute_to_recorded_transfer() @test {
    StagingReuseFixture fixture = make_staging_reuse_fixture();
    defer destroy_staging_reuse_fixture(&fixture);
    gpu::vk::VkDeviceState* state = (gpu::vk::VkDeviceState*)fixture.device.backend_state;
    if (state.compute_family == state.transfer_family) {
        io::eprintfn(
            "staging_range_reuse_recorded_compute_to_recorded_transfer: N/A graphics=%d compute=%d transfer=%d",
            state.graphics_family,
            state.compute_family,
            state.transfer_family,
        );
        return;
    }
    char[STAGING_REUSE_BYTES] first_payload;
    char[STAGING_REUSE_BYTES] second_payload;
    fill_staging_payload(first_payload[..], 41);
    fill_staging_payload(second_payload[..], 173);
    ulong fallbacks = gpu::vk::dedicated_fallback_count(state);
    recorded_staging_upload(
        fixture: &fixture,
        queue:   gpu::QueueKind.COMPUTE,
        index:   0,
        payload: first_payload[..],
    );
    recorded_staging_upload(
        fixture: &fixture,
        queue:   gpu::QueueKind.TRANSFER,
        index:   1,
        payload: second_payload[..],
    );
    assert(gpu::vk::dedicated_fallback_count(state) == fallbacks, "recorded family switch stays on the staging ring");
    io::eprintfn(
        "staging_range_reuse_recorded_compute_to_recorded_transfer: PASS graphics=%d compute=%d transfer=%d",
        state.graphics_family,
        state.compute_family,
        state.transfer_family,
    );
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

<* An unrelated blocking helper's completion must never make an unsubmitted
   ticket poll ready. *>
fn void cmd_readback_ticket_not_ready_before_submit() @test {
    TransferDiagnosticCapture capture;
    gpu::Device device = make_transfer_test_device(capture: &capture);
    defer gpu::destroy_device(&device)!!;

    // Settle the timeline to a known baseline.
    gpu::FrameToken frame_token = gpu::begin_frame(&device)!!;
    gpu::end_frame(&frame_token)!!;

    gpu::BufferDesc src_desc = {
        .size        = 256,
        .usage       = { .storage, .transfer_src, .transfer_dst },
        .memory_kind = gpu::MemoryKind.DEVICE,
        .debug_name  = "ticket_src",
    };
    gpu::BufferHandle src = gpu::create_buffer(&device, &src_desc)!!;
    defer gpu::destroy_buffer(&device, src)!!;

    gpu::CommandList cmd = gpu::begin_commands(&device, gpu::QueueKind.GRAPHICS)!!;
    uint before = capture.calls;
    fault zero_size = @catch(gpu::cmd_readback_buffer(&cmd, src, 0, 0));
    assert(zero_size == gpu::INVALID_ARGUMENT, "zero ticket size keeps INVALID_ARGUMENT");
    assert_transfer_rejection(
        capture:             &capture,
        before:              before,
        returned:            zero_size,
        operation:           "cmd_readback_buffer",
        rejected_field:      "size",
        invariant:           "readback size must be greater than zero",
        resource_kind:       gpu::DebugResourceKind.BUFFER,
        has_public_identity: false,
        resource_index:      0,
        resource_generation: 0,
        resource_name:       null,
    );

    before = capture.calls;
    fault range_fault = @catch(gpu::cmd_readback_buffer(&cmd, src, 200, 64));
    assert(range_fault == gpu::INVALID_ARGUMENT, "ticket range past source keeps INVALID_ARGUMENT");
    assert_transfer_rejection(
        capture:             &capture,
        before:              before,
        returned:            range_fault,
        operation:           "cmd_readback_buffer",
        rejected_field:      "offset/size",
        invariant:           "readback range must fit the source buffer",
        resource_kind:       gpu::DebugResourceKind.BUFFER,
        has_public_identity: true,
        resource_index:      src.index,
        resource_generation: src.generation,
        resource_name:       "ticket_src",
    );

    gpu::ReadbackTicket ticket = gpu::cmd_readback_buffer(&cmd, src, 0, 256)!!;
    gpu::end_commands(&cmd)!!;
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
    before = capture.calls;
    fault short_dest = @catch(gpu::resolve_readback(&device, &ticket, dest[:255]));
    assert(short_dest == gpu::INVALID_ARGUMENT, "short ticket destination keeps INVALID_ARGUMENT");
    assert_transfer_rejection(
        capture:             &capture,
        before:              before,
        returned:            short_dest,
        operation:           "resolve_readback",
        rejected_field:      "dest",
        invariant:           "destination must cover the complete ticket span",
        resource_kind:       gpu::DebugResourceKind.NONE,
        has_public_identity: false,
        resource_index:      0,
        resource_generation: 0,
        resource_name:       null,
    );

    before = capture.calls;
    fault not_ready = @catch(gpu::resolve_readback(&device, &ticket, dest[..]));
    assert(not_ready == gpu::READBACK_NOT_READY, "resolve on an unsubmitted ticket faults READBACK_NOT_READY");
    assert_transfer_rejection(
        capture:             &capture,
        before:              before,
        returned:            not_ready,
        operation:           "resolve_readback",
        rejected_field:      "ticket.value",
        invariant:           "ticket timeline value must be signaled before resolve",
        resource_kind:       gpu::DebugResourceKind.NONE,
        has_public_identity: false,
        resource_index:      0,
        resource_generation: 0,
        resource_name:       null,
    );
}

fn void oversized_upload_uses_dedicated_staging() @test {
    // Tiny arena: every upload larger than 1 KiB falls back to a dedicated buffer.
    TransferDiagnosticCapture capture;
    gpu::Device device = make_transfer_test_device(
        staging_size:  1024,
        readback_size: 1024,
        capture:       &capture,
    );
    defer gpu::destroy_device(&device)!!;
    gpu::vk::VkDeviceState* state = (gpu::vk::VkDeviceState*)device.backend_state;
    ulong fallbacks = gpu::vk::dedicated_fallback_count(state);
    uint diagnostic_count = capture.calls;

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
    assert(gpu::vk::dedicated_fallback_count(state) == fallbacks + 2, "oversized upload and readback each use a dedicated fallback");
    assert(capture.calls == diagnostic_count, "successful dedicated fallback emits no contract diagnostic");
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
        gpu::FrameToken frame_token = gpu::begin_frame(&device)!!;
        gpu::CommandList cmd = gpu::begin_commands(&device, gpu::QueueKind.GRAPHICS)!!;
        gpu::cmd_upload_buffer(&cmd, buffer, 0, data[..])!!;
        gpu::BufferBarrier next_use = {
            .buffer = buffer, .offset = 0, .size = 1024,
            .before_stage = gpu::Stage.TRANSFER, .after_stage = gpu::Stage.COMPUTE_SHADER,
            .before_hazard = gpu::Hazard.TRANSFER_WRITE, .after_hazard = gpu::Hazard.SHADER_READ,
        };
        gpu::cmd_buffer_barrier(&cmd, &next_use)!!;
        gpu::end_commands(&cmd)!!;
        gpu::CommandList[1] lists = { cmd };
        gpu::SubmitDesc submit = { .command_lists = lists[..] };
        gpu::submit(&device, &submit)!!;
        gpu::end_frame(&frame_token)!!;
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
    gpu::end_commands(&cmd)!!;
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

    gpu::FrameToken frame_token = gpu::begin_frame(&device)!!;
    gpu::GpuSpan root_span = gpu::alloc_frame_span(&frame_token, SampleRoot::size, 16)!!;
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
    gpu::end_commands(&cmd)!!;
    gpu::CommandList[1] lists = { cmd };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::submit(&device, &submit)!!;
    gpu::wait_queue_idle(&device, gpu::QueueKind.COMPUTE)!!;

    gpu::GpuSpan out_span = gpu::get_buffer_span(&device, sample_out)!!;
    float* sampled = (float*)out_span.cpu;
    assert(quantize_unorm8(sampled[1]) == 90, "sampled green matches helper upload");
    assert(quantize_unorm8(sampled[2]) == 160, "sampled blue matches helper upload");
    gpu::end_frame(&frame_token)!!;
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
    TransferDiagnosticCapture capture;
    gpu::Device device = make_transfer_test_device(capture: &capture);
    defer gpu::destroy_device(&device)!!;
    gpu::vk::VkDeviceState* state = (gpu::vk::VkDeviceState*)device.backend_state;

    gpu::TextureDesc tex_desc = {
        .dimension = gpu::TextureDimension.TEX_2D,
        .width     = UR_TEX,
        .height    = UR_TEX,
        .format    = gpu::Format.RGBA8_UNORM,
        .usage     = { .sampled, .transfer_dst },
        .debug_name = "helper_upload_texture",
    };
    gpu::TextureHandle texture = gpu::create_texture(&device, &tex_desc)!!;
    defer gpu::destroy_texture(&device, texture)!!;

    char[16] short_data;
    uint before = capture.calls;
    ulong helper_before = state.helper_counter;
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
    assert_transfer_rejection(
        capture:             &capture,
        before:              before,
        returned:            len_fault,
        operation:           "upload_texture_data",
        rejected_field:      "data",
        invariant:           "data length must match the selected texture mip",
        resource_kind:       gpu::DebugResourceKind.TEXTURE,
        has_public_identity: true,
        resource_index:      texture.index,
        resource_generation: texture.generation,
        resource_name:       "helper_upload_texture",
    );
    assert(state.helper_counter == helper_before, "invalid texture length reserves no helper value");

    before = capture.calls;
    helper_before = state.helper_counter;
    fault mip_fault = @catch(gpu::upload_texture_data(
        device:      &device,
        dst:         texture,
        mip:         1,
        data:        short_data[..],
        next_stage:  gpu::Stage.COMPUTE_SHADER,
        next_hazard: gpu::Hazard.SHADER_READ,
        next_layout: gpu::TextureLayout.SHADER_READ,
    ));
    assert(mip_fault == gpu::INVALID_ARGUMENT, "invalid bare mip faults INVALID_ARGUMENT");
    assert_transfer_rejection(
        capture:             &capture,
        before:              before,
        returned:            mip_fault,
        operation:           "upload_texture_data",
        rejected_field:      "mip",
        invariant:           "mip must exist in the texture mip chain",
        resource_kind:       gpu::DebugResourceKind.TEXTURE,
        has_public_identity: true,
        resource_index:      texture.index,
        resource_generation: texture.generation,
        resource_name:       "helper_upload_texture",
    );
    assert(state.helper_counter == helper_before, "invalid bare mip reserves no helper value");

    gpu::BufferDesc buf_desc = {
        .size = 64, .usage = { .transfer_dst }, .memory_kind = gpu::MemoryKind.DEVICE, .debug_name = "helper_upload_buffer",
    };
    gpu::BufferHandle buffer = gpu::create_buffer(&device, &buf_desc)!!;
    defer gpu::destroy_buffer(&device, buffer)!!;

    char[128] too_big;
    char[] empty_data = {};
    before = capture.calls;
    helper_before = state.helper_counter;
    fault empty_fault = @catch(gpu::upload_buffer_data(
        device:      &device,
        dst:         buffer,
        offset:      0,
        data:        empty_data,
        next_stage:  gpu::Stage.TRANSFER,
        next_hazard: gpu::Hazard.TRANSFER_READ,
    ));
    assert(empty_fault == gpu::INVALID_ARGUMENT, "empty upload faults INVALID_ARGUMENT");
    assert_transfer_rejection(
        capture:             &capture,
        before:              before,
        returned:            empty_fault,
        operation:           "upload_buffer_data",
        rejected_field:      "data",
        invariant:           "upload data must not be empty",
        resource_kind:       gpu::DebugResourceKind.BUFFER,
        has_public_identity: false,
        resource_index:      0,
        resource_generation: 0,
        resource_name:       null,
    );
    assert(state.helper_counter == helper_before, "empty upload reserves no helper value");

    gpu::BufferDesc no_transfer_buffer_desc = {
        .size        = 64,
        .usage       = { .storage },
        .memory_kind = gpu::MemoryKind.DEVICE,
        .debug_name  = "no_transfer_upload_buffer",
    };
    gpu::BufferHandle no_transfer_buffer = gpu::create_buffer(&device, &no_transfer_buffer_desc)!!;
    defer gpu::destroy_buffer(&device, no_transfer_buffer)!!;
    char[8] small_upload;
    before = capture.calls;
    helper_before = state.helper_counter;
    fault buffer_usage_fault = @catch(gpu::upload_buffer_data(
        device:      &device,
        dst:         no_transfer_buffer,
        offset:      0,
        data:        small_upload[..],
        next_stage:  gpu::Stage.TRANSFER,
        next_hazard: gpu::Hazard.TRANSFER_READ,
    ));
    assert(buffer_usage_fault == gpu::INVALID_ARGUMENT, "buffer without transfer_dst faults INVALID_ARGUMENT");
    assert_transfer_rejection(
        capture:             &capture,
        before:              before,
        returned:            buffer_usage_fault,
        operation:           "upload_buffer_data",
        rejected_field:      "BufferDesc.usage",
        invariant:           "destination buffer must include transfer_dst",
        resource_kind:       gpu::DebugResourceKind.BUFFER,
        has_public_identity: true,
        resource_index:      no_transfer_buffer.index,
        resource_generation: no_transfer_buffer.generation,
        resource_name:       "no_transfer_upload_buffer",
    );
    assert(state.helper_counter == helper_before, "invalid buffer usage reserves no helper value");

    before = capture.calls;
    helper_before = state.helper_counter;
    fault bounds_fault = @catch(gpu::upload_buffer_data(
        device:      &device,
        dst:         buffer,
        offset:      0,
        data:        too_big[..],
        next_stage:  gpu::Stage.TRANSFER,
        next_hazard: gpu::Hazard.TRANSFER_READ,
    ));
    assert(bounds_fault == gpu::INVALID_ARGUMENT, "past-the-end upload faults INVALID_ARGUMENT");
    assert_transfer_rejection(
        capture:             &capture,
        before:              before,
        returned:            bounds_fault,
        operation:           "upload_buffer_data",
        rejected_field:      "offset/data",
        invariant:           "upload range must fit the destination buffer",
        resource_kind:       gpu::DebugResourceKind.BUFFER,
        has_public_identity: true,
        resource_index:      buffer.index,
        resource_generation: buffer.generation,
        resource_name:       "helper_upload_buffer",
    );
    assert(state.helper_counter == helper_before, "invalid buffer range reserves no helper value");

    gpu::TextureDesc no_transfer_desc = {
        .dimension  = gpu::TextureDimension.TEX_2D,
        .width      = UR_TEX,
        .height     = UR_TEX,
        .format     = gpu::Format.RGBA8_UNORM,
        .usage      = { .sampled },
        .debug_name = "no_transfer_upload_texture",
    };
    gpu::TextureHandle no_transfer_texture = gpu::create_texture(&device, &no_transfer_desc)!!;
    defer gpu::destroy_texture(&device, no_transfer_texture)!!;
    char[UR_TEX_BYTES] full_upload;
    before = capture.calls;
    helper_before = state.helper_counter;
    fault texture_usage_fault = @catch(gpu::upload_texture_data(
        device:      &device,
        dst:         no_transfer_texture,
        mip:         0,
        data:        full_upload[..],
        next_stage:  gpu::Stage.COMPUTE_SHADER,
        next_hazard: gpu::Hazard.SHADER_READ,
        next_layout: gpu::TextureLayout.SHADER_READ,
    ));
    assert(texture_usage_fault == gpu::INVALID_ARGUMENT, "texture without transfer_dst faults INVALID_ARGUMENT");
    assert_transfer_rejection(
        capture:             &capture,
        before:              before,
        returned:            texture_usage_fault,
        operation:           "upload_texture_data",
        rejected_field:      "TextureDesc.usage",
        invariant:           "destination texture must include transfer_dst",
        resource_kind:       gpu::DebugResourceKind.TEXTURE,
        has_public_identity: true,
        resource_index:      no_transfer_texture.index,
        resource_generation: no_transfer_texture.generation,
        resource_name:       "no_transfer_upload_texture",
    );
    assert(state.helper_counter == helper_before, "invalid texture usage reserves no helper value");
    gpu::vk::TextureSlot* no_transfer_slot = state.textures.get(no_transfer_texture)!!;
    assert(
        gpu::vk::texture_layout(no_transfer_slot) == gpu::TextureLayout.UNDEFINED,
        "invalid texture usage preserves the tracked layout"
    );

    char[8] readback_dest;
    before = capture.calls;
    fault usage_fault = @catch(gpu::readback_buffer_data(
        device:      &device,
        src:         buffer,
        offset:      0,
        out_data:    readback_dest[..],
        from_stage:  gpu::Stage.TRANSFER,
        from_hazard: gpu::Hazard.TRANSFER_WRITE,
    ));
    assert(usage_fault == gpu::INVALID_ARGUMENT, "readback source without transfer_src keeps INVALID_ARGUMENT");
    assert_transfer_rejection(
        capture:             &capture,
        before:              before,
        returned:            usage_fault,
        operation:           "readback_buffer_data",
        rejected_field:      "BufferDesc.usage",
        invariant:           "source buffer must include transfer_src",
        resource_kind:       gpu::DebugResourceKind.BUFFER,
        has_public_identity: true,
        resource_index:      buffer.index,
        resource_generation: buffer.generation,
        resource_name:       "helper_upload_buffer",
    );

    // cmd-level upload into a texture that is not TRANSFER_DST still faults.
    gpu::FrameToken frame_token = gpu::begin_frame(&device)!!;
    gpu::CommandList cmd = gpu::begin_commands(&device, gpu::QueueKind.GRAPHICS)!!;
    char[UR_TEX_BYTES] full;
    gpu::TextureUploadDesc upload = { .texture = texture, .mip = 0 };
    gpu::TextureUploadDesc invalid_mip_upload = { .texture = texture, .mip = 1 };
    before = capture.calls;
    fault desc_mip_fault = @catch(gpu::cmd_upload_texture(&cmd, &invalid_mip_upload, full[..]));
    assert(desc_mip_fault == gpu::INVALID_ARGUMENT, "invalid descriptor mip faults INVALID_ARGUMENT");
    assert_transfer_rejection(
        capture:             &capture,
        before:              before,
        returned:            desc_mip_fault,
        operation:           "cmd_upload_texture",
        rejected_field:      "TextureUploadDesc.mip",
        invariant:           "mip must exist in the texture mip chain",
        resource_kind:       gpu::DebugResourceKind.TEXTURE,
        has_public_identity: true,
        resource_index:      texture.index,
        resource_generation: texture.generation,
        resource_name:       "helper_upload_texture",
    );

    before = capture.calls;
    fault layout_fault = @catch(gpu::cmd_upload_texture(&cmd, &upload, full[..]));
    assert(layout_fault == gpu::INVALID_RESOURCE_STATE, "cmd upload into UNDEFINED layout faults INVALID_RESOURCE_STATE");
    assert_transfer_rejection(
        capture:             &capture,
        before:              before,
        returned:            layout_fault,
        operation:           "cmd_upload_texture",
        rejected_field:      "TextureUploadDesc.texture",
        invariant:           "destination texture must be in TRANSFER_DST layout",
        resource_kind:       gpu::DebugResourceKind.TEXTURE,
        has_public_identity: true,
        resource_index:      texture.index,
        resource_generation: texture.generation,
        resource_name:       "helper_upload_texture",
    );
    gpu::end_commands(&cmd)!!;
    gpu::end_frame(&frame_token)!!;
}

fn void transfer_callback_presence_preserves_upload_fault() @test {
    TransferDiagnosticCapture capture;
    gpu::Device callback_device = make_transfer_test_device(capture: &capture);
    gpu::BufferDesc desc = { .size = 16, .usage = { .transfer_dst }, .memory_kind = gpu::MemoryKind.DEVICE };
    gpu::BufferHandle callback_buffer = gpu::create_buffer(&callback_device, &desc)!!;
    char[32] payload;
    fault callback_fault = @catch(gpu::upload_buffer_data(
        device:      &callback_device,
        dst:         callback_buffer,
        offset:      0,
        data:        payload[..],
        next_stage:  gpu::Stage.TRANSFER,
        next_hazard: gpu::Hazard.TRANSFER_READ,
    ));
    assert(capture.calls == 1, "callback upload rejection emits exactly once");
    gpu::destroy_buffer(&callback_device, callback_buffer)!!;
    gpu::destroy_device(&callback_device)!!;

    gpu::Device plain_device = make_transfer_test_device();
    gpu::BufferHandle plain_buffer = gpu::create_buffer(&plain_device, &desc)!!;
    fault plain_fault = @catch(gpu::upload_buffer_data(
        device:      &plain_device,
        dst:         plain_buffer,
        offset:      0,
        data:        payload[..],
        next_stage:  gpu::Stage.TRANSFER,
        next_hazard: gpu::Hazard.TRANSFER_READ,
    ));
    gpu::destroy_buffer(&plain_device, plain_buffer)!!;
    gpu::destroy_device(&plain_device)!!;
    assert(plain_fault == callback_fault, "callback presence does not change upload fault");
}
