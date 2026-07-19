module gpu_test;

import gpu @public;
import gpu::vk @public;
import vk;
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
    int                       validation_id_number;
}

struct StagingReuseFixture {
    gpu::Device device;
    OwnedSpan[2] buffers;
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
    capture.validation_id_number = message.validation_id_number;
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
    return gpu::create_device_from_desc(&desc)!!;
}

fn StagingReuseFixture make_staging_reuse_fixture() {
    StagingReuseFixture fixture;
    fixture.device = make_transfer_test_device(staging_size: STAGING_REUSE_BYTES);
    for (uint i = 0; i < 2; i++) {
        fixture.buffers[i] = allocate_owned_span(
            device:       &fixture.device,
            size:         STAGING_REUSE_BYTES,
            memory_class: gpu::MemoryClass.CPU_READ,
            access:       { .graphics, .compute, .transfer },
            debug_name:   "staging_reuse_readback",
        )!!;
        fixture.spans[i] = fixture.buffers[i].span;
    }
    return fixture;
}

fn void destroy_staging_reuse_fixture(StagingReuseFixture* fixture) {
    free_owned_span(&fixture.device, &fixture.buffers[1])!!;
    free_owned_span(&fixture.device, &fixture.buffers[0])!!;
    gpu::destroy_device(&fixture.device)!!;
}

fn void fill_staging_payload(char[] payload, uint seed) {
    foreach (i, &value : payload) *value = (char)(seed + i * 17);
}

fn void verify_staging_payload(StagingReuseFixture* fixture, uint index, char[] expected) {
    gpu::invalidate_mapped_span(&fixture.device, fixture.spans[index])!!;
    char* actual = (char*)test_span_mapping(&fixture.device, fixture.spans[index]).ptr;
    foreach (i, value : expected) assert(actual[i] == value, "reused staging range preserves payload bytes");
}

fn void blocking_staging_upload(StagingReuseFixture* fixture, char[] payload) {
    gpu::upload_buffer_data(
        device:      &fixture.device,
        dst:         fixture.spans[0],
        data:        payload,
        next_stage:  gpu::Stage.HOST,
        next_hazard: gpu::Hazard.HOST_READ,
    )!!;
    verify_staging_payload(fixture, 0, payload);
}

<* Waits for frame retirement so the next upload can reuse physical offset zero. *>
fn void recorded_staging_upload(StagingReuseFixture* fixture, gpu::QueueKind queue, uint index, char[] payload) {
    gpu::FrameToken frame = gpu::begin_frame(&fixture.device)!!;
    gpu::CommandList commands = gpu::begin_commands(gpu::get_queue(&fixture.device, queue)!!)!!;
    defer discard_test_commands(&commands)!!;
    gpu::cmd_upload_buffer(
        commands: &commands,
        dst:      fixture.spans[index],
        data:     payload,
    )!!;
    gpu::BufferBarrier host_read = {
        .span = fixture.spans[index],
        .before_stage = gpu::Stage.TRANSFER, .after_stage = gpu::Stage.HOST,
        .before_hazard = gpu::Hazard.TRANSFER_WRITE, .after_hazard = gpu::Hazard.HOST_READ,
    };
    gpu::cmd_buffer_barrier(&commands, &host_read)!!;
    gpu::ExecutableCommandList commands_executable = gpu::end_commands(&commands)!!;
    defer discard_test_executable(&commands_executable)!!;
    gpu::ExecutableCommandList[1] lists = { commands_executable };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    submit_test_queue(&fixture.device, queue, &submit)!!;
    wait_test_queue(&fixture.device, queue)!!;
    gpu::end_frame(&frame)!!;
    wait_test_queue(&fixture.device, gpu::QueueKind.GRAPHICS)!!;
    verify_staging_payload(fixture, index, payload);
}

fn void run_blocking_to_recorded_reuse(StagingReuseFixture* fixture, gpu::QueueKind queue, uint first_seed, uint second_seed) {
    char[STAGING_REUSE_BYTES] first_payload;
    char[STAGING_REUSE_BYTES] second_payload;
    fill_staging_payload(first_payload[..], first_seed);
    fill_staging_payload(second_payload[..], second_seed);
    gpu::vk::VkDeviceState* state = (gpu::vk::VkDeviceState*)gpu::device_backend_state(fixture.device);
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
    gpu::vk::VkDeviceState* state = (gpu::vk::VkDeviceState*)gpu::device_backend_state(fixture.device);
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
    gpu::vk::VkDeviceState* state = (gpu::vk::VkDeviceState*)gpu::device_backend_state(fixture.device);
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
    gpu::vk::VkDeviceState* state = (gpu::vk::VkDeviceState*)gpu::device_backend_state(fixture.device);
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
    defer destroy_test_device(&device)!!;

    OwnedSpan buffer = allocate_owned_span(
        device:       &device,
        size:         4096,
        memory_class: gpu::MemoryClass.GPU_PRIVATE,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "device_buffer",
    )!!;
    defer free_owned_span(&device, &buffer)!!;
    gpu::GpuSpan buffer_span = buffer.span;

    char[4096] data;
    for (uint i = 0; i < 4096; i++) data[i] = (char)(i * 7);
    gpu::upload_buffer_data(
        device:      &device,
        dst:         buffer_span,
        data:        data[..],
        next_stage:  gpu::Stage.TRANSFER,
        next_hazard: gpu::Hazard.TRANSFER_READ,
    )!!;

    char[4096] out;
    gpu::readback_buffer_data(
        device:      &device,
        src:         buffer_span,
        out_data:    out[..],
        from_stage:  gpu::Stage.TRANSFER,
        from_hazard: gpu::Hazard.TRANSFER_READ,
    )!!;
    for (uint i = 0; i < 4096; i++) {
        assert(out[i] == (char)(i * 7), "device-memory round trip preserves bytes");
    }
}

fn void transfer_lengths_must_match_exact_spans() @test {
    gpu::Device device = make_transfer_test_device();
    defer destroy_test_device(&device)!!;

    OwnedSpan buffer = allocate_owned_span(
        device:       &device,
        size:         16,
        memory_class: gpu::MemoryClass.GPU_PRIVATE,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "exact_transfer_span",
    )!!;
    defer free_owned_span(&device, &buffer)!!;
    gpu::GpuSpan span = buffer.span;
    char[15] shorter;
    char[17] longer;

    gpu::CommandList commands = gpu::begin_commands(
        gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!,
    )!!;
    defer discard_test_commands(&commands)!!;
    fault cmd_short = @catch(gpu::cmd_upload_buffer(&commands, span, shorter[..]));
    assert(cmd_short == gpu::INVALID_ARGUMENT, "recorded upload rejects data shorter than its span");
    fault cmd_long = @catch(gpu::cmd_upload_buffer(&commands, span, longer[..]));
    assert(cmd_long == gpu::INVALID_ARGUMENT, "recorded upload rejects data longer than its span");

    fault upload_short = @catch(gpu::upload_buffer_data(
        device:      &device,
        dst:         span,
        data:        shorter[..],
        next_stage:  gpu::Stage.TRANSFER,
        next_hazard: gpu::Hazard.TRANSFER_READ,
    ));
    assert(upload_short == gpu::INVALID_ARGUMENT, "blocking upload rejects data shorter than its span");
    fault upload_long = @catch(gpu::upload_buffer_data(
        device:      &device,
        dst:         span,
        data:        longer[..],
        next_stage:  gpu::Stage.TRANSFER,
        next_hazard: gpu::Hazard.TRANSFER_READ,
    ));
    assert(upload_long == gpu::INVALID_ARGUMENT, "blocking upload rejects data longer than its span");

    fault readback_short = @catch(gpu::readback_buffer_data(
        device:      &device,
        src:         span,
        out_data:    shorter[..],
        from_stage:  gpu::Stage.TRANSFER,
        from_hazard: gpu::Hazard.TRANSFER_WRITE,
    ));
    assert(readback_short == gpu::INVALID_ARGUMENT, "blocking readback rejects output shorter than its span");
    fault readback_long = @catch(gpu::readback_buffer_data(
        device:      &device,
        src:         span,
        out_data:    longer[..],
        from_stage:  gpu::Stage.TRANSFER,
        from_hazard: gpu::Hazard.TRANSFER_WRITE,
    ));
    assert(readback_long == gpu::INVALID_ARGUMENT, "blocking readback rejects output longer than its span");
}

fn void transfer_rejections_report_resolved_command_and_lookup_context() @test {
    TransferDiagnosticCapture capture;
    gpu::Device device = make_transfer_test_device(capture: &capture);
    defer destroy_test_device(&device)!!;

    OwnedSpan buffer = allocate_owned_span(
        device:       &device,
        size:         64,
        memory_class: gpu::MemoryClass.GPU_PRIVATE,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "transfer_diagnostic_buffer",
    )!!;
    defer free_owned_span(&device, &buffer)!!;

    char[16] data;
    gpu::CommandList commands = gpu::begin_commands(
        gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!,
    )!!;
    defer discard_test_commands(&commands)!!;
    gpu::CommandList ended_alias = commands;
    gpu::ExecutableCommandList executable = gpu::end_commands(&commands)!!;
    defer discard_test_executable(&executable)!!;
    uint before = capture.calls;
    fault command_state = @catch(gpu::cmd_upload_buffer(
        &ended_alias,
        buffer.span.checked_subspan(0, data.len)!!,
        data[..],
    ));
    assert(command_state == gpu::COMMAND_RECORDING_ERROR, "ended upload command keeps COMMAND_RECORDING_ERROR");
    assert_transfer_rejection(
        capture:             &capture,
        before:              before,
        returned:            command_state,
        operation:           "cmd_upload_buffer",
        rejected_field:      "CommandList.state",
        invariant:           "command must be recording outside a render pass",
        resource_kind:       gpu::DebugResourceKind.COMMAND_LIST,
        has_public_identity: true,
        resource_index:      ended_alias.handle.index,
        resource_generation: ended_alias.handle.generation,
        resource_name:       null,
    );

    gpu::CommandList lookup_commands = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!)!!;
    defer discard_test_commands(&lookup_commands)!!;
    gpu::GpuSpan missing_span = {
        .index      = uint::max,
        .generation = 1,
        .size       = data.len,
    };
    before = capture.calls;
    fault invalid_buffer = @catch(gpu::cmd_upload_buffer(&lookup_commands, missing_span, data[..]));
    assert(invalid_buffer == gpu::INVALID_HANDLE, "missing upload span keeps INVALID_HANDLE");
    assert_transfer_rejection(
        capture:             &capture,
        before:              before,
        returned:            invalid_buffer,
        operation:           "cmd_upload_buffer",
        rejected_field:      "GpuSpan.identity",
        invariant:           "must identify one live span owner",
        resource_kind:       gpu::DebugResourceKind.GPU_SPAN,
        has_public_identity: false,
        resource_index:      0,
        resource_generation: 0,
        resource_name:       null,
    );
}

fn void stale_allocation_span_rejects_after_slot_reuse() @test {
    TransferDiagnosticCapture capture;
    gpu::Device device = make_transfer_test_device(capture: &capture);
    defer destroy_test_device(&device)!!;

    OwnedSpan first = allocate_owned_span(
        device:       &device,
        size:         16,
        memory_class: gpu::MemoryClass.GPU_PRIVATE,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "stale_allocation",
    )!!;
    gpu::GpuSpan stale = first.span;
    free_owned_span(&device, &first)!!;

    OwnedSpan replacement = allocate_owned_span(
        device:       &device,
        size:         16,
        memory_class: gpu::MemoryClass.GPU_PRIVATE,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "replacement_allocation",
    )!!;
    defer free_owned_span(&device, &replacement)!!;
    assert(replacement.span.index == stale.index
        && replacement.span.generation != stale.generation,
        "allocation slot reuse advances the generation");

    char[16] data;
    gpu::CommandList commands = gpu::begin_commands(
        gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!,
    )!!;
    defer discard_test_commands(&commands)!!;

    uint before = capture.calls;
    fault stale_fault = @catch(gpu::cmd_upload_buffer(
        &commands,
        stale,
        data[..],
    ));
    assert(stale_fault == gpu::INVALID_HANDLE,
        "a stale span cannot address the allocation reusing its slot");
    assert_transfer_rejection(
        capture:             &capture,
        before:              before,
        returned:            stale_fault,
        operation:           "cmd_upload_buffer",
        rejected_field:      "GpuSpan.identity",
        invariant:           "must identify one live span owner",
        resource_kind:       gpu::DebugResourceKind.ALLOCATION,
        has_public_identity: false,
        resource_index:      0,
        resource_generation: 0,
        resource_name:       null,
    );
}

fn void stale_transfer_command_owners_report_without_identity() @test {
    TransferDiagnosticCapture capture;
    gpu::Device device = make_transfer_test_device(capture: &capture);
    defer destroy_test_device(&device)!!;

    gpu::CommandList stale = gpu::begin_commands(
        gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!,
    )!!;
    defer discard_test_commands(&stale)!!;
    gpu::CommandList stale_alias = stale;
    gpu::ExecutableCommandList stale_executable = gpu::end_commands(&stale)!!;
    defer discard_test_executable(&stale_executable)!!;
    gpu::ExecutableCommandList[1] lists = { stale_executable };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    submit_test_queue(&device, gpu::QueueKind.GRAPHICS, &submit)!!;
    wait_test_queue(&device, gpu::QueueKind.GRAPHICS)!!;

    char[4] data;
    gpu::TextureUploadDesc texture_upload;
    uint before = capture.calls;
    fault upload_buffer = @catch(gpu::cmd_upload_buffer(&stale_alias, {}, data[..]));
    assert(upload_buffer == gpu::INVALID_HANDLE, "stale buffer upload keeps INVALID_HANDLE");
    assert_transfer_rejection(
        capture:             &capture,
        before:              before,
        returned:            upload_buffer,
        operation:           "cmd_upload_buffer",
        rejected_field:      "CommandList.handle",
        invariant:           "command list handle must identify a live command list",
        resource_kind:       gpu::DebugResourceKind.COMMAND_LIST,
        has_public_identity: false,
        resource_index:      0,
        resource_generation: 0,
        resource_name:       null,
    );

    before = capture.calls;
    fault upload_texture = @catch(gpu::cmd_upload_texture(&stale_alias, &texture_upload, data[..]));
    assert(upload_texture == gpu::INVALID_HANDLE, "stale texture upload keeps INVALID_HANDLE");
    assert_transfer_rejection(
        capture:             &capture,
        before:              before,
        returned:            upload_texture,
        operation:           "cmd_upload_texture",
        rejected_field:      "CommandList.handle",
        invariant:           "command list handle must identify a live command list",
        resource_kind:       gpu::DebugResourceKind.COMMAND_LIST,
        has_public_identity: false,
        resource_index:      0,
        resource_generation: 0,
        resource_name:       null,
    );

}

fn void helper_backend_result_diagnostics_preserve_public_fault_metadata() @test {
    TransferDiagnosticCapture capture;
    gpu::Device device = make_transfer_test_device(capture: &capture);
    defer destroy_test_device(&device)!!;
    gpu::vk::VkDeviceState* state = (gpu::vk::VkDeviceState*)gpu::device_backend_state(device);

    uint before = capture.calls;
    fault cleanup_fault = @catch(gpu::vk::check_helper_backend_result(
        state:           state,
        operation:       "upload_buffer_data",
        result:          vk::Result.ERROR_DEVICE_LOST,
        fault_is_public: false,
    ));
    assert(cleanup_fault == gpu::DEVICE_LOST, "cleanup result mapping keeps DEVICE_LOST");
    assert(capture.calls == before + 1, "cleanup backend failure emits exactly once");
    assert(capture.category.backend, "cleanup result is a backend diagnostic");
    assert(!capture.has_fault, "swallowed cleanup diagnostic omits a public fault");
    assert(((ZString)&capture.operation[0]).str_view() == "upload_buffer_data", "cleanup operation is public owner");
    assert(((ZString)&capture.backend_text[0]).str_view() == "Vulkan device lost", "cleanup diagnostic preserves backend text");
    assert(capture.validation_id_number == (int)vk::Result.ERROR_DEVICE_LOST, "cleanup diagnostic preserves backend result");

    before = capture.calls;
    fault reset_fault = @catch(gpu::vk::check_helper_backend_result(
        state:           state,
        operation:       "readback_texture_data",
        result:          vk::Result.ERROR_DEVICE_LOST,
        fault_is_public: true,
    ));
    assert(reset_fault == gpu::DEVICE_LOST, "propagated reset result keeps DEVICE_LOST");
    assert(capture.calls == before + 1, "propagated backend failure emits exactly once");
    assert(capture.category.backend, "propagated result is a backend diagnostic");
    assert(capture.has_fault && capture.public_fault == reset_fault, "propagated result carries matching public fault");
    assert(((ZString)&capture.operation[0]).str_view() == "readback_texture_data", "reset operation is public owner");
    assert(((ZString)&capture.backend_text[0]).str_view() == "Vulkan device lost", "propagated diagnostic preserves backend text");
    assert(capture.validation_id_number == (int)vk::Result.ERROR_DEVICE_LOST, "propagated diagnostic preserves backend result");
}

fn void transfer_allocation_exhaustion_reports_at_public_owner() @test {
    TransferDiagnosticCapture capture;
    gpu::Device device = make_transfer_test_device(
        staging_size:  64,
        readback_size: 64,
        capture:       &capture,
    );
    defer destroy_test_device(&device)!!;
    gpu::vk::VkDeviceState* state = (gpu::vk::VkDeviceState*)gpu::device_backend_state(device);

    OwnedSpan buffer = allocate_owned_span(
        device:       &device,
        size:         128,
        memory_class: gpu::MemoryClass.GPU_PRIVATE,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "exhaustion_buffer",
    )!!;
    defer free_owned_span(&device, &buffer)!!;
    gpu::CommandList commands = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!)!!;
    defer discard_test_commands(&commands)!!;
    gpu::TextureDesc texture_desc = {
        .dimension  = gpu::TextureDimension.TEX_2D,
        .access    = { .graphics, .compute, .transfer },
        .width      = 2,
        .height     = 2,
        .format     = gpu::Format.RGBA8_UNORM,
        .usage      = { .transfer_src, .transfer_dst },
        .debug_name = "exhaustion_texture",
    };
    gpu::TextureHandle texture = gpu::create_texture(&device, &texture_desc)!!;
    defer gpu::destroy_texture(&device, texture)!!;

    usz staging_size = state.staging_arena.size;
    uint dedicated_count = state.dedicated_staging.count;
    state.staging_arena.size = 1;
    state.dedicated_staging.count = gpu::vk::DEDICATED_RETIRE_CAP;
    foreach (&entry : state.dedicated_staging.entries) entry.timeline_value = ulong::max;

    char[16] data;
    uint before = capture.calls;
    gpu::GpuSpan dst = buffer.span.checked_subspan(0, data.len)!!;
    fault staging_full = @catch(gpu::cmd_upload_buffer(&commands, dst, data[..]));
    state.staging_arena.size = staging_size;
    state.dedicated_staging.count = dedicated_count;
    assert(staging_full == gpu::ARENA_FULL, "staging exhaustion keeps ARENA_FULL");
    assert_transfer_rejection(
        capture:             &capture,
        before:              before,
        returned:            staging_full,
        operation:           "cmd_upload_buffer",
        rejected_field:      "data",
        invariant:           "staging capacity must be available for the upload",
        resource_kind:       gpu::DebugResourceKind.ALLOCATION,
        has_public_identity: true,
        resource_index:      buffer.span.index,
        resource_generation: buffer.span.generation,
        resource_name:       "exhaustion_buffer",
    );
    gpu::TextureBarrier to_dst = {
        .texture       = texture,
        .before_stage  = gpu::Stage.HOST,
        .after_stage   = gpu::Stage.TRANSFER,
        .before_hazard = gpu::Hazard.HOST_WRITE,
        .after_hazard  = gpu::Hazard.TRANSFER_WRITE,
        .old_layout    = gpu::TextureLayout.UNDEFINED,
        .new_layout    = gpu::TextureLayout.TRANSFER_DST,
    };
    gpu::cmd_texture_barrier(&commands, &to_dst)!!;
    state.staging_arena.size = 1;
    state.dedicated_staging.count = gpu::vk::DEDICATED_RETIRE_CAP;
    foreach (&entry : state.dedicated_staging.entries) entry.timeline_value = ulong::max;
    gpu::TextureUploadDesc upload = { .texture = texture };
    before = capture.calls;
    fault texture_staging_full = @catch(gpu::cmd_upload_texture(&commands, &upload, data[..]));
    state.staging_arena.size = staging_size;
    state.dedicated_staging.count = dedicated_count;
    assert(texture_staging_full == gpu::ARENA_FULL, "texture staging exhaustion keeps ARENA_FULL");
    assert_transfer_rejection(
        capture:             &capture,
        before:              before,
        returned:            texture_staging_full,
        operation:           "cmd_upload_texture",
        rejected_field:      "data",
        invariant:           "staging capacity must be available for the upload",
        resource_kind:       gpu::DebugResourceKind.TEXTURE,
        has_public_identity: true,
        resource_index:      texture.index,
        resource_generation: texture.generation,
        resource_name:       "exhaustion_texture",
    );

    state.staging_arena.size = 1;
    state.dedicated_staging.count = gpu::vk::DEDICATED_RETIRE_CAP;
    foreach (&entry : state.dedicated_staging.entries) entry.timeline_value = ulong::max;
    before = capture.calls;
    fault helper_buffer_full = @catch(gpu::upload_buffer_data(
        device:      &device,
        dst:         buffer.span.checked_subspan(0, data.len)!!,
        data:        data[..],
        next_stage:  gpu::Stage.TRANSFER,
        next_hazard: gpu::Hazard.TRANSFER_READ,
    ));
    state.staging_arena.size = staging_size;
    state.dedicated_staging.count = dedicated_count;
    assert(helper_buffer_full == gpu::ARENA_FULL, "blocking buffer upload exhaustion keeps ARENA_FULL");
    assert_transfer_rejection(
        capture:             &capture,
        before:              before,
        returned:            helper_buffer_full,
        operation:           "upload_buffer_data",
        rejected_field:      "data",
        invariant:           "staging capacity must be available for the upload",
        resource_kind:       gpu::DebugResourceKind.ALLOCATION,
        has_public_identity: true,
        resource_index:      buffer.span.index,
        resource_generation: buffer.span.generation,
        resource_name:       "exhaustion_buffer",
    );

    state.staging_arena.size = 1;
    state.dedicated_staging.count = gpu::vk::DEDICATED_RETIRE_CAP;
    foreach (&entry : state.dedicated_staging.entries) entry.timeline_value = ulong::max;
    before = capture.calls;
    fault helper_texture_full = @catch(gpu::upload_texture_data(
        device:      &device,
        dst:         texture,
        mip:         0,
        data:        data[..],
        next_stage:  gpu::Stage.TRANSFER,
        next_hazard: gpu::Hazard.TRANSFER_READ,
        next_layout: gpu::TextureLayout.TRANSFER_DST,
    ));
    state.staging_arena.size = staging_size;
    state.dedicated_staging.count = dedicated_count;
    assert(helper_texture_full == gpu::ARENA_FULL, "blocking texture upload exhaustion keeps ARENA_FULL");
    assert_transfer_rejection(
        capture:             &capture,
        before:              before,
        returned:            helper_texture_full,
        operation:           "upload_texture_data",
        rejected_field:      "data",
        invariant:           "staging capacity must be available for the upload",
        resource_kind:       gpu::DebugResourceKind.TEXTURE,
        has_public_identity: true,
        resource_index:      texture.index,
        resource_generation: texture.generation,
        resource_name:       "exhaustion_texture",
    );

    usz blocking_readback_size = state.readback_arena.size;
    dedicated_count = state.dedicated_staging.count;
    state.readback_arena.size = 1;
    state.dedicated_staging.count = gpu::vk::DEDICATED_RETIRE_CAP;
    foreach (&entry : state.dedicated_staging.entries) entry.timeline_value = ulong::max;
    char[16] out_data;
    before = capture.calls;
    fault readback_full = @catch(gpu::readback_buffer_data(
        device:      &device,
        src:         buffer.span.checked_subspan(0, out_data.len)!!,
        out_data:    out_data[..],
        from_stage:  gpu::Stage.TRANSFER,
        from_hazard: gpu::Hazard.TRANSFER_READ,
    ));
    state.readback_arena.size = blocking_readback_size;
    state.dedicated_staging.count = dedicated_count;
    assert(readback_full == gpu::ARENA_FULL, "blocking readback exhaustion keeps ARENA_FULL");
    assert_transfer_rejection(
        capture:             &capture,
        before:              before,
        returned:            readback_full,
        operation:           "readback_buffer_data",
        rejected_field:      "out_data",
        invariant:           "readback capacity must be available for the destination",
        resource_kind:       gpu::DebugResourceKind.ALLOCATION,
        has_public_identity: true,
        resource_index:      buffer.span.index,
        resource_generation: buffer.span.generation,
        resource_name:       "exhaustion_buffer",
    );

}

fn void helper_recording_fault_does_not_poison_frame_slot() @test {
    TransferDiagnosticCapture capture;
    gpu::Device device = make_transfer_test_device(
        staging_size: 64,
        capture:      &capture,
    );
    defer destroy_test_device(&device)!!;
    gpu::vk::VkDeviceState* state =
        (gpu::vk::VkDeviceState*)gpu::device_backend_state(device);

    OwnedSpan buffer = allocate_owned_span(
        device:       &device,
        size:         128,
        memory_class: gpu::MemoryClass.GPU_PRIVATE,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "helper_fault_buffer",
    )!!;
    defer free_owned_span(&device, &buffer)!!;

    usz staging_size = state.staging_arena.size;
    uint dedicated_count = state.dedicated_staging.count;
    state.staging_arena.size = 1;
    state.dedicated_staging.count = gpu::vk::DEDICATED_RETIRE_CAP;
    foreach (&entry : state.dedicated_staging.entries) {
        entry.timeline_value = ulong::max;
    }

    char[16] data;
    fault helper_fault = @catch(gpu::upload_buffer_data(
        device:      &device,
        dst:         buffer.span.checked_subspan(0, data.len)!!,
        data:        data[..],
        next_stage:  gpu::Stage.TRANSFER,
        next_hazard: gpu::Hazard.TRANSFER_READ,
    ));
    state.staging_arena.size = staging_size;
    state.dedicated_staging.count = dedicated_count;
    assert(helper_fault == gpu::ARENA_FULL,
        "helper recording fault reaches the public caller");

    for (uint i = 0; i < 3; i++) {
        gpu::FrameToken frame = gpu::begin_frame(&device)!!;
        gpu::end_frame(&frame)!!;
    }
}

fn void oversized_upload_uses_dedicated_staging() @test {
    // Tiny arena: every upload larger than 1 KiB falls back to a dedicated buffer.
    TransferDiagnosticCapture capture;
    gpu::Device device = make_transfer_test_device(
        staging_size:  1024,
        readback_size: 1024,
        capture:       &capture,
    );
    defer destroy_test_device(&device)!!;
    gpu::vk::VkDeviceState* state = (gpu::vk::VkDeviceState*)gpu::device_backend_state(device);
    ulong fallbacks = gpu::vk::dedicated_fallback_count(state);
    uint diagnostic_count = capture.calls;

    const usz BIG = 256 * 1024;
    OwnedSpan buffer = allocate_owned_span(
        device:       &device,
        size:         BIG,
        memory_class: gpu::MemoryClass.GPU_PRIVATE,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "oversized_transfer_buffer",
    )!!;
    defer free_owned_span(&device, &buffer)!!;
    gpu::GpuSpan buffer_span = buffer.span;

    @pool() {
        char[] data = mem::talloc_array(char, (sz)BIG);
        for (usz i = 0; i < BIG; i++) data[i] = (char)(i % 251);
        gpu::upload_buffer_data(
            device:      &device,
            dst:         buffer_span,
            data:        data,
            next_stage:  gpu::Stage.TRANSFER,
            next_hazard: gpu::Hazard.TRANSFER_READ,
        )!!;

        char[] out = mem::talloc_array(char, (sz)BIG);
        gpu::readback_buffer_data(
            device:      &device,
            src:         buffer_span,
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
    defer destroy_test_device(&device)!!;

    OwnedSpan buffer = allocate_owned_span(
        device:       &device,
        size:         1024,
        memory_class: gpu::MemoryClass.GPU_PRIVATE,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "staging_ring_buffer",
    )!!;
    defer free_owned_span(&device, &buffer)!!;
    gpu::GpuSpan buffer_span = buffer.span;

    char[1024] data;
    for (uint frame = 0; frame < 12; frame++) {
        for (uint i = 0; i < 1024; i++) data[i] = (char)(frame + i);
        gpu::FrameToken frame_token = gpu::begin_frame(&device)!!;
        gpu::CommandList cmd = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!)!!;
        defer discard_test_commands(&cmd)!!;
        gpu::cmd_upload_buffer(&cmd, buffer_span, data[..])!!;
        gpu::BufferBarrier next_use = {
            .span = buffer_span,
            .before_stage = gpu::Stage.TRANSFER, .after_stage = gpu::Stage.COMPUTE_SHADER,
            .before_hazard = gpu::Hazard.TRANSFER_WRITE, .after_hazard = gpu::Hazard.SHADER_READ,
        };
        gpu::cmd_buffer_barrier(&cmd, &next_use)!!;
        gpu::ExecutableCommandList cmd_executable = gpu::end_commands(&cmd)!!;
        defer discard_test_executable(&cmd_executable)!!;
        gpu::ExecutableCommandList[1] lists = { cmd_executable };
        gpu::SubmitDesc submit = { .command_lists = lists[..] };
        submit_test_queue(&device, gpu::QueueKind.GRAPHICS, &submit)!!;
        gpu::end_frame(&frame_token)!!;
    }
    wait_test_queue(&device, gpu::QueueKind.GRAPHICS)!!;
}

fn void out_of_frame_uploads_do_not_alias_staging() @test {
    // Two cmd-level uploads recorded outside any frame into one list: their
    // staging ranges must not alias (ranges hold until a timeline advance).
    gpu::Device device = make_transfer_test_device(staging_size: 4096);
    defer destroy_test_device(&device)!!;

    OwnedSpan first = allocate_owned_span(
        device:       &device,
        size:         1024,
        memory_class: gpu::MemoryClass.GPU_PRIVATE,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "out_of_frame_first",
    )!!;
    defer free_owned_span(&device, &first)!!;
    OwnedSpan second = allocate_owned_span(
        device:       &device,
        size:         1024,
        memory_class: gpu::MemoryClass.GPU_PRIVATE,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "out_of_frame_second",
    )!!;
    defer free_owned_span(&device, &second)!!;
    gpu::GpuSpan first_span = first.span;
    gpu::GpuSpan second_span = second.span;

    char[1024] a;
    char[1024] b;
    for (uint i = 0; i < 1024; i++) {
        a[i] = (char)11;
        b[i] = (char)77;
    }

    gpu::CommandList cmd = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!)!!;
    defer discard_test_commands(&cmd)!!;
    gpu::cmd_upload_buffer(&cmd, first_span, a[..])!!;
    gpu::cmd_upload_buffer(&cmd, second_span, b[..])!!;
    gpu::BufferBarrier next_use = {
        .span = first_span,
        .before_stage = gpu::Stage.TRANSFER, .after_stage = gpu::Stage.TRANSFER,
        .before_hazard = gpu::Hazard.TRANSFER_WRITE, .after_hazard = gpu::Hazard.TRANSFER_READ,
    };
    gpu::cmd_buffer_barrier(&cmd, &next_use)!!;
    gpu::ExecutableCommandList cmd_executable = gpu::end_commands(&cmd)!!;
    defer discard_test_executable(&cmd_executable)!!;
    gpu::ExecutableCommandList[1] lists = { cmd_executable };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    submit_test_queue(&device, gpu::QueueKind.GRAPHICS, &submit)!!;
    wait_test_queue(&device, gpu::QueueKind.GRAPHICS)!!;

    char[1024] out;
    gpu::readback_buffer_data(
        device:      &device,
        src:         first_span,
        out_data:    out[..],
        from_stage:  gpu::Stage.TRANSFER,
        from_hazard: gpu::Hazard.TRANSFER_READ,
    )!!;
    assert(out[0] == (char)11 && out[1023] == (char)11, "first upload's staging was not overwritten by the second");
}

fn void texture_upload_helper_sampled() @test {
    gpu::Device device = make_transfer_test_device();
    defer destroy_test_device(&device)!!;

    gpu::TextureDesc tex_desc = {
        .dimension = gpu::TextureDimension.TEX_2D,
        .access    = { .graphics, .compute, .transfer },
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
    OwnedSpan sample_out = allocate_owned_span(
        device:       &device,
        size:         UR_TEX * UR_TEX * 16,
        memory_class: gpu::MemoryClass.CPU_READ,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "sample_output",
    )!!;
    defer free_owned_span(&device, &sample_out)!!;

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
    SampleRoot* root = (SampleRoot*)test_span_mapping(&device, root_span).ptr;
    root.output_gpu    = gpu::get_span_address(&device, sample_out.span)!!;
    root.texture_index = tex_index;
    root.sampler_index = smp_index;
    root.width         = UR_TEX;
    root.height        = UR_TEX;

    gpu::CommandList cmd = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.COMPUTE)!!)!!;
    defer discard_test_commands(&cmd)!!;
    gpu::cmd_dispatch(
        commands: &cmd,
        pipeline: pipeline,
        root:     test_span_address(&device, root_span),
        groups:   { 1, 1, 1 },
    )!!;
    gpu::BufferBarrier out_bar = {
        .span = sample_out.span,
        .before_stage = gpu::Stage.COMPUTE_SHADER, .after_stage = gpu::Stage.HOST,
        .before_hazard = gpu::Hazard.SHADER_WRITE, .after_hazard = gpu::Hazard.HOST_READ,
    };
    gpu::cmd_buffer_barrier(&cmd, &out_bar)!!;
    gpu::ExecutableCommandList cmd_executable = gpu::end_commands(&cmd)!!;
    defer discard_test_executable(&cmd_executable)!!;
    gpu::ExecutableCommandList[1] lists = { cmd_executable };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    submit_test_queue(&device, gpu::QueueKind.COMPUTE, &submit)!!;
    wait_test_queue(&device, gpu::QueueKind.COMPUTE)!!;

    gpu::invalidate_mapped_span(&device, sample_out.span)!!;
    float* sampled = (float*)test_span_mapping(&device, sample_out.span).ptr;
    assert(quantize_unorm8(sampled[1]) == 90, "sampled green matches helper upload");
    assert(quantize_unorm8(sampled[2]) == 160, "sampled blue matches helper upload");
    gpu::end_frame(&frame_token)!!;
}

fn void texture_readback_helper_restores_layout() @test {
    gpu::Device device = make_transfer_test_device();
    defer destroy_test_device(&device)!!;

    gpu::TextureDesc tex_desc = {
        .dimension = gpu::TextureDimension.TEX_2D,
        .access    = { .graphics, .compute, .transfer },
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
    defer destroy_test_device(&device)!!;
    gpu::vk::VkDeviceState* state = (gpu::vk::VkDeviceState*)gpu::device_backend_state(device);

    gpu::TextureDesc tex_desc = {
        .dimension = gpu::TextureDimension.TEX_2D,
        .access    = { .graphics, .compute, .transfer },
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

    OwnedSpan buffer = allocate_owned_span(
        device:       &device,
        size:         64,
        memory_class: gpu::MemoryClass.GPU_PRIVATE,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "helper_upload_buffer",
    )!!;
    defer free_owned_span(&device, &buffer)!!;

    char[128] too_big;
    char[] empty_data = {};
    before = capture.calls;
    helper_before = state.helper_counter;
    fault empty_fault = @catch(gpu::upload_buffer_data(
        device:      &device,
        dst:         buffer.span,
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
        invariant:           "data length must equal the destination span size",
        resource_kind:       gpu::DebugResourceKind.ALLOCATION,
        has_public_identity: true,
        resource_index:      buffer.span.index,
        resource_generation: buffer.span.generation,
        resource_name:       "helper_upload_buffer",
    );
    assert(state.helper_counter == helper_before, "empty upload reserves no helper value");

    before = capture.calls;
    helper_before = state.helper_counter;
    fault bounds_fault = @catch(gpu::upload_buffer_data(
        device:      &device,
        dst:         buffer.span.unchecked_subspan(0, too_big.len),
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
        rejected_field:      "dst",
        invariant:           "span range must fit its live owner",
        resource_kind:       gpu::DebugResourceKind.ALLOCATION,
        has_public_identity: true,
        resource_index:      buffer.span.index,
        resource_generation: buffer.span.generation,
        resource_name:       "helper_upload_buffer",
    );
    assert(state.helper_counter == helper_before, "invalid buffer range reserves no helper value");

    gpu::TextureDesc no_transfer_desc = {
        .dimension  = gpu::TextureDimension.TEX_2D,
        .access    = { .graphics, .compute, .transfer },
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

    // cmd-level upload into a texture that is not TRANSFER_DST still faults.
    gpu::FrameToken frame_token = gpu::begin_frame(&device)!!;
    gpu::CommandList cmd = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!)!!;
    defer discard_test_commands(&cmd)!!;
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
    gpu::ExecutableCommandList cmd_executable = gpu::end_commands(&cmd)!!;
    defer discard_test_executable(&cmd_executable)!!;
    gpu::end_frame(&frame_token)!!;
}

fn void transfer_access_rejects_before_helper_or_arena_mutation() @test {
    TransferDiagnosticCapture capture;
    gpu::Device device = make_transfer_test_device(capture: &capture);
    defer destroy_test_device(&device)!!;
    gpu::vk::VkDeviceState* state =
        (gpu::vk::VkDeviceState*)gpu::device_backend_state(device);

    OwnedSpan upload = allocate_owned_span(
        device:       &device,
        size:         16,
        memory_class: gpu::MemoryClass.GPU_PRIVATE,
        access:       { .compute },
        debug_name:   "compute_upload_only",
    )!!;
    defer free_owned_span(&device, &upload)!!;

    OwnedSpan readback = allocate_owned_span(
        device:       &device,
        size:         16,
        memory_class: gpu::MemoryClass.GPU_PRIVATE,
        access:       { .compute },
        debug_name:   "compute_readback_only",
    )!!;
    defer free_owned_span(&device, &readback)!!;

    char[8] data;
    ulong helper_before = state.helper_counter;
    uint staging_before = state.staging_arena.retire_count;
    ulong fallback_before = gpu::vk::dedicated_fallback_count(state);
    uint before = capture.calls;
    fault blocking_upload = @catch(gpu::upload_buffer_data(
        device:      &device,
        dst:         upload.span.checked_subspan(0, data.len)!!,
        data:        data[..],
        next_stage:  gpu::Stage.COMPUTE_SHADER,
        next_hazard: gpu::Hazard.SHADER_READ,
    ));
    assert(blocking_upload == gpu::INVALID_ARGUMENT,
        "blocking upload rejects a resource outside the helper queue role");
    assert_transfer_rejection(
        capture:             &capture,
        before:              before,
        returned:            blocking_upload,
        operation:           "upload_buffer_data",
        rejected_field:      "GpuSpan",
        invariant:           "access must include the operation queue role",
        resource_kind:       gpu::DebugResourceKind.ALLOCATION,
        has_public_identity: true,
        resource_index:      upload.span.index,
        resource_generation: upload.span.generation,
        resource_name:       "compute_upload_only",
    );
    assert(state.helper_counter == helper_before
        && state.staging_arena.retire_count == staging_before
        && gpu::vk::dedicated_fallback_count(state) == fallback_before,
        "blocking upload access rejection consumes no helper or staging state");

    char[8] out_data;
    helper_before = state.helper_counter;
    uint readback_before = state.readback_arena.retire_count;
    before = capture.calls;
    fault blocking_readback = @catch(gpu::readback_buffer_data(
        device:      &device,
        src:         readback.span.checked_subspan(0, out_data.len)!!,
        out_data:    out_data[..],
        from_stage:  gpu::Stage.COMPUTE_SHADER,
        from_hazard: gpu::Hazard.SHADER_WRITE,
    ));
    assert(blocking_readback == gpu::INVALID_ARGUMENT,
        "blocking readback rejects a resource outside the helper queue role");
    assert_transfer_rejection(
        capture:             &capture,
        before:              before,
        returned:            blocking_readback,
        operation:           "readback_buffer_data",
        rejected_field:      "GpuSpan",
        invariant:           "access must include the operation queue role",
        resource_kind:       gpu::DebugResourceKind.ALLOCATION,
        has_public_identity: true,
        resource_index:      readback.span.index,
        resource_generation: readback.span.generation,
        resource_name:       "compute_readback_only",
    );
    assert(state.helper_counter == helper_before
        && state.readback_arena.retire_count == readback_before,
        "blocking readback access rejection consumes no helper or arena state");

    gpu::FrameToken frame = gpu::begin_frame(&device)!!;
    gpu::CommandList commands = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!)!!;
    defer discard_test_commands(&commands)!!;
    gpu::vk::CommandRecord* record = state.commands.get(commands.handle)!!;
    gpu::Queue recorded_queue = record.queue;
    record.queue.roles = { .graphics };
    staging_before = state.staging_arena.retire_count;
    before = capture.calls;
    fault command_upload = @catch(gpu::cmd_upload_buffer(
        commands: &commands,
        dst:      upload.span.checked_subspan(0, data.len)!!,
        data:     data[..],
    ));
    assert(command_upload == gpu::INVALID_ARGUMENT,
        "recorded upload rejects a resource outside the recording role");
    assert_transfer_rejection(
        capture:             &capture,
        before:              before,
        returned:            command_upload,
        operation:           "cmd_upload_buffer",
        rejected_field:      "GpuSpan",
        invariant:           "access must include the operation queue role",
        resource_kind:       gpu::DebugResourceKind.ALLOCATION,
        has_public_identity: true,
        resource_index:      upload.span.index,
        resource_generation: upload.span.generation,
        resource_name:       "compute_upload_only",
    );
    assert(state.staging_arena.retire_count == staging_before,
        "recorded upload access rejection consumes no staging range");

    record.queue = recorded_queue;
    gpu::ExecutableCommandList commands_executable = gpu::end_commands(&commands)!!;

    defer discard_test_executable(&commands_executable)!!;
    gpu::end_frame(&frame)!!;
}

fn void transfer_callback_presence_preserves_upload_fault() @test {
    TransferDiagnosticCapture capture;
    gpu::Device callback_device = make_transfer_test_device(capture: &capture);
    OwnedSpan callback_buffer = allocate_owned_span(
        device:       &callback_device,
        size:         16,
        memory_class: gpu::MemoryClass.GPU_PRIVATE,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "callback_upload_buffer",
    )!!;
    char[32] payload;
    fault callback_fault = @catch(gpu::upload_buffer_data(
        device:      &callback_device,
        dst:         callback_buffer.span.unchecked_subspan(0, payload.len),
        data:        payload[..],
        next_stage:  gpu::Stage.TRANSFER,
        next_hazard: gpu::Hazard.TRANSFER_READ,
    ));
    assert(capture.calls == 1, "callback upload rejection emits exactly once");
    free_owned_span(&callback_device, &callback_buffer)!!;
    gpu::destroy_device(&callback_device)!!;

    gpu::Device plain_device = make_transfer_test_device();
    OwnedSpan plain_buffer = allocate_owned_span(
        device:       &plain_device,
        size:         16,
        memory_class: gpu::MemoryClass.GPU_PRIVATE,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "plain_upload_buffer",
    )!!;
    fault plain_fault = @catch(gpu::upload_buffer_data(
        device:      &plain_device,
        dst:         plain_buffer.span.unchecked_subspan(0, payload.len),
        data:        payload[..],
        next_stage:  gpu::Stage.TRANSFER,
        next_hazard: gpu::Hazard.TRANSFER_READ,
    ));
    free_owned_span(&plain_device, &plain_buffer)!!;
    gpu::destroy_device(&plain_device)!!;
    assert(plain_fault == callback_fault, "callback presence does not change upload fault");
}
