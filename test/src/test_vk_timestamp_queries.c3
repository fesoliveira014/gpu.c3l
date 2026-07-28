module gpu_test;

import gpu::internal @public;
import gpu @public;

const char[*] TIMESTAMP_COMPUTE_SPIRV =
    $embed("shaders/root_pointer.comp.spv");
const uint TIMESTAMP_DISPATCH_GROUPS = 8192;
const uint TIMESTAMP_QUERY_COUNT = 2;
const usz TIMESTAMP_RESULT_SIZE =
    TIMESTAMP_QUERY_COUNT * ulong::size;

struct TimestampValidationCapture {
    uint errors;
}

fn void capture_timestamp_validation_error(
    gpu::DebugMessage* message,
    void* user_data,
) {
    if (message.category.backend
        && message.category.validation
        && message.severity == gpu::DebugMessageSeverity.ERROR) {
        TimestampValidationCapture* capture =
            (TimestampValidationCapture*)user_data;
        capture.errors++;
    }
}

fn void timestamp_pool_creation_rejects_unsupported_device() @test {
    gpu::RuntimeDesc desc = {
        .contract_validation      = gpu::ContractValidation.FULL,
        .enable_vulkan_validation = true,
        .application_name         = "gpu_timestamp_unsupported_test",
    };
    gpu::Device device = create_test_device(&desc)!!;
    defer destroy_test_device(&device)!!;
    gpu::internal::DeviceSlot* device_slot =
        &gpu::internal::device_slots[device.index];
    gpu::TimestampCaps saved_caps = device_slot.data.caps.timestamps;
    defer device_slot.data.caps.timestamps = saved_caps;
    device_slot.data.caps.timestamps = {};

    gpu::internal::vk::VkDeviceState* state = device_slot.data.state;
    uint next_new = state.timestamp_pools.next_new;
    gpu::TimestampPoolDesc pool_desc = {
        .capacity   = TIMESTAMP_QUERY_COUNT,
        .debug_name = "unsupported_timestamp_pool",
    };

    assert(
        @catch(gpu::create_timestamp_pool(&device, &pool_desc))
            == gpu::UNSUPPORTED_FEATURE,
        "a device without timestamp-capable roles rejects pool creation"
    );
    assert(
        state.timestamp_pools.next_new == next_new,
        "unsupported pool creation publishes no backend slot"
    );
}

fn void run_timestamp_query_measurement(
    TimestampValidationCapture* validation_capture,
) {
    gpu::RuntimeDesc desc = {
        .contract_validation      = gpu::ContractValidation.FULL,
        .enable_vulkan_validation = true,
        .application_name         = "gpu_timestamp_query_test",
        .debug_callback           = &capture_timestamp_validation_error,
        .debug_user_data          = validation_capture,
    };
    gpu::Device device = create_test_device(&desc)!!;
    defer destroy_test_device(&device)!!;
    gpu::DeviceCaps caps = gpu::get_device_caps(&device)!!;
    if (!caps.timestamps.queues.compute) return;

    gpu::TimestampPoolDesc pool_desc = {
        .capacity   = TIMESTAMP_QUERY_COUNT,
        .debug_name = "compute_timestamp_pool",
    };
    gpu::TimestampPoolHandle pool =
        gpu::create_timestamp_pool(&device, &pool_desc)!!;
    defer gpu::destroy_timestamp_pool(&device, pool)!!;

    OwnedSpan readback = allocate_owned_span(
        device:       &device,
        size:         TIMESTAMP_RESULT_SIZE,
        memory_class: gpu::MemoryClass.CPU_READ,
        access:       { .compute },
        debug_name:   "timestamp_readback",
    )!!;
    defer free_owned_span(&device, &readback)!!;

    gpu::ShaderDesc shader = {
        .spirv       = TIMESTAMP_COMPUTE_SPIRV[..],
        .entry_point = "main",
    };
    gpu::ComputePipelineDesc pipeline_desc = { .shader = shader };
    gpu::PipelineHandle pipeline =
        gpu::create_compute_pipeline(&device, &pipeline_desc)!!;
    defer gpu::destroy_pipeline(&device, pipeline)!!;

    gpu::Queue queue =
        gpu::get_queue(&device, gpu::QueueKind.COMPUTE)!!;
    gpu::CommandAllocator allocator =
        gpu::create_command_allocator(&device, queue)!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::CommandList commands = gpu::begin_commands(&allocator)!!;
    defer discard_test_commands(&commands)!!;

    gpu::cmd_reset_timestamps(
        commands: &commands,
        pool:     pool,
        first:    0,
        count:    TIMESTAMP_QUERY_COUNT,
    )!!;
    gpu::cmd_write_timestamp(
        commands: &commands,
        pool:     pool,
        index:    0,
        stage:    { .compute },
    )!!;
    bind_test_compute_pipeline(&commands, pipeline);
    gpu::cmd_dispatch(
        commands: &commands,
        root:     (gpu::GpuAddress)0,
        groups:   { TIMESTAMP_DISPATCH_GROUPS, 1, 1 },
    )!!;
    gpu::cmd_write_timestamp(
        commands: &commands,
        pool:     pool,
        index:    1,
        stage:    { .compute },
    )!!;
    gpu::cmd_resolve_timestamps(
        commands: &commands,
        pool:     pool,
        first:    0,
        count:    TIMESTAMP_QUERY_COUNT,
        dst:      readback.span,
    )!!;
    gpu::Barrier host_read = {
        .before = { .transfer },
        .after  = { .host },
    };
    gpu::cmd_barrier(&commands, &host_read)!!;

    gpu::ExecutableCommandList executable =
        gpu::end_commands(&commands)!!;
    defer discard_test_executable(&executable)!!;
    gpu::ExecutableCommandList[1] lists = { executable };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::CompletionPoint completion =
        gpu::submit(queue, &submit)!!;
    gpu::wait_completion(completion, ulong::max)!!;
    gpu::invalidate_mapped_span(&device, readback.span)!!;

    ulong* resolved =
        (ulong*)test_span_mapping(&device, readback.span).ptr;
    double resolved_delta = gpu::timestamp_delta_ns(
        caps:  &caps.timestamps,
        queue: gpu::QueueKind.COMPUTE,
        begin: resolved[0],
        end:   resolved[1],
    )!!;
    assert(resolved_delta > 0.0,
        "resolved timestamps measure positive compute duration");

    ulong[TIMESTAMP_QUERY_COUNT] direct;
    gpu::read_timestamps(
        device: &device,
        pool:   pool,
        first:  0,
        count:  TIMESTAMP_QUERY_COUNT,
        out:    direct[..],
    )!!;
    assert(
        direct[0] == resolved[0] && direct[1] == resolved[1],
        "direct host reads match resolved query values"
    );
    double direct_delta = gpu::timestamp_delta_ns(
        caps:  &caps.timestamps,
        queue: gpu::QueueKind.COMPUTE,
        begin: direct[0],
        end:   direct[1],
    )!!;
    assert(direct_delta > 0.0,
        "direct timestamps measure positive compute duration");
}

fn void timestamp_queries_measure_compute_work() @test {
    TimestampValidationCapture validation_capture = {};
    run_timestamp_query_measurement(&validation_capture);
    assert(validation_capture.errors == 0,
        "timestamp workflow emits no Vulkan validation errors");
}
