module gpu_test;

import gpu::internal @public;
import gpu @public;
import gpu::internal::vk @public;
import vk;
import std::io;

const char[*] GENERATED_SCRATCH_SPIRV =
    $embed("shaders/root_pointer.comp.spv");

struct ZeroGeneratedExecutionCapture {
    uint              calls;
    vk::DeviceAddress preprocess_address;
    vk::DeviceSize    preprocess_size;
}

uint zero_generated_requirement_calls;
ZeroGeneratedExecutionCapture* zero_generated_execution_capture;

fn void zero_generated_requirements(
    vk::Device device,
    vk::GeneratedCommandsMemoryRequirementsInfoEXT* info,
    vk::MemoryRequirements2* requirements,
) {
    (void)device;
    (void)info;
    zero_generated_requirement_calls++;
    requirements.memory_requirements = {};
}

fn void capture_zero_generated_execute(
    vk::CommandBuffer command_buffer,
    uint is_preprocessed,
    vk::GeneratedCommandsInfoEXT* info,
) {
    (void)command_buffer;
    (void)is_preprocessed;
    ZeroGeneratedExecutionCapture* capture = zero_generated_execution_capture;
    capture.calls++;
    capture.preprocess_address = info.preprocess_address;
    capture.preprocess_size = info.preprocess_size;
}

fn void generated_scratch_reservation_is_driver_compatible_and_bounded() @test {
    gpu::RuntimeDesc runtime_desc = {
        .application_name = "generated_scratch_reservation_test",
    };
    gpu::Device device = create_test_device(&runtime_desc)!!;
    defer destroy_test_device(&device)!!;
    if (!gpu::get_device_caps(&device)!!.generated_work) {
        io::printfn(
            "generated-scratch-reservation: NOT EXERCISED (generated work unavailable)",
        );
        return;
    }

    gpu::ShaderDesc shader_desc = {
        .spirv       = GENERATED_SCRATCH_SPIRV[..],
        .entry_point = "main",
    };
    gpu::ShaderDesc shader = shader_desc;
    gpu::ComputePipelineDesc pipeline_desc = {
        .shader = shader,
    };
    gpu::PipelineHandle pipeline = gpu::create_compute_pipeline(
        &device,
        &pipeline_desc,
    )!!;
    defer gpu::destroy_pipeline(&device, pipeline)!!;

    gpu::Queue queue = gpu::get_queue(
        &device,
        gpu::QueueKind.COMPUTE,
    )!!;
    gpu::GeneratedWorkReservationDesc reservation = {
        .pipeline              = pipeline,
        .kind                  = gpu::GeneratedWorkKind.DISPATCH,
        .max_commands_per_list = 8,
        .concurrent_lists      = 1,
    };
    vk::VkDeviceState* state = gpu::internal::device_slots[device.index].data.state;
    vk::MemoryRequirements requirements = vk::generated_scratch_requirements(
        state,
        &reservation,
    )!!;
    gpu::CommandAllocatorDesc allocator_desc = {
        .command_buffer_capacity = 1,
    };
    gpu::CommandAllocator allocator = gpu::create_command_allocator(
        &device,
        queue,
        &allocator_desc,
    )!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    vk::CommandAllocatorSlot* cold_slot = state.command_allocators.get(
        allocator.handle,
    )!!;
    assert(cold_slot.reservations.len == 0
            && cold_slot.preprocess_index_storage.len == 0,
        "an allocator without a reservation owns no generated-work storage");
    gpu::reserve_generated_work(&allocator, &reservation)!!;
    assert(@catch(gpu::destroy_pipeline(&device, pipeline))
        == gpu::RESOURCE_IN_USE,
        "a live reservation retains its exact pipeline key");

    gpu::CommandList commands = gpu::begin_commands(&allocator)!!;
    defer discard_test_commands(&commands)!!;
    assert(@catch(gpu::reserve_generated_work(&allocator, &reservation))
        == gpu::RESOURCE_IN_USE,
        "reservation replacement requires a quiescent allocator");

    vk::CommandRecord* record = vk::resolve_command(&commands)!!;
    vk::CommandAllocatorSlot* slot = state.command_allocators.get(
        record.allocator,
    )!!;
    assert(slot.reservations.len
            == (usz)vk::GENERATED_PREPROCESS_PER_LIST
        && slot.reservations.len == slot.preprocess_index_storage.len
        && record.scratch.generated_preprocess_indices.len
            == (usz)vk::GENERATED_PREPROCESS_PER_LIST,
        "the first reservation publishes the fixed backend-owned tables");
    vk::GeneratedPreprocessBuffer* reserved =
        &slot.reservations[0];
    assert(reserved.reservation_pipeline == pipeline
            && reserved.reservation_kind
                == gpu::GeneratedWorkKind.DISPATCH
            && reserved.reservation_max_commands == 8,
        "reservation retains its exact pipeline, kind, and count key");
    assert(reserved.size >= requirements.size
            && (requirements.alignment <= 1
                || reserved.address % requirements.alignment == 0),
        "reservation-time allocation satisfies the exact driver requirements");

    assert(@catch(vk::acquire_generated_preprocess_buffer(
        state:        state,
        record:       record,
        pipeline:     pipeline,
        kind:         gpu::GeneratedWorkKind.DISPATCH,
        max_count:    9,
    )) == gpu::GENERATED_SCRATCH_EXHAUSTED,
        "work above the reserved command count faults deterministically");
    vk::GeneratedPreprocessBuffer* acquired =
        vk::acquire_generated_preprocess_buffer(
            state:        state,
            record:       record,
            pipeline:     pipeline,
            kind:         gpu::GeneratedWorkKind.DISPATCH,
            max_count:    8,
        )!!;
    assert(acquired.address == reserved.address,
        "generated recording consumes the exact reserved buffer");
    assert(@catch(vk::acquire_generated_preprocess_buffer(
        state:        state,
        record:       record,
        pipeline:     pipeline,
        kind:         gpu::GeneratedWorkKind.DISPATCH,
        max_count:    1,
    )) == gpu::GENERATED_SCRATCH_EXHAUSTED,
        "concurrent use beyond the reserved list count is deterministic");

    gpu::discard_commands(&commands)!!;
    gpu::CommandList retry = gpu::begin_commands(&allocator)!!;
    vk::CommandRecord* retry_record = vk::resolve_command(&retry)!!;
    vk::GeneratedPreprocessBuffer* reacquired =
        vk::acquire_generated_preprocess_buffer(
            state:        state,
            record:       retry_record,
            pipeline:     pipeline,
            kind:         gpu::GeneratedWorkKind.DISPATCH,
            max_count:    1,
        )!!;
    assert(reacquired.address == reserved.address,
        "discard returns reserved preprocess storage for retry");
    gpu::discard_commands(&retry)!!;
    gpu::release_generated_work(
        &allocator,
        pipeline,
        gpu::GeneratedWorkKind.DISPATCH,
    )!!;
    io::printfn("generated-scratch-reservation: EXERCISED");
}

fn void generated_scratch_replacement_capacity_and_allocator_isolation() @test {
    gpu::RuntimeDesc runtime_desc = {
        .application_name = "generated_scratch_allocator_ownership_test",
    };
    gpu::Device device = create_test_device(&runtime_desc)!!;
    defer destroy_test_device(&device)!!;
    if (!gpu::get_device_caps(&device)!!.generated_work) {
        io::printfn(
            "generated-scratch-ownership: NOT EXERCISED (generated work unavailable)",
        );
        return;
    }

    gpu::ShaderDesc shader_desc = {
        .spirv       = GENERATED_SCRATCH_SPIRV[..],
        .entry_point = "main",
    };
    gpu::ShaderDesc shader = shader_desc;
    gpu::ComputePipelineDesc pipeline_desc = {
        .shader = shader,
    };
    gpu::PipelineHandle pipeline = gpu::create_compute_pipeline(
        &device,
        &pipeline_desc,
    )!!;
    defer gpu::destroy_pipeline(&device, pipeline)!!;
    gpu::PipelineHandle alias_pipeline = gpu::create_compute_pipeline(
        &device,
        &pipeline_desc,
    )!!;
    defer gpu::destroy_pipeline(&device, alias_pipeline)!!;

    gpu::Queue queue = gpu::get_queue(
        &device,
        gpu::QueueKind.COMPUTE,
    )!!;
    gpu::GeneratedWorkReservationDesc reservation = {
        .pipeline              = pipeline,
        .kind                  = gpu::GeneratedWorkKind.DISPATCH,
        .max_commands_per_list = 8,
        .concurrent_lists      = 1,
    };
    gpu::GeneratedWorkReservationDesc alias_reservation = reservation;
    alias_reservation.pipeline = alias_pipeline;
    vk::VkDeviceState* state = gpu::internal::device_slots[device.index].data.state;
    vk::MemoryRequirements requirements = vk::generated_scratch_requirements(
        state,
        &reservation,
    )!!;
    vk::MemoryRequirements alias_requirements =
        vk::generated_scratch_requirements(
            state,
            &alias_reservation,
        )!!;
    assert(requirements.size > 0 && alias_requirements.size > 0,
        "generated work reports a non-empty preprocess requirement");

    gpu::CommandAllocatorDesc replacement_desc = {
        .command_buffer_capacity = 1,
    };
    gpu::CommandAllocator reserved_allocator = gpu::create_command_allocator(
        &device,
        queue,
        &replacement_desc,
    )!!;
    gpu::PipelineHandle stale_pipeline = gpu::create_compute_pipeline(
        &device,
        &pipeline_desc,
    )!!;
    gpu::destroy_pipeline(&device, stale_pipeline)!!;
    gpu::PipelineHandle recycled_pipeline = gpu::create_compute_pipeline(
        &device,
        &pipeline_desc,
    )!!;
    defer gpu::destroy_pipeline(&device, recycled_pipeline)!!;
    assert(stale_pipeline.index == recycled_pipeline.index
            && stale_pipeline.generation != recycled_pipeline.generation,
        "pipeline slots recycle with a new generation");
    assert(@catch(gpu::release_generated_work(
        &reserved_allocator,
        stale_pipeline,
        gpu::GeneratedWorkKind.DISPATCH,
    )) == gpu::INVALID_HANDLE,
        "release rejects a stale recycled pipeline before reservation lookup");
    gpu::reserve_generated_work(&reserved_allocator, &reservation)!!;
    vk::CommandAllocatorSlot* reserved_slot = state.command_allocators.get(
        reserved_allocator.handle,
    )!!;
    vk::GeneratedPreprocessBuffer original = reserved_slot.reservations[0];

    reservation.concurrent_lists = vk::GENERATED_PREPROCESS_PER_LIST + 1;
    assert(@catch(gpu::reserve_generated_work(
        &reserved_allocator,
        &reservation,
    )) == gpu::COMMAND_ALLOCATOR_CAPACITY_EXCEEDED,
        "concurrency above the fixed table returns the capacity fault");
    assert(reserved_slot.reservations[0].buffer == original.buffer
            && reserved_slot.reservations[0].allocation == original.allocation
            && reserved_slot.reservations[0].reservation_max_commands == 8,
        "failed replacement preserves the published reservation exactly");

    reservation.concurrent_lists      = 1;
    reservation.max_commands_per_list = 7;
    gpu::reserve_generated_work(&reserved_allocator, &reservation)!!;
    assert(reserved_slot.reservations[0].reservation_pipeline == pipeline
            && reserved_slot.reservations[0].reservation_kind
                == gpu::GeneratedWorkKind.DISPATCH
            && reserved_slot.reservations[0].reservation_max_commands == 7,
        "successful replacement publishes the new exact reservation");
    reservation.max_commands_per_list = 8;
    gpu::reserve_generated_work(&reserved_allocator, &reservation)!!;
    alias_reservation.concurrent_lists = vk::GENERATED_PREPROCESS_PER_LIST;
    assert(@catch(gpu::reserve_generated_work(
        &reserved_allocator,
        &alias_reservation,
    )) == gpu::COMMAND_ALLOCATOR_CAPACITY_EXCEEDED,
        "a distinct key cannot exceed the reservation slot ceiling");

    gpu::CommandList reserved_commands = gpu::begin_commands(
        &reserved_allocator,
    )!!;
    assert(@catch(gpu::release_generated_work(
        &reserved_allocator,
        pipeline,
        gpu::GeneratedWorkKind.DISPATCH,
    )) == gpu::RESOURCE_IN_USE,
        "release requires a quiescent allocator");
    assert(reserved_slot.reservations[0].reservation_pipeline == pipeline,
        "failed release preserves the reservation");
    gpu::discard_commands(&reserved_commands)!!;

    gpu::CommandAllocatorDesc unreserved_desc = {
        .command_buffer_capacity = 1,
    };
    gpu::CommandAllocator unreserved_allocator = gpu::create_command_allocator(
        &device,
        queue,
        &unreserved_desc,
    )!!;
    gpu::CommandList unreserved_commands = gpu::begin_commands(
        &unreserved_allocator,
    )!!;
    vk::CommandRecord* unreserved_record = vk::resolve_command(
        &unreserved_commands,
    )!!;
    uint unreserved_indices =
        unreserved_record.scratch.generated_preprocess_count;
    bool reserved_in_use = reserved_slot.reservations[0].reservation_in_use;
    assert(@catch(vk::acquire_generated_preprocess_buffer(
        state:        state,
        record:       unreserved_record,
        pipeline:     pipeline,
        kind:         gpu::GeneratedWorkKind.DISPATCH,
        max_count:    1,
    )) == gpu::GENERATED_SCRATCH_EXHAUSTED,
        "an allocator cannot borrow another allocator's reservation");
    assert(unreserved_record.scratch.generated_preprocess_count
            == unreserved_indices
            && reserved_slot.reservations[0].reservation_in_use
                == reserved_in_use
            && reserved_slot.reservations[0].reservation_pipeline == pipeline,
        "allocator-isolation failure leaves both allocators untouched");
    gpu::discard_commands(&unreserved_commands)!!;
    gpu::destroy_command_allocator(&unreserved_allocator)!!;

    gpu::release_generated_work(
        &reserved_allocator,
        pipeline,
        gpu::GeneratedWorkKind.DISPATCH,
    )!!;
    assert(@catch(gpu::release_generated_work(
        &reserved_allocator,
        pipeline,
        gpu::GeneratedWorkKind.DISPATCH,
    )) == gpu::INVALID_RESOURCE_STATE,
        "releasing an absent exact key returns INVALID_RESOURCE_STATE");
    gpu::destroy_command_allocator(&reserved_allocator)!!;
    io::printfn("generated-scratch-ownership: EXERCISED");
}

fn void zero_size_generated_scratch_still_requires_reservation() @test {
    gpu::RuntimeDesc runtime_desc = {
        .application_name = "zero_size_generated_scratch_test",
    };
    gpu::Device device = create_test_device(&runtime_desc)!!;
    defer destroy_test_device(&device)!!;
    if (!gpu::get_device_caps(&device)!!.generated_work) return;

    gpu::ShaderDesc shader_desc = {
        .spirv       = GENERATED_SCRATCH_SPIRV[..],
        .entry_point = "main",
    };
    gpu::ShaderDesc shader = shader_desc;
    gpu::ComputePipelineDesc pipeline_desc = { .shader = shader };
    gpu::PipelineHandle pipeline = gpu::create_compute_pipeline(
        &device,
        &pipeline_desc,
    )!!;
    defer gpu::destroy_pipeline(&device, pipeline)!!;

    gpu::Queue queue = gpu::get_queue(
        &device,
        gpu::QueueKind.COMPUTE,
    )!!;
    gpu::CommandAllocatorDesc allocator_desc = {
        .command_buffer_capacity = 1,
    };
    gpu::CommandAllocator reserved_allocator = gpu::create_command_allocator(
        &device,
        queue,
        &allocator_desc,
    )!!;
    defer gpu::destroy_command_allocator(&reserved_allocator)!!;
    gpu::GeneratedWorkReservationDesc reservation = {
        .pipeline              = pipeline,
        .kind                  = gpu::GeneratedWorkKind.DISPATCH,
        .max_commands_per_list = 8,
        .concurrent_lists      = 1,
    };
    vk::VkDeviceState* state = gpu::internal::device_slots[device.index].data.state;
    vk::PFN_vkGetGeneratedCommandsMemoryRequirementsEXT original =
        state.device_dispatch.generated_work.get_memory_requirements;
    zero_generated_requirement_calls = 0;
    {
        state.device_dispatch.generated_work.get_memory_requirements =
            &zero_generated_requirements;
        defer state.device_dispatch.generated_work.get_memory_requirements =
            original;
        gpu::reserve_generated_work(&reserved_allocator, &reservation)!!;
    }
    assert(zero_generated_requirement_calls == 1,
        "zero-size reservation performs one native requirements query");

    OwnedSpan records = allocate_owned_span(
        device:       &device,
        size:         gpu::GeneratedDispatchRecord::size,
        memory_class: gpu::MemoryClass.GPU_PRIVATE,
        access:       { .compute },
        debug_name:   "zero_size_generated_records",
    )!!;
    defer free_owned_span(&device, &records)!!;
    OwnedSpan count = allocate_owned_span(
        device:       &device,
        size:         4,
        memory_class: gpu::MemoryClass.GPU_PRIVATE,
        access:       { .compute },
        debug_name:   "zero_size_generated_count",
    )!!;
    defer free_owned_span(&device, &count)!!;
    ZeroGeneratedExecutionCapture execution_capture;
    zero_generated_execution_capture = &execution_capture;
    vk::PFN_vkCmdExecuteGeneratedCommandsEXT original_execute =
        state.device_dispatch.generated_work.cmd_execute_generated_commands;
    state.device_dispatch.generated_work.cmd_execute_generated_commands =
        &capture_zero_generated_execute;
    defer {
        state.device_dispatch.generated_work.cmd_execute_generated_commands =
            original_execute;
        zero_generated_execution_capture = null;
    }

    gpu::CommandList reserved_commands = gpu::begin_commands(
        &reserved_allocator,
    )!!;
    bind_test_compute_pipeline(&reserved_commands, pipeline);
    vk::CommandRecord* reserved_record = vk::resolve_command(
        &reserved_commands,
    )!!;
    gpu::cmd_dispatch_generated(
        commands:           &reserved_commands,
        records:            records.span,
        count_span:         count.span,
        max_dispatch_count: 1,
    )!!;
    assert(execution_capture.calls == 1
            && execution_capture.preprocess_address == 0
            && execution_capture.preprocess_size == 0
            && reserved_record.scratch.generated_preprocess_count == 1
            && zero_generated_requirement_calls == 1,
        "zero-size public recording consumes a slot without preprocess storage");
    vk::CommandAllocatorSlot* reserved_slot = state.command_allocators.get(
        reserved_allocator.handle,
    )!!;
    assert(reserved_slot.reservations[0].reservation_in_use
            && reserved_slot.generated_reservations_in_use == 1,
        "zero-size public recording retains the exact reservation");
    gpu::discard_commands(&reserved_commands)!!;
    assert(!reserved_slot.reservations[0].reservation_in_use
            && reserved_slot.generated_reservations_in_use == 0,
        "discard returns the zero-size reservation slot");

    gpu::CommandAllocator unreserved_allocator = gpu::create_command_allocator(
        &device,
        queue,
        &allocator_desc,
    )!!;
    gpu::CommandList unreserved_commands = gpu::begin_commands(
        &unreserved_allocator,
    )!!;
    bind_test_compute_pipeline(&unreserved_commands, pipeline);
    vk::CommandRecord* unreserved_record = vk::resolve_command(
        &unreserved_commands,
    )!!;
    assert(@catch(gpu::cmd_dispatch_generated(
        commands:           &unreserved_commands,
        records:            records.span,
        count_span:         count.span,
        max_dispatch_count: 1,
    )) == gpu::GENERATED_SCRATCH_EXHAUSTED,
        "zero-size generated work cannot bypass explicit reservation");
    assert(execution_capture.calls == 1
            && unreserved_record.scratch.generated_preprocess_count == 0
            && zero_generated_requirement_calls == 1,
        "unreserved zero-size recording faults before native execution");
    gpu::discard_commands(&unreserved_commands)!!;
    gpu::destroy_command_allocator(&unreserved_allocator)!!;
    gpu::release_generated_work(
        &reserved_allocator,
        pipeline,
        gpu::GeneratedWorkKind.DISPATCH,
    )!!;
}
