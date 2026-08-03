module gpu::internal::vk;

import gpu::internal @public;
import gpu @public;
import vk;
import std::atomic::types;

const uint TIMESTAMP_COMMAND_POOL_CAPACITY = 8;
const usz TIMESTAMP_COMMAND_DESTINATION_SIZE = 256;
const vk::QueryPool TIMESTAMP_COMMAND_NATIVE_POOL =
    (vk::QueryPool)(uptr)0x4970;

struct TimestampCommandProbe {
    uint                    reset_calls;
    uint                    write_calls;
    uint                    resolve_calls;
    vk::CommandBuffer       reset_command_buffer;
    vk::CommandBuffer       write_command_buffer;
    vk::CommandBuffer       resolve_command_buffer;
    vk::QueryPool           reset_pool;
    vk::QueryPool           write_pool;
    vk::QueryPool           resolve_pool;
    uint                    reset_first;
    uint                    reset_count;
    uint                    write_index;
    vk::PipelineStageFlags2 write_stage;
    uint                    resolve_first;
    uint                    resolve_count;
    vk::Buffer              resolve_destination;
    vk::DeviceSize          resolve_offset;
    vk::DeviceSize          resolve_stride;
    vk::QueryResultFlags    resolve_flags;
}

struct TimestampCommandFixture {
    gpu::Runtime             runtime;
    gpu::Device              device;
    VkDeviceState*           state;
    gpu::GpuAllocation       allocation;
    gpu::GpuSpan             destination;
    gpu::TimestampPoolHandle pool;
    gpu::CommandAllocator    allocator;
    gpu::CommandList         commands;
    CommandRecord*           record;
}

fn void timestamp_probe_reset(
    void* user_data,
    vk::CommandBuffer command_buffer,
    vk::QueryPool pool,
    uint first,
    uint count,
) {
    TimestampCommandProbe* probe = (TimestampCommandProbe*)user_data;
    probe.reset_calls++;
    probe.reset_command_buffer = command_buffer;
    probe.reset_pool = pool;
    probe.reset_first = first;
    probe.reset_count = count;
}

fn void timestamp_probe_write(
    void* user_data,
    vk::CommandBuffer command_buffer,
    vk::PipelineStageFlags2 stage,
    vk::QueryPool pool,
    uint index,
) {
    TimestampCommandProbe* probe = (TimestampCommandProbe*)user_data;
    probe.write_calls++;
    probe.write_command_buffer = command_buffer;
    probe.write_stage = stage;
    probe.write_pool = pool;
    probe.write_index = index;
}

fn void timestamp_probe_resolve(
    void* user_data,
    vk::CommandBuffer command_buffer,
    vk::QueryPool pool,
    uint first,
    uint count,
    vk::Buffer destination,
    vk::DeviceSize offset,
    vk::DeviceSize stride,
    vk::QueryResultFlags flags,
) {
    TimestampCommandProbe* probe = (TimestampCommandProbe*)user_data;
    probe.resolve_calls++;
    probe.resolve_command_buffer = command_buffer;
    probe.resolve_pool = pool;
    probe.resolve_first = first;
    probe.resolve_count = count;
    probe.resolve_destination = destination;
    probe.resolve_offset = offset;
    probe.resolve_stride = stride;
    probe.resolve_flags = flags;
}

fn TimestampCommandNativeOps timestamp_probe_ops(
    TimestampCommandProbe* probe,
) {
    return {
        .user_data = probe,
        .reset     = &timestamp_probe_reset,
        .write     = &timestamp_probe_write,
        .resolve   = &timestamp_probe_resolve,
    };
}

fn TimestampCommandFixture? create_timestamp_command_fixture(
    gpu::ContractValidation contract,
    uint reference_capacity,
) {
    gpu::RuntimeDesc runtime_desc = {
        .contract_validation = contract,
        .application_name    = "timestamp_command_test",
    };
    gpu::Runtime runtime = gpu::create_runtime(&runtime_desc)!;
    defer catch (void)gpu::destroy_runtime(&runtime);

    gpu::AdapterList adapters = gpu::enumerate_adapters(&runtime)!;
    gpu::Device device;
    for (uint i = 0; i < adapters.count; i++) {
        gpu::Adapter adapter = adapters.get(i)!;
        gpu::DeviceSupport support =
            gpu::supports_device_desc(&adapter)!;
        if (!support.supported) continue;
        gpu::Device? candidate = gpu::create_device(&adapter);
        if (catch err = candidate) {
            if (err == gpu::UNSUPPORTED_FEATURE) continue;
            return err~;
        }
        device = candidate;
        break;
    }
    if (!device.is_valid()) return gpu::UNSUPPORTED_FEATURE~;
    defer catch (void)gpu::destroy_device(&device);

    VkDeviceState* state =
        gpu::internal::device_slots[device.index].data.state;
    state.caps.timestamps = {
        .queues              = { .graphics },
        .graphics_valid_bits = 64,
        .period_ns           = 1.0f,
    };
    gpu::TimestampPoolHandle pool = state.timestamp_pools.alloc({
        .pool     = TIMESTAMP_COMMAND_NATIVE_POOL,
        .capacity = TIMESTAMP_COMMAND_POOL_CAPACITY,
    })!;
    defer catch (void)state.timestamp_pools.free(pool);

    gpu::AllocationDesc allocation_desc = {
        .size         = TIMESTAMP_COMMAND_DESTINATION_SIZE,
        .alignment    = 16,
        .memory_class = gpu::MemoryClass.GPU_PRIVATE,
        .access       = { .graphics },
        .debug_name   = "timestamp_command_destination",
    };
    gpu::GpuAllocation allocation =
        gpu::allocate_memory(&device, &allocation_desc)!;
    defer catch (void)gpu::free_allocation(&device, &allocation);
    gpu::GpuSpan destination =
        gpu::get_allocation_span(&device, allocation)!;

    gpu::Queue queue =
        gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!;
    gpu::CommandAllocatorDesc allocator_desc = {
        .command_buffer_capacity          = 1,
        .max_resource_references_per_list = reference_capacity,
    };
    gpu::CommandAllocator allocator = gpu::create_command_allocator(
        device: &device,
        queue:  queue,
        desc:   &allocator_desc,
    )!;
    defer catch (void)gpu::destroy_command_allocator(&allocator);
    gpu::CommandList commands = gpu::begin_commands(&allocator)!;
    defer catch (void)gpu::discard_commands(&commands);
    CommandRecord* record = resolve_command(&commands)!;
    QueueCompletionState* completion =
        queue_completion_state(state, record.queue.id)!;
    completion.queue_flags = (vk::QueueFlags)(
        (uint)vk::QueueFlagBits.GRAPHICS_BIT
            | (uint)vk::QueueFlagBits.COMPUTE_BIT
            | (uint)vk::QueueFlagBits.TRANSFER_BIT
    );
    return {
        .runtime     = runtime,
        .device      = device,
        .state       = state,
        .allocation  = allocation,
        .destination = destination,
        .pool        = pool,
        .allocator   = allocator,
        .commands    = commands,
        .record      = record,
    };
}

fn void? destroy_timestamp_command_fixture(
    TimestampCommandFixture* fixture,
) {
    if (fixture == null) return;
    if (fixture.commands.is_valid()) {
        gpu::discard_commands(&fixture.commands)!;
    }
    if (fixture.allocator.handle.is_valid()) {
        gpu::destroy_command_allocator(&fixture.allocator)!;
    }
    if (fixture.allocation.is_valid()) {
        gpu::free_allocation(
            &fixture.device,
            &fixture.allocation,
        )!;
    }
    if (fixture.pool.is_valid()) {
        fixture.state.timestamp_pools.free(fixture.pool)!;
        fixture.pool = {};
    }
    if (fixture.device.is_valid()) gpu::destroy_device(&fixture.device)!;
    if (fixture.runtime.is_valid()) gpu::destroy_runtime(&fixture.runtime)!;
}

fn void timestamp_stage_mapping_is_one_to_one() @test {
    assert(timestamp_stage_to_vk({ .all = true }) ==
        (vk::PipelineStageFlags2)
            vk::PipelineStageFlagBits2.STAGE_2_ALL_COMMANDS_BIT);
    assert(timestamp_stage_to_vk({ .transfer = true }) ==
        (vk::PipelineStageFlags2)
            vk::PipelineStageFlagBits2.STAGE_2_ALL_TRANSFER_BIT);
    assert(timestamp_stage_to_vk({ .compute = true }) ==
        (vk::PipelineStageFlags2)
            vk::PipelineStageFlagBits2.STAGE_2_COMPUTE_SHADER_BIT);
    assert(timestamp_stage_to_vk({ .vertex_shader = true }) ==
        (vk::PipelineStageFlags2)
            vk::PipelineStageFlagBits2.STAGE_2_VERTEX_SHADER_BIT);
    assert(timestamp_stage_to_vk({ .fragment_shader = true }) ==
        (vk::PipelineStageFlags2)
            vk::PipelineStageFlagBits2.STAGE_2_FRAGMENT_SHADER_BIT);
    assert(timestamp_stage_to_vk({ .color_output = true }) ==
        (vk::PipelineStageFlags2)
            vk::PipelineStageFlagBits2.STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT);
    assert(timestamp_stage_to_vk({ .depth_output = true }) ==
        (vk::PipelineStageFlags2)
            vk::PipelineStageFlagBits2.STAGE_2_LATE_FRAGMENT_TESTS_BIT);
    assert(timestamp_stage_to_vk({ .indirect = true }) ==
        (vk::PipelineStageFlags2)
            vk::PipelineStageFlagBits2.STAGE_2_DRAW_INDIRECT_BIT);
    assert(timestamp_stage_to_vk({ .acceleration_structure_build = true }) ==
        (vk::PipelineStageFlags2)vk::PipelineStageFlagBits2
            .STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR);
}

fn void timestamp_stage_shape_rejects_invalid_masks() @test {
    assert(!timestamp_stage_mask_valid({}));
    assert(!timestamp_stage_mask_valid({ .host = true }));
    assert(!timestamp_stage_mask_valid({ .present = true }));
    assert(!timestamp_stage_mask_valid({
        .compute = true,
        .transfer = true,
    }));
    assert(timestamp_stage_mask_valid({ .acceleration_structure_build }));
    assert(!timestamp_stage_mask_valid((gpu::StageMask)(1 << 12)));
}

fn void timestamp_destination_requires_packed_ulong_alignment() @test {
    assert(timestamp_destination_valid(0, ulong::size, ulong::size));
    assert(timestamp_destination_valid(
        ulong::size,
        2 * ulong::size,
        ulong::size,
    ));
    assert(!timestamp_destination_valid(4, ulong::size, ulong::size));
    assert(!timestamp_destination_valid(0, ulong::size - 1, ulong::size));
}

fn void timestamp_commands_emit_exact_native_parameters() @test {
    TimestampCommandFixture fixture =
        create_timestamp_command_fixture(
            gpu::ContractValidation.FULL,
            2,
        )!!;
    defer destroy_timestamp_command_fixture(&fixture)!!;
    TimestampCommandProbe probe;
    TimestampCommandNativeOps ops = timestamp_probe_ops(&probe);
    TimestampPoolSlot* pool_slot =
        fixture.state.timestamp_pools.get(fixture.pool)!!;
    AllocationSlot* allocation_slot =
        fixture.state.allocations.get(fixture.allocation)!!;
    gpu::GpuSpan destination = fixture.destination;
    destination.offset = 16;
    destination.size = 3 * ulong::size;

    full_reset_timestamps_with(
        record: fixture.record,
        pool:   fixture.pool,
        first:  2,
        count:  3,
        ops:    &ops,
    )!!;
    full_write_timestamp_with(
        record: fixture.record,
        pool:   fixture.pool,
        index:  5,
        stage:  { .compute },
        ops:    &ops,
    )!!;
    full_resolve_timestamps_with(
        record:      fixture.record,
        pool:        fixture.pool,
        first:       1,
        count:       3,
        destination: destination,
        ops:         &ops,
    )!!;

    assert(probe.reset_calls == 1
        && probe.reset_command_buffer == fixture.record.command_buffer
        && probe.reset_pool == TIMESTAMP_COMMAND_NATIVE_POOL
        && probe.reset_first == 2
        && probe.reset_count == 3);
    assert(probe.write_calls == 1
        && probe.write_command_buffer == fixture.record.command_buffer
        && probe.write_pool == TIMESTAMP_COMMAND_NATIVE_POOL
        && probe.write_index == 5
        && probe.write_stage == (vk::PipelineStageFlags2)
            vk::PipelineStageFlagBits2.STAGE_2_COMPUTE_SHADER_BIT);
    assert(probe.resolve_calls == 1
        && probe.resolve_command_buffer == fixture.record.command_buffer
        && probe.resolve_pool == TIMESTAMP_COMMAND_NATIVE_POOL
        && probe.resolve_first == 1
        && probe.resolve_count == 3
        && probe.resolve_destination == allocation_slot.backing.buffer
        && probe.resolve_offset == 16
        && probe.resolve_stride == ulong::size
        && (uint)probe.resolve_flags == (
            (uint)vk::QueryResultFlagBits.RESULT_64_BIT
                | (uint)vk::QueryResultFlagBits.WAIT_BIT
        ));
    assert(fixture.record.scratch.reference_count == 2);
    assert(pool_slot.tracked_command_references.load(
            AtomicOrdering.RELAXED,
        ) == 1);
    assert(allocation_slot.tracked_command_references.load(
            AtomicOrdering.RELAXED,
        ) == 1);

    gpu::discard_commands(&fixture.commands)!!;
    assert(pool_slot.tracked_command_references.load(
            AtomicOrdering.RELAXED,
        ) == 0);
    assert(allocation_slot.tracked_command_references.load(
            AtomicOrdering.RELAXED,
        ) == 0);
}

fn void timestamp_write_emits_every_stage_mapping() @test {
    TimestampCommandFixture fixture =
        create_timestamp_command_fixture(
            gpu::ContractValidation.FULL,
            1,
        )!!;
    defer destroy_timestamp_command_fixture(&fixture)!!;
    TimestampCommandProbe probe;
    TimestampCommandNativeOps ops = timestamp_probe_ops(&probe);
    gpu::StageMask[8] stages = {
        { .all },
        { .transfer },
        { .compute },
        { .vertex_shader },
        { .fragment_shader },
        { .color_output },
        { .depth_output },
        { .indirect },
    };
    vk::PipelineStageFlags2[8] expected = {
        (vk::PipelineStageFlags2)
            vk::PipelineStageFlagBits2.STAGE_2_ALL_COMMANDS_BIT,
        (vk::PipelineStageFlags2)
            vk::PipelineStageFlagBits2.STAGE_2_ALL_TRANSFER_BIT,
        (vk::PipelineStageFlags2)
            vk::PipelineStageFlagBits2.STAGE_2_COMPUTE_SHADER_BIT,
        (vk::PipelineStageFlags2)
            vk::PipelineStageFlagBits2.STAGE_2_VERTEX_SHADER_BIT,
        (vk::PipelineStageFlags2)
            vk::PipelineStageFlagBits2.STAGE_2_FRAGMENT_SHADER_BIT,
        (vk::PipelineStageFlags2)
            vk::PipelineStageFlagBits2.STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        (vk::PipelineStageFlags2)
            vk::PipelineStageFlagBits2.STAGE_2_LATE_FRAGMENT_TESTS_BIT,
        (vk::PipelineStageFlags2)
            vk::PipelineStageFlagBits2.STAGE_2_DRAW_INDIRECT_BIT,
    };

    foreach (i, stage : stages) {
        full_write_timestamp_with(
            record: fixture.record,
            pool:   fixture.pool,
            index:  (uint)i,
            stage:  stage,
            ops:    &ops,
        )!!;
        assert(probe.write_calls == i + 1
            && probe.write_command_buffer == fixture.record.command_buffer
            && probe.write_pool == TIMESTAMP_COMMAND_NATIVE_POOL
            && probe.write_index == i
            && probe.write_stage == expected[i]);
    }
    assert(fixture.record.scratch.reference_count == 1);
}

fn void timestamp_transfer_stage_accepts_implicit_graphics_support() @test {
    TimestampCommandFixture fixture =
        create_timestamp_command_fixture(
            gpu::ContractValidation.FULL,
            1,
        )!!;
    defer destroy_timestamp_command_fixture(&fixture)!!;
    TimestampCommandProbe probe;
    TimestampCommandNativeOps ops = timestamp_probe_ops(&probe);
    QueueCompletionState* completion = queue_completion_state(
        fixture.state,
        fixture.record.queue.id,
    )!!;
    completion.queue_flags = (vk::QueueFlags)
        vk::QueueFlagBits.GRAPHICS_BIT;

    full_write_timestamp_with(
        record: fixture.record,
        pool:   fixture.pool,
        index:  0,
        stage:  { .transfer },
        ops:    &ops,
    )!!;

    assert(probe.write_calls == 1
        && probe.write_stage == (vk::PipelineStageFlags2)
            vk::PipelineStageFlagBits2.STAGE_2_ALL_TRANSFER_BIT);
}

fn void timestamp_full_write_rejects_stage_absent_from_recording_queue() @test {
    TimestampCommandFixture fixture =
        create_timestamp_command_fixture(
            gpu::ContractValidation.FULL,
            1,
        )!!;
    defer destroy_timestamp_command_fixture(&fixture)!!;
    TimestampCommandProbe probe;
    TimestampCommandNativeOps ops = timestamp_probe_ops(&probe);
    QueueCompletionState* completion = queue_completion_state(
        fixture.state,
        fixture.record.queue.id,
    )!!;
    completion.queue_flags = (vk::QueueFlags)
        vk::QueueFlagBits.COMPUTE_BIT;

    assert(@catch(full_write_timestamp_with(
        record: fixture.record,
        pool:   fixture.pool,
        index:  0,
        stage:  { .fragment_shader },
        ops:    &ops,
    )) == gpu::INVALID_ARGUMENT,
        "a stage the recording queue family cannot execute is rejected");
    assert(probe.write_calls == 0
        && fixture.record.scratch.reference_count == 0,
        "the rejected stage reaches no native write and tracks nothing");

    full_write_timestamp_with(
        record: fixture.record,
        pool:   fixture.pool,
        index:  0,
        stage:  { .compute },
        ops:    &ops,
    )!!;

    assert(probe.write_calls == 1
        && probe.write_stage == (vk::PipelineStageFlags2)
            vk::PipelineStageFlagBits2.STAGE_2_COMPUTE_SHADER_BIT,
        "a stage the same queue family does execute still records");
}

fn void timestamp_full_write_rejects_invalid_masks_without_emission() @test {
    TimestampCommandFixture fixture =
        create_timestamp_command_fixture(
            gpu::ContractValidation.FULL,
            1,
        )!!;
    defer destroy_timestamp_command_fixture(&fixture)!!;
    TimestampCommandProbe probe;
    TimestampCommandNativeOps ops = timestamp_probe_ops(&probe);
    gpu::StageMask[6] invalid = {
        {},
        { .host },
        { .present },
        { .compute, .transfer },
        { .acceleration_structure_build },
        (gpu::StageMask)(1 << 12),
    };

    foreach (stage : invalid) {
        assert(@catch(full_write_timestamp_with(
            record: fixture.record,
            pool:   fixture.pool,
            index:  0,
            stage:  stage,
            ops:    &ops,
        )) == gpu::INVALID_ARGUMENT);
    }
    assert(probe.write_calls == 0);
    assert(fixture.record.scratch.reference_count == 0);
}

fn void timestamp_trusted_write_rejects_invalid_masks_without_emission() @test {
    TimestampCommandFixture fixture =
        create_timestamp_command_fixture(
            gpu::ContractValidation.TRUSTED,
            1,
        )!!;
    defer destroy_timestamp_command_fixture(&fixture)!!;
    TimestampCommandProbe probe;
    TimestampCommandNativeOps ops = timestamp_probe_ops(&probe);
    gpu::StageMask[5] invalid = {
        {},
        { .host },
        { .present },
        { .compute, .transfer },
        (gpu::StageMask)(1 << 12),
    };

    foreach (stage : invalid) {
        assert(@catch(trusted_write_timestamp_with(
            record: fixture.record,
            pool:   fixture.pool,
            index:  0,
            stage:  stage,
            ops:    &ops,
        )) == gpu::INVALID_ARGUMENT);
    }
    assert(probe.write_calls == 0);
    assert(fixture.record.scratch.reference_count == 0);
}

fn void timestamp_range_validation_prevents_native_emission() @test {
    TimestampCommandFixture fixture =
        create_timestamp_command_fixture(
            gpu::ContractValidation.TRUSTED,
            1,
        )!!;
    defer destroy_timestamp_command_fixture(&fixture)!!;
    TimestampCommandProbe probe;
    TimestampCommandNativeOps ops = timestamp_probe_ops(&probe);

    assert(@catch(trusted_reset_timestamps_with(
        record: fixture.record,
        pool:   fixture.pool,
        first:  0,
        count:  0,
        ops:    &ops,
    )) == gpu::INVALID_ARGUMENT);
    assert(@catch(trusted_reset_timestamps_with(
        record: fixture.record,
        pool:   fixture.pool,
        first:  uint::max,
        count:  2,
        ops:    &ops,
    )) == gpu::INVALID_ARGUMENT);
    assert(@catch(trusted_reset_timestamps_with(
        record: fixture.record,
        pool:   fixture.pool,
        first:  TIMESTAMP_COMMAND_POOL_CAPACITY,
        count:  1,
        ops:    &ops,
    )) == gpu::INVALID_ARGUMENT);
    assert(@catch(trusted_write_timestamp_with(
        record: fixture.record,
        pool:   fixture.pool,
        index:  TIMESTAMP_COMMAND_POOL_CAPACITY,
        stage:  { .compute },
        ops:    &ops,
    )) == gpu::INVALID_ARGUMENT);
    assert(@catch(trusted_resolve_timestamps_with(
        record:      fixture.record,
        pool:        fixture.pool,
        first:       0,
        count:       0,
        destination: fixture.destination,
        ops:         &ops,
    )) == gpu::INVALID_ARGUMENT);
    assert(probe.reset_calls == 0
        && probe.write_calls == 0
        && probe.resolve_calls == 0);
}

fn void timestamp_render_phase_allows_only_writes() @test {
    TimestampCommandFixture fixture =
        create_timestamp_command_fixture(
            gpu::ContractValidation.FULL,
            1,
        )!!;
    defer destroy_timestamp_command_fixture(&fixture)!!;
    TimestampCommandProbe probe;
    TimestampCommandNativeOps ops = timestamp_probe_ops(&probe);
    fixture.record.state =
        gpu::internal::CommandState.RECORDING_RENDER_PASS;
    defer fixture.record.state = gpu::internal::CommandState.RECORDING;

    assert(@catch(full_reset_timestamps_with(
        record: fixture.record,
        pool:   fixture.pool,
        first:  0,
        count:  1,
        ops:    &ops,
    )) == gpu::COMMAND_RECORDING_ERROR);
    assert(@catch(trusted_reset_timestamps_with(
        record: fixture.record,
        pool:   fixture.pool,
        first:  0,
        count:  1,
        ops:    &ops,
    )) == gpu::COMMAND_RECORDING_ERROR);
    assert(@catch(full_resolve_timestamps_with(
        record:      fixture.record,
        pool:        fixture.pool,
        first:       0,
        count:       1,
        destination: fixture.destination,
        ops:         &ops,
    )) == gpu::COMMAND_RECORDING_ERROR);
    assert(@catch(trusted_resolve_timestamps_with(
        record:      fixture.record,
        pool:        fixture.pool,
        first:       0,
        count:       1,
        destination: fixture.destination,
        ops:         &ops,
    )) == gpu::COMMAND_RECORDING_ERROR);
    full_write_timestamp_with(
        record: fixture.record,
        pool:   fixture.pool,
        index:  0,
        stage:  { .fragment_shader },
        ops:    &ops,
    )!!;
    trusted_write_timestamp_with(
        record: fixture.record,
        pool:   fixture.pool,
        index:  1,
        stage:  { .fragment_shader },
        ops:    &ops,
    )!!;
    assert(probe.reset_calls == 0
        && probe.resolve_calls == 0
        && probe.write_calls == 2);
}

fn void timestamp_full_mode_rejects_unsupported_recording_role() @test {
    TimestampCommandFixture fixture =
        create_timestamp_command_fixture(
            gpu::ContractValidation.FULL,
            2,
        )!!;
    defer destroy_timestamp_command_fixture(&fixture)!!;
    TimestampCommandProbe probe;
    TimestampCommandNativeOps ops = timestamp_probe_ops(&probe);
    fixture.record.queue.roles = { .transfer };
    fixture.state.caps.timestamps.queues = { .graphics };

    assert(@catch(full_reset_timestamps_with(
        record: fixture.record,
        pool:   fixture.pool,
        first:  0,
        count:  1,
        ops:    &ops,
    )) == gpu::UNSUPPORTED_FEATURE);
    assert(@catch(full_write_timestamp_with(
        record: fixture.record,
        pool:   fixture.pool,
        index:  0,
        stage:  { .compute },
        ops:    &ops,
    )) == gpu::UNSUPPORTED_FEATURE);
    assert(@catch(full_resolve_timestamps_with(
        record:      fixture.record,
        pool:        fixture.pool,
        first:       0,
        count:       1,
        destination: fixture.destination,
        ops:         &ops,
    )) == gpu::UNSUPPORTED_FEATURE);
    assert(probe.reset_calls == 0
        && probe.write_calls == 0
        && probe.resolve_calls == 0);
    assert(fixture.record.scratch.reference_count == 0);
}

fn void timestamp_destination_validation_is_transactional() @test {
    TimestampCommandFixture fixture =
        create_timestamp_command_fixture(
            gpu::ContractValidation.FULL,
            2,
        )!!;
    defer destroy_timestamp_command_fixture(&fixture)!!;
    TimestampCommandProbe probe;
    TimestampCommandNativeOps ops = timestamp_probe_ops(&probe);
    AllocationSlot* allocation_slot =
        fixture.state.allocations.get(fixture.allocation)!!;
    gpu::GpuSpan destination = fixture.destination;
    destination.offset = 4;
    destination.size = 2 * ulong::size;
    assert(@catch(full_resolve_timestamps_with(
        record:      fixture.record,
        pool:        fixture.pool,
        first:       0,
        count:       2,
        destination: destination,
        ops:         &ops,
    )) == gpu::INVALID_ARGUMENT);

    destination = fixture.destination;
    destination.size = ulong::size;
    assert(@catch(full_resolve_timestamps_with(
        record:      fixture.record,
        pool:        fixture.pool,
        first:       0,
        count:       2,
        destination: destination,
        ops:         &ops,
    )) == gpu::INVALID_ARGUMENT);

    destination = fixture.destination;
    allocation_slot.backing.usage.transfer_dst = false;
    assert(@catch(full_resolve_timestamps_with(
        record:      fixture.record,
        pool:        fixture.pool,
        first:       0,
        count:       1,
        destination: destination,
        ops:         &ops,
    )) == gpu::INVALID_ARGUMENT);
    allocation_slot.backing.usage.transfer_dst = true;

    fixture.record.queue.roles = { .graphics };
    allocation_slot.access = { .compute };
    allocation_slot.backing.access = { .compute };
    assert(@catch(full_resolve_timestamps_with(
        record:      fixture.record,
        pool:        fixture.pool,
        first:       0,
        count:       1,
        destination: destination,
        ops:         &ops,
    )) == gpu::INVALID_ARGUMENT);
    allocation_slot.access = { .graphics };
    allocation_slot.backing.access = { .graphics };

    assert(probe.resolve_calls == 0);
    assert(fixture.record.scratch.reference_count == 0);
    assert(allocation_slot.tracked_command_references.load(
            AtomicOrdering.RELAXED,
        ) == 0);
}

fn void timestamp_full_resolve_capacity_rejection_is_transactional() @test {
    TimestampCommandFixture fixture =
        create_timestamp_command_fixture(
            gpu::ContractValidation.FULL,
            1,
        )!!;
    defer destroy_timestamp_command_fixture(&fixture)!!;
    TimestampCommandProbe probe;
    TimestampCommandNativeOps ops = timestamp_probe_ops(&probe);
    TimestampPoolSlot* pool_slot =
        fixture.state.timestamp_pools.get(fixture.pool)!!;
    AllocationSlot* allocation_slot =
        fixture.state.allocations.get(fixture.allocation)!!;

    assert(@catch(full_resolve_timestamps_with(
        record:      fixture.record,
        pool:        fixture.pool,
        first:       0,
        count:       1,
        destination: fixture.destination,
        ops:         &ops,
    )) == gpu::COMMAND_ALLOCATOR_CAPACITY_EXCEEDED);
    assert(probe.resolve_calls == 0);
    assert(fixture.record.scratch.reference_count == 0);
    assert(pool_slot.tracked_command_references.load(
            AtomicOrdering.RELAXED,
        ) == 0);
    assert(allocation_slot.tracked_command_references.load(
            AtomicOrdering.RELAXED,
        ) == 0);

    full_reset_timestamps_with(
        record: fixture.record,
        pool:   fixture.pool,
        first:  0,
        count:  1,
        ops:    &ops,
    )!!;
    assert(probe.reset_calls == 1);
    assert(fixture.record.scratch.reference_count == 1);
    assert(pool_slot.tracked_command_references.load(
            AtomicOrdering.RELAXED,
        ) == 1);
}

fn void timestamp_trusted_mode_performs_zero_reference_work() @test {
    TimestampCommandFixture fixture =
        create_timestamp_command_fixture(
            gpu::ContractValidation.TRUSTED,
            1,
        )!!;
    defer destroy_timestamp_command_fixture(&fixture)!!;
    TimestampCommandProbe probe;
    TimestampCommandNativeOps ops = timestamp_probe_ops(&probe);
    TimestampPoolSlot* pool_slot =
        fixture.state.timestamp_pools.get(fixture.pool)!!;
    AllocationSlot* allocation_slot =
        fixture.state.allocations.get(fixture.allocation)!!;

    trusted_reset_timestamps_with(
        record: fixture.record,
        pool:   fixture.pool,
        first:  0,
        count:  2,
        ops:    &ops,
    )!!;
    trusted_write_timestamp_with(
        record: fixture.record,
        pool:   fixture.pool,
        index:  2,
        stage:  { .compute },
        ops:    &ops,
    )!!;
    trusted_resolve_timestamps_with(
        record:      fixture.record,
        pool:        fixture.pool,
        first:       0,
        count:       2,
        destination: fixture.destination,
        ops:         &ops,
    )!!;

    assert(probe.reset_calls == 1
        && probe.write_calls == 1
        && probe.resolve_calls == 1);
    assert(fixture.record.scratch.reference_count == 0);
    assert(pool_slot.tracked_command_references.load(
            AtomicOrdering.RELAXED,
        ) == 0);
    assert(allocation_slot.tracked_command_references.load(
            AtomicOrdering.RELAXED,
        ) == 0);
}
