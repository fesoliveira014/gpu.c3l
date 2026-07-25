module gpu_test;

import gpu::internal @public;
import gpu::internal::vk @public;
import gpu @public;
import std::atomic;
import std::thread;

const uint DEVICE_REGISTRY_THREAD_COUNT = 8;
const uint COMMAND_PIN_RECORD_CAPACITY = 64;

struct CommandPinRecordCell {
    gpu::internal::CommandRecord record;
    uint                         generation;
    bool                         used;
}

struct DeviceRegistryWorkerArgs {
    Atomic{uint}* ready;
    Atomic{bool}* release;
    DeviceFixture fixture;
    gpu::Device   device;
    bool          failed;
}

struct BlockingDestroyFixture {
    gpu::internal::vk::VkDeviceState state;
}

struct BlockingOperationFixture {
    gpu::internal::vk::VkDeviceState state;
}

struct CommandPinFixture {
    gpu::internal::vk::VkDeviceState state;
    Atomic{bool} command_entered;
    Atomic{bool} command_release;
    Atomic{bool} discard_entered;
    Atomic{bool} discard_release;
    uint         pins_during_command;
    uint         pins_during_submit;
    uint         next_command;
    gpu::Device  device;
    CommandPinRecordCell[COMMAND_PIN_RECORD_CAPACITY] records;
    bool         block_command;
    bool         reject_discard;
    bool         reject_end;
    bool         reject_command;
    bool         block_discard;
    bool         reuse_command_index;
    bool         submit_busy;
    bool         lose_during_submit;
    bool         return_null_backend_state;
    bool         return_null_backend_command;
}

struct StaleCommandWorkerArgs {
    gpu::CommandList commands;
    fault            result;
    bool             failed;
}

struct DiscardCommandWorkerArgs {
    gpu::CommandList commands;
    fault            result;
    bool             failed;
}

struct DestroyWorkerArgs {
    gpu::Device device;
    bool        failed;
}

struct DeviceOperationWorkerArgs {
    gpu::Device device;
    bool        failed;
}

struct RuntimeRetainWorkerArgs {
    gpu::Runtime  runtime;
    Atomic{uint}* ready;
    Atomic{bool}* release;
    bool          failed;
}

fn void? command_pin_begin(
    gpu::CommandAllocator* allocator,
    gpu::CommandList* out_commands,
) {
    gpu::Device device = allocator.device;
    gpu::internal::DeviceData* data = gpu::internal::resolve_pinned_device(&device);
    if (data == null || out_commands == null) return gpu::INVALID_HANDLE~;
    CommandPinFixture* fixture = (CommandPinFixture*)data.state;
    if (data.command_ops == null
        || fixture.return_null_backend_state
        || fixture.return_null_backend_command) {
        return gpu::BACKEND_ERROR~;
    }
    fixture.device = device;
    fixture.next_command++;
    uint index = fixture.reuse_command_index ? 1 : fixture.next_command;
    if (index >= fixture.records.len) return gpu::SLOT_TABLE_FULL~;
    CommandPinRecordCell* cell = &fixture.records[index];
    if (cell.used) return gpu::DEVICE_BUSY~;
    cell.generation++;
    if (cell.generation == 0) cell.generation = 1;
    gpu::internal::CommandListHandle identity = {
        .owner      = gpu::internal::resource_owner_tag(
            data.resource_owner,
            gpu::internal::ResourceOwnerKind.COMMAND,
        ),
        .index      = index,
        .generation = cell.generation,
    };
    cell.record = {
        .ops             = data.command_ops,
        .backend_state   = fixture,
        .backend_command = fixture,
        .device          = device,
        .identity        = identity,
        .state           = gpu::internal::CommandState.RECORDING,
    };
    cell.used = true;
    gpu::internal::publish_command_list(&cell.record, out_commands);
}

fn void? command_pin_end(gpu::internal::CommandRecord* record) {
    CommandPinFixture* fixture =
        (CommandPinFixture*)record.backend_state;
    if (fixture.reject_end) return gpu::DEVICE_BUSY~;
    if (record.state != gpu::internal::CommandState.RECORDING) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    record.state = gpu::internal::CommandState.EXECUTABLE;
}

fn void? command_pin_discard(
    gpu::internal::CommandRecord* record,
    bool executable,
) {
    if (record == null || !record.identity.is_valid()) {
        return gpu::INVALID_HANDLE~;
    }
    CommandPinFixture* fixture = (CommandPinFixture*)record.backend_state;
    if (fixture == null) return gpu::INVALID_HANDLE~;
    if (fixture.reject_discard) return gpu::DEVICE_BUSY~;
    bool state_matches = executable
        ? record.state == gpu::internal::CommandState.EXECUTABLE
        : record.state == gpu::internal::CommandState.RECORDING
            || record.state
                == gpu::internal::CommandState.RECORDING_RENDER_PASS;
    if (!state_matches) return gpu::COMMAND_RECORDING_ERROR~;
    uint index = record.identity.index;
    record.state = gpu::internal::CommandState.INACTIVE;
    fixture.discard_entered.store(true, AtomicOrdering.RELEASE);
    while (fixture.block_discard
        && !fixture.discard_release.load(AtomicOrdering.ACQUIRE)) {
        thread::yield();
    }
    fixture.records[index].record = {};
    fixture.records[index].used = false;
}

fn int discard_command_worker(void* raw_args) {
    DiscardCommandWorkerArgs* args = (DiscardCommandWorkerArgs*)raw_args;
    if (catch result = gpu::discard_commands(&args.commands)) {
        args.result = result;
        args.failed = true;
    }
    return args.failed ? 1 : 0;
}

fn void? command_pin_barrier(
    gpu::internal::CommandRecord* commands,
    gpu::Barrier* barrier,
) {
    CommandPinFixture* fixture =
        (CommandPinFixture*)commands.backend_state;
    if (fixture == null) return gpu::INVALID_HANDLE~;
    gpu::internal::DeviceSlotControl control =
        (gpu::internal::DeviceSlotControl)gpu::internal::device_slots[fixture.device.index].control.load(
            AtomicOrdering.ACQUIRE,
        );
    fixture.pins_during_command = control.pins;
    fixture.command_entered.store(true, AtomicOrdering.RELEASE);
    while (fixture.block_command
        && !fixture.command_release.load(AtomicOrdering.ACQUIRE)) {
        thread::yield();
    }
    if (fixture.reject_command) return gpu::INVALID_HANDLE~;
}

fn void? command_pin_copy_buffer(
    gpu::internal::CommandRecord* record,
    gpu::BufferCopyDesc* desc,
) {
    (void)record;
    (void)desc;
    return gpu::UNSUPPORTED_FEATURE~;
}

fn void? command_pin_fill_buffer(
    gpu::internal::CommandRecord* record,
    gpu::GpuSpan span,
    uint value,
) {
    (void)record;
    (void)span;
    (void)value;
    return gpu::UNSUPPORTED_FEATURE~;
}

fn void? command_pin_copy_buffer_to_texture(
    gpu::internal::CommandRecord* record,
    gpu::BufferTextureCopyDesc* desc,
) {
    (void)record;
    (void)desc;
    return gpu::UNSUPPORTED_FEATURE~;
}

fn void? command_pin_copy_texture_to_buffer(
    gpu::internal::CommandRecord* record,
    gpu::TextureBufferCopyDesc* desc,
) {
    (void)record;
    (void)desc;
    return gpu::UNSUPPORTED_FEATURE~;
}

fn void? command_pin_texture_barrier(
    gpu::internal::CommandRecord* record,
    gpu::TextureBarrier* barrier,
) {
    (void)record;
    (void)barrier;
    return gpu::UNSUPPORTED_FEATURE~;
}

fn void? command_pin_bind_pipeline(
    gpu::internal::CommandRecord* record,
    gpu::PipelineHandle pipeline,
) {
    (void)record;
    (void)pipeline;
    return gpu::UNSUPPORTED_FEATURE~;
}

fn void? command_pin_set_depth_state(
    gpu::internal::CommandRecord* record,
    gpu::DepthState* depth,
) {
    (void)record;
    (void)depth;
    return gpu::UNSUPPORTED_FEATURE~;
}

fn void? command_pin_set_raster_state(
    gpu::internal::CommandRecord* record,
    gpu::DynamicRasterState* raster,
) {
    (void)record;
    (void)raster;
    return gpu::UNSUPPORTED_FEATURE~;
}

fn void? command_pin_set_graphics_state(
    gpu::internal::CommandRecord* record,
    gpu::GraphicsState* state,
) {
    (void)record;
    (void)state;
    return gpu::UNSUPPORTED_FEATURE~;
}

fn void? command_pin_dispatch(
    gpu::internal::CommandRecord* record,
    gpu::GpuAddress root,
    gpu::Vec3u groups,
) {
    (void)record;
    (void)root;
    (void)groups;
    return gpu::UNSUPPORTED_FEATURE~;
}

fn void? command_pin_begin_render_pass(
    gpu::internal::CommandRecord* record,
    gpu::RenderPassDesc* desc,
) {
    (void)record;
    (void)desc;
    return gpu::UNSUPPORTED_FEATURE~;
}

fn void? command_pin_begin_render_pass_with_state(
    gpu::internal::CommandRecord* record,
    gpu::RenderPassDesc* desc,
    gpu::GraphicsState* state,
) {
    (void)record;
    (void)desc;
    (void)state;
    return gpu::UNSUPPORTED_FEATURE~;
}

fn void? command_pin_set_viewport(
    gpu::internal::CommandRecord* record,
    gpu::Viewport* viewport,
) {
    (void)record;
    (void)viewport;
    return gpu::UNSUPPORTED_FEATURE~;
}

fn void? command_pin_set_scissor(
    gpu::internal::CommandRecord* record,
    gpu::ScissorRect* scissor,
) {
    (void)record;
    (void)scissor;
    return gpu::UNSUPPORTED_FEATURE~;
}

fn void? command_pin_end_render_pass(gpu::internal::CommandRecord* record) {
    (void)record;
    return gpu::UNSUPPORTED_FEATURE~;
}

fn void? command_pin_draw(
    gpu::internal::CommandRecord* record,
    gpu::GpuAddress vertex_root,
    gpu::GpuAddress fragment_root,
    uint vertex_count,
    uint instance_count,
) {
    (void)record;
    (void)vertex_root;
    (void)fragment_root;
    (void)vertex_count;
    (void)instance_count;
    return gpu::UNSUPPORTED_FEATURE~;
}

fn void? command_pin_draw_indexed(
    gpu::internal::CommandRecord* record,
    gpu::GpuAddress vertex_root,
    gpu::GpuAddress fragment_root,
    gpu::GpuSpan index_span,
    uint index_count,
    uint instance_count,
    gpu::IndexType index_type,
) {
    (void)record;
    (void)vertex_root;
    (void)fragment_root;
    (void)index_span;
    (void)index_count;
    (void)instance_count;
    (void)index_type;
    return gpu::UNSUPPORTED_FEATURE~;
}

fn void? command_pin_draw_indirect(
    gpu::internal::CommandRecord* record,
    gpu::GpuAddress vertex_root,
    gpu::GpuAddress fragment_root,
    gpu::GpuSpan args,
    uint draw_count,
) {
    (void)record;
    (void)vertex_root;
    (void)fragment_root;
    (void)args;
    (void)draw_count;
    return gpu::UNSUPPORTED_FEATURE~;
}

fn void? command_pin_draw_indexed_indirect(
    gpu::internal::CommandRecord* record,
    gpu::GpuAddress vertex_root,
    gpu::GpuAddress fragment_root,
    gpu::GpuSpan args,
    uint draw_count,
    gpu::GpuSpan index_span,
    gpu::IndexType index_type,
) {
    (void)record;
    (void)vertex_root;
    (void)fragment_root;
    (void)args;
    (void)draw_count;
    (void)index_span;
    (void)index_type;
    return gpu::UNSUPPORTED_FEATURE~;
}

fn void? command_pin_draw_indexed_indirect_count(
    gpu::internal::CommandRecord* record,
    gpu::GpuAddress vertex_root,
    gpu::GpuAddress fragment_root,
    gpu::GpuSpan args,
    gpu::GpuSpan count_span,
    uint max_draw_count,
    gpu::GpuSpan index_span,
    gpu::IndexType index_type,
) {
    (void)record;
    (void)vertex_root;
    (void)fragment_root;
    (void)args;
    (void)count_span;
    (void)max_draw_count;
    (void)index_span;
    (void)index_type;
    return gpu::UNSUPPORTED_FEATURE~;
}

fn void? command_pin_dispatch_indirect(
    gpu::internal::CommandRecord* record,
    gpu::GpuAddress root,
    gpu::GpuSpan args,
) {
    (void)record;
    (void)root;
    (void)args;
    return gpu::UNSUPPORTED_FEATURE~;
}

fn void? command_pin_draw_generated(
    gpu::internal::CommandRecord* record,
    gpu::GpuSpan records,
    gpu::GpuSpan count_span,
    uint max_draw_count,
) {
    (void)record;
    (void)records;
    (void)count_span;
    (void)max_draw_count;
    return gpu::UNSUPPORTED_FEATURE~;
}

fn void? command_pin_draw_indexed_generated(
    gpu::internal::CommandRecord* record,
    gpu::GpuSpan records,
    gpu::GpuSpan count_span,
    uint max_draw_count,
    gpu::GpuSpan index_span,
    gpu::IndexType index_type,
) {
    (void)record;
    (void)records;
    (void)count_span;
    (void)max_draw_count;
    (void)index_span;
    (void)index_type;
    return gpu::UNSUPPORTED_FEATURE~;
}

fn void? command_pin_dispatch_generated(
    gpu::internal::CommandRecord* record,
    gpu::GpuSpan records,
    gpu::GpuSpan count_span,
    uint max_dispatch_count,
) {
    (void)record;
    (void)records;
    (void)count_span;
    (void)max_dispatch_count;
    return gpu::UNSUPPORTED_FEATURE~;
}

fn void? command_pin_begin_label(
    gpu::internal::CommandRecord* record,
    ZString label,
    float[4] color,
) {
    (void)record;
    (void)label;
    (void)color;
    return gpu::UNSUPPORTED_FEATURE~;
}

fn void? command_pin_end_label(gpu::internal::CommandRecord* record) {
    (void)record;
    return gpu::UNSUPPORTED_FEATURE~;
}

fn gpu::CompletionPoint? command_pin_submit(
    gpu::Queue queue,
    gpu::SubmitDesc* desc,
) {
    gpu::Device device = queue.device;
    CommandPinFixture* fixture =
        (CommandPinFixture*)gpu::internal::device_backend_state_ptr(&device);
    if (fixture == null) return gpu::INVALID_HANDLE~;
    gpu::internal::DeviceSlotControl control =
        (gpu::internal::DeviceSlotControl)gpu::internal::device_slots[device.index].control.load(
            AtomicOrdering.ACQUIRE,
        );
    fixture.pins_during_submit = control.pins;
    if (fixture.submit_busy) return gpu::DEVICE_BUSY~;
    gpu::internal::CommandRecord*[gpu::internal::MAX_DEVICE_COMMANDS] records;
    foreach (i, &commands : desc.command_lists) {
        records[i] = gpu::internal::executable_command(commands)!;
    }
    if (fixture.lose_during_submit) gpu::internal::mark_device_lost(&device);
    foreach (record : records[:desc.command_lists.len]) {
        record.state = gpu::internal::CommandState.SUBMITTED;
        fixture.records[record.identity.index].used = false;
        record.state = gpu::internal::CommandState.INACTIVE;
    }
    gpu::internal::DeviceOperation operation =
        gpu::internal::borrow_retained_device_operation(&device, true)!;
    gpu::internal::release_device_children(
        &operation,
        (ulong)desc.command_lists.len,
    );
    (void)gpu::internal::release_retained_device_pins(
        &operation,
        (uint)desc.command_lists.len,
    );
    gpu::internal::end_device_operation(&operation);
    return {
        .device  = device,
        .payload = 1ull << 10,
    };
}

gpu::internal::BackendVTable blocking_device_vtable = {
};

gpu::internal::BackendVTable blocking_operation_vtable = {
};

gpu::internal::CommandOps command_pin_ops = {
    .copy_buffer                 = &command_pin_copy_buffer,
    .fill_buffer                 = &command_pin_fill_buffer,
    .copy_buffer_to_texture      = &command_pin_copy_buffer_to_texture,
    .copy_texture_to_buffer      = &command_pin_copy_texture_to_buffer,
    .barrier                     = &command_pin_barrier,
    .texture_barrier             = &command_pin_texture_barrier,
    .bind_pipeline               = &command_pin_bind_pipeline,
    .set_depth_state             = &command_pin_set_depth_state,
    .set_raster_state            = &command_pin_set_raster_state,
    .set_graphics_state          = &command_pin_set_graphics_state,
    .dispatch                    = &command_pin_dispatch,
    .begin_render_pass           = &command_pin_begin_render_pass,
    .begin_render_pass_with_state = &command_pin_begin_render_pass_with_state,
    .set_viewport                = &command_pin_set_viewport,
    .set_scissor                 = &command_pin_set_scissor,
    .end_render_pass             = &command_pin_end_render_pass,
    .draw                        = &command_pin_draw,
    .draw_indexed                = &command_pin_draw_indexed,
    .draw_indirect               = &command_pin_draw_indirect,
    .draw_indexed_indirect       = &command_pin_draw_indexed_indirect,
    .draw_indexed_indirect_count = &command_pin_draw_indexed_indirect_count,
    .dispatch_indirect           = &command_pin_dispatch_indirect,
    .draw_generated              = &command_pin_draw_generated,
    .draw_indexed_generated      = &command_pin_draw_indexed_generated,
    .dispatch_generated          = &command_pin_dispatch_generated,
    .begin_label                 = &command_pin_begin_label,
    .end_label                   = &command_pin_end_label,
};

gpu::internal::BackendVTable command_pin_vtable = {
    .begin_commands            = &command_pin_begin,
    .end_commands              = &command_pin_end,
    .discard_commands          = &command_pin_discard,
    .submit                    = &command_pin_submit,
    .command_ops               = &command_pin_ops,
};

fn gpu::CommandList? begin_command_pin(gpu::Device* device) {
    if (device == null || !device.is_valid()) return gpu::INVALID_HANDLE~;
    gpu::internal::DeviceData* data = gpu::internal::resolve_pinned_device(device);
    if (data == null) return gpu::INVALID_HANDLE~;
    gpu::CommandAllocator allocator = {
        .device = *device,
        .queue = gpu::get_queue(device, gpu::QueueKind.GRAPHICS)!,
        .handle = {
            .owner = gpu::internal::resource_owner_tag(
                data.resource_owner,
                gpu::internal::ResourceOwnerKind.COMMAND_ALLOCATOR,
            ),
            .generation = 1,
        },
    };
    return gpu::begin_commands(&allocator);
}

fn int destroy_device_worker(void* raw_args) {
    DestroyWorkerArgs* args = (DestroyWorkerArgs*)raw_args;
    if (catch destroy_fault = gpu::destroy_device(&args.device)) {
        (void)destroy_fault;
        args.failed = true;
    }
    return args.failed ? 1 : 0;
}

fn int device_operation_worker(void* raw_args) {
    DeviceOperationWorkerArgs* args = (DeviceOperationWorkerArgs*)raw_args;
    if (catch operation_fault = gpu::get_memory_stats(&args.device)) {
        (void)operation_fault;
        args.failed = true;
    }
    return args.failed ? 1 : 0;
}

fn int stale_command_worker(void* raw_args) {
    StaleCommandWorkerArgs* args = (StaleCommandWorkerArgs*)raw_args;
    gpu::Barrier barrier = {
        .before = { .compute = true },
        .after = { .compute = true },
    };
    if (catch command_fault = gpu::cmd_barrier(
        &args.commands,
        &barrier,
    )) {
        args.result = command_fault;
    } else {
        args.failed = true;
    }
    return args.failed ? 1 : 0;
}

fn int runtime_retain_worker(void* raw_args) {
    RuntimeRetainWorkerArgs* args = (RuntimeRetainWorkerArgs*)raw_args;
    bool retained = true;
    if (catch retain_fault = gpu::internal::runtime_retain_device(&args.runtime)) {
        (void)retain_fault;
        args.failed = true;
        retained = false;
    }

    (void)args.ready.add(1, AtomicOrdering.RELEASE);
    while (!args.release.load(AtomicOrdering.ACQUIRE)) {
        thread::yield();
    }
    if (retained) {
        if (catch release_fault = gpu::internal::runtime_release_device(&args.runtime)) {
            (void)release_fault;
            args.failed = true;
        }
    }
    return args.failed ? 1 : 0;
}

fn int device_registry_worker(void* raw_args) {
    DeviceRegistryWorkerArgs* args = (DeviceRegistryWorkerArgs*)raw_args;
    gpu::Device? registered = gpu::internal::register_device(
        state:  &args.fixture.state,
        caps:   {},
        vtable: &device_fixture_vtable,
    );
    if (catch registration_fault = registered) {
        (void)registration_fault;
        args.failed = true;
    } else {
        args.device = registered;
    }

    (void)args.ready.add(1, AtomicOrdering.RELEASE);
    while (!args.release.load(AtomicOrdering.ACQUIRE)) {
        thread::yield();
    }
    if (!args.failed) {
        gpu::Device device = args.device;
        gpu::destroy_device(&device)!!;
    }
    return args.failed ? 1 : 0;
}

fn void command_allocator_public_lifecycle_is_transactional() @test {
    CommandPinFixture fixture;
    fixture.state.resource_owner =
        gpu::internal::allocate_device_resource_owner()!!;
    gpu::Device device = gpu::internal::register_device(
        state:          &fixture.state,
        caps:           {},
        vtable:         &command_pin_vtable,
        resource_owner: fixture.state.resource_owner,
    )!!;
    gpu::Queue queue = gpu::get_queue(
        &device,
        gpu::QueueKind.GRAPHICS,
    )!!;
    gpu::CommandAllocator allocator = gpu::create_command_allocator(
        &device,
        queue,
    )!!;
    gpu::internal::DeviceSlot* slot = &gpu::internal::device_slots[device.index];

    assert(allocator.device == device && allocator.queue == queue,
        "allocator carries its exact device and queue identity");
    assert(fixture.state.allocator_desc.command_buffer_capacity
            == gpu::DEFAULT_COMMAND_ALLOCATOR_CAPACITY
        && fixture.state.allocator_desc.max_resource_references_per_list
            == gpu::DEFAULT_COMMAND_REFERENCES_PER_LIST
        && fixture.state.allocator_desc
            .max_generated_preprocess_buffers_per_list
            == gpu::DEFAULT_COMMAND_PREPROCESS_PER_LIST,
        "zero descriptor fields normalize before backend dispatch");
    assert(slot.live_children.load(AtomicOrdering.ACQUIRE) == 1,
        "a published allocator retains one device child");

    fixture.state.reject_allocator_destroy = true;
    gpu::CommandAllocator original = allocator;
    assert(@catch(gpu::destroy_command_allocator(&allocator))
        == gpu::RESOURCE_IN_USE,
        "backend destroy failure remains retryable");
    assert(allocator.handle == original.handle
        && slot.live_children.load(AtomicOrdering.ACQUIRE) == 1,
        "failed destroy preserves the token and child retain");

    fixture.state.reject_allocator_destroy = false;
    gpu::destroy_command_allocator(&allocator)!!;
    assert(!allocator.handle.is_valid()
        && slot.live_children.load(AtomicOrdering.ACQUIRE) == 0,
        "successful destroy consumes the token and releases its child");
    gpu::destroy_device(&device)!!;
}

fn void command_allocator_creation_rejects_before_publication() @test {
    CommandPinFixture first_fixture;
    CommandPinFixture second_fixture;
    first_fixture.state.resource_owner =
        gpu::internal::allocate_device_resource_owner()!!;
    gpu::Device first = gpu::internal::register_device(
        state:          &first_fixture.state,
        caps:           {},
        vtable:         &command_pin_vtable,
        resource_owner: first_fixture.state.resource_owner,
    )!!;
    gpu::Device second = gpu::internal::register_device(
        state:  &second_fixture.state,
        caps:   {},
        vtable: &command_pin_vtable,
    )!!;
    gpu::Queue foreign = gpu::get_queue(
        &second,
        gpu::QueueKind.GRAPHICS,
    )!!;
    gpu::CommandAllocatorDesc oversized = {
        .command_buffer_capacity = gpu::MAX_COMMAND_ALLOCATOR_CAPACITY + 1,
    };

    assert(@catch(gpu::create_command_allocator(&first, foreign))
        == gpu::INVALID_HANDLE,
        "a foreign queue is rejected before backend creation");
    gpu::Queue queue = gpu::get_queue(
        &first,
        gpu::QueueKind.GRAPHICS,
    )!!;
    assert(@catch(gpu::create_command_allocator(
        &first,
        queue,
        &oversized,
    )) == gpu::INVALID_ARGUMENT,
        "an oversized descriptor is rejected before backend creation");
    assert(first_fixture.state.allocator_creates == 0
        && gpu::internal::device_slots[first.index].live_children.load(
            AtomicOrdering.ACQUIRE,
        ) == 0,
        "validation failures publish no backend allocator or child");

    first_fixture.state.reject_allocator_create = true;
    assert(@catch(gpu::create_command_allocator(&first, queue))
        == gpu::DEVICE_BUSY,
        "backend create faults propagate exactly");
    assert(gpu::internal::device_slots[first.index].live_children.load(
            AtomicOrdering.ACQUIRE,
        ) == 0,
        "backend create failure retains no child");

    first_fixture.state.reject_allocator_create = false;
    first_fixture.state.return_invalid_allocator = true;
    assert(@catch(gpu::create_command_allocator(&first, queue))
        == gpu::BACKEND_ERROR,
        "an invalid backend handle fails publication");
    first_fixture.state.return_invalid_allocator = false;
    first_fixture.state.return_foreign_allocator = true;
    assert(@catch(gpu::create_command_allocator(&first, queue))
        == gpu::BACKEND_ERROR,
        "a foreign backend handle fails publication");
    assert(first_fixture.state.allocator_destroy_calls == 2
        && gpu::internal::device_slots[first.index].live_children.load(
            AtomicOrdering.ACQUIRE,
        ) == 0,
        "invalid backend publication rolls back exactly once without a child");
    gpu::destroy_device(&second)!!;
    gpu::destroy_device(&first)!!;
}

fn void command_calls_use_the_retained_lifetime_pin() @test {
    CommandPinFixture fixture;
    gpu::Device device = gpu::internal::register_device(
        state:  &fixture.state,
        caps:   {},
        vtable: &command_pin_vtable,
    )!!;
    gpu::CommandList commands = begin_command_pin(&device)!!;
    gpu::internal::CommandRecord* record = commands.record;
    uint record_index = record.identity.index;
    gpu::internal::DeviceSlot* slot = &gpu::internal::device_slots[device.index];
    gpu::internal::DeviceSlotControl after_begin =
        (gpu::internal::DeviceSlotControl)slot.control.load(AtomicOrdering.ACQUIRE);

    gpu::Barrier barrier = {
        .before = { .compute = true },
        .after = { .compute = true },
    };
    gpu::cmd_barrier(&commands, &barrier)!!;
    gpu::ExecutableCommandList executable = gpu::end_commands(&commands)!!;
    gpu::internal::DeviceSlotControl after_end =
        (gpu::internal::DeviceSlotControl)slot.control.load(AtomicOrdering.ACQUIRE);
    bool record_transferred =
        record.state == gpu::internal::CommandState.EXECUTABLE
        && executable.record == record
        && executable.generation == record.identity.generation;
    gpu::discard_executable_commands(&executable)!!;
    gpu::internal::DeviceSlotControl after_discard =
        (gpu::internal::DeviceSlotControl)slot.control.load(AtomicOrdering.ACQUIRE);

    assert(after_begin.pins == 1,
        "begin transfers its operation pin to the recording token");
    assert(record_transferred,
        "end transfers the same stable record to its executable phase");
    assert(fixture.pins_during_command == 1,
        "hot command recording uses the retained lifetime without another pin");
    assert(after_end.pins == 1,
        "the executable token retains the same lifetime pin");
    assert(after_discard.pins == 0,
        "discard releases the command lifetime pin exactly once");
    assert(record.state == gpu::internal::CommandState.INACTIVE
        && !fixture.records[record_index].used,
        "successful discard invalidates the record before lifetime release");
    assert(slot.live_children.load(AtomicOrdering.ACQUIRE) == 0,
        "discard releases the command child exactly once");
    gpu::destroy_device(&device)!!;
}

fn void warm_command_dispatch_skips_resolution_paths() @test {
    CommandPinFixture fixture;
    gpu::Device device = gpu::internal::register_device(
        state:  &fixture.state,
        caps:   {},
        vtable: &command_pin_vtable,
    )!!;
    gpu::CommandList commands = begin_command_pin(&device)!!;
    gpu::reset_command_resolution_stats();

    gpu::Barrier barrier = {
        .before = { .compute = true },
        .after  = { .compute = true },
    };
    gpu::cmd_barrier(&commands, &barrier)!!;
    gpu::CommandResolutionStats stats = gpu::command_resolution_stats();

    assert(stats.device_registry_resolutions == 0,
        "warm recording does not resolve the device registry");
    assert(stats.retained_pin_operations == 0,
        "warm recording does not borrow or acquire a device pin");
    assert(stats.lifecycle_vtable_dispatches == 0,
        "warm recording does not load lifecycle dispatch");
    assert(stats.command_table_lookups == 0,
        "warm recording does not resolve the backend command table");
    assert(stats.pipeline_table_lookups == 0
        && stats.pipeline_cache_lookups == 0,
        "a warm barrier performs no pipeline resolution");
    assert(stats.command_policy_selections == 0,
        "warm recording does not reselect command policy");
    assert(stats.encoder_cell_computations == 0,
        "removed encoder-cell proof remains zero");
    assert(stats.encoder_lease_comparisons == 0,
        "removed encoder-lease proof remains zero");
    assert(stats.recording_commands == 1,
        "the counter distinguishes one recording operation from resolution");

    gpu::discard_commands(&commands)!!;
    gpu::destroy_device(&device)!!;
}

fn void null_and_stale_direct_command_tokens_are_rejected() @test {
    CommandPinFixture fixture;
    gpu::Device device = gpu::internal::register_device(
        state:  &fixture.state,
        caps:   {},
        vtable: &command_pin_vtable,
    )!!;
    gpu::CommandList commands = begin_command_pin(&device)!!;
    gpu::Barrier barrier = {
        .before = { .compute = true },
        .after  = { .compute = true },
    };

    gpu::CommandList null_record = commands;
    null_record.record = null;
    gpu::CommandList zero_generation = commands;
    zero_generation.generation = 0;
    gpu::CommandList zero_device_identity = commands;
    zero_device_identity.device_identity = 0;
    gpu::CommandList reserved_device_identity = commands;
    reserved_device_identity.device_identity |= 0x4000_0000;
    gpu::CommandList stale_generation = commands;
    stale_generation.generation++;

    assert(@catch(gpu::cmd_barrier(&null_record, &barrier))
        == gpu::INVALID_HANDLE,
        "a null direct record is rejected");
    assert(@catch(gpu::cmd_barrier(&zero_generation, &barrier))
        == gpu::INVALID_HANDLE,
        "a zero direct generation is rejected");
    assert(!zero_generation.is_valid(),
        "zero generation is part of the invalid token sentinel");
    assert(@catch(gpu::cmd_barrier(&zero_device_identity, &barrier))
        == gpu::INVALID_HANDLE,
        "a zero device identity is rejected");
    assert(!zero_device_identity.is_valid(),
        "zero device identity is part of the invalid token sentinel");
    assert(@catch(gpu::cmd_barrier(&reserved_device_identity, &barrier))
        == gpu::INVALID_HANDLE,
        "reserved device identity bits are rejected");
    assert(@catch(gpu::cmd_barrier(&stale_generation, &barrier))
        == gpu::INVALID_HANDLE,
        "a stale direct generation is rejected");
    assert(!fixture.command_entered.load(AtomicOrdering.ACQUIRE),
        "malformed selectors never reach the backend");
    gpu::discard_commands(&commands)!!;
    gpu::destroy_device(&device)!!;
}

fn void recording_alias_is_rejected_after_end_phase_transition() @test {
    CommandPinFixture fixture;
    gpu::Device device = gpu::internal::register_device(
        state:  &fixture.state,
        caps:   {},
        vtable: &command_pin_vtable,
    )!!;
    gpu::CommandList commands = begin_command_pin(&device)!!;
    gpu::CommandList recording_alias = commands;
    gpu::ExecutableCommandList executable = gpu::end_commands(&commands)!!;
    gpu::Barrier barrier = {
        .before = { .compute = true },
        .after  = { .compute = true },
    };

    assert(@catch(gpu::cmd_barrier(&recording_alias, &barrier))
        == gpu::COMMAND_RECORDING_ERROR,
        "a recording alias is rejected after the record becomes executable");
    gpu::discard_executable_commands(&executable)!!;
    gpu::destroy_device(&device)!!;
}

fn void failed_command_lifecycle_preserves_record_state() @test {
    CommandPinFixture fixture = {
        .reject_discard = true,
        .reject_end     = true,
    };
    gpu::Device device = gpu::internal::register_device(
        state:  &fixture.state,
        caps:   {},
        vtable: &command_pin_vtable,
    )!!;
    gpu::CommandList commands = begin_command_pin(&device)!!;
    gpu::CommandList original = commands;
    gpu::internal::CommandRecord* record = commands.record;

    fault end_fault = @catch(gpu::end_commands(&commands));
    fault discard_fault = @catch(gpu::discard_commands(&commands));

    assert(end_fault == gpu::DEVICE_BUSY
        && discard_fault == gpu::DEVICE_BUSY,
        "failed end and discard preserve their backend faults");
    assert(commands.record == original.record
        && commands.generation == original.generation,
        "failed lifecycle calls leave the recording token unchanged");
    assert(record.state == gpu::internal::CommandState.RECORDING,
        "failed lifecycle calls leave the record recording");

    fixture.reject_end = false;
    fixture.reject_discard = false;
    gpu::ExecutableCommandList executable = gpu::end_commands(&commands)!!;
    gpu::discard_executable_commands(&executable)!!;
    gpu::destroy_device(&device)!!;
}

fn void executable_alias_of_recording_token_faults_command_state() @test {
    CommandPinFixture fixture;
    gpu::Device device = gpu::internal::register_device(
        state:  &fixture.state,
        caps:   {},
        vtable: &command_pin_vtable,
    )!!;
    gpu::CommandList commands = begin_command_pin(&device)!!;
    gpu::ExecutableCommandList forged = {
        .record          = commands.record,
        .generation      = commands.generation,
        .device_identity = commands.device_identity,
    };

    assert(@catch(gpu::discard_executable_commands(&forged))
        == gpu::COMMAND_RECORDING_ERROR,
        "an executable alias of a recording token faults command state");
    gpu::discard_commands(&commands)!!;
    gpu::destroy_device(&device)!!;
}

fn void failed_record_publication_rolls_back_begin() @test {
    CommandPinFixture fixture;
    gpu::internal::BackendVTable missing_ops = command_pin_vtable;
    missing_ops.command_ops = null;
    gpu::Device device = gpu::internal::register_device(
        state:  &fixture.state,
        caps:   {},
        vtable: &missing_ops,
    )!!;
    gpu::internal::DeviceSlot* slot = &gpu::internal::device_slots[device.index];
    gpu::Queue queue = gpu::get_queue(
        &device,
        gpu::QueueKind.GRAPHICS,
    )!!;

    fault begin_fault = @catch(begin_command_pin(&device));
    gpu::internal::DeviceSlotControl control =
        (gpu::internal::DeviceSlotControl)slot.control.load(AtomicOrdering.ACQUIRE);

    assert(begin_fault == gpu::BACKEND_ERROR,
        "missing command operations prevent record publication");
    assert(control.pins == 0
        && slot.live_children.load(AtomicOrdering.ACQUIRE) == 0,
        "failed publication rolls back its record, child, and retained pin");
    assert(fixture.next_command == 0,
        "failed publication exposes no command record");
    gpu::destroy_device(&device)!!;
}

fn void missing_backend_publication_pointers_roll_back_begin() @test {
    for (uint i = 0; i < 2; i++) {
        CommandPinFixture fixture = {
            .return_null_backend_state   = i == 0,
            .return_null_backend_command = i == 1,
        };
        gpu::Device device = gpu::internal::register_device(
            state:  &fixture.state,
            caps:   {},
            vtable: &command_pin_vtable,
        )!!;
        gpu::internal::DeviceSlot* slot =
            &gpu::internal::device_slots[device.index];

        fault begin_fault = @catch(begin_command_pin(&device));
        gpu::internal::DeviceSlotControl control =
            (gpu::internal::DeviceSlotControl)slot.control.load(
                AtomicOrdering.ACQUIRE,
            );

        assert(begin_fault == gpu::BACKEND_ERROR,
            "missing backend publication pointers fail begin");
        assert(control.pins == 0
            && slot.live_children.load(AtomicOrdering.ACQUIRE) == 0,
            "partial publication rolls back record, child, and retained pin");
        assert(fixture.next_command == 0,
            "partial publication exposes no active record");
        gpu::destroy_device(&device)!!;
    }
}

fn void registration_accepts_per_device_command_operations() @test {
    CommandPinFixture fixture;
    gpu::internal::BackendVTable missing_ops = command_pin_vtable;
    missing_ops.command_ops = null;
    gpu::Device device = gpu::internal::register_device(
        state:       &fixture.state,
        caps:        {},
        vtable:      &missing_ops,
        command_ops: &command_pin_ops,
    )!!;
    gpu::internal::DeviceSlot* slot = &gpu::internal::device_slots[device.index];
    assert(
        slot.data.command_ops == &command_pin_ops,
        "the per-device table overrides the backend default"
    );
    gpu::destroy_device(&device)!!;
}

fn void failed_submit_preserves_command_lifetime_pins() @test {
    CommandPinFixture fixture = { .submit_busy = true };
    gpu::Device device = gpu::internal::register_device(
        state:  &fixture.state,
        caps:   {},
        vtable: &command_pin_vtable,
    )!!;
    gpu::CommandList first = begin_command_pin(&device)!!;
    gpu::CommandList second = begin_command_pin(&device)!!;
    gpu::ExecutableCommandList first_executable = gpu::end_commands(&first)!!;
    gpu::ExecutableCommandList second_executable = gpu::end_commands(&second)!!;
    gpu::ExecutableCommandList[2] lists = {
        first_executable,
        second_executable,
    };
    gpu::SubmitDesc desc = { .command_lists = lists[..] };

    fault submit_fault = @catch(gpu::submit(gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!, &desc));
    gpu::internal::DeviceSlot* slot = &gpu::internal::device_slots[device.index];
    gpu::internal::DeviceSlotControl after_failure =
        (gpu::internal::DeviceSlotControl)slot.control.load(AtomicOrdering.ACQUIRE);
    ulong children_after_failure =
        slot.live_children.load(AtomicOrdering.ACQUIRE);

    fixture.submit_busy = false;
    gpu::submit(gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!, &desc)!!;
    gpu::internal::DeviceSlotControl after_success =
        (gpu::internal::DeviceSlotControl)slot.control.load(AtomicOrdering.ACQUIRE);

    assert(submit_fault == gpu::DEVICE_BUSY,
        "retryable submit failure preserves executable tokens");
    assert(fixture.pins_during_submit == 3,
        "submit observes two command pins and its short operation pin");
    assert(after_failure.pins == 2,
        "failed submit preserves both command lifetime pins");
    assert(children_after_failure == 2,
        "failed submit preserves both command children");
    assert(after_success.pins == 0,
        "successful submit releases both command lifetime pins");
    assert(slot.live_children.load(AtomicOrdering.ACQUIRE) == 0,
        "successful submit releases both command children");
    gpu::destroy_device(&device)!!;
}

fn void device_loss_after_backend_submit_still_consumes_every_token() @test {
    CommandPinFixture fixture = { .lose_during_submit = true };
    gpu::Device device = gpu::internal::register_device(
        state:  &fixture.state,
        caps:   {},
        vtable: &command_pin_vtable,
    )!!;
    gpu::CommandList first = begin_command_pin(&device)!!;
    gpu::CommandList second = begin_command_pin(&device)!!;
    gpu::ExecutableCommandList first_executable = gpu::end_commands(&first)!!;
    gpu::ExecutableCommandList second_executable = gpu::end_commands(&second)!!;
    gpu::ExecutableCommandList[2] lists = {
        first_executable,
        second_executable,
    };
    gpu::SubmitDesc desc = { .command_lists = lists[..] };

    gpu::submit(
        gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!,
        &desc,
    )!!;
    gpu::internal::DeviceSlot* slot = &gpu::internal::device_slots[device.index];
    gpu::internal::DeviceSlotControl control =
        (gpu::internal::DeviceSlotControl)slot.control.load(AtomicOrdering.ACQUIRE);

    assert(!lists[0].is_valid() && !lists[1].is_valid(),
        "successful submit consumes every token after concurrent device loss");
    assert(control.pins == 0,
        "post-submit device loss cannot strand command lifetime pins");
    assert(slot.live_children.load(AtomicOrdering.ACQUIRE) == 0,
        "post-submit device loss cannot strand command children");
    gpu::destroy_device(&device)!!;
}

fn void duplicate_backend_discard_cannot_underflow_command_accounting() @test {
    CommandPinFixture fixture;
    gpu::Device device = gpu::internal::register_device(
        state:  &fixture.state,
        caps:   {},
        vtable: &command_pin_vtable,
    )!!;
    gpu::CommandList commands = begin_command_pin(&device)!!;
    gpu::CommandList duplicate = commands;
    gpu::internal::DeviceSlot* slot = &gpu::internal::device_slots[device.index];

    gpu::discard_commands(&commands)!!;
    fault duplicate_fault = @catch(gpu::discard_commands(&duplicate));
    gpu::internal::DeviceSlotControl after_duplicate =
        (gpu::internal::DeviceSlotControl)slot.control.load(AtomicOrdering.ACQUIRE);

    assert(after_duplicate.pins == 0,
        "a duplicate backend discard cannot underflow retained pins");
    assert(slot.live_children.load(AtomicOrdering.ACQUIRE) == 0,
        "a duplicate backend discard cannot underflow command children");
    assert(duplicate_fault == gpu::INVALID_HANDLE,
        "a consumed command alias is rejected before backend discard");
    gpu::destroy_device(&device)!!;
}

fn void command_record_is_not_reused_before_discard_returns() @test {
    CommandPinFixture fixture = {
        .block_discard       = true,
        .reuse_command_index = true,
    };
    gpu::Device device = gpu::internal::register_device(
        state:  &fixture.state,
        caps:   {},
        vtable: &command_pin_vtable,
    )!!;
    gpu::Queue queue = gpu::get_queue(
        &device,
        gpu::QueueKind.GRAPHICS,
    )!!;
    gpu::CommandList first = begin_command_pin(&device)!!;
    DiscardCommandWorkerArgs worker_args = { .commands = first };
    Thread worker;
    worker.create(&discard_command_worker, &worker_args)!!;
    while (!fixture.discard_entered.load(AtomicOrdering.ACQUIRE)) {
        thread::yield();
    }

    fault busy = @catch(begin_command_pin(&device));
    fixture.discard_release.store(true, AtomicOrdering.RELEASE);
    assert(worker.join()!! == 0 && !worker_args.failed,
        "the prior discard completes before record reuse");
    gpu::CommandList next = begin_command_pin(&device)!!;
    assert(busy == gpu::DEVICE_BUSY,
        "a discarding record remains unavailable");
    assert(next.record == first.record
        && next.generation != first.generation,
        "the backend generation-advances a record only after discard");

    gpu::discard_commands(&next)!!;
    gpu::destroy_device(&device)!!;
}

fn void lost_commands_can_be_discarded() @test {
    CommandPinFixture fixture;
    gpu::Device device = gpu::internal::register_device(
        state:  &fixture.state,
        caps:   {},
        vtable: &command_pin_vtable,
    )!!;
    gpu::CommandList commands = begin_command_pin(&device)!!;
    gpu::internal::mark_device_lost(&device);

    gpu::Barrier barrier = {
        .before = { .compute = true },
        .after = { .compute = true },
    };
    gpu::cmd_barrier(&commands, &barrier)!!;
    fault end_fault = @catch(gpu::end_commands(&commands));
    fault destroy_fault = @catch(gpu::destroy_device(&device));
    uint destroy_calls_before_discard =
        fixture.state.destroy_calls.load(AtomicOrdering.ACQUIRE);
    gpu::discard_commands(&commands)!!;
    gpu::internal::DeviceSlot* slot = &gpu::internal::device_slots[device.index];
    gpu::internal::DeviceSlotControl after_discard =
        (gpu::internal::DeviceSlotControl)slot.control.load(AtomicOrdering.ACQUIRE);

    assert(fixture.command_entered.load(AtomicOrdering.ACQUIRE),
        "recording reaches the trusted backend after device loss");
    assert(end_fault == gpu::DEVICE_LOST,
        "a lifecycle boundary reports loss without consuming the token");
    assert(destroy_fault == gpu::DEVICE_BUSY,
        "a lost device remains busy while a command retains its lifetime pin");
    assert(destroy_calls_before_discard == 0,
        "device teardown does not begin before the lost command is discarded");
    assert(after_discard.pins == 0,
        "discard releases a lost command lifetime pin");
    assert(slot.live_children.load(AtomicOrdering.ACQUIRE) == 0,
        "discard releases a lost command child");
    gpu::destroy_device(&device)!!;
}

fn void lost_executable_commands_can_be_discarded() @test {
    CommandPinFixture fixture;
    gpu::Device device = gpu::internal::register_device(
        state:  &fixture.state,
        caps:   {},
        vtable: &command_pin_vtable,
    )!!;
    gpu::CommandList commands = begin_command_pin(&device)!!;
    gpu::ExecutableCommandList executable = gpu::end_commands(&commands)!!;
    gpu::internal::mark_device_lost(&device);

    gpu::discard_executable_commands(&executable)!!;
    gpu::internal::DeviceSlot* slot =
        &gpu::internal::device_slots[device.index];
    gpu::internal::DeviceSlotControl after_discard =
        (gpu::internal::DeviceSlotControl)slot.control.load(
            AtomicOrdering.ACQUIRE,
        );

    assert(after_discard.pins == 0,
        "executable discard releases a lost command lifetime pin");
    assert(slot.live_children.load(AtomicOrdering.ACQUIRE) == 0,
        "executable discard releases a lost command child");
    gpu::destroy_device(&device)!!;
}

fn void stale_commands_do_not_use_unrelated_lifetime_pins() @test {
    CommandPinFixture fixture = {
        .reject_command = true,
    };
    gpu::Device device = gpu::internal::register_device(
        state:  &fixture.state,
        caps:   {},
        vtable: &command_pin_vtable,
    )!!;
    gpu::CommandList discarded = begin_command_pin(&device)!!;
    gpu::CommandList stale = discarded;
    gpu::discard_commands(&discarded)!!;
    gpu::CommandList live = begin_command_pin(&device)!!;

    StaleCommandWorkerArgs worker_args = { .commands = stale };
    Thread worker;
    worker.create(&stale_command_worker, &worker_args)!!;
    int worker_result = worker.join()!!;
    gpu::discard_commands(&live)!!;
    gpu::destroy_device(&device)!!;

    assert(fixture.pins_during_command == 0,
        "a stale command does not reach backend pin observation");
    assert(!fixture.command_entered.load(AtomicOrdering.ACQUIRE),
        "a stale command alias is rejected before backend access");
    assert(worker_result == 0 && !worker_args.failed,
        "the stale command reports its root validation fault");
    assert(worker_args.result == gpu::INVALID_HANDLE,
        "the stale command remains invalid");
    assert(fixture.state.destroy_calls.load(AtomicOrdering.ACQUIRE) == 1,
        "a stale alias retains no device lifetime pin");
}

fn void stale_command_faults_before_reused_device_callback() @test {
    CommandPinFixture retired_fixture;
    gpu::Device retired = gpu::internal::register_device(
        state:  &retired_fixture.state,
        caps:   {},
        vtable: &command_pin_vtable,
    )!!;
    gpu::CommandList commands = begin_command_pin(&retired)!!;
    gpu::CommandList stale = commands;
    gpu::discard_commands(&commands)!!;
    gpu::Device retired_identity = retired;
    gpu::destroy_device(&retired)!!;

    CommandPinFixture replacement_fixture;
    gpu::Device replacement = gpu::internal::register_device(
        state:  &replacement_fixture.state,
        caps:   {},
        vtable: &command_pin_vtable,
    )!!;
    gpu::Barrier barrier = {
        .before = { .compute = true },
        .after  = { .compute = true },
    };
    gpu::CommandList poisoned = stale;
    poisoned.record = (gpu::internal::CommandRecord*)(uptr)1;
    fault poisoned_fault = @catch(gpu::cmd_barrier(&poisoned, &barrier));
    fault stale_fault = @catch(gpu::cmd_barrier(&stale, &barrier));

    assert(stale.is_valid(),
        "a copied valid-origin token remains distinct from the zero sentinel");
    assert(replacement.index == retired_identity.index
        && replacement.generation != retired_identity.generation,
        "the replacement device reuses the retired static slot generation");
    assert(stale_fault == gpu::INVALID_HANDLE,
        "the retired device identity invalidates the copied token");
    assert(poisoned_fault == gpu::INVALID_HANDLE,
        "the stale device identity faults before record pointer dereference");
    assert(!retired_fixture.command_entered.load(AtomicOrdering.ACQUIRE)
        && !replacement_fixture.command_entered.load(AtomicOrdering.ACQUIRE),
        "stale validation faults before either recording callback");
    gpu::destroy_device(&replacement)!!;
}

fn void stale_command_generation_does_not_use_reused_slot_pin() @test {
    CommandPinFixture fixture = {
        .reject_command      = true,
        .reuse_command_index = true,
    };
    gpu::Device device = gpu::internal::register_device(
        state:  &fixture.state,
        caps:   {},
        vtable: &command_pin_vtable,
    )!!;
    gpu::CommandList first = begin_command_pin(&device)!!;
    gpu::CommandList stale = first;
    gpu::discard_commands(&first)!!;
    gpu::CommandList live = begin_command_pin(&device)!!;

    gpu::Barrier barrier = {
        .before = { .compute = true },
        .after = { .compute = true },
    };
    fault stale_fault = @catch(gpu::cmd_barrier(&stale, &barrier));

    assert(stale.record == live.record,
        "the command slot is reused");
    assert(stale.generation != live.generation,
        "the reused command slot advances its generation");
    assert(fixture.pins_during_command == 0,
        "stale-generation validation does not reach backend pin observation");
    assert(stale_fault == gpu::INVALID_HANDLE,
        "the stale generation remains invalid");
    gpu::discard_commands(&live)!!;
    gpu::destroy_device(&device)!!;
}

fn void device_handles_are_compact_structured_tokens() @test {
    assert(gpu::Device::size == ulong::size, "Device stays one machine word");

    gpu::Device device = {
        .index      = 7,
        .generation = 11,
    };
    assert(device.is_valid(), "a nonzero generation is valid");
    assert(device.index == 7, "the device index round-trips");
    assert(device.generation == 11, "the device generation round-trips");
    assert(device.reserved == 0, "valid devices keep reserved bits clear");
}

fn void forged_device_tokens_are_rejected() @test {
    gpu::Device zero_generation = {
        .index      = 7,
        .generation = 0,
    };
    assert(
        @catch(gpu::get_device_caps(&zero_generation)) == gpu::INVALID_HANDLE,
        "zero-generation device tokens are rejected"
    );

    gpu::Device reserved = {
        .index      = 0,
        .generation = 1,
        .reserved   = 1,
    };
    assert(
        @catch(gpu::get_device_caps(&reserved)) == gpu::INVALID_HANDLE,
        "reserved device bits are rejected"
    );

    gpu::Device out_of_range = {
        .index      = gpu::internal::MAX_DEVICES,
        .generation = 1,
    };
    assert(
        @catch(gpu::get_device_caps(&out_of_range)) == gpu::INVALID_HANDLE,
        "out-of-range device slots are rejected"
    );
}

fn void device_loss_isolated_to_marked_slot() @test {
    DeviceFixture first_fixture;
    DeviceFixture second_fixture;
    gpu::Device first = fixture_device(&first_fixture);
    gpu::Device second = fixture_device(&second_fixture);

    gpu::internal::mark_device_lost(&first);

    assert(
        @catch(gpu::get_device_caps(&first)) == gpu::DEVICE_LOST,
        "the affected device stays lost"
    );
    (void)gpu::get_device_caps(&second)!!;

    gpu::Device stale = first;
    gpu::internal::device_slots[first.index].live_children.store(1, AtomicOrdering.RELAXED);
    gpu::destroy_device(&first)!!;
    DeviceFixture replacement_fixture;
    gpu::Device replacement = fixture_device(&replacement_fixture);
    gpu::internal::mark_device_lost(&stale);
    (void)gpu::get_device_caps(&replacement)!!;

    gpu::destroy_device(&second)!!;
    gpu::destroy_device(&replacement)!!;
}

fn void exhausted_device_slots_are_retired() @test {
    gpu::internal::DeviceSlot* slot = &gpu::internal::device_slots[0];
    gpu::internal::DeviceSlotControl control = (gpu::internal::DeviceSlotControl)slot.control.load(AtomicOrdering.RELAXED);
    assert(control.state == gpu::internal::DEVICE_SLOT_FREE, "the exhaustion fixture starts from a free slot");

    control.generation = gpu::internal::MAX_DEVICE_GENERATION - 1;
    slot.control.store((ulong)control, AtomicOrdering.RELAXED);

    DeviceFixture penultimate_fixture;
    gpu::Device penultimate = fixture_device(&penultimate_fixture);
    assert(penultimate.index == 0, "the seeded slot publishes its current generation");
    assert(penultimate.generation == gpu::internal::MAX_DEVICE_GENERATION - 1, "registration does not advance generation");
    gpu::destroy_device(&penultimate)!!;

    control = (gpu::internal::DeviceSlotControl)slot.control.load(AtomicOrdering.RELAXED);
    assert(control.state == gpu::internal::DEVICE_SLOT_FREE, "the final generation remains publishable");
    assert(control.generation == gpu::internal::MAX_DEVICE_GENERATION, "successful destruction advances to the final generation");

    DeviceFixture final_fixture;
    gpu::Device final = fixture_device(&final_fixture);
    assert(final.index == 0, "the final generation publishes from the seeded slot");
    assert(final.generation == gpu::internal::MAX_DEVICE_GENERATION, "registration preserves the final generation");
    gpu::destroy_device(&final)!!;

    control = (gpu::internal::DeviceSlotControl)slot.control.load(AtomicOrdering.RELAXED);
    assert(control.state == gpu::internal::DEVICE_SLOT_RETIRED, "destruction retires the exhausted slot");
}

fn void concurrent_registry_mutation_keeps_devices_distinct() @test {
    Atomic{uint} ready;
    Atomic{bool} release;
    DeviceRegistryWorkerArgs[DEVICE_REGISTRY_THREAD_COUNT] args;
    Thread[DEVICE_REGISTRY_THREAD_COUNT] threads;

    for (uint i = 0; i < DEVICE_REGISTRY_THREAD_COUNT; i++) {
        args[i].ready = &ready;
        args[i].release = &release;
        threads[i].create(&device_registry_worker, &args[i])!!;
    }
    while (ready.load(AtomicOrdering.ACQUIRE) < DEVICE_REGISTRY_THREAD_COUNT) {
        thread::yield();
    }

    gpu::Device[DEVICE_REGISTRY_THREAD_COUNT] published;
    bool all_published = true;
    bool all_distinct = true;
    for (uint i = 0; i < DEVICE_REGISTRY_THREAD_COUNT; i++) {
        published[i] = args[i].device;
        if (args[i].failed || !published[i].is_valid()) all_published = false;
        for (uint j = 0; j < i; j++) {
            if ((ulong)published[i] == (ulong)published[j]) all_distinct = false;
        }
    }

    release.store(true, AtomicOrdering.RELEASE);
    bool all_joined = true;
    foreach (&worker : threads) {
        if (worker.join()!! != 0) all_joined = false;
    }

    assert(all_published, "every concurrent registration succeeds");
    assert(all_distinct, "concurrent registrations publish distinct devices");
    assert(all_joined, "every concurrent destruction succeeds");
    foreach (&stale : published) {
        assert(
            @catch(gpu::get_device_caps(stale)) == gpu::INVALID_HANDLE,
            "destroyed concurrent devices become stale"
        );
    }
}

fn void registration_proceeds_while_another_device_is_closing() @test {
    BlockingDestroyFixture blocking_fixture;
    blocking_fixture.state.block_destroy = true;
    gpu::Device closing = gpu::internal::register_device(
        state:  &blocking_fixture.state,
        caps:   {},
        vtable: &blocking_device_vtable,
        owner:  &blocking_fixture.state.owner,
    )!!;
    gpu::Device stale = closing;
    gpu::Queue closing_queue = {
        .device = stale,
        .id     = 0,
        .roles  = { .graphics },
    };
    gpu::CompletionPoint closing_point = gpu::internal::make_completion_point(
        closing_queue,
        1,
    )!!;

    DestroyWorkerArgs destroy_args = { .device = closing };
    Thread destroyer;
    destroyer.create(&destroy_device_worker, &destroy_args)!!;
    while (!blocking_fixture.state.destroy_entered.load(AtomicOrdering.ACQUIRE)) {
        thread::yield();
    }
    fault closing_fault = @catch(gpu::get_device_caps(&stale));
    fault poll_fault = @catch(gpu::poll_completion(closing_point));

    gpu::Device concurrent_destroy = stale;
    fault concurrent_destroy_fault = @catch(gpu::destroy_device(&concurrent_destroy));
    bool concurrent_destroy_preserved = concurrent_destroy.is_valid();

    DeviceFixture survivor_fixture;
    gpu::Device survivor;
    bool survivor_created = true;
    gpu::Device? registered = gpu::internal::register_device(
        state:  &survivor_fixture.state,
        caps:   {},
        vtable: &device_fixture_vtable,
    );
    if (catch registration_fault = registered) {
        (void)registration_fault;
        survivor_created = false;
    } else {
        survivor = registered;
    }

    blocking_fixture.state.destroy_release.store(true, AtomicOrdering.RELEASE);
    int destroy_result = destroyer.join()!!;

    assert(survivor_created, "registration succeeds while another slot is closing");
    assert(destroy_result == 0 && !destroy_args.failed, "the closing device finishes destruction");
    assert(closing_fault == gpu::DEVICE_BUSY, "operations reject a closing device with DEVICE_BUSY");
    assert(poll_fault == gpu::DEVICE_BUSY,
        "completion polling rejects a closing device with DEVICE_BUSY");

    assert(concurrent_destroy_fault == gpu::DEVICE_BUSY, "a second destruction attempt is retryable");
    assert(concurrent_destroy_preserved, "a busy destruction attempt preserves its token");
    assert(
        @catch(gpu::get_device_caps(&stale)) == gpu::INVALID_HANDLE,
        "the destroyed device remains stale"
    );
    if (survivor_created) {
        (void)gpu::get_device_caps(&survivor)!!;
        gpu::destroy_device(&survivor)!!;
    }
}

fn void active_operations_make_device_destruction_retryable() @test {
    BlockingOperationFixture fixture;
    fixture.state.block_memory_stats = true;
    gpu::Device device = gpu::internal::register_device(
        state:  &fixture.state,
        caps:   {},
        vtable: &blocking_operation_vtable,
    )!!;

    DeviceOperationWorkerArgs operation_args = { .device = device };
    Thread operation_thread;
    operation_thread.create(&device_operation_worker, &operation_args)!!;
    while (!fixture.state.memory_stats_entered.load(AtomicOrdering.ACQUIRE)) {
        thread::yield();
    }

    fault destroy_fault = @catch(gpu::destroy_device(&device));
    bool handle_preserved = device.is_valid();
    uint calls_before_retry =
        fixture.state.destroy_calls.load(AtomicOrdering.ACQUIRE);

    fixture.state.memory_stats_release.store(true, AtomicOrdering.RELEASE);
    int operation_result = operation_thread.join()!!;

    bool retry_succeeded = false;
    if (device.is_valid()) {
        if (catch retry_fault = gpu::destroy_device(&device)) {
            (void)retry_fault;
        } else {
            retry_succeeded = true;
        }
    }

    assert(destroy_fault == gpu::DEVICE_BUSY, "active operations make destruction retryable");
    assert(handle_preserved, "a busy destruction attempt preserves the device token");
    assert(calls_before_retry == 0, "a busy destruction attempt does not tear down the backend");
    assert(operation_result == 0 && !operation_args.failed, "the pinned operation completes normally");
    assert(retry_succeeded, "destruction succeeds after the active operation releases its pin");
    assert(fixture.state.destroy_calls.load(AtomicOrdering.ACQUIRE) == 1, "only the successful retry tears down the backend");
}

fn void live_children_take_precedence_over_active_operations() @test {
    BlockingOperationFixture fixture;
    fixture.state.block_memory_stats = true;
    gpu::Device device = gpu::internal::register_device(
        state:  &fixture.state,
        caps:   {},
        vtable: &blocking_operation_vtable,
    )!!;
    gpu::internal::DeviceSlot* slot = &gpu::internal::device_slots[device.index];
    slot.live_children.store(1, AtomicOrdering.RELEASE);

    DeviceOperationWorkerArgs operation_args = { .device = device };
    Thread operation_thread;
    operation_thread.create(&device_operation_worker, &operation_args)!!;
    while (!fixture.state.memory_stats_entered.load(AtomicOrdering.ACQUIRE)) {
        thread::yield();
    }

    fault destroy_fault = @catch(gpu::destroy_device(&device));
    fixture.state.memory_stats_release.store(true, AtomicOrdering.RELEASE);
    int operation_result = operation_thread.join()!!;
    slot.live_children.store(0, AtomicOrdering.RELEASE);
    gpu::destroy_device(&device)!!;

    assert(destroy_fault == gpu::RESOURCE_IN_USE,
        "live children take precedence over active operation pins");
    assert(operation_result == 0 && !operation_args.failed,
        "the active operation completes after child rejection");
}

fn void backend_busy_device_destruction_is_retryable() @test {
    BlockingOperationFixture fixture;
    fixture.state.destroy_busy = true;
    gpu::Device device = gpu::internal::register_device(
        state:  &fixture.state,
        caps:   {},
        vtable: &blocking_operation_vtable,
    )!!;
    gpu::Device original = device;

    fault destroy_fault = @catch(gpu::destroy_device(&device));
    gpu::internal::DeviceSlotControl control = (gpu::internal::DeviceSlotControl)gpu::internal::device_slots[device.index].control.load(AtomicOrdering.ACQUIRE);

    assert(destroy_fault == gpu::DEVICE_BUSY, "incomplete backend work makes destruction retryable");
    assert(device == original, "backend busy preserves the device token");
    assert(control.state == gpu::internal::DEVICE_SLOT_LIVE, "backend busy never publishes closing state");
    assert(control.generation == original.generation, "backend busy preserves generation");
    assert(fixture.state.check_calls.load(AtomicOrdering.ACQUIRE) == 1, "the readiness check runs once");
    assert(fixture.state.destroy_calls.load(AtomicOrdering.ACQUIRE) == 0, "backend busy prevents teardown");

    fixture.state.destroy_busy = false;
    gpu::destroy_device(&device)!!;
    assert(fixture.state.check_calls.load(AtomicOrdering.ACQUIRE) == 2, "the retry checks readiness again");
    assert(fixture.state.destroy_calls.load(AtomicOrdering.ACQUIRE) == 1, "only the ready retry tears down the backend");
}

fn void readiness_check_runs_after_new_operations_are_closed() @test {
    BlockingOperationFixture fixture;
    fixture.state.block_check = true;
    gpu::Device device = gpu::internal::register_device(
        state:  &fixture.state,
        caps:   {},
        vtable: &blocking_operation_vtable,
    )!!;
    gpu::Device stale = device;

    DestroyWorkerArgs destroy_args = { .device = device };
    Thread destroyer;
    destroyer.create(&destroy_device_worker, &destroy_args)!!;
    while (!fixture.state.check_entered.load(AtomicOrdering.ACQUIRE)) {
        thread::yield();
    }

    fault operation_fault = @catch(gpu::get_device_caps(&stale));
    fixture.state.check_release.store(true, AtomicOrdering.RELEASE);
    int destroy_result = destroyer.join()!!;

    assert(operation_fault == gpu::DEVICE_BUSY,
        "the readiness check runs only after new operations are closed");
    assert(destroy_result == 0 && !destroy_args.failed,
        "destruction completes after the readiness check");
    assert(fixture.state.destroy_calls.load(AtomicOrdering.ACQUIRE) == 1,
        "the backend tears down once after readiness succeeds");
}

fn void child_published_while_closing_blocks_destruction() @test {
    BlockingOperationFixture fixture;
    fixture.state.publish_child_on_check = true;
    gpu::Device device = gpu::internal::register_device(
        state:  &fixture.state,
        caps:   {},
        vtable: &blocking_operation_vtable,
        owner:  &fixture.state.owner,
    )!!;
    gpu::Device original = device;

    fault destroy_fault = @catch(gpu::destroy_device(&device));
    gpu::internal::DeviceSlot* slot = &gpu::internal::device_slots[original.index];
    gpu::internal::DeviceSlotControl control = (gpu::internal::DeviceSlotControl)slot.control.load(
        AtomicOrdering.ACQUIRE,
    );
    ulong live_children = slot.live_children.load(AtomicOrdering.ACQUIRE);

    if (destroy_fault == gpu::RESOURCE_IN_USE) {
        (void)slot.live_children.sub(1, AtomicOrdering.RELEASE);
        fixture.state.publish_child_on_check = false;
        gpu::destroy_device(&device)!!;
    }

    assert(destroy_fault == gpu::RESOURCE_IN_USE,
        "a child published before closing completes blocks destruction");
    assert(device == gpu::DEVICE_INVALID,
        "the retry succeeds after the raced child is released");
    assert(control.state == gpu::internal::DEVICE_SLOT_LIVE,
        "the failed attempt restores the live state");
    assert(control.generation == original.generation,
        "the failed attempt preserves generation");
    assert(live_children == 1,
        "the late child publication remains owned by the live device");
}

fn void successful_device_destruction_advances_generation() @test {
    DeviceFixture fixture;
    gpu::Device device = fixture_device(&fixture);
    uint index = device.index;
    uint generation = device.generation;

    gpu::destroy_device(&device)!!;

    gpu::internal::DeviceSlotControl control = (gpu::internal::DeviceSlotControl)gpu::internal::device_slots[index].control.load(AtomicOrdering.ACQUIRE);
    assert(control.state == gpu::internal::DEVICE_SLOT_FREE, "successful destruction releases the slot");
    assert(control.generation == generation + 1, "successful destruction advances generation exactly once");
}

fn void runtime_device_retains_are_atomic() @test {
    gpu::RuntimeDesc desc = {
        .application_name = "gpu_device_registry_test",
    };
    gpu::Runtime runtime = gpu::create_runtime(&desc)!!;

    Atomic{uint} ready;
    Atomic{bool} release;
    RuntimeRetainWorkerArgs[DEVICE_REGISTRY_THREAD_COUNT] args;
    Thread[DEVICE_REGISTRY_THREAD_COUNT] threads;
    for (uint i = 0; i < DEVICE_REGISTRY_THREAD_COUNT; i++) {
        args[i].runtime = runtime;
        args[i].ready = &ready;
        args[i].release = &release;
        threads[i].create(&runtime_retain_worker, &args[i])!!;
    }
    while (ready.load(AtomicOrdering.ACQUIRE) < DEVICE_REGISTRY_THREAD_COUNT) {
        thread::yield();
    }

    gpu::internal::RuntimeData* runtime_data = gpu::internal::resolve_runtime(&runtime);
    uint retained_count = runtime_data.live_device_count.load(AtomicOrdering.ACQUIRE);
    release.store(true, AtomicOrdering.RELEASE);

    bool all_joined = true;
    foreach (&worker : threads) {
        if (worker.join()!! != 0) all_joined = false;
    }
    uint final_count = runtime_data.live_device_count.load(AtomicOrdering.ACQUIRE);

    assert(retained_count == DEVICE_REGISTRY_THREAD_COUNT, "concurrent retains lose no updates");
    assert(all_joined, "concurrent releases return without faults");
    assert(final_count == 0, "concurrent releases restore the runtime count");
    gpu::destroy_runtime(&runtime)!!;
}
