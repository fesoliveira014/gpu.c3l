module gpu_test;

import gpu::internal @public;
import gpu;
import gpu::internal::vk @public;
import vk;
import std::core::mem;
import std::core::mem::alloc;
import std::core::mem::allocators;

struct CommandAllocatorCreateProbe {
    uint pool_creates;
    uint pool_destroys;
    uint buffer_allocations;
    uint family;
    uint buffer_count;
    bool fail_pool;
    bool fail_buffers;
}

struct CommandRecordingProbe {
    vk::Result        reset_result;
    vk::Result        begin_result;
    vk::Result        end_result;
    vk::CommandBuffer command_buffer;
    uint              reset_calls;
    uint              begin_calls;
    uint              end_calls;
}

struct FailOnNAllocator (Allocator) {
    uint fail_on;
    uint attempts;
    uint acquires;
    uint releases;
    uint outstanding;
}

fn void*? FailOnNAllocator.acquire(
    &self,
    sz size,
    AllocInitType init_type,
    sz alignment,
) @dynamic {
    self.attempts++;
    if (self.attempts == self.fail_on) return mem::OUT_OF_MEMORY~;
    void* pointer = allocators::LIBC_ALLOCATOR.acquire(
        size,
        init_type,
        alignment,
    )!;
    self.acquires++;
    self.outstanding++;
    return pointer;
}

fn void*? FailOnNAllocator.resize(
    &self,
    void* pointer,
    sz size,
    sz alignment,
) @dynamic {
    return allocators::LIBC_ALLOCATOR.resize(pointer, size, alignment);
}

fn void FailOnNAllocator.release(
    &self,
    void* pointer,
    bool aligned,
) @dynamic {
    self.releases++;
    self.outstanding--;
    allocators::LIBC_ALLOCATOR.release(pointer, aligned);
}

fn vk::CommandPool? probe_create_command_pool(
    void* user_data,
    vk::Device device,
    uint family,
) {
    (void)device;
    CommandAllocatorCreateProbe* probe =
        (CommandAllocatorCreateProbe*)user_data;
    probe.pool_creates++;
    probe.family = family;
    if (probe.fail_pool) return gpu::BACKEND_ERROR~;
    return (vk::CommandPool)(uptr)0x10;
}

fn void probe_destroy_command_pool(
    void* user_data,
    vk::Device device,
    vk::CommandPool pool,
) {
    (void)device;
    (void)pool;
    CommandAllocatorCreateProbe* probe =
        (CommandAllocatorCreateProbe*)user_data;
    probe.pool_destroys++;
}

fn vk::Result probe_allocate_command_buffers(
    void* user_data,
    vk::Device device,
    vk::CommandBufferAllocateInfo* info,
    vk::CommandBuffer* command_buffers,
) {
    (void)device;
    CommandAllocatorCreateProbe* probe =
        (CommandAllocatorCreateProbe*)user_data;
    probe.buffer_allocations++;
    probe.buffer_count = info.command_buffer_count;
    if (probe.fail_buffers) return vk::Result.ERROR_OUT_OF_DEVICE_MEMORY;
    for (uint i = 0; i < info.command_buffer_count; i++) {
        command_buffers[i] = (vk::CommandBuffer)(uptr)(i + 1);
    }
    return vk::Result.SUCCESS;
}

fn vk::Result probe_reset_command_buffer(
    void* user_data,
    vk::CommandBuffer command_buffer,
    vk::CommandBufferResetFlags flags,
) {
    (void)flags;
    CommandRecordingProbe* probe = (CommandRecordingProbe*)user_data;
    probe.reset_calls++;
    probe.command_buffer = command_buffer;
    return probe.reset_result;
}

fn vk::Result probe_begin_command_buffer(
    void* user_data,
    vk::CommandBuffer command_buffer,
    vk::CommandBufferBeginInfo* begin,
) {
    (void)begin;
    CommandRecordingProbe* probe = (CommandRecordingProbe*)user_data;
    probe.begin_calls++;
    probe.command_buffer = command_buffer;
    return probe.begin_result;
}

fn vk::Result probe_end_command_buffer(
    void* user_data,
    vk::CommandBuffer command_buffer,
) {
    CommandRecordingProbe* probe = (CommandRecordingProbe*)user_data;
    probe.end_calls++;
    probe.command_buffer = command_buffer;
    return probe.end_result;
}

fn gpu::internal::vk::CommandRecordingOps command_recording_ops(
    CommandRecordingProbe* probe,
) {
    return {
        .user_data = probe,
        .reset     = &probe_reset_command_buffer,
        .begin     = &probe_begin_command_buffer,
        .end       = &probe_end_command_buffer,
    };
}

fn gpu::internal::vk::CommandAllocatorSlot allocator_slot(uint queue_id) {
    return {
        .queue = {
            .device = { .generation = 1 },
            .id = queue_id,
            .roles = { .graphics },
        },
    };
}

fn void command_allocator_pool_allows_individual_buffer_reset() @test {
    uint flags = (uint)gpu::internal::vk::command_allocator_pool_create_flags();
    assert(
        (flags & (uint)vk::CommandPoolCreateFlagBits.TRANSIENT_BIT) != 0,
        "allocator pools are transient"
    );
    assert(
        (flags & (uint)vk::CommandPoolCreateFlagBits.RESET_COMMAND_BUFFER_BIT)
            != 0,
        "allocator buffers may be reset individually"
    );
}

fn void command_allocator_table_recycles_generation_checked_slots() @test {
    gpu::internal::vk::CommandAllocatorTable table = {
        .owner = 17,
        .slots = mem::new_array(gpu::internal::vk::CommandAllocatorCell, 1),
    };
    defer mem::free(table.slots.ptr);
    gpu::CommandAllocatorHandle first = table.alloc(allocator_slot(3))!!;
    assert(first.owner == 17 && table.get(first)!!.queue.id == 3,
        "allocation publishes exact owner and slot state");

    table.free(first)!!;
    assert(@catch(table.get(first)) == gpu::INVALID_HANDLE,
        "freed allocator generation is stale");
    gpu::CommandAllocatorHandle second = table.alloc(allocator_slot(5))!!;
    assert(second.index == first.index
        && second.generation != first.generation,
        "free-list reuse advances generation");
    assert(table.get(second)!!.queue.id == 5,
        "recycled slot publishes replacement state");
}

fn void command_allocator_table_rejects_foreign_owner() @test {
    gpu::internal::vk::CommandAllocatorTable table = {
        .owner = 29,
        .slots = mem::new_array(gpu::internal::vk::CommandAllocatorCell, 1),
    };
    defer mem::free(table.slots.ptr);
    gpu::CommandAllocatorHandle handle = table.alloc(allocator_slot(1))!!;
    handle.owner++;
    assert(@catch(table.get(handle)) == gpu::INVALID_HANDLE,
        "allocator table rejects a coincident foreign slot");
}

fn void command_claim_cancellation_preserves_concurrent_table_mutation() @test {
    gpu::internal::vk::CommandCell[2] cells;
    gpu::internal::vk::CommandTable table = {
        .slots = cells[..],
        .owner = 41,
    };
    gpu::Device device = { .generation = 1 };
    gpu::internal::CommandRecord* first = table.claim(
        device:        device,
        ops:           null,
        backend_state: null,
        value:         {
            .command_buffer = (vk::CommandBuffer)(uptr)1,
        },
    )!!;
    gpu::internal::CommandListHandle first_identity = first.identity;
    gpu::internal::CommandRecord* concurrent = table.claim(
        device:        device,
        ops:           null,
        backend_state: null,
        value:         {
            .command_buffer = (vk::CommandBuffer)(uptr)2,
        },
    )!!;
    gpu::internal::CommandListHandle concurrent_identity = concurrent.identity;

    table.cancel_claim(first_identity)!!;
    assert(table.get_record(concurrent_identity)!! == concurrent,
        "cancelling an earlier claim preserves the concurrent record");
    gpu::internal::CommandRecord* replacement = table.claim(
        device:        device,
        ops:           null,
        backend_state: null,
        value:         {
            .command_buffer = (vk::CommandBuffer)(uptr)3,
        },
    )!!;
    assert(replacement.identity.index == first_identity.index
        && replacement.identity.generation != first_identity.generation,
        "the cancelled cell rejoins the current free list with new identity");
    assert(table.get_record(concurrent_identity)!! == concurrent
        && concurrent.identity == concurrent_identity,
        "replacement publication leaves the interleaved claim unchanged");
    assert(@catch(table.claim(
        device:        device,
        ops:           null,
        backend_state: null,
        value:         {},
    )) == gpu::SLOT_TABLE_FULL,
        "interleaved cancellation and reuse preserve exact table capacity");
}

fn void command_allocator_table_reports_fixed_capacity() @test {
    gpu::internal::vk::CommandAllocatorTable table = {
        .owner = 41,
        .slots = mem::new_array(gpu::internal::vk::CommandAllocatorCell, 1),
    };
    defer mem::free(table.slots.ptr);
    (void)table.alloc(allocator_slot(1))!!;
    assert(@catch(table.alloc(allocator_slot(2))) == gpu::SLOT_TABLE_FULL,
        "live allocator slots cannot exceed table capacity");
}

fn void command_allocator_table_retires_exhausted_generations() @test {
    gpu::internal::vk::CommandAllocatorCell[1] cells = {{
        .generation = gpu::internal::vk::HANDLE_GENERATION_MAX - 1,
    }};
    gpu::internal::vk::CommandAllocatorTable table = {
        .owner     = 43,
        .slots     = cells[..],
        .free_head = 1,
        .next_new  = 1,
    };
    gpu::CommandAllocatorHandle allocator = table.alloc(
        allocator_slot(1),
    )!!;
    assert(allocator.generation == gpu::internal::vk::HANDLE_GENERATION_MAX,
        "the final allocator generation can be published once");

    table.free(allocator)!!;
    assert(cells[0].generation == gpu::internal::vk::HANDLE_GENERATION_MAX,
        "an exhausted allocator generation remains retired");
    assert(@catch(table.alloc(allocator_slot(2))) == gpu::SLOT_TABLE_FULL,
        "a retired allocator slot is never reused");
}

fn void command_reference_index_capacity_is_power_of_two_and_overflow_safe() @test {
    assert(gpu::internal::vk::command_reference_index_capacity(0)!! == 0
        && gpu::internal::vk::command_reference_index_capacity(1)!! == 2
        && gpu::internal::vk::command_reference_index_capacity(3)!! == 8,
        "the reference index keeps at least half of its cells empty");
    assert(@catch(gpu::internal::vk::command_reference_index_capacity(uint::max))
        == gpu::INVALID_ARGUMENT,
        "reference index capacity rejects unrepresentable powers of two");
}

fn void command_allocator_host_allocation_failure_is_transactional() @test {
    gpu::internal::vk::VkDeviceState state = {
        .host_allocator = &allocators::NULL_ALLOCATOR,
    };
    gpu::CommandAllocatorDesc desc = {
        .command_buffer_capacity = 1,
        .max_resource_references_per_list = 1,
        .max_generated_preprocess_buffers_per_list = 1,
    };
    gpu::internal::vk::CommandAllocatorSlot slot;
    assert(@catch(gpu::internal::vk::initialize_command_allocator_slot(
        state: &state,
        queue: {},
        desc:  &desc,
        slot:  &slot,
    )) == gpu::OUT_OF_HOST_MEMORY,
        "allocator host exhaustion returns a public fault");
    assert(slot.command_buffers.ptr == null
        && slot.scratch.ptr == null
        && slot.reference_storage.ptr == null
        && slot.reference_index_storage.ptr == null
        && slot.preprocess_index_storage.ptr == null
        && slot.available.ptr == null
        && slot.reservations.ptr == null
        && !slot.mutex.is_initialized(),
        "failed host allocation leaves no owned allocator state");
}

fn void command_allocator_native_creation_is_transactional() @test {
    gpu::internal::vk::QueueCompletionState[1] completions = {{
        .native_id = 7ull << 32,
        .timeline  = (vk::Semaphore)(uptr)0x20,
        .bound     = true,
    }};
    gpu::internal::vk::VkDeviceState state = {
        .host_allocator   = &allocators::LIBC_ALLOCATOR,
        .queue_completions = completions[..],
    };
    gpu::CommandAllocatorDesc desc = {
        .command_buffer_capacity = 2,
        .max_resource_references_per_list = 1,
        .max_generated_preprocess_buffers_per_list = 1,
    };
    gpu::Queue queue = { .id = 0 };
    gpu::internal::vk::CommandAllocatorSlot slot;
    CommandAllocatorCreateProbe probe = { .fail_pool = true };
    gpu::internal::vk::CommandPoolOps ops = {
        .user_data = &probe,
        .create    = &probe_create_command_pool,
        .destroy   = &probe_destroy_command_pool,
        .allocate  = &probe_allocate_command_buffers,
    };

    assert(@catch(gpu::internal::vk::initialize_command_allocator_slot_with_ops(
        state: &state,
        queue: queue,
        desc:  &desc,
        slot:  &slot,
        ops:   &ops,
    )) == gpu::BACKEND_ERROR,
        "pool creation failure propagates without publication");
    assert(probe.pool_creates == 1 && probe.pool_destroys == 0
        && probe.buffer_allocations == 0
        && slot.command_buffers.ptr == null,
        "failed pool creation releases every host allocation exactly once");

    probe = { .fail_buffers = true };
    assert(@catch(gpu::internal::vk::initialize_command_allocator_slot_with_ops(
        state: &state,
        queue: queue,
        desc:  &desc,
        slot:  &slot,
        ops:   &ops,
    )) == gpu::OUT_OF_DEVICE_MEMORY,
        "native command-buffer allocation failure remains specific");
    assert(probe.pool_creates == 1 && probe.pool_destroys == 1
        && probe.buffer_allocations == 1 && probe.family == 7
        && probe.buffer_count == 2 && slot.command_buffers.ptr == null,
        "buffer allocation failure destroys the exact-family pool once");

    probe = {};
    gpu::internal::vk::initialize_command_allocator_slot_with_ops(
        state: &state,
        queue: queue,
        desc:  &desc,
        slot:  &slot,
        ops:   &ops,
    )!!;
    assert(probe.pool_creates == 1 && probe.buffer_allocations == 1
        && probe.family == 7 && probe.buffer_count == 2
        && slot.command_buffers[0] != null
        && slot.command_buffers[1] != null,
        "successful creation uses one exact-family pool and one batch allocation");
    gpu::internal::vk::destroy_command_allocator_slot_with_ops(&state, &slot, &ops);
    assert(probe.pool_destroys == 1 && slot.command_buffers.ptr == null,
        "successful slot cleanup destroys the injected pool once");
}

fn void command_allocator_host_failure_sweep_is_transactional() @test {
    gpu::internal::vk::QueueCompletionState[1] completions = {{
        .native_id = 7ull << 32,
        .timeline  = (vk::Semaphore)(uptr)0x20,
        .bound     = true,
    }};
    gpu::CommandAllocatorDesc desc = {
        .command_buffer_capacity = 2,
        .max_resource_references_per_list = 1,
        .max_generated_preprocess_buffers_per_list = 1,
    };
    gpu::Queue queue = { .id = 0 };

    for (uint fail_on = 1; fail_on <= 7; fail_on++) {
        FailOnNAllocator fail_allocator = { .fail_on = fail_on };
        gpu::internal::vk::VkDeviceState state = {
            .host_allocator = &fail_allocator,
            .queue_completions = completions[..],
            .validation_policy = { .track_lifetimes = true },
        };
        gpu::internal::vk::CommandAllocatorSlot slot;
        CommandAllocatorCreateProbe probe;
        gpu::internal::vk::CommandPoolOps ops = {
            .user_data = &probe,
            .create    = &probe_create_command_pool,
            .destroy   = &probe_destroy_command_pool,
            .allocate  = &probe_allocate_command_buffers,
        };

        assert(@catch(gpu::internal::vk::initialize_command_allocator_slot_with_ops(
            state: &state,
            queue: queue,
            desc:  &desc,
            slot:  &slot,
            ops:   &ops,
        )) == gpu::OUT_OF_HOST_MEMORY,
            "each allocator host allocation propagates exact exhaustion");
        assert(fail_allocator.attempts == fail_on
            && fail_allocator.acquires == fail_on - 1
            && fail_allocator.releases == fail_allocator.acquires
            && fail_allocator.outstanding == 0,
            "each partial host allocation set is released exactly once");
        assert(probe.pool_creates == 0 && probe.pool_destroys == 0
            && probe.buffer_allocations == 0,
            "native creation begins only after every host allocation succeeds");
        assert(slot.command_buffers.ptr == null
            && slot.scratch.ptr == null
            && slot.reference_storage.ptr == null
            && slot.reference_index_storage.ptr == null
            && slot.preprocess_index_storage.ptr == null
            && slot.available.ptr == null
            && slot.reservations.ptr == null
            && !slot.mutex.is_initialized(),
            "host rollback leaves no owned allocator state");
    }

    FailOnNAllocator success_allocator = { .fail_on = 8 };
    gpu::internal::vk::VkDeviceState state = {
        .host_allocator = &success_allocator,
        .queue_completions = completions[..],
        .validation_policy = { .track_lifetimes = true },
    };
    gpu::internal::vk::CommandAllocatorSlot slot;
    CommandAllocatorCreateProbe probe;
    gpu::internal::vk::CommandPoolOps ops = {
        .user_data = &probe,
        .create    = &probe_create_command_pool,
        .destroy   = &probe_destroy_command_pool,
        .allocate  = &probe_allocate_command_buffers,
    };
    gpu::internal::vk::initialize_command_allocator_slot_with_ops(
        state: &state,
        queue: queue,
        desc:  &desc,
        slot:  &slot,
        ops:   &ops,
    )!!;
    assert(success_allocator.attempts == 7
        && success_allocator.acquires == 7
        && success_allocator.releases == 0
        && success_allocator.outstanding == 7,
        "successful creation performs the exact fixed host allocation set");
    gpu::internal::vk::destroy_command_allocator_slot_with_ops(&state, &slot, &ops);
    assert(success_allocator.releases == 7
        && success_allocator.outstanding == 0
        && probe.pool_creates == 1 && probe.pool_destroys == 1
        && probe.buffer_allocations == 1,
        "successful cleanup releases every host allocation and native pool");
}

fn void command_allocator_defaults_preallocate_exact_storage() @test {
    gpu::RuntimeDesc runtime_desc = {
        .backend = gpu::BackendKind.VULKAN,
        .application_name = "command_allocator_defaults_test",
    };
    gpu::Device device = create_test_device(&runtime_desc)!!;
    defer destroy_test_device(&device)!!;
    gpu::Queue queue = gpu::get_queue(
        &device,
        gpu::QueueKind.GRAPHICS,
    )!!;
    gpu::CommandAllocatorDesc desc = {
        .debug_name = "graphics_allocator",
    };
    gpu::CommandAllocator allocator = gpu::create_command_allocator(
        &device,
        queue,
        &desc,
    )!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::internal::vk::VkDeviceState* state = (gpu::internal::vk::VkDeviceState*)
        gpu::internal::device_backend_state_ptr(&device);
    gpu::internal::vk::CommandAllocatorSlot* slot = state.command_allocators.get(
        allocator.handle,
    )!!;

    assert(allocator.device == device && allocator.queue == queue,
        "public allocator preserves its exact selected queue");
    assert(slot.queue == queue
        && slot.command_buffers.len
            == gpu::DEFAULT_COMMAND_ALLOCATOR_CAPACITY
        && slot.scratch.len == slot.command_buffers.len
        && slot.available_count == slot.command_buffers.len,
        "zero capacities eagerly create the documented fixed buffer set");
    assert(slot.reference_storage.len == 0
        && slot.reference_index_storage.len == 0
        && slot.scratch[0].reference_index.cells.len == 0,
        "tracking-off allocators retain no resource-reference storage");
    assert(slot.preprocess_index_storage.len
            == gpu::DEFAULT_COMMAND_ALLOCATOR_CAPACITY
                * gpu::DEFAULT_COMMAND_PREPROCESS_PER_LIST
        && slot.reservations.len == slot.preprocess_index_storage.len,
        "generated index and reservation tables use fixed default scaling");
    assert(gpu::internal::vk::slot_name(&slot.debug_name[0]).str_view()
        == "graphics_allocator",
        "allocator debug names are copied into stable backend storage");
}

fn void command_allocator_capacity_and_destroy_are_non_waiting() @test {
    gpu::RuntimeDesc runtime_desc = {
        .backend = gpu::BackendKind.VULKAN,
        .application_name = "command_allocator_capacity_test",
    };
    gpu::Device device = create_test_device(&runtime_desc)!!;
    defer destroy_test_device(&device)!!;
    gpu::Queue queue = gpu::get_queue(
        &device,
        gpu::QueueKind.GRAPHICS,
    )!!;
    gpu::CommandAllocatorDesc desc = {
        .command_buffer_capacity = 1,
    };
    gpu::CommandAllocator allocator = gpu::create_command_allocator(
        &device,
        queue,
        &desc,
    )!!;
    assert(@catch(gpu::destroy_device(&device)) == gpu::RESOURCE_IN_USE,
        "an empty live allocator remains a device child");

    gpu::CommandList commands = gpu::begin_commands(&allocator)!!;
    assert(@catch(gpu::begin_commands(&allocator)) == gpu::DEVICE_BUSY,
        "begin reports temporary exhaustion at fixed buffer capacity");
    assert(@catch(gpu::destroy_command_allocator(&allocator))
        == gpu::RESOURCE_IN_USE,
        "destroy preserves an allocator with a live recording");
    gpu::discard_commands(&commands)!!;

    commands = gpu::begin_commands(&allocator)!!;
    gpu::ExecutableCommandList executable = gpu::end_commands(&commands)!!;
    assert(@catch(gpu::destroy_command_allocator(&allocator))
        == gpu::RESOURCE_IN_USE,
        "destroy preserves an allocator with executable work");
    gpu::discard_executable_commands(&executable)!!;

    gpu::internal::vk::VkDeviceState* state = (gpu::internal::vk::VkDeviceState*)
        gpu::internal::device_backend_state_ptr(&device);
    gpu::internal::vk::reset_completion_work_counters(state);
    gpu::internal::vk::CompletionWorkCounters before =
        gpu::internal::vk::completion_work_counters(state);
    gpu::destroy_command_allocator(&allocator)!!;
    gpu::internal::vk::CompletionWorkCounters after =
        gpu::internal::vk::completion_work_counters(state);
    assert(after.native_counter_queries == before.native_counter_queries
        && after.native_wait_calls == before.native_wait_calls,
        "quiescent allocator destruction performs no query or wait");
    assert(!allocator.handle.is_valid(),
        "successful allocator destruction consumes the public value");
}

fn void command_allocator_destroy_never_waits_for_an_active_operation() @test {
    gpu::RuntimeDesc runtime_desc = {
        .backend          = gpu::BackendKind.VULKAN,
        .application_name = "command_allocator_nonblocking_destroy_test",
    };
    gpu::Device device = create_test_device(&runtime_desc)!!;
    defer destroy_test_device(&device)!!;
    gpu::Queue queue = gpu::get_queue(
        &device,
        gpu::QueueKind.GRAPHICS,
    )!!;
    gpu::CommandAllocator allocator = gpu::create_command_allocator(
        &device,
        queue,
    )!!;
    gpu::CommandAllocator original = allocator;
    gpu::internal::vk::VkDeviceState* state = (gpu::internal::vk::VkDeviceState*)
        gpu::internal::device_backend_state_ptr(&device);
    gpu::internal::vk::CommandAllocatorSlot* slot = state.command_allocators.get(
        allocator.handle,
    )!!;

    slot.mutex.lock()!!;
    fault destroy_fault = @catch(gpu::destroy_command_allocator(&allocator));
    slot.mutex.unlock();

    assert(destroy_fault == gpu::RESOURCE_IN_USE,
        "destroy returns immediately while an allocator operation owns the slot");
    assert(allocator.device == original.device
        && allocator.queue == original.queue
        && allocator.handle == original.handle,
        "nonblocking destroy preserves the public allocator for retry");
    gpu::destroy_command_allocator(&allocator)!!;
}

fn void command_table_exhaustion_restores_allocator_claim() @test {
    gpu::RuntimeDesc runtime_desc = {
        .backend = gpu::BackendKind.VULKAN,
        .application_name = "command_table_exhaustion_test",
    };
    gpu::Device device = create_test_device(&runtime_desc)!!;
    defer destroy_test_device(&device)!!;
    gpu::Queue queue = gpu::get_queue(
        &device,
        gpu::QueueKind.GRAPHICS,
    )!!;
    gpu::CommandAllocatorDesc desc = {
        .command_buffer_capacity = 1,
        .max_resource_references_per_list = 1,
        .max_generated_preprocess_buffers_per_list = 1,
    };
    gpu::CommandAllocator allocator = gpu::create_command_allocator(
        &device,
        queue,
        &desc,
    )!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::internal::vk::VkDeviceState* state = (gpu::internal::vk::VkDeviceState*)
        gpu::internal::device_backend_state_ptr(&device);
    gpu::internal::vk::CommandAllocatorSlot* slot = state.command_allocators.get(
        allocator.handle,
    )!!;
    uint saved_next_new = state.commands.next_new;
    uint saved_free_head = state.commands.free_head;
    uint available_index = slot.available[0];
    gpu::internal::vk::CommandScratch* scratch = &slot.scratch[available_index];

    state.commands.next_new = (uint)state.commands.slots.len;
    state.commands.free_head = 0;
    fault begin_fault = @catch(gpu::begin_commands(&allocator));
    state.commands.next_new = saved_next_new;
    state.commands.free_head = saved_free_head;

    assert(begin_fault == gpu::SLOT_TABLE_FULL,
        "command table exhaustion remains the exact public fault");
    assert(slot.available_count == 1
        && slot.available[0] == available_index
        && slot.live_recordings == 0
        && slot.live_executables == 0
        && slot.in_flight_buffers == 0,
        "failed command publication restores the exact allocator claim");
    assert(!scratch.in_use
        && scratch.reference_count == 0
        && scratch.generated_preprocess_count == 0
        && !slot.recording_thread_set,
        "failed command publication restores scratch and thread ownership");

    gpu::CommandList commands = gpu::begin_commands(&allocator)!!;
    assert(slot.available_count == 0 && slot.live_recordings == 1
        && scratch.in_use && slot.recording_thread_set,
        "the restored allocator buffer is immediately reusable");
    gpu::discard_commands(&commands)!!;
}

fn void native_reset_failure_restores_unpublished_allocator_claim() @test {
    gpu::RuntimeDesc runtime_desc = {
        .backend          = gpu::BackendKind.VULKAN,
        .application_name = "command_reset_failure_test",
    };
    gpu::Device device = create_test_device(&runtime_desc)!!;
    defer destroy_test_device(&device)!!;
    gpu::Queue queue = gpu::get_queue(
        &device,
        gpu::QueueKind.GRAPHICS,
    )!!;
    gpu::CommandAllocatorDesc desc = { .command_buffer_capacity = 1 };
    gpu::CommandAllocator allocator = gpu::create_command_allocator(
        &device,
        queue,
        &desc,
    )!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::internal::vk::VkDeviceState* state = (gpu::internal::vk::VkDeviceState*)
        gpu::internal::device_backend_state_ptr(&device);
    gpu::internal::vk::CommandAllocatorSlot* slot = state.command_allocators.get(
        allocator.handle,
    )!!;
    uint available_index = slot.available[0];
    gpu::internal::vk::CommandScratch* scratch = &slot.scratch[available_index];
    gpu::CommandList token = {
        .record     = (void*)(uptr)91,
        .generation = 5,
    };
    CommandRecordingProbe probe = {
        .reset_result = vk::Result.ERROR_OUT_OF_DEVICE_MEMORY,
    };
    gpu::internal::vk::CommandRecordingOps ops = command_recording_ops(&probe);

    assert(@catch(gpu::internal::vk::begin_commands_with_injected_ops(
        &allocator,
        &token,
        &ops,
    )) == gpu::OUT_OF_DEVICE_MEMORY,
        "native reset failure remains the exact public fault");
    assert(probe.reset_calls == 1 && probe.begin_calls == 0
        && probe.command_buffer == slot.command_buffers[available_index],
        "reset failure stops before native begin on the claimed buffer");
    assert(slot.available_count == 1
        && slot.available[0] == available_index
        && slot.live_recordings == 0
        && slot.live_executables == 0
        && slot.in_flight_buffers == 0
        && !slot.recording_thread_set,
        "reset failure restores allocator availability and thread ownership");
    assert(!scratch.in_use
        && scratch.reference_count == 0
        && scratch.generated_preprocess_count == 0,
        "reset failure restores the exact scratch unit");
    assert(token.record == (void*)(uptr)91 && token.generation == 5,
        "reset failure publishes neither a command record nor a token");

    probe.reset_result = vk::Result.SUCCESS;
    gpu::internal::vk::VkCommandRecord* record =
        gpu::internal::vk::begin_commands_with_injected_ops(
            &allocator,
            &token,
            &ops,
        )!!;
    assert(probe.reset_calls == 2 && probe.begin_calls == 1
        && record.scratch == scratch && scratch.in_use
        && slot.available_count == 0 && slot.live_recordings == 1,
        "the reset-failed allocator unit succeeds on immediate retry");
    gpu::internal::vk::vk_discard_commands(record.root, false)!!;
}

fn void native_begin_failure_restores_unpublished_allocator_claim() @test {
    gpu::RuntimeDesc runtime_desc = {
        .backend          = gpu::BackendKind.VULKAN,
        .application_name = "command_begin_failure_test",
    };
    gpu::Device device = create_test_device(&runtime_desc)!!;
    defer destroy_test_device(&device)!!;
    gpu::Queue queue = gpu::get_queue(
        &device,
        gpu::QueueKind.GRAPHICS,
    )!!;
    gpu::CommandAllocatorDesc desc = { .command_buffer_capacity = 1 };
    gpu::CommandAllocator allocator = gpu::create_command_allocator(
        &device,
        queue,
        &desc,
    )!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::internal::vk::VkDeviceState* state = (gpu::internal::vk::VkDeviceState*)
        gpu::internal::device_backend_state_ptr(&device);
    gpu::internal::vk::CommandAllocatorSlot* slot = state.command_allocators.get(
        allocator.handle,
    )!!;
    uint available_index = slot.available[0];
    gpu::internal::vk::CommandScratch* scratch = &slot.scratch[available_index];
    gpu::CommandList token = {
        .record     = (void*)(uptr)93,
        .generation = 7,
    };
    CommandRecordingProbe probe = {
        .begin_result = vk::Result.ERROR_OUT_OF_DEVICE_MEMORY,
    };
    gpu::internal::vk::CommandRecordingOps ops = command_recording_ops(&probe);

    assert(@catch(gpu::internal::vk::begin_commands_with_injected_ops(
        &allocator,
        &token,
        &ops,
    )) == gpu::OUT_OF_DEVICE_MEMORY,
        "native begin failure remains the exact public fault");
    assert(probe.reset_calls == 1 && probe.begin_calls == 1
        && probe.command_buffer == slot.command_buffers[available_index],
        "begin failure follows one successful reset on the claimed buffer");
    assert(slot.available_count == 1
        && slot.available[0] == available_index
        && slot.live_recordings == 0
        && slot.live_executables == 0
        && slot.in_flight_buffers == 0
        && !slot.recording_thread_set,
        "begin failure restores allocator availability and thread ownership");
    assert(!scratch.in_use
        && scratch.reference_count == 0
        && scratch.generated_preprocess_count == 0,
        "begin failure restores the exact scratch unit");
    assert(token.record == (void*)(uptr)93 && token.generation == 7,
        "begin failure publishes neither a command record nor a token");

    probe.begin_result = vk::Result.SUCCESS;
    gpu::internal::vk::VkCommandRecord* record =
        gpu::internal::vk::begin_commands_with_injected_ops(
            &allocator,
            &token,
            &ops,
        )!!;
    assert(probe.reset_calls == 2 && probe.begin_calls == 2
        && record.scratch == scratch && scratch.in_use
        && slot.available_count == 0 && slot.live_recordings == 1,
        "the begin-failed allocator unit succeeds on immediate retry");
    gpu::internal::vk::vk_discard_commands(record.root, false)!!;
}

fn void native_end_failure_preserves_recording_for_retry() @test {
    gpu::RuntimeDesc runtime_desc = {
        .backend          = gpu::BackendKind.VULKAN,
        .application_name = "command_end_failure_test",
    };
    gpu::Device device = create_test_device(&runtime_desc)!!;
    defer destroy_test_device(&device)!!;
    gpu::Queue queue = gpu::get_queue(
        &device,
        gpu::QueueKind.GRAPHICS,
    )!!;
    gpu::CommandAllocatorDesc desc = { .command_buffer_capacity = 1 };
    gpu::CommandAllocator allocator = gpu::create_command_allocator(
        &device,
        queue,
        &desc,
    )!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::internal::vk::VkDeviceState* state = (gpu::internal::vk::VkDeviceState*)
        gpu::internal::device_backend_state_ptr(&device);
    gpu::internal::vk::CommandAllocatorSlot* slot = state.command_allocators.get(
        allocator.handle,
    )!!;
    uint available_index = slot.available[0];
    gpu::internal::vk::CommandScratch* scratch = &slot.scratch[available_index];
    CommandRecordingProbe probe;
    gpu::internal::vk::CommandRecordingOps ops = command_recording_ops(&probe);
    gpu::CommandList token;
    gpu::internal::vk::VkCommandRecord* record =
        gpu::internal::vk::begin_commands_with_injected_ops(
            &allocator,
            &token,
            &ops,
        )!!;
    gpu::internal::CommandListHandle handle = record.root.identity;
    void* token_record = token.record;
    uint token_generation = token.generation;
    uint next_new = state.commands.next_new;
    uint free_head = state.commands.free_head;
    probe.end_result = vk::Result.ERROR_OUT_OF_DEVICE_MEMORY;

    assert(@catch(gpu::internal::vk::end_commands_with_injected_ops(
        state,
        record,
        &ops,
    )) == gpu::OUT_OF_DEVICE_MEMORY,
        "native end failure remains the exact public fault");
    assert(probe.end_calls == 1
        && probe.command_buffer == slot.command_buffers[available_index],
        "end failure targets the exact allocator-owned native buffer");
    assert(record.root.state == gpu::internal::CommandState.RECORDING
        && record.allocator == allocator.handle
        && record.allocator_buffer == available_index
        && record.scratch == scratch,
        "end failure leaves the command record recording and unchanged");
    assert(slot.available_count == 0
        && slot.live_recordings == 1
        && slot.live_executables == 0
        && slot.in_flight_buffers == 0
        && slot.recording_thread_set
        && scratch.in_use
        && scratch.reference_count == 0
        && scratch.generated_preprocess_count == 0,
        "end failure preserves allocator, thread, and scratch ownership");
    assert(state.commands.next_new == next_new
        && state.commands.free_head == free_head
        && state.commands.get(handle)!! == record
        && token.record == token_record
        && token.generation == token_generation,
        "end failure preserves the published recording token and table entry");

    probe.end_result = vk::Result.SUCCESS;
    gpu::internal::vk::end_commands_with_injected_ops(state, record, &ops)!!;
    assert(probe.end_calls == 2
        && record.root.state == gpu::internal::CommandState.EXECUTABLE
        && slot.live_recordings == 0
        && slot.live_executables == 1
        && !slot.recording_thread_set
        && scratch.in_use,
        "the same recording succeeds on immediate end retry");
    gpu::internal::vk::vk_discard_commands(record.root, true)!!;
}

fn void tracking_allocator_wires_non_overlapping_reference_slices() @test {
    gpu::RuntimeDesc runtime_desc = {
        .backend                  = gpu::BackendKind.VULKAN,
        .contract_validation      = gpu::ContractValidation.FULL,
        .track_resource_lifetimes = true,
        .application_name         = "command_allocator_tracking_test",
    };
    gpu::Device device = create_test_device(&runtime_desc)!!;
    defer destroy_test_device(&device)!!;
    gpu::Queue queue = gpu::get_queue(
        &device,
        gpu::QueueKind.GRAPHICS,
    )!!;
    gpu::CommandAllocatorDesc desc = {
        .command_buffer_capacity          = 2,
        .max_resource_references_per_list = 3,
    };
    gpu::CommandAllocator allocator = gpu::create_command_allocator(
        &device,
        queue,
        &desc,
    )!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::internal::vk::VkDeviceState* state = (gpu::internal::vk::VkDeviceState*)
        gpu::internal::device_backend_state_ptr(&device);
    gpu::internal::vk::CommandAllocatorSlot* slot = state.command_allocators.get(
        allocator.handle,
    )!!;

    assert(slot.reference_storage.len == 6
        && slot.reference_index_storage.len == 16
        && slot.scratch[0].references.len == 3
        && slot.scratch[1].references.len == 3
        && slot.scratch[0].reference_index.cells.len == 8
        && slot.scratch[1].reference_index.cells.len == 8,
        "tracking storage scales as buffers times references and index cells");
    assert(slot.scratch[0].references.ptr == slot.reference_storage.ptr
        && slot.scratch[1].references.ptr == &slot.reference_storage[3]
        && slot.scratch[0].reference_index.cells.ptr
            == slot.reference_index_storage.ptr
        && slot.scratch[1].reference_index.cells.ptr
            == &slot.reference_index_storage[8],
        "each command buffer receives stable non-overlapping reference storage");
}
