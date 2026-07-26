module gpu::internal::vk @private;

import gpu::internal @public;
import gpu @public;
import vk;
import vma;
import std::thread;
import std::core::env;
import std::atomic;
import std::atomic::types;

alias CreateCommandPoolFn @private = fn vk::CommandPool? (void*, vk::Device, uint);
alias DestroyCommandPoolFn @private = fn void (void*, vk::Device, vk::CommandPool);
alias AllocateCommandBuffersFn @private = fn vk::Result (
    void*,
    vk::Device,
    vk::CommandBufferAllocateInfo*,
    vk::CommandBuffer*,
);
alias ResetCommandBufferFn @private = fn vk::Result (
    void*,
    vk::CommandBuffer,
    vk::CommandBufferResetFlags,
);
alias BeginCommandBufferFn @private = fn vk::Result (
    void*,
    vk::CommandBuffer,
    vk::CommandBufferBeginInfo*,
);
alias EndCommandBufferFn @private = fn vk::Result (
    void*,
    vk::CommandBuffer,
);

const uint MAX_COMMAND_ALLOCATORS = 256;
const uint COMMAND_ALLOCATOR_FREE_NONE = 0;

struct GeneratedPreprocessBuffer {
    vk::Buffer             buffer;
    vma::Allocation        allocation;
    vk::DeviceAddress      address;
    vk::DeviceSize         size;
    gpu::PipelineHandle    reservation_pipeline;
    gpu::GeneratedWorkKind reservation_kind;
    uint                   reservation_max_commands;
    gpu::CommandAllocatorHandle reservation_allocator;
    uint                   reservation_index;
    bool                   reservation_owned;
    bool                   reservation_in_use;
}

struct PreparedBufferCopy {
    ResolvedSpan src;
    ResolvedSpan dst;
    gpu::GpuSpan src_span;
    gpu::GpuSpan dst_span;
}

struct PreparedBufferFill {
    ResolvedSpan span;
    gpu::GpuSpan source;
    uint         value;
}

struct PreparedBufferTextureCopy {
    ResolvedSpan      buffer;
    gpu::GpuSpan      buffer_span;
    TextureSlot*      texture;
    gpu::TextureHandle texture_handle;
    vk::BufferImageCopy region;
}

struct PreparedDispatchIndirect {
    BoundPipeline   pipeline;
    ResolvedSpan    args;
    gpu::GpuSpan    args_span;
    gpu::GpuAddress root;
}

struct PreparedGeneratedDispatch {
    BoundPipeline                  pipeline;
    ResolvedSpan                   records;
    ResolvedSpan                   count;
    gpu::GpuSpan                   records_span;
    gpu::GpuSpan                   count_span;
    vk::IndirectCommandsLayoutEXT layout;
    uint                           max_count;
}

struct RootPushObservation @if(env::TESTING) {
    gpu::GpuAddress compute;
    ulong           compute_layout;
    gpu::GpuAddress vertex;
    gpu::GpuAddress fragment;
}

Atomic{ulong} last_compute_root @if(env::TESTING);
Atomic{ulong} last_compute_layout @if(env::TESTING);
Atomic{ulong} last_vertex_root @if(env::TESTING);
Atomic{ulong} last_fragment_root @if(env::TESTING);

struct CommandPoolOps @private {
    void*                    user_data;
    CreateCommandPoolFn      create;
    DestroyCommandPoolFn     destroy;
    AllocateCommandBuffersFn allocate;
}

struct CommandRecordingOps @private {
    void*                user_data;
    ResetCommandBufferFn reset;
    BeginCommandBufferFn begin;
    EndCommandBufferFn   end;
}

struct CommandScratch {
    CommandResourceReference[] references;
    uint                       reference_count;
    uint[]                     generated_preprocess_indices;
    uint                       generated_preprocess_count;
    bool                       in_use;
}

struct CommandAllocatorSlot {
    gpu::Queue                   queue;
    vk::CommandPool             pool;
    vk::CommandBuffer[]          command_buffers;
    CommandScratch[]             scratch;
    CommandResourceReference[]   reference_storage;
    uint[]                       preprocess_index_storage;
    uint[]                       available;
    uint                         available_count;
    GeneratedPreprocessBuffer[]  reservations;
    usz                          generated_preprocess_budget;
    usz                          generated_preprocess_committed;
    uint                         generated_reservations_in_use;
    uint                         live_recordings;
    uint                         live_executables;
    uint                         in_flight_buffers;
    Atomic{uint}                 resolution_pins;
    RecordingThreadIdentity      recording_thread_identity;
    bool                         recording_thread_set;
    Mutex                        mutex;
    char[DEBUG_NAME_CAP]         debug_name;
}

struct CommandAllocatorCell {
    CommandAllocatorSlot value;
    uint                 generation;
    uint                 next_free;
    bool                 used;
}

struct CommandAllocatorTable {
    CommandAllocatorCell[] slots;
    ulong                  owner;
    uint                   free_head;
    uint                   next_new;
}

fn gpu::CommandAllocatorHandle? CommandAllocatorTable.alloc(
    &self,
    CommandAllocatorSlot value,
) {
    uint index;
    bool found;
    while (self.free_head != COMMAND_ALLOCATOR_FREE_NONE) {
        index = self.free_head - 1;
        self.free_head = self.slots[index].next_free;
        if (can_allocate_handle_generation(self.slots[index].generation)) {
            found = true;
            break;
        }
        self.slots[index].next_free = COMMAND_ALLOCATOR_FREE_NONE;
    }
    if (!found) {
        if (self.next_new >= self.slots.len) return gpu::SLOT_TABLE_FULL~;
        index = self.next_new++;
        self.slots[index].generation = 0;
    }
    CommandAllocatorCell* cell = &self.slots[index];
    cell.value = value;
    cell.generation = next_handle_generation(cell.generation)!;
    cell.used = true;
    return {
        .owner      = self.owner,
        .index      = index,
        .generation = cell.generation,
    };
}

fn CommandAllocatorSlot*? CommandAllocatorTable.get(
    &self,
    gpu::CommandAllocatorHandle handle,
) {
    if (handle.owner != self.owner || handle.index >= self.next_new) {
        return gpu::INVALID_HANDLE~;
    }
    CommandAllocatorCell* cell = &self.slots[handle.index];
    if (!cell.used || cell.generation != handle.generation) {
        return gpu::INVALID_HANDLE~;
    }
    return &cell.value;
}

fn void? CommandAllocatorTable.free(
    &self,
    gpu::CommandAllocatorHandle handle,
) {
    self.get(handle)!;
    CommandAllocatorCell* cell = &self.slots[handle.index];
    cell.value = {};
    cell.used = false;
    if (!should_recycle_handle_generation(cell.generation)) {
        cell.generation = HANDLE_GENERATION_MAX;
        cell.next_free = COMMAND_ALLOCATOR_FREE_NONE;
        return;
    }
    cell.generation = next_handle_generation(cell.generation)!;
    cell.next_free = self.free_head;
    self.free_head = handle.index + 1;
}

fn void reset_root_push_observation() @if(env::TESTING) {
    last_compute_root.store(0, AtomicOrdering.RELAXED);
    last_compute_layout.store(0, AtomicOrdering.RELAXED);
    last_vertex_root.store(0, AtomicOrdering.RELAXED);
    last_fragment_root.store(0, AtomicOrdering.RELAXED);
}

fn RootPushObservation root_push_observation() @if(env::TESTING) {
    return {
        .compute = (gpu::GpuAddress)last_compute_root.load(
            AtomicOrdering.RELAXED,
        ),
        .compute_layout = last_compute_layout.load(AtomicOrdering.RELAXED),
        .vertex = (gpu::GpuAddress)last_vertex_root.load(
            AtomicOrdering.RELAXED,
        ),
        .fragment = (gpu::GpuAddress)last_fragment_root.load(
            AtomicOrdering.RELAXED,
        ),
    };
}

fn vk::CommandPoolCreateFlags command_allocator_pool_create_flags() {
    return (vk::CommandPoolCreateFlags)(
        (uint)vk::CommandPoolCreateFlagBits.TRANSIENT_BIT
        | (uint)vk::CommandPoolCreateFlagBits.RESET_COMMAND_BUFFER_BIT
    );
}

fn vk::CommandPool? create_command_pool_real(void* user_data, vk::Device device, uint family) @private {
    VkDeviceState* state = (VkDeviceState*)user_data;
    vk::CommandPoolCreateInfo info = vk::command_pool_create_info()
        .set_queue_family_index(family)
        .set_flags(command_allocator_pool_create_flags());
    vk::CommandPool pool;
    check_device_result(state, vk::create_command_pool(
        device:        device,
        p_create_info: &info,
        p_allocator:   null,
        p_command_pool: &pool,
    ))!;
    return pool;
}

fn void destroy_command_pool_real(void* user_data, vk::Device device, vk::CommandPool pool) @private {
    vk::destroy_command_pool(device, pool, null);
}

fn vk::Result allocate_command_buffers_real(
    void* user_data,
    vk::Device device,
    vk::CommandBufferAllocateInfo* info,
    vk::CommandBuffer* command_buffers,
) @private {
    return vk::allocate_command_buffers(device, info, command_buffers);
}

fn CommandPoolOps command_pool_ops_real(VkDeviceState* state) @private {
    return {
        .user_data = state,
        .create    = &create_command_pool_real,
        .destroy   = &destroy_command_pool_real,
        .allocate  = &allocate_command_buffers_real,
    };
}

fn void destroy_command_allocator_slot_with_ops(
    VkDeviceState* state,
    CommandAllocatorSlot* slot,
    CommandPoolOps* ops,
) @private {
    free_reserved_generated_preprocess_buffers(
        state,
        slot.reservations,
    );
    if (slot.pool != null) ops.destroy(ops.user_data, state.device, slot.pool);
    if (slot.command_buffers.ptr != null) {
        alloc::free(state.host_allocator, slot.command_buffers.ptr);
    }
    if (slot.scratch.ptr != null) alloc::free(state.host_allocator, slot.scratch.ptr);
    if (slot.reference_storage.ptr != null) {
        alloc::free(state.host_allocator, slot.reference_storage.ptr);
    }
    if (slot.preprocess_index_storage.ptr != null) {
        alloc::free(state.host_allocator, slot.preprocess_index_storage.ptr);
    }
    if (slot.available.ptr != null) alloc::free(state.host_allocator, slot.available.ptr);
    if (slot.mutex.is_initialized()) (void)slot.mutex.destroy();
    *slot = {};
}

fn void destroy_command_allocator_slot(
    VkDeviceState* state,
    CommandAllocatorSlot* slot,
) @private {
    CommandPoolOps ops = command_pool_ops_real(state);
    destroy_command_allocator_slot_with_ops(state, slot, &ops);
}

fn uint allocator_reservation_capacity(gpu::CommandAllocatorDesc* desc) {
    return desc.command_buffer_capacity
        * desc.max_generated_preprocess_buffers_per_list;
}

fn void? initialize_command_allocator_slot_with_ops(
    VkDeviceState* state,
    gpu::Queue queue,
    gpu::CommandAllocatorDesc* desc,
    CommandAllocatorSlot* slot,
    CommandPoolOps* ops,
) @private {
    *slot = {};
    defer catch destroy_command_allocator_slot_with_ops(state, slot, ops);
    slot.queue = queue;
    slot.generated_preprocess_budget = desc.generated_preprocess_bytes;
    copy_debug_name(&slot.debug_name[0], desc.debug_name);

    uint capacity = desc.command_buffer_capacity;
    vk::CommandBuffer[]? command_buffers = alloc::new_array_try(
        state.host_allocator,
        vk::CommandBuffer,
        capacity,
    );
    if (catch command_buffers) {
        return gpu::OUT_OF_HOST_MEMORY~;
    }
    slot.command_buffers = command_buffers;
    CommandScratch[]? scratch = alloc::new_array_try(
        state.host_allocator,
        CommandScratch,
        capacity,
    );
    if (catch scratch) return gpu::OUT_OF_HOST_MEMORY~;
    slot.scratch = scratch;
    uint[]? available = alloc::new_array_try(
        state.host_allocator,
        uint,
        capacity,
    );
    if (catch available) return gpu::OUT_OF_HOST_MEMORY~;
    slot.available = available;

    uint reference_capacity =
        contract_uses_full_validation(state.validation_policy)
        ? desc.max_resource_references_per_list
        : 0;
    if (reference_capacity > 0) {
        CommandResourceReference[]? reference_storage = alloc::new_array_try(
            state.host_allocator,
            CommandResourceReference,
            (usz)capacity * (usz)reference_capacity,
        );
        if (catch reference_storage) {
            return gpu::OUT_OF_HOST_MEMORY~;
        }
        slot.reference_storage = reference_storage;
    }
    uint preprocess_capacity =
        desc.max_generated_preprocess_buffers_per_list;
    if (preprocess_capacity > 0) {
        uint[]? preprocess_index_storage = alloc::new_array_try(
            state.host_allocator,
            uint,
            (usz)capacity * (usz)preprocess_capacity,
        );
        if (catch preprocess_index_storage) {
            return gpu::OUT_OF_HOST_MEMORY~;
        }
        slot.preprocess_index_storage = preprocess_index_storage;
        GeneratedPreprocessBuffer[]? reservations = alloc::new_array_try(
            state.host_allocator,
            GeneratedPreprocessBuffer,
            allocator_reservation_capacity(desc),
        );
        if (catch reservations) {
            return gpu::OUT_OF_HOST_MEMORY~;
        }
        slot.reservations = reservations;
    }
    for (uint i = 0; i < capacity; i++) {
        if (reference_capacity > 0) {
            usz start = (usz)i * (usz)reference_capacity;
            slot.scratch[i].references = slot.reference_storage[
                start:reference_capacity
            ];
        }
        if (preprocess_capacity > 0) {
            usz start = (usz)i * (usz)preprocess_capacity;
            slot.scratch[i].generated_preprocess_indices =
                slot.preprocess_index_storage[start:preprocess_capacity];
        }
        slot.available[i] = capacity - 1 - i;
    }
    slot.available_count = capacity;

    QueueCompletionState* completion = queue_completion_state(
        state,
        queue.id,
    )!;
    slot.pool = ops.create(
        ops.user_data,
        state.device,
        native_queue_family(completion.native_id),
    )!;
    vk::CommandBufferAllocateInfo allocate_info =
        vk::command_buffer_allocate_info()
            .set_command_pool(slot.pool)
            .set_level(vk::CommandBufferLevel.PRIMARY)
            .set_command_buffer_count(capacity);
    check_device_result(state, ops.allocate(
        ops.user_data,
        state.device,
        &allocate_info,
        slot.command_buffers.ptr,
    ))!;
}

fn void? initialize_command_allocator_slot(
    VkDeviceState* state,
    gpu::Queue queue,
    gpu::CommandAllocatorDesc* desc,
    CommandAllocatorSlot* slot,
) @private {
    CommandPoolOps ops = command_pool_ops_real(state);
    return initialize_command_allocator_slot_with_ops(
        state: state,
        queue: queue,
        desc:  desc,
        slot:  slot,
        ops:   &ops,
    );
}

fn gpu::CommandAllocatorHandle? vk_create_command_allocator(
    VkDeviceState* state,
    gpu::Queue queue,
    gpu::CommandAllocatorDesc* desc,
) {
    CommandAllocatorSlot slot;
    initialize_command_allocator_slot(
        state: state,
        queue: queue,
        desc:  desc,
        slot:  &slot,
    )!;
    bool owns_slot = true;
    defer catch {
        if (owns_slot) destroy_command_allocator_slot(state, &slot);
    }
    gpu::CommandAllocatorHandle handle;
    {
        state.resource_mutex.lock()!!;
        defer state.resource_mutex.unlock();
        handle = state.command_allocators.alloc(slot)!;
        slot = {};
        owns_slot = false;
        CommandAllocatorSlot* published = state.command_allocators.get(
            handle,
        )!!;
        if (catch published.mutex.init()) {
            destroy_command_allocator_slot(state, published);
            state.command_allocators.free(handle)!!;
            return gpu::BACKEND_ERROR~;
        }
    }
    return handle;
}

fn void? vk_destroy_command_allocator(
    VkDeviceState* state,
    gpu::CommandAllocatorHandle handle,
) {
    CommandAllocatorSlot removed;
    {
        state.resource_mutex.lock()!!;
        defer state.resource_mutex.unlock();
        CommandAllocatorSlot* slot = state.command_allocators.get(handle)!;
        if (slot.resolution_pins.load(AtomicOrdering.ACQUIRE) != 0) {
            return gpu::RESOURCE_IN_USE~;
        }
        if (!slot.mutex.try_lock()) return gpu::RESOURCE_IN_USE~;
        if (slot.live_recordings != 0
            || slot.live_executables != 0
            || slot.in_flight_buffers != 0
            || slot.generated_reservations_in_use != 0
            || slot.available_count != slot.command_buffers.len) {
            slot.mutex.unlock();
            return gpu::RESOURCE_IN_USE~;
        }
        slot.mutex.unlock();
        (void)slot.mutex.destroy();
        removed = *slot;
        state.command_allocators.free(handle)!;
    }
    destroy_command_allocator_slot(state, &removed);
}

fn bool command_allocator_quiescent_locked(CommandAllocatorSlot* slot) {
    return slot.live_recordings == 0
        && slot.live_executables == 0
        && slot.in_flight_buffers == 0
        && slot.generated_reservations_in_use == 0
        && slot.available_count == slot.command_buffers.len;
}

fn CommandAllocatorSlot*? resolve_command_allocator(
    VkDeviceState* state,
    gpu::CommandAllocator* allocator,
) {
    if (allocator == null || allocator.device != state.owner) {
        return gpu::INVALID_HANDLE~;
    }
    CommandAllocatorSlot* slot = state.command_allocators.get(
        allocator.handle,
    )!;
    if (slot.queue != allocator.queue) return gpu::INVALID_HANDLE~;
    return slot;
}

fn CommandAllocatorSlot*? lock_command_allocator(
    VkDeviceState* state,
    gpu::CommandAllocator* allocator,
) {
    CommandAllocatorSlot* slot;
    {
        state.resource_mutex.lock()!!;
        defer state.resource_mutex.unlock();
        slot = resolve_command_allocator(state, allocator)!;
        (void)slot.resolution_pins.add(1, AtomicOrdering.RELEASE);
    }
    bool pinned = true;
    defer catch {
        if (pinned) {
            (void)slot.resolution_pins.sub(1, AtomicOrdering.RELEASE);
        }
    }
    slot.mutex.lock()!;
    (void)slot.resolution_pins.sub(1, AtomicOrdering.RELEASE);
    pinned = false;
    return slot;
}

fn CommandAllocatorSlot*? lock_command_allocator_handle(
    VkDeviceState* state,
    gpu::CommandAllocatorHandle handle,
) {
    CommandAllocatorSlot* slot;
    {
        state.resource_mutex.lock()!!;
        defer state.resource_mutex.unlock();
        slot = state.command_allocators.get(handle)!;
        (void)slot.resolution_pins.add(1, AtomicOrdering.RELEASE);
    }
    bool pinned = true;
    defer catch {
        if (pinned) {
            (void)slot.resolution_pins.sub(1, AtomicOrdering.RELEASE);
        }
    }
    slot.mutex.lock()!;
    (void)slot.resolution_pins.sub(1, AtomicOrdering.RELEASE);
    pinned = false;
    return slot;
}

fn CommandAllocatorSlot*? lock_command_allocator_for_pipeline(
    VkDeviceState* state,
    gpu::CommandAllocator* allocator,
    gpu::PipelineHandle pipeline,
) {
    CommandAllocatorSlot* slot;
    {
        state.resource_mutex.lock()!!;
        defer state.resource_mutex.unlock();
        slot = resolve_command_allocator(state, allocator)!;
        (void)state.pipelines.get(pipeline)!;
        (void)slot.resolution_pins.add(1, AtomicOrdering.RELEASE);
    }
    bool pinned = true;
    defer catch {
        if (pinned) {
            (void)slot.resolution_pins.sub(1, AtomicOrdering.RELEASE);
        }
    }
    slot.mutex.lock()!;
    (void)slot.resolution_pins.sub(1, AtomicOrdering.RELEASE);
    pinned = false;
    return slot;
}

fn vk::MemoryRequirements? generated_scratch_requirements(
    VkDeviceState* state,
    gpu::GeneratedScratchDesc* desc,
) {
    PipelineSlot* slot = state.pipelines.get(desc.pipeline)!;
    if (slot.entry >= state.pipeline_cache.entries.len
        || !state.pipeline_cache.entries[slot.entry].used) {
        return gpu::INVALID_HANDLE~;
    }
    PipelineKind pipeline_kind = (PipelineKind)state.pipeline_cache
        .entries[slot.entry].key.kind;
    vk::IndirectCommandsLayoutEXT layout;
    switch (desc.kind) {
        case gpu::GeneratedWorkKind.DRAW:
            if (pipeline_kind != PipelineKind.GRAPHICS) {
                return gpu::INVALID_ARGUMENT~;
            }
            layout = state.generated_draw_layout;
        case gpu::GeneratedWorkKind.DRAW_INDEXED:
            if (pipeline_kind != PipelineKind.GRAPHICS) {
                return gpu::INVALID_ARGUMENT~;
            }
            layout = state.generated_indexed_draw_layout;
        case gpu::GeneratedWorkKind.DISPATCH:
            if (pipeline_kind != PipelineKind.COMPUTE) {
                return gpu::INVALID_ARGUMENT~;
            }
            layout = slot.generated_dispatch_layout;
        default:
            return gpu::INVALID_ARGUMENT~;
    }
    if (layout == null) return gpu::BACKEND_ERROR~;

    vk::GeneratedCommandsPipelineInfoEXT pipeline_info =
        vk::generated_commands_pipeline_info_ext()
            .set_pipeline(slot.pipeline);
    vk::GeneratedCommandsMemoryRequirementsInfoEXT requirements_info =
        vk::generated_commands_memory_requirements_info_ext()
            .set_next(&pipeline_info)
            .set_indirect_commands_layout(layout)
            .set_max_sequence_count(desc.max_commands_per_list);
    vk::MemoryRequirements2 requirements2 = vk::memory_requirements2();
    state.device_dispatch.generated_work.get_memory_requirements(
        state.device,
        &requirements_info,
        &requirements2,
    );
    vk::MemoryRequirements requirements = requirements2.memory_requirements;
    if (requirements.size > 0
        && (requirements.alignment == 0
            || requirements.memory_type_bits == 0)) {
        return gpu::BACKEND_ERROR~;
    }
    return requirements;
}

fn bool generated_reservation_matches(
    GeneratedPreprocessBuffer* buffer,
    gpu::PipelineHandle pipeline,
    gpu::GeneratedWorkKind kind,
) {
    return buffer.reservation_pipeline == pipeline
        && buffer.reservation_kind == kind;
}

fn void? preflight_generated_reservation(
    CommandAllocatorSlot* slot,
    gpu::PipelineHandle pipeline,
    gpu::GeneratedWorkKind kind,
    uint count,
    vk::DeviceSize size,
) {
    usz previous_bytes;
    uint previous_count;
    uint owned_count;
    foreach (&buffer : slot.reservations) {
        if (buffer.reservation_owned) owned_count++;
        if (!generated_reservation_matches(buffer, pipeline, kind)) continue;
        previous_count++;
        previous_bytes += (usz)buffer.size;
    }
    usz available_slots = slot.reservations.len - owned_count + previous_count;
    if ((usz)count > available_slots
        || size > 0 && (usz)count > usz::max / (usz)size) {
        return gpu::COMMAND_ALLOCATOR_CAPACITY_EXCEEDED~;
    }
    if (previous_bytes > slot.generated_preprocess_committed) {
        return gpu::BACKEND_ERROR~;
    }
    usz retained_bytes = slot.generated_preprocess_committed - previous_bytes;
    usz requested_bytes = (usz)size * (usz)count;
    if (retained_bytes > slot.generated_preprocess_budget
        || requested_bytes
            > slot.generated_preprocess_budget - retained_bytes) {
        return gpu::COMMAND_ALLOCATOR_CAPACITY_EXCEEDED~;
    }
}

fn bool generated_scratch_references_pipeline(
    VkDeviceState* state,
    gpu::PipelineHandle pipeline,
) {
    foreach (&cell : state.command_allocators.slots[
        :state.command_allocators.next_new
    ]) {
        if (!cell.used) continue;
        cell.value.mutex.lock()!!;
        bool found;
        foreach (&buffer : cell.value.reservations) {
            if (buffer.reservation_pipeline == pipeline) {
                found = true;
                break;
            }
        }
        cell.value.mutex.unlock();
        if (found) return true;
    }
    return false;
}

fn void? vk_reserve_generated_scratch(
    VkDeviceState* state,
    gpu::CommandAllocator* allocator,
    gpu::GeneratedScratchDesc* desc,
) {
    if (!state.generated_work.enabled) return gpu::UNSUPPORTED_FEATURE~;
    vk::MemoryRequirements requirements;
    {
        state.resource_mutex.lock()!!;
        defer state.resource_mutex.unlock();
        (void)resolve_command_allocator(
            state,
            allocator,
        )!;
        requirements = generated_scratch_requirements(state, desc)!;
    }
    {
        CommandAllocatorSlot* slot = lock_command_allocator(
            state,
            allocator,
        )!;
        defer slot.mutex.unlock();
        if (!command_allocator_quiescent_locked(slot)) {
            return gpu::RESOURCE_IN_USE~;
        }
        preflight_generated_reservation(
            slot,
            desc.pipeline,
            desc.kind,
            desc.preprocess_buffer_count,
            requirements.size,
        )!;
    }

    GeneratedPreprocessBuffer[] reserved;
    if (desc.preprocess_buffer_count > 0) {
        GeneratedPreprocessBuffer[]? reserved_result = alloc::new_array_try(
            state.host_allocator,
            GeneratedPreprocessBuffer,
            desc.preprocess_buffer_count,
        );
        if (catch reserved_result) {
            return gpu::OUT_OF_HOST_MEMORY~;
        }
        reserved = reserved_result;
    }
    defer catch free_reserved_generated_preprocess_buffers(state, reserved);
    for (uint i = 0; i < desc.preprocess_buffer_count; i++) {
        if (requirements.size > 0) {
            reserved[i] = allocate_generated_preprocess_buffer(
                state:            state,
                size:             requirements.size,
                alignment:        requirements.alignment,
                memory_type_bits: requirements.memory_type_bits,
            )!;
        }
        reserved[i].reservation_pipeline = desc.pipeline;
        reserved[i].reservation_kind = desc.kind;
        reserved[i].reservation_max_commands =
            desc.max_commands_per_list;
        reserved[i].reservation_owned = true;
    }

    {
        CommandAllocatorSlot* slot = lock_command_allocator_for_pipeline(
            state,
            allocator,
            desc.pipeline,
        )!;
        defer slot.mutex.unlock();
        if (!command_allocator_quiescent_locked(slot)) {
            return gpu::RESOURCE_IN_USE~;
        }
        preflight_generated_reservation(
            slot,
            desc.pipeline,
            desc.kind,
            desc.preprocess_buffer_count,
            requirements.size,
        )!;
        foreach (&buffer : slot.reservations) {
            if (generated_reservation_matches(
                buffer,
                desc.pipeline,
                desc.kind,
            )) {
                if (buffer.buffer != null) {
                    state.allocator.destroy_buffer(
                        buffer.buffer,
                        buffer.allocation,
                    );
                }
                slot.generated_preprocess_committed -= (usz)buffer.size;
                *buffer = {};
            }
        }
        usz next_reserved;
        foreach (&buffer : reserved) {
            while (slot.reservations[next_reserved].reservation_owned) {
                next_reserved++;
            }
            buffer.reservation_allocator = allocator.handle;
            buffer.reservation_index = (uint)next_reserved;
            slot.generated_preprocess_committed += (usz)buffer.size;
            slot.reservations[next_reserved++] = *buffer;
            *buffer = {};
        }
    }
    free_reserved_generated_preprocess_buffers(state, reserved);
}

fn void? vk_release_generated_scratch(
    VkDeviceState* state,
    gpu::CommandAllocator* allocator,
    gpu::PipelineHandle pipeline,
    gpu::GeneratedWorkKind kind,
) {
    if (!state.generated_work.enabled) return gpu::UNSUPPORTED_FEATURE~;
    CommandAllocatorSlot* slot = lock_command_allocator_for_pipeline(
        state,
        allocator,
        pipeline,
    )!;
    defer slot.mutex.unlock();
    if (!command_allocator_quiescent_locked(slot)) {
        return gpu::RESOURCE_IN_USE~;
    }
    bool found;
    foreach (&buffer : slot.reservations) {
        if (!generated_reservation_matches(buffer, pipeline, kind)) continue;
        found = true;
        if (buffer.buffer != null) {
            state.allocator.destroy_buffer(buffer.buffer, buffer.allocation);
        }
        slot.generated_preprocess_committed -= (usz)buffer.size;
        *buffer = {};
    }
    if (!found) return gpu::INVALID_RESOURCE_STATE~;
}

fn void return_allocator_buffer_locked(
    CommandAllocatorSlot* slot,
    uint buffer_index,
) {
    CommandScratch* scratch = &slot.scratch[buffer_index];
    scratch.reference_count = 0;
    scratch.generated_preprocess_count = 0;
    scratch.in_use = false;
    slot.available[slot.available_count++] = buffer_index;
}

fn vk::Result reset_command_buffer_real(
    void* user_data,
    vk::CommandBuffer command_buffer,
    vk::CommandBufferResetFlags flags,
) @private {
    (void)user_data;
    return vk::reset_command_buffer(command_buffer, flags);
}

fn vk::Result begin_command_buffer_real(
    void* user_data,
    vk::CommandBuffer command_buffer,
    vk::CommandBufferBeginInfo* begin,
) @private {
    (void)user_data;
    return vk::begin_command_buffer(command_buffer, begin);
}

fn vk::Result end_command_buffer_real(
    void* user_data,
    vk::CommandBuffer command_buffer,
) @private {
    (void)user_data;
    return vk::end_command_buffer(command_buffer);
}

fn CommandRecordingOps command_recording_ops_real() @private {
    return {
        .reset = &reset_command_buffer_real,
        .begin = &begin_command_buffer_real,
        .end   = &end_command_buffer_real,
    };
}

fn void? vk_begin_commands_with_ops(
    VkDeviceState* state,
    gpu::CommandAllocator* allocator,
    gpu::CommandList* out_commands,
    CommandRecordingOps* ops,
) @private {
    CommandAllocatorSlot* slot = lock_command_allocator(
        state,
        allocator,
    )!;
    defer slot.mutex.unlock();

    RecordingThreadIdentity current_identity =
        current_recording_thread_identity();
    if (contract_uses_full_validation(state.validation_policy)
        && slot.recording_thread_set
        && !recording_thread_identity_matches(
            slot.recording_thread_identity,
        )) {
        return gpu::RESOURCE_IN_USE~;
    }
    if (slot.available_count == 0) return gpu::DEVICE_BUSY~;
    uint buffer_index = slot.available[--slot.available_count];
    CommandScratch* scratch = &slot.scratch[buffer_index];
    scratch.in_use = true;
    scratch.reference_count = 0;
    scratch.generated_preprocess_count = 0;
    slot.live_recordings++;
    if (!slot.recording_thread_set) {
        slot.recording_thread_identity = current_identity;
        slot.recording_thread_set = true;
    }
    bool claimed = true;
    defer catch {
        if (claimed) {
            return_allocator_buffer_locked(slot, buffer_index);
            slot.live_recordings--;
            if (slot.live_recordings == 0) slot.recording_thread_set = false;
        }
    }

    vk::CommandBuffer command_buffer = slot.command_buffers[buffer_index];
    CommandRecord backend_record = {
        .command_buffer   = command_buffer,
        .allocator        = allocator.handle,
        .allocator_buffer = buffer_index,
        .queue            = allocator.queue,
        .scratch          = scratch,
    };
    CommandRecord* record;
    {
        state.command_mutex.lock()!!;
        defer state.command_mutex.unlock();
        record = state.commands.claim(
            device:       allocator.device,
            ops:          state.command_ops,
            device_state: state,
            value:        backend_record,
        )!;
    }
    bool record_claimed = true;
    defer catch {
        if (record_claimed) {
            state.command_mutex.lock()!!;
            (void)state.commands.cancel_claim(record.identity);
            state.command_mutex.unlock();
        }
    }
    check_device_result(state, ops.reset(
        ops.user_data,
        command_buffer,
        {},
    ))!;
    vk::CommandBufferBeginInfo begin = vk::command_buffer_begin_info()
        .set_flags(vk::CommandBufferUsageFlagBits.ONE_TIME_SUBMIT_BIT);
    check_device_result(state, ops.begin(
        ops.user_data,
        command_buffer,
        &begin,
    ))!;

    record.state = gpu::internal::CommandState.RECORDING;
    gpu::internal::publish_command_list(record, out_commands);
    record_claimed = false;
    claimed = false;
}

fn void? vk_begin_commands(
    VkDeviceState* state,
    gpu::CommandAllocator* allocator,
    gpu::CommandList* out_commands,
) {
    CommandRecordingOps ops = command_recording_ops_real();
    return vk_begin_commands_with_ops(
        state:        state,
        allocator:    allocator,
        out_commands: out_commands,
        ops:          &ops,
    );
}

fn void? vk_end_commands_with_ops(
    VkDeviceState* state,
    CommandRecord* commands,
    CommandRecordingOps* ops,
) @private {
    CommandRecord* record = commands;
    if (commands.state != gpu::internal::CommandState.RECORDING) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    if (commands.device_state != state) return gpu::INVALID_HANDLE~;
    CommandAllocatorSlot* slot = lock_command_allocator_handle(
        state,
        record.allocator,
    )!;
    defer slot.mutex.unlock();
    check_device_result(state, ops.end(
        ops.user_data,
        record.command_buffer,
    ))!;
    commands.state = gpu::internal::CommandState.EXECUTABLE;
    slot.live_recordings--;
    slot.live_executables++;
    if (slot.live_recordings == 0) slot.recording_thread_set = false;
}

fn void? vk_end_commands(
    VkDeviceState* state,
    CommandRecord* commands,
) {
    CommandRecordingOps ops = command_recording_ops_real();
    return vk_end_commands_with_ops(state, commands, &ops);
}

fn CommandRecord*? begin_commands_with_injected_ops(
    VkDeviceState* state,
    gpu::CommandAllocator* allocator,
    gpu::CommandList* out_commands,
    CommandRecordingOps* ops,
) @if(env::TESTING) {
    vk_begin_commands_with_ops(
        state:        state,
        allocator:    allocator,
        out_commands: out_commands,
        ops:          ops,
    )!;
    CommandRecord* record =
        gpu::internal::recording_command(out_commands)!;
    return record;
}

fn void? end_commands_with_injected_ops(
    VkDeviceState* state,
    CommandRecord* record,
    CommandRecordingOps* ops,
) @if(env::TESTING) {
    if (record == null
        || record.device_state != state) {
        return gpu::INVALID_HANDLE~;
    }
    return vk_end_commands_with_ops(state, record, ops);
}

fn void? vk_discard_commands(
    VkDeviceState* state,
    CommandRecord* command,
    bool executable,
) {
    if (command.device_state != state) return gpu::INVALID_HANDLE~;
    gpu::internal::CommandListHandle handle = command.identity;
    gpu::CommandAllocatorHandle allocator_handle;
    {
        state.command_mutex.lock()!!;
        defer state.command_mutex.unlock();
        CommandRecord* record = state.commands.get(handle)!;
        bool state_matches = executable
            ? record.state == gpu::internal::CommandState.EXECUTABLE
            : record.state == gpu::internal::CommandState.RECORDING
                || record.state == gpu::internal::CommandState.RECORDING_RENDER_PASS;
        if (!state_matches) return gpu::COMMAND_RECORDING_ERROR~;
        allocator_handle = record.allocator;
    }
    CommandAllocatorSlot* slot = lock_command_allocator_handle(
        state,
        allocator_handle,
    )!;
    defer slot.mutex.unlock();
    state.command_mutex.lock()!!;
    defer state.command_mutex.unlock();
    CommandRecord* record = state.commands.get(handle)!;
    bool state_matches = executable
        ? record.state == gpu::internal::CommandState.EXECUTABLE
        : record.state == gpu::internal::CommandState.RECORDING
            || record.state == gpu::internal::CommandState.RECORDING_RENDER_PASS;
    if (!state_matches || record.allocator != allocator_handle) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    command.state = gpu::internal::CommandState.INACTIVE;
    clear_command_scratch_references(record.scratch);
    recycle_generated_preprocess_indices_locked(
        state,
        slot,
        record.scratch.generated_preprocess_indices,
        record.scratch.generated_preprocess_count,
    );
    if (executable) {
        slot.live_executables--;
    } else {
        slot.live_recordings--;
        if (slot.live_recordings == 0) slot.recording_thread_set = false;
    }
    return_allocator_buffer_locked(slot, record.allocator_buffer);
    (void)state.commands.free(handle);
}

fn PreparedBufferCopy? lower_buffer_copy(
    CommandRecord* commands,
    gpu::BufferCopyDesc* desc,
) {
    if (desc == null) return gpu::INVALID_ARGUMENT~;
    CommandRecord* record = commands;
    if (record.state != gpu::internal::CommandState.RECORDING) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    VkDeviceState* state = commands.device_state;
    return {
        .src = resolve_span_slot(
            state:          state,
            span:           &desc.src,
            required:       0,
            operation:      "cmd_copy_buffer",
            rejected_field: "BufferCopyDesc.src",
        )!,
        .dst = resolve_span_slot(
            state:          state,
            span:           &desc.dst,
            required:       0,
            operation:      "cmd_copy_buffer",
            rejected_field: "BufferCopyDesc.dst",
        )!,
        .src_span = desc.src,
        .dst_span = desc.dst,
    };
}

fn bool buffer_copy_is_safe(PreparedBufferCopy* copy) {
    if (copy.src.size != copy.dst.size) return false;
    return copy.src.slot.buffer != copy.dst.slot.buffer
        || copy.src.offset >= copy.dst.offset + copy.dst.size
        || copy.dst.offset >= copy.src.offset + copy.src.size;
}

fn PreparedBufferCopy? validate_and_lower_buffer_copy(
    CommandRecord* commands,
    gpu::BufferCopyDesc* desc,
) {
    CommandRecord* full_record = commands;
    if (full_record.state != gpu::internal::CommandState.RECORDING) {
        VkDeviceState* full_state = commands.device_state;
        uint command_index = 0;
        uint command_generation = 0;
        bool has_public_identity = false;
        foreach (i, &cell : full_state.commands.slots) {
            if (&cell.record != full_record) continue;
            command_index = (uint)i;
            command_generation = cell.generation;
            has_public_identity = cell.used;
            break;
        }
        return report_contract_failure(
            state:               full_state,
            operation:           "cmd_copy_buffer",
            public_fault:        gpu::COMMAND_RECORDING_ERROR,
            resource_kind:       gpu::DebugResourceKind.COMMAND_LIST,
            has_public_identity: has_public_identity,
            index:               command_index,
            generation:          command_generation,
            debug_name:          null,
            rejected_field:      "CommandList.state",
            invariant:           "command must be recording outside a render pass",
            backend_text:        null,
        )~;
    }
    PreparedBufferCopy copy = lower_buffer_copy(commands, desc)!;
    VkDeviceState* state = commands.device_state;
    CommandRecord* record = commands;
    validate_span_transfer_usage(
        state:     state,
        resolved:  &copy.src,
        span:      &copy.src_span,
        source:    true,
        operation: "cmd_copy_buffer",
    )!;
    validate_span_transfer_usage(
        state:     state,
        resolved:  &copy.dst,
        span:      &copy.dst_span,
        source:    false,
        operation: "cmd_copy_buffer",
    )!;
    if (copy.src.size != copy.dst.size) {
        return report_contract_failure(
            state:               state,
            operation:           "cmd_copy_buffer",
            public_fault:        gpu::INVALID_ARGUMENT,
            resource_kind:       span_resource_kind(state, &copy.dst_span),
            has_public_identity: true,
            index:               copy.dst_span.index,
            generation:          copy.dst_span.generation,
            debug_name:          slot_name(&copy.dst.slot.debug_name[0]),
            rejected_field:      "BufferCopyDesc.dst",
            invariant:           "source and destination spans must have equal sizes",
            backend_text:        null,
        )~;
    }
    if (!buffer_copy_is_safe(&copy)) {
        return report_contract_failure(
            state:               state,
            operation:           "cmd_copy_buffer",
            public_fault:        gpu::INVALID_ARGUMENT,
            resource_kind:       span_resource_kind(state, &copy.src_span),
            has_public_identity: true,
            index:               copy.src_span.index,
            generation:          copy.src_span.generation,
            debug_name:          slot_name(&copy.src.slot.debug_name[0]),
            rejected_field:      "BufferCopyDesc",
            invariant:           "source and destination spans must not overlap",
            backend_text:        null,
        )~;
    }
    validate_span_recording_access_only(
        state:     state,
        record:    record,
        resolved:  &copy.src,
        span:      &copy.src_span,
        operation: "cmd_copy_buffer",
    )!;
    validate_span_recording_access_only(
        state:     state,
        record:    record,
        resolved:  &copy.dst,
        span:      &copy.dst_span,
        operation: "cmd_copy_buffer",
    )!;
    return copy;
}

fn void? track_buffer_copy_references(
    CommandRecord* commands,
    PreparedBufferCopy* copy,
) {
    VkDeviceState* state = commands.device_state;
    CommandRecord* record = commands;
    CommandResourceReference[2] references = {
        {
            .owner      = copy.src_span.owner,
            .index      = copy.src_span.index,
            .generation = copy.src_span.generation,
        },
        {
            .owner      = copy.dst_span.owner,
            .index      = copy.dst_span.index,
            .generation = copy.dst_span.generation,
        },
    };
    preflight_command_references(record, references[..])!;
    track_span_command_reference(state, record, &copy.src_span)!;
    track_span_command_reference(state, record, &copy.dst_span)!;
}

fn void record_buffer_copy(
    CommandRecord* commands,
    PreparedBufferCopy* copy,
) {
    CommandRecord* record = commands;
    vk::BufferCopy region = {
        .src_offset = (vk::DeviceSize)copy.src.offset,
        .dst_offset = (vk::DeviceSize)copy.dst.offset,
        .size       = (vk::DeviceSize)copy.src.size,
    };
    vk::cmd_copy_buffer(
        command_buffer: record.command_buffer,
        src_buffer:     copy.src.slot.buffer,
        dst_buffer:     copy.dst.slot.buffer,
        region_count:   1,
        p_regions:      &region,
    );
}

fn void? trusted_copy_buffer(
    CommandRecord* commands,
    gpu::BufferCopyDesc* desc,
) {
    PreparedBufferCopy copy = lower_buffer_copy(commands, desc)!;
    if (!buffer_copy_is_safe(&copy)) return gpu::INVALID_ARGUMENT~;
    record_buffer_copy(commands, &copy);
}

fn void? full_copy_buffer(
    CommandRecord* commands,
    gpu::BufferCopyDesc* desc,
) {
    PreparedBufferCopy copy = validate_and_lower_buffer_copy(commands, desc)!;
    CommandRecord* record = commands;
    uint checkpoint = record.scratch.reference_count;
    defer catch rollback_command_references(
        record,
        checkpoint,
    );
    track_buffer_copy_references(commands, &copy)!;
    record_buffer_copy(commands, &copy);
}

fn PreparedBufferFill? lower_buffer_fill(
    CommandRecord* commands,
    gpu::GpuSpan dst,
    uint value,
) {
    CommandRecord* record = commands;
    if (record.state != gpu::internal::CommandState.RECORDING) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    VkDeviceState* state = commands.device_state;
    ResolvedSpan resolved = resolve_span_slot(
        state:          state,
        span:           &dst,
        required:       0,
        operation:      "cmd_fill_buffer",
        rejected_field: "dst",
    )!;
    return {
        .span   = resolved,
        .source = dst,
        .value  = value,
    };
}

fn PreparedBufferFill? validate_and_lower_buffer_fill(
    CommandRecord* commands,
    gpu::GpuSpan dst,
    uint value,
) {
    PreparedBufferFill fill = lower_buffer_fill(commands, dst, value)!;
    VkDeviceState* state = commands.device_state;
    CommandRecord* record = commands;
    if (fill.span.offset % 4 != 0 || fill.span.size % 4 != 0) {
        bool offset_misaligned = fill.span.offset % 4 != 0;
        return report_contract_failure(
            state:               state,
            operation:           "cmd_fill_buffer",
            public_fault:        gpu::INVALID_ARGUMENT,
            resource_kind:       span_resource_kind(state, &fill.source),
            has_public_identity: true,
            index:               fill.source.index,
            generation:          fill.source.generation,
            debug_name:          slot_name(&fill.span.slot.debug_name[0]),
            rejected_field:      offset_misaligned
                ? "GpuSpan.offset"
                : "GpuSpan.size",
            invariant:           offset_misaligned
                ? "span offset must be 4-byte aligned"
                : "span size must be 4-byte aligned",
            backend_text:        null,
        )~;
    }
    validate_span_transfer_usage(
        state:     state,
        resolved:  &fill.span,
        span:      &fill.source,
        source:    false,
        operation: "cmd_fill_buffer",
    )!;
    validate_span_recording_access_only(
        state:     state,
        record:    record,
        resolved:  &fill.span,
        span:      &fill.source,
        operation: "cmd_fill_buffer",
    )!;
    return fill;
}

fn void record_buffer_fill(
    CommandRecord* commands,
    PreparedBufferFill* fill,
) {
    vk::cmd_fill_buffer(
        command_buffer: commands.command_buffer,
        dst_buffer:     fill.span.slot.buffer,
        dst_offset:     (vk::DeviceSize)fill.span.offset,
        size:           (vk::DeviceSize)fill.span.size,
        data:           fill.value,
    );
}

fn void? trusted_fill_buffer(
    CommandRecord* commands,
    gpu::GpuSpan dst,
    uint value,
) {
    PreparedBufferFill fill = lower_buffer_fill(commands, dst, value)!;
    if (fill.span.offset % 4 != 0 || fill.span.size % 4 != 0) {
        return gpu::INVALID_ARGUMENT~;
    }
    record_buffer_fill(commands, &fill);
}

fn void? full_fill_buffer(
    CommandRecord* commands,
    gpu::GpuSpan dst,
    uint value,
) {
    PreparedBufferFill fill = validate_and_lower_buffer_fill(
        commands,
        dst,
        value,
    )!;
    CommandRecord* record = commands;
    uint checkpoint = record.scratch.reference_count;
    defer catch rollback_command_references(
        record,
        checkpoint,
    );
    track_span_command_reference(
        commands.device_state,
        record,
        &fill.source,
    )!;
    record_buffer_fill(commands, &fill);
}

<* Validate and translate a buffer-image copy region. *>
fn vk::BufferImageCopy? resolve_texture_copy_region(
    ResolvedSpan* resolved,
    TextureSlot* slot,
    uint row_length_texels,
    uint mip,
    uint base_layer,
    uint layer_count,
    uint x,
    uint y,
    uint width,
    uint height,
) @private {
    if (mip >= slot.mip_levels) return gpu::INVALID_ARGUMENT~;
    uint layers = layer_count == 0 ? 1 : layer_count;
    if ((ulong)base_layer + (ulong)layers > (ulong)slot.array_layers) {
        return gpu::INVALID_ARGUMENT~;
    }

    MipExtent extent = texture_mip_extent(slot, mip);
    if (width == 0) width = extent.width;
    if (height == 0) height = extent.height;
    if ((ulong)x + (ulong)width > (ulong)extent.width
        || (ulong)y + (ulong)height > (ulong)extent.height) {
        return gpu::INVALID_ARGUMENT~;
    }
    if (row_length_texels != 0 && row_length_texels < width) {
        return gpu::INVALID_ARGUMENT~;
    }

    usz texel_size = format_texel_size(slot.format);
    usz row_texels = row_length_texels == 0 ? width : row_length_texels;
    if (texel_size == 0 || row_texels == 0 || height == 0 || layers == 0) {
        return gpu::INVALID_ARGUMENT~;
    }
    if (row_texels > usz::max / texel_size) return gpu::INVALID_ARGUMENT~;
    usz copy_bytes = row_texels * texel_size;
    if ((usz)height > usz::max / copy_bytes) return gpu::INVALID_ARGUMENT~;
    copy_bytes *= height;
    if ((usz)layers > usz::max / copy_bytes) return gpu::INVALID_ARGUMENT~;
    copy_bytes *= layers;
    if (copy_bytes > resolved.size) return gpu::INVALID_ARGUMENT~;
    if (resolved.offset % texel_size != 0 || resolved.offset % 4 != 0) {
        return gpu::INVALID_ARGUMENT~;
    }

    return {
        .buffer_offset       = (vk::DeviceSize)resolved.offset,
        .buffer_row_length   = row_length_texels,
        .buffer_image_height = 0,
        .image_subresource   = {
            .aspect_mask      = format_aspect(slot.format),
            .mip_level        = mip,
            .base_array_layer = base_layer,
            .layer_count      = layers,
        },
        .image_offset = { .x = (int)x, .y = (int)y, .z = 0 },
        .image_extent = { .width = width, .height = height, .depth = 1 },
    };
}

fn PreparedBufferTextureCopy? lower_buffer_to_texture_copy(
    CommandRecord* commands,
    gpu::BufferTextureCopyDesc* desc,
) {
    if (desc == null) return gpu::INVALID_ARGUMENT~;
    CommandRecord* record = commands;
    if (record.state != gpu::internal::CommandState.RECORDING) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    VkDeviceState* state = commands.device_state;
    TextureSlot* texture = state.textures.get(desc.texture)!;
    ResolvedSpan buffer = resolve_span_slot(
        state:          state,
        span:           &desc.src,
        required:       0,
        operation:      "cmd_copy_buffer_to_texture",
        rejected_field: "GpuSpan",
    )!;
    return {
        .buffer         = buffer,
        .buffer_span    = desc.src,
        .texture        = texture,
        .texture_handle = desc.texture,
        .region         = resolve_texture_copy_region(
            resolved:          &buffer,
            slot:              texture,
            row_length_texels: desc.row_length_texels,
            mip:               desc.mip,
            base_layer:        desc.base_layer,
            layer_count:       desc.layer_count,
            x:                 desc.x,
            y:                 desc.y,
            width:             desc.width,
            height:            desc.height,
        )!,
    };
}

fn PreparedBufferTextureCopy? validate_and_lower_buffer_to_texture_copy(
    CommandRecord* commands,
    gpu::BufferTextureCopyDesc* desc,
) {
    PreparedBufferTextureCopy copy = lower_buffer_to_texture_copy(commands, desc)!;
    VkDeviceState* state = commands.device_state;
    CommandRecord* record = commands;
    validate_span_recording_access_only(
        state:     state,
        record:    record,
        resolved:  &copy.buffer,
        span:      &copy.buffer_span,
        operation: "cmd_copy_buffer_to_texture",
    )!;
    validate_texture_recording_access_only(
        state:     state,
        record:    record,
        slot:      copy.texture,
        handle:    copy.texture_handle,
        operation: "cmd_copy_buffer_to_texture",
    )!;
    validate_span_transfer_usage(
        state:     state,
        resolved:  &copy.buffer,
        span:      &copy.buffer_span,
        source:    true,
        operation: "cmd_copy_buffer_to_texture",
    )!;
    if (!copy.texture.usage.transfer_dst) return gpu::INVALID_ARGUMENT~;
    return copy;
}

fn PreparedBufferTextureCopy? lower_texture_to_buffer_copy(
    CommandRecord* commands,
    gpu::TextureBufferCopyDesc* desc,
) {
    if (desc == null) return gpu::INVALID_ARGUMENT~;
    CommandRecord* record = commands;
    if (record.state != gpu::internal::CommandState.RECORDING) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    VkDeviceState* state = commands.device_state;
    TextureSlot* texture = state.textures.get(desc.texture)!;
    ResolvedSpan buffer = resolve_span_slot(
        state:          state,
        span:           &desc.dst,
        required:       0,
        operation:      "cmd_copy_texture_to_buffer",
        rejected_field: "GpuSpan",
    )!;
    return {
        .buffer         = buffer,
        .buffer_span    = desc.dst,
        .texture        = texture,
        .texture_handle = desc.texture,
        .region         = resolve_texture_copy_region(
            resolved:          &buffer,
            slot:              texture,
            row_length_texels: desc.row_length_texels,
            mip:               desc.mip,
            base_layer:        desc.base_layer,
            layer_count:       desc.layer_count,
            x:                 desc.x,
            y:                 desc.y,
            width:             desc.width,
            height:            desc.height,
        )!,
    };
}

fn PreparedBufferTextureCopy? validate_and_lower_texture_to_buffer_copy(
    CommandRecord* commands,
    gpu::TextureBufferCopyDesc* desc,
) {
    PreparedBufferTextureCopy copy = lower_texture_to_buffer_copy(commands, desc)!;
    VkDeviceState* state = commands.device_state;
    CommandRecord* record = commands;
    validate_texture_recording_access_only(
        state:     state,
        record:    record,
        slot:      copy.texture,
        handle:    copy.texture_handle,
        operation: "cmd_copy_texture_to_buffer",
    )!;
    validate_span_recording_access_only(
        state:     state,
        record:    record,
        resolved:  &copy.buffer,
        span:      &copy.buffer_span,
        operation: "cmd_copy_texture_to_buffer",
    )!;
    validate_span_transfer_usage(
        state:     state,
        resolved:  &copy.buffer,
        span:      &copy.buffer_span,
        source:    false,
        operation: "cmd_copy_texture_to_buffer",
    )!;
    if (!copy.texture.usage.transfer_src) return gpu::INVALID_ARGUMENT~;
    return copy;
}

fn void? track_buffer_texture_copy_references(
    CommandRecord* commands,
    PreparedBufferTextureCopy* copy,
) {
    VkDeviceState* state = commands.device_state;
    CommandRecord* record = commands;
    CommandResourceReference[2] references = {
        {
            .owner      = copy.buffer_span.owner,
            .index      = copy.buffer_span.index,
            .generation = copy.buffer_span.generation,
        },
        {
            .owner      = copy.texture_handle.owner,
            .index      = copy.texture_handle.index,
            .generation = copy.texture_handle.generation,
        },
    };
    preflight_command_references(record, references[..])!;
    track_span_command_reference(state, record, &copy.buffer_span)!;
    track_texture_command_reference(state, record, copy.texture_handle)!;
}

fn vk::ImageLayout texture_copy_layout(
    TextureSlot* texture,
    bool destination,
) {
    return texture_layout_to_vk(
        destination
            ? gpu::TextureLayout.TRANSFER_DESTINATION
            : gpu::TextureLayout.TRANSFER_SOURCE,
        texture.format,
    );
}

fn void record_buffer_to_texture_copy(
    CommandRecord* commands,
    PreparedBufferTextureCopy* copy,
) {
    vk::cmd_copy_buffer_to_image(
        command_buffer:  commands.command_buffer,
        src_buffer:      copy.buffer.slot.buffer,
        dst_image:       copy.texture.image,
        dst_image_layout: texture_copy_layout(
            copy.texture,
            true,
        ),
        region_count:    1,
        p_regions:       &copy.region,
    );
}

fn void record_texture_to_buffer_copy(
    CommandRecord* commands,
    PreparedBufferTextureCopy* copy,
) {
    vk::cmd_copy_image_to_buffer(
        command_buffer:  commands.command_buffer,
        src_image:       copy.texture.image,
        src_image_layout: texture_copy_layout(
            copy.texture,
            false,
        ),
        dst_buffer:      copy.buffer.slot.buffer,
        region_count:    1,
        p_regions:       &copy.region,
    );
}

fn void? trusted_copy_buffer_to_texture(
    CommandRecord* commands,
    gpu::BufferTextureCopyDesc* desc,
) {
    PreparedBufferTextureCopy copy = lower_buffer_to_texture_copy(commands, desc)!;
    record_buffer_to_texture_copy(commands, &copy);
}

fn void? full_copy_buffer_to_texture(
    CommandRecord* commands,
    gpu::BufferTextureCopyDesc* desc,
) {
    PreparedBufferTextureCopy copy =
        validate_and_lower_buffer_to_texture_copy(commands, desc)!;
    CommandRecord* record = commands;
    uint checkpoint = record.scratch.reference_count;
    defer catch rollback_command_references(record, checkpoint);
    track_buffer_texture_copy_references(commands, &copy)!;
    record_buffer_to_texture_copy(commands, &copy);
}

fn void? trusted_copy_texture_to_buffer(
    CommandRecord* commands,
    gpu::TextureBufferCopyDesc* desc,
) {
    PreparedBufferTextureCopy copy = lower_texture_to_buffer_copy(commands, desc)!;
    record_texture_to_buffer_copy(commands, &copy);
}

fn void? full_copy_texture_to_buffer(
    CommandRecord* commands,
    gpu::TextureBufferCopyDesc* desc,
) {
    PreparedBufferTextureCopy copy =
        validate_and_lower_texture_to_buffer_copy(commands, desc)!;
    CommandRecord* record = commands;
    uint checkpoint = record.scratch.reference_count;
    defer catch rollback_command_references(record, checkpoint);
    track_buffer_texture_copy_references(commands, &copy)!;
    record_texture_to_buffer_copy(commands, &copy);
}

fn ResolvedSpan? lower_indirect_span(
    VkDeviceState* state,
    gpu::GpuSpan* span,
    usz arg_size,
    uint count,
) {
    usz required = gpu::internal::indirect_argument_bytes(
        argument_size: arg_size,
        count:         count,
        max_count:     uint::max,
    )!;
    ResolvedSpan resolved = resolve_span_slot(state, span, required)!;
    if (resolved.offset % 4 != 0) return gpu::INVALID_ARGUMENT~;
    return resolved;
}

<* Resolve an indirect argument span and validate its command semantics. *>
fn ResolvedSpan? validate_and_lower_indirect_span(
    VkDeviceState* state,
    CommandRecord* record,
    gpu::GpuSpan* span,
    usz arg_size,
    uint count,
    ZString operation,
    ZString rejected_field,
    uint max_count = uint::max,
) {
    if (count > max_count && operation != null) {
        return report_contract_failure(
            state:               state,
            operation:           operation,
            public_fault:        gpu::INVALID_ARGUMENT,
            resource_kind:       gpu::DebugResourceKind.NONE,
            has_public_identity: false,
            index:               0,
            generation:          0,
            debug_name:          null,
            rejected_field:      rejected_field,
            invariant:           "count must not exceed DeviceCaps.max_draw_indirect_count",
            backend_text:        null,
        )~;
    }
    usz? required_result = gpu::internal::indirect_argument_bytes(
        argument_size: arg_size,
        count:         count,
        max_count:     max_count,
    );
    if (catch size_fault = required_result) {
        if (operation == null) return size_fault~;
        return report_contract_failure(
            state:               state,
            operation:           operation,
            public_fault:        size_fault,
            resource_kind:       gpu::DebugResourceKind.NONE,
            has_public_identity: false,
            index:               0,
            generation:          0,
            debug_name:          null,
            rejected_field:      rejected_field,
            invariant:           "indirect argument byte size must not overflow",
            backend_text:        null,
        )~;
    }
    usz required = required_result;
    ResolvedSpan resolved = resolve_span_slot(
        state:          state,
        span:           span,
        required:       required,
        operation:      operation,
        rejected_field: rejected_field,
    )!;
    validate_span_recording_access_only(
        state:     state,
        record:    record,
        resolved:  &resolved,
        span:      span,
        operation: operation,
    )!;
    if (!resolved.slot.usage.indirect) return gpu::INVALID_ARGUMENT~;
    if (resolved.offset % 4 != 0) return gpu::INVALID_ARGUMENT~;
    return resolved;
}

fn BoundPipeline? bound_pipeline_snapshot(CommandRecord* record) {
    BoundPipeline pipeline = record.logical_pipeline;
    if (!pipeline.handle.is_valid() || pipeline.pipeline == null
        || pipeline.layout == null) {
        return gpu::INVALID_RESOURCE_STATE~;
    }
    return pipeline;
}

fn BoundPipeline*? active_bound_pipeline(
    VkDeviceState* state,
    CommandRecord* record,
    PipelineKind expected_kind,
    ZString operation,
) @private {
    BoundPipeline* pipeline = &record.logical_pipeline;
    if (!pipeline.handle.is_valid()) return gpu::COMMAND_RECORDING_ERROR~;
    if (pipeline.kind != expected_kind) {
        ZString invariant = expected_kind == PipelineKind.COMPUTE
            ? "pipeline kind must match the COMPUTE bind point"
            : "pipeline kind must match the GRAPHICS bind point";
        return report_contract_failure(
            state:               state,
            operation:           operation,
            public_fault:        gpu::INVALID_ARGUMENT,
            resource_kind:       gpu::DebugResourceKind.PIPELINE,
            has_public_identity: true,
            index:               pipeline.handle.index,
            generation:          pipeline.handle.generation,
            debug_name:          null,
            rejected_field:      "pipeline",
            invariant:           invariant,
            backend_text:        null,
        )~;
    }
    return pipeline;
}

fn void? validate_pipeline_render_compatibility(
    BoundPipeline* pipeline,
    RenderCompatibility* render,
) @private {
    if (pipeline.kind != PipelineKind.GRAPHICS) return;
    RenderCompatibility* expected = &pipeline.render;
    if (expected.color_count != render.color_count
        || expected.depth_format != render.depth_format
        || expected.sample_count != render.sample_count) {
        return gpu::INVALID_ARGUMENT~;
    }
    for (uint i = 0; i < render.color_count; i++) {
        if (expected.color_formats[i] != render.color_formats[i]) {
            return gpu::INVALID_ARGUMENT~;
        }
    }
}

fn void? validate_active_pipeline_render_compatibility(
    VkDeviceState* state,
    CommandRecord* record,
    RenderCompatibility* render,
) @private {
    BoundPipeline* pipeline = &record.logical_pipeline;
    if (!pipeline.handle.is_valid()) return;
    return validate_pipeline_render_compatibility(pipeline, render);
}

fn RenderCompatibility pipeline_render_compatibility(
    PipelineKey* key,
) @private {
    RenderCompatibility compatibility = {
        .color_count  = key.color_target_count,
        .depth_format = (gpu::Format)key.depth_format,
        .sample_count = (gpu::SampleCount)key.sample_count,
    };
    for (uint i = 0; i < key.color_target_count; i++) {
        compatibility.color_formats[i] =
            (gpu::Format)key.color_formats[i];
    }
    return compatibility;
}

fn BoundPipeline? lower_bound_pipeline(
    CommandRecord* commands,
    gpu::PipelineHandle pipeline,
) {
    CommandRecord* record = commands;
    if (record.state != gpu::internal::CommandState.RECORDING
        && record.state != gpu::internal::CommandState.RECORDING_RENDER_PASS) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    VkDeviceState* state = commands.device_state;
    validate_descriptor_heap_binding(state)!;
    PipelineCell* cell = state.pipelines.get_cell(pipeline)!;
    PipelineSlot* slot = &cell.value;
    if (slot.entry >= state.pipeline_cache.entries.len
        || !state.pipeline_cache.entries[slot.entry].used) {
        return gpu::INVALID_HANDLE~;
    }
    PipelineCacheEntry* entry = &state.pipeline_cache.entries[slot.entry];
    return {
        .handle                    = pipeline,
        .pipeline                  = slot.pipeline,
        .layout                    = slot.layout,
        .generated_dispatch_layout = slot.generated_dispatch_layout,
        .cache_entry               = slot.entry,
        .kind                      = (PipelineKind)entry.key.kind,
        .render                    = pipeline_render_compatibility(&entry.key),
    };
}

fn BoundPipeline? validate_and_lower_bound_pipeline(
    CommandRecord* commands,
    gpu::PipelineHandle pipeline,
) {
    BoundPipeline bound = lower_bound_pipeline(commands, pipeline)!;
    VkDeviceState* state = commands.device_state;
    CommandRecord* record = commands;
    if (record.state == gpu::internal::CommandState.RECORDING_RENDER_PASS) {
        validate_pipeline_render_compatibility(
            &bound,
            &record.active_render,
        )!;
    }
    return bound;
}

fn void? track_bound_pipeline(
    CommandRecord* commands,
    BoundPipeline* pipeline,
) {
    CommandRecord* record = commands;
    return track_command_reference(
        commands.device_state,
        record,
        pipeline.handle.owner,
        pipeline.handle.index,
        pipeline.handle.generation,
    );
}

fn void? record_bound_pipeline(
    CommandRecord* commands,
    BoundPipeline* bound,
) {
    VkDeviceState* state = commands.device_state;
    CommandRecord* record = commands;
    NativePipelineBinding* native = bound.kind == PipelineKind.COMPUTE
        ? &record.native_compute_pipeline
        : &record.native_graphics_pipeline;
    vk::PipelineBindPoint bind_point = bound.kind == PipelineKind.COMPUTE
        ? vk::PipelineBindPoint.COMPUTE
        : vk::PipelineBindPoint.GRAPHICS;
    bool native_match = native.pipeline != null
        && native.pipeline == bound.pipeline
        && native.cache_entry == bound.cache_entry;
    if (!native_match) {
        vk::cmd_bind_pipeline(record.command_buffer, bind_point, bound.pipeline);
        *native = {
            .pipeline    = bound.pipeline,
            .cache_entry = bound.cache_entry,
        };
    }
    bind_descriptor_heap(
        state:      state,
        record:     record,
        layout:     bound.layout,
        bind_point: bind_point,
    )!;
    record.logical_pipeline = *bound;
}

fn void update_full_color_state_for_bind(
    CommandRecord* record,
    BoundPipeline* bound,
) @private {
    if (!record.color_state_initialized
        || bound.kind != PipelineKind.GRAPHICS) return;
    if (!color_compatibility_matches(
        &record.color_state,
        &bound.render,
    )) {
        record.color_state_initialized = false;
        record.color_state = {};
    }
}

fn void? trusted_bind_pipeline(
    CommandRecord* commands,
    gpu::PipelineHandle pipeline,
) {
    BoundPipeline bound = lower_bound_pipeline(commands, pipeline)!;
    record_bound_pipeline(commands, &bound)!;
}

fn void? full_bind_pipeline(
    CommandRecord* commands,
    gpu::PipelineHandle pipeline,
) {
    BoundPipeline bound = validate_and_lower_bound_pipeline(commands, pipeline)!;
    CommandRecord* record = commands;
    uint checkpoint = record.scratch.reference_count;
    defer catch rollback_command_references(
        record,
        checkpoint,
    );
    track_bound_pipeline(commands, &bound)!;
    record_bound_pipeline(commands, &bound)!;
    update_full_color_state_for_bind(record, &bound);
}

fn void push_compute_root(
    CommandRecord* record,
    BoundPipeline* pipeline,
    gpu::GpuAddress root,
) @private {
    gpu::RootPush push = { .root_gpu = root };
    $if env::TESTING:
        last_compute_root.store((ulong)push.root_gpu, AtomicOrdering.RELAXED);
        last_compute_layout.store((ulong)pipeline.layout, AtomicOrdering.RELAXED);
    $endif
    vk::cmd_push_constants(
        command_buffer: record.command_buffer,
        layout:         pipeline.layout,
        stage_flags:    vk::ShaderStageFlagBits.COMPUTE_BIT,
        offset:         0,
        size:           gpu::RootPush::size,
        p_values:       &push,
    );
}

fn void? vk_cmd_dispatch(
    CommandRecord* commands,
    gpu::GpuAddress root,
    gpu::Vec3u groups,
) @private {
    CommandRecord* record = commands;
    if (record.state != gpu::internal::CommandState.RECORDING) return gpu::COMMAND_RECORDING_ERROR~;
    VkDeviceState* state = commands.device_state;
    BoundPipeline* pipeline = active_bound_pipeline(
        state:         state,
        record:        record,
        expected_kind: PipelineKind.COMPUTE,
        operation:     "cmd_dispatch",
    )!;
    if (catch limit_fault = gpu::internal::validate_dispatch_groups(groups, state.caps.max_compute_work_group_count)) {
        return report_contract_failure(
            state:               state,
            operation:           "cmd_dispatch",
            public_fault:        limit_fault,
            resource_kind:       gpu::DebugResourceKind.NONE,
            has_public_identity: false,
            index:               0,
            generation:          0,
            debug_name:          null,
            rejected_field:      "groups",
            invariant:           "group counts must not exceed DeviceCaps.max_compute_work_group_count",
            backend_text:        null,
        )~;
    }
    push_compute_root(record, pipeline, root);
    vk::cmd_dispatch(
        command_buffer: record.command_buffer,
        group_count_x:  groups.x,
        group_count_y:  groups.y,
        group_count_z:  groups.z,
    );
}

fn void? trusted_dispatch(
    CommandRecord* commands,
    gpu::GpuAddress root,
    gpu::Vec3u groups,
) {
    CommandRecord* record = commands;
    if (record.state != gpu::internal::CommandState.RECORDING) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    BoundPipeline pipeline = bound_pipeline_snapshot(record)!;
    push_compute_root(record, &pipeline, root);
    vk::cmd_dispatch(
        command_buffer: record.command_buffer,
        group_count_x:  groups.x,
        group_count_y:  groups.y,
        group_count_z:  groups.z,
    );
}

fn void? full_dispatch(
    CommandRecord* commands,
    gpu::GpuAddress root,
    gpu::Vec3u groups,
) {
    if (commands == null) return gpu::INVALID_ARGUMENT~;
    return vk_cmd_dispatch(commands, root, groups);
}

fn PreparedDispatchIndirect? lower_dispatch_indirect(
    CommandRecord* commands,
    gpu::GpuAddress root,
    gpu::GpuSpan args,
) {
    CommandRecord* record = commands;
    if (record.state != gpu::internal::CommandState.RECORDING) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    VkDeviceState* state = commands.device_state;
    return {
        .pipeline  = bound_pipeline_snapshot(record)!,
        .args      = lower_indirect_span(
            state:    state,
            span:     &args,
            arg_size: gpu::DispatchIndirectCommand::size,
            count:    1,
        )!,
        .args_span = args,
        .root      = root,
    };
}

fn PreparedDispatchIndirect? full_prepare_dispatch_indirect(
    CommandRecord* commands,
    gpu::GpuAddress root,
    gpu::GpuSpan args,
) {
    if (commands == null) return gpu::INVALID_ARGUMENT~;
    CommandRecord* record = commands;
    if (record == null || record.state != gpu::internal::CommandState.RECORDING) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    VkDeviceState* state = commands.device_state;
    BoundPipeline pipeline = *active_bound_pipeline(
        state:         state,
        record:        record,
        expected_kind: PipelineKind.COMPUTE,
        operation:     "cmd_dispatch_indirect",
    )!;
    return {
        .pipeline  = pipeline,
        .args      = validate_and_lower_indirect_span(
            state:          state,
            record:         record,
            span:           &args,
            arg_size:       gpu::DispatchIndirectCommand::size,
            count:          1,
            operation:      "cmd_dispatch_indirect",
            rejected_field: "args",
        )!,
        .args_span = args,
        .root      = root,
    };
}

fn void? track_dispatch_indirect(
    CommandRecord* commands,
    PreparedDispatchIndirect* dispatch,
) {
    return track_span_command_reference(
        commands.device_state,
        commands,
        &dispatch.args_span,
    );
}

fn void record_dispatch_indirect(
    CommandRecord* commands,
    PreparedDispatchIndirect* dispatch,
) {
    CommandRecord* record = commands;
    push_compute_root(record, &dispatch.pipeline, dispatch.root);
    vk::cmd_dispatch_indirect(
        record.command_buffer,
        dispatch.args.slot.buffer,
        (vk::DeviceSize)dispatch.args.offset,
    );
}

fn void? trusted_dispatch_indirect(
    CommandRecord* commands,
    gpu::GpuAddress root,
    gpu::GpuSpan args,
) {
    PreparedDispatchIndirect dispatch = lower_dispatch_indirect(
        commands,
        root,
        args,
    )!;
    record_dispatch_indirect(commands, &dispatch);
}

fn void? full_dispatch_indirect(
    CommandRecord* commands,
    gpu::GpuAddress root,
    gpu::GpuSpan args,
) {
    PreparedDispatchIndirect dispatch = full_prepare_dispatch_indirect(
        commands,
        root,
        args,
    )!;
    track_dispatch_indirect(commands, &dispatch)!;
    record_dispatch_indirect(commands, &dispatch);
}

fn void? validate_generated_work_count(
    VkDeviceState* state,
    uint count,
    ZString operation,
    ZString rejected_field,
) {
    if (count <= state.caps.max_generated_work_count) return;
    return report_contract_failure(
        state:               state,
        operation:           operation,
        public_fault:        gpu::INVALID_ARGUMENT,
        resource_kind:       gpu::DebugResourceKind.NONE,
        has_public_identity: false,
        index:               0,
        generation:          0,
        debug_name:          null,
        rejected_field:      rejected_field,
        invariant:           "count must not exceed DeviceCaps.max_generated_work_count",
        backend_text:        null,
    )~;
}

fn ResolvedSpan? lower_generated_work_records(
    VkDeviceState* state,
    gpu::GpuSpan* records,
    usz record_size,
    uint max_count,
) {
    usz required = gpu::internal::generated_work_bytes(
        record_size,
        max_count,
        uint::max,
    )!;
    ResolvedSpan resolved = resolve_span_slot(state, records, required)!;
    if (resolved.offset % 8 != 0) return gpu::INVALID_ARGUMENT~;
    return resolved;
}

fn ResolvedSpan? validate_and_lower_generated_work_records(
    VkDeviceState* state,
    CommandRecord* record,
    gpu::GpuSpan* records,
    usz record_size,
    uint max_count,
    ZString operation,
) {
    validate_generated_work_count(state, max_count, operation, "max_count")!;
    usz required = gpu::internal::generated_work_bytes(
        record_size,
        max_count,
        state.caps.max_generated_work_count,
    )!;
    ResolvedSpan resolved = resolve_span_slot(
        state:          state,
        span:           records,
        required:       required,
        operation:      operation,
        rejected_field: "records",
    )!;
    validate_span_recording_access_only(
        state:     state,
        record:    record,
        resolved:  &resolved,
        span:      records,
        operation: operation,
    )!;
    if (!resolved.slot.usage.indirect || !resolved.slot.usage.addressable) {
        return gpu::INVALID_ARGUMENT~;
    }
    if (resolved.offset % 8 != 0) return gpu::INVALID_ARGUMENT~;
    return resolved;
}

fn void set_generated_dispatch_layout_read_hook(
    VkDeviceState* state,
    GeneratedDispatchLayoutReadHook hook,
    void* user_data,
) @if(env::TESTING) {
    state.generated_dispatch_layout_read_hook = hook;
    state.generated_dispatch_layout_read_user_data = user_data;
    state.generated_dispatch_layout_read_observed =
        (vk::IndirectCommandsLayoutEXT)null;
}

fn void? execute_generated_work(
    VkDeviceState* state,
    CommandRecord* record,
    BoundPipeline* pipeline,
    gpu::GeneratedWorkKind kind,
    vk::IndirectCommandsLayoutEXT layout,
    vk::ShaderStageFlags shader_stages,
    ResolvedSpan* records,
    ResolvedSpan* count,
    uint max_count,
    ResolvedSpan* index_span = null,
    gpu::IndexType index_type = gpu::IndexType.U32,
) {
    vk::GeneratedCommandsPipelineInfoEXT pipeline_info =
        vk::generated_commands_pipeline_info_ext()
            .set_pipeline(pipeline.pipeline);
    GeneratedPreprocessBuffer* preprocess =
        acquire_generated_preprocess_buffer(
            state:     state,
            record:    record,
            pipeline:  pipeline.handle,
            kind:      kind,
            max_count: max_count,
        )!;
    vk::GeneratedCommandsInfoEXT info = vk::generated_commands_info_ext()
        .set_next(&pipeline_info)
        .set_shader_stages(shader_stages)
        .set_indirect_commands_layout(layout)
        .set_indirect_address((vk::DeviceAddress)(records.slot.gpu_base + (gpu::GpuAddress)records.offset))
        .set_indirect_address_size((vk::DeviceSize)records.size)
        .set_max_sequence_count(max_count)
        .set_sequence_count_address((vk::DeviceAddress)(count.slot.gpu_base + (gpu::GpuAddress)count.offset));
    if (preprocess.size > 0) {
        info = info
            .set_preprocess_address(preprocess.address)
            .set_preprocess_size(preprocess.size);
    }
    if (index_span != null) {
        bind_index_buffer(
            cb:         record.command_buffer,
            index_span: index_span,
            index_type: index_type,
        );
    }
    state.device_dispatch.generated_work.cmd_execute_generated_commands(
        record.command_buffer,
        vk::FALSE,
        &info,
    );
}

fn vk::IndirectCommandsLayoutEXT? snapshot_generated_dispatch_layout(
    VkDeviceState* state,
    BoundPipeline* pipeline,
) {
    $if env::TESTING:
        if (state.generated_dispatch_layout_read_hook != null) {
            state.generated_dispatch_layout_read_hook(
                state.generated_dispatch_layout_read_user_data,
            );
        }
    $endif
    vk::IndirectCommandsLayoutEXT layout =
        pipeline.generated_dispatch_layout;
    $if env::TESTING:
        state.generated_dispatch_layout_read_observed = layout;
    $endif
    if (layout == null) return gpu::BACKEND_ERROR~;
    return layout;
}

fn PreparedGeneratedDispatch? lower_generated_dispatch(
    CommandRecord* commands,
    gpu::GpuSpan records,
    gpu::GpuSpan count_span,
    uint max_dispatch_count,
) {
    CommandRecord* record = commands;
    if (record.state != gpu::internal::CommandState.RECORDING) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    VkDeviceState* state = commands.device_state;
    PreparedGeneratedDispatch dispatch = {
        .pipeline     = bound_pipeline_snapshot(record)!,
        .records_span = records,
        .count_span   = count_span,
        .max_count    = max_dispatch_count,
    };
    if (!state.generated_work.enabled) return gpu::UNSUPPORTED_FEATURE~;
    if (max_dispatch_count == 0) return dispatch;

    dispatch.records = lower_generated_work_records(
        state:       state,
        records:     &records,
        record_size: gpu::GeneratedDispatchRecord::size,
        max_count:   max_dispatch_count,
    )!;
    dispatch.count = lower_indirect_span(
        state:    state,
        span:     &count_span,
        arg_size: uint::size,
        count:    1,
    )!;
    dispatch.layout = snapshot_generated_dispatch_layout(
        state,
        &dispatch.pipeline,
    )!;
    return dispatch;
}

fn PreparedGeneratedDispatch? full_prepare_generated_dispatch(
    CommandRecord* commands,
    gpu::GpuSpan records,
    gpu::GpuSpan count_span,
    uint max_dispatch_count,
) {
    if (commands == null) return gpu::INVALID_ARGUMENT~;
    CommandRecord* record = commands;
    if (record == null || record.state != gpu::internal::CommandState.RECORDING) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    VkDeviceState* state = commands.device_state;
    PreparedGeneratedDispatch dispatch = {
        .pipeline = *active_bound_pipeline(
            state:         state,
            record:        record,
            expected_kind: PipelineKind.COMPUTE,
            operation:     "cmd_dispatch_generated",
        )!,
        .records_span = records,
        .count_span   = count_span,
        .max_count    = max_dispatch_count,
    };
    if (!state.generated_work.enabled) return gpu::UNSUPPORTED_FEATURE~;
    if (max_dispatch_count == 0) return dispatch;

    dispatch.records = validate_and_lower_generated_work_records(
        state:       state,
        record:      record,
        records:     &records,
        record_size: gpu::GeneratedDispatchRecord::size,
        max_count:   max_dispatch_count,
        operation:   "cmd_dispatch_generated",
    )!;
    dispatch.count = validate_and_lower_indirect_span(
        state:          state,
        record:         record,
        span:           &count_span,
        arg_size:       uint::size,
        count:          1,
        operation:      "cmd_dispatch_generated",
        rejected_field: "count_span",
    )!;
    dispatch.layout = snapshot_generated_dispatch_layout(
        state,
        &dispatch.pipeline,
    )!;
    return dispatch;
}

fn void? track_generated_dispatch(
    CommandRecord* commands,
    PreparedGeneratedDispatch* dispatch,
) {
    VkDeviceState* state = commands.device_state;
    CommandRecord* record = commands;
    CommandResourceReference[2] references = {
        {
            .owner      = dispatch.records_span.owner,
            .index      = dispatch.records_span.index,
            .generation = dispatch.records_span.generation,
        },
        {
            .owner      = dispatch.count_span.owner,
            .index      = dispatch.count_span.index,
            .generation = dispatch.count_span.generation,
        },
    };
    preflight_command_references(record, references[..])!;
    track_span_command_reference(state, record, &dispatch.records_span)!;
    track_span_command_reference(state, record, &dispatch.count_span)!;
}

fn void? record_generated_dispatch(
    CommandRecord* commands,
    PreparedGeneratedDispatch* dispatch,
) {
    if (dispatch.max_count == 0) return;
    VkDeviceState* state = commands.device_state;
    CommandRecord* record = commands;
    execute_generated_work(
        state:         state,
        record:        record,
        pipeline:      &dispatch.pipeline,
        kind:          gpu::GeneratedWorkKind.DISPATCH,
        layout:        dispatch.layout,
        shader_stages: vk::ShaderStageFlagBits.COMPUTE_BIT,
        records:       &dispatch.records,
        count:         &dispatch.count,
        max_count:     dispatch.max_count,
    )!;
}

fn void? trusted_dispatch_generated(
    CommandRecord* commands,
    gpu::GpuSpan records,
    gpu::GpuSpan count_span,
    uint max_dispatch_count,
) {
    PreparedGeneratedDispatch dispatch = lower_generated_dispatch(
        commands:           commands,
        records:            records,
        count_span:         count_span,
        max_dispatch_count: max_dispatch_count,
    )!;
    record_generated_dispatch(commands, &dispatch)!;
}

fn void? full_dispatch_generated(
    CommandRecord* commands,
    gpu::GpuSpan records,
    gpu::GpuSpan count_span,
    uint max_dispatch_count,
) {
    PreparedGeneratedDispatch dispatch = full_prepare_generated_dispatch(
        commands:           commands,
        records:            records,
        count_span:         count_span,
        max_dispatch_count: max_dispatch_count,
    )!;
    if (dispatch.max_count == 0) return;
    CommandRecord* record = commands;
    uint checkpoint = record.scratch.reference_count;
    defer catch rollback_command_references(
        record,
        checkpoint,
    );
    track_generated_dispatch(commands, &dispatch)!;
    record_generated_dispatch(commands, &dispatch)!;
}

fn void recycle_generated_preprocess_indices_locked(
    VkDeviceState* state,
    CommandAllocatorSlot* slot,
    uint[] indices,
    uint count,
) {
    for (uint i = 0; i < count; i++) {
        uint index = indices[i];
        if (index < slot.reservations.len) {
            GeneratedPreprocessBuffer* reservation = &slot.reservations[index];
            if (reservation.reservation_in_use) {
                reservation.reservation_in_use = false;
                slot.generated_reservations_in_use--;
            }
        }
        indices[i] = 0;
    }
}

fn void free_generated_preprocess_buffers(
    VkDeviceState* state,
    GeneratedPreprocessBuffer[] buffers,
) {
    foreach (&buffer : buffers) {
        if (buffer.buffer != null && !buffer.reservation_owned) {
            state.allocator.destroy_buffer(buffer.buffer, buffer.allocation);
        }
    }
    if (buffers.ptr != null) alloc::free(state.host_allocator, buffers.ptr);
}

fn void free_reserved_generated_preprocess_buffers(
    VkDeviceState* state,
    GeneratedPreprocessBuffer[] buffers,
) {
    foreach (&buffer : buffers) {
        if (buffer.buffer != null) {
            state.allocator.destroy_buffer(buffer.buffer, buffer.allocation);
        }
    }
    if (buffers.ptr != null) alloc::free(state.host_allocator, buffers.ptr);
}

fn GeneratedPreprocessBuffer? allocate_generated_preprocess_buffer(
    VkDeviceState* state,
    vk::DeviceSize size,
    vk::DeviceSize alignment,
    uint memory_type_bits,
) {
    vk::BufferUsageFlags2CreateInfoKHR usage2 = {
        .s_type = vk::StructureType.BUFFER_USAGE_FLAGS_2_CREATE_INFO_KHR,
        .usage = (vk::BufferUsageFlags2KHR)(
            (ulong)vk::BufferUsageFlagBits2KHR.USAGE_2_PREPROCESS_BUFFER_BIT_EXT
            | (ulong)vk::BufferUsageFlagBits2KHR.USAGE_2_SHADER_DEVICE_ADDRESS_BIT_KHR
        ),
    };
    vk::BufferCreateInfo buffer_info = vk::buffer_create_info()
        .set_next(&usage2)
        .set_size(size)
        .set_usage(vk::BufferUsageFlagBits.SHADER_DEVICE_ADDRESS_BIT);
    vma::AllocationCreateInfo allocation_info = {
        .usage            = vma::MemoryUsage.AUTO_PREFER_DEVICE,
        .memory_type_bits = memory_type_bits,
    };
    GeneratedPreprocessBuffer buffer;
    bool owns_pair;
    defer catch {
        if (owns_pair) state.allocator.destroy_buffer(buffer.buffer, buffer.allocation);
    }
    check_device_result(state, state.allocator.create_buffer_with_alignment(
        bi:            &buffer_info,
        ci:            &allocation_info,
        min_alignment: alignment,
        out_buffer:    &buffer.buffer,
        out_alloc:     &buffer.allocation,
        out_info:      null,
    ))!;
    owns_pair = true;
    vk::BufferDeviceAddressInfo address_info =
        vk::buffer_device_address_info().set_buffer(buffer.buffer);
    buffer.address = vk::get_buffer_device_address(state.device, &address_info);
    buffer.size = size;
    if (buffer.address == 0) return gpu::BACKEND_ERROR~;
    owns_pair = false;
    return buffer;
}

fn GeneratedPreprocessBuffer*? acquire_generated_preprocess_buffer(
    VkDeviceState* state,
    CommandRecord* record,
    gpu::PipelineHandle pipeline,
    gpu::GeneratedWorkKind kind,
    uint max_count,
) {
    CommandAllocatorSlot* slot = lock_command_allocator_handle(
        state,
        record.allocator,
    )!;
    defer slot.mutex.unlock();
    if (record.scratch.generated_preprocess_count
            >= record.scratch.generated_preprocess_indices.len) {
        return gpu::COMMAND_ALLOCATOR_CAPACITY_EXCEEDED~;
    }
    foreach (index, &reserved : slot.reservations) {
        if (!generated_reservation_matches(reserved, pipeline, kind)
            || max_count > reserved.reservation_max_commands
            || reserved.reservation_in_use) continue;
        reserved.reservation_in_use = true;
        slot.generated_reservations_in_use++;
        record.scratch.generated_preprocess_indices[
            record.scratch.generated_preprocess_count++
        ] = (uint)index;
        return reserved;
    }
    return gpu::GENERATED_SCRATCH_EXHAUSTED~;
}

fn vk::IndirectCommandsLayoutEXT? create_generated_work_layout(
    VkDeviceState* state,
    vk::PipelineLayout pipeline_layout,
    vk::ShaderStageFlags shader_stages,
    uint root_size,
    vk::IndirectCommandsTokenTypeEXT action_type,
    uint action_offset,
    uint stride,
) {
    vk::PushConstantRange range = {
        .stage_flags = shader_stages,
        .offset      = 0,
        .size        = root_size,
    };
    vk::IndirectCommandsPushConstantTokenEXT push = {
        .update_range = range,
    };
    vk::IndirectCommandsLayoutTokenEXT[2] tokens = {
        vk::indirect_commands_layout_token_ext()
            .set_type(vk::IndirectCommandsTokenTypeEXT.PUSH_CONSTANT_EXT)
            .set_data({ .p_push_constant = &push })
            .set_offset(0),
        vk::indirect_commands_layout_token_ext()
            .set_type(action_type)
            .set_offset(action_offset),
    };
    vk::IndirectCommandsLayoutCreateInfoEXT info =
        vk::indirect_commands_layout_create_info_ext()
            .set_shader_stages(shader_stages)
            .set_indirect_stride(stride)
            .set_pipeline_layout(pipeline_layout)
            .set_tokens(tokens[..]);
    vk::IndirectCommandsLayoutEXT layout;
    check_backend_result(
        state:     state,
        operation: "create_generated_work_layout",
        result:    state.device_dispatch.generated_work.create_layout(
            state.device,
            &info,
            null,
            &layout,
        ),
    )!;
    return layout;
}
