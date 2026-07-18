module gpu::vk @private;

import gpu @public;
import vk;
import vma;
import std::core::mem;
import std::atomic;
import std::atomic::types;

// Backend objects whose release must outlive in-flight frames. Growable FIFO
// keyed by retire_timeline_value; push order is non-decreasing, so head-drain
// is exact.

const uint DEFERRED_QUEUE_INITIAL_CAP = 64;

enum DeferredKind : int {
    BUFFER,
    IMAGE,
    PLACED_IMAGE,
    IMAGE_VIEW,
    PIPELINE,
    SHADER_MODULE,
}

struct DeferredEntry {
    DeferredKind    kind;
    ulong           timeline_value;
    void*              object;
    vma::Allocation    allocation;
    gpu::GpuAllocation placement;
    usz                placement_offset;
    usz                placement_size;
}

struct DeferredQueue {
    DeferredEntry[] entries;
    uint            head;
    uint            count;
}

<* Drain accounting, whitebox-exposed via deferred_stats. *>
struct DeferredStats {
    ulong pushed;
    ulong drained;
    ulong drain_calls;
    ulong host_queries;
}

fn void free_deferred_entry(VkDeviceState* state, DeferredEntry* entry) @private {
    switch (entry.kind) {
        case DeferredKind.BUFFER:
            state.allocator.destroy_buffer((vk::Buffer)entry.object, entry.allocation);
        case DeferredKind.IMAGE:
            state.allocator.destroy_image((vk::Image)entry.object, entry.allocation);
        case DeferredKind.PLACED_IMAGE:
            vk::destroy_image(state.device, (vk::Image)entry.object, null);
            AllocationSlot* slot = state.allocations.get(entry.placement)!!;
            slot.live_placements--;
        case DeferredKind.IMAGE_VIEW:
            vk::destroy_image_view(state.device, (vk::ImageView)entry.object, null);
        case DeferredKind.PIPELINE:
            vk::destroy_pipeline(state.device, (vk::Pipeline)entry.object, null);
        case DeferredKind.SHADER_MODULE:
            vk::destroy_shader_module(state.device, (vk::ShaderModule)entry.object, null);
    }
}

<* Caller must hold resource_mutex. *>
fn void grow_deferred_queue(VkDeviceState* state) @private {
    DeferredQueue* queue = &state.deferred;
    usz new_cap = queue.entries.len == 0 ? DEFERRED_QUEUE_INITIAL_CAP : queue.entries.len * 2;
    DeferredEntry[] grown = alloc::new_array(state.host_allocator, DeferredEntry, (sz)new_cap);
    for (uint i = 0; i < queue.count; i++) {
        grown[i] = queue.entries[(queue.head + i) % (uint)queue.entries.len];
    }
    if (queue.entries.ptr != null) alloc::free(state.host_allocator, queue.entries.ptr);
    queue.entries = grown;
    queue.head = 0;
}

fn void drain_deferred_to_bound(VkDeviceState* state, ulong bound) @private {
    DeferredQueue* queue = &state.deferred;
    while (queue.count > 0) {
        DeferredEntry* entry = &queue.entries[queue.head];
        if (entry.timeline_value > bound) break;
        free_deferred_entry(state, entry);
        queue.head = (queue.head + 1) % (uint)queue.entries.len;
        queue.count--;
        state.deferred_stats.drained++;
    }
}

fn bool deferred_placement_overlaps(
    VkDeviceState* state,
    gpu::GpuAllocation placement,
    usz offset,
    usz size,
) {
    DeferredQueue* queue = &state.deferred;
    for (uint i = 0; i < queue.count; i++) {
        DeferredEntry* entry =
            &queue.entries[(queue.head + i) % (uint)queue.entries.len];
        if (entry.kind != DeferredKind.PLACED_IMAGE
            || entry.placement != placement) {
            continue;
        }
        if (offset < entry.placement_offset + entry.placement_size
            && entry.placement_offset < offset + size) {
            return true;
        }
    }
    return false;
}

fn void deferred_drain_shadow(VkDeviceState* state) @private {
    state.deferred_stats.drain_calls++;
    drain_deferred_to_bound(state, state.signaled_shadow);
}

<* Caller must hold resource_mutex; also runs an opportunistic head-drain. *>
fn void enqueue_deferred(
    VkDeviceState* state,
    DeferredKind kind,
    ulong timeline_value,
    void* object,
    vma::Allocation allocation,
    gpu::GpuAllocation placement = {},
    usz placement_offset = 0,
    usz placement_size = 0,
) @private {
    DeferredQueue* queue = &state.deferred;
    if (queue.count == queue.entries.len) grow_deferred_queue(state);
    uint idx = (queue.head + queue.count) % (uint)queue.entries.len;
    queue.entries[idx] = {
        .kind             = kind,
        .timeline_value   = timeline_value,
        .object           = object,
        .allocation       = allocation,
        .placement        = placement,
        .placement_offset = placement_offset,
        .placement_size   = placement_size,
    };
    queue.count++;
    state.deferred_stats.pushed++;
    deferred_drain_shadow(state);
}

<* Caller must hold resource_mutex. Cheapest tier first; tiers 2/3 free only
   against proven-signaled bounds, tier 1 is a pure skip. *>
fn void? deferred_drain(VkDeviceState* state) @private {
    DeferredQueue* queue = &state.deferred;
    state.deferred_stats.drain_calls++;
    if (queue.count == 0) return;

    DeferredEntry* head = &queue.entries[queue.head];
    ulong counter = state.timeline_counter.load(AtomicOrdering.RELAXED);
    if (head.timeline_value > counter) return;

    if (head.timeline_value > state.signaled_shadow) {
        ulong queried = 0;
        state.deferred_stats.host_queries++;
        check_device_result(state, vk::get_semaphore_counter_value(state.device, state.frame_timeline, &queried))!;
        if (queried > state.signaled_shadow) state.signaled_shadow = queried;
    }

    drain_deferred_to_bound(state, state.signaled_shadow);
}

<* Teardown only: an idle device makes every entry free-able. *>
fn void deferred_drain_all(VkDeviceState* state) @private {
    DeferredQueue* queue = &state.deferred;
    while (queue.count > 0) {
        DeferredEntry* entry = &queue.entries[queue.head];
        free_deferred_entry(state, entry);
        queue.head = (queue.head + 1) % (uint)queue.entries.len;
        queue.count--;
        state.deferred_stats.drained++;
    }
}

fn void destroy_deferred_queue(VkDeviceState* state) @private {
    if (state.deferred.entries.ptr != null) {
        alloc::free(state.host_allocator, state.deferred.entries.ptr);
        state.deferred.entries = {};
    }
}

<* Live entry count, exposed for whitebox drain accounting in tests. *>
fn uint deferred_queue_count(VkDeviceState* state) {
    return state.deferred.count;
}

<* Snapshot of drain counters, exposed for whitebox accounting in tests. *>
fn DeferredStats deferred_stats(VkDeviceState* state) {
    return state.deferred_stats;
}
