module gpu::vk;

import gpu;
import vk;
import vma;
import std::core::mem;

// Backend objects whose release must outlive in-flight frames. Growable FIFO
// keyed by retire_timeline_value; push order is non-decreasing, so head-drain
// is exact.

const uint DEFERRED_QUEUE_INITIAL_CAP = 64;

enum DeferredKind : int {
    BUFFER,
    IMAGE,
    IMAGE_VIEW,
    PIPELINE,
    SHADER_MODULE,
    SEMAPHORE,
}

struct DeferredEntry {
    DeferredKind    kind;
    ulong           timeline_value;
    void*           object;
    vma::Allocation allocation;
}

struct DeferredQueue {
    DeferredEntry[] entries;
    uint            head;
    uint            count;
}

fn void free_deferred_entry(VkDeviceState* state, DeferredEntry* entry) @private {
    switch (entry.kind) {
        case DeferredKind.BUFFER:
            state.allocator.destroy_buffer((vk::Buffer)entry.object, entry.allocation);
        case DeferredKind.IMAGE:
            state.allocator.destroy_image((vk::Image)entry.object, entry.allocation);
        case DeferredKind.IMAGE_VIEW:
            vk::destroy_image_view(state.device, (vk::ImageView)entry.object, null);
        case DeferredKind.PIPELINE:
            vk::destroy_pipeline(state.device, (vk::Pipeline)entry.object, null);
        case DeferredKind.SHADER_MODULE:
            vk::destroy_shader_module(state.device, (vk::ShaderModule)entry.object, null);
        case DeferredKind.SEMAPHORE:
            vk::destroy_semaphore(state.device, (vk::Semaphore)entry.object, null);
    }
}

<* Growth allocates on the destroying thread; the test TrackingAllocator
   tolerates that only because storm tests drain immediately. *>
fn void grow_deferred_queue(DeferredQueue* queue) @private {
    usz new_cap = queue.entries.len == 0 ? DEFERRED_QUEUE_INITIAL_CAP : queue.entries.len * 2;
    DeferredEntry[] grown = mem::new_array(DeferredEntry, (sz)new_cap);
    for (uint i = 0; i < queue.count; i++) {
        grown[i] = queue.entries[(queue.head + i) % (uint)queue.entries.len];
    }
    if (queue.entries.ptr != null) mem::free(queue.entries.ptr);
    queue.entries = grown;
    queue.head = 0;
}

<* Caller must hold resource_mutex; also runs an opportunistic head-drain. *>
fn void enqueue_deferred(VkDeviceState* state, DeferredKind kind, ulong timeline_value, void* object, vma::Allocation allocation) @private {
    DeferredQueue* queue = &state.deferred;
    if (queue.count == queue.entries.len) grow_deferred_queue(queue);
    uint idx = (queue.head + queue.count) % (uint)queue.entries.len;
    queue.entries[idx] = { .kind = kind, .timeline_value = timeline_value, .object = object, .allocation = allocation };
    queue.count++;
    deferred_drain(state);
}

<* Caller must hold resource_mutex. *>
fn void deferred_drain(VkDeviceState* state) @private {
    DeferredQueue* queue = &state.deferred;
    if (queue.count == 0) return;
    ulong current = 0;
    vk::get_semaphore_counter_value(state.device, state.frame_timeline, &current);
    while (queue.count > 0) {
        DeferredEntry* entry = &queue.entries[queue.head];
        if (entry.timeline_value > current) break;
        free_deferred_entry(state, entry);
        queue.head = (queue.head + 1) % (uint)queue.entries.len;
        queue.count--;
    }
}

<* Teardown only: an idle device makes every entry free-able. *>
fn void deferred_drain_all(VkDeviceState* state) @private {
    DeferredQueue* queue = &state.deferred;
    while (queue.count > 0) {
        DeferredEntry* entry = &queue.entries[queue.head];
        free_deferred_entry(state, entry);
        queue.head = (queue.head + 1) % (uint)queue.entries.len;
        queue.count--;
    }
}

fn void destroy_deferred_queue(VkDeviceState* state) @private {
    if (state.deferred.entries.ptr != null) {
        mem::free(state.deferred.entries.ptr);
        state.deferred.entries = {};
    }
}

<* Live entry count, exposed for whitebox drain accounting in tests. *>
fn uint deferred_queue_count(VkDeviceState* state) {
    return state.deferred.count;
}
