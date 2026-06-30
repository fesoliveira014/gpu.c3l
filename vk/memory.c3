module gpu::vk;

import gpu;
import vk;
import std::core::mem;

// Per-frame upload arena: a mapped, addressable buffer bump-allocated within a
// frame, reset once its timeline value retires.
struct FrameArenaState {
    gpu::BufferHandle backing_buffer;
    gpu::GpuAddress   gpu_base;
    void*             cpu_base;
    usz               size;
    usz               cursor;
    ulong             frame_timeline_value;
}

<* Create the frame timeline semaphore and one arena buffer per frame-in-flight.
   Arena buffers are real device buffers, so the device's buffer sweep frees
   them; only the host array and the semaphore are released here. *>
fn void? create_frame_resources(VkDeviceState* state, gpu::DeviceDesc* desc) {
    state.frames_in_flight = desc.frames_in_flight == 0 ? 2 : desc.frames_in_flight;
    state.frame_index = 0;
    state.timeline_counter = 0;

    vk::SemaphoreTypeCreateInfo type_info = vk::semaphore_type_create_info()
        .set_semaphore_type(vk::SemaphoreType.TIMELINE)
        .set_initial_value(0);
    vk::SemaphoreCreateInfo semaphore_info = vk::semaphore_create_info().set_next(&type_info);
    check(vk::create_semaphore(state.device, &semaphore_info, null, &state.frame_timeline))!;

    state.frame_arenas = mem::new_array(FrameArenaState, state.frames_in_flight);
    gpu::Device handle = { .backend = gpu::BackendKind.VULKAN, .vtable = &VK_VTABLE, .backend_state = (void*)state };
    for (uint i = 0; i < state.frames_in_flight; i++) {
        gpu::BufferDesc arena_desc = {
            .size        = gpu::ARENA_SIZE,
            .usage       = { .storage, .addressable },
            .memory_kind = gpu::MemoryKind.FRAME_UPLOAD,
            .debug_name  = "frame_arena",
        };
        gpu::BufferHandle buffer = vk_create_buffer(&handle, &arena_desc)!;
        gpu::GpuSpan span = vk_get_buffer_span(&handle, buffer)!;
        state.frame_arenas[i] = {
            .backing_buffer = buffer,
            .gpu_base       = span.gpu,
            .cpu_base       = span.cpu,
            .size           = span.size,
        };
    }
}

fn void destroy_frame_resources(VkDeviceState* state) @private {
    if (state.frame_arenas.ptr != null) {
        mem::free(state.frame_arenas.ptr);
        state.frame_arenas = {};
    }
    if (state.frame_timeline != null) {
        vk::destroy_semaphore(state.device, state.frame_timeline, null);
        state.frame_timeline = (vk::Semaphore)null;
    }
}

// True once the device timeline has reached the slot's recorded value.
fn bool frame_arena_retired(VkDeviceState* state, FrameArenaState* slot) {
    ulong value = 0;
    vk::get_semaphore_counter_value(state.device, state.frame_timeline, &value);
    return value >= slot.frame_timeline_value;
}

<* Reset an arena slot's cursor. Rejects if the slot has not yet retired. *>
fn void? reset_frame_arena(VkDeviceState* state, uint index) {
    FrameArenaState* slot = &state.frame_arenas[index];
    if (!frame_arena_retired(state, slot)) return gpu::INVALID_RESOURCE_STATE~;
    slot.cursor = 0;
}

fn void? vk_begin_frame(gpu::Device* device) @private {
    VkDeviceState* state = (VkDeviceState*)device.backend_state;
    state.frame_index = (state.frame_index + 1) % state.frames_in_flight;
    FrameArenaState* slot = &state.frame_arenas[state.frame_index];

    if (slot.frame_timeline_value > 0) {
        ulong wait_value = slot.frame_timeline_value;
        vk::SemaphoreWaitInfo wait_info = vk::semaphore_wait_info()
            .set_semaphore_count(1)
            .set_semaphores((&state.frame_timeline)[:1])
            .set_values((&wait_value)[:1]);
        check(vk::wait_semaphores(state.device, &wait_info, 1_000_000_000))!;
    }

    reset_frame_arena(state, state.frame_index)!;
    state.allocator.set_current_frame_index(state.frame_index);
}

fn gpu::GpuSpan? vk_alloc_frame_span(gpu::Device* device, usz size, usz align) @private {
    VkDeviceState* state = (VkDeviceState*)device.backend_state;
    FrameArenaState* slot = &state.frame_arenas[state.frame_index];

    usz aligned = gpu::align_up(slot.cursor, align);
    if (aligned + size > slot.size) return gpu::ARENA_FULL~;

    gpu::GpuSpan span = {
        .gpu    = (gpu::GpuAddress)((ulong)slot.gpu_base + aligned),
        .cpu    = slot.cpu_base == null ? null : (char*)slot.cpu_base + aligned,
        .size   = size,
        .buffer = slot.backing_buffer,
        .offset = aligned,
        .kind   = gpu::MemoryKind.FRAME_UPLOAD,
    };
    slot.cursor = aligned + size;
    return span;
}

fn void? vk_end_frame(gpu::Device* device) @private {
    VkDeviceState* state = (VkDeviceState*)device.backend_state;
    FrameArenaState* slot = &state.frame_arenas[state.frame_index];

    state.timeline_counter++;
    slot.frame_timeline_value = state.timeline_counter;

    vk::SemaphoreSignalInfo signal_info = vk::semaphore_signal_info()
        .set_semaphore(state.frame_timeline)
        .set_value(state.timeline_counter);
    check(vk::signal_semaphore(state.device, &signal_info))!;
}
