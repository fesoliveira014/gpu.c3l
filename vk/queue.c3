module gpu::vk;

import gpu;
import vk;
import std::thread;
import std::core::mem;

struct QueueFamilies {
    uint graphics;
    uint compute;
    uint compute_index;
    uint transfer;
    bool async_compute;
}

<* Compute tiers: a second queue in the main family when it has one, else a
   compute-capable non-graphics family, else the graphics queue itself. A
   transfer-only family is used for transfer when present. *>
fn QueueFamilies? pick_queue_families(vk::QueueFamilyProperties[] families) {
    int main_family = -1;
    int compute_only = -1;
    int transfer_only = -1;
    foreach (i, family : families) {
        uint flags = (uint)family.queue_flags;
        bool graphics = (flags & (uint)vk::QueueFlagBits.GRAPHICS_BIT) != 0;
        bool compute  = (flags & (uint)vk::QueueFlagBits.COMPUTE_BIT) != 0;
        bool transfer = (flags & (uint)vk::QueueFlagBits.TRANSFER_BIT) != 0;
        if (graphics && compute && main_family < 0) main_family = (int)i;
        if (compute && !graphics && compute_only < 0) compute_only = (int)i;
        if (transfer && !graphics && !compute && transfer_only < 0) transfer_only = (int)i;
    }

    if (main_family < 0) return gpu::UNSUPPORTED_FEATURE~;
    uint main = (uint)main_family;
    QueueFamilies picked = {
        .graphics = main,
        .compute = main,
        .compute_index = 0,
        .transfer = transfer_only < 0 ? main : (uint)transfer_only,
    };
    if (families[main].queue_count >= 2) {
        picked.compute_index = 1;
        picked.async_compute = true;
    } else if (compute_only >= 0) {
        picked.compute = (uint)compute_only;
        picked.async_compute = true;
    }
    return picked;
}

fn QueueFamilies? select_queue_families(vk::PhysicalDevice physical_device) {
    uint count = 0;
    vk::get_physical_device_queue_family_properties(physical_device, &count, null);
    if (count == 0) return gpu::UNSUPPORTED_FEATURE~;

    @pool() {
        vk::QueueFamilyProperties[] families = mem::talloc_array(vk::QueueFamilyProperties, (sz)count);
        vk::get_physical_device_queue_family_properties(physical_device, &count, families.ptr);
        return pick_queue_families(families[:count]);
    };
}

// Graphics, compute, transfer — the most distinct families a device uses.
const uint MAX_QUEUE_FAMILIES = 3;

<* Distinct queue families in use, for CONCURRENT-shared resources. *>
fn uint used_queue_families(VkDeviceState* state, uint* out) {
    uint count = 0;
    out[count++] = state.graphics_family;
    if (state.compute_family != state.graphics_family) out[count++] = state.compute_family;
    if (state.transfer_family != state.graphics_family && state.transfer_family != state.compute_family) {
        out[count++] = state.transfer_family;
    }
    return count;
}

fn vk::Queue queue_for(VkDeviceState* state, gpu::QueueKind queue) {
    switch (queue) {
        case gpu::QueueKind.TRANSFER: return state.transfer_queue;
        case gpu::QueueKind.COMPUTE:  return state.compute_queue;
        default:                      return state.graphics_queue;
    }
}

<* Submit command lists to their queue via vkQueueSubmit2, waiting on and
   signaling the requested timeline values at their stages. Empty wait/signal/
   command arrays are left unset so the slice setters are never handed a
   zero-length slice. *>
fn Mutex* queue_mutex_for(VkDeviceState* state, gpu::QueueKind kind) @private {
    switch (kind) {
        case gpu::QueueKind.TRANSFER: return &state.transfer_queue_mutex;
        case gpu::QueueKind.COMPUTE:  return state.compute_queue_distinct ? &state.compute_queue_mutex : &state.graphics_queue_mutex;
        default:                      return &state.graphics_queue_mutex;
    }
}

fn void? vk_submit(gpu::Device* device, gpu::SubmitDesc* desc) @private {
    VkDeviceState* state = (VkDeviceState*)device.backend_state;
    if (desc.command_lists.len == 0 && desc.waits.len == 0 && desc.signals.len == 0) return;

    @pool() {
        vk::SubmitInfo2 submit = vk::submit_info2();

        if (desc.command_lists.len > 0) {
            vk::CommandBufferSubmitInfo[] cmds = mem::talloc_array(vk::CommandBufferSubmitInfo, desc.command_lists.len);
            foreach (i, &cl : desc.command_lists) {
                if (cl.state != gpu::CommandState.EXECUTABLE) return gpu::COMMAND_RECORDING_ERROR~;
                cmds[i] = vk::command_buffer_submit_info().set_command_buffer((vk::CommandBuffer)cl.backend);
            }
            submit = submit.set_command_buffer_infos(cmds);
        }

        // A named swapchain couples this submit to its pending acquire: wait
        // the binary acquire semaphore, signal the image's present semaphore.
        SwapchainSlot* swapchain = null;
        if (desc.swapchain.is_valid()) {
            swapchain = state.swapchains.get(desc.swapchain)!;
            if (!swapchain.acquire_pending) return gpu::INVALID_RESOURCE_STATE~;
        }

        usz wait_count = desc.waits.len + (swapchain != null ? 1 : 0);
        if (wait_count > 0) {
            vk::SemaphoreSubmitInfo[] waits = mem::talloc_array(vk::SemaphoreSubmitInfo, (sz)wait_count);
            foreach (i, w : desc.waits) {
                SemaphoreSlot* slot = state.semaphores.get(w.semaphore)!;
                waits[i] = vk::semaphore_submit_info()
                    .set_semaphore(slot.semaphore)
                    .set_value((ulong)w.value)
                    .set_stage_mask(stage_to_vk(w.stage));
            }
            if (swapchain != null) {
                waits[desc.waits.len] = vk::semaphore_submit_info()
                    .set_semaphore(swapchain.pending_acquire_semaphore)
                    .set_stage_mask(vk::PipelineStageFlagBits2.STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT);
            }
            submit = submit.set_wait_semaphore_infos(waits);
        }

        usz signal_count = desc.signals.len + (swapchain != null ? 1 : 0);
        if (signal_count > 0) {
            vk::SemaphoreSubmitInfo[] signals = mem::talloc_array(vk::SemaphoreSubmitInfo, (sz)signal_count);
            foreach (i, s : desc.signals) {
                SemaphoreSlot* slot = state.semaphores.get(s.semaphore)!;
                signals[i] = vk::semaphore_submit_info()
                    .set_semaphore(slot.semaphore)
                    .set_value((ulong)s.value)
                    .set_stage_mask(stage_to_vk(s.stage));
            }
            if (swapchain != null) {
                signals[desc.signals.len] = vk::semaphore_submit_info()
                    .set_semaphore(swapchain.present_semaphores[swapchain.pending_image])
                    .set_stage_mask(vk::PipelineStageFlagBits2.STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT);
            }
            submit = submit.set_signal_semaphore_infos(signals);
        }

        gpu::QueueKind kind = desc.command_lists.len > 0 ? desc.command_lists[0].queue : gpu::QueueKind.GRAPHICS;
        Mutex* queue_mutex = queue_mutex_for(state, kind);
        queue_mutex.lock()!!;
        defer queue_mutex.unlock();
        check(vk::queue_submit2(queue_for(state, kind), 1, &submit, (vk::Fence)null))!;
        foreach (&cl : desc.command_lists) cl.state = gpu::CommandState.SUBMITTED;
    };
}

fn void? vk_wait_queue_idle(gpu::Device* device, gpu::QueueKind queue) @private {
    VkDeviceState* state = (VkDeviceState*)device.backend_state;
    Mutex* queue_mutex = queue_mutex_for(state, queue);
    queue_mutex.lock()!!;
    defer queue_mutex.unlock();
    check(vk::queue_wait_idle(queue_for(state, queue)))!;
}
