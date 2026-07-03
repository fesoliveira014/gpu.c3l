module gpu::vk;

import gpu;
import vk;
import std::core::mem;

struct QueueFamilies {
    uint graphics;
    uint compute;
    uint transfer;
}

<* A graphics+compute family backs both; a transfer-only family is used for
   transfer when present, else the main family is reused. *>
fn QueueFamilies? select_queue_families(vk::PhysicalDevice physical_device) {
    uint count = 0;
    vk::get_physical_device_queue_family_properties(physical_device, &count, null);
    if (count == 0) return gpu::UNSUPPORTED_FEATURE~;

    @pool() {
        vk::QueueFamilyProperties[] families = mem::talloc_array(vk::QueueFamilyProperties, (sz)count);
        vk::get_physical_device_queue_family_properties(physical_device, &count, families.ptr);

        int main_family = -1;
        int transfer_only = -1;
        foreach (i, family : families) {
            uint flags = (uint)family.queue_flags;
            bool graphics = (flags & (uint)vk::QueueFlagBits.GRAPHICS_BIT) != 0;
            bool compute  = (flags & (uint)vk::QueueFlagBits.COMPUTE_BIT) != 0;
            bool transfer = (flags & (uint)vk::QueueFlagBits.TRANSFER_BIT) != 0;
            if (graphics && compute && main_family < 0) main_family = (int)i;
            if (transfer && !graphics && !compute && transfer_only < 0) transfer_only = (int)i;
        }

        if (main_family < 0) return gpu::UNSUPPORTED_FEATURE~;
        uint main = (uint)main_family;
        uint transfer = transfer_only < 0 ? main : (uint)transfer_only;
        return (QueueFamilies){ .graphics = main, .compute = main, .transfer = transfer };
    };
}

fn vk::Queue queue_for(VkDeviceState* state, gpu::QueueKind queue) {
    switch (queue) {
        case gpu::QueueKind.TRANSFER: return state.transfer_queue;
        default:                      return state.graphics_queue;
    }
}

<* Submit command lists to their queue via vkQueueSubmit2, waiting on and
   signaling the requested timeline values at their stages. Empty wait/signal/
   command arrays are left unset so the slice setters are never handed a
   zero-length slice. *>
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
        check(vk::queue_submit2(queue_for(state, kind), 1, &submit, (vk::Fence)null))!;
        foreach (&cl : desc.command_lists) cl.state = gpu::CommandState.SUBMITTED;
    };
}

fn void? vk_wait_queue_idle(gpu::Device* device, gpu::QueueKind queue) @private {
    VkDeviceState* state = (VkDeviceState*)device.backend_state;
    check(vk::queue_wait_idle(queue_for(state, queue)))!;
}
