module gpu::vk;

import gpu @public;
import vk;
import std::thread;
import std::core::mem;

// Graphics, compute, transfer — the most distinct families a device uses.
const uint MAX_QUEUE_FAMILIES = 3;

struct QueueFamilies {
    uint graphics;
    uint compute;
    uint compute_index;
    uint transfer;
    bool async_compute;
}

<* Submit accounting, whitebox-exposed via submit_stats. *>
struct SubmitStats {
    ulong queue_submits;
    ulong empty_signal_submits;
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

<* Distinct queue families in use, for CONCURRENT-shared resources. *>
fn uint used_queue_families(VkDeviceState* state, uint[] out_families) {
    uint count = 0;
    out_families[count++] = state.graphics_family;
    if (state.compute_family != state.graphics_family) out_families[count++] = state.compute_family;
    if (state.transfer_family != state.graphics_family && state.transfer_family != state.compute_family) {
        out_families[count++] = state.transfer_family;
    }
    return count;
}

<* Selected graphics and compute families, graphics first and deduplicated. *>
fn uint graphics_compute_queue_families(VkDeviceState* state, uint[] out_families) @private {
    uint count = 0;
    out_families[count++] = state.graphics_family;
    if (state.compute_family != state.graphics_family) out_families[count++] = state.compute_family;
    return count;
}

fn vk::Queue queue_for(VkDeviceState* state, gpu::QueueKind queue) {
    switch (queue) {
        case gpu::QueueKind.TRANSFER: return state.transfer_queue;
        case gpu::QueueKind.COMPUTE:  return state.compute_queue;
        default:                      return state.graphics_queue;
    }
}

<* One mutex per distinct VkQueue: an aliased queue must share the graphics
   mutex or two locks would guard one queue. *>
fn Mutex* queue_mutex_for(VkDeviceState* state, gpu::QueueKind kind) @private {
    switch (kind) {
        case gpu::QueueKind.TRANSFER:
            return state.transfer_queue == state.graphics_queue ? &state.graphics_queue_mutex : &state.transfer_queue_mutex;
        case gpu::QueueKind.COMPUTE:
            return state.compute_queue_distinct ? &state.compute_queue_mutex : &state.graphics_queue_mutex;
        default:
            return &state.graphics_queue_mutex;
    }
}

<* Submit command lists to their queue via vkQueueSubmit2, waiting on and
   signaling the requested timeline values at their stages. Empty wait/signal/
   command arrays are left unset so the slice setters are never handed a
   zero-length slice. Queue-used flags and the off-frame marker are set
   unconditionally so retirement also covers off-frame submissions. A submit
   on a distinct compute/transfer queue also piggybacks one internal aux
   signal, so end_frame needs no per-queue empty submit of its own. *>
fn void? vk_submit(gpu::Device* device, gpu::SubmitDesc* desc) @private {
    VkDeviceState* state = (VkDeviceState*)gpu::device_backend_state_ptr(device);
    if (desc.command_lists.len == 0 && desc.waits.len == 0 && desc.signals.len == 0) return;

    @pool() {
        vk::SubmitInfo2 submit = vk::submit_info2();
        gpu::QueueKind kind = gpu::QueueKind.GRAPHICS;
        CommandRecord*[] records;
        bool claimed = false;

        if (desc.command_lists.len > 0) {
            records = mem::talloc_array(CommandRecord*, (sz)desc.command_lists.len);
            state.command_mutex.lock()!!;
            {
                defer state.command_mutex.unlock();
                foreach (i, &token : desc.command_lists) {
                    if (token.device != device) return gpu::INVALID_ARGUMENT~;
                    for (sz j = 0; j < i; j++) {
                        if ((ulong)desc.command_lists[j].handle == (ulong)token.handle) {
                            return gpu::COMMAND_RECORDING_ERROR~;
                        }
                    }
                    CommandRecord* record = state.commands.get(token.handle)!;
                    if (record.state != CommandState.EXECUTABLE) return gpu::COMMAND_RECORDING_ERROR~;
                    if (i == 0) {
                        kind = record.queue;
                    } else if (record.queue != kind) {
                        return gpu::INVALID_ARGUMENT~;
                    }
                    records[i] = record;
                }
                foreach (record : records) record.state = CommandState.SUBMITTING;
                claimed = true;
            }

            vk::CommandBufferSubmitInfo[] cmds = mem::talloc_array(
                vk::CommandBufferSubmitInfo,
                (sz)records.len,
            );
            foreach (i, record : records) {
                cmds[i] = vk::command_buffer_submit_info()
                    .set_command_buffer(record.command_buffer);
            }
            submit = submit.set_command_buffer_infos(cmds);
        }

        defer catch {
            if (claimed) {
                state.command_mutex.lock()!!;
                defer state.command_mutex.unlock();
                foreach (record : records) {
                    if (record.state == CommandState.SUBMITTING) {
                        record.state = CommandState.EXECUTABLE;
                    }
                }
            }
        }

        bool piggyback_compute = kind == gpu::QueueKind.COMPUTE
            && state.compute_queue_distinct;
        bool piggyback_transfer = kind == gpu::QueueKind.TRANSFER
            && state.transfer_queue != state.graphics_queue;
        bool piggyback = piggyback_compute || piggyback_transfer;

        SwapchainSlot* swapchain = null;
        if (desc.swapchain.is_valid()) {
            swapchain = state.swapchains.get(desc.swapchain)!;
            swapchain_submit_guard(
                swapchain.acquire_pending,
                swapchain.acquire_consumed,
                kind,
            )!;
        }

        usz wait_count = desc.waits.len + (swapchain != null ? 1 : 0);
        if (wait_count > 0) {
            vk::SemaphoreSubmitInfo[] waits = mem::talloc_array(
                vk::SemaphoreSubmitInfo,
                (sz)wait_count,
            );
            foreach (i, wait : desc.waits) {
                SemaphoreSlot* slot = state.semaphores.get(wait.semaphore)!;
                waits[i] = vk::semaphore_submit_info()
                    .set_semaphore(slot.semaphore)
                    .set_value((ulong)wait.value)
                    .set_stage_mask(stage_to_vk(wait.stage));
            }
            if (swapchain != null) {
                waits[desc.waits.len] = vk::semaphore_submit_info()
                    .set_semaphore(swapchain.pending_acquire_semaphore)
                    .set_stage_mask(
                        vk::PipelineStageFlagBits2.STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                    );
            }
            submit = submit.set_wait_semaphore_infos(waits);
        }

        usz signal_count = desc.signals.len
            + (swapchain != null ? 1 : 0)
            + (piggyback ? 1 : 0);
        vk::SemaphoreSubmitInfo* aux_signal = null;
        if (signal_count > 0) {
            vk::SemaphoreSubmitInfo[] signals = mem::talloc_array(
                vk::SemaphoreSubmitInfo,
                (sz)signal_count,
            );
            foreach (i, signal : desc.signals) {
                SemaphoreSlot* slot = state.semaphores.get(signal.semaphore)!;
                signals[i] = vk::semaphore_submit_info()
                    .set_semaphore(slot.semaphore)
                    .set_value((ulong)signal.value)
                    .set_stage_mask(stage_to_vk(signal.stage));
            }
            usz next_signal = desc.signals.len;
            if (swapchain != null) {
                signals[next_signal++] = vk::semaphore_submit_info()
                    .set_semaphore(
                        swapchain.present_semaphores[swapchain.pending_image],
                    )
                    .set_stage_mask(
                        vk::PipelineStageFlagBits2.STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                    );
            }
            if (piggyback) {
                vk::Semaphore aux_semaphore = piggyback_compute
                    ? state.aux_compute_timeline
                    : state.aux_transfer_timeline;
                signals[next_signal] = vk::semaphore_submit_info()
                    .set_semaphore(aux_semaphore)
                    .set_stage_mask(
                        vk::PipelineStageFlagBits2.STAGE_2_ALL_COMMANDS_BIT,
                    );
                aux_signal = &signals[next_signal++];
            }
            submit = submit.set_signal_semaphore_infos(signals);
        }

        Mutex* queue_mutex = queue_mutex_for(state, kind);
        {
            queue_mutex.lock()!!;
            defer queue_mutex.unlock();
            if (aux_signal != null) {
                aux_signal.value = (piggyback_compute
                    ? state.aux_compute_counter
                    : state.aux_transfer_counter) + 1;
            }
            check_backend_result(
                state:     state,
                operation: "submit",
                result:    vk::queue_submit2(
                    queue_for(state, kind),
                    1,
                    &submit,
                    (vk::Fence)null,
                ),
            )!;
            state.submit_stats.queue_submits++;
            if (swapchain != null) swapchain.acquire_consumed = true;
            if (piggyback_compute) {
                state.aux_compute_counter++;
                state.aux_compute_last_signaled = state.aux_compute_counter;
                state.compute_used_since_frame = true;
            }
            if (piggyback_transfer) {
                state.aux_transfer_counter++;
                state.aux_transfer_last_signaled = state.aux_transfer_counter;
                state.transfer_used_since_frame = true;
            }
            if (!state.frame_active) {
                state.off_frame_pending.store(true, AtomicOrdering.RELAXED);
            }
            foreach (record : records) {
                commit_pending_layouts(state, record);
                record.state = CommandState.SUBMITTED;
            }
        }

        if (claimed) {
            state.command_mutex.lock()!!;
            defer state.command_mutex.unlock();
            foreach (token : desc.command_lists) {
                release_command(state, token.handle);
            }
            claimed = false;
        }
    };
}

fn void? vk_wait_queue_idle(gpu::Device* device, gpu::QueueKind queue) @private {
    VkDeviceState* state = (VkDeviceState*)gpu::device_backend_state_ptr(device);
    Mutex* queue_mutex = queue_mutex_for(state, queue);
    queue_mutex.lock()!!;
    defer queue_mutex.unlock();
    check_backend_result(
        state:     state,
        operation: "wait_queue_idle",
        result:    vk::queue_wait_idle(queue_for(state, queue)),
    )!;
}

<* Snapshot of submit counters, exposed for whitebox accounting in tests. *>
fn SubmitStats submit_stats(VkDeviceState* state) {
    return state.submit_stats;
}
