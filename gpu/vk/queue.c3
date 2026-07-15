module gpu::vk @private;

import gpu @public;
import vk;
import std::thread;
import std::core::mem;

const uint MAX_QUEUE_FAMILIES = 4;
const uint MAX_SELECTED_QUEUE_INDICES = 3;

struct QueueFamilies {
    gpu::QueueRoles selected;
    uint graphics;
    uint graphics_index;
    uint compute;
    uint compute_index;
    uint transfer;
    uint transfer_index;
    uint present;
    uint present_index;
    bool presentation;
    bool async_compute;
}

<* Submit accounting, whitebox-exposed via submit_stats. *>
struct SubmitStats {
    ulong queue_submits;
    ulong empty_signal_submits;
}

struct SubmitInputs {
    CommandRecord*[] records;
    SemaphoreSlot*[] waits;
    SemaphoreSlot*[] signals;
    SwapchainSlot*   swapchain;
    gpu::QueueKind   kind;
}

fn bool queue_family_supports_graphics(
    vk::QueueFamilyProperties properties,
) @private {
    return ((uint)properties.queue_flags
        & (uint)vk::QueueFlagBits.GRAPHICS_BIT) != 0;
}

fn bool queue_family_supports_compute(
    vk::QueueFamilyProperties properties,
) @private {
    return ((uint)properties.queue_flags
        & (uint)vk::QueueFlagBits.COMPUTE_BIT) != 0;
}

fn bool queue_family_supports_transfer(
    vk::QueueFamilyProperties properties,
) @private {
    uint flags = (uint)properties.queue_flags;
    uint transfer_flags = (uint)vk::QueueFlagBits.GRAPHICS_BIT
        | (uint)vk::QueueFlagBits.COMPUTE_BIT
        | (uint)vk::QueueFlagBits.TRANSFER_BIT;
    return (flags & transfer_flags) != 0;
}

fn bool queue_family_is_transfer_only(
    vk::QueueFamilyProperties properties,
) @private {
    uint flags = (uint)properties.queue_flags;
    return (flags & (uint)vk::QueueFlagBits.TRANSFER_BIT) != 0
        && (flags & (uint)vk::QueueFlagBits.GRAPHICS_BIT) == 0
        && (flags & (uint)vk::QueueFlagBits.COMPUTE_BIT) == 0;
}

fn uint selectable_queue_count(uint count) @private {
    return count < MAX_SELECTED_QUEUE_INDICES
        ? count
        : MAX_SELECTED_QUEUE_INDICES;
}

fn bool graphics_compute_alias(QueueFamilies selection) @private {
    return selection.selected.graphics
        && selection.selected.compute
        && selection.graphics == selection.compute
        && selection.graphics_index == selection.compute_index;
}

fn bool graphics_transfer_alias(QueueFamilies selection) @private {
    return selection.selected.graphics
        && selection.selected.transfer
        && selection.graphics == selection.transfer
        && selection.graphics_index == selection.transfer_index;
}

fn bool compute_transfer_alias(QueueFamilies selection) @private {
    return selection.selected.compute
        && selection.selected.transfer
        && selection.compute == selection.transfer
        && selection.compute_index == selection.transfer_index;
}

fn bool selection_satisfies_requirements(
    QueueFamilies selection,
    gpu::QueueRequirements requirements,
) @private {
    bool graphics_alias = requirements.counts.graphics > 0
        && ((requirements.counts.compute > 0
                && graphics_compute_alias(selection))
            || (requirements.counts.transfer > 0
                && graphics_transfer_alias(selection)));
    bool compute_alias = requirements.counts.compute > 0
        && ((requirements.counts.graphics > 0
                && graphics_compute_alias(selection))
            || (requirements.counts.transfer > 0
                && compute_transfer_alias(selection)));
    bool transfer_alias = requirements.counts.transfer > 0
        && ((requirements.counts.graphics > 0
                && graphics_transfer_alias(selection))
            || (requirements.counts.compute > 0
                && compute_transfer_alias(selection)));
    return !(requirements.distinct.graphics && graphics_alias)
        && !(requirements.distinct.compute && compute_alias)
        && !(requirements.distinct.transfer && transfer_alias);
}

fn bool assign_presentation_queue(
    QueueFamilies* selection,
    bool[] presentation_support,
) @private {
    if (selection.selected.graphics
        && presentation_support[selection.graphics]) {
        selection.present = selection.graphics;
        selection.present_index = selection.graphics_index;
        selection.presentation = true;
        return true;
    }
    if (selection.selected.compute
        && presentation_support[selection.compute]) {
        selection.present = selection.compute;
        selection.present_index = selection.compute_index;
        selection.presentation = true;
        return true;
    }
    if (selection.selected.transfer
        && presentation_support[selection.transfer]) {
        selection.present = selection.transfer;
        selection.present_index = selection.transfer_index;
        selection.presentation = true;
        return true;
    }
    foreach (family, supported : presentation_support) {
        if (!supported) continue;
        selection.present = (uint)family;
        selection.present_index = 0;
        selection.presentation = true;
        return true;
    }
    return false;
}

fn uint compute_selection_tier(
    QueueFamilies selection,
    vk::QueueFamilyProperties[] families,
) @private {
    if (!selection.selected.compute) return 0;
    if (selection.selected.graphics
        && selection.graphics == selection.compute) {
        return selection.graphics_index == selection.compute_index ? 2 : 0;
    }
    uint compute_flags = (uint)families[selection.compute].queue_flags;
    bool graphics_capable =
        (compute_flags & (uint)vk::QueueFlagBits.GRAPHICS_BIT) != 0;
    if (!selection.selected.graphics) return graphics_capable ? 1 : 0;
    return graphics_capable ? 3 : 1;
}

fn uint transfer_selection_tier(
    QueueFamilies selection,
    vk::QueueFamilyProperties[] families,
) @private {
    if (!selection.selected.transfer) return 0;
    if (queue_family_is_transfer_only(families[selection.transfer])) return 0;
    if (graphics_transfer_alias(selection)) return 1;
    if (compute_transfer_alias(selection)) return 2;
    if (selection.selected.graphics && selection.graphics == selection.transfer) return 3;
    if (selection.selected.compute && selection.compute == selection.transfer) return 4;
    return 5;
}

fn uint presentation_selection_tier(QueueFamilies selection) @private {
    if (!selection.presentation) return 0;
    if (selection.selected.graphics
        && selection.present == selection.graphics
        && selection.present_index == selection.graphics_index) {
        return 0;
    }
    if (selection.selected.compute
        && selection.present == selection.compute
        && selection.present_index == selection.compute_index) {
        return 1;
    }
    if (selection.selected.transfer
        && selection.present == selection.transfer
        && selection.present_index == selection.transfer_index) {
        return 2;
    }
    return 3;
}

fn bool selection_is_preferred(
    QueueFamilies candidate,
    QueueFamilies current,
    vk::QueueFamilyProperties[] families,
) @private {
    uint candidate_tier = compute_selection_tier(candidate, families);
    uint current_tier = compute_selection_tier(current, families);
    if (candidate_tier != current_tier) return candidate_tier < current_tier;

    candidate_tier = transfer_selection_tier(candidate, families);
    current_tier = transfer_selection_tier(current, families);
    if (candidate_tier != current_tier) return candidate_tier < current_tier;

    candidate_tier = presentation_selection_tier(candidate);
    current_tier = presentation_selection_tier(current);
    if (candidate_tier != current_tier) return candidate_tier < current_tier;

    if (candidate.selected.graphics
        && candidate.graphics != current.graphics) {
        return candidate.graphics < current.graphics;
    }
    if (candidate.selected.graphics
        && candidate.graphics_index != current.graphics_index) {
        return candidate.graphics_index < current.graphics_index;
    }
    if (candidate.selected.compute
        && candidate.compute != current.compute) {
        return candidate.compute < current.compute;
    }
    if (candidate.selected.compute
        && candidate.compute_index != current.compute_index) {
        return candidate.compute_index < current.compute_index;
    }
    if (candidate.selected.transfer
        && candidate.transfer != current.transfer) {
        return candidate.transfer < current.transfer;
    }
    if (candidate.selected.transfer
        && candidate.transfer_index != current.transfer_index) {
        return candidate.transfer_index < current.transfer_index;
    }
    if (candidate.presentation && candidate.present != current.present) {
        return candidate.present < current.present;
    }
    if (candidate.presentation
        && candidate.present_index != current.present_index) {
        return candidate.present_index < current.present_index;
    }
    return false;
}

fn QueueFamilies? pick_queue_families_impl(
    vk::QueueFamilyProperties[] families,
    bool[] presentation_support,
    bool require_presentation,
    gpu::QueueRequirements requirements,
) @private {
    if (!gpu::queue_requirements_are_valid(requirements)) {
        return gpu::INVALID_ARGUMENT~;
    }
    if (requirements.counts.graphics > 1
        || requirements.counts.compute > 1
        || requirements.counts.transfer > 1) {
        return gpu::UNSUPPORTED_FEATURE~;
    }
    if (require_presentation && presentation_support.len != families.len) {
        return gpu::INVALID_ARGUMENT~;
    }

    QueueFamilies base = {
        .selected = {
            .graphics = requirements.counts.graphics > 0,
            .compute  = requirements.counts.compute > 0,
            .transfer = requirements.counts.transfer > 0,
        },
    };
    QueueFamilies best;
    bool found;
    usz graphics_family_count = base.selected.graphics ? families.len : 1;
    for (usz graphics_family = 0;
        graphics_family < graphics_family_count;
        graphics_family++) {
        if (base.selected.graphics
            && !queue_family_supports_graphics(families[graphics_family])) {
            continue;
        }
        uint graphics_count = base.selected.graphics
            ? selectable_queue_count(families[graphics_family].queue_count)
            : 1;
        for (uint graphics_index = 0;
            graphics_index < graphics_count;
            graphics_index++) {
            usz compute_family_count = base.selected.compute ? families.len : 1;
            for (usz compute_family = 0;
                compute_family < compute_family_count;
                compute_family++) {
                if (base.selected.compute
                    && !queue_family_supports_compute(families[compute_family])) {
                    continue;
                }
                uint compute_count = base.selected.compute
                    ? selectable_queue_count(families[compute_family].queue_count)
                    : 1;
                for (uint compute_index = 0;
                    compute_index < compute_count;
                    compute_index++) {
                    usz transfer_family_count = base.selected.transfer ? families.len : 1;
                    for (usz transfer_family = 0;
                        transfer_family < transfer_family_count;
                        transfer_family++) {
                        if (base.selected.transfer
                            && !queue_family_supports_transfer(families[transfer_family])) {
                            continue;
                        }
                        uint transfer_count = base.selected.transfer
                            ? selectable_queue_count(families[transfer_family].queue_count)
                            : 1;
                        for (uint transfer_index = 0;
                            transfer_index < transfer_count;
                            transfer_index++) {
                            QueueFamilies candidate = base;
                            if (candidate.selected.graphics) {
                                candidate.graphics = (uint)graphics_family;
                                candidate.graphics_index = graphics_index;
                            }
                            if (candidate.selected.compute) {
                                candidate.compute = (uint)compute_family;
                                candidate.compute_index = compute_index;
                            }
                            if (candidate.selected.transfer) {
                                candidate.transfer = (uint)transfer_family;
                                candidate.transfer_index = transfer_index;
                            }
                            candidate.async_compute =
                                candidate.selected.graphics
                                && candidate.selected.compute
                                && !graphics_compute_alias(candidate);
                            if (!selection_satisfies_requirements(
                                candidate,
                                requirements,
                            )) {
                                continue;
                            }
                            if (require_presentation) {
                                if (!assign_presentation_queue(
                                    &candidate,
                                    presentation_support,
                                )) {
                                    continue;
                                }
                            }
                            if (!found || selection_is_preferred(
                                candidate,
                                best,
                                families,
                            )) {
                                best = candidate;
                                found = true;
                            }
                        }
                    }
                }
            }
        }
    }
    return found ? best : gpu::UNSUPPORTED_FEATURE~;
}

fn QueueFamilies? pick_queue_families_for_requirements(
    vk::QueueFamilyProperties[] families,
    gpu::QueueRequirements requirements,
) {
    return pick_queue_families_impl(
        families:             families,
        presentation_support: {},
        require_presentation: false,
        requirements:         requirements,
    );
}

fn QueueFamilies? pick_queue_families_for_surface_requirements(
    vk::QueueFamilyProperties[] families,
    bool[] presentation_support,
    gpu::QueueRequirements requirements,
) {
    return pick_queue_families_impl(
        families:             families,
        presentation_support: presentation_support,
        require_presentation: true,
        requirements:         requirements,
    );
}

fn QueueFamilies? pick_queue_families(vk::QueueFamilyProperties[] families) {
    return pick_queue_families_for_requirements(
        families,
        gpu::default_queue_requirements(),
    );
}

fn QueueFamilies? pick_queue_families_for_surface(
    vk::QueueFamilyProperties[] families,
    bool[] presentation_support,
) {
    return pick_queue_families_for_surface_requirements(
        families,
        presentation_support,
        gpu::default_queue_requirements(),
    );
}

fn uint requested_queue_count(
    QueueFamilies* selection,
    uint family,
    bool include_presentation,
) {
    uint count;
    if (selection.selected.graphics && selection.graphics == family) {
        count = selection.graphics_index + 1;
    }
    if (selection.selected.compute
        && selection.compute == family
        && count <= selection.compute_index) {
        count = selection.compute_index + 1;
    }
    if (selection.selected.transfer
        && selection.transfer == family
        && count <= selection.transfer_index) {
        count = selection.transfer_index + 1;
    }
    if (include_presentation
        && selection.presentation
        && selection.present == family
        && count <= selection.present_index) {
        count = selection.present_index + 1;
    }
    return count;
}

fn uint append_queue_family(
    uint[] out_families,
    uint count,
    uint family,
) @private {
    foreach (selected : out_families[:count]) {
        if (selected == family) return count;
    }
    out_families[count] = family;
    return count + 1;
}

fn uint selected_device_queue_families(
    QueueFamilies* selection,
    bool include_presentation,
    uint[] out_families,
) {
    uint count;
    if (selection.selected.graphics) {
        count = append_queue_family(out_families, count, selection.graphics);
    }
    if (selection.selected.compute) {
        count = append_queue_family(out_families, count, selection.compute);
    }
    if (selection.selected.transfer) {
        count = append_queue_family(out_families, count, selection.transfer);
    }
    if (include_presentation && selection.presentation) {
        count = append_queue_family(out_families, count, selection.present);
    }
    return count;
}

fn QueueFamilies? select_queue_families(
    vk::PhysicalDevice physical_device,
    gpu::QueueRequirements* requirements = null,
) {
    uint count = 0;
    vk::get_physical_device_queue_family_properties(physical_device, &count, null);
    if (count == 0) return gpu::UNSUPPORTED_FEATURE~;

    @pool() {
        vk::QueueFamilyProperties[] families = mem::talloc_array(vk::QueueFamilyProperties, (sz)count);
        vk::get_physical_device_queue_family_properties(physical_device, &count, families.ptr);
        return requirements == null
            ? pick_queue_families(families[:count])
            : pick_queue_families_for_requirements(families[:count], *requirements);
    };
}

fn QueueFamilies? select_queue_families_for_surface(
    VkRuntimeState* runtime,
    vk::PhysicalDevice physical_device,
    vk::SurfaceKHR surface,
    gpu::QueueRequirements* requirements = null,
) {
    if (runtime == null || runtime.dispatch.surface.get_surface_support == null) {
        return gpu::UNSUPPORTED_FEATURE~;
    }
    uint count = 0;
    vk::get_physical_device_queue_family_properties(physical_device, &count, null);
    if (count == 0) return gpu::UNSUPPORTED_FEATURE~;

    @pool() {
        vk::QueueFamilyProperties[] families =
            mem::talloc_array(vk::QueueFamilyProperties, (sz)count);
        bool[] presentation_support = mem::talloc_array(bool, (sz)count);
        vk::get_physical_device_queue_family_properties(
            physical_device,
            &count,
            families.ptr,
        );
        for (uint index = 0; index < count; index++) {
            uint supported = 0;
            check_surface_result(runtime.dispatch.surface.get_surface_support(
                physical_device,
                index,
                surface,
                &supported,
            ))!;
            presentation_support[index] = supported != 0;
        }
        return requirements == null
            ? pick_queue_families_for_surface(families[:count], presentation_support)
            : pick_queue_families_for_surface_requirements(
                families[:count],
                presentation_support,
                *requirements,
            );
    };
}

fn bool? vk_supports_queue_requirements(
    void* backend_state,
    uint adapter_index,
    gpu::QueueRequirements* requirements,
    void* backend_surface,
) {
    VkRuntimeState* state = (VkRuntimeState*)backend_state;
    if (state == null || requirements == null || adapter_index >= state.adapters.len) {
        return gpu::INVALID_HANDLE~;
    }

    vk::PhysicalDevice physical_device = state.adapters[adapter_index].physical_device;
    QueueFamilies? selection = backend_surface == null
        ? select_queue_families(physical_device, requirements)
        : select_queue_families_for_surface(
            runtime:         state,
            physical_device: physical_device,
            surface:         (vk::SurfaceKHR)backend_surface,
            requirements:    requirements,
        );
    if (catch selection_fault = selection) {
        if (selection_fault == gpu::UNSUPPORTED_FEATURE) return false;
        return selection_fault~;
    }
    return true;
}

<* Distinct queue families in use, for CONCURRENT-shared resources. *>
fn uint used_queue_families(VkDeviceState* state, uint[] out_families) {
    uint count;
    if (state.selected_queue_roles.graphics) {
        count = append_queue_family(out_families, count, state.graphics_family);
    }
    if (state.selected_queue_roles.compute) {
        count = append_queue_family(out_families, count, state.compute_family);
    }
    if (state.selected_queue_roles.transfer) {
        count = append_queue_family(out_families, count, state.transfer_family);
    }
    return count;
}

<* Selected shader families, or transfer when it is the only admitted role. *>
fn uint graphics_compute_queue_families(VkDeviceState* state, uint[] out_families) @private {
    uint count;
    if (state.selected_queue_roles.graphics) {
        count = append_queue_family(out_families, count, state.graphics_family);
    }
    if (state.selected_queue_roles.compute) {
        count = append_queue_family(out_families, count, state.compute_family);
    }
    if (count == 0 && state.selected_queue_roles.transfer) {
        count = append_queue_family(out_families, count, state.transfer_family);
    }
    return count;
}

<* Selected graphics and private presentation families for swapchain sharing. *>
fn uint swapchain_queue_families(
    VkDeviceState* state,
    uint[] out_families,
) {
    uint count = 0;
    if (state.selected_queue_roles.graphics) {
        out_families[count++] = state.graphics_family;
    }
    if (!state.selected_queue_roles.graphics
        || state.presentation.family != state.graphics_family) {
        out_families[count++] = state.presentation.family;
    }
    return count;
}

<* Selected graphics and private presentation queues used by a swapchain. *>
fn uint swapchain_idle_queues(
    VkDeviceState* state,
    vk::Queue[] out_queues,
) {
    uint count = 0;
    if (state.selected_queue_roles.graphics) {
        out_queues[count++] = state.graphics_queue;
    }
    if (!state.selected_queue_roles.graphics
        || state.presentation.family != state.graphics_family) {
        out_queues[count++] = state.presentation.queue;
    }
    return count;
}

fn gpu::QueueKind frame_queue_kind(VkDeviceState* state) @private {
    if (state.selected_queue_roles.graphics) return gpu::QueueKind.GRAPHICS;
    if (state.selected_queue_roles.compute) return gpu::QueueKind.COMPUTE;
    return gpu::QueueKind.TRANSFER;
}

fn vk::Queue queue_for(VkDeviceState* state, gpu::QueueKind queue) {
    switch (queue) {
        case gpu::QueueKind.TRANSFER: return state.transfer_queue;
        case gpu::QueueKind.COMPUTE:  return state.compute_queue;
        default:                      return state.graphics_queue;
    }
}

fn vk::Queue frame_queue_for(VkDeviceState* state) @private {
    return queue_for(state, frame_queue_kind(state));
}

fn bool queue_uses_frame_aux(VkDeviceState* state, gpu::QueueKind kind) @private {
    return kind != frame_queue_kind(state)
        && queue_for(state, kind) != frame_queue_for(state);
}

<* Aliased semantic roles share the mutex for their native queue. *>
fn Mutex* queue_mutex_for(VkDeviceState* state, gpu::QueueKind kind) @private {
    switch (kind) {
        case gpu::QueueKind.TRANSFER:
            if (state.selected_queue_roles.graphics
                && state.transfer_queue == state.graphics_queue) {
                return &state.graphics_queue_mutex;
            }
            if (state.selected_queue_roles.compute
                && state.transfer_queue == state.compute_queue) {
                return queue_mutex_for(state, gpu::QueueKind.COMPUTE);
            }
            return &state.transfer_queue_mutex;
        case gpu::QueueKind.COMPUTE:
            return state.selected_queue_roles.graphics
                && state.compute_queue == state.graphics_queue
                ? &state.graphics_queue_mutex
                : &state.compute_queue_mutex;
        default:
            return &state.graphics_queue_mutex;
    }
}

fn Mutex* frame_queue_mutex_for(VkDeviceState* state) @private {
    return queue_mutex_for(state, frame_queue_kind(state));
}

fn Mutex* present_queue_mutex_for(VkDeviceState* state) @private {
    if (state.selected_queue_roles.graphics
        && state.presentation.family == state.graphics_family) {
        return &state.graphics_queue_mutex;
    }
    if (state.selected_queue_roles.compute
        && state.presentation.family == state.compute_family) {
        return queue_mutex_for(state, gpu::QueueKind.COMPUTE);
    }
    if (state.selected_queue_roles.transfer
        && state.presentation.family == state.transfer_family) {
        return queue_mutex_for(state, gpu::QueueKind.TRANSFER);
    }
    return &state.presentation.queue_mutex;
}

fn Mutex* native_queue_mutex_for(VkDeviceState* state, vk::Queue queue) @private {
    if (state.selected_queue_roles.graphics && queue == state.graphics_queue) {
        return &state.graphics_queue_mutex;
    }
    if (state.selected_queue_roles.compute && queue == state.compute_queue) {
        return queue_mutex_for(state, gpu::QueueKind.COMPUTE);
    }
    if (state.selected_queue_roles.transfer && queue == state.transfer_queue) {
        return queue_mutex_for(state, gpu::QueueKind.TRANSFER);
    }
    return present_queue_mutex_for(state);
}

fn SubmitInputs? resolve_submit_inputs(
    VkDeviceState* state,
    gpu::Device* device,
    gpu::SubmitDesc* desc,
) {
    SubmitInputs inputs = { .kind = frame_queue_kind(state) };

    if (desc.command_lists.len > 0) {
        inputs.records = mem::talloc_array(CommandRecord*, (sz)desc.command_lists.len);
        state.command_mutex.lock()!!;
        {
            defer state.command_mutex.unlock();
            foreach (i, &token : desc.command_lists) {
                if (token.device != *device) return gpu::INVALID_HANDLE~;
                for (sz j = 0; j < i; j++) {
                    if (desc.command_lists[j].handle == token.handle) {
                        return gpu::COMMAND_RECORDING_ERROR~;
                    }
                }
                CommandRecord* record = state.commands.get(token.handle)!;
                if (record.state != CommandState.EXECUTABLE) return gpu::COMMAND_RECORDING_ERROR~;
                if (i == 0) {
                    inputs.kind = record.queue;
                } else if (record.queue != inputs.kind) {
                    return gpu::INVALID_ARGUMENT~;
                }
                inputs.records[i] = record;
            }
        }
    }

    if (desc.swapchain.is_valid()) {
        inputs.swapchain = state.presentation.swapchains.get(desc.swapchain)!;
        swapchain_submit_guard(
            inputs.swapchain.acquire_pending,
            inputs.swapchain.acquire_consumed,
            inputs.kind,
        )!;
    }

    if (desc.waits.len > 0) {
        inputs.waits = mem::talloc_array(SemaphoreSlot*, (sz)desc.waits.len);
        foreach (i, wait : desc.waits) {
            inputs.waits[i] = state.semaphores.get(wait.semaphore)!;
        }
    }

    if (desc.signals.len > 0) {
        inputs.signals = mem::talloc_array(SemaphoreSlot*, (sz)desc.signals.len);
        foreach (i, signal : desc.signals) {
            inputs.signals[i] = state.semaphores.get(signal.semaphore)!;
        }
    }

    return inputs;
}
<* Submit command lists to their queue via vkQueueSubmit2, waiting on and
   signaling the requested timeline values at their stages. Empty wait/signal/
   command arrays are left unset so the slice setters are never handed a
   zero-length slice. Queue-used flags and the off-frame marker are set
   unconditionally so retirement also covers off-frame submissions. A submit
   on a non-anchor compute/transfer queue piggybacks one internal aux signal,
   so end_frame needs no per-queue empty submit of its own. *>
fn void? vk_submit(gpu::Device* device, gpu::SubmitDesc* desc) @private {
    VkDeviceState* state = (VkDeviceState*)gpu::device_backend_state_ptr(device);
    if (desc.command_lists.len == 0 && desc.waits.len == 0 && desc.signals.len == 0) return;

    @pool() {
        SubmitInputs inputs = resolve_submit_inputs(state, device, desc)!;
        vk::SubmitInfo2 submit = vk::submit_info2();
        gpu::QueueKind kind = inputs.kind;
        CommandRecord*[] records = inputs.records;
        bool claimed = false;

        if (records.len > 0) {
            state.command_mutex.lock()!!;
            {
                defer state.command_mutex.unlock();
                foreach (i, &token : desc.command_lists) {
                    CommandRecord* record = state.commands.get(token.handle)!;
                    if (record != records[i]) return gpu::COMMAND_RECORDING_ERROR~;
                    if (record.state != CommandState.EXECUTABLE) return gpu::COMMAND_RECORDING_ERROR~;
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
            && queue_uses_frame_aux(state, kind);
        bool piggyback_transfer = kind == gpu::QueueKind.TRANSFER
            && queue_uses_frame_aux(state, kind);
        bool piggyback = piggyback_compute || piggyback_transfer;

        SwapchainSlot* swapchain = inputs.swapchain;
        usz wait_count = desc.waits.len + (swapchain != null ? 1 : 0);
        if (wait_count > 0) {
            vk::SemaphoreSubmitInfo[] waits = mem::talloc_array(
                vk::SemaphoreSubmitInfo,
                (sz)wait_count,
            );
            foreach (i, wait : desc.waits) {
                SemaphoreSlot* slot = inputs.waits[i];
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
                SemaphoreSlot* slot = inputs.signals[i];
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
