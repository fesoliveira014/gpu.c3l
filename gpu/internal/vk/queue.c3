module gpu::internal::vk @private;

import gpu::internal @public;
import gpu @public;
import vk;
import std::thread;
import std::core::mem;
import std::atomic;
import std::atomic::types;

alias QueueFamilyFlags @private = vk::QueueFlags;
alias QueueSubmit2Fn @private = fn vk::Result (
    vk::Queue,
    uint,
    vk::SubmitInfo2*,
    vk::Fence,
);
alias SubmissionCheckpointHook @if(env::TESTING) = fn void (
    void*, SubmissionCheckpoint, gpu::CompletionPoint
);

const uint MAX_QUEUE_FAMILIES = gpu::internal::MAX_SELECTED_QUEUE_IDENTITIES + 4;

enum SubmissionCheckpoint : uint {
    NATIVE_ACCEPTED,
    BATCHES_PUBLISHED,
    TOKEN_PUBLISHED,
}

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
    ulong completion_point_publications;
    ulong completion_point_allocations;
    ulong completion_wait_calls;
}

struct SubmitWorkCounters @if(env::TESTING || $feature(SUBMIT_WORK_STATS)) {
    ulong authoritative_token_resolutions;
    ulong duplicate_token_visits;
    ulong epoch_reset_cells;
    ulong command_mutex_acquisitions;
    ulong queue_submission_mutex_acquisitions;
    ulong rollback_mutex_acquisitions;
    ulong native_submissions;
}

<* Advisory relaxed counters for completion-path work, whitebox-exposed via
   completion_work_counters and reset_completion_work_counters. *>
struct CompletionWorkCounters @if(env::TESTING || $feature(COMPLETION_RETIREMENT_STATS)) {
    ulong native_counter_queries;
    ulong native_wait_calls;
    ulong retirement_passes;
    ulong retirement_lock_acquisitions;
    ulong cached_poll_hits;
    ulong cached_wait_hits;
}

struct QueueCompletionState {
    vk::Queue                     queue;
    vk::Semaphore                 timeline;
    Mutex                         submission_mutex;
    Mutex                         retirement_mutex;
    QueueSubmitScratch*           submit_scratch;
    CommandRecord* submitted_head;
    CommandRecord* submitted_tail;
    uint                          submitted_count;
    gpu::QueueRoles               roles;
    QueueFamilyFlags              queue_flags;
    ulong                         native_id;
    ulong                         next_sequence;
    Atomic{ulong}                 published_sequence;
    Atomic{ulong}                 retired_sequence;
    bool                          bound;
}

struct ResolvedCompletionWait {
    QueueCompletionState*   source;
    vk::PipelineStageFlags2 stages;
    ulong                   sequence;
}

struct QueueSubmitScratch {
    ResolvedCompletionWait[gpu::internal::MAX_SUBMIT_COMPLETION_WAITS]
        completion_waits;
    CommandRecord*[gpu::internal::MAX_SUBMIT_COMMAND_LISTS]
        records;
    vk::CommandBufferSubmitInfo[gpu::internal::MAX_SUBMIT_COMMAND_LISTS]
        command_buffer_infos;
    vk::SemaphoreSubmitInfo[gpu::internal::MAX_SUBMIT_COMPLETION_WAITS + 1]
        wait_infos;
}

struct SubmitInputs {
    CommandRecord*[] records;
    ResolvedCompletionWait[]        completion_waits;
    SwapchainSlot*                  swapchain;
    vk::PipelineStageFlags2         readiness_stages;
}

fn vk::Result real_queue_submit2(
    vk::Queue queue,
    uint submit_count,
    vk::SubmitInfo2* submits,
    vk::Fence fence,
) @private {
    return vk::queue_submit2(queue, submit_count, submits, fence);
}

fn vk::QueueFlags? query_queue_family_flags(
    vk::PhysicalDevice physical_device,
    uint family,
) @private {
    uint count;
    vk::get_physical_device_queue_family_properties(
        physical_device,
        &count,
        null,
    );
    if (count == 0 || family >= count) return gpu::BACKEND_ERROR~;

    @pool() {
        vk::QueueFamilyProperties[] families =
            mem::talloc_array(vk::QueueFamilyProperties, (sz)count);
        vk::get_physical_device_queue_family_properties(
            physical_device,
            &count,
            families.ptr,
        );
        if (family >= count) return gpu::BACKEND_ERROR~;
        return families[family].queue_flags;
    };
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

fn bool queue_family_supports_kind(
    vk::QueueFamilyProperties properties,
    gpu::QueueKind kind,
) @private {
    switch (kind) {
        case gpu::QueueKind.GRAPHICS:
            return queue_family_supports_graphics(properties);
        case gpu::QueueKind.COMPUTE:
            return queue_family_supports_compute(properties);
        case gpu::QueueKind.TRANSFER:
            return queue_family_supports_transfer(properties);
    }
}

fn bool queue_role_is_available(
    vk::QueueFamilyProperties[] families,
    gpu::QueueKind kind,
) @private {
    foreach (properties : families) {
        if (properties.queue_count > 0
            && queue_family_supports_kind(properties, kind)) {
            return true;
        }
    }
    return false;
}

fn bool queue_role_is_selected(
    gpu::QueueRoles selected,
    gpu::QueueKind kind,
) @private {
    switch (kind) {
        case gpu::QueueKind.GRAPHICS: return selected.graphics;
        case gpu::QueueKind.COMPUTE:  return selected.compute;
        case gpu::QueueKind.TRANSFER: return selected.transfer;
    }
}

fn ulong native_queue_id(uint family, uint index) @private {
    return ((ulong)family << 32) | (ulong)index;
}

fn uint native_queue_family(ulong id) @private => (uint)(id >> 32);
fn uint native_queue_index(ulong id) @private => (uint)id;

fn ulong selected_native_queue(
    QueueFamilies* selection,
    gpu::QueueKind kind,
) @private {
    switch (kind) {
        case gpu::QueueKind.GRAPHICS:
            return native_queue_id(selection.graphics, selection.graphics_index);
        case gpu::QueueKind.COMPUTE:
            return native_queue_id(selection.compute, selection.compute_index);
        case gpu::QueueKind.TRANSFER:
            return native_queue_id(selection.transfer, selection.transfer_index);
    }
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

fn bool selection_satisfies_request(
    QueueFamilies selection,
    gpu::QueueRequest request,
) @private {
    bool graphics_alias = request.required.graphics
        && ((request.required.compute && graphics_compute_alias(selection))
            || (request.required.transfer && graphics_transfer_alias(selection)));
    bool compute_alias = request.required.compute
        && ((request.required.graphics && graphics_compute_alias(selection))
            || (request.required.transfer && compute_transfer_alias(selection)));
    bool transfer_alias = request.required.transfer
        && ((request.required.graphics && graphics_transfer_alias(selection))
            || (request.required.compute && compute_transfer_alias(selection)));
    return !(request.distinct.graphics && graphics_alias)
        && !(request.distinct.compute && compute_alias)
        && !(request.distinct.transfer && transfer_alias);
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

fn uint selected_native_identity_count(QueueFamilies selection) @private {
    uint count;
    if (selection.selected.graphics) count++;
    if (selection.selected.compute && !graphics_compute_alias(selection)) count++;
    if (selection.selected.transfer
        && !graphics_transfer_alias(selection)
        && !compute_transfer_alias(selection)) {
        count++;
    }
    return count;
}

fn uint compute_preference_tier(
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

fn uint transfer_preference_tier(
    QueueFamilies selection,
    vk::QueueFamilyProperties[] families,
) @private {
    if (!selection.selected.transfer) return 0;
    if (queue_family_is_transfer_only(families[selection.transfer])) return 0;
    if (graphics_transfer_alias(selection)) return 1;
    if (compute_transfer_alias(selection)) return 2;
    if (selection.selected.graphics
        && selection.graphics == selection.transfer) {
        return 3;
    }
    if (selection.selected.compute
        && selection.compute == selection.transfer) {
        return 4;
    }
    return 5;
}

fn uint distinct_selection_tier(
    QueueFamilies selection,
    vk::QueueFamilyProperties[] families,
    gpu::QueueRequest request,
) @private {
    uint tier;
    if (request.distinct.compute) {
        uint flags = (uint)families[selection.compute].queue_flags;
        if ((flags & (uint)vk::QueueFlagBits.GRAPHICS_BIT) != 0) tier++;
    }
    if (request.distinct.transfer
        && !queue_family_is_transfer_only(families[selection.transfer])) {
        tier++;
    }
    return tier;
}

fn bool selection_is_preferred(
    QueueFamilies candidate,
    QueueFamilies current,
    vk::QueueFamilyProperties[] families,
    gpu::QueueRequest request,
) @private {
    uint candidate_tier = presentation_selection_tier(candidate);
    uint current_tier = presentation_selection_tier(current);
    if (candidate_tier != current_tier) return candidate_tier < current_tier;

    candidate_tier = distinct_selection_tier(candidate, families, request);
    current_tier = distinct_selection_tier(current, families, request);
    if (candidate_tier != current_tier) return candidate_tier < current_tier;

    candidate_tier = compute_preference_tier(candidate, families);
    current_tier = compute_preference_tier(current, families);
    if (candidate_tier != current_tier) return candidate_tier < current_tier;

    candidate_tier = transfer_preference_tier(candidate, families);
    current_tier = transfer_preference_tier(current, families);
    if (candidate_tier != current_tier) return candidate_tier < current_tier;

    candidate_tier = selected_native_identity_count(candidate);
    current_tier = selected_native_identity_count(current);
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

fn uint selectable_queue_count(uint count) @private {
    return count < gpu::internal::MAX_SELECTED_QUEUE_IDENTITIES
        ? count
        : gpu::internal::MAX_SELECTED_QUEUE_IDENTITIES;
}

fn QueueFamilies? pick_queue_families_impl(
    vk::QueueFamilyProperties[] families,
    bool[] presentation_support,
    bool require_presentation,
    gpu::QueueRequest request,
) @private {
    if (!gpu::internal::queue_request_is_valid(request)) {
        return gpu::INVALID_ARGUMENT~;
    }
    if (require_presentation && !request.required.graphics) {
        return gpu::INVALID_ARGUMENT~;
    }
    if ((request.required.graphics
            && !queue_role_is_available(families, gpu::QueueKind.GRAPHICS))
        || (request.required.compute
            && !queue_role_is_available(families, gpu::QueueKind.COMPUTE))
        || (request.required.transfer
            && !queue_role_is_available(families, gpu::QueueKind.TRANSFER))) {
        return gpu::UNSUPPORTED_FEATURE~;
    }
    if (require_presentation && presentation_support.len != families.len) {
        return gpu::INVALID_ARGUMENT~;
    }

    QueueFamilies base = {
        .selected = request.required,
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
                    usz transfer_family_count =
                        base.selected.transfer ? families.len : 1;
                    for (usz transfer_family = 0;
                        transfer_family < transfer_family_count;
                        transfer_family++) {
                        if (base.selected.transfer
                            && !queue_family_supports_transfer(
                                families[transfer_family],
                            )) {
                            continue;
                        }
                        uint transfer_count = base.selected.transfer
                            ? selectable_queue_count(
                                families[transfer_family].queue_count,
                            )
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
                            if (!selection_satisfies_request(
                                candidate,
                                request,
                            )) {
                                continue;
                            }
                            if (require_presentation
                                && !assign_presentation_queue(
                                    &candidate,
                                    presentation_support,
                                )) {
                                continue;
                            }
                            if (!found || selection_is_preferred(
                                candidate: candidate,
                                current:   best,
                                families:  families,
                                request:   request,
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
    if (!found) return gpu::UNSUPPORTED_FEATURE~;
    return best;
}

fn QueueFamilies? pick_queue_families_for_request(
    vk::QueueFamilyProperties[] families,
    gpu::QueueRequest request,
) {
    return pick_queue_families_impl(
        families:             families,
        presentation_support: {},
        require_presentation: false,
        request:              request,
    );
}

fn QueueFamilies? pick_queue_families_for_surface_request(
    vk::QueueFamilyProperties[] families,
    bool[] presentation_support,
    gpu::QueueRequest request,
) {
    return pick_queue_families_impl(
        families:             families,
        presentation_support: presentation_support,
        require_presentation: true,
        request:              request,
    );
}

fn QueueFamilies? pick_queue_families(vk::QueueFamilyProperties[] families) {
    return pick_queue_families_for_request(
        families,
        gpu::internal::default_queue_request(),
    );
}

fn QueueFamilies? pick_queue_families_for_surface(
    vk::QueueFamilyProperties[] families,
    bool[] presentation_support,
) {
    return pick_queue_families_for_surface_request(
        families,
        presentation_support,
        gpu::internal::default_queue_request(),
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
    gpu::QueueRequest* request = null,
) {
    uint count = 0;
    vk::get_physical_device_queue_family_properties(physical_device, &count, null);
    if (count == 0) return gpu::UNSUPPORTED_FEATURE~;

    @pool() {
        vk::QueueFamilyProperties[] families = mem::talloc_array(vk::QueueFamilyProperties, (sz)count);
        vk::get_physical_device_queue_family_properties(physical_device, &count, families.ptr);
        return request == null
            ? pick_queue_families(families[:count])
            : pick_queue_families_for_request(families[:count], *request);
    };
}

fn QueueFamilies? select_queue_families_for_surface(
    VkRuntimeState* runtime,
    vk::PhysicalDevice physical_device,
    vk::SurfaceKHR surface,
    gpu::QueueRequest* request = null,
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
        return request == null
            ? pick_queue_families_for_surface(families[:count], presentation_support)
            : pick_queue_families_for_surface_request(
                families[:count],
                presentation_support,
                *request,
            );
    };
}

fn bool? vk_supports_queue_request(
    VkRuntimeState* state,
    uint adapter_index,
    gpu::QueueRequest* request,
    void* backend_surface,
) {
    if (state == null || request == null || adapter_index >= state.adapters.len) {
        return gpu::INVALID_HANDLE~;
    }

    vk::PhysicalDevice physical_device = state.adapters[adapter_index].physical_device;
    QueueFamilies? selection = backend_surface == null
        ? select_queue_families(physical_device, request)
        : select_queue_families_for_surface(
            runtime:         state,
            physical_device: physical_device,
            surface:         (vk::SurfaceKHR)backend_surface,
            request:         request,
        );
    if (catch selection_fault = selection) {
        if (selection_fault == gpu::UNSUPPORTED_FEATURE) return false;
        return selection_fault~;
    }
    return true;
}

fn uint resource_queue_families(
    VkDeviceState* state,
    gpu::QueueRoles access,
    uint[] out_families,
) {
    uint count;
    if (access.graphics) {
        count = append_queue_family(out_families, count, state.graphics_family);
    }
    if (access.compute) {
        count = append_queue_family(out_families, count, state.compute_family);
    }
    if (access.transfer) {
        count = append_queue_family(out_families, count, state.transfer_family);
    }
    return count;
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

<* Selected graphics and private presentation families for swapchain sharing. *>
fn uint swapchain_queue_families(
    VkDeviceState* state,
    uint[] out_families,
) {
    uint count;
    if (state.selected_queue_roles.graphics) {
        count = append_queue_family(
            out_families,
            count,
            state.graphics_family,
        );
    }
    return append_queue_family(
        out_families,
        count,
        state.presentation.family,
    );
}

fn void? bind_queue_completion_role(
    VkDeviceState* state,
    gpu::internal::QueueSelection* selection,
    gpu::QueueKind kind,
) @private {
    if (!queue_role_is_selected(selection.selected, kind)) return;

    uint identity = gpu::internal::queue_selection_id(selection, kind);
    ulong native_id = selected_native_queue(&state.queues, kind);
    QueueCompletionState* completion = &state.queue_completions[identity];
    if (completion.bound) {
        if (completion.native_id != native_id) return gpu::BACKEND_ERROR~;
        return;
    }

    completion.queue_flags = query_queue_family_flags(
        state.physical_device,
        native_queue_family(native_id),
    )!;
    completion.native_id = native_id;
    completion.bound = true;
    vk::get_device_queue(
        device:             state.device,
        queue_family_index: native_queue_family(native_id),
        queue_index:        native_queue_index(native_id),
        p_queue:            &completion.queue,
    );
}

fn void? create_queue_completions(
    VkDeviceState* state,
    gpu::internal::QueueSelection* selection,
) @private {
    if (state == null || selection == null || selection.identity_count == 0) {
        return gpu::INVALID_ARGUMENT~;
    }
    state.queue_completions = mem::new_array(
        QueueCompletionState,
        (usz)selection.identity_count,
    );
    foreach (&completion : state.queue_completions) {
        completion.submission_mutex.init()!;
        completion.retirement_mutex.init()!;
        completion.submit_scratch = mem::new(QueueSubmitScratch);
        completion.published_sequence.store(0, AtomicOrdering.RELAXED);
        completion.retired_sequence.store(0, AtomicOrdering.RELAXED);
    }

    bind_queue_completion_role(
        state,
        selection,
        gpu::QueueKind.GRAPHICS,
    )!;
    bind_queue_completion_role(
        state,
        selection,
        gpu::QueueKind.COMPUTE,
    )!;
    bind_queue_completion_role(
        state,
        selection,
        gpu::QueueKind.TRANSFER,
    )!;

    if (selection.selected.graphics) {
        state.graphics_completion_id = gpu::internal::queue_selection_id(
            selection,
            gpu::QueueKind.GRAPHICS,
        );
    }
    if (selection.selected.compute) {
        state.compute_completion_id = gpu::internal::queue_selection_id(
            selection,
            gpu::QueueKind.COMPUTE,
        );
    }
    if (selection.selected.transfer) {
        state.transfer_completion_id = gpu::internal::queue_selection_id(
            selection,
            gpu::QueueKind.TRANSFER,
        );
    }

    foreach (identity, &completion : state.queue_completions) {
        if (!completion.bound) return gpu::BACKEND_ERROR~;
        completion.roles = selection.identity_roles[identity];
        completion.timeline = create_private_timeline(state)!;
    }
}

fn void destroy_queue_completions(VkDeviceState* state) @private {
    if (state == null || state.queue_completions.ptr == null) return;
    foreach (&completion : state.queue_completions) {
        completion.published_sequence.store(0, AtomicOrdering.RELAXED);
        completion.retired_sequence.store(0, AtomicOrdering.RELAXED);
        if (completion.timeline != null && state.device != null) {
            vk::destroy_semaphore(state.device, completion.timeline, null);
            completion.timeline = (vk::Semaphore)null;
        }
        if (completion.submission_mutex.is_initialized()) {
            (void)completion.submission_mutex.destroy();
        }
        if (completion.retirement_mutex.is_initialized()) {
            (void)completion.retirement_mutex.destroy();
        }
        if (completion.submit_scratch != null) {
            mem::free(completion.submit_scratch);
            completion.submit_scratch = null;
        }
    }
    mem::free(state.queue_completions.ptr);
    state.queue_completions = {};
}

fn QueueCompletionState*? queue_completion_state(
    VkDeviceState* state,
    uint queue_id,
) {
    if (state == null
        || state.queue_completions.ptr == null
        || queue_id >= state.queue_completions.len) {
        return gpu::INVALID_HANDLE~;
    }
    QueueCompletionState* completion = &state.queue_completions[queue_id];
    if (!completion.bound || completion.timeline == null) {
        return gpu::INVALID_HANDLE~;
    }
    return completion;
}

fn uint primary_completion_id(
    VkDeviceState* state,
    gpu::QueueKind kind,
) @private {
    switch (kind) {
        case gpu::QueueKind.COMPUTE:  return state.compute_completion_id;
        case gpu::QueueKind.TRANSFER: return state.transfer_completion_id;
        default:                      return state.graphics_completion_id;
    }
}

fn QueueCompletionState*? queue_completion_for_kind(
    VkDeviceState* state,
    gpu::QueueKind kind,
) @private {
    QueueCompletionState* completion = queue_completion_state(
        state,
        primary_completion_id(state, kind),
    )!;
    bool supported;
    switch (kind) {
        case gpu::QueueKind.GRAPHICS: supported = completion.roles.graphics;
        case gpu::QueueKind.COMPUTE:  supported = completion.roles.compute;
        case gpu::QueueKind.TRANSFER: supported = completion.roles.transfer;
    }
    if (!supported) return gpu::INVALID_HANDLE~;
    return completion;
}

fn gpu::Queue? queue_token_for_kind(
    VkDeviceState* state,
    gpu::QueueKind kind,
) @private {
    QueueCompletionState* completion = queue_completion_for_kind(state, kind)!;
    return {
        .device = state.owner,
        .id     = primary_completion_id(state, kind),
        .roles  = completion.roles,
    };
}

fn void? require_queue_completion_headroom_with_query(
    VkDeviceState* state,
    QueueCompletionState* completion,
    SemaphoreCounterQueryFn query_fn,
) @private {
    if (completion.next_sequence == gpu::internal::MAX_COMPLETION_SEQUENCE) {
        return gpu::SLOT_TABLE_FULL~;
    }

    ulong prospective = completion.next_sequence + 1;
    ulong completed = completion.retired_sequence.load(
        AtomicOrdering.ACQUIRE,
    );
    ulong limit = state.max_timeline_semaphore_value_difference;
    if (prospective <= completed || prospective - completed <= limit) return;

    note_completion_counter_query(state);
    vk::Result result = query_fn(
        state.device,
        completion.timeline,
        &completed,
    );
    check_backend_result(
        state:     state,
        operation: "submit",
        result:    result,
    )!;
    retire_observed_queue(state, completion, completed);
    completed = completion.retired_sequence.load(AtomicOrdering.ACQUIRE);
    if (prospective > completed && prospective - completed > limit) {
        return gpu::DEVICE_BUSY~;
    }
}

fn void reach_submission_checkpoint(
    VkDeviceState* state,
    SubmissionCheckpoint checkpoint,
    gpu::CompletionPoint point,
) @if(env::TESTING) {
    if (state.submission_checkpoint_hook != null) {
        state.submission_checkpoint_hook(
            state.submission_checkpoint_user_data,
            checkpoint,
            point,
        );
    }
}

fn void set_submission_checkpoint_hook(
    VkDeviceState* state,
    SubmissionCheckpointHook hook,
    void* user_data,
) @if(env::TESTING) {
    state.submission_checkpoint_hook = hook;
    state.submission_checkpoint_user_data = user_data;
}

<* Reserve the next point while holding the queue's submission lock. *>
fn gpu::CompletionPoint? reserve_queue_completion_locked(
    VkDeviceState* state,
    gpu::Queue queue,
) @private {
    if (state == null || queue.device != state.owner) {
        return gpu::INVALID_HANDLE~;
    }
    QueueCompletionState* completion = queue_completion_state(
        state,
        queue.id,
    )!;
    if ((uint)completion.roles != (uint)queue.roles) {
        return gpu::INVALID_HANDLE~;
    }
    return gpu::internal::next_completion_point(queue, &completion.next_sequence);
}

fn gpu::CompletionPoint? reserve_queue_completion(
    VkDeviceState* state,
    gpu::Queue queue,
) {
    QueueCompletionState* completion = queue_completion_state(
        state,
        queue.id,
    )!;
    completion.submission_mutex.lock()!!;
    defer completion.submission_mutex.unlock();
    return reserve_queue_completion_locked(state, queue);
}

fn void? cancel_queue_completion_locked(
    VkDeviceState* state,
    gpu::CompletionPoint point,
) @private {
    if (state == null || point.device != state.owner || !point.is_valid()) {
        return gpu::INVALID_HANDLE~;
    }
    QueueCompletionState* completion = queue_completion_state(
        state,
        gpu::internal::completion_point_queue_id(point),
    )!;
    ulong sequence = gpu::internal::completion_point_sequence(point);
    ulong published = completion.published_sequence.load(
        AtomicOrdering.ACQUIRE,
    );
    if (sequence != completion.next_sequence || sequence <= published) {
        return gpu::INVALID_HANDLE~;
    }
    completion.next_sequence--;
}

<* Publish a point only after its native queue submission succeeds. *>
fn void? publish_queue_completion_locked(
    VkDeviceState* state,
    gpu::CompletionPoint point,
) @private {
    if (state == null || point.device != state.owner || !point.is_valid()) {
        return gpu::INVALID_HANDLE~;
    }
    uint queue_id = gpu::internal::completion_point_queue_id(point);
    ulong sequence = gpu::internal::completion_point_sequence(point);
    QueueCompletionState* completion = queue_completion_state(
        state,
        queue_id,
    )!;
    ulong published = completion.published_sequence.load(
        AtomicOrdering.ACQUIRE,
    );
    if (sequence != published + 1 || sequence > completion.next_sequence) {
        return gpu::INVALID_HANDLE~;
    }
    completion.published_sequence.store(sequence, AtomicOrdering.RELEASE);
}

fn void? publish_queue_completion(
    VkDeviceState* state,
    gpu::CompletionPoint point,
) {
    QueueCompletionState* completion = queue_completion_state(
        state,
        gpu::internal::completion_point_queue_id(point),
    )!;
    completion.submission_mutex.lock()!!;
    defer completion.submission_mutex.unlock();
    return publish_queue_completion_locked(state, point);
}

fn QueueCompletionState*? published_queue_completion_state(
    VkDeviceState* state,
    uint queue_id,
    ulong sequence,
) @private {
    QueueCompletionState* completion = queue_completion_state(
        state,
        queue_id,
    )!;
    if (sequence == 0
        || sequence > completion.published_sequence.load(
            AtomicOrdering.ACQUIRE,
        )) {
        return gpu::INVALID_HANDLE~;
    }
    return completion;
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
        || state.presentation.queue != state.graphics_queue) {
        out_queues[count++] = state.presentation.queue;
    }
    return count;
}

<* Aliased semantic roles share the selected queue identity's submission mutex. *>
fn Mutex* queue_mutex_for(VkDeviceState* state, gpu::QueueKind kind) @private {
    QueueCompletionState* completion = queue_completion_for_kind(state, kind)!!;
    return &completion.submission_mutex;
}

fn Mutex* native_queue_mutex_for(VkDeviceState* state, vk::Queue queue) @private {
    foreach (&completion : state.queue_completions) {
        if (completion.bound && completion.queue == queue) {
            return &completion.submission_mutex;
        }
    }
    return &state.presentation.queue_mutex;
}

fn Mutex* present_queue_mutex_for(VkDeviceState* state) @private {
    return native_queue_mutex_for(state, state.presentation.queue);
}

fn void note_submit_authoritative_resolution(VkDeviceState* state) @private {
    $if env::TESTING || $feature(SUBMIT_WORK_STATS):
        (void)state.submit_authoritative_token_resolutions.add(
            1,
            AtomicOrdering.RELAXED,
        );
    $endif
}

fn void note_submit_duplicate_token_visit(VkDeviceState* state) @private {
    $if env::TESTING || $feature(SUBMIT_WORK_STATS):
        (void)state.submit_duplicate_token_visits.add(
            1,
            AtomicOrdering.RELAXED,
        );
    $endif
}

fn void note_submit_epoch_reset_cell(VkDeviceState* state) @private {
    $if env::TESTING || $feature(SUBMIT_WORK_STATS):
        (void)state.submit_epoch_reset_cells.add(
            1,
            AtomicOrdering.RELAXED,
        );
    $endif
}

fn void note_submit_command_mutex_acquisition(VkDeviceState* state) @private {
    $if env::TESTING || $feature(SUBMIT_WORK_STATS):
        (void)state.submit_command_mutex_acquisitions.add(
            1,
            AtomicOrdering.RELAXED,
        );
    $endif
}

fn void note_submit_queue_submission_mutex_acquisition(
    VkDeviceState* state,
) @private {
    $if env::TESTING || $feature(SUBMIT_WORK_STATS):
        (void)state.submit_queue_submission_mutex_acquisitions.add(
            1,
            AtomicOrdering.RELAXED,
        );
    $endif
}

fn void note_submit_rollback_mutex_acquisition(VkDeviceState* state) @private {
    $if env::TESTING || $feature(SUBMIT_WORK_STATS):
        (void)state.submit_rollback_mutex_acquisitions.add(
            1,
            AtomicOrdering.RELAXED,
        );
    $endif
}

fn void note_submit_native_submission(VkDeviceState* state) @private {
    $if env::TESTING || $feature(SUBMIT_WORK_STATS):
        (void)state.submit_native_submissions.add(
            1,
            AtomicOrdering.RELAXED,
        );
    $endif
}

fn void reset_submit_work_counters(
    VkDeviceState* state,
) @if(env::TESTING || $feature(SUBMIT_WORK_STATS)) {
    state.submit_authoritative_token_resolutions.store(
        0,
        AtomicOrdering.RELAXED,
    );
    state.submit_duplicate_token_visits.store(0, AtomicOrdering.RELAXED);
    state.submit_epoch_reset_cells.store(0, AtomicOrdering.RELAXED);
    state.submit_command_mutex_acquisitions.store(0, AtomicOrdering.RELAXED);
    state.submit_queue_submission_mutex_acquisitions.store(
        0,
        AtomicOrdering.RELAXED,
    );
    state.submit_rollback_mutex_acquisitions.store(0, AtomicOrdering.RELAXED);
    state.submit_native_submissions.store(0, AtomicOrdering.RELAXED);
}

fn SubmitWorkCounters submit_work_counters(
    VkDeviceState* state,
) @if(env::TESTING || $feature(SUBMIT_WORK_STATS)) {
    return {
        .authoritative_token_resolutions =
            state.submit_authoritative_token_resolutions.load(
                AtomicOrdering.RELAXED,
            ),
        .duplicate_token_visits =
            state.submit_duplicate_token_visits.load(AtomicOrdering.RELAXED),
        .epoch_reset_cells =
            state.submit_epoch_reset_cells.load(AtomicOrdering.RELAXED),
        .command_mutex_acquisitions =
            state.submit_command_mutex_acquisitions.load(
                AtomicOrdering.RELAXED,
            ),
        .queue_submission_mutex_acquisitions =
            state.submit_queue_submission_mutex_acquisitions.load(
                AtomicOrdering.RELAXED,
            ),
        .rollback_mutex_acquisitions =
            state.submit_rollback_mutex_acquisitions.load(
                AtomicOrdering.RELAXED,
            ),
        .native_submissions =
            state.submit_native_submissions.load(AtomicOrdering.RELAXED),
    };
}

fn ulong next_submit_visit_epoch_locked(VkDeviceState* state) @private {
    if (state.next_submit_visit_epoch == ulong::max) {
        foreach (&cell : state.commands.slots[:state.commands.next_new]) {
            note_submit_epoch_reset_cell(state);
            if (cell.used) cell.record.submit_visit_epoch = 0;
        }
        state.next_submit_visit_epoch = 1;
        return 1;
    }
    state.next_submit_visit_epoch++;
    return state.next_submit_visit_epoch;
}

fn CommandRecord*[]? claim_submit_records_locked(
    VkDeviceState* state,
    gpu::Queue queue,
    gpu::ExecutableCommandList[] tokens,
    CommandRecord*[] storage,
) @private {
    CommandRecord*[] records = storage[:tokens.len];
    if (tokens.len == 0) return records;

    ulong epoch = next_submit_visit_epoch_locked(state);
    foreach (i, &token : tokens) {
        note_submit_authoritative_resolution(state);
        CommandRecord* record =
            gpu::internal::direct_command_record(
                token.record,
                token.generation,
                token.device_identity,
            )!;
        if (record.device != queue.device
            || record.device_state != state) {
            return gpu::INVALID_HANDLE~;
        }
        if (record.state != gpu::internal::CommandState.EXECUTABLE) {
            return gpu::COMMAND_RECORDING_ERROR~;
        }
        CommandRecord* backend = record;
        if (backend.queue != queue) return gpu::INVALID_ARGUMENT~;
        note_submit_duplicate_token_visit(state);
        if (backend.submit_visit_epoch == epoch) {
            return gpu::COMMAND_RECORDING_ERROR~;
        }
        backend.submit_visit_epoch = epoch;
        records[i] = record;
    }
    foreach (record : records) {
        record.state = gpu::internal::CommandState.SUBMITTING;
    }
    return records;
}

fn void rollback_submit_claim(
    VkDeviceState* state,
    CommandRecord*[] records,
) @private {
    note_submit_rollback_mutex_acquisition(state);
    state.command_mutex.lock()!!;
    defer state.command_mutex.unlock();
    foreach (record : records) {
        if (record.state == gpu::internal::CommandState.SUBMITTING) {
            record.state = gpu::internal::CommandState.EXECUTABLE;
        }
    }
}

fn SubmitInputs? resolve_and_claim_submit_inputs(
    VkDeviceState* state,
    gpu::Queue queue,
    gpu::SubmitDesc* desc,
    QueueCompletionState* target,
    ResolvedCompletionWait[] completion_wait_storage,
    CommandRecord*[] record_storage,
) {
    SubmitInputs inputs;
    gpu::Device device = queue.device;
    bool has_readiness = desc.readiness != gpu::SWAPCHAIN_READINESS_INVALID;
    if (has_readiness != ((uint)desc.readiness_before != 0)) {
        return gpu::INVALID_ARGUMENT~;
    }

    if (has_readiness) {
        if (!desc.readiness.is_valid()
            || desc.readiness.device != device) {
            return gpu::INVALID_HANDLE~;
        }
        if (desc.readiness_before.indirect) return gpu::INVALID_ARGUMENT~;
        inputs.readiness_stages = submit_wait_stages_to_vk(
            desc.readiness_before,
            target.queue_flags,
        )!;
        gpu::SwapchainHandle handle = swapchain_handle_from_readiness(
            state,
            desc.readiness,
        )!;
        inputs.swapchain = state.presentation.swapchains.get(handle)!;
        swapchain_submit_guard(
            acquire_pending:      inputs.swapchain.acquire_pending,
            readiness_consumed:   inputs.swapchain.readiness_consumed,
            expected_acquisition: inputs.swapchain.acquisition,
            provided_acquisition: swapchain_readiness_acquisition(
                desc.readiness,
            ),
            roles:                queue.roles,
        )!;
    }

    if (desc.completion_waits.len > 0) {
        inputs.completion_waits = completion_wait_storage[
            :desc.completion_waits.len
        ];
        foreach (i, wait : desc.completion_waits) {
            gpu::CompletionPoint point = wait.point;
            if (point.device != device) return gpu::INVALID_HANDLE~;
            ulong sequence = gpu::internal::completion_point_sequence(point);
            QueueCompletionState* source = published_queue_completion_state(
                state,
                gpu::internal::completion_point_queue_id(point),
                sequence,
            )!;
            vk::PipelineStageFlags2 stages = submit_wait_stages_to_vk(
                wait.before,
                target.queue_flags,
                state.generated_work.enabled,
            )!;
            inputs.completion_waits[i] = {
                .source   = source,
                .stages   = stages,
                .sequence = sequence,
            };
        }
    }

    if (desc.command_lists.len > 0) {
        note_submit_command_mutex_acquisition(state);
        state.command_mutex.lock()!!;
        defer state.command_mutex.unlock();
        inputs.records = claim_submit_records_locked(
            state,
            queue,
            desc.command_lists,
            record_storage,
        )!;
    }

    return inputs;
}

fn usz append_completion_wait_infos(
    vk::SemaphoreSubmitInfo[] waits,
    usz next_wait,
    ResolvedCompletionWait[] completion_waits,
    QueueCompletionState* target,
) @private {
    foreach (&wait : completion_waits) {
        if (wait.source == target) continue;
        waits[next_wait++] = vk::semaphore_submit_info()
            .set_semaphore(wait.source.timeline)
            .set_value(wait.sequence)
            .set_stage_mask(wait.stages);
    }
    return next_wait;
}

<* Submit executable command lists and signal the queue-owned completion
   timeline. Preparation and native failure leave every token retryable. *>
fn gpu::CompletionPoint? vk_submit_with_queries(
    VkDeviceState* state,
    gpu::Queue queue,
    gpu::SubmitDesc* desc,
    QueueSubmit2Fn submit_fn,
    SemaphoreCounterQueryFn counter_query_fn,
) @private {
    gpu::CompletionPoint submitted;
    vk::SemaphoreSubmitInfo[2] signal_infos;

    QueueCompletionState* completion = queue_completion_state(
        state,
        queue.id,
    )!;
    QueueSubmitScratch* scratch = completion.submit_scratch;
    if (scratch == null) return gpu::BACKEND_ERROR~;
    note_submit_queue_submission_mutex_acquisition(state);
    completion.submission_mutex.lock()!!;
    defer completion.submission_mutex.unlock();

    SubmitInputs inputs = resolve_and_claim_submit_inputs(
        state:  state,
        queue:  queue,
        desc:   desc,
        target: completion,
        completion_wait_storage: scratch.completion_waits[..],
        record_storage:          scratch.records[..],
    )!;
    vk::SubmitInfo2 submit = vk::submit_info2();
    CommandRecord*[] records = inputs.records;
    bool claimed = records.len > 0;
    defer catch {
        if (claimed) rollback_submit_claim(state, records);
    }

    if (records.len > 0) {
        foreach (i, record : records) {
            scratch.command_buffer_infos[i] = vk::command_buffer_submit_info()
                .set_command_buffer(
                    record.command_buffer,
                );
        }
        submit = submit.set_command_buffer_infos(
            scratch.command_buffer_infos[:records.len],
        );
    }

    SwapchainSlot* swapchain = inputs.swapchain;
    usz wait_count = desc.completion_waits.len
        + (swapchain != null ? 1 : 0);
    if (wait_count > 0) {
        usz next_wait;
        next_wait = append_completion_wait_infos(
            waits:            scratch.wait_infos[..],
            next_wait:        next_wait,
            completion_waits: inputs.completion_waits,
            target:           completion,
        );
        if (swapchain != null) {
            scratch.wait_infos[next_wait++] = vk::semaphore_submit_info()
                .set_semaphore(swapchain.pending_acquire_semaphore)
                .set_stage_mask(inputs.readiness_stages);
        }
        if (next_wait > 0) {
            submit = submit.set_wait_semaphore_infos(
                scratch.wait_infos[:next_wait],
            );
        }
    }

    usz signal_count = (usz)(swapchain != null ? 1 : 0) + 1;
    usz next_signal;
    if (swapchain != null) {
        signal_infos[next_signal++] = vk::semaphore_submit_info()
            .set_semaphore(
                swapchain.present_semaphores[swapchain.pending_image],
            )
            .set_stage_mask(
                vk::PipelineStageFlagBits2.STAGE_2_ALL_COMMANDS_BIT,
            );
    }

    vk::SemaphoreSubmitInfo* completion_signal =
        &signal_infos[next_signal++];
    *completion_signal = vk::semaphore_submit_info()
        .set_semaphore(completion.timeline)
        .set_stage_mask(
            vk::PipelineStageFlagBits2.STAGE_2_ALL_COMMANDS_BIT,
        );
    submit = submit.set_signal_semaphore_infos(
        signal_infos[:signal_count],
    );

    drain_queue_if_needed_with_query(
        state,
        completion,
        counter_query_fn,
    )!;
    require_queue_completion_headroom_with_query(
        state:      state,
        completion: completion,
        query_fn:   counter_query_fn,
    )!;
    gpu::CompletionPoint point = reserve_queue_completion_locked(
        state,
        queue,
    )!;
    completion_signal.value = gpu::internal::completion_point_sequence(point);

    note_submit_native_submission(state);
    vk::Result result = submit_fn(
        completion.queue,
        1,
        &submit,
        (vk::Fence)null,
    );
    if (result != vk::Result.SUCCESS) {
        cancel_queue_completion_locked(state, point)!!;
        check_backend_result(
            state:     state,
            operation: "submit",
            result:    result,
        )!;
    }

    $if env::TESTING:
        reach_submission_checkpoint(
            state,
            SubmissionCheckpoint.NATIVE_ACCEPTED,
            point,
        );
    $endif

    completion.retirement_mutex.lock()!!;
    publish_pending_commands_locked(
        state,
        completion,
        records,
        point,
    );
    completion.retirement_mutex.unlock();
    $if env::TESTING:
        reach_submission_checkpoint(
            state,
            SubmissionCheckpoint.BATCHES_PUBLISHED,
            point,
        );
    $endif
    (void)state.submit_queue_submits.add(1, AtomicOrdering.RELAXED);
    (void)state.submit_completion_point_publications.add(
        1,
        AtomicOrdering.RELAXED,
    );
    if (swapchain != null) {
        commit_swapchain_submit(swapchain, point);
    }
    foreach (record : records) {
        record.state = gpu::internal::CommandState.SUBMITTED;
    }
    publish_queue_completion_locked(state, point)!!;
    $if env::TESTING:
        reach_submission_checkpoint(
            state,
            SubmissionCheckpoint.TOKEN_PUBLISHED,
            point,
        );
    $endif
    submitted = point;
    claimed = false;
    return submitted;
}

fn gpu::CompletionPoint? vk_submit_with(
    VkDeviceState* state,
    gpu::Queue queue,
    gpu::SubmitDesc* desc,
    QueueSubmit2Fn submit_fn,
) @private {
    return vk_submit_with_queries(
        state:            state,
        queue:            queue,
        desc:             desc,
        submit_fn:        submit_fn,
        counter_query_fn: &real_semaphore_counter_query,
    );
}

fn gpu::CompletionPoint? vk_submit(
    VkDeviceState* state,
    gpu::Queue queue,
    gpu::SubmitDesc* desc,
) {
    return vk_submit_with(state, queue, desc, &real_queue_submit2);
}


<* Snapshot of submit counters, exposed for whitebox accounting in tests. *>
fn SubmitStats submit_stats(VkDeviceState* state) {
    SubmitStats stats = {
        .queue_submits = state.submit_queue_submits.load(
            AtomicOrdering.RELAXED,
        ),
        .completion_point_publications =
            state.submit_completion_point_publications.load(
                AtomicOrdering.RELAXED,
            ),
    };
    stats.completion_wait_calls = state.submit_completion_wait_calls.load(
        AtomicOrdering.RELAXED,
    );
    return stats;
}
