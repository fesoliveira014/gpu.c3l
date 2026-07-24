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
const uint MAX_SELECTED_QUEUE_INDICES = 3;

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
    gpu::QueueCounts counts;
    gpu::QueueRoles distinct;
    ulong[gpu::internal::MAX_REQUESTED_QUEUES_PER_ROLE] graphics_ids;
    ulong[gpu::internal::MAX_REQUESTED_QUEUES_PER_ROLE] compute_ids;
    ulong[gpu::internal::MAX_REQUESTED_QUEUES_PER_ROLE] transfer_ids;
}
struct QueueChoice {
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


struct QueueCandidatePool {
    uint count;
    ulong[] ids;
    uint[] owners;
    uint[] graphics;
    uint[] compute;
    uint[] transfer;
    gpu::QueueCounts candidate_counts;
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
    gpu::internal::CommandRecord* submitted_head;
    gpu::internal::CommandRecord* submitted_tail;
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
    gpu::internal::CommandRecord*[gpu::internal::MAX_SUBMIT_COMMAND_LISTS]
        records;
    vk::CommandBufferSubmitInfo[gpu::internal::MAX_SUBMIT_COMMAND_LISTS]
        command_buffer_infos;
    vk::SemaphoreSubmitInfo[gpu::internal::MAX_SUBMIT_COMPLETION_WAITS + 1]
        wait_infos;
}

struct SubmitInputs {
    gpu::internal::CommandRecord*[] records;
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

fn uint available_queue_count(
    vk::QueueFamilyProperties[] families,
    gpu::QueueKind kind,
) @private {
    uint count;
    foreach (properties : families) {
        bool supported;
        switch (kind) {
            case gpu::QueueKind.GRAPHICS:
                supported = queue_family_supports_graphics(properties);
            case gpu::QueueKind.COMPUTE:
                supported = queue_family_supports_compute(properties);
            case gpu::QueueKind.TRANSFER:
                supported = queue_family_supports_transfer(properties);
        }
        if (!supported) continue;
        uint remaining = gpu::internal::MAX_REQUESTED_QUEUES_PER_ROLE - count;
        count += properties.queue_count < remaining
            ? properties.queue_count
            : remaining;
        if (count == gpu::internal::MAX_REQUESTED_QUEUES_PER_ROLE) break;
    }
    return count;
}

fn ulong native_queue_id(uint family, uint index) @private {
    return ((ulong)family << 32) | (ulong)index;
}

fn uint native_queue_family(ulong id) @private => (uint)(id >> 32);
fn uint native_queue_index(ulong id) @private => (uint)id;

fn uint required_queue_count(
    gpu::QueueRequirements requirements,
    gpu::QueueKind kind,
) @private {
    switch (kind) {
        case gpu::QueueKind.GRAPHICS: return requirements.counts.graphics;
        case gpu::QueueKind.COMPUTE:  return requirements.counts.compute;
        case gpu::QueueKind.TRANSFER: return requirements.counts.transfer;
    }
}

fn uint selected_native_queue_count(
    QueueFamilies* selection,
    gpu::QueueKind kind,
) @private {
    switch (kind) {
        case gpu::QueueKind.GRAPHICS: return selection.counts.graphics;
        case gpu::QueueKind.COMPUTE:  return selection.counts.compute;
        case gpu::QueueKind.TRANSFER: return selection.counts.transfer;
    }
}

fn ulong selected_native_queue(
    QueueFamilies* selection,
    gpu::QueueKind kind,
    uint index,
) @private {
    switch (kind) {
        case gpu::QueueKind.GRAPHICS: return selection.graphics_ids[index];
        case gpu::QueueKind.COMPUTE:  return selection.compute_ids[index];
        case gpu::QueueKind.TRANSFER: return selection.transfer_ids[index];
    }
}

fn ulong representative_native_queue(
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

fn bool queue_kind_is_distinct(
    gpu::QueueRoles distinct,
    gpu::QueueKind kind,
) @private {
    switch (kind) {
        case gpu::QueueKind.GRAPHICS: return distinct.graphics;
        case gpu::QueueKind.COMPUTE:  return distinct.compute;
        case gpu::QueueKind.TRANSFER: return distinct.transfer;
    }
}

fn bool queue_kinds_must_not_alias(
    gpu::QueueRoles distinct,
    gpu::QueueKind first,
    gpu::QueueKind second,
) @private {
    return first != second
        && (queue_kind_is_distinct(distinct, first)
            || queue_kind_is_distinct(distinct, second));
}

fn bool selected_role_contains(
    QueueFamilies* selection,
    gpu::QueueKind kind,
    ulong id,
) @private {
    uint count = selected_native_queue_count(selection, kind);
    for (uint index = 0; index < count; index++) {
        if (selected_native_queue(selection, kind, index) == id) return true;
    }
    return false;
}

fn bool selected_queue_conflicts(
    QueueFamilies* selection,
    gpu::QueueKind kind,
    ulong id,
) @private {
    gpu::QueueKind[3] kinds = {
        gpu::QueueKind.GRAPHICS,
        gpu::QueueKind.COMPUTE,
        gpu::QueueKind.TRANSFER,
    };
    foreach (other : kinds) {
        if (queue_kinds_must_not_alias(selection.distinct, kind, other)
            && selected_role_contains(selection, other, id)) {
            return true;
        }
    }
    return false;
}

fn void append_selected_native_queue(
    QueueFamilies* selection,
    gpu::QueueKind kind,
    ulong id,
) @private {
    switch (kind) {
        case gpu::QueueKind.GRAPHICS:
            selection.graphics_ids[selection.counts.graphics++] = id;
        case gpu::QueueKind.COMPUTE:
            selection.compute_ids[selection.counts.compute++] = id;
        case gpu::QueueKind.TRANSFER:
            selection.transfer_ids[selection.counts.transfer++] = id;
    }
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

fn void? fill_selected_queue_role(
    QueueFamilies* selection,
    vk::QueueFamilyProperties[] families,
    gpu::QueueKind kind,
    uint target_count,
) @private {
    foreach (family, properties : families) {
        if (!queue_family_supports_kind(properties, kind)) continue;
        for (uint index = 0;
            index < properties.queue_count
                && selected_native_queue_count(selection, kind) < target_count;
            index++) {
            ulong id = native_queue_id((uint)family, index);
            if (selected_role_contains(selection, kind, id)
                || selected_queue_conflicts(selection, kind, id)) {
                continue;
            }
            append_selected_native_queue(selection, kind, id);
        }
    }
    if (selected_native_queue_count(selection, kind) != target_count) {
        return gpu::UNSUPPORTED_FEATURE~;
    }
}

fn uint queue_candidate_count(
    QueueCandidatePool* pool,
    gpu::QueueKind kind,
) @private {
    switch (kind) {
        case gpu::QueueKind.GRAPHICS: return pool.candidate_counts.graphics;
        case gpu::QueueKind.COMPUTE:  return pool.candidate_counts.compute;
        case gpu::QueueKind.TRANSFER: return pool.candidate_counts.transfer;
    }
}

fn uint queue_candidate_capacity(
    QueueCandidatePool* pool,
    gpu::QueueKind kind,
) @private {
    switch (kind) {
        case gpu::QueueKind.GRAPHICS: return (uint)pool.graphics.len;
        case gpu::QueueKind.COMPUTE:  return (uint)pool.compute.len;
        case gpu::QueueKind.TRANSFER: return (uint)pool.transfer.len;
    }
}

fn QueueCandidatePool create_queue_candidate_pool(
    gpu::QueueRequirements requirements,
) @private {
    uint slot_capacity = requirements.counts.graphics
        + requirements.counts.compute
        + requirements.counts.transfer;
    uint candidate_capacity = slot_capacity * 3;
    QueueCandidatePool pool = {
        .ids    = mem::talloc_array(ulong, (sz)candidate_capacity),
        .owners = mem::talloc_array(uint, (sz)candidate_capacity),
    };
    if (requirements.counts.graphics > 0) {
        pool.graphics = mem::talloc_array(uint, (sz)slot_capacity);
    }
    if (requirements.counts.compute > 0) {
        pool.compute = mem::talloc_array(uint, (sz)slot_capacity);
    }
    if (requirements.counts.transfer > 0) {
        pool.transfer = mem::talloc_array(uint, (sz)slot_capacity);
    }
    return pool;
}

fn uint queue_candidate_index(
    QueueCandidatePool* pool,
    gpu::QueueKind kind,
    uint index,
) @private {
    switch (kind) {
        case gpu::QueueKind.GRAPHICS: return pool.graphics[index];
        case gpu::QueueKind.COMPUTE:  return pool.compute[index];
        case gpu::QueueKind.TRANSFER: return pool.transfer[index];
    }
}

fn uint queue_candidate_pool_index(
    QueueCandidatePool* pool,
    ulong id,
) @private {
    for (uint index = 0; index < pool.count; index++) {
        if (pool.ids[index] == id) return index;
    }
    uint index = pool.count++;
    pool.ids[index] = id;
    return index;
}

fn bool queue_candidates_contain(
    QueueCandidatePool* pool,
    gpu::QueueKind kind,
    uint candidate,
) @private {
    uint count = queue_candidate_count(pool, kind);
    for (uint index = 0; index < count; index++) {
        if (queue_candidate_index(pool, kind, index) == candidate) return true;
    }
    return false;
}

fn void append_queue_candidate(
    QueueCandidatePool* pool,
    gpu::QueueKind kind,
    ulong id,
) @private {
    if (queue_candidate_count(pool, kind)
        == queue_candidate_capacity(pool, kind)) {
        return;
    }
    uint candidate = queue_candidate_pool_index(pool, id);
    if (queue_candidates_contain(pool, kind, candidate)) return;
    switch (kind) {
        case gpu::QueueKind.GRAPHICS:
            pool.graphics[pool.candidate_counts.graphics++] = candidate;
        case gpu::QueueKind.COMPUTE:
            pool.compute[pool.candidate_counts.compute++] = candidate;
        case gpu::QueueKind.TRANSFER:
            pool.transfer[pool.candidate_counts.transfer++] = candidate;
    }
}

fn void collect_queue_candidates(
    QueueCandidatePool* pool,
    QueueFamilies* selection,
    vk::QueueFamilyProperties[] families,
    gpu::QueueRequirements requirements,
) @private {
    gpu::QueueKind[3] kinds = {
        gpu::QueueKind.GRAPHICS,
        gpu::QueueKind.COMPUTE,
        gpu::QueueKind.TRANSFER,
    };
    foreach (kind : kinds) {
        if (required_queue_count(requirements, kind) == 0) continue;
        append_queue_candidate(
            pool,
            kind,
            representative_native_queue(selection, kind),
        );
        foreach (family, properties : families) {
            if (!queue_family_supports_kind(properties, kind)) continue;
            for (uint index = 0;
                index < properties.queue_count
                    && queue_candidate_count(pool, kind)
                        < queue_candidate_capacity(pool, kind);
                index++) {
                append_queue_candidate(
                    pool,
                    kind,
                    native_queue_id((uint)family, index),
                );
            }
        }
    }
}

fn bool try_assign_queue_slot(
    QueueCandidatePool* pool,
    gpu::QueueKind[] slot_kinds,
    uint slot,
    uint[] slot_candidates,
    bool[] visited,
    uint[] pending_slots,
    uint[] parent_slots,
    bool[] visited_slots,
) @private {
    for (uint index = 0; index < slot_candidates.len; index++) {
        visited_slots[index] = false;
    }

    uint pending_index;
    uint pending_count = 1;
    pending_slots[0] = slot;
    visited_slots[slot] = true;
    while (pending_index < pending_count) {
        uint current_slot = pending_slots[pending_index++];
        gpu::QueueKind kind = slot_kinds[current_slot];
        uint candidate_count = queue_candidate_count(pool, kind);
        for (uint index = 0; index < candidate_count; index++) {
            uint candidate = queue_candidate_index(pool, kind, index);
            if (visited[candidate]) continue;
            visited[candidate] = true;

            uint owner = pool.owners[candidate];
            if (owner == 0) {
                uint assigned_slot = current_slot;
                uint replacement = candidate;
                while (true) {
                    uint previous;
                    if (assigned_slot != slot) {
                        previous = slot_candidates[assigned_slot];
                    }
                    pool.owners[replacement] = assigned_slot + 1;
                    slot_candidates[assigned_slot] = replacement;
                    if (assigned_slot == slot) return true;
                    replacement = previous;
                    assigned_slot = parent_slots[assigned_slot];
                }
            }

            uint owned_slot = owner - 1;
            if (visited_slots[owned_slot]) continue;
            visited_slots[owned_slot] = true;
            parent_slots[owned_slot] = current_slot;
            pending_slots[pending_count++] = owned_slot;
        }
    }
    return false;
}

fn void set_representative_native_queue(
    QueueFamilies* selection,
    gpu::QueueKind kind,
    ulong id,
) @private {
    uint family = native_queue_family(id);
    uint index = native_queue_index(id);
    switch (kind) {
        case gpu::QueueKind.GRAPHICS:
            selection.graphics = family;
            selection.graphics_index = index;
        case gpu::QueueKind.COMPUTE:
            selection.compute = family;
            selection.compute_index = index;
        case gpu::QueueKind.TRANSFER:
            selection.transfer = family;
            selection.transfer_index = index;
    }
}

fn uint requested_distinct_role_count(
    gpu::QueueRequirements requirements,
) @private {
    return (requirements.distinct.graphics ? 1 : 0)
        + (requirements.distinct.compute ? 1 : 0)
        + (requirements.distinct.transfer ? 1 : 0);
}

fn bool requests_multiple_queues(
    gpu::QueueRequirements requirements,
) @private {
    return requirements.counts.graphics > 1
        || requirements.counts.compute > 1
        || requirements.counts.transfer > 1;
}

fn void? assign_mutually_distinct_queues(
    QueueFamilies* selection,
    vk::QueueFamilyProperties[] families,
    gpu::QueueRequirements requirements,
) @private {
    QueueCandidatePool pool = create_queue_candidate_pool(requirements);
    collect_queue_candidates(
        pool:         &pool,
        selection:    selection,
        families:     families,
        requirements: requirements,
    );

    uint slot_count = requirements.counts.graphics
        + requirements.counts.compute
        + requirements.counts.transfer;
    gpu::QueueKind[] slot_kinds =
        mem::talloc_array(gpu::QueueKind, (sz)slot_count);
    uint[] slot_candidates = mem::talloc_array(uint, (sz)slot_count);
    uint next_slot;
    gpu::QueueKind[3] kinds = {
        gpu::QueueKind.GRAPHICS,
        gpu::QueueKind.COMPUTE,
        gpu::QueueKind.TRANSFER,
    };
    foreach (kind : kinds) {
        uint count = required_queue_count(requirements, kind);
        if (queue_candidate_count(&pool, kind) < count) {
            return gpu::UNSUPPORTED_FEATURE~;
        }
        for (uint index = 0; index < count; index++) {
            slot_kinds[next_slot++] = kind;
        }
    }

    bool[] visited = mem::talloc_array(bool, (sz)pool.count);
    uint[] pending_slots = mem::talloc_array(uint, (sz)slot_count);
    uint[] parent_slots = mem::talloc_array(uint, (sz)slot_count);
    bool[] visited_slots = mem::talloc_array(bool, (sz)slot_count);
    for (uint candidate = 0; candidate < pool.count; candidate++) {
        pool.owners[candidate] = 0;
    }
    for (uint slot = 0; slot < slot_count; slot++) {
        for (uint candidate = 0; candidate < pool.count; candidate++) {
            visited[candidate] = false;
        }
        if (!try_assign_queue_slot(
            pool:            &pool,
            slot_kinds:       slot_kinds,
            slot:             slot,
            slot_candidates: slot_candidates,
            visited:          visited,
            pending_slots:    pending_slots,
            parent_slots:     parent_slots,
            visited_slots:    visited_slots,
        )) {
            return gpu::UNSUPPORTED_FEATURE~;
        }
    }

    selection.counts = {};
    selection.distinct = requirements.distinct;
    for (uint slot = 0; slot < slot_count; slot++) {
        append_selected_native_queue(
            selection,
            slot_kinds[slot],
            pool.ids[slot_candidates[slot]],
        );
    }
    foreach (kind : kinds) {
        if (selected_native_queue_count(selection, kind) == 0) continue;
        set_representative_native_queue(
            selection,
            kind,
            selected_native_queue(selection, kind, 0),
        );
    }
    selection.async_compute = selection.counts.graphics > 0
        && selection.counts.compute > 0
        && selected_native_queue(selection, gpu::QueueKind.GRAPHICS, 0)
            != selected_native_queue(selection, gpu::QueueKind.COMPUTE, 0);
}

fn void? fill_selected_queue_role_from_pool(
    QueueFamilies* selection,
    QueueCandidatePool* pool,
    gpu::QueueKind kind,
    uint target_count,
) @private {
    uint candidate_count = queue_candidate_count(pool, kind);
    for (uint index = 0;
        index < candidate_count
            && selected_native_queue_count(selection, kind) < target_count;
        index++) {
        ulong id = pool.ids[queue_candidate_index(pool, kind, index)];
        if (selected_role_contains(selection, kind, id)
            || selected_queue_conflicts(selection, kind, id)) {
            continue;
        }
        append_selected_native_queue(selection, kind, id);
    }
    if (selected_native_queue_count(selection, kind) != target_count) {
        return gpu::UNSUPPORTED_FEATURE~;
    }
}

fn void? assign_one_distinct_queue_role(
    QueueFamilies* selection,
    vk::QueueFamilyProperties[] families,
    gpu::QueueRequirements requirements,
) @private {
    gpu::QueueKind distinct_kind;
    gpu::QueueKind[2] other_kinds;
    if (requirements.distinct.graphics) {
        distinct_kind = gpu::QueueKind.GRAPHICS;
        other_kinds = {
            gpu::QueueKind.COMPUTE,
            gpu::QueueKind.TRANSFER,
        };
    } else if (requirements.distinct.compute) {
        distinct_kind = gpu::QueueKind.COMPUTE;
        other_kinds = {
            gpu::QueueKind.GRAPHICS,
            gpu::QueueKind.TRANSFER,
        };
    } else {
        distinct_kind = gpu::QueueKind.TRANSFER;
        other_kinds = {
            gpu::QueueKind.GRAPHICS,
            gpu::QueueKind.COMPUTE,
        };
    }

    QueueCandidatePool pool = create_queue_candidate_pool(requirements);
    collect_queue_candidates(
        pool:         &pool,
        selection:    selection,
        families:     families,
        requirements: requirements,
    );
    uint distinct_required = required_queue_count(requirements, distinct_kind);
    uint first_required = required_queue_count(requirements, other_kinds[0]);
    uint second_required = required_queue_count(requirements, other_kinds[1]);
    uint first_available = queue_candidate_count(&pool, other_kinds[0]);
    uint second_available = queue_candidate_count(&pool, other_kinds[1]);
    if (queue_candidate_count(&pool, distinct_kind) < distinct_required
        || first_available < first_required
        || second_available < second_required) {
        return gpu::UNSUPPORTED_FEATURE~;
    }

    uint first_slack = first_available - first_required;
    uint second_slack = second_available - second_required;
    uint[4] category_counts;
    uint distinct_available = queue_candidate_count(&pool, distinct_kind);
    for (uint index = 0; index < distinct_available; index++) {
        uint candidate = queue_candidate_index(&pool, distinct_kind, index);
        uint category;
        if (queue_candidates_contain(&pool, other_kinds[0], candidate)) {
            category |= 1;
        }
        if (queue_candidates_contain(&pool, other_kinds[1], candidate)) {
            category |= 2;
        }
        category_counts[category]++;
    }

    uint[4] quotas;
    quotas[0] = category_counts[0] < distinct_required
        ? category_counts[0]
        : distinct_required;
    uint remaining = distinct_required - quotas[0];
    bool found = remaining == 0;
    uint max_both = category_counts[3];
    if (max_both > remaining) max_both = remaining;
    if (max_both > first_slack) max_both = first_slack;
    if (max_both > second_slack) max_both = second_slack;
    for (uint both = 0; !found && both <= max_both; both++) {
        uint after_both = remaining - both;
        uint first_capacity = category_counts[1];
        uint second_capacity = category_counts[2];
        uint first_remaining_slack = first_slack - both;
        uint second_remaining_slack = second_slack - both;
        if (first_capacity > first_remaining_slack) {
            first_capacity = first_remaining_slack;
        }
        if (second_capacity > second_remaining_slack) {
            second_capacity = second_remaining_slack;
        }
        if (after_both > first_capacity + second_capacity) continue;
        uint first = after_both > second_capacity
            ? after_both - second_capacity
            : 0;
        if (first > first_capacity) continue;
        quotas[1] = first;
        quotas[2] = after_both - first;
        quotas[3] = both;
        found = true;
    }
    if (!found) return gpu::UNSUPPORTED_FEATURE~;

    selection.counts = {};
    selection.distinct = requirements.distinct;
    for (uint index = 0; index < distinct_available; index++) {
        uint candidate = queue_candidate_index(&pool, distinct_kind, index);
        uint category;
        if (queue_candidates_contain(&pool, other_kinds[0], candidate)) {
            category |= 1;
        }
        if (queue_candidates_contain(&pool, other_kinds[1], candidate)) {
            category |= 2;
        }
        if (quotas[category] == 0) continue;
        append_selected_native_queue(
            selection,
            distinct_kind,
            pool.ids[candidate],
        );
        quotas[category]--;
    }
    fill_selected_queue_role_from_pool(
        selection:    selection,
        pool:         &pool,
        kind:         other_kinds[0],
        target_count: first_required,
    )!;
    fill_selected_queue_role_from_pool(
        selection:    selection,
        pool:         &pool,
        kind:         other_kinds[1],
        target_count: second_required,
    )!;

    gpu::QueueKind[3] kinds = {
        gpu::QueueKind.GRAPHICS,
        gpu::QueueKind.COMPUTE,
        gpu::QueueKind.TRANSFER,
    };
    foreach (kind : kinds) {
        if (selected_native_queue_count(selection, kind) == 0) continue;
        set_representative_native_queue(
            selection,
            kind,
            selected_native_queue(selection, kind, 0),
        );
    }
    selection.async_compute = selection.counts.graphics > 0
        && selection.counts.compute > 0
        && selected_native_queue(selection, gpu::QueueKind.GRAPHICS, 0)
            != selected_native_queue(selection, gpu::QueueKind.COMPUTE, 0);
}

fn void? assign_requested_queues(
    QueueFamilies* selection,
    vk::QueueFamilyProperties[] families,
    gpu::QueueRequirements requirements,
) @private {
    selection.counts = {};
    if (requests_multiple_queues(requirements)
        && requested_distinct_role_count(requirements) >= 2) {
        @pool() {
            assign_mutually_distinct_queues(selection, families, requirements)!;
        };
        return;
    }

    if (requests_multiple_queues(requirements)
        && requested_distinct_role_count(requirements) == 1) {
        @pool() {
            assign_one_distinct_queue_role(selection, families, requirements)!;
        };
        return;
    }

    selection.distinct = requirements.distinct;
    gpu::QueueKind[3] kinds = {
        gpu::QueueKind.GRAPHICS,
        gpu::QueueKind.COMPUTE,
        gpu::QueueKind.TRANSFER,
    };
    foreach (kind : kinds) {
        if (required_queue_count(requirements, kind) > 0) {
            append_selected_native_queue(
                selection,
                kind,
                representative_native_queue(selection, kind),
            );
        }
    }
    foreach (kind : kinds) {
        if (!queue_kind_is_distinct(requirements.distinct, kind)) continue;
        fill_selected_queue_role(
            selection:    selection,
            families:     families,
            kind:         kind,
            target_count: required_queue_count(requirements, kind),
        )!;
    }
    foreach (kind : kinds) {
        if (queue_kind_is_distinct(requirements.distinct, kind)) continue;
        fill_selected_queue_role(
            selection:    selection,
            families:     families,
            kind:         kind,
            target_count: required_queue_count(requirements, kind),
        )!;
    }
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

fn bool queue_choice_graphics_compute_alias(QueueChoice selection) @private {
    return selection.selected.graphics
        && selection.selected.compute
        && selection.graphics == selection.compute
        && selection.graphics_index == selection.compute_index;
}

fn bool queue_choice_graphics_transfer_alias(QueueChoice selection) @private {
    return selection.selected.graphics
        && selection.selected.transfer
        && selection.graphics == selection.transfer
        && selection.graphics_index == selection.transfer_index;
}

fn bool queue_choice_compute_transfer_alias(QueueChoice selection) @private {
    return selection.selected.compute
        && selection.selected.transfer
        && selection.compute == selection.transfer
        && selection.compute_index == selection.transfer_index;
}

fn bool selection_satisfies_requirements(
    QueueChoice selection,
    gpu::QueueRequirements requirements,
) @private {
    bool graphics_alias = requirements.counts.graphics > 0
        && ((requirements.counts.compute > 0
                && queue_choice_graphics_compute_alias(selection))
            || (requirements.counts.transfer > 0
                && queue_choice_graphics_transfer_alias(selection)));
    bool compute_alias = requirements.counts.compute > 0
        && ((requirements.counts.graphics > 0
                && queue_choice_graphics_compute_alias(selection))
            || (requirements.counts.transfer > 0
                && queue_choice_compute_transfer_alias(selection)));
    bool transfer_alias = requirements.counts.transfer > 0
        && ((requirements.counts.graphics > 0
                && queue_choice_graphics_transfer_alias(selection))
            || (requirements.counts.compute > 0
                && queue_choice_compute_transfer_alias(selection)));
    return !(requirements.distinct.graphics && graphics_alias)
        && !(requirements.distinct.compute && compute_alias)
        && !(requirements.distinct.transfer && transfer_alias);
}

fn bool assign_presentation_queue(
    QueueChoice* selection,
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
    QueueChoice selection,
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
    QueueChoice selection,
    vk::QueueFamilyProperties[] families,
) @private {
    if (!selection.selected.transfer) return 0;
    if (queue_family_is_transfer_only(families[selection.transfer])) return 0;
    if (queue_choice_graphics_transfer_alias(selection)) return 1;
    if (queue_choice_compute_transfer_alias(selection)) return 2;
    if (selection.selected.graphics && selection.graphics == selection.transfer) return 3;
    if (selection.selected.compute && selection.compute == selection.transfer) return 4;
    return 5;
}

fn uint presentation_selection_tier(QueueChoice selection) @private {
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
    QueueChoice candidate,
    QueueChoice current,
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
    if (!gpu::internal::queue_requirements_are_valid(requirements)) {
        return gpu::INVALID_ARGUMENT~;
    }
    if (require_presentation && requirements.counts.graphics == 0) {
        return gpu::INVALID_ARGUMENT~;
    }
    if (available_queue_count(families, gpu::QueueKind.GRAPHICS)
            < requirements.counts.graphics
        || available_queue_count(families, gpu::QueueKind.COMPUTE)
            < requirements.counts.compute
        || available_queue_count(families, gpu::QueueKind.TRANSFER)
            < requirements.counts.transfer) {
        return gpu::UNSUPPORTED_FEATURE~;
    }
    if (require_presentation && presentation_support.len != families.len) {
        return gpu::INVALID_ARGUMENT~;
    }

    QueueChoice base = {
        .selected = {
            .graphics = requirements.counts.graphics > 0,
            .compute  = requirements.counts.compute > 0,
            .transfer = requirements.counts.transfer > 0,
        },
    };
    QueueChoice best;
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
                            QueueChoice candidate = base;
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
                                && !queue_choice_graphics_compute_alias(candidate);
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
    if (!found) return gpu::UNSUPPORTED_FEATURE~;
    QueueFamilies selection = {
        .selected       = best.selected,
        .graphics       = best.graphics,
        .graphics_index = best.graphics_index,
        .compute        = best.compute,
        .compute_index  = best.compute_index,
        .transfer       = best.transfer,
        .transfer_index = best.transfer_index,
        .present        = best.present,
        .present_index  = best.present_index,
        .presentation   = best.presentation,
        .async_compute  = best.async_compute,
    };
    assign_requested_queues(&selection, families, requirements)!;
    return selection;
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
        gpu::internal::default_queue_requirements(),
    );
}

fn QueueFamilies? pick_queue_families_for_surface(
    vk::QueueFamilyProperties[] families,
    bool[] presentation_support,
) {
    return pick_queue_families_for_surface_requirements(
        families,
        presentation_support,
        gpu::internal::default_queue_requirements(),
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
    gpu::QueueKind[3] kinds = {
        gpu::QueueKind.GRAPHICS,
        gpu::QueueKind.COMPUTE,
        gpu::QueueKind.TRANSFER,
    };
    foreach (kind : kinds) {
        uint selected_count = selected_native_queue_count(selection, kind);
        for (uint index = 0; index < selected_count; index++) {
            ulong id = selected_native_queue(selection, kind, index);
            uint queue_index = native_queue_index(id);
            if (native_queue_family(id) == family && count <= queue_index) {
                count = queue_index + 1;
            }
        }
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
    gpu::QueueKind[3] kinds = {
        gpu::QueueKind.GRAPHICS,
        gpu::QueueKind.COMPUTE,
        gpu::QueueKind.TRANSFER,
    };
    foreach (kind : kinds) {
        uint selected_count = selected_native_queue_count(selection, kind);
        for (uint index = 0; index < selected_count; index++) {
            ulong id = selected_native_queue(selection, kind, index);
            count = append_queue_family(
                out_families,
                count,
                native_queue_family(id),
            );
        }
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

fn uint append_selected_role_families(
    QueueFamilies* selection,
    gpu::QueueKind kind,
    uint[] out_families,
    uint count,
) @private {
    uint selected_count = selected_native_queue_count(selection, kind);
    for (uint index = 0; index < selected_count; index++) {
        ulong id = selected_native_queue(selection, kind, index);
        count = append_queue_family(
            out_families,
            count,
            native_queue_family(id),
        );
    }
    return count;
}

fn uint resource_queue_families(
    VkDeviceState* state,
    gpu::QueueRoles access,
    uint[] out_families,
) {
    uint count;
    if (access.graphics) {
        count = append_queue_family(out_families, count, state.graphics_family);
        count = append_selected_role_families(
            selection:    &state.queues,
            kind:         gpu::QueueKind.GRAPHICS,
            out_families: out_families,
            count:        count,
        );
    }
    if (access.compute) {
        count = append_queue_family(out_families, count, state.compute_family);
        count = append_selected_role_families(
            selection:    &state.queues,
            kind:         gpu::QueueKind.COMPUTE,
            out_families: out_families,
            count:        count,
        );
    }
    if (access.transfer) {
        count = append_queue_family(out_families, count, state.transfer_family);
        count = append_selected_role_families(
            selection:    &state.queues,
            kind:         gpu::QueueKind.TRANSFER,
            out_families: out_families,
            count:        count,
        );
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
    gpu::QueueKind[3] kinds = {
        gpu::QueueKind.GRAPHICS,
        gpu::QueueKind.COMPUTE,
        gpu::QueueKind.TRANSFER,
    };
    foreach (kind : kinds) {
        count = append_selected_role_families(
            selection:    &state.queues,
            kind:         kind,
            out_families: out_families,
            count:        count,
        );
    }
    return count;
}

<* Selected strict-heap families, or transfer when it is the only admitted role. *>
fn uint strict_heap_queue_families(VkDeviceState* state, uint[] out_families) @private {
    uint count;
    if (state.selected_queue_roles.graphics) {
        count = append_queue_family(out_families, count, state.graphics_family);
    }
    if (state.selected_queue_roles.compute) {
        count = append_queue_family(out_families, count, state.compute_family);
    }
    count = append_selected_role_families(
        selection:    &state.queues,
        kind:         gpu::QueueKind.GRAPHICS,
        out_families: out_families,
        count:        count,
    );
    count = append_selected_role_families(
        selection:    &state.queues,
        kind:         gpu::QueueKind.COMPUTE,
        out_families: out_families,
        count:        count,
    );
    if (count == 0 && state.selected_queue_roles.transfer) {
        count = append_queue_family(out_families, count, state.transfer_family);
        count = append_selected_role_families(
            selection:    &state.queues,
            kind:         gpu::QueueKind.TRANSFER,
            out_families: out_families,
            count:        count,
        );
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
        count = append_selected_role_families(
            selection:    &state.queues,
            kind:         gpu::QueueKind.GRAPHICS,
            out_families: out_families,
            count:        count,
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
    uint count = selected_native_queue_count(&state.queues, kind);
    for (uint index = 0; index < count; index++) {
        uint identity = gpu::internal::queue_selection_id(selection, kind, index);
        ulong native_id = selected_native_queue(&state.queues, kind, index);
        QueueCompletionState* completion = &state.queue_completions[identity];
        if (completion.bound) {
            if (completion.native_id != native_id) return gpu::BACKEND_ERROR~;
            continue;
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

    if (selection.counts.graphics > 0) {
        state.graphics_completion_id = gpu::internal::queue_selection_id(
            selection,
            gpu::QueueKind.GRAPHICS,
            0,
        );
    }
    if (selection.counts.compute > 0) {
        state.compute_completion_id = gpu::internal::queue_selection_id(
            selection,
            gpu::QueueKind.COMPUTE,
            0,
        );
    }
    if (selection.counts.transfer > 0) {
        state.transfer_completion_id = gpu::internal::queue_selection_id(
            selection,
            gpu::QueueKind.TRANSFER,
            0,
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
            if (cell.used) cell.value.submit_visit_epoch = 0;
        }
        state.next_submit_visit_epoch = 1;
        return 1;
    }
    state.next_submit_visit_epoch++;
    return state.next_submit_visit_epoch;
}

fn gpu::internal::CommandRecord*[]? claim_submit_records_locked(
    VkDeviceState* state,
    gpu::Queue queue,
    gpu::ExecutableCommandList[] tokens,
    gpu::internal::CommandRecord*[] storage,
) @private {
    gpu::internal::CommandRecord*[] records = storage[:tokens.len];
    if (tokens.len == 0) return records;

    ulong epoch = next_submit_visit_epoch_locked(state);
    foreach (i, &token : tokens) {
        note_submit_authoritative_resolution(state);
        if (token.device != queue.device) return gpu::INVALID_HANDLE~;
        gpu::internal::CommandRecord* record =
            state.commands.get_record(token.handle)!;
        if (record.device != queue.device
            || record.identity != token.handle
            || record.backend_state != state) {
            return gpu::INVALID_HANDLE~;
        }
        if (record.state != gpu::internal::CommandState.EXECUTABLE) {
            return gpu::COMMAND_RECORDING_ERROR~;
        }
        VkCommandRecord* backend = backend_command(record);
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
    gpu::internal::CommandRecord*[] records,
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
    gpu::internal::CommandRecord*[] record_storage,
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
            vk::PipelineStageFlags2 stages = submit_wait_scope_to_vk(
                stages:         wait.before,
                consumers:      wait.consumers,
                flags:          target.queue_flags,
                generated_work_enabled: state.generated_work.enabled,
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
    gpu::Queue queue,
    gpu::SubmitDesc* desc,
    QueueSubmit2Fn submit_fn,
    SemaphoreCounterQueryFn counter_query_fn,
) @private {
    gpu::Device device = queue.device;
    VkDeviceState* state = (VkDeviceState*)gpu::internal::device_backend_state_ptr(&device);
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
    gpu::internal::CommandRecord*[] records = inputs.records;
    bool claimed = records.len > 0;
    defer catch {
        if (claimed) rollback_submit_claim(state, records);
    }

    if (records.len > 0) {
        foreach (i, record : records) {
            scratch.command_buffer_infos[i] = vk::command_buffer_submit_info()
                .set_command_buffer(
                    backend_command(record).command_buffer,
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
    gpu::Queue queue,
    gpu::SubmitDesc* desc,
    QueueSubmit2Fn submit_fn,
) @private {
    return vk_submit_with_queries(
        queue:            queue,
        desc:             desc,
        submit_fn:        submit_fn,
        counter_query_fn: &real_semaphore_counter_query,
    );
}

fn gpu::CompletionPoint? vk_submit(
    gpu::Queue queue,
    gpu::SubmitDesc* desc,
) @private {
    return vk_submit_with(queue, desc, &real_queue_submit2);
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
