module gpu_bench;

import gpu::internal::vk @public;
import std::atomic::types;
import std::io;
import std::time;

const uint COMMAND_REFERENCE_MAX = 4_096;
const uint COMMAND_REFERENCE_INDEX_MAX = COMMAND_REFERENCE_MAX * 2;
const uint COMMAND_REFERENCE_REPEAT_COUNT = 100_000;
const uint COMMAND_REFERENCE_MIXED_UNIQUE = 2_048;
const uint COMMAND_REFERENCE_COLLISION_COUNT = 64;
const uint[6] COMMAND_REFERENCE_SIZES = { 1, 8, 64, 256, 1_024, 4_096 };

vk::CommandResourceReference[COMMAND_REFERENCE_MAX] benchmark_references;
vk::CommandReferenceIndexCell[COMMAND_REFERENCE_INDEX_MAX] benchmark_cells;
vk::BufferCell[COMMAND_REFERENCE_MAX] benchmark_buffer_cells;
Atomic{uint}[COMMAND_REFERENCE_MAX] benchmark_retain_counts;

fn void initialize_benchmark_state(vk::VkDeviceState* state) {
    state.buffers = {
        .slots    = benchmark_buffer_cells[..],
        .owner    = 1,
        .next_new = COMMAND_REFERENCE_MAX,
    };
    foreach (&cell : benchmark_buffer_cells) {
        cell.generation = 1;
        cell.used = true;
    }
    state.resource_mutex.init()!!;
}

fn uint reference_bucket(
    vk::CommandResourceReference* reference,
    uint cell_count,
) {
    ulong hash = vk::command_reference_hash(reference);
    return (uint)((hash ^ (hash >> 32)) & (ulong)(cell_count - 1));
}

fn void clear_reference_cells(uint count) {
    for (uint i = 0; i < count; i++) benchmark_cells[i] = {};
}

fn void track_reference(
    vk::VkDeviceState* state,
    vk::CommandScratch* scratch,
    uint index,
) {
    vk::CommandRecord record = { .scratch = scratch };
    vk::track_command_reference(
        state:      state,
        record:     &record,
        owner:      state.buffers.owner,
        index:      index,
        generation: 1,
    )!!;
}

fn void publish_collision_reference(
    vk::VkDeviceState* state,
    vk::CommandScratch* scratch,
    vk::CommandResourceReference* reference,
    uint reference_index,
) {
    vk::CommandReferenceProbe probe = vk::probe_command_reference(
        state,
        scratch,
        reference,
    )!!;
    assert(!probe.found);
    benchmark_retain_counts[reference_index].store(1, AtomicOrdering.RELAXED);
    reference.tracked_command_references = &benchmark_retain_counts[reference_index];
    state.resource_mutex.lock()!!;
    vk::note_command_reference_mutex_acquisition(state);
    defer state.resource_mutex.unlock();
    vk::note_command_reference_retain(state);
    scratch.references[reference_index] = *reference;
    scratch.reference_count++;
    vk::publish_command_reference(
        state:           state,
        scratch:         scratch,
        reference:       reference,
        probe:           &probe,
        reference_index: reference_index,
    );
}

fn void run_unique_case(vk::VkDeviceState* state, uint count) {
    uint cell_count = vk::command_reference_index_capacity(count)!!;
    clear_reference_cells(cell_count);
    vk::CommandScratch scratch = {
        .references = benchmark_references[:count],
        .reference_index = {
            .cells = benchmark_cells[:cell_count],
            .epoch = 1,
        },
    };
    vk::reset_command_reference_work_counters(state);
    vk::reset_command_recording_work_counters(state);
    std::time::Clock start = std::time::clock::now();
    for (uint i = 0; i < count; i++) track_reference(state, &scratch, i);
    long elapsed_ns = (long)start.to_now();
    vk::clear_command_scratch_references(state, &scratch);
    vk::CommandReferenceWorkCounters work =
        vk::command_reference_work_counters(state);
    ulong host_allocations =
        vk::command_recording_work_counters(state).host_allocations;
    assert(work.lookups == count
        && work.index_publications == count
        && work.probes <= (ulong)count * 8
        && work.equality_checks <= work.probes
        && work.duplicate_hits == 0
        && work.resource_mutex_acquisitions <= count
        && work.retains == count
        && work.releases == count
        && host_allocations == 0
        && scratch.reference_count == 0,
        "unique reference-index work remains linearly bounded");
    io::printfn(
        "reference_index unique=%d lookups=%d probes=%d equality=%d publications=%d mutex=%d retains=%d releases=%d host_allocations=%d probes_per_reference=%.3f equality_per_reference=%.3f ns/reference=%.3f",
        count,
        work.lookups,
        work.probes,
        work.equality_checks,
        work.index_publications,
        work.resource_mutex_acquisitions,
        work.retains,
        work.releases,
        host_allocations,
        (double)work.probes / (double)count,
        (double)work.equality_checks / (double)count,
        (double)elapsed_ns / (double)count,
    );
}

fn void run_repeated_case(vk::VkDeviceState* state) {
    clear_reference_cells(2);
    vk::CommandScratch scratch = {
        .references = benchmark_references[:1],
        .reference_index = {
            .cells = benchmark_cells[:2],
            .epoch = 1,
        },
    };
    vk::reset_command_reference_work_counters(state);
    vk::reset_command_recording_work_counters(state);
    track_reference(state, &scratch, 7);
    std::time::Clock start = std::time::clock::now();
    for (uint i = 0; i < COMMAND_REFERENCE_REPEAT_COUNT; i++) {
        track_reference(state, &scratch, 7);
    }
    long elapsed_ns = (long)start.to_now();
    vk::clear_command_scratch_references(state, &scratch);
    vk::CommandReferenceWorkCounters work =
        vk::command_reference_work_counters(state);
    ulong host_allocations =
        vk::command_recording_work_counters(state).host_allocations;
    assert(work.lookups == COMMAND_REFERENCE_REPEAT_COUNT + 1
        && work.probes <= COMMAND_REFERENCE_REPEAT_COUNT + 1
        && work.equality_checks <= COMMAND_REFERENCE_REPEAT_COUNT
        && work.duplicate_hits == COMMAND_REFERENCE_REPEAT_COUNT
        && work.index_publications == 1
        && work.resource_mutex_acquisitions <= 1
        && work.retains == 1
        && work.releases == 1
        && host_allocations == 0,
        "repeated references remain one-probe duplicate hits");
    io::printfn(
        "reference_index repeated=%d lookups=%d probes=%d equality=%d duplicates=%d publications=%d mutex=%d retains=%d releases=%d host_allocations=%d ns/reference=%.3f",
        COMMAND_REFERENCE_REPEAT_COUNT,
        work.lookups,
        work.probes,
        work.equality_checks,
        work.duplicate_hits,
        work.index_publications,
        work.resource_mutex_acquisitions,
        work.retains,
        work.releases,
        host_allocations,
        (double)elapsed_ns / (double)COMMAND_REFERENCE_REPEAT_COUNT,
    );
}

fn void run_mixed_case(vk::VkDeviceState* state) {
    uint operation_count = COMMAND_REFERENCE_MIXED_UNIQUE * 2;
    uint cell_count = vk::command_reference_index_capacity(
        COMMAND_REFERENCE_MIXED_UNIQUE,
    )!!;
    clear_reference_cells(cell_count);
    vk::CommandScratch scratch = {
        .references = benchmark_references[:COMMAND_REFERENCE_MIXED_UNIQUE],
        .reference_index = {
            .cells = benchmark_cells[:cell_count],
            .epoch = 1,
        },
    };
    vk::reset_command_reference_work_counters(state);
    vk::reset_command_recording_work_counters(state);
    std::time::Clock start = std::time::clock::now();
    for (uint i = 0; i < operation_count; i++) {
        uint reference_index = i % COMMAND_REFERENCE_MIXED_UNIQUE;
        track_reference(state, &scratch, reference_index);
    }
    long elapsed_ns = (long)start.to_now();
    vk::clear_command_scratch_references(state, &scratch);
    vk::CommandReferenceWorkCounters work =
        vk::command_reference_work_counters(state);
    ulong host_allocations =
        vk::command_recording_work_counters(state).host_allocations;
    assert(work.lookups == operation_count
        && work.index_publications == COMMAND_REFERENCE_MIXED_UNIQUE
        && work.duplicate_hits == COMMAND_REFERENCE_MIXED_UNIQUE
        && work.resource_mutex_acquisitions <= COMMAND_REFERENCE_MIXED_UNIQUE
        && work.retains == COMMAND_REFERENCE_MIXED_UNIQUE
        && work.releases == COMMAND_REFERENCE_MIXED_UNIQUE
        && host_allocations == 0
        && work.probes <= (ulong)operation_count * 8
        && work.equality_checks <= work.probes,
        "mixed unique and duplicate tracking remains linearly bounded");
    io::printfn(
        "reference_index mixed=%d unique=%d lookups=%d probes=%d equality=%d duplicates=%d publications=%d mutex=%d retains=%d releases=%d host_allocations=%d probes_per_reference=%.3f equality_per_reference=%.3f ns/reference=%.3f",
        operation_count,
        COMMAND_REFERENCE_MIXED_UNIQUE,
        work.lookups,
        work.probes,
        work.equality_checks,
        work.duplicate_hits,
        work.index_publications,
        work.resource_mutex_acquisitions,
        work.retains,
        work.releases,
        host_allocations,
        (double)work.probes / (double)operation_count,
        (double)work.equality_checks / (double)operation_count,
        (double)elapsed_ns / (double)operation_count,
    );
}

fn void run_capacity_case(vk::VkDeviceState* state) {
    uint cell_count = vk::command_reference_index_capacity(
        COMMAND_REFERENCE_MAX,
    )!!;
    clear_reference_cells(cell_count);
    vk::CommandScratch scratch = {
        .references = benchmark_references[..],
        .reference_index = {
            .cells = benchmark_cells[..],
            .epoch = 1,
        },
    };
    vk::CommandRecord record = { .scratch = &scratch };
    vk::reset_command_reference_work_counters(state);
    vk::reset_command_recording_work_counters(state);
    std::time::Clock start = std::time::clock::now();
    for (uint i = 0; i < COMMAND_REFERENCE_MAX - 1; i++) {
        track_reference(state, &scratch, i);
    }
    long elapsed_ns = (long)start.to_now();
    vk::CommandResourceReference[2] candidates = {{
        .owner      = state.buffers.owner,
        .index      = COMMAND_REFERENCE_MAX - 1,
        .generation = 1,
    }, {
        .owner      = state.buffers.owner,
        .index      = COMMAND_REFERENCE_MAX,
        .generation = 1,
    }};
    fault capacity_fault = @catch(vk::preflight_command_references(
        state,
        &record,
        candidates[..],
    ));
    vk::CommandReferenceWorkCounters work =
        vk::command_reference_work_counters(state);
    ulong host_allocations =
        vk::command_recording_work_counters(state).host_allocations;
    assert(capacity_fault == gpu::COMMAND_ALLOCATOR_CAPACITY_EXCEEDED
        && scratch.reference_count == COMMAND_REFERENCE_MAX - 1
        && work.resource_mutex_acquisitions <= COMMAND_REFERENCE_MAX - 1
        && work.retains == COMMAND_REFERENCE_MAX - 1
        && work.index_publications == COMMAND_REFERENCE_MAX - 1
        && host_allocations == 0,
        "near-capacity preflight faults before any partial retain");
    vk::clear_command_scratch_references(state, &scratch);
    work = vk::command_reference_work_counters(state);
    assert(work.releases == COMMAND_REFERENCE_MAX - 1,
        "near-capacity cleanup releases every retained identity once");
    io::printfn(
        "reference_index near_capacity=%d additional=2 capacity_fault=true lookups=%d probes=%d equality=%d publications=%d mutex=%d retains=%d releases=%d host_allocations=%d probes_per_reference=%.3f equality_per_reference=%.3f ns/reference=%.3f",
        COMMAND_REFERENCE_MAX - 1,
        work.lookups,
        work.probes,
        work.equality_checks,
        work.index_publications,
        work.resource_mutex_acquisitions,
        work.retains,
        work.releases,
        host_allocations,
        (double)work.probes / (double)(COMMAND_REFERENCE_MAX - 1),
        (double)work.equality_checks / (double)(COMMAND_REFERENCE_MAX - 1),
        (double)elapsed_ns / (double)(COMMAND_REFERENCE_MAX - 1),
    );
}

fn void run_collision_case(vk::VkDeviceState* state) {
    uint cell_count = COMMAND_REFERENCE_COLLISION_COUNT * 2;
    clear_reference_cells(cell_count);
    vk::CommandScratch scratch = {
        .references = benchmark_references[:COMMAND_REFERENCE_COLLISION_COUNT],
        .reference_index = {
            .cells = benchmark_cells[:cell_count],
            .epoch = 1,
        },
    };
    vk::CommandResourceReference seed = {
        .owner      = 3,
        .index      = 11,
        .generation = 1,
    };
    uint bucket = reference_bucket(&seed, cell_count);
    uint generation = 1;
    for (uint i = 0; i < COMMAND_REFERENCE_COLLISION_COUNT; i++) {
        benchmark_references[i] = seed;
        while (true) {
            benchmark_references[i].generation = generation++;
            if (reference_bucket(&benchmark_references[i], cell_count)
                    == bucket) break;
        }
    }
    vk::reset_command_reference_work_counters(state);
    vk::reset_command_recording_work_counters(state);
    std::time::Clock start = std::time::clock::now();
    for (uint i = 0; i < COMMAND_REFERENCE_COLLISION_COUNT; i++) {
        publish_collision_reference(
            state,
            &scratch,
            &benchmark_references[i],
            i,
        );
    }
    long elapsed_ns = (long)start.to_now();
    vk::clear_command_scratch_references(state, &scratch);
    vk::CommandReferenceWorkCounters work =
        vk::command_reference_work_counters(state);
    ulong host_allocations =
        vk::command_recording_work_counters(state).host_allocations;
    ulong expected_probes =
        (ulong)COMMAND_REFERENCE_COLLISION_COUNT
        * (COMMAND_REFERENCE_COLLISION_COUNT + 1) / 2;
    assert(work.lookups == COMMAND_REFERENCE_COLLISION_COUNT
        && work.index_publications == COMMAND_REFERENCE_COLLISION_COUNT
        && work.probes <= expected_probes
        && work.equality_checks <= work.probes
        && work.resource_mutex_acquisitions <= COMMAND_REFERENCE_COLLISION_COUNT
        && work.retains == COMMAND_REFERENCE_COLLISION_COUNT
        && work.releases == COMMAND_REFERENCE_COLLISION_COUNT
        && host_allocations == 0,
        "forced collisions remain bounded by the fixed index");
    io::printfn(
        "reference_index collisions=%d lookups=%d probes=%d equality=%d publications=%d mutex=%d retains=%d releases=%d host_allocations=%d probes_per_reference=%.3f equality_per_reference=%.3f ns/reference=%.3f",
        COMMAND_REFERENCE_COLLISION_COUNT,
        work.lookups,
        work.probes,
        work.equality_checks,
        work.index_publications,
        work.resource_mutex_acquisitions,
        work.retains,
        work.releases,
        host_allocations,
        (double)work.probes / (double)COMMAND_REFERENCE_COLLISION_COUNT,
        (double)work.equality_checks / (double)COMMAND_REFERENCE_COLLISION_COUNT,
        (double)elapsed_ns / (double)COMMAND_REFERENCE_COLLISION_COUNT,
    );
}

fn int main() {
    vk::VkDeviceState state;
    initialize_benchmark_state(&state);
    defer (void)state.resource_mutex.destroy();
    io::printfn(
        "iterations=unique:1,8,64,256,1024,4096;mixed:4096;near_capacity:4095;repeated:%d;collisions:%d units=ns/reference",
        COMMAND_REFERENCE_REPEAT_COUNT,
        COMMAND_REFERENCE_COLLISION_COUNT,
    );
    foreach (count : COMMAND_REFERENCE_SIZES) run_unique_case(&state, count);
    run_repeated_case(&state);
    run_mixed_case(&state);
    run_collision_case(&state);
    run_capacity_case(&state);
    io::printn("reference_index_check status=pass");
    return 0;
}
