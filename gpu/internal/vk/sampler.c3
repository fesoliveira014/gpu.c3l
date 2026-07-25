module gpu::internal::vk @private;

import gpu::internal @public;
import gpu @public;
import vk;
import std::core::mem;

const uint SAMPLER_CELL_NONE = uint::max;

struct SamplerCell {
    ulong             hash;
    SamplerKey        key;
    vk::Sampler       native;
    gpu::SamplerIndex index;
    uint              next_in_bucket;
}

struct SamplerTable {
    SamplerCell[] slots;
    uint[]        bucket_heads;
    uint          count;
}

struct SamplerLookupStats @if(env::TESTING || $feature(SAMPLER_INTERN_STATS)) {
    ulong lookup_probes;
}

struct SamplerLookupProbeResult
    @if(env::TESTING || $feature(SAMPLER_INTERN_STATS)) {
    uint  occupancy;
    uint  bucket_count;
    ulong probes;
    ulong empty_bucket_miss_probes;
}

fn uint sampler_bucket(ulong hash, uint bucket_count) {
    return bucket_for(hash, bucket_count);
}

fn void link_sampler_cell(SamplerTable* table, uint index) {
    uint bucket = sampler_bucket(
        table.slots[index].hash,
        (uint)table.bucket_heads.len,
    );
    table.slots[index].next_in_bucket = table.bucket_heads[bucket];
    table.bucket_heads[bucket] = index + 1;
}

fn void note_sampler_lookup_probe(
    VkDeviceState* state,
) @if(env::TESTING || $feature(SAMPLER_INTERN_STATS)) {
    state.sampler_lookup_probes++;
}

fn void reset_sampler_lookup_stats(
    VkDeviceState* state,
) @if(env::TESTING || $feature(SAMPLER_INTERN_STATS)) {
    state.sampler_lookup_probes = 0;
}

fn SamplerLookupStats sampler_lookup_stats(
    VkDeviceState* state,
) @if(env::TESTING || $feature(SAMPLER_INTERN_STATS)) {
    return { .lookup_probes = state.sampler_lookup_probes };
}

fn ulong sampler_key_hash(VkDeviceState* state, SamplerKey* key) {
    $if env::TESTING:
        if (state.sampler_hash_override_enabled) {
            return state.sampler_hash_override;
        }
    $endif
    return hash_sampler_key(key);
}

fn uint find_sampler_cell(
    VkDeviceState* state,
    SamplerTable* table,
    ulong hash,
    SamplerKey* key,
) {
    uint current = table.bucket_heads[sampler_bucket(
        hash,
        (uint)table.bucket_heads.len,
    )];
    while (current != 0) {
        $if env::TESTING || $feature(SAMPLER_INTERN_STATS):
            note_sampler_lookup_probe(state);
        $endif
        SamplerCell* cell = &table.slots[current - 1];
        if (cell.hash == hash && sampler_key_equal(&cell.key, key)) {
            return current - 1;
        }
        current = cell.next_in_bucket;
    }
    return SAMPLER_CELL_NONE;
}

fn bool sampler_table_index_consistent(
    SamplerTable* table,
) @if(env::TESTING || $feature(SAMPLER_INTERN_STATS)) {
    @pool() {
        bool[] visited = mem::talloc_array(bool, (sz)table.count);
        for (uint index = 0; index < table.count; index++) {
            visited[index] = false;
        }
        uint linked_count;
        foreach (uint bucket, uint current : table.bucket_heads) {
            while (current != 0) {
                uint index = current - 1;
                if (index >= table.count || visited[index]) return false;
                SamplerCell* cell = &table.slots[index];
                if (sampler_bucket(
                    cell.hash,
                    (uint)table.bucket_heads.len,
                ) != bucket) {
                    return false;
                }
                visited[index] = true;
                linked_count++;
                current = cell.next_in_bucket;
            }
        }
        return linked_count == table.count;
    };
}

fn SamplerTable create_sampler_lookup_probe_table(
    uint occupancy,
) @if(env::TESTING || $feature(SAMPLER_INTERN_STATS)) {
    SamplerTable table = {
        .slots        = mem::new_array(SamplerCell, occupancy),
        .bucket_heads = mem::new_array(uint, next_pow2(occupancy * 2)),
    };
    for (uint index = 0; index < occupancy; index++) {
        SamplerKey key = { .mip_lod_bias = (float)(index + 1) };
        table.slots[index] = {
            .hash  = hash_sampler_key(&key),
            .key   = key,
            .index = { .value = index + 1 },
        };
        link_sampler_cell(&table, index);
        table.count++;
    }
    return table;
}

fn void destroy_sampler_lookup_probe_table(
    SamplerTable* table,
) @if(env::TESTING || $feature(SAMPLER_INTERN_STATS)) {
    mem::free(table.slots.ptr);
    mem::free(table.bucket_heads.ptr);
    *table = {};
}

fn SamplerLookupProbeResult probe_sampler_lookup(
    VkDeviceState* state,
    SamplerTable* table,
) @if(env::TESTING || $feature(SAMPLER_INTERN_STATS)) {
    SamplerLookupProbeResult result = {
        .occupancy    = table.count,
        .bucket_count = (uint)table.bucket_heads.len,
    };
    if (table.count == 0) return result;

    SamplerCell* expected = &table.slots[0];
    reset_sampler_lookup_stats(state);
    uint found = find_sampler_cell(
        state: state,
        table: table,
        hash:  expected.hash,
        key:   &expected.key,
    );
    result.probes = sampler_lookup_stats(state).lookup_probes;
    if (found != 0) result.probes = 0;
    bool observed_empty_bucket;
    foreach (uint bucket, uint head : table.bucket_heads) {
        if (head != 0) continue;
        SamplerKey missing_key;
        reset_sampler_lookup_stats(state);
        uint missing = find_sampler_cell(
            state: state,
            table: table,
            hash:  (ulong)bucket,
            key:   &missing_key,
        );
        if (missing != SAMPLER_CELL_NONE) {
            result.empty_bucket_miss_probes = ulong::max;
        } else {
            result.empty_bucket_miss_probes =
                sampler_lookup_stats(state).lookup_probes;
        }
        observed_empty_bucket = true;
        break;
    }
    if (!observed_empty_bucket) {
        result.empty_bucket_miss_probes = ulong::max;
    }
    return result;
}

fn void set_sampler_hash_override(
    VkDeviceState* state,
    bool enabled,
    ulong hash,
) @if(env::TESTING) {
    state.sampler_hash_override_enabled = enabled;
    state.sampler_hash_override = hash;
}

fn vk::Filter sampler_filter_to_vk(gpu::Filter filter) @private {
    return filter == gpu::Filter.LINEAR ? vk::Filter.LINEAR : vk::Filter.NEAREST;
}

fn vk::SamplerAddressMode sampler_address_mode_to_vk(
    gpu::AddressMode mode,
) @private {
    switch (mode) {
        case gpu::AddressMode.REPEAT:          return vk::SamplerAddressMode.REPEAT;
        case gpu::AddressMode.MIRRORED_REPEAT: return vk::SamplerAddressMode.MIRRORED_REPEAT;
        case gpu::AddressMode.CLAMP_TO_EDGE:   return vk::SamplerAddressMode.CLAMP_TO_EDGE;
        case gpu::AddressMode.CLAMP_TO_BORDER: return vk::SamplerAddressMode.CLAMP_TO_BORDER;
        default:                               return vk::SamplerAddressMode.REPEAT;
    }
}

fn void? create_sampler_table(VkDeviceState* state) @private {
    if (!state.strict_heap.enabled) return;
    vk::PhysicalDeviceProperties properties;
    vk::get_physical_device_properties(state.physical_device, &properties);
    uint capacity = properties.limits.max_sampler_allocation_count;
    if (capacity > gpu::MAX_SHADER_HEAP_CAPACITY) {
        capacity = gpu::MAX_SHADER_HEAP_CAPACITY;
    }
    if (capacity == 0) return gpu::UNSUPPORTED_FEATURE~;
    state.samplers.slots = mem::new_array(SamplerCell, capacity);
    state.samplers.bucket_heads = mem::new_array(
        uint,
        next_pow2(capacity * 2),
    );
}

fn void destroy_sampler_table(VkDeviceState* state) @private {
    SamplerTable* table = &state.samplers;
    if (table.slots.ptr != null) {
        foreach (&cell : table.slots[:table.count]) {
            if (cell.native != null) {
                vk::destroy_sampler(state.device, cell.native, null);
                cell.native = (vk::Sampler)null;
            }
        }
        mem::free(table.slots.ptr);
    }
    if (table.bucket_heads.ptr != null) mem::free(table.bucket_heads.ptr);
    *table = {};
}

fn void set_sampler_create_result(
    VkDeviceState* state,
    vk::Result result,
) @if(env::TESTING) {
    state.sampler_create_result = result;
}

fn float sampler_last_create_max_anisotropy(
    VkDeviceState* state,
) @if(env::TESTING) {
    return state.sampler_last_create_max_anisotropy;
}

fn gpu::SamplerIndex? vk_intern_sampler(
    VkDeviceState* state,
    gpu::SamplerDesc* desc,
) {
    SamplerKey key = canonical_sampler_key(desc);
    state.resource_mutex.lock()!!;
    defer state.resource_mutex.unlock();

    SamplerTable* table = &state.samplers;
    ulong hash = sampler_key_hash(state, &key);
    uint existing = find_sampler_cell(
        state: state,
        table: table,
        hash:  hash,
        key:   &key,
    );
    if (existing != SAMPLER_CELL_NONE) return table.slots[existing].index;
    if (table.count >= table.slots.len) return gpu::SLOT_TABLE_FULL~;
    DescriptorHeapState* heap = &state.strict_heap.descriptors;
    if (heap.sampler_next_new >= state.strict_heap.sampler_capacity) {
        return report_contract_failure(
            state:               state,
            operation:           "intern_sampler",
            public_fault:        gpu::DESCRIPTOR_HEAP_FULL,
            resource_kind:       gpu::DebugResourceKind.SAMPLER,
            has_public_identity: false,
            index:               0,
            generation:          0,
            debug_name:          desc.debug_name,
            rejected_field:      "RuntimeDesc.sampler_heap_capacity",
            invariant:           "must exceed interned samplers",
            backend_text:        null,
        )~;
    }

    vk::SamplerCreateInfo sampler_info = vk::sampler_create_info()
        .set_mag_filter(sampler_filter_to_vk(key.mag_filter))
        .set_min_filter(sampler_filter_to_vk(key.min_filter))
        .set_mipmap_mode(key.mip_filter == gpu::Filter.LINEAR
            ? vk::SamplerMipmapMode.LINEAR
            : vk::SamplerMipmapMode.NEAREST)
        .set_address_mode_u(sampler_address_mode_to_vk(key.address_u))
        .set_address_mode_v(sampler_address_mode_to_vk(key.address_v))
        .set_address_mode_w(sampler_address_mode_to_vk(key.address_w))
        .set_mip_lod_bias(key.mip_lod_bias)
        .set_anisotropy_enable(key.anisotropy_enable != 0 ? vk::TRUE : vk::FALSE)
        .set_max_anisotropy(key.max_anisotropy)
        .set_compare_enable(key.compare_enable != 0 ? vk::TRUE : vk::FALSE)
        .set_compare_op(compare_op_to_vk(key.compare))
        .set_min_lod(key.min_lod)
        .set_max_lod(key.max_lod);

    vk::Sampler native;
    vk::Result create_result;
    $if env::TESTING:
        state.sampler_last_create_max_anisotropy = sampler_info.max_anisotropy;
        create_result = state.sampler_create_result;
        if (create_result == vk::Result.SUCCESS) {
            create_result = vk::create_sampler(
                device:        state.device,
                p_create_info: &sampler_info,
                p_allocator:   null,
                p_sampler:     &native,
            );
        } else {
            state.sampler_create_result = vk::Result.SUCCESS;
        }
    $else
        create_result = vk::create_sampler(
            device:        state.device,
            p_create_info: &sampler_info,
            p_allocator:   null,
            p_sampler:     &native,
        );
    $endif
    check_backend_result(
        state:     state,
        operation: "intern_sampler",
        result:    create_result,
    )!;
    set_object_name(
        state:         state,
        object_type:   vk::ObjectType.SAMPLER,
        object_handle: (ulong)native,
        name:          desc.debug_name,
    );

    uint slot = heap.sampler_next_new;
    write_sampler_descriptor(state, slot, native);
    gpu::SamplerIndex index = { .value = slot + 1 };
    uint cell_index = table.count;
    table.slots[cell_index] = {
        .hash   = hash,
        .key    = key,
        .native = native,
        .index  = index,
    };
    link_sampler_cell(table, cell_index);
    table.count++;
    heap.sampler_next_new++;
    return index;
}
