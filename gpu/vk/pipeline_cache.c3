module gpu::vk @private;

import gpu @public;
import vk;
import vma;
import std::thread;
import std::core::mem;
import std::hash::fnv64a;

alias PipelineCacheCreateFn @private = fn vk::Result (void*, vk::Device, vk::PipelineCacheCreateInfo*, vk::PipelineCache*);

// Descriptor-keyed pipeline dedup cache: entries own the vk::Pipeline,
// refcounted by the slots that alias them; layouts are device-owned and
// shared. Monomorphic table, same c3c 0.8.0 rationale as PipelineTable.

const uint PIPELINE_ENTRY_NONE = uint::max;

enum PipelineKind : int {
    COMPUTE,
    GRAPHICS,
}

struct BlendKey {
    int src_color;
    int dst_color;
    int color_op;
    int src_alpha;
    int dst_alpha;
    int alpha_op;
}
$assert BlendKey::size == 24;

struct ColorTargetKey {
    int                 format;
    int                 blend_enable;
    BlendKey            blend;
    gpu::ColorWriteMask write_mask;
}
$assert ColorTargetKey::size == 36;

// Canonical packed key over immutable pipeline state. Zero-init plus field-wise
// assignment keeps the byte view deterministic for hashing. Command-time state
// and debug_name stay out.
struct PipelineKey {
    ShaderId vertex_shader;
    ShaderId fragment_shader;
    uint  color_target_count;
    ColorTargetKey[gpu::MAX_COLOR_ATTACHMENTS] color_targets;
    int   depth_format;
    int   sample_count;
    int   polygon_mode;
    int   kind;
    uint  reserved;
}
$assert PipelineKey::size == 320;

struct PipelineCacheEntry {
    ulong                          hash;
    PipelineKey                    key;
    vk::Pipeline                   pipeline;
    vk::PipelineLayout             layout;
    vk::IndirectCommandsLayoutEXT  generated_dispatch_layout;
    uint                           refcount;
    uint                           next_free;
    uint                           next_in_bucket;
    bool                           used;
}

struct PipelineCacheTable {
    PipelineCacheEntry[]  entries;
    // Pow2-sized, +1-encoded like next_free/next_in_bucket; 0 = empty bucket.
    uint[]                bucket_heads;
    uint                  free_head;
    uint                  next_new;
}

struct PipelineCacheCreateOps @private {
    void*                 user_data;
    PipelineCacheCreateFn create;
}

fn fault report_pipeline_cache_backend_failure(
    VkDeviceState* state,
    ZString operation,
    fault public_fault,
    ZString backend_text,
    int backend_result,
) {
    if (state.debug_callback != null) {
        gpu::DebugMessage message = {
            .severity             = gpu::DebugMessageSeverity.ERROR,
            .category             = { .backend },
            .operation            = operation,
            .has_fault            = true,
            .public_fault         = public_fault,
            .backend_text         = backend_text,
            .validation_id_number = backend_result,
        };
        gpu::dispatch_debug_message(state.debug_callback, state.debug_user_data, &message);
    }
    return public_fault;
}

fn void? check_pipeline_cache_terminal_result(
    VkDeviceState* state,
    ZString operation,
    vk::Result result,
) {
    return check_backend_result(state, operation, result);
}

fn void? check_pipeline_cache_data_backend_result(
    VkDeviceState* state,
    ZString operation,
    vk::Result result,
) {
    if (result != vk::Result.INCOMPLETE) {
        return check_pipeline_cache_terminal_result(state, operation, result);
    }
    return report_pipeline_cache_backend_failure(
        state:          state,
        operation:      operation,
        public_fault:   gpu::INVALID_ARGUMENT,
        backend_text:   "Vulkan returned incomplete pipeline cache data",
        backend_result: (int)result,
    )~;
}

fn vk::Result create_pipeline_cache_real(
    void* user_data,
    vk::Device device,
    vk::PipelineCacheCreateInfo* info,
    vk::PipelineCache* out_cache,
) @private {
    return vk::create_pipeline_cache(device, info, null, out_cache);
}

fn PipelineCacheCreateOps pipeline_cache_create_ops_real() @private {
    return { .create = &create_pipeline_cache_real };
}

fn void? create_driver_pipeline_cache_with_ops(
    VkDeviceState* state,
    char[] cache_data,
    PipelineCacheCreateOps* ops,
) @private {
    vk::PipelineCacheCreateInfo cache_info = vk::pipeline_cache_create_info();
    cache_info.initial_data_size = (sz)cache_data.len;
    cache_info.p_initial_data    = cache_data.ptr;
    vk::Result first_result = ops.create(
        ops.user_data,
        state.device,
        &cache_info,
        &state.driver_cache,
    );
    if (first_result == vk::Result.SUCCESS) return;
    if (cache_data.len == 0 || !pipeline_cache_blob_retryable(first_result)) {
        return check_pipeline_cache_terminal_result(state, "create_device", first_result);
    }

    cache_info.initial_data_size = 0;
    cache_info.p_initial_data    = null;
    return check_pipeline_cache_terminal_result(
        state:     state,
        operation: "create_device",
        result:    ops.create(
            ops.user_data,
            state.device,
            &cache_info,
            &state.driver_cache,
        ),
    );
}

fn void? create_driver_pipeline_cache(VkDeviceState* state, char[] cache_data) @private {
    PipelineCacheCreateOps ops = pipeline_cache_create_ops_real();
    return create_driver_pipeline_cache_with_ops(state, cache_data, &ops);
}

fn PipelineKey build_graphics_key(
    gpu::GraphicsPipelineDesc* desc,
    ShaderId vertex_shader,
    ShaderId fragment_shader,
) @private {
    PipelineKey key = {
        .vertex_shader   = vertex_shader,
        .fragment_shader = fragment_shader,
    };
    key.kind                   = (int)PipelineKind.GRAPHICS;
    key.polygon_mode           = (int)desc.polygon_mode;
    key.depth_format           = (int)desc.depth_format;
    key.sample_count           = (int)desc.sample_count;
    key.color_target_count     = (uint)desc.colors.len;
    foreach (i, &target : desc.colors) {
        ColorTargetKey* color = &key.color_targets[i];
        color.format = (int)target.format;
        color.write_mask = target.write_mask;
        if (target.blend.enable && (uint)color.write_mask != 0) {
            color.blend_enable = 1;
            color.blend = {
                .src_color = (int)target.blend.src_color,
                .dst_color = (int)target.blend.dst_color,
                .color_op  = (int)target.blend.color_op,
                .src_alpha = (int)target.blend.src_alpha,
                .dst_alpha = (int)target.blend.dst_alpha,
                .alpha_op  = (int)target.blend.alpha_op,
            };
        }
    }
    return key;
}

fn PipelineKey build_compute_key(ShaderId shader) @private {
    return {
        .vertex_shader   = shader,
        .fragment_shader = SHADER_ID_INVALID,
        .kind            = (int)PipelineKind.COMPUTE,
    };
}

fn ulong hash_key(PipelineKey* key) @private {
    return fnv64a::hash(((char*)key)[:PipelineKey::size]);
}

fn bool key_equals(PipelineKey* a, PipelineKey* b) @private {
    return mem::equals(((char*)a)[:PipelineKey::size], ((char*)b)[:PipelineKey::size]);
}

fn uint bucket_for(ulong hash, uint bucket_count) @private {
    return (uint)((hash ^ (hash >> 32)) & (ulong)(bucket_count - 1));
}

<* Sizes bucket_heads from the resolved pipeline capacity. *>
fn uint next_pow2(uint value) @private {
    if (value <= 1) return 1;
    value--;
    value |= value >> 1;
    value |= value >> 2;
    value |= value >> 4;
    value |= value >> 8;
    value |= value >> 16;
    return value + 1;
}

fn void link_entry(PipelineCacheTable* cache, uint idx) @private {
    uint bucket = bucket_for(cache.entries[idx].hash, (uint)cache.bucket_heads.len);
    cache.entries[idx].next_in_bucket = cache.bucket_heads[bucket];
    cache.bucket_heads[bucket] = idx + 1;
}

fn void unlink_entry(PipelineCacheTable* cache, uint idx) @private {
    uint bucket = bucket_for(cache.entries[idx].hash, (uint)cache.bucket_heads.len);
    uint* cur = &cache.bucket_heads[bucket];
    while (*cur != idx + 1) cur = &cache.entries[*cur - 1].next_in_bucket;
    *cur = cache.entries[idx].next_in_bucket;
}

<* Index of the live entry matching hash+key, or PIPELINE_ENTRY_NONE. *>
fn uint find_entry(
    VkDeviceState* state,
    ulong hash,
    PipelineKey* key,
) {
    PipelineCacheTable* cache = &state.pipeline_cache;
    uint bucket = bucket_for(hash, (uint)cache.bucket_heads.len);
    for (uint cur = cache.bucket_heads[bucket]; cur != 0; ) {
        note_pipeline_key_probe(state);
        PipelineCacheEntry* entry = &cache.entries[cur - 1];
        if (entry.hash == hash
            && key_equals(&entry.key, key)) {
            return cur - 1;
        }
        cur = entry.next_in_bucket;
    }
    return PIPELINE_ENTRY_NONE;
}

fn uint? insert_entry(
    VkDeviceState* state,
    PipelineCacheEntry value,
) {
    $if env::TESTING:
        if (state.pipeline_cache_insert_failure) {
            return gpu::SLOT_TABLE_FULL~;
        }
    $endif
    PipelineCacheTable* cache = &state.pipeline_cache;
    uint idx;
    if (cache.free_head != PIPELINE_FREE_NONE) {
        idx = cache.free_head - 1;
        cache.free_head = cache.entries[idx].next_free;
    } else if (cache.next_new < cache.entries.len) {
        idx = cache.next_new++;
    } else {
        return gpu::SLOT_TABLE_FULL~;
    }
    cache.entries[idx] = value;
    cache.entries[idx].used = true;
    cache.entries[idx].refcount = 1;
    link_entry(cache, idx);
    return idx;
}

<* Drop one alias and destroy the native pipeline with the last one. *>
fn void release_entry(VkDeviceState* state, uint entry_index) {
    PipelineCacheEntry* entry = &state.pipeline_cache.entries[entry_index];
    if (--entry.refcount > 0) return;
    vk::destroy_pipeline(state.device, entry.pipeline, null);
    release_shader_identity_locked(state, entry.key.vertex_shader)!!;
    if (entry.key.fragment_shader != SHADER_ID_INVALID) {
        release_shader_identity_locked(state, entry.key.fragment_shader)!!;
    }
    unlink_entry(&state.pipeline_cache, entry_index);
    entry.used = false;
    entry.next_free = state.pipeline_cache.free_head;
    state.pipeline_cache.free_head = entry_index + 1;
}

fn ulong pipeline_shader_module_create_count(VkDeviceState* state) {
    return state.pipeline_shader_modules_created;
}

fn ulong pipeline_native_create_count(VkDeviceState* state) {
    return state.pipelines_created;
}

<* Deterministic partial-batch rollback probe. *>
fn void set_pipeline_creation_failure(VkDeviceState* state, usz index) {
    state.pipeline_creation_failure_enabled = true;
    state.pipeline_creation_failure_index = index;
}

fn void clear_pipeline_creation_failure(VkDeviceState* state) {
    state.pipeline_creation_failure_enabled = false;
    state.pipeline_creation_failure_index = 0;
}

fn void set_pipeline_shader_create_result(
    VkDeviceState* state,
    vk::Result result,
) @if(env::TESTING) {
    state.pipeline_shader_create_result = result;
}

fn void set_pipeline_native_create_result(
    VkDeviceState* state,
    vk::Result result,
) @if(env::TESTING) {
    state.pipeline_native_create_result = result;
}

fn void set_pipeline_cache_insert_failure(
    VkDeviceState* state,
    bool enabled,
) @if(env::TESTING) {
    state.pipeline_cache_insert_failure = enabled;
}

fn bool pipeline_creation_should_fail(VkDeviceState* state, usz index) {
    return state.pipeline_creation_failure_enabled
        && state.pipeline_creation_failure_index == index;
}

<* Live entry count, exposed for dedup accounting in tests. *>
fn uint pipeline_cache_entry_count(VkDeviceState* state) {
    uint count = 0;
    foreach (&entry : state.pipeline_cache.entries[:state.pipeline_cache.next_new]) {
        if (entry.used) count++;
    }
    return count;
}

<* Whitebox check: every bucket-linked entry is used and hashes back to its
   own bucket, and the total linked count matches the live entry count. *>
fn bool pipeline_cache_index_consistent(VkDeviceState* state) {
    PipelineCacheTable* cache = &state.pipeline_cache;
    uint bucket_count = (uint)cache.bucket_heads.len;
    uint linked = 0;
    foreach (bucket, head : cache.bucket_heads) {
        for (uint cur = head; cur != 0; ) {
            PipelineCacheEntry* entry = &cache.entries[cur - 1];
            if (!entry.used) return false;
            if (bucket_for(entry.hash, bucket_count) != (uint)bucket) return false;
            linked++;
            cur = entry.next_in_bucket;
        }
    }
    return linked == pipeline_cache_entry_count(state);
}

fn PipelineIdentityProbeResult? probe_pipeline_identity_boundary(
    VkDeviceState* state,
    gpu::ShaderCode* code,
) @if(env::TESTING || $feature(PIPELINE_CACHE_STATS)) {
    state.resource_mutex.lock()!!;
    defer state.resource_mutex.unlock();

    reset_pipeline_identity_stats(state);
    ShaderId shader = retain_shader_identity_locked(state, code)!;
    PipelineIdentityStats interning = pipeline_identity_stats(state);
    PipelineKey key = build_compute_key(shader);
    ulong hash = hash_key(&key);
    uint? inserted_optional = insert_entry(
        state,
        {
            .hash = hash,
            .key  = key,
        },
    );
    if (catch insert_fault = inserted_optional) {
        release_shader_identity_locked(state, shader)!;
        return insert_fault~;
    }
    uint inserted = inserted_optional;

    reset_pipeline_identity_stats(state);
    uint found = find_entry(state, hash, &key);
    if (found != inserted) {
        release_entry(state, inserted);
        return gpu::BACKEND_ERROR~;
    }
    release_entry(state, inserted);
    return {
        .interning = interning,
        .lookup    = pipeline_identity_stats(state),
    };
}

<* Spec-recommended pre-screen; the driver retry is the real safety net. *>
fn bool cache_blob_compatible(VkDeviceState* state, char[] blob) @private {
    if (blob.len == 0) return true;
    if (blob.len < 32) return false;
    uint header_len = *(uint*)blob.ptr;
    uint header_version = *(uint*)&blob[4];
    if (header_len != 32 || header_version != 1) return false;
    vk::PhysicalDeviceProperties props;
    vk::get_physical_device_properties(state.physical_device, &props);
    if (*(uint*)&blob[8] != props.vendor_id) return false;
    if (*(uint*)&blob[12] != props.device_id) return false;
    return mem::equals(blob[16:16], (char[])props.pipeline_cache_uuid[0:16]);
}

<* Shared layouts, cache entry storage, and the driver VkPipelineCache. Requires
   the descriptor heap (set layout) to exist. *>
fn void? create_pipeline_shared(VkDeviceState* state, char[] initial_cache_data, uint capacity) {
    uint push_stages = (uint)vk::ShaderStageFlagBits.VERTEX_BIT | (uint)vk::ShaderStageFlagBits.FRAGMENT_BIT;
    vk::PushConstantRange range = {
        .stage_flags = (vk::ShaderStageFlags)push_stages,
        .offset      = 0,
        .size        = GRAPHICS_ROOT_PUSH_SIZE,
    };
    vk::PipelineLayoutCreateInfo layout_info = vk::pipeline_layout_create_info()
        .set_set_layouts((&state.strict_heap.descriptors.set_layout)[:1])
        .set_push_constant_ranges((&range)[:1]);
    check_backend_result(
        state:     state,
        operation: "create_device",
        result:    vk::create_pipeline_layout(state.device, &layout_info, null, &state.graphics_layout),
    )!;
    set_object_name(
        state:         state,
        object_type:   vk::ObjectType.PIPELINE_LAYOUT,
        object_handle: (ulong)state.graphics_layout,
        name:          "gpu_graphics_layout",
    );

    vk::PushConstantRange compute_root = {
        .stage_flags = vk::ShaderStageFlagBits.COMPUTE_BIT,
        .offset      = 0,
        .size        = gpu::RootPush::size,
    };
    vk::PipelineLayoutCreateInfo compute_layout_info =
        vk::pipeline_layout_create_info()
            .set_set_layouts((&state.strict_heap.descriptors.set_layout)[:1])
            .set_push_constant_ranges((&compute_root)[:1]);
    check_backend_result(
        state:     state,
        operation: "create_device",
        result:    vk::create_pipeline_layout(
            state.device,
            &compute_layout_info,
            null,
            &state.compute_layout,
        ),
    )!;
    set_object_name(
        state:         state,
        object_type:   vk::ObjectType.PIPELINE_LAYOUT,
        object_handle: (ulong)state.compute_layout,
        name:          "gpu_compute_layout",
    );
    if (state.generated_work.enabled) {
        state.generated_dispatch_layout = create_generated_work_layout(
            state:           state,
            pipeline_layout: state.compute_layout,
            shader_stages:   vk::ShaderStageFlagBits.COMPUTE_BIT,
            root_size:       gpu::RootPush::size,
            action_type:     vk::IndirectCommandsTokenTypeEXT.DISPATCH_EXT,
            action_offset:   8,
            stride:          gpu::GeneratedDispatchRecord::size,
        )!;
        state.generated_draw_layout = create_generated_work_layout(
            state:           state,
            pipeline_layout: state.graphics_layout,
            shader_stages:   (vk::ShaderStageFlags)push_stages,
            root_size:       gpu::GraphicsRootPush::size,
            action_type:     vk::IndirectCommandsTokenTypeEXT.DRAW_EXT,
            action_offset:   16,
            stride:          gpu::GeneratedDrawRecord::size,
        )!;
        state.generated_indexed_draw_layout = create_generated_work_layout(
            state:           state,
            pipeline_layout: state.graphics_layout,
            shader_stages:   (vk::ShaderStageFlags)push_stages,
            root_size:       gpu::GraphicsRootPush::size,
            action_type:     vk::IndirectCommandsTokenTypeEXT.DRAW_INDEXED_EXT,
            action_offset:   16,
            stride:          gpu::GeneratedDrawIndexedRecord::size,
        )!;
    }

    char[] cache_data = initial_cache_data;
    if (!cache_blob_compatible(state, cache_data)) cache_data = {};
    create_driver_pipeline_cache(state, cache_data)!;

    state.pipeline_cache.entries = mem::new_array(PipelineCacheEntry, capacity);
    state.pipeline_cache.bucket_heads = mem::new_array(uint, next_pow2(capacity * 2));
    create_shader_store(state, capacity)!;
}

fn usz? vk_get_pipeline_cache_size(gpu::Device* device) @private {
    VkDeviceState* state = (VkDeviceState*)gpu::device_backend_state_ptr(device);
    sz size = 0;
    check_backend_result(
        state:     state,
        operation: "get_pipeline_cache_size",
        result:    vk::get_pipeline_cache_data(state.device, state.driver_cache, &size, null),
    )!;
    return (usz)size;
}

fn usz? vk_get_pipeline_cache_data(gpu::Device* device, char[] out) @private {
    VkDeviceState* state = (VkDeviceState*)gpu::device_backend_state_ptr(device);
    sz required = 0;
    check_backend_result(
        state:     state,
        operation: "get_pipeline_cache_data",
        result:    vk::get_pipeline_cache_data(state.device, state.driver_cache, &required, null),
    )!;
    if ((usz)required > out.len) {
        return report_contract_failure(
            state:               state,
            operation:           "get_pipeline_cache_data",
            public_fault:        gpu::INVALID_ARGUMENT,
            resource_kind:       gpu::DebugResourceKind.DEVICE,
            has_public_identity: false,
            index:               0,
            generation:          0,
            debug_name:          null,
            rejected_field:      "out",
            invariant:           "output buffer must fit the pipeline cache data",
            backend_text:        null,
        )~;
    }
    sz written = required;
    check_pipeline_cache_data_backend_result(
        state:     state,
        operation: "get_pipeline_cache_data",
        result:    vk::get_pipeline_cache_data(state.device, state.driver_cache, &written, out.ptr),
    )!;
    return (usz)written;
}

<* Teardown counterpart; safe on a partial state. Destroys surviving entries
   regardless of refcount — the device owns everything at this point. *>
fn void destroy_pipeline_shared(VkDeviceState* state) {
    if (state.pipeline_cache.entries.ptr != null) {
        foreach (&entry : state.pipeline_cache.entries[:state.pipeline_cache.next_new]) {
            if (entry.used) {
                vk::destroy_pipeline(state.device, entry.pipeline, null);
            }
        }
        mem::free(state.pipeline_cache.entries.ptr);
        state.pipeline_cache.entries = {};
    }
    if (state.pipeline_cache.bucket_heads.ptr != null) {
        mem::free(state.pipeline_cache.bucket_heads.ptr);
        state.pipeline_cache.bucket_heads = {};
    }
    destroy_shader_store(state);
    if (state.generated_dispatch_layout != null) {
        state.device_dispatch.generated_work.destroy_layout(
            state.device,
            state.generated_dispatch_layout,
            null,
        );
        state.generated_dispatch_layout = (vk::IndirectCommandsLayoutEXT)null;
    }
    if (state.compute_layout != null) {
        vk::destroy_pipeline_layout(state.device, state.compute_layout, null);
        state.compute_layout = (vk::PipelineLayout)null;
    }
    if (state.generated_indexed_draw_layout != null) {
        state.device_dispatch.generated_work.destroy_layout(
            state.device,
            state.generated_indexed_draw_layout,
            null,
        );
        state.generated_indexed_draw_layout = (vk::IndirectCommandsLayoutEXT)null;
    }
    if (state.generated_draw_layout != null) {
        state.device_dispatch.generated_work.destroy_layout(
            state.device,
            state.generated_draw_layout,
            null,
        );
        state.generated_draw_layout = (vk::IndirectCommandsLayoutEXT)null;
    }
    if (state.graphics_layout != null) {
        vk::destroy_pipeline_layout(state.device, state.graphics_layout, null);
        state.graphics_layout = (vk::PipelineLayout)null;
    }
    if (state.driver_cache != null) {
        vk::destroy_pipeline_cache(state.device, state.driver_cache, null);
        state.driver_cache = (vk::PipelineCache)null;
    }
}
