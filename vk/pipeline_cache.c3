module gpu::vk;

import gpu;
import vk;
import vma;
import std::thread;
import std::core::mem;
import std::hash::fnv64a;

// Descriptor-keyed pipeline dedup cache: entries own the vk::Pipeline,
// refcounted by the slots that alias them; layouts are device-owned and
// shared. Monomorphic table, same c3c 0.8.0 rationale as PipelineTable.

const uint PIPELINE_ENTRY_NONE = uint::max;
const uint COMPUTE_LAYOUT_CAP  = 32;

enum PipelineKind : int {
    COMPUTE,
    GRAPHICS,
}

// Canonical packed key over the immutable pipeline state. Field order keeps
// the struct padding-free; zero-init plus field-wise assignment keeps the
// byte view deterministic for hashing. Dynamic-able state (viewport, scissor,
// cull, front face, depth test/write/compare) and debug_name stay out.
struct PipelineKey {
    ulong vertex_shader;
    ulong fragment_shader;
    uint  color_format_count;
    uint  push_constant_size;
    int[gpu::MAX_COLOR_ATTACHMENTS] color_formats;
    int   depth_format;
    int   topology;
    int   polygon_mode;
    int   blend_enable;
    int[6] blend;
    int   kind;
    int   reserved;
    float bias_constant;
    float bias_slope;
    float bias_clamp;
    int   reserved2;
}
$assert PipelineKey::size == 120;

struct PipelineCacheEntry {
    ulong              hash;
    PipelineKey        key;
    vk::Pipeline       pipeline;
    vk::PipelineLayout layout;
    uint               refcount;
    uint               next_free;
    bool               used;
}

struct PipelineCacheTable {
    PipelineCacheEntry[] entries;
    uint                 free_head;
    uint                 next_new;
}

struct ComputeLayoutEntry {
    uint               push_size;
    vk::PipelineLayout layout;
    bool               used;
}

fn PipelineKey build_graphics_key(gpu::GraphicsPipelineDesc* desc) @private {
    PipelineKey key;
    key.kind               = (int)PipelineKind.GRAPHICS;
    key.vertex_shader      = (ulong)desc.vertex_shader;
    key.fragment_shader    = (ulong)desc.fragment_shader;
    key.topology           = (int)desc.topology;
    key.polygon_mode       = (int)desc.raster.polygon_mode;
    key.bias_constant      = desc.raster.depth_bias_constant;
    key.bias_slope         = desc.raster.depth_bias_slope;
    key.bias_clamp         = desc.raster.depth_bias_clamp;
    key.depth_format       = (int)desc.depth_format;
    key.color_format_count = (uint)desc.color_formats.len;
    foreach (i, format : desc.color_formats) key.color_formats[i] = (int)format;
    if (desc.blend.enable) {
        key.blend_enable = 1;
        key.blend = {
            (int)desc.blend.src_color, (int)desc.blend.dst_color, (int)desc.blend.color_op,
            (int)desc.blend.src_alpha, (int)desc.blend.dst_alpha, (int)desc.blend.alpha_op,
        };
    }
    return key;
}

fn PipelineKey build_compute_key(gpu::ComputePipelineDesc* desc) @private {
    PipelineKey key;
    key.kind               = (int)PipelineKind.COMPUTE;
    key.vertex_shader      = (ulong)desc.shader;
    key.push_constant_size = desc.push_constant_size;
    return key;
}

fn ulong hash_key(PipelineKey* key) @private {
    return fnv64a::hash(((char*)key)[:PipelineKey::size]);
}

fn bool key_equals(PipelineKey* a, PipelineKey* b) @private {
    return mem::equals(((char*)a)[:PipelineKey::size], ((char*)b)[:PipelineKey::size]);
}

<* Index of the live entry matching hash+key, or PIPELINE_ENTRY_NONE. The hash
   is a fast reject; equality is confirmed on the full key. *>
fn uint find_entry(PipelineCacheTable* cache, ulong hash, PipelineKey* key) {
    foreach (i, &entry : cache.entries[:cache.next_new]) {
        if (!entry.used || entry.hash != hash) continue;
        if (key_equals(&entry.key, key)) return (uint)i;
    }
    return PIPELINE_ENTRY_NONE;
}

fn uint? insert_entry(PipelineCacheTable* cache, PipelineCacheEntry value) {
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
    return idx;
}

<* Single release choke point: drop one alias, enqueue the backend pipeline
   for deferred release when the last alias goes. Layouts are shared and
   never destroyed here. *>
fn void release_entry(VkDeviceState* state, uint entry_index) {
    PipelineCacheEntry* entry = &state.pipeline_cache.entries[entry_index];
    if (--entry.refcount > 0) return;
    enqueue_deferred(state, DeferredKind.PIPELINE, retire_timeline_value(state), (void*)entry.pipeline, (vma::Allocation)null);
    entry.used = false;
    entry.next_free = state.pipeline_cache.free_head;
    state.pipeline_cache.free_head = entry_index + 1;
}

<* Live entry count, exposed for dedup accounting in tests. *>
fn uint pipeline_cache_entry_count(VkDeviceState* state) {
    uint count = 0;
    foreach (&entry : state.pipeline_cache.entries[:state.pipeline_cache.next_new]) {
        if (entry.used) count++;
    }
    return count;
}

<* Compute layout for `push_size`, created on first use. All compute layouts
   share the global heap set-0 layout and differ only in push range size. *>
fn vk::PipelineLayout? get_compute_layout(VkDeviceState* state, uint push_size) {
    state.resource_mutex.lock()!!;
    defer state.resource_mutex.unlock();
    foreach (&entry : state.compute_layouts[:COMPUTE_LAYOUT_CAP]) {
        if (entry.used && entry.push_size == push_size) return entry.layout;
    }
    foreach (&entry : state.compute_layouts[:COMPUTE_LAYOUT_CAP]) {
        if (entry.used) continue;
        vk::PushConstantRange range = {
            .stage_flags = vk::ShaderStageFlagBits.COMPUTE_BIT,
            .offset      = 0,
            .size        = push_size,
        };
        vk::PipelineLayoutCreateInfo layout_info = vk::pipeline_layout_create_info()
            .set_set_layouts((&state.heap.set_layout)[:1])
            .set_push_constant_ranges((&range)[:1]);
        vk::PipelineLayout layout;
        check(vk::create_pipeline_layout(state.device, &layout_info, null, &layout))!;
        set_object_name(
            state:         state,
            object_type:   vk::ObjectType.PIPELINE_LAYOUT,
            object_handle: (ulong)layout,
            name:          "gpu_compute_layout",
        );
        *entry = { .push_size = push_size, .layout = layout, .used = true };
        return layout;
    }
    return gpu::SLOT_TABLE_FULL~;
}

<* Shared graphics layout, cache entry storage, and the driver VkPipelineCache.
   Requires the descriptor heap (set layout) to exist. *>
fn void? create_pipeline_shared(VkDeviceState* state, char[] initial_cache_data) {
    uint push_stages = (uint)vk::ShaderStageFlagBits.VERTEX_BIT | (uint)vk::ShaderStageFlagBits.FRAGMENT_BIT;
    vk::PushConstantRange range = {
        .stage_flags = (vk::ShaderStageFlags)push_stages,
        .offset      = 0,
        .size        = GRAPHICS_ROOT_PUSH_SIZE,
    };
    vk::PipelineLayoutCreateInfo layout_info = vk::pipeline_layout_create_info()
        .set_set_layouts((&state.heap.set_layout)[:1])
        .set_push_constant_ranges((&range)[:1]);
    check(vk::create_pipeline_layout(state.device, &layout_info, null, &state.graphics_layout))!;
    set_object_name(
        state:         state,
        object_type:   vk::ObjectType.PIPELINE_LAYOUT,
        object_handle: (ulong)state.graphics_layout,
        name:          "gpu_graphics_layout",
    );

    // The driver validates the blob header and ignores incompatible data.
    vk::PipelineCacheCreateInfo cache_info = vk::pipeline_cache_create_info();
    cache_info.initial_data_size = (sz)initial_cache_data.len;
    cache_info.p_initial_data    = initial_cache_data.ptr;
    check(vk::create_pipeline_cache(state.device, &cache_info, null, &state.driver_cache))!;

    state.pipeline_cache.entries = mem::new_array(PipelineCacheEntry, gpu::MAX_PIPELINES);
}

fn usz? vk_get_pipeline_cache_size(gpu::Device* device) @private {
    VkDeviceState* state = (VkDeviceState*)device.backend_state;
    sz size = 0;
    check(vk::get_pipeline_cache_data(state.device, state.driver_cache, &size, null))!;
    return (usz)size;
}

fn usz? vk_get_pipeline_cache_data(gpu::Device* device, char[] out) @private {
    VkDeviceState* state = (VkDeviceState*)device.backend_state;
    sz required = 0;
    check(vk::get_pipeline_cache_data(state.device, state.driver_cache, &required, null))!;
    if ((usz)required > out.len) return gpu::INVALID_ARGUMENT~;
    sz written = required;
    check(vk::get_pipeline_cache_data(state.device, state.driver_cache, &written, out.ptr))!;
    return (usz)written;
}

<* Teardown counterpart; safe on a partial state. Destroys surviving entries
   regardless of refcount — the device owns everything at this point. *>
fn void destroy_pipeline_shared(VkDeviceState* state) {
    if (state.pipeline_cache.entries.ptr != null) {
        foreach (&entry : state.pipeline_cache.entries[:state.pipeline_cache.next_new]) {
            if (entry.used) vk::destroy_pipeline(state.device, entry.pipeline, null);
        }
        mem::free(state.pipeline_cache.entries.ptr);
        state.pipeline_cache.entries = {};
    }
    foreach (&entry : state.compute_layouts[:COMPUTE_LAYOUT_CAP]) {
        if (entry.used) {
            vk::destroy_pipeline_layout(state.device, entry.layout, null);
            entry.used = false;
        }
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
