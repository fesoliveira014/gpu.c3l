module gpu::internal @private;

import std::hash::fnv64a;

const uint SPIRV_MAGIC @private                    = 0x07230203;
const usz SPIRV_HEADER_SIZE @private               = 20;
const usz SHADER_ENTRY_POINT_CAP @private          = gpu::MAX_SHADER_ENTRY_POINT_BYTES + 1;
const uint SPIRV_OPCODE_MASK @private              = 0xFFFF;
const uint SPIRV_WORD_COUNT_SHIFT @private         = 16;
const uint SPIRV_OP_CAPABILITY @private            = 17;
const uint SPIRV_CAPABILITY_CLIP_DISTANCE @private = 32;
const uint SPIRV_CAPABILITY_SAMPLE_RATE_SHADING @private = 35;

enum ShaderRole : int {
    COMPUTE,
    VERTEX,
    FRAGMENT,
    RAY_GENERATION,
    MISS,
    CLOSEST_HIT,
    ANY_HIT,
    INTERSECTION,
    CALLABLE,
    TASK,
    MESH,
}

fn ulong shader_identity_hash(char[] spirv) @private {
    ulong digest = fnv64a::hash(spirv);
    return digest == 0 ? 1 : digest;
}

<* Whether the module's leading capability instructions name the capability. Pure. *>
fn bool spirv_declares_capability(uint[] words, uint capability) @private {
    usz index = SPIRV_HEADER_SIZE / uint::size;
    while (index + 1 < words.len) {
        uint opcode = words[index] & SPIRV_OPCODE_MASK;
        uint word_count = words[index] >> SPIRV_WORD_COUNT_SHIFT;
        if (opcode != SPIRV_OP_CAPABILITY || word_count < 2) return false;
        if (words[index + 1] == capability) return true;
        index += word_count;
    }
    return false;
}

<* Whether the name is null, which selects `main`, or between 1 and
   MAX_SHADER_ENTRY_POINT_BYTES bytes long. *>
fn bool shader_entry_point_valid(ZString entry_point) @private {
    if (entry_point == null) return true;
    usz length = (usz)entry_point.len();
    return length > 0 && length <= gpu::MAX_SHADER_ENTRY_POINT_BYTES;
}

<* Whether the fraction lies in [0, 1]; NaN and infinities are not. *>
fn bool min_sample_shading_valid(float fraction) @private {
    return fraction >= 0.0f && fraction <= 1.0f;
}

fn bool polygon_mode_valid(gpu::PolygonMode mode) @private {
    switch (mode) {
        case gpu::PolygonMode.FILL:
        case gpu::PolygonMode.LINE:
            return true;
        default:
            return false;
    }
}

<* Whether every entry is a defined, color-renderable format. *>
fn bool color_formats_valid(gpu::Format[] formats) @private {
    foreach (format : formats) {
        if (format_is_depth_stencil(format) || !texture_format_in_backend_profile(format)) {
            return false;
        }
    }
    return true;
}

fn bool depth_format_valid(gpu::Format format) @private {
    return format == gpu::Format.UNDEFINED || format_is_depth_stencil(format);
}

<* Whether a hit-group array holds only defined kinds with the intersection
   shader the kind requires. *>
fn bool ray_tracing_hit_groups_valid(gpu::RayTracingHitGroupDesc[] groups) @private {
    foreach (&group : groups) {
        switch (group.kind) {
            case gpu::RayTracingHitGroupKind.TRIANGLES:
                if (group.intersection_shader != null) return false;
            case gpu::RayTracingHitGroupKind.PROCEDURAL:
                if (group.intersection_shader == null) return false;
            default:
                return false;
        }
    }
    return true;
}

<* Whether a binding-table region is the canonical empty value with stride 0,
   or a nonempty span. *>
fn bool ray_tracing_shader_binding_table_region_valid(
    gpu::RayTracingShaderBindingTableRegion* region,
) @private {
    gpu::GpuSpan span = region.span;
    bool empty = span.owner == 0
        && span.index == 0
        && span.generation == 0
        && span.offset == 0
        && span.size == 0;
    return empty ? region.stride == 0 : span.size > 0;
}

fn bool ray_tracing_shader_binding_table_valid(
    gpu::RayTracingShaderBindingTable* table,
) @private {
    return ray_tracing_shader_binding_table_region_valid(&table.ray_generation)
        && ray_tracing_shader_binding_table_region_valid(&table.miss)
        && ray_tracing_shader_binding_table_region_valid(&table.hit)
        && ray_tracing_shader_binding_table_region_valid(&table.callable);
}

<* Whether the descriptor has ray generation shaders, no null slice with a
   length, and counts that fit `uint` per slice, as group totals and as stage
   totals. *>
fn bool ray_tracing_pipeline_desc_valid(gpu::RayTracingPipelineDesc* desc) @private {
    if (desc.ray_generation_shaders.len == 0) return false;
    if ((desc.ray_generation_shaders.ptr == null)
        || (desc.miss_shaders.len > 0 && desc.miss_shaders.ptr == null)
        || (desc.hit_groups.len > 0 && desc.hit_groups.ptr == null)
        || (desc.callable_shaders.len > 0 && desc.callable_shaders.ptr == null)) {
        return false;
    }
    if (desc.ray_generation_shaders.len > uint::max
        || desc.miss_shaders.len > uint::max
        || desc.hit_groups.len > uint::max
        || desc.callable_shaders.len > uint::max) {
        return false;
    }
    ulong groups = (ulong)desc.ray_generation_shaders.len
        + desc.miss_shaders.len
        + desc.hit_groups.len
        + desc.callable_shaders.len;
    if (groups > uint::max) return false;
    ulong stages = (ulong)desc.ray_generation_shaders.len
        + desc.miss_shaders.len
        + desc.callable_shaders.len;
    foreach (&group : desc.hit_groups) {
        if (group.closest_hit_shader != null) stages++;
        if (group.any_hit_shader != null) stages++;
        if (group.intersection_shader != null) stages++;
    }
    return stages <= uint::max;
}

<* Whether every shader in the descriptor has a valid entry-point name. Assumes
   `ray_tracing_pipeline_desc_valid`. *>
fn bool ray_tracing_entry_points_valid(gpu::RayTracingPipelineDesc* desc) @private {
    foreach (&shader : desc.ray_generation_shaders) {
        if (!shader_entry_point_valid(shader.entry_point)) return false;
    }
    foreach (&shader : desc.miss_shaders) {
        if (!shader_entry_point_valid(shader.entry_point)) return false;
    }
    foreach (&shader : desc.callable_shaders) {
        if (!shader_entry_point_valid(shader.entry_point)) return false;
    }
    foreach (&group : desc.hit_groups) {
        if (group.closest_hit_shader != null
            && !shader_entry_point_valid(group.closest_hit_shader.entry_point)) return false;
        if (group.any_hit_shader != null
            && !shader_entry_point_valid(group.any_hit_shader.entry_point)) return false;
        if (group.intersection_shader != null
            && !shader_entry_point_valid(group.intersection_shader.entry_point)) return false;
    }
    return true;
}
