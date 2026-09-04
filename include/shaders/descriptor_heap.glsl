// gpu.c3l descriptor heap shader ABI.
//
// Set/binding convention (must match gpu/internal/vk/descriptor_heap.c3):
//   set 0, binding 0  sampled 2D images
//   set 0, binding 1  storage 2D images
//   set 0, binding 2  samplers
//   set 0, binding 3  sampled 3D images
//   set 0, binding 4  storage 3D images
//   set 0, binding 5  acceleration structures
//   set 0, binding 6  sampled cube images
//
// TextureIndex/SamplerIndex values are generation-free uints. Zero is invalid;
// live values encode the zero-based heap slot plus one.
//
// Layout convention: a texture must sit in the layout its usage implies when
// shaders access it — sampled-only textures in SHADER_READ, storage-capable
// textures in GENERAL.

#ifndef GPU_DESCRIPTOR_HEAP_GLSL
#define GPU_DESCRIPTOR_HEAP_GLSL

#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_shader_image_load_formatted : require

layout(set = 0, binding = 0) uniform texture2D gpu_texture_heap[];
layout(set = 0, binding = 1) uniform image2D gpu_storage_heap[];
layout(set = 0, binding = 2) uniform sampler gpu_sampler_heap[];
layout(set = 0, binding = 3) uniform texture3D gpu_texture_3d_heap[];
layout(set = 0, binding = 4) uniform image3D gpu_storage_3d_heap[];
layout(set = 0, binding = 6) uniform textureCube gpu_texture_cube_heap[];
// Aliased view of the sampler binding for depth-compare (shadow) access;
// SPIR-V samplers are untyped, so both views share binding 2.
layout(set = 0, binding = 2) uniform samplerShadow gpu_shadow_sampler_heap[];

#define GPU_HEAP_SLOT(index) ((index) - 1u)

// Explicit-LOD sampling: usable from compute, where derivatives don't exist.
vec4 sample_texture_2d(uint tex_index, uint smp_index, vec2 uv) {
    return textureLod(
        sampler2D(
            gpu_texture_heap[nonuniformEXT(GPU_HEAP_SLOT(tex_index))],
            gpu_sampler_heap[nonuniformEXT(GPU_HEAP_SLOT(smp_index))]),
        uv,
        0.0);
}

// Implicit-LOD sampling: fragment-stage use (derivatives drive mip selection).
vec4 sample_texture_2d_implicit(uint tex_index, uint smp_index, vec2 uv) {
    return texture(
        sampler2D(
            gpu_texture_heap[nonuniformEXT(GPU_HEAP_SLOT(tex_index))],
            gpu_sampler_heap[nonuniformEXT(GPU_HEAP_SLOT(smp_index))]),
        uv);
}

// Explicit-LOD sampling: usable from compute, where derivatives don't exist.
vec4 sample_texture_3d(uint tex_index, uint smp_index, vec3 uvw) {
    return textureLod(
        sampler3D(
            gpu_texture_3d_heap[nonuniformEXT(GPU_HEAP_SLOT(tex_index))],
            gpu_sampler_heap[nonuniformEXT(GPU_HEAP_SLOT(smp_index))]),
        uvw,
        0.0);
}

// Implicit-LOD sampling: fragment-stage use (derivatives drive mip selection).
vec4 sample_texture_3d_implicit(
    uint tex_index,
    uint smp_index,
    vec3 uvw
) {
    return texture(
        sampler3D(
            gpu_texture_3d_heap[nonuniformEXT(GPU_HEAP_SLOT(tex_index))],
            gpu_sampler_heap[nonuniformEXT(GPU_HEAP_SLOT(smp_index))]),
        uvw);
}

// Explicit-LOD cube sampling by direction: usable from compute.
vec4 sample_texture_cube_lod(
    uint tex_index,
    uint smp_index,
    vec3 dir,
    float lod
) {
    return textureLod(
        samplerCube(
            gpu_texture_cube_heap[nonuniformEXT(GPU_HEAP_SLOT(tex_index))],
            gpu_sampler_heap[nonuniformEXT(GPU_HEAP_SLOT(smp_index))]),
        dir,
        lod);
}

vec4 sample_texture_cube(uint tex_index, uint smp_index, vec3 dir) {
    return sample_texture_cube_lod(tex_index, smp_index, dir, 0.0);
}

// Implicit-LOD cube sampling: fragment-stage use.
vec4 sample_texture_cube_implicit(uint tex_index, uint smp_index, vec3 dir) {
    return texture(
        samplerCube(
            gpu_texture_cube_heap[nonuniformEXT(GPU_HEAP_SLOT(tex_index))],
            gpu_sampler_heap[nonuniformEXT(GPU_HEAP_SLOT(smp_index))]),
        dir);
}

// Depth-compare fetch: coord.xy samples, coord.z is the reference depth.
// Explicit LOD — safe in any stage. The sampler must be compare-enabled.
float sample_shadow_2d(uint tex_index, uint smp_index, vec3 coord) {
    return textureLod(
        sampler2DShadow(
            gpu_texture_heap[nonuniformEXT(GPU_HEAP_SLOT(tex_index))],
            gpu_shadow_sampler_heap[nonuniformEXT(GPU_HEAP_SLOT(smp_index))]),
        coord,
        0.0);
}

vec4 load_storage_texture(uint tex_index, ivec2 coord) {
    return imageLoad(gpu_storage_heap[nonuniformEXT(GPU_HEAP_SLOT(tex_index))], coord);
}

void store_storage_texture(uint tex_index, ivec2 coord, vec4 value) {
    imageStore(gpu_storage_heap[nonuniformEXT(GPU_HEAP_SLOT(tex_index))], coord, value);
}

vec4 load_storage_texture_3d(uint tex_index, ivec3 coord) {
    return imageLoad(
        gpu_storage_3d_heap[nonuniformEXT(GPU_HEAP_SLOT(tex_index))],
        coord);
}

void store_storage_texture_3d(uint tex_index, ivec3 coord, vec4 value) {
    imageStore(
        gpu_storage_3d_heap[nonuniformEXT(GPU_HEAP_SLOT(tex_index))],
        coord,
        value);
}

#endif
