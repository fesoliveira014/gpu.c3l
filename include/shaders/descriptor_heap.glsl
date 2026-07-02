// gpu.c3l descriptor heap shader ABI.
//
// Set/binding convention (must match vk/descriptor_heap.c3):
//   set 0, binding 0  sampled images
//   set 0, binding 1  storage images
//   set 0, binding 2  samplers
//
// TextureIndex/SamplerIndex values are packed uints: slot in the low 16 bits,
// generation in the high 16. Pass them straight from material records; the
// helpers mask the slot. Both backend descriptor paths use identical GLSL.

#ifndef GPU_DESCRIPTOR_HEAP_GLSL
#define GPU_DESCRIPTOR_HEAP_GLSL

#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_shader_image_load_formatted : require

layout(set = 0, binding = 0) uniform texture2D gpu_texture_heap[];
layout(set = 0, binding = 1) uniform image2D gpu_storage_heap[];
layout(set = 0, binding = 2) uniform sampler gpu_sampler_heap[];

#define GPU_HEAP_SLOT_MASK 0xFFFFu

// Explicit-LOD sampling: usable from compute, where derivatives don't exist.
vec4 sample_texture_2d(uint tex_index, uint smp_index, vec2 uv) {
    return textureLod(
        sampler2D(
            gpu_texture_heap[nonuniformEXT(tex_index & GPU_HEAP_SLOT_MASK)],
            gpu_sampler_heap[nonuniformEXT(smp_index & GPU_HEAP_SLOT_MASK)]),
        uv,
        0.0);
}

vec4 load_storage_texture(uint tex_index, ivec2 coord) {
    return imageLoad(gpu_storage_heap[nonuniformEXT(tex_index & GPU_HEAP_SLOT_MASK)], coord);
}

void store_storage_texture(uint tex_index, ivec2 coord, vec4 value) {
    imageStore(gpu_storage_heap[nonuniformEXT(tex_index & GPU_HEAP_SLOT_MASK)], coord, value);
}

#endif
