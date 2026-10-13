// gpu.c3l multisample descriptor heap access.
//
// Include this file in addition to descriptor_heap.glsl in stages that read
// multisample textures. A multisample texture created with the sampled usage is
// published at set 0, binding 0 like any sampled view; these aliases type the
// access by sample. The aliases sit in their own include because declaring them
// adds an unused variable to every module that includes them.
//
// A multisample texture is read with texelFetch only: it has no sampler, no
// filtering and no mip levels. A depth texture returns its depth in .x.

#ifndef GPU_MULTISAMPLE_HEAP_GLSL
#define GPU_MULTISAMPLE_HEAP_GLSL

#include "descriptor_heap.glsl"

layout(set = 0, binding = 0) uniform texture2DMS gpu_texture_ms_heap[];
// Integer view of the same binding for unsigned-integer color formats.
layout(set = 0, binding = 0) uniform utexture2DMS gpu_utexture_ms_heap[];

// One sample of a float, normalized or depth texel.
vec4 gpu_fetch_ms(uint tex_index, ivec2 texel, int sample_index) {
    return texelFetch(
        gpu_texture_ms_heap[nonuniformEXT(GPU_HEAP_SLOT(tex_index))],
        texel,
        sample_index);
}

// One sample of an unsigned-integer texel; the first component only.
uint gpu_fetch_ms_uint(uint tex_index, ivec2 texel, int sample_index) {
    return texelFetch(
        gpu_utexture_ms_heap[nonuniformEXT(GPU_HEAP_SLOT(tex_index))],
        texel,
        sample_index).x;
}

#endif
