#version 460
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference : require

layout(set = 0, binding = 5, rgba8) uniform image2D bad_heap[];
layout(buffer_reference, std430) buffer Result { uint value; };
layout(push_constant) uniform Push { uint64_t root_gpu; } pc;
layout(local_size_x = 1) in;

void main() {
    if (pc.root_gpu == 0ul) return;
    Result(pc.root_gpu).value = uint(imageLoad(
        bad_heap[nonuniformEXT(0u)],
        ivec2(0)).x);
}
