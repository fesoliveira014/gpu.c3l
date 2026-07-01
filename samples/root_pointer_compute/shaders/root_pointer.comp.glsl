#version 460
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

layout(local_size_x = 64) in;

layout(buffer_reference, std430) readonly  buffer InBuf  { float v[]; };
layout(buffer_reference, std430) writeonly buffer OutBuf { float v[]; };
layout(buffer_reference, std430) buffer Root {
    uint64_t input_gpu;
    uint64_t output_gpu;
    uint count;
    uint _pad0;
    uint _pad1;
    uint _pad2;
};
layout(push_constant) uniform Push { uint64_t root_gpu; };

void main() {
    Root root = Root(root_gpu);
    uint i = gl_GlobalInvocationID.x;
    if (i < root.count) {
        OutBuf(root.output_gpu).v[i] = InBuf(root.input_gpu).v[i] * 2.0;
    }
}
