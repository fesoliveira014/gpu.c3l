#version 460
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

layout(local_size_x = 1) in;

struct NestedRoot {
    uint64_t root_gpu;
};

layout(push_constant) uniform Push {
    NestedRoot root;
} pc;

void main() {
    if (pc.root.root_gpu == 1) memoryBarrier();
}
