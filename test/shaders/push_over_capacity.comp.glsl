// Reflection-validation fixture: a compute push block of 144 bytes, past
// ROOT_PUSH_CAPACITY. Pipeline creation must fault SHADER_INVALID.
#version 460
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

layout(local_size_x = 1) in;

layout(push_constant) uniform Push {
    uint64_t root_gpu;
    uint _pad0;
    uint _pad1;
    vec4 rows[8];
} pc;

void main() {
    if (pc.root_gpu == 0ul && pc.rows[7].x > 2.0) return;
}
