// Reflection-validation fixture: a graphics push block with a member between
// the two root addresses. Pipeline creation must fault SHADER_INVALID.
#version 460
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

layout(push_constant) uniform Push {
    uint64_t vertex_root_gpu;
    uint extra;
    uint64_t fragment_root_gpu;
} pc;

void main() {
    gl_Position = vec4(float(pc.extra), 0.0, 0.0, 1.0);
    if (pc.vertex_root_gpu == pc.fragment_root_gpu) gl_Position.y = 1.0;
}
