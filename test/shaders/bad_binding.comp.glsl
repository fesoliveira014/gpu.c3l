// Reflection-validation fixture: set-0 binding 0 as a combined image sampler,
// where the convention requires a sampled image array. create_shader must
// fault SHADER_INVALID.
#version 460

layout(local_size_x = 1) in;

layout(set = 0, binding = 0) uniform sampler2D wrong_type;

void main() {
    if (textureLod(wrong_type, vec2(0.0), 0.0).x > 2.0) return;
}
