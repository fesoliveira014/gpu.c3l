// Reflection-validation fixture: a push_constant block larger than both
// root-push ABIs (8B compute / 16B graphics). pipeline creation must fault
// SHADER_INVALID.
#version 460

layout(local_size_x = 1) in;

layout(push_constant) uniform OversizedPush {
    vec4 a;
    vec4 b;
} pc;

void main() {
    if (pc.a.x > 2.0 && pc.b.x > 2.0) return;
}
