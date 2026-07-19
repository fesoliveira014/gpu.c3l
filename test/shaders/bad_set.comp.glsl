// Reflection-validation fixture: declares a binding in set 1, which the heap
// convention forbids. pipeline creation must fault SHADER_INVALID.
#version 460

layout(local_size_x = 1) in;

layout(set = 1, binding = 0) uniform sampler2D foreign_texture;

void main() {
    // Reference the binding so no tool strips it from the module.
    if (textureLod(foreign_texture, vec2(0.0), 0.0).x > 2.0) return;
}
