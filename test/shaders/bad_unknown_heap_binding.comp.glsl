#version 460

layout(local_size_x = 1) in;
layout(set = 0, binding = 5) uniform sampler2D unknown_binding;

void main() {
    if (textureLod(unknown_binding, vec2(0.5), 0.0).x > 2.0) return;
}
