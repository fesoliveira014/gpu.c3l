#version 460

#extension GL_EXT_samplerless_texture_functions : require

layout(local_size_x = 1) in;
layout(set = 0, binding = 5) uniform texture2D unknown_binding;

void main() {
    if (textureSize(unknown_binding, 0).x > 2) return;
}
