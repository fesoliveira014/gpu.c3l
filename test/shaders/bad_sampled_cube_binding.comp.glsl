#version 460

layout(local_size_x = 1) in;
layout(set = 0, binding = 6, rgba8) uniform image2D wrong_cube_type;

void main() {
    imageStore(wrong_cube_type, ivec2(0), vec4(1.0));
}
