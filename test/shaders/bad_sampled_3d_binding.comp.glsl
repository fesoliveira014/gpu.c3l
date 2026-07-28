#version 460

layout(local_size_x = 1) in;
layout(set = 0, binding = 3, rgba8) uniform image3D wrong_sampled_type;

void main() {
    imageStore(wrong_sampled_type, ivec3(0), vec4(1.0));
}
