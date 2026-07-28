#version 460

layout(local_size_x = 1) in;
layout(set = 0, binding = 2) uniform sampler heap_sampler;
layout(set = 0, binding = 4) uniform texture3D wrong_storage_type;

void main() {
    if (textureLod(
        sampler3D(wrong_storage_type, heap_sampler),
        vec3(0.5),
        0.0).x > 2.0) return;
}
