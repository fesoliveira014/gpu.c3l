#version 460
#include "generated/shader_abi.glsl"
#include "generated/bindless_abi.glsl"
#include "descriptor_heap.glsl"

layout(local_size_x = 4, local_size_y = 4, local_size_z = 1) in;

layout(buffer_reference, std430) writeonly buffer OutBuf { vec4 texels[]; };
layout(push_constant) uniform Push {
    uint64_t root_gpu;
} pc;

void main() {
    VolumeHeapRoot root = VolumeHeapRoot(pc.root_gpu);
    uvec3 p = gl_GlobalInvocationID;
    if (p.x >= root.width || p.y >= root.height || p.z >= root.depth) return;

    uint area = root.width * root.height;
    uint volume_index = p.z * area + p.y * root.width + p.x;
    vec4 volume_value = vec4(
        float(p.x) / 255.0,
        float(p.y) / 255.0,
        float(p.z) / 255.0,
        1.0);
    store_storage_texture_3d(
        root.texture_3d_index,
        ivec3(p),
        volume_value);
    memoryBarrierImage();
    OutBuf(root.output_gpu).texels[volume_index] =
        load_storage_texture_3d(root.texture_3d_index, ivec3(p));

    if (p.z == 0) {
        uint image_index = p.y * root.width + p.x;
        vec4 image_value = vec4(
            float(p.x) / 255.0,
            float(p.y) / 255.0,
            17.0 / 255.0,
            1.0);
        store_storage_texture(
            root.texture_2d_index,
            ivec2(p.xy),
            image_value);
        memoryBarrierImage();
        OutBuf(root.output_gpu).texels[root.width * root.height * root.depth
            + image_index] =
            load_storage_texture(root.texture_2d_index, ivec2(p.xy));
    }
}
