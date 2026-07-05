#version 460
#include "generated/shader_abi.glsl"
#include "generated/gpu_driven_abi.glsl"

layout(local_size_x = BUILD_WORKGROUP) in;

layout(buffer_reference, std430) readonly buffer Instances { Instance items[]; };
layout(buffer_reference, std430) writeonly buffer Args { DrawIndexedIndirectCommand cmds[]; };
layout(buffer_reference, std430) writeonly buffer CountBuf { uint value; };
layout(push_constant) uniform Push { RootPush pc; };

// Slot i drives instance i (gl_DrawID == i in the vertex stage); culled
// instances draw zero instances rather than compacting.
void main() {
    BuildRoot root = BuildRoot(pc.root_gpu);
    uint i = gl_GlobalInvocationID.x;
    if (i >= root.instance_count) return;

    Instance inst = Instances(root.instances_gpu).items[i];
    vec2 spotlight = vec2(cos(root.time) * 0.5, sin(root.time * 0.7) * 0.5);
    bool visible = distance(inst.pos, spotlight) < 0.55;

    Args(root.args_gpu).cmds[i] = DrawIndexedIndirectCommand(6u, visible ? 1u : 0u, 0u, 0, 0u);
    if (i == 0u) CountBuf(root.count_gpu).value = root.instance_count;
}
