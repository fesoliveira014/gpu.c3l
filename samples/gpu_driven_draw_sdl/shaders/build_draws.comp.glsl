#version 460
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#include "indirect_commands.glsl"

layout(local_size_x = 64) in;

struct Instance {
    vec2 pos;
    float scale;
    float _pad;
    vec4 color;
};

layout(buffer_reference, std430) readonly buffer Instances { Instance items[]; };
layout(buffer_reference, std430) writeonly buffer Args { DrawIndexedIndirectCommand cmds[]; };
layout(buffer_reference, std430) writeonly buffer CountBuf { uint value; };
layout(buffer_reference, std430) buffer Root {
    uint64_t instances_gpu;
    uint64_t args_gpu;
    uint64_t count_gpu;
    uint instance_count;
    float time;
    uint _pad0;
    uint _pad1;
};
layout(push_constant) uniform Push { uint64_t root_gpu; };

// Slot i drives instance i (gl_DrawID == i in the vertex stage); culled
// instances draw zero instances rather than compacting.
void main() {
    Root root = Root(root_gpu);
    uint i = gl_GlobalInvocationID.x;
    if (i >= root.instance_count) return;

    Instance inst = Instances(root.instances_gpu).items[i];
    vec2 spotlight = vec2(cos(root.time) * 0.5, sin(root.time * 0.7) * 0.5);
    bool visible = distance(inst.pos, spotlight) < 0.55;

    Args(root.args_gpu).cmds[i] = DrawIndexedIndirectCommand(6u, visible ? 1u : 0u, 0u, 0, 0u);
    if (i == 0u) CountBuf(root.count_gpu).value = root.instance_count;
}
