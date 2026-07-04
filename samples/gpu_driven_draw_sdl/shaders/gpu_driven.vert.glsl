#version 460
#extension GL_ARB_shader_draw_parameters : require
#include "generated/shader_abi.glsl"
#include "generated/gpu_driven_abi.glsl"

layout(buffer_reference, std430) readonly buffer Corners { vec2 items[]; };
layout(buffer_reference, std430) readonly buffer Instances { Instance items[]; };
layout(push_constant) uniform Push { GraphicsRootPush pc; };

layout(location = 0) out vec4 v_color;

// One multi-draw serves every quad; gl_DrawID picks this draw's instance
// record from the shared table.
void main() {
    DrawRoot root = DrawRoot(pc.vertex_root_gpu);
    Instance inst = Instances(root.instances_gpu).items[gl_DrawID];
    vec2 corner = Corners(root.corners_gpu).items[gl_VertexIndex];
    gl_Position = vec4(inst.pos + corner * inst.scale, 0.0, 1.0);
    v_color = inst.color;
}
