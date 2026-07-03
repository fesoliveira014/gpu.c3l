#version 460
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_ARB_shader_draw_parameters : require

struct Instance {
    vec2 pos;
    float scale;
    float _pad;
    vec4 color;
};

layout(buffer_reference, std430) readonly buffer Corners { vec2 items[]; };
layout(buffer_reference, std430) readonly buffer Instances { Instance items[]; };
layout(buffer_reference, std430) readonly buffer Root {
    uint64_t corners_gpu;
    uint64_t instances_gpu;
};
layout(push_constant) uniform Push {
    uint64_t vertex_root_gpu;
    uint64_t fragment_root_gpu;
};

layout(location = 0) out vec4 v_color;

// One multi-draw serves every quad; gl_DrawID picks this draw's instance
// record from the shared table.
void main() {
    Root root = Root(vertex_root_gpu);
    Instance inst = Instances(root.instances_gpu).items[gl_DrawID];
    vec2 corner = Corners(root.corners_gpu).items[gl_VertexIndex];
    gl_Position = vec4(inst.pos + corner * inst.scale, 0.0, 1.0);
    v_color = inst.color;
}
