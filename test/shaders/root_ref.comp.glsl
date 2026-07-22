#version 460
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require

layout(local_size_x = 1) in;
layout(buffer_reference, scalar) buffer RootReference {
    uint value;
};
layout(push_constant) uniform Push {
    RootReference root_gpu;
} pc;

void main() {
    if (pc.root_gpu.value == 0u) {
        return;
    }
}
