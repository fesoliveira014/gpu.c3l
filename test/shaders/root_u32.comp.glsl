#version 460

layout(local_size_x = 1) in;

layout(push_constant) uniform Push {
    uint root_gpu;
} pc;

void main() {
    if (pc.root_gpu == 1) memoryBarrier();
}
