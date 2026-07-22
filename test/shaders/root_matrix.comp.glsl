#version 460

layout(local_size_x = 1) in;

layout(push_constant) uniform Push {
    mat2 root_gpu;
} pc;

void main() {
    if (pc.root_gpu[0][0] == 1.0) memoryBarrier();
}
