#version 460
layout(local_size_x = 1) in;
layout(push_constant) uniform Push {
    uint low;
    uint high;
} pc;

void main() {
    if (pc.low == 0u && pc.high == 0u) {
        return;
    }
}
