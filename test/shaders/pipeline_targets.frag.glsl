#version 460

layout(location = 0) out vec4 out_first_color;
layout(location = 1) out vec4 out_second_color;

void main() {
    out_first_color = vec4(1.0, 0.0, 0.0, 0.0);
    out_second_color = vec4(0.0, 1.0, 0.0, 0.5);
}
