#version 460

layout(location = 0) out vec4 first_color;
layout(location = 1) out vec4 second_color;

void main() {
    first_color = vec4(1.0, 0.0, 0.0, 0.0);
    second_color = vec4(0.0, 1.0, 0.0, 0.5);
}
