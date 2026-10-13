#version 460

layout(location = 0) out vec4 out_color;

void main() {
    out_color = vec4(float(bitCount(gl_SampleMaskIn[0])) / 255.0, 0.0, 0.0, 1.0);
}
