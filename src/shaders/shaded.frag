#version 450

layout(location = 0) in vec3 v_pos_norm;

layout(location = 0) out vec4 o_color;

void main() {
    o_color = vec4(v_pos_norm, 1.0);
}
