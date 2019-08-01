#version 450

layout(set = 0, binding = 0) uniform Matrix {
    mat4 u_matrix;
};

layout(location = 0) in vec3 a_pos;

layout(location = 0) out vec3 v_pos_norm;

void main() {
    v_pos_norm = a_pos / length(a_pos) * 0.5 + 0.5;
    gl_Position = u_matrix * vec4(a_pos, 1.0);
}
