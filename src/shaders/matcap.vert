#version 450

layout(set = 0, binding = 0, std140) uniform GlobalMatrix {
    mat4 u_projection_matrix;
    mat4 u_view_matrix;
};

layout(location = 0) in vec4 a_position;
layout(location = 1) in vec4 a_normal;

layout(location = 0) out vec2 v_matcap_tex_coords;

float remap(float value, vec2 from, vec2 to) {
    return (value - from.x) / (from.y - from.x) * (to.y - to.x) + to.x;
}

void main() {
    // FIXME: @Optimization Should we assume it is normalized already?
    vec4 viewspace_normal = u_view_matrix * normalize(a_normal);

    v_matcap_tex_coords = vec2(remap(viewspace_normal.x, vec2(-1, 1), vec2(0, 1)),
                               remap(viewspace_normal.y, vec2(-1, 1), vec2(0, 1)));

    gl_Position = u_projection_matrix * u_view_matrix * a_position;
}
