#version 450

layout(set = 0, binding = 0) uniform Transform {
    vec2 u_translate;
    vec2 u_scale;
};

layout(location = 0) in vec2 a_pos;
layout(location = 1) in vec2 a_tex_coords;
layout(location = 2) in uint a_color;

layout(location = 0) out vec2 v_tex_coords;
layout(location = 1) out vec4 v_color;

void main() {
    v_tex_coords = a_tex_coords;
    v_color = vec4(a_color & 0xFF,
                   (a_color >> 8) & 0xFF,
                   (a_color >> 16) & 0xFF,
                   (a_color >> 24) & 0xFF) / 255.0;
    gl_Position = vec4(a_pos.xy * u_scale + u_translate, 0.0, 1.0);
}
