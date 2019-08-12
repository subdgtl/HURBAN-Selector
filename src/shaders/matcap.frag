#version 450

layout(set = 1, binding = 0) uniform texture2D u_matcap_texture;
layout(set = 1, binding = 1) uniform sampler u_matcap_sampler;

layout(location = 0) in vec2 v_matcap_tex_coords;

layout(location = 0) out vec4 o_color;

void main() {
    o_color = texture(sampler2D(u_matcap_texture, u_matcap_sampler), v_matcap_tex_coords);
}
