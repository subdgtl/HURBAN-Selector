#version 450

layout(set = 0, binding = 0, std140) uniform BlitPass {
    uint u_blit_sampler;
};

layout(set = 1, binding = 0) uniform sampler u_sampler;
layout(set = 1, binding = 1) uniform sampler u_shadow_sampler;
layout(set = 2, binding = 0) uniform texture2D u_blit_texture;

layout(location = 0) in vec2 v_tex_coord;

layout(location = 0) out vec4 f_color;

const uint BLIT_SAMPLER_COLOR = 0;
const uint BLIT_SAMPLER_DEPTH = 1;

void main() {
    if (u_blit_sampler == BLIT_SAMPLER_COLOR) {
        f_color = texture(sampler2D(u_blit_texture, u_sampler), v_tex_coord);
    } else if (u_blit_sampler == BLIT_SAMPLER_DEPTH) {
        // float depth = texture(sampler2DShadow(u_blit_texture, u_shadow_sampler), vec3(v_tex_coord, 0));
        float depth = texture(sampler2D(u_blit_texture, u_sampler), v_tex_coord).r;
        f_color = vec4(vec3(depth), 1);
    }
}
