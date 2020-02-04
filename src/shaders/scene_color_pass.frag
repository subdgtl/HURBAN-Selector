#version 450

layout(set = 1, binding = 0) uniform sampler u_sampler;
layout(set = 1, binding = 1) uniform samplerShadow u_shadow_sampler;
layout(set = 2, binding = 0) uniform texture2D u_matcap_texture;
layout(set = 3, binding = 0) uniform texture2D u_shadow_map_texture;

layout(set = 4, binding = 0, std140) uniform ColorPass {
    vec4 u_shading_mode_flat_color;
    vec3 u_shading_mode_edges_color;
    uint u_shading_mode;
    bool u_collect_shadows;
};

layout(location = 0) in vec2 v_matcap_tex_coords;
layout(location = 1) in vec3 v_barycentric;
layout(location = 2) in vec4 v_frag_pos_light_space;

layout(location = 0) out vec4 f_color;

const uint SHADING_MODE_FLAT = 0x01;
const uint SHADING_MODE_SHADED = 0x02;
const uint SHADING_MODE_EDGES = 0x04;

const float EDGE_THICKNESS_MIN = 0.75;
const float EDGE_THICKNESS_MAX = 1.00;

const float SHADOW_INTENSITY = 0.3;

void main() {

    // -- Compute edge color --

    // Find which edge this pixel is the closest to by finding the
    // barycentric coordinate of the farthest away vertex within this
    // triangle.
    float opposite_vertex = min(v_barycentric.x, min(v_barycentric.y, v_barycentric.z));
    float d_opposite_vertex = fwidth(opposite_vertex);

    // Dividing by the fragment derivative will both sharpen the edge
    // (the derivative becomes larger once we get further away from
    // the edge) and keep the line of constant width in screenspace.
    float thickness = opposite_vertex / d_opposite_vertex;

    // Smoothen the alpha value so the lines are smooth even without
    // antialiasing.
    float edge_alpha = 1.0 - smoothstep(EDGE_THICKNESS_MIN, EDGE_THICKNESS_MAX, thickness);

    // -- Sample matcap color --

    vec4 matcap_color = texture(sampler2D(u_matcap_texture, u_sampler), v_matcap_tex_coords);

    // -- Compute shadow --

    // Since this is not a glsl builtin, we have to perform perspective divide
    // and remap depth from [-1, 1] to [0, 1] ourselves
    vec3 frag_pos_light_space = (v_frag_pos_light_space.xyz / v_frag_pos_light_space.w) * 0.5 + 0.5;

    float shadow = 0.0;
    vec2 shadow_map_texel_size = 1.0 / textureSize(sampler2DShadow(u_shadow_map_texture,
                                                                   u_shadow_sampler), 0);

    // Protect against sampling beyong depth 1.0 (the light's far plane).
    if (u_collect_shadows && frag_pos_light_space.z <= 1.0) {
        // Use Percentage Closer Filtering (PCF) - sample the depth texture sixteen
        // times, each time between texels.
        // https://developer.nvidia.com/gpugems/gpugems/part-ii-lighting-and-shadows/chapter-11-shadow-map-antialiasing
        for (float x = -1.5; x <= 1.5; x += 1.0) {
            for (float y = -1.5; y <= 1.5; y += 1.0) {
                vec2 offset = vec2(x, y) * shadow_map_texel_size;
                vec3 lookup_coords = vec3(frag_pos_light_space.xy + offset, frag_pos_light_space.z);

                // Protect against oversampling. If we were to sample outside the
                // shadow map, let's not accumulate any shadow.
                if (lookup_coords.x >= 0.0
                    && lookup_coords.y <= 1.0
                    && lookup_coords.y >= 0.0
                    && lookup_coords.y <= 1.0)
                {
                    // Accumulate shadow if the depth comparison succeeds
                    shadow += texture(sampler2DShadow(u_shadow_map_texture,
                                                      u_shadow_sampler), lookup_coords);
                }
            }
        }
    }
    shadow /= 16.0;

    // -- Mix colors --

    f_color = vec4(0);

    // Add flat color
    if (bool(u_shading_mode & SHADING_MODE_FLAT)) {
        f_color += u_shading_mode_flat_color;
    }

    // Alpha blend shaded color
    if (bool(u_shading_mode & SHADING_MODE_SHADED)) {
        f_color += mix(f_color, vec4(matcap_color.rgb, 1), matcap_color.a);
    }

    // Apply shadows
    f_color.rgb *= (1.0 - shadow * SHADOW_INTENSITY);
    f_color.a = mix(f_color.a, 1.0, shadow * SHADOW_INTENSITY);

    // Alpha blend edge color
    if (bool(u_shading_mode & SHADING_MODE_EDGES)) {
        f_color = mix(f_color, vec4(u_shading_mode_edges_color, 1), edge_alpha);
    }
}
