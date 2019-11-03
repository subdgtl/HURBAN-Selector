#version 450

layout(set = 1, binding = 0, std140) uniform Shading {
    vec4 u_edge_color_and_face_alpha;
    uint u_shading_mode;
};

layout(set = 2, binding = 0) uniform texture2D u_matcap_texture;
layout(set = 2, binding = 1) uniform sampler u_matcap_sampler;

layout(location = 0) in vec2 v_matcap_tex_coords;
layout(location = 1) in vec3 v_barycentric;

layout(location = 0) out vec4 f_color;

const uint SHADING_MODE_SHADED = 0x01;
const uint SHADING_MODE_EDGES = 0x02;

const float EDGE_THICKNESS_MIN = 0.75;
const float EDGE_THICKNESS_MAX = 1.00;

void main() {
    vec3 edge_color = u_edge_color_and_face_alpha.rgb;
    float face_alpha = u_edge_color_and_face_alpha.a;

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

    vec4 matcap_color = texture(sampler2D(u_matcap_texture, u_matcap_sampler), v_matcap_tex_coords);

    bool shaded_mode_enabled = bool(u_shading_mode & SHADING_MODE_SHADED);
    bool edges_mode_enabled = bool(u_shading_mode & SHADING_MODE_EDGES);

    if (shaded_mode_enabled && edges_mode_enabled) {
        f_color = mix(vec4(matcap_color.rgb, face_alpha), vec4(edge_color, 1), edge_alpha);
    } else if (shaded_mode_enabled) {
        f_color = vec4(matcap_color.rgb, face_alpha);
    } else if (edges_mode_enabled) {
        f_color = vec4(edge_color, edge_alpha);
    }
}
