#version 450

layout(set = 0, binding = 0, std140) uniform ShadowPass {
    mat4 u_light_space_matrix;
};

layout(location = 0) in vec4 a_position;

void main() {
    gl_Position = u_light_space_matrix * a_position;

    // These are manual OpenGL to Vulkan NDC space corrections. We can't just
    // use the correction matrix as our projection matrix for shadow casting is
    // ortographic.
    gl_Position.y = -gl_Position.y;
    gl_Position.z = (gl_Position.z + gl_Position.w) / 2.0;
}
