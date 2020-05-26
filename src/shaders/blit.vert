#version 450

layout(location = 0) out vec2 v_tex_coord;

void main() {
    // Produce one CCW fullscreen triangle in WebGPU NDC coordinates.
    // https://gpuweb.github.io/gpuweb/#coordinate-systems
    switch (gl_VertexIndex % 3) {
    case 0:
        v_tex_coord = vec2(0, -1);
        gl_Position = vec4(-1, 3, 0, 1);
        break;
    case 1:
        v_tex_coord = vec2(0, 1);
        gl_Position = vec4(-1, -1, 0, 1);
        break;
    case 2:
        v_tex_coord = vec2(2, 1);
        gl_Position = vec4(3, -1, 0, 1);
        break;
    }
}
