#version 450

layout(location = 0) out vec2 v_tex_coord;

void main() {
    // Produce one CCW fullscreen triangle. Note that while WebGPU now
    // has a freshly specified coordinate system for NDC (similar to
    // D3D12), our wgpu-rs has not caught up yet and still use
    // Vulkan's coordinate systems. Vulkan's NDC coordinate system is
    // right handed with the Y axis growing downwards, so CCW in
    // Vulkan is CW in OpenGL.
    // https://matthewwellings.com/blog/the-new-vulkan-coordinate-system/

    // FIXME: Fix coordinates in this shader once wgpu-rs uses the
    // coordinate systems as specified by WebGPU.

    switch (gl_VertexIndex % 3) {
    case 0:
        v_tex_coord = vec2(2, 0);
        gl_Position = vec4(3, -1, 0, 1);
        break;
    case 1:
        v_tex_coord = vec2(0, 0);
        gl_Position = vec4(-1, -1, 0, 1);
        break;
    case 2:
        v_tex_coord = vec2(0, 2);
        gl_Position = vec4(-1, 3, 0, 1);
        break;
    }
}
