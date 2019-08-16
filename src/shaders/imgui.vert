#version 450

layout(set = 0, binding = 0) uniform Transform {
    vec2 u_Translate;
    vec2 u_Scale;
};

layout(location = 0) in vec2 a_Pos;
layout(location = 1) in vec2 a_UV;
layout(location = 2) in uint a_Color;

layout(location = 0) out vec2 v_UV;
layout(location = 1) out vec4 v_Color;

void main() {
    v_UV = a_UV;
    v_Color = vec4(a_Color & 0xFF,
                   (a_Color >> 8) & 0xFF,
                   (a_Color >> 16) & 0xFF,
                   (a_Color >> 24) & 0xFF) / 255.0;
    gl_Position = vec4(a_Pos.xy * u_Scale + u_Translate, 0.0, 1.0);
}
