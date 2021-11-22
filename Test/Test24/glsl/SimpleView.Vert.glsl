#version 330 core
layout(location = 0) in vec3 vInPosition;
layout(location = 1) in vec3 vInNormal;
layout(location = 2) in vec2 vInTexCrd;
out vec3 vOutNormal;
out vec2 vOutTexCrd;
uniform mat4x4 uMatMVP;
uniform mat3x3 uMatNo;
void main()
{
    gl_Position = uMatMVP * vec4(vInPosition,1.0);
    vOutNormal  = uMatNo  * vInNormal;
    vOutTexCrd  = vInTexCrd;
}
