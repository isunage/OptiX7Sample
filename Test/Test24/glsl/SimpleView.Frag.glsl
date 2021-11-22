#version 330 core
/*in */
in vec3 vOutNormal;
in vec2 vOutTexCrd;
/*out*/
layout(location = 0) out vec4 fOutNormal;
layout(location = 1) out vec4 fOutTexCrd;
void main()
{
    fOutNormal = vec4(vec3(0.5f)+0.5f*vOutNormal,1.0f);
    fOutTexCrd = vec4(vOutTexCrd, 1.0f-0.5f*(vOutTexCrd.x+vOutTexCrd.y),1.0f);
}
