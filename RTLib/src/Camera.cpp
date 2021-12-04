#include <RTLib/ext/Camera.h>
void rtlib::ext::Camera::getUVW(float3& u,float3& v,float3& w)const noexcept{
    w = m_LookAt - m_Eye;
    //front
    //u = rtlib::normalize(rtlib::cross(w,m_Vup));
    u = rtlib::normalize(rtlib::cross(m_Vup,w));
    v = rtlib::normalize(rtlib::cross(w,u));
    auto vlen = rtlib::length(w)*std::tanf(RTLIB_M_PI*m_FovY/360.0f);
    auto ulen = vlen * m_Aspect;
    u*=ulen;
    v*=vlen;
}
std::tuple<float3,float3,float3> rtlib::ext::Camera::getUVW()const noexcept{
    std::tuple<float3,float3,float3> uvw;
    this->getUVW(std::get<0>(uvw),std::get<1>(uvw),std::get<2>(uvw));
    return uvw;
}