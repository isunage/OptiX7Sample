#include "..\include\TestLib\RTCamera.h"

auto test::RTPinholeCamera::GetViewMatrix() const noexcept -> rtlib::Matrix3x3 
{
    float3 w = m_LookAt - m_Eye;
    //front
    //u = rtlib::normalize(rtlib::cross(w,m_Vup));
    float3 u = rtlib::normalize(rtlib::cross(m_Vup, w));
    float3 v = rtlib::normalize(rtlib::cross(w, u));
    auto vlen = rtlib::length(w) * std::tanf(RTLIB_M_PI * m_FovY / 360.0f);
    auto ulen = vlen * m_Aspect;
    u *= ulen;
    v *= vlen;
    return rtlib::Matrix3x3(u,v,w);
}
