#ifndef RTLIB_CAMERA_H
#define RTLIB_CAMERA_H
#include <tuple>
#include "VectorFunction.h"
namespace rtlib{
    class Camera{
        float3 m_Eye;
        float3 m_LookAt;
        float3 m_Vup;
        float  m_FovY;
        float  m_Aspect;
    public:
        Camera()noexcept :m_Eye{}, m_LookAt{}, m_Vup{}, m_FovY{}, m_Aspect{}{}
        Camera(const float3& eye,
               const float3& lookAt, 
               const float3& vup, 
               const float   fovY, 
               const float   aspect)noexcept
        :m_Eye{ eye },
         m_LookAt{lookAt},
         m_Vup{vup},
         m_FovY{fovY},
         m_Aspect{aspect}{}
        //Direction
        inline float3 getDirection()const noexcept{
            return normalize(m_LookAt-m_Eye);
        }
        inline void   setDirection(const float3& direction)noexcept{
            auto len = length(m_LookAt-m_Eye);
            m_Eye += len*normalize(direction);
        }
        //Get And Set
        RTLIB_DECLARE_GET_AND_SET_BY_REFERENCE(Camera,float3,   Eye,   m_Eye);
        RTLIB_DECLARE_GET_AND_SET_BY_REFERENCE(Camera,float3,LookAt,m_LookAt);
        RTLIB_DECLARE_GET_AND_SET_BY_REFERENCE(Camera,float3,   Vup,   m_Vup);
        RTLIB_DECLARE_GET_AND_SET_BY_VALUE(Camera,float,  FovY,m_FovY  );
        RTLIB_DECLARE_GET_AND_SET_BY_VALUE(Camera,float,Aspect,m_Aspect);
        //getUVW
        void getUVW(float3& u, float3& v, float3& w)const noexcept;
        std::tuple<float3,float3,float3> getUVW( )const noexcept;
    };
}
#endif