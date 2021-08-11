#ifndef RTLIB_CAMERA_H
#define RTLIB_CAMERA_H
#include <tuple>
#include <cmath>
#include <iostream>
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
         m_Aspect{aspect}{
            //std::cout << "Camera Eye (x:" << m_Eye.x    << " y:" <<m_Eye.y    << " z:" <<    m_Eye.z << ")" << std::endl;
			//std::cout << "Camera  At (x:" << m_LookAt.x << " y:" <<m_LookAt.y << " z:" << m_LookAt.z << ")" << std::endl;
			//std::cout << "Camera  up (x:" <<    m_Vup.x << " y:" <<   m_Vup.y << " z:" <<    m_Vup.z << ")" << std::endl;
         }
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
    enum class CameraMovement :uint8_t {
        eForward  = 0,
        eBackward = 1,
        eLeft     = 2,
        eRight    = 3,
        eUp       = 4,
        eDown     = 5,
    };
    struct CameraController {
    private:
        inline static constexpr float defaultYaw         = -90.0f;
        inline static constexpr float defaultPitch       = 0.0f;
        inline static constexpr float defaultSpeed       = 1.0f;
        inline static constexpr float defaultSensitivity = 0.025f;
        inline static constexpr float defaultZoom        = 45.0f;
    private:
        float3 m_Position;
        float3 m_Front;
        float3 m_Up;
        float3 m_Right;
        float  m_Yaw;
        float  m_Pitch;
        float  m_MovementSpeed;
        float  m_MouseSensitivity;
        float  m_Zoom;
    public:
        CameraController(
            const float3& position = make_float3(0.0f, 0.0f, 0.0f),
            const float3& up       = make_float3(0.0f, 1.0f, 0.0f),
            float yaw = defaultYaw,
            float pitch = defaultPitch)noexcept :
            m_Position{ position },
            m_Up{ up },
            m_Yaw{ yaw },
            m_Pitch{ pitch },
            m_MouseSensitivity{ defaultSensitivity },
            m_MovementSpeed{ defaultSpeed },
            m_Zoom{ defaultZoom }{
            UpdateCameraVectors();
        }
        void SetCamera(const Camera& camera)noexcept
        {
            m_Position = camera.getEye();
            m_Front    = camera.getLookAt() - m_Position;
            m_Up       = camera.getVup();
            m_Right    = rtlib::normalize(rtlib::cross(m_Up, m_Front));
        }
        auto GetCamera(float fovY, float aspect)const noexcept -> Camera{
            return Camera(m_Position, m_Position+m_Front,m_Up,fovY,aspect);
        }
        void ProcessKeyboard(CameraMovement mode, float deltaTime)noexcept {
            float velocity = m_MovementSpeed * deltaTime;
            if (mode == CameraMovement::eForward) {
                m_Position += m_Front * velocity;
            }
            if (mode == CameraMovement::eBackward) {
                m_Position -= m_Front * velocity;
            }
            if (mode == CameraMovement::eLeft) {
                m_Position += m_Right * velocity;
            }
            if (mode == CameraMovement::eRight) {
                m_Position -= m_Right * velocity;
            }
            if (mode == CameraMovement::eUp) {
                m_Position += m_Up    * velocity;
            }
            if (mode == CameraMovement::eDown) {
                m_Position -= m_Up    * velocity;
            }
        }
        void ProcessMouseMovement(float xoffset, float yoffset, bool constrainPitch = true)noexcept {
            xoffset *= m_MouseSensitivity;
            yoffset *= m_MouseSensitivity;
            m_Yaw   -= xoffset;
            m_Pitch += yoffset;
            if (constrainPitch) {
                if (m_Pitch > 89.0f) {
                    m_Pitch = 89.0f;
                }
                if (m_Pitch < -89.0f) {
                    m_Pitch = -89.0f;
                }
            }
            UpdateCameraVectors();
        }
        void ProcessMouseScroll(float yoffset)noexcept
        {
            float next_zoom = m_Zoom - yoffset;
            if (next_zoom >= 1.0f && next_zoom <= 45.0f)
                m_Zoom = next_zoom;
            if (next_zoom <= 1.0f)
                m_Zoom = 1.0f;
            if (next_zoom >= 45.0f)
                m_Zoom = 45.0f;
        }
        void SetMouseSensitivity(float sensitivity)noexcept {
            m_MouseSensitivity = sensitivity;
        }
        void SetMovementSpeed(float speed)noexcept {
            m_MovementSpeed    = speed;
        }
    private:
        // Calculates the front vector from the Camera's (updated) Euler Angles
        void UpdateCameraVectors()noexcept
        {
            // Calculate the new Front vector
            float3 front;
            float yaw   = RTLIB_M_PI * (m_Yaw) / 180.0f;
            float pitch = RTLIB_M_PI * (m_Pitch) / 180.0f;
            front.x = cos(yaw) * cos(pitch);
            front.y = sin(pitch);
            front.z = sin(yaw) * cos(pitch);
            m_Front = rtlib::normalize(front);
            m_Right = rtlib::normalize(rtlib::cross(m_Up, m_Front));
        }
    };
}
#endif