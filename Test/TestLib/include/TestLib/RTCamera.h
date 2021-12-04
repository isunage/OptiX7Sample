#ifndef TEST_RT_CAMERA_H
#define TEST_RT_CAMERA_H
#include <RTLib/math/VectorFunction.h>
#include <RTLib/math/Matrix.h>
namespace test
{
    class RTCamera
    {
    public:
        RTCamera()noexcept {}
        virtual auto GetDirection()const noexcept -> float3            = 0;
        virtual void SetDirection(const float3& direction)noexcept     = 0;
        virtual auto GetViewMatrix()const noexcept -> rtlib::Matrix3x3 = 0;

        virtual ~RTCamera()noexcept {}
    };
    class RTCameraController
    {
    public:
        RTCameraController(
            const float3& position = make_float3(0.0f, 0.0f, 0.0f),
            const float3& up = make_float3(0.0f, 1.0f, 0.0f),
            float yaw = defaultYaw,
            float pitch = defaultPitch) noexcept : m_Position{ position },
            m_Up{ up },
            m_Yaw{ yaw },
            m_Pitch{ pitch },
            m_MouseSensitivity{ defaultSensitivity },
            m_MovementSpeed{ defaultSpeed },
            m_Zoom{ defaultZoom }
        {
            UpdateCameraVectors();
        }
        void SetCamera(const Camera& camera) noexcept
        {
            m_Position = camera.getEye();
            m_Front = camera.getLookAt() - m_Position;
            m_Up = camera.getVup();
            m_Right = rtlib::normalize(rtlib::cross(m_Up, m_Front));
        }
        auto GetCamera(float fovY, float aspect) const noexcept -> Camera
        {
            return Camera(m_Position, m_Position + m_Front, m_Up, fovY, aspect);
        }
        void ProcessKeyboard(CameraMovement mode, float deltaTime) noexcept
        {
            float velocity = m_MovementSpeed * deltaTime;
            if (mode == CameraMovement::eForward)
            {
                m_Position += m_Front * velocity;
            }
            if (mode == CameraMovement::eBackward)
            {
                m_Position -= m_Front * velocity;
            }
            if (mode == CameraMovement::eLeft)
            {
                m_Position += m_Right * velocity;
            }
            if (mode == CameraMovement::eRight)
            {
                m_Position -= m_Right * velocity;
            }
            if (mode == CameraMovement::eUp)
            {
                m_Position += m_Up * velocity;
            }
            if (mode == CameraMovement::eDown)
            {
                m_Position -= m_Up * velocity;
            }
        }
        void ProcessMouseMovement(float xoffset, float yoffset, bool constrainPitch = true) noexcept
        {
            xoffset *= m_MouseSensitivity;
            yoffset *= m_MouseSensitivity;
            m_Yaw -= xoffset;
            m_Pitch += yoffset;
            if (constrainPitch)
            {
                if (m_Pitch > 89.0f)
                {
                    m_Pitch = 89.0f;
                }
                if (m_Pitch < -89.0f)
                {
                    m_Pitch = -89.0f;
                }
            }
            UpdateCameraVectors();
        }
        void ProcessMouseScroll(float yoffset) noexcept
        {
            float next_zoom = m_Zoom - yoffset;
            if (next_zoom >= 1.0f && next_zoom <= 45.0f)
                m_Zoom = next_zoom;
            if (next_zoom <= 1.0f)
                m_Zoom = 1.0f;
            if (next_zoom >= 45.0f)
                m_Zoom = 45.0f;
        }
        void SetMouseSensitivity(float sensitivity) noexcept
        {
            m_MouseSensitivity = sensitivity;
        }
        void SetMovementSpeed(float speed) noexcept
        {
            m_MovementSpeed = speed;
        }

        virtual ~RTCameraController()noexcept {}
    };
	class RTPinholeCamera : public RTCamera
	{
    public:
        RTPinholeCamera() noexcept : m_Eye{}, m_LookAt{}, m_Vup{}, m_FovY{}, m_Aspect{} {}
        RTPinholeCamera(const float3& eye,const float3& lookAt,const float3& vup,
            const float fovY,
            const float aspect) noexcept
            : m_Eye{ eye },
            m_LookAt{ lookAt },
            m_Vup{ vup },
            m_FovY{ fovY },
            m_Aspect{ aspect }{}
        //Get And Set
        RTLIB_DECLARE_GET_AND_SET_BY_REFERENCE(RTPinholeCamera, float3, Eye, m_Eye);
        RTLIB_DECLARE_GET_AND_SET_BY_REFERENCE(RTPinholeCamera, float3, LookAt, m_LookAt);
        RTLIB_DECLARE_GET_AND_SET_BY_REFERENCE(RTPinholeCamera, float3, Vup, m_Vup);
        RTLIB_DECLARE_GET_AND_SET_BY_VALUE(Camera, float, FovY, m_FovY);
        RTLIB_DECLARE_GET_AND_SET_BY_VALUE(Camera, float, Aspect, m_Aspect);
        //getUVW
        //Direction
        virtual float3 GetDirection() const noexcept override
        {
            return rtlib::normalize(m_LookAt - m_Eye);
        }
        virtual  void SetDirection(const float3& direction) noexcept override
        {
            auto len = rtlib::length(m_LookAt - m_Eye);
            m_Eye += len * rtlib::normalize(direction);
        }
        virtual auto GetViewMatrix()const noexcept -> rtlib::Matrix3x3 override;

        virtual ~RTPinholeCamera()noexcept {}
    private:
        float3 m_Eye;
        float3 m_LookAt;
        float3 m_Vup;
        float  m_FovY;
        float  m_Aspect;
	};

}
#endif