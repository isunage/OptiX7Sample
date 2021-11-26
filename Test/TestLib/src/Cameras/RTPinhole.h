#ifndef TEST_RT_CAMERA_PINHOLE_H
#define TEST_RT_CAMERA_PINHOLE_H
#include <TestLib/RTCamera.h>
#include <TestLib/RTProperties.h>
#include <RTLib/ext/Math/VectorFunction.h>
#include <string>
namespace test
{
    class RTPinhole: public RTCamera
    {
    public:
        RTPinhole()noexcept;
        virtual auto GetTypeName()  const noexcept -> RTString override;
        virtual auto GetPluginName()const noexcept -> RTString override;
        virtual auto GetID()        const noexcept -> RTString override;
        virtual auto GetProperties()const noexcept -> const RTProperties & override;
        virtual auto GetJsonAsData()const noexcept ->       nlohmann::json override;
        auto GetEye()const noexcept -> RTPoint;
        void SetEye(const RTPoint& p)noexcept ;
        auto GetLookAt()const noexcept -> RTVector;
        void SetLookAt(const RTVector& d)noexcept;
        auto GetVup()const noexcept -> RTVector;
        void SetVup(const RTVector& d)noexcept;
        auto GetFovY()const noexcept -> RTFloat;
        void SetFovY(const RTFloat fovY)noexcept;
        void SetAspect(const RTFloat aspect)noexcept;
        auto GetAspect()const noexcept -> RTFloat;
        virtual ~RTPinhole()noexcept {}
    private:
        std::string  m_ID;
        RTProperties m_Properties;
    };
}
#endif