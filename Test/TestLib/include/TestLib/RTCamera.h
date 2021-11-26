#ifndef TEST_RT_CAMERA_H
#define TEST_RT_CAMERA_H
#include <TestLib/RTInterface.h>
#include <nlohmann/json.hpp>
#include <memory>
#include <string>
namespace test {
    class RTProperties;
    class RTCamera :public RTInterface
    {
    public:
        RTCamera()noexcept :RTInterface() {}
        virtual ~RTCamera()noexcept {}
    };
    using RTCameraPtr = std::shared_ptr<RTCamera>;
    class RTCameraReader
    {
        auto LoadJsonFromString(const std::string& jsonStr)noexcept -> RTCameraPtr {
            return LoadJsonFromData(nlohmann::json::parse(jsonStr));
        }
        virtual auto LoadJsonFromData(const nlohmann::json& json)noexcept -> RTCameraPtr = 0;
        virtual ~RTCameraReader()noexcept {}
    };
    using RTCameraReaderPtr = std::shared_ptr<RTCameraReader>;
}
#endif
