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
    public:
        virtual auto GetPluginName()const noexcept -> std::string = 0;
        auto LoadJsonFromString(const std::string& jsonStr)noexcept -> RTCameraPtr {
            return LoadJsonFromData(nlohmann::json::parse(jsonStr));
        }
        virtual auto LoadJsonFromData(const nlohmann::json& json)noexcept -> RTCameraPtr = 0;
        virtual ~RTCameraReader()noexcept {}
    };
    using RTCameraReaderPtr = std::shared_ptr<RTCameraReader>;
    class RTCameraCache
    {
    public:
        RTCameraCache()noexcept;
        bool AddReader(const RTCameraReaderPtr& reader)noexcept;
        bool HasReader(const std::string& id)const noexcept;
        auto GetReader(const std::string& id)const->RTCameraReaderPtr;
        auto LoadJsonFromData(const nlohmann::json& json)noexcept -> RTCameraPtr;
        ~RTCameraCache() {}
    private:
        std::unordered_map<std::string, RTCameraReaderPtr> m_Readers;
    };
    auto GetDefaultCameraCache() noexcept -> std::shared_ptr<RTCameraCache>;
}
#endif
