#ifndef TEST_RT_MATERIAL_H
#define TEST_RT_MATERIAL_H
#include <TestLib/RTInterface.h>
#include <nlohmann/json.hpp>
#include <memory>
#include <string>
namespace test{
    class RTProperties;
    class RTMaterial:public RTInterface
    {
    public:
        RTMaterial()noexcept:RTInterface(){}
        virtual ~RTMaterial()noexcept {}
    };
    using RTMaterialPtr = std::shared_ptr<RTMaterial>;
    class RTMaterialReader
    {
    public:
        auto LoadJsonFromString(const std::string& jsonStr)noexcept -> RTMaterialPtr {
            return LoadJsonFromData(nlohmann::json::parse(jsonStr));
        }
        virtual auto LoadJsonFromData(const nlohmann::json& json)noexcept -> RTMaterialPtr = 0;
        virtual ~RTMaterialReader()noexcept {}
    };
    using RTMaterialReaderPtr = std::shared_ptr<RTMaterialReader>;
    class RTMaterialCache
    {
    public:
        RTMaterialCache()noexcept;
        bool AddMaterial(const RTMaterialPtr& material)noexcept;
        bool HasMaterial(const std::string&   id)const noexcept;
        auto GetMaterial(const std::string&   id)const ->RTMaterialPtr;
        ~RTMaterialCache()noexcept;
    private:
        std::unordered_map<std::string, RTMaterialPtr> m_BaseMap;
    };
}
#endif