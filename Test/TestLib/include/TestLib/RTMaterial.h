#ifndef TEST_RT_MATERIAL_H
#define TEST_RT_MATERIAL_H
#include <TestLib/RTInterface.h>
#include <nlohmann/json.hpp>
#include <memory>
#include <string>
namespace test{
    class RTProperties;
    class RTTextureCache;
    class RTMaterial:public RTInterface
    {
    public:
        RTMaterial()noexcept:RTInterface(){}
        virtual auto GetID()        const noexcept -> std::string = 0;
        virtual void SetID(const std::string&)noexcept = 0;
        virtual ~RTMaterial()noexcept {}
    };
    using RTMaterialPtr = std::shared_ptr<RTMaterial>;
    class RTMaterialReader
    {
    public:
        auto LoadJsonFromString(const std::string& jsonStr)noexcept -> RTMaterialPtr {
            return LoadJsonFromData(nlohmann::json::parse(jsonStr));
        }
        virtual auto GetPluginName()const noexcept -> std::string = 0;
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
        bool AddReader(const RTMaterialReaderPtr& reader)noexcept;
        bool HasReader(const std::string& id)const noexcept;
        auto GetReader(const std::string& id)const->RTMaterialReaderPtr;
        auto LoadJsonFromData(const nlohmann::json& json)noexcept -> RTMaterialPtr;
        ~RTMaterialCache()noexcept;
    private:
        std::unordered_map<std::string, RTMaterialPtr      > m_Materials;
        std::unordered_map<std::string, RTMaterialReaderPtr> m_Readers;
    };
    auto GetDefaultMaterialCache(const std::shared_ptr<RTTextureCache>& texCache) noexcept -> std::shared_ptr<RTMaterialCache>;
}
#endif