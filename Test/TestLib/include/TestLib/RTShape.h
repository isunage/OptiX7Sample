#ifndef TEST_RT_SHAPE_H
#define TEST_RT_SHAPE_H
#include <TestLib/RTMaterial.h>
#include <TestLib/RTInterface.h>
#include <nlohmann/json.hpp>
#include <memory>
#include <string>
namespace test {
    class RTMaterial;
    class RTProperties;
    class RTShape :public RTInterface
    {
    public:
        RTShape()noexcept :RTInterface() {}
        virtual auto GetMaterial()const noexcept   -> std::shared_ptr<RTMaterial> = 0;
        virtual auto GetID()        const noexcept -> std::string = 0;
        virtual void SetID(const std::string&)noexcept = 0;
        virtual ~RTShape()noexcept {}
    };
    using RTShapePtr = std::shared_ptr<RTShape>;
    class RTShapeReader
    {
    public:
        auto LoadJsonFromString(const std::string& jsonStr)noexcept -> RTShapePtr {
            return   LoadJsonFromData(nlohmann::json::parse(jsonStr));
        }
        virtual auto GetPluginName()const noexcept -> std::string = 0;
        virtual auto LoadJsonFromData(const nlohmann::json& json)noexcept -> RTShapePtr = 0;
        virtual ~RTShapeReader()noexcept {}
    };
    using RTShapeReaderPtr = std::shared_ptr<RTShapeReader>;
    class RTShapeCache
    {
    public:
        RTShapeCache(const std::shared_ptr<RTMaterialCache>& cache)noexcept;
        bool AddReader(const RTShapeReaderPtr& reader)noexcept;
        bool HasReader(const std::string& id)const noexcept;
        auto GetReader(const std::string& id)const->RTShapeReaderPtr;
        auto LoadJsonFromData(const nlohmann::json& json)noexcept -> RTShapePtr;
        ~RTShapeCache()noexcept {}
    private:
        std::unordered_map<std::string, RTShapeReaderPtr> m_Readers  = {};
        std::shared_ptr<RTMaterialCache>                  m_MatCache = nullptr;
    };

    auto GetDefaultShapeCache(const std::shared_ptr<RTMaterialCache>& cache) noexcept -> std::shared_ptr<RTShapeCache>;
}
#endif