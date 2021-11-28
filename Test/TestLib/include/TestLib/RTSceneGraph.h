#ifndef TEST_RT_SCENE_GRAPH_H
#define TEST_RT_SCENE_GRAPH_H
#include <TestLib/RTShape.h>
#include <TestLib/RTMaterial.h>
#include <TestLib/RTInterface.h>
#include <nlohmann/json.hpp>
#include <memory>
#include <string>
namespace test {
    class RTProperties;
    class RTSceneGraph :public RTInterface
    {
    public:
        RTSceneGraph()noexcept :RTInterface() {}
        virtual auto GetTransforms()const noexcept -> rtlib::Matrix4x4 = 0;
        virtual auto GetID()const noexcept -> std::string              = 0;
        virtual void SetID(const std::string&)noexcept                 = 0;
        virtual ~RTSceneGraph()noexcept {}
    };
    using RTSceneGraphPtr = std::shared_ptr<RTSceneGraph>;
    class RTSceneGraphReader
    {
    public:
        auto LoadJsonFromString(const std::string& jsonStr)noexcept -> RTSceneGraphPtr {
            return LoadJsonFromData(nlohmann::json::parse(jsonStr));
        }
        virtual auto GetPluginName()const noexcept -> std::string = 0;
        virtual auto LoadJsonFromData(const nlohmann::json& json)noexcept -> RTSceneGraphPtr = 0;
        virtual ~RTSceneGraphReader()noexcept {}
    };
    using RTSceneGraphReaderPtr = std::shared_ptr<RTSceneGraphReader>;
    class RTSceneGraphCache
    {
    public:
        RTSceneGraphCache()noexcept;
        bool AddSceneGraph(const RTSceneGraphPtr& texture)noexcept;
        bool HasSceneGraph(const std::string& id)const noexcept;
        auto GetSceneGraph(const std::string& id)const->RTSceneGraphPtr;
        bool AddReader(const RTSceneGraphReaderPtr& reader)noexcept;
        bool HasReader(const std::string& id)const noexcept;
        auto GetReader(const std::string& id)const->RTSceneGraphReaderPtr;
        auto LoadJsonFromData(const nlohmann::json& json)noexcept -> RTSceneGraphPtr;
        ~RTSceneGraphCache()noexcept;
    private:
        std::unordered_map<std::string, RTSceneGraphPtr>       m_Graphs;
        std::unordered_map<std::string, RTSceneGraphReaderPtr> m_Readers;
    };
    auto GetDefaultSceneGraphCache(const std::shared_ptr<RTShapeCache>& sphCache, const std::shared_ptr<RTMaterialCache>& matCache) noexcept -> std::shared_ptr<RTSceneGraphCache>;
}
#endif
