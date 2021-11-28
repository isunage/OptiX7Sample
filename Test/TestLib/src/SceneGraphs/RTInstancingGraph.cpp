#include "RTInstancingGraph.h"

test::RTInstancingGraph::RTInstancingGraph() noexcept : RTSceneGraph()
{
}

auto test::RTInstancingGraph::GetTypeName() const noexcept -> RTString
{
    return "SceneGraph";
}

auto test::RTInstancingGraph::GetPluginName() const noexcept -> RTString
{
    return "Instancing";
}

auto test::RTInstancingGraph::GetID() const noexcept -> RTString
{
    if (m_Properties.HasString("ID"))
    {
        return m_Properties.GetString("ID");
    }
    else
    {
        return "";
    }
}

auto test::RTInstancingGraph::GetProperties() const noexcept -> const RTProperties &
{
    return m_Properties;
}

auto test::RTInstancingGraph::GetJsonAsData() const noexcept -> nlohmann::json
{
    nlohmann::json data;
    data = GetProperties().GetJsonAsData();
    data["Type"] = GetTypeName();
    data["Plugin"] = GetPluginName();
    data["BaseGraph"] = GetBaseGraph()->GetID();
    data["Materials"] = nlohmann::json();
    for (auto &material : m_Materials)
    {
        if (material->GetID() == "")
        {

            data["Materials"].push_back(material->GetJsonAsData());
        }
        else
        {
            data["Materials"].push_back(material->GetID());
        }
    }
    return data;
}

auto test::RTInstancingGraph::GetMaterials() const noexcept -> const std::vector<RTMaterialPtr> &
{
    return m_Materials;
}

void test::RTInstancingGraph::SetID(const std::string &id) noexcept
{
    m_Properties.SetString("ID", id);
}

auto test::RTInstancingGraph::GetBaseGraph() const noexcept -> RTSceneGraphPtr
{
    return m_BaseGraph;
}

void test::RTInstancingGraph::SetBaseGraph(const RTSceneGraphPtr &shp) noexcept
{
    m_BaseGraph = shp;
}

auto test::RTInstancingGraph::GetTransforms() const noexcept -> RTMat4x4
{
    if (m_Properties.HasMat4x4("Transforms"))
    {
        return m_Properties.GetMat4x4("Transforms");
    }
    return RTMat4x4::Identity();
}

auto test::RTInstancingGraph::SetTransforms(const RTMat4x4 &mat) noexcept
{
    m_Properties.SetMat4x4("Transforms", mat);
}

struct test::RTInstancingGraphReader::Impl
{
    std::weak_ptr<RTMaterialCache> matCache;
    std::weak_ptr<RTSceneGraphCache> gphCache;
};

test::RTInstancingGraphReader::RTInstancingGraphReader(const std::shared_ptr<RTSceneGraphCache> &gphCache, const std::shared_ptr<RTMaterialCache> &matCache) noexcept
{
    m_Impl = std::make_unique<test::RTInstancingGraphReader::Impl>();
    m_Impl->matCache = matCache;
    m_Impl->gphCache = gphCache;
}

auto test::RTInstancingGraphReader::GetPluginName() const noexcept -> RTString
{
    return "Instancing";
}

auto test::RTInstancingGraphReader::LoadJsonFromData(const nlohmann::json &json) noexcept -> RTSceneGraphPtr
{
    if (!json.contains("Type") || !json["Type"].is_string() || json["Type"].get<std::string>() != "SceneGraph")
    {
        return nullptr;
    }

    if (!json.contains("Plugin") || !json["Plugin"].is_string() || json["Plugin"].get<std::string>() != "Instancing")
    {
        return nullptr;
    }
    if (!json.contains("BaseGraph") || !json["BaseGraph"].is_string())
    {
        return nullptr;
    }

    auto instancing = std::make_shared<test::RTInstancingGraph>();

    {
        auto &baseGraphJson = json["BaseGraph"];
        auto baseGraph = m_Impl->gphCache.lock()->LoadJsonFromData(baseGraphJson);
        if (!baseGraph)
        {
            return nullptr;
        }
        instancing->m_BaseGraph = baseGraph;
    }
    
    if (instancing->m_BaseGraph->GetPluginName() != "ShapeArray")
    {
        if(json.contains("Materials")){
            return nullptr;
        }
    }

    instancing->m_Properties.LoadMat4x4("Transforms", json);
    if (json.contains("Materials"))
    {
        auto &materialsJson = json["Materials"];
        if (!materialsJson.is_array())
        {
            return nullptr;
        }
        for (auto &materialJson : materialsJson)
        {
            auto material = m_Impl->matCache.lock()->LoadJsonFromData(materialJson);
            if (!material)
            {
                return nullptr;
            }
            instancing->m_Materials.push_back(
                material);
        }
    }
    return instancing;
}

test::RTInstancingGraphReader::~RTInstancingGraphReader() noexcept
{
}
