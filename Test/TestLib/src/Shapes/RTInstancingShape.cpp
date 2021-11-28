#include "RTInstancingShape.h"

test::RTInstancingShape::RTInstancingShape() noexcept : RTShape()
{
}

auto test::RTInstancingShape::GetTypeName() const noexcept -> RTString 
{
    return "Shape";
}

auto test::RTInstancingShape::GetPluginName() const noexcept -> RTString 
{
    return "Instancing";
}

auto test::RTInstancingShape::GetID() const noexcept -> RTString
{
    if (m_Properties.HasString("ID"))
    {
        return m_Properties.GetString("ID");
    }
    else {
        return "";
    }
}

auto test::RTInstancingShape::GetProperties() const noexcept -> const RTProperties & 
{
    return m_Properties;
}

auto test::RTInstancingShape::GetJsonAsData() const noexcept -> nlohmann::json 
{
    nlohmann::json data;
    data = GetProperties().GetJsonAsData();
    data["Type"]      = GetTypeName();
    data["Plugin"]    = GetPluginName();
    data["BaseShape"] = GetBaseShape()->GetID();
    if (GetMaterial()) {
        if (GetMaterial()->GetID() != "") {
            data["Material"] = GetMaterial()->GetID();
        }
        else {
            data["Material"] = GetMaterial()->GetJsonAsData();
        }
    }
    return data;
}

auto test::RTInstancingShape::GetMaterial() const noexcept -> RTMaterialPtr
{
    if (m_Material) {
        return m_Material;
    }
    else {
        return m_BaseShape->GetMaterial();
    }
}

void test::RTInstancingShape::SetID(const std::string& id) noexcept
{
    m_Properties.SetString("ID", id);
}

auto test::RTInstancingShape::GetBaseShape() const noexcept -> RTShapePtr
{
    return m_BaseShape;
}

void test::RTInstancingShape::SetBaseShape(const RTShapePtr& shp) noexcept
{
    m_BaseShape = shp;
}

auto test::RTInstancingShape::GetFlipNormals() const noexcept -> RTBool
{
    if (m_Properties.HasBool("FlipNormals")) {
        return m_Properties.GetBool("FlipNormals");
    }
    else
    {
        return m_BaseShape->GetFlipNormals();
    }
}

auto test::RTInstancingShape::GetTransforms() const noexcept -> RTMat4x4
{
    if (m_Properties.HasMat4x4("Transforms")) {
        return m_Properties.GetMat4x4("Transforms");
    }
    else {
        return RTMat4x4::Identity();
    }
}

void test::RTInstancingShape::SetFlipNormals(const RTBool& val) noexcept
{
    m_Properties.SetBool("FlipNormals", val);
}

auto test::RTInstancingShape::SetTransforms(const RTMat4x4& mat) noexcept
{
    m_Properties.SetMat4x4("Transforms", mat);
}

struct test::RTInstancingShapeReader::Impl {
    std::weak_ptr<RTMaterialCache> matCache;
    std::weak_ptr<   RTShapeCache> shpCache;
};

test::RTInstancingShapeReader::RTInstancingShapeReader(const std::shared_ptr< RTShapeCache>& shpCache, const std::shared_ptr<RTMaterialCache>& matCache) noexcept
{
    m_Impl = std::make_unique<test::RTInstancingShapeReader::Impl>();
    m_Impl->matCache = matCache;
    m_Impl->shpCache = shpCache;
    
}

auto test::RTInstancingShapeReader::GetPluginName() const noexcept -> RTString 
{
    return "Instancing";
}

auto test::RTInstancingShapeReader::LoadJsonFromData(const nlohmann::json& json) noexcept -> RTShapePtr 
{
    if (!json.contains("Type") || !json["Type"].is_string() || json["Type"].get<std::string>() != "Shape") {
        return nullptr;
    }

    if (!json.contains("Plugin") || !json["Plugin"].is_string() || json["Plugin"].get<std::string>() != "Instancing") {
        return nullptr;
    }
    if (!json.contains("BaseShape") || !json["BaseShape"].is_string()) {
        return nullptr;
    }

    auto box = std::make_shared<test::RTInstancingShape>();
    
    {
        auto& baseShapeJson = json["BaseShape"];
        auto  baseShape = m_Impl->shpCache.lock()->LoadJsonFromData(baseShapeJson);
        if (!baseShape) {
            return nullptr;
        }
        box->m_BaseShape = baseShape;
    }

    box->m_Properties.LoadMat4x4("Transforms", json);

    box->m_Properties.LoadBool( "FlipNormals", json);

    if (json.contains("Material")) {
        auto& meterialJson = json["Material"];
        auto  material     = m_Impl->matCache.lock()->LoadJsonFromData(meterialJson);
        if (!material) {
            return nullptr;
        }
        box->m_Material = material;
    }
    return box;
}

test::RTInstancingShapeReader::~RTInstancingShapeReader() noexcept
{
}
