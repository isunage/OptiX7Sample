#include "RTSphere.h"

test::RTSphere::RTSphere() noexcept : RTShape()
{
}

auto test::RTSphere::GetTypeName() const noexcept -> RTString
{
    return "Shape";
}

auto test::RTSphere::GetPluginName() const noexcept -> RTString
{
    return "Sphere";
}

auto test::RTSphere::GetID() const noexcept -> RTString
{
    if (m_Properties.HasString("ID"))
    {
        return m_Properties.GetString("ID");
    }
    else {
        return "";
    }
}

auto test::RTSphere::GetProperties() const noexcept -> const RTProperties&
{
    return m_Properties;
}

auto test::RTSphere::GetJsonAsData() const noexcept -> nlohmann::json
{
    nlohmann::json data;
    data = GetProperties().GetJsonAsData();
    data["Type"] = GetTypeName();
    data["Plugin"] = GetPluginName();
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

auto test::RTSphere::GetMaterial() const noexcept -> RTMaterialPtr
{
    return m_Material;
}

void test::RTSphere::SetID(const std::string& id) noexcept
{
    m_Properties.SetString("ID", id);
}

auto test::RTSphere::GetCenter() const noexcept -> RTPoint
{
    if (m_Properties.HasPoint("Center")) {
        return m_Properties.GetPoint("Center");
    }
    else {
        return make_float3(0.0f);
    }
}

auto test::RTSphere::GetRadius() const noexcept -> RTFloat
{
    if (m_Properties.HasFloat("Radius")) {
        return m_Properties.GetFloat("Radius");
    }
    else {
        return 1.0f;
    }
}

auto test::RTSphere::GetFlipNormals() const noexcept -> RTBool
{
    if (m_Properties.HasBool("FlipNormals")) {
        return m_Properties.GetBool("FlipNormals");
    }
    else
    {
        return false;
    }
}

auto test::RTSphere::GetTransforms() const noexcept -> RTMat4x4
{
    if (m_Properties.HasMat4x4("Transforms")) {
        return m_Properties.GetMat4x4("Transforms");
    }
    else
    {
        return RTMat4x4::Identity();
    }
}

void test::RTSphere::SetCenter(const RTPoint& center) noexcept
{
    m_Properties.SetPoint("Center",center);
}

void test::RTSphere::SetRadius(const RTFloat& radius) noexcept
{
    m_Properties.SetFloat("Radius", radius);
}

void test::RTSphere::SetFlipNormals(const RTBool& val) noexcept
{
    m_Properties.SetBool("FlipNormals", val);
}

auto test::RTSphere::SetTransforms(const RTMat4x4& mat) noexcept
{
    m_Properties.SetMat4x4("Transforms", mat);
}

struct test::RTSphereReader::Impl {
    std::weak_ptr<RTMaterialCache> matCache;
};

test::RTSphereReader::RTSphereReader(const std::shared_ptr<RTMaterialCache>& matCache) noexcept
{
    m_Impl = std::make_unique<test::RTSphereReader::Impl>();
    m_Impl->matCache = matCache;
}

auto test::RTSphereReader::GetPluginName() const noexcept -> RTString
{
    return "Sphere";
}

auto test::RTSphereReader::LoadJsonFromData(const nlohmann::json& json) noexcept -> RTShapePtr
{
    if (!json.contains("Type") || !json["Type"].is_string() || json["Type"].get<std::string>() != "Shape") {
        return nullptr;
    }

    if (!json.contains("Plugin") || !json["Plugin"].is_string() || json["Plugin"].get<std::string>() != "Sphere") {
        return nullptr;
    }
    
    if (!json.contains("Material")) {
        return nullptr;
    }

    auto sphere = std::make_shared<test::RTSphere>();

    sphere->m_Properties.LoadMat4x4("Transforms", json);

    sphere->m_Properties.LoadBool( "FlipNormals", json);

    if (!sphere->m_Properties.LoadPoint("Center", json)) {
        return nullptr;
    }

    if (!sphere->m_Properties.LoadFloat("Radius", json)) {
        return nullptr;
    }

    auto& meterialJson = json["Material"];
    auto  material = m_Impl->matCache.lock()->LoadJsonFromData(meterialJson);
    if (!material) {
        return nullptr;
    }
    sphere->m_Material = material;
    return sphere;
}

test::RTSphereReader::~RTSphereReader() noexcept
{
}
