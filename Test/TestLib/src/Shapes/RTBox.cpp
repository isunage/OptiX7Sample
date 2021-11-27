#include "RTBox.h"

test::RTBox::RTBox() noexcept : RTShape()
{
}

auto test::RTBox::GetTypeName() const noexcept -> RTString 
{
    return "Shape";
}

auto test::RTBox::GetPluginName() const noexcept -> RTString 
{
    return "Box";
}

auto test::RTBox::GetID() const noexcept -> RTString
{
    if (m_Properties.HasString("ID"))
    {
        return m_Properties.GetString("ID");
    }
    else {
        return "";
    }
}

auto test::RTBox::GetProperties() const noexcept -> const RTProperties & 
{
    return m_Properties;
}

auto test::RTBox::GetJsonAsData() const noexcept -> nlohmann::json 
{
    nlohmann::json data;
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
    data["Properties"] = GetProperties().GetJsonData();
    return data;
}

auto test::RTBox::GetMaterial() const noexcept -> RTMaterialPtr
{
    return m_Material;
}

void test::RTBox::SetID(const std::string& id) noexcept
{
    m_Properties.SetString("ID", id);
}

auto test::RTBox::GetFlipNormals() const noexcept -> RTBool
{
    if (m_Properties.HasBool("FlipNormals")) {
        return m_Properties.GetBool("FlipNormals");
    }
    else
    {
        return false;
    }
}

auto test::RTBox::GetTransforms() const noexcept -> RTMat4x4
{
    if (m_Properties.HasMat4x4("Transforms")) {
        return m_Properties.GetMat4x4("Transforms");
    }
    else
    {
        return RTMat4x4::Identity();
    }
}

void test::RTBox::SetFlipNormals(const RTBool& val) noexcept
{
    m_Properties.SetBool("FlipNormals", val);
}

auto test::RTBox::SetTransforms(const RTMat4x4& mat) noexcept
{
    m_Properties.SetMat4x4("Transforms", mat);
}

struct test::RTBoxReader::Impl {
    std::weak_ptr<RTMaterialCache> matCache;
};

test::RTBoxReader::RTBoxReader(const std::shared_ptr<RTMaterialCache>& matCache) noexcept
{
    m_Impl = std::make_unique<test::RTBoxReader::Impl>();
    m_Impl->matCache = matCache;
}

auto test::RTBoxReader::GetPluginName() const noexcept -> RTString 
{
    return "Box";
}

auto test::RTBoxReader::LoadJsonFromData(const nlohmann::json& json) noexcept -> RTShapePtr 
{
    if (!json.contains("Type") || !json["Type"].is_string() || json["Type"].get<std::string>() != "Shape") {
        return nullptr;
    }

    if (!json.contains("Plugin") || !json["Plugin"].is_string() || json["Plugin"].get<std::string>() != "Box") {
        return nullptr;
    }

    if (!json.contains("Properties") || !json["Properties"].is_object()) {
        return nullptr;
    }

    if (!json.contains("Material")) {
        return nullptr;
    }

    auto& propertiesJson = json["Properties"];
    auto box = std::make_shared<test::RTBox>();
    if (!box->m_Properties.LoadMat4x4("Transforms", propertiesJson)) {
        return nullptr;
    }

    if (!box->m_Properties.LoadBool("FlipNormals", propertiesJson)) {
        return nullptr;
    }

    auto& meterialJson = json["Material"];
    auto  material = m_Impl->matCache.lock()->LoadJsonFromData(meterialJson);
    if (!material) {
        return nullptr;
    }
    box->m_Material = material;
    return box;
}

test::RTBoxReader::~RTBoxReader() noexcept
{
}
