#include "RTDiffuse.h"

test::RTDiffuse::RTDiffuse() noexcept
{
}

auto test::RTDiffuse::GetTypeName() const noexcept -> RTString 
{
    return "Material";
}

auto test::RTDiffuse::GetPluginName() const noexcept -> RTString 
{
    return "Diffuse";
}

auto test::RTDiffuse::GetID() const noexcept -> RTString 
{
    if (m_Properties.HasString("ID")){ 
        return m_Properties.GetString("ID");
    }
    return "";
}

auto test::RTDiffuse::GetProperties() const noexcept -> const RTProperties & 
{
    return m_Properties;
}

auto test::RTDiffuse::GetJsonAsData() const noexcept -> nlohmann::json 
{
	nlohmann::json data;
    data = GetProperties().GetJsonAsData();
	data["Type"] = GetTypeName();
	data["Plugin"] = GetPluginName();
	return data;
}

auto test::RTDiffuse::GetReflectance() const noexcept -> std::variant<RTTexturePtr, RTColor, RTFloat>
{
    if (m_Properties.HasTexture("Reflectance")) {
        return m_Properties.GetTexture("Reflectance");
    }
    if (m_Properties.HasColor("Reflectance")) {
        return m_Properties.GetColor("Reflectance");
    }
    if (m_Properties.HasFloat("Reflectance")) {
        return m_Properties.GetFloat("Reflectance");
    }
    return 1.0f;
}

void test::RTDiffuse::SetID(const RTString& id) noexcept
{
    m_Properties.SetString("ID", id);
}

void test::RTDiffuse::SetReflectance(const RTTexturePtr& texture) noexcept
{
    m_Properties.SetTexture("Reflectance", texture);
}

void test::RTDiffuse::SetReflectance(const RTColor& color) noexcept
{
    m_Properties.SetColor("Reflectance"  , color);
}

void test::RTDiffuse::SetReflectance(const RTFloat& fv) noexcept
{
    m_Properties.SetFloat("Reflectance"  , fv);
}
struct test::RTDiffuseReader::Impl
{
    std::weak_ptr<RTMaterialCache> matCache;
    std::weak_ptr<RTTextureCache > texCache;
};
test::RTDiffuseReader::RTDiffuseReader(const std::shared_ptr<RTMaterialCache>& matCache, const std::shared_ptr<RTTextureCache>& texCache) noexcept
{
    m_Impl = std::make_unique<test::RTDiffuseReader::Impl>();
    m_Impl->matCache = matCache;
    m_Impl->texCache = texCache;
}

auto test::RTDiffuseReader::GetPluginName() const noexcept -> RTString 
{
    return "Diffuse";
}

auto test::RTDiffuseReader::LoadJsonFromData(const nlohmann::json& json) noexcept -> RTMaterialPtr 
{
    if (!json.contains("Type") || !json["Type"].is_string() || json["Type"].get<std::string>() != "Material") {
        return nullptr;
    }

    if (!json.contains("Plugin") || !json["Plugin"].is_string() || json["Plugin"].get<std::string>() != "Diffuse") {
        return nullptr;
    }

    auto diffuse = std::make_shared<test::RTDiffuse>();

    if (!diffuse->m_Properties.LoadFloat("Reflectance", json) &&
        !diffuse->m_Properties.LoadColor("Reflectance", json) &&
        !diffuse->m_Properties.LoadTexture("Reflectance", json, m_Impl->texCache.lock())) {
        return nullptr;
    }
    return diffuse;
}

test::RTDiffuseReader::~RTDiffuseReader() noexcept
{
}
