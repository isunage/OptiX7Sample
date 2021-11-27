#include "RTDeltaReflection.h"

test::RTDeltaReflection::RTDeltaReflection() noexcept : RTMaterial()
{
}

auto test::RTDeltaReflection::GetTypeName() const noexcept -> RTString
{
	return "Material";
}

auto test::RTDeltaReflection::GetPluginName() const noexcept -> RTString
{
	return "DeltaReflection";
}

auto test::RTDeltaReflection::GetID() const noexcept -> RTString 
{
    if (m_Properties.HasString("ID")){ 
        return m_Properties.GetString("ID");
    }
    return "";
}

auto test::RTDeltaReflection::GetProperties() const noexcept -> const RTProperties &
{
	return m_Properties;
}
 
auto test::RTDeltaReflection::GetJsonAsData() const noexcept ->       nlohmann::json 
{
	nlohmann::json data;
	data["Type"]       = GetTypeName();
	data["Plugin"]     = GetPluginName();
	data["Properties"] = GetProperties().GetJsonData();
	return data;
}

void test::RTDeltaReflection::SetID(const RTString& id) noexcept
{
	m_Properties.SetString("ID", id);
}

auto test::RTDeltaReflection::GetReflectance() const noexcept -> std::variant<RTTexturePtr, RTColor, RTFloat>
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

void test::RTDeltaReflection::SetReflectance(const RTTexturePtr& texture) noexcept
{
	m_Properties.SetTexture("Reflectance", texture);
}

void test::RTDeltaReflection::SetReflectance(const RTColor& color) noexcept
{
	m_Properties.SetColor("Reflectance", color);
}

void test::RTDeltaReflection::SetReflectance(const RTFloat& fv) noexcept
{
	m_Properties.SetFloat("Reflectance", fv);
}
struct test::RTDeltaReflectionReader::Impl
{
	std::weak_ptr<RTMaterialCache> matCache;
	std::weak_ptr<RTTextureCache > texCache;
};
test::RTDeltaReflectionReader::RTDeltaReflectionReader(const std::shared_ptr<RTMaterialCache>& matCache, const std::shared_ptr<RTTextureCache>& texCache) noexcept
{
	m_Impl = std::make_unique<RTDeltaReflectionReader::Impl>();
	m_Impl->matCache = matCache;
	m_Impl->texCache = texCache;
}

auto test::RTDeltaReflectionReader::GetPluginName() const noexcept -> RTString
{
	return "DeltaReflection";
}

auto test::RTDeltaReflectionReader::LoadJsonFromData(const nlohmann::json& json) noexcept -> RTMaterialPtr 
{
    if (!json.contains("Type") || !json["Type"].is_string() || json["Type"].get<std::string>() != "Material") {
        return nullptr;
    }

    if (!json.contains("Plugin") || !json["Plugin"].is_string() || json["Plugin"].get<std::string>() != "DeltaReflection") {
        return nullptr;
    }
    if (!json.contains("Properties") || !json["Properties"].is_object()) {
        return nullptr;
    }

    auto& propertiesJson = json["Properties"];
    auto reflectance = std::make_shared<test::RTDeltaReflection>();

    if (!reflectance->m_Properties.LoadFloat(  "Reflectance", propertiesJson) &&
        !reflectance->m_Properties.LoadColor(  "Reflectance", propertiesJson) &&
        !reflectance->m_Properties.LoadTexture("Reflectance", propertiesJson, m_Impl->texCache.lock())) {
        return nullptr;
    }
    return reflectance;
}

test::RTDeltaReflectionReader::~RTDeltaReflectionReader() noexcept
{
}
