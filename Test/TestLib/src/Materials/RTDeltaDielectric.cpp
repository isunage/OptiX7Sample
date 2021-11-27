#include "RTDeltaDielectric.h"

test::RTDeltaDielectric::RTDeltaDielectric() noexcept
{
}

auto test::RTDeltaDielectric::GetTypeName() const noexcept -> RTString 
{
	return "Material";
}

auto test::RTDeltaDielectric::GetPluginName() const noexcept -> RTString 
{

	return "DeltaDielectric";
}

auto test::RTDeltaDielectric::GetID() const noexcept -> RTString 
{
    if (m_Properties.HasString("ID")){ 
        return m_Properties.GetString("ID");
    }
    return "";
}

auto test::RTDeltaDielectric::GetProperties() const noexcept -> const RTProperties & 
{
	return m_Properties;
}

auto test::RTDeltaDielectric::GetJsonAsData() const noexcept ->       nlohmann::json
{
	nlohmann::json data;
	data["Type"] = GetTypeName();
	data["Plugin"] = GetPluginName();
	data["Properties"] = GetProperties().GetJsonData();
	return data;
}

auto test::RTDeltaDielectric::GetReflectance() const noexcept -> std::variant<RTTexturePtr, RTColor, RTFloat>
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

auto test::RTDeltaDielectric::GetTransmittance() const noexcept -> std::variant<RTColor, RTFloat>
{
	if (m_Properties.HasColor("Transmittance")) {
		return m_Properties.GetColor("Transmittance");
	}
	if (m_Properties.HasFloat("Transmittance")) {
		return m_Properties.GetFloat("Transmittance");
	}
	return 0.0f;
}

auto test::RTDeltaDielectric::GetIOR() const noexcept -> RTFloat
{
	if (m_Properties.HasFloat("IOR")) {
		return m_Properties.GetFloat("IOR");
	}
	return 1.0f;
}

void test::RTDeltaDielectric::SetID(const RTString& id) noexcept
{
	return m_Properties.SetString("ID",id);
}

void test::RTDeltaDielectric::SetReflectance(const RTTexturePtr& texture) noexcept
{
	m_Properties.SetTexture("Reflectance", texture);
}

void test::RTDeltaDielectric::SetReflectance(const RTColor& color) noexcept
{
	m_Properties.SetColor("Reflectance", color);
}

void test::RTDeltaDielectric::SetReflectance(const RTFloat& fv) noexcept
{
	m_Properties.SetFloat("Reflectance", fv);
}

void test::RTDeltaDielectric::SetTransmittance(const RTColor& color) noexcept
{
	m_Properties.SetColor("Transmittance", color);
}

void test::RTDeltaDielectric::SetTransmittance(const RTFloat& fv) noexcept
{
	m_Properties.SetFloat("Transmittance", fv);
}

void test::RTDeltaDielectric::SetIOR(const RTFloat& fv) noexcept
{
	m_Properties.SetFloat("IOR", fv);
}
struct test::RTDeltaDielectricReader::Impl
{
	std::weak_ptr<RTMaterialCache> matCache;
	std::weak_ptr<RTTextureCache > texCache;
};
test::RTDeltaDielectricReader::RTDeltaDielectricReader(const std::shared_ptr<RTMaterialCache>& matCache, const std::shared_ptr<RTTextureCache>& texCache) noexcept
{
	m_Impl           = std::make_unique<test::RTDeltaDielectricReader::Impl>();
	m_Impl->matCache = matCache;
	m_Impl->texCache = texCache;
}

auto test::RTDeltaDielectricReader::GetPluginName() const noexcept -> RTString
{

	return "DeltaDielectric";
}


auto test::RTDeltaDielectricReader::LoadJsonFromData(const nlohmann::json& json) noexcept -> RTMaterialPtr 
{
	if (!json.contains("Type") || !json["Type"].is_string() || json["Type"].get<std::string>() != "Material") {

		return nullptr;
	}

	if (!json.contains("Plugin") || !json["Plugin"].is_string() || json["Plugin"].get<std::string>() != "DeltaDielectric") {
		return nullptr;
	}

	if (!json.contains("Properties") || !json["Properties"].is_object()) {
		return nullptr;
	}

	auto& propertiesJson = json["Properties"];
	auto dielectric = std::make_shared<test::RTDeltaDielectric>();

	if (!dielectric->m_Properties.LoadFloat("Reflectance", propertiesJson) &&
		!dielectric->m_Properties.LoadColor("Reflectance", propertiesJson) &&
		!dielectric->m_Properties.LoadTexture("Reflectance", propertiesJson, m_Impl->texCache.lock())) {
		return nullptr;
	}
	if (!dielectric->m_Properties.LoadFloat(  "Transmittance", propertiesJson) &&
		!dielectric->m_Properties.LoadColor(  "Transmittance", propertiesJson) &&
		!dielectric->m_Properties.LoadTexture("Transmittance", propertiesJson, m_Impl->texCache.lock())) {
		return nullptr;
	}
	if (!dielectric->m_Properties.LoadFloat("IOR", propertiesJson)) {
		return nullptr;
	}
	return dielectric;
}

test::RTDeltaDielectricReader::~RTDeltaDielectricReader() noexcept
{
}
