#include "RTCheckTexture.h"
test::RTCheckTexture::RTCheckTexture()noexcept {}

auto test::RTCheckTexture::GetTypeName() const noexcept -> RTString {
	return "Texture";
};

auto test::RTCheckTexture::GetPluginName() const noexcept -> RTString {
	return "Check";
}
auto test::RTCheckTexture::GetID()         const noexcept -> RTString {
	if (m_Properties.HasString("ID")) {
		return m_Properties.GetString("ID");
	}
	else {
		return "";
	}
}
auto test::RTCheckTexture::GetProperties() const noexcept -> const RTProperties& {
	return m_Properties;
}
auto test::RTCheckTexture::GetJsonAsData() const noexcept ->       nlohmann::json {
	nlohmann::json data;
	data = GetProperties().GetJsonAsData();
	data["Type"] = GetTypeName();
	data["Plugin"] = GetPluginName();
	return data;
}
auto test::RTCheckTexture::GetTexture0()const noexcept -> std::variant<RTColor, RTTexturePtr> {
	if (m_Properties.HasTexture("Texture0"))
	{
		return m_Properties.GetTexture("Texture0");
	}
	if (m_Properties.HasColor("Texture0"))
	{
		return m_Properties.GetColor("Texture0");
	}
	else {
		return make_float3(0.0f);
	}
}
auto test::RTCheckTexture::GetTexture1()const noexcept -> std::variant<RTColor, RTTexturePtr> {
	if (m_Properties.HasTexture("Texture1"))
	{
		return m_Properties.GetTexture("Texture1");
	}
	if (m_Properties.HasColor("Texture1"))
	{
		return m_Properties.GetColor("Texture1");
	}
	else {
		return make_float3(0.0f);
	}
}
auto test::RTCheckTexture::GetFrequency()const noexcept -> RTFloat {
	return m_Properties.GetFloat("Frequency");
}
void test::RTCheckTexture::SetTexture0(const RTColor& color)noexcept {
	m_Properties.SetColor(  "Texture0", color);
}
void test::RTCheckTexture::SetTexture0(const RTTexturePtr& texture)noexcept {
	m_Properties.SetTexture("Texture0", texture);
}
void test::RTCheckTexture::SetTexture1(const RTColor& color)noexcept {
	m_Properties.SetColor(  "Texture1", color);
}
void test::RTCheckTexture::SetTexture1(const RTTexturePtr& texture)noexcept {
	m_Properties.SetTexture("Texture1", texture);
}
void test::RTCheckTexture::SetID(const RTString& id)  noexcept {
	m_Properties.SetString(       "ID", id);
}
void test::RTCheckTexture::SetFrequency(const RTFloat& freq) noexcept {
	m_Properties.SetFloat("Frequency", freq);
}

struct test::RTCheckTextureReader::Impl
{
	std::weak_ptr< RTTextureCache> texCache;
};

test::RTCheckTextureReader::RTCheckTextureReader(const std::shared_ptr< RTTextureCache>& texCache)noexcept {
	m_Impl		     = std::make_unique<test::RTCheckTextureReader::Impl>();
	m_Impl->texCache = texCache;
}

auto test::RTCheckTextureReader::GetPluginName() const noexcept -> RTString
{
	return "Check";
}

auto test::RTCheckTextureReader::LoadJsonFromData(const nlohmann::json& json)noexcept -> RTTexturePtr {
	if (!json.contains("Type") || !json["Type"].is_string() || json["Type"].get<std::string>() != "Texture") {
		return nullptr;
	}

	if (!json.contains("Plugin") || !json["Plugin"].is_string() || json["Plugin"].get<std::string>() != "Check") {
		return nullptr;
	}

	auto check = std::make_shared<test::RTCheckTexture>();
	if (!check->m_Properties.LoadColor  ("Texture0", json)&&
		!check->m_Properties.LoadTexture("Texture0", json, m_Impl->texCache.lock())) {
		return nullptr;
	}
	if (!check->m_Properties.LoadColor  ("Texture1", json) &&
		!check->m_Properties.LoadTexture("Texture1", json, m_Impl->texCache.lock())) {
		return nullptr;
	}
	if (!check->m_Properties.LoadFloat("Frequency" , json)) {
		return nullptr;
	}
	// if (check->m_Properties.LoadString("ID", json)) {
	// 	m_Impl->texCache.lock()->AddTexture(check);
	// }
	return check;
}
test::RTCheckTextureReader::~RTCheckTextureReader()noexcept {

}