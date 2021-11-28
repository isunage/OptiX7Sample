#include "RTBlendTexture.h"
test::RTBlendTexture::RTBlendTexture()noexcept {}

auto test::RTBlendTexture::GetTypeName() const noexcept -> RTString {
	return "Texture";
};

auto test::RTBlendTexture::GetPluginName() const noexcept -> RTString {
	return "Blend";
}
auto test::RTBlendTexture::GetID()         const noexcept -> RTString {
	if (m_Properties.HasString("ID")) {
		return m_Properties.GetString("ID");
	}
	else {
		return "";
	}
}
auto test::RTBlendTexture::GetProperties() const noexcept -> const RTProperties& {
	return m_Properties;
}
auto test::RTBlendTexture::GetJsonAsData() const noexcept ->       nlohmann::json {
	nlohmann::json data;
	data               = GetProperties().GetJsonAsData();
	data["Type"]       = GetTypeName();
	data["Plugin"]     = GetPluginName();
	return data;
}
void test::RTBlendTexture::SetID(const std::string& id) noexcept
{
}
auto test::RTBlendTexture::GetTexture0()const noexcept -> std::variant<RTColor, RTTexturePtr> {
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
auto test::RTBlendTexture::GetTexture1()const noexcept -> std::variant<RTColor, RTTexturePtr> {
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

auto test::RTBlendTexture::GetBlendMode() const noexcept -> RTString
{
	if (m_Properties.HasString("BlendMode")) {
		return m_Properties.GetString("BlendMode");
	}
	else {
		return "Mul";
	}
}

void test::RTBlendTexture::SetTexture0(const RTColor& color)noexcept {
	m_Properties.SetColor(  "Texture0", color);
}
void test::RTBlendTexture::SetTexture0(const RTTexturePtr& texture)noexcept {
	m_Properties.SetTexture("Texture0", texture);
}
void test::RTBlendTexture::SetTexture1(const RTColor& color)noexcept {
	m_Properties.SetColor(  "Texture1", color);
}
void test::RTBlendTexture::SetTexture1(const RTTexturePtr& texture)noexcept {
	m_Properties.SetTexture("Texture1", texture);
}
void test::RTBlendTexture::SetBlendMode(const RTString& mode) noexcept
{
	m_Properties.SetString("BlendMode", mode);
}

struct test::RTBlendTextureReader::Impl
{
	std::weak_ptr< RTTextureCache> texCache;
};

test::RTBlendTextureReader::RTBlendTextureReader(const std::shared_ptr< RTTextureCache>& texCache)noexcept {
	m_Impl		     = std::make_unique<test::RTBlendTextureReader::Impl>();
	m_Impl->texCache = texCache;
}

auto test::RTBlendTextureReader::GetPluginName() const noexcept -> RTString
{
	return "Blend";
}

auto test::RTBlendTextureReader::LoadJsonFromData(const nlohmann::json& json)noexcept -> RTTexturePtr {
	if (!json.contains("Type") || !json["Type"].is_string() || json["Type"].get<std::string>() != "Texture") {
		return nullptr;
	}

	if (!json.contains("Plugin") || !json["Plugin"].is_string() || json["Plugin"].get<std::string>() != "Blend") {
		return nullptr;
	}
	auto blend = std::make_shared<test::RTBlendTexture>();
	if (!blend->m_Properties.LoadColor  ("Texture0", json)&&
		!blend->m_Properties.LoadTexture("Texture0", json, m_Impl->texCache.lock())) {
		return nullptr;
	}
	if (!blend->m_Properties.LoadColor  ("Texture1", json) &&
		!blend->m_Properties.LoadTexture("Texture1", json, m_Impl->texCache.lock())) {
		return nullptr;
	}

	if (!blend->m_Properties.LoadString("BlendMode", json)) {
		return nullptr;
	}
	// if (check->m_Properties.LoadString("ID", json)) {
	// 	m_Impl->texCache.lock()->AddTexture(check);
	// }
	return blend;
}
test::RTBlendTextureReader::~RTBlendTextureReader()noexcept {

}