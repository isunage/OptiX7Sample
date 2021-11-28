#include "RTColorTexture.h"

test::RTColorTexture::RTColorTexture() noexcept
{
}

auto test::RTColorTexture::GetTypeName() const noexcept -> RTString 
{
	return "Texture";
}

auto test::RTColorTexture::GetPluginName() const noexcept -> RTString 
{
	return "Color";
}

auto test::RTColorTexture::GetID() const noexcept -> RTString 
{
	if (m_Properties.HasString("ID")) {
		return m_Properties.GetString("ID");
	}
	else {
		return "";
	}
}

auto test::RTColorTexture::GetProperties() const noexcept -> const RTProperties & 
{
	return m_Properties;
}

auto test::RTColorTexture::GetJsonAsData() const noexcept -> nlohmann::json 
{
	nlohmann::json data;
	data = GetProperties().GetJsonAsData();
	data["Type"] = GetTypeName();
	data["Plugin"] = GetPluginName();
	return data;
}

auto test::RTColorTexture::GetColor() const noexcept -> RTColor
{
	if (m_Properties.HasColor("Color")) {
		return m_Properties.GetColor("Color");
	}
	else {
		return make_float3(0.0f);
	}
}

void test::RTColorTexture::SetColor(const RTColor& color) noexcept
{
	m_Properties.SetColor("Color", color);
}

void test::RTColorTexture::SetID(const RTString& id) noexcept
{
	m_Properties.SetString("ID", id);
}

struct test::RTColorTextureReader::Impl
{
	std::weak_ptr<RTTextureCache> texCache;
};

test::RTColorTextureReader::RTColorTextureReader(const std::shared_ptr<RTTextureCache>& texCache) noexcept
{
	m_Impl = std::make_unique<test::RTColorTextureReader::Impl>();
	m_Impl->texCache = texCache;
}

auto test::RTColorTextureReader::GetPluginName() const noexcept -> RTString 
{
	return "Color";
}

auto test::RTColorTextureReader::LoadJsonFromData(const nlohmann::json& json) noexcept -> RTTexturePtr 
{
	if (!json.contains("Type") || !json["Type"].is_string() || json["Type"].get<std::string>() != "Texture") {
		return nullptr;
	}

	if (!json.contains("Plugin") || !json["Plugin"].is_string() || json["Plugin"].get<std::string>() != "Color") {
		return nullptr;
	}

	auto color = std::make_shared<test::RTColorTexture>();
	if (!color->m_Properties.LoadColor("Color"   , json)) {
		return nullptr;
	}
	// if (color->m_Properties.LoadString("ID"      , json)) {
	// 	m_Impl->texCache.lock()->AddTexture(color);
	// }
	return color;
}

test::RTColorTextureReader::~RTColorTextureReader() noexcept
{
}
