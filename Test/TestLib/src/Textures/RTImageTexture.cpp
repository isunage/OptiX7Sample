#include "RTImageTexture.h"

test::RTImageTexture::RTImageTexture() noexcept
{
}

auto test::RTImageTexture::GetTypeName() const noexcept -> RTString 
{
	return "Texture";
}

auto test::RTImageTexture::GetPluginName() const noexcept -> RTString 
{
	return "Image";
}

auto test::RTImageTexture::GetID() const noexcept -> RTString 
{
	if (m_Properties.HasString("ID")) {
		return m_Properties.GetString("ID");
	}
	else {
		return "";
	}
}

auto test::RTImageTexture::GetProperties() const noexcept -> const RTProperties & 
{
	return m_Properties;
}

auto test::RTImageTexture::GetJsonAsData() const noexcept -> nlohmann::json 
{
	nlohmann::json data;
	data["Type"]   = GetTypeName();
	data["Plugin"] = GetPluginName();
	data["Properties"] = GetProperties().GetJsonData();
	return data;
}

auto test::RTImageTexture::GetFilename() const noexcept -> RTString
{
	if (m_Properties.HasString("Filename")) {
		return m_Properties.GetString("Filename");
	}
	else {
		return "";
	}
}

auto test::RTImageTexture::GetMagFilter() const noexcept -> RTString
{
	if (m_Properties.HasString("MagFilter")) {
		return m_Properties.GetString("MagFilter");
	}
	else {
		//Default
		return "";
	}
}

auto test::RTImageTexture::GetMinFilter() const noexcept -> RTString
{
	if (m_Properties.HasString("MinFilter")) {
		return m_Properties.GetString("MinFilter");
	}
	else {
		//Default
		return "";
	}
}

auto test::RTImageTexture::GetWarpModeS() const noexcept -> RTString
{
	if (m_Properties.HasString("WarpModeS")) {
		return m_Properties.GetString("WarpModeS");
	}
	else {
		//Default
		return "";
	}
}

auto test::RTImageTexture::GetWarpModeT() const noexcept -> RTString
{
	if (m_Properties.HasString("WarpModeT")) {
		return m_Properties.GetString("WarpModeT");
	}
	else {
		return "";
	}
}

void test::RTImageTexture::SetID(const RTString& id) noexcept
{
	m_Properties.SetString("ID", id);
}

void test::RTImageTexture::SetFilename(const RTString& filename) noexcept
{
	m_Properties.SetString("Filename", filename);
}

void test::RTImageTexture::SetMagFilter(const RTString& filtername) noexcept
{
	m_Properties.SetString("MagFilter", filtername);
}

void test::RTImageTexture::SetMinFilter(const RTString& filtername) noexcept
{
	m_Properties.SetString("MinFilter", filtername);
}

void test::RTImageTexture::SetWarpModeS(const RTString& warpModeS) noexcept
{
	m_Properties.SetString("WarpModeS", warpModeS);
}

void test::RTImageTexture::SetWarpModeT(const RTString& warpModeT) noexcept
{
	m_Properties.SetString("WarpModeT", warpModeT);
}

struct test::RTImageTextureReader::Impl {
	std::weak_ptr<RTTextureCache> texCache;
};

test::RTImageTextureReader::RTImageTextureReader(const std::shared_ptr<RTTextureCache>& texCache) noexcept
{
	m_Impl			 = std::make_unique<RTImageTextureReader::Impl>();
	m_Impl->texCache = texCache;
}

auto test::RTImageTextureReader::GetPluginName() const noexcept -> RTString 
{
	return "Image";
}

auto test::RTImageTextureReader::LoadJsonFromData(const nlohmann::json& json) noexcept -> RTTexturePtr 
{
	if (!json.contains("Type") || !json["Type"].is_string() || json["Type"].get<std::string>() != "Texture") {
		return nullptr;
	}

	if (!json.contains("Plugin") || !json["Plugin"].is_string() || json["Plugin"].get<std::string>() != "Image") {
		return nullptr;
	}
	if (!json.contains("Properties") || !json["Properties"].is_object()) {
		return nullptr;
	}

	auto& propertiesJson = json["Properties"];
	auto image = std::make_shared<test::RTImageTexture>();
	if (!image->m_Properties.LoadString("Filename" ,propertiesJson)) {
		return nullptr;
	}
	if (!image->m_Properties.LoadString("MagFilter", propertiesJson)) {
		return nullptr;
	}
	if (!image->m_Properties.LoadString("MinFilter", propertiesJson)) {
		return nullptr;
	}
	if (!image->m_Properties.LoadString("WarpModeS", propertiesJson)) {
		return nullptr;
	}
	if (!image->m_Properties.LoadString("WarpModeT", propertiesJson)) {
		return nullptr;
	}
	// if (image->m_Properties.LoadString("ID", propertiesJson)) {
	// 	m_Impl->texCache.lock()->AddTexture(image);
	// }
	return image;
}

test::RTImageTextureReader::~RTImageTextureReader() noexcept
{
}
