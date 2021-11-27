#include "..\include\TestLib\RTTexture.h"
#include "Textures/RTImageTexture.h"
#include "Textures/RTColorTexture.h"
#include "Textures/RTCheckTexture.h"
#include "Textures/RTBlendTexture.h"
#include <TestLib/RTProperties.h>
test::RTTextureCache::RTTextureCache() noexcept
{
}

bool test::RTTextureCache::AddTexture(const RTTexturePtr& texture) noexcept
{
	if (!texture) { return false; }
	if (!texture->GetProperties().HasString("ID")) {
		return false;
	}
	m_Textures[texture->GetProperties().GetString("ID")] = texture;
	return true;
}

bool test::RTTextureCache::HasTexture(const std::string& id) const noexcept
{
	return m_Textures.count(id);
}

auto test::RTTextureCache::GetTexture(const std::string& id) const -> RTTexturePtr
{
	return m_Textures.at(id);
}

bool test::RTTextureCache::AddReader(const RTTextureReaderPtr& reader) noexcept
{
	if (m_Readers.count(reader->GetPluginName()) > 0) {
		return false;
	}
	else {
		m_Readers[reader->GetPluginName()] = reader;
		return true;
	}
}

bool test::RTTextureCache::HasReader(const std::string& id) const noexcept
{
	return m_Readers.count(id) > 0;
}

auto test::RTTextureCache::GetReader(const std::string& id) const -> RTTextureReaderPtr
{
	return m_Readers.at(id);
}

auto test::RTTextureCache::LoadJsonFromData(const nlohmann::json& json) noexcept -> RTTexturePtr
{
	if (json.is_string()) {
		if (HasTexture(json.get<std::string>())) {
			return GetTexture(json.get<std::string>());
		}
		else {
			//‚½‚¾‚µA‚Ü‚¾“Ç‚Ýž‚Ü‚ê‚Ä‚¢‚È‚¢‰Â”\«‚ª‚ ‚é
			return nullptr;
		}
	}
	else if(json.is_object()){
		RTTexturePtr ptr;
		for (auto& [id, reader] : m_Readers) {
			if (ptr = reader->LoadJsonFromData(json)) {
				return ptr;
			}
		}
	}
	return nullptr;
}

test::RTTextureCache::~RTTextureCache() noexcept
{
}

auto test::GetDefaultTextureCache() noexcept -> std::shared_ptr<RTTextureCache>
{
	auto cache = std::make_shared<RTTextureCache>();
	cache->AddReader(std::make_shared<test::RTImageTextureReader>(cache));
	cache->AddReader(std::make_shared<test::RTColorTextureReader>(cache));
	cache->AddReader(std::make_shared<test::RTCheckTextureReader>(cache));
	cache->AddReader(std::make_shared<test::RTBlendTextureReader>(cache));
	return cache;
}
