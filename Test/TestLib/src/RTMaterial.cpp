#include "..\include\TestLib\RTMaterial.h"
#include "Materials/RTPhong.h"
#include "Materials/RTDiffuse.h"
#include "Materials/RTDeltaDielectric.h"
#include "Materials/RTDeltaReflection.h"
#include <TestLib/RTProperties.h>
test::RTMaterialCache::RTMaterialCache() noexcept
{
}

bool test::RTMaterialCache::AddMaterial(const RTMaterialPtr& material) noexcept
{
	if (!material) { return false; }
	if (!material->GetProperties().HasString("ID")) {
		return false;

	}
	if (material->GetProperties().GetString("ID") == "") {
		return false;
	}
	m_Materials[material->GetProperties().GetString("ID")] = material;
	return true;
}

bool test::RTMaterialCache::HasMaterial(const std::string& id) const noexcept
{
	return m_Materials.count(id) > 0;
}

auto test::RTMaterialCache::GetMaterial(const std::string& id) const -> RTMaterialPtr
{
	return m_Materials.at(id);
}

bool test::RTMaterialCache::AddReader(const RTMaterialReaderPtr& reader) noexcept
{
	if (m_Readers.count(reader->GetPluginName()) > 0) {
		return false;
	}
	else {
		m_Readers[reader->GetPluginName()] = reader;
		return true;
	}
}

bool test::RTMaterialCache::HasReader(const std::string& id) const noexcept
{
	return m_Readers.count(id) > 0;
}

auto test::RTMaterialCache::GetReader(const std::string& id) const -> RTMaterialReaderPtr
{
	return m_Readers.at(id);
}

auto test::RTMaterialCache::LoadJsonFromData(const nlohmann::json& json) noexcept -> RTMaterialPtr
{
	if (json.is_string()) {
		if (HasMaterial(json.get<std::string>())) {
			return GetMaterial(json.get<std::string>());
		}
		else {
			//‚½‚¾‚µA‚Ü‚¾“Ç‚Ýž‚Ü‚ê‚Ä‚¢‚È‚¢‰Â”\«‚ª‚ ‚é
			return nullptr;
		}
	}
	else if (json.is_object()) {
		RTMaterialPtr ptr;
		for (auto& [id, reader] : m_Readers) {
			if (ptr = reader->LoadJsonFromData(json)) {
				return ptr;
			}
		}
	}
	return nullptr;
}

test::RTMaterialCache::~RTMaterialCache() noexcept
{
}

auto test::GetDefaultMaterialCache(const std::shared_ptr<RTTextureCache>& texCache) noexcept -> std::shared_ptr<RTMaterialCache>
{
	auto matCache = std::make_shared<RTMaterialCache>();
	matCache->AddReader(std::make_shared<RTPhongReader>(matCache, texCache));
	matCache->AddReader(std::make_shared<RTDiffuseReader>(matCache, texCache));
	matCache->AddReader(std::make_shared<RTDeltaDielectricReader>(matCache, texCache));
	matCache->AddReader(std::make_shared<RTDeltaReflectionReader>(matCache, texCache));
	return matCache;
}
