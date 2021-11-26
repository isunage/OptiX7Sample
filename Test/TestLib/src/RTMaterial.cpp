#include "..\include\TestLib\RTMaterial.h"
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
	m_BaseMap[material->GetProperties().GetString("ID")] = material;
}

bool test::RTMaterialCache::HasMaterial(const std::string& id) const noexcept
{
	return m_BaseMap.count(id) > 0;
}

auto test::RTMaterialCache::GetMaterial(const std::string& id) const -> RTMaterialPtr
{
	return m_BaseMap.at(id);
}

test::RTMaterialCache::~RTMaterialCache() noexcept
{
}
