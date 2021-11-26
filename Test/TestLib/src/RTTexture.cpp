#include "..\include\TestLib\RTTexture.h"
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
	m_BaseMap[texture->GetProperties().GetString("ID")] = texture;
}

bool test::RTTextureCache::HasTexture(const std::string& id) const noexcept
{
	return m_BaseMap.count(id);
}

auto test::RTTextureCache::GetTexture(const std::string& id) const -> RTTexturePtr
{
	return m_BaseMap.at(id);
}

test::RTTextureCache::~RTTextureCache() noexcept
{
}
