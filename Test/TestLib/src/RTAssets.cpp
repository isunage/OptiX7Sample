#include "../include/TestLib/RTAssets.h"

auto test::RTAssetManager::GetAsset(const std::string& keyName) const -> test::RTAssetPtr
{
	return m_Assets.at(keyName);
}

auto test::RTAssetManager::GetAssets() const -> const std::unordered_map<std::string, test::RTAssetPtr>&
{
	// TODO: return ステートメントをここに挿入します
	return m_Assets;
}

auto test::RTAssetManager::GetAssets() -> std::unordered_map<std::string, test::RTAssetPtr>&
{
	// TODO: return ステートメントをここに挿入します
	return m_Assets;
}

bool test::RTAssetManager::HasAsset(const std::string& keyName) const
{
	return m_Assets.count(keyName)>0;
}
