#include "../include/TestLib/Assets/ImgAssets.h"
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
bool test::assets::ImgAsset::Load(const rtlib::ext::VariableMap& params)
{
	if (m_Valid)
	{
		Free();
	}
	m_Valid   = true;
	auto path = params.GetString("path");
	int x, y, comp;
	auto img  = stbi_load(path.c_str(), &x, &y, &comp, 4);
	if (!img) {
		return false;
	}
	auto pixels = std::vector<uchar4>(x * y);
	for (unsigned int j = 0; j < y; ++j)
	{
		std::memcpy(&pixels[x * (y-1-j)], &img[4 * (x * j)], sizeof(uchar4) * x);
	}
	m_Image = rtlib::ext::CustomImage2D(x, y, pixels);
	return true;
}

void test::assets::ImgAsset::Free()
{
	m_Valid = false;
	m_Image.Reset();
}

bool test::assets::ImgAsset::IsValid() const
{
	return m_Valid;
}

bool test::assets::ImgAssetManager::LoadAsset(const std::string& key, const rtlib::ext::VariableMap& params)
{
	auto asset = test::assets::ImgAsset::New();
	bool cond  = asset->Load(params);
	if (cond) {
		GetAssets()[key] = asset;
	}
	return cond;
}

void test::assets::ImgAssetManager::FreeAsset(const std::string& key)
{
	if (HasAsset(key))
	{
		auto& asset = GetAssets().at(key);
		asset->Free();
		GetAssets().erase(key);
	}
}
