#include "RTShapeTest.h"
#include <TestLib/RTTexture.h>
#include <nlohmann/json.hpp>
#include <fstream>
#include <string>
#include <iostream>
int main()
{
	auto phongPath    = test::GetShapeTestBaseDir() / "Textures\\RTCheckTexture.json";
	std::fstream phongJsonFile(phongPath, std::ios::binary | std::ios::in);
	auto phongJsonStr = std::string(
		(std::istreambuf_iterator<char>(phongJsonFile)), (std::istreambuf_iterator<char>())
	);
	auto phongJsonData  = nlohmann::json::parse(phongJsonStr);
	auto texCache       = test::GetDefaultTextureCache();
	auto phongMaterial0 = texCache->LoadJsonFromData(phongJsonData["Sample0"]);
	auto phongMaterial1 = texCache->LoadJsonFromData(phongJsonData["Sample1"]);
	auto phongMaterial2 = texCache->LoadJsonFromData(phongJsonData["Sample2"]);

	return 0;
}