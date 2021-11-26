#include "RTShapeTest.h"
#include "Materials/RTPhong.h"
#include "Materials/RTDiffuse.h"
#include "Materials/RTDeltaReflection.h"
#include "Materials/RTDeltaDielectric.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <string>
#include <iostream>
int main()
{
	auto phongPath    = test::GetShapeTestBaseDir() / "Materials\\RTDeltaDielectric.json";
	std::fstream phongJsonFile(phongPath, std::ios::binary | std::ios::in);
	auto phongJsonStr = std::string(
		(std::istreambuf_iterator<char>(phongJsonFile)), (std::istreambuf_iterator<char>())
	);
	auto phongJsonData = nlohmann::json::parse(phongJsonStr);
	auto texCache	   = std::make_shared<test::RTTextureCache> ();
	auto matCache	   = std::make_shared<test::RTMaterialCache>();
	auto phongReader   = std::make_shared<test::RTDeltaDielectricReader>(matCache, texCache);
	auto phongMaterial = phongReader->LoadJsonFromData(phongJsonData.front());

	return 0.0f;
}