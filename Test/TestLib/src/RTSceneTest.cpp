#include <RTSceneTest.h>
#include <TestLib/RTScene.h>
#include <nlohmann/json.hpp>
#include <fstream>
#include <string>
#include <queue>
#include <iostream>
int main()
{
	auto jsonPath = test::GetSceneTestBaseDir()/"RTSceneTest.json";
	std::fstream jsonFile(jsonPath, std::ios::binary | std::ios::in);
	auto jsonStr = std::string(
		(std::istreambuf_iterator<char>(jsonFile)), (std::istreambuf_iterator<char>())
	);
	auto jsonData    = nlohmann::json::parse(jsonStr);
	auto sceneReader = std::make_shared<test::RTSceneReader>();
	auto scene = sceneReader->LoadJsonFromData(jsonData);
	if (!scene) {
		return -1;
	}
	std::cout << scene->GetJsonAsString() << std::endl;
	return 0;
}