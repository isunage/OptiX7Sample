#include <RTTextureTest.h>
#include <TestLib/RTTexture.h>
#include <nlohmann/json.hpp>
#include <fstream>
#include <string>
#include <queue>
#include <iostream>
int main()
{
	auto phongPath      = test::GetTextureTestBaseDir() / "RTTextureTest.json";
	std::fstream phongJsonFile(phongPath, std::ios::binary | std::ios::in);
	auto phongJsonStr   = std::string(
		(std::istreambuf_iterator<char>(phongJsonFile)), (std::istreambuf_iterator<char>())
	);
	auto phongJsonData  = nlohmann::json::parse(phongJsonStr);
	auto texCache       = test::GetDefaultTextureCache();
	std::queue<std::tuple<std::string, nlohmann::json>> jsonStack;
	for (auto& textureJson : phongJsonData.items()) {
		jsonStack.push({ textureJson.key(),textureJson.value() });
	}
	while (!jsonStack.empty()) {
		auto top     = jsonStack.front();
		jsonStack.pop();
		auto texture = texCache->LoadJsonFromData(std::get<1>(top));
		std::cout << std::get<0>(top) << (texture ? "  : SUCC" : ": FAIL") << std::endl;
		if (!texture) {
			jsonStack.push(top);
		}
	}
	return 0;
}