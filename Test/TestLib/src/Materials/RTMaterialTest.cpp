#include <RTMaterialTest.h>
#include <TestLib/RTMaterial.h>
#include <TestLib/RTTexture.h>
#include <nlohmann/json.hpp>
#include <fstream>
#include <string>
#include <queue>
#include <iostream>
int main()
{
	auto jsonPath = test::GetMaterialTestBaseDir()/"RTMaterialTest.json";
	std::fstream jsonFile(jsonPath, std::ios::binary | std::ios::in);
	auto jsonStr = std::string(
		(std::istreambuf_iterator<char>(jsonFile)), (std::istreambuf_iterator<char>())
	);
	auto jsonData = nlohmann::json::parse(jsonStr);
	auto texCache = test::GetDefaultTextureCache();
	auto matCache = test::GetDefaultMaterialCache(texCache);
	std::queue<std::tuple<std::string, nlohmann::json>> jsonStack;
	for (auto& materialJson : jsonData.items()) {
		jsonStack.push({ materialJson.key(),materialJson.value() });
	}
	size_t i = 0;
	while (!jsonStack.empty()) {
		if (i > 255) { return -1; }
		auto top = jsonStack.front();
		jsonStack.pop();
		auto material = matCache->LoadJsonFromData(std::get<1>(top));
		std::cout << std::get<0>(top) << (material ? "  : SUCC" : ": FAIL") << std::endl;
		if (!material) {
			jsonStack.push(top);
		}
		++i;
	}

	return 0;
}