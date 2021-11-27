#include <RTCameraTest.h>
#include <RTPinhole.h>
#include <nlohmann/json.hpp>
#include <fstream>
#include <string>
#include <queue>
#include <iostream>
int main()
{
	auto jsonPath = test::GetCameraTestBaseDir() / "RTCameraTest.json";
	std::fstream jsonFile(jsonPath, std::ios::binary | std::ios::in);
	auto jsonStr = std::string(
		(std::istreambuf_iterator<char>(jsonFile)), (std::istreambuf_iterator<char>())
	);
	auto jsonData = nlohmann::json::parse(jsonStr);
	auto camCache = test::GetDefaultCameraCache();
	std::queue<std::tuple<std::string, nlohmann::json>> jsonStack;
	for (auto& materialJson : jsonData.items()) {
		jsonStack.push({ materialJson.key(),materialJson.value() });
	}
	size_t i = 0;
	while (!jsonStack.empty()) {
		if (i > 255) { return -1; }
		auto top = jsonStack.front();
		jsonStack.pop();
		auto camera = camCache->LoadJsonFromData(std::get<1>(top));
		std::cout << std::get<0>(top) << (camera ? "  : SUCC" : ": FAIL") << std::endl;
		if (!camera) {
			jsonStack.push(top);
		}
		++i;
	}

	return 0;
}