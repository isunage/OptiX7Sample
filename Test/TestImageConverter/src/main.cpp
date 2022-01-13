#include <TestLib/RTUtils.h>
#include <filesystem>
int main(int argc, const char* argv[])
{
	if (argc < 4)
	{
		return -1;
	}
	std::string  inputFilename = argv[1];
	std::string         option = argv[2];
	if (option == "-o")
	{
		std::string outputFilename = argv[3];
		int wid, hei, com;
		auto inputPixels = test::LoadImgFromPathAsFloatData(inputFilename, wid, hei, com, 3);
		if (!inputPixels) {
			std::cout << "FATAL\n";
			return -7;
		}
		size_t imgSize = (size_t)wid * hei;
		auto outputPixels = std::unique_ptr<float3[]>(new float3[imgSize]);
		std::cout << wid << " " << hei << std::endl;
		std::array<float3, 9> arrData = {};
		for (int j = 0; j < hei; ++j) {
			for (int i = 0; i < wid; ++i) {
				size_t arrCount = 0;
				for (int t = std::max(j - 1, 0); t <= std::min(j + 1, hei - 1); ++t)
				{
					for (int s = std::max(i - 1, 0); s <= std::min(i + 1, wid - 1); ++s)
					{
						size_t imgIdx = (size_t)t * wid + s;
						arrData[arrCount] = make_float3(inputPixels[3 * imgIdx + 0], inputPixels[3 * imgIdx + 1], inputPixels[3 * imgIdx + 2]);
						++arrCount;
					}
				}

				size_t imgIdx = (size_t)(hei - 1 - j) * wid + i;
				std::sort(arrData.begin(), arrData.begin() + arrCount, [](const auto& a, const auto& b) {
					auto getLuminance = [](const auto& c)->float {
						return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
					};
					return getLuminance(a) > getLuminance(b);
					});

				outputPixels[imgIdx] = arrData[arrCount / 2];
			}
		}
		return test::SaveExrImgFromData(outputFilename.c_str(), outputPixels.get(), wid, hei, 1) == true ? 0 : -3;
	}
	if (option == "-g")
	{
		std::string outputFilename = argv[3];
		int wid, hei, com;
		auto inputPixels = test::LoadImgFromPathAsFloatData(inputFilename, wid, hei, com, 3);
		if (!inputPixels) {
			std::cout << "FATAL\n";
			return -7;
		}
		size_t imgSize = (size_t)wid * hei;
		auto outputPixels = std::unique_ptr<float3[]>(new float3[imgSize]);
		std::cout << wid << " " << hei << std::endl;
		std::array<float3, 9> arrData = {};
		auto getLuminance = [](const auto& c)->float {
			return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
		};
		for (int j = 0; j < hei; ++j) {
			for (int i = 0; i < wid; ++i) {

				auto inCol1 = make_float3(inputPixels[3 * (wid * j + i) + 0], inputPixels[3 * (wid * j + i) + 1], inputPixels[3 * (wid * j + i) + 2]);
				size_t imgIdx = (size_t)(hei - 1 - j) * wid + i;
				float luminance = getLuminance(inCol1);
				outputPixels[imgIdx] = make_float3(luminance, luminance, luminance);
			}
		}
		return test::SaveExrImgFromData(outputFilename.c_str(), outputPixels.get(), wid, hei, 1) == true ? 0 : -3;
	}
	if (argc > 3 && option == "-m") {
		bool saveComp = (argc > 6) && std::string(argv[4]) == "-o";
		std::cout << "#OK!\n";
		std::string  inputFilename2 = argv[3];
		int wid, hei, com;
		auto inputPixels1 = test::LoadImgFromPathAsFloatData(inputFilename, wid, hei, com, 3);
		if (!inputPixels1) {
			std::cout << "FATAL\n";
			return -7;
		}
		auto inputPixels2 = test::LoadImgFromPathAsFloatData(inputFilename2, wid, hei, com, 3);
		if (!inputPixels2) {
			std::cout << "FATAL\n";
			return -7;
		}
		std::cout << "#OK!\n";
		size_t imgSize = (size_t)wid * hei;
		auto getLuminance = [](const float3& c)->float {
			return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
		};
		std::string outputFilename;
		std::unique_ptr<float3[]> outputPixels;
		if (saveComp) { 
			outputFilename = argv[5];
			outputPixels = std::unique_ptr<float3[]>(new float3[imgSize]);
		}
		float rel_mse = 0.0f;
		for (int j = 0; j < hei; ++j) {
			for (int i = 0; i < wid; ++i) {
				auto inCol1 = make_float3(inputPixels1[3 * (wid * j + i) + 0], inputPixels1[3 * (wid * j + i) + 1], inputPixels1[3 * (wid * j + i) + 2]);
				auto inCol2 = make_float3(inputPixels2[3 * (wid * j + i) + 0], inputPixels2[3 * (wid * j + i) + 1], inputPixels2[3 * (wid * j + i) + 2]);
				float inLum1 = getLuminance(inCol1);
				float inLum2 = getLuminance(inCol2);
				float comp   = fabsf(inLum1 /(inLum1 +1e-15f) -1.0f);
				size_t imgIdx = (size_t)(hei - 1 - j) * wid + i;
				if (saveComp) {
					outputPixels[imgIdx] = make_float3(comp, comp, comp);
				}
				rel_mse += comp * comp;
			}
		}
		std::cout << "Rel MSE: " << rel_mse / static_cast<float>(wid * hei) << std::endl;
		if (saveComp) {
			return test::SaveExrImgFromData(outputFilename.c_str(), outputPixels.get(), wid, hei, 1) == true ? 0 : -3;
		}
		return 0;
	}
}