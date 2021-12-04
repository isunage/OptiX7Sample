#define  STB_IMAGE_IMPLEMENTATION
#include <TestLib/RTUtils.h>
#include <stb_image.h>
#include <tinyexr.h>
#include <filesystem>
#include <memory>
auto test::LoadImgFromPathAsUcharData(const std::string& filePath, int& width, int& height, int& comp, int requestComp) -> std::unique_ptr<unsigned char[]>
{
	std::filesystem::path path(filePath);
	if (path.extension() == ".png" || path.extension() == ".jpg" || path.extension() == ".jpeg" || path.extension() == ".bmp")
	{
		int t_w, t_h, t_c;
		std::string pathString = path.string();
		auto pixels = stbi_load(pathString.c_str(), &t_w, &t_h, &t_c, requestComp);
		if (pixels) {
			auto img = std::unique_ptr<unsigned char[]>(new unsigned char[t_w * t_h * t_c]);
			std::memcpy(img.get(), pixels, sizeof(unsigned char) * t_w * t_h * t_c);
			width    = t_w;
			height   = t_h;
			comp     = t_c;
			return std::move(img);
		}
		else
		{
			return nullptr;
		}
	}
}

auto test::LoadImgFromPathAsFloatData(const std::string& filePath, int& width, int& height, int& comp, int requestComp) -> std::unique_ptr<float[]>
{
	std::filesystem::path path(filePath);
	if (path.extension() == ".png" || path.extension() == ".jpg" || path.extension() == ".jpeg" || path.extension() == ".bmp")
	{
		int t_w, t_h, t_c;
		std::string pathString = path.string();
		auto pixels = stbi_loadf(pathString.c_str(), &t_w, &t_h, &t_c, requestComp);
		if (pixels) {
			auto img = std::unique_ptr<float[]>(new float[t_w * t_h * t_c]);
			std::memcpy(img.get(), pixels, sizeof(float) * t_w * t_h * t_c);
			width    = t_w;
			height   = t_h;
			comp     = t_c;
			return std::move(img);
		}
		else
		{
			return nullptr;
		}
	}
}
