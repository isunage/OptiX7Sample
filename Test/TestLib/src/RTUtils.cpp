#define  STB_IMAGE_IMPLEMENTATION
#define  STB_IMAGE_WRITE_IMPLEMENTATION
#include <TestLib/RTUtils.h>
#include <stb_image.h>
#include <stb_image_write.h>
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
			int r_c = requestComp ? requestComp : t_c;
			auto img = std::unique_ptr<unsigned char[]>(new unsigned char[t_w * t_h * r_c]);
			std::memcpy(img.get(), pixels, sizeof(unsigned char) * t_w * t_h * r_c);
			width    = t_w;
			height   = t_h;
			comp     = t_c;
			stbi_image_free(pixels);
			return std::move(img);
		}
		else
		{
			return nullptr;
		}
	}
	else {
		return nullptr;
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
			stbi_image_free(pixels);
			return std::move(img);
		}
		else
		{
			return nullptr;
		}
	}
	else {
		return nullptr;
	}
}

bool test::SavePngImgFromCUDA(const char* filename, const rtlib::CUDABuffer<uchar4>& buffer, size_t width, size_t height)
{
	return false;
}

bool test::SavePngImgFromCUDA(const char* filename, const rtlib::CUDATexture2D<uchar4>& texture)
{
	return false;
}

bool test::SavePngImgFromGL(  const char* filename, const rtlib::GLBuffer<uchar4>& buffer, size_t width, size_t height)
{
	std::vector<uchar4> imageData(width * height);
	{
		std::unique_ptr<uchar4[]> imagePixels(new uchar4[width * height]);
		buffer.bind();
		uchar4* pMapData = (uchar4*)glMapBuffer(buffer.getTarget(), GL_READ_ONLY);
		memcpy(imagePixels.get(), pMapData, sizeof(uchar4) * width * height);
		glUnmapBuffer(buffer.getTarget());
		buffer.unbind();

		for (int i = 0; i < height; ++i)
		{
			std::memcpy(imageData.data() + (height - 1 - i) * width, imagePixels.get() + i * width, sizeof(uchar4) * width);
		}
	}
	return stbi_write_png(filename, width, height, 4, imageData.data(), width * 4);
}

bool test::SavePngImgFromGL(  const char* filename, const rtlib::GLTexture2D<uchar4>& texture)
{
	size_t width = texture.getViews()[0].width;
	size_t height = texture.getViews()[0].height;
	std::vector<uchar4> imageData(width * height);
	{
		std::unique_ptr<uchar4[]> imagePixels(new uchar4[width * height]);
		texture.bind();
		glGetTexImage(texture.getTarget(), 0, GL_RGBA, GL_UNSIGNED_BYTE, imagePixels.get());
		texture.unbind();

		for (int i = 0; i < height; ++i)
		{
			std::memcpy(imageData.data() + (height - 1 - i) * width, imagePixels.get() + i * width, sizeof(uchar4) * width);
		}
	}
	return stbi_write_png(filename, width, height, 4, imageData.data(), width * 4);
}

bool test::SaveExrImgFromCUDA(const char* filename, const rtlib::CUDABuffer<float3>& buffer, size_t width, size_t height, size_t numSample)
{
	std::vector<float3> img;
	buffer.download(img);
	EXRHeader header;
	InitEXRHeader(&header);
	EXRImage  image;
	InitEXRImage(&image);
	image.num_channels = 3;
	std::vector<float> images[3];
	for (auto& image : images)
	{
		image.resize(width * height);
	}
	for (int j = 0; j < height; ++j)
	{
		for (int i = 0; i < width; ++i)
		{
			images[0][width * j + i] = img[width * (height - 1 - j) + i].x / static_cast<float>(numSample);
			images[1][width * j + i] = img[width * (height - 1 - j) + i].y / static_cast<float>(numSample);
			images[2][width * j + i] = img[width * (height - 1 - j) + i].z / static_cast<float>(numSample);
		}
	}
	float* image_ptr[3];
	image_ptr[0] = images[2].data();
	image_ptr[1] = images[1].data();
	image_ptr[2] = images[0].data();

	image.images = (unsigned char**)image_ptr;
	image.width = width;
	image.height = height;

	header.num_channels = 3;
	auto header_channels = std::unique_ptr<EXRChannelInfo[]>(new EXRChannelInfo[header.num_channels]);
	header.channels = header_channels.get();
	strncpy(header_channels[0].name, "B", 255); header_channels[0].name[strlen("B")] = '\0';
	strncpy(header_channels[1].name, "G", 255); header_channels[1].name[strlen("G")] = '\0';
	strncpy(header_channels[2].name, "R", 255); header_channels[2].name[strlen("R")] = '\0';
	auto header_pixel_types = std::unique_ptr<int[]>(new int[header.num_channels]);
	auto header_requested_pixel_types = std::unique_ptr<int[]>(new int[header.num_channels]);
	header.pixel_types = header_pixel_types.get();
	header.requested_pixel_types = header_requested_pixel_types.get();
	for (int i = 0; i < header.num_channels; ++i)
	{
		header_pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT;
		header_requested_pixel_types[i] = TINYEXR_PIXELTYPE_HALF;
	}
	const char* err = nullptr;
	int ret = SaveEXRImageToFile(&image, &header, filename, &err);
	if (ret != TINYEXR_SUCCESS) {
		std::cout << err << std::endl;
		return ret;
	}
	return true;
}
