#include "../include/TestLib/RTUtils.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
#include <memory>
#include <vector>
bool test::SavePNGFromCUDA(const char* filename, const rtlib::CUDABuffer<uchar4>& buffer, size_t width, size_t height)
{
	return false;
}

bool test::SavePNGFromCUDA(const char* filename, const rtlib::CUDATexture2D<uchar4>& texture)
{
	return false;
}

bool test::SavePNGFromGL(const char* filename, const rtlib::GLBuffer<uchar4>& buffer, size_t width, size_t height)
{

	return false;
}

bool test::SavePNGFromGL(const char* filename, const rtlib::GLTexture2D<uchar4>& texture)
{
	size_t width  = texture.getViews()[0].width;
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
