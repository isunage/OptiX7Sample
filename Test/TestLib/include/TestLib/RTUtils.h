#ifndef TEST_RT_UTILS_H
#define TEST_RT_UTILS_H
#include <RTLib/core/CUDA.h>
#include <RTLib/core/GL.h>
#include <string>
#include <memory>
namespace test
{
    auto LoadImgFromPathAsUcharData(const std::string& filePath, int& width, int& height, int& comp, int requestComp)->std::unique_ptr<unsigned char[]>;
    auto LoadImgFromPathAsFloatData(const std::string& filePath, int& width, int& height, int& comp, int requestComp)->std::unique_ptr<float[]>;
	//Image
	bool SavePngImgFromCUDA(const char* filename, const rtlib::CUDABuffer<uchar4>   & buffer , size_t width, size_t height);
	bool SavePngImgFromCUDA(const char* filename, const rtlib::CUDATexture2D<uchar4>& texture);
	bool SavePngImgFromGL  (const char* filename, const rtlib::GLBuffer<uchar4>     & buffer , size_t width, size_t height);
	bool SavePngImgFromGL  (const char* filename, const rtlib::GLTexture2D<uchar4>  & texture);
	bool SaveExrImgFromData(const char* filename, const float3*                       data   , size_t width, size_t height, size_t numSample = 1);
	bool SaveExrImgFromCUDA(const char* filename, const rtlib::CUDABuffer<float3>   & buffer , size_t width, size_t height, size_t numSample = 1);
}
#endif