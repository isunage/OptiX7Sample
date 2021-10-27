#ifndef RT_UTILS_H
#define RT_UTILS_H
#include <RTLib/GL.h>
#include <RTLib/CUDA.h>
namespace test {
	bool SavePNGFromCUDA(const char* filename, const rtlib::CUDABuffer<uchar4>   & buffer, size_t width, size_t height);
	bool SavePNGFromCUDA(const char* filename, const rtlib::CUDATexture2D<uchar4>& texture);
	bool SavePNGFromGL  (const char* filename, const rtlib::GLBuffer<uchar4>     & buffer, size_t width, size_t height);
	bool SavePNGFromGL  (const char* filename, const rtlib::GLTexture2D<uchar4>  & texture);
	bool SaveEXRFromCUDA(const char* filename, const rtlib::CUDABuffer<float3>& buffer, size_t width, size_t height, size_t numSample = 1);
}
#endif