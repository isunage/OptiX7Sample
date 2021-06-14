#define __CUDACC__
#include <cuda_runtime.h>
#include <RTLib/Random.h>
#include <RTLib/VectorFunction.h>
extern "C" __global__ void rgbKernel(uchar4* inBuffer,uchar4* outBuffer, int width, int height){
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   int j = blockIdx.y * blockDim.y + threadIdx.y;
   if(i<width&&j<height){
       outBuffer[j*width+i] = rtlib::srgb_to_rgba(inBuffer[j*width+i]);
   }
}
extern "C" __global__ void blurKernel(uchar4* inBuffer,uchar4* outBuffer, int width, int height){
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   int j = blockIdx.y * blockDim.y + threadIdx.y;
   if(i<width&&j<height){
       unsigned long long seed = static_cast<unsigned long long>(j)*width+i;
       auto rng = rtlib::Xorshift128(seed);
       auto random_v = rtlib::random_float2(-5.0f,5.0f,rng);
       auto new_i    = rtlib::clamp((int)(i+random_v.x),0,width-1);
       auto new_j    = rtlib::clamp((int)(j+random_v.y),0,height-1);
       outBuffer[j*width+i] = inBuffer[new_j*width+new_i];
   }
};