
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <stb/stb_image.h>
#include <stb/stb_image_write.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <RTLib/Core.h>
#include <RTLib/Config.h>
#include <RTLib/VectorFunction.h>
#include <RTLib/Exceptions.h>
#include <iostream>
//NVRTC Example
static constexpr size_t kNumBlocks = 64;
int main(){
    const char* programSrc =
    "#include <cuda_runtime.h>\n"
    "#include <RTLib/Random.h>\n"
    "#include <RTLib/VectorFunction.h>\n"
    "extern \"C\" __global__ void rgbKernel(uchar4* inBuffer,uchar4* outBuffer, int width, int height){\n"
    "   int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "   int j = blockIdx.y * blockDim.y + threadIdx.y;\n"
    "   if(i<width&&j<height){\n"
    "       outBuffer[j*width+i]   = rtlib::srgb_to_rgba(inBuffer[j*width+i]);\n"
    "   }\n"
    "}"
    "extern \"C\" __global__ void blurKernel(uchar4* inBuffer,uchar4* outBuffer, int width, int height){\n"
    "   int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "   int j = blockIdx.y * blockDim.y + threadIdx.y;\n"
    "   if(i<width&&j<height){\n"
    "       unsigned long long seed = static_cast<unsigned long long>(j)*width+i;\n"
    "       auto rng = rtlib::Xorshift128(seed);\n"
    "       auto random_v = rtlib::random_float2(-5.0f,5.0f,rng);\n"
    "       auto new_i    = rtlib::clamp((int)(i+random_v.x),0,width-1);\n"
    "       auto new_j    = rtlib::clamp((int)(j+random_v.y),0,height-1);\n"
    "       outBuffer[j*width+i]   = inBuffer[new_j*width+new_i];\n"
    "   }\n"
    "}";
    try{
        RTLIB_CUDA_CHECK( cudaFree(0));
        auto program = rtlib::NVRTCProgram();
        {
            program.create(programSrc,"simpleKernel");
            const char* nvrtc_options[]      = {RTLIB_NVRTC_OPTIONS };
            const char* cuda_include_dirs[]  = {RTLIB_CUDA_INCLUDE_DIRS};
            const char* optix_include_dir    =  RTLIB_INCLUDE_DIR;
            const char* rtlib_include_dir    =  RTLIB_INCLUDE_DIR;
            std::vector<std::string> includeOptions;
            includeOptions.reserve(2+std::size(cuda_include_dirs));
            includeOptions.push_back(std::string("-I")+optix_include_dir);
            includeOptions.push_back(std::string("-I")+rtlib_include_dir);
            for(const auto& cuda_include_dir:cuda_include_dirs){
                includeOptions.push_back(std::string("-I")+cuda_include_dir);
            }
            std::vector<std::string> options = {};
            options.reserve(includeOptions.size()+options.size());
            for(const auto& nvrtc_option:nvrtc_options){
                options.push_back(std::string(nvrtc_option));
            }
            std::copy(includeOptions.begin(),includeOptions.end(),std::back_inserter(options));
            try{
                program.compile(options);
            }catch(rtlib::NVRTCException& nvrtcErr){
                std::cout << program.getLog() << std::endl;
                std::cerr << nvrtcErr.what()  << std::endl;
            }
        }
        std::string ptx = program.getPTX();
        CUdevice   cuDevice;
        CUcontext  cuContext;
        RTLIB_CU_CHECK(cuInit(0));
        RTLIB_CU_CHECK(cuDeviceGet(&cuDevice,0));
        RTLIB_CU_CHECK(cuCtxCreate(&cuContext,0,cuDevice));
        auto cuModule     = rtlib::CUDAModule(ptx.data());
        auto simpleKernel = cuModule.getFunction("blurKernel");
        {
            int width = 0,height = 0,comp = 0;
            auto data = stbi_load("C:\\Users\\shums\\Desktop\\image.png", &width, &height, &comp, 4);
            size_t n = width * height;
            uint2 numThreads = (make_uint2(width - 1, height - 1) / kNumBlocks) + make_uint2(1);
            //std::cout << numThreads << std::endl;
            auto buff_in  = rtlib::CUDABuffer<uchar4>(reinterpret_cast<uchar4*>(data), n);
            auto buff_out = rtlib::CUDABuffer<uchar4>(); buff_out.allocate(n);
            {
                stbi_image_free(data);
                data = nullptr;
            }
            auto buff_in_ptr  =  buff_in.getDevicePtr();
            auto buff_out_ptr = buff_out.getDevicePtr();
            void* args[] = {
                reinterpret_cast<void*>(&buff_in_ptr),
                reinterpret_cast<void*>(&buff_out_ptr),
                reinterpret_cast<void*>(&width),
                reinterpret_cast<void*>(&height),
            };
            simpleKernel.launch(make_uint3(kNumBlocks, kNumBlocks, 1), make_uint3(numThreads, 1), 0, nullptr, args, nullptr);
            RTLIB_CU_CHECK(cuCtxSynchronize());
            std::vector<uchar4> imgOutput;
            buff_out.download(imgOutput);
            stbi_write_png("tekitou_gray.png",width,height,4,reinterpret_cast<void*>(imgOutput.data()),0);
        }
    }catch(std::runtime_error& err){
        std::cout << err.what() << std::endl;
    }
    return 0;
}
