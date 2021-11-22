#include <RTLib/CUDA.h>
#include <RTLib/Config.h>
#include <RTLib/Exceptions.h>
#include <RTLib/ext/Math.h>
#include <RTLib/ext/Math/VectorFunction.h>
#include <RTLib/ext/Math/Random.h>
#include <iostream>
#include <string_view>
#include <stdexcept>
#include <random>
#include <array>
static constexpr size_t kNumThreads = 128;
static constexpr size_t kNumBlocks  = 32;
int main(){
    const char* programSrc =
    "#include <cuda_runtime.h>\n"
    "extern \"C\" __global__ void add(int* a, int* b, size_t n){\n"
    "   size_t i = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "   if(i<n){ a[i] += b[i]; }\n"
    "}";
    try{
        RTLIB_CUDA_CHECK( cudaFree(0));
        auto program = rtlib::NVRTCProgram();
        {
            program.create(programSrc,"add");
            const char* nvrtc_options[]      = {RTLIB_NVRTC_OPTIONS };
            const char* cuda_include_dirs[]  = {RTLIB_CUDA_INCLUDE_DIRS};
            const char* optix_include_dir    = RTLIB_INCLUDE_DIR;
            std::vector<std::string> includeOptions;
            includeOptions.reserve(1+std::size(cuda_include_dirs));
            includeOptions.push_back(std::string("-I")+optix_include_dir);
            for(const auto& cuda_include_dir:cuda_include_dirs){
                includeOptions.push_back(std::string("-I")+cuda_include_dir);
            }
            std::vector<std::string> options = {};
            options.reserve(includeOptions.size()+options.size());
            for(const auto& nvrtc_option:nvrtc_options){
                options.push_back(std::string(nvrtc_option));
            }
            std::copy(includeOptions.begin(),includeOptions.end(),std::back_inserter(options));
            program.compile(options);
        }
        std::string ptx = {};
        try{
            ptx = program.getPTX();
        }catch(rtlib::NVRTCException& nvrtcErr){
            std::cout << program.getLog() << std::endl;
            std::cerr << nvrtcErr.what()  << std::endl;
        }
        CUdevice   cuDevice;
        CUcontext  cuContext;
        CUmodule   module;
        CUfunction addFunction;
        RTLIB_CU_CHECK(cuInit(0));
        RTLIB_CU_CHECK(cuDeviceGet(&cuDevice,0));
        RTLIB_CU_CHECK(cuCtxCreate(&cuContext,0,cuDevice));
        RTLIB_CU_CHECK(cuModuleLoadDataEx(&module,ptx.data(),0,0,0));
        RTLIB_CU_CHECK(cuModuleGetFunction(&addFunction,module,"add"));
        size_t n   = kNumThreads * kNumBlocks;
        auto buffA = rtlib::CUDABuffer<int>();
        auto buffB = rtlib::CUDABuffer<int>();
        std::vector<int> data(n);
        {
            buffA.allocate(n);
            buffB.allocate(n);
            for (auto i = 0; i < n; ++i) {
                data[i] = i;
            }
            buffA.upload(data.data(), data.size());
            buffB.upload(data.data(), data.size());
        }
        {
            auto buffAPtr = buffA.getDevicePtr();
            auto buffBPtr = buffB.getDevicePtr();
            void* args[] = {
                reinterpret_cast<void*>(&buffAPtr),
                reinterpret_cast<void*>(&buffBPtr),
                reinterpret_cast<void*>(&n),
            };
            RTLIB_CU_CHECK(cuLaunchKernel(
                addFunction,
                kNumBlocks, 1, 1,
                kNumThreads,1, 1,
                0, nullptr,
                args, nullptr));
            RTLIB_CU_CHECK(cuCtxSynchronize());
            buffA.download(data);
            for (auto i = 0; i < 10; ++i) {
                std::cout << data[i] << ",";
            }
            std::cout << "...";
            for (auto i = n-10; i < n-1; ++i) {
                std::cout << data[i] << ",";
            }
            std::cout << data[n - 1] << std::endl;
        }
    }catch(std::runtime_error& err){
        std::cout << err.what() << std::endl;
    }
    return 0;
}
