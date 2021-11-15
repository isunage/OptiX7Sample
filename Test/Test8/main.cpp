#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <RTLib/Optix.h>
#include <RTLib/Config.h>
#include <RTLib/VectorFunction.h>
#include <RTLib/Exceptions.h>
#include <GLFW/glfw3.h>
#include <stb_image.h>
#include <stb_image_write.h>
#include <iostream>
#include <string_view>
static inline constexpr std::string_view cuSource =
"#include <cuda_runtime.h>\n"
"#include <optix.h>\n"
"struct Params{\n"
"   uchar4* img_pixel;\n"
"   int     img_width;\n"
"   int     img_height;\n"
"};\n"
"extern \"C\" __constant__ Params params;\n"
"extern \"C\" void __raygen__rg(){ \n"
"   auto launchIndex = optixGetLaunchIndex(); \n"
"   params.img_pixel[params.img_width*launchIndex.y+launchIndex.x] = uchar4{255,255,0,255}; \n"
"}\n"
"extern \"C\" void __miss__ms(){}\n"
"extern \"C\" void __closesthit__ch(){}\n"
"extern \"C\" void __anyhit__ah(){}\n";
int main(){
    struct Params {
        uchar4* img_pixel;
        int     img_width;
        int     img_height;
    };
    try{
        int width  = 1024;
        int height = 1024;
        //一番最初に呼び出す
        RTLIB_CUDA_CHECK( cudaFree(0));
        RTLIB_OPTIX_CHECK(optixInit());
        //OPX型はすべて参照型で対応する実体の参照を保持
        //contextはcopy/move不可
        auto context       = rtlib::OPXContext({0,0,OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL,4});
        auto pipelineCompileOptions = OptixPipelineCompileOptions{};
        {            
            pipelineCompileOptions.usesMotionBlur         = false;
            pipelineCompileOptions.traversableGraphFlags  = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
            pipelineCompileOptions.numAttributeValues     = 3;
            pipelineCompileOptions.numPayloadValues       = 3;
            pipelineCompileOptions.pipelineLaunchParamsVariableName = "params";
            pipelineCompileOptions.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
            pipelineCompileOptions.exceptionFlags         = OPTIX_EXCEPTION_FLAG_NONE;
        }
        //contextはcopy不可
        auto pipeline = context.createPipeline(pipelineCompileOptions);
        auto program  = rtlib::NVRTCProgram(std::string(cuSource),"sampleProgram");
        {
            const char* nvrtc_options[]      = {RTLIB_NVRTC_OPTIONS };
            const char* cuda_include_dirs[]  = {RTLIB_CUDA_INCLUDE_DIRS};
            const char* optix_include_dir    =  RTLIB_OPTIX_INCLUDE_DIR;
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
                std::cerr << "Failed To NVRTC Compile Program!\n";
                std::cout << program.getLog() << std::endl;
                std::cout << nvrtcErr.what()  << std::endl;
            }
            //std::cout << program.getPTX() << "\n";
        }
        auto moduleCompileOptions = OptixModuleCompileOptions{};
        {
            moduleCompileOptions.optLevel         = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
            moduleCompileOptions.debugLevel       = static_cast<OptixCompileDebugLevel>(rtlib::OPXCompileDebugLevel::Minimal);
            moduleCompileOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
            moduleCompileOptions.numBoundValues   = 0;
            moduleCompileOptions.boundValues      = 0;
        }
        auto module     = pipeline.createModule(program.getPTX(),moduleCompileOptions);
        auto raygenPG   = pipeline.createRaygenPG(  { module,"__raygen__rg"     });
        auto missPG     = pipeline.createMissPG(    { module,"__miss__ms"       });
        auto hitgroupPG = pipeline.createHitgroupPG({ module,"__closesthit__ch" },{ module,"__anyhit__ah"     },{});
        auto pipelineLinkOptions = OptixPipelineLinkOptions{};
        {
            pipelineLinkOptions.maxTraceDepth = 1;
            pipelineLinkOptions.debugLevel    = static_cast<OptixCompileDebugLevel>(rtlib::OPXCompileDebugLevel::Minimal);
        }
        pipeline.link(pipelineLinkOptions);
        auto  raygenRecord              = raygenPG.getSBTRecord<int>();
        auto  missRecord                = missPG.getSBTRecord<int>();
        auto  hitgroupRecord            = hitgroupPG.getSBTRecord<int>();
        auto  d_RaygenBuffer            = rtlib::CUDABuffer<rtlib::SBTRecord<int>>(raygenRecord);
        auto  d_MissBuffer              = rtlib::CUDABuffer<rtlib::SBTRecord<int>>(missRecord);
        auto  d_HitgroupBuffer          = rtlib::CUDABuffer<rtlib::SBTRecord<int>>(hitgroupRecord);
        OptixShaderBindingTable sbt     = {};
        sbt.raygenRecord                = reinterpret_cast<CUdeviceptr>(d_RaygenBuffer.getDevicePtr());
        sbt.missRecordBase              = reinterpret_cast<CUdeviceptr>(d_MissBuffer.getDevicePtr());
        sbt.missRecordCount             = 1;
        sbt.missRecordStrideInBytes     = sizeof(missRecord);
        sbt.hitgroupRecordBase          = reinterpret_cast<CUdeviceptr>(d_HitgroupBuffer.getDevicePtr());
        sbt.hitgroupRecordCount         = 1;
        sbt.hitgroupRecordStrideInBytes = sizeof(hitgroupRecord);

        auto d_pixel  = rtlib::CUDABuffer<uchar4>();
        d_pixel.allocate(width * height);
        auto params       = Params();
        params.img_pixel  = d_pixel.getDevicePtr();
        params.img_width  = width;
        params.img_height = height;
        auto d_params     = rtlib::CUDABuffer<Params>(params);
        CUstream stream;
        RTLIB_CU_CHECK(cuStreamCreate(&stream, CU_STREAM_DEFAULT));
        pipeline.launch(stream, d_params.getDevicePtr(), sbt, width, height, 1);
        cuStreamSynchronize(stream);
        RTLIB_CU_CHECK(cuStreamDestroy(stream));
        auto img_pixels = std::vector<uchar4>();
        d_pixel.download(img_pixels);
        stbi_write_bmp("tekitou.bmp", width, height, 4, img_pixels.data());
    }catch(std::runtime_error& err){
        std::cerr << err.what() << std::endl;
    }
}