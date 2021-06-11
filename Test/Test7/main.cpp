#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <stb_image.h>
#include <stb_image_write.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <RTLib/Camera.h>
#include <RTLib/Core.h>
#include <RTLib/Config.h>
#include <RTLib/VectorFunction.h>
#include <RTLib/Exceptions.h>
#include <RTLib/Utils.h>
#include <Test7Config.h>
#include <iostream>
#include <fstream>
#include <array>
#include <vector>
#include <string_view>
#include "cuda/RayTrace.h"
struct Vertex{
    float3 position;
};
int main(){ 
    //static constexpr float3 vertices[] = { float3{-0.5f,-0.5f,0.0f},float3{0.5f,-0.5f,0.0f},float3{0.0f,0.5f,0.0f}};
    //static constexpr uint3   indices[] = {{0,1,2}};
    //auto box = Box{};
    //box.x0   =-0.5f;box.y0   =-0.5f;box.z0   =-0.5f;
    //box.x1   = 0.5f;box.y1   = 0.5f;box.z1   = 0.5f;
    //auto vertices = box.getVertices();
    //auto indices  = box.getIndices();
    auto rect       = rtlib::utils::Rect{};
    rect.x0         = -0.5f; rect.y0 = -0.5f; rect.z = 0.0f;
    rect.x1         =  0.5f; rect.y0 =  0.5f; 
    auto vertices   =  rect.getVertices();
    auto indices    =  rect.getIndices();
    try{
        auto width  = 1024;
        auto height = 1024;
        auto camera = rtlib::Camera({ 0.0f,0.0f,2.0f }, { 0.0f,0.0f,0.0f }, { 0.0f,1.0f,3.0f },45.0f,1.0f);
        //一番最初に呼び出す
        RTLIB_CUDA_CHECK( cudaFree(0));
        RTLIB_OPTIX_CHECK(optixInit());
        //OPX型はすべて参照型で対応する実体の参照を保持
        //contextはcopy/move不可
        auto context                                    = rtlib::OPXContext({0,0,OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL,4});
        auto d_vertices                                 = rtlib::CUDABuffer<float3>(vertices);
        auto d_pVertices                                = reinterpret_cast<CUdeviceptr>(d_vertices.getDevicePtr());
        auto d_indices                                  = rtlib::CUDABuffer<uint3>( indices);
        auto accelBuildOptions                          = OptixAccelBuildOptions();
        accelBuildOptions.buildFlags                    = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
        accelBuildOptions.motionOptions                 = {};
        accelBuildOptions.operation                     = OPTIX_BUILD_OPERATION_BUILD;
        auto geometryFlags                              = std::vector<unsigned int>{
            OPTIX_GEOMETRY_FLAG_NONE
        };
        auto triBuildInput                              = OptixBuildInput();
        triBuildInput.type                              = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        triBuildInput.triangleArray.vertexBuffers       = &d_pVertices;
        triBuildInput.triangleArray.numVertices         = std::size(vertices);
        triBuildInput.triangleArray.vertexFormat        = OPTIX_VERTEX_FORMAT_FLOAT3;
        triBuildInput.triangleArray.vertexStrideInBytes = sizeof(float3);
        triBuildInput.triangleArray.indexBuffer         = reinterpret_cast<CUdeviceptr>(d_indices.getDevicePtr());
        triBuildInput.triangleArray.numIndexTriplets    = std::size(indices);
        triBuildInput.triangleArray.indexFormat         = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        triBuildInput.triangleArray.indexStrideInBytes  = sizeof(uint3);
        triBuildInput.triangleArray.numSbtRecords       = 1;
        triBuildInput.triangleArray.flags               = geometryFlags.data();
        auto [gasBuffer,  gasHandle]                    = context.buildAccel(accelBuildOptions, triBuildInput);
        //instance1->GAS
        auto instance1                                  = OptixInstance{
            {
                0.5f,0.0f,0.0f,-0.6f,
                0.0f,0.5f,0.0f, 0.0f,
                0.0f,0.0f,0.5f, 0.0f
            },
        };
        {
            instance1.instanceId        = 0;
            instance1.visibilityMask    = 255;
            instance1.sbtOffset         = 0;
            instance1.flags             = OPTIX_INSTANCE_FLAG_NONE;
            instance1.traversableHandle = gasHandle;
        }
        auto instance2                                  = OptixInstance{
            {
                0.5f,0.0f,0.0f,+0.6f,
                0.0f,0.5f,0.0f, 0.0f,
                0.0f,0.0f,0.5f, 0.0f
            },
        };
        {
            //instance2->GAS
            instance2.instanceId        = 1;
            instance2.visibilityMask    = 255;
            instance2.sbtOffset         = 1;
            instance2.flags             = OPTIX_INSTANCE_FLAG_NONE;
            instance2.traversableHandle = gasHandle;
        }
        std::vector<OptixInstance> instances            = { instance1,instance2 };
        auto instanceBuffer                             = rtlib::CUDABuffer<OptixInstance>(instances);
        OptixBuildInput instBuildInput                  = {};
        instBuildInput.type                             = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
        instBuildInput.instanceArray.instances          = reinterpret_cast<CUdeviceptr>(instanceBuffer.getDevicePtr());
        instBuildInput.instanceArray.numInstances       = instances.size();
        auto [iasBuffer, iasHandle]                     = context.buildAccel(accelBuildOptions, instBuildInput);
        auto pipelineCompileOptions = OptixPipelineCompileOptions{};
        {            
            pipelineCompileOptions.usesMotionBlur                   = false;
            pipelineCompileOptions.traversableGraphFlags            = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
            pipelineCompileOptions.numAttributeValues               = 3;
            pipelineCompileOptions.numPayloadValues                 = 3;
            pipelineCompileOptions.pipelineLaunchParamsVariableName = "params";
            pipelineCompileOptions.usesPrimitiveTypeFlags           = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
            pipelineCompileOptions.exceptionFlags                   = OPTIX_EXCEPTION_FLAG_NONE;
        }
        auto cuSource   = std::string();
        {
            auto cuFile = std::ifstream(TEST_TEST7_CUDA_PATH"/RayTrace.cu", std::ios::binary);
            cuSource    = std::string((std::istreambuf_iterator<char>(cuFile)), (std::istreambuf_iterator<char>()));

        }
        //contextはcopy不可
        auto pipeline = context.createPipeline(pipelineCompileOptions);
        auto program  = rtlib::NVRTCProgram(std::string(cuSource),"sampleProgram");
        {
            const char* nvrtc_options[]      = {RTLIB_NVRTC_OPTIONS };
            const char* cuda_include_dirs[]  = { TEST_TEST7_CUDA_PATH, RTLIB_CUDA_INCLUDE_DIRS};
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
            moduleCompileOptions.debugLevel       = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;
            moduleCompileOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
            moduleCompileOptions.numBoundValues   = 0;
            moduleCompileOptions.boundValues      = 0;
        }
        auto module                               = pipeline.createModule(program.getPTX(),moduleCompileOptions);
        auto raygenPG                             = pipeline.createRaygenPG({ module,"__raygen__rg" });
        auto missPG                               = pipeline.createMissPG({ module,"__miss__ms" });
        auto hitgroupPG                           = pipeline.createHitgroupPG({ module,"__closesthit__ch" },{ module,    "__anyhit__ah" },{});
        auto pipelineLinkOptions                  = OptixPipelineLinkOptions{};
        {
            pipelineLinkOptions.maxTraceDepth     = 1;
            pipelineLinkOptions.debugLevel        = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;
        }
        pipeline.link(pipelineLinkOptions);
        auto raygenRecord                   = raygenPG.getSBTRecord<RayGenData>();
        raygenRecord.data.eye               = camera.getEye();
        auto [u, v, w]                      = camera.getUVW();
        raygenRecord.data.u                 = u;
        raygenRecord.data.v                 = v;
        raygenRecord.data.w                 = w;
        auto missRecord                     = missPG.getSBTRecord<MissData>();
        missRecord.data.bgColor             = float4{ 1.0f,0.0f,0.0f,1.0f };
        auto hitgroupRecords                = std::vector<rtlib::SBTRecord<HitgroupData>>();
        hitgroupRecords.push_back(hitgroupPG.getSBTRecord<HitgroupData>());
        hitgroupRecords.push_back(hitgroupPG.getSBTRecord<HitgroupData>());
        hitgroupRecords[0].data.vertices    = d_vertices.getDevicePtr();
        hitgroupRecords[0].data.indices     = d_indices.getDevicePtr();
        hitgroupRecords[0].data.emission    = make_float3(1.0f, 1.0f, 0.0f);
        hitgroupRecords[1].data.vertices    = d_vertices.getDevicePtr();
        hitgroupRecords[1].data.indices     = d_indices.getDevicePtr();
        hitgroupRecords[1].data.emission    = make_float3(0.0f, 1.0f, 1.0f);
        auto    d_RaygenBuffer              = rtlib::CUDABuffer<rtlib::SBTRecord<RayGenData>>(raygenRecord);
        auto      d_MissBuffer              = rtlib::CUDABuffer<rtlib::SBTRecord<MissData>>(missRecord);
        auto  d_HitgroupBuffer              = rtlib::CUDABuffer<rtlib::SBTRecord<HitgroupData>>(hitgroupRecords);
        OptixShaderBindingTable sbt         = {};
        sbt.raygenRecord                    = reinterpret_cast<CUdeviceptr>(d_RaygenBuffer.getDevicePtr());
        sbt.missRecordBase                  = reinterpret_cast<CUdeviceptr>(d_MissBuffer.getDevicePtr());
        sbt.missRecordCount                 = 1;
        sbt.missRecordStrideInBytes         = sizeof(rtlib::SBTRecord<MissData>);
        sbt.hitgroupRecordBase              = reinterpret_cast<CUdeviceptr>(d_HitgroupBuffer.getDevicePtr());
        sbt.hitgroupRecordCount             = hitgroupRecords.size();
        sbt.hitgroupRecordStrideInBytes     = sizeof(rtlib::SBTRecord<HitgroupData>);
        auto d_pixel                        = rtlib::CUDABuffer<uchar4>();
        d_pixel.allocate(width * height);
        auto params                         = Params();
        params.image                        = d_pixel.getDevicePtr();
        params.width                        = width;
        params.height                       = height;
        params.gasHandle                    = iasHandle;
        auto d_params                       = rtlib::CUDABuffer<Params>(params);
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