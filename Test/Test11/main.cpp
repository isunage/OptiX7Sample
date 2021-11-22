#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <stb_image.h>
#include <stb_image_write.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <RTLib/ext/Camera.h>
#include <RTLib/Optix.h>
#include <RTLib/Config.h>
#include <RTLib/ext/Math/VectorFunction.h>
#include <RTLib/Exceptions.h>
#include <RTLib/Utils.h>
#include <Test11Config.h>
#include <tiny_obj_loader.h>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <array>
#include <string_view>
#include "cuda/RayTrace.h"
struct AABB {
    float3 min = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
    float3 max = make_float3(0.0f, 0.0f, 0.0f);
    void Update(const float3& vertex) {
        min = rtlib::min(min, vertex);
        max = rtlib::max(max, vertex);
    }
};
struct ShapeInfo {
    std::string        name    = {};
    std::vector<uint3> indices = {};
    uint32_t           matID   =  0;
};
struct MaterialInfo{
    std::string        diffTexName = {};
    float3             diffColor   = {};
    std::string        emitTexName = {};
    float3             emitColor   = {};
};
int main() {
    //static constexpr float3 vertices[]           = { float3{-0.5f,-0.5f,0.0f},float3{0.5f,-0.5f,0.0f},float3{0.0f,0.5f,0.0f}};
    //static constexpr uint3   indices[]           = {{0,1,2}};
    std::vector<float3>        vertices            = {};
    std::vector<float2>        texCoords           = {};
    std::vector<uint3>         indices             = {};
    std::vector<ShapeInfo>     shapeInfos          = {};
    std::vector<MaterialInfo>  materialInfos       = {};
    std::unordered_map<std::string, size_t> texMap = {};
    AABB                       aabb                = {};
    {
        std::string                      err       = {};
        std::string                      warn      = {};
        tinyobj::attrib_t                attrib    = {};
        std::vector<tinyobj::shape_t>    shapes    = {};
        std::vector<tinyobj::material_t> materials = {};
        bool res = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, TEST_TEST11_DATA_PATH"/Models/CornellBox/CornellBox-Original.obj", TEST_TEST11_DATA_PATH"/Models/CornellBox/");
        std::cout << warn << "\n";
        std::cout << err  << "\n";
        assert(res);
        {
            {
                for (const auto& shape : shapes) {
                    size_t numMeshes = shape.mesh.num_face_vertices.size();
                    std::unordered_map<size_t, size_t> materialIDMap = {};
                    std::vector<ShapeInfo>             tmpShapeInfos = {};
                    for (size_t f = 0; f < numMeshes; ++f) {
                        size_t idx = 0;
                        try {
                            idx = materialIDMap.at(shape.mesh.material_ids[f]);
                        }
                        catch (...) {
                            idx = tmpShapeInfos.size();
                            materialIDMap[shape.mesh.material_ids[f]] = idx;
                            tmpShapeInfos.push_back(ShapeInfo{});
                            tmpShapeInfos[idx].matID = shape.mesh.material_ids[f];
                            tmpShapeInfos[idx].name  = shape.name + std::to_string(idx);
                        }
                        tmpShapeInfos[idx].indices.push_back(make_uint3(shape.mesh.indices[3 * f + 0].vertex_index,
                                                                        shape.mesh.indices[3 * f + 1].vertex_index,
                                                                        shape.mesh.indices[3 * f + 2].vertex_index));
                    }
                    for (auto&& tmpShapeInfo : tmpShapeInfos) {
                        shapeInfos.emplace_back(tmpShapeInfo);
                    }
                }
            }
            {
                size_t numVertices = attrib.vertices.size() / 3;
                vertices.resize(numVertices);
                for (size_t v = 0; v < numVertices; ++v) {
                    vertices[v].x = attrib.vertices[3 * v + 0];
                    vertices[v].y = attrib.vertices[3 * v + 1];
                    vertices[v].z = attrib.vertices[3 * v + 2];
                }
            }
            {
                texCoords.resize(vertices.size());
                for (const auto& shape : shapes) {
                    for (auto& meshInd:shape.mesh.indices) {
                        if (meshInd.texcoord_index >= 0) {
                            texCoords[meshInd.vertex_index] = make_float2(
                                attrib.texcoords[2 * meshInd.texcoord_index + 0],
                               -attrib.texcoords[2 * meshInd.texcoord_index + 1]
                            );
                        }
                        else {
                            texCoords[meshInd.vertex_index] = make_float2(
                                0.5f,
                                0.5f
                            );
                        }
                        
                    }
                }
            }
            {
                for (size_t v = 0; v < vertices.size(); ++v) {
                    aabb.Update(vertices[v]);
                }
            }
            {
                materialInfos.resize(materials.size());
                for (size_t i = 0; i < materialInfos.size(); ++i) {
                    if (!materials[i].diffuse_texname.empty()) {
                        materialInfos[i].diffTexName = TEST_TEST11_DATA_PATH"/Models/CornellBox/" + materials[i].diffuse_texname;
                        materialInfos[i].diffColor   = make_float3(1.0f);
                    }
                    else {
                        materialInfos[i].diffTexName = TEST_TEST11_DATA_PATH"/Textures/white.png";
                        materialInfos[i].diffColor   = make_float3(materials[i].diffuse[0], materials[i].diffuse[1], materials[i].diffuse[2]);
                    }
                    if (!materials[i].emissive_texname.empty()) {
                        materialInfos[i].emitTexName = TEST_TEST11_DATA_PATH"/Models/CornellBox/" + materials[i].emissive_texname;
                        materialInfos[i].emitColor   = make_float3(1.0f);
                    }
                    else {
                        materialInfos[i].emitTexName = TEST_TEST11_DATA_PATH"/Textures/white.png";
                        materialInfos[i].emitColor   = make_float3(materials[i].emission[0], materials[i].emission[1], materials[i].emission[2]);
                    }
                }
            }
            {
                size_t i = 0;
                for (auto& materialInfo : materialInfos) {
                    if (texMap.count(materialInfo.diffTexName) == 0) {
                        texMap[materialInfo.diffTexName] = i;
                        ++i;
                    }
                    if (texMap.count(materialInfo.emitTexName) == 0) {
                        texMap[materialInfo.emitTexName] = i;
                        ++i;
                    }
                }
            }
            //shapeInfos.resize(30);
        }
        //shapeInfos.resize(100);
    }
    try{
        int width                                    = 1024;
        int height                                   = 1024;
        auto camera = rtlib::ext::Camera({ 0.0f,1.0f, 5.0f }, { 0.0f,1.0f, 0.0f }, { 0.0f,1.0f,0.0f }, 30.0f, 1.0f);
        //一番最初に呼び出す
        RTLIB_CUDA_CHECK( cudaFree(0));
        RTLIB_OPTIX_CHECK(optixInit());
        //OPX型はすべて参照型で対応する実体の参照を保持
        //contextはcopy/move不可
        auto context      = rtlib::OPXContext({0,0,OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL,4});
        auto vertexBuffer = rtlib::CUDABuffer<float3>(vertices);
        auto texCrdBuffer = rtlib::CUDABuffer<float2>(texCoords);
        auto textures     = std::vector<rtlib::CUDATexture2D<uchar4>>{ texMap.size() };
        {
            {
                for (auto& [TexName, TexId] : texMap) {
                    int  width, height, comp;
                    try {
                        auto img = stbi_load(TexName.c_str(), &width, &height, &comp, 4);
                        textures[TexId].allocate(width, height,cudaTextureReadMode::cudaReadModeElementType);
                        textures[TexId].upload(img, width, height);
                        stbi_image_free(img);
                    }
                    catch (std::runtime_error& err) {
                        std::cout << err.what() << "\n";
                    }
                }
            }
        }
        auto d_pVertices                             = reinterpret_cast<CUdeviceptr>(vertexBuffer.getDevicePtr());
        auto indexBuffers                            = std::vector< rtlib::CUDABuffer<uint3>>(shapeInfos.size());
        for (size_t idxBuffID = 0; idxBuffID < indexBuffers.size();++idxBuffID) {
            indexBuffers[idxBuffID].allocate(shapeInfos[idxBuffID].indices.size());
            indexBuffers[idxBuffID].upload(shapeInfos[idxBuffID].indices.data(), shapeInfos[idxBuffID].indices.size());
        }
        auto accelBuildOptions                       = OptixAccelBuildOptions();
        accelBuildOptions.buildFlags                 = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
        accelBuildOptions.motionOptions              = {};
        accelBuildOptions.operation                  = OPTIX_BUILD_OPERATION_BUILD;
        auto geometryFlags                           = std::vector<unsigned int>{
            OPTIX_GEOMETRY_FLAG_NONE
        };
        std::vector<OptixBuildInput> buildInputs(indexBuffers.size());
        for (size_t idxBuffID = 0; idxBuffID < indexBuffers.size(); ++idxBuffID) {
            buildInputs[idxBuffID].type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
            buildInputs[idxBuffID].triangleArray.vertexBuffers       = &d_pVertices;
            buildInputs[idxBuffID].triangleArray.numVertices         = std::size(vertices);
            buildInputs[idxBuffID].triangleArray.vertexFormat        = OPTIX_VERTEX_FORMAT_FLOAT3;
            buildInputs[idxBuffID].triangleArray.vertexStrideInBytes = sizeof(float3);
            buildInputs[idxBuffID].triangleArray.indexBuffer         = reinterpret_cast<CUdeviceptr>(indexBuffers[idxBuffID].getDevicePtr());
            buildInputs[idxBuffID].triangleArray.numIndexTriplets    = indexBuffers[idxBuffID].getCount();
            buildInputs[idxBuffID].triangleArray.indexFormat         = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
            buildInputs[idxBuffID].triangleArray.indexStrideInBytes  = sizeof(uint3);
            buildInputs[idxBuffID].triangleArray.numSbtRecords       = 1;
            buildInputs[idxBuffID].triangleArray.flags               = geometryFlags.data();
        }
        auto pipelineCompileOptions                                  = OptixPipelineCompileOptions{};
        auto [outputBuffer,  traversableHandle]                      = context.buildAccel(accelBuildOptions, buildInputs);
        {            
            pipelineCompileOptions.usesMotionBlur                    = false;
            pipelineCompileOptions.traversableGraphFlags             = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
            pipelineCompileOptions.numAttributeValues                = 3;
            pipelineCompileOptions.numPayloadValues                  = 3;
            pipelineCompileOptions.pipelineLaunchParamsVariableName  = "params";
            pipelineCompileOptions.usesPrimitiveTypeFlags            = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
            pipelineCompileOptions.exceptionFlags                    = OPTIX_EXCEPTION_FLAG_NONE;
        }
        auto cuSource   = std::string();
        {
            auto cuFile = std::ifstream(TEST_TEST11_CUDA_PATH"/RayTrace.cu", std::ios::binary);
            cuSource    = std::string((std::istreambuf_iterator<char>(cuFile)), (std::istreambuf_iterator<char>()));

        }
        //contextはcopy不可
        auto pipeline = context.createPipeline(pipelineCompileOptions);
        auto program  = rtlib::NVRTCProgram(std::string(cuSource),"sampleProgram");
        {
            const char* nvrtc_options[]      = {RTLIB_NVRTC_OPTIONS };
            const char* cuda_include_dirs[]  = { TEST_TEST11_CUDA_PATH, RTLIB_CUDA_INCLUDE_DIRS};
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
        auto module                     = pipeline.createModule(program.getPTX(),moduleCompileOptions);
        auto raygenPG                   = pipeline.createRaygenPG({ module,"__raygen__rg" });
        auto missPG                     = pipeline.createMissPG({ module,"__miss__ms" });
        auto hitgroupPG                 = pipeline.createHitgroupPG({ module,"__closesthit__ch" },{ module,    "__anyhit__ah" },{});
        auto pipelineLinkOptions        = OptixPipelineLinkOptions{};
        {
            pipelineLinkOptions.maxTraceDepth = 1;
            pipelineLinkOptions.debugLevel    = static_cast<OptixCompileDebugLevel>(rtlib::OPXCompileDebugLevel::Minimal);
        }
        pipeline.link(pipelineLinkOptions);
        auto      raygenRecord          = raygenPG.getSBTRecord<RayGenData>();
        raygenRecord.data.eye           = camera.getEye();
        auto [u, v, w]                  = camera.getUVW();
        raygenRecord.data.u             = u;
        raygenRecord.data.v             = v;
        raygenRecord.data.w             = w;
        auto        missRecord          = missPG.getSBTRecord<MissData>();
        missRecord.data.bgColor         = float4{ 0.0f,0.0f,0.0f,1.0f };
        std::vector<rtlib::SBTRecord<HitgroupData>> hitgroupRecords(indexBuffers.size());
        for (size_t idxBuffID = 0; idxBuffID < indexBuffers.size(); ++idxBuffID) {
            hitgroupRecords[idxBuffID]                 = hitgroupPG.getSBTRecord<HitgroupData>();
            hitgroupRecords[idxBuffID].data.vertices   = vertexBuffer.getDevicePtr();
            hitgroupRecords[idxBuffID].data.indices    = indexBuffers[idxBuffID].getDevicePtr();
            hitgroupRecords[idxBuffID].data.texCoords  = texCrdBuffer.getDevicePtr();
            hitgroupRecords[idxBuffID].data.diffuse    = materialInfos[shapeInfos[idxBuffID].matID].diffColor;
            auto diffTexName                           = materialInfos[shapeInfos[idxBuffID].matID].diffTexName;
            hitgroupRecords[idxBuffID].data.diffuseTex = textures[texMap[diffTexName]].getHandle();
            hitgroupRecords[idxBuffID].data.emission   = materialInfos[shapeInfos[idxBuffID].matID].emitColor;
            auto emitTexName                           = materialInfos[shapeInfos[idxBuffID].matID].emitTexName;
            hitgroupRecords[idxBuffID].data.emissionTex= textures[texMap[emitTexName]].getHandle();
        }
        auto    d_RaygenBuffer          = rtlib::CUDABuffer<rtlib::SBTRecord<RayGenData>>(raygenRecord);
        auto      d_MissBuffer          = rtlib::CUDABuffer<rtlib::SBTRecord<MissData>>(missRecord);
        auto  d_HitgroupBuffer          = rtlib::CUDABuffer<rtlib::SBTRecord<HitgroupData>>(hitgroupRecords);
        OptixShaderBindingTable sbt     = {};
        sbt.raygenRecord                = reinterpret_cast<CUdeviceptr>(d_RaygenBuffer.getDevicePtr());
        sbt.missRecordBase              = reinterpret_cast<CUdeviceptr>(  d_MissBuffer.getDevicePtr());
        sbt.missRecordCount             = 1;
        sbt.missRecordStrideInBytes     = sizeof(missRecord);
        sbt.hitgroupRecordBase          = reinterpret_cast<CUdeviceptr>(d_HitgroupBuffer.getDevicePtr());
        sbt.hitgroupRecordCount         = d_HitgroupBuffer.getCount();
        sbt.hitgroupRecordStrideInBytes = sizeof(rtlib::SBTRecord<HitgroupData>);
        auto d_pixel                    = rtlib::CUDABuffer<uchar4>();
        d_pixel.allocate(width * height);
        auto d_seeds                    = rtlib::CUDABuffer<unsigned int>();
        {
            std::random_device rd;
            std::mt19937 mt(rd());
            std::vector<unsigned int> seeds(width* height);
            std::generate(seeds.begin(), seeds.end(), mt);
            d_seeds = rtlib::CUDABuffer<unsigned int>(seeds);
        }
        auto params                     = Params();
        params.image                    = d_pixel.getDevicePtr();
        params.seed                     = d_seeds.getDevicePtr();
        params.width                    = width;
        params.height                   = height;
        params.gasHandle                = traversableHandle;
        auto d_params                   = rtlib::CUDABuffer<Params>(params);
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