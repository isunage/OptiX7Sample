#include <Test15Config.h>
#include <cuda/RayTrace.h>
#include <RTLib/Optix.h>
#include <RTLib/Utils.h>
#include <stb_image.h>
#include <stb_image_write.h>
#include "../include/PathTracer.h"
#include <random>
#include <iostream>
#include <fstream>
#include <string>
int main() {
    int  width = 512;
    int  height = 512;
    auto cameraController = rtlib::CameraController({ 0.0f,1.0f, 5.0f });
    test::PathTracer tracer = {};
    tracer.InitCUDA();
    tracer.InitOPX();
    auto objMeshGroup   = std::make_shared<test::ObjMeshGroup>();
    if (!objMeshGroup->Load(TEST_TEST15_DATA_PATH"/Models/CornellBox/CornellBox-Mirror.obj", TEST_TEST15_DATA_PATH"/Models/CornellBox/")) {
        return -1;
    }
    auto& materialSet   = objMeshGroup->GetMaterialSet();
    {
        for (auto& material : materialSet->materials) {
            auto diffTex = material.diffTex != "" ? material.diffTex : std::string(TEST_TEST15_DATA_PATH"/Textures/white.png");
            auto specTex = material.specTex != "" ? material.specTex : std::string(TEST_TEST15_DATA_PATH"/Textures/white.png");
            auto emitTex = material.emitTex != "" ? material.emitTex : std::string(TEST_TEST15_DATA_PATH"/Textures/white.png");
            if (!tracer.HasTexture(material.diffTex)) {
                 tracer.LoadTexture(material.diffTex, diffTex);
            }
            if (!tracer.HasTexture(material.specTex)) {
                 tracer.LoadTexture(material.specTex, specTex);
            }
            if (!tracer.HasTexture(material.emitTex)) {
                 tracer.LoadTexture(material.emitTex, emitTex);
            }
        }
    }
    //GAS1: World
    auto worldGASHandle = std::make_shared<test::GASHandle>();
    {
        OptixAccelBuildOptions accelOptions = {};
        accelOptions.buildFlags             = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
        accelOptions.operation              = OPTIX_BUILD_OPERATION_BUILD;
        for (auto& name : objMeshGroup->GetMeshGroup()->GetUniqueNames()) {
            if(name!="light"){
                worldGASHandle->meshes.push_back(objMeshGroup->GetMeshGroup()->LoadMesh(name));
            }
        }
        worldGASHandle->Build(tracer.GetOPXContext().get(), accelOptions);
    }
    //GAS2: Light
    auto lightGASHandle = std::make_shared<test::GASHandle>();
    {
        OptixAccelBuildOptions accelOptions = {};
        accelOptions.buildFlags             = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
        accelOptions.operation              = OPTIX_BUILD_OPERATION_BUILD;
        lightGASHandle->meshes.push_back(objMeshGroup->GetMeshGroup()->LoadMesh("light"));
        lightGASHandle->Build(tracer.GetOPXContext().get(), accelOptions);
    }
    tracer.SetGASHandle("CornellBox-World", worldGASHandle);
    tracer.SetGASHandle("CornellBox-Light", lightGASHandle);
    //IAS1: FirstIAS
    auto firstIASHandle = std::make_shared<test::IASHandle>();
    {
        OptixAccelBuildOptions accelOptions = {};
        accelOptions.buildFlags             = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
        accelOptions.operation              = OPTIX_BUILD_OPERATION_BUILD;
        auto worldInstance                  = tracer.GetInstance("CornellBox-World");
        auto lightInstance                  = tracer.GetInstance("CornellBox-Light");
        lightInstance.instance.sbtOffset    = worldInstance.baseGASHandle->sbtCount * RAY_TYPE_COUNT;
        firstIASHandle->instanceSets.resize(1);
        firstIASHandle->instanceSets[0]     = std::make_shared<test::InstanceSet>();
        firstIASHandle->instanceSets[0]->SetInstance(worldInstance);
        firstIASHandle->instanceSets[0]->SetInstance(lightInstance);
        firstIASHandle->instanceSets[0]->instanceBuffer.Upload();
        firstIASHandle->Build(tracer.GetOPXContext().get(), accelOptions);
    }
    tracer.SetIASHandle("FirstIAS", firstIASHandle);
    //pipeline: init
    auto mainPipeline   = std::make_shared<test::Pipeline>();
    {
        OptixPipelineCompileOptions pipelineCompileOptions      = {};
        pipelineCompileOptions.pipelineLaunchParamsVariableName = "params";
        pipelineCompileOptions.numAttributeValues               = 3;
        pipelineCompileOptions.numPayloadValues                 = 3;
        pipelineCompileOptions.usesPrimitiveTypeFlags           = 0;
        pipelineCompileOptions.usesMotionBlur                   = false;
        pipelineCompileOptions.traversableGraphFlags            = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
        mainPipeline->pipeline                                  = tracer.GetOPXContext()->createPipeline(pipelineCompileOptions);
    }

    {
        mainPipeline->width  = width;
        mainPipeline->height = height;
        mainPipeline->depth  = 2;
    }
    //module: Load
    {
        auto ptxSource = std::string();
        {
            auto ptxFile = std::ifstream(TEST_TEST15_CUDA_PATH"/RayTrace.ptx", std::ios::binary);
            ptxSource = std::string((std::istreambuf_iterator<char>(ptxFile)), (std::istreambuf_iterator<char>()));
        }
        OptixModuleCompileOptions moduleCompileOptions = {};
        moduleCompileOptions.maxRegisterCount          = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
        moduleCompileOptions.numBoundValues            = 0;
#ifndef NDEBUG
        moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
        moduleCompileOptions.optLevel   = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
#else
        moduleCompileOptions.optLevel   = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
        moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;
#endif
        try {
            mainPipeline->modules["RayTrace"] = mainPipeline->pipeline.createModule(ptxSource, moduleCompileOptions);
        }
        catch (rtlib::OptixException& err) {
            std::cout << err.what() << std::endl;
        }
    }
    //program group: init
    {
        auto& rayTraceModule                               = mainPipeline->modules["RayTrace"];
        mainPipeline->raygenPG                             = mainPipeline->pipeline.createRaygenPG({ rayTraceModule,"__raygen__rg" });
        mainPipeline->missPGs.resize(RAY_TYPE_COUNT); 
        mainPipeline->missPGs[RAY_TYPE_RADIANCE]           = mainPipeline->pipeline.createMissPG({ rayTraceModule,"__miss__radiance" });
        mainPipeline->missPGs[RAY_TYPE_OCCLUSION]          = mainPipeline->pipeline.createMissPG({ rayTraceModule,"__miss__occluded" });
        mainPipeline->hitGroupPGs.resize(MATERIAL_TYPE_COUNT);
        mainPipeline->hitGroupPGs[MATERIAL_TYPE_DIFFUSE]   = mainPipeline->pipeline.createHitgroupPG({ rayTraceModule ,"__closesthit__radiance_for_diffuse" }, {}, {});
        mainPipeline->hitGroupPGs[MATERIAL_TYPE_SPECULAR]  = mainPipeline->pipeline.createHitgroupPG({ rayTraceModule ,"__closesthit__radiance_for_specular" }, {}, {});
        mainPipeline->hitGroupPGs[MATERIAL_TYPE_EMISSION]  = mainPipeline->pipeline.createHitgroupPG({ rayTraceModule ,"__closesthit__radiance_for_emission" }, {}, {});
        mainPipeline->hitGroupPGs[MATERIAL_TYPE_OCCLUSION] = mainPipeline->pipeline.createHitgroupPG({ rayTraceModule ,"__closesthit__occluded" }, {}, {});
    }
    //pipeline link
    {
        OptixPipelineLinkOptions pipelineLinkOptions = {};
        pipelineLinkOptions.maxTraceDepth = 2;
#ifndef NDEBUG
        pipelineLinkOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;
#else
        pipelineLinkOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;
#endif
        mainPipeline->pipeline.link(pipelineLinkOptions);
    }
    //SBTRecord
    {
        mainPipeline->raygenBuffer.cpuHandle.resize(1);
        auto camera    = cameraController.GetCamera(30.0f, 1.0f);
        auto [u, v, w] = camera.getUVW();
        mainPipeline->raygenBuffer.cpuHandle[0] = mainPipeline->raygenPG.getSBTRecord<RayGenData>();
        mainPipeline->raygenBuffer.cpuHandle[0].data.eye = camera.getEye();
        mainPipeline->raygenBuffer.cpuHandle[0].data.u   = u;
        mainPipeline->raygenBuffer.cpuHandle[0].data.v   = v;
        mainPipeline->raygenBuffer.cpuHandle[0].data.w   = w;
        mainPipeline->raygenBuffer.Upload();
        mainPipeline->missBuffer.cpuHandle.resize(RAY_TYPE_COUNT);
        mainPipeline->missBuffer.cpuHandle[RAY_TYPE_RADIANCE]               = mainPipeline->missPGs[RAY_TYPE_RADIANCE].getSBTRecord<MissData>();
        mainPipeline->missBuffer.cpuHandle[RAY_TYPE_RADIANCE].data.bgColor  = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        mainPipeline->missBuffer.cpuHandle[RAY_TYPE_OCCLUSION]              = mainPipeline->missPGs[RAY_TYPE_OCCLUSION].getSBTRecord<MissData>();
        mainPipeline->missBuffer.cpuHandle[RAY_TYPE_OCCLUSION].data.bgColor = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        mainPipeline->missBuffer.Upload();
        mainPipeline->hitGBuffer.cpuHandle.resize(RAY_TYPE_COUNT* firstIASHandle->sbtCount);
        auto& cpuHgRecords = mainPipeline->hitGBuffer.cpuHandle;
        for (auto& [name, iasHandle] : tracer.m_IASHandles) {
            size_t sbtOffset = 0;
            for (auto& instanceSet : iasHandle->instanceSets) {
                for (auto& baseGASHandle : instanceSet->baseGASHandles) {
                    for (auto& mesh : baseGASHandle->meshes) {
                        for (size_t i = 0; i < mesh->GetUniqueResource()->materials.size(); ++i) {
                            auto materialId = mesh->GetUniqueResource()->materials[i];
                            auto& material  = materialSet->materials[materialId];
                            HitgroupData radianceHgData = {};
                            {
                                radianceHgData.vertices = mesh->GetSharedResource()->vertexBuffer.gpuHandle.getDevicePtr();
                                radianceHgData.indices = mesh->GetUniqueResource()->triIndBuffer.gpuHandle.getDevicePtr();
                                radianceHgData.texCoords = mesh->GetSharedResource()->texCrdBuffer.gpuHandle.getDevicePtr();
                                radianceHgData.diffuseTex = tracer.GetTexture(material.diffTex).getHandle();
                                radianceHgData.specularTex = tracer.GetTexture(material.specTex).getHandle();
                                radianceHgData.emissionTex = tracer.GetTexture(material.emitTex).getHandle();
                                radianceHgData.diffuse = material.diffCol;
                                radianceHgData.specular = material.specCol;
                                radianceHgData.emission = material.emitCol;
                                radianceHgData.shinness = material.shinness;
                            }
                            if (material.type == test::PhongMaterialType::eDiffuse) {
                                cpuHgRecords[RAY_TYPE_COUNT * sbtOffset + RAY_TYPE_COUNT * i + RAY_TYPE_RADIANCE] = mainPipeline->hitGroupPGs[MATERIAL_TYPE_DIFFUSE].getSBTRecord<HitgroupData>(radianceHgData);
                            }
                            else if (material.type == test::PhongMaterialType::eSpecular) {
                                cpuHgRecords[RAY_TYPE_COUNT * sbtOffset + RAY_TYPE_COUNT * i + RAY_TYPE_RADIANCE] = mainPipeline->hitGroupPGs[MATERIAL_TYPE_SPECULAR].getSBTRecord<HitgroupData>(radianceHgData);
                            }
                            else {
                                cpuHgRecords[RAY_TYPE_COUNT * sbtOffset + RAY_TYPE_COUNT * i + RAY_TYPE_RADIANCE] = mainPipeline->hitGroupPGs[MATERIAL_TYPE_EMISSION].getSBTRecord<HitgroupData>(radianceHgData);
                            }
                            cpuHgRecords[RAY_TYPE_COUNT * sbtOffset + RAY_TYPE_COUNT * i + RAY_TYPE_OCCLUSION]    = mainPipeline->hitGroupPGs[MATERIAL_TYPE_OCCLUSION].getSBTRecord<HitgroupData>();
                        }
                        sbtOffset += mesh->GetUniqueResource()->materials.size();
                    }
                }
            }
        }
        mainPipeline->hitGBuffer.Upload();
        mainPipeline->shaderbindingTable = {};
        mainPipeline->shaderbindingTable.raygenRecord                = reinterpret_cast<CUdeviceptr>(mainPipeline->raygenBuffer.gpuHandle.getDevicePtr());
        mainPipeline->shaderbindingTable.missRecordBase              = reinterpret_cast<CUdeviceptr>(mainPipeline->missBuffer.gpuHandle.getDevicePtr());
        mainPipeline->shaderbindingTable.missRecordCount             = mainPipeline->missBuffer.gpuHandle.getCount();
        mainPipeline->shaderbindingTable.missRecordStrideInBytes     = sizeof(rtlib::SBTRecord<MissData>);
        mainPipeline->shaderbindingTable.hitgroupRecordBase          = reinterpret_cast<CUdeviceptr>(mainPipeline->hitGBuffer.gpuHandle.getDevicePtr());
        mainPipeline->shaderbindingTable.hitgroupRecordCount         = mainPipeline->hitGBuffer.gpuHandle.getCount();
        mainPipeline->shaderbindingTable.hitgroupRecordStrideInBytes = sizeof(rtlib::SBTRecord<HitgroupData>);
    }
    auto frameBuffer = rtlib::CUDABuffer<uchar4>(std::vector<uchar4>(width * height));
    auto accumBuffer = rtlib::CUDABuffer<float3>(std::vector<float3>(width * height));
    auto seedBuffer  = rtlib::CUDABuffer<unsigned int>();
    {
        std::vector<unsigned int> seeds(width*height);
        std::random_device rd;
        std::mt19937 mt(rd());
        std::generate(seeds.begin(),seeds.end(),mt);
        seedBuffer.allocate(seeds.size());
        seedBuffer.upload(seeds);
    }
    {
        auto light         = ParallelLight();
        {
            auto lightMesh = lightGASHandle->meshes[0];
            auto lightVertices = std::vector<float3>();
            for (auto& index : lightMesh->GetUniqueResource()->triIndBuffer.cpuHandle) {
                lightVertices.push_back(lightMesh->GetSharedResource()->vertexBuffer.cpuHandle[index.x]);
                lightVertices.push_back(lightMesh->GetSharedResource()->vertexBuffer.cpuHandle[index.y]);
                lightVertices.push_back(lightMesh->GetSharedResource()->vertexBuffer.cpuHandle[index.z]);
            }
            auto lightAABB = rtlib::utils::AABB(lightVertices);
            std::cout << "AABBMin=(" << lightAABB.min.x << "," << lightAABB.min.y << "," << lightAABB.min.z << ")" << std::endl;
            std::cout << "AABBMax=(" << lightAABB.max.x << "," << lightAABB.max.y << "," << lightAABB.max.z << ")" << std::endl;
            auto lightV3 = lightAABB.max - lightAABB.min;
            light.corner = lightAABB.min;
            light.v1 = make_float3(0.0f, 0.0f, lightV3.z);
            light.v2 = make_float3(lightV3.x, 0.0f, 0.0f);
            light.normal = make_float3(0.0f, -1.0f, 0.0f);
            auto lightMaterial = materialSet->materials[lightMesh->GetUniqueResource()->materials[0]];
            light.emission = lightMaterial.emitCol;
        }
        Params params = {};
        {
            params.frameBuffer     = frameBuffer.getDevicePtr();
            params.accumBuffer     = accumBuffer.getDevicePtr();
            params.seed            = seedBuffer.getDevicePtr();
            params.width           = width;
            params.height          = height;
            params.gasHandle       = firstIASHandle->handle;
            params.light           = light;
            params.samplePerALL    = 100;
            params.samplePerLaunch = 100;
        }
        mainPipeline->paramsBuffer.cpuHandle.push_back(params);
        mainPipeline->paramsBuffer.Upload();
    }
    tracer.SetPipeline("Main", mainPipeline);

    {
        CUstream stream;
        RTLIB_CU_CHECK(cuStreamCreate(&stream, CU_STREAM_DEFAULT));
        mainPipeline->Launch(stream);
        cuStreamSynchronize(stream);
        RTLIB_CU_CHECK(cuStreamDestroy(stream));
        auto img_pixels = std::vector<uchar4>();
        frameBuffer.download(img_pixels);
        stbi_write_bmp("tekitou.bmp", width, height, 4, img_pixels.data());
    }
    return 0;
}