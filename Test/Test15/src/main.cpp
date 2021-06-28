#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <Test15Config.h>
#include <cuda/RayTrace.h>
#include <RTLib/Optix.h>
#include <stb_image.h>
#include <stb_image_write.h>
#include "../include/PathTracer.h"
#include <random>
#include <iostream>
#include <fstream>
#include <string>
int main() {
    test::PathTracer tracer = {};
    int width  = 512;
    int height = 512;
    auto cameraController = rtlib::CameraController({ 0.0f,1.0f, 5.0f });
    RTLIB_CUDA_CHECK(cudaFree(0));
    RTLIB_OPTIX_CHECK(optixInit());
    tracer.m_OPXContext = std::make_shared<rtlib::OPXContext>(rtlib::OPXContext::Desc{ 0, 0, OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL, 4 });
    auto objMeshGroup = std::make_shared<test::ObjMeshGroup>();
    if (!objMeshGroup->Load(TEST_TEST15_DATA_PATH"/Models/CornellBox/CornellBox-Glossy.obj", TEST_TEST15_DATA_PATH"/Models/CornellBox/")) {
        return -1;
    }
    auto& materialSet   = objMeshGroup->GetMaterialSet();
    {
        for (auto& material : materialSet->materials) {
            auto diffTex = material.diffTex != "" ? material.diffTex : std::string(TEST_TEST15_DATA_PATH"/Textures/white.png");
            auto specTex = material.specTex != "" ? material.specTex : std::string(TEST_TEST15_DATA_PATH"/Textures/white.png");
            auto emitTex = material.emitTex != "" ? material.emitTex : std::string(TEST_TEST15_DATA_PATH"/Textures/white.png");
            if (tracer.m_Textures.count(material.diffTex)==0) {
                int texWidth, texHeight, texComp;
                auto img = stbi_load(diffTex.c_str(), &texWidth, &texHeight, &texComp, 4);
                tracer.m_Textures[material.diffTex] = rtlib::CUDATexture2D<uchar4>();
                tracer.m_Textures[material.diffTex].allocate(texWidth, texHeight, cudaTextureReadMode::cudaReadModeElementType);
                tracer.m_Textures[material.diffTex].upload(img, texWidth, texHeight);
                stbi_image_free(img);
            }
            if (tracer.m_Textures.count(material.specTex) == 0) {
                int texWidth, texHeight, texComp;
                auto img = stbi_load(specTex.c_str(), &texWidth, &texHeight, &texComp, 4);
                tracer.m_Textures[material.specTex] = rtlib::CUDATexture2D<uchar4>();
                tracer.m_Textures[material.specTex].allocate(texWidth, texHeight, cudaTextureReadMode::cudaReadModeElementType);
                tracer.m_Textures[material.specTex].upload(img, texWidth, texHeight);
                stbi_image_free(img);
            }
            if (tracer.m_Textures.count(material.emitTex) == 0) {
                int texWidth, texHeight, texComp;
                auto img = stbi_load(emitTex.c_str(), &texWidth, &texHeight, &texComp, 4);
                tracer.m_Textures[material.emitTex] = rtlib::CUDATexture2D<uchar4>();
                tracer.m_Textures[material.emitTex].allocate(texWidth, texHeight, cudaTextureReadMode::cudaReadModeElementType);
                tracer.m_Textures[material.emitTex].upload(img, texWidth, texHeight);
                stbi_image_free(img);
            }
        }
    }
    tracer.m_GASHandles["CornellBox-Glossy"] = std::make_shared<test::GASHandle>();
    {
        auto& gasHandle = tracer.m_GASHandles["CornellBox-Glossy"];
        gasHandle->sbtCount = 0;
        for (auto& name : objMeshGroup->GetMeshGroup()->GetUniqueNames()) {
            gasHandle->meshes.push_back(objMeshGroup->GetMeshGroup()->LoadMesh(name));
        }
        OptixAccelBuildOptions accelOptions = {};
        accelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
        accelOptions.operation  = OPTIX_BUILD_OPERATION_BUILD;
        gasHandle->Build(tracer.m_OPXContext.get(), accelOptions);
    }
    tracer.m_IASHandle = std::make_shared<test::IASHandle>();
    {
        tracer.m_IASHandle->instanceSets.resize(1);
        tracer.m_IASHandle->instanceSets[0] = std::make_shared<test::InstanceSet>();
        tracer.m_IASHandle->instanceSets[0]->baseGASHandles.resize(1);
        tracer.m_IASHandle->instanceSets[0]->instanceBuffer.cpuHandle.resize(1);
        //instanceSets0
        //--instanceSets[0].baseGASHandles[0]
        tracer.m_IASHandle->instanceSets[0]->baseGASHandles[0] = tracer.m_GASHandles["CornellBox-Glossy"];
        tracer.m_IASHandle->instanceSets[0]->instanceBuffer.cpuHandle[0].traversableHandle = tracer.m_GASHandles["CornellBox-Glossy"]->handle;
        tracer.m_IASHandle->instanceSets[0]->instanceBuffer.cpuHandle[0].instanceId        = 0;
        tracer.m_IASHandle->instanceSets[0]->instanceBuffer.cpuHandle[0].sbtOffset         = 0;
        tracer.m_IASHandle->instanceSets[0]->instanceBuffer.cpuHandle[0].visibilityMask    = 255;
        tracer.m_IASHandle->instanceSets[0]->instanceBuffer.cpuHandle[0].flags             = OPTIX_INSTANCE_FLAG_NONE;
        {
            float transform[12] = {
                1.0f,0.0f,0.0f,0.0f,
                0.0f,1.0f,0.0f,0.0f,
                0.0f,0.0f,1.0f,0.0f
            };
            std::memcpy(tracer.m_IASHandle->instanceSets[0]->instanceBuffer.cpuHandle[0].transform, transform, sizeof(float) * 12);
        }
        tracer.m_IASHandle->instanceSets[0]->instanceBuffer.Upload();
        OptixAccelBuildOptions accelOptions = {};
        accelOptions.buildFlags             = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
        accelOptions.operation              = OPTIX_BUILD_OPERATION_BUILD;
        tracer.m_IASHandle->Build(tracer.m_OPXContext.get(), accelOptions);
    }
    //pipeline: init
    {
        tracer.m_Pipelines["Main"] = {};
        OptixPipelineCompileOptions pipelineCompileOptions = {};
        pipelineCompileOptions.pipelineLaunchParamsVariableName = "params";
        pipelineCompileOptions.numAttributeValues = 3;
        pipelineCompileOptions.numPayloadValues = 3;
        pipelineCompileOptions.usesPrimitiveTypeFlags = 0;
        pipelineCompileOptions.usesMotionBlur = false;
        pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
        tracer.m_Pipelines["Main"].pipeline = tracer.m_OPXContext->createPipeline(pipelineCompileOptions);
    }
    //module: Load
    {
        auto ptxSource = std::string();
        {
            auto ptxFile = std::ifstream(TEST_TEST15_CUDA_PATH"/RayTrace.ptx", std::ios::binary);
            ptxSource = std::string((std::istreambuf_iterator<char>(ptxFile)), (std::istreambuf_iterator<char>()));
        }
        OptixModuleCompileOptions moduleCompileOptions = {};
        moduleCompileOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
        moduleCompileOptions.numBoundValues = 0;
#ifndef NDEBUG
        moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
        moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
#else
        moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
        moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;
#endif
        try {
            tracer.m_OPXModules["RayTrace"] = tracer.m_Pipelines["Main"].pipeline.createModule(ptxSource, moduleCompileOptions);
        }
        catch (rtlib::OptixException& err) {
            std::cout << err.what() << std::endl;
        }
    }
    //program group: init
    {
        tracer.m_Pipelines["Main"].raygenPG = tracer.m_Pipelines["Main"].pipeline.createRaygenPG({ tracer.m_OPXModules["RayTrace"],"__raygen__rg" });
        tracer.m_Pipelines["Main"].missPGs.resize(RAY_TYPE_COUNT);
        tracer.m_Pipelines["Main"].missPGs[RAY_TYPE_RADIANCE]  = tracer.m_Pipelines["Main"].pipeline.createMissPG({ tracer.m_OPXModules["RayTrace"],"__miss__radiance" });
        tracer.m_Pipelines["Main"].missPGs[RAY_TYPE_OCCLUSION] = tracer.m_Pipelines["Main"].pipeline.createMissPG({ tracer.m_OPXModules["RayTrace"],"__miss__occluded" });
        tracer.m_Pipelines["Main"].hitGroupPGs.resize(MATERIAL_TYPE_COUNT);
        tracer.m_Pipelines["Main"].hitGroupPGs[MATERIAL_TYPE_DIFFUSE]   = tracer.m_Pipelines["Main"].pipeline.createHitgroupPG({ tracer.m_OPXModules["RayTrace"] ,"__closesthit__radiance_for_diffuse" }, {}, {});
        tracer.m_Pipelines["Main"].hitGroupPGs[MATERIAL_TYPE_SPECULAR]  = tracer.m_Pipelines["Main"].pipeline.createHitgroupPG({ tracer.m_OPXModules["RayTrace"] ,"__closesthit__radiance_for_specular" }, {}, {});
        tracer.m_Pipelines["Main"].hitGroupPGs[MATERIAL_TYPE_EMISSION]  = tracer.m_Pipelines["Main"].pipeline.createHitgroupPG({ tracer.m_OPXModules["RayTrace"] ,"__closesthit__radiance_for_emission" }, {}, {});
        tracer.m_Pipelines["Main"].hitGroupPGs[MATERIAL_TYPE_OCCLUSION] = tracer.m_Pipelines["Main"].pipeline.createHitgroupPG({ tracer.m_OPXModules["RayTrace"] ,"__closesthit__occluded" }, {}, {});
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
        tracer.m_Pipelines["Main"].pipeline.link(pipelineLinkOptions);
    }
    //SBTRecord
    {
        tracer.m_Pipelines["Main"].raygenBuffer.cpuHandle.resize(1);
        auto camera    = cameraController.GetCamera(30.0f, 1.0f);
        auto [u, v, w] = camera.getUVW();
        tracer.m_Pipelines["Main"].raygenBuffer.cpuHandle[0] = tracer.m_Pipelines["Main"].raygenPG.getSBTRecord<RayGenData>();
        tracer.m_Pipelines["Main"].raygenBuffer.cpuHandle[0].data.eye = camera.getEye();
        tracer.m_Pipelines["Main"].raygenBuffer.cpuHandle[0].data.u   = u;
        tracer.m_Pipelines["Main"].raygenBuffer.cpuHandle[0].data.v   = v;
        tracer.m_Pipelines["Main"].raygenBuffer.cpuHandle[0].data.w   = w;
        tracer.m_Pipelines["Main"].raygenBuffer.Upload();
        tracer.m_Pipelines["Main"].missBuffer.cpuHandle.resize(RAY_TYPE_COUNT);
        tracer.m_Pipelines["Main"].missBuffer.cpuHandle[RAY_TYPE_RADIANCE] = tracer.m_Pipelines["Main"].missPGs[RAY_TYPE_RADIANCE].getSBTRecord<MissData>();
        tracer.m_Pipelines["Main"].missBuffer.cpuHandle[RAY_TYPE_RADIANCE].data.bgColor = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        tracer.m_Pipelines["Main"].missBuffer.cpuHandle[RAY_TYPE_OCCLUSION] = tracer.m_Pipelines["Main"].missPGs[RAY_TYPE_OCCLUSION].getSBTRecord<MissData>();
        tracer.m_Pipelines["Main"].missBuffer.cpuHandle[RAY_TYPE_OCCLUSION].data.bgColor = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        tracer.m_Pipelines["Main"].missBuffer.Upload();
        tracer.m_Pipelines["Main"].hitGBuffer.cpuHandle.resize(RAY_TYPE_COUNT* tracer.m_IASHandle->sbtCount);
        size_t sbtOffset = 0;
        auto& cpuHgRecords = tracer.m_Pipelines["Main"].hitGBuffer.cpuHandle;
        for (auto& instanceSet : tracer.m_IASHandle->instanceSets) {
            for (auto& baseGASHandle : instanceSet->baseGASHandles) {
                for (auto& mesh : baseGASHandle->meshes) {
                    for (size_t i = 0; i < mesh->GetUniqueResource()->materials.size(); ++i) {
                        auto materialId = mesh->GetUniqueResource()->materials[i];
                        auto& material  = materialSet->materials[materialId];
                        if (material.type == test::PhongMaterialType::eDiffuse) {
                            cpuHgRecords[RAY_TYPE_COUNT * sbtOffset + RAY_TYPE_COUNT * i + RAY_TYPE_RADIANCE]              = tracer.m_Pipelines["Main"].hitGroupPGs[MATERIAL_TYPE_DIFFUSE].getSBTRecord<HitgroupData>();
                        }
                        else if (material.type == test::PhongMaterialType::eSpecular) {
                            cpuHgRecords[RAY_TYPE_COUNT * sbtOffset + RAY_TYPE_COUNT * i + RAY_TYPE_RADIANCE]              = tracer.m_Pipelines["Main"].hitGroupPGs[MATERIAL_TYPE_SPECULAR].getSBTRecord<HitgroupData>();
                        }
                        else {
                            cpuHgRecords[RAY_TYPE_COUNT * sbtOffset + RAY_TYPE_COUNT * i + RAY_TYPE_RADIANCE]              = tracer.m_Pipelines["Main"].hitGroupPGs[MATERIAL_TYPE_EMISSION].getSBTRecord<HitgroupData>();
                        }
                        cpuHgRecords[RAY_TYPE_COUNT * sbtOffset + RAY_TYPE_COUNT * i + RAY_TYPE_RADIANCE].data.vertices    = mesh->GetSharedResource()->vertexBuffer.gpuHandle.getDevicePtr();
                        cpuHgRecords[RAY_TYPE_COUNT * sbtOffset + RAY_TYPE_COUNT * i + RAY_TYPE_RADIANCE].data.indices     = mesh->GetUniqueResource()->triIndBuffer.gpuHandle.getDevicePtr();
                        cpuHgRecords[RAY_TYPE_COUNT * sbtOffset + RAY_TYPE_COUNT * i + RAY_TYPE_RADIANCE].data.texCoords   = mesh->GetSharedResource()->texCrdBuffer.gpuHandle.getDevicePtr();
                        cpuHgRecords[RAY_TYPE_COUNT * sbtOffset + RAY_TYPE_COUNT * i + RAY_TYPE_RADIANCE].data.diffuseTex  = tracer.m_Textures[material.diffTex].getHandle();
                        cpuHgRecords[RAY_TYPE_COUNT * sbtOffset + RAY_TYPE_COUNT * i + RAY_TYPE_RADIANCE].data.specularTex = tracer.m_Textures[material.specTex].getHandle();
                        cpuHgRecords[RAY_TYPE_COUNT * sbtOffset + RAY_TYPE_COUNT * i + RAY_TYPE_RADIANCE].data.emissionTex = tracer.m_Textures[material.emitTex].getHandle();
                        cpuHgRecords[RAY_TYPE_COUNT * sbtOffset + RAY_TYPE_COUNT * i + RAY_TYPE_RADIANCE].data.diffuse     = material.diffCol;
                        cpuHgRecords[RAY_TYPE_COUNT * sbtOffset + RAY_TYPE_COUNT * i + RAY_TYPE_RADIANCE].data.specular    = material.specCol;
                        cpuHgRecords[RAY_TYPE_COUNT * sbtOffset + RAY_TYPE_COUNT * i + RAY_TYPE_RADIANCE].data.emission    = material.emitCol;
                        cpuHgRecords[RAY_TYPE_COUNT * sbtOffset + RAY_TYPE_COUNT * i + RAY_TYPE_RADIANCE].data.shinness    = material.shinness;
                        cpuHgRecords[RAY_TYPE_COUNT * sbtOffset + RAY_TYPE_COUNT * i + RAY_TYPE_OCCLUSION]                 = tracer.m_Pipelines["Main"].hitGroupPGs[MATERIAL_TYPE_OCCLUSION].getSBTRecord<HitgroupData>();
                    }
                    sbtOffset += mesh->GetUniqueResource()->materials.size();
                }
            }
        }
        tracer.m_Pipelines["Main"].hitGBuffer.Upload();
        tracer.m_Pipelines["Main"].shaderbindingTable = {};
        tracer.m_Pipelines["Main"].shaderbindingTable.raygenRecord                = reinterpret_cast<CUdeviceptr>(tracer.m_Pipelines["Main"].raygenBuffer.gpuHandle.getDevicePtr());
        tracer.m_Pipelines["Main"].shaderbindingTable.missRecordBase              = reinterpret_cast<CUdeviceptr>(tracer.m_Pipelines["Main"].missBuffer.gpuHandle.getDevicePtr());
        tracer.m_Pipelines["Main"].shaderbindingTable.missRecordCount             = tracer.m_Pipelines["Main"].missBuffer.gpuHandle.getCount();
        tracer.m_Pipelines["Main"].shaderbindingTable.missRecordStrideInBytes     = sizeof(rtlib::SBTRecord<MissData>);
        tracer.m_Pipelines["Main"].shaderbindingTable.hitgroupRecordBase          = reinterpret_cast<CUdeviceptr>(tracer.m_Pipelines["Main"].hitGBuffer.gpuHandle.getDevicePtr());
        tracer.m_Pipelines["Main"].shaderbindingTable.hitgroupRecordCount         = tracer.m_Pipelines["Main"].hitGBuffer.gpuHandle.getCount();
        tracer.m_Pipelines["Main"].shaderbindingTable.hitgroupRecordStrideInBytes = sizeof(rtlib::SBTRecord<HitgroupData>);
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
        auto lightMesh = test::MeshPtr();
        for (auto& mesh : tracer.m_GASHandles["CornellBox-Glossy"]->meshes) {
            if (mesh->GetUniqueResource()->name == "light") {
                std::cout << "found!\n";
                lightMesh = mesh;
            }
        }
        auto light         = ParallelLight();
        light.corner       = make_float3(-0.239999995, 1.58000004,-0.21999999);
        light.v1           = make_float3(0.469999999 , 0.00000000 ,0.0000000);
        light.v2           = make_float3(0.00000000  ,0.00000000 , 0.379999995 );
        light.normal       = make_float3(0.0f, -1.0f, 0.0f);
        auto lightMaterial = materialSet->materials[lightMesh->GetUniqueResource()->materials[0]];
        light.emission     = lightMaterial.emitCol;
        tracer.m_Pipelines["Main"].paramsBuffer.cpuHandle.resize(1);
        tracer.m_Pipelines["Main"].paramsBuffer.cpuHandle[0].frameBuffer     = frameBuffer.getDevicePtr();
        tracer.m_Pipelines["Main"].paramsBuffer.cpuHandle[0].accumBuffer     = accumBuffer.getDevicePtr();
        tracer.m_Pipelines["Main"].paramsBuffer.cpuHandle[0].seed            = seedBuffer.getDevicePtr();
        tracer.m_Pipelines["Main"].paramsBuffer.cpuHandle[0].width           = width;
        tracer.m_Pipelines["Main"].paramsBuffer.cpuHandle[0].height          = height;
        tracer.m_Pipelines["Main"].paramsBuffer.cpuHandle[0].gasHandle       = tracer.m_IASHandle->handle;
        tracer.m_Pipelines["Main"].paramsBuffer.cpuHandle[0].light           = light;
        tracer.m_Pipelines["Main"].paramsBuffer.cpuHandle[0].samplePerALL    = 100;
        tracer.m_Pipelines["Main"].paramsBuffer.cpuHandle[0].samplePerLaunch = 100;
        tracer.m_Pipelines["Main"].paramsBuffer.Upload();
    }
    {
        CUstream stream;
        RTLIB_CU_CHECK(cuStreamCreate(&stream, CU_STREAM_DEFAULT));
        tracer.m_Pipelines["Main"].cuStream = stream;
        tracer.m_Pipelines["Main"].pipeline.launch(tracer.m_Pipelines["Main"].cuStream,tracer.m_Pipelines["Main"].paramsBuffer.gpuHandle.getDevicePtr(),tracer.m_Pipelines["Main"].shaderbindingTable,width,height,2);
        cuStreamSynchronize(stream);
        RTLIB_CU_CHECK(cuStreamDestroy(stream));
        auto img_pixels = std::vector<uchar4>();
        frameBuffer.download(img_pixels);
        stbi_write_bmp("tekitou.bmp", width, height, 4, img_pixels.data());
    }
    return 0;
}