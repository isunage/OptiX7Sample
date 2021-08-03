#include <Test17Config.h>
#include <cuda/RayTrace.h>
#include <RTLib/Optix.h>
#include <RTLib/Utils.h>
#include <GLFW/glfw3.h>
#include <stb_image.h>
#include <stb_image_write.h>
#include "../include/PathTracer.h"
#include <random>
#include <iostream>
#include <fstream>
#include <string>

struct WindowState {
    float  curTime = 0.0f;
    float  delTime = 0.0f;
    float2 curCurPos = {};
    float2 delCurPos = {};
};
int main() {
    int  width  = 768;
    int  height = 768;
    auto cameraController = rtlib::CameraController({ 0.0f,1.0f, 5.0f });
	cameraController.SetMouseSensitivity(0.125f);
	cameraController.SetMovementSpeed(50.0f);
    test::PathTracer tracer = {};
    tracer.InitCUDA();
    tracer.InitOPX();
    {
        if (glfwInit() != GLFW_TRUE) {
            throw std::runtime_error("Failed To Initialize GLFW!");
        }
        glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        GLFWwindow* window = glfwCreateWindow(width, height, "title", nullptr, nullptr);
        if (!window) {
            throw std::runtime_error("Failed To Create Window!");
        }
        glfwMakeContextCurrent(window);
        if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
            throw std::runtime_error("Failed To Load GLAD!");
        }
    }
    auto objMeshGroup   = std::make_shared<test::ObjMeshGroup>();
    if (!objMeshGroup->Load(TEST_TEST17_DATA_PATH"/Models/Sponza/sponza.obj", TEST_TEST17_DATA_PATH"/Models/Sponza/")) {
        return -1;
    }
    auto& materialSet   = objMeshGroup->GetMaterialSet();

    {
        for (auto& material : materialSet->materials) {
            auto diffTex = material.diffTex != "" ? material.diffTex : std::string(TEST_TEST17_DATA_PATH"/Textures/white.png");
            auto specTex = material.specTex != "" ? material.specTex : std::string(TEST_TEST17_DATA_PATH"/Textures/white.png");
            auto emitTex = material.emitTex != "" ? material.emitTex : std::string(TEST_TEST17_DATA_PATH"/Textures/white.png");
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
	bool isLightFound   = false;
    //GAS1: World
    auto worldGASHandle = std::make_shared<test::GASHandle>();
    {
		bool isLightFound = false;
        OptixAccelBuildOptions accelOptions = {};
        accelOptions.buildFlags             = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
        accelOptions.operation              = OPTIX_BUILD_OPERATION_BUILD;
        for (auto& name : objMeshGroup->GetMeshGroup()->GetUniqueNames()) {
            if(name!="light"){
                worldGASHandle->meshes.push_back(objMeshGroup->GetMeshGroup()->LoadMesh(name));
			}
			else {
				isLightFound = true;
			}
        }
        worldGASHandle->Build(tracer.GetOPXContext().get(), accelOptions);

    }
    //GAS2: Light
    auto lightGASHandle = std::make_shared<test::GASHandle>();
    if(isLightFound){
        OptixAccelBuildOptions accelOptions = {};
        accelOptions.buildFlags             = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
        accelOptions.operation              = OPTIX_BUILD_OPERATION_BUILD;
        lightGASHandle->meshes.push_back(objMeshGroup->GetMeshGroup()->LoadMesh("light"));
        lightGASHandle->Build(tracer.GetOPXContext().get(), accelOptions);

	}
	else {
		rtlib::utils::AABB aabb = {};
		for (auto& vertex : objMeshGroup->GetMeshGroup()->GetSharedResource()->vertexBuffer.cpuHandle) {
			aabb.Update(vertex);
		}
		OptixAccelBuildOptions accelOptions = {};
		accelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
		accelOptions.operation  = OPTIX_BUILD_OPERATION_BUILD;
		auto lightMesh = test::Mesh::New();
		lightMesh->SetSharedResource(test::MeshSharedResource::New());
		lightMesh->GetSharedResource()->name = "light";
		lightMesh->GetSharedResource()->vertexBuffer.cpuHandle = {
			{aabb.min.x,aabb.max.y+1e-3f,aabb.min.z},
			{aabb.max.x,aabb.max.y+1e-3f,aabb.min.z},
			{aabb.max.x,aabb.max.y+1e-3f,aabb.max.z},
			{aabb.min.x,aabb.max.y+1e-3f,aabb.max.z}
		};
		lightMesh->GetSharedResource()->texCrdBuffer.cpuHandle = {
			{0.0f,0.0f},
			{1.0f,0.0f},
			{1.0f,1.0f},
			{0.0f,1.0f},
		};
		lightMesh->GetSharedResource()->normalBuffer.cpuHandle = {
			{0.0f,-1.0f,0.0f},
			{0.0f,-1.0f,0.0f},
			{0.0f,-1.0f,0.0f},
			{0.0f,-1.0f,0.0f},
		};
		unsigned int curMaterialSetCount = materialSet->materials.size();
		auto lightMaterial = test::PhongMaterial{};
		{
			lightMaterial.name     = "light";
			lightMaterial.type     = test::PhongMaterialType::eEmission;
			lightMaterial.diffCol  = { 10.0f,10.0f,10.0f };
			lightMaterial.diffTex  = "";
			lightMaterial.emitCol  = { 10.0f,10.0f,10.0f };
			lightMaterial.emitTex  = "";
			lightMaterial.specCol  = { 0.0f,0.0f,0.0f };
			lightMaterial.specTex  = "";
			lightMaterial.shinness = 0.0f;
			lightMaterial.shinTex  = "";
			lightMaterial.tranCol  = { 0.0f,0.0f,0.0f };
			lightMaterial.refrInd  = 0.0f;
		}
		materialSet->materials.push_back(
			lightMaterial
		);
		lightMesh->GetSharedResource()->vertexBuffer.Upload();
		lightMesh->GetSharedResource()->texCrdBuffer.Upload();
		lightMesh->GetSharedResource()->normalBuffer.Upload();
		lightMesh->SetUniqueResource(test::MeshUniqueResource::New());
		lightMesh->GetUniqueResource()->name      = "light";
		lightMesh->GetUniqueResource()->materials = {
			curMaterialSetCount
		};
		lightMesh->GetUniqueResource()->matIndBuffer.cpuHandle = {
			0,0,
		};
		lightMesh->GetUniqueResource()->triIndBuffer.cpuHandle = {
			{0,1,2},
			{2,3,0}
		};

		lightMesh->GetUniqueResource()->matIndBuffer.Upload();
		lightMesh->GetUniqueResource()->triIndBuffer.Upload();
		lightGASHandle->meshes.push_back(lightMesh);
		lightGASHandle->Build(tracer.GetOPXContext().get(), accelOptions);
	}
    tracer.SetGASHandle("CornellBox-World", worldGASHandle);
    tracer.SetGASHandle("CornellBox-Light", lightGASHandle);
    //IAS1: FirstIAS
    auto firstIASHandle                     = std::make_shared<test::IASHandle>();
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
    {
        auto tracePipeline = std::make_shared<test::Pipeline>();
        {
            OptixPipelineCompileOptions pipelineCompileOptions = {};
            pipelineCompileOptions.pipelineLaunchParamsVariableName = "params";
            pipelineCompileOptions.numAttributeValues = 3;
            pipelineCompileOptions.numPayloadValues = 3;
            pipelineCompileOptions.usesPrimitiveTypeFlags = 0;
            pipelineCompileOptions.usesMotionBlur = false;
            pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
            tracePipeline->pipeline = tracer.GetOPXContext()->createPipeline(pipelineCompileOptions);
        }
        {
            tracePipeline->width = width;
            tracePipeline->height = height;
            tracePipeline->depth = 2;
        }
        //module: Load
        {
            auto ptxSource = std::string();
            {
                auto ptxFile = std::ifstream(TEST_TEST17_CUDA_PATH"/RayTrace.ptx", std::ios::binary);
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
                tracePipeline->modules["RayTrace"] = tracePipeline->pipeline.createModule(ptxSource, moduleCompileOptions);
            }
            catch (rtlib::OptixException& err) {
                std::cout << err.what() << std::endl;
            }
        }
        //program group: init
        {
            auto& rayTraceModule = tracePipeline->modules["RayTrace"];
            tracePipeline->raygenPG = tracePipeline->pipeline.createRaygenPG({ rayTraceModule,"__raygen__rg" });
            tracePipeline->missPGs.resize(RAY_TYPE_COUNT);
            tracePipeline->missPGs[RAY_TYPE_RADIANCE] = tracePipeline->pipeline.createMissPG({ rayTraceModule,"__miss__radiance" });
            tracePipeline->missPGs[RAY_TYPE_OCCLUSION] = tracePipeline->pipeline.createMissPG({ rayTraceModule,"__miss__occluded" });
            tracePipeline->hitGroupPGs.resize(MATERIAL_TYPE_COUNT);
            tracePipeline->hitGroupPGs[MATERIAL_TYPE_DIFFUSE]   = tracePipeline->pipeline.createHitgroupPG({ rayTraceModule ,"__closesthit__radiance_for_diffuse" }, {}, {});
            tracePipeline->hitGroupPGs[MATERIAL_TYPE_SPECULAR]  = tracePipeline->pipeline.createHitgroupPG({ rayTraceModule ,"__closesthit__radiance_for_specular" }, {}, {});
            tracePipeline->hitGroupPGs[MATERIAL_TYPE_REFRACTION]= tracePipeline->pipeline.createHitgroupPG({ rayTraceModule ,"__closesthit__radiance_for_refraction" }, {}, {});
            tracePipeline->hitGroupPGs[MATERIAL_TYPE_EMISSION]  = tracePipeline->pipeline.createHitgroupPG({ rayTraceModule ,"__closesthit__radiance_for_emission" }, {}, {});
            tracePipeline->hitGroupPGs[MATERIAL_TYPE_OCCLUSION] = tracePipeline->pipeline.createHitgroupPG({ rayTraceModule ,"__closesthit__occluded" }, {}, {});
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
            tracePipeline->pipeline.link(pipelineLinkOptions);
        }
        //SBTRecord
        {
            tracePipeline->raygenBuffer.cpuHandle.resize(1);
            auto camera = cameraController.GetCamera(30.0f, 1.0f);
            auto [u, v, w] = camera.getUVW();
            tracePipeline->raygenBuffer.cpuHandle[0] = tracePipeline->raygenPG.getSBTRecord<RayGenData>();
            tracePipeline->raygenBuffer.cpuHandle[0].data.eye = camera.getEye();
            tracePipeline->raygenBuffer.cpuHandle[0].data.u = u;
            tracePipeline->raygenBuffer.cpuHandle[0].data.v = v;
            tracePipeline->raygenBuffer.cpuHandle[0].data.w = w;
            tracePipeline->raygenBuffer.Upload();
            tracePipeline->missBuffer.cpuHandle.resize(RAY_TYPE_COUNT);
            tracePipeline->missBuffer.cpuHandle[RAY_TYPE_RADIANCE] = tracePipeline->missPGs[RAY_TYPE_RADIANCE].getSBTRecord<MissData>();
            tracePipeline->missBuffer.cpuHandle[RAY_TYPE_RADIANCE].data.bgColor = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            tracePipeline->missBuffer.cpuHandle[RAY_TYPE_OCCLUSION] = tracePipeline->missPGs[RAY_TYPE_OCCLUSION].getSBTRecord<MissData>();
            tracePipeline->missBuffer.cpuHandle[RAY_TYPE_OCCLUSION].data.bgColor = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            tracePipeline->missBuffer.Upload();
            tracePipeline->hitGBuffer.cpuHandle.resize(RAY_TYPE_COUNT * firstIASHandle->sbtCount);
            auto& cpuHgRecords = tracePipeline->hitGBuffer.cpuHandle;
            for (auto& [name, iasHandle] : tracer.m_IASHandles) {
                size_t sbtOffset = 0;
                for (auto& instanceSet : iasHandle->instanceSets) {
                    for (auto& baseGASHandle : instanceSet->baseGASHandles) {
                        for (auto& mesh : baseGASHandle->meshes) {
                            for (size_t i = 0; i < mesh->GetUniqueResource()->materials.size(); ++i) {
                                auto materialId = mesh->GetUniqueResource()->materials[i];
                                auto& material = materialSet->materials[materialId];
                                HitgroupData radianceHgData = {};
                                {
                                    radianceHgData.vertices    = mesh->GetSharedResource()->vertexBuffer.gpuHandle.getDevicePtr();
                                    radianceHgData.indices     = mesh->GetUniqueResource()->triIndBuffer.gpuHandle.getDevicePtr();
                                    radianceHgData.texCoords   = mesh->GetSharedResource()->texCrdBuffer.gpuHandle.getDevicePtr();
                                    radianceHgData.diffuseTex  = tracer.GetTexture(material.diffTex).getHandle();
                                    radianceHgData.specularTex = tracer.GetTexture(material.specTex).getHandle();
                                    radianceHgData.emissionTex = tracer.GetTexture(material.emitTex).getHandle();
                                    radianceHgData.diffuse  = material.diffCol;
                                    radianceHgData.specular = material.specCol;
                                    radianceHgData.emission = material.emitCol;
                                    radianceHgData.shinness = material.shinness;
                                    radianceHgData.transmit = material.tranCol;
                                    radianceHgData.refrInd  = material.refrInd;
                                }
                                if (material.type == test::PhongMaterialType::eDiffuse) {
                                    cpuHgRecords[RAY_TYPE_COUNT * sbtOffset + RAY_TYPE_COUNT * i + RAY_TYPE_RADIANCE] = tracePipeline->hitGroupPGs[MATERIAL_TYPE_DIFFUSE].getSBTRecord<HitgroupData>(radianceHgData);
                                }
                                else if (material.type == test::PhongMaterialType::eSpecular) {
                                    cpuHgRecords[RAY_TYPE_COUNT * sbtOffset + RAY_TYPE_COUNT * i + RAY_TYPE_RADIANCE] = tracePipeline->hitGroupPGs[MATERIAL_TYPE_SPECULAR].getSBTRecord<HitgroupData>(radianceHgData);
                                }
                                else if (material.type == test::PhongMaterialType::eRefract){
                                    cpuHgRecords[RAY_TYPE_COUNT * sbtOffset + RAY_TYPE_COUNT * i + RAY_TYPE_RADIANCE] = tracePipeline->hitGroupPGs[MATERIAL_TYPE_REFRACTION].getSBTRecord<HitgroupData>(radianceHgData);
                                }
                                else {
                                    cpuHgRecords[RAY_TYPE_COUNT * sbtOffset + RAY_TYPE_COUNT * i + RAY_TYPE_RADIANCE] = tracePipeline->hitGroupPGs[MATERIAL_TYPE_EMISSION].getSBTRecord<HitgroupData>(radianceHgData);
                                }
                                cpuHgRecords[RAY_TYPE_COUNT * sbtOffset + RAY_TYPE_COUNT * i + RAY_TYPE_OCCLUSION]    = tracePipeline->hitGroupPGs[MATERIAL_TYPE_OCCLUSION].getSBTRecord<HitgroupData>();
                            }
                            sbtOffset += mesh->GetUniqueResource()->materials.size();
                        }
                    }
                }
            }
            tracePipeline->hitGBuffer.Upload();
            tracePipeline->shaderbindingTable = {};
            tracePipeline->shaderbindingTable.raygenRecord = reinterpret_cast<CUdeviceptr>(tracePipeline->raygenBuffer.gpuHandle.getDevicePtr());
            tracePipeline->shaderbindingTable.missRecordBase = reinterpret_cast<CUdeviceptr>(tracePipeline->missBuffer.gpuHandle.getDevicePtr());
            tracePipeline->shaderbindingTable.missRecordCount = tracePipeline->missBuffer.gpuHandle.getCount();
            tracePipeline->shaderbindingTable.missRecordStrideInBytes = sizeof(rtlib::SBTRecord<MissData>);
            tracePipeline->shaderbindingTable.hitgroupRecordBase = reinterpret_cast<CUdeviceptr>(tracePipeline->hitGBuffer.gpuHandle.getDevicePtr());
            tracePipeline->shaderbindingTable.hitgroupRecordCount = tracePipeline->hitGBuffer.gpuHandle.getCount();
            tracePipeline->shaderbindingTable.hitgroupRecordStrideInBytes = sizeof(rtlib::SBTRecord<HitgroupData>);
        }
        tracer.SetPipeline("Trace", tracePipeline);
    }
    {
        auto debugPipeline = std::make_shared<test::Pipeline>();
        {
            OptixPipelineCompileOptions pipelineCompileOptions = {};
            pipelineCompileOptions.pipelineLaunchParamsVariableName = "params";
            pipelineCompileOptions.numAttributeValues = 3;
            pipelineCompileOptions.numPayloadValues = 3;
            pipelineCompileOptions.usesPrimitiveTypeFlags = 0;
            pipelineCompileOptions.usesMotionBlur = false;
            pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
            debugPipeline->pipeline = tracer.GetOPXContext()->createPipeline(pipelineCompileOptions);
        }
        {
            debugPipeline->width  = width;
            debugPipeline->height = height;
            debugPipeline->depth  = 1;
        }
        //module: Load
        {
            auto ptxSource = std::string();
            {
                auto ptxFile = std::ifstream(TEST_TEST17_CUDA_PATH"/RayDebug.ptx", std::ios::binary);
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
                debugPipeline->modules["RayDebug"] = debugPipeline->pipeline.createModule(ptxSource, moduleCompileOptions);
            }
            catch (rtlib::OptixException& err) {
                std::cout << err.what() << std::endl;
            }
        }
        //program group: init
        {
            auto& rayDebugModule   = debugPipeline->modules["RayDebug"];
            debugPipeline->raygenPG = debugPipeline->pipeline.createRaygenPG({ rayDebugModule,"__raygen__debug" });
            debugPipeline->missPGs.resize(RAY_TYPE_COUNT);
            debugPipeline->missPGs[RAY_TYPE_RADIANCE]  = debugPipeline->pipeline.createMissPG({ rayDebugModule,"__miss__debug" });
            debugPipeline->missPGs[RAY_TYPE_OCCLUSION] = debugPipeline->pipeline.createMissPG({ rayDebugModule,"__miss__debug" });
            debugPipeline->hitGroupPGs.resize(RAY_TYPE_COUNT);
            debugPipeline->hitGroupPGs[RAY_TYPE_RADIANCE]  = debugPipeline->pipeline.createHitgroupPG({ rayDebugModule ,"__closesthit__debug" }, {}, {});
            debugPipeline->hitGroupPGs[RAY_TYPE_OCCLUSION] = debugPipeline->pipeline.createHitgroupPG({ rayDebugModule ,"__closesthit__debug" }, {}, {});
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
            debugPipeline->pipeline.link(pipelineLinkOptions);
        }
        //SBTRecord
        {
            debugPipeline->raygenBuffer.cpuHandle.resize(1);
            auto camera = cameraController.GetCamera(30.0f, 1.0f);
            auto [u, v, w] = camera.getUVW();
            debugPipeline->raygenBuffer.cpuHandle[0] = debugPipeline->raygenPG.getSBTRecord<RayGenData>();
            debugPipeline->raygenBuffer.cpuHandle[0].data.eye = camera.getEye();
            debugPipeline->raygenBuffer.cpuHandle[0].data.u = u;
            debugPipeline->raygenBuffer.cpuHandle[0].data.v = v;
            debugPipeline->raygenBuffer.cpuHandle[0].data.w = w;
            debugPipeline->raygenBuffer.Upload();
            debugPipeline->missBuffer.cpuHandle.resize(RAY_TYPE_COUNT);
            debugPipeline->missBuffer.cpuHandle[RAY_TYPE_RADIANCE] = debugPipeline->missPGs[RAY_TYPE_RADIANCE].getSBTRecord<MissData>();
            debugPipeline->missBuffer.cpuHandle[RAY_TYPE_RADIANCE].data.bgColor = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            debugPipeline->missBuffer.cpuHandle[RAY_TYPE_OCCLUSION] = debugPipeline->missPGs[RAY_TYPE_OCCLUSION].getSBTRecord<MissData>();
            debugPipeline->missBuffer.cpuHandle[RAY_TYPE_OCCLUSION].data.bgColor = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            debugPipeline->missBuffer.Upload();
            debugPipeline->hitGBuffer.cpuHandle.resize(RAY_TYPE_COUNT * firstIASHandle->sbtCount);
            auto& cpuHgRecords = debugPipeline->hitGBuffer.cpuHandle;
            for (auto& [name, iasHandle] : tracer.m_IASHandles) {
                size_t sbtOffset = 0;
                for (auto& instanceSet : iasHandle->instanceSets) {
                    for (auto& baseGASHandle : instanceSet->baseGASHandles) {
                        for (auto& mesh : baseGASHandle->meshes) {
                            for (size_t i = 0; i < mesh->GetUniqueResource()->materials.size(); ++i) {
                                auto materialId = mesh->GetUniqueResource()->materials[i];
                                auto& material = materialSet->materials[materialId];
                                HitgroupData radianceHgData = {};
                                {
                                    radianceHgData.vertices = mesh->GetSharedResource()->vertexBuffer.gpuHandle.getDevicePtr();
                                    radianceHgData.indices = mesh->GetUniqueResource()->triIndBuffer.gpuHandle.getDevicePtr();
                                    radianceHgData.texCoords = mesh->GetSharedResource()->texCrdBuffer.gpuHandle.getDevicePtr();
                                    radianceHgData.diffuseTex = tracer.GetTexture(material.diffTex).getHandle();
                                    radianceHgData.specularTex = tracer.GetTexture(material.specTex).getHandle();
                                    radianceHgData.emissionTex = tracer.GetTexture(material.emitTex).getHandle();
                                    radianceHgData.diffuse  = material.diffCol;
                                    radianceHgData.specular = material.specCol;
                                    radianceHgData.emission = material.emitCol;
                                    radianceHgData.transmit = material.tranCol;
                                    radianceHgData.shinness = material.shinness;
                                    //printf("%lf %lf %lf\n",radianceHgData.transmit.x,radianceHgData.transmit.y,radianceHgData.transmit.z);
                                }
                                cpuHgRecords[RAY_TYPE_COUNT * sbtOffset + RAY_TYPE_COUNT * i + RAY_TYPE_RADIANCE]  = debugPipeline->hitGroupPGs[RAY_TYPE_RADIANCE].getSBTRecord<HitgroupData>(radianceHgData);
                                cpuHgRecords[RAY_TYPE_COUNT * sbtOffset + RAY_TYPE_COUNT * i + RAY_TYPE_OCCLUSION] = debugPipeline->hitGroupPGs[RAY_TYPE_OCCLUSION].getSBTRecord<HitgroupData>();
                            }
                            sbtOffset += mesh->GetUniqueResource()->materials.size();
                        }
                    }
                }
            }
            debugPipeline->hitGBuffer.Upload();
            debugPipeline->shaderbindingTable = {};
            debugPipeline->shaderbindingTable.raygenRecord = reinterpret_cast<CUdeviceptr>(debugPipeline->raygenBuffer.gpuHandle.getDevicePtr());
            debugPipeline->shaderbindingTable.missRecordBase = reinterpret_cast<CUdeviceptr>(debugPipeline->missBuffer.gpuHandle.getDevicePtr());
            debugPipeline->shaderbindingTable.missRecordCount = debugPipeline->missBuffer.gpuHandle.getCount();
            debugPipeline->shaderbindingTable.missRecordStrideInBytes = sizeof(rtlib::SBTRecord<MissData>);
            debugPipeline->shaderbindingTable.hitgroupRecordBase = reinterpret_cast<CUdeviceptr>(debugPipeline->hitGBuffer.gpuHandle.getDevicePtr());
            debugPipeline->shaderbindingTable.hitgroupRecordCount = debugPipeline->hitGBuffer.gpuHandle.getCount();
            debugPipeline->shaderbindingTable.hitgroupRecordStrideInBytes = sizeof(rtlib::SBTRecord<HitgroupData>);
        }
        tracer.SetPipeline("Debug", debugPipeline);
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
    Params params = {};
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
        {
            params.frameBuffer     = frameBuffer.getDevicePtr();
            params.accumBuffer     = accumBuffer.getDevicePtr();
            params.seed            = seedBuffer.getDevicePtr();
            params.width           = width;
            params.height          = height;
            params.gasHandle       = firstIASHandle->handle;
            params.light           = light;
            params.samplePerALL    = 0;
            params.samplePerLaunch = 1;
        }
    }
    auto&  curPipeline = tracer.m_Pipelines["Trace"];
    {
        constexpr std::array<float, 5 * 4> screenVertices = {
            -1.0f,-1.0f,0.0f,1.0f, 0.0f,
             1.0f,-1.0f,0.0f,0.0f, 0.0f,
             1.0f, 1.0f,0.0f,0.0f, 1.0f,
            -1.0f, 1.0f,0.0f,1.0f, 1.0f
        };
        constexpr std::array<uint32_t, 6>  screenIndices = {
            0,1,2,
            2,3,0
        };
        constexpr std::string_view vsSource =
            "#version 330 core\n"
            "layout(location=0) in vec3 position;\n"
            "layout(location=1) in vec2 texCoord;\n"
            "out vec2 uv;\n"
            "void main(){\n"
            "   gl_Position = vec4(position,1.0f);\n"
            "   uv = texCoord;\n"
            "}\n";
        constexpr std::string_view fsSource =
            "#version 330 core\n"
            "uniform sampler2D tex;\n"
            "in vec2 uv;\n"
            "layout(location=0) out vec3 color;\n"
            "void main(){\n"
            "   color = texture2D(tex,uv).xyz;\n"
            "}\n";
        auto window  = glfwGetCurrentContext();
        WindowState windowState = {};
        glfwSetWindowUserPointer(window, &windowState);
        auto program = rtlib::GLProgram();
        {
            program.create();
            auto vs = rtlib::GLVertexShader(std::string(vsSource));
            auto fs = rtlib::GLFragmentShader(std::string(fsSource));
            if (!vs.compile()) {
                std::cout << vs.getLog() << std::endl;
            }
            if (!fs.compile()) {
                std::cout << fs.getLog() << std::endl;
            }
            program.attach(vs);
            program.attach(fs);
            if (!program.link()) {
                std::cout << program.getLog() << std::endl;
            }
        }
        auto screenVBO = rtlib::GLBuffer(screenVertices, GL_ARRAY_BUFFER, GL_STATIC_DRAW);
        auto screenIBO = rtlib::GLBuffer(screenIndices, GL_ELEMENT_ARRAY_BUFFER, GL_STATIC_DRAW);
        auto glTexture = rtlib::GLTexture2D<uchar4>();
        {
            glTexture.allocate({ (size_t)width, (size_t)height });
            glTexture.setParameteri(GL_TEXTURE_MAG_FILTER, GL_LINEAR, false);
            glTexture.setParameteri(GL_TEXTURE_MIN_FILTER, GL_LINEAR, false);
            glTexture.setParameteri(GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE, false);
            glTexture.setParameteri(GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE, false);
        }
        auto screenVAO = GLuint(0);
        {
            glGenVertexArrays(1, &screenVAO);
            glBindVertexArray(screenVAO);
            screenVBO.bind();
            screenIBO.bind();
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 5, (void*)(sizeof(float) * 0));
            glEnableVertexAttribArray(0);
            glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 5, (void*)(sizeof(float) * 3));
            glEnableVertexAttribArray(1);
            glBindVertexArray(0);
            screenVBO.unbind();
            screenIBO.unbind();
        }

        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        GLint texLoc  = program.getUniformLocation("tex");
        glfwSetCursorPosCallback(window, [](GLFWwindow* wnd, double xPos, double yPos) {
            WindowState* pWindowState = (WindowState*)glfwGetWindowUserPointer(wnd);
            pWindowState->delCurPos.x = xPos - pWindowState->curCurPos.x;
            pWindowState->delCurPos.y = yPos - pWindowState->curCurPos.y;;
            pWindowState->curCurPos.x = xPos;
            pWindowState->curCurPos.y = yPos;
        });
        glfwSetTime(0.0f);
        {
            double xPos, yPos;
            glfwGetCursorPos(window, &xPos, &yPos);
            windowState.curCurPos.x = xPos;
            windowState.curCurPos.y = yPos;
            windowState.delCurPos.x = 0.0f;
            windowState.delCurPos.y = 0.0f;
        }
        curPipeline->paramsBuffer.cpuHandle.push_back(params);
        curPipeline->paramsBuffer.Upload();
        CUstream stream = nullptr;
        RTLIB_CUDA_CHECK(cudaStreamCreate(&stream));
        auto   frameBufferGL  = rtlib::GLInteropBuffer<uchar4>(width * height, GL_PIXEL_UNPACK_BUFFER, GL_DYNAMIC_DRAW, stream);
        bool   isResized      = false;
        bool   isUpdated      = false;
        bool   isFixedLight   = false;
        bool   isMovedCamera  = false;
        while (!glfwWindowShouldClose(window)) {

            if (isResized) {
                {
                    std::random_device rd;
                    std::mt19937 mt(rd());
                    std::vector<unsigned int> seeds(width * height);
                    std::generate(seeds.begin(), seeds.end(), mt);
                    seedBuffer.resize(width * height);
                    seedBuffer.upload(seeds);
                }
                frameBufferGL.resize(width * height);
                accumBuffer.resize(width * height);
                curPipeline->paramsBuffer.cpuHandle[0].accumBuffer = accumBuffer.getDevicePtr();
                curPipeline->paramsBuffer.cpuHandle[0].seed        = seedBuffer.getDevicePtr();
                curPipeline->paramsBuffer.cpuHandle[0].width       = width;
                curPipeline->paramsBuffer.cpuHandle[0].height      = height;
            }
            if (isMovedCamera) {
                auto camera = cameraController.GetCamera(30.0f, 1.0f);
                curPipeline->raygenBuffer.cpuHandle[0].data.eye = camera.getEye();
                auto [u, v, w] = camera.getUVW();
                curPipeline->raygenBuffer.cpuHandle[0].data.u = u;
                curPipeline->raygenBuffer.cpuHandle[0].data.v = v;
                curPipeline->raygenBuffer.cpuHandle[0].data.w = w;
                curPipeline->raygenBuffer.Upload();
                isUpdated = true;
            }
            if (isUpdated) {
                frameBufferGL.upload(std::vector<uchar4>(width * height));
                accumBuffer.upload(std::vector<float3>(  width * height));
                curPipeline->paramsBuffer.cpuHandle[0].samplePerALL = 0;
            }
            {
                curPipeline->width  = width;
                curPipeline->height = height;
                curPipeline->paramsBuffer.cpuHandle[0].frameBuffer = frameBufferGL.map();
                curPipeline->paramsBuffer.Upload();
                curPipeline->Launch(stream);
                cuStreamSynchronize(stream);
                frameBufferGL.unmap();
                curPipeline->paramsBuffer.cpuHandle[0].samplePerALL += curPipeline->paramsBuffer.cpuHandle[0].samplePerLaunch;
            }

            {
                glfwPollEvents();
                if (isResized) {
                    glTexture.reset();
                    glTexture.allocate({ (size_t)width,(size_t)height }, GL_TEXTURE_2D);
                    glTexture.setParameteri(GL_TEXTURE_MAG_FILTER, GL_LINEAR, false);
                    glTexture.setParameteri(GL_TEXTURE_MIN_FILTER, GL_LINEAR, false);
                    glTexture.setParameteri(GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE, false);
                    glTexture.setParameteri(GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE, false);
                }
                glTexture.upload(0, frameBufferGL.getHandle(), 0, 0, width, height);
                {
                    glViewport(0, 0, width, height);
                }
                glClear(GL_COLOR_BUFFER_BIT);
                program.use();
                glActiveTexture(GL_TEXTURE0);
                glTexture.bind();
                glUniform1i(texLoc, 0);
                glBindVertexArray(screenVAO);
                glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);
                glfwSwapBuffers(window);
                isUpdated = false;
                isResized = false;
                isMovedCamera = false;
                {
                    int tWidth, tHeight;
                    glfwGetWindowSize(window, &tWidth, &tHeight);
                    if (width != tWidth || height != tHeight) {
                        std::cout << width << "->" << tWidth << "\n";
                        std::cout << height << "->" << tHeight << "\n";
                        width = tWidth;
                        height = tHeight;
                        isResized = true;
                        isUpdated = true;
                    }
                    else {
                        isResized = false;
                    }
                    float prevTime = glfwGetTime();
                    windowState.delTime = windowState.curTime - prevTime;
                    windowState.curTime = prevTime;
                    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
                        cameraController.ProcessKeyboard(rtlib::CameraMovement::eForward, windowState.delTime);
                        isMovedCamera = true;
                    }
                    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
                        cameraController.ProcessKeyboard(rtlib::CameraMovement::eBackward, windowState.delTime);
                        isMovedCamera = true;
                    }
                    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
                        cameraController.ProcessKeyboard(rtlib::CameraMovement::eLeft, windowState.delTime);
                        isMovedCamera = true;
                    }
                    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
                        cameraController.ProcessKeyboard(rtlib::CameraMovement::eRight, windowState.delTime);
                        isMovedCamera = true;
                    }
                    if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS) {
                        cameraController.ProcessKeyboard(rtlib::CameraMovement::eLeft, windowState.delTime);
                        isMovedCamera = true;
                    }
                    if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS) {
                        cameraController.ProcessKeyboard(rtlib::CameraMovement::eRight, windowState.delTime);
                        isMovedCamera = true;
                    }
                    if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS) {
                        cameraController.ProcessKeyboard(rtlib::CameraMovement::eUp, windowState.delTime);
                        isMovedCamera = true;
                    }
                    if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS) {
                        cameraController.ProcessKeyboard(rtlib::CameraMovement::eDown, windowState.delTime);
                        isMovedCamera = true;
                    }
                    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
                        cameraController.ProcessMouseMovement(-windowState.delCurPos.x, windowState.delCurPos.y);
                        isMovedCamera = true;
                    }
                }

            }
        }
        auto img_pixels = std::vector<uchar4>();
        frameBufferGL.download(img_pixels);
        stbi_write_bmp("tekitou.bmp", width, height, 4, img_pixels.data());
        program.destroy();
        glDeleteVertexArrays(1, &screenVAO);
        screenVBO.reset();
        screenIBO.reset();
        glTexture.reset();
        glfwDestroyWindow(window);
        window = nullptr;
        glfwTerminate();
    }
    return 0;
}