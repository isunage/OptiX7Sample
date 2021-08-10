#include <Test18Config.h>
#include <RTLib/GL.h>
#include <RTLib/CUDA.h>
#include <RTLib/CUDA_GL.h>
#include <RTLib/Camera.h>
#include <RTLib/Utils.h>
#include <RTLib/ext/RectRenderer.h>
#include <cuda/RayTrace.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include "../include/PathTracer.h"
#include "../include/SceneBuilder.h"
#include <fstream>
#include <unordered_map>
#include <random>
#include <sstream>
#include <string>
class Test18Application {
public:
	bool InitGLFW(int gl_version_major, int gl_version_minor) {
		if (glfwInit() == GLFW_FALSE) {
			return false;
		}
		glfwWindowHint(GLFW_VERSION_MAJOR, 3);
		glfwWindowHint(GLFW_VERSION_MINOR, 3);
		glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API);
		glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, gl_version_major);
		glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, gl_version_minor);
		glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
		glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
		std::stringstream ss;
		ss << "#version " << gl_version_major << gl_version_minor << "0 core";
		m_GlslVersion = ss.str();
		return true;
	}
	bool InitWindow(int width, int height, const std::string& title) {
		m_Window = glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr);
		if (!m_Window) {
			return false;
		}
		glfwMakeContextCurrent(m_Window);
		glfwSetWindowUserPointer(m_Window, this);
		glfwSetMouseButtonCallback(m_Window,ImGui_ImplGlfw_MouseButtonCallback);
		glfwSetKeyCallback(m_Window,ImGui_ImplGlfw_KeyCallback);
		glfwSetCharCallback(m_Window,ImGui_ImplGlfw_CharCallback);
		glfwSetScrollCallback(m_Window,ImGui_ImplGlfw_ScrollCallback);
		glfwSetCursorPosCallback(m_Window, cursorPosCallback);
		glfwSetFramebufferSizeCallback(m_Window, frameBufferSizeCallback);
		glfwGetFramebufferSize(m_Window, &m_FbWidth, &m_FbHeight);
		return true;
	}
	bool InitGLAD() {
		if (!glfwGetCurrentContext()) {
			return false;
		}
		if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
			return false;
		}
		return true;
	}
	bool InitImGui() {
		IMGUI_CHECKVERSION();
		ImGui::CreateContext();
		ImGuiIO& io = ImGui::GetIO(); (void)io;
		ImGui::StyleColorsDark();
		if (!ImGui_ImplGlfw_InitForOpenGL(m_Window, false)) {
			return false;
		}
		if (!ImGui_ImplOpenGL3_Init(m_GlslVersion.c_str())) {
			return false;
		}
		return true;
	}
	void InitOptix (){
		m_Tracer.InitCUDA();
    	m_Tracer.InitOPX();
	}
	void InitCamera(){
		m_CameraController = rtlib::CameraController({ 0.0f,1.0f, 5.0f });
		m_CameraController.SetMouseSensitivity(0.125f);
		m_CameraController.SetMovementSpeed(50.0f);
	}
	void LoadScene(){
		auto objMeshGroup = std::make_shared<test::ObjMeshGroup>();
		if (!objMeshGroup->Load(TEST_TEST18_DATA_PATH"/Models/Sponza/sponza.obj", TEST_TEST18_DATA_PATH"/Models/Sponza/")) {
			throw std::runtime_error("Failed To Load Model!");
		}
		m_MaterialSet = objMeshGroup->GetMaterialSet();

		{
			for (auto& material : m_MaterialSet->materials) {
				auto diffTex = material.diffTex != "" ? material.diffTex : std::string(TEST_TEST18_DATA_PATH"/Textures/white.png");
				auto specTex = material.specTex != "" ? material.specTex : std::string(TEST_TEST18_DATA_PATH"/Textures/white.png");
				auto emitTex = material.emitTex != "" ? material.emitTex : std::string(TEST_TEST18_DATA_PATH"/Textures/white.png");
				if (!m_Tracer.HasTexture(material.diffTex)) {
					 m_Tracer.LoadTexture(material.diffTex, diffTex);
				}
				if (!m_Tracer.HasTexture(material.specTex)) {
				  	 m_Tracer.LoadTexture(material.specTex, specTex);
				}
				if (!m_Tracer.HasTexture(material.emitTex)) {
					m_Tracer.LoadTexture(material.emitTex, emitTex);
				}
			}
		}
		bool isLightFound   = false;
		//GAS1: World
		auto worldGASHandle = std::make_shared<rtlib::ext::GASHandle>();
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
			worldGASHandle->Build(m_Tracer.GetOPXContext().get(), accelOptions);

		}
		//GAS2: Light
		auto lightGASHandle = std::make_shared<rtlib::ext::GASHandle>();
		if(isLightFound){
			OptixAccelBuildOptions accelOptions = {};
			accelOptions.buildFlags             = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
			accelOptions.operation              = OPTIX_BUILD_OPERATION_BUILD;
			lightGASHandle->meshes.push_back(objMeshGroup->GetMeshGroup()->LoadMesh("light"));
			lightGASHandle->Build(m_Tracer.GetOPXContext().get(), accelOptions);
		}
		else {
			rtlib::utils::AABB aabb = {};
			for (auto& vertex : objMeshGroup->GetMeshGroup()->GetSharedResource()->vertexBuffer.cpuHandle) {
				aabb.Update(vertex);
			}
			OptixAccelBuildOptions accelOptions = {};
			accelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
			accelOptions.operation  = OPTIX_BUILD_OPERATION_BUILD;
			auto lightMesh = rtlib::ext::Mesh::New();
			lightMesh->SetSharedResource(rtlib::ext::MeshSharedResource::New());
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
			unsigned int curMaterialSetCount = m_MaterialSet->materials.size();
			auto lightMaterial               = test::PhongMaterial{};
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
			m_MaterialSet->materials.push_back(
				lightMaterial
			);
			lightMesh->GetSharedResource()->vertexBuffer.Upload();
			lightMesh->GetSharedResource()->texCrdBuffer.Upload();
			lightMesh->GetSharedResource()->normalBuffer.Upload();
			lightMesh->SetUniqueResource(rtlib::ext::MeshUniqueResource::New());
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
			lightGASHandle->Build(m_Tracer.GetOPXContext().get(), accelOptions);
		}
		m_Tracer.SetGASHandle("Sponza-World", worldGASHandle);
		m_Tracer.SetGASHandle("Light", lightGASHandle);
		//IAS1: First
		auto firstIASHandle                     = std::make_shared<rtlib::ext::IASHandle>();
		{
			OptixAccelBuildOptions accelOptions = {};
			accelOptions.buildFlags             = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
			accelOptions.operation              = OPTIX_BUILD_OPERATION_BUILD;
			auto worldInstance                  = m_Tracer.GetInstance("Sponza-World");
			auto lightInstance                  = m_Tracer.GetInstance("Light");
			lightInstance.instance.sbtOffset    = worldInstance.baseGASHandle->sbtCount * RAY_TYPE_COUNT;
			firstIASHandle->instanceSets.resize(1);
			firstIASHandle->instanceSets[0]     = std::make_shared<rtlib::ext::InstanceSet>();
			firstIASHandle->instanceSets[0]->SetInstance(worldInstance);
			firstIASHandle->instanceSets[0]->SetInstance(lightInstance);
			firstIASHandle->instanceSets[0]->instanceBuffer.Upload();
			firstIASHandle->Build(m_Tracer.GetOPXContext().get(), accelOptions);

		}
		m_Tracer.SetIASHandle("First", firstIASHandle);
	}
	void InitLight() {
		m_Light = ParallelLight();
		{
			auto& lightGASHandle = m_Tracer.m_GASHandles["Light"];
			auto  lightMesh      = lightGASHandle->meshes[0];
			auto  lightVertices  = std::vector<float3>();
			for (auto& index : lightMesh->GetUniqueResource()->triIndBuffer.cpuHandle) {
				lightVertices.push_back(lightMesh->GetSharedResource()->vertexBuffer.cpuHandle[index.x]);
				lightVertices.push_back(lightMesh->GetSharedResource()->vertexBuffer.cpuHandle[index.y]);
				lightVertices.push_back(lightMesh->GetSharedResource()->vertexBuffer.cpuHandle[index.z]);
			}
			auto lightAABB = rtlib::utils::AABB(lightVertices);
			auto lightV3 = lightAABB.max - lightAABB.min;
			m_Light.corner = lightAABB.min;
			m_Light.v1 = make_float3(0.0f, 0.0f, lightV3.z);
			m_Light.v2 = make_float3(lightV3.x, 0.0f, 0.0f);
			m_Light.normal = make_float3(0.0f, -1.0f, 0.0f);
			auto lightMaterial = m_MaterialSet->materials[lightMesh->GetUniqueResource()->materials[0]];
			m_Light.emission = lightMaterial.emitCol;
		}
	}
	void InitFrameResources(){
		RTLIB_CUDA_CHECK(cudaStreamCreate(&m_Stream));
	 	m_FrameBuffer   = rtlib::CUDABuffer<uchar4>(std::vector<uchar4>(m_FbWidth * m_FbHeight));
		auto frameBufferGL = rtlib::GLInteropBuffer<uchar4>(m_FbWidth * m_FbHeight, GL_PIXEL_UNPACK_BUFFER, GL_DYNAMIC_DRAW, m_Stream);

		m_FrameBufferGL = rtlib::GLInteropBuffer<uchar4>(m_FbWidth * m_FbHeight, GL_PIXEL_UNPACK_BUFFER, GL_DYNAMIC_DRAW, m_Stream);

    	m_AccumBuffer   = rtlib::CUDABuffer<float3>(std::vector<float3>(m_FbWidth * m_FbHeight));
   		m_SeedBuffer    = rtlib::CUDABuffer<unsigned int>();
		{
			std::vector<unsigned int> seeds(m_FbWidth * m_FbHeight);
			std::random_device rd;
			std::mt19937 mt(rd());
			std::generate(seeds.begin(), seeds.end(), mt);
			m_SeedBuffer.allocate(seeds.size());
			m_SeedBuffer.upload(seeds);
		}
		m_GLTexture = rtlib::GLTexture2D<uchar4>();
		{
			m_GLTexture.allocate({ (size_t)m_FbWidth, (size_t)m_FbHeight });
			m_GLTexture.setParameteri(GL_TEXTURE_MAG_FILTER, GL_LINEAR, false);
			m_GLTexture.setParameteri(GL_TEXTURE_MIN_FILTER, GL_LINEAR, false);
			m_GLTexture.setParameteri(GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE, false);
			m_GLTexture.setParameteri(GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE, false);
		}
	}
	void InitTracePipeline()
    {
        auto tracePipeline = std::make_shared<test::Pipeline>();
        {
			tracePipeline->SetContext(m_Tracer.GetOPXContext());

			OptixPipelineCompileOptions compileOptions = {};

			compileOptions.pipelineLaunchParamsVariableName = "params";
			compileOptions.numAttributeValues               = 3;
			compileOptions.numPayloadValues                 = 3;
			compileOptions.usesPrimitiveTypeFlags           = 0;
			compileOptions.usesMotionBlur                   = false;
			compileOptions.traversableGraphFlags            = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;

			tracePipeline->InitPipeline(compileOptions);
        }
        {
            tracePipeline->width  = m_FbWidth;
            tracePipeline->height = m_FbHeight;
            tracePipeline->depth  = 2;
        }
        //module: Load
        {
            OptixModuleCompileOptions moduleCompileOptions = {};
            moduleCompileOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
            moduleCompileOptions.numBoundValues = 0;
#ifndef NDEBUG
			moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;
			moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
#else
			moduleCompileOptions.optLevel   = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
			moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
#endif
			tracePipeline->LoadModuleFromPtxFile("RayTrace", TEST_TEST18_CUDA_PATH"/RayTrace.ptx", moduleCompileOptions);
        }
        //program group: init
        {
			tracePipeline->LoadRgProgramGroupFromModule({ "RayTrace", "__raygen__rg" });
			tracePipeline->LoadMsProgramGroupFromModule("Radiance"  , { "RayTrace" , "__miss__radiance" });
			tracePipeline->LoadMsProgramGroupFromModule("Occlusion" , { "RayTrace" , "__miss__occluded" });
			tracePipeline->LoadHgProgramGroupFromModule("Diffuse"   , {"RayTrace" ,"__closesthit__radiance_for_diffuse"}, {}, {});
			tracePipeline->LoadHgProgramGroupFromModule("Specular"  , { "RayTrace" ,"__closesthit__radiance_for_specular" }, {}, {});
			tracePipeline->LoadHgProgramGroupFromModule("Refraction", { "RayTrace" ,"__closesthit__radiance_for_refraction" }, {}, {});
			tracePipeline->LoadHgProgramGroupFromModule("Emission"  , { "RayTrace" ,"__closesthit__radiance_for_emission" }, {}, {});
			tracePipeline->LoadHgProgramGroupFromModule("Occlusion" , { "RayTrace" ,"__closesthit__occluded" }, {}, {});
        }
        //pipeline link
        {
			OptixPipelineLinkOptions linkOptions = {};
			linkOptions.maxTraceDepth = 2;
#ifndef NDEBUG
			linkOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;
#else
			linkOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
#endif

			tracePipeline->LinkPipeline(linkOptions);
        }
        //SBTRecord
        {
            tracePipeline->raygenBuffer.cpuHandle.resize(1);
            auto camera = m_CameraController.GetCamera(30.0f, 1.0f);
            auto [u, v, w] = camera.getUVW();
            tracePipeline->raygenBuffer.cpuHandle[0]          = tracePipeline->rgProgramGroup.getSBTRecord<RayGenData>();
            tracePipeline->raygenBuffer.cpuHandle[0].data.eye = camera.getEye();
            tracePipeline->raygenBuffer.cpuHandle[0].data.u   = u;
            tracePipeline->raygenBuffer.cpuHandle[0].data.v   = v;
            tracePipeline->raygenBuffer.cpuHandle[0].data.w   = w;
            tracePipeline->raygenBuffer.Upload();
            tracePipeline->missBuffer.cpuHandle.resize(RAY_TYPE_COUNT);
            tracePipeline->missBuffer.cpuHandle[RAY_TYPE_RADIANCE]  = tracePipeline->msProgramGroups["Radiance"].getSBTRecord<MissData>();
            tracePipeline->missBuffer.cpuHandle[RAY_TYPE_RADIANCE].data.bgColor = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            tracePipeline->missBuffer.cpuHandle[RAY_TYPE_OCCLUSION] = tracePipeline->msProgramGroups["Occlusion"].getSBTRecord<MissData>();
            tracePipeline->missBuffer.cpuHandle[RAY_TYPE_OCCLUSION].data.bgColor = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            tracePipeline->missBuffer.Upload();
            tracePipeline->hitGBuffer.cpuHandle.resize(RAY_TYPE_COUNT * m_Tracer.m_IASHandles["First"]->sbtCount);
            auto& cpuHgRecords = tracePipeline->hitGBuffer.cpuHandle;
            for (auto& [name, iasHandle] : m_Tracer.m_IASHandles) {
                size_t sbtOffset = 0;
                for (auto& instanceSet : iasHandle->instanceSets) {
                    for (auto& baseGASHandle : instanceSet->baseGASHandles) {
                        for (auto& mesh : baseGASHandle->meshes) {
                            for (size_t i = 0; i < mesh->GetUniqueResource()->materials.size(); ++i) {
                                auto materialId = mesh->GetUniqueResource()->materials[i];
                                auto& material = m_MaterialSet->materials[materialId];
                                HitgroupData radianceHgData = {};
                                {
                                    radianceHgData.vertices    = mesh->GetSharedResource()->vertexBuffer.gpuHandle.getDevicePtr();
                                    radianceHgData.indices     = mesh->GetUniqueResource()->triIndBuffer.gpuHandle.getDevicePtr();
                                    radianceHgData.texCoords   = mesh->GetSharedResource()->texCrdBuffer.gpuHandle.getDevicePtr();
                                    radianceHgData.diffuseTex  = m_Tracer.GetTexture(material.diffTex).getHandle();
                                    radianceHgData.specularTex = m_Tracer.GetTexture(material.specTex).getHandle();
                                    radianceHgData.emissionTex = m_Tracer.GetTexture(material.emitTex).getHandle();
                                    radianceHgData.diffuse  = material.diffCol;
                                    radianceHgData.specular = material.specCol;
                                    radianceHgData.emission = material.emitCol;
                                    radianceHgData.shinness = material.shinness;
                                    radianceHgData.transmit = material.tranCol;
                                    radianceHgData.refrInd  = material.refrInd;
                                }
								if (material.name == "light") {
									m_LightHgRecIndex = RAY_TYPE_COUNT * sbtOffset + RAY_TYPE_COUNT * i + RAY_TYPE_RADIANCE;
								}
                                if (material.type == test::PhongMaterialType::eDiffuse) {
                                    cpuHgRecords[RAY_TYPE_COUNT * sbtOffset + RAY_TYPE_COUNT * i + RAY_TYPE_RADIANCE] = tracePipeline->hgProgramGroups["Diffuse"].getSBTRecord<HitgroupData>(radianceHgData);
                                }
                                else if (material.type == test::PhongMaterialType::eSpecular) {
                                    cpuHgRecords[RAY_TYPE_COUNT * sbtOffset + RAY_TYPE_COUNT * i + RAY_TYPE_RADIANCE] = tracePipeline->hgProgramGroups["Specular"].getSBTRecord<HitgroupData>(radianceHgData);
                                }
                                else if (material.type == test::PhongMaterialType::eRefract){
                                    cpuHgRecords[RAY_TYPE_COUNT * sbtOffset + RAY_TYPE_COUNT * i + RAY_TYPE_RADIANCE] = tracePipeline->hgProgramGroups["Refraction"].getSBTRecord<HitgroupData>(radianceHgData);
                                }
                                else {
                                    cpuHgRecords[RAY_TYPE_COUNT * sbtOffset + RAY_TYPE_COUNT * i + RAY_TYPE_RADIANCE] = tracePipeline->hgProgramGroups["Emission"].getSBTRecord<HitgroupData>(radianceHgData);
                                }
                                cpuHgRecords[RAY_TYPE_COUNT * sbtOffset + RAY_TYPE_COUNT * i + RAY_TYPE_OCCLUSION]    = tracePipeline->hgProgramGroups["Occlusion"].getSBTRecord<HitgroupData>();

                            }
                            sbtOffset += mesh->GetUniqueResource()->materials.size();
                        }
                    }
                }
            }
            tracePipeline->hitGBuffer.Upload();
			{
				tracePipeline->shaderbindingTable = {};
				tracePipeline->shaderbindingTable.raygenRecord = reinterpret_cast<CUdeviceptr>(tracePipeline->raygenBuffer.gpuHandle.getDevicePtr());
				tracePipeline->shaderbindingTable.missRecordBase = reinterpret_cast<CUdeviceptr>(tracePipeline->missBuffer.gpuHandle.getDevicePtr());
				tracePipeline->shaderbindingTable.missRecordCount = tracePipeline->missBuffer.gpuHandle.getCount();
				tracePipeline->shaderbindingTable.missRecordStrideInBytes = sizeof(rtlib::SBTRecord<MissData>);
				tracePipeline->shaderbindingTable.hitgroupRecordBase = reinterpret_cast<CUdeviceptr>(tracePipeline->hitGBuffer.gpuHandle.getDevicePtr());
				tracePipeline->shaderbindingTable.hitgroupRecordCount = tracePipeline->hitGBuffer.gpuHandle.getCount();
				tracePipeline->shaderbindingTable.hitgroupRecordStrideInBytes = sizeof(rtlib::SBTRecord<HitgroupData>);
			}

			{
				tracePipeline->paramsBuffer.cpuHandle.resize(1);
				auto& params = tracePipeline->paramsBuffer.cpuHandle[0];
				{
					params.frameBuffer = m_FrameBuffer.getDevicePtr();
					params.accumBuffer = m_AccumBuffer.getDevicePtr();
					params.seed = m_SeedBuffer.getDevicePtr();
					params.width = m_FbWidth;
					params.height = m_FbHeight;
					params.maxTraceDepth = m_MaxTraceDepth;
					params.gasHandle = m_Tracer.m_IASHandles["First"]->handle;
					params.light = m_Light;
					params.samplePerALL = 0;
					params.samplePerLaunch = m_SamplePerLaunch;
				}
			}
        }
        m_Tracer.SetPipeline("Trace", tracePipeline);

    }
	void InitDebugPipeline() {
		auto debugPipeline = std::make_shared<test::Pipeline>();
		{
			debugPipeline->SetContext(m_Tracer.GetOPXContext());

			OptixPipelineCompileOptions compileOptions = {};

			debugPipeline->width  = m_FbWidth;
			debugPipeline->height = m_FbHeight;
			debugPipeline->depth  = 1;
			//compileOptions
			compileOptions.pipelineLaunchParamsVariableName = "params";
			compileOptions.numAttributeValues               = 3;
			compileOptions.numPayloadValues                 = 3;
			compileOptions.usesPrimitiveTypeFlags           = 0;
			compileOptions.usesMotionBlur                   = false;
			compileOptions.traversableGraphFlags            = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;

			debugPipeline->InitPipeline(compileOptions);
		}
		//module: Load
		{
			OptixModuleCompileOptions moduleCompileOptions = {};
			moduleCompileOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
			moduleCompileOptions.numBoundValues   = 0;
#ifndef NDEBUG
			moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;
			moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
#else
			moduleCompileOptions.optLevel   = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
			moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
#endif
			debugPipeline->LoadModuleFromPtxFile("RayDebug", TEST_TEST18_CUDA_PATH"/RayDebug.ptx", moduleCompileOptions);
		}
		//program group: init
		{
			debugPipeline->LoadRgProgramGroupFromModule({ "RayDebug", "__raygen__debug" });
			debugPipeline->LoadMsProgramGroupFromModule( "Radiance" , { "RayDebug" , "__miss__debug" });
			debugPipeline->LoadMsProgramGroupFromModule( "Occlusion", { "RayDebug" , "__miss__debug" });
			debugPipeline->LoadHgProgramGroupFromModule( "Radiance" , { "RayDebug" ,"__closesthit__debug" }, {}, {});
			debugPipeline->LoadHgProgramGroupFromModule( "Occlusion", { "RayDebug" ,"__closesthit__debug" }, {}, {});
		}
		//pipeline link
		{
			OptixPipelineLinkOptions linkOptions = {};
			linkOptions.maxTraceDepth = 2;
#ifndef NDEBUG
			linkOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;
#else
			linkOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
#endif

			debugPipeline->LinkPipeline(linkOptions);
		}
		//SBTRecord
		{
			debugPipeline->raygenBuffer.cpuHandle.resize(1);
			auto camera    = m_CameraController.GetCamera(30.0f, 1.0f);
			auto [u, v, w] = camera.getUVW();
			debugPipeline->raygenBuffer.cpuHandle[0] = debugPipeline->rgProgramGroup.getSBTRecord<RayGenData>();
			debugPipeline->raygenBuffer.cpuHandle[0].data.eye = camera.getEye();
			debugPipeline->raygenBuffer.cpuHandle[0].data.u = u;
			debugPipeline->raygenBuffer.cpuHandle[0].data.v = v;
			debugPipeline->raygenBuffer.cpuHandle[0].data.w = w;
			debugPipeline->raygenBuffer.Upload();
			debugPipeline->missBuffer.cpuHandle.resize(RAY_TYPE_COUNT);
			debugPipeline->missBuffer.cpuHandle[RAY_TYPE_RADIANCE]  = debugPipeline->msProgramGroups["Radiance"].getSBTRecord<MissData>();
			debugPipeline->missBuffer.cpuHandle[RAY_TYPE_RADIANCE].data.bgColor = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
			debugPipeline->missBuffer.cpuHandle[RAY_TYPE_OCCLUSION] = debugPipeline->msProgramGroups["Occlusion"].getSBTRecord<MissData>();
			debugPipeline->missBuffer.cpuHandle[RAY_TYPE_OCCLUSION].data.bgColor = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
			debugPipeline->missBuffer.Upload();
			debugPipeline->hitGBuffer.cpuHandle.resize(RAY_TYPE_COUNT * m_Tracer.m_IASHandles["First"]->sbtCount);
			auto& cpuHgRecords = debugPipeline->hitGBuffer.cpuHandle;
			for (auto& [name, iasHandle] : m_Tracer.m_IASHandles) {
				size_t sbtOffset = 0;
				for (auto& instanceSet : iasHandle->instanceSets) {
					for (auto& baseGASHandle : instanceSet->baseGASHandles) {
						for (auto& mesh : baseGASHandle->meshes) {
							for (size_t i = 0; i < mesh->GetUniqueResource()->materials.size(); ++i) {
								auto materialId = mesh->GetUniqueResource()->materials[i];
								auto& material = m_MaterialSet->materials[materialId];
								HitgroupData radianceHgData = {};
								{
									radianceHgData.vertices    = mesh->GetSharedResource()->vertexBuffer.gpuHandle.getDevicePtr();
									radianceHgData.indices     = mesh->GetUniqueResource()->triIndBuffer.gpuHandle.getDevicePtr();
									radianceHgData.texCoords   = mesh->GetSharedResource()->texCrdBuffer.gpuHandle.getDevicePtr();
									radianceHgData.diffuseTex  = m_Tracer.GetTexture(material.diffTex).getHandle();
									radianceHgData.specularTex = m_Tracer.GetTexture(material.specTex).getHandle();
									radianceHgData.emissionTex = m_Tracer.GetTexture(material.emitTex).getHandle();
									radianceHgData.diffuse     = material.diffCol;
									radianceHgData.specular    = material.specCol;
									radianceHgData.emission    = material.emitCol;
									radianceHgData.transmit    = material.tranCol;
									radianceHgData.shinness    = material.shinness;
									//printf("%lf %lf %lf\n",radianceHgData.transmit.x,radianceHgData.transmit.y,radianceHgData.transmit.z);
								}
								cpuHgRecords[RAY_TYPE_COUNT * sbtOffset + RAY_TYPE_COUNT * i + RAY_TYPE_RADIANCE]  = debugPipeline->hgProgramGroups["Radiance" ].getSBTRecord<HitgroupData>(radianceHgData);
								cpuHgRecords[RAY_TYPE_COUNT * sbtOffset + RAY_TYPE_COUNT * i + RAY_TYPE_OCCLUSION] = debugPipeline->hgProgramGroups["Occlusion"].getSBTRecord<HitgroupData>();
							}
							sbtOffset += mesh->GetUniqueResource()->materials.size();
						}
					}
				}
			}
			debugPipeline->hitGBuffer.Upload();
			{
				debugPipeline->shaderbindingTable = {};
				debugPipeline->shaderbindingTable.raygenRecord                = reinterpret_cast<CUdeviceptr>(debugPipeline->raygenBuffer.gpuHandle.getDevicePtr());
				debugPipeline->shaderbindingTable.missRecordBase              = reinterpret_cast<CUdeviceptr>(debugPipeline->missBuffer.gpuHandle.getDevicePtr());
				debugPipeline->shaderbindingTable.missRecordCount             = debugPipeline->missBuffer.gpuHandle.getCount();
				debugPipeline->shaderbindingTable.missRecordStrideInBytes     = sizeof(rtlib::SBTRecord<MissData>);
				debugPipeline->shaderbindingTable.hitgroupRecordBase          = reinterpret_cast<CUdeviceptr>(debugPipeline->hitGBuffer.gpuHandle.getDevicePtr());
				debugPipeline->shaderbindingTable.hitgroupRecordCount         = debugPipeline->hitGBuffer.gpuHandle.getCount();
				debugPipeline->shaderbindingTable.hitgroupRecordStrideInBytes = sizeof(rtlib::SBTRecord<HitgroupData>);
			}

			{
				debugPipeline->paramsBuffer.cpuHandle.resize(1);
				auto& params = debugPipeline->paramsBuffer.cpuHandle[0];
				{
					params.frameBuffer     = m_FrameBuffer.getDevicePtr();
					params.accumBuffer     = m_AccumBuffer.getDevicePtr();
					params.seed            = m_SeedBuffer.getDevicePtr();
					params.width           = m_FbWidth;
					params.height          = m_FbHeight;
					params.maxTraceDepth   = m_MaxTraceDepth;
					params.gasHandle       = m_Tracer.m_IASHandles["First"]->handle;
					params.light           = m_Light;
					params.samplePerALL    = 0;
					params.samplePerLaunch = m_SamplePerLaunch;
				}
			}
		}
		m_Tracer.SetPipeline("Debug", debugPipeline);
	}
	void MainLoop() {
		PrepareMainLoop();
		while (!glfwWindowShouldClose(m_Window)) {
			{
				if (m_Resized     ) {
					this->OnResize();
					m_Resized    = false;
					//ResizeはFlushする必要がある
					m_FlushFrame = true;
				}
				if (m_UpdateCamera) {
					this->OnUpdateCamera();
					m_UpdateCamera = false;
					//Cameraの移動はFlushする必要がある
					m_FlushFrame   = true;
				}
				if (m_UpdateLight ) {
					this->OnUpdateLight();
					m_UpdateLight  = false;
					m_FlushFrame   = true;
				}
				if (m_FlushFrame  ) {
					this->OnFlushFrame();
					m_FlushFrame   = false;
					//Flushした場合、Paramの再設定が必要
					m_UpdateParams = true;
				}
				if (m_UpdateParams) {
					this->OnUpdateParams();
					m_UpdateParams = false;
				}
				this->OnLaunch();
				this->OnRenderFrame();
				this->OnRenderImGUI();
				this->OnUpdateTime();
				this->OnGetInputs();
				glfwSwapBuffers(m_Window);
				glfwPollEvents();
			}
		}
	}
	void CleanUpImGui() {
		ImGui_ImplOpenGL3_Shutdown();
		ImGui_ImplGlfw_Shutdown();
		ImGui::DestroyContext();
	}
	void CleanUpWindow() {
		glfwDestroyWindow(m_Window);
		m_Window = nullptr;
	}
	void CleanUpGLFW() {
		glfwTerminate();
	}
private:
	void PrepareMainLoop() {
		
		m_RectRenderer = std::make_shared<rtlib::ext::RectRenderer>();
		m_RectRenderer->init();
		glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
		glfwSetTime(0.0f);
		{
			double xPos, yPos;
			glfwGetCursorPos(m_Window, &xPos, &yPos);
			this->m_DelCursorPos.x = 0.0f;
			this->m_DelCursorPos.y = 0.0f;
			this->m_CurCursorPos.x = xPos;
			this->m_CurCursorPos.y = yPos;
		}
	}
	void OnResize() {
		std::random_device rd;
		std::mt19937 mt(rd());
		std::vector<unsigned int> seeds(m_FbWidth * m_FbHeight);
		std::generate(seeds.begin(), seeds.end(), mt);

		m_SeedBuffer.resize(m_FbWidth * m_FbHeight);
		m_SeedBuffer.upload(seeds);
		m_FrameBufferGL.resize(m_FbWidth * m_FbHeight);
		m_AccumBuffer.resize(m_FbWidth * m_FbHeight);
		{
			m_GLTexture.reset();
			m_GLTexture.allocate({ (size_t)m_FbWidth,(size_t)m_FbHeight }, GL_TEXTURE_2D);
			m_GLTexture.setParameteri(GL_TEXTURE_MAG_FILTER, GL_LINEAR, false);
			m_GLTexture.setParameteri(GL_TEXTURE_MIN_FILTER, GL_LINEAR, false);
			m_GLTexture.setParameteri(GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE, false);
			m_GLTexture.setParameteri(GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE, false);
		}
		m_Tracer.m_Pipelines["Trace"]->paramsBuffer.cpuHandle[0].accumBuffer = m_AccumBuffer.getDevicePtr();
		m_Tracer.m_Pipelines["Trace"]->paramsBuffer.cpuHandle[0].seed = m_SeedBuffer.getDevicePtr();
		m_Tracer.m_Pipelines["Trace"]->paramsBuffer.cpuHandle[0].width = m_FbWidth;
		m_Tracer.m_Pipelines["Trace"]->paramsBuffer.cpuHandle[0].height = m_FbHeight;
		m_Tracer.m_Pipelines["Debug"]->paramsBuffer.cpuHandle[0].accumBuffer = m_AccumBuffer.getDevicePtr();
		m_Tracer.m_Pipelines["Debug"]->paramsBuffer.cpuHandle[0].seed = m_SeedBuffer.getDevicePtr();
		m_Tracer.m_Pipelines["Debug"]->paramsBuffer.cpuHandle[0].width = m_FbWidth;
		m_Tracer.m_Pipelines["Debug"]->paramsBuffer.cpuHandle[0].height = m_FbHeight;
	}
	void OnUpdateCamera()
	{
		auto camera    = m_CameraController.GetCamera(30.0f, 1.0f);
		auto [u, v, w] = camera.getUVW();
		m_Tracer.m_Pipelines["Trace"]->raygenBuffer.cpuHandle[0].data.eye = camera.getEye();
		m_Tracer.m_Pipelines["Trace"]->raygenBuffer.cpuHandle[0].data.u   = u;
		m_Tracer.m_Pipelines["Trace"]->raygenBuffer.cpuHandle[0].data.v   = v;
		m_Tracer.m_Pipelines["Trace"]->raygenBuffer.cpuHandle[0].data.w   = w;
		m_Tracer.m_Pipelines["Trace"]->updateRG = true;

		m_Tracer.m_Pipelines["Debug"]->raygenBuffer.cpuHandle[0].data.eye = camera.getEye();
		m_Tracer.m_Pipelines["Debug"]->raygenBuffer.cpuHandle[0].data.u   = u;
		m_Tracer.m_Pipelines["Debug"]->raygenBuffer.cpuHandle[0].data.v   = v;
		m_Tracer.m_Pipelines["Debug"]->raygenBuffer.cpuHandle[0].data.w   = w;
		m_Tracer.m_Pipelines["Debug"]->updateRG = true;
	}
	void OnUpdateLight() {
		m_Tracer.m_Pipelines["Trace"]->hitGBuffer.cpuHandle[m_LightHgRecIndex].data.emission = m_Light.emission;
		m_Tracer.m_Pipelines["Trace"]->hitGBuffer.cpuHandle[m_LightHgRecIndex].data.diffuse  = m_Light.emission;
		m_Tracer.m_Pipelines["Trace"]->updateHG = true;

		m_Tracer.m_Pipelines["Debug"]->hitGBuffer.cpuHandle[m_LightHgRecIndex].data.emission = m_Light.emission;
		m_Tracer.m_Pipelines["Debug"]->hitGBuffer.cpuHandle[m_LightHgRecIndex].data.diffuse  = m_Light.emission;
		m_Tracer.m_Pipelines["Debug"]->updateHG = true;
	}
	void OnFlushFrame() {
		//Frameの再取得
		m_FrameBufferGL.upload(std::vector<uchar4>(m_FbWidth * m_FbHeight));
		m_AccumBuffer.upload(std::vector<float3>(m_FbWidth * m_FbHeight));
		m_Tracer.m_Pipelines["Trace"]->paramsBuffer.cpuHandle[0].samplePerALL = 0;
		m_Tracer.m_Pipelines["Debug"]->paramsBuffer.cpuHandle[0].samplePerALL = 0;
	}
	void OnUpdateParams()
	{
		//paramsの再設定
		{
			m_Tracer.m_Pipelines["Trace"]->width = m_FbWidth;
			m_Tracer.m_Pipelines["Trace"]->height = m_FbHeight;
			auto& params           = m_Tracer.m_Pipelines["Trace"]->paramsBuffer.cpuHandle[0];
			params.frameBuffer     = m_FrameBuffer.getDevicePtr();
			params.accumBuffer     = m_AccumBuffer.getDevicePtr();
			params.seed            = m_SeedBuffer.getDevicePtr();
			params.width           = m_FbWidth;
			params.height          = m_FbHeight;
			params.maxTraceDepth   = m_MaxTraceDepth;
			params.gasHandle       = m_Tracer.m_IASHandles["First"]->handle;
			params.light           = m_Light;
			params.samplePerLaunch = m_SamplePerLaunch;
		}
		{
			m_Tracer.m_Pipelines["Debug"]->width = m_FbWidth;
			m_Tracer.m_Pipelines["Debug"]->height = m_FbHeight;
			auto& params = m_Tracer.m_Pipelines["Debug"]->paramsBuffer.cpuHandle[0];
			params.frameBuffer     = m_FrameBuffer.getDevicePtr();
			params.accumBuffer     = m_AccumBuffer.getDevicePtr();
			params.seed            = m_SeedBuffer.getDevicePtr();
			params.width           = m_FbWidth;
			params.height          = m_FbHeight;
			params.maxTraceDepth   = m_MaxTraceDepth;
			params.gasHandle       = m_Tracer.m_IASHandles["First"]->handle;
			params.light           = m_Light;
			params.samplePerLaunch = m_SamplePerLaunch;
		}
	}
	void OnUpdateTime() {
		float prevTime = glfwGetTime();
		m_DelTime      = prevTime - m_CurTime;
		m_CurTime      = prevTime;
	}
	void OnLaunch() {
		auto& curPipeline     = m_Tracer.m_Pipelines[curPipelineName];
		auto& params          = curPipeline->paramsBuffer.cpuHandle[0];
		params.frameBuffer    = m_FrameBufferGL.map();
		curPipeline->updatePm = true;
		curPipeline->Update();
		curPipeline->Launch(m_Stream);
		cuStreamSynchronize(m_Stream);
		m_FrameBufferGL.unmap();
		curPipeline->paramsBuffer.cpuHandle[0].samplePerALL += curPipeline->paramsBuffer.cpuHandle[0].samplePerLaunch;
	}
	void OnRenderFrame() {
		m_GLTexture.upload(0, m_FrameBufferGL.getHandle(), 0, 0, m_FbWidth, m_FbHeight);
		glClearColor(0.8f, 0.8f, 0.8f, 0.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glViewport(0, 0, m_FbWidth, m_FbHeight);
		m_RectRenderer->draw(m_GLTexture.getID());
	}
	void OnRenderImGUI() {
		{
			ImGui_ImplOpenGL3_NewFrame();
			ImGui_ImplGlfw_NewFrame();
			ImGui::NewFrame();
			{
				ImGui::PushStyleColor(ImGuiCol_TitleBgActive, ImVec4(0.0f, 0.7f, 0.2f, 1.0f));
				ImGui::PushStyleColor(ImGuiCol_TitleBg, ImVec4(0.0f, 0.3f, 0.1f, 1.0f));
				ImGui::SetNextWindowPos(ImVec2(20, 20), ImGuiCond_Once);
				ImGui::SetNextWindowSize(ImVec2(400, 300), ImGuiCond_Once);

				ImGui::Begin("TraceConfig", nullptr, ImGuiWindowFlags_MenuBar);

				ImGui::BeginChild(ImGui::GetID((void*)0), ImVec2(350, 200), ImGuiWindowFlags_NoTitleBar);
				ImGui::Text("Fps: %.2f", 1.0f / m_DelTime);
				{
					int samplePerLaunch = m_SamplePerLaunch;
					if (ImGui::SliderInt("samplePerLaunch", &samplePerLaunch, 1, 10)) {
						m_SamplePerLaunch = samplePerLaunch;
						m_UpdateParams = true;
					}
				}
				{
					int maxTraceDepth = m_MaxTraceDepth;
					if (ImGui::SliderInt("maxTraceDepth", &maxTraceDepth, 1, 10)) {
						m_MaxTraceDepth = maxTraceDepth;
						m_FlushFrame = true;
					}
				}
				{
					float emission[3] = { m_Light.emission.x, m_Light.emission.y, m_Light.emission.z };
					if (ImGui::SliderFloat3("light.Color", emission, 0.0f, 10.0f)) {
						m_Light.emission.x = emission[0];
						m_Light.emission.y = emission[1];
						m_Light.emission.z = emission[2];
						m_UpdateLight = true;
					}
				}
				if (ImGui::Button(prvPipelineName.c_str())) {
					std::swap(curPipelineName, prvPipelineName);
					m_UpdateCamera  = true;
				}
				ImGui::EndChild();

				ImGui::End();

				ImGui::PopStyleColor();
				ImGui::PopStyleColor();
			}

			// Rendering
			ImGui::Render();
			ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

		}
	}
	void OnGetInputs() {
		if (glfwGetKey(m_Window, GLFW_KEY_W) == GLFW_PRESS) {
			m_CameraController.ProcessKeyboard(rtlib::CameraMovement::eForward, m_DelTime);
			m_UpdateCamera = true;
		}
		if (glfwGetKey(m_Window, GLFW_KEY_S) == GLFW_PRESS) {
			m_CameraController.ProcessKeyboard(rtlib::CameraMovement::eBackward, m_DelTime);
			m_UpdateCamera = true;
		}
		if (glfwGetKey(m_Window, GLFW_KEY_A) == GLFW_PRESS) {
			m_CameraController.ProcessKeyboard(rtlib::CameraMovement::eLeft, m_DelTime);
			m_UpdateCamera = true;
		}
		if (glfwGetKey(m_Window, GLFW_KEY_D) == GLFW_PRESS) {
			m_CameraController.ProcessKeyboard(rtlib::CameraMovement::eRight, m_DelTime);
			m_UpdateCamera = true;
		}
		if (glfwGetKey(m_Window, GLFW_KEY_LEFT) == GLFW_PRESS) {
			m_CameraController.ProcessKeyboard(rtlib::CameraMovement::eLeft, m_DelTime);
			m_UpdateCamera = true;
		}
		if (glfwGetKey(m_Window, GLFW_KEY_RIGHT) == GLFW_PRESS) {
			m_CameraController.ProcessKeyboard(rtlib::CameraMovement::eRight, m_DelTime);
			m_UpdateCamera = true;
		}
		if (glfwGetKey(m_Window, GLFW_KEY_UP) == GLFW_PRESS) {
			m_CameraController.ProcessKeyboard(rtlib::CameraMovement::eUp, m_DelTime);
			m_UpdateCamera = true;
		}
		if (glfwGetKey(m_Window, GLFW_KEY_DOWN) == GLFW_PRESS) {
			m_CameraController.ProcessKeyboard(rtlib::CameraMovement::eDown, m_DelTime);
			m_UpdateCamera = true;
		}
		if (glfwGetMouseButton(m_Window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS) {
			m_CameraController.ProcessMouseMovement(-m_DelCursorPos.x, m_DelCursorPos.y);
			m_UpdateCamera = true;
		}
	}
private:
	static void frameBufferSizeCallback(GLFWwindow* window, int fbWidth, int fbHeight)
	{
		Test18Application* app = reinterpret_cast<Test18Application*>(glfwGetWindowUserPointer(window));
		if (app) {
			if (fbWidth != app->m_FbWidth || fbHeight != app->m_FbHeight)
			{
				app->m_FbWidth  = fbWidth;
				app->m_FbHeight = fbHeight;
				app->m_Resized  = true;
			}
		}
	}
	static void cursorPosCallback(GLFWwindow* window,  double xPos, double yPos)
	{
		Test18Application* app = reinterpret_cast<Test18Application*>(glfwGetWindowUserPointer(window));
		if (app) {
			app->m_DelCursorPos.x = xPos - app->m_CurCursorPos.x;
			app->m_DelCursorPos.y = yPos - app->m_CurCursorPos.y;;
			app->m_CurCursorPos.x = xPos;
			app->m_CurCursorPos.y = yPos;
		}
	}
private:
	GLFWwindow*                     m_Window             = nullptr;
	int			      	            m_FbWidth              = 0;
	int                             m_FbHeight             = 0;
	std::string                     m_Title              = {};
	bool                            m_Resized            = false;
	bool                            m_UpdateCamera       = false;
	bool                            m_UpdateLight        = false;
	float2                          m_DelCursorPos       = {};
	float2                          m_CurCursorPos       = {};
	float                           m_CurTime            = 0.0f;
	float                           m_DelTime            = 0.0f;
	std::string                     m_GlslVersion        = {};

	rtlib::GLTexture2D<uchar4>      m_GLTexture          = {};
	std::shared_ptr<rtlib::ext::RectRenderer>        
		                            m_RectRenderer       = {};
	rtlib::CameraController         m_CameraController   = {};
	test::PathTracer                m_Tracer             = {};
	test::MaterialSetPtr            m_MaterialSet        = nullptr;
	CUstream                        m_Stream             = nullptr;
	rtlib::CUDABuffer<uchar4>       m_FrameBuffer        = {};
	rtlib::GLInteropBuffer<uchar4>  m_FrameBufferGL      = {};
	rtlib::CUDABuffer<float3>       m_AccumBuffer        = {};
	rtlib::CUDABuffer<unsigned int> m_SeedBuffer         = {};
	uint32_t                        m_LightHgRecIndex    = 0;
	ParallelLight                   m_Light              = {};
	uint32_t                        m_MaxTraceDepth      = 4;
	uint32_t                        m_SamplePerLaunch    = 1;
	bool                            m_FlushFrame         = false;
	bool                            m_UpdateParams       = false;
	std::string                     curPipelineName      = "Trace";
	std::string                     prvPipelineName      = "Debug";
};
int main() {
	Test18Application app = {};
	app.InitGLFW(4, 4);
	app.InitWindow(1024, 1024, "title");
	app.InitGLAD();
	app.InitImGui();
	app.InitOptix();
	app.LoadScene();
	app.InitLight();
	app.InitCamera();
	app.InitFrameResources();
	app.InitTracePipeline();
	app.InitDebugPipeline();
	app.MainLoop();
	app.CleanUpImGui();
	app.CleanUpWindow();
	app.CleanUpGLFW();
}