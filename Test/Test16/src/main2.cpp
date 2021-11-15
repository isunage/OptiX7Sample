#include <Test16Config.h>
#include <RTLib/GL.h>
#include <RTLib/CUDA.h>
#include <RTLib/CUDA_GL.h>
#include <RTLib/ext/Camera.h>
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

class Application {
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
		m_Width = width;
		m_Height = height;
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
		m_CameraController = rtlib::ext::CameraController({ 0.0f,1.0f, 5.0f });
		m_CameraController.SetMouseSensitivity(0.125f);
		m_CameraController.SetMovementSpeed(50.0f);
	}
	void LoadObjModel(){
		auto objMeshGroup = std::make_shared<test::ObjMeshGroup>();
		if (!objMeshGroup->Load(TEST_TEST16_DATA_PATH"/Models/Sponza/sponza.obj", TEST_TEST16_DATA_PATH"/Models/Sponza/")) {
			throw std::runtime_error("Failed To Load Model!");
		}
		m_MaterialSet = objMeshGroup->GetMaterialSet();

		{
			for (auto& material : m_MaterialSet->materials) {
				auto diffTex = material.diffTex != "" ? material.diffTex : std::string(TEST_TEST16_DATA_PATH"/Textures/white.png");
				auto specTex = material.specTex != "" ? material.specTex : std::string(TEST_TEST16_DATA_PATH"/Textures/white.png");
				auto emitTex = material.emitTex != "" ? material.emitTex : std::string(TEST_TEST16_DATA_PATH"/Textures/white.png");
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

		    //IAS1: FirstIAS
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
		m_Tracer.SetIASHandle("FirstIAS", firstIASHandle);
	}
	void InitFrameResources(){
		RTLIB_CUDA_CHECK(cudaStreamCreate(&m_Stream));
	 	m_FrameBuffer   = rtlib::CUDABuffer<uchar4>(std::vector<uchar4>(m_Width * m_Height));
		auto frameBufferGL = rtlib::GLInteropBuffer<uchar4>(m_Width * m_Height, GL_PIXEL_UNPACK_BUFFER, GL_DYNAMIC_DRAW, m_Stream);

		m_FrameBufferGL = rtlib::GLInteropBuffer<uchar4>(m_Width * m_Height, GL_PIXEL_UNPACK_BUFFER, GL_DYNAMIC_DRAW, m_Stream);

    	m_AccumBuffer   = rtlib::CUDABuffer<float3>(std::vector<float3>(m_Width * m_Height));
   		m_SeedBuffer    = rtlib::CUDABuffer<unsigned int>();
		{
			std::vector<unsigned int> seeds(m_Width * m_Height);
			std::random_device rd;
			std::mt19937 mt(rd());
			std::generate(seeds.begin(), seeds.end(), mt);
			m_SeedBuffer.allocate(seeds.size());
			m_SeedBuffer.upload(seeds);
		}
		m_GLTexture = rtlib::GLTexture2D<uchar4>();
		{
			m_GLTexture.allocate({ (size_t)m_Width, (size_t)m_Height });
			m_GLTexture.setParameteri(GL_TEXTURE_MAG_FILTER, GL_LINEAR, false);
			m_GLTexture.setParameteri(GL_TEXTURE_MIN_FILTER, GL_LINEAR, false);
			m_GLTexture.setParameteri(GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE, false);
			m_GLTexture.setParameteri(GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE, false);
		}
	}
	void InitRayTracePipeline()
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
            tracePipeline->pipeline = m_Tracer.GetOPXContext()->createPipeline(pipelineCompileOptions);
        }
        {
            tracePipeline->width  = m_Width;
            tracePipeline->height = m_Height;
            tracePipeline->depth  = 2;
        }
        //module: Load
        {
            auto ptxSource = std::string();
            {
                auto ptxFile = std::ifstream(TEST_TEST16_CUDA_PATH"/RayTrace.ptx", std::ios::binary);
                ptxSource = std::string((std::istreambuf_iterator<char>(ptxFile)), (std::istreambuf_iterator<char>()));
            }
            OptixModuleCompileOptions moduleCompileOptions = {};
            moduleCompileOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
            moduleCompileOptions.numBoundValues = 0;
#ifndef NDEBUG
            moduleCompileOptions.debugLevel = static_cast<OptixCompileDebugLevel>(rtlib::OPXCompileDebugLevel::None);
            moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
#else
            moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
            moduleCompileOptions.debugLevel = static_cast<OptixCompileDebugLevel>(rtlib::OPXCompileDebugLevel::Minimal);
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
            pipelineLinkOptions.debugLevel = static_cast<OptixCompileDebugLevel>(rtlib::OPXCompileDebugLevel::Minimal);
#else
            pipelineLinkOptions.debugLevel = static_cast<OptixCompileDebugLevel>(rtlib::OPXCompileDebugLevel::Minimal);
#endif
            tracePipeline->pipeline.link(pipelineLinkOptions);
        }
        //SBTRecord
        {
            tracePipeline->raygenBuffer.cpuHandle.resize(1);
            auto camera = m_CameraController.GetCamera(30.0f, 1.0f);
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
            tracePipeline->hitGBuffer.cpuHandle.resize(RAY_TYPE_COUNT * m_Tracer.m_IASHandles["FirstIAS"]->sbtCount);
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
        m_Tracer.SetPipeline("Trace", tracePipeline);
    }
	void InitRayTraceParams(){
		auto& lightGASHandle = m_Tracer.m_GASHandles["Light"];
		{
			auto light = ParallelLight();
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
				auto lightMaterial = m_MaterialSet->materials[lightMesh->GetUniqueResource()->materials[0]];
				light.emission = lightMaterial.emitCol;
			}
			{
				m_Params.frameBuffer     = m_FrameBuffer.getDevicePtr();
				m_Params.accumBuffer     = m_AccumBuffer.getDevicePtr();
				m_Params.seed            = m_SeedBuffer.getDevicePtr();
				m_Params.width           = m_Width;
				m_Params.height          = m_Height;
				m_Params.gasHandle       = m_Tracer.m_IASHandles["FirstIAS"]->handle;
				m_Params.light           = light;
				m_Params.samplePerALL    = 0;
				m_Params.samplePerLaunch = 1;
			}
		}
	}
	void MainLoop() {
		float x = 0.0f;
		float y = 0.0f;
		bool   isResized = false;
		bool   isUpdated = false;
		bool   isFixedLight = false;
		bool   isMovedCamera = false;
		PrepareMainLoop();
		auto& curPipeline = m_Tracer.m_Pipelines["Trace"];
		while (!glfwWindowShouldClose(m_Window)) {
			{
				if (isResized) {
					{
						std::random_device rd;
						std::mt19937 mt(rd());
						std::vector<unsigned int> seeds(m_Width * m_Height);
						std::generate(seeds.begin(), seeds.end(), mt);
						m_SeedBuffer.resize(m_Width * m_Height);
						m_SeedBuffer.upload(seeds);
					}
					m_FrameBufferGL.resize(m_Width * m_Height);
					m_AccumBuffer.resize(m_Width * m_Height);
					curPipeline->paramsBuffer.cpuHandle[0].accumBuffer = m_AccumBuffer.getDevicePtr();
					curPipeline->paramsBuffer.cpuHandle[0].seed   = m_SeedBuffer.getDevicePtr();
					curPipeline->paramsBuffer.cpuHandle[0].width  = m_Width;
					curPipeline->paramsBuffer.cpuHandle[0].height = m_Height;
				}
				if (isMovedCamera) {
					auto camera = m_CameraController.GetCamera(30.0f, 1.0f);
					auto [u, v, w] = camera.getUVW();
					curPipeline->raygenBuffer.cpuHandle[0].data.eye = camera.getEye();
					curPipeline->raygenBuffer.cpuHandle[0].data.u = u;
					curPipeline->raygenBuffer.cpuHandle[0].data.v = v;
					curPipeline->raygenBuffer.cpuHandle[0].data.w = w;
					curPipeline->raygenBuffer.Upload();
					isUpdated = true;
				}
				if (isUpdated) {
					m_FrameBufferGL.upload(std::vector<uchar4>(m_Width * m_Height));
					m_AccumBuffer.upload(std::vector<float3>(m_Width * m_Height));
					curPipeline->paramsBuffer.cpuHandle[0].samplePerALL = 0;
				}
				{
					curPipeline->width  = m_Width;
					curPipeline->height = m_Height;
					curPipeline->paramsBuffer.cpuHandle[0].frameBuffer = m_FrameBufferGL.map();
					curPipeline->paramsBuffer.Upload();
					curPipeline->Launch(m_Stream);
					cuStreamSynchronize(m_Stream);
					m_FrameBufferGL.unmap();
					curPipeline->paramsBuffer.cpuHandle[0].samplePerALL += curPipeline->paramsBuffer.cpuHandle[0].samplePerLaunch;
				}

				{
					glfwPollEvents();
					if (isResized) {
						m_GLTexture.reset();
						m_GLTexture.allocate({ (size_t)m_Width,(size_t)m_Height }, GL_TEXTURE_2D);
						m_GLTexture.setParameteri(GL_TEXTURE_MAG_FILTER, GL_LINEAR, false);
						m_GLTexture.setParameteri(GL_TEXTURE_MIN_FILTER, GL_LINEAR, false);
						m_GLTexture.setParameteri(GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE, false);
						m_GLTexture.setParameteri(GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE, false);
					}
					m_GLTexture.upload(0, m_FrameBufferGL.getHandle(), 0, 0, m_Width, m_Height);
					{
						glViewport(0, 0, m_Width, m_Height);
					}
					glClear(GL_COLOR_BUFFER_BIT);
					m_RectRenderer->draw(m_GLTexture.getID());

					glfwSwapBuffers(m_Window);
					isUpdated = false;
					isResized = false;
					isMovedCamera = false;
					{
						int tWidth, tHeight;
						glfwGetWindowSize(m_Window, &tWidth, &tHeight);
						if (m_Width != tWidth || m_Height != tHeight) {
							std::cout << m_Width  << "->" << tWidth << "\n";
							std::cout << m_Height << "->" << tHeight << "\n";
							m_Width = tWidth;
							m_Height = tHeight;
							isResized = true;
							isUpdated = true;
						}
						else {
							isResized = false;
						}
						float prevTime = glfwGetTime();
						m_DelTime = m_CurTime - prevTime;
						m_CurTime = prevTime;
						if (glfwGetKey(m_Window, GLFW_KEY_W) == GLFW_PRESS) {
							m_CameraController.ProcessKeyboard(rtlib::ext::CameraMovement::eForward, m_DelTime);
							isMovedCamera = true;
						}
						if (glfwGetKey(m_Window, GLFW_KEY_S) == GLFW_PRESS) {
							m_CameraController.ProcessKeyboard(rtlib::ext::CameraMovement::eBackward, m_DelTime);
							isMovedCamera = true;
						}
						if (glfwGetKey(m_Window, GLFW_KEY_A) == GLFW_PRESS) {
							m_CameraController.ProcessKeyboard(rtlib::ext::CameraMovement::eLeft, m_DelTime);
							isMovedCamera = true;
						}
						if (glfwGetKey(m_Window, GLFW_KEY_D) == GLFW_PRESS) {
							m_CameraController.ProcessKeyboard(rtlib::ext::CameraMovement::eRight, m_DelTime);
							isMovedCamera = true;
						}
						if (glfwGetKey(m_Window, GLFW_KEY_LEFT) == GLFW_PRESS) {
							m_CameraController.ProcessKeyboard(rtlib::ext::CameraMovement::eLeft, m_DelTime);
							isMovedCamera = true;
						}
						if (glfwGetKey(m_Window, GLFW_KEY_RIGHT) == GLFW_PRESS) {
							m_CameraController.ProcessKeyboard(rtlib::ext::CameraMovement::eRight, m_DelTime);
							isMovedCamera = true;
						}
						if (glfwGetKey(m_Window, GLFW_KEY_UP) == GLFW_PRESS) {
							m_CameraController.ProcessKeyboard(rtlib::ext::CameraMovement::eUp,   m_DelTime);
							isMovedCamera = true;
						}
						if (glfwGetKey(m_Window, GLFW_KEY_DOWN) == GLFW_PRESS) {
							m_CameraController.ProcessKeyboard(rtlib::ext::CameraMovement::eDown, m_DelTime);
							isMovedCamera = true;
						}
						if (glfwGetMouseButton(m_Window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
							m_CameraController.ProcessMouseMovement(-m_DelCursorPos.x, m_DelCursorPos.y);
							isMovedCamera = true;
						}
					}

				}

			}
			glfwPollEvents();
			
		}
	}
    void MainLoop2() {
		float x = 0.0f;
		float y = 0.0f;
		while (!glfwWindowShouldClose(m_Window)) {
			glfwPollEvents();
			ImGui_ImplOpenGL3_NewFrame();
			ImGui_ImplGlfw_NewFrame();
			ImGui::NewFrame();
			ImGui::PushStyleColor(ImGuiCol_TitleBgActive, ImVec4(0.0f, 0.7f, 0.2f, 1.0f));
			ImGui::PushStyleColor(ImGuiCol_TitleBg, ImVec4(0.0f, 0.3f, 0.1f, 1.0f));
			ImGui::SetNextWindowPos(ImVec2(20, 20));
			ImGui::SetNextWindowSize(ImVec2(280, 300));

			ImGui::Begin("config 1", nullptr, ImGuiWindowFlags_MenuBar);

			if (ImGui::BeginMenuBar()) {
				if (ImGui::BeginMenu("File"))
				{
					if (ImGui::MenuItem("Save")) {

					}
					if (ImGui::MenuItem("Load")) {

					}

					ImGui::EndMenu();
				}
				ImGui::EndMenuBar();
			}


			static std::vector<float> items(10);

			if (ImGui::Button("add")) {
				items.push_back(0.0f);
			}
			if (ImGui::Button("remove")) {
				if (items.empty() == false) {
					items.pop_back();
				}
			}

			ImGui::BeginChild(ImGui::GetID((void*)0), ImVec2(250, 100), ImGuiWindowFlags_NoTitleBar);
			for (int i = 0; i < items.size() ; ++i) {
				char name[16];
				sprintf(name, "item %d", i);
				ImGui::SliderFloat(name, &items[i], 0.0f, 10.0f);
			}
			ImGui::EndChild();

			ImGui::End();

			ImGui::PopStyleColor();
			ImGui::PopStyleColor();
			// Rendering
			ImGui::Render();
			int display_w, display_h;
			glfwGetFramebufferSize(m_Window, &display_w, &display_h);
			glClearColor(0.8f, 0.8f, 0.8f, 0.0f);
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
			glViewport(0, 0, display_w, display_h);
			ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
			glfwSwapBuffers(m_Window);
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
		auto& curPipeline = m_Tracer.m_Pipelines["Trace"];
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
		curPipeline->paramsBuffer.cpuHandle.push_back(m_Params);
		curPipeline->paramsBuffer.Upload();
	}
private:
	static void cursorPosCallback(GLFWwindow* window,  double xPos, double yPos)
	{
		Application* app = reinterpret_cast<Application*>(glfwGetWindowUserPointer(window));
		if (app) {
			app->m_DelCursorPos.x = xPos - app->m_CurCursorPos.x;
			app->m_DelCursorPos.y = yPos - app->m_CurCursorPos.y;;
			app->m_CurCursorPos.x = xPos;
			app->m_CurCursorPos.y = yPos;
		}
	}
private:
	GLFWwindow*                     m_Window           = nullptr;
	int			      	            m_Width            = 0;
	int                             m_Height           = 0;
	float2                          m_DelCursorPos     = {};
	float2                          m_CurCursorPos     = {};
	float                           m_CurTime          = 0.0f;
	float                           m_DelTime          = 0.0f;
	std::string                     m_Title            = {};
	std::string                     m_GlslVersion      = {};
	rtlib::GLTexture2D<uchar4>      m_GLTexture        = {};
	std::shared_ptr<rtlib::ext::RectRenderer>        
		                            m_RectRenderer     = {};

	rtlib::ext::CameraController         m_CameraController = {};
	test::PathTracer                m_Tracer           = {};
	test::MaterialSetPtr            m_MaterialSet      = nullptr;
	CUstream                        m_Stream           = nullptr;
	rtlib::CUDABuffer<uchar4>       m_FrameBuffer      = {};
	rtlib::GLInteropBuffer<uchar4>  m_FrameBufferGL    = {};
	rtlib::CUDABuffer<float3>       m_AccumBuffer      = {};
	rtlib::CUDABuffer<unsigned int> m_SeedBuffer       = {};
	Params                          m_Params           = {};    
	
};
int main() {
	Application app = {};
	app.InitGLFW(4, 4);
	app.InitWindow(1024, 1024, "title");
	app.InitGLAD();
	app.InitImGui();
	app.InitOptix();
	app.LoadObjModel();
	app.InitRayTracePipeline();
	app.InitFrameResources();
	app.InitRayTraceParams();
	app.InitCamera();
	app.MainLoop();
	app.CleanUpImGui();
	app.CleanUpWindow();
	app.CleanUpGLFW();
}