#include <TestPGConfig.h>
#include <RTLib/GL.h>
#include <RTLib/CUDA.h>
#include <RTLib/CUDA_GL.h>
#include <RTLib/ext/Camera.h>
#include <RTLib/Utils.h>
#include <RTLib/VectorFunction.h>
#include <RTLib/ext/RectRenderer.h>
#include <RTLib/ext/Resources/CUDA.h>
#include <cuda/RayTrace.h>
#include <GLFW/glfw3.h>
#include <stb_image_write.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include "../include/RTPathGuidingUtils.h"
#include "../include/RTTracer.h"
#include "../include/SceneBuilder.h"
#include <filesystem>
#include <fstream>
#include <unordered_map>
#include <random>
#include <sstream>
#include <chrono>
#include <string>
namespace test {
	std::string SpecifyMaterialType(const rtlib::ext::VariableMap& material) {
		auto emitCol  = material.GetFloat3As<float3>("emitCol");
		auto tranCol  = material.GetFloat3As<float3>("tranCol");
		auto refrIndx = material.GetFloat1("refrIndx");
		auto shinness = material.GetFloat1("shinness");
		auto illum    = material.GetUInt32("illum");
		if (illum == 7) {
			return "Refraction";
		}else {
			return "Phong";
		}
	}
}
class Test20Application {
private:
	static inline constexpr std::array<float,3> kDefaultLightColor          = {10.0f,10.0f,10.0f};
	static inline constexpr uint32_t            kDefaultSamplePerLaunch     = 1;
	static inline constexpr uint32_t            kDefaultMaxTraceDepth       = 4;
	static inline constexpr std::string_view    tracePipelineSubPassNames[] = { "Def","Pg"};
	static inline constexpr std::string_view    debugPipelineFrameNames[]   = { "Diffuse","Specular","Emission","Transmit","TexCoord","Normal","Depth","STree"};
	enum EventFlags {
		eNone             = 0,
		eOnFlushFrame     = (1 << 0),
		eOnResize         = (1 << 1),
		eOnUpdateParams   = (1 << 2),
		eOnUpdateCamera   = (1 << 3),
		eOnUpdateLight    = (1 << 4),
		eFlushFrame       = eOnFlushFrame   | eOnUpdateParams,
		eUpdateCamera     = eOnUpdateCamera | eFlushFrame,
		eUpdateLight      = eOnUpdateLight  | eFlushFrame,
		eResize           = eOnResize       | eUpdateCamera,// Flush Frame + UpdateCamera
	};
public:
	void Initialize() {
		this->InitGLFW(4, 4);
		this->InitWindow(1024, 1024, "title");
		this->InitGLAD();
		this->InitImGui();
		this->InitOptix();
		this->LoadScene();
		this->InitLight();
		this->InitCamera();
		this->InitFrameResources();
		this->InitSDTree();
		this->InitTracePipeline();
		this->InitDebugPipeline();
	}
	void MainLoop() {
		PrepareMainLoop();
		while (!glfwWindowShouldClose(m_Window)) {
			{ 
				ProcessFrame();
				glfwSwapBuffers(m_Window);
				glfwPollEvents();
			}
		}
	}
	void ProcessFrame() {
		//PG
		if (m_CurSubPassName == "Pg") {
			this->OnBeginPG();
		}
		//Update
		this->OnUpdate();
		//Launch
		this->OnLaunch();
		//Render: Frame
		this->OnRenderFrame();
		//PG
		if (m_CurSubPassName == "Pg") {
			this->OnEndPG();
		}
		//Render: ImGui
		this->OnRenderImGui();
		//Update: Time
		this->OnUpdateTime();
		//GetInputs
		if (m_CurSubPassName == "Def") {
			this->OnReactInputs();
		}
	}
	void CleanUp() {
		this->CleanUpImGui();
		this->CleanUpWindow();
		this->CleanUpGLFW();
	}
private:
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
		glfwSetMouseButtonCallback(m_Window, ImGui_ImplGlfw_MouseButtonCallback);
		glfwSetKeyCallback(m_Window, ImGui_ImplGlfw_KeyCallback);
		glfwSetCharCallback(m_Window, ImGui_ImplGlfw_CharCallback);
		glfwSetScrollCallback(m_Window, ImGui_ImplGlfw_ScrollCallback);
		glfwSetCursorPosCallback(m_Window, cursorPosCallback);
		glfwSetFramebufferSizeCallback(m_Window, frameBufferSizeCallback);
		glfwGetFramebufferSize(m_Window, &m_FbWidth, &m_FbHeight);
		m_FbAspect = static_cast<float>(m_FbWidth) / static_cast<float>(m_FbHeight);
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
	void InitOptix() {
		RTLIB_CUDA_CHECK(cudaFree(0));
		RTLIB_OPTIX_CHECK(optixInit());
		m_Tracer.SetContext(std::make_shared<rtlib::OPXContext>(rtlib::OPXContext::Desc{ 0, 0, OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL, 4 }));
	}
	void InitCamera() {
		m_CameraController = rtlib::ext::CameraController({ 0.0f,1.0f, 5.0f });
		m_CameraController.SetMouseSensitivity(0.001f);
		m_CameraController.SetMovementSpeed(10.0f);
	}
	void LoadScene(){
		std::vector<std::pair<std::string, std::string>> objInfos = {
			//{TEST_TEST_PG_DATA_PATH"/Models/Lumberyard/Exterior/exterior.obj"  , TEST_TEST_PG_DATA_PATH"/Models/Lumberyard/Exterior/"},
			//{TEST_TEST_PG_DATA_PATH"/Models/Lumberyard/Interior/interior.obj"  , TEST_TEST_PG_DATA_PATH"/Models/Lumberyard/Interior/"},
			//{TEST_TEST_PG_DATA_PATH"/Models/Sponza/Sponza.obj"                 , TEST_TEST_PG_DATA_PATH"/Models/Sponza/"    },
			//{TEST_TEST_PG_DATA_PATH"/Models/CornellBox/CornellBox-Original.obj", TEST_TEST_PG_DATA_PATH"/Models/CornellBox/"},
			{TEST_TEST_PG_DATA_PATH"/Models/CornellBox/CornellBox-Water.obj"     , TEST_TEST_PG_DATA_PATH"/Models/CornellBox/"},
		};
		m_MaterialSet = rtlib::ext::VariableMapListPtr(new rtlib::ext::VariableMapList());
		{
			size_t materialOffset = 0;
			for (auto& objInfo : objInfos) {
				auto objMeshGroup = std::make_shared<test::ObjMeshGroup>();
				if (!objMeshGroup->Load2(objInfo.first, objInfo.second)) {
					throw std::runtime_error("Failed To Load Model!");
				}
				auto  meshGroup   = objMeshGroup->GetMeshGroup();
				auto  materialSet = objMeshGroup->GetMaterialList();
				for (auto& [name, uniqueResource] : meshGroup->GetUniqueResources()) {
					for (auto& material : uniqueResource->materials) {
						material += materialOffset;
					}
				}
				m_Tracer.AddMeshGroup(std::filesystem::path(objInfo.first).filename().string(), meshGroup);
				m_MaterialSet->resize(materialOffset + materialSet->size());
				for (size_t i = 0; i < materialSet->size(); ++i) {
					(*m_MaterialSet)[materialOffset + i] = (*materialSet)[i];
				}
				materialOffset+= materialSet->size();
			}
		}
		{
			for (auto& material : *m_MaterialSet) {
				auto diffTex = material.GetString("diffTex") != "" ? material.GetString("diffTex") : std::string(TEST_TEST_PG_DATA_PATH"/Textures/white.png");
				auto specTex = material.GetString("specTex") != "" ? material.GetString("specTex") : std::string(TEST_TEST_PG_DATA_PATH"/Textures/white.png");
				auto emitTex = material.GetString("emitTex") != "" ? material.GetString("emitTex") : std::string(TEST_TEST_PG_DATA_PATH"/Textures/white.png");

				if (!m_Tracer.LoadTexture(material.GetString("diffTex"), diffTex)) {
					material.SetString("diffTex", "");
				}
				if (material.GetString("diffTex") != "") {
					auto diffCol    = material.GetFloat3As<float3>("diffCol");
					auto avgDiffCol = rtlib::dot(diffCol, make_float3(1.0f)) / 3.0f;
					if (avgDiffCol == 0.0f)
					{
						material.SetFloat3("diffCol", { 1.0f,1.0f,1.0f });
					}
				}
				if (!m_Tracer.LoadTexture(material.GetString("specTex"), specTex)) {
					material.SetString("specTex", "");
				}
				if (material.GetString("specTex") != "") {
					auto specCol    = material.GetFloat3As<float3>("specCol");
					auto avgSpecCol = rtlib::dot(specCol, make_float3(1.0f)) / 3.0f;
					if (avgSpecCol == 0.0f)
					{
						material.SetFloat3("specCol", { 1.0f,1.0f,1.0f });
					}
				}
				if (!m_Tracer.LoadTexture(material.GetString("emitTex"), emitTex)) {
					material.SetString("emitTex", "");
				}
				if (material.GetString("emitTex") != "") {
					auto emitCol    = material.GetFloat3As<float3>("emitCol");
					auto avgEmitCol = rtlib::dot(emitCol, make_float3(1.0f)) / 3.0f;
					if (avgEmitCol == 0.0f)
					{
						material.SetFloat3("emitCol", { 1.0f,1.0f,1.0f });
					}
				}

			}
		}
		bool isLightFound = false;
		//GAS1: World
		m_Tracer.NewGASHandle("world");
		auto worldGASHandle = m_Tracer.GetGASHandle("world");
		{
			OptixAccelBuildOptions accelOptions = {};
			accelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
			accelOptions.operation  = OPTIX_BUILD_OPERATION_BUILD;
			for (auto& [name, meshGroup]  : m_Tracer.GetMeshGroups()) {

				if (!meshGroup->GetSharedResource()->vertexBuffer.HasGpuComponent("CUDA"))
				{
					meshGroup->GetSharedResource()->vertexBuffer.AddGpuComponent<rtlib::ext::resources::CUDABufferComponent<float3>>("CUDA");
				}
				if (!meshGroup->GetSharedResource()->normalBuffer.HasGpuComponent("CUDA"))
				{
					meshGroup->GetSharedResource()->normalBuffer.AddGpuComponent<rtlib::ext::resources::CUDABufferComponent<float3>>("CUDA");
				}
				if (!meshGroup->GetSharedResource()->texCrdBuffer.HasGpuComponent("CUDA"))
				{
					meshGroup->GetSharedResource()->texCrdBuffer.AddGpuComponent<rtlib::ext::resources::CUDABufferComponent<float2>>("CUDA");
				}
				for (auto& [meshUniqueName, meshUniqueResource]: meshGroup->GetUniqueResources()) {
					if (!meshUniqueResource->matIndBuffer.HasGpuComponent("CUDA"))
					{
						meshUniqueResource->matIndBuffer.AddGpuComponent<rtlib::ext::resources::CUDABufferComponent<uint32_t>>("CUDA");
					}
					if (!meshUniqueResource->triIndBuffer.HasGpuComponent("CUDA"))
					{
						meshUniqueResource->triIndBuffer.AddGpuComponent<rtlib::ext::resources::CUDABufferComponent<uint3>>("CUDA");
					}
					if (meshUniqueName != "light") {
						worldGASHandle->AddMesh(meshGroup->LoadMesh(meshUniqueName));
					}
					else {
						isLightFound = true;
					}
				}
			}
			worldGASHandle->Build(m_Tracer.GetContext().get(), accelOptions);
		}
		//GAS2: Light
		m_Tracer.NewGASHandle("light");
		auto lightGASHandle = m_Tracer.GetGASHandle("light");
		if (isLightFound) {
			OptixAccelBuildOptions accelOptions = {};
			accelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
			accelOptions.operation  = OPTIX_BUILD_OPERATION_BUILD;
			for (auto& [name, meshGroup] : m_Tracer.GetMeshGroups()) {
				lightGASHandle->AddMesh(meshGroup->LoadMesh("light"));
			}
			lightGASHandle->Build(m_Tracer.GetContext().get(), accelOptions);
		}
		else {
			rtlib::utils::AABB aabb = {};
			for (auto& [name, meshGroup] : m_Tracer.GetMeshGroups()) {
				for (auto& vertex : meshGroup->GetSharedResource()->vertexBuffer) {
					aabb.Update(vertex);
				}
			}
			OptixAccelBuildOptions accelOptions = {};
			accelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
			accelOptions.operation  = OPTIX_BUILD_OPERATION_BUILD;
			auto lightMesh          = rtlib::ext::Mesh::New();
			lightMesh->SetSharedResource(rtlib::ext::MeshSharedResource::New());
			lightMesh->GetSharedResource()->name = "light";
			lightMesh->GetSharedResource()->vertexBuffer = {
				{aabb.min.x,aabb.max.y - 1e-3f,aabb.min.z},
				{aabb.max.x,aabb.max.y - 1e-3f,aabb.min.z},
				{aabb.max.x,aabb.max.y - 1e-3f,aabb.max.z},
				{aabb.min.x,aabb.max.y - 1e-3f,aabb.max.z}
			};
			lightMesh->GetSharedResource()->texCrdBuffer = { {0.0f,0.0f}      , {1.0f,0.0f}      , {1.0f,1.0f}      , {0.0f,1.0f},     };
			lightMesh->GetSharedResource()->normalBuffer = { {0.0f,-1.0f,0.0f}, {0.0f,-1.0f,0.0f}, {0.0f,-1.0f,0.0f}, {0.0f,-1.0f,0.0f}};
			unsigned int curMaterialSetCount = m_MaterialSet->size();
			auto lightMaterial = rtlib::ext::VariableMap{};
			{
				lightMaterial.SetString("name"    , "light");
				lightMaterial.SetFloat3("diffCol" , kDefaultLightColor);
				lightMaterial.SetString("diffTex" , "");
				lightMaterial.SetFloat3("emitCol" , kDefaultLightColor);
				lightMaterial.SetString("emitTex" , "");
				lightMaterial.SetFloat3("specCol" , kDefaultLightColor);
				lightMaterial.SetString("specTex" , "");
				lightMaterial.SetFloat1("shinness", 0.0f);
				lightMaterial.SetString("shinTex" , "");
				lightMaterial.SetFloat3("tranCol" , kDefaultLightColor);
				lightMaterial.SetFloat1("refrIndx", 0.0f);
				lightMaterial.SetUInt32("illum"   , 2);
			}
			m_MaterialSet->push_back( lightMaterial );
			lightMesh->GetSharedResource()->vertexBuffer.AddGpuComponent<rtlib::ext::resources::CUDABufferComponent<float3>>("CUDA");
			lightMesh->GetSharedResource()->normalBuffer.AddGpuComponent<rtlib::ext::resources::CUDABufferComponent<float3>>("CUDA");
			lightMesh->GetSharedResource()->texCrdBuffer.AddGpuComponent<rtlib::ext::resources::CUDABufferComponent<float2>>("CUDA");
			lightMesh->SetUniqueResource(rtlib::ext::MeshUniqueResource::New());
			lightMesh->GetUniqueResource()->name                   = "light";
			lightMesh->GetUniqueResource()->materials              = { curMaterialSetCount };
			lightMesh->GetUniqueResource()->matIndBuffer = { 0,0 };
			lightMesh->GetUniqueResource()->triIndBuffer = { {0,1,2}, {2,3,0} };
			lightMesh->GetUniqueResource()->matIndBuffer.AddGpuComponent<rtlib::ext::resources::CUDABufferComponent<uint32_t>>("CUDA");
			lightMesh->GetUniqueResource()->triIndBuffer.AddGpuComponent<rtlib::ext::resources::CUDABufferComponent<uint3>>("CUDA");
			//AddMesh
			lightGASHandle->AddMesh(lightMesh);
			//Build
			lightGASHandle->Build(m_Tracer.GetContext().get(), accelOptions);
		}
		//IAS1: First
		m_Tracer.NewIASHandle("TLAS");
		m_Tracer.SetTLASName( "TLAS");
		auto tlasHandle = m_Tracer.GetIASHandle("TLAS");
		{
			OptixAccelBuildOptions accelOptions = {};
			accelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
			accelOptions.operation  = OPTIX_BUILD_OPERATION_BUILD;
			//World
			auto worldInstance = rtlib::ext::Instance();
			worldInstance.Init(m_Tracer.GetGASHandle("world"));
			worldInstance.SetSbtOffset(0);
			//Light
			auto lightInstance = rtlib::ext::Instance();
			lightInstance.Init(m_Tracer.GetGASHandle("light"));
			lightInstance.SetSbtOffset(worldInstance.GetSbtCount() * RAY_TYPE_COUNT);
			//InstanceSet
			auto instanceSet   = std::make_shared<rtlib::ext::InstanceSet>();
			instanceSet->SetInstance(worldInstance);
			instanceSet->SetInstance(lightInstance);
			instanceSet->Upload();
			//AddInstanceSet
			tlasHandle->AddInstanceSet(instanceSet);
			//Build
			tlasHandle->Build(m_Tracer.GetContext().get(), accelOptions);
		}
	}
	void InitLight() {
		m_Light = ParallelLight();
		{
			auto  lightGASHandle = m_Tracer.GetGASHandle("light");
			auto  lightMesh      = lightGASHandle->GetMesh(0);
			auto  lightVertices  = std::vector<float3>();
			for (auto& index : lightMesh->GetUniqueResource()->triIndBuffer) {
				lightVertices.push_back(lightMesh->GetSharedResource()->vertexBuffer[index.x]);
				lightVertices.push_back(lightMesh->GetSharedResource()->vertexBuffer[index.y]);
				lightVertices.push_back(lightMesh->GetSharedResource()->vertexBuffer[index.z]);
			}
			auto lightAABB     = rtlib::utils::AABB(lightVertices);
			auto lightV3       = lightAABB.max - lightAABB.min;
			m_Light.corner     = lightAABB.min;
			m_Light.v1         = make_float3(0.0f, 0.0f, lightV3.z);
			m_Light.v2         = make_float3(lightV3.x, 0.0f, 0.0f);
			m_Light.normal     = make_float3(0.0f, -1.0f, 0.0f);
			auto lightMaterial = (*m_MaterialSet)[lightMesh->GetUniqueResource()->materials[0]];
			m_Light.emission   = lightMaterial.GetFloat3As<float3>("emitCol");
		}
	}
	void InitFrameResources() {
		RTLIB_CUDA_CHECK(cudaStreamCreate(&m_Stream));
		m_FrameBuffer   = rtlib::CUDABuffer<uchar4>(std::vector<uchar4>(m_FbWidth * m_FbHeight));
		m_FrameBufferGL = rtlib::GLInteropBuffer<uchar4>(m_FbWidth * m_FbHeight, GL_PIXEL_UNPACK_BUFFER, GL_DYNAMIC_DRAW, m_Stream);
		m_RefImage	    = std::vector<float3>(m_FbWidth * m_FbHeight);
		for(auto frameName: debugPipelineFrameNames){
			m_DebugBuffers[  frameName.data()] = rtlib::CUDABuffer<uchar4>(std::vector<uchar4>(m_FbWidth * m_FbHeight));
			m_DebugBufferGLs[frameName.data()] = rtlib::GLInteropBuffer<uchar4>(m_FbWidth * m_FbHeight, GL_PIXEL_UNPACK_BUFFER, GL_DYNAMIC_DRAW, m_Stream);
		}
		m_AccumBuffer    = rtlib::CUDABuffer<float3>(std::vector<float3>(m_FbWidth * m_FbHeight));
		m_AccumBufferPG  = rtlib::CUDABuffer<float3>(std::vector<float3>(m_FbWidth * m_FbHeight));
		m_SeedBuffer     = rtlib::CUDABuffer<unsigned int>();
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
			m_GLTexture.setParameteri(GL_TEXTURE_MAG_FILTER, GL_LINEAR       , false);
			m_GLTexture.setParameteri(GL_TEXTURE_MIN_FILTER, GL_LINEAR       , false);
			m_GLTexture.setParameteri(GL_TEXTURE_WRAP_S    , GL_CLAMP_TO_EDGE, false);
			m_GLTexture.setParameteri(GL_TEXTURE_WRAP_T    , GL_CLAMP_TO_EDGE, false);
		}
	}
	void InitSDTree() {
		m_WorldAABB = rtlib::utils::AABB();

		for (auto& mesh : m_Tracer.GetGASHandle("world")->GetMeshes()) {
			for (auto& index : mesh->GetUniqueResource()->triIndBuffer)
			{
				m_WorldAABB.Update(mesh->GetSharedResource()->vertexBuffer[index.x]);
				m_WorldAABB.Update(mesh->GetSharedResource()->vertexBuffer[index.y]);
				m_WorldAABB.Update(mesh->GetSharedResource()->vertexBuffer[index.z]);
			}
		}

		m_SdTree = std::make_unique<test::RTSTreeWrapper>(m_WorldAABB.min, m_WorldAABB.max);
		m_SdTree->Upload();

	}
	void InitTracePipeline()
	{
		auto tracePipeline2 = std::make_shared<test::RTPipeline<RayGenData, MissData, HitgroupData, RayTraceParams>>();
		{
			OptixPipelineCompileOptions compileOptions = {};

			compileOptions.pipelineLaunchParamsVariableName = "params";
			compileOptions.numAttributeValues               = 3;
			compileOptions.numPayloadValues                 = 8;
			compileOptions.usesPrimitiveTypeFlags           = 0;
			compileOptions.usesMotionBlur                   = false;
			compileOptions.traversableGraphFlags            = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;

			OptixPipelineLinkOptions linkOptions = {};

			linkOptions.maxTraceDepth = 1;
#ifndef NDEBUG
			linkOptions.debugLevel = static_cast<OptixCompileDebugLevel>(rtlib::OPXCompileDebugLevel::Minimal);
#else
			linkOptions.debugLevel = static_cast<OptixCompileDebugLevel>(rtlib::OPXCompileDebugLevel::None);
#endif

			tracePipeline2->Init(m_Tracer.GetContext(), compileOptions, linkOptions);
		}
		//module: Load
		{
			OptixModuleCompileOptions moduleCompileOptions = {};
			moduleCompileOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
			moduleCompileOptions.numBoundValues = 0;
#ifndef NDEBUG
			moduleCompileOptions.debugLevel = static_cast<OptixCompileDebugLevel>(rtlib::OPXCompileDebugLevel::Minimal);
			moduleCompileOptions.optLevel   = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
#else
			moduleCompileOptions.optLevel   = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
			moduleCompileOptions.debugLevel = static_cast<OptixCompileDebugLevel>(rtlib::OPXCompileDebugLevel::None);
#endif
			tracePipeline2->LoadModuleFromPtxFile("RayGuiding", TEST_TEST_PG_CUDA_PATH"/RayGuiding.ptx", moduleCompileOptions);
		}
		//program group: init
		{
			tracePipeline2->LoadRayGProgramGroupFromModule("Default"   , { "RayGuiding", "__raygen__def" });
			tracePipeline2->LoadRayGProgramGroupFromModule("Guiding"   , { "RayGuiding", "__raygen__pg" });
			tracePipeline2->LoadMissProgramGroupFromModule("Radiance"  , { "RayGuiding" ,"__miss__radiance" });
			tracePipeline2->LoadMissProgramGroupFromModule("Occlusion" , { "RayGuiding" ,"__miss__occluded" });
			tracePipeline2->LoadHitGProgramGroupFromModule("DiffuseDef", { "RayGuiding" ,"__closesthit__radiance_for_diffuse_def" }, {}, {});
			tracePipeline2->LoadHitGProgramGroupFromModule("DiffusePg" , { "RayGuiding" ,"__closesthit__radiance_for_diffuse_pg" }, {}, {});
			tracePipeline2->LoadHitGProgramGroupFromModule("PhongDef"  , { "RayGuiding" ,"__closesthit__radiance_for_phong_def" }, {}, {});
			tracePipeline2->LoadHitGProgramGroupFromModule("PhongPg"   , { "RayGuiding" ,"__closesthit__radiance_for_phong_pg" }, {}, {});
			tracePipeline2->LoadHitGProgramGroupFromModule("Specular"  , { "RayGuiding" ,"__closesthit__radiance_for_specular" }, {}, {});
			tracePipeline2->LoadHitGProgramGroupFromModule("Refraction", { "RayGuiding" ,"__closesthit__radiance_for_refraction" }, {}, {});
			tracePipeline2->LoadHitGProgramGroupFromModule("Emission"  , { "RayGuiding" ,"__closesthit__radiance_for_emission" }, {}, {});
			tracePipeline2->LoadHitGProgramGroupFromModule("Occlusion" , { "RayGuiding" ,"__closesthit__occluded" }, {}, {});
			tracePipeline2->Link();
		}
		//SBTRecord
		{
			//RGData: Default
			{
				auto camera    = m_CameraController.GetCamera(m_CameraFovY, m_FbAspect);
				auto [u, v, w] = camera.getUVW();
				RayGenData rayGData = {};
				rayGData.eye = camera.getEye();
				rayGData.u = u;
				rayGData.v = v;
				rayGData.w = w;
				tracePipeline2->NewRayGRecordBuffer("Def");
				tracePipeline2->AddRayGRecordFromPG("Def", "Default", rayGData);
				tracePipeline2->GetRayGRecordBuffer("Def")->Upload();
			}
			//RGData: Guiding
			{
				auto camera = m_CameraController.GetCamera(m_CameraFovY, m_FbAspect);
				auto [u, v, w] = camera.getUVW();
				RayGenData rayGData = {};
				rayGData.eye = camera.getEye();
				rayGData.u = u;
				rayGData.v = v;
				rayGData.w = w;
				tracePipeline2->NewRayGRecordBuffer("Pg");
				tracePipeline2->AddRayGRecordFromPG("Pg", "Guiding", rayGData);
				tracePipeline2->GetRayGRecordBuffer("Pg")->Upload();
			}
			//MSData: Radiance and Occlusion
			{
				tracePipeline2->NewMissRecordBuffer("Def", RAY_TYPE_COUNT);
				tracePipeline2->AddMissRecordFromPG("Def", RAY_TYPE_RADIANCE, "Radiance", { make_float4(0.0f, 0.0f, 0.0f, 0.0f) });
				tracePipeline2->AddMissRecordFromPG("Def", RAY_TYPE_OCCLUSION, "Occlusion", { make_float4(0.0f, 0.0f, 0.0f, 0.0f) });
				tracePipeline2->GetMissRecordBuffer("Def")->Upload();
			}
			//HGData: Default
			{
				tracePipeline2->NewHitGRecordBuffer("Def", RAY_TYPE_COUNT * m_Tracer.GetTLAS()->GetSbtCount());
				{
					size_t sbtOffset = 0;
					for (auto& instanceSet : m_Tracer.GetTLAS()->GetInstanceSets()) {
						for (auto& baseGASHandle : instanceSet->baseGASHandles) {
							for (auto& mesh : baseGASHandle->GetMeshes()) {
								auto cudaVertexBuffer = mesh->GetSharedResource()->vertexBuffer.GetGpuComponent<rtlib::ext::resources::CUDABufferComponent<float3>>("CUDA");
								auto cudaNormalBuffer = mesh->GetSharedResource()->normalBuffer.GetGpuComponent<rtlib::ext::resources::CUDABufferComponent<float3>>("CUDA");
								auto cudaTexCrdBuffer = mesh->GetSharedResource()->texCrdBuffer.GetGpuComponent<rtlib::ext::resources::CUDABufferComponent<float2>>("CUDA");
								auto cudaTriIndBuffer = mesh->GetUniqueResource()->triIndBuffer.GetGpuComponent<rtlib::ext::resources::CUDABufferComponent<uint3>>("CUDA");
								for (size_t i = 0; i < mesh->GetUniqueResource()->materials.size(); ++i) {
									auto materialId = mesh->GetUniqueResource()->materials[i];
									auto& material = (*m_MaterialSet)[materialId];
									HitgroupData radianceHgData = {};
									{
										radianceHgData.vertices = cudaVertexBuffer->GetHandle().getDevicePtr();
										radianceHgData.normals = cudaNormalBuffer->GetHandle().getDevicePtr();
										radianceHgData.texCoords = cudaTexCrdBuffer->GetHandle().getDevicePtr();
										radianceHgData.indices = cudaTriIndBuffer->GetHandle().getDevicePtr();
										radianceHgData.diffuseTex  = m_Tracer.GetTexture(material.GetString("diffTex")).getHandle();
										radianceHgData.specularTex = m_Tracer.GetTexture(material.GetString("specTex")).getHandle();
										radianceHgData.emissionTex = m_Tracer.GetTexture(material.GetString("emitTex")).getHandle();
										radianceHgData.diffuse     = material.GetFloat3As<float3>("diffCol");
										radianceHgData.specular    = material.GetFloat3As<float3>("specCol");
										radianceHgData.emission    = material.GetFloat3As<float3>("emitCol");
										radianceHgData.shinness    = material.GetFloat1("shinness");
										radianceHgData.transmit    = material.GetFloat3As<float3>("tranCol");
										radianceHgData.refrInd     = material.GetFloat1("refrIndx");
									}
									auto typeString = test::SpecifyMaterialType(material);
									if (typeString == "Phong" || typeString == "Diffuse") {
										typeString += "Def";
									}
									if (material.GetString("name") == "light") {
										m_LightHgRecIndex = RAY_TYPE_COUNT * sbtOffset + RAY_TYPE_COUNT * i + RAY_TYPE_RADIANCE;
									}
									tracePipeline2->AddHitGRecordFromPG("Def", RAY_TYPE_COUNT * sbtOffset + RAY_TYPE_COUNT * i + RAY_TYPE_RADIANCE , typeString , radianceHgData);
									tracePipeline2->AddHitGRecordFromPG("Def", RAY_TYPE_COUNT * sbtOffset + RAY_TYPE_COUNT * i + RAY_TYPE_OCCLUSION, "Occlusion", {});

								}
								sbtOffset += mesh->GetUniqueResource()->materials.size();
							}
						}
					}
				}
				tracePipeline2->GetHitGRecordBuffer("Def")->Upload();
			}
			{
				tracePipeline2->NewHitGRecordBuffer("Pg", RAY_TYPE_COUNT * m_Tracer.GetTLAS()->GetSbtCount());
				{
					size_t sbtOffset = 0;
					for (auto& instanceSet : m_Tracer.GetTLAS()->GetInstanceSets()) {
						for (auto& baseGASHandle : instanceSet->baseGASHandles) {
							for (auto& mesh : baseGASHandle->GetMeshes()) {
								auto cudaVertexBuffer = mesh->GetSharedResource()->vertexBuffer.GetGpuComponent<rtlib::ext::resources::CUDABufferComponent<float3>>("CUDA");
								auto cudaNormalBuffer = mesh->GetSharedResource()->normalBuffer.GetGpuComponent<rtlib::ext::resources::CUDABufferComponent<float3>>("CUDA");
								auto cudaTexCrdBuffer = mesh->GetSharedResource()->texCrdBuffer.GetGpuComponent<rtlib::ext::resources::CUDABufferComponent<float2>>("CUDA");
								auto cudaTriIndBuffer = mesh->GetUniqueResource()->triIndBuffer.GetGpuComponent<rtlib::ext::resources::CUDABufferComponent<uint3>>("CUDA");
								for (size_t i = 0; i < mesh->GetUniqueResource()->materials.size(); ++i) {
									auto materialId = mesh->GetUniqueResource()->materials[i];
									auto& material  = (*m_MaterialSet)[materialId];
									HitgroupData radianceHgData = {};
									{
										radianceHgData.vertices = cudaVertexBuffer->GetHandle().getDevicePtr();
										radianceHgData.normals = cudaNormalBuffer->GetHandle().getDevicePtr();
										radianceHgData.texCoords = cudaTexCrdBuffer->GetHandle().getDevicePtr();
										radianceHgData.indices = cudaTriIndBuffer->GetHandle().getDevicePtr();
										radianceHgData.diffuseTex  = m_Tracer.GetTexture(material.GetString("diffTex")).getHandle();
										radianceHgData.specularTex = m_Tracer.GetTexture(material.GetString("specTex")).getHandle();
										radianceHgData.emissionTex = m_Tracer.GetTexture(material.GetString("emitTex")).getHandle();
										radianceHgData.diffuse     = material.GetFloat3As<float3>("diffCol");
										radianceHgData.specular    = material.GetFloat3As<float3>("specCol");
										radianceHgData.emission    = material.GetFloat3As<float3>("emitCol");
										radianceHgData.shinness    = material.GetFloat1("shinness");
										radianceHgData.transmit    = material.GetFloat3As<float3>("tranCol");
										radianceHgData.refrInd     = material.GetFloat1("refrIndx");
									}
									auto typeString = test::SpecifyMaterialType(material);
									if (material.GetString("name") == "light") {
										m_LightHgRecIndex = RAY_TYPE_COUNT * sbtOffset + RAY_TYPE_COUNT * i + RAY_TYPE_RADIANCE;
									}
									if (typeString == "Phong" || typeString == "Diffuse") {
										typeString += "Pg";
									}
									tracePipeline2->AddHitGRecordFromPG("Pg", RAY_TYPE_COUNT * sbtOffset + RAY_TYPE_COUNT * i + RAY_TYPE_RADIANCE , typeString , radianceHgData);
									tracePipeline2->AddHitGRecordFromPG("Pg", RAY_TYPE_COUNT * sbtOffset + RAY_TYPE_COUNT * i + RAY_TYPE_OCCLUSION, "Occlusion", {});

								}
								sbtOffset += mesh->GetUniqueResource()->materials.size();
							}
						}
					}
				}
				tracePipeline2->GetHitGRecordBuffer("Pg")->Upload();
			}
			{
				RayTraceParams params  = {};
				params.frameBuffer     = m_FrameBuffer.getDevicePtr();
				params.accumBuffer     = m_AccumBuffer.getDevicePtr();
				params.accumBuffer2    = nullptr;
				params.seedBuffer      = m_SeedBuffer.getDevicePtr();
				params.width           = m_FbWidth;
				params.height          = m_FbHeight;
				params.maxTraceDepth   = kDefaultMaxTraceDepth;
				params.gasHandle       = m_Tracer.GetTLAS()->GetHandle();
				params.light           = m_Light;
				params.samplePerALL    = 0;
				params.samplePerALL2   = 0;
				params.samplePerLaunch = kDefaultSamplePerLaunch;
				params.isBuilt         = false;
				
				tracePipeline2->NewSubPass("Def");
				tracePipeline2->AddRayGRecordBufferToSubPass("Def", "Def");
				tracePipeline2->AddMissRecordBufferToSubPass("Def", "Def");
				tracePipeline2->AddHitGRecordBufferToSubPass("Def", "Def");
				tracePipeline2->GetSubPass("Def")->InitShaderBindingTable();
				tracePipeline2->GetSubPass("Def")->InitParams(params);
				tracePipeline2->GetSubPass("Def")->SetTraceCallDepth(1);
			}
			{
				RayTraceParams params  = {};
				params.frameBuffer     = m_FrameBuffer.getDevicePtr();
				params.accumBuffer     = m_AccumBuffer.getDevicePtr();
				params.accumBuffer2    = m_AccumBufferPG.getDevicePtr();
				params.seedBuffer      = m_SeedBuffer.getDevicePtr();
				params.width           = m_FbWidth;
				params.height          = m_FbHeight;
				params.maxTraceDepth   = kDefaultMaxTraceDepth;
				params.gasHandle       = m_Tracer.GetTLAS()->GetHandle();
				params.sdTree          = m_SdTree->GetGpuHandle();
				params.light           = m_Light;
				params.samplePerALL    = 0;
				params.samplePerALL2   = 0;
				params.samplePerLaunch = kDefaultSamplePerLaunch;
				params.isBuilt         = false;

				tracePipeline2->NewSubPass("Pg");
				tracePipeline2->NewSubPass("Pg");
				tracePipeline2->AddRayGRecordBufferToSubPass("Pg", "Pg");
				tracePipeline2->AddMissRecordBufferToSubPass("Pg", "Def");
				tracePipeline2->AddHitGRecordBufferToSubPass("Pg", "Pg");
				tracePipeline2->GetSubPass("Pg")->InitShaderBindingTable();
				tracePipeline2->GetSubPass("Pg")->InitParams(params);
				tracePipeline2->GetSubPass("Pg")->SetTraceCallDepth(1);
			}
		}
		m_Tracer.SetTracePipeline(tracePipeline2);

	}
	void InitDebugPipeline() {
		auto debugPipeline2 = std::make_shared<test::RTPipeline<RayGenData, MissData, HitgroupData, RayDebugParams>>();
		{
			OptixPipelineCompileOptions compileOptions = {};

			compileOptions.pipelineLaunchParamsVariableName = "params";
			compileOptions.numAttributeValues     = 3;
			compileOptions.numPayloadValues       = 8;
			compileOptions.usesPrimitiveTypeFlags = 0;
			compileOptions.usesMotionBlur         = false;
			compileOptions.traversableGraphFlags  = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;

			OptixPipelineLinkOptions linkOptions = {};

			linkOptions.maxTraceDepth = 2;
#ifndef NDEBUG
			linkOptions.debugLevel = static_cast<OptixCompileDebugLevel>(rtlib::OPXCompileDebugLevel::Minimal);
#else
			linkOptions.debugLevel = static_cast<OptixCompileDebugLevel>(rtlib::OPXCompileDebugLevel::None);
#endif

			debugPipeline2->Init(m_Tracer.GetContext(), compileOptions, linkOptions);
		}
		//module: Load
		{
			OptixModuleCompileOptions moduleCompileOptions = {};
			moduleCompileOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
			moduleCompileOptions.numBoundValues = 0;
#ifndef NDEBUG
			moduleCompileOptions.debugLevel = static_cast<OptixCompileDebugLevel>(rtlib::OPXCompileDebugLevel::Minimal);
			moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
#else
			moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
			moduleCompileOptions.debugLevel = static_cast<OptixCompileDebugLevel>(rtlib::OPXCompileDebugLevel::None);
#endif
			debugPipeline2->LoadModuleFromPtxFile("RayDebug", TEST_TEST_PG_CUDA_PATH"/RayDebug.ptx", moduleCompileOptions);
		}
		//program group: init
		{
			debugPipeline2->LoadRayGProgramGroupFromModule("Default"   , { "RayDebug" , "__raygen__debug" });
			debugPipeline2->LoadMissProgramGroupFromModule("Radiance"  , { "RayDebug" , "__miss__debug"   });
			debugPipeline2->LoadMissProgramGroupFromModule("Occlusion" , { "RayDebug" , "__miss__debug"   });
			debugPipeline2->LoadHitGProgramGroupFromModule("Radiance"  , { "RayDebug" , "__closesthit__debug" }, {}, {});
			debugPipeline2->LoadHitGProgramGroupFromModule("Occlusion" , { "RayDebug" , "__closesthit__debug" }, {}, {});
			debugPipeline2->Link();
		}
		//SBTRecord
		{
			{
				auto camera = m_CameraController.GetCamera(m_CameraFovY, m_FbAspect);
				auto [u, v, w] = camera.getUVW();
				RayGenData rayGData = {};
				rayGData.eye = camera.getEye();
				rayGData.u = u;
				rayGData.v = v;
				rayGData.w = w;
				debugPipeline2->NewRayGRecordBuffer("Def");
				debugPipeline2->AddRayGRecordFromPG("Def", "Default", rayGData);
				debugPipeline2->GetRayGRecordBuffer("Def")->Upload();
			}
			{
				debugPipeline2->NewMissRecordBuffer("Def", RAY_TYPE_COUNT);
				debugPipeline2->AddMissRecordFromPG("Def", RAY_TYPE_RADIANCE, "Radiance", { make_float4(0.0f, 0.0f, 0.0f, 0.0f) });
				debugPipeline2->AddMissRecordFromPG("Def", RAY_TYPE_OCCLUSION, "Occlusion", { make_float4(0.0f, 0.0f, 0.0f, 0.0f) });
				debugPipeline2->GetMissRecordBuffer("Def")->Upload();
			}
			{
				debugPipeline2->NewHitGRecordBuffer("Def", RAY_TYPE_COUNT * m_Tracer.GetTLAS()->GetSbtCount());
				{
					size_t sbtOffset = 0;
					for (auto& instanceSet : m_Tracer.GetTLAS()->GetInstanceSets()) {
						for (auto& baseGASHandle : instanceSet->baseGASHandles) {
							for (auto& mesh : baseGASHandle->GetMeshes()) {
								auto cudaVertexBuffer = mesh->GetSharedResource()->vertexBuffer.GetGpuComponent<rtlib::ext::resources::CUDABufferComponent<float3>>("CUDA");
								auto cudaNormalBuffer = mesh->GetSharedResource()->normalBuffer.GetGpuComponent<rtlib::ext::resources::CUDABufferComponent<float3>>("CUDA");
								auto cudaTexCrdBuffer = mesh->GetSharedResource()->texCrdBuffer.GetGpuComponent<rtlib::ext::resources::CUDABufferComponent<float2>>("CUDA");
								auto cudaTriIndBuffer = mesh->GetUniqueResource()->triIndBuffer.GetGpuComponent<rtlib::ext::resources::CUDABufferComponent<uint3>>("CUDA");
								for (size_t i = 0; i < mesh->GetUniqueResource()->materials.size(); ++i) {
									auto materialId = mesh->GetUniqueResource()->materials[i];
									auto& material = (*m_MaterialSet)[materialId];
									HitgroupData radianceHgData = {};
									{
										radianceHgData.vertices = cudaVertexBuffer->GetHandle().getDevicePtr();
										radianceHgData.normals = cudaNormalBuffer->GetHandle().getDevicePtr();
										radianceHgData.texCoords = cudaTexCrdBuffer->GetHandle().getDevicePtr();
										radianceHgData.indices = cudaTriIndBuffer->GetHandle().getDevicePtr();
										radianceHgData.diffuseTex  = m_Tracer.GetTexture(material.GetString("diffTex")).getHandle();
										radianceHgData.specularTex = m_Tracer.GetTexture(material.GetString("specTex")).getHandle();
										radianceHgData.emissionTex = m_Tracer.GetTexture(material.GetString("emitTex")).getHandle();
										radianceHgData.diffuse     = material.GetFloat3As<float3>("diffCol");
										radianceHgData.specular    = material.GetFloat3As<float3>("specCol");
										radianceHgData.emission    = material.GetFloat3As<float3>("emitCol");
										radianceHgData.shinness    = material.GetFloat1("shinness");
										radianceHgData.transmit    = material.GetFloat3As<float3>("tranCol");
										radianceHgData.refrInd = material.GetFloat1("refrIndx");
									}
									if (material.GetString("name") == "light") {
										m_LightHgRecIndex = RAY_TYPE_COUNT * sbtOffset + RAY_TYPE_COUNT * i + RAY_TYPE_RADIANCE;
									}
									debugPipeline2->AddHitGRecordFromPG("Def", RAY_TYPE_COUNT * sbtOffset + RAY_TYPE_COUNT * i + RAY_TYPE_RADIANCE, "Radiance", radianceHgData);
									debugPipeline2->AddHitGRecordFromPG("Def", RAY_TYPE_COUNT * sbtOffset + RAY_TYPE_COUNT * i + RAY_TYPE_OCCLUSION, "Occlusion", {});

								}
								sbtOffset += mesh->GetUniqueResource()->materials.size();
							}
						}
					}
				}
				debugPipeline2->GetHitGRecordBuffer("Def")->Upload();
			}
			{
				RayDebugParams params = {};
				params.width = m_FbWidth;
				params.height = m_FbHeight;
				params.gasHandle = m_Tracer.GetTLAS()->GetHandle();
				params.light = m_Light;

				debugPipeline2->NewSubPass("Def");
				debugPipeline2->AddRayGRecordBufferToSubPass("Def", "Def");
				debugPipeline2->AddMissRecordBufferToSubPass("Def", "Def");
				debugPipeline2->AddHitGRecordBufferToSubPass("Def", "Def");
				debugPipeline2->GetSubPass("Def")->InitShaderBindingTable();
				debugPipeline2->GetSubPass("Def")->InitParams(params);
				debugPipeline2->GetSubPass("Def")->SetTraceCallDepth(1);
			}
		}
		m_Tracer.SetDebugPipeline(debugPipeline2);
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
	//ProcessFrame
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
	void OnUpdate() {
		if (m_EventFlags & EventFlags::eOnResize) {
			this->OnResize();
		}
		if (m_EventFlags & EventFlags::eOnUpdateCamera) {
			this->OnUpdateCamera();
		}
		if (m_EventFlags & EventFlags::eOnUpdateLight) {
			this->OnUpdateLight();
		}
		if (m_EventFlags & EventFlags::eOnFlushFrame) {
			this->OnFlushFrame();
		}
		if (m_EventFlags & EventFlags::eOnUpdateParams) {
			this->OnUpdateParams();
		}
		m_EventFlags = EventFlags::eNone;
	}
	void OnLaunch() {
		if (m_CurPipelineName  == "Trace") {
			auto& curPipeline   = m_Tracer.GetTracePipeline();
			auto& params        = curPipeline->GetSubPass(m_CurSubPassName)->GetParams();
			params.frameBuffer  = m_FrameBufferGL.map();
			params.samplePerALL = m_SamplePerAll;
			if (m_CurSubPassName == "Pg") {
				params.samplePerALL2 = m_SamplePerAllPg;
			}
			auto start = std::chrono::system_clock::now();
			curPipeline->Launch(m_FbWidth, m_FbHeight, m_CurSubPassName, m_Stream);
			m_FrameBufferGL.unmap();
			auto end   = std::chrono::system_clock::now();
			m_SamplePerAll     += params.samplePerLaunch;
			if (m_CurSubPassName == "Pg") {
				m_SamplePerAllPg += params.samplePerLaunch;
			}
			if (m_RequiredVariance)
			{
				const size_t numSamples = m_FbWidth * m_FbHeight;
				m_AccumBuffer.download(m_AccumImage);
				float aver1 = 0.0f;
				float aver2 = 0.0f;
				for (auto& pixel: m_AccumImage) {
					if (isnan(pixel.x) || isnan(pixel.y) || isnan(pixel.z)) {
						printf("Bug!\n");
					}
					float pixelGray = rtlib::dot(make_float3(1.0f), pixel) / (3.0f* (float)m_SamplePerAll);
					aver1 += pixelGray;
					aver2 += pixelGray* pixelGray;
				}
				aver1 /= (float)numSamples;
				aver2 /= (float)numSamples;
				m_Variance = aver2 - aver1 * aver1;
				if (m_CurSubPassName == "Pg") {
					auto variancePg  = 0.0f;
					m_AccumBufferPG.download(m_AccumImagePg);
					{
						float aver1 = 0.0f;
						float aver2 = 0.0f;
						for (auto& pixel : m_AccumImagePg) {
							if (isnan(pixel.x) || isnan(pixel.y) || isnan(pixel.z)) {
								printf("Bug!\n");
							}
							float pixelGray = rtlib::dot(make_float3(1.0f), pixel) / (3.0f * (float)m_SamplePerAllPg);
							aver1 += pixelGray;
							aver2 += pixelGray * pixelGray;
						}
						aver1 /= (float)(m_AccumImagePg.size());
						aver2 /= (float)(m_AccumImagePg.size());
						variancePg = aver2 - aver1 * aver1;
					}
					m_VariancePg = variancePg;
					printf("Spp %d: Time = %d (ms), Variance = %.9f\n", m_SamplePerAllPg, std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count(), m_VariancePg);
				}
				else {
					printf("Spp %d: Time = %d (ms), Variance = %.9f\n", m_SamplePerAll  , std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count(), m_Variance  );
				}
			}
		}
		else {
			auto& curPipeline     = m_Tracer.GetDebugPipeline();
			auto& params          = curPipeline->GetSubPass("Def")->GetParams();
			params.diffuseBuffer  = m_DebugBufferGLs["Diffuse"].map();
			params.specularBuffer = m_DebugBufferGLs["Specular"].map();
			params.emissionBuffer = m_DebugBufferGLs["Emission"].map();
			params.transmitBuffer = m_DebugBufferGLs["Transmit"].map();
			params.texCoordBuffer = m_DebugBufferGLs["TexCoord"].map();
			params.normalBuffer   = m_DebugBufferGLs["Normal"].map();
			params.depthBuffer    = m_DebugBufferGLs["Depth"].map();
			params.sTreeColBuffer = m_DebugBufferGLs["STree"].map();
			curPipeline->Launch(m_FbWidth, m_FbHeight, "Def", m_Stream);
			m_DebugBufferGLs["Diffuse"].unmap();
			m_DebugBufferGLs["Specular"].unmap();
			m_DebugBufferGLs["Emission"].unmap();
			m_DebugBufferGLs["Transmit"].unmap();
			m_DebugBufferGLs["TexCoord"].unmap();
			m_DebugBufferGLs["Normal"].unmap();
			m_DebugBufferGLs["Depth"].unmap();
			m_DebugBufferGLs["STree"].unmap();
		}

	}
	void OnRenderFrame() {
		if (m_CurPipelineName == "Trace") {
			m_GLTexture.upload(0, m_FrameBufferGL.getHandle(), 0, 0, m_FbWidth, m_FbHeight);
		}
		else {
			m_GLTexture.upload(0, m_DebugBufferGLs[m_DebugFrameName].getHandle(), 0, 0, m_FbWidth, m_FbHeight);
		}
		glClearColor(0.8f, 0.8f, 0.8f, 0.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glViewport(0, 0, m_FbWidth, m_FbHeight);
		m_RectRenderer->draw(m_GLTexture.getID());
	}
	void OnRenderImGui() {
		{
			ImGui_ImplOpenGL3_NewFrame();
			ImGui_ImplGlfw_NewFrame();
			ImGui::NewFrame();
			{
				ImGui::PushStyleColor(ImGuiCol_TitleBgActive, ImVec4(0.0f, 0.7f, 0.2f, 1.0f));
				ImGui::PushStyleColor(ImGuiCol_TitleBg, ImVec4(0.0f, 0.3f, 0.1f, 1.0f));
				ImGui::SetNextWindowPos(ImVec2(20, 20), ImGuiCond_Once);
				ImGui::SetNextWindowSize(ImVec2(500, 500), ImGuiCond_Once);

				ImGui::Begin("TraceConfig", nullptr, ImGuiWindowFlags_MenuBar);
				{
					ImGui::BeginChild(ImGui::GetID((void*)0), ImVec2(450, 450), ImGuiWindowFlags_NoTitleBar);
					{
						ImGui::Text("Fps  : %.2f", 1.0f / m_DelTime);
						ImGui::Text("Smp  : %3d", m_SamplePerAll);
						ImGui::Text("Smp/s: %.2f", m_SamplePerLaunch / (float)m_DelTime);
						//Camera and Light
						if(m_CurSubPassName!="Pg") {

							float camFovY = m_CameraFovY;
							if (ImGui::SliderFloat("Camera.FovY", &camFovY, -90.0f, 90.0f)) {
								m_CameraFovY      = camFovY;
								m_EventFlags     |= EventFlags::eUpdateCamera;
							}
							float emission[3] = { m_Light.emission.x, m_Light.emission.y, m_Light.emission.z };
							if (ImGui::SliderFloat3("light.Color", emission, 0.0f, 10.0f)) {
								m_Light.emission.x = emission[0];
								m_Light.emission.y = emission[1];
								m_Light.emission.z = emission[2];
								m_EventFlags      |= EventFlags::eUpdateLight;
							}
						}
						else {
							ImGui::Text("Camera.FovY  : %.2f", m_CameraFovY);
							ImGui::Text("light.Color  : (%.2f, %.2f, %.2f)", m_Light.emission.x, m_Light.emission.y, m_Light.emission.z);
						}

						if (m_CurPipelineName == "Trace" && m_CurSubPassName == "Def")
						{
							auto  curPipeline = m_Tracer.GetTracePipeline();
							auto  curSubpass  = curPipeline->GetSubPass(m_CurSubPassName);
							auto& curParams   = curSubpass->GetParams();
							{
								int samplePerLaunch = curParams.samplePerLaunch;
								if (ImGui::SliderInt("samplePerLaunch", &samplePerLaunch, 1, 10)) {
									curParams.samplePerLaunch = samplePerLaunch;
									m_SamplePerLaunch         = samplePerLaunch;
								}
							}
							{
								int maxTraceDepth = curParams.maxTraceDepth;
								if (ImGui::SliderInt("maxTraceDepth", &maxTraceDepth, 1, 10)) {
									curParams.maxTraceDepth   = maxTraceDepth;
									m_EventFlags             |= EventFlags::eFlushFrame;
								}
							}
							{
								int sppBudget = m_SppBudget;
								if (ImGui::SliderInt("sppBudget", &sppBudget, 1, 10000)) {
									m_SppBudget = sppBudget;
								}
							}
							if (ImGui::Button("Save")) {
							}
							ImGui::SameLine();
							if (ImGui::Button("Run")) {
								OnPreparePG();
							}
						}
						if (m_CurPipelineName == "Debug") {
							OnSelectDebugFrame();
						}
						if (ImGui::Button(m_PrvPipelineName.c_str())) {
							std::swap(m_CurPipelineName, m_PrvPipelineName);
							m_EventFlags |= EventFlags::eUpdateCamera;
						}
					}
					ImGui::EndChild();
				}
				ImGui::End();
				ImGui::PopStyleColor();
				ImGui::PopStyleColor();
			}
			// Rendering
			ImGui::Render();
			ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
		}
	}
	void OnUpdateTime() {
		float prevTime = glfwGetTime();
		m_DelTime = prevTime - m_CurTime;
		m_CurTime = prevTime;
	}
	void OnReactInputs() {
		if (glfwGetKey(m_Window, GLFW_KEY_W) == GLFW_PRESS) {
			m_CameraController.ProcessKeyboard(rtlib::ext::CameraMovement::eForward, m_DelTime);
			m_EventFlags |= EventFlags::eUpdateCamera;
		}
		if (glfwGetKey(m_Window, GLFW_KEY_S) == GLFW_PRESS) {
			m_CameraController.ProcessKeyboard(rtlib::ext::CameraMovement::eBackward, m_DelTime);
			m_EventFlags |= EventFlags::eUpdateCamera;
		}
		if (glfwGetKey(m_Window, GLFW_KEY_A) == GLFW_PRESS) {
			m_CameraController.ProcessKeyboard(rtlib::ext::CameraMovement::eLeft, m_DelTime);
			m_EventFlags |= EventFlags::eUpdateCamera;
		}
		if (glfwGetKey(m_Window, GLFW_KEY_D) == GLFW_PRESS) {
			m_CameraController.ProcessKeyboard(rtlib::ext::CameraMovement::eRight, m_DelTime);
			m_EventFlags |= EventFlags::eUpdateCamera;
		}
		if (glfwGetKey(m_Window, GLFW_KEY_LEFT) == GLFW_PRESS) {
			m_CameraController.ProcessKeyboard(rtlib::ext::CameraMovement::eLeft, m_DelTime);
			m_EventFlags |= EventFlags::eUpdateCamera;
		}
		if (glfwGetKey(m_Window, GLFW_KEY_RIGHT) == GLFW_PRESS) {
			m_CameraController.ProcessKeyboard(rtlib::ext::CameraMovement::eRight, m_DelTime);
			m_EventFlags |= EventFlags::eUpdateCamera;
		}
		if (glfwGetKey(m_Window, GLFW_KEY_UP) == GLFW_PRESS) {
			m_CameraController.ProcessKeyboard(rtlib::ext::CameraMovement::eUp, m_DelTime);
			m_EventFlags |= EventFlags::eUpdateCamera;
		}
		if (glfwGetKey(m_Window, GLFW_KEY_DOWN) == GLFW_PRESS) {
			m_CameraController.ProcessKeyboard(rtlib::ext::CameraMovement::eDown, m_DelTime);
			m_EventFlags |= EventFlags::eUpdateCamera;
		}
		if (glfwGetMouseButton(m_Window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS) {
			m_CameraController.ProcessMouseMovement(-m_DelCursorPos.x, m_DelCursorPos.y);
			m_EventFlags |= EventFlags::eUpdateCamera;
		}
	}
	//UpdateTask
	void OnResize() {
		std::random_device rd;
		std::mt19937 mt(rd());
		std::vector<unsigned int> seeds(m_FbWidth * m_FbHeight);
		std::generate(seeds.begin(), seeds.end(), mt);

		m_SeedBuffer.resize(m_FbWidth * m_FbHeight);
		m_SeedBuffer.upload(seeds);

		m_FrameBuffer.resize(m_FbWidth * m_FbHeight);
		m_RefImage.resize(m_FbWidth * m_FbHeight);
		for (auto& [name, debugBuffer] : m_DebugBuffers)
		{
			debugBuffer.resize(m_FbWidth * m_FbHeight);
		}

		m_FrameBufferGL.resize(m_FbWidth * m_FbHeight);
		for (auto& [name, debugBufferGL] : m_DebugBufferGLs)
		{
			debugBufferGL.resize(m_FbWidth * m_FbHeight);
		}

		m_AccumBuffer.resize(m_FbWidth * m_FbHeight);
		{
			m_GLTexture.reset();
			m_GLTexture.allocate({ (size_t)m_FbWidth,(size_t)m_FbHeight }, GL_TEXTURE_2D);
			m_GLTexture.setParameteri(GL_TEXTURE_MAG_FILTER, GL_LINEAR, false);
			m_GLTexture.setParameteri(GL_TEXTURE_MIN_FILTER, GL_LINEAR, false);
			m_GLTexture.setParameteri(GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE, false);
			m_GLTexture.setParameteri(GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE, false);
		}
		m_Tracer.GetTracePipeline()->GetSubPass("Def")->GetParams().accumBuffer = m_AccumBuffer.getDevicePtr();
		m_Tracer.GetTracePipeline()->GetSubPass("Def")->GetParams().seedBuffer        = m_SeedBuffer.getDevicePtr();
		m_Tracer.GetTracePipeline()->GetSubPass("Def")->GetParams().width       = m_FbWidth;
		m_Tracer.GetTracePipeline()->GetSubPass("Def")->GetParams().height      = m_FbHeight;

		m_Tracer.GetTracePipeline()->GetSubPass("Pg")->GetParams().accumBuffer  = m_AccumBuffer.getDevicePtr();
		m_Tracer.GetTracePipeline()->GetSubPass("Pg")->GetParams().seedBuffer         = m_SeedBuffer.getDevicePtr();
		m_Tracer.GetTracePipeline()->GetSubPass("Pg")->GetParams().width        = m_FbWidth;
		m_Tracer.GetTracePipeline()->GetSubPass("Pg")->GetParams().height       = m_FbHeight;

		m_Tracer.GetDebugPipeline()->GetSubPass("Def")->GetParams().width       = m_FbWidth;
		m_Tracer.GetDebugPipeline()->GetSubPass("Def")->GetParams().height      = m_FbHeight;
	}
	void OnUpdateCamera()
	{
		auto camera = m_CameraController.GetCamera(m_CameraFovY, m_FbAspect);
		auto [u, v, w] = camera.getUVW();

		m_Tracer.GetTracePipeline()->GetRayGRecordBuffer("Def")->GetData().eye = camera.getEye();
		m_Tracer.GetTracePipeline()->GetRayGRecordBuffer("Def")->GetData().u = u;
		m_Tracer.GetTracePipeline()->GetRayGRecordBuffer("Def")->GetData().v = v;
		m_Tracer.GetTracePipeline()->GetRayGRecordBuffer("Def")->GetData().w = w;
		m_Tracer.GetTracePipeline()->GetRayGRecordBuffer("Def")->Upload();

		m_Tracer.GetTracePipeline()->GetRayGRecordBuffer("Pg")->GetData().eye = camera.getEye();
		m_Tracer.GetTracePipeline()->GetRayGRecordBuffer("Pg")->GetData().u = u;
		m_Tracer.GetTracePipeline()->GetRayGRecordBuffer("Pg")->GetData().v = v;
		m_Tracer.GetTracePipeline()->GetRayGRecordBuffer("Pg")->GetData().w = w;
		m_Tracer.GetTracePipeline()->GetRayGRecordBuffer("Pg")->Upload();

		m_Tracer.GetDebugPipeline()->GetRayGRecordBuffer("Def")->GetData().eye = camera.getEye();
		m_Tracer.GetDebugPipeline()->GetRayGRecordBuffer("Def")->GetData().u = u;
		m_Tracer.GetDebugPipeline()->GetRayGRecordBuffer("Def")->GetData().v = v;
		m_Tracer.GetDebugPipeline()->GetRayGRecordBuffer("Def")->GetData().w = w;
		m_Tracer.GetDebugPipeline()->GetRayGRecordBuffer("Def")->Upload();
	}
	void OnUpdateLight() {
		m_Tracer.GetTracePipeline()->GetHitGRecordBuffer("Def")->GetData(m_LightHgRecIndex).emission = m_Light.emission;
		m_Tracer.GetTracePipeline()->GetHitGRecordBuffer("Def")->GetData(m_LightHgRecIndex).diffuse  = m_Light.emission;
		m_Tracer.GetTracePipeline()->GetHitGRecordBuffer("Def")->Upload();

		m_Tracer.GetTracePipeline()->GetHitGRecordBuffer("Pg")->GetData(m_LightHgRecIndex).emission = m_Light.emission;
		m_Tracer.GetTracePipeline()->GetHitGRecordBuffer("Pg")->GetData(m_LightHgRecIndex).diffuse  = m_Light.emission;
		m_Tracer.GetTracePipeline()->GetHitGRecordBuffer("Pg")->Upload();

		m_Tracer.GetDebugPipeline()->GetHitGRecordBuffer("Def")->GetData(m_LightHgRecIndex).emission = m_Light.emission;
		m_Tracer.GetDebugPipeline()->GetHitGRecordBuffer("Def")->GetData(m_LightHgRecIndex).diffuse  = m_Light.emission;
		m_Tracer.GetDebugPipeline()->GetHitGRecordBuffer("Def")->Upload();
	}
	void OnFlushFrame() {
		//Frame�̍Ď擾
		m_FrameBufferGL.upload(std::vector<uchar4>(m_FbWidth * m_FbHeight));
		m_AccumBuffer.upload(std::vector<float3>(m_FbWidth * m_FbHeight));
		for (auto& [name, debugBufferGL] : m_DebugBufferGLs)
		{
			debugBufferGL.upload(std::vector<uchar4>(m_FbWidth * m_FbHeight));
		}
		m_SamplePerAll = 0;
	}
	void OnUpdateParams()
	{

		//params�̍Đݒ�
		if (m_CurPipelineName == "Trace") {
			auto& params          = m_Tracer.GetTracePipeline()->GetSubPass(m_CurSubPassName)->GetParams();
			params.frameBuffer    = m_FrameBuffer.getDevicePtr();
			params.accumBuffer    = m_AccumBuffer.getDevicePtr();
			params.seedBuffer     = m_SeedBuffer.getDevicePtr();
			params.width          = m_FbWidth;
			params.height         = m_FbHeight;
			params.gasHandle      = m_Tracer.GetTLAS()->GetHandle();
			params.sdTree         = m_SdTree->GetGpuHandle();
			params.light          = m_Light;
			params.samplePerALL   = m_SamplePerAll;
			params.samplePerLaunch= m_SamplePerLaunch;
			if (m_CurSubPassName == "Pg")
			{
				params.accumBuffer2  = m_AccumBufferPG.getDevicePtr();
				params.samplePerALL2 = m_SamplePerAllPg;
			}
		}
		else {
			auto& params          = m_Tracer.GetDebugPipeline()->GetSubPass("Def")->GetParams();
			params.diffuseBuffer  = m_DebugBuffers[ "Diffuse"].getDevicePtr();
			params.specularBuffer = m_DebugBuffers["Specular"].getDevicePtr();
			params.emissionBuffer = m_DebugBuffers["Emission"].getDevicePtr();
			params.transmitBuffer = m_DebugBuffers["Transmit"].getDevicePtr();
			params.texCoordBuffer = m_DebugBuffers["TexCoord"].getDevicePtr();
			params.normalBuffer   = m_DebugBuffers[  "Normal"].getDevicePtr();
			params.depthBuffer    = m_DebugBuffers[   "Depth"].getDevicePtr();
			params.sTreeColBuffer = m_DebugBuffers[   "STree"].getDevicePtr();
			params.width          = m_FbWidth;
			params.height         = m_FbHeight;
			params.gasHandle      = m_Tracer.GetTLAS()->GetHandle();
			params.sdTree         = m_SdTree->GetGpuHandle();
			params.light          = m_Light;
		}
	}
	//GUI
	void OnSelectDebugFrame(){
		int debugMode = 0;
		{
			for(auto& frameName: debugPipelineFrameNames){
				if(m_DebugFrameName==frameName){
					break;
				}
				debugMode++;
			}
		}
		{
			int i = 0;
			for(auto& frameName: debugPipelineFrameNames){
				ImGui::RadioButton(frameName.data(), &debugMode, i);
				if (i == std::size(debugPipelineFrameNames) / 2 -1) {
					ImGui::NewLine();
				}else if(i != std::size(debugPipelineFrameNames) - 1) {
					ImGui::SameLine();
				}
				i++;
			}
		}
		{
			m_DebugFrameName = debugPipelineFrameNames[debugMode];
		}
	}
	//PathGuiding
	void OnPreparePG() {
		auto& curPipeline  = m_Tracer.GetTracePipeline();
		auto& curPgParams  = curPipeline->GetSubPass("Pg")->GetParams();
		auto& curDefParams = curPipeline->GetSubPass("Def")->GetParams();
		OnClearSDTree();
		OnUpLoadSDTree();
		m_AccumBufferPG.resize(m_FbWidth * m_FbHeight);
		m_AccumBufferPG.upload(std::vector<float3>(m_FbWidth * m_FbHeight));
		m_AccumImagePg            = std::vector<float3>(m_FbWidth * m_FbHeight);
		m_SamplePerAllPg          = 0;
		m_VariancePg              = 0.0f;
		m_NumPasses               = (int)std::ceil(m_SppBudget / (float)m_SamplePerLaunch);
		m_TmpPasses               = 0;
		m_RenderPasses            = 0;
		m_CurPasses               = 0;
		m_IsFinalIteration        = false;
		m_IsAccumulated           = false;
		m_CurIteration            = 0;
		m_CurVariance             = std::numeric_limits<float>::max();
		curPgParams.accumBuffer2  = m_AccumBufferPG.getDevicePtr();
		curPgParams.samplePerALL2 = m_SamplePerAllPg;
		curPgParams.maxTraceDepth = curDefParams.maxTraceDepth;
		curPgParams.isBuilt       = false;

		m_CurSubPassName          = "Pg";
		m_EventFlags             |= EventFlags::eFlushFrame;
	}
	void OnBeginPG() {
		//printf("m_RenderPasses=%d\n", m_RenderPasses);
		if (m_RenderPasses >= m_NumPasses) {
			OnUpLoadSDTree();
			m_CurSubPassName = "Def";
			m_EventFlags |= EventFlags::eFlushFrame;
			return;
		}
		if (m_TmpPasses == 0 && !m_IsAccumulated) {
			//printf("Iteration :%d\n", m_CurIteration);
			//Sampling開始時には、設定を行う
			m_RemainPasses = m_NumPasses - m_RenderPasses;
			m_CurPasses    = std::min<uint32_t>(m_RemainPasses, 1 << m_CurIteration);
			if (m_RemainPasses - m_CurPasses < 2 * m_CurPasses)
			{
				m_CurPasses = m_RemainPasses;
			}
			if (!m_IsFinalIteration) {
				if (m_CurPasses >= m_RemainPasses) {
					//printf("Final Iteration!\n");
					m_IsFinalIteration = true;
				}
				this->OnDwLoadSDTree();
				this->OnResetSDTree();
				this->OnUpLoadSDTree();
				m_EventFlags |= EventFlags::eFlushFrame;
			}
			m_IsAccumulated = true;
		}
	}
	void OnEndPG() {
		m_RenderPasses++;
		if (m_IsAccumulated) {
			m_TmpPasses++;
			if (m_TmpPasses >= m_CurPasses) {
				m_IsAccumulated = false;
			}
		}
		if (!m_IsAccumulated) {
			m_PrvVariance = m_CurVariance;
			m_CurVariance = m_CurPasses * m_Variance / m_RemainPasses;
			//printf("Extrapolated var: Prv: %f Cur: %f Total: %f/%d\n", m_PrvVariance, m_CurVariance, m_VariancePg, m_SamplePerAllPg);
			m_RemainPasses -= m_CurPasses;

			auto& curParams = m_Tracer.GetTracePipeline()->GetSubPass("Pg")->GetParams();
			curParams.isBuilt = true;

			
			this->OnDwLoadSDTree();
			this->OnBuildSDTree();
			this->OnUpLoadSDTree();
			//Next Iteration
			m_TmpPasses = 0;
			m_CurIteration++;
		}
	}
	//SDTree
	void OnUpLoadSDTree() {
		m_SdTree->Upload();
	}
	void OnDwLoadSDTree() {
		m_SdTree->Download();
	}
	void OnClearSDTree()  {
		m_SdTree->Clear();
	}
	void OnResetSDTree()  {
		m_SdTree->Reset(m_CurIteration, m_SamplePerLaunch);
	}
	void OnBuildSDTree()  {
		m_SdTree->Build();
	}
private:
	static void frameBufferSizeCallback(GLFWwindow* window, int fbWidth, int fbHeight)
	{
		Test20Application* app = reinterpret_cast<Test20Application*>(glfwGetWindowUserPointer(window));
		if (app) {
			if (fbWidth != app->m_FbWidth || fbHeight != app->m_FbHeight)
			{
				app->m_FbWidth     = fbWidth;
				app->m_FbHeight    = fbHeight;
				app->m_FbAspect    = static_cast<float>(fbWidth) / static_cast<float>(fbHeight);
				app->m_EventFlags |= EventFlags::eResize;
			}
		}
	}
	static void cursorPosCallback(GLFWwindow* window, double xPos, double yPos)
	{
		Test20Application* app = reinterpret_cast<Test20Application*>(glfwGetWindowUserPointer(window));
		if (app) {
			app->m_DelCursorPos.x = xPos - app->m_CurCursorPos.x;
			app->m_DelCursorPos.y = yPos - app->m_CurCursorPos.y;
			app->m_CurCursorPos.x = xPos;
			app->m_CurCursorPos.y = yPos;
		}
	}
private:
	using RectRendererPtr    = std::shared_ptr<rtlib::ext::RectRenderer>;
	using CUDABufferMap      = std::unordered_map<std::string, rtlib::CUDABuffer<uchar4>>;
	using GLInteropBufferMap = std::unordered_map<std::string, rtlib::GLInteropBuffer<uchar4>>;
	using SDTreePtr          = std::unique_ptr<test::RTSTreeWrapper>;
private:
	GLFWwindow*                     m_Window             = nullptr;
	int			      	            m_FbWidth            = 0;
	int                             m_FbHeight           = 0;
	float                           m_FbAspect           = 1.0f;
	std::string                     m_Title              = {};
	std::string                     m_GlslVersion        = {};
	//Inputs
	float2                          m_DelCursorPos       = {};
	float2                          m_CurCursorPos       = {};
	float                           m_CurTime            = 0.0f;
	float                           m_DelTime            = 0.0f;
	float                           m_CameraFovY         = 30.0f;
	//State
	rtlib::GLTexture2D<uchar4>      m_GLTexture          = {};
	RectRendererPtr                 m_RectRenderer       = {};
	rtlib::ext::CameraController         m_CameraController   = {};
	test::RTTracer                  m_Tracer             = {};
	rtlib::ext::VariableMapListPtr     m_MaterialSet        = nullptr;
	CUstream                        m_Stream             = nullptr;
	//Trace
	rtlib::CUDABuffer<uchar4>       m_FrameBuffer        = {};
	rtlib::GLInteropBuffer<uchar4>  m_FrameBufferGL      = {};
	rtlib::CUDABuffer<float3>       m_AccumBuffer        = {};
	std::vector<float3>             m_AccumImage         = {};
	rtlib::CUDABuffer<unsigned int> m_SeedBuffer         = {};
	//Variance
	std::vector<float3>             m_RefImage           = {};
	std::vector<float3>             m_CurImage           = {};
	float                           m_CurVariance        = 0.0f;
	float                           m_PrvVariance        = std::numeric_limits<float>::max();
	float                           m_CurError           = 0.0f;
	float                           m_PrvError           = std::numeric_limits<float>::max();
	//Guiding
	SDTreePtr                       m_SdTree             = nullptr;
	rtlib::utils::AABB              m_WorldAABB          = {};
	rtlib::CUDABuffer<float3>       m_AccumBufferPG      = {};
	std::vector<float3>             m_AccumImagePg       = {};
	float                           m_VariancePg         = 0.0f;
	uint32_t                        m_SamplePerAllPg     = 0;
	uint32_t                        m_SppBudget          = 0;
	uint32_t                        m_NumPasses          = 0;
	uint32_t                        m_TmpPasses          = 0;
	uint32_t                        m_CurPasses          = 0;
	uint32_t                        m_RenderPasses       = 0;
	uint32_t                        m_RemainPasses       = 0;
	uint32_t                        m_CurIteration       = 0;
	bool                            m_IsAccumulated      = false;
	bool                            m_IsFinalIteration   = false;
	//Debug
	CUDABufferMap                   m_DebugBuffers       = {};
	GLInteropBufferMap              m_DebugBufferGLs     = {};
	//Light
	uint32_t                        m_LightHgRecIndex    = 0;
	ParallelLight                   m_Light              = {};
	//Config 
	uint32_t                        m_SamplePerAll       = 0;
	uint32_t                        m_SamplePerLaunch    = kDefaultSamplePerLaunch;
	float                           m_Variance           = 0.0f;
	//State
	int                             m_EventFlags         = EventFlags::eNone;
	bool                            m_RequiredVariance   = true;
	std::string                     m_CurPipelineName    = "Trace";
	std::string                     m_PrvPipelineName    = "Debug";
	std::string                     m_CurSubPassName     = "Def";
	std::string                     m_DebugFrameName     = "Diffuse";
};
void  Test20Main() {
	Test20Application app;
	app.Initialize();
	app.MainLoop();
	app.CleanUp();
}
int main() {
	//TestPathGuide();
	Test20Main();
}