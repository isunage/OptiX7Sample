#include <Test19Config.h>
#include <RTLib/core/GL.h>
#include <RTLib/core/CUDA.h>
#include <RTLib/core/CUDA_GL.h>
#include <RTLib/ext/Camera.h>
#include <RTLib/ext/Utils.h>
#include <RTLib/ext/RectRenderer.h>
#include <RTLib/ext/Resources/CUDA.h>
#include <cuda/RayTrace.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include "../include/RTTracer.h"
#include "../include/SceneBuilder.h"
#include <fstream>
#include <unordered_map>
#include <random>
#include <sstream>
#include <string>
#define TEST19_USE_NEE
namespace test {
	std::string SpecifyMaterialType(const rtlib::ext::VariableMap& material) {
		auto emitCol  = material.GetFloat3As<float3>("emitCol");
		auto tranCol  = material.GetFloat3As<float3>("tranCol");
		auto refrIndx = material.GetFloat1("refrIndx");
		auto shinness = material.GetFloat1("shinness");
		if (emitCol.x + emitCol.y + emitCol.z != 0.0f) {
			return "Emission";
		}
		else if (refrIndx > 1.61f &&
			tranCol.x + tranCol.y + tranCol.z != 0.0f) {
			return "Refraction";
		}
		else if (shinness > 300) {
			return "Specular";
		}
		else {
			return "Diffuse";
		}
	}
}
class Test19Application {
private:
	static inline constexpr uint32_t kDefaultSamplePerLaunch = 1;
	static inline constexpr uint32_t kDefaultMaxTraceDepth = 4;
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
		m_Tracer2.SetContext(std::make_shared<rtlib::OPXContext>(rtlib::OPXContext::Desc{ 0, 0, OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL, 4 }));
	}
	void InitCamera() {
		m_CameraController = rtlib::ext::CameraController({ 0.0f,1.0f, 5.0f });
		m_CameraController.SetMouseSensitivity(0.125f);
		m_CameraController.SetMovementSpeed(50.0f);
	}
	void LoadScene() {
		auto objMeshGroup = std::make_shared<test::ObjMeshGroup>();
		if (!objMeshGroup->Load(TEST_TEST19_DATA_PATH"/Models/Sponza/sponza.obj", TEST_TEST19_DATA_PATH"/Models/Sponza/")) {
			throw std::runtime_error("Failed To Load Model!");
		}
		m_MaterialSet = objMeshGroup->GetMaterialList();

		{
			for (auto& material : *m_MaterialSet) {
				auto diffTex = material.GetString("diffTex") != "" ? material.GetString("diffTex") : std::string(TEST_TEST19_DATA_PATH"/Textures/white.png");
				auto specTex = material.GetString("specTex") != "" ? material.GetString("specTex") : std::string(TEST_TEST19_DATA_PATH"/Textures/white.png");
				auto emitTex = material.GetString("emitTex") != "" ? material.GetString("emitTex") : std::string(TEST_TEST19_DATA_PATH"/Textures/white.png");
				if (!m_Tracer2.HasTexture(material.GetString("diffTex"))) {
					m_Tracer2.LoadTexture(material.GetString("diffTex"), diffTex);
				}
				if (!m_Tracer2.HasTexture(material.GetString("specTex"))) {
					m_Tracer2.LoadTexture(material.GetString("specTex"), specTex);
				}
				if (!m_Tracer2.HasTexture(material.GetString("emitTex"))) {
					m_Tracer2.LoadTexture(material.GetString("emitTex"), emitTex);
				}
			}
		}
		bool isLightFound = false;
		//GAS1: World
		m_Tracer2.NewGASHandle("world");
		auto worldGASHandle = m_Tracer2.GetGASHandle("world");
		{
			bool isLightFound = false;
			OptixAccelBuildOptions accelOptions = {};
			accelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
			accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
			for (auto& [name, meshGroup] : m_Tracer2.GetMeshGroups()) {
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
				for (auto& [meshUniqueName, meshUniqueResource] : meshGroup->GetUniqueResources()) {
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
				worldGASHandle->Build(m_Tracer2.GetContext().get(), accelOptions);
			}
		}
		//GAS2: Light
		m_Tracer2.NewGASHandle("light");
		auto lightGASHandle = m_Tracer2.GetGASHandle("light");
		if (isLightFound) {
			OptixAccelBuildOptions accelOptions = {};
			accelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
			accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
			lightGASHandle->AddMesh(objMeshGroup->GetMeshGroup()->LoadMesh("light"));
			lightGASHandle->Build(m_Tracer2.GetContext().get(), accelOptions);
		}
		else {
			rtlib::utils::AABB aabb = {};
			for (auto& vertex : objMeshGroup->GetMeshGroup()->GetSharedResource()->vertexBuffer) {
				aabb.Update(vertex);
			}
			OptixAccelBuildOptions accelOptions = {};
			accelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
			accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
			auto lightMesh = rtlib::ext::Mesh::New();
			lightMesh->SetSharedResource(rtlib::ext::MeshSharedResource::New());
			lightMesh->GetSharedResource()->name = "light";
			lightMesh->GetSharedResource()->vertexBuffer = {
				{aabb.min.x,aabb.max.y + 1e-3f,aabb.min.z},
				{aabb.max.x,aabb.max.y + 1e-3f,aabb.min.z},
				{aabb.max.x,aabb.max.y + 1e-3f,aabb.max.z},
				{aabb.min.x,aabb.max.y + 1e-3f,aabb.max.z}
			};
			lightMesh->GetSharedResource()->texCrdBuffer = {
				{0.0f,0.0f},
				{1.0f,0.0f},
				{1.0f,1.0f},
				{0.0f,1.0f},
			};
			lightMesh->GetSharedResource()->normalBuffer = {
				{0.0f,-1.0f,0.0f},
				{0.0f,-1.0f,0.0f},
				{0.0f,-1.0f,0.0f},
				{0.0f,-1.0f,0.0f},
			};
			unsigned int curMaterialSetCount = m_MaterialSet->size();
			auto lightMaterial = rtlib::ext::VariableMap{};
			{
				lightMaterial.SetString("name", "light");
				lightMaterial.SetFloat3("diffCol",{ 10.0f, 10.0f, 10.0f });
				lightMaterial.SetString("diffTex", "");
				lightMaterial.SetFloat3("emitCol", { 10.0f, 10.0f, 10.0f });
				lightMaterial.SetString("emitTex", "");
				lightMaterial.SetFloat3("specCol", { 10.0f, 10.0f, 10.0f });
				lightMaterial.SetString("specTex", "");
				lightMaterial.SetFloat1("shinness",0.0f);
				lightMaterial.SetString("shinTex", "");
				lightMaterial.SetFloat3("tranCol", {  0.0f,  0.0f,  0.0f });
				lightMaterial.SetFloat1("refrIndx", 0.0f);
			}
			m_MaterialSet->push_back(
				lightMaterial
			);
			lightMesh->GetSharedResource()->vertexBuffer.AddGpuComponent<rtlib::ext::resources::CUDABufferComponent<float3>>("CUDA");
			lightMesh->GetSharedResource()->normalBuffer.AddGpuComponent<rtlib::ext::resources::CUDABufferComponent<float3>>("CUDA");
			lightMesh->GetSharedResource()->texCrdBuffer.AddGpuComponent<rtlib::ext::resources::CUDABufferComponent<float2>>("CUDA");
			lightMesh->SetUniqueResource(rtlib::ext::MeshUniqueResource::New());
			lightMesh->GetUniqueResource()->name = "light";
			lightMesh->GetUniqueResource()->materials = { curMaterialSetCount };
			lightMesh->GetUniqueResource()->matIndBuffer = { 0,0 };
			lightMesh->GetUniqueResource()->triIndBuffer = { {0,1,2}, {2,3,0} };
			lightMesh->GetUniqueResource()->matIndBuffer.AddGpuComponent<rtlib::ext::resources::CUDABufferComponent<uint32_t>>("CUDA");
			lightMesh->GetUniqueResource()->triIndBuffer.AddGpuComponent<rtlib::ext::resources::CUDABufferComponent<uint3>>("CUDA");
			lightGASHandle->AddMesh(lightMesh);
			lightGASHandle->Build(m_Tracer2.GetContext().get(), accelOptions);
		}
		//IAS1: First
		m_Tracer2.NewIASHandle("TLAS");
		m_Tracer2.SetTLASName("TLAS");
		auto tlasHandle = m_Tracer2.GetIASHandle("TLAS");
		{
			OptixAccelBuildOptions accelOptions = {};
			accelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
			accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
			auto worldInstance = rtlib::ext::Instance();
			worldInstance.Init(m_Tracer2.GetGASHandle("world"));
			auto lightInstance = rtlib::ext::Instance();
			lightInstance.Init(m_Tracer2.GetGASHandle("light"));
			lightInstance.instance.sbtOffset = worldInstance.baseGASHandle->GetSbtCount() * RAY_TYPE_COUNT;
			tlasHandle->GetInstanceSets().resize(1);
			tlasHandle->GetInstanceSets()[0] = std::make_shared<rtlib::ext::InstanceSet>();
			tlasHandle->GetInstanceSets()[0]->SetInstance(worldInstance);
			tlasHandle->GetInstanceSets()[0]->SetInstance(lightInstance);
			tlasHandle->GetInstanceSets()[0]->instanceBuffer.Upload();
			tlasHandle->Build(m_Tracer2.GetContext().get(), accelOptions);

		}
	}
	void InitLight() {
		m_Light = ParallelLight();
		{
			auto  lightGASHandle = m_Tracer2.GetGASHandle("light");
			auto  lightMesh = lightGASHandle->GetMesh(0);
			auto  lightVertices = std::vector<float3>();
			for (auto& index : lightMesh->GetUniqueResource()->triIndBuffer) {
				lightVertices.push_back(lightMesh->GetSharedResource()->vertexBuffer[index.x]);
				lightVertices.push_back(lightMesh->GetSharedResource()->vertexBuffer[index.y]);
				lightVertices.push_back(lightMesh->GetSharedResource()->vertexBuffer[index.z]);
			}
			auto lightAABB = rtlib::utils::AABB(lightVertices);
			auto lightV3 = lightAABB.max - lightAABB.min;
			m_Light.corner = lightAABB.min;
			m_Light.v1 = make_float3(0.0f, 0.0f, lightV3.z);
			m_Light.v2 = make_float3(lightV3.x, 0.0f, 0.0f);
			m_Light.normal = make_float3(0.0f, -1.0f, 0.0f);
			auto lightMaterial = (*m_MaterialSet)[lightMesh->GetUniqueResource()->materials[0]];
			m_Light.emission   = lightMaterial.GetFloat3As<float3>("emitCol");
		}
	}
	void InitFrameResources() {
		RTLIB_CUDA_CHECK(cudaStreamCreate(&m_Stream));
		m_FrameBuffer = rtlib::CUDABuffer<uchar4>(std::vector<uchar4>(m_FbWidth * m_FbHeight));
		m_DebugBuffers["Diffuse"] = rtlib::CUDABuffer<uchar4>(std::vector<uchar4>(m_FbWidth * m_FbHeight));
		m_DebugBuffers["Specular"] = rtlib::CUDABuffer<uchar4>(std::vector<uchar4>(m_FbWidth * m_FbHeight));
		m_DebugBuffers["Transmit"] = rtlib::CUDABuffer<uchar4>(std::vector<uchar4>(m_FbWidth * m_FbHeight));
		m_DebugBuffers["Emission"] = rtlib::CUDABuffer<uchar4>(std::vector<uchar4>(m_FbWidth * m_FbHeight));
		m_DebugBuffers["TexCoord"] = rtlib::CUDABuffer<uchar4>(std::vector<uchar4>(m_FbWidth * m_FbHeight));
		m_DebugBuffers["Normal"] = rtlib::CUDABuffer<uchar4>(std::vector<uchar4>(m_FbWidth * m_FbHeight));
		m_DebugBuffers["Depth"] = rtlib::CUDABuffer<uchar4>(std::vector<uchar4>(m_FbWidth * m_FbHeight));

		m_FrameBufferGL = rtlib::GLInteropBuffer<uchar4>(m_FbWidth * m_FbHeight, GL_PIXEL_UNPACK_BUFFER, GL_DYNAMIC_DRAW, m_Stream);
		m_DebugBufferGLs["Diffuse"] = rtlib::GLInteropBuffer<uchar4>(m_FbWidth * m_FbHeight, GL_PIXEL_UNPACK_BUFFER, GL_DYNAMIC_DRAW, m_Stream);
		m_DebugBufferGLs["Specular"] = rtlib::GLInteropBuffer<uchar4>(m_FbWidth * m_FbHeight, GL_PIXEL_UNPACK_BUFFER, GL_DYNAMIC_DRAW, m_Stream);
		m_DebugBufferGLs["Transmit"] = rtlib::GLInteropBuffer<uchar4>(m_FbWidth * m_FbHeight, GL_PIXEL_UNPACK_BUFFER, GL_DYNAMIC_DRAW, m_Stream);
		m_DebugBufferGLs["Emission"] = rtlib::GLInteropBuffer<uchar4>(m_FbWidth * m_FbHeight, GL_PIXEL_UNPACK_BUFFER, GL_DYNAMIC_DRAW, m_Stream);
		m_DebugBufferGLs["TexCoord"] = rtlib::GLInteropBuffer<uchar4>(m_FbWidth * m_FbHeight, GL_PIXEL_UNPACK_BUFFER, GL_DYNAMIC_DRAW, m_Stream);
		m_DebugBufferGLs["Normal"] = rtlib::GLInteropBuffer<uchar4>(m_FbWidth * m_FbHeight, GL_PIXEL_UNPACK_BUFFER, GL_DYNAMIC_DRAW, m_Stream);
		m_DebugBufferGLs["Depth"] = rtlib::GLInteropBuffer<uchar4>(m_FbWidth * m_FbHeight, GL_PIXEL_UNPACK_BUFFER, GL_DYNAMIC_DRAW, m_Stream);
		m_AccumBuffer = rtlib::CUDABuffer<float3>(std::vector<float3>(m_FbWidth * m_FbHeight));
		m_SeedBuffer = rtlib::CUDABuffer<unsigned int>();
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
		auto tracePipeline2= std::make_shared<test::RTPipeline<RayGenData,MissData,HitgroupData,RayTraceParams>>();
		{
			OptixPipelineCompileOptions compileOptions = {};

			compileOptions.pipelineLaunchParamsVariableName = "params";
			compileOptions.numAttributeValues = 3;
			compileOptions.numPayloadValues = 8;
			compileOptions.usesPrimitiveTypeFlags = 0;
			compileOptions.usesMotionBlur = false;
			compileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;

			OptixPipelineLinkOptions linkOptions = {};

			linkOptions.maxTraceDepth = 2;
#ifndef NDEBUG
			linkOptions.debugLevel = static_cast<OptixCompileDebugLevel>(rtlib::OPXCompileDebugLevel::Minimal);
#else
			linkOptions.debugLevel = static_cast<OptixCompileDebugLevel>(rtlib::OPXCompileDebugLevel::None);
#endif
			
			tracePipeline2->Init(m_Tracer2.GetContext(),compileOptions,linkOptions);
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
			tracePipeline2->LoadModuleFromPtxFile("RayTrace", TEST_TEST19_CUDA_PATH"/RayTrace.ptx", moduleCompileOptions);
		}
		//program group: init
		{
			tracePipeline2->LoadRayGProgramGroupFromModule("Default"   , { "RayTrace", "__raygen__rg" });
			tracePipeline2->LoadMissProgramGroupFromModule("Radiance"  , { "RayTrace" , "__miss__radiance" });
			tracePipeline2->LoadMissProgramGroupFromModule("Occlusion" , { "RayTrace" , "__miss__occluded" });
			tracePipeline2->LoadHitGProgramGroupFromModule("DiffuseDef", { "RayTrace" ,"__closesthit__radiance_for_diffuse_def" }, {}, {});
			tracePipeline2->LoadHitGProgramGroupFromModule("DiffuseNee", { "RayTrace" ,"__closesthit__radiance_for_diffuse_nee" }, {}, {});
			tracePipeline2->LoadHitGProgramGroupFromModule("Specular"  , { "RayTrace" ,"__closesthit__radiance_for_specular" }, {}, {});
			tracePipeline2->LoadHitGProgramGroupFromModule("Refraction", { "RayTrace" ,"__closesthit__radiance_for_refraction" }, {}, {});
			tracePipeline2->LoadHitGProgramGroupFromModule("Emission"  , { "RayTrace" ,"__closesthit__radiance_for_emission" }, {}, {});
			tracePipeline2->LoadHitGProgramGroupFromModule("Occlusion" , { "RayTrace" ,"__closesthit__occluded" }, {}, {});
			tracePipeline2->Link();
		}
		//SBTRecord
		{
			{
				auto camera = m_CameraController.GetCamera(m_CameraFovY, m_FbAspect);
				auto [u, v, w] = camera.getUVW();
				RayGenData rayGData = {};
				rayGData.eye = camera.getEye();
				rayGData.u   = u;
				rayGData.v   = v;
				rayGData.w   = w;
				tracePipeline2->NewRayGRecordBuffer("Def");
				tracePipeline2->AddRayGRecordFromPG("Def","Default",rayGData);
				tracePipeline2->GetRayGRecordBuffer("Def")->Upload();
			}
			{
				tracePipeline2->NewMissRecordBuffer("Def", RAY_TYPE_COUNT);
				tracePipeline2->AddMissRecordFromPG("Def", RAY_TYPE_RADIANCE , "Radiance" , {make_float4(0.0f, 0.0f, 0.0f, 0.0f)});
				tracePipeline2->AddMissRecordFromPG("Def", RAY_TYPE_OCCLUSION, "Occlusion", {make_float4(0.0f, 0.0f, 0.0f, 0.0f)});
				tracePipeline2->GetMissRecordBuffer("Def")->Upload();
			}
			{
				tracePipeline2->NewHitGRecordBuffer("Def", RAY_TYPE_COUNT * m_Tracer2.GetTLAS()->GetSbtCount());
				{
					size_t sbtOffset = 0;
					for (auto& instanceSet : m_Tracer2.GetTLAS()->GetInstanceSets()) {
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
										radianceHgData.texCoords = cudaTexCrdBuffer->GetHandle().getDevicePtr();
										radianceHgData.indices = cudaTriIndBuffer->GetHandle().getDevicePtr();
										radianceHgData.diffuseTex  = m_Tracer2.GetTexture(material.GetString("diffTex")).getHandle();
										radianceHgData.specularTex = m_Tracer2.GetTexture(material.GetString("specTex")).getHandle();
										radianceHgData.emissionTex = m_Tracer2.GetTexture(material.GetString("emitTex")).getHandle();
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
									if (typeString == "Diffuse") {
										typeString += "Def";
									}
									tracePipeline2->AddHitGRecordFromPG("Def", RAY_TYPE_COUNT * sbtOffset + RAY_TYPE_COUNT * i + RAY_TYPE_RADIANCE , typeString ,  radianceHgData);
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
				tracePipeline2->NewHitGRecordBuffer("Nee", RAY_TYPE_COUNT * m_Tracer2.GetTLAS()->GetSbtCount());
				{
					size_t sbtOffset = 0;
					for (auto& instanceSet : m_Tracer2.GetTLAS()->GetInstanceSets()) {
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
										radianceHgData.texCoords = cudaTexCrdBuffer->GetHandle().getDevicePtr();
										radianceHgData.indices = cudaTriIndBuffer->GetHandle().getDevicePtr();
										radianceHgData.diffuseTex  = m_Tracer2.GetTexture(material.GetString("diffTex")).getHandle();
										radianceHgData.specularTex = m_Tracer2.GetTexture(material.GetString("specTex")).getHandle();
										radianceHgData.emissionTex = m_Tracer2.GetTexture(material.GetString("emitTex")).getHandle();
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
									if (typeString == "Diffuse") {
										typeString += "Nee";
									}
									tracePipeline2->AddHitGRecordFromPG("Nee", RAY_TYPE_COUNT * sbtOffset + RAY_TYPE_COUNT * i + RAY_TYPE_RADIANCE , typeString ,  radianceHgData);
									tracePipeline2->AddHitGRecordFromPG("Nee", RAY_TYPE_COUNT * sbtOffset + RAY_TYPE_COUNT * i + RAY_TYPE_OCCLUSION, "Occlusion", {});

								}
								sbtOffset += mesh->GetUniqueResource()->materials.size();
							}
						}
					}
				}
				tracePipeline2->GetHitGRecordBuffer("Nee")->Upload();
			}
			{
				RayTraceParams params = {};
				params.frameBuffer = m_FrameBuffer.getDevicePtr();
				params.accumBuffer = m_AccumBuffer.getDevicePtr();
				params.seedBuffer = m_SeedBuffer.getDevicePtr();
				params.width = m_FbWidth;
				params.height = m_FbHeight;
				params.maxTraceDepth = kDefaultMaxTraceDepth;
				params.gasHandle = m_Tracer2.GetTLAS()->GetHandle();
				params.light = m_Light;
				params.samplePerALL = 0;
				params.samplePerLaunch = kDefaultSamplePerLaunch;

				tracePipeline2->NewSubPass("Def");
				tracePipeline2->AddRayGRecordBufferToSubPass("Def", "Def");
				tracePipeline2->AddMissRecordBufferToSubPass("Def", "Def");
				tracePipeline2->AddHitGRecordBufferToSubPass("Def", "Def");
				tracePipeline2->GetSubPass("Def")->InitShaderBindingTable();
				tracePipeline2->GetSubPass("Def")->InitParams(params);
				tracePipeline2->GetSubPass("Def")->SetTraceCallDepth(1);
			}
			{
				RayTraceParams params = {};
				params.frameBuffer = m_FrameBuffer.getDevicePtr();
				params.accumBuffer = m_AccumBuffer.getDevicePtr();
				params.seedBuffer = m_SeedBuffer.getDevicePtr();
				params.width = m_FbWidth;
				params.height = m_FbHeight;
				params.maxTraceDepth = kDefaultMaxTraceDepth;
				params.gasHandle = m_Tracer2.GetTLAS()->GetHandle();
				params.light = m_Light;
				params.samplePerALL = 0;
				params.samplePerLaunch = kDefaultSamplePerLaunch;

				tracePipeline2->NewSubPass("Nee");
				tracePipeline2->NewSubPass("Nee");
				tracePipeline2->AddRayGRecordBufferToSubPass("Nee", "Def");
				tracePipeline2->AddMissRecordBufferToSubPass("Nee", "Def");
				tracePipeline2->AddHitGRecordBufferToSubPass("Nee", "Nee");
				tracePipeline2->GetSubPass("Nee")->InitShaderBindingTable();
				tracePipeline2->GetSubPass("Nee")->InitParams(params);
				tracePipeline2->GetSubPass("Nee")->SetTraceCallDepth(2);
			}
		}
		m_Tracer2.SetTracePipeline(tracePipeline2);

	}
	void InitDebugPipeline() {
		auto debugPipeline2 = std::make_shared<test::RTPipeline<RayGenData, MissData, HitgroupData, RayDebugParams>>();
		{
			OptixPipelineCompileOptions compileOptions = {};

			compileOptions.pipelineLaunchParamsVariableName = "params";
			compileOptions.numAttributeValues = 3;
			compileOptions.numPayloadValues = 8;
			compileOptions.usesPrimitiveTypeFlags = 0;
			compileOptions.usesMotionBlur = false;
			compileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;

			OptixPipelineLinkOptions linkOptions = {};

			linkOptions.maxTraceDepth = 2;
#ifndef NDEBUG
			linkOptions.debugLevel = static_cast<OptixCompileDebugLevel>(rtlib::OPXCompileDebugLevel::Minimal);
#else
			linkOptions.debugLevel = static_cast<OptixCompileDebugLevel>(rtlib::OPXCompileDebugLevel::None);
#endif

			debugPipeline2->Init(m_Tracer2.GetContext(), compileOptions, linkOptions);
		}
		//module: Load
		{
			OptixModuleCompileOptions moduleCompileOptions = {};
			moduleCompileOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
			moduleCompileOptions.numBoundValues   = 0;
#ifndef NDEBUG
			moduleCompileOptions.debugLevel = static_cast<OptixCompileDebugLevel>(rtlib::OPXCompileDebugLevel::Minimal);
			moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
#else
			moduleCompileOptions.optLevel   = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
			moduleCompileOptions.debugLevel = static_cast<OptixCompileDebugLevel>(rtlib::OPXCompileDebugLevel::None);
#endif
			debugPipeline2->LoadModuleFromPtxFile("RayDebug", TEST_TEST19_CUDA_PATH"/RayDebug.ptx", moduleCompileOptions);
		}
		//program group: init
		{
			debugPipeline2->LoadRayGProgramGroupFromModule("Default"  , { "RayDebug" , "__raygen__debug" });
			debugPipeline2->LoadMissProgramGroupFromModule("Radiance" , { "RayDebug" , "__miss__debug" });
			debugPipeline2->LoadMissProgramGroupFromModule("Occlusion", { "RayDebug" , "__miss__debug" });
			debugPipeline2->LoadHitGProgramGroupFromModule("Radiance" , { "RayDebug" , "__closesthit__debug" }, {}, {});
			debugPipeline2->LoadHitGProgramGroupFromModule("Occlusion", { "RayDebug" , "__closesthit__debug" }, {}, {});
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
				debugPipeline2->NewHitGRecordBuffer("Def", RAY_TYPE_COUNT * m_Tracer2.GetTLAS()->GetSbtCount());
				{
					size_t sbtOffset = 0;
					for (auto& instanceSet : m_Tracer2.GetTLAS()->GetInstanceSets()) {
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
										radianceHgData.texCoords = cudaTexCrdBuffer->GetHandle().getDevicePtr();
										radianceHgData.indices = cudaTriIndBuffer->GetHandle().getDevicePtr();
										radianceHgData.diffuseTex  = m_Tracer2.GetTexture(material.GetString("diffTex")).getHandle();
										radianceHgData.specularTex = m_Tracer2.GetTexture(material.GetString("specTex")).getHandle();
										radianceHgData.emissionTex = m_Tracer2.GetTexture(material.GetString("emitTex")).getHandle();
										radianceHgData.diffuse = material.GetFloat3As<float3>("diffCol");
										radianceHgData.specular = material.GetFloat3As<float3>("specCol");
										radianceHgData.emission = material.GetFloat3As<float3>("emitCol");
										radianceHgData.shinness = material.GetFloat1("shinness");
										radianceHgData.transmit = material.GetFloat3As<float3>("tranCol");
										radianceHgData.refrInd = material.GetFloat1("refrIndx");
									}
									if (material.GetString("name") == "light") {
										m_LightHgRecIndex = RAY_TYPE_COUNT * sbtOffset + RAY_TYPE_COUNT * i + RAY_TYPE_RADIANCE;
									}
									debugPipeline2->AddHitGRecordFromPG("Def", RAY_TYPE_COUNT * sbtOffset + RAY_TYPE_COUNT * i + RAY_TYPE_RADIANCE,  "Radiance" , radianceHgData);
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
				params.width     = m_FbWidth;
				params.height    = m_FbHeight;
				params.gasHandle = m_Tracer2.GetTLAS()->GetHandle();
				params.light     = m_Light;

				debugPipeline2->NewSubPass("Def");
				debugPipeline2->AddRayGRecordBufferToSubPass("Def", "Def");
				debugPipeline2->AddMissRecordBufferToSubPass("Def", "Def");
				debugPipeline2->AddHitGRecordBufferToSubPass("Def", "Def");
				debugPipeline2->GetSubPass("Def")->InitShaderBindingTable();
				debugPipeline2->GetSubPass("Def")->InitParams(params);
				debugPipeline2->GetSubPass("Def")->SetTraceCallDepth(1);
			}
		}
		m_Tracer2.SetDebugPipeline(debugPipeline2);
	}
	void MainLoop() {
		PrepareMainLoop();
		while (!glfwWindowShouldClose(m_Window)) {
			{
				if (m_Resized) {
					this->OnResize();
					m_Resized = false;
					m_UpdateCamera = true;
					//Resize��Flush����K�v������
					m_FlushFrame = true;
				}
				if (m_UpdateCamera) {
					this->OnUpdateCamera();
					m_UpdateCamera = false;
					//Camera�̈ړ���Flush����K�v������
					m_FlushFrame = true;
				}
				if (m_UpdateLight) {
					this->OnUpdateLight();
					m_UpdateLight = false;
					m_FlushFrame = true;
				}
				if (m_FlushFrame) {
					this->OnFlushFrame();
					m_FlushFrame = false;
					//Flush�����ꍇ�AParam�̍Đݒ肪�K�v
					m_UpdateParams = true;
				}
				if (m_UpdateParams) {
					this->OnUpdateParams();
					m_UpdateParams = false;
				}
				this->OnLaunch();
				this->OnRenderFrame();
				this->OnRenderImGui();
				this->OnUpdateTime();
				this->OnReactInputs();
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

		m_FrameBuffer.resize(m_FbWidth * m_FbHeight);
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
		m_Tracer2.GetTracePipeline()->GetSubPass("Def")->GetParams().accumBuffer = m_AccumBuffer.getDevicePtr();
		m_Tracer2.GetTracePipeline()->GetSubPass("Def")->GetParams().seedBuffer        = m_SeedBuffer.getDevicePtr();
		m_Tracer2.GetTracePipeline()->GetSubPass("Def")->GetParams().width       = m_FbWidth;
		m_Tracer2.GetTracePipeline()->GetSubPass("Def")->GetParams().height      = m_FbHeight;
		m_Tracer2.GetTracePipeline()->GetSubPass("Nee")->GetParams().accumBuffer = m_AccumBuffer.getDevicePtr();
		m_Tracer2.GetTracePipeline()->GetSubPass("Nee")->GetParams().seedBuffer   = m_SeedBuffer.getDevicePtr();
		m_Tracer2.GetTracePipeline()->GetSubPass("Nee")->GetParams().width  = m_FbWidth;
		m_Tracer2.GetTracePipeline()->GetSubPass("Nee")->GetParams().height = m_FbHeight;
		m_Tracer2.GetDebugPipeline()->GetSubPass("Def")->GetParams().width  = m_FbWidth;
		m_Tracer2.GetDebugPipeline()->GetSubPass("Def")->GetParams().height = m_FbHeight;
	}
	void OnUpdateCamera()
	{
		auto camera = m_CameraController.GetCamera(m_CameraFovY, m_FbAspect);
		auto [u, v, w] = camera.getUVW();
		m_Tracer2.GetTracePipeline()->GetRayGRecordBuffer("Def")->GetData().eye = camera.getEye();
		m_Tracer2.GetTracePipeline()->GetRayGRecordBuffer("Def")->GetData().u = u;
		m_Tracer2.GetTracePipeline()->GetRayGRecordBuffer("Def")->GetData().v = v;
		m_Tracer2.GetTracePipeline()->GetRayGRecordBuffer("Def")->GetData().w = w;
		m_Tracer2.GetTracePipeline()->GetRayGRecordBuffer("Def")->Upload();

		m_Tracer2.GetDebugPipeline()->GetRayGRecordBuffer("Def")->GetData().eye = camera.getEye();
		m_Tracer2.GetDebugPipeline()->GetRayGRecordBuffer("Def")->GetData().u = u;
		m_Tracer2.GetDebugPipeline()->GetRayGRecordBuffer("Def")->GetData().v = v;
		m_Tracer2.GetDebugPipeline()->GetRayGRecordBuffer("Def")->GetData().w = w;
		m_Tracer2.GetDebugPipeline()->GetRayGRecordBuffer("Def")->Upload();
	}
	void OnUpdateLight() {
		m_Tracer2.GetTracePipeline()->GetHitGRecordBuffer("Def")->GetData(m_LightHgRecIndex).emission = m_Light.emission;
		m_Tracer2.GetTracePipeline()->GetHitGRecordBuffer("Def")->GetData(m_LightHgRecIndex).diffuse  = m_Light.emission;
		m_Tracer2.GetTracePipeline()->GetHitGRecordBuffer("Def")->Upload();

		m_Tracer2.GetTracePipeline()->GetHitGRecordBuffer("Nee")->GetData(m_LightHgRecIndex).emission = m_Light.emission;
		m_Tracer2.GetTracePipeline()->GetHitGRecordBuffer("Nee")->GetData(m_LightHgRecIndex).diffuse  = m_Light.emission;
		m_Tracer2.GetTracePipeline()->GetHitGRecordBuffer("Nee")->Upload();

		m_Tracer2.GetDebugPipeline()->GetHitGRecordBuffer("Def")->GetData(m_LightHgRecIndex).emission = m_Light.emission;
		m_Tracer2.GetDebugPipeline()->GetHitGRecordBuffer("Def")->GetData(m_LightHgRecIndex).diffuse = m_Light.emission;
		m_Tracer2.GetDebugPipeline()->GetHitGRecordBuffer("Def")->Upload();
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
			auto& params        = m_Tracer2.GetTracePipeline()->GetSubPass(m_CurSubPassName)->GetParams();
			params.frameBuffer  = m_FrameBuffer.getDevicePtr();
			params.accumBuffer  = m_AccumBuffer.getDevicePtr();
			params.seedBuffer         = m_SeedBuffer.getDevicePtr();
			params.width        = m_FbWidth;
			params.height       = m_FbHeight;
			//params.maxTraceDepth   = m_MaxTraceDepth;
			params.gasHandle    = m_Tracer2.GetTLAS()->GetHandle();
			params.light        = m_Light;
			params.samplePerALL = m_SamplePerAll;
		}
		else {
			auto& params          = m_Tracer2.GetDebugPipeline()->GetSubPass("Def")->GetParams();
			params.diffuseBuffer  = m_DebugBuffers["Diffuse"].getDevicePtr();
			params.specularBuffer = m_DebugBuffers["Specular"].getDevicePtr();
			params.emissionBuffer = m_DebugBuffers["Emission"].getDevicePtr();
			params.transmitBuffer = m_DebugBuffers["Transmit"].getDevicePtr();
			params.texCoordBuffer = m_DebugBuffers["TexCoord"].getDevicePtr();
			params.normalBuffer   = m_DebugBuffers["Normal"].getDevicePtr();
			params.depthBuffer    = m_DebugBuffers["Depth"].getDevicePtr();
			params.width          = m_FbWidth;
			params.height         = m_FbHeight;
			params.gasHandle      = m_Tracer2.GetTLAS()->GetHandle();
			params.light          = m_Light;
		}
	}
	void OnUpdateTime() {
		float prevTime = glfwGetTime();
		m_DelTime = prevTime - m_CurTime;
		m_CurTime = prevTime;
	}
	void OnLaunch() {
		if (m_CurPipelineName == "Trace") {
			auto& curPipeline   = m_Tracer2.GetTracePipeline();
			auto& params        = curPipeline->GetSubPass(m_CurSubPassName)->GetParams();
			params.frameBuffer  = m_FrameBufferGL.map();
			params.samplePerALL = m_SamplePerAll;
			curPipeline->Launch(m_FbWidth, m_FbHeight, m_CurSubPassName, m_Stream);
			m_FrameBufferGL.unmap();
			m_SamplePerAll     += params.samplePerLaunch;
		}
		else {
			auto& curPipeline     = m_Tracer2.GetDebugPipeline();
			auto& params          = curPipeline->GetSubPass("Def")->GetParams();
			params.diffuseBuffer  = m_DebugBufferGLs["Diffuse"].map();
			params.specularBuffer = m_DebugBufferGLs["Specular"].map();
			params.emissionBuffer = m_DebugBufferGLs["Emission"].map();
			params.transmitBuffer = m_DebugBufferGLs["Transmit"].map();
			params.texCoordBuffer = m_DebugBufferGLs["TexCoord"].map();
			params.normalBuffer   = m_DebugBufferGLs["Normal"].map();
			params.depthBuffer    = m_DebugBufferGLs["Depth"].map();
			curPipeline->Launch(m_FbWidth, m_FbHeight, "Def", m_Stream);
			m_DebugBufferGLs["Diffuse"].unmap();
			m_DebugBufferGLs["Specular"].unmap();
			m_DebugBufferGLs["Emission"].unmap();
			m_DebugBufferGLs["Transmit"].unmap();
			m_DebugBufferGLs["TexCoord"].unmap();
			m_DebugBufferGLs["Normal"].unmap();
			m_DebugBufferGLs["Depth"].unmap();
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
				ImGui::SetNextWindowSize(ImVec2(400, 300), ImGuiCond_Once);

				ImGui::Begin("TraceConfig", nullptr, ImGuiWindowFlags_MenuBar);

				ImGui::BeginChild(ImGui::GetID((void*)0), ImVec2(350, 200), ImGuiWindowFlags_NoTitleBar);
				ImGui::Text("Fps: %.2f", 1.0f / m_DelTime);
				if (m_CurPipelineName == "Trace")
				{
					auto  curPipeline = m_Tracer2.GetTracePipeline();
					auto  curSubpass  = curPipeline->GetSubPass(m_CurSubPassName);
					auto& curParams   = curSubpass->GetParams();
					ImGui::Text("Smp  : %3d", curParams.samplePerALL);
					ImGui::Text("Smp/s: %.2f", curParams.samplePerLaunch / m_DelTime);
					{
						int samplePerLaunch = curParams.samplePerLaunch;
						if (ImGui::SliderInt("samplePerLaunch", &samplePerLaunch, 1, 10)) {
							curParams.samplePerLaunch = samplePerLaunch;
						}
					}
					{
						int maxTraceDepth = curParams.maxTraceDepth;
						if (ImGui::SliderInt("maxTraceDepth", &maxTraceDepth, 1, 10)) {
							curParams.maxTraceDepth = maxTraceDepth;
							m_FlushFrame = true;
						}
					}
				}
				else {
					static int mode = 0;
					ImGui::RadioButton("Diffuse", &mode, 0);
					ImGui::SameLine();
					ImGui::RadioButton("Specular", &mode, 1);
					ImGui::SameLine();
					ImGui::RadioButton("Transmit", &mode, 2);
					ImGui::SameLine();
					ImGui::RadioButton("Emission", &mode, 3);
					ImGui::NewLine();
					ImGui::RadioButton("TexCoord", &mode, 4);
					ImGui::SameLine();
					ImGui::RadioButton("Normal", &mode, 5);
					ImGui::SameLine();
					ImGui::RadioButton("Depth", &mode, 6);
					switch (mode) {
					case 0:
						m_DebugFrameName = "Diffuse";
						break;
					case 1:
						m_DebugFrameName = "Specular";
						break;
					case 2:
						m_DebugFrameName = "Transmit";
						break;
					case 3:
						m_DebugFrameName = "Emission";
						break;
					case 4:
						m_DebugFrameName = "TexCoord";
						break;
					case 5:
						m_DebugFrameName = "Normal";
						break;
					case 6:
						m_DebugFrameName = "Depth";
						break;
					}
				}
				//Camera
				{

					{
						float camFovY = m_CameraFovY;
						if (ImGui::SliderFloat("Camera.FovY", &camFovY, -90.0f, 90.0f)) {
							m_CameraFovY = camFovY;
							m_UpdateCamera = true;
						}
					}
				}
				//Light
				{
					float emission[3] = { m_Light.emission.x, m_Light.emission.y, m_Light.emission.z };
					if (ImGui::SliderFloat3("light.Color", emission, 0.0f, 10.0f)) {
						m_Light.emission.x = emission[0];
						m_Light.emission.y = emission[1];
						m_Light.emission.z = emission[2];
						m_UpdateLight = true;
					}
				}
				if (ImGui::Button(m_PrvPipelineName.c_str())) {
					std::swap(m_CurPipelineName, m_PrvPipelineName);
					m_UpdateCamera = true;
				}
				if (m_CurPipelineName == "Trace") {
					ImGui::SameLine();
					if (ImGui::Button(m_PrvSubPassName.c_str())) {
						std::swap(m_CurSubPassName, m_PrvSubPassName);
						m_FlushFrame = true;
					}
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
	void OnReactInputs() {
		if (glfwGetKey(m_Window, GLFW_KEY_W) == GLFW_PRESS) {
			m_CameraController.ProcessKeyboard(rtlib::ext::CameraMovement::eForward, m_DelTime);
			m_UpdateCamera = true;
		}
		if (glfwGetKey(m_Window, GLFW_KEY_S) == GLFW_PRESS) {
			m_CameraController.ProcessKeyboard(rtlib::ext::CameraMovement::eBackward, m_DelTime);
			m_UpdateCamera = true;
		}
		if (glfwGetKey(m_Window, GLFW_KEY_A) == GLFW_PRESS) {
			m_CameraController.ProcessKeyboard(rtlib::ext::CameraMovement::eLeft, m_DelTime);
			m_UpdateCamera = true;
		}
		if (glfwGetKey(m_Window, GLFW_KEY_D) == GLFW_PRESS) {
			m_CameraController.ProcessKeyboard(rtlib::ext::CameraMovement::eRight, m_DelTime);
			m_UpdateCamera = true;
		}
		if (glfwGetKey(m_Window, GLFW_KEY_LEFT) == GLFW_PRESS) {
			m_CameraController.ProcessKeyboard(rtlib::ext::CameraMovement::eLeft, m_DelTime);
			m_UpdateCamera = true;
		}
		if (glfwGetKey(m_Window, GLFW_KEY_RIGHT) == GLFW_PRESS) {
			m_CameraController.ProcessKeyboard(rtlib::ext::CameraMovement::eRight, m_DelTime);
			m_UpdateCamera = true;
		}
		if (glfwGetKey(m_Window, GLFW_KEY_UP) == GLFW_PRESS) {
			m_CameraController.ProcessKeyboard(rtlib::ext::CameraMovement::eUp, m_DelTime);
			m_UpdateCamera = true;
		}
		if (glfwGetKey(m_Window, GLFW_KEY_DOWN) == GLFW_PRESS) {
			m_CameraController.ProcessKeyboard(rtlib::ext::CameraMovement::eDown, m_DelTime);
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
		Test19Application* app = reinterpret_cast<Test19Application*>(glfwGetWindowUserPointer(window));
		if (app) {
			if (fbWidth != app->m_FbWidth || fbHeight != app->m_FbHeight)
			{
				app->m_FbWidth = fbWidth;
				app->m_FbHeight = fbHeight;
				app->m_FbAspect = static_cast<float>(fbWidth) / static_cast<float>(fbHeight);
				app->m_Resized = true;
			}
		}
	}
	static void cursorPosCallback(GLFWwindow* window, double xPos, double yPos)
	{
		Test19Application* app = reinterpret_cast<Test19Application*>(glfwGetWindowUserPointer(window));
		if (app) {
			app->m_DelCursorPos.x = xPos - app->m_CurCursorPos.x;
			app->m_DelCursorPos.y = yPos - app->m_CurCursorPos.y;;
			app->m_CurCursorPos.x = xPos;
			app->m_CurCursorPos.y = yPos;
		}
	}
private:
	using CUDABufferMap = std::unordered_map<std::string, rtlib::CUDABuffer<uchar4>>;
	using GLInteropBufferMap = std::unordered_map<std::string, rtlib::GLInteropBuffer<uchar4>>;
private:
	GLFWwindow* m_Window = nullptr;
	int			      	            m_FbWidth = 0;
	int                             m_FbHeight = 0;
	float                           m_FbAspect = 1.0f;
	std::string                     m_Title = {};
	std::string                     m_GlslVersion = {};
	//Inputs
	float2                          m_DelCursorPos = {};
	float2                          m_CurCursorPos = {};
	float                           m_CurTime = 0.0f;
	float                           m_DelTime = 0.0f;
	float                           m_CameraFovY = 30.0f;
	//State
	bool                            m_Resized = false;
	bool                            m_UpdateCamera = false;
	bool                            m_UpdateLight = false;

	rtlib::GLTexture2D<uchar4>      m_GLTexture = {};
	std::shared_ptr<rtlib::ext::RectRenderer>
		m_RectRenderer = {};
	rtlib::ext::CameraController         m_CameraController = {};
	test::RTTracer                  m_Tracer2 = {};
	rtlib::ext::VariableMapListPtr     m_MaterialSet = nullptr;
	CUstream                        m_Stream = nullptr;
	//Trace
	rtlib::CUDABuffer<uchar4>       m_FrameBuffer   = {};
	rtlib::GLInteropBuffer<uchar4>  m_FrameBufferGL = {};
	rtlib::CUDABuffer<float3>       m_AccumBuffer   = {};
	rtlib::CUDABuffer<unsigned int> m_SeedBuffer    = {};
	//Debug
	CUDABufferMap                   m_DebugBuffers   = {};
	GLInteropBufferMap              m_DebugBufferGLs = {};
	//Light
	uint32_t                        m_LightHgRecIndex = 0;
	ParallelLight                   m_Light = {};
	//Config
	uint32_t                        m_SamplePerAll = 0;
	//State
	bool                            m_FlushFrame      = false;
	bool                            m_UpdateParams    = false;
	std::string                     m_CurPipelineName = "Trace";
	std::string                     m_PrvPipelineName = "Debug";
	std::string                     m_CurSubPassName  = "Def";
	std::string                     m_PrvSubPassName  = "Nee";
	std::string                     m_DebugFrameName  = "Diffuse";
};
int main() {
	Test19Application app = {};
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