#include <Test18Config.h>
#include <RTLib/core/GL.h>
#include <RTLib/core/CUDA.h>
#include <RTLib/core/CUDA_GL.h>
#include <RTLib/ext/Camera.h>
#include <RTLib/ext/Utils.h>
#include <RTLib/ext/RectRenderer.h>
#include <RTLib/ext/Resources/CUDA.h>
#include <RTLib/ext/TraversalHandle.h>
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
#define TEST18_USE_NEE
class Test18Application {
private:
	static inline constexpr uint32_t kDefaultSamplePerLaunch = 1;
	static inline constexpr uint32_t kDefaultMaxTraceDepth   = 4;
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
	void InitOptix (){
		m_Tracer.InitCUDA();
    	m_Tracer.InitOPX();
	}
	void InitCamera(){
		m_CameraController = rtlib::ext::CameraController({ 0.0f,1.0f, 5.0f });
		m_CameraController.SetMouseSensitivity(0.125f);
		m_CameraController.SetMovementSpeed(50.0f);
	}
	void LoadScene(){
		auto objMeshGroup = std::make_shared<test::ObjMeshGroup>();
		if (!objMeshGroup->Load(TEST_TEST18_DATA_PATH"/Models/Sponza/sponza.obj", TEST_TEST18_DATA_PATH"/Models/Sponza/")) {
			throw std::runtime_error("Failed To Load Model!");
		}
		m_MaterialSet = objMeshGroup->GetMaterialSet();
		objMeshGroup->GetMeshGroup()->GetSharedResource()->vertexBuffer.AddGpuComponent < rtlib::ext::resources::CUDABufferComponent <float3>> ("CUDA");
		objMeshGroup->GetMeshGroup()->GetSharedResource()->normalBuffer.AddGpuComponent < rtlib::ext::resources::CUDABufferComponent <float3>> ("CUDA");
		objMeshGroup->GetMeshGroup()->GetSharedResource()->texCrdBuffer.AddGpuComponent < rtlib::ext::resources::CUDABufferComponent <float2>> ("CUDA");
		for (auto& [name, uniqueResource] : objMeshGroup->GetMeshGroup()->GetUniqueResources()) {
			uniqueResource->triIndBuffer.AddGpuComponent< rtlib::ext::resources::CUDABufferComponent<uint3>>("CUDA");
			uniqueResource->matIndBuffer.AddGpuComponent< rtlib::ext::resources::CUDABufferComponent<uint32_t>>("CUDA");
		}
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
					worldGASHandle->AddMesh(objMeshGroup->GetMeshGroup()->LoadMesh(name));
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
			lightGASHandle->AddMesh(objMeshGroup->GetMeshGroup()->LoadMesh("light"));
			lightGASHandle->Build(m_Tracer.GetOPXContext().get(), accelOptions);
		}
		else {
			rtlib::utils::AABB aabb = {};
			for (auto& vertex : objMeshGroup->GetMeshGroup()->GetSharedResource()->vertexBuffer) {
				aabb.Update(vertex);
			}
			OptixAccelBuildOptions accelOptions = {};
			accelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
			accelOptions.operation  = OPTIX_BUILD_OPERATION_BUILD;
			auto lightMesh = rtlib::ext::Mesh::New();
			lightMesh->SetSharedResource(rtlib::ext::MeshSharedResource::New());
			lightMesh->GetSharedResource()->name = "light";
			lightMesh->GetSharedResource()->vertexBuffer = {
				{aabb.min.x,aabb.max.y+1e-3f,aabb.min.z},
				{aabb.max.x,aabb.max.y+1e-3f,aabb.min.z},
				{aabb.max.x,aabb.max.y+1e-3f,aabb.max.z},
				{aabb.min.x,aabb.max.y+1e-3f,aabb.max.z}
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
			lightMesh->GetSharedResource()->vertexBuffer.AddGpuComponent<rtlib::ext::resources::CUDABufferComponent<float3>>("CUDA");
			lightMesh->GetSharedResource()->texCrdBuffer.AddGpuComponent< rtlib::ext::resources::CUDABufferComponent<float2>>("CUDA");
			lightMesh->GetSharedResource()->normalBuffer.AddGpuComponent< rtlib::ext::resources::CUDABufferComponent<float3>>("CUDA");
			lightMesh->SetUniqueResource(rtlib::ext::MeshUniqueResource::New());
			lightMesh->GetUniqueResource()->name      = "light";
			lightMesh->GetUniqueResource()->materials = {
				curMaterialSetCount
			};
			lightMesh->GetUniqueResource()->matIndBuffer = {
				0,0,
			};
			lightMesh->GetUniqueResource()->triIndBuffer = {
				{0,1,2},
				{2,3,0}
			};

			lightMesh->GetUniqueResource()->matIndBuffer.AddGpuComponent< rtlib::ext::resources::CUDABufferComponent<uint32_t>>("CUDA");
			lightMesh->GetUniqueResource()->triIndBuffer.AddGpuComponent< rtlib::ext::resources::CUDABufferComponent<uint3>>("CUDA");
			lightGASHandle->AddMesh(lightMesh);
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
			lightInstance.instance.sbtOffset    = worldInstance.baseGASHandle->GetSbtCount() * RAY_TYPE_COUNT;
			firstIASHandle->GetInstanceSets().resize(1);
			firstIASHandle->GetInstanceSets()[0]     = std::make_shared<rtlib::ext::InstanceSet>();
			firstIASHandle->GetInstanceSets()[0]->SetInstance(worldInstance);
			firstIASHandle->GetInstanceSets()[0]->SetInstance(lightInstance);
			firstIASHandle->GetInstanceSets()[0]->instanceBuffer.Upload();
			firstIASHandle->Build(m_Tracer.GetOPXContext().get(), accelOptions);

		}
		m_Tracer.SetIASHandle("Top", firstIASHandle);
	}
	void InitLight() {
		m_Light = ParallelLight();
		{
			auto& lightGASHandle = m_Tracer.m_GASHandles["Light"];
			auto  lightMesh      = lightGASHandle->GetMesh(0);
			auto  lightVertices  = std::vector<float3>();
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
			auto lightMaterial = m_MaterialSet->materials[lightMesh->GetUniqueResource()->materials[0]];
			m_Light.emission = lightMaterial.emitCol;
		}
	}
	void InitFrameResources(){
		RTLIB_CUDA_CHECK(cudaStreamCreate(&m_Stream));
	 	m_FrameBuffer                = rtlib::CUDABuffer<uchar4>(std::vector<uchar4>(m_FbWidth * m_FbHeight));
		m_DebugBuffers["Diffuse"]    = rtlib::CUDABuffer<uchar4>(std::vector<uchar4>(m_FbWidth * m_FbHeight));
		m_DebugBuffers["Specular"]   = rtlib::CUDABuffer<uchar4>(std::vector<uchar4>(m_FbWidth * m_FbHeight));
		m_DebugBuffers["Transmit"]   = rtlib::CUDABuffer<uchar4>(std::vector<uchar4>(m_FbWidth * m_FbHeight));
		m_DebugBuffers["Emission"]   = rtlib::CUDABuffer<uchar4>(std::vector<uchar4>(m_FbWidth * m_FbHeight));
		m_DebugBuffers["TexCoord"]   = rtlib::CUDABuffer<uchar4>(std::vector<uchar4>(m_FbWidth * m_FbHeight));
		m_DebugBuffers["Normal"]     = rtlib::CUDABuffer<uchar4>(std::vector<uchar4>(m_FbWidth * m_FbHeight));
		m_DebugBuffers["Depth"]      = rtlib::CUDABuffer<uchar4>(std::vector<uchar4>(m_FbWidth * m_FbHeight));

		m_FrameBufferGL              = rtlib::GLInteropBuffer<uchar4>(m_FbWidth * m_FbHeight, GL_PIXEL_UNPACK_BUFFER, GL_DYNAMIC_DRAW, m_Stream);
		m_DebugBufferGLs["Diffuse"]  = rtlib::GLInteropBuffer<uchar4>(m_FbWidth * m_FbHeight, GL_PIXEL_UNPACK_BUFFER, GL_DYNAMIC_DRAW, m_Stream);
		m_DebugBufferGLs["Specular"] = rtlib::GLInteropBuffer<uchar4>(m_FbWidth * m_FbHeight, GL_PIXEL_UNPACK_BUFFER, GL_DYNAMIC_DRAW, m_Stream);
		m_DebugBufferGLs["Transmit"] = rtlib::GLInteropBuffer<uchar4>(m_FbWidth * m_FbHeight, GL_PIXEL_UNPACK_BUFFER, GL_DYNAMIC_DRAW, m_Stream);
		m_DebugBufferGLs["Emission"] = rtlib::GLInteropBuffer<uchar4>(m_FbWidth * m_FbHeight, GL_PIXEL_UNPACK_BUFFER, GL_DYNAMIC_DRAW, m_Stream);
		m_DebugBufferGLs["TexCoord"] = rtlib::GLInteropBuffer<uchar4>(m_FbWidth * m_FbHeight, GL_PIXEL_UNPACK_BUFFER, GL_DYNAMIC_DRAW, m_Stream);
		m_DebugBufferGLs["Normal"]   = rtlib::GLInteropBuffer<uchar4>(m_FbWidth * m_FbHeight, GL_PIXEL_UNPACK_BUFFER, GL_DYNAMIC_DRAW, m_Stream);
		m_DebugBufferGLs["Depth"]    = rtlib::GLInteropBuffer<uchar4>(m_FbWidth * m_FbHeight, GL_PIXEL_UNPACK_BUFFER, GL_DYNAMIC_DRAW, m_Stream);
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
        auto tracePipeline = std::make_shared<test::Pipeline<RayTraceParams>>();
        {
			tracePipeline->SetContext(m_Tracer.GetOPXContext());

			OptixPipelineCompileOptions compileOptions = {};

			compileOptions.pipelineLaunchParamsVariableName = "params";
			compileOptions.numAttributeValues               = 3;
			compileOptions.numPayloadValues                 = 8;
			compileOptions.usesPrimitiveTypeFlags           = 0;
			compileOptions.usesMotionBlur                   = false;
			compileOptions.traversableGraphFlags            = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;

			tracePipeline->InitPipeline(compileOptions);
        }
        {
            tracePipeline->width  = m_FbWidth;
            tracePipeline->height = m_FbHeight;
            tracePipeline->subPasses["Default"].depth = 1;
			tracePipeline->subPasses["NEE"].depth     = 2;
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
			tracePipeline->LoadModuleFromPtxFile("RayTrace", TEST_TEST18_CUDA_PATH"/RayTrace.ptx", moduleCompileOptions);
        }
        //program group: init
        {
			tracePipeline->LoadRgProgramGroupFromModule({ "RayTrace", "__raygen__rg" });
			tracePipeline->LoadMsProgramGroupFromModule("Radiance"  , { "RayTrace" , "__miss__radiance" });
			tracePipeline->LoadMsProgramGroupFromModule("Occlusion" , { "RayTrace" , "__miss__occluded" });
			tracePipeline->LoadHgProgramGroupFromModule("DiffuseDef", { "RayTrace" ,"__closesthit__radiance_for_diffuse_non_nee" }, {}, {});
			tracePipeline->LoadHgProgramGroupFromModule("DiffuseNee", { "RayTrace" ,"__closesthit__radiance_for_diffuse" }, {}, {});
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
			linkOptions.debugLevel = static_cast<OptixCompileDebugLevel>(rtlib::OPXCompileDebugLevel::Minimal);
#else
			linkOptions.debugLevel = static_cast<OptixCompileDebugLevel>(rtlib::OPXCompileDebugLevel::None);
#endif

			tracePipeline->LinkPipeline(linkOptions);
        }
        //SBTRecord
        {
			{
				auto& rayGBuffer = tracePipeline->rayGBuffers["Default"];
				rayGBuffer.cpuHandle.resize(1);
				auto camera = m_CameraController.GetCamera(m_CameraFovY, m_FbAspect);
				auto [u, v, w] = camera.getUVW();
				rayGBuffer.cpuHandle[0] = tracePipeline->rgProgramGroup.getSBTRecord<RayGenData>();
				rayGBuffer.cpuHandle[0].data.eye = camera.getEye();
				rayGBuffer.cpuHandle[0].data.u = u;
				rayGBuffer.cpuHandle[0].data.v = v;
				rayGBuffer.cpuHandle[0].data.w = w;
				rayGBuffer.Upload();
			}
			{
				auto& missBuffer = tracePipeline->missBuffers["Default"];
				missBuffer.cpuHandle.resize(RAY_TYPE_COUNT);
				missBuffer.cpuHandle[RAY_TYPE_RADIANCE] = tracePipeline->msProgramGroups["Radiance"].getSBTRecord<MissData>();
				missBuffer.cpuHandle[RAY_TYPE_RADIANCE].data.bgColor = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
				missBuffer.cpuHandle[RAY_TYPE_OCCLUSION] = tracePipeline->msProgramGroups["Occlusion"].getSBTRecord<MissData>();
				missBuffer.cpuHandle[RAY_TYPE_OCCLUSION].data.bgColor = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
				missBuffer.Upload();
			}
			{
				auto& defHitGBuffer = tracePipeline->hitGBuffers["Default"];
				defHitGBuffer.cpuHandle.resize(RAY_TYPE_COUNT * m_Tracer.m_IASHandles["Top"]->GetSbtCount());
				auto& cpuHgRecords = defHitGBuffer.cpuHandle;
				{
					auto& iasHandle = m_Tracer.m_IASHandles["Top"];
					size_t sbtOffset = 0;
					for (auto& instanceSet : iasHandle->GetInstanceSets()) {
						for (auto& baseGASHandle : instanceSet->baseGASHandles) {
							for (auto& mesh : baseGASHandle->GetMeshes()) {
								for (size_t i = 0; i < mesh->GetUniqueResource()->materials.size(); ++i) {
									auto materialId = mesh->GetUniqueResource()->materials[i];
									auto& material = m_MaterialSet->materials[materialId];
									HitgroupData radianceHgData = {};
									{
										radianceHgData.vertices = mesh->GetSharedResource()->vertexBuffer.GetGpuComponent<rtlib::ext::resources::CUDABufferComponent<float3>>("CUDA")->GetHandle().getDevicePtr();
										radianceHgData.indices = mesh->GetUniqueResource()->triIndBuffer.GetGpuComponent<rtlib::ext::resources::CUDABufferComponent<uint3>>("CUDA")->GetHandle().getDevicePtr();
										radianceHgData.texCoords = mesh->GetSharedResource()->texCrdBuffer.GetGpuComponent<rtlib::ext::resources::CUDABufferComponent<float2>>("CUDA")->GetHandle().getDevicePtr();
										radianceHgData.diffuseTex = m_Tracer.GetTexture(material.diffTex).getHandle();
										radianceHgData.specularTex = m_Tracer.GetTexture(material.specTex).getHandle();
										radianceHgData.emissionTex = m_Tracer.GetTexture(material.emitTex).getHandle();
										radianceHgData.diffuse = material.diffCol;
										radianceHgData.specular = material.specCol;
										radianceHgData.emission = material.emitCol;
										radianceHgData.shinness = material.shinness;
										radianceHgData.transmit = material.tranCol;
										radianceHgData.refrInd = material.refrInd;
									}
									if (material.name == "light") {
										m_LightHgRecIndex = RAY_TYPE_COUNT * sbtOffset + RAY_TYPE_COUNT * i + RAY_TYPE_RADIANCE;
									}
									if (material.type == test::PhongMaterialType::eDiffuse) {
										cpuHgRecords[RAY_TYPE_COUNT * sbtOffset + RAY_TYPE_COUNT * i + RAY_TYPE_RADIANCE] = tracePipeline->hgProgramGroups["DiffuseDef"].getSBTRecord<HitgroupData>(radianceHgData);
									}
									else if (material.type == test::PhongMaterialType::eSpecular) {
										cpuHgRecords[RAY_TYPE_COUNT * sbtOffset + RAY_TYPE_COUNT * i + RAY_TYPE_RADIANCE] = tracePipeline->hgProgramGroups["Specular"].getSBTRecord<HitgroupData>(radianceHgData);
									}
									else if (material.type == test::PhongMaterialType::eRefract) {
										cpuHgRecords[RAY_TYPE_COUNT * sbtOffset + RAY_TYPE_COUNT * i + RAY_TYPE_RADIANCE] = tracePipeline->hgProgramGroups["Refraction"].getSBTRecord<HitgroupData>(radianceHgData);
									}
									else {
										cpuHgRecords[RAY_TYPE_COUNT * sbtOffset + RAY_TYPE_COUNT * i + RAY_TYPE_RADIANCE] = tracePipeline->hgProgramGroups["Emission"].getSBTRecord<HitgroupData>(radianceHgData);
									}
									cpuHgRecords[RAY_TYPE_COUNT * sbtOffset + RAY_TYPE_COUNT * i + RAY_TYPE_OCCLUSION] = tracePipeline->hgProgramGroups["Occlusion"].getSBTRecord<HitgroupData>();

								}
								sbtOffset += mesh->GetUniqueResource()->materials.size();
							}
						}
					}
				}
				defHitGBuffer.Upload();
			}
			{
				auto& neeHitGBuffer = tracePipeline->hitGBuffers["NEE"];
				neeHitGBuffer.cpuHandle.resize(RAY_TYPE_COUNT * m_Tracer.m_IASHandles["Top"]->GetSbtCount());
				auto& cpuHgRecords = neeHitGBuffer.cpuHandle;
				{
					auto& iasHandle = m_Tracer.m_IASHandles["Top"];
					size_t sbtOffset = 0;
					for (auto& instanceSet : iasHandle->GetInstanceSets()) {
						for (auto& baseGASHandle : instanceSet->baseGASHandles) {
							for (auto& mesh : baseGASHandle->GetMeshes()) {
								for (size_t i = 0; i < mesh->GetUniqueResource()->materials.size(); ++i) {
									auto materialId = mesh->GetUniqueResource()->materials[i];
									auto& material = m_MaterialSet->materials[materialId];
									HitgroupData radianceHgData = {};
									{
										radianceHgData.vertices  = mesh->GetSharedResource()->vertexBuffer.GetGpuComponent<rtlib::ext::resources::CUDABufferComponent<float3>>("CUDA")->GetHandle().getDevicePtr();
										radianceHgData.indices   = mesh->GetUniqueResource()->triIndBuffer.GetGpuComponent<rtlib::ext::resources::CUDABufferComponent<uint3>>("CUDA")->GetHandle().getDevicePtr();
										radianceHgData.texCoords = mesh->GetSharedResource()->texCrdBuffer.GetGpuComponent<rtlib::ext::resources::CUDABufferComponent<float2>>("CUDA")->GetHandle().getDevicePtr();
										radianceHgData.diffuseTex = m_Tracer.GetTexture(material.diffTex).getHandle();
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
										cpuHgRecords[RAY_TYPE_COUNT * sbtOffset + RAY_TYPE_COUNT * i + RAY_TYPE_RADIANCE] = tracePipeline->hgProgramGroups["DiffuseNee"].getSBTRecord<HitgroupData>(radianceHgData);
									}
									else if (material.type == test::PhongMaterialType::eSpecular) {
										cpuHgRecords[RAY_TYPE_COUNT * sbtOffset + RAY_TYPE_COUNT * i + RAY_TYPE_RADIANCE] = tracePipeline->hgProgramGroups["Specular"].getSBTRecord<HitgroupData>(radianceHgData);
									}
									else if (material.type == test::PhongMaterialType::eRefract) {
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
				neeHitGBuffer.Upload();
			}
			{
				tracePipeline->InitShaderBindingTable("Default", "Default", "Default", "Default");
				tracePipeline->InitShaderBindingTable("NEE"    , "Default", "Default", "NEE"    );
			}
			{
				{
					tracePipeline->subPasses["Default"].paramsBuffer.cpuHandle.resize(1);
					auto& params = tracePipeline->subPasses["Default"].paramsBuffer.cpuHandle[0];
					{
						params.frameBuffer = m_FrameBuffer.getDevicePtr();
						params.accumBuffer = m_AccumBuffer.getDevicePtr();
						params.seedBuffer = m_SeedBuffer.getDevicePtr();
						params.width = m_FbWidth;
						params.height = m_FbHeight;
						params.maxTraceDepth = kDefaultMaxTraceDepth;
						params.gasHandle = m_Tracer.m_IASHandles["Top"]->GetHandle();
						params.light = m_Light;
						params.samplePerALL = 0;
						params.samplePerLaunch = kDefaultSamplePerLaunch;
					}
				}
				{
					tracePipeline->subPasses["NEE"].paramsBuffer.cpuHandle.resize(1);
					auto& params = tracePipeline->subPasses["NEE"].paramsBuffer.cpuHandle[0];
					{
						params.frameBuffer     = m_FrameBuffer.getDevicePtr();
						params.accumBuffer     = m_AccumBuffer.getDevicePtr();
						params.seedBuffer            = m_SeedBuffer.getDevicePtr();
						params.width           = m_FbWidth;
						params.height          = m_FbHeight;
						params.maxTraceDepth   = kDefaultMaxTraceDepth;
						params.gasHandle       = m_Tracer.m_IASHandles["Top"]->GetHandle();
						params.light           = m_Light;
						params.samplePerALL    = 0;
						params.samplePerLaunch = kDefaultSamplePerLaunch;
					}
				}
			}
        }
        m_Tracer.SetTracePipeline(tracePipeline);

    }
	void InitDebugPipeline() {
		auto debugPipeline = std::make_shared<test::Pipeline<RayDebugParams>>();
		{
			debugPipeline->SetContext(m_Tracer.GetOPXContext());

			OptixPipelineCompileOptions compileOptions = {};

			debugPipeline->width  = m_FbWidth;
			debugPipeline->height = m_FbHeight;
			debugPipeline->subPasses["Default"].depth = 1;
			//compileOptions
			compileOptions.pipelineLaunchParamsVariableName = "params";
			compileOptions.numAttributeValues               = 3;
			compileOptions.numPayloadValues                 = 2;
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
			moduleCompileOptions.debugLevel = static_cast<OptixCompileDebugLevel>(rtlib::OPXCompileDebugLevel::Minimal);
			moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
#else
			moduleCompileOptions.optLevel   = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
			moduleCompileOptions.debugLevel = static_cast<OptixCompileDebugLevel>(rtlib::OPXCompileDebugLevel::None);
#endif
			debugPipeline->LoadModuleFromPtxFile("RayDebug", TEST_TEST18_CUDA_PATH"/RayDebug.ptx", moduleCompileOptions);
		}
		//program group: init
		{
			debugPipeline->LoadRgProgramGroupFromModule({"RayDebug", "__raygen__debug" });
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
			linkOptions.debugLevel    = static_cast<OptixCompileDebugLevel>(rtlib::OPXCompileDebugLevel::Minimal);
#else
			linkOptions.debugLevel    = static_cast<OptixCompileDebugLevel>(rtlib::OPXCompileDebugLevel::None);
#endif

			debugPipeline->LinkPipeline(linkOptions);
		}
		//SBTRecord
		{
			auto& rayGBuffer = debugPipeline->rayGBuffers["Default"];
			rayGBuffer.cpuHandle.resize(1);
			auto camera    = m_CameraController.GetCamera(m_CameraFovY, m_FbAspect);
			auto [u, v, w] = camera.getUVW();
			rayGBuffer.cpuHandle[0] = debugPipeline->rgProgramGroup.getSBTRecord<RayGenData>();
			rayGBuffer.cpuHandle[0].data.eye = camera.getEye();
			rayGBuffer.cpuHandle[0].data.u = u;
			rayGBuffer.cpuHandle[0].data.v = v;
			rayGBuffer.cpuHandle[0].data.w = w;
			rayGBuffer.Upload();
			auto& missBuffer = debugPipeline->missBuffers["Default"];
			missBuffer.cpuHandle.resize(RAY_TYPE_COUNT);
			missBuffer.cpuHandle[RAY_TYPE_RADIANCE]               = debugPipeline->msProgramGroups["Radiance"].getSBTRecord<MissData>();
			missBuffer.cpuHandle[RAY_TYPE_RADIANCE].data.bgColor  = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
			missBuffer.cpuHandle[RAY_TYPE_OCCLUSION]              = debugPipeline->msProgramGroups["Occlusion"].getSBTRecord<MissData>();
			missBuffer.cpuHandle[RAY_TYPE_OCCLUSION].data.bgColor = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
			missBuffer.Upload();
			auto& hitGBuffer = debugPipeline->hitGBuffers["Default"];
			hitGBuffer.cpuHandle.resize(RAY_TYPE_COUNT * m_Tracer.m_IASHandles["Top"]->GetSbtCount());
			auto& cpuHgRecords = hitGBuffer.cpuHandle;
			{
				auto& iasHandle  = m_Tracer.m_IASHandles["Top"];
				size_t sbtOffset = 0;
				for (auto& instanceSet : iasHandle->GetInstanceSets()) {
					for (auto& baseGASHandle : instanceSet->baseGASHandles) {
						for (auto& mesh : baseGASHandle->GetMeshes()) {
							for (size_t i = 0; i < mesh->GetUniqueResource()->materials.size(); ++i) {
								auto materialId = mesh->GetUniqueResource()->materials[i];
								auto& material  = m_MaterialSet->materials[materialId];
								HitgroupData radianceHgData = {};
								{
									radianceHgData.vertices    = mesh->GetSharedResource()->vertexBuffer.GetGpuComponent<rtlib::ext::resources::CUDABufferComponent<float3>>("CUDA")->GetHandle().getDevicePtr();
									radianceHgData.indices     = mesh->GetUniqueResource()->triIndBuffer.GetGpuComponent<rtlib::ext::resources::CUDABufferComponent<uint3>>("CUDA")->GetHandle().getDevicePtr();
									radianceHgData.texCoords   = mesh->GetSharedResource()->texCrdBuffer.GetGpuComponent<rtlib::ext::resources::CUDABufferComponent<float2>>("CUDA")->GetHandle().getDevicePtr();
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
								cpuHgRecords[RAY_TYPE_COUNT * sbtOffset + RAY_TYPE_COUNT * i + RAY_TYPE_OCCLUSION] = debugPipeline->hgProgramGroups["Occlusion"].getSBTRecord<HitgroupData>(radianceHgData);
							}
							sbtOffset += mesh->GetUniqueResource()->materials.size();
						}
					}
				}
			}
			hitGBuffer.Upload();
			{
				debugPipeline->InitShaderBindingTable("Default", "Default", "Default", "Default");
			}

			{
				debugPipeline->subPasses["Default"].paramsBuffer.cpuHandle.resize(1);
				auto& params = debugPipeline->subPasses["Default"].paramsBuffer.cpuHandle[0];
				{
					params.width           = m_FbWidth;
					params.height          = m_FbHeight;
					params.gasHandle       = m_Tracer.m_IASHandles["Top"]->GetHandle();
					params.light           = m_Light;
				}
			}
		}
		m_Tracer.SetDebugPipeline(debugPipeline);
	}
	void MainLoop() {
		PrepareMainLoop();
		while (!glfwWindowShouldClose(m_Window)) {
			{
				if (m_Resized     ) {
					this->OnResize();
					m_Resized      = false;
					m_UpdateCamera = true;
					//Resize��Flush����K�v������
					m_FlushFrame   = true;
				}
				if (m_UpdateCamera) {
					this->OnUpdateCamera();
					m_UpdateCamera = false;
					//Camera�̈ړ���Flush����K�v������
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
		m_Tracer.GetTracePipeline()->subPasses["Default"].paramsBuffer.cpuHandle[0].accumBuffer = m_AccumBuffer.getDevicePtr();
		m_Tracer.GetTracePipeline()->subPasses["Default"].paramsBuffer.cpuHandle[0].seedBuffer        = m_SeedBuffer.getDevicePtr();
		m_Tracer.GetTracePipeline()->subPasses["Default"].paramsBuffer.cpuHandle[0].width       = m_FbWidth;
		m_Tracer.GetTracePipeline()->subPasses["Default"].paramsBuffer.cpuHandle[0].height      = m_FbHeight;
		m_Tracer.GetTracePipeline()->subPasses["NEE"].paramsBuffer.cpuHandle[0].accumBuffer     = m_AccumBuffer.getDevicePtr();
		m_Tracer.GetTracePipeline()->subPasses["NEE"].paramsBuffer.cpuHandle[0].seedBuffer            = m_SeedBuffer.getDevicePtr();
		m_Tracer.GetTracePipeline()->subPasses["NEE"].paramsBuffer.cpuHandle[0].width           = m_FbWidth;
		m_Tracer.GetTracePipeline()->subPasses["NEE"].paramsBuffer.cpuHandle[0].height          = m_FbHeight;
		m_Tracer.GetDebugPipeline()->subPasses["Default"].paramsBuffer.cpuHandle[0].width       = m_FbWidth;
		m_Tracer.GetDebugPipeline()->subPasses["Default"].paramsBuffer.cpuHandle[0].height      = m_FbHeight;
	}
	void OnUpdateCamera()
	{
		auto camera    = m_CameraController.GetCamera(m_CameraFovY, m_FbAspect);
		auto [u, v, w] = camera.getUVW();
		m_Tracer.GetTracePipeline()->rayGBuffers["Default"].cpuHandle[0].data.eye = camera.getEye();
		m_Tracer.GetTracePipeline()->rayGBuffers["Default"].cpuHandle[0].data.u   = u;
		m_Tracer.GetTracePipeline()->rayGBuffers["Default"].cpuHandle[0].data.v   = v;
		m_Tracer.GetTracePipeline()->rayGBuffers["Default"].cpuHandle[0].data.w   = w;
		m_Tracer.GetTracePipeline()->updateRG = true;

		m_Tracer.GetDebugPipeline()->rayGBuffers["Default"].cpuHandle[0].data.eye = camera.getEye();
		m_Tracer.GetDebugPipeline()->rayGBuffers["Default"].cpuHandle[0].data.u   = u;
		m_Tracer.GetDebugPipeline()->rayGBuffers["Default"].cpuHandle[0].data.v   = v;
		m_Tracer.GetDebugPipeline()->rayGBuffers["Default"].cpuHandle[0].data.w   = w;
		m_Tracer.GetDebugPipeline()->updateRG = true;
	}
	void OnUpdateLight() {
		m_Tracer.GetTracePipeline()->hitGBuffers["Default"].cpuHandle[m_LightHgRecIndex].data.emission = m_Light.emission;
		m_Tracer.GetTracePipeline()->hitGBuffers["Default"].cpuHandle[m_LightHgRecIndex].data.diffuse  = m_Light.emission;
		m_Tracer.GetTracePipeline()->hitGBuffers["NEE"].cpuHandle[m_LightHgRecIndex].data.emission     = m_Light.emission;
		m_Tracer.GetTracePipeline()->hitGBuffers["NEE"].cpuHandle[m_LightHgRecIndex].data.diffuse      = m_Light.emission;
		m_Tracer.GetTracePipeline()->updateHG = true;
		m_Tracer.GetDebugPipeline()->hitGBuffers["Default"].cpuHandle[m_LightHgRecIndex].data.emission = m_Light.emission;
		m_Tracer.GetDebugPipeline()->hitGBuffers["Default"].cpuHandle[m_LightHgRecIndex].data.diffuse  = m_Light.emission;
		m_Tracer.GetDebugPipeline()->updateHG = true;
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
		if(m_CurPipelineName=="Trace") {
			m_Tracer.GetTracePipeline()->width  = m_FbWidth;
			m_Tracer.GetTracePipeline()->height = m_FbHeight;
			auto& params           = m_Tracer.GetTracePipeline()->subPasses[m_CurSubPassName].paramsBuffer.cpuHandle[0];
			params.frameBuffer     = m_FrameBuffer.getDevicePtr();
			params.accumBuffer     = m_AccumBuffer.getDevicePtr();
			params.seedBuffer            = m_SeedBuffer.getDevicePtr();
			params.width           = m_FbWidth;
			params.height          = m_FbHeight;
			//params.maxTraceDepth   = m_MaxTraceDepth;
			params.gasHandle       = m_Tracer.m_IASHandles["Top"]->GetHandle();
			params.light           = m_Light;
			params.samplePerALL    = m_SamplePerAll;
		}else{
			m_Tracer.GetDebugPipeline()->width  = m_FbWidth;
			m_Tracer.GetDebugPipeline()->height = m_FbHeight;
			auto& params = m_Tracer.GetDebugPipeline()->subPasses["Default"].paramsBuffer.cpuHandle[0];
			params.diffuseBuffer   = m_DebugBuffers["Diffuse"].getDevicePtr();
			params.specularBuffer  = m_DebugBuffers["Specular"].getDevicePtr();
			params.emissionBuffer  = m_DebugBuffers["Emission"].getDevicePtr();
			params.transmitBuffer  = m_DebugBuffers["Transmit"].getDevicePtr();
			params.texCoordBuffer  = m_DebugBuffers["TexCoord"].getDevicePtr();
			params.normalBuffer    = m_DebugBuffers["Normal"].getDevicePtr();
			params.depthBuffer     = m_DebugBuffers["Depth"].getDevicePtr();
			params.width           = m_FbWidth;
			params.height          = m_FbHeight;
			params.gasHandle       = m_Tracer.m_IASHandles["Top"]->GetHandle();
			params.light           = m_Light;
		}
	}
	void OnUpdateTime() {
		float prevTime = glfwGetTime();
		m_DelTime      = prevTime - m_CurTime;
		m_CurTime      = prevTime;
	}
	void OnLaunch() {
		if (m_CurPipelineName == "Trace") {
			auto  curPipeline   = m_Tracer.GetTracePipeline();
			auto& params        = curPipeline->subPasses[m_CurSubPassName].paramsBuffer.cpuHandle[0];
			params.frameBuffer  = m_FrameBufferGL.map();
			params.samplePerALL = m_SamplePerAll;
			curPipeline->updatePm = true;
			curPipeline->Update();
			curPipeline->Launch(m_CurSubPassName,m_Stream);
			cuStreamSynchronize(m_Stream);
			m_FrameBufferGL.unmap();
			m_SamplePerAll += curPipeline->subPasses[m_CurSubPassName].paramsBuffer.cpuHandle[0].samplePerLaunch;
		}
		else {
			
			auto  curPipeline     = m_Tracer.GetDebugPipeline();
			auto& params          = curPipeline->subPasses["Default"].paramsBuffer.cpuHandle[0];
			params.diffuseBuffer  = m_DebugBufferGLs["Diffuse"].map();
			params.specularBuffer = m_DebugBufferGLs["Specular"].map();
			params.emissionBuffer = m_DebugBufferGLs["Emission"].map();
			params.transmitBuffer = m_DebugBufferGLs["Transmit"].map();
			params.texCoordBuffer = m_DebugBufferGLs["TexCoord"].map();
			params.normalBuffer   = m_DebugBufferGLs["Normal"].map();
			params.depthBuffer    = m_DebugBufferGLs["Depth"].map();

			curPipeline->updatePm = true;
			curPipeline->Update();
			curPipeline->Launch("Default",m_Stream);
			cuStreamSynchronize(m_Stream);
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
		}else{
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
					auto  curPipeline = m_Tracer.GetTracePipeline();
					auto& curSubpass  = curPipeline->subPasses[m_CurSubPassName];
					auto& curParams   = curSubpass.paramsBuffer.cpuHandle[0];
					ImGui::Text("Smp  : %3d" , curParams.samplePerALL);
					ImGui::Text("Smp/s: %.2f", curParams.samplePerLaunch / m_DelTime);
					{
						int samplePerLaunch =  curParams.samplePerLaunch;
						if (ImGui::SliderInt("samplePerLaunch", &samplePerLaunch, 1, 10)) {
							curParams.samplePerLaunch = samplePerLaunch;
						}
					}
					{
						int maxTraceDepth = curParams.maxTraceDepth;
						if (ImGui::SliderInt("maxTraceDepth", &maxTraceDepth, 1, 10)) {
							curParams.maxTraceDepth = maxTraceDepth;
							m_FlushFrame  = true;
						}
					}
				}
				else {
					static int mode = 0;
					ImGui::RadioButton("Diffuse" , &mode, 0); 
					ImGui::SameLine(); 
					ImGui::RadioButton("Specular", &mode, 1);
					ImGui::SameLine();
					ImGui::RadioButton("Transmit", &mode, 2);
					ImGui::SameLine();
					ImGui::RadioButton("Emission", &mode, 3);
					ImGui::NewLine();
					ImGui::RadioButton("TexCoord", &mode, 4);
					ImGui::SameLine();
					ImGui::RadioButton("Normal"  , &mode, 5);
					ImGui::SameLine();
					ImGui::RadioButton("Depth"   , &mode, 6);
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
							m_CameraFovY   = camFovY;
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
					m_UpdateCamera  = true;
				}
				if (m_CurPipelineName == "Trace") {
					ImGui::SameLine();
					if (ImGui::Button(m_PrvSubPassName.c_str())) {
						std::swap(m_CurSubPassName, m_PrvSubPassName);
						m_FlushFrame  = true;
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
		Test18Application* app = reinterpret_cast<Test18Application*>(glfwGetWindowUserPointer(window));
		if (app) {
			if (fbWidth != app->m_FbWidth || fbHeight != app->m_FbHeight)
			{
				app->m_FbWidth  = fbWidth;
				app->m_FbHeight = fbHeight;
				app->m_FbAspect = static_cast<float>(fbWidth) / static_cast<float>(fbHeight);
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
	using CUDABufferMap      = std::unordered_map<std::string, rtlib::CUDABuffer<uchar4>>;
	using GLInteropBufferMap = std::unordered_map<std::string, rtlib::GLInteropBuffer<uchar4>>;
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
	bool                            m_Resized            = false;
	bool                            m_UpdateCamera       = false;
	bool                            m_UpdateLight        = false;

	rtlib::GLTexture2D<uchar4>      m_GLTexture          = {};
	std::shared_ptr<rtlib::ext::RectRenderer>        
		                            m_RectRenderer       = {};
	rtlib::ext::CameraController         m_CameraController   = {};
	test::PathTracer                m_Tracer             = {};
	test::MaterialSetPtr            m_MaterialSet        = nullptr;
	CUstream                        m_Stream             = nullptr;
	//Trace
	rtlib::CUDABuffer<uchar4>       m_FrameBuffer        = {};
	rtlib::GLInteropBuffer<uchar4>  m_FrameBufferGL      = {};
	rtlib::CUDABuffer<float3>       m_AccumBuffer        = {};
	rtlib::CUDABuffer<unsigned int> m_SeedBuffer         = {};
	//Debug
	CUDABufferMap                   m_DebugBuffers       = {};
	GLInteropBufferMap              m_DebugBufferGLs     = {};
	//Light
	uint32_t                        m_LightHgRecIndex    = 0;
	ParallelLight                   m_Light              = {};
	//Config
	uint32_t                        m_SamplePerAll       = 0;
	//State
	bool                            m_FlushFrame         = false;
	bool                            m_UpdateParams       = false;
	std::string                     m_CurPipelineName    = "Trace";
	std::string                     m_PrvPipelineName    = "Debug";
	std::string                     m_CurSubPassName     = "Default";
	std::string                     m_PrvSubPassName     = "NEE";
	std::string                     m_DebugFrameName     = "Diffuse";
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