#include "../include/GuidePathOPXTracer.h"
#include <RTPathGuidingUtils.h>
#include <RTLib/ext/Utils.h>
#include <RayTrace.h>
#include <fstream>
#include <random>
using namespace std::string_literals;
using namespace test24_guide_path;
using RayGRecord = rtlib::SBTRecord<RayGenData>;
using MissRecord = rtlib::SBTRecord<MissData>;
using HitGRecord = rtlib::SBTRecord<HitgroupData>;
// RTTracer ����Čp������܂���
struct Test24GuidePathOPXTracer::Impl {
	Impl(
		ContextPtr context,
		FramebufferPtr framebuffer,
		CameraControllerPtr cameraController,
		TextureAssetManager textureManager,
		rtlib::ext::IASHandlePtr topLevelAS,
		const std::vector<rtlib::ext::VariableMap>& materials,
		const float3& bgLightColor,
		const unsigned int& eventFlags,
		const unsigned int& maxTraceDepth
	) :
		m_Context{ context },
		m_Framebuffer{ framebuffer },
		m_CameraController{ cameraController },
		m_TextureManager{ textureManager },
		m_TopLevelAS{ topLevelAS },
		m_Materials{ materials },
		m_BgLightColor{ bgLightColor },
		m_EventFlags{ eventFlags },
		m_Pipeline{},
		m_Modules{},
		m_RGProgramGroups{},
		m_HGProgramGroups{},
		m_MSProgramGroups{},
		m_Params{},
		m_SeedBuffer{},
		m_MeshLights{},
		m_MaxTraceDepth{ maxTraceDepth }{}
	~Impl() {}

	auto GetMeshLightList()const noexcept -> MeshLightList
	{
		MeshLightList list;
		list.data = m_MeshLights.gpuHandle.getDevicePtr();
		list.count = m_MeshLights.cpuHandle.size();
		return list;
	}

	bool HasEvent(unsigned int eventFlag)const noexcept
	{
		return (m_EventFlags & eventFlag) == eventFlag;
	}

	std::weak_ptr<test::RTContext> m_Context;
	std::weak_ptr<test::RTFramebuffer> m_Framebuffer;
	std::weak_ptr<rtlib::ext::CameraController> m_CameraController;
	std::weak_ptr<test::RTTextureAssetManager>  m_TextureManager;
	std::weak_ptr<rtlib::ext::IASHandle> m_TopLevelAS;
	const std::vector<rtlib::ext::VariableMap>& m_Materials;
	const float3& m_BgLightColor;
	const unsigned int& m_EventFlags;
	const unsigned int& m_MaxTraceDepth;
	rtlib::ext::Camera m_Camera = {};
	OptixShaderBindingTable m_ShaderBindingTable = {};
	Pipeline  m_Pipeline;
	ModuleMap m_Modules;
	RGProgramGroupMap m_RGProgramGroups;
	MSProgramGroupMap m_MSProgramGroups;
	HGProgramGroupMap m_HGProgramGroups;
	rtlib::CUDAUploadBuffer<RayGRecord> m_RGRecordBuffer;
	rtlib::CUDAUploadBuffer<MissRecord> m_MSRecordBuffers;
	rtlib::CUDAUploadBuffer<HitGRecord> m_HGRecordBuffers;
	rtlib::CUDAUploadBuffer<RayTraceParams> m_Params;
	rtlib::CUDABuffer<unsigned int> m_SeedBuffer;
	rtlib::CUDAUploadBuffer<MeshLight> m_MeshLights;
	std::shared_ptr<RTSTreeWrapper> m_STree = nullptr;
	unsigned int m_LightHgRecIndex = 0;
	unsigned int m_SampleForRemain = 0;
	unsigned int m_SampleForPass   = 0;
};

Test24GuidePathOPXTracer::Test24GuidePathOPXTracer(
	ContextPtr context,
	FramebufferPtr framebuffer,
	CameraControllerPtr cameraController,
	TextureAssetManager textureManager,
	rtlib::ext::IASHandlePtr topLevelAS,
	const std::vector<rtlib::ext::VariableMap>& materials,
	const float3& bgLightColor,
	const unsigned int& eventFlags,
	const unsigned int& maxTraceDepth) :test::RTTracer() {
	m_Impl = std::make_unique<Test24GuidePathOPXTracer::Impl>(
		context,
		framebuffer,
		cameraController,
		textureManager,
		topLevelAS,
		materials,
		bgLightColor,
		eventFlags,
		maxTraceDepth);

	GetVariables()->SetBool("Started" , false);
	GetVariables()->SetBool("Launched", false);
	GetVariables()->SetUInt32("SamplePerAll"     , 0);
	GetVariables()->SetUInt32("SamplePerTmp"     , 0);
	GetVariables()->SetUInt32("SamplePerLaunch"  , 1);
	GetVariables()->SetUInt32("SampleForBudget"  , 1024);
	GetVariables()->SetUInt32("CurIteration"     , 0);
	GetVariables()->SetUInt32("IterationForBuilt", 0);
	GetVariables()->SetFloat1("RatioForBudget"   , 0.5f);
}

void Test24GuidePathOPXTracer::Initialize()
{
	this->InitFrameResources();
	this->InitPipeline();
	this->InitShaderBindingTable();
	this->InitLight();
	this->InitSdTree();
	this->InitLaunchParams();
}

void Test24GuidePathOPXTracer::Launch(int width, int height, void* pdata)
{

	UserData* pUserData = (UserData*)pdata;
	if (!pUserData)
	{
		return;
	}
	if (width != this->m_Impl->m_Framebuffer.lock()->GetWidth() || height != this->m_Impl->m_Framebuffer.lock()->GetHeight()) {
		return;
	}
	if (this->OnLaunchBegin  (width, height, pUserData)) {
		this->OnLaunchExecute(width, height, pUserData);
		this->OnLaunchEnd(    width, height, pUserData);
	}
}

void Test24GuidePathOPXTracer::CleanUp()
{
	this->FreePipeline();
	this->FreeShaderBindingTable();
	this->FreeSdTree();
	this->FreeLight();
	this->FreeLaunchParams();
	this->FreeFrameResources();
	this->m_Impl->m_LightHgRecIndex = 0;
}

void Test24GuidePathOPXTracer::Update()
{
	if (this->m_Impl->HasEvent(TEST24_EVENT_FLAG_BIT_UPDATE_CAMERA))
	{
		float aspect = static_cast<float>(this->m_Impl->m_Framebuffer.lock()->GetWidth()) / static_cast<float>(this->m_Impl->m_Framebuffer.lock()->GetHeight());
		this->m_Impl->m_Camera = this->m_Impl->m_CameraController.lock()->GetCamera(aspect);
		this->m_Impl->m_RGRecordBuffer.cpuHandle[0].data.eye = this->m_Impl->m_Camera.getEye();
		auto [u, v, w] = this->m_Impl->m_Camera.getUVW();
		this->m_Impl->m_RGRecordBuffer.cpuHandle[0].data.u = u;
		this->m_Impl->m_RGRecordBuffer.cpuHandle[0].data.v = v;
		this->m_Impl->m_RGRecordBuffer.cpuHandle[0].data.w = w;
		this->m_Impl->m_RGRecordBuffer.Upload();
	}
	if (this->m_Impl->HasEvent(TEST24_EVENT_FLAG_BIT_FLUSH_FRAME  ))
	{
		std::vector<float3> zeroAccumValues(this->m_Impl->m_Framebuffer.lock()->GetWidth() * this->m_Impl->m_Framebuffer.lock()->GetHeight(), make_float3(0.0f));
		cudaMemcpy(this->m_Impl->m_Framebuffer.lock()->GetComponent<test::RTCUDABufferFBComponent<float3>>("RAccum")->GetHandle().getDevicePtr(), zeroAccumValues.data(), sizeof(float3) * this->m_Impl->m_Framebuffer.lock()->GetWidth() * this->m_Impl->m_Framebuffer.lock()->GetHeight(), cudaMemcpyHostToDevice);
		this->m_Impl->m_Params.cpuHandle[0].maxTraceDepth = this->m_Impl->m_MaxTraceDepth;
		this->m_Impl->m_Params.cpuHandle[0].samplePerALL = 0;
		GetVariables()->SetUInt32("SamplePerAll", 0);
	}
	auto samplePerAll = GetVariables()->GetUInt32("SamplePerAll");
	bool shouldRegen = ((samplePerAll + this->m_Impl->m_Params.cpuHandle[0].samplePerLaunch) / 1024 != samplePerAll / 1024);
	if (this->m_Impl->HasEvent(TEST24_EVENT_FLAG_BIT_RESIZE_FRAME  ) || shouldRegen)
	{
		std::cout << "Regen!\n";
		std::vector<unsigned int> seedData(this->m_Impl->m_Framebuffer.lock()->GetWidth() * this->m_Impl->m_Framebuffer.lock()->GetHeight());
		std::random_device rd;
		std::mt19937 mt(rd());
		std::generate(seedData.begin(), seedData.end(), mt);
		this->m_Impl->m_SeedBuffer.resize(this->m_Impl->m_Framebuffer.lock()->GetWidth() * this->m_Impl->m_Framebuffer.lock()->GetHeight());
		this->m_Impl->m_SeedBuffer.upload(seedData);
	}
	if (this->m_Impl->HasEvent(TEST24_EVENT_FLAG_BIT_UPDATE_LIGHT ))
	{
		auto lightColor = make_float4(this->m_Impl->m_BgLightColor, 1.0f);
		this->m_Impl->m_MSRecordBuffers.cpuHandle[RAY_TYPE_RADIANCE].data.bgColor = lightColor;
		RTLIB_CUDA_CHECK(cudaMemcpy(this->m_Impl->m_MSRecordBuffers.gpuHandle.getDevicePtr() + RAY_TYPE_RADIANCE, &this->m_Impl->m_MSRecordBuffers.cpuHandle[RAY_TYPE_RADIANCE], sizeof(MissRecord), cudaMemcpyHostToDevice));
	}
}

bool Test24GuidePathOPXTracer::ShouldLock() const noexcept
{
	return false;
}

Test24GuidePathOPXTracer::~Test24GuidePathOPXTracer() {
}

void Test24GuidePathOPXTracer::InitLight()
{
	auto ChooseNEE = [](const rtlib::ext::MeshPtr& mesh)->bool {
		return (mesh->GetUniqueResource()->triIndBuffer.Size() < 200 || mesh->GetUniqueResource()->triIndBuffer.Size() > 230);
	};
	auto lightGASHandle = m_Impl->m_TopLevelAS.lock()->GetInstanceSets()[0]->GetInstance(1).baseGASHandle;
	for (auto& mesh : lightGASHandle->GetMeshes())
	{
		//Select NEE Light
		if (!ChooseNEE(mesh)) {
			mesh->GetUniqueResource()->variables.SetBool("useNEE", false);
		}
		else {
			mesh->GetUniqueResource()->variables.SetBool("useNEE", true);
			std::cout << "Name: " << mesh->GetUniqueResource()->name << " LightCount: " << mesh->GetUniqueResource()->triIndBuffer.Size() << std::endl;
			MeshLight meshLight = {};
			if (!m_Impl->m_TextureManager.lock()->GetAsset(m_Impl->m_Materials[mesh->GetUniqueResource()->materials[0]].GetString("emitTex")).HasGpuComponent("CUDATexture")) {
				this->m_Impl->m_TextureManager.lock()->GetAsset(m_Impl->m_Materials[mesh->GetUniqueResource()->materials[0]].GetString("emitTex")).AddGpuComponent<rtlib::ext::resources::CUDATextureImage2DComponent<uchar4>>("CUDATexture");
			}
			meshLight.emission = m_Impl->m_Materials[mesh->GetUniqueResource()->materials[0]].GetFloat3As<float3>("emitCol");
			meshLight.emissionTex = m_Impl->m_TextureManager.lock()->GetAsset(m_Impl->m_Materials[mesh->GetUniqueResource()->materials[0]].GetString("emitTex")).GetGpuComponent<rtlib::ext::resources::CUDATextureImage2DComponent<uchar4>>("CUDATexture")->GetHandle().getHandle();
			meshLight.vertices = mesh->GetSharedResource()->vertexBuffer.GetGpuComponent<rtlib::ext::resources::CUDABufferComponent<float3>>("CUDA")->GetHandle().getDevicePtr();
			meshLight.normals = mesh->GetSharedResource()->normalBuffer.GetGpuComponent<rtlib::ext::resources::CUDABufferComponent<float3>>("CUDA")->GetHandle().getDevicePtr();
			meshLight.texCoords = mesh->GetSharedResource()->texCrdBuffer.GetGpuComponent<rtlib::ext::resources::CUDABufferComponent<float2>>("CUDA")->GetHandle().getDevicePtr();
			meshLight.indices = mesh->GetUniqueResource()->triIndBuffer.GetGpuComponent<rtlib::ext::resources::CUDABufferComponent<uint3>>("CUDA")->GetHandle().getDevicePtr();
			meshLight.indCount = mesh->GetUniqueResource()->triIndBuffer.Size();
			m_Impl->m_MeshLights.cpuHandle.push_back(meshLight);
		}
	}
	m_Impl->m_MeshLights.Alloc();
	m_Impl->m_MeshLights.Upload();
}

void Test24GuidePathOPXTracer::InitPipeline()
{
	auto rayGuidePtxFile = std::ifstream(TEST_TEST24_GUIDE_PATH_OPX_CUDA_PATH "/RayGuide.ptx", std::ios::binary);
	if (!rayGuidePtxFile.is_open()) {
		throw std::runtime_error("Failed To Load RayGuide.ptx!");
	}
	auto rayGuidePtxData = std::string((std::istreambuf_iterator<char>(rayGuidePtxFile)), (std::istreambuf_iterator<char>()));
	rayGuidePtxFile.close();

	auto guideCompileOptions = OptixPipelineCompileOptions{};
	{
		guideCompileOptions.pipelineLaunchParamsVariableName = "params";
		guideCompileOptions.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_TRIANGLE;
		guideCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
		guideCompileOptions.usesMotionBlur = false;
		guideCompileOptions.numAttributeValues = 3;
		guideCompileOptions.numPayloadValues = 8;
	}
	auto guideLinkOptions = OptixPipelineLinkOptions{};
	{
#ifndef NDEBUG
		guideLinkOptions.debugLevel = static_cast<OptixCompileDebugLevel>(rtlib::OPXCompileDebugLevel::Minimal);
#else
		guideLinkOptions.debugLevel = static_cast<OptixCompileDebugLevel>(rtlib::OPXCompileDebugLevel::None);
#endif
		guideLinkOptions.maxTraceDepth = 2;
	}
	auto guideModuleOptions = OptixModuleCompileOptions{};
	{
#ifndef NDEBUG
		guideModuleOptions.debugLevel = static_cast<OptixCompileDebugLevel>(rtlib::OPXCompileDebugLevel::Minimal);
		guideModuleOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
#else
		guideModuleOptions.debugLevel = static_cast<OptixCompileDebugLevel>(rtlib::OPXCompileDebugLevel::None);
		guideModuleOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
#endif
	}
	guideModuleOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
	m_Impl->m_Pipeline = m_Impl->m_Context.lock()->GetOPX7Handle()->createPipeline(guideCompileOptions);
	m_Impl->m_Modules["RayGuide"] = m_Impl->m_Pipeline.createModule(rayGuidePtxData, guideModuleOptions);
	m_Impl->m_RGProgramGroups["Guide.Default"] = m_Impl->m_Pipeline.createRaygenPG({ m_Impl->m_Modules["RayGuide"], RTLIB_RAYGEN_PROGRAM_STR(def) });
	m_Impl->m_RGProgramGroups["Guide.Guiding.Default"] = m_Impl->m_Pipeline.createRaygenPG({ m_Impl->m_Modules["RayGuide"], RTLIB_RAYGEN_PROGRAM_STR(pg_def) });
	m_Impl->m_MSProgramGroups["Guide.Radiance"] = m_Impl->m_Pipeline.createMissPG({ m_Impl->m_Modules["RayGuide"], RTLIB_MISS_PROGRAM_STR(radiance) });
	m_Impl->m_MSProgramGroups["Guide.Occluded"] = m_Impl->m_Pipeline.createMissPG({ m_Impl->m_Modules["RayGuide"], RTLIB_MISS_PROGRAM_STR(occluded) });
	m_Impl->m_HGProgramGroups["Guide.Radiance.Diffuse.Guiding.Default"] = m_Impl->m_Pipeline.createHitgroupPG({ m_Impl->m_Modules["RayGuide"], RTLIB_CLOSESTHIT_PROGRAM_STR(radiance_for_diffuse_pg_def) }, {}, {});
	m_Impl->m_HGProgramGroups["Guide.Radiance.Phong.Guiding.Default"] = m_Impl->m_Pipeline.createHitgroupPG({ m_Impl->m_Modules["RayGuide"], RTLIB_CLOSESTHIT_PROGRAM_STR(radiance_for_phong_pg_def) }, {}, {});
	m_Impl->m_HGProgramGroups["Guide.Radiance.Emission"] = m_Impl->m_Pipeline.createHitgroupPG({ m_Impl->m_Modules["RayGuide"], RTLIB_CLOSESTHIT_PROGRAM_STR(radiance_for_emission) }, {}, {});
	m_Impl->m_HGProgramGroups["Guide.Radiance.Specular"] = m_Impl->m_Pipeline.createHitgroupPG({ m_Impl->m_Modules["RayGuide"], RTLIB_CLOSESTHIT_PROGRAM_STR(radiance_for_specular) }, {}, {});
	m_Impl->m_HGProgramGroups["Guide.Radiance.Refraction"] = m_Impl->m_Pipeline.createHitgroupPG({ m_Impl->m_Modules["RayGuide"], RTLIB_CLOSESTHIT_PROGRAM_STR(radiance_for_refraction) }, {}, {});
	m_Impl->m_HGProgramGroups["Guide.Occluded"] = m_Impl->m_Pipeline.createHitgroupPG({ m_Impl->m_Modules["RayGuide"], RTLIB_CLOSESTHIT_PROGRAM_STR(occluded) }, {}, {});
	m_Impl->m_Pipeline.link(guideLinkOptions);
}

void Test24GuidePathOPXTracer::InitFrameResources()
{
	std::vector<unsigned int> seedData(this->m_Impl->m_Framebuffer.lock()->GetWidth() * this->m_Impl->m_Framebuffer.lock()->GetHeight());
	std::random_device rd;
	std::mt19937 mt(rd());
	std::generate(seedData.begin(), seedData.end(), mt);
	this->m_Impl->m_SeedBuffer = rtlib::CUDABuffer<uint32_t>(seedData);
}


void Test24GuidePathOPXTracer::InitShaderBindingTable()
{
	float aspect = static_cast<float>(this->m_Impl->m_Framebuffer.lock()->GetWidth()) / static_cast<float>(this->m_Impl->m_Framebuffer.lock()->GetHeight());
	auto tlas = this->m_Impl->m_TopLevelAS.lock();
	this->m_Impl->m_Camera = this->m_Impl->m_CameraController.lock()->GetCamera(aspect);
	auto& materials = this->m_Impl->m_Materials;
	this->m_Impl->m_RGRecordBuffer.Alloc(1);
	this->m_Impl->m_RGRecordBuffer.cpuHandle[0] = this->m_Impl->m_RGProgramGroups["Guide.Guiding.Default"].getSBTRecord<RayGenData>();
	this->m_Impl->m_RGRecordBuffer.cpuHandle[0].data.eye = this->m_Impl->m_Camera.getEye();
	auto [u, v, w] = this->m_Impl->m_Camera.getUVW();
	this->m_Impl->m_RGRecordBuffer.cpuHandle[0].data.u = u;
	this->m_Impl->m_RGRecordBuffer.cpuHandle[0].data.v = v;
	this->m_Impl->m_RGRecordBuffer.cpuHandle[0].data.w = w;
	this->m_Impl->m_RGRecordBuffer.Upload();
	this->m_Impl->m_MSRecordBuffers.Alloc(RAY_TYPE_COUNT);
	this->m_Impl->m_MSRecordBuffers.cpuHandle[RAY_TYPE_RADIANCE] = this->m_Impl->m_MSProgramGroups["Guide.Radiance"].getSBTRecord<MissData>();
	this->m_Impl->m_MSRecordBuffers.cpuHandle[RAY_TYPE_RADIANCE].data.bgColor = make_float4(this->m_Impl->m_BgLightColor, 1.0f);
	this->m_Impl->m_MSRecordBuffers.cpuHandle[RAY_TYPE_OCCLUSION] = this->m_Impl->m_MSProgramGroups["Guide.Occluded"].getSBTRecord<MissData>();
	this->m_Impl->m_MSRecordBuffers.Upload();
	this->m_Impl->m_HGRecordBuffers.Alloc(tlas->GetSbtCount() * RAY_TYPE_COUNT);
	{
		size_t sbtOffset = 0;
		for (auto& instanceSet : tlas->GetInstanceSets())
		{
			for (auto& baseGASHandle : instanceSet->baseGASHandles)
			{
				for (auto& mesh : baseGASHandle->GetMeshes())
				{
					if (!mesh->GetSharedResource()->vertexBuffer.HasGpuComponent("CUDA")) {
						throw std::runtime_error("VertexBuffer of Mesh '" + mesh->GetUniqueResource()->name + "' Has No CUDA Component!");
					}
					if (!mesh->GetSharedResource()->normalBuffer.HasGpuComponent("CUDA")) {
						throw std::runtime_error("NormalBuffer of Mesh '" + mesh->GetUniqueResource()->name + "' Has No CUDA Component!");
					}
					if (!mesh->GetSharedResource()->texCrdBuffer.HasGpuComponent("CUDA")) {
						throw std::runtime_error("TexCrdBuffer of Mesh '" + mesh->GetUniqueResource()->name + "' Has No CUDA Component!");
					}
					if (!mesh->GetUniqueResource()->triIndBuffer.HasGpuComponent("CUDA")) {
						throw std::runtime_error("TriIndBuffer of Mesh '" + mesh->GetUniqueResource()->name + "' Has No CUDA Component!");
					}
					auto cudaVertexBuffer = mesh->GetSharedResource()->vertexBuffer.GetGpuComponent<rtlib::ext::resources::CUDABufferComponent<float3>>("CUDA");
					auto cudaNormalBuffer = mesh->GetSharedResource()->normalBuffer.GetGpuComponent<rtlib::ext::resources::CUDABufferComponent<float3>>("CUDA");
					auto cudaTexCrdBuffer = mesh->GetSharedResource()->texCrdBuffer.GetGpuComponent<rtlib::ext::resources::CUDABufferComponent<float2>>("CUDA");
					auto cudaTriIndBuffer = mesh->GetUniqueResource()->triIndBuffer.GetGpuComponent<rtlib::ext::resources::CUDABufferComponent<uint3>>("CUDA");
					for (size_t i = 0; i < mesh->GetUniqueResource()->materials.size(); ++i)
					{
						auto materialId = mesh->GetUniqueResource()->materials[i];
						auto& material = materials[materialId];
						if (!this->m_Impl->m_TextureManager.lock()->GetAsset(material.GetString("diffTex")).HasGpuComponent("CUDATexture")) {
							 this->m_Impl->m_TextureManager.lock()->GetAsset(material.GetString("diffTex")).AddGpuComponent<rtlib::ext::resources::CUDATextureImage2DComponent<uchar4>>("CUDATexture");
						}
						if (!this->m_Impl->m_TextureManager.lock()->GetAsset(material.GetString("specTex")).HasGpuComponent("CUDATexture")) {
							 this->m_Impl->m_TextureManager.lock()->GetAsset(material.GetString("specTex")).AddGpuComponent<rtlib::ext::resources::CUDATextureImage2DComponent<uchar4>>("CUDATexture");
						}
						if (!this->m_Impl->m_TextureManager.lock()->GetAsset(material.GetString("emitTex")).HasGpuComponent("CUDATexture")) {
							 this->m_Impl->m_TextureManager.lock()->GetAsset(material.GetString("emitTex")).AddGpuComponent<rtlib::ext::resources::CUDATextureImage2DComponent<uchar4>>("CUDATexture");
						}
						HitgroupData radianceHgData = {};
						{
							radianceHgData.vertices = cudaVertexBuffer->GetHandle().getDevicePtr();
							radianceHgData.normals = cudaNormalBuffer->GetHandle().getDevicePtr();
							radianceHgData.texCoords = cudaTexCrdBuffer->GetHandle().getDevicePtr();
							radianceHgData.indices = cudaTriIndBuffer->GetHandle().getDevicePtr();
							radianceHgData.diffuseTex = this->m_Impl->m_TextureManager.lock()->GetAsset(material.GetString("diffTex")).GetGpuComponent<rtlib::ext::resources::CUDATextureImage2DComponent<uchar4>>("CUDATexture")->GetHandle().getHandle();
							radianceHgData.specularTex = this->m_Impl->m_TextureManager.lock()->GetAsset(material.GetString("specTex")).GetGpuComponent<rtlib::ext::resources::CUDATextureImage2DComponent<uchar4>>("CUDATexture")->GetHandle().getHandle();
							radianceHgData.emissionTex = this->m_Impl->m_TextureManager.lock()->GetAsset(material.GetString("emitTex")).GetGpuComponent<rtlib::ext::resources::CUDATextureImage2DComponent<uchar4>>("CUDATexture")->GetHandle().getHandle();
							radianceHgData.diffuse = material.GetFloat3As<float3>("diffCol");
							radianceHgData.specular = material.GetFloat3As<float3>("specCol");
							radianceHgData.emission = material.GetFloat3As<float3>("emitCol");
							radianceHgData.shinness = material.GetFloat1("shinness");
							radianceHgData.transmit = material.GetFloat3As<float3>("tranCol");
							radianceHgData.refrInd = material.GetFloat1("refrIndx");
						}
						if (material.GetString("name") == "light")
						{
							this->m_Impl->m_LightHgRecIndex = RAY_TYPE_COUNT * sbtOffset + RAY_TYPE_COUNT * i + RAY_TYPE_RADIANCE;
						}
						std::string typeString = test24::SpecifyMaterialType(material);
						if (typeString == "Phong" || typeString == "Diffuse")
						{
							typeString += ".Guiding.Default";
						}
						this->m_Impl->m_HGRecordBuffers.cpuHandle[RAY_TYPE_COUNT * sbtOffset + RAY_TYPE_COUNT * i + RAY_TYPE_RADIANCE ] = this->m_Impl->m_HGProgramGroups[std::string("Guide.Radiance.") + typeString].getSBTRecord<HitgroupData>(radianceHgData);
						this->m_Impl->m_HGRecordBuffers.cpuHandle[RAY_TYPE_COUNT * sbtOffset + RAY_TYPE_COUNT * i + RAY_TYPE_OCCLUSION] = this->m_Impl->m_HGProgramGroups["Guide.Occluded"].getSBTRecord<HitgroupData>({});
					}
					sbtOffset += mesh->GetUniqueResource()->materials.size();
				}
			}
		}
	}
	this->m_Impl->m_HGRecordBuffers.Upload();
	this->m_Impl->m_ShaderBindingTable.raygenRecord = reinterpret_cast<CUdeviceptr>(this->m_Impl->m_RGRecordBuffer.gpuHandle.getDevicePtr());
	this->m_Impl->m_ShaderBindingTable.missRecordBase = reinterpret_cast<CUdeviceptr>(this->m_Impl->m_MSRecordBuffers.gpuHandle.getDevicePtr());
	this->m_Impl->m_ShaderBindingTable.missRecordCount = this->m_Impl->m_MSRecordBuffers.cpuHandle.size();
	this->m_Impl->m_ShaderBindingTable.missRecordStrideInBytes = sizeof(MissRecord);
	this->m_Impl->m_ShaderBindingTable.hitgroupRecordBase = reinterpret_cast<CUdeviceptr>(this->m_Impl->m_HGRecordBuffers.gpuHandle.getDevicePtr());
	this->m_Impl->m_ShaderBindingTable.hitgroupRecordCount = this->m_Impl->m_HGRecordBuffers.cpuHandle.size();
	this->m_Impl->m_ShaderBindingTable.hitgroupRecordStrideInBytes = sizeof(HitGRecord);
}

void Test24GuidePathOPXTracer::InitLaunchParams()
{
	auto tlas = this->m_Impl->m_TopLevelAS.lock();
	this->m_Impl->m_Params.Alloc(1);
	this->m_Impl->m_Params.cpuHandle[0].gasHandle       = tlas->GetHandle();
	this->m_Impl->m_Params.cpuHandle[0].width           = this->m_Impl->m_Framebuffer.lock()->GetWidth();
	this->m_Impl->m_Params.cpuHandle[0].height          = this->m_Impl->m_Framebuffer.lock()->GetHeight();
	this->m_Impl->m_Params.cpuHandle[0].sdTree          = this->m_Impl->m_STree->GetGpuHandle();
	this->m_Impl->m_Params.cpuHandle[0].light           = this->m_Impl->GetMeshLightList();
	this->m_Impl->m_Params.cpuHandle[0].accumBuffer     = this->m_Impl->m_Framebuffer.lock()->GetComponent<test::RTCUDABufferFBComponent<float3>>("RAccum")->GetHandle().getDevicePtr();
	this->m_Impl->m_Params.cpuHandle[0].frameBuffer     = nullptr;
	this->m_Impl->m_Params.cpuHandle[0].seedBuffer      = this->m_Impl->m_SeedBuffer.getDevicePtr();
	this->m_Impl->m_Params.cpuHandle[0].maxTraceDepth   = this->m_Impl->m_MaxTraceDepth;
	this->m_Impl->m_Params.cpuHandle[0].samplePerLaunch = GetVariables()->GetUInt32("SamplePerLaunch");
	this->m_Impl->m_Params.Upload();
}

void Test24GuidePathOPXTracer::FreeLight()
{
}

void Test24GuidePathOPXTracer::InitSdTree()
{
	auto worldAABB = rtlib::utils::AABB();
	for (auto& instanceSet : m_Impl->m_TopLevelAS.lock()->GetInstanceSets()) {
		for (auto& gasHandle : instanceSet->baseGASHandles)
		{
			for (auto& mesh : gasHandle->GetMeshes())
			{
				for (auto& index : mesh->GetUniqueResource()->triIndBuffer)
				{
					worldAABB.Update(mesh->GetSharedResource()->vertexBuffer[index.x]);
					worldAABB.Update(mesh->GetSharedResource()->vertexBuffer[index.y]);
					worldAABB.Update(mesh->GetSharedResource()->vertexBuffer[index.z]);
				}
			}
		}
		
	}

	m_Impl->m_STree = std::make_shared<RTSTreeWrapper>(worldAABB.min, worldAABB.max, 20);
	m_Impl->m_STree->Upload();

}

void Test24GuidePathOPXTracer::FreeSdTree()
{
	m_Impl->m_STree->Clear();
	m_Impl->m_STree.reset();
}

void Test24GuidePathOPXTracer::FreePipeline()
{
	this->m_Impl->m_RGProgramGroups    = {};
	this->m_Impl->m_HGProgramGroups    = {};
	this->m_Impl->m_MSProgramGroups    = {};
	this->m_Impl->m_Pipeline = {};
}

void Test24GuidePathOPXTracer::FreeShaderBindingTable()
{
	this->m_Impl->m_ShaderBindingTable = {};
	this->m_Impl->m_RGRecordBuffer.Reset();
	this->m_Impl->m_MSRecordBuffers.Reset();
	this->m_Impl->m_HGRecordBuffers.Reset();
}

void Test24GuidePathOPXTracer::FreeLaunchParams()
{
	this->m_Impl->m_Params.Reset();
}

bool Test24GuidePathOPXTracer::OnLaunchBegin(int width, int height, UserData* pUserData)
{
	if (GetVariables()->GetBool("Started"))
	{
		auto sampleForBudget = GetVariables()->GetUInt32("SampleForBudget");
		auto samplePerLaunch = GetVariables()->GetUInt32("SamplePerLaunch");
		GetVariables()->SetUInt32("SamplePerAll", 0);
		GetVariables()->SetUInt32("SamplePerTmp", 0);
		GetVariables()->SetUInt32("CurIteration", 0);
		GetVariables()->SetBool("Launched", true);
		GetVariables()->SetBool( "Started", false);
		this->m_Impl->m_SampleForRemain = ((sampleForBudget - 1 + samplePerLaunch) / samplePerLaunch) * samplePerLaunch;
		this->m_Impl->m_SampleForPass   = 0;
		std::random_device rd;
		std::mt19937 mt(rd());
		std::vector<unsigned int> seedData(width * height);
		std::generate(std::begin(seedData), std::end(seedData), mt);
		this->m_Impl->m_SeedBuffer.upload(seedData);
		this->m_Impl->m_STree->Clear();
		this->m_Impl->m_STree->Upload();
	}
	if (!GetVariables()->GetBool("Launched")) {
		return false;
	}
	auto curIteration    = GetVariables()->GetUInt32("CurIteration");
	auto samplePerAll    = GetVariables()->GetUInt32("SamplePerAll");
	auto samplePerTmp    = GetVariables()->GetUInt32("SamplePerTmp");
	auto samplePerLaunch = GetVariables()->GetUInt32("SamplePerLaunch");
	auto sampleForBudget = GetVariables()->GetUInt32("SampleForBudget");
	auto ratioForBudget  = GetVariables()->GetFloat1("RatioForBudget");
	if (samplePerTmp == 0)
	{
		//CurIteration > 0 -> Reset
		this->m_Impl->m_SampleForRemain = this->m_Impl->m_SampleForRemain - this->m_Impl->m_SampleForPass;
		this->m_Impl->m_SampleForPass = std::min<uint32_t>(this->m_Impl->m_SampleForRemain, (1 << curIteration) * samplePerLaunch);
		if ((this->m_Impl->m_SampleForRemain - this->m_Impl->m_SampleForPass < 2 * this->m_Impl->m_SampleForPass) || (samplePerAll >= ratioForBudget * static_cast<float>(sampleForBudget)))
		{
			std::cout << "Final: this->m_Impl->m_SamplePerAll=" << samplePerAll << std::endl;
			this->m_Impl->m_SampleForPass = this->m_Impl->m_SampleForRemain;
		}
		/*Remain>Pass -> Not Final Iteration*/
		if (this->m_Impl->m_SampleForRemain > this->m_Impl->m_SampleForPass)
		{
			this->m_Impl->m_STree->Download();
			this->m_Impl->m_STree->Reset(curIteration, samplePerLaunch);
			this->m_Impl->m_STree->Upload();
		}
	}
	std::cout << "CurIteration: " << curIteration << " SamplePerTmp: " << samplePerTmp << std::endl;
	return true;
}

void Test24GuidePathOPXTracer::OnLaunchExecute(int width, int height, UserData* pUserData)
{
	auto curIteration      = GetVariables()->GetUInt32("CurIteration");
	auto iterationForBuilt = GetVariables()->GetUInt32("IterationForBuilt");
	auto samplePerAll      = GetVariables()->GetUInt32("SamplePerAll");
	auto samplePerTmp      = GetVariables()->GetUInt32("SamplePerTmp");
	auto samplePerLaunch   = GetVariables()->GetUInt32("SamplePerLaunch");

	this->m_Impl->m_Params.cpuHandle[0].width           = width;
	this->m_Impl->m_Params.cpuHandle[0].height          = height;
	this->m_Impl->m_Params.cpuHandle[0].sdTree          = this->m_Impl->m_STree->GetGpuHandle();
	this->m_Impl->m_Params.cpuHandle[0].accumBuffer     = this->m_Impl->m_Framebuffer.lock()->GetComponent<test::RTCUDABufferFBComponent<float3>>("RAccum")->GetHandle().getDevicePtr();
	this->m_Impl->m_Params.cpuHandle[0].frameBuffer     = this->m_Impl->m_Framebuffer.lock()->GetComponent<test::RTCUGLBufferFBComponent<uchar4>>("RFrame")->GetHandle().map();
	this->m_Impl->m_Params.cpuHandle[0].seedBuffer      = this->m_Impl->m_SeedBuffer.getDevicePtr();
	this->m_Impl->m_Params.cpuHandle[0].isBuilt         = curIteration > iterationForBuilt;
	this->m_Impl->m_Params.cpuHandle[0].isFinal         = this->m_Impl->m_SampleForPass >= this->m_Impl->m_SampleForRemain;
	this->m_Impl->m_Params.cpuHandle[0].samplePerALL    = samplePerAll;
	this->m_Impl->m_Params.cpuHandle[0].samplePerLaunch = samplePerLaunch;
	cudaMemcpyAsync(this->m_Impl->m_Params.gpuHandle.getDevicePtr(), &this->m_Impl->m_Params.cpuHandle[0], sizeof(RayTraceParams), cudaMemcpyHostToDevice, pUserData->stream);
	//cudaMemcpy(this->m_Impl->m_Params.gpuHandle.getDevicePtr(), &this->m_Impl->m_Params.cpuHandle[0], sizeof(RayTraceParams), cudaMemcpyHostToDevice);
	this->m_Impl->m_Pipeline.launch(pUserData->stream, this->m_Impl->m_Params.gpuHandle.getDevicePtr(), this->m_Impl->m_ShaderBindingTable, width, height, 1);
	auto* parmams = this->m_Impl->m_Params.cpuHandle.data();
	if (pUserData->isSync)
	{
		cuStreamSynchronize(pUserData->stream);
	}
	this->m_Impl->m_Framebuffer.lock()->GetComponent<test::RTCUGLBufferFBComponent<uchar4>>("RFrame")->GetHandle().unmap();

	samplePerAll += samplePerLaunch;
	samplePerTmp += samplePerLaunch;
	this->m_Impl->m_Params.cpuHandle[0].samplePerALL = samplePerAll;

	GetVariables()->SetUInt32("SamplePerAll", samplePerAll);
	GetVariables()->SetUInt32("SamplePerTmp", samplePerTmp);
}

void Test24GuidePathOPXTracer::OnLaunchEnd(int width, int height, UserData* pUserData)
{
	auto samplePerAll    = GetVariables()->GetUInt32("SamplePerAll");
	auto samplePerTmp    = GetVariables()->GetUInt32("SamplePerTmp");
	auto sampleForBudget = GetVariables()->GetUInt32("SampleForBudget");
	auto curIteration    = GetVariables()->GetUInt32("CurIteration");

	if (samplePerTmp >= this->m_Impl->m_SampleForPass)
	{
		this->m_Impl->m_STree->Download();
		this->m_Impl->m_STree->Build();
		this->m_Impl->m_STree->Upload();

		curIteration++;
		GetVariables()->SetUInt32("SamplePerTmp", 0);
		GetVariables()->SetUInt32("CurIteration", curIteration);
	}
	if (samplePerAll >= sampleForBudget)
	{
		GetVariables()->SetBool("Launched", false);
		GetVariables()->SetBool("Started" , false);
		GetVariables()->SetUInt32("SamplePerAll",0);
		pUserData->finished = true;
	}
	else {
		pUserData->finished = false;
	}
}

void Test24GuidePathOPXTracer::FreeFrameResources()
{
	this->m_Impl->m_SeedBuffer.reset();
}

