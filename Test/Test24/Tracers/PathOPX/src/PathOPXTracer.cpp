#include "../include/PathOPXTracer.h"
#include <RayTrace.h>
#include <fstream>
#include <random>
using namespace std::string_literals;
using namespace test24_path;
inline auto SpecifyMaterialType(const rtlib::ext::VariableMap& material) -> std::string
{
	auto emitCol = material.GetFloat3As<float3>("emitCol");
	auto specCol = material.GetFloat3As<float3>("specCol");
	auto tranCol = material.GetFloat3As<float3>("tranCol");
	auto refrIndx = material.GetFloat1("refrIndx");
	auto shinness = material.GetFloat1("shinness");
	auto illum = material.GetUInt32("illum");
	if (illum == 7)
	{
		return "Refraction";
	}
	else if (emitCol.x + emitCol.y + emitCol.z > 0.0f)
	{
		return "Emission";
	}
	else
	{
		return "Diffuse";
	}
};
using RayGRecord = rtlib::SBTRecord<RayGenData>;
using MissRecord = rtlib::SBTRecord<MissData>;
using HitGRecord = rtlib::SBTRecord<HitgroupData>;
// RTTracer ����Čp������܂���
struct Test24PathOPXTracer::Impl {
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
		m_MaxTraceDepth{ maxTraceDepth }{}
	~Impl() {
	
	}

	auto GetMeshLightList()const noexcept -> MeshLightList
	{
		MeshLightList list;
		list.data  = m_MeshLights.gpuHandle.getDevicePtr();
		list.count = m_MeshLights.cpuHandle.size();
		return list;
	}

	ContextPtr m_Context;
	FramebufferPtr m_Framebuffer;
	CameraControllerPtr m_CameraController;
	TextureAssetManager m_TextureManager;
	rtlib::ext::IASHandlePtr m_TopLevelAS;
	const std::vector<rtlib::ext::VariableMap>& m_Materials;
	const float3& m_BgLightColor;
	const unsigned int& m_EventFlags;
	const unsigned int& m_MaxTraceDepth;
	rtlib::ext::Camera m_Camera = {};
	OptixShaderBindingTable m_ShaderBindingTable = {};
	Pipeline m_Pipeline = {};
	ModuleMap m_Modules = {};
	RGProgramGroupMap m_RGProgramGroups = {};
	MSProgramGroupMap m_MSProgramGroups = {};
	HGProgramGroupMap m_HGProgramGroups = {};
	rtlib::CUDAUploadBuffer<RayGRecord> m_RGRecordBuffer = {};
	rtlib::CUDAUploadBuffer<MissRecord> m_MSRecordBuffers = {};
	rtlib::CUDAUploadBuffer<HitGRecord> m_HGRecordBuffers = {};
	rtlib::CUDAUploadBuffer<RayTraceParams> m_Params = {};
	rtlib::CUDABuffer<unsigned int> m_SeedBuffer = {};
	rtlib::CUDAUploadBuffer<MeshLight> m_MeshLights = {};
	unsigned int m_LightHgRecIndex = 0;
	unsigned int m_SamplePerAll = 0;
};

Test24PathOPXTracer::Test24PathOPXTracer(
	ContextPtr context, 
	FramebufferPtr framebuffer, 
	CameraControllerPtr cameraController, 
	TextureAssetManager textureManager, 
	rtlib::ext::IASHandlePtr topLevelAS, 
	const std::vector<rtlib::ext::VariableMap>& materials, 
	const float3& bgLightColor,
	const unsigned int& eventFlags,
	const unsigned int& maxTraceDepth) :test::RTTracer() {
	m_Impl = std::make_unique<Test24PathOPXTracer::Impl>(
		context,
		framebuffer,
		cameraController,
		textureManager,
		topLevelAS,
		materials,
		bgLightColor,
		eventFlags,
		maxTraceDepth);
}

void Test24PathOPXTracer::Initialize()
{
	this->InitFrameResources();
	this->InitPipeline();
	this->InitShaderBindingTable();
	this->InitLight();
	this->InitLaunchParams();
}

 void Test24PathOPXTracer::Launch(int width, int height, void* pdata)
{

	 UserData* pUserData = (UserData*)pdata;
	 if (!pUserData)
	 {
		 return;
	 }
	 if (width != m_Impl->m_Framebuffer->GetWidth() || height != m_Impl->m_Framebuffer->GetHeight()) {
		 return;
	 }
	this->m_Impl->m_Params.cpuHandle[0].width       = width;
	this->m_Impl->m_Params.cpuHandle[0].height      = height;
	this->m_Impl->m_Params.cpuHandle[0].accumBuffer = m_Impl->m_Framebuffer->GetComponent<test::RTCUDABufferFBComponent<float3>>("RAccum")->GetHandle().getDevicePtr();
	this->m_Impl->m_Params.cpuHandle[0].frameBuffer = m_Impl->m_Framebuffer->GetComponent<test::RTCUGLBufferFBComponent<uchar4>>("RFrame")->GetHandle().map();
	this->m_Impl->m_Params.cpuHandle[0].seedBuffer  = this->m_Impl->m_SeedBuffer.getDevicePtr();
	this->m_Impl->m_Params.cpuHandle[0].isBuilt     = false;
	this->m_Impl->m_Params.cpuHandle[0].samplePerLaunch = pUserData->samplePerLaunch;
	cudaMemcpyAsync(this->m_Impl->m_Params.gpuHandle.getDevicePtr(), &this->m_Impl->m_Params.cpuHandle[0], sizeof(RayTraceParams), cudaMemcpyHostToDevice, pUserData->stream);
	this->m_Impl->m_Pipeline.launch(pUserData->stream, this->m_Impl->m_Params.gpuHandle.getDevicePtr(), this->m_Impl->m_ShaderBindingTable, width, height, 1);
	if (pUserData->isSync)
	{
		RTLIB_CU_CHECK(cuStreamSynchronize(pUserData->stream));
	}
	m_Impl->m_Framebuffer->GetComponent<test::RTCUGLBufferFBComponent<uchar4>>("RFrame")->GetHandle().unmap();
	this->m_Impl->m_Params.cpuHandle[0].samplePerALL += this->m_Impl->m_Params.cpuHandle[0].samplePerLaunch;
	this->m_Impl->m_SamplePerAll = pUserData->samplePerAll = this->m_Impl->m_Params.cpuHandle[0].samplePerALL;
}

 void Test24PathOPXTracer::CleanUp()
{
	this->FreePipeline();
	this->FreeShaderBindingTable();
	this->FreeLight();
	this->FreeLaunchParams();
	this->FreeFrameResources();
	this->m_Impl->m_LightHgRecIndex = 0;
}

 void Test24PathOPXTracer::Update()
{
	 if ((this->m_Impl->m_EventFlags & TEST24_EVENT_FLAG_UPDATE_CAMERA) == TEST24_EVENT_FLAG_UPDATE_CAMERA)
	{
		float aspect = static_cast<float>(m_Impl->m_Framebuffer->GetWidth()) / static_cast<float>(m_Impl->m_Framebuffer->GetHeight());
		this->m_Impl->m_Camera = this->m_Impl->m_CameraController->GetCamera(aspect);
		this->m_Impl->m_RGRecordBuffer.cpuHandle[0].data.eye = this->m_Impl->m_Camera.getEye();
		auto [u, v, w] = this->m_Impl->m_Camera.getUVW();
		this->m_Impl->m_RGRecordBuffer.cpuHandle[0].data.u = u;
		this->m_Impl->m_RGRecordBuffer.cpuHandle[0].data.v = v;
		this->m_Impl->m_RGRecordBuffer.cpuHandle[0].data.w = w;
		this->m_Impl->m_RGRecordBuffer.Upload();
	}
	 if ((this->m_Impl->m_EventFlags & TEST24_EVENT_FLAG_FLUSH_FRAME) == TEST24_EVENT_FLAG_FLUSH_FRAME)
	{
		std::vector<float3> zeroAccumValues(this->m_Impl->m_Framebuffer->GetWidth() * this->m_Impl->m_Framebuffer->GetHeight(),make_float3(0.0f));
		cudaMemcpy(m_Impl->m_Framebuffer->GetComponent<test::RTCUDABufferFBComponent<float3>>("RAccum")->GetHandle().getDevicePtr(), zeroAccumValues.data(),sizeof(float3)* this->m_Impl->m_Framebuffer->GetWidth() * this->m_Impl->m_Framebuffer->GetHeight(),cudaMemcpyHostToDevice);
		this->m_Impl->m_Params.cpuHandle[0].maxTraceDepth = this->m_Impl->m_MaxTraceDepth;
		this->m_Impl->m_Params.cpuHandle[0].samplePerALL  = 0;
		this->m_Impl->m_SamplePerAll                      = 0;
	}
	bool shouldRegen = ((this->m_Impl->m_SamplePerAll + this->m_Impl->m_Params.cpuHandle[0].samplePerLaunch) / 1024 != this->m_Impl->m_SamplePerAll / 1024);
	if (((this->m_Impl->m_EventFlags & TEST24_EVENT_FLAG_RESIZE_FRAME) == TEST24_EVENT_FLAG_RESIZE_FRAME) || shouldRegen)
	{
		std::cout << "Regen!\n";
		std::vector<unsigned int> seedData(this->m_Impl->m_Framebuffer->GetWidth() * this->m_Impl->m_Framebuffer->GetHeight());
		std::random_device rd;
		std::mt19937 mt(rd());
		std::generate(seedData.begin(), seedData.end(), mt);
		this->m_Impl->m_SeedBuffer.resize(this->m_Impl->m_Framebuffer->GetWidth() * this->m_Impl->m_Framebuffer->GetHeight());
		this->m_Impl->m_SeedBuffer.upload(seedData);
	}
	if ((this->m_Impl->m_EventFlags & TEST24_EVENT_FLAG_UPDATE_LIGHT) == TEST24_EVENT_FLAG_UPDATE_LIGHT)
	{
		auto lightColor = make_float4(this->m_Impl->m_BgLightColor, 1.0f);
		this->m_Impl->m_MSRecordBuffers.cpuHandle[RAY_TYPE_RADIANCE].data.bgColor = lightColor;
		RTLIB_CUDA_CHECK(cudaMemcpy(this->m_Impl->m_MSRecordBuffers.gpuHandle.getDevicePtr() + RAY_TYPE_RADIANCE, &this->m_Impl->m_MSRecordBuffers.cpuHandle[RAY_TYPE_RADIANCE], sizeof(MissRecord), cudaMemcpyHostToDevice));
	}
}

 Test24PathOPXTracer::~Test24PathOPXTracer() {
	 m_Impl.reset();
 }

 void Test24PathOPXTracer::InitLight()
 {
	 auto ChooseNEE = [](const rtlib::ext::MeshPtr& mesh)->bool {
		 return (mesh->GetUniqueResource()->triIndBuffer.Size() < 200 || mesh->GetUniqueResource()->triIndBuffer.Size() > 230);
	 };
	 auto lightGASHandle = m_Impl->m_TopLevelAS->GetInstanceSets()[0]->GetInstance(1).baseGASHandle;
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
			 if (!m_Impl->m_TextureManager->GetAsset(m_Impl->m_Materials[mesh->GetUniqueResource()->materials[0]].GetString("emitTex")).HasGpuComponent("CUDATexture")) {
				 this->m_Impl->m_TextureManager->GetAsset(m_Impl->m_Materials[mesh->GetUniqueResource()->materials[0]].GetString("emitTex")).AddGpuComponent<rtlib::ext::resources::CUDATextureImage2DComponent<uchar4>>("CUDATexture");
			 }
			 meshLight.emission    = m_Impl->m_Materials[mesh->GetUniqueResource()->materials[0]].GetFloat3As<float3>("emitCol");
			 meshLight.emissionTex = m_Impl->m_TextureManager->GetAsset(m_Impl->m_Materials[mesh->GetUniqueResource()->materials[0]].GetString("emitTex")).GetGpuComponent<rtlib::ext::resources::CUDATextureImage2DComponent<uchar4>>("CUDATexture")->GetHandle().getHandle();
			 meshLight.vertices    = mesh->GetSharedResource()->vertexBuffer.GetGpuComponent<rtlib::ext::resources::CUDABufferComponent<float3>>("CUDA")->GetHandle().getDevicePtr();
			 meshLight.normals     = mesh->GetSharedResource()->normalBuffer.GetGpuComponent<rtlib::ext::resources::CUDABufferComponent<float3>>("CUDA")->GetHandle().getDevicePtr();
			 meshLight.texCoords   = mesh->GetSharedResource()->texCrdBuffer.GetGpuComponent<rtlib::ext::resources::CUDABufferComponent<float2>>("CUDA")->GetHandle().getDevicePtr();
			 meshLight.indices     = mesh->GetUniqueResource()->triIndBuffer.GetGpuComponent<rtlib::ext::resources::CUDABufferComponent<uint3>>("CUDA")->GetHandle().getDevicePtr();
			 meshLight.indCount    = mesh->GetUniqueResource()->triIndBuffer.Size();
			 m_Impl->m_MeshLights.cpuHandle.push_back(meshLight);
		 }
	 }
	 m_Impl->m_MeshLights.Alloc();
	 m_Impl->m_MeshLights.Upload();
 }

 void Test24PathOPXTracer::InitPipeline()
{
	auto rayTracePtxFile = std::ifstream(TEST_TEST24_PATH_OPX_CUDA_PATH "/RayTrace.ptx", std::ios::binary);
	if (!rayTracePtxFile.is_open())
		throw std::runtime_error("Failed To Load RayTrace.ptx!");
	auto rayTracePtxData = std::string((std::istreambuf_iterator<char>(rayTracePtxFile)), (std::istreambuf_iterator<char>()));
	rayTracePtxFile.close();

	auto traceCompileOptions = OptixPipelineCompileOptions{};
	{
		traceCompileOptions.pipelineLaunchParamsVariableName = "params";
		traceCompileOptions.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_TRIANGLE;
		traceCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
		traceCompileOptions.usesMotionBlur = false;
		traceCompileOptions.numAttributeValues = 3;
		traceCompileOptions.numPayloadValues = 8;
	}
	auto traceLinkOptions = OptixPipelineLinkOptions{};
	{
#ifndef NDEBUG
		traceLinkOptions.debugLevel = static_cast<OptixCompileDebugLevel>(rtlib::OPXCompileDebugLevel::Minimal);
#else
		traceLinkOptions.debugLevel = static_cast<OptixCompileDebugLevel>(rtlib::OPXCompileDebugLevel::None);
#endif
		traceLinkOptions.maxTraceDepth = 2;
	}
	auto traceModuleOptions = OptixModuleCompileOptions{};
	{
#ifndef NDEBUG
		traceModuleOptions.debugLevel = static_cast<OptixCompileDebugLevel>(rtlib::OPXCompileDebugLevel::Minimal);
		traceModuleOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
#else
		traceModuleOptions.debugLevel = static_cast<OptixCompileDebugLevel>(rtlib::OPXCompileDebugLevel::None);
		traceModuleOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
#endif
	}
	traceModuleOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
	this->m_Impl->m_Pipeline = this->m_Impl->m_Context->GetOPX7Handle()->createPipeline(traceCompileOptions);
	this->m_Impl->m_Modules["RayTrace"] = this->m_Impl->m_Pipeline.createModule(rayTracePtxData, traceModuleOptions);
	this->m_Impl->m_RGProgramGroups["Trace.Default"] = this->m_Impl->m_Pipeline.createRaygenPG({ this->m_Impl->m_Modules["RayTrace"], RTLIB_RAYGEN_PROGRAM_STR(def) });
	this->m_Impl->m_MSProgramGroups["Trace.Radiance"] = this->m_Impl->m_Pipeline.createMissPG({ this->m_Impl->m_Modules["RayTrace"], RTLIB_MISS_PROGRAM_STR(radiance) });
	this->m_Impl->m_MSProgramGroups["Trace.Occluded"] = this->m_Impl->m_Pipeline.createMissPG({ this->m_Impl->m_Modules["RayTrace"], RTLIB_MISS_PROGRAM_STR(occluded) });
	this->m_Impl->m_HGProgramGroups["Trace.Radiance.Diffuse.Default"] = this->m_Impl->m_Pipeline.createHitgroupPG({ this->m_Impl->m_Modules["RayTrace"], RTLIB_CLOSESTHIT_PROGRAM_STR(radiance_for_diffuse_def) }, {}, {});
	this->m_Impl->m_HGProgramGroups["Trace.Radiance.Phong.Default"] = this->m_Impl->m_Pipeline.createHitgroupPG({ this->m_Impl->m_Modules["RayTrace"], RTLIB_CLOSESTHIT_PROGRAM_STR(radiance_for_phong_def) }, {}, {});
	this->m_Impl->m_HGProgramGroups["Trace.Radiance.Emission"] = this->m_Impl->m_Pipeline.createHitgroupPG({ this->m_Impl->m_Modules["RayTrace"], RTLIB_CLOSESTHIT_PROGRAM_STR(radiance_for_emission) }, {}, {});
	this->m_Impl->m_HGProgramGroups["Trace.Radiance.Specular"] = this->m_Impl->m_Pipeline.createHitgroupPG({ this->m_Impl->m_Modules["RayTrace"], RTLIB_CLOSESTHIT_PROGRAM_STR(radiance_for_specular) }, {}, {});
	this->m_Impl->m_HGProgramGroups["Trace.Radiance.Refraction"] = this->m_Impl->m_Pipeline.createHitgroupPG({ this->m_Impl->m_Modules["RayTrace"], RTLIB_CLOSESTHIT_PROGRAM_STR(radiance_for_refraction) }, {}, {});
	this->m_Impl->m_HGProgramGroups["Trace.Occluded"] = this->m_Impl->m_Pipeline.createHitgroupPG({ this->m_Impl->m_Modules["RayTrace"], RTLIB_CLOSESTHIT_PROGRAM_STR(occluded) }, {}, {});
	this->m_Impl->m_Pipeline.link(traceLinkOptions);
}

 void Test24PathOPXTracer::InitFrameResources()
{
	std::vector<unsigned int> seedData(this->m_Impl->m_Framebuffer->GetWidth() * this->m_Impl->m_Framebuffer->GetHeight());
	std::random_device rd;
	std::mt19937 mt(rd());
	std::generate(seedData.begin(), seedData.end(), mt);
	this->m_Impl->m_SeedBuffer = rtlib::CUDABuffer<uint32_t>(seedData);
}

 void Test24PathOPXTracer::InitShaderBindingTable()
{
	float aspect = static_cast<float>(m_Impl->m_Framebuffer->GetWidth()) / static_cast<float>(m_Impl->m_Framebuffer->GetHeight());
	auto tlas = this->m_Impl->m_TopLevelAS;
	this->m_Impl->m_Camera = this->m_Impl->m_CameraController->GetCamera(aspect);
	auto& materials = this->m_Impl->m_Materials;
	this->m_Impl->m_RGRecordBuffer.Alloc(1);
	this->m_Impl->m_RGRecordBuffer.cpuHandle[0] = this->m_Impl->m_RGProgramGroups["Trace.Default"].getSBTRecord<RayGenData>();
	this->m_Impl->m_RGRecordBuffer.cpuHandle[0].data.eye = this->m_Impl->m_Camera.getEye();
	auto [u, v, w] = this->m_Impl->m_Camera.getUVW();
	this->m_Impl->m_RGRecordBuffer.cpuHandle[0].data.u = u;
	this->m_Impl->m_RGRecordBuffer.cpuHandle[0].data.v = v;
	this->m_Impl->m_RGRecordBuffer.cpuHandle[0].data.w = w;
	this->m_Impl->m_RGRecordBuffer.Upload();
	this->m_Impl->m_MSRecordBuffers.Alloc(RAY_TYPE_COUNT);
	this->m_Impl->m_MSRecordBuffers.cpuHandle[RAY_TYPE_RADIANCE] = this->m_Impl->m_MSProgramGroups["Trace.Radiance"].getSBTRecord<MissData>();
	this->m_Impl->m_MSRecordBuffers.cpuHandle[RAY_TYPE_RADIANCE].data.bgColor = make_float4(this->m_Impl->m_BgLightColor, 1.0f);
	this->m_Impl->m_MSRecordBuffers.cpuHandle[RAY_TYPE_OCCLUSION] = this->m_Impl->m_MSProgramGroups["Trace.Occluded"].getSBTRecord<MissData>();
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
						if (!this->m_Impl->m_TextureManager->GetAsset(material.GetString("diffTex")).HasGpuComponent("CUDATexture")) {
							this->m_Impl->m_TextureManager->GetAsset(material.GetString("diffTex")).AddGpuComponent<rtlib::ext::resources::CUDATextureImage2DComponent<uchar4>>("CUDATexture");
						}
						if (!this->m_Impl->m_TextureManager->GetAsset(material.GetString("specTex")).HasGpuComponent("CUDATexture")) {
							this->m_Impl->m_TextureManager->GetAsset(material.GetString("specTex")).AddGpuComponent<rtlib::ext::resources::CUDATextureImage2DComponent<uchar4>>("CUDATexture");
						}
						if (!this->m_Impl->m_TextureManager->GetAsset(material.GetString("emitTex")).HasGpuComponent("CUDATexture")) {
							this->m_Impl->m_TextureManager->GetAsset(material.GetString("emitTex")).AddGpuComponent<rtlib::ext::resources::CUDATextureImage2DComponent<uchar4>>("CUDATexture");
						}
						HitgroupData radianceHgData = {};
						{
							radianceHgData.vertices  = cudaVertexBuffer->GetHandle().getDevicePtr();
							radianceHgData.normals   = cudaNormalBuffer->GetHandle().getDevicePtr();
							radianceHgData.texCoords = cudaTexCrdBuffer->GetHandle().getDevicePtr();
							radianceHgData.indices   = cudaTriIndBuffer->GetHandle().getDevicePtr();
							radianceHgData.diffuseTex  = this->m_Impl->m_TextureManager->GetAsset(material.GetString("diffTex")).GetGpuComponent<rtlib::ext::resources::CUDATextureImage2DComponent<uchar4>>("CUDATexture")->GetHandle().getHandle();
							radianceHgData.specularTex = this->m_Impl->m_TextureManager->GetAsset(material.GetString("specTex")).GetGpuComponent<rtlib::ext::resources::CUDATextureImage2DComponent<uchar4>>("CUDATexture")->GetHandle().getHandle();
							radianceHgData.emissionTex = this->m_Impl->m_TextureManager->GetAsset(material.GetString("emitTex")).GetGpuComponent<rtlib::ext::resources::CUDATextureImage2DComponent<uchar4>>("CUDATexture")->GetHandle().getHandle();
							radianceHgData.diffuse = material.GetFloat3As<float3>("diffCol");
							radianceHgData.specular = material.GetFloat3As<float3>("specCol");
							radianceHgData.emission = material.GetFloat3As<float3>("emitCol");
							radianceHgData.shinness = material.GetFloat1("shinness");
							radianceHgData.transmit = material.GetFloat3As<float3>("tranCol");
							radianceHgData.refrInd  = material.GetFloat1("refrIndx");
						}
						if (material.GetString("name") == "light")
						{
							this->m_Impl->m_LightHgRecIndex = RAY_TYPE_COUNT * sbtOffset + RAY_TYPE_COUNT * i + RAY_TYPE_RADIANCE;
						}
						std::string typeString = SpecifyMaterialType(material);
						if (typeString == "Phong" || typeString == "Diffuse")
						{
							typeString += ".Default";
						}
						this->m_Impl->m_HGRecordBuffers.cpuHandle[RAY_TYPE_COUNT * sbtOffset + RAY_TYPE_COUNT * i + RAY_TYPE_RADIANCE] = this->m_Impl->m_HGProgramGroups[std::string("Trace.Radiance.") + typeString].getSBTRecord<HitgroupData>(radianceHgData);
						this->m_Impl->m_HGRecordBuffers.cpuHandle[RAY_TYPE_COUNT * sbtOffset + RAY_TYPE_COUNT * i + RAY_TYPE_OCCLUSION] = this->m_Impl->m_HGProgramGroups["Trace.Occluded"].getSBTRecord<HitgroupData>({});
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

 void Test24PathOPXTracer::InitLaunchParams()
{
	auto tlas = this->m_Impl->m_TopLevelAS;
	this->m_Impl->m_Params.Alloc(1);
	this->m_Impl->m_Params.cpuHandle[0].gasHandle = tlas->GetHandle();
	this->m_Impl->m_Params.cpuHandle[0].width  = this->m_Impl->m_Framebuffer->GetWidth();
	this->m_Impl->m_Params.cpuHandle[0].height = this->m_Impl->m_Framebuffer->GetHeight();
	this->m_Impl->m_Params.cpuHandle[0].light  = this->m_Impl->GetMeshLightList();
	this->m_Impl->m_Params.cpuHandle[0].accumBuffer = m_Impl->m_Framebuffer->GetComponent<test::RTCUDABufferFBComponent<float3>>("RAccum")->GetHandle().getDevicePtr();
	this->m_Impl->m_Params.cpuHandle[0].frameBuffer = nullptr;
	this->m_Impl->m_Params.cpuHandle[0].seedBuffer  = this->m_Impl->m_SeedBuffer.getDevicePtr();
	this->m_Impl->m_Params.cpuHandle[0].isBuilt     = false;
	this->m_Impl->m_Params.cpuHandle[0].maxTraceDepth = this->m_Impl->m_MaxTraceDepth;
	this->m_Impl->m_Params.cpuHandle[0].samplePerLaunch = 1;

	this->m_Impl->m_Params.Upload();
}

 void Test24PathOPXTracer::FreeLight()
 {
 }

 void Test24PathOPXTracer::FreePipeline()
{
	 this->m_Impl->m_RGProgramGroups = {};
	 this->m_Impl->m_HGProgramGroups = {};
	 this->m_Impl->m_MSProgramGroups = {};
}

 void Test24PathOPXTracer::FreeShaderBindingTable()
{
	this->m_Impl->m_ShaderBindingTable = {};
	this->m_Impl->m_RGRecordBuffer.Reset();
	this->m_Impl->m_MSRecordBuffers.Reset();
	this->m_Impl->m_HGRecordBuffers.Reset();
}

 void Test24PathOPXTracer::FreeLaunchParams()
{
	this->m_Impl->m_Params.Reset();
}

 void Test24PathOPXTracer::FreeFrameResources()
{
	this->m_Impl->m_SeedBuffer.reset();
}
