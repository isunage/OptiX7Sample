#include "..\include\DebugOPXTracer.h"
#include <Test24DebugOPXConfig.h>
#include <RayTrace.h>
#include <unordered_map>
#include <fstream>

using RayGRecord = rtlib::SBTRecord<RayGenData>;
using MissRecord = rtlib::SBTRecord<MissData>;
using HitGRecord = rtlib::SBTRecord<HitgroupData>;
using Pipeline   = rtlib::OPXPipeline;
using ModuleMap  = std::unordered_map<std::string, rtlib::OPXModule>;
using RGProgramGroupMap = std::unordered_map<std::string, rtlib::OPXRaygenPG>;
using MSProgramGroupMap = std::unordered_map<std::string, rtlib::OPXMissPG>;
using HGProgramGroupMap = std::unordered_map<std::string, rtlib::OPXHitgroupPG>;

// RTTracer ����Čp������܂���
struct Test24DebugOPXTracer::Impl
{
	Impl(
		ContextPtr Context,
		FramebufferPtr Framebuffer,
		CameraControllerPtr CameraController,
		TextureAssetManager TextureManager,
		rtlib::ext::IASHandlePtr TopLevelAS,
		const std::vector<rtlib::ext::VariableMap>& Materials,
		const float3& BgLightColor,
		const bool& ResizedFrame,
		const bool& UpdateCamera,
		const bool& UpdateBgLight) :m_Context{ Context }, m_Framebuffer{ Framebuffer }, m_CameraController{ CameraController }, m_TextureManager{ TextureManager }, m_Materials{ Materials }, m_TopLevelAS{TopLevelAS}, m_BgLightColor{ BgLightColor }, m_ResizedFrame{ ResizedFrame }, m_UpdateCamera{ UpdateCamera }, m_UpdateBgLight{ UpdateBgLight }
	{
	}
	~Impl() {
	}
	ContextPtr m_Context;
	FramebufferPtr m_Framebuffer;
	CameraControllerPtr m_CameraController;
	TextureAssetManager m_TextureManager;
	rtlib::ext::IASHandlePtr m_TopLevelAS;
	const std::vector<rtlib::ext::VariableMap>& m_Materials;
	const float3& m_BgLightColor;
	const bool& m_ResizedFrame;
	const bool& m_UpdateCamera;
	const bool& m_UpdateBgLight;

	unsigned int m_LightHgRecIndex = 0;
	rtlib::ext::Camera m_Camera = {};
	Pipeline  m_Pipeline = {};
	ModuleMap m_Modules = {};
	RGProgramGroupMap m_RGProgramGroups = {};
	MSProgramGroupMap m_MSProgramGroups = {};
	HGProgramGroupMap m_HGProgramGroups = {};
	OptixShaderBindingTable m_ShaderBindingTable = {};
	rtlib::CUDAUploadBuffer<RayGRecord> m_RGRecordBuffer = {};
	rtlib::CUDAUploadBuffer<MissRecord> m_MSRecordBuffers = {};
	rtlib::CUDAUploadBuffer<HitGRecord> m_HGRecordBuffers = {};
	rtlib::CUDAUploadBuffer<RayDebugParams> m_Params = {};
};

Test24DebugOPXTracer::Test24DebugOPXTracer(
	ContextPtr Context, FramebufferPtr Framebuffer, CameraControllerPtr CameraController, TextureAssetManager TextureManager,
	rtlib::ext::IASHandlePtr TopLevelAS, const std::vector<rtlib::ext::VariableMap>& Materials, const float3& BgLightColor, const  bool& ResizedFrame, const bool& UpdateCamera, const bool& UpdateBgLight)
{
	m_Impl = std::make_unique<Test24DebugOPXTracer::Impl>(
		Context,Framebuffer,CameraController,TextureManager,TopLevelAS,Materials,BgLightColor, ResizedFrame, UpdateCamera,UpdateBgLight
	);
}

void Test24DebugOPXTracer::Initialize()
{
	this->InitPipeline();
	this->InitShaderBindingTable();
	this->InitLaunchParams();
}

void Test24DebugOPXTracer::Launch(int width, int height, void* userData)
{
	UserData* pUserData = (UserData*)userData;
	if (!pUserData)
	{
		return;
	}
	if (width != m_Impl->m_Framebuffer->GetWidth() || height != m_Impl->m_Framebuffer->GetHeight()) {
		return;
	}
	this->m_Impl->m_Params.cpuHandle[0].width          = width;
	this->m_Impl->m_Params.cpuHandle[0].height         = height;
	this->m_Impl->m_Params.cpuHandle[0].diffuseBuffer  = m_Impl->m_Framebuffer->GetComponent<test::RTCUGLBufferFBComponent<uchar4>>("DDiffuse")->GetHandle().map();
	this->m_Impl->m_Params.cpuHandle[0].specularBuffer = m_Impl->m_Framebuffer->GetComponent<test::RTCUGLBufferFBComponent<uchar4>>("DSpecular")->GetHandle().map();
	this->m_Impl->m_Params.cpuHandle[0].emissionBuffer = m_Impl->m_Framebuffer->GetComponent<test::RTCUGLBufferFBComponent<uchar4>>("DEmission")->GetHandle().map();
	this->m_Impl->m_Params.cpuHandle[0].transmitBuffer = m_Impl->m_Framebuffer->GetComponent<test::RTCUGLBufferFBComponent<uchar4>>("DTransmit")->GetHandle().map();
	this->m_Impl->m_Params.cpuHandle[0].normalBuffer   = m_Impl->m_Framebuffer->GetComponent<test::RTCUGLBufferFBComponent<uchar4>>("DNormal")->GetHandle().map();
	this->m_Impl->m_Params.cpuHandle[0].depthBuffer    = m_Impl->m_Framebuffer->GetComponent<test::RTCUGLBufferFBComponent<uchar4>>("DDistance")->GetHandle().map();
	this->m_Impl->m_Params.cpuHandle[0].texCoordBuffer = m_Impl->m_Framebuffer->GetComponent<test::RTCUGLBufferFBComponent<uchar4>>("DTexCoord")->GetHandle().map();
	//TODO
	this->m_Impl->m_Params.cpuHandle[0].sTreeColBuffer = m_Impl->m_Framebuffer->GetComponent<test::RTCUGLBufferFBComponent<uchar4>>("DSTreeCol")->GetHandle().map();
	cudaMemcpyAsync(this->m_Impl->m_Params.gpuHandle.getDevicePtr(), &this->m_Impl->m_Params.cpuHandle[0], sizeof(RayDebugParams), cudaMemcpyHostToDevice, pUserData->stream);
	this->m_Impl->m_Pipeline.launch(pUserData->stream, this->m_Impl->m_Params.gpuHandle.getDevicePtr(), this->m_Impl->m_ShaderBindingTable, width, height, 1);
	if (pUserData->isSync)
	{
		RTLIB_CU_CHECK(cuStreamSynchronize(pUserData->stream));
	}
	m_Impl->m_Framebuffer->GetComponent<test::RTCUGLBufferFBComponent<uchar4>>("DDiffuse")->GetHandle().unmap();
	m_Impl->m_Framebuffer->GetComponent<test::RTCUGLBufferFBComponent<uchar4>>("DSpecular")->GetHandle().unmap();
	m_Impl->m_Framebuffer->GetComponent<test::RTCUGLBufferFBComponent<uchar4>>("DEmission")->GetHandle().unmap();
	m_Impl->m_Framebuffer->GetComponent<test::RTCUGLBufferFBComponent<uchar4>>("DTransmit")->GetHandle().unmap();
	m_Impl->m_Framebuffer->GetComponent<test::RTCUGLBufferFBComponent<uchar4>>("DNormal")->GetHandle().unmap();
	m_Impl->m_Framebuffer->GetComponent<test::RTCUGLBufferFBComponent<uchar4>>("DDistance")->GetHandle().unmap();
	m_Impl->m_Framebuffer->GetComponent<test::RTCUGLBufferFBComponent<uchar4>>("DTexCoord")->GetHandle().unmap();
	m_Impl->m_Framebuffer->GetComponent<test::RTCUGLBufferFBComponent<uchar4>>("DSTreeCol")->GetHandle().unmap();
}

void Test24DebugOPXTracer::CleanUp()
{
	this->FreePipeline();
	this->FreeShaderBindingTable();
	this->FreeLaunchParams();
	this->m_Impl->m_LightHgRecIndex = 0;
}

void Test24DebugOPXTracer::Update()
{
	if (this->m_Impl->m_UpdateCamera || this->m_Impl->m_ResizedFrame)
	{
		float aspect  =(float) m_Impl->m_Framebuffer->GetWidth() / (float)m_Impl->m_Framebuffer->GetHeight();
		this->m_Impl->m_Camera = this->m_Impl->m_CameraController->GetCamera(aspect);
		this->m_Impl->m_RGRecordBuffer.cpuHandle[0].data.eye = this->m_Impl->m_Camera.getEye();
		auto [u, v, w] = this->m_Impl->m_Camera.getUVW();
		CUdeviceptr prvRayGenPtr = (CUdeviceptr)this->m_Impl->m_RGRecordBuffer.gpuHandle.getDevicePtr();
		this->m_Impl->m_RGRecordBuffer.cpuHandle[0].data.u = u;
		this->m_Impl->m_RGRecordBuffer.cpuHandle[0].data.v = v;
		this->m_Impl->m_RGRecordBuffer.cpuHandle[0].data.w = w;
		this->m_Impl->m_RGRecordBuffer.Upload();
		CUdeviceptr curRayGenPtr = (CUdeviceptr)this->m_Impl->m_RGRecordBuffer.gpuHandle.getDevicePtr();
	}
	if (this->m_Impl->m_UpdateBgLight)
	{
		auto lightColor = make_float4(this->m_Impl->m_BgLightColor, 1.0f);
		this->m_Impl->m_MSRecordBuffers.cpuHandle[RAY_TYPE_RADIANCE].data.bgColor = lightColor;
		RTLIB_CUDA_CHECK(cudaMemcpy(this->m_Impl->m_MSRecordBuffers.gpuHandle.getDevicePtr() + RAY_TYPE_RADIANCE, &this->m_Impl->m_MSRecordBuffers.cpuHandle[RAY_TYPE_RADIANCE], sizeof(MissRecord), cudaMemcpyHostToDevice));
	}
}

Test24DebugOPXTracer::~Test24DebugOPXTracer() {}

void Test24DebugOPXTracer::InitPipeline()
{
	auto rayDebugPtxFile = std::ifstream(TEST_TEST24_DEBUG_OPX_CUDA_PATH "/RayDebug.ptx", std::ios::binary);
	if (!rayDebugPtxFile.is_open())
		throw std::runtime_error("Failed To Load RayDebug.ptx!");
	auto rayDebugPtxData = std::string((std::istreambuf_iterator<char>(rayDebugPtxFile)), (std::istreambuf_iterator<char>()));
	rayDebugPtxFile.close();

	auto debugCompileOptions = OptixPipelineCompileOptions{};
	{
		debugCompileOptions.pipelineLaunchParamsVariableName = "params";
		debugCompileOptions.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_TRIANGLE;
		debugCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
		debugCompileOptions.usesMotionBlur = false;
		debugCompileOptions.numAttributeValues = 3;
		debugCompileOptions.numPayloadValues = 8;
	}
	auto debugLinkOptions = OptixPipelineLinkOptions{};
	{
#ifndef NDEBUG
		debugLinkOptions.debugLevel = static_cast<OptixCompileDebugLevel>(rtlib::OPXCompileDebugLevel::Minimal);
#else
		debugLinkOptions.debugLevel = static_cast<OptixCompileDebugLevel>(rtlib::OPXCompileDebugLevel::None);
#endif
		debugLinkOptions.maxTraceDepth = 1;
	}
	auto debugModuleOptions = OptixModuleCompileOptions{};
	{
#ifndef NDEBUG
		debugModuleOptions.debugLevel = static_cast<OptixCompileDebugLevel>(rtlib::OPXCompileDebugLevel::Minimal);
		debugModuleOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
#else
		debugModuleOptions.debugLevel = static_cast<OptixCompileDebugLevel>(rtlib::OPXCompileDebugLevel::None);
		debugModuleOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
#endif
		debugModuleOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
	}
	this->m_Impl->m_Pipeline = this->m_Impl->m_Context->GetOPX7Handle()->createPipeline(debugCompileOptions);
	this->m_Impl->m_Modules["RayDebug"] = this->m_Impl->m_Pipeline.createModule(rayDebugPtxData, debugModuleOptions);
	this->m_Impl->m_RGProgramGroups["Debug.Default"] = this->m_Impl->m_Pipeline.createRaygenPG({ this->m_Impl->m_Modules["RayDebug"], "__raygen__debug" });
	this->m_Impl->m_MSProgramGroups["Debug.Default"] = this->m_Impl->m_Pipeline.createMissPG({ this->m_Impl->m_Modules["RayDebug"], "__miss__debug" });
	this->m_Impl->m_HGProgramGroups["Debug.Default"] = this->m_Impl->m_Pipeline.createHitgroupPG({ this->m_Impl->m_Modules["RayDebug"], "__closesthit__debug" }, {}, {});
	this->m_Impl->m_Pipeline.link(debugLinkOptions);
}

void Test24DebugOPXTracer::InitShaderBindingTable()
{
	auto tlas = this->m_Impl->m_TopLevelAS;
	auto camera = this->m_Impl->m_Camera;
	auto& materials = this->m_Impl->m_Materials;
	this->m_Impl->m_RGRecordBuffer.cpuHandle.resize(1);
	this->m_Impl->m_RGRecordBuffer.cpuHandle[0] = this->m_Impl->m_RGProgramGroups["Debug.Default"].getSBTRecord<RayGenData>();
	this->m_Impl->m_RGRecordBuffer.cpuHandle[0].data.eye = camera.getEye();
	auto [u, v, w] = camera.getUVW();
	this->m_Impl->m_RGRecordBuffer.cpuHandle[0].data.u = u;
	this->m_Impl->m_RGRecordBuffer.cpuHandle[0].data.v = v;
	this->m_Impl->m_RGRecordBuffer.cpuHandle[0].data.w = w;
	this->m_Impl->m_RGRecordBuffer.Upload();
	this->m_Impl->m_MSRecordBuffers.cpuHandle.resize(RAY_TYPE_COUNT);
	this->m_Impl->m_MSRecordBuffers.cpuHandle[RAY_TYPE_RADIANCE] = this->m_Impl->m_MSProgramGroups["Debug.Default"].getSBTRecord<MissData>();
	this->m_Impl->m_MSRecordBuffers.cpuHandle[RAY_TYPE_RADIANCE].data.bgColor = make_float4(this->m_Impl->m_BgLightColor, 1.0f);
	this->m_Impl->m_MSRecordBuffers.cpuHandle[RAY_TYPE_OCCLUSION] = this->m_Impl->m_MSProgramGroups["Debug.Default"].getSBTRecord<MissData>();
	this->m_Impl->m_MSRecordBuffers.Upload();
	this->m_Impl->m_HGRecordBuffers.cpuHandle.resize(tlas->GetSbtCount() * RAY_TYPE_COUNT);
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
					auto cudaTriIndBuffer = mesh->GetUniqueResource()->triIndBuffer.GetGpuComponent<rtlib::ext::resources::CUDABufferComponent<uint3>>( "CUDA");
					for (size_t i = 0; i < mesh->GetUniqueResource()->materials.size(); ++i)
					{
						auto materialId = mesh->GetUniqueResource()->materials[i];
						auto& material = materials[materialId];
						HitgroupData radianceHgData = {};
						{
							if (!this->m_Impl->m_TextureManager->GetAsset(material.GetString("diffTex")).HasGpuComponent("CUDATexture")) {
								this->m_Impl->m_TextureManager->GetAsset(material.GetString("diffTex")).AddGpuComponent<rtlib::ext::resources::CUDATextureImage2DComponent<uchar4>>("CUDATexture");
							}
							if (!this->m_Impl->m_TextureManager->GetAsset(material.GetString("specTex")).HasGpuComponent("CUDATexture")) {
								this->m_Impl->m_TextureManager->GetAsset(material.GetString("specTex")).AddGpuComponent<rtlib::ext::resources::CUDATextureImage2DComponent<uchar4>>("CUDATexture");
							}
							if (!this->m_Impl->m_TextureManager->GetAsset(material.GetString("emitTex")).HasGpuComponent("CUDATexture")) {
								this->m_Impl->m_TextureManager->GetAsset(material.GetString("emitTex")).AddGpuComponent<rtlib::ext::resources::CUDATextureImage2DComponent<uchar4>>("CUDATexture");
							}
							radianceHgData.vertices  = cudaVertexBuffer->GetHandle().getDevicePtr();
							radianceHgData.normals   = cudaNormalBuffer->GetHandle().getDevicePtr();
							radianceHgData.texCoords = cudaTexCrdBuffer->GetHandle().getDevicePtr();
							radianceHgData.indices   = cudaTriIndBuffer->GetHandle().getDevicePtr();
							radianceHgData.diffuseTex  = this->m_Impl->m_TextureManager->GetAsset(material.GetString("diffTex")).GetGpuComponent<rtlib::ext::resources::CUDATextureImage2DComponent<uchar4>>("CUDATexture")->GetHandle().getHandle();
							radianceHgData.specularTex = this->m_Impl->m_TextureManager->GetAsset(material.GetString("specTex")).GetGpuComponent<rtlib::ext::resources::CUDATextureImage2DComponent<uchar4>>("CUDATexture")->GetHandle().getHandle();
							radianceHgData.emissionTex = this->m_Impl->m_TextureManager->GetAsset(material.GetString("emitTex")).GetGpuComponent<rtlib::ext::resources::CUDATextureImage2DComponent<uchar4>>("CUDATexture")->GetHandle().getHandle();
							radianceHgData.diffuse  = material.GetFloat3As<float3>("diffCol");
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
						this->m_Impl->m_HGRecordBuffers.cpuHandle[RAY_TYPE_COUNT * sbtOffset + RAY_TYPE_COUNT * i + RAY_TYPE_RADIANCE] = this->m_Impl->m_HGProgramGroups["Debug.Default"].getSBTRecord<HitgroupData>(radianceHgData);
						this->m_Impl->m_HGRecordBuffers.cpuHandle[RAY_TYPE_COUNT * sbtOffset + RAY_TYPE_COUNT * i + RAY_TYPE_OCCLUSION] = this->m_Impl->m_HGProgramGroups["Debug.Default"].getSBTRecord<HitgroupData>({});
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

void Test24DebugOPXTracer::InitLaunchParams()
{
	auto tlas = this->m_Impl->m_TopLevelAS;
	this->m_Impl->m_Params.cpuHandle.resize(1);
	this->m_Impl->m_Params.cpuHandle[0].gasHandle = tlas->GetHandle();
	this->m_Impl->m_Params.cpuHandle[0].width  = this->m_Impl->m_Framebuffer->GetWidth();
	this->m_Impl->m_Params.cpuHandle[0].height = this->m_Impl->m_Framebuffer->GetHeight();
	this->m_Impl->m_Params.cpuHandle[0].diffuseBuffer = nullptr;
	this->m_Impl->m_Params.cpuHandle[0].specularBuffer = nullptr;
	this->m_Impl->m_Params.cpuHandle[0].emissionBuffer = nullptr;
	this->m_Impl->m_Params.cpuHandle[0].transmitBuffer = nullptr;
	this->m_Impl->m_Params.cpuHandle[0].normalBuffer = nullptr;
	this->m_Impl->m_Params.cpuHandle[0].depthBuffer = nullptr;
	this->m_Impl->m_Params.cpuHandle[0].texCoordBuffer = nullptr;
	this->m_Impl->m_Params.cpuHandle[0].sTreeColBuffer = nullptr;
	this->m_Impl->m_Params.Upload();
}

void Test24DebugOPXTracer::FreePipeline()
{
	this->m_Impl->m_ShaderBindingTable = {};
	this->m_Impl->m_RGRecordBuffer.Reset();
	this->m_Impl->m_MSRecordBuffers.Reset();
	this->m_Impl->m_HGRecordBuffers.Reset();
}

void Test24DebugOPXTracer::FreeShaderBindingTable()
{
	this->m_Impl->m_ShaderBindingTable = {};
	this->m_Impl->m_RGRecordBuffer.Reset();
	this->m_Impl->m_MSRecordBuffers.Reset();
	this->m_Impl->m_HGRecordBuffers.Reset();
}

void Test24DebugOPXTracer::FreeLaunchParams()
{
	this->m_Impl->m_Params.Reset();
}
