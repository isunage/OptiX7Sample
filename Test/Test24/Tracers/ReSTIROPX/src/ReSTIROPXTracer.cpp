#include "..\include\ReSTIROPXTracer.h"
#include <Test24ReSTIROPXConfig.h>
#include <RayTrace.h>
#include <unordered_map>
#include <fstream>
#include <random>
using namespace test24_restir;
using RayGRecord = rtlib::SBTRecord<RayGenData>;
using MissRecord = rtlib::SBTRecord<MissData>;
using HitGRecord = rtlib::SBTRecord<HitgroupData>;
using RayGRecord2 = rtlib::SBTRecord<RayGenData>;
using MissRecord2 = rtlib::SBTRecord<MissData2>;
using HitGRecord2 = rtlib::SBTRecord<HitgroupData2>;
using Pipeline = rtlib::OPXPipeline;
using ModuleMap = std::unordered_map<std::string, rtlib::OPXModule>;
using RGProgramGroupMap = std::unordered_map<std::string, rtlib::OPXRaygenPG>;
using MSProgramGroupMap = std::unordered_map<std::string, rtlib::OPXMissPG>;
using HGProgramGroupMap = std::unordered_map<std::string, rtlib::OPXHitgroupPG>;

// RTTracer ����Čp������܂���
struct Test24ReSTIROPXTracer::Impl
{
	struct  FirstPipeline
	{
		Pipeline  m_Pipeline = {};
		ModuleMap m_Modules = {};
		RGProgramGroupMap m_RGProgramGroups = {};
		MSProgramGroupMap m_MSProgramGroups = {};
		HGProgramGroupMap m_HGProgramGroups = {};
		OptixShaderBindingTable m_ShaderBindingTable = {};
		rtlib::CUDAUploadBuffer<RayGRecord>     m_RGRecordBuffer = {};
		rtlib::CUDAUploadBuffer<MissRecord>     m_MSRecordBuffers = {};
		rtlib::CUDAUploadBuffer<HitGRecord>     m_HGRecordBuffers = {};
		rtlib::CUDAUploadBuffer<RayFirstParams> m_Params = {};
	};
	struct SecondPipeline
	{
		Pipeline  m_Pipeline = {};
		ModuleMap m_Modules = {};
		RGProgramGroupMap m_RGProgramGroups = {};
		MSProgramGroupMap m_MSProgramGroups = {};
		HGProgramGroupMap m_HGProgramGroups = {};
		OptixShaderBindingTable m_ShaderBindingTableForInit = {};
		OptixShaderBindingTable m_ShaderBindingTableForDraw = {};
		rtlib::CUDAUploadBuffer<RayGRecord2>     m_RGRecordBuffers = {};
		rtlib::CUDAUploadBuffer<MissRecord2>     m_MSRecordBuffers = {};
		rtlib::CUDAUploadBuffer<HitGRecord2>     m_HGRecordBuffers = {};
		rtlib::CUDAUploadBuffer<RaySecondParams> m_Params = {};
	};
	Impl(
		ContextPtr Context,
		FramebufferPtr Framebuffer,
		CameraControllerPtr CameraController,
		TextureAssetManager TextureManager,
		rtlib::ext::IASHandlePtr TopLevelAS,
		const std::vector<rtlib::ext::VariableMap>& Materials,
		const float3& BgLightColor,
		const unsigned int& EventFlags) :m_Context{ Context }, m_Framebuffer{ Framebuffer }, m_CameraController{ CameraController }, m_TextureManager{ TextureManager }, m_Materials{ Materials }, m_TopLevelAS{ TopLevelAS }, m_BgLightColor{ BgLightColor }, m_EventFlags{ EventFlags }
	{
	}
	~Impl() {
	}
	auto GetMeshLightList()const noexcept -> MeshLightList
	{
		MeshLightList list;
		list.data = m_MeshLights.gpuHandle.getDevicePtr();
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
	FirstPipeline       m_First;
	SecondPipeline      m_Second;
	rtlib::CUDAUploadBuffer<MeshLight> m_MeshLights;
	rtlib::CUDABuffer<Reservoir>       m_ResvBuffers[3];
	rtlib::CUDABuffer<unsigned int>    m_SeedBuffer;
	unsigned int        m_LightHgRecIndex = 0;
	unsigned int        m_PrvReservoirIdx = 0;
	unsigned int        m_CurReservoirIdx = 1;
	unsigned int        m_FnlReservoirIdx = 2;
	unsigned int        m_SamplePerAll    = 0;
	unsigned int        m_SamplePerLaunch = 1;
	unsigned int        m_NumCandidates   = 1;
	rtlib::ext::Camera  m_Camera          = {};
};

Test24ReSTIROPXTracer::Test24ReSTIROPXTracer(
	ContextPtr Context, FramebufferPtr Framebuffer, CameraControllerPtr CameraController, TextureAssetManager TextureManager,
	rtlib::ext::IASHandlePtr TopLevelAS, const std::vector<rtlib::ext::VariableMap>& Materials, const float3& BgLightColor, const unsigned int& eventFlags)
{
	m_Impl = std::make_unique<Test24ReSTIROPXTracer::Impl>(
		Context, Framebuffer, CameraController, TextureManager, TopLevelAS, Materials, BgLightColor, eventFlags
		);
}

void Test24ReSTIROPXTracer::Initialize()
{
	this->InitPipeline();
	this->InitFrameResources();
	this->InitLight();
	this->InitShaderBindingTable();
	this->InitLaunchParams();
}

void Test24ReSTIROPXTracer::Launch(int width, int height, void* userData)
{
	UserData* pUserData = (UserData*)userData;
	if (!pUserData)
	{
		return;
	}
	if (width != m_Impl->m_Framebuffer->GetWidth() || height != m_Impl->m_Framebuffer->GetHeight()) {
		return;
	}

	m_Impl->m_NumCandidates = pUserData->numCandidates;

	this->m_Impl->m_First.m_Params.cpuHandle[0].width = width;
	this->m_Impl->m_First.m_Params.cpuHandle[0].height = height;
	this->m_Impl->m_First.m_Params.cpuHandle[0].posiBuffer = m_Impl->m_Framebuffer->GetComponent<test::RTCUDABufferFBComponent<float3>>("GPosition")->GetHandle().getDevicePtr();
	this->m_Impl->m_First.m_Params.cpuHandle[0].normBuffer = m_Impl->m_Framebuffer->GetComponent<test::RTCUDABufferFBComponent<float3>>("GNormal")->GetHandle().getDevicePtr();
	this->m_Impl->m_First.m_Params.cpuHandle[0].emitBuffer = m_Impl->m_Framebuffer->GetComponent<test::RTCUDABufferFBComponent<float3>>("GEmission")->GetHandle().getDevicePtr();
	this->m_Impl->m_First.m_Params.cpuHandle[0].diffBuffer = m_Impl->m_Framebuffer->GetComponent<test::RTCUDABufferFBComponent<float3>>("GDiffuse")->GetHandle().getDevicePtr();
	//TODO
	cudaMemcpyAsync(this->m_Impl->m_First.m_Params.gpuHandle.getDevicePtr(), &this->m_Impl->m_First.m_Params.cpuHandle[0], sizeof(RayFirstParams), cudaMemcpyHostToDevice, pUserData->stream);
	this->m_Impl->m_First.m_Pipeline.launch(pUserData->stream, this->m_Impl->m_First.m_Params.gpuHandle.getDevicePtr(), this->m_Impl->m_First.m_ShaderBindingTable, width, height, 1);
	RTLIB_CU_CHECK(cuStreamSynchronize(pUserData->stream));
	
	this->m_Impl->m_Second.m_Params.cpuHandle[0].width           = width;
	this->m_Impl->m_Second.m_Params.cpuHandle[0].height          = height;
	this->m_Impl->m_Second.m_Params.cpuHandle[0].samplePerALL    = m_Impl->m_SamplePerAll;
	this->m_Impl->m_Second.m_Params.cpuHandle[0].samplePerLaunch = m_Impl->m_SamplePerLaunch;
	this->m_Impl->m_Second.m_Params.cpuHandle[0].numCandidates   = m_Impl->m_NumCandidates;
	this->m_Impl->m_Second.m_Params.cpuHandle[0].seedBuffer      = m_Impl->m_SeedBuffer.getDevicePtr();
	this->m_Impl->m_Second.m_Params.cpuHandle[0].posiBuffer      = m_Impl->m_Framebuffer->GetComponent<test::RTCUDABufferFBComponent<float3>>("GPosition")->GetHandle().getDevicePtr();
	this->m_Impl->m_Second.m_Params.cpuHandle[0].normBuffer      = m_Impl->m_Framebuffer->GetComponent<test::RTCUDABufferFBComponent<float3>>("GNormal")->GetHandle().getDevicePtr();
	this->m_Impl->m_Second.m_Params.cpuHandle[0].emitBuffer      = m_Impl->m_Framebuffer->GetComponent<test::RTCUDABufferFBComponent<float3>>("GEmission")->GetHandle().getDevicePtr();
	this->m_Impl->m_Second.m_Params.cpuHandle[0].diffBuffer      = m_Impl->m_Framebuffer->GetComponent<test::RTCUDABufferFBComponent<float3>>("GDiffuse")->GetHandle().getDevicePtr();
	this->m_Impl->m_Second.m_Params.cpuHandle[0].accumBuffer     = m_Impl->m_Framebuffer->GetComponent<test::RTCUDABufferFBComponent<float3>>("RAccum")->GetHandle().getDevicePtr();
	this->m_Impl->m_Second.m_Params.cpuHandle[0].resvBuffer      = m_Impl->m_ResvBuffers[m_Impl->m_FnlReservoirIdx].getDevicePtr();
	this->m_Impl->m_Second.m_Params.cpuHandle[0].frameBuffer     = m_Impl->m_Framebuffer->GetComponent<test::RTCUGLBufferFBComponent<uchar4>>("RFrame")->GetHandle().map();
	this->m_Impl->m_Second.m_Params.cpuHandle[0].meshLights      = m_Impl->GetMeshLightList();

	cudaMemcpyAsync(this->m_Impl->m_Second.m_Params.gpuHandle.getDevicePtr(), &this->m_Impl->m_Second.m_Params.cpuHandle[0], sizeof(RaySecondParams), cudaMemcpyHostToDevice, pUserData->stream);
	this->m_Impl->m_Second.m_Pipeline.launch(pUserData->stream, this->m_Impl->m_Second.m_Params.gpuHandle.getDevicePtr()+0 , this->m_Impl->m_Second.m_ShaderBindingTableForInit, width, height, 1);
	RTLIB_CU_CHECK(cuStreamSynchronize(pUserData->stream));

	this->m_Impl->m_Second.m_Pipeline.launch(pUserData->stream, this->m_Impl->m_Second.m_Params.gpuHandle.getDevicePtr()+0 , this->m_Impl->m_Second.m_ShaderBindingTableForDraw, width, height, 1);
	RTLIB_CU_CHECK(cuStreamSynchronize(pUserData->stream));

	m_Impl->m_Framebuffer->GetComponent<test::RTCUGLBufferFBComponent<uchar4>>("RFrame")->GetHandle().unmap();
	this->m_Impl->m_Second.m_Params.cpuHandle[0].samplePerALL += this->m_Impl->m_Second.m_Params.cpuHandle[0].samplePerLaunch;
	this->m_Impl->m_SamplePerAll = this->m_Impl->m_Second.m_Params.cpuHandle[0].samplePerALL;
}

void Test24ReSTIROPXTracer::CleanUp()
{
	this->FreeLaunchParams();
	this->FreeShaderBindingTable();
	this->FreeLight();
	this->FreeFrameResources();
	this->FreePipeline();
	this->m_Impl->m_LightHgRecIndex = 0;
}

void Test24ReSTIROPXTracer::Update()
{
	if ((this->m_Impl->m_EventFlags  & TEST24_EVENT_FLAG_UPDATE_CAMERA) == TEST24_EVENT_FLAG_UPDATE_CAMERA)
	{
		float aspect = (float)m_Impl->m_Framebuffer->GetWidth() / (float)m_Impl->m_Framebuffer->GetHeight();
		this->m_Impl->m_Camera = this->m_Impl->m_CameraController->GetCamera(aspect);
		this->m_Impl->m_First.m_RGRecordBuffer.cpuHandle[0].data.pinhole[1] = this->m_Impl->m_First.m_RGRecordBuffer.cpuHandle[0].data.pinhole[0];
		this->m_Impl->m_First.m_RGRecordBuffer.cpuHandle[0].data.pinhole[0].eye = this->m_Impl->m_Camera.getEye();
		auto [u, v, w] = this->m_Impl->m_Camera.getUVW();
		CUdeviceptr prvRayGenPtr = (CUdeviceptr)this->m_Impl->m_First.m_RGRecordBuffer.gpuHandle.getDevicePtr();
		this->m_Impl->m_First.m_RGRecordBuffer.cpuHandle[0].data.pinhole[0].u = u;
		this->m_Impl->m_First.m_RGRecordBuffer.cpuHandle[0].data.pinhole[0].v = v;
		this->m_Impl->m_First.m_RGRecordBuffer.cpuHandle[0].data.pinhole[0].w = w;
		this->m_Impl->m_First.m_RGRecordBuffer.Upload();
		CUdeviceptr curRayGenPtr = (CUdeviceptr)this->m_Impl->m_First.m_RGRecordBuffer.gpuHandle.getDevicePtr();
	}
	if ((this->m_Impl->m_EventFlags  & TEST24_EVENT_FLAG_FLUSH_FRAME)   == TEST24_EVENT_FLAG_FLUSH_FRAME)
	{
		std::vector<float3> zeroAccumValues(this->m_Impl->m_Framebuffer->GetWidth() * this->m_Impl->m_Framebuffer->GetHeight(), make_float3(0.0f));
		cudaMemcpy(this->m_Impl->m_Framebuffer->GetComponent<test::RTCUDABufferFBComponent<float3>>("RAccum")->GetHandle().getDevicePtr(), zeroAccumValues.data(), sizeof(float3) * this->m_Impl->m_Framebuffer->GetWidth() * this->m_Impl->m_Framebuffer->GetHeight(), cudaMemcpyHostToDevice);
		this->m_Impl->m_Second.m_Params.cpuHandle[0].samplePerALL = 0;
		this->m_Impl->m_SamplePerAll = 0;
	}
	bool shouldRegen = ((this->m_Impl->m_SamplePerAll + this->m_Impl->m_Second.m_Params.cpuHandle[0].samplePerLaunch) / 1024 != this->m_Impl->m_SamplePerAll / 1024);
	if (((this->m_Impl->m_EventFlags & TEST24_EVENT_FLAG_RESIZE_FRAME)  == TEST24_EVENT_FLAG_RESIZE_FRAME) || shouldRegen)
	{
		std::cout << "Regen!\n";
		std::vector<unsigned int> seedData(this->m_Impl->m_Framebuffer->GetWidth() * this->m_Impl->m_Framebuffer->GetHeight());
		std::random_device rd;
		std::mt19937 mt(rd());
		std::generate(seedData.begin(), seedData.end(), mt);
		this->m_Impl->m_SeedBuffer.resize(this->m_Impl->m_Framebuffer->GetWidth() * this->m_Impl->m_Framebuffer->GetHeight());
		this->m_Impl->m_SeedBuffer.upload(seedData);

		std::vector<Reservoir> reservoirs(this->m_Impl->m_Framebuffer->GetWidth() * this->m_Impl->m_Framebuffer->GetHeight());
		this->m_Impl->m_ResvBuffers[0].resize(this->m_Impl->m_Framebuffer->GetWidth() * this->m_Impl->m_Framebuffer->GetHeight());
		this->m_Impl->m_ResvBuffers[1].resize(this->m_Impl->m_Framebuffer->GetWidth() * this->m_Impl->m_Framebuffer->GetHeight());
		this->m_Impl->m_ResvBuffers[2].resize(this->m_Impl->m_Framebuffer->GetWidth() * this->m_Impl->m_Framebuffer->GetHeight());
		this->m_Impl->m_ResvBuffers[0].upload(reservoirs);
		this->m_Impl->m_ResvBuffers[1].upload(reservoirs);
		this->m_Impl->m_ResvBuffers[2].upload(reservoirs);
	}
	if ((this->m_Impl->m_EventFlags  & TEST24_EVENT_FLAG_UPDATE_LIGHT)  == TEST24_EVENT_FLAG_UPDATE_LIGHT)
	{
		auto lightColor = make_float4(this->m_Impl->m_BgLightColor, 1.0f);
		this->m_Impl->m_First.m_MSRecordBuffers.cpuHandle[RAY_TYPE_RADIANCE].data.bgColor = lightColor;
		RTLIB_CUDA_CHECK(cudaMemcpy(this->m_Impl->m_First.m_MSRecordBuffers.gpuHandle.getDevicePtr() + RAY_TYPE_RADIANCE, &this->m_Impl->m_First.m_MSRecordBuffers.cpuHandle[RAY_TYPE_RADIANCE], sizeof(MissRecord), cudaMemcpyHostToDevice));
	}
}

Test24ReSTIROPXTracer::~Test24ReSTIROPXTracer() {}

void Test24ReSTIROPXTracer::InitFrameResources()
{
	std::vector<unsigned int> seedData(this->m_Impl->m_Framebuffer->GetWidth() * this->m_Impl->m_Framebuffer->GetHeight());
	std::random_device rd;
	std::mt19937 mt(rd());
	std::generate(seedData.begin(), seedData.end(), mt);
	this->m_Impl->m_SeedBuffer = rtlib::CUDABuffer<uint32_t>(seedData);

	std::vector<Reservoir> reservoirs(this->m_Impl->m_Framebuffer->GetWidth() * this->m_Impl->m_Framebuffer->GetHeight());
	this->m_Impl->m_ResvBuffers[0] = rtlib::CUDABuffer<Reservoir>(reservoirs);
	this->m_Impl->m_ResvBuffers[1] = rtlib::CUDABuffer<Reservoir>(reservoirs);
	this->m_Impl->m_ResvBuffers[2] = rtlib::CUDABuffer<Reservoir>(reservoirs);
}

void Test24ReSTIROPXTracer::InitPipeline()
{
	/**First **/
	{
		auto rayReSTIRPtxFile = std::ifstream(TEST_TEST24_RESTIR_OPX_CUDA_PATH "/RayFirst.ptx", std::ios::binary);
		if (!rayReSTIRPtxFile.is_open())
			throw std::runtime_error("Failed To Load RayFirst.ptx!");
		auto rayReSTIRPtxData = std::string((std::istreambuf_iterator<char>(rayReSTIRPtxFile)), (std::istreambuf_iterator<char>()));
		rayReSTIRPtxFile.close();

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
		this->m_Impl->m_First.m_Pipeline = this->m_Impl->m_Context->GetOPX7Handle()->createPipeline(debugCompileOptions);
		this->m_Impl->m_First.m_Modules["RayFirst"] = this->m_Impl->m_First.m_Pipeline.createModule(rayReSTIRPtxData, debugModuleOptions);
		this->m_Impl->m_First.m_RGProgramGroups["First.Default"] = this->m_Impl->m_First.m_Pipeline.createRaygenPG({ this->m_Impl->m_First.m_Modules["RayFirst"], "__raygen__first" });
		this->m_Impl->m_First.m_MSProgramGroups["First.Default"] = this->m_Impl->m_First.m_Pipeline.createMissPG({ this->m_Impl->m_First.m_Modules["RayFirst"], "__miss__first" });
		this->m_Impl->m_First.m_HGProgramGroups["First.Default"] = this->m_Impl->m_First.m_Pipeline.createHitgroupPG({ this->m_Impl->m_First.m_Modules["RayFirst"], "__closesthit__first" }, {}, {});
		this->m_Impl->m_First.m_Pipeline.link(debugLinkOptions);
	}
	/**Second**/
	{
		auto rayReSTIRPtxFile = std::ifstream(TEST_TEST24_RESTIR_OPX_CUDA_PATH "/RaySecond.ptx", std::ios::binary);
		if (!rayReSTIRPtxFile.is_open())
			throw std::runtime_error("Failed To Load RaySecond.ptx!");
		auto rayReSTIRPtxData = std::string((std::istreambuf_iterator<char>(rayReSTIRPtxFile)), (std::istreambuf_iterator<char>()));
		rayReSTIRPtxFile.close();

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
		this->m_Impl->m_Second.m_Pipeline = this->m_Impl->m_Context->GetOPX7Handle()->createPipeline(debugCompileOptions);
		this->m_Impl->m_Second.m_Modules["RaySecond"] = this->m_Impl->m_Second.m_Pipeline.createModule(rayReSTIRPtxData, debugModuleOptions);
		this->m_Impl->m_Second.m_RGProgramGroups["Second.Init"] = this->m_Impl->m_Second.m_Pipeline.createRaygenPG({ this->m_Impl->m_Second.m_Modules["RaySecond"], "__raygen__init" });
		this->m_Impl->m_Second.m_RGProgramGroups["Second.Draw"] = this->m_Impl->m_Second.m_Pipeline.createRaygenPG({ this->m_Impl->m_Second.m_Modules["RaySecond"], "__raygen__draw" });
		this->m_Impl->m_Second.m_MSProgramGroups["Second.Occluded"] = this->m_Impl->m_Second.m_Pipeline.createMissPG({ this->m_Impl->m_Second.m_Modules["RaySecond"], "__miss__occluded" });
		this->m_Impl->m_Second.m_HGProgramGroups["Second.Occluded"] = this->m_Impl->m_Second.m_Pipeline.createHitgroupPG({ this->m_Impl->m_Second.m_Modules["RaySecond"], "__closesthit__occluded" }, {}, {});
		this->m_Impl->m_Second.m_Pipeline.link(debugLinkOptions);
	}
}

void Test24ReSTIROPXTracer::InitShaderBindingTable()
{
	/**First **/
	{
		float aspect = static_cast<float>(m_Impl->m_Framebuffer->GetWidth()) / static_cast<float>(m_Impl->m_Framebuffer->GetHeight());
		auto tlas = this->m_Impl->m_TopLevelAS;
		this->m_Impl->m_Camera = this->m_Impl->m_CameraController->GetCamera(aspect);
		auto& materials = this->m_Impl->m_Materials;
		this->m_Impl->m_First.m_RGRecordBuffer.Alloc(RAY_TYPE_COUNT);
		this->m_Impl->m_First.m_RGRecordBuffer.cpuHandle[0] = this->m_Impl->m_First.m_RGProgramGroups["First.Default"].getSBTRecord<RayGenData>();
		this->m_Impl->m_First.m_RGRecordBuffer.cpuHandle[0].data.pinhole[0].eye = this->m_Impl->m_Camera.getEye();
		auto [u, v, w] = this->m_Impl->m_Camera.getUVW();
		this->m_Impl->m_First.m_RGRecordBuffer.cpuHandle[0].data.pinhole[0].u = u;
		this->m_Impl->m_First.m_RGRecordBuffer.cpuHandle[0].data.pinhole[0].v = v;
		this->m_Impl->m_First.m_RGRecordBuffer.cpuHandle[0].data.pinhole[0].w = w;
		this->m_Impl->m_First.m_RGRecordBuffer.Upload();
		this->m_Impl->m_First.m_MSRecordBuffers.Alloc(RAY_TYPE_COUNT);
		this->m_Impl->m_First.m_MSRecordBuffers.cpuHandle[RAY_TYPE_RADIANCE] = this->m_Impl->m_First.m_MSProgramGroups["First.Default"].getSBTRecord<MissData>();
		this->m_Impl->m_First.m_MSRecordBuffers.cpuHandle[RAY_TYPE_RADIANCE].data.bgColor = make_float4(this->m_Impl->m_BgLightColor, 1.0f);
		this->m_Impl->m_First.m_MSRecordBuffers.cpuHandle[RAY_TYPE_OCCLUSION] = this->m_Impl->m_First.m_MSProgramGroups["First.Default"].getSBTRecord<MissData>();
		this->m_Impl->m_First.m_MSRecordBuffers.Upload();
		this->m_Impl->m_First.m_HGRecordBuffers.Alloc(tlas->GetSbtCount() * RAY_TYPE_COUNT);
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
								radianceHgData.vertices = cudaVertexBuffer->GetHandle().getDevicePtr();
								radianceHgData.normals = cudaNormalBuffer->GetHandle().getDevicePtr();
								radianceHgData.texCoords = cudaTexCrdBuffer->GetHandle().getDevicePtr();
								radianceHgData.indices = cudaTriIndBuffer->GetHandle().getDevicePtr();
								radianceHgData.diffuseTex = this->m_Impl->m_TextureManager->GetAsset(material.GetString("diffTex")).GetGpuComponent<rtlib::ext::resources::CUDATextureImage2DComponent<uchar4>>("CUDATexture")->GetHandle().getHandle();
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
							this->m_Impl->m_First.m_HGRecordBuffers.cpuHandle[RAY_TYPE_COUNT * sbtOffset + RAY_TYPE_COUNT * i + RAY_TYPE_RADIANCE]  = this->m_Impl->m_First.m_HGProgramGroups["First.Default"].getSBTRecord<HitgroupData>(radianceHgData);
							this->m_Impl->m_First.m_HGRecordBuffers.cpuHandle[RAY_TYPE_COUNT * sbtOffset + RAY_TYPE_COUNT * i + RAY_TYPE_OCCLUSION] = this->m_Impl->m_First.m_HGProgramGroups["First.Default"].getSBTRecord<HitgroupData>({});
						}
						sbtOffset += mesh->GetUniqueResource()->materials.size();
					}
				}
			}
		}
		this->m_Impl->m_First.m_HGRecordBuffers.Upload();
		this->m_Impl->m_First.m_ShaderBindingTable.raygenRecord = reinterpret_cast<CUdeviceptr>(this->m_Impl->m_First.m_RGRecordBuffer.gpuHandle.getDevicePtr());
		this->m_Impl->m_First.m_ShaderBindingTable.missRecordBase = reinterpret_cast<CUdeviceptr>(this->m_Impl->m_First.m_MSRecordBuffers.gpuHandle.getDevicePtr());
		this->m_Impl->m_First.m_ShaderBindingTable.missRecordCount = this->m_Impl->m_First.m_MSRecordBuffers.cpuHandle.size();
		this->m_Impl->m_First.m_ShaderBindingTable.missRecordStrideInBytes = sizeof(MissRecord);
		this->m_Impl->m_First.m_ShaderBindingTable.hitgroupRecordBase = reinterpret_cast<CUdeviceptr>(this->m_Impl->m_First.m_HGRecordBuffers.gpuHandle.getDevicePtr());
		this->m_Impl->m_First.m_ShaderBindingTable.hitgroupRecordCount = this->m_Impl->m_First.m_HGRecordBuffers.cpuHandle.size();
		this->m_Impl->m_First.m_ShaderBindingTable.hitgroupRecordStrideInBytes = sizeof(HitGRecord);
	}
	/**Second**/
	{
		auto tlas = this->m_Impl->m_TopLevelAS;
		auto camera = this->m_Impl->m_Camera;
		auto& materials = this->m_Impl->m_Materials;
		/*RayGen*/
		this->m_Impl->m_Second.m_RGRecordBuffers.Alloc(2);
		/*RayGen-0-Init*/
		auto eye = camera.getEye();
		auto [u, v, w] = camera.getUVW();
		this->m_Impl->m_Second.m_RGRecordBuffers.cpuHandle[0] = this->m_Impl->m_Second.m_RGProgramGroups["Second.Init"].getSBTRecord<RayGenData>();
		this->m_Impl->m_Second.m_RGRecordBuffers.cpuHandle[0].data.pinhole[0].eye = eye;
		this->m_Impl->m_Second.m_RGRecordBuffers.cpuHandle[0].data.pinhole[0].u = u;
		this->m_Impl->m_Second.m_RGRecordBuffers.cpuHandle[0].data.pinhole[0].v = v;
		this->m_Impl->m_Second.m_RGRecordBuffers.cpuHandle[0].data.pinhole[0].w = w;
		/*RayGen-1-Draw*/
		this->m_Impl->m_Second.m_RGRecordBuffers.cpuHandle[1] = this->m_Impl->m_Second.m_RGProgramGroups["Second.Draw"].getSBTRecord<RayGenData>();
		this->m_Impl->m_Second.m_RGRecordBuffers.cpuHandle[1].data.pinhole[0].eye = eye;
		this->m_Impl->m_Second.m_RGRecordBuffers.cpuHandle[1].data.pinhole[0].u = u;
		this->m_Impl->m_Second.m_RGRecordBuffers.cpuHandle[1].data.pinhole[0].v = v;
		this->m_Impl->m_Second.m_RGRecordBuffers.cpuHandle[1].data.pinhole[0].w = w;
		this->m_Impl->m_Second.m_RGRecordBuffers.Upload();
		/*Miss*/
		this->m_Impl->m_Second.m_MSRecordBuffers.Alloc(RAY_TYPE_COUNT);
		this->m_Impl->m_Second.m_MSRecordBuffers.cpuHandle[RAY_TYPE_RADIANCE]  = this->m_Impl->m_Second.m_MSProgramGroups["Second.Occluded"].getSBTRecord<MissData2>();
		this->m_Impl->m_Second.m_MSRecordBuffers.cpuHandle[RAY_TYPE_OCCLUSION] = this->m_Impl->m_Second.m_MSProgramGroups["Second.Occluded"].getSBTRecord<MissData2>();
		this->m_Impl->m_Second.m_MSRecordBuffers.Upload();
		/*Hitgroup*/
		this->m_Impl->m_Second.m_HGRecordBuffers.Alloc(tlas->GetSbtCount() * RAY_TYPE_COUNT);
		{
			size_t sbtOffset = 0;
			for (auto& instanceSet : tlas->GetInstanceSets())
			{
				for (auto& baseGASHandle : instanceSet->baseGASHandles)
				{
					for (auto& mesh : baseGASHandle->GetMeshes())
					{
						for (size_t i = 0; i < mesh->GetUniqueResource()->materials.size(); ++i)
						{
							this->m_Impl->m_Second.m_HGRecordBuffers.cpuHandle[RAY_TYPE_COUNT * sbtOffset + RAY_TYPE_COUNT * i + RAY_TYPE_RADIANCE]  = this->m_Impl->m_Second.m_HGProgramGroups["Second.Occluded"].getSBTRecord<HitgroupData2>({});
							this->m_Impl->m_Second.m_HGRecordBuffers.cpuHandle[RAY_TYPE_COUNT * sbtOffset + RAY_TYPE_COUNT * i + RAY_TYPE_OCCLUSION] = this->m_Impl->m_Second.m_HGProgramGroups["Second.Occluded"].getSBTRecord<HitgroupData2>({});
						}
						sbtOffset += mesh->GetUniqueResource()->materials.size();
					}
				}
			}
		}
		this->m_Impl->m_Second.m_HGRecordBuffers.Upload();
		/*ShaderBindingTable*/
		/*ShaderBindingTable-Init*/
		this->m_Impl->m_Second.m_ShaderBindingTableForInit.raygenRecord    = reinterpret_cast<CUdeviceptr>(this->m_Impl->m_Second.m_RGRecordBuffers.gpuHandle.getDevicePtr() + 0);
		this->m_Impl->m_Second.m_ShaderBindingTableForInit.missRecordBase  = reinterpret_cast<CUdeviceptr>(this->m_Impl->m_Second.m_MSRecordBuffers.gpuHandle.getDevicePtr());
		this->m_Impl->m_Second.m_ShaderBindingTableForInit.missRecordCount = this->m_Impl->m_Second.m_MSRecordBuffers.cpuHandle.size();
		this->m_Impl->m_Second.m_ShaderBindingTableForInit.missRecordStrideInBytes = sizeof(MissRecord2);
		this->m_Impl->m_Second.m_ShaderBindingTableForInit.hitgroupRecordBase = reinterpret_cast<CUdeviceptr>(this->m_Impl->m_Second.m_HGRecordBuffers.gpuHandle.getDevicePtr());
		this->m_Impl->m_Second.m_ShaderBindingTableForInit.hitgroupRecordCount = this->m_Impl->m_Second.m_HGRecordBuffers.cpuHandle.size();
		this->m_Impl->m_Second.m_ShaderBindingTableForInit.hitgroupRecordStrideInBytes = sizeof(HitGRecord2);
		/*ShaderBindingTable-Draw*/
		this->m_Impl->m_Second.m_ShaderBindingTableForDraw.raygenRecord = reinterpret_cast<CUdeviceptr>(this->m_Impl->m_Second.m_RGRecordBuffers.gpuHandle.getDevicePtr() + 1);
		this->m_Impl->m_Second.m_ShaderBindingTableForDraw.missRecordBase = reinterpret_cast<CUdeviceptr>(this->m_Impl->m_Second.m_MSRecordBuffers.gpuHandle.getDevicePtr());
		this->m_Impl->m_Second.m_ShaderBindingTableForDraw.missRecordCount = this->m_Impl->m_Second.m_MSRecordBuffers.cpuHandle.size();
		this->m_Impl->m_Second.m_ShaderBindingTableForDraw.missRecordStrideInBytes = sizeof(MissRecord2);
		this->m_Impl->m_Second.m_ShaderBindingTableForDraw.hitgroupRecordBase = reinterpret_cast<CUdeviceptr>(this->m_Impl->m_Second.m_HGRecordBuffers.gpuHandle.getDevicePtr());
		this->m_Impl->m_Second.m_ShaderBindingTableForDraw.hitgroupRecordCount = this->m_Impl->m_Second.m_HGRecordBuffers.cpuHandle.size();
		this->m_Impl->m_Second.m_ShaderBindingTableForDraw.hitgroupRecordStrideInBytes = sizeof(HitGRecord2);
	}
}

void Test24ReSTIROPXTracer::InitLight()
{
	auto ChooseNEE = [](const rtlib::ext::MeshPtr& mesh)->bool {
		return true;
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
			meshLight.indices     = mesh->GetUniqueResource()->triIndBuffer.GetGpuComponent<rtlib::ext::resources::CUDABufferComponent<uint3>>( "CUDA")->GetHandle().getDevicePtr();
			meshLight.indCount    = mesh->GetUniqueResource()->triIndBuffer.Size();
			m_Impl->m_MeshLights.cpuHandle.push_back(meshLight);
		}
	}
	m_Impl->m_MeshLights.Alloc();
	m_Impl->m_MeshLights.Upload();
}

void Test24ReSTIROPXTracer::InitLaunchParams()
{
	auto tlas = this->m_Impl->m_TopLevelAS;
	this->m_Impl->m_First.m_Params.Alloc(1);
	this->m_Impl->m_First.m_Params.cpuHandle[0].gasHandle  = tlas->GetHandle();
	this->m_Impl->m_First.m_Params.cpuHandle[0].width      = this->m_Impl->m_Framebuffer->GetWidth();
	this->m_Impl->m_First.m_Params.cpuHandle[0].height     = this->m_Impl->m_Framebuffer->GetHeight();
	this->m_Impl->m_First.m_Params.cpuHandle[0].posiBuffer = m_Impl->m_Framebuffer->GetComponent<test::RTCUDABufferFBComponent<float3>>("GPosition")->GetHandle().getDevicePtr();
	this->m_Impl->m_First.m_Params.cpuHandle[0].normBuffer = m_Impl->m_Framebuffer->GetComponent<test::RTCUDABufferFBComponent<float3>>("GNormal")->GetHandle().getDevicePtr();
	this->m_Impl->m_First.m_Params.cpuHandle[0].emitBuffer = m_Impl->m_Framebuffer->GetComponent<test::RTCUDABufferFBComponent<float3>>("GEmission")->GetHandle().getDevicePtr();
	this->m_Impl->m_First.m_Params.cpuHandle[0].diffBuffer = m_Impl->m_Framebuffer->GetComponent<test::RTCUDABufferFBComponent<float3>>("GDiffuse")->GetHandle().getDevicePtr();
	this->m_Impl->m_First.m_Params.Upload();
	this->m_Impl->m_Second.m_Params.Alloc(1);
	this->m_Impl->m_Second.m_Params.cpuHandle[0].gasHandle       = tlas->GetHandle();
	this->m_Impl->m_Second.m_Params.cpuHandle[0].width           = this->m_Impl->m_Framebuffer->GetWidth();
	this->m_Impl->m_Second.m_Params.cpuHandle[0].height          = this->m_Impl->m_Framebuffer->GetHeight();
	this->m_Impl->m_Second.m_Params.cpuHandle[0].samplePerALL    = m_Impl->m_SamplePerAll;
	this->m_Impl->m_Second.m_Params.cpuHandle[0].samplePerLaunch = m_Impl->m_SamplePerLaunch;
	this->m_Impl->m_Second.m_Params.cpuHandle[0].numCandidates   = m_Impl->m_NumCandidates;
	this->m_Impl->m_Second.m_Params.cpuHandle[0].seedBuffer      = m_Impl->m_SeedBuffer.getDevicePtr();
	this->m_Impl->m_Second.m_Params.cpuHandle[0].posiBuffer      = m_Impl->m_Framebuffer->GetComponent<test::RTCUDABufferFBComponent<float3>>("GPosition")->GetHandle().getDevicePtr();
	this->m_Impl->m_Second.m_Params.cpuHandle[0].normBuffer      = m_Impl->m_Framebuffer->GetComponent<test::RTCUDABufferFBComponent<float3>>("GNormal"  )->GetHandle().getDevicePtr();
	this->m_Impl->m_Second.m_Params.cpuHandle[0].emitBuffer      = m_Impl->m_Framebuffer->GetComponent<test::RTCUDABufferFBComponent<float3>>("GEmission")->GetHandle().getDevicePtr();
	this->m_Impl->m_Second.m_Params.cpuHandle[0].diffBuffer      = m_Impl->m_Framebuffer->GetComponent<test::RTCUDABufferFBComponent<float3>>("GDiffuse" )->GetHandle().getDevicePtr();
	this->m_Impl->m_Second.m_Params.cpuHandle[0].resvBuffer      = m_Impl->m_ResvBuffers[m_Impl->m_FnlReservoirIdx].getDevicePtr();
	this->m_Impl->m_Second.m_Params.cpuHandle[0].accumBuffer     = m_Impl->m_Framebuffer->GetComponent<test::RTCUDABufferFBComponent<float3>>("RAccum"   )->GetHandle().getDevicePtr();
	this->m_Impl->m_Second.m_Params.cpuHandle[0].frameBuffer     = nullptr;
	this->m_Impl->m_Second.m_Params.cpuHandle[0].meshLights      = m_Impl->GetMeshLightList();
	this->m_Impl->m_Second.m_Params.Upload();
}

void Test24ReSTIROPXTracer::FreeFrameResources()
{
}

void Test24ReSTIROPXTracer::FreePipeline()
{
	this->m_Impl->m_First.m_RGProgramGroups.clear();
	this->m_Impl->m_First.m_HGProgramGroups.clear();
	this->m_Impl->m_First.m_MSProgramGroups.clear();
	this->m_Impl->m_Second.m_RGProgramGroups.clear();
	this->m_Impl->m_Second.m_HGProgramGroups.clear();
	this->m_Impl->m_Second.m_MSProgramGroups.clear();
}

void Test24ReSTIROPXTracer::FreeShaderBindingTable()
{
	this->m_Impl->m_First.m_ShaderBindingTable = {};
	this->m_Impl->m_First.m_RGRecordBuffer.Reset();
	this->m_Impl->m_First.m_MSRecordBuffers.Reset();
	this->m_Impl->m_First.m_HGRecordBuffers.Reset();
}

void Test24ReSTIROPXTracer::FreeLight()
{

}

void Test24ReSTIROPXTracer::FreeLaunchParams()
{
	this->m_Impl->m_First.m_Params.Reset();
}

