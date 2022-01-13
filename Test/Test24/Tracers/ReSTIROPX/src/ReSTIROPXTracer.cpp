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
using CUDAModuleMap = std::unordered_map<std::string, rtlib::CUDAModule>;
using CUDAFunctionMap = std::unordered_map<std::string, rtlib::CUDAFunction>;
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
		CUDAModuleMap m_CUDAModules = {};
		CUDAFunctionMap m_CUDAFunctions = {};
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
		FramebufferSharedPtr Framebuffer,
		CameraControllerPtr CameraController,
		TextureAssetManager TextureManager,
		rtlib::ext::IASHandlePtr TopLevelAS,
		const std::vector<rtlib::ext::VariableMap>& Materials,
		const float3& BgLightColor,
		const unsigned int& EventFlags) :m_Context{ Context }, m_SharedFramebuffer{ Framebuffer }, m_CameraController{ CameraController }, m_TextureManager{ TextureManager }, m_Materials{ Materials }, m_TopLevelAS{ TopLevelAS }, m_BgLightColor{ BgLightColor },m_CurEventFlags{ EventFlags }, m_PrvEventFlags{ TEST24_EVENT_FLAG_NONE }
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
	FramebufferSharedPtr m_SharedFramebuffer;
	FramebufferUniquePtr m_UniqueFramebuffer;
	CameraControllerPtr  m_CameraController;
	TextureAssetManager  m_TextureManager;
	rtlib::ext::IASHandlePtr m_TopLevelAS;
	const std::vector<rtlib::ext::VariableMap>& m_Materials;
	const float3& m_BgLightColor;
	const unsigned int& m_CurEventFlags;
	unsigned int        m_PrvEventFlags;
	FirstPipeline       m_First;
	SecondPipeline      m_Second;
	rtlib::CUDAUploadBuffer<MeshLight>     m_MeshLights;
	unsigned int        m_LightHgRecIndex = 0;
	unsigned int        m_PrvReservoirIdx = 0;
	unsigned int        m_CurReservoirIdx = 1;
	unsigned int        m_FnlReservoirIdx = 2;
	bool                m_UpdateMotion    = false;
	rtlib::ext::Camera  m_Camera          = {};
};

Test24ReSTIROPXTracer::Test24ReSTIROPXTracer(
	ContextPtr Context, FramebufferSharedPtr Framebuffer, CameraControllerPtr CameraController, TextureAssetManager TextureManager,
	rtlib::ext::IASHandlePtr TopLevelAS, const std::vector<rtlib::ext::VariableMap>& Materials, const float3& BgLightColor,
	const unsigned int& eventFlags):test::RTTracer()
{
	m_Impl = std::make_unique<Test24ReSTIROPXTracer::Impl>(
		Context, Framebuffer, CameraController, TextureManager, TopLevelAS, Materials, BgLightColor,eventFlags
		);
	//MOVABLE
	this->GetVariables()->SetUInt32( "SampleForBudget", 1024);
	this->GetVariables()->SetUInt32(    "SamplePerAll", 0);
	this->GetVariables()->SetUInt32( "SamplePerLaunch", 1);
	this->GetVariables()->SetUInt32(  "NumCandidates" , 32);
	this->GetVariables()->SetBool(           "Started", false);
	this->GetVariables()->SetBool(          "Launched", false);
	this->GetVariables()->SetBool(     "ReuseTemporal", true);
	this->GetVariables()->SetBool(      "ReuseSpatial", true);
	this->GetVariables()->SetUInt32(   "RangeSpatial" , 30);
	this->GetVariables()->SetUInt32(  "SampleSpatial" , 5);
	this->GetVariables()->SetUInt32("IterationSpatial", 4);

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
	if (width != m_Impl->m_SharedFramebuffer->GetWidth() || height != m_Impl->m_SharedFramebuffer->GetHeight()) {
		return;
	}

	if (GetVariables()->GetBool("Started")) {

		GetVariables()->SetBool("Started", false);
		GetVariables()->SetBool("Launched", true);
		GetVariables()->SetUInt32("SamplePerAll", 0);
	}


	auto samplePerAll    = GetVariables()->GetUInt32("SamplePerAll");
	auto samplePerLaunch = GetVariables()->GetUInt32("SamplePerLaunch");

	this->m_Impl->m_First.m_Params.cpuHandle[0].width        = width;
	this->m_Impl->m_First.m_Params.cpuHandle[0].height       = height;
	this->m_Impl->m_First.m_Params.cpuHandle[0].posiBuffer   = m_Impl->m_SharedFramebuffer->GetComponent<test::RTCUDABufferFBComponent<float3>>("GPosition")->GetHandle().getDevicePtr();
	this->m_Impl->m_First.m_Params.cpuHandle[0].normBuffer   = m_Impl->m_SharedFramebuffer->GetComponent<test::RTCUDABufferFBComponent<float3>>("GNormal"  )->GetHandle().getDevicePtr();
	this->m_Impl->m_First.m_Params.cpuHandle[0].emitBuffer   = m_Impl->m_SharedFramebuffer->GetComponent<test::RTCUDABufferFBComponent<float3>>("GEmission")->GetHandle().getDevicePtr();
	this->m_Impl->m_First.m_Params.cpuHandle[0].diffBuffer   = m_Impl->m_SharedFramebuffer->GetComponent<test::RTCUDABufferFBComponent<float3>>("GDiffuse" )->GetHandle().getDevicePtr();
	this->m_Impl->m_First.m_Params.cpuHandle[0].distBuffer   = m_Impl->m_SharedFramebuffer->GetComponent<test::RTCUDABufferFBComponent<float >>("GDistance")->GetHandle().getDevicePtr();
	this->m_Impl->m_First.m_Params.cpuHandle[0].seedBuffer   = m_Impl->m_UniqueFramebuffer->GetComponent<test::RTCUDABufferFBComponent<unsigned int>>("Seed")->GetHandle().getDevicePtr();
	this->m_Impl->m_First.m_Params.cpuHandle[0].motiBuffer   = m_Impl->m_UniqueFramebuffer->GetComponent<test::RTCUDABufferFBComponent<int2>>("Motion2D")->GetHandle().getDevicePtr();
	this->m_Impl->m_First.m_Params.cpuHandle[0].updateMotion = m_Impl->m_UpdateMotion;
	//TODO
	cudaMemcpyAsync(this->m_Impl->m_First.m_Params.gpuHandle.getDevicePtr(), &this->m_Impl->m_First.m_Params.cpuHandle[0], sizeof(RayFirstParams), cudaMemcpyHostToDevice, pUserData->stream);
	this->m_Impl->m_First.m_Pipeline.launch(pUserData->stream, this->m_Impl->m_First.m_Params.gpuHandle.getDevicePtr(), this->m_Impl->m_First.m_ShaderBindingTable, width, height, 1);
	RTLIB_CU_CHECK(cuStreamSynchronize(pUserData->stream));

	this->m_Impl->m_Second.m_Params.cpuHandle[0].width           = width;
	this->m_Impl->m_Second.m_Params.cpuHandle[0].height          = height;
	this->m_Impl->m_Second.m_Params.cpuHandle[0].samplePerALL    = samplePerAll;
	this->m_Impl->m_Second.m_Params.cpuHandle[0].samplePerLaunch = samplePerLaunch;
	this->m_Impl->m_Second.m_Params.cpuHandle[0].numCandidates   = GetVariables()->GetUInt32("NumCandidates");
	this->m_Impl->m_Second.m_Params.cpuHandle[0].seedBuffer      = m_Impl->m_UniqueFramebuffer->GetComponent<test::RTCUDABufferFBComponent<unsigned int>>("Seed")->GetHandle().getDevicePtr();
	this->m_Impl->m_Second.m_Params.cpuHandle[0].motiBuffer      = m_Impl->m_UniqueFramebuffer->GetComponent<test::RTCUDABufferFBComponent<int2>>  ("Motion2D"  )->GetHandle().getDevicePtr();
	this->m_Impl->m_Second.m_Params.cpuHandle[0].curPosiBuffer   = m_Impl->m_SharedFramebuffer->GetComponent<test::RTCUDABufferFBComponent<float3>>("GPosition" )->GetHandle().getDevicePtr();
	this->m_Impl->m_Second.m_Params.cpuHandle[0].curNormBuffer   = m_Impl->m_SharedFramebuffer->GetComponent<test::RTCUDABufferFBComponent<float3>>("GNormal"   )->GetHandle().getDevicePtr();
	this->m_Impl->m_Second.m_Params.cpuHandle[0].prvPosiBuffer   = m_Impl->m_UniqueFramebuffer->GetComponent<test::RTCUDABufferFBComponent<float3>>("GPosition2")->GetHandle().getDevicePtr();
	this->m_Impl->m_Second.m_Params.cpuHandle[0].prvNormBuffer   = m_Impl->m_UniqueFramebuffer->GetComponent<test::RTCUDABufferFBComponent<float3>>("GNormal2"  )->GetHandle().getDevicePtr();
	this->m_Impl->m_Second.m_Params.cpuHandle[0].curDiffBuffer   = m_Impl->m_SharedFramebuffer->GetComponent<test::RTCUDABufferFBComponent<float3>>("GDiffuse"  )->GetHandle().getDevicePtr();
	this->m_Impl->m_Second.m_Params.cpuHandle[0].prvDiffBuffer   = m_Impl->m_UniqueFramebuffer->GetComponent<test::RTCUDABufferFBComponent<float3>>("GDiffuse2" )->GetHandle().getDevicePtr();
	this->m_Impl->m_Second.m_Params.cpuHandle[0].emitBuffer      = m_Impl->m_SharedFramebuffer->GetComponent<test::RTCUDABufferFBComponent<float3>>("GEmission" )->GetHandle().getDevicePtr();
	this->m_Impl->m_Second.m_Params.cpuHandle[0].accumBuffer     = m_Impl->m_SharedFramebuffer->GetComponent<test::RTCUDABufferFBComponent<float3>>("RAccum"    )->GetHandle().getDevicePtr();
	this->m_Impl->m_Second.m_Params.cpuHandle[0].distBuffer      = m_Impl->m_SharedFramebuffer->GetComponent<test::RTCUDABufferFBComponent<float >>("GDistance" )->GetHandle().getDevicePtr();
	this->m_Impl->m_Second.m_Params.cpuHandle[0].resvBuffer      = m_Impl->m_UniqueFramebuffer->GetComponent<test::RTCUDABufferFBComponent<Reservoir<LightRec>>>("Reservoir" + std::to_string(m_Impl->m_CurReservoirIdx))->GetHandle().getDevicePtr();
	this->m_Impl->m_Second.m_Params.cpuHandle[0].tempBuffer      = m_Impl->m_UniqueFramebuffer->GetComponent<test::RTCUDABufferFBComponent<ReservoirState>>("ResvState")->GetHandle().getDevicePtr();
	this->m_Impl->m_Second.m_Params.cpuHandle[0].frameBuffer     = m_Impl->m_SharedFramebuffer->GetComponent<test::RTCUGLBufferFBComponent<uchar4>>("RFrame")->GetHandle().map();
	this->m_Impl->m_Second.m_Params.cpuHandle[0].meshLights      = m_Impl->GetMeshLightList();

	cudaMemcpyAsync(this->m_Impl->m_Second.m_Params.gpuHandle.getDevicePtr(), &this->m_Impl->m_Second.m_Params.cpuHandle[0], sizeof(RaySecondParams), cudaMemcpyHostToDevice, pUserData->stream);
	this->m_Impl->m_Second.m_Pipeline.launch(pUserData->stream, this->m_Impl->m_Second.m_Params.gpuHandle.getDevicePtr() , this->m_Impl->m_Second.m_ShaderBindingTableForInit, width, height, 1);
	RTLIB_CU_CHECK(cuStreamSynchronize(pUserData->stream));

	if(m_Impl->m_UpdateMotion && GetVariables()->GetBool("ReuseTemporal")) {
		unsigned int kNumBlocks = 64;
		uint2 numThreads    = (make_uint2(width - 1, height - 1) / kNumBlocks) + make_uint2(1);
		auto params         = m_Impl->m_Second.m_Params.gpuHandle.getDevicePtr();
		auto prvResvBuffer  = m_Impl->m_UniqueFramebuffer->GetComponent<test::RTCUDABufferFBComponent<Reservoir<LightRec>>>("Reservoir" + std::to_string(m_Impl->m_PrvReservoirIdx))->GetHandle().getDevicePtr();
		auto curResvBuffer  = m_Impl->m_UniqueFramebuffer->GetComponent<test::RTCUDABufferFBComponent<Reservoir<LightRec>>>("Reservoir" + std::to_string(m_Impl->m_CurReservoirIdx))->GetHandle().getDevicePtr();
		auto tmpStatBuffer  = m_Impl->m_UniqueFramebuffer->GetComponent<test::RTCUDABufferFBComponent<ReservoirState>>("ResvState")->GetHandle().getDevicePtr();
		auto camEye         = m_Impl->m_Camera.getEye();
		void* args[] = {
			reinterpret_cast<void*>(&prvResvBuffer),
			reinterpret_cast<void*>(&curResvBuffer),
			reinterpret_cast<void*>(&tmpStatBuffer),
			reinterpret_cast<void*>(&params),
			reinterpret_cast<void*>(&camEye),
			reinterpret_cast<void*>(&width ),
			reinterpret_cast<void*>(&height),
		};
		this->m_Impl->m_Second.m_CUDAFunctions["Second.TemporalCache"].launch(make_uint3(kNumBlocks, kNumBlocks, 1), make_uint3(numThreads, 1), 0, pUserData->stream, args, nullptr);
		RTLIB_CU_CHECK(cuStreamSynchronize(pUserData->stream));
	}
	if (GetVariables()->GetBool("ReuseSpatial")) {
		unsigned int kNumBlocks = 64;
		int range = GetVariables()->GetUInt32("RangeSpatial");
		
		uint2 numThreads = (make_uint2(width - 1, height - 1) / kNumBlocks) + make_uint2(1);
		auto params = m_Impl->m_Second.m_Params.gpuHandle.getDevicePtr();
		int  sample = GetVariables()->GetUInt32("SampleSpatial");
		for (int i  = 0; i < this->GetVariables()->GetUInt32("IterationSpatial"); ++i) {
			auto inResvBuffer  = m_Impl->m_UniqueFramebuffer->GetComponent<test::RTCUDABufferFBComponent<Reservoir<LightRec>>>("Reservoir" + std::to_string(m_Impl->m_CurReservoirIdx))->GetHandle().getDevicePtr();
			auto outResvBuffer = m_Impl->m_UniqueFramebuffer->GetComponent<test::RTCUDABufferFBComponent<Reservoir<LightRec>>>("Reservoir" + std::to_string(m_Impl->m_FnlReservoirIdx))->GetHandle().getDevicePtr();
			auto tmpStatBuffer = m_Impl->m_UniqueFramebuffer->GetComponent<test::RTCUDABufferFBComponent<ReservoirState>>("ResvState")->GetHandle().getDevicePtr();
			void* args[] = {
				reinterpret_cast<void*>(&inResvBuffer),
				reinterpret_cast<void*>(&outResvBuffer),
				reinterpret_cast<void*>(&tmpStatBuffer),
				reinterpret_cast<void*>(&params),
				reinterpret_cast<void*>(&width),
				reinterpret_cast<void*>(&height),
				reinterpret_cast<void*>(&sample),
				reinterpret_cast<void*>(&range),
			};
			this->m_Impl->m_Second.m_CUDAFunctions["Second.SpatialCache"].launch(make_uint3(kNumBlocks, kNumBlocks, 1), make_uint3(numThreads, 1), 0, pUserData->stream, args, nullptr);
			RTLIB_CU_CHECK(cuStreamSynchronize(pUserData->stream));
			std::swap(m_Impl->m_CurReservoirIdx, m_Impl->m_FnlReservoirIdx);
		}
		

		this->m_Impl->m_Second.m_Params.cpuHandle[0].resvBuffer = m_Impl->m_UniqueFramebuffer->GetComponent<test::RTCUDABufferFBComponent<Reservoir<LightRec>>>("Reservoir" + std::to_string(m_Impl->m_CurReservoirIdx))->GetHandle().getDevicePtr();
		cudaMemcpyAsync(this->m_Impl->m_Second.m_Params.gpuHandle.getDevicePtr(), &this->m_Impl->m_Second.m_Params.cpuHandle[0], sizeof(RaySecondParams), cudaMemcpyHostToDevice, pUserData->stream);
	}
	this->m_Impl->m_Second.m_Pipeline.launch(pUserData->stream, this->m_Impl->m_Second.m_Params.gpuHandle.getDevicePtr() , this->m_Impl->m_Second.m_ShaderBindingTableForDraw, width, height, 1);
	RTLIB_CU_CHECK(cuStreamSynchronize(pUserData->stream));

	m_Impl->m_SharedFramebuffer->GetComponent<test::RTCUGLBufferFBComponent<uchar4>>("RFrame")->GetHandle().unmap();

	samplePerAll += samplePerLaunch;

	this->m_Impl->m_Second.m_Params.cpuHandle[0].samplePerALL = samplePerAll;
	GetVariables()->SetUInt32("SamplePerAll", samplePerAll);
	std::swap(m_Impl->m_CurReservoirIdx, m_Impl->m_PrvReservoirIdx);

	if (GetVariables()->GetBool("Launched")) {
		auto sampleForBudget = GetVariables()->GetUInt32("SampleForBudget");
		if (samplePerAll >= sampleForBudget) {
			GetVariables()->SetBool("Launched", false);
			GetVariables()->SetBool("Started", false);
			GetVariables()->SetUInt32("SamplePerAll", 0);
			pUserData->finished = true;
		}
		else {
			pUserData->finished = false;
		}
	}
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

	m_Impl->m_UpdateMotion = true;
	if (this->m_Impl->m_CurEventFlags   & TEST24_EVENT_FLAG_BIT_UPDATE_CAMERA)
	{
		float aspect = (float)m_Impl->m_SharedFramebuffer->GetWidth() / (float)m_Impl->m_SharedFramebuffer->GetHeight();
		this->m_Impl->m_Camera = this->m_Impl->m_CameraController->GetCamera(aspect);
		auto eye       = this->m_Impl->m_Camera.getEye();
		auto [u, v, w] = this->m_Impl->m_Camera.getUVW();

		this->m_Impl->m_First.m_RGRecordBuffer.cpuHandle[0].data.pinhole[FRAME_TYPE_PREVIOUS] = this->m_Impl->m_First.m_RGRecordBuffer.cpuHandle[0].data.pinhole[FRAME_TYPE_CURRENT];
		this->m_Impl->m_First.m_RGRecordBuffer.cpuHandle[0].data.pinhole[FRAME_TYPE_CURRENT].eye = eye;
		this->m_Impl->m_First.m_RGRecordBuffer.cpuHandle[0].data.pinhole[FRAME_TYPE_CURRENT].u   = u;
		this->m_Impl->m_First.m_RGRecordBuffer.cpuHandle[0].data.pinhole[FRAME_TYPE_CURRENT].v   = v;
		this->m_Impl->m_First.m_RGRecordBuffer.cpuHandle[0].data.pinhole[FRAME_TYPE_CURRENT].w   = w;
		this->m_Impl->m_First.m_RGRecordBuffer.Upload();

		for (int i = 0; i < 2; ++i) {
			this->m_Impl->m_Second.m_RGRecordBuffers.cpuHandle[i].data.pinhole[FRAME_TYPE_PREVIOUS]    = this->m_Impl->m_Second.m_RGRecordBuffers.cpuHandle[0].data.pinhole[FRAME_TYPE_CURRENT];
			this->m_Impl->m_Second.m_RGRecordBuffers.cpuHandle[i].data.pinhole[FRAME_TYPE_CURRENT].eye = eye;
			this->m_Impl->m_Second.m_RGRecordBuffers.cpuHandle[i].data.pinhole[FRAME_TYPE_CURRENT].u   = u;
			this->m_Impl->m_Second.m_RGRecordBuffers.cpuHandle[i].data.pinhole[FRAME_TYPE_CURRENT].v   = v;
			this->m_Impl->m_Second.m_RGRecordBuffers.cpuHandle[i].data.pinhole[FRAME_TYPE_CURRENT].w   = w;
		}

		this->m_Impl->m_Second.m_RGRecordBuffers.Upload();
	}
	else {
		this->m_Impl->m_First.m_RGRecordBuffer.cpuHandle[0].data.pinhole[FRAME_TYPE_PREVIOUS] = this->m_Impl->m_First.m_RGRecordBuffer.cpuHandle[0].data.pinhole[FRAME_TYPE_CURRENT];
		//GPU 側でUpdateするため不要
		//this->m_Impl->m_First.m_RGRecordBuffer.Upload();

		for (int i = 0; i < 2; ++i) {
			this->m_Impl->m_Second.m_RGRecordBuffers.cpuHandle[i].data.pinhole[FRAME_TYPE_PREVIOUS] = this->m_Impl->m_Second.m_RGRecordBuffers.cpuHandle[i].data.pinhole[FRAME_TYPE_CURRENT];
		}
		//GPU 側でUpdateするため不要
		//this->m_Impl->m_Second.m_RGRecordBuffers.Upload();
	}

	if (this->m_Impl->m_CurEventFlags   & TEST24_EVENT_FLAG_BIT_FLUSH_FRAME)
	{
		std::vector<float3> zeroAccumValues(this->m_Impl->m_SharedFramebuffer->GetWidth() * this->m_Impl->m_SharedFramebuffer->GetHeight(), make_float3(0.0f));
		cudaMemcpy(this->m_Impl->m_SharedFramebuffer->GetComponent<test::RTCUDABufferFBComponent<float3>>("RAccum")->GetHandle().getDevicePtr(), zeroAccumValues.data(), sizeof(float3) * this->m_Impl->m_SharedFramebuffer->GetWidth() * this->m_Impl->m_SharedFramebuffer->GetHeight(), cudaMemcpyHostToDevice);
		this->m_Impl->m_Second.m_Params.cpuHandle[0].samplePerALL = 0;
		GetVariables()->SetUInt32("SamplePerAll", 0);
	}

	if (this->m_Impl->m_CurEventFlags   & TEST24_EVENT_FLAG_BIT_RESIZE_FRAME) {
		std::cout << "Regen!\n";
		m_Impl->m_UniqueFramebuffer->Resize(m_Impl->m_SharedFramebuffer->GetWidth(), m_Impl->m_SharedFramebuffer->GetHeight());

		m_Impl->m_UniqueFramebuffer->GetComponent<test::RTCUDABufferFBComponent<float3>>("GPosition2")->GetHandle().upload(std::vector<float3>(m_Impl->m_SharedFramebuffer->GetWidth() * m_Impl->m_SharedFramebuffer->GetHeight(), { 0.0f,0.0f,0.0f }));

		m_Impl->m_UniqueFramebuffer->GetComponent<test::RTCUDABufferFBComponent<float3>>("GNormal2"  )->GetHandle().upload(std::vector<float3>(m_Impl->m_SharedFramebuffer->GetWidth() * m_Impl->m_SharedFramebuffer->GetHeight(), { 0.0f,0.0f,0.0f }));

		m_Impl->m_UniqueFramebuffer->GetComponent<test::RTCUDABufferFBComponent<float3>>("GDiffuse2")->GetHandle().upload(std::vector<float3>( m_Impl->m_SharedFramebuffer->GetWidth() * m_Impl->m_SharedFramebuffer->GetHeight(), { 0.0f,0.0f,0.0f }));

		std::vector<unsigned int> seedData(this->m_Impl->m_SharedFramebuffer->GetWidth() * this->m_Impl->m_SharedFramebuffer->GetHeight());
		std::random_device rd;
		std::mt19937 mt(rd());
		std::generate(seedData.begin(), seedData.end(), mt);
		m_Impl->m_UniqueFramebuffer->GetComponent<test::RTCUDABufferFBComponent<unsigned int>>("Seed")->GetHandle().upload(seedData);

		std::vector<Reservoir<LightRec>> reservoirs(this->m_Impl->m_SharedFramebuffer->GetWidth() * this->m_Impl->m_SharedFramebuffer->GetHeight());
		m_Impl->m_UniqueFramebuffer->GetComponent<test::RTCUDABufferFBComponent<Reservoir<LightRec>>>("Reservoir0")->GetHandle().upload(reservoirs);
		m_Impl->m_UniqueFramebuffer->GetComponent<test::RTCUDABufferFBComponent<Reservoir<LightRec>>>("Reservoir1")->GetHandle().upload(reservoirs);
		m_Impl->m_UniqueFramebuffer->GetComponent<test::RTCUDABufferFBComponent<Reservoir<LightRec>>>("Reservoir2")->GetHandle().upload(reservoirs);

		std::vector<ReservoirState> resvStates(this->m_Impl->m_SharedFramebuffer->GetWidth() * this->m_Impl->m_SharedFramebuffer->GetHeight(), { 0.0f });
		m_Impl->m_UniqueFramebuffer->GetComponent<test::RTCUDABufferFBComponent<ReservoirState>>("ResvState")->GetHandle().upload(resvStates);

		std::vector<int2> motionData(this->m_Impl->m_SharedFramebuffer->GetWidth() * this->m_Impl->m_SharedFramebuffer->GetHeight(), { 0,0 });
		m_Impl->m_UniqueFramebuffer->GetComponent<test::RTCUDABufferFBComponent<int2>>("Motion2D")->GetHandle().upload(motionData);
		m_Impl->m_UpdateMotion = false;
	}

	auto samplePerAll = GetVariables()->GetUInt32("SamplePerAll");
	bool shouldRegen = ((samplePerAll + this->m_Impl->m_Second.m_Params.cpuHandle[0].samplePerLaunch) / 1024 != samplePerAll / 1024);
	if (shouldRegen)
	{
		std::cout << "Regen!\n";
		std::vector<unsigned int> seedData(this->m_Impl->m_SharedFramebuffer->GetWidth() * this->m_Impl->m_SharedFramebuffer->GetHeight());
		std::random_device rd;
		std::mt19937 mt(rd());
		std::generate(seedData.begin(), seedData.end(), mt);
		m_Impl->m_UniqueFramebuffer->GetComponent<test::RTCUDABufferFBComponent<unsigned int>>("Seed")->GetHandle().upload(seedData);
	}
	if (this->m_Impl->m_CurEventFlags   & TEST24_EVENT_FLAG_BIT_UPDATE_LIGHT)
	{
		auto lightColor = make_float4(this->m_Impl->m_BgLightColor, 1.0f);
		this->m_Impl->m_First.m_MSRecordBuffers.cpuHandle[RAY_TYPE_RADIANCE].data.bgColor = lightColor;
		RTLIB_CUDA_CHECK(cudaMemcpy(this->m_Impl->m_First.m_MSRecordBuffers.gpuHandle.getDevicePtr() + RAY_TYPE_RADIANCE, &this->m_Impl->m_First.m_MSRecordBuffers.cpuHandle[RAY_TYPE_RADIANCE], sizeof(MissRecord), cudaMemcpyHostToDevice));
		m_Impl->m_UpdateMotion    = false;
	}

	if (this->m_Impl->m_CurEventFlags & TEST24_EVENT_FLAG_BIT_CHANGE_TRACE) {
		m_Impl->m_UpdateMotion    = false;
	}
	this->m_Impl->m_PrvEventFlags = this->m_Impl->m_CurEventFlags;
}

Test24ReSTIROPXTracer::~Test24ReSTIROPXTracer() {}

void Test24ReSTIROPXTracer::InitFrameResources()
{
	m_Impl->m_UniqueFramebuffer = std::unique_ptr<test::RTFramebuffer>(new test::RTFramebuffer(m_Impl->m_SharedFramebuffer->GetWidth(), m_Impl->m_SharedFramebuffer->GetHeight()));

	m_Impl->m_UniqueFramebuffer->AddComponent<test::RTCUDABufferFBComponent<float3>>("GPosition2");
	m_Impl->m_UniqueFramebuffer->GetComponent<test::RTCUDABufferFBComponent<float3>>("GPosition2")->GetHandle().upload(std::vector<float3>(m_Impl->m_SharedFramebuffer->GetWidth()* m_Impl->m_SharedFramebuffer->GetHeight(), { 0.0f,0.0f,0.0f }));

	m_Impl->m_UniqueFramebuffer->AddComponent<test::RTCUDABufferFBComponent<float3>>("GNormal2");
	m_Impl->m_UniqueFramebuffer->GetComponent<test::RTCUDABufferFBComponent<float3>>("GNormal2")->GetHandle().upload(std::vector<float3>(m_Impl->m_SharedFramebuffer->GetWidth() * m_Impl->m_SharedFramebuffer->GetHeight(), { 0.0f,0.0f,0.0f }));

	m_Impl->m_UniqueFramebuffer->AddComponent<test::RTCUDABufferFBComponent<float3>>("GDiffuse2");
	m_Impl->m_UniqueFramebuffer->GetComponent<test::RTCUDABufferFBComponent<float3>>("GDiffuse2")->GetHandle().upload(std::vector<float3>(m_Impl->m_SharedFramebuffer->GetWidth() * m_Impl->m_SharedFramebuffer->GetHeight(), { 0.0f,0.0f,0.0f }));

	std::vector<unsigned int> seedData(this->m_Impl->m_SharedFramebuffer->GetWidth() * this->m_Impl->m_SharedFramebuffer->GetHeight());
	std::random_device rd;
	std::mt19937 mt(rd());
	std::generate(seedData.begin(), seedData.end(), mt);
	m_Impl->m_UniqueFramebuffer->AddComponent<test::RTCUDABufferFBComponent<unsigned int>>("Seed");
	m_Impl->m_UniqueFramebuffer->GetComponent<test::RTCUDABufferFBComponent<unsigned int>>("Seed")->GetHandle().upload(seedData);

	std::vector<Reservoir<LightRec>> reservoirs(this->m_Impl->m_SharedFramebuffer->GetWidth() * this->m_Impl->m_SharedFramebuffer->GetHeight());
	m_Impl->m_UniqueFramebuffer->AddComponent<test::RTCUDABufferFBComponent<Reservoir<LightRec>>>("Reservoir0");
	m_Impl->m_UniqueFramebuffer->AddComponent<test::RTCUDABufferFBComponent<Reservoir<LightRec>>>("Reservoir1");
	m_Impl->m_UniqueFramebuffer->AddComponent<test::RTCUDABufferFBComponent<Reservoir<LightRec>>>("Reservoir2");
	m_Impl->m_UniqueFramebuffer->GetComponent<test::RTCUDABufferFBComponent<Reservoir<LightRec>>>("Reservoir0")->GetHandle().upload(reservoirs);
	m_Impl->m_UniqueFramebuffer->GetComponent<test::RTCUDABufferFBComponent<Reservoir<LightRec>>>("Reservoir1")->GetHandle().upload(reservoirs);
	m_Impl->m_UniqueFramebuffer->GetComponent<test::RTCUDABufferFBComponent<Reservoir<LightRec>>>("Reservoir2")->GetHandle().upload(reservoirs);
	
	std::vector<ReservoirState> resvStates(this->m_Impl->m_SharedFramebuffer->GetWidth() * this->m_Impl->m_SharedFramebuffer->GetHeight(), { 0.0f });
	m_Impl->m_UniqueFramebuffer->AddComponent<test::RTCUDABufferFBComponent<ReservoirState>>("ResvState");
	m_Impl->m_UniqueFramebuffer->GetComponent<test::RTCUDABufferFBComponent<ReservoirState>>("ResvState")->GetHandle().upload(resvStates);

	std::vector<int2> motionData(this->m_Impl->m_SharedFramebuffer->GetWidth() * this->m_Impl->m_SharedFramebuffer->GetHeight(), { 0,0 });
	m_Impl->m_UniqueFramebuffer->AddComponent<test::RTCUDABufferFBComponent<int2>>("Motion2D");
	m_Impl->m_UniqueFramebuffer->GetComponent<test::RTCUDABufferFBComponent<int2>>("Motion2D")->GetHandle().upload(motionData);
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
	{
		auto rayReSTIRPtxFile = std::ifstream(TEST_TEST24_RESTIR_OPX_CUDA_PATH "/ReservoirReuse.ptx", std::ios::binary);
		if (!rayReSTIRPtxFile.is_open())
			throw std::runtime_error("Failed To Load RaySecond.ptx!");
		auto rayReSTIRPtxData = std::string((std::istreambuf_iterator<char>(rayReSTIRPtxFile)), (std::istreambuf_iterator<char>()));
		rayReSTIRPtxFile.close(); 
		this->m_Impl->m_Second.m_CUDAModules[ "ReservoirReuse"]        = rtlib::CUDAModule(rayReSTIRPtxData.data());
		this->m_Impl->m_Second.m_CUDAFunctions["Second.SpatialCache"]  = this->m_Impl->m_Second.m_CUDAModules["ReservoirReuse"].getFunction( "combineSpatialReservoirs");
		this->m_Impl->m_Second.m_CUDAFunctions["Second.TemporalCache"] = this->m_Impl->m_Second.m_CUDAModules["ReservoirReuse"].getFunction("combineTemporalReservoirs");
	}
}

void Test24ReSTIROPXTracer::InitShaderBindingTable()
{
	/**First **/
	{
		float aspect = static_cast<float>(m_Impl->m_SharedFramebuffer->GetWidth()) / static_cast<float>(m_Impl->m_SharedFramebuffer->GetHeight());
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
	auto lightGASHandle = m_Impl->m_TopLevelAS->GetInstanceSets()[0]->GetInstance(1).baseGASHandle;
	for (auto& mesh : lightGASHandle->GetMeshes())
	{
		//Select NEE Light
		if (mesh->GetUniqueResource()->variables.GetBool("useNEE")) {
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
	auto samplePerAll = GetVariables()->GetUInt32("SamplePerAll");
	auto samplePerLaunch = GetVariables()->GetUInt32("SamplePerLaunch");

	auto tlas = this->m_Impl->m_TopLevelAS;
	this->m_Impl->m_First.m_Params.Alloc(1);
	this->m_Impl->m_First.m_Params.cpuHandle[0].gasHandle    = tlas->GetHandle();
	this->m_Impl->m_First.m_Params.cpuHandle[0].width        = this->m_Impl->m_SharedFramebuffer->GetWidth();
	this->m_Impl->m_First.m_Params.cpuHandle[0].height       = this->m_Impl->m_SharedFramebuffer->GetHeight();
	this->m_Impl->m_First.m_Params.cpuHandle[0].posiBuffer   = m_Impl->m_SharedFramebuffer->GetComponent<test::RTCUDABufferFBComponent<float3>>("GPosition")->GetHandle().getDevicePtr();
	this->m_Impl->m_First.m_Params.cpuHandle[0].normBuffer   = m_Impl->m_SharedFramebuffer->GetComponent<test::RTCUDABufferFBComponent<float3>>("GNormal"  )->GetHandle().getDevicePtr();
	this->m_Impl->m_First.m_Params.cpuHandle[0].emitBuffer   = m_Impl->m_SharedFramebuffer->GetComponent<test::RTCUDABufferFBComponent<float3>>("GEmission")->GetHandle().getDevicePtr();
	this->m_Impl->m_First.m_Params.cpuHandle[0].diffBuffer   = m_Impl->m_SharedFramebuffer->GetComponent<test::RTCUDABufferFBComponent<float3>>("GDiffuse" )->GetHandle().getDevicePtr();
	this->m_Impl->m_First.m_Params.cpuHandle[0].distBuffer   = m_Impl->m_SharedFramebuffer->GetComponent<test::RTCUDABufferFBComponent<float >>("GDistance")->GetHandle().getDevicePtr();
	this->m_Impl->m_First.m_Params.cpuHandle[0].seedBuffer   = m_Impl->m_UniqueFramebuffer->GetComponent<test::RTCUDABufferFBComponent<unsigned int>>("Seed")->GetHandle().getDevicePtr();
	this->m_Impl->m_First.m_Params.cpuHandle[0].motiBuffer   = m_Impl->m_UniqueFramebuffer->GetComponent<test::RTCUDABufferFBComponent<int2>>("Motion2D")->GetHandle().getDevicePtr();
	this->m_Impl->m_First.m_Params.cpuHandle[0].updateMotion = m_Impl->m_UpdateMotion;
	this->m_Impl->m_First.m_Params.Upload();
	this->m_Impl->m_Second.m_Params.Alloc(1);
	this->m_Impl->m_Second.m_Params.cpuHandle[0].gasHandle       = tlas->GetHandle();
	this->m_Impl->m_Second.m_Params.cpuHandle[0].width           = this->m_Impl->m_SharedFramebuffer->GetWidth();
	this->m_Impl->m_Second.m_Params.cpuHandle[0].height          = this->m_Impl->m_SharedFramebuffer->GetHeight();
	this->m_Impl->m_Second.m_Params.cpuHandle[0].samplePerALL    = samplePerAll;
	this->m_Impl->m_Second.m_Params.cpuHandle[0].samplePerLaunch = samplePerLaunch;
	this->m_Impl->m_Second.m_Params.cpuHandle[0].numCandidates   = GetVariables()->GetUInt32("NumCandidates");
	this->m_Impl->m_Second.m_Params.cpuHandle[0].seedBuffer      = m_Impl->m_UniqueFramebuffer->GetComponent<test::RTCUDABufferFBComponent<unsigned int>>("Seed")->GetHandle().getDevicePtr();
	this->m_Impl->m_Second.m_Params.cpuHandle[0].motiBuffer      = m_Impl->m_UniqueFramebuffer->GetComponent<test::RTCUDABufferFBComponent<int2>>(  "Motion2D" )->GetHandle().getDevicePtr();
	this->m_Impl->m_Second.m_Params.cpuHandle[0].curPosiBuffer   = m_Impl->m_SharedFramebuffer->GetComponent<test::RTCUDABufferFBComponent<float3>>("GPosition")->GetHandle().getDevicePtr();
	this->m_Impl->m_Second.m_Params.cpuHandle[0].curNormBuffer   = m_Impl->m_SharedFramebuffer->GetComponent<test::RTCUDABufferFBComponent<float3>>("GNormal"  )->GetHandle().getDevicePtr();
	this->m_Impl->m_Second.m_Params.cpuHandle[0].prvPosiBuffer   = m_Impl->m_UniqueFramebuffer->GetComponent<test::RTCUDABufferFBComponent<float3>>("GPosition2")->GetHandle().getDevicePtr();
	this->m_Impl->m_Second.m_Params.cpuHandle[0].prvNormBuffer   = m_Impl->m_UniqueFramebuffer->GetComponent<test::RTCUDABufferFBComponent<float3>>("GNormal2")->GetHandle().getDevicePtr();
	this->m_Impl->m_Second.m_Params.cpuHandle[0].emitBuffer      = m_Impl->m_SharedFramebuffer->GetComponent<test::RTCUDABufferFBComponent<float3>>("GEmission")->GetHandle().getDevicePtr();
	this->m_Impl->m_Second.m_Params.cpuHandle[0].curDiffBuffer   = m_Impl->m_SharedFramebuffer->GetComponent<test::RTCUDABufferFBComponent<float3>>("GDiffuse" )->GetHandle().getDevicePtr();
	this->m_Impl->m_Second.m_Params.cpuHandle[0].prvDiffBuffer   = m_Impl->m_UniqueFramebuffer->GetComponent<test::RTCUDABufferFBComponent<float3>>("GDiffuse2")->GetHandle().getDevicePtr();
	this->m_Impl->m_Second.m_Params.cpuHandle[0].distBuffer      = m_Impl->m_SharedFramebuffer->GetComponent<test::RTCUDABufferFBComponent<float >>("GDistance")->GetHandle().getDevicePtr();
	this->m_Impl->m_Second.m_Params.cpuHandle[0].resvBuffer      = m_Impl->m_UniqueFramebuffer->GetComponent<test::RTCUDABufferFBComponent<Reservoir<LightRec>>>("Reservoir"+std::to_string(m_Impl->m_FnlReservoirIdx))->GetHandle().getDevicePtr();
	this->m_Impl->m_Second.m_Params.cpuHandle[0].accumBuffer     = m_Impl->m_SharedFramebuffer->GetComponent<test::RTCUDABufferFBComponent<float3>>("RAccum"   )->GetHandle().getDevicePtr();
	this->m_Impl->m_Second.m_Params.cpuHandle[0].frameBuffer     = nullptr;
	this->m_Impl->m_Second.m_Params.cpuHandle[0].meshLights      = m_Impl->GetMeshLightList();
	this->m_Impl->m_Second.m_Params.Upload();
}

void Test24ReSTIROPXTracer::FreeFrameResources()
{
	this->m_Impl->m_UniqueFramebuffer.reset();
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

