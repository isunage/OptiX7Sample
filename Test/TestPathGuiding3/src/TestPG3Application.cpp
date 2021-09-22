#include "../include/TestPG3Application.h"
#include <RTLib/Utils.h>
#include <random>
//Init

class TestPG3SimpleTracer :public test::RTTracer {
public:
	struct UserData {
		STree        sTree;
		uchar4* frameBuffer;
		unsigned int samplePerAll;
		unsigned int samplePerLaunch;
	};
private:
	using RayGRecord = rtlib::SBTRecord<RayGenData>;
	using MissRecord = rtlib::SBTRecord<MissData>;
	using HitGRecord = rtlib::SBTRecord<HitgroupData>;
public:
	TestPG3SimpleTracer(TestPG3Application* app) {
		m_ParentApp = app;
	}
	// RTTracer を介して継承されました
	virtual void Initialize() override
	{
		this->InitFrameResources();
		this->InitShaderBindingTable();
		this->InitLaunchParams();
	}
	virtual void Launch(const test::RTTraceConfig& config) override {
		UserData* pUserData = (UserData*)config.pUserData;
		if (!pUserData) {
			return;
		}
		m_Params.cpuHandle[0].width = config.width;
		m_Params.cpuHandle[0].height = config.height;
		m_Params.cpuHandle[0].sdTree = pUserData->sTree;
		m_Params.cpuHandle[0].accumBuffer = m_AccumBuffer.getDevicePtr();
		m_Params.cpuHandle[0].frameBuffer = pUserData->frameBuffer;
		m_Params.cpuHandle[0].seedBuffer = m_SeedBuffer.getDevicePtr();
		m_Params.cpuHandle[0].isBuilt = false;
		m_Params.cpuHandle[0].samplePerLaunch = pUserData->samplePerLaunch;
		cudaMemcpyAsync(m_Params.gpuHandle.getDevicePtr(), &m_Params.cpuHandle[0], sizeof(RayTraceParams), cudaMemcpyHostToDevice, config.stream);
		auto& pipeline = m_ParentApp->GetOPXPipeline("Trace");
		pipeline.launch(config.stream, m_Params.gpuHandle.getDevicePtr(), m_ShaderBindingTable, config.width, config.height, config.depth);
		if (config.isSync) {
			RTLIB_CU_CHECK(cuStreamSynchronize(config.stream));
		}
		m_Params.cpuHandle[0].samplePerALL += m_Params.cpuHandle[0].samplePerLaunch;
		pUserData->samplePerAll = m_Params.cpuHandle[0].samplePerALL;
	}
	virtual void CleanUp() override {
		m_ParentApp = nullptr;
		m_ShaderBindingTable = {};
		m_RGRecordBuffer.Reset();
		m_MSRecordBuffers.Reset();
		m_HGRecordBuffers.Reset();
		m_Params.Reset();
		m_LightHgRecIndex = 0;
	}
	virtual void Update()  override {
		if (m_ParentApp->IsCameraUpdated() || m_ParentApp->IsFrameResized()) {
			auto camera = m_ParentApp->GetCamera();
			m_RGRecordBuffer.cpuHandle[0].data.eye = camera.getEye();
			auto [u, v, w] = camera.getUVW();
			m_RGRecordBuffer.cpuHandle[0].data.u = u;
			m_RGRecordBuffer.cpuHandle[0].data.v = v;
			m_RGRecordBuffer.cpuHandle[0].data.w = w;
			m_RGRecordBuffer.Upload();
		}
		if (m_ParentApp->IsCameraUpdated() || m_ParentApp->IsFrameResized() || m_Params.cpuHandle[0].maxTraceDepth != m_ParentApp->GetMaxTraceDepth() || m_ParentApp->IsLightUpdated()) {
			m_Params.cpuHandle[0].maxTraceDepth = m_ParentApp->GetMaxTraceDepth();
			m_AccumBuffer.resize(m_ParentApp->GetFbWidth() * m_ParentApp->GetFbHeight());
			m_AccumBuffer.upload(std::vector<float3>(m_ParentApp->GetFbWidth() * m_ParentApp->GetFbHeight()));
			m_Params.cpuHandle[0].samplePerALL = 0;
		}
		if (m_ParentApp->IsFrameResized()) {
			std::vector<unsigned int> seedData(m_ParentApp->GetFbWidth() * m_ParentApp->GetFbHeight());
			std::random_device rd;
			std::mt19937 mt(rd());
			std::generate(seedData.begin(), seedData.end(), mt);
			m_SeedBuffer.resize(m_ParentApp->GetFbWidth() * m_ParentApp->GetFbHeight());
			m_SeedBuffer.upload(seedData);
		}
		if (m_ParentApp->IsLightUpdated()) {
			auto light = m_ParentApp->GetLight();
			m_Params.cpuHandle[0].light = light;
			m_HGRecordBuffers.cpuHandle[m_LightHgRecIndex].data.emission = light.emission;
			m_HGRecordBuffers.cpuHandle[m_LightHgRecIndex].data.diffuse = light.emission;
			RTLIB_CUDA_CHECK(cudaMemcpy(m_HGRecordBuffers.gpuHandle.getDevicePtr() + m_LightHgRecIndex, &m_HGRecordBuffers.cpuHandle[m_LightHgRecIndex], sizeof(HitGRecord), cudaMemcpyHostToDevice));
		}
	}
	virtual ~TestPG3SimpleTracer() {}
private:
	void InitFrameResources() {
		std::vector<unsigned int> seedData(m_ParentApp->GetFbWidth() * m_ParentApp->GetFbHeight());
		std::random_device rd;
		std::mt19937 mt(rd());
		std::generate(seedData.begin(), seedData.end(), mt);
		m_AccumBuffer = rtlib::CUDABuffer<float3>(std::vector<float3>(m_ParentApp->GetFbWidth() * m_ParentApp->GetFbHeight()));
		m_SeedBuffer = rtlib::CUDABuffer<uint32_t>(seedData);
	}
	void InitShaderBindingTable() {
		auto SpecifyMaterialType = [](const rtlib::ext::Material& material) {
			auto emitCol = material.GetFloat3As<float3>("emitCol");
			auto tranCol = material.GetFloat3As<float3>("tranCol");
			auto refrIndx = material.GetFloat1("refrIndx");
			auto shinness = material.GetFloat1("shinness");
			auto illum = material.GetUInt32("illum");
			if (illum == 7) {
				return "Refraction";
			}
			else {
				return "Phong";
			}
		};
		auto  tlas = m_ParentApp->GetTLAS();
		auto  camera = m_ParentApp->GetCamera();
		auto& materials = m_ParentApp->GetMaterials();
		m_RGRecordBuffer.cpuHandle.resize(1);
		m_RGRecordBuffer.cpuHandle[0] = m_ParentApp->GetRGProgramGroup("Trace.Default").getSBTRecord<RayGenData>();
		m_RGRecordBuffer.cpuHandle[0].data.eye = camera.getEye();
		auto [u, v, w] = camera.getUVW();
		m_RGRecordBuffer.cpuHandle[0].data.u = u;
		m_RGRecordBuffer.cpuHandle[0].data.v = v;
		m_RGRecordBuffer.cpuHandle[0].data.w = w;
		m_RGRecordBuffer.Upload();
		m_MSRecordBuffers.cpuHandle.resize(RAY_TYPE_COUNT);
		m_MSRecordBuffers.cpuHandle[RAY_TYPE_RADIANCE] = m_ParentApp->GetMSProgramGroup("Trace.Radiance").getSBTRecord<MissData>();
		m_MSRecordBuffers.cpuHandle[RAY_TYPE_RADIANCE].data.bgColor = { 0,0,0,0 };
		m_MSRecordBuffers.cpuHandle[RAY_TYPE_OCCLUSION] = m_ParentApp->GetMSProgramGroup("Trace.Occluded").getSBTRecord<MissData>();
		m_MSRecordBuffers.Upload();
		m_HGRecordBuffers.cpuHandle.resize(tlas->GetSbtCount() * RAY_TYPE_COUNT);
		{
			size_t sbtOffset = 0;
			for (auto& instanceSet : tlas->GetInstanceSets()) {
				for (auto& baseGASHandle : instanceSet->baseGASHandles) {
					for (auto& mesh : baseGASHandle->GetMeshes()) {
						for (size_t i = 0; i < mesh->GetUniqueResource()->materials.size(); ++i) {
							auto materialId = mesh->GetUniqueResource()->materials[i];
							auto& material = materials[materialId];
							HitgroupData radianceHgData = {};
							{
								radianceHgData.vertices = mesh->GetSharedResource()->vertexBuffer.gpuHandle.getDevicePtr();
								radianceHgData.normals = mesh->GetSharedResource()->normalBuffer.gpuHandle.getDevicePtr();
								radianceHgData.texCoords = mesh->GetSharedResource()->texCrdBuffer.gpuHandle.getDevicePtr();
								radianceHgData.indices = mesh->GetUniqueResource()->triIndBuffer.gpuHandle.getDevicePtr();
								radianceHgData.diffuseTex = m_ParentApp->GetTexture(material.GetString("diffTex")).getHandle();
								radianceHgData.specularTex = m_ParentApp->GetTexture(material.GetString("specTex")).getHandle();
								radianceHgData.emissionTex = m_ParentApp->GetTexture(material.GetString("emitTex")).getHandle();
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
							std::string typeString = SpecifyMaterialType(material);
							if (typeString == "Phong" || typeString == "Diffuse") {
								typeString += ".Default";
							}
							m_HGRecordBuffers.cpuHandle[RAY_TYPE_COUNT * sbtOffset + RAY_TYPE_COUNT * i + RAY_TYPE_RADIANCE] = m_ParentApp->GetHGProgramGroup(std::string("Trace.Radiance.") + typeString).getSBTRecord<HitgroupData>(radianceHgData);
							m_HGRecordBuffers.cpuHandle[RAY_TYPE_COUNT * sbtOffset + RAY_TYPE_COUNT * i + RAY_TYPE_OCCLUSION] = m_ParentApp->GetHGProgramGroup("Trace.Occluded").getSBTRecord<HitgroupData>({});

						}
						sbtOffset += mesh->GetUniqueResource()->materials.size();
					}
				}
			}
		}
		m_HGRecordBuffers.Upload();
		m_ShaderBindingTable.raygenRecord = reinterpret_cast<CUdeviceptr>(m_RGRecordBuffer.gpuHandle.getDevicePtr());
		m_ShaderBindingTable.missRecordBase = reinterpret_cast<CUdeviceptr>(m_MSRecordBuffers.gpuHandle.getDevicePtr());
		m_ShaderBindingTable.missRecordCount = m_MSRecordBuffers.cpuHandle.size();
		m_ShaderBindingTable.missRecordStrideInBytes = sizeof(MissRecord);
		m_ShaderBindingTable.hitgroupRecordBase = reinterpret_cast<CUdeviceptr>(m_HGRecordBuffers.gpuHandle.getDevicePtr());
		m_ShaderBindingTable.hitgroupRecordCount = m_HGRecordBuffers.cpuHandle.size();
		m_ShaderBindingTable.hitgroupRecordStrideInBytes = sizeof(HitGRecord);
	}
	void InitLaunchParams() {
		auto  tlas = m_ParentApp->GetTLAS();
		m_Params.cpuHandle.resize(1);
		m_Params.cpuHandle[0].gasHandle = tlas->GetHandle();
		m_Params.cpuHandle[0].sdTree = m_ParentApp->GetSTree()->GetGpuHandle();
		m_Params.cpuHandle[0].width = m_ParentApp->GetFbWidth();
		m_Params.cpuHandle[0].height = m_ParentApp->GetFbHeight();
		m_Params.cpuHandle[0].light = m_ParentApp->GetLight();
		m_Params.cpuHandle[0].accumBuffer = m_AccumBuffer.getDevicePtr();
		m_Params.cpuHandle[0].frameBuffer = nullptr;
		m_Params.cpuHandle[0].seedBuffer = m_SeedBuffer.getDevicePtr();
		m_Params.cpuHandle[0].isBuilt = false;
		m_Params.cpuHandle[0].maxTraceDepth = m_ParentApp->GetMaxTraceDepth();
		m_Params.cpuHandle[0].samplePerLaunch = 1;

		m_Params.Upload();
	}
private:
	TestPG3Application* m_ParentApp = nullptr;
	OptixShaderBindingTable                  m_ShaderBindingTable = {};
	rtlib::CUDAUploadBuffer<RayGRecord>      m_RGRecordBuffer = {};
	rtlib::CUDAUploadBuffer<MissRecord>      m_MSRecordBuffers = {};
	rtlib::CUDAUploadBuffer<HitGRecord>      m_HGRecordBuffers = {};
	rtlib::CUDAUploadBuffer<RayTraceParams>  m_Params = {};
	rtlib::CUDABuffer<float3>                m_AccumBuffer = {};
	rtlib::CUDABuffer<unsigned int>          m_SeedBuffer = {};
	unsigned int                             m_LightHgRecIndex = 0;
	unsigned int                             m_SamplePerAll = 0;
};
class TestPG3DebugTracer :public test::RTTracer
{
public:
	struct UserData {
		STree   sTree;
		uchar4* diffuseBuffer; //8
		uchar4* specularBuffer;
		uchar4* transmitBuffer;
		uchar4* emissionBuffer;
		uchar4* texCoordBuffer;
		uchar4* normalBuffer;
		uchar4* depthBuffer;
		uchar4* sTreeColBuffer;
	};
private:
	using RayGRecord = rtlib::SBTRecord<RayGenData>;
	using MissRecord = rtlib::SBTRecord<MissData>;
	using HitGRecord = rtlib::SBTRecord<HitgroupData>;
public:
	TestPG3DebugTracer(TestPG3Application* app) {
		m_ParentApp = app;
	}
	// RTTracer を介して継承されました
	virtual void Initialize() override
	{
		this->InitShaderBindingTable();
		this->InitLaunchParams();
	}
	virtual void Launch(const test::RTTraceConfig& config) override {
		UserData* pUserData             = (UserData*)config.pUserData;
		if (!pUserData) {
			return;
		}

		m_Params.cpuHandle[0].width          = config.width;
		m_Params.cpuHandle[0].height         = config.height;
		m_Params.cpuHandle[0].sdTree         = pUserData->sTree;
		m_Params.cpuHandle[0].diffuseBuffer  = pUserData->diffuseBuffer;
		m_Params.cpuHandle[0].specularBuffer = pUserData->specularBuffer;
		m_Params.cpuHandle[0].emissionBuffer = pUserData->emissionBuffer;
		m_Params.cpuHandle[0].transmitBuffer = pUserData->transmitBuffer;
		m_Params.cpuHandle[0].normalBuffer   = pUserData->normalBuffer;
		m_Params.cpuHandle[0].depthBuffer    = pUserData->depthBuffer;
		m_Params.cpuHandle[0].texCoordBuffer = pUserData->texCoordBuffer;
		m_Params.cpuHandle[0].sTreeColBuffer = pUserData->sTreeColBuffer;
		cudaMemcpyAsync(m_Params.gpuHandle.getDevicePtr(), &m_Params.cpuHandle[0], sizeof(RayDebugParams), cudaMemcpyHostToDevice, config.stream);

		auto& pipeline = m_ParentApp->GetOPXPipeline("Debug");
		pipeline.launch(config.stream, m_Params.gpuHandle.getDevicePtr(), m_ShaderBindingTable, config.width, config.height, config.depth);
		if (config.isSync) {
			RTLIB_CU_CHECK(cuStreamSynchronize(config.stream));
		}
	}
	virtual void CleanUp() override {
		m_ParentApp = nullptr;
		m_ShaderBindingTable = {};
		m_RGRecordBuffer.Reset();
		m_MSRecordBuffers.Reset();
		m_HGRecordBuffers.Reset();
		m_Params.Reset();
		m_LightHgRecIndex = 0;
	}
	virtual void Update()  override {
		if (m_ParentApp->IsCameraUpdated()||m_ParentApp->IsFrameResized()) {
			auto camera = m_ParentApp->GetCamera();
			m_RGRecordBuffer.cpuHandle[0].data.eye = camera.getEye();
			auto [u, v, w] = camera.getUVW();
			CUdeviceptr prvRayGenPtr = (CUdeviceptr)m_RGRecordBuffer.gpuHandle.getDevicePtr();
			m_RGRecordBuffer.cpuHandle[0].data.u = u;
			m_RGRecordBuffer.cpuHandle[0].data.v = v;
			m_RGRecordBuffer.cpuHandle[0].data.w = w;
			m_RGRecordBuffer.Upload();
			CUdeviceptr curRayGenPtr = (CUdeviceptr)m_RGRecordBuffer.gpuHandle.getDevicePtr();
		}
		if (m_ParentApp->IsLightUpdated()) {
			auto light = m_ParentApp->GetLight();
			m_Params.cpuHandle[0].light = light;
			m_HGRecordBuffers.cpuHandle[m_LightHgRecIndex].data.emission = light.emission;
			m_HGRecordBuffers.cpuHandle[m_LightHgRecIndex].data.diffuse  = light.emission;
			RTLIB_CUDA_CHECK(cudaMemcpy(m_HGRecordBuffers.gpuHandle.getDevicePtr() + m_LightHgRecIndex, &m_HGRecordBuffers.cpuHandle[m_LightHgRecIndex], sizeof(HitGRecord), cudaMemcpyHostToDevice));
		}
	}
	virtual ~TestPG3DebugTracer() {}
private:
	void InitShaderBindingTable() {
		auto  tlas = m_ParentApp->GetTLAS();
		auto  camera = m_ParentApp->GetCamera();
		auto& materials = m_ParentApp->GetMaterials();
		m_RGRecordBuffer.cpuHandle.resize(1);
		m_RGRecordBuffer.cpuHandle[0] = m_ParentApp->GetRGProgramGroup("Debug.Default").getSBTRecord<RayGenData>();
		m_RGRecordBuffer.cpuHandle[0].data.eye = camera.getEye();
		auto [u, v, w] = camera.getUVW();
		m_RGRecordBuffer.cpuHandle[0].data.u = u;
		m_RGRecordBuffer.cpuHandle[0].data.v = v;
		m_RGRecordBuffer.cpuHandle[0].data.w = w;
		m_RGRecordBuffer.Upload();
		m_MSRecordBuffers.cpuHandle.resize(RAY_TYPE_COUNT);
		m_MSRecordBuffers.cpuHandle[RAY_TYPE_RADIANCE]  = m_ParentApp->GetMSProgramGroup("Debug.Default").getSBTRecord<MissData>();
		m_MSRecordBuffers.cpuHandle[RAY_TYPE_RADIANCE].data.bgColor = { 0,0,0,0 };
		m_MSRecordBuffers.cpuHandle[RAY_TYPE_OCCLUSION] = m_ParentApp->GetMSProgramGroup("Debug.Default").getSBTRecord<MissData>();
		m_MSRecordBuffers.Upload();
		m_HGRecordBuffers.cpuHandle.resize(tlas->GetSbtCount() * RAY_TYPE_COUNT);
		{
			size_t sbtOffset = 0;
			for (auto& instanceSet : tlas->GetInstanceSets()) {
				for (auto& baseGASHandle : instanceSet->baseGASHandles) {
					for (auto& mesh : baseGASHandle->GetMeshes()) {
						for (size_t i = 0; i < mesh->GetUniqueResource()->materials.size(); ++i) {
							auto materialId = mesh->GetUniqueResource()->materials[i];
							auto& material = materials[materialId];
							HitgroupData radianceHgData = {};
							{
								radianceHgData.vertices    = mesh->GetSharedResource()->vertexBuffer.gpuHandle.getDevicePtr();
								radianceHgData.normals     = mesh->GetSharedResource()->normalBuffer.gpuHandle.getDevicePtr();
								radianceHgData.texCoords   = mesh->GetSharedResource()->texCrdBuffer.gpuHandle.getDevicePtr();
								radianceHgData.indices     = mesh->GetUniqueResource()->triIndBuffer.gpuHandle.getDevicePtr();
								radianceHgData.diffuseTex  = m_ParentApp->GetTexture(material.GetString("diffTex")).getHandle();
								radianceHgData.specularTex = m_ParentApp->GetTexture(material.GetString("specTex")).getHandle();
								radianceHgData.emissionTex = m_ParentApp->GetTexture(material.GetString("emitTex")).getHandle();
								radianceHgData.diffuse  = material.GetFloat3As<float3>("diffCol");
								radianceHgData.specular = material.GetFloat3As<float3>("specCol");
								radianceHgData.emission = material.GetFloat3As<float3>("emitCol");
								radianceHgData.shinness = material.GetFloat1("shinness");
								radianceHgData.transmit = material.GetFloat3As<float3>("tranCol");
								radianceHgData.refrInd  = material.GetFloat1("refrIndx");
							}
							if (material.GetString("name") == "light") {
								m_LightHgRecIndex = RAY_TYPE_COUNT * sbtOffset + RAY_TYPE_COUNT * i + RAY_TYPE_RADIANCE;
							}
							m_HGRecordBuffers.cpuHandle[RAY_TYPE_COUNT * sbtOffset + RAY_TYPE_COUNT * i + RAY_TYPE_RADIANCE]  = m_ParentApp->GetHGProgramGroup("Debug.Default").getSBTRecord<HitgroupData>(radianceHgData);
							m_HGRecordBuffers.cpuHandle[RAY_TYPE_COUNT * sbtOffset + RAY_TYPE_COUNT * i + RAY_TYPE_OCCLUSION] = m_ParentApp->GetHGProgramGroup("Debug.Default").getSBTRecord<HitgroupData>({});;

						}
						sbtOffset += mesh->GetUniqueResource()->materials.size();
					}
				}
			}
		}
		m_HGRecordBuffers.Upload();
		m_ShaderBindingTable.raygenRecord            = reinterpret_cast<CUdeviceptr>(m_RGRecordBuffer.gpuHandle.getDevicePtr());
		m_ShaderBindingTable.missRecordBase          = reinterpret_cast<CUdeviceptr>(m_MSRecordBuffers.gpuHandle.getDevicePtr());
		m_ShaderBindingTable.missRecordCount         = m_MSRecordBuffers.cpuHandle.size();
		m_ShaderBindingTable.missRecordStrideInBytes = sizeof(MissRecord);
		m_ShaderBindingTable.hitgroupRecordBase          = reinterpret_cast<CUdeviceptr>(m_HGRecordBuffers.gpuHandle.getDevicePtr());
		m_ShaderBindingTable.hitgroupRecordCount         = m_HGRecordBuffers.cpuHandle.size();
		m_ShaderBindingTable.hitgroupRecordStrideInBytes = sizeof(HitGRecord);
	}
	void InitLaunchParams() {
		auto  tlas = m_ParentApp->GetTLAS();
		m_Params.cpuHandle.resize(1);
		m_Params.cpuHandle[0].gasHandle      = tlas->GetHandle();
		m_Params.cpuHandle[0].sdTree         = m_ParentApp->GetSTree()->GetGpuHandle();
		m_Params.cpuHandle[0].width          = m_ParentApp->GetFbWidth();
		m_Params.cpuHandle[0].height         = m_ParentApp->GetFbHeight();
		m_Params.cpuHandle[0].light          = m_ParentApp->GetLight();
		m_Params.cpuHandle[0].diffuseBuffer  = nullptr;
		m_Params.cpuHandle[0].specularBuffer = nullptr;
		m_Params.cpuHandle[0].emissionBuffer = nullptr;
		m_Params.cpuHandle[0].transmitBuffer = nullptr;
		m_Params.cpuHandle[0].normalBuffer   = nullptr;
		m_Params.cpuHandle[0].depthBuffer    = nullptr;
		m_Params.cpuHandle[0].texCoordBuffer = nullptr;
		m_Params.cpuHandle[0].sTreeColBuffer = nullptr;
		m_Params.Upload();
	}
private:
	TestPG3Application*                      m_ParentApp          = nullptr;
	OptixShaderBindingTable                  m_ShaderBindingTable = {};
	rtlib::CUDAUploadBuffer<RayGRecord>      m_RGRecordBuffer     = {};
	rtlib::CUDAUploadBuffer<MissRecord>      m_MSRecordBuffers    = {};
	rtlib::CUDAUploadBuffer<HitGRecord>      m_HGRecordBuffers    = {};
	rtlib::CUDAUploadBuffer<RayDebugParams>  m_Params             = {};
	unsigned int                             m_LightHgRecIndex    = 0;
};
class TestPG3GuideTracer :public test::RTTracer {
public:
	struct UserData {
		STree        sTree;
		uchar4*      frameBuffer;
		unsigned int samplePerAll;
		unsigned int samplePerLaunch;
	};
private:
	using RayGRecord = rtlib::SBTRecord<RayGenData>;
	using MissRecord = rtlib::SBTRecord<MissData>;
	using HitGRecord = rtlib::SBTRecord<HitgroupData>;
public:
	TestPG3GuideTracer(TestPG3Application* app) {
		m_ParentApp = app;
	}
	// RTTracer を介して継承されました
	virtual void Initialize() override
	{
		this->InitFrameResources();
		this->InitShaderBindingTable();
		this->InitLaunchParams();
	}
	virtual void Launch(const test::RTTraceConfig& config) override {
		UserData* pUserData = (UserData*)config.pUserData;
		if (!pUserData) {
			return;
		}
		m_Params.cpuHandle[0].width       = config.width;
		m_Params.cpuHandle[0].height      = config.height;
		m_Params.cpuHandle[0].sdTree      = pUserData->sTree;
		m_Params.cpuHandle[0].accumBuffer = m_AccumBuffer.getDevicePtr();
		m_Params.cpuHandle[0].frameBuffer = pUserData->frameBuffer;
		m_Params.cpuHandle[0].seedBuffer  = m_SeedBuffer.getDevicePtr();
		m_Params.cpuHandle[0].isBuilt     = false;
		m_Params.cpuHandle[0].samplePerLaunch = pUserData->samplePerLaunch;
		cudaMemcpyAsync(m_Params.gpuHandle.getDevicePtr(), &m_Params.cpuHandle[0], sizeof(RayTraceParams), cudaMemcpyHostToDevice, config.stream);
		auto& pipeline = m_ParentApp->GetOPXPipeline("Trace");
		pipeline.launch(config.stream, m_Params.gpuHandle.getDevicePtr(), m_ShaderBindingTable, config.width, config.height, config.depth);
		if (config.isSync) {
			RTLIB_CU_CHECK(cuStreamSynchronize(config.stream));
		}
		m_Params.cpuHandle[0].samplePerALL += m_Params.cpuHandle[0].samplePerLaunch;
		pUserData->samplePerAll = m_Params.cpuHandle[0].samplePerALL;
	}
	virtual void CleanUp() override {
		m_ParentApp = nullptr;
		m_ShaderBindingTable = {};
		m_RGRecordBuffer.Reset();
		m_MSRecordBuffers.Reset();
		m_HGRecordBuffers.Reset();
		m_Params.Reset();
		m_LightHgRecIndex = 0;
	}
	virtual void Update()  override {
		if (m_ParentApp->IsCameraUpdated() || m_ParentApp->IsFrameResized()) {
			auto camera = m_ParentApp->GetCamera();
			m_RGRecordBuffer.cpuHandle[0].data.eye = camera.getEye();
			auto [u, v, w] = camera.getUVW();
			m_RGRecordBuffer.cpuHandle[0].data.u = u;
			m_RGRecordBuffer.cpuHandle[0].data.v = v;
			m_RGRecordBuffer.cpuHandle[0].data.w = w;
			m_RGRecordBuffer.Upload();
		}
		if (m_ParentApp->IsCameraUpdated() || m_ParentApp->IsFrameResized() || m_Params.cpuHandle[0].maxTraceDepth != m_ParentApp->GetMaxTraceDepth() || m_ParentApp->IsLightUpdated()) {
			m_Params.cpuHandle[0].maxTraceDepth = m_ParentApp->GetMaxTraceDepth();
			m_AccumBuffer.resize(m_ParentApp->GetFbWidth() * m_ParentApp->GetFbHeight());
			m_AccumBuffer.upload(std::vector<float3>(m_ParentApp->GetFbWidth() * m_ParentApp->GetFbHeight()));
			m_Params.cpuHandle[0].samplePerALL = 0;
		}
		if (m_ParentApp->IsFrameResized()) {
			std::vector<unsigned int> seedData(m_ParentApp->GetFbWidth() * m_ParentApp->GetFbHeight());
			std::random_device rd;
			std::mt19937 mt(rd());
			std::generate(seedData.begin(), seedData.end(), mt);
			m_SeedBuffer.resize(m_ParentApp->GetFbWidth() * m_ParentApp->GetFbHeight());
			m_SeedBuffer.upload(seedData);
		}
		if (m_ParentApp->IsLightUpdated()) {
			auto light = m_ParentApp->GetLight();
			m_Params.cpuHandle[0].light = light;
			m_HGRecordBuffers.cpuHandle[m_LightHgRecIndex].data.emission = light.emission;
			m_HGRecordBuffers.cpuHandle[m_LightHgRecIndex].data.diffuse = light.emission;
			RTLIB_CUDA_CHECK(cudaMemcpy(m_HGRecordBuffers.gpuHandle.getDevicePtr() + m_LightHgRecIndex, &m_HGRecordBuffers.cpuHandle[m_LightHgRecIndex], sizeof(HitGRecord), cudaMemcpyHostToDevice));
		}
	}
	virtual ~TestPG3GuideTracer() {}
private:
	void InitFrameResources() {
		std::vector<unsigned int> seedData(m_ParentApp->GetFbWidth() * m_ParentApp->GetFbHeight());
		std::random_device rd;
		std::mt19937 mt(rd());
		std::generate(seedData.begin(), seedData.end(), mt);
		m_AccumBuffer = rtlib::CUDABuffer<float3>(std::vector<float3>(m_ParentApp->GetFbWidth() * m_ParentApp->GetFbHeight()));
		m_SeedBuffer = rtlib::CUDABuffer<uint32_t>(seedData);
	}
	void InitShaderBindingTable() {
		auto SpecifyMaterialType = [](const rtlib::ext::Material& material) {
			auto emitCol = material.GetFloat3As<float3>("emitCol");
			auto tranCol = material.GetFloat3As<float3>("tranCol");
			auto refrIndx = material.GetFloat1("refrIndx");
			auto shinness = material.GetFloat1("shinness");
			auto illum = material.GetUInt32("illum");
			if (illum == 7) {
				return "Refraction";
			}
			else {
				return "Phong";
			}
		};
		auto  tlas = m_ParentApp->GetTLAS();
		auto  camera = m_ParentApp->GetCamera();
		auto& materials = m_ParentApp->GetMaterials();
		m_RGRecordBuffer.cpuHandle.resize(1);
		m_RGRecordBuffer.cpuHandle[0] = m_ParentApp->GetRGProgramGroup("Trace.Default").getSBTRecord<RayGenData>();
		m_RGRecordBuffer.cpuHandle[0].data.eye = camera.getEye();
		auto [u, v, w] = camera.getUVW();
		m_RGRecordBuffer.cpuHandle[0].data.u = u;
		m_RGRecordBuffer.cpuHandle[0].data.v = v;
		m_RGRecordBuffer.cpuHandle[0].data.w = w;
		m_RGRecordBuffer.Upload();
		m_MSRecordBuffers.cpuHandle.resize(RAY_TYPE_COUNT);
		m_MSRecordBuffers.cpuHandle[RAY_TYPE_RADIANCE] = m_ParentApp->GetMSProgramGroup("Trace.Radiance").getSBTRecord<MissData>();
		m_MSRecordBuffers.cpuHandle[RAY_TYPE_RADIANCE].data.bgColor = { 0,0,0,0 };
		m_MSRecordBuffers.cpuHandle[RAY_TYPE_OCCLUSION] = m_ParentApp->GetMSProgramGroup("Trace.Occluded").getSBTRecord<MissData>();
		m_MSRecordBuffers.Upload();
		m_HGRecordBuffers.cpuHandle.resize(tlas->GetSbtCount() * RAY_TYPE_COUNT);
		{
			size_t sbtOffset = 0;
			for (auto& instanceSet : tlas->GetInstanceSets()) {
				for (auto& baseGASHandle : instanceSet->baseGASHandles) {
					for (auto& mesh : baseGASHandle->GetMeshes()) {
						for (size_t i = 0; i < mesh->GetUniqueResource()->materials.size(); ++i) {
							auto materialId = mesh->GetUniqueResource()->materials[i];
							auto& material = materials[materialId];
							HitgroupData radianceHgData = {};
							{
								radianceHgData.vertices = mesh->GetSharedResource()->vertexBuffer.gpuHandle.getDevicePtr();
								radianceHgData.normals = mesh->GetSharedResource()->normalBuffer.gpuHandle.getDevicePtr();
								radianceHgData.texCoords = mesh->GetSharedResource()->texCrdBuffer.gpuHandle.getDevicePtr();
								radianceHgData.indices = mesh->GetUniqueResource()->triIndBuffer.gpuHandle.getDevicePtr();
								radianceHgData.diffuseTex = m_ParentApp->GetTexture(material.GetString("diffTex")).getHandle();
								radianceHgData.specularTex = m_ParentApp->GetTexture(material.GetString("specTex")).getHandle();
								radianceHgData.emissionTex = m_ParentApp->GetTexture(material.GetString("emitTex")).getHandle();
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
							std::string typeString = SpecifyMaterialType(material);
							if (typeString == "Phong" || typeString == "Diffuse") {
								typeString += ".Default";
							}
							m_HGRecordBuffers.cpuHandle[RAY_TYPE_COUNT * sbtOffset + RAY_TYPE_COUNT * i + RAY_TYPE_RADIANCE] = m_ParentApp->GetHGProgramGroup(std::string("Trace.Radiance.") + typeString).getSBTRecord<HitgroupData>(radianceHgData);
							m_HGRecordBuffers.cpuHandle[RAY_TYPE_COUNT * sbtOffset + RAY_TYPE_COUNT * i + RAY_TYPE_OCCLUSION] = m_ParentApp->GetHGProgramGroup("Trace.Occluded").getSBTRecord<HitgroupData>({});

						}
						sbtOffset += mesh->GetUniqueResource()->materials.size();
					}
				}
			}
		}
		m_HGRecordBuffers.Upload();
		m_ShaderBindingTable.raygenRecord = reinterpret_cast<CUdeviceptr>(m_RGRecordBuffer.gpuHandle.getDevicePtr());
		m_ShaderBindingTable.missRecordBase = reinterpret_cast<CUdeviceptr>(m_MSRecordBuffers.gpuHandle.getDevicePtr());
		m_ShaderBindingTable.missRecordCount = m_MSRecordBuffers.cpuHandle.size();
		m_ShaderBindingTable.missRecordStrideInBytes = sizeof(MissRecord);
		m_ShaderBindingTable.hitgroupRecordBase = reinterpret_cast<CUdeviceptr>(m_HGRecordBuffers.gpuHandle.getDevicePtr());
		m_ShaderBindingTable.hitgroupRecordCount = m_HGRecordBuffers.cpuHandle.size();
		m_ShaderBindingTable.hitgroupRecordStrideInBytes = sizeof(HitGRecord);
	}
	void InitLaunchParams() {
		auto  tlas = m_ParentApp->GetTLAS();
		m_Params.cpuHandle.resize(1);
		m_Params.cpuHandle[0].gasHandle = tlas->GetHandle();
		m_Params.cpuHandle[0].sdTree = m_ParentApp->GetSTree()->GetGpuHandle();
		m_Params.cpuHandle[0].width = m_ParentApp->GetFbWidth();
		m_Params.cpuHandle[0].height = m_ParentApp->GetFbHeight();
		m_Params.cpuHandle[0].light = m_ParentApp->GetLight();
		m_Params.cpuHandle[0].accumBuffer = m_AccumBuffer.getDevicePtr();
		m_Params.cpuHandle[0].frameBuffer = nullptr;
		m_Params.cpuHandle[0].seedBuffer = m_SeedBuffer.getDevicePtr();
		m_Params.cpuHandle[0].isBuilt = false;
		m_Params.cpuHandle[0].maxTraceDepth = m_ParentApp->GetMaxTraceDepth();
		m_Params.cpuHandle[0].samplePerLaunch = 1;

		m_Params.Upload();
	}
private:
	TestPG3Application* m_ParentApp = nullptr;
	OptixShaderBindingTable                  m_ShaderBindingTable = {};
	rtlib::CUDAUploadBuffer<RayGRecord>      m_RGRecordBuffer = {};
	rtlib::CUDAUploadBuffer<MissRecord>      m_MSRecordBuffers = {};
	rtlib::CUDAUploadBuffer<HitGRecord>      m_HGRecordBuffers = {};
	rtlib::CUDAUploadBuffer<RayTraceParams>  m_Params = {};
	rtlib::CUDABuffer<float3>                m_AccumBuffer = {};
	rtlib::CUDABuffer<unsigned int>          m_SeedBuffer = {};
	unsigned int                             m_LightHgRecIndex = 0;
	unsigned int                             m_SamplePerAll = 0;
};

void  TestPG3Application::InitGLFW() {
	if (glfwInit() == GLFW_FALSE) {
		throw std::runtime_error("Failed To Init GLFW!");
	}
}

void  TestPG3Application::InitWindow() {
	glfwWindowHint(GLFW_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 4);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
	m_Window = glfwCreateWindow(m_FbWidth, m_FbHeight, m_Title, nullptr, nullptr);
	if (!m_Window) {
		throw std::runtime_error("Failed To Create Window!");
	}
	glfwMakeContextCurrent(m_Window);
	glfwSetWindowUserPointer(m_Window, this);
	glfwSetMouseButtonCallback(m_Window, ImGui_ImplGlfw_MouseButtonCallback);
	glfwSetKeyCallback(m_Window, ImGui_ImplGlfw_KeyCallback);
	glfwSetCharCallback(m_Window, ImGui_ImplGlfw_CharCallback);
	glfwSetScrollCallback(m_Window, ImGui_ImplGlfw_ScrollCallback);
	glfwSetCursorPosCallback(m_Window, CursorPositionCallback);
	glfwSetFramebufferSizeCallback(m_Window, FrameBufferSizeCallback);
	glfwGetFramebufferSize(m_Window, &m_FbWidth, &m_FbHeight);
	m_FbAspect = static_cast<float>(m_FbWidth) / static_cast<float>(m_FbHeight);
}

void  TestPG3Application::InitGLAD() {
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
		throw std::runtime_error("Failed To Init GLAD!");
	}
}

void TestPG3Application::InitCUDA() {
	RTLIB_CUDA_CHECK(cudaFree(0));
	RTLIB_OPTIX_CHECK(optixInit());
	m_Context = std::make_shared<rtlib::OPXContext>(
		rtlib::OPXContext::Desc{ 0,0,OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL,4 }
	);
}

void TestPG3Application::InitGui() {
	//Renderer
	m_Renderer = std::make_unique<rtlib::ext::RectRenderer>();
	m_Renderer->init();
	//ImGui
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO(); (void)io;
	ImGui::StyleColorsDark();
	if (!ImGui_ImplGlfw_InitForOpenGL(m_Window, false)) {
		throw std::runtime_error("Failed To Init ImGui For GLFW + OpenGL!");
	}
	int major = glfwGetWindowAttrib(m_Window, GLFW_CONTEXT_VERSION_MAJOR);
	int minor = glfwGetWindowAttrib(m_Window, GLFW_CONTEXT_VERSION_MINOR);
	std::string glslVersion = std::string("#version ") + std::to_string(major) + std::to_string(minor) + "0 core";
	if (!ImGui_ImplOpenGL3_Init(glslVersion.c_str())) {
		throw std::runtime_error("Failed To Init ImGui For GLFW3");
	}
}

void TestPG3Application::InitPipelines() {
	{
		auto rayTracePtxFile = std::ifstream(TEST_TEST_PG_CUDA_PATH"/RayTrace.ptx", std::ios::binary);
		if (!rayTracePtxFile.is_open()) throw std::runtime_error("Failed To Load RayTrace.ptx!");
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
			traceLinkOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;
#else
			traceLinkOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
#endif
			traceLinkOptions.maxTraceDepth = 2;
		}
		auto traceModuleOptions = OptixModuleCompileOptions{};
		{
#ifndef NDEBUG
			traceModuleOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;
			traceModuleOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
#else
			traceModuleOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
			traceModuleOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
#endif
		}
		traceModuleOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
		m_Pipelines["Trace"] = m_Context->createPipeline(traceCompileOptions);
		m_Modules["RayTrace"] = m_Pipelines["Trace"].createModule(rayTracePtxData, traceModuleOptions);
		m_RGProgramGroups["Trace.Default"] = m_Pipelines["Trace"].createRaygenPG({ m_Modules["RayTrace"]  , "__raygen__def" });
		m_RGProgramGroups["Trace.Guiding"] = m_Pipelines["Trace"].createRaygenPG({ m_Modules["RayTrace"]  , "__raygen__pg" });
		m_MSProgramGroups["Trace.Radiance"] = m_Pipelines["Trace"].createMissPG({ m_Modules["RayTrace"]  , "__miss__radiance" });
		m_MSProgramGroups["Trace.Occluded"] = m_Pipelines["Trace"].createMissPG({ m_Modules["RayTrace"]  , "__miss__occluded" });
		m_HGProgramGroups["Trace.Radiance.Diffuse.Default"] = m_Pipelines["Trace"].createHitgroupPG({ m_Modules["RayTrace"], "__closesthit__radiance_for_diffuse_def" }, {}, {});
		m_HGProgramGroups["Trace.Radiance.Diffuse.Guiding"] = m_Pipelines["Trace"].createHitgroupPG({ m_Modules["RayTrace"], "__closesthit__radiance_for_diffuse_pg" }, {}, {});
		m_HGProgramGroups["Trace.Radiance.Phong.Default"] = m_Pipelines["Trace"].createHitgroupPG({ m_Modules["RayTrace"], "__closesthit__radiance_for_phong_def" }, {}, {});
		m_HGProgramGroups["Trace.Radiance.Phong.Guiding"] = m_Pipelines["Trace"].createHitgroupPG({ m_Modules["RayTrace"], "__closesthit__radiance_for_phong_pg" }, {}, {});
		m_HGProgramGroups["Trace.Radiance.Emission"] = m_Pipelines["Trace"].createHitgroupPG({ m_Modules["RayTrace"] , "__closesthit__radiance_for_emission" }, {}, {});
		m_HGProgramGroups["Trace.Radiance.Specular"] = m_Pipelines["Trace"].createHitgroupPG({ m_Modules["RayTrace"] , "__closesthit__radiance_for_specular" }, {}, {});
		m_HGProgramGroups["Trace.Radiance.Refraction"] = m_Pipelines["Trace"].createHitgroupPG({ m_Modules["RayTrace"] , "__closesthit__radiance_for_refraction" }, {}, {});
		m_HGProgramGroups["Trace.Occluded"] = m_Pipelines["Trace"].createHitgroupPG({ m_Modules["RayTrace"], "__closesthit__occluded" }, {}, {});
		m_Pipelines["Trace"].link(traceLinkOptions);
	}

	{
		auto rayDebugPtxFile = std::ifstream(TEST_TEST_PG_CUDA_PATH"/RayDebug.ptx", std::ios::binary);
		if (!rayDebugPtxFile.is_open()) throw std::runtime_error("Failed To Load RayDebug.ptx!");
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
			debugLinkOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;
#else
			debugLinkOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
#endif
			debugLinkOptions.maxTraceDepth = 1;
		}
		auto debugModuleOptions = OptixModuleCompileOptions{};
		{
#ifndef NDEBUG
			debugModuleOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;
			debugModuleOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
#else
			debugModuleOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
			debugModuleOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
#endif
			debugModuleOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
		}
		m_Pipelines["Debug"]               = m_Context->createPipeline(debugCompileOptions);
		m_Modules["RayDebug"]              = m_Pipelines["Debug"].createModule(rayDebugPtxData, debugModuleOptions);
		m_RGProgramGroups["Debug.Default"] = m_Pipelines["Debug"].createRaygenPG({	 m_Modules["RayDebug"] , "__raygen__debug" });
		m_MSProgramGroups["Debug.Default"] = m_Pipelines["Debug"].createMissPG({	 m_Modules["RayDebug"] , "__miss__debug" });
		m_HGProgramGroups["Debug.Default"] = m_Pipelines["Debug"].createHitgroupPG({ m_Modules["RayDebug"] , "__closesthit__debug" }, {}, {});
		m_Pipelines["Debug"].link(debugLinkOptions);
	}

}

void TestPG3Application::InitAssets() {
	auto objModelPath = std::filesystem::canonical(std::filesystem::path(TEST_TEST_PG_DATA_PATH"/Models/CornellBox/CornellBox-Water.obj"));
	if (!m_ObjModelAssets.LoadAsset("CornellBox-Water", objModelPath.string())) {
		throw std::runtime_error("Failed To Load Obj Model!");
	}
	auto smpTexPath = std::filesystem::canonical(std::filesystem::path(TEST_TEST_PG_DATA_PATH"/Textures/white.png"));
	if (!m_TextureAssets.LoadAsset("", smpTexPath.string())) {
		throw std::runtime_error("Failed To Load White Texture!");
	}
	for (auto& [name,objModel] : m_ObjModelAssets.GetAssets()) {
		for (auto& material : objModel.materials) {
			auto diffTexPath = material.GetString("diffTex");
			auto specTexPath = material.GetString("specTex");
			auto emitTexPath = material.GetString("emitTex");
			auto shinTexPath = material.GetString("shinTex");
			if (diffTexPath  != "") {
				if (!m_TextureAssets.LoadAsset(diffTexPath, diffTexPath)) {
					material.SetString("diffTex", "");
				}
			}
			if (specTexPath != "") {
				if (!m_TextureAssets.LoadAsset(specTexPath, specTexPath)) {
					material.SetString("specTex", "");
				}
			}
			if (emitTexPath != "") {
				if (!m_TextureAssets.LoadAsset(emitTexPath, emitTexPath)) {
					material.SetString("emitTex", "");
				}
			}
			if (shinTexPath != "") {
				if (!m_TextureAssets.LoadAsset(shinTexPath, shinTexPath)) {
					material.SetString("shinTex", "");
				}
			}
		}
	}
}

void TestPG3Application::InitAccelerationStructures()
{
	{
		size_t materialSize = 0;
		for (auto& [name, objModel] : m_ObjModelAssets.GetAssets()) {
			materialSize += objModel.materials.size();
		}
		m_Materials.resize(materialSize + 1);
		size_t materialOffset = 0;
		for (auto& [name, objModel] : m_ObjModelAssets.GetAssets()) {
			auto& materials = objModel.materials;
			std::copy(std::begin(materials), std::end(materials), m_Materials.begin() + materialOffset);
			materialOffset += materials.size();
		}
	}

	m_GASHandles["World"] = std::make_shared<rtlib::ext::GASHandle>();
	m_GASHandles["Light"] = std::make_shared<rtlib::ext::GASHandle>();
	{
		OptixAccelBuildOptions accelBuildOptions = {};
		accelBuildOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
		accelBuildOptions.operation  = OPTIX_BUILD_OPERATION_BUILD;

		bool hasLight = false;
		{
		size_t materialOffset = 0;
		for (auto& [name, objModel] : m_ObjModelAssets.GetAssets()) {
			for (auto& [name, meshUniqueResource] : objModel.meshGroup->GetUniqueResources()) {
				auto mesh = rtlib::ext::Mesh::New();
				mesh->SetUniqueResource(meshUniqueResource);
				mesh->SetSharedResource(objModel.meshGroup->GetSharedResource());
				for (auto& matIdx : mesh->GetUniqueResource()->matIndBuffer.cpuHandle) {
					matIdx += materialOffset;
				}
				if (name == "light") {
					hasLight = true;
					m_GASHandles["Light"]->AddMesh(mesh);
				}
				else {
					m_GASHandles["World"]->AddMesh(mesh);
				}
			}
			materialOffset += objModel.materials.size();
		}
	}
	if (!hasLight) {
		/*Generate Light*/
		rtlib::utils::AABB aabb = {};
		for (auto& mesh : m_GASHandles["World"]->GetMeshes()) {
			for (auto& vertex : mesh->GetSharedResource()->vertexBuffer.cpuHandle) {
				aabb.Update(vertex);
			}
		}
		OptixAccelBuildOptions accelOptions = {};
		accelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
		accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
		auto lightMesh = rtlib::ext::Mesh::New();
		lightMesh->SetSharedResource(rtlib::ext::MeshSharedResource::New());
		lightMesh->GetSharedResource()->name = "light";
		lightMesh->GetSharedResource()->vertexBuffer.cpuHandle = {
			{aabb.min.x,aabb.max.y - 1e-3f,aabb.min.z},
			{aabb.max.x,aabb.max.y - 1e-3f,aabb.min.z},
			{aabb.max.x,aabb.max.y - 1e-3f,aabb.max.z},
			{aabb.min.x,aabb.max.y - 1e-3f,aabb.max.z}
		};
		lightMesh->GetSharedResource()->texCrdBuffer.cpuHandle = { {0.0f,0.0f}      , {1.0f,0.0f}      , {1.0f,1.0f}      , {0.0f,1.0f}, };
		lightMesh->GetSharedResource()->normalBuffer.cpuHandle = { {0.0f,-1.0f,0.0f}, {0.0f,-1.0f,0.0f}, {0.0f,-1.0f,0.0f}, {0.0f,-1.0f,0.0f} };

		auto& lightMaterial = m_Materials.back();
		{
			lightMaterial.SetString("name", "light");
			lightMaterial.SetFloat3("diffCol", kDefaultLightColor);
			lightMaterial.SetString("diffTex", "");
			lightMaterial.SetFloat3("emitCol", kDefaultLightColor);
			lightMaterial.SetString("emitTex", "");
			lightMaterial.SetFloat3("specCol", kDefaultLightColor);
			lightMaterial.SetString("specTex", "");
			lightMaterial.SetFloat1("shinness", 0.0f);
			lightMaterial.SetString("shinTex", "");
			lightMaterial.SetFloat3("tranCol", kDefaultLightColor);
			lightMaterial.SetFloat1("refrIndx", 0.0f);
			lightMaterial.SetUInt32("illum", 2);
		}
		lightMesh->GetSharedResource()->vertexBuffer.Upload();
		lightMesh->GetSharedResource()->texCrdBuffer.Upload();
		lightMesh->GetSharedResource()->normalBuffer.Upload();
		lightMesh->SetUniqueResource(rtlib::ext::MeshUniqueResource::New());
		lightMesh->GetUniqueResource()->name = "light";
		lightMesh->GetUniqueResource()->materials = { (unsigned int)m_Materials.size() - 1 };
		lightMesh->GetUniqueResource()->matIndBuffer.cpuHandle = { 0,0 };
		lightMesh->GetUniqueResource()->triIndBuffer.cpuHandle = { {0,1,2}, {2,3,0} };
		lightMesh->GetUniqueResource()->matIndBuffer.Upload();
		lightMesh->GetUniqueResource()->triIndBuffer.Upload();
		//AddMesh
		m_GASHandles["Light"]->AddMesh(lightMesh);
	}
		m_GASHandles["World"]->Build(m_Context.get(), accelBuildOptions);
		m_GASHandles["Light"]->Build(m_Context.get(), accelBuildOptions);
	}
	m_IASHandles["TopLevel"] = std::make_shared<rtlib::ext::IASHandle>();
	{
		OptixAccelBuildOptions accelOptions = {};
		accelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
		accelOptions.operation  = OPTIX_BUILD_OPERATION_BUILD;
		//World
		auto worldInstance = rtlib::ext::Instance();
		worldInstance.Init(m_GASHandles["World"]);
		worldInstance.SetSbtOffset(0);
		//Light
		auto lightInstance = rtlib::ext::Instance();
		lightInstance.Init(m_GASHandles["Light"]);
		lightInstance.SetSbtOffset(worldInstance.GetSbtCount() * RAY_TYPE_COUNT);
		//InstanceSet
		auto instanceSet = std::make_shared<rtlib::ext::InstanceSet>();
		instanceSet->SetInstance(worldInstance);
		instanceSet->SetInstance(lightInstance);
		instanceSet->Upload();
		//AddInstanceSet
		m_IASHandles["TopLevel"]->AddInstanceSet(instanceSet);
		//Build
		m_IASHandles["TopLevel"]->Build(m_Context.get(), accelOptions);

	}
}

void TestPG3Application::InitLight()
{
	auto  lightGASHandle = m_GASHandles["Light"];
	auto  lightMesh      = lightGASHandle->GetMesh(0);
	auto  lightVertices  = std::vector<float3>();
	for (auto& index : lightMesh->GetUniqueResource()->triIndBuffer.cpuHandle) {
		lightVertices.push_back(lightMesh->GetSharedResource()->vertexBuffer.cpuHandle[index.x]);
		lightVertices.push_back(lightMesh->GetSharedResource()->vertexBuffer.cpuHandle[index.y]);
		lightVertices.push_back(lightMesh->GetSharedResource()->vertexBuffer.cpuHandle[index.z]);
	}
	auto lightAABB           = rtlib::utils::AABB(lightVertices);
	auto lightV3             = lightAABB.max - lightAABB.min;
	m_ParallelLight.corner   = lightAABB.min;
	m_ParallelLight.v1	     = make_float3(0.0f, 0.0f, lightV3.z);
	m_ParallelLight.v2       = make_float3(lightV3.x, 0.0f, 0.0f);
	m_ParallelLight.normal   = make_float3(0.0f, -1.0f, 0.0f);
	auto lightMaterial       = m_Materials[lightMesh->GetUniqueResource()->materials[0]];
	m_ParallelLight.emission = lightMaterial.GetFloat3As<float3>("emitCol");
}

void TestPG3Application::InitCamera()
{
	m_CameraController = rtlib::CameraController({ 0.0f,1.0f, 5.0f });
	m_CameraController.SetMouseSensitivity(0.125f);
	m_CameraController.SetMovementSpeed(10.0f);
}

void TestPG3Application::InitSTree()
{
	auto worldAABB = rtlib::utils::AABB();

	for (auto& mesh : m_GASHandles["World"]->GetMeshes()) {
		for (auto& index : mesh->GetUniqueResource()->triIndBuffer.cpuHandle)
		{
			worldAABB.Update(mesh->GetSharedResource()->vertexBuffer.cpuHandle[index.x]);
			worldAABB.Update(mesh->GetSharedResource()->vertexBuffer.cpuHandle[index.y]);
			worldAABB.Update(mesh->GetSharedResource()->vertexBuffer.cpuHandle[index.z]);
		}
	}

	m_STree = std::make_shared<test::RTSTreeWrapper>(worldAABB.min, worldAABB.max);
	m_STree->Upload();
}

void TestPG3Application::InitFrameResources() {
	m_RenderTexture = std::make_unique<rtlib::GLTexture2D<uchar4>>();
	m_RenderTexture->allocate({ (size_t)m_FbWidth,(size_t)m_FbHeight, nullptr }, GL_TEXTURE_2D);
	m_RenderTexture->setParameteri(GL_TEXTURE_MAG_FILTER, GL_LINEAR, false);
	m_RenderTexture->setParameteri(GL_TEXTURE_MIN_FILTER, GL_LINEAR, false);
	m_RenderTexture->setParameteri(GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE, false);
	m_RenderTexture->setParameteri(GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE, false);

	m_DebugTexture = std::make_unique<rtlib::GLTexture2D<uchar4>>();
	m_DebugTexture->allocate({ (size_t)m_FbWidth,(size_t)m_FbHeight, nullptr }, GL_TEXTURE_2D);
	m_DebugTexture->setParameteri(GL_TEXTURE_MAG_FILTER, GL_LINEAR, false);
	m_DebugTexture->setParameteri(GL_TEXTURE_MIN_FILTER, GL_LINEAR, false);
	m_DebugTexture->setParameteri(GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE, false);
	m_DebugTexture->setParameteri(GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE, false);

	m_FrameBuffer = std::make_unique<test::RTFrameBuffer>(m_FbWidth, m_FbHeight);
	m_FrameBuffer->AddCUGLBuffer("Default");
	m_FrameBuffer->GetCUGLBuffer("Default").upload(
		std::vector<uchar4>(m_FbWidth * m_FbHeight, make_uchar4(0, 0, 0, 255))
	);
	for (auto& debugFrameName : kDebugFrameNames) {
		m_FrameBuffer->AddCUGLBuffer(debugFrameName);
		m_FrameBuffer->GetCUGLBuffer(debugFrameName).upload(
			std::vector<uchar4>(m_FbWidth * m_FbHeight, make_uchar4(0,0,0, 255))
		);
	}
}
void TestPG3Application::InitTracers()
{
	m_DebugActor = std::shared_ptr<test::RTTracer>(
		new TestPG3DebugTracer(this)
	);
	m_DebugActor->Initialize();
	m_TraceActor = std::shared_ptr<test::RTTracer>(
		new TestPG3SimpleTracer(this)
	);
	m_TraceActor->Initialize();
}
//ShouldClose
void TestPG3Application::PrepareLoop()
{
	m_CurTime = glfwGetTime();
}

bool TestPG3Application::QuitLoop() {
	if (!m_Window) return false;
	return glfwWindowShouldClose(m_Window);
}

void TestPG3Application::Trace()
{
	if (m_LaunchDebug) {
		TestPG3DebugTracer::UserData userData = {};
		userData.sTree = m_STree->GetGpuHandle();
		userData.diffuseBuffer = m_FrameBuffer->GetCUGLBuffer("Diffuse").map();
		userData.specularBuffer = m_FrameBuffer->GetCUGLBuffer("Specular").map();
		userData.emissionBuffer = m_FrameBuffer->GetCUGLBuffer("Emission").map();
		userData.transmitBuffer = m_FrameBuffer->GetCUGLBuffer("Transmit").map();
		userData.normalBuffer = m_FrameBuffer->GetCUGLBuffer("Normal").map();
		userData.texCoordBuffer = m_FrameBuffer->GetCUGLBuffer("TexCoord").map();
		userData.depthBuffer = m_FrameBuffer->GetCUGLBuffer("Depth").map();
		userData.sTreeColBuffer = m_FrameBuffer->GetCUGLBuffer("STree").map();

		test::RTTraceConfig traceConfig = {};
		traceConfig.width = m_FbWidth;
		traceConfig.height = m_FbHeight;
		traceConfig.depth = 1;
		traceConfig.isSync = true;
		traceConfig.pUserData = &userData;
		m_DebugActor->Launch(traceConfig);

		m_FrameBuffer->GetCUGLBuffer("Diffuse").unmap();
		m_FrameBuffer->GetCUGLBuffer("Specular").unmap();
		m_FrameBuffer->GetCUGLBuffer("Emission").unmap();
		m_FrameBuffer->GetCUGLBuffer("Transmit").unmap();
		m_FrameBuffer->GetCUGLBuffer("Normal").unmap();
		m_FrameBuffer->GetCUGLBuffer("TexCoord").unmap();
		m_FrameBuffer->GetCUGLBuffer("Depth").unmap();
		m_FrameBuffer->GetCUGLBuffer("STree").unmap();
	}
	TestPG3SimpleTracer::UserData userData = {};
	userData.sTree           = m_STree->GetGpuHandle();
	userData.frameBuffer     = m_FrameBuffer->GetCUGLBuffer("Default").map();
	userData.samplePerLaunch = m_SamplePerLaunch;
	userData.samplePerAll    = 0;

	test::RTTraceConfig traceConfig = {};
	traceConfig.width     = m_FbWidth;
	traceConfig.height    = m_FbHeight;
	traceConfig.depth     = 1;
	traceConfig.isSync    = true;
	traceConfig.pUserData = &userData;
	m_TraceActor->Launch(traceConfig);

	m_FrameBuffer->GetCUGLBuffer("Default").unmap();
	m_SamplePerALL = userData.samplePerAll;
}

void TestPG3Application::DrawFrame() {
	m_DebugTexture->upload( 0, m_FrameBuffer->GetCUGLBuffer( m_CurDebugFrame).getHandle(), 0, 0, m_FbWidth, m_FbHeight);
	m_RenderTexture->upload(0, m_FrameBuffer->GetCUGLBuffer(m_CurRenderFrame).getHandle(), 0, 0, m_FbWidth, m_FbHeight);
	m_Renderer->draw(m_RenderTexture->getID());
}
void TestPG3Application::DrawGui() {
	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();

	ImGui::PushStyleColor(ImGuiCol_TitleBgActive, ImVec4(0.0f, 0.7f, 0.2f, 1.0f));
	ImGui::PushStyleColor(ImGuiCol_TitleBg, ImVec4(0.0f, 0.3f, 0.1f, 1.0f));

	ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Once);
	ImGui::SetNextWindowSize(ImVec2(500, 500), ImGuiCond_Once);

	ImGui::Begin("GlobalSetting",nullptr,ImGuiWindowFlags_MenuBar);
	{
		float lightColor[3] = { m_ParallelLight.emission.x,m_ParallelLight.emission.y,m_ParallelLight.emission.z };
		if (ImGui::SliderFloat3("LightColor", lightColor, 0.0f, 10.0f)) {
			m_ParallelLight.emission = make_float3(lightColor[0], lightColor[1], lightColor[2]);
			m_UpdateLight = true;
		}

		float cameraFovY = m_FovY;
		if (ImGui::SliderFloat("CameraFovY", &cameraFovY, -90.0f, 90.0f)) {
			m_FovY = cameraFovY;
			m_UpdateCamera = true;
		}
		int isLockedUpdate = m_LockUpdate;
		if (ImGui::RadioButton("UnLockUpdate", &isLockedUpdate, 0)) {
			 m_LockUpdate = false;
		}
		ImGui::SameLine();
		if (ImGui::RadioButton("  LockUpdate", &isLockedUpdate, 1)) {
			m_LockUpdate  = true;
		}
	}
	ImGui::End();

	ImGui::Begin("AppSetting", nullptr, ImGuiWindowFlags_MenuBar);
	ImGui::Text( "FrameSize: (   %4d,    %4d)",        m_FbWidth,       m_FbHeight);
	ImGui::Text( "CurCursor: (%4.3lf, %4.3lf)", m_CurCursorPos.x, m_CurCursorPos.y);
	ImGui::Text( "DelCursor: (%4.3lf, %4.3lf)", m_DelCursorPos.x, m_DelCursorPos.y);
	ImGui::End();

	ImGui::Begin("TraceSetting", nullptr, ImGuiWindowFlags_MenuBar);
	ImGui::Text("      spp:     %4d", m_SamplePerALL);
	ImGui::Text("  spp/sec:  %4.3lf", (float)m_SamplePerLaunch / m_DelTime);
	{
		int samplePerLaunch   = m_SamplePerLaunch;
		if (ImGui::SliderInt("samplePerLaunch", &samplePerLaunch, 1, 10)) {
			m_SamplePerLaunch = samplePerLaunch;
		}
		int maxTraceDepth     = m_MaxTraceDepth;
		if (ImGui::SliderInt("maxTraceDepth",     &maxTraceDepth, 1, 10)) {
			m_MaxTraceDepth   = maxTraceDepth;
		}
	}
	ImGui::End();

	ImGui::Begin("DebugSetting", nullptr, ImGuiWindowFlags_MenuBar);
	{
		static int curFrameIdx = 0;
		{
			int i = 0;
			for (auto kDebugFrameName : kDebugFrameNames) {
				if (ImGui::RadioButton(kDebugFrameName, &curFrameIdx, i++)) {
					m_CurDebugFrame = kDebugFrameName;
				}
				if (i != 4 && i!=8) {
					ImGui::SameLine();
				}
				else {
					ImGui::NewLine();
				}
			}
		}

	}
	if (ImGui::Button("Launch")) {
		m_LaunchDebug = true;
	}
	else {
		m_LaunchDebug = false;
	}
	ImGui::NewLine();
	ImGui::Image(reinterpret_cast<void*>(m_DebugTexture->getID()), { 256 * m_FbAspect,256 }, { 1,1 }, {0,0});
	ImGui::End();

	ImGui::PopStyleColor();
	ImGui::PopStyleColor();
	ImGui::Render();
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}
void TestPG3Application::PollEvents()
{
	double prvTime = m_CurTime;
	m_CurTime      = glfwGetTime();
	m_DelTime      = m_CurTime - prvTime;

	if (!m_LockUpdate) {
		if (glfwGetWindowAttrib(m_Window, GLFW_RESIZABLE) == GLFW_FALSE) {
			glfwSetWindowAttrib(m_Window, GLFW_RESIZABLE, GLFW_TRUE);
		}
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
	else {
		if (glfwGetWindowAttrib(m_Window, GLFW_RESIZABLE) == GLFW_TRUE) {
			glfwSetWindowAttrib(m_Window, GLFW_RESIZABLE, GLFW_FALSE);
		}
	}
	glfwPollEvents();
}
void TestPG3Application::Update()
{
	if (m_ResizeFrame) {
		m_FrameBuffer->Resize(m_FbWidth, m_FbHeight);
		m_RenderTexture->reset();
		m_RenderTexture->allocate({(size_t) m_FbWidth,(size_t)m_FbHeight });
		m_RenderTexture->setParameteri(GL_TEXTURE_MAG_FILTER, GL_LINEAR, false);
		m_RenderTexture->setParameteri(GL_TEXTURE_MIN_FILTER, GL_LINEAR, false);
		m_RenderTexture->setParameteri(GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE, false);
		m_RenderTexture->setParameteri(GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE, false);
		m_DebugTexture->reset();
		m_DebugTexture->allocate( { (size_t)m_FbWidth,(size_t)m_FbHeight });
		m_DebugTexture->setParameteri(GL_TEXTURE_MAG_FILTER, GL_LINEAR, false);
		m_DebugTexture->setParameteri(GL_TEXTURE_MIN_FILTER, GL_LINEAR, false);
		m_DebugTexture->setParameteri(GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE, false);
		m_DebugTexture->setParameteri(GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE, false);

	}
	m_TraceActor->Update();
	m_DebugActor->Update();
	//m_GuideActor->Update();
	m_UpdateCamera = false;
	m_UpdateLight  = false;
	m_ResizeFrame  = false;
}
void TestPG3Application::LockUpdate()
{
	m_LockUpdate = true;
}
void TestPG3Application::UnLockUpdate()
{
	m_LockUpdate = false;
}
//Free
void TestPG3Application::FreeGLFW() {
	glfwTerminate();
}

void TestPG3Application::FreeWindow() {
	glfwDestroyWindow(m_Window);
	m_Window = nullptr;
}

void TestPG3Application::FreeCUDA() {
	m_Context.reset();
}

void TestPG3Application::FreeGui() {
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();
	//Renderer
	m_Renderer->reset();
	m_Renderer.reset();
}

void TestPG3Application::FreePipelines() {
	m_HGProgramGroups.clear();
	m_MSProgramGroups.clear();
	m_RGProgramGroups.clear();
	m_Modules.clear();
	m_Pipelines.clear();

}

void TestPG3Application::FreeAssets() {
	m_TextureAssets.Reset();
	m_ObjModelAssets.Reset();
}

void TestPG3Application::FreeAccelerationStructures()
{
	m_GASHandles.clear();
	m_IASHandles.clear();
}

void TestPG3Application::FreeLight()
{
	m_ParallelLight = {};
}

void TestPG3Application::FreeCamera()
{
	m_CameraController = {};
}

void TestPG3Application::FreeSTree()
{
	m_STree.reset();
}

void TestPG3Application::FreeFrameResources() {
	m_RenderTexture->reset();
	m_FrameBuffer->CleanUp();
}

void TestPG3Application::FreeTracers()
{
	m_DebugActor->CleanUp();
	m_TraceActor->CleanUp();
}

auto TestPG3Application::GetOPXContext() const -> std::shared_ptr<rtlib::OPXContext>
{
	return m_Context;
}

auto TestPG3Application::GetOPXPipeline(const std::string& name) ->rtlib::OPXPipeline&
{
	return m_Pipelines.at(name);
}

auto TestPG3Application::GetRGProgramGroup(const std::string& name)  -> rtlib::OPXRaygenPG&
{
	return m_RGProgramGroups.at(name);
}

auto TestPG3Application::GetMSProgramGroup(const std::string& name) -> rtlib::OPXMissPG&
{
	return m_MSProgramGroups.at(name);
}

auto TestPG3Application::GetHGProgramGroup(const std::string& name) -> rtlib::OPXHitgroupPG&
{
	return m_HGProgramGroups.at(name);
}

auto TestPG3Application::GetTLAS() const -> rtlib::ext::IASHandlePtr
{
	return m_IASHandles.at("TopLevel");
}

auto TestPG3Application::GetMaterials() const -> const std::vector<rtlib::ext::Material>&
{
	// TODO: return ステートメントをここに挿入します
	return m_Materials;
}

auto TestPG3Application::GetCamera() const -> rtlib::Camera
{
	return m_CameraController.GetCamera(m_FovY, m_FbAspect);
}

auto TestPG3Application::GetLight() const -> ParallelLight
{
	return m_ParallelLight;
}

auto TestPG3Application::GetSTree() const -> std::shared_ptr<test::RTSTreeWrapper>
{
	return m_STree;
}

auto TestPG3Application::GetTexture(const std::string& name) const -> const rtlib::CUDATexture2D<uchar4>&
{
	// TODO: return ステートメントをここに挿入します
	return m_TextureAssets.GetAsset(name);
}
