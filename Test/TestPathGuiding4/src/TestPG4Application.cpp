#include "../include/TestPG4Application.h"
#include "../include/RTUtils.h"
#include <RTLib/Optix.h>
#include <RTLib/Utils.h>
#include <RTLib/ext/Resources/CUDA.h>
#include <nlohmann/json.hpp>
#include <fstream>
#include <string>
#include <random>
using namespace std::string_literals;
static auto SpecifyMaterialType(const rtlib::ext::VariableMap &material) -> std::string
{
	auto emitCol  = material.GetFloat3As<float3>("emitCol");
	auto specCol  = material.GetFloat3As<float3>("specCol");
	auto tranCol  = material.GetFloat3As<float3>("tranCol");
	auto refrIndx = material.GetFloat1("refrIndx");
	auto shinness = material.GetFloat1("shinness");
	auto illum    = material.GetUInt32("illum");
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
		return "Phong";
	}
};
//SimpleTracer
class TestPG4SimpleTracer : public test::RTTracer
{
public:
	struct UserData
	{
		uchar4 *frameBuffer;
		unsigned int samplePerAll;
		unsigned int samplePerLaunch;
	};

private:
	using RayGRecord = rtlib::SBTRecord<RayGenData>;
	using MissRecord = rtlib::SBTRecord<MissData>;
	using HitGRecord = rtlib::SBTRecord<HitgroupData>;
	using Pipeline = rtlib::OPXPipeline;
	using ModuleMap = std::unordered_map<std::string, rtlib::OPXModule>;
	using RGProgramGroupMap = std::unordered_map<std::string, rtlib::OPXRaygenPG>;
	using MSProgramGroupMap = std::unordered_map<std::string, rtlib::OPXMissPG>;
	using HGProgramGroupMap = std::unordered_map<std::string, rtlib::OPXHitgroupPG>;

public:
	TestPG4SimpleTracer(TestPG4Application *app)
	{
		m_ParentApp = app;
	}
	// RTTracer ����Čp������܂���
	virtual void Initialize() override
	{
		this->InitFrameResources();
		this->InitPipeline();
		this->InitShaderBindingTable();
		this->InitLaunchParams();
	}
	virtual void Launch(const test::RTTraceConfig &config) override
	{
		UserData *pUserData = (UserData *)config.pUserData;
		if (!pUserData)
		{
			return;
		}
		m_Params.cpuHandle[0].width = config.width;
		m_Params.cpuHandle[0].height = config.height;
		m_Params.cpuHandle[0].sdTree = m_ParentApp->GetSTree()->GetGpuHandle();
		m_Params.cpuHandle[0].accumBuffer = m_ParentApp->GetAccumBuffer().getDevicePtr();
		m_Params.cpuHandle[0].frameBuffer = pUserData->frameBuffer;
		m_Params.cpuHandle[0].seedBuffer  = m_SeedBuffer.getDevicePtr();
		m_Params.cpuHandle[0].isBuilt = false;
		m_Params.cpuHandle[0].samplePerLaunch = pUserData->samplePerLaunch;
		cudaMemcpyAsync(m_Params.gpuHandle.getDevicePtr(), &m_Params.cpuHandle[0], sizeof(RayTraceParams), cudaMemcpyHostToDevice, config.stream);
		m_Pipeline.launch(config.stream, m_Params.gpuHandle.getDevicePtr(), m_ShaderBindingTable, config.width, config.height, config.depth);
		if (config.isSync)
		{
			RTLIB_CU_CHECK(cuStreamSynchronize(config.stream));
		}
		m_Params.cpuHandle[0].samplePerALL += m_Params.cpuHandle[0].samplePerLaunch;
		m_SamplePerAll = pUserData->samplePerAll = m_Params.cpuHandle[0].samplePerALL;
	}
	virtual void CleanUp() override
	{
		m_ParentApp       = nullptr;
		this->FreePipeline();
		this->FreeShaderBindingTable();
		this->FreeLaunchParams();
		this->FreeFrameResources();
		m_LightHgRecIndex = 0;
	}
	virtual void Update() override
	{
		if (m_ParentApp->IsCameraUpdated() || m_ParentApp->IsFrameResized())
		{
			auto camera = m_ParentApp->GetCamera();
			m_RGRecordBuffer.cpuHandle[0].data.eye = camera.getEye();
			auto [u, v, w] = camera.getUVW();
			m_RGRecordBuffer.cpuHandle[0].data.u = u;
			m_RGRecordBuffer.cpuHandle[0].data.v = v;
			m_RGRecordBuffer.cpuHandle[0].data.w = w;
			m_RGRecordBuffer.Upload();
		}
		if (m_ParentApp->IsCameraUpdated() || m_ParentApp->IsFrameResized() || m_ParentApp->IsFrameFlushed() || m_ParentApp->IsTraceChanged() || m_ParentApp->IsLightUpdated())
		{
			m_Params.cpuHandle[0].maxTraceDepth = m_ParentApp->GetMaxTraceDepth();
			m_Params.cpuHandle[0].samplePerALL = 0;
		}
		bool shouldRegen = ((m_SamplePerAll + m_Params.cpuHandle[0].samplePerLaunch) / 1024 != m_SamplePerAll / 1024);
		if (m_ParentApp->IsFrameResized() || shouldRegen)
		{
			std::cout << "Regen!\n";
			std::vector<unsigned int> seedData(m_ParentApp->GetFbWidth() * m_ParentApp->GetFbHeight());
			std::random_device rd;
			std::mt19937 mt(rd());
			std::generate(seedData.begin(), seedData.end(), mt);
			m_SeedBuffer.resize(m_ParentApp->GetFbWidth() * m_ParentApp->GetFbHeight());
			m_SeedBuffer.upload(seedData);
		}
		if (m_ParentApp->IsLightUpdated())
		{
			auto lightColor = make_float4(m_ParentApp->GetBackGroundLight(), 1.0f);
			m_MSRecordBuffers.cpuHandle[RAY_TYPE_RADIANCE].data.bgColor = lightColor;
			RTLIB_CUDA_CHECK(cudaMemcpy(m_MSRecordBuffers.gpuHandle.getDevicePtr() + RAY_TYPE_RADIANCE, &m_MSRecordBuffers.cpuHandle[RAY_TYPE_RADIANCE], sizeof(MissRecord), cudaMemcpyHostToDevice));
		}
	}
	virtual ~TestPG4SimpleTracer() {}
private:
	void InitPipeline()
	{
		auto rayTracePtxFile = std::ifstream(TEST_TEST_PG_CUDA_PATH "/RayTrace.ptx", std::ios::binary);
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
			traceLinkOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;
#else
			traceLinkOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;
#endif
			traceLinkOptions.maxTraceDepth = 2;
		}
		auto traceModuleOptions = OptixModuleCompileOptions{};
		{
#ifndef NDEBUG
			traceModuleOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;
			traceModuleOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
#else
			traceModuleOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;
			traceModuleOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
#endif
		}
		traceModuleOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
		m_Pipeline = m_ParentApp->GetOPXContext()->createPipeline(traceCompileOptions);
		m_Modules["RayTrace"] = m_Pipeline.createModule(rayTracePtxData, traceModuleOptions);
		m_RGProgramGroups["Trace.Default"] = m_Pipeline.createRaygenPG({m_Modules["RayTrace"], RTLIB_RAYGEN_PROGRAM_STR(def)});
		m_MSProgramGroups["Trace.Radiance"] = m_Pipeline.createMissPG({m_Modules["RayTrace"], RTLIB_MISS_PROGRAM_STR(radiance)});
		m_MSProgramGroups["Trace.Occluded"] = m_Pipeline.createMissPG({m_Modules["RayTrace"], RTLIB_MISS_PROGRAM_STR(occluded)});
		m_HGProgramGroups["Trace.Radiance.Diffuse.Default"] = m_Pipeline.createHitgroupPG({m_Modules["RayTrace"], RTLIB_CLOSESTHIT_PROGRAM_STR(radiance_for_diffuse_def)}, {}, {});
		m_HGProgramGroups["Trace.Radiance.Phong.Default"] = m_Pipeline.createHitgroupPG({m_Modules["RayTrace"], RTLIB_CLOSESTHIT_PROGRAM_STR(radiance_for_phong_def)}, {}, {});
		m_HGProgramGroups["Trace.Radiance.Emission"] = m_Pipeline.createHitgroupPG({m_Modules["RayTrace"], RTLIB_CLOSESTHIT_PROGRAM_STR(radiance_for_emission)}, {}, {});
		m_HGProgramGroups["Trace.Radiance.Specular"] = m_Pipeline.createHitgroupPG({m_Modules["RayTrace"], RTLIB_CLOSESTHIT_PROGRAM_STR(radiance_for_specular)}, {}, {});
		m_HGProgramGroups["Trace.Radiance.Refraction"] = m_Pipeline.createHitgroupPG({m_Modules["RayTrace"], RTLIB_CLOSESTHIT_PROGRAM_STR(radiance_for_refraction)}, {}, {});
		m_HGProgramGroups["Trace.Occluded"] = m_Pipeline.createHitgroupPG({m_Modules["RayTrace"], RTLIB_CLOSESTHIT_PROGRAM_STR(occluded)}, {}, {});
		m_Pipeline.link(traceLinkOptions);
	}
	void InitFrameResources()
	{
		std::vector<unsigned int> seedData(m_ParentApp->GetFbWidth() * m_ParentApp->GetFbHeight());
		std::random_device rd;
		std::mt19937 mt(rd());
		std::generate(seedData.begin(), seedData.end(), mt);
		m_SeedBuffer = rtlib::CUDABuffer<uint32_t>(seedData);
	}
	void InitShaderBindingTable()
	{
		auto tlas = m_ParentApp->GetTLAS();
		auto camera = m_ParentApp->GetCamera();
		auto &materials = m_ParentApp->GetMaterials();
		m_RGRecordBuffer.cpuHandle.resize(1);
		m_RGRecordBuffer.cpuHandle[0] = m_RGProgramGroups["Trace.Default"].getSBTRecord<RayGenData>();
		m_RGRecordBuffer.cpuHandle[0].data.eye = camera.getEye();
		auto [u, v, w] = camera.getUVW();
		m_RGRecordBuffer.cpuHandle[0].data.u = u;
		m_RGRecordBuffer.cpuHandle[0].data.v = v;
		m_RGRecordBuffer.cpuHandle[0].data.w = w;
		m_RGRecordBuffer.Upload();
		m_MSRecordBuffers.cpuHandle.resize(RAY_TYPE_COUNT);
		m_MSRecordBuffers.cpuHandle[RAY_TYPE_RADIANCE] = m_MSProgramGroups["Trace.Radiance"].getSBTRecord<MissData>();
		m_MSRecordBuffers.cpuHandle[RAY_TYPE_RADIANCE].data.bgColor = make_float4(m_ParentApp->GetBackGroundLight(),1.0f);
		m_MSRecordBuffers.cpuHandle[RAY_TYPE_OCCLUSION] = m_MSProgramGroups["Trace.Occluded"].getSBTRecord<MissData>();
		m_MSRecordBuffers.Upload();
		m_HGRecordBuffers.cpuHandle.resize(tlas->GetSbtCount() * RAY_TYPE_COUNT);
		{
			size_t sbtOffset = 0;
			for (auto &instanceSet : tlas->GetInstanceSets())
			{
				for (auto &baseGASHandle : instanceSet->baseGASHandles)
				{
					for (auto &mesh : baseGASHandle->GetMeshes())
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
						auto cudaTriIndBuffer = mesh->GetUniqueResource()->triIndBuffer.GetGpuComponent<rtlib::ext::resources::CUDABufferComponent<uint3>> ("CUDA");
						for (size_t i = 0; i < mesh->GetUniqueResource()->materials.size(); ++i)
						{
							auto materialId = mesh->GetUniqueResource()->materials[i];
							auto &material = materials[materialId];
							HitgroupData radianceHgData = {};
							{
								radianceHgData.vertices   = cudaVertexBuffer->GetHandle().getDevicePtr();
								radianceHgData.normals    = cudaNormalBuffer->GetHandle().getDevicePtr();
								radianceHgData.texCoords  = cudaTexCrdBuffer->GetHandle().getDevicePtr();
								radianceHgData.indices    = cudaTriIndBuffer->GetHandle().getDevicePtr();
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
							if (material.GetString("name") == "light")
							{
								m_LightHgRecIndex = RAY_TYPE_COUNT * sbtOffset + RAY_TYPE_COUNT * i + RAY_TYPE_RADIANCE;
							}
							std::string typeString = SpecifyMaterialType(material);
							if (typeString == "Phong" || typeString == "Diffuse")
							{
								typeString += ".Default";
							}
							m_HGRecordBuffers.cpuHandle[RAY_TYPE_COUNT * sbtOffset + RAY_TYPE_COUNT * i + RAY_TYPE_RADIANCE] = m_HGProgramGroups[std::string("Trace.Radiance.") + typeString].getSBTRecord<HitgroupData>(radianceHgData);
							m_HGRecordBuffers.cpuHandle[RAY_TYPE_COUNT * sbtOffset + RAY_TYPE_COUNT * i + RAY_TYPE_OCCLUSION] = m_HGProgramGroups["Trace.Occluded"].getSBTRecord<HitgroupData>({});
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
	void InitLaunchParams()
	{
		auto tlas = m_ParentApp->GetTLAS();
		m_Params.cpuHandle.resize(1);
		m_Params.cpuHandle[0].gasHandle = tlas->GetHandle();
		m_Params.cpuHandle[0].sdTree    = m_ParentApp->GetSTree()->GetGpuHandle();
		m_Params.cpuHandle[0].width     = m_ParentApp->GetFbWidth();
		m_Params.cpuHandle[0].height    = m_ParentApp->GetFbHeight();
		m_Params.cpuHandle[0].light     = m_ParentApp->GetMeshLightList();
		m_Params.cpuHandle[0].accumBuffer =  m_ParentApp->GetAccumBuffer().getDevicePtr();
		m_Params.cpuHandle[0].frameBuffer = nullptr;
		m_Params.cpuHandle[0].seedBuffer = m_SeedBuffer.getDevicePtr();
		m_Params.cpuHandle[0].isBuilt = false;
		m_Params.cpuHandle[0].maxTraceDepth = m_ParentApp->GetMaxTraceDepth();
		m_Params.cpuHandle[0].samplePerLaunch = 1;

		m_Params.Upload();
	}
	void FreePipeline()
	{
		m_ShaderBindingTable = {};
		m_RGRecordBuffer.Reset();
		m_MSRecordBuffers.Reset();
		m_HGRecordBuffers.Reset();
	}
	void FreeShaderBindingTable()
	{
		m_ShaderBindingTable = {};
		m_RGRecordBuffer.Reset();
		m_MSRecordBuffers.Reset();
		m_HGRecordBuffers.Reset();
	}
	void FreeLaunchParams()
	{
		m_Params.Reset();
	}
	void FreeFrameResources()
	{
		m_SeedBuffer.reset();
	}

private:
	TestPG4Application *m_ParentApp = nullptr;
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
	unsigned int m_LightHgRecIndex = 0;
	unsigned int m_SamplePerAll = 0;
};
//  NEETracer
class TestPG4SimpleNEETracer : public test::RTTracer
{
public:
	struct UserData
	{
		uchar4 *frameBuffer;
		unsigned int samplePerAll;
		unsigned int samplePerLaunch;
	};

private:
	using RayGRecord = rtlib::SBTRecord<RayGenData>;
	using MissRecord = rtlib::SBTRecord<MissData>;
	using HitGRecord = rtlib::SBTRecord<HitgroupData>;
	using Pipeline = rtlib::OPXPipeline;
	using ModuleMap = std::unordered_map<std::string, rtlib::OPXModule>;
	using RGProgramGroupMap = std::unordered_map<std::string, rtlib::OPXRaygenPG>;
	using MSProgramGroupMap = std::unordered_map<std::string, rtlib::OPXMissPG>;
	using HGProgramGroupMap = std::unordered_map<std::string, rtlib::OPXHitgroupPG>;

public:
	TestPG4SimpleNEETracer(TestPG4Application *app)
	{
		m_ParentApp = app;
	}
	// RTTracer ����Čp������܂���
	virtual void Initialize() override
	{
		this->InitFrameResources();
		this->InitPipeline();
		this->InitShaderBindingTable();
		this->InitLaunchParams();
	}
	virtual void Launch(const test::RTTraceConfig &config) override
	{
		UserData *pUserData = (UserData *)config.pUserData;
		if (!pUserData)
		{
			return;
		}
		m_Params.cpuHandle[0].width = config.width;
		m_Params.cpuHandle[0].height = config.height;
		m_Params.cpuHandle[0].accumBuffer = m_ParentApp->GetAccumBuffer().getDevicePtr();
		m_Params.cpuHandle[0].frameBuffer = pUserData->frameBuffer;
		m_Params.cpuHandle[0].seedBuffer = m_SeedBuffer.getDevicePtr();
		m_Params.cpuHandle[0].samplePerLaunch = pUserData->samplePerLaunch;
		cudaMemcpyAsync(m_Params.gpuHandle.getDevicePtr(), &m_Params.cpuHandle[0], sizeof(RayTraceParams), cudaMemcpyHostToDevice, config.stream);
		m_Pipeline.launch(config.stream, m_Params.gpuHandle.getDevicePtr(), m_ShaderBindingTable, config.width, config.height, config.depth);
		if (config.isSync)
		{
			RTLIB_CU_CHECK(cuStreamSynchronize(config.stream));
		}
		m_Params.cpuHandle[0].samplePerALL += m_Params.cpuHandle[0].samplePerLaunch;
		m_SamplePerAll = pUserData->samplePerAll = m_Params.cpuHandle[0].samplePerALL;
	}
	virtual void CleanUp() override
	{
		m_ParentApp = nullptr;
		this->FreePipeline();
		this->FreeShaderBindingTable();
		this->FreeLaunchParams();
		this->FreeFrameResources();
		m_LightHgRecIndex = 0;
	}
	virtual void Update() override
	{
		if (m_ParentApp->IsCameraUpdated() || m_ParentApp->IsFrameResized())
		{
			auto camera = m_ParentApp->GetCamera();
			m_RGRecordBuffer.cpuHandle[0].data.eye = camera.getEye();
			auto [u, v, w] = camera.getUVW();
			m_RGRecordBuffer.cpuHandle[0].data.u = u;
			m_RGRecordBuffer.cpuHandle[0].data.v = v;
			m_RGRecordBuffer.cpuHandle[0].data.w = w;
			m_RGRecordBuffer.Upload();
		}
		if (m_ParentApp->IsCameraUpdated() || m_ParentApp->IsFrameResized() || m_ParentApp->IsFrameFlushed() || m_ParentApp->IsTraceChanged() || m_ParentApp->IsLightUpdated())
		{
			m_Params.cpuHandle[0].maxTraceDepth = m_ParentApp->GetMaxTraceDepth();
			m_Params.cpuHandle[0].samplePerALL = 0;
		}
		bool shouldRegen = ((m_SamplePerAll + m_Params.cpuHandle[0].samplePerLaunch) / 1024 != m_SamplePerAll / 1024);
		if (m_ParentApp->IsFrameResized() || shouldRegen)
		{
			std::cout << "Regen!\n";
			std::vector<unsigned int> seedData(m_ParentApp->GetFbWidth() * m_ParentApp->GetFbHeight());
			std::random_device rd;
			std::mt19937 mt(rd());
			std::generate(seedData.begin(), seedData.end(), mt);
			m_SeedBuffer.resize(m_ParentApp->GetFbWidth() * m_ParentApp->GetFbHeight());
			m_SeedBuffer.upload(seedData);
		}
		if (m_ParentApp->IsLightUpdated())
		{

			auto lightColor = make_float4(m_ParentApp->GetBackGroundLight(), 1.0f);
			m_MSRecordBuffers.cpuHandle[RAY_TYPE_RADIANCE].data.bgColor = lightColor;
			RTLIB_CUDA_CHECK(cudaMemcpy(m_MSRecordBuffers.gpuHandle.getDevicePtr() + RAY_TYPE_RADIANCE, &m_MSRecordBuffers.cpuHandle[RAY_TYPE_RADIANCE], sizeof(MissRecord), cudaMemcpyHostToDevice));
		}
	}
	virtual ~TestPG4SimpleNEETracer() {}

private:
	void InitPipeline()
	{
		auto rayTracePtxFile = std::ifstream(TEST_TEST_PG_CUDA_PATH "/RayTrace.ptx", std::ios::binary);
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
			traceLinkOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;
#else
			traceLinkOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;
#endif
			traceLinkOptions.maxTraceDepth = 2;
		}
		auto traceModuleOptions = OptixModuleCompileOptions{};
		{
#ifndef NDEBUG
			traceModuleOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;
			traceModuleOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
#else
			traceModuleOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;
			traceModuleOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
#endif
		}
		traceModuleOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
		m_Pipeline = m_ParentApp->GetOPXContext()->createPipeline(traceCompileOptions);
		m_Modules["RayTrace"] = m_Pipeline.createModule(rayTracePtxData, traceModuleOptions);
		m_RGProgramGroups["Trace.Default"] = m_Pipeline.createRaygenPG({m_Modules["RayTrace"], RTLIB_RAYGEN_PROGRAM_STR(def)});
		m_MSProgramGroups["Trace.Radiance"] = m_Pipeline.createMissPG({m_Modules["RayTrace"], RTLIB_MISS_PROGRAM_STR(radiance)});
		m_MSProgramGroups["Trace.Occluded"] = m_Pipeline.createMissPG({m_Modules["RayTrace"], RTLIB_MISS_PROGRAM_STR(occluded)});
		m_HGProgramGroups["Trace.Radiance.Diffuse.NEE.Default"] = m_Pipeline.createHitgroupPG({ m_Modules["RayTrace"], RTLIB_CLOSESTHIT_PROGRAM_STR(radiance_for_diffuse_nee_with_def_light) }, {}, {});
		m_HGProgramGroups["Trace.Radiance.Diffuse.NEE.NEE"]     = m_Pipeline.createHitgroupPG({ m_Modules["RayTrace"], RTLIB_CLOSESTHIT_PROGRAM_STR(radiance_for_diffuse_nee_with_nee_light)}, {}, {});
		m_HGProgramGroups["Trace.Radiance.Phong.NEE.Default"]   = m_Pipeline.createHitgroupPG({ m_Modules["RayTrace"], RTLIB_CLOSESTHIT_PROGRAM_STR(radiance_for_phong_nee_with_def_light)}, {}, {});
		m_HGProgramGroups["Trace.Radiance.Phong.NEE.NEE"]       = m_Pipeline.createHitgroupPG({ m_Modules["RayTrace"], RTLIB_CLOSESTHIT_PROGRAM_STR(radiance_for_phong_nee_with_nee_light) }, {}, {});
		m_HGProgramGroups["Trace.Radiance.Emission.Default"]    = m_Pipeline.createHitgroupPG({ m_Modules["RayTrace"], RTLIB_CLOSESTHIT_PROGRAM_STR(radiance_for_emission_with_def_light) }, {}, {});
		m_HGProgramGroups["Trace.Radiance.Emission.NEE"]        = m_Pipeline.createHitgroupPG({ m_Modules["RayTrace"], RTLIB_CLOSESTHIT_PROGRAM_STR(radiance_for_emission_with_nee_light) }, {}, {});
		m_HGProgramGroups["Trace.Radiance.Specular"] = m_Pipeline.createHitgroupPG({m_Modules["RayTrace"], RTLIB_CLOSESTHIT_PROGRAM_STR(radiance_for_specular)}, {}, {});
		m_HGProgramGroups["Trace.Radiance.Refraction"] = m_Pipeline.createHitgroupPG({m_Modules["RayTrace"], RTLIB_CLOSESTHIT_PROGRAM_STR(radiance_for_refraction)}, {}, {});
		m_HGProgramGroups["Trace.Occluded"] = m_Pipeline.createHitgroupPG({m_Modules["RayTrace"], RTLIB_CLOSESTHIT_PROGRAM_STR(occluded)}, {}, {});
		m_Pipeline.link(traceLinkOptions);
	}
	void InitFrameResources()
	{
		std::vector<unsigned int> seedData(m_ParentApp->GetFbWidth() * m_ParentApp->GetFbHeight());
		std::random_device rd;
		std::mt19937 mt(rd());
		std::generate(seedData.begin(), seedData.end(), mt);
		m_SeedBuffer = rtlib::CUDABuffer<uint32_t>(seedData);
	}
	void InitShaderBindingTable()
	{
		auto tlas = m_ParentApp->GetTLAS();
		auto camera = m_ParentApp->GetCamera();
		auto &materials = m_ParentApp->GetMaterials();
		m_RGRecordBuffer.cpuHandle.resize(1);
		m_RGRecordBuffer.cpuHandle[0] = m_RGProgramGroups["Trace.Default"].getSBTRecord<RayGenData>();
		m_RGRecordBuffer.cpuHandle[0].data.eye = camera.getEye();
		auto [u, v, w] = camera.getUVW();
		m_RGRecordBuffer.cpuHandle[0].data.u = u;
		m_RGRecordBuffer.cpuHandle[0].data.v = v;
		m_RGRecordBuffer.cpuHandle[0].data.w = w;
		m_RGRecordBuffer.Upload();
		m_MSRecordBuffers.cpuHandle.resize(RAY_TYPE_COUNT);
		m_MSRecordBuffers.cpuHandle[RAY_TYPE_RADIANCE] = m_MSProgramGroups["Trace.Radiance"].getSBTRecord<MissData>();
		m_MSRecordBuffers.cpuHandle[RAY_TYPE_RADIANCE].data.bgColor = make_float4(m_ParentApp->GetBackGroundLight(), 1.0f);
		m_MSRecordBuffers.cpuHandle[RAY_TYPE_OCCLUSION] = m_MSProgramGroups["Trace.Occluded"].getSBTRecord<MissData>();
		m_MSRecordBuffers.Upload();
		m_HGRecordBuffers.cpuHandle.resize(tlas->GetSbtCount() * RAY_TYPE_COUNT);
		{
			size_t sbtOffset = 0;
			for (auto &instanceSet : tlas->GetInstanceSets())
			{
				for (auto &baseGASHandle : instanceSet->baseGASHandles)
				{
					for (auto &mesh : baseGASHandle->GetMeshes())
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
							auto &material = materials[materialId];
							HitgroupData radianceHgData = {};
							{
								radianceHgData.vertices = cudaVertexBuffer->GetHandle().getDevicePtr();
								radianceHgData.normals = cudaNormalBuffer->GetHandle().getDevicePtr();
								radianceHgData.texCoords = cudaTexCrdBuffer->GetHandle().getDevicePtr();
								radianceHgData.indices = cudaTriIndBuffer->GetHandle().getDevicePtr();
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
							if (material.GetString("name") == "light")
							{
								m_LightHgRecIndex = RAY_TYPE_COUNT * sbtOffset + RAY_TYPE_COUNT * i + RAY_TYPE_RADIANCE;
							}
							std::string typeString = SpecifyMaterialType(material);
							if (typeString == "Phong" || typeString == "Diffuse")
							{
								typeString += ".NEE";
								if (mesh->GetUniqueResource()->variables.GetBool("hasLight") && mesh->GetUniqueResource()->variables.GetBool("useNEE"))
								{
									typeString += ".NEE";
								}
								else
								{
									typeString += ".Default";
								}
							}
							if (typeString == "Emission")
							{
								if (mesh->GetUniqueResource()->variables.GetBool("hasLight") && mesh->GetUniqueResource()->variables.GetBool("useNEE"))
								{
									typeString += ".NEE";
								}
								else
								{
									typeString += ".Default";
								}
							}
							m_HGRecordBuffers.cpuHandle[RAY_TYPE_COUNT * sbtOffset + RAY_TYPE_COUNT * i + RAY_TYPE_RADIANCE] = m_HGProgramGroups[std::string("Trace.Radiance.") + typeString].getSBTRecord<HitgroupData>(radianceHgData);
							m_HGRecordBuffers.cpuHandle[RAY_TYPE_COUNT * sbtOffset + RAY_TYPE_COUNT * i + RAY_TYPE_OCCLUSION] = m_HGProgramGroups["Trace.Occluded"].getSBTRecord<HitgroupData>({});
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
	void InitLaunchParams()
	{
		auto tlas = m_ParentApp->GetTLAS();
		m_Params.cpuHandle.resize(1);
		m_Params.cpuHandle[0].gasHandle = tlas->GetHandle();
		m_Params.cpuHandle[0].width = m_ParentApp->GetFbWidth();
		m_Params.cpuHandle[0].height = m_ParentApp->GetFbHeight();
		m_Params.cpuHandle[0].light = m_ParentApp->GetMeshLightList();
		m_Params.cpuHandle[0].accumBuffer = m_AccumBuffer.getDevicePtr();
		m_Params.cpuHandle[0].frameBuffer = nullptr;
		m_Params.cpuHandle[0].seedBuffer = m_SeedBuffer.getDevicePtr();
		m_Params.cpuHandle[0].maxTraceDepth = m_ParentApp->GetMaxTraceDepth();
		m_Params.cpuHandle[0].samplePerLaunch = 1;

		m_Params.Upload();
	}
	void FreePipeline()
	{
		m_ShaderBindingTable = {};
		m_RGRecordBuffer.Reset();
		m_MSRecordBuffers.Reset();
		m_HGRecordBuffers.Reset();
	}
	void FreeShaderBindingTable()
	{
		m_ShaderBindingTable = {};
		m_RGRecordBuffer.Reset();
		m_MSRecordBuffers.Reset();
		m_HGRecordBuffers.Reset();
	}
	void FreeLaunchParams()
	{
		m_Params.Reset();
	}
	void FreeFrameResources()
	{
		m_AccumBuffer.reset();
		m_SeedBuffer.reset();
	}

private:
	TestPG4Application *m_ParentApp = nullptr;
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
	rtlib::CUDABuffer<float3> m_AccumBuffer = {};
	rtlib::CUDABuffer<unsigned int> m_SeedBuffer = {};
	unsigned int m_LightHgRecIndex = 0;
	unsigned int m_SamplePerAll = 0;
};
// GuideTracer
class TestPG4GuideTracer : public test::RTTracer
{
public:
	struct UserData
	{
		uchar4 *frameBuffer;
		unsigned int samplePerAll;
		unsigned int samplePerLaunch;
		unsigned int sampleForBudget;
		unsigned int iterationForBuilt;
	};
private:
	using RayGRecord = rtlib::SBTRecord<RayGenData>;
	using MissRecord = rtlib::SBTRecord<MissData>;
	using HitGRecord = rtlib::SBTRecord<HitgroupData>;
	using Pipeline = rtlib::OPXPipeline;
	using ModuleMap = std::unordered_map<std::string, rtlib::OPXModule>;
	using RGProgramGroupMap = std::unordered_map<std::string, rtlib::OPXRaygenPG>;
	using MSProgramGroupMap = std::unordered_map<std::string, rtlib::OPXMissPG>;
	using HGProgramGroupMap = std::unordered_map<std::string, rtlib::OPXHitgroupPG>;

public:
	TestPG4GuideTracer(TestPG4Application *app)
	{
		m_ParentApp = app;
	}
	// RTTracer ����Čp������܂���
	virtual void Initialize() override
	{
		this->InitFrameResources();
		this->InitPipeline();
		this->InitShaderBindingTable();
		this->InitLaunchParams();
	}
	virtual void Launch(const test::RTTraceConfig &config) override
	{
		OnLaunchBegin(config);
		OnLaunchExecute(config);
		OnLaunchEnd(config);
	}
	virtual void CleanUp() override
	{
		m_ParentApp = nullptr;
		this->FreePipeline();
		this->FreeShaderBindingTable();
		this->FreeLaunchParams();
		this->FreeFrameResources();
		m_LightHgRecIndex = 0;
	}
	virtual void Update() override
	{
		if (m_ParentApp->IsCameraUpdated() || m_ParentApp->IsFrameResized())
		{
			auto camera = m_ParentApp->GetCamera();
			m_RGRecordBuffer.cpuHandle[0].data.eye = camera.getEye();
			auto [u, v, w] = camera.getUVW();
			m_RGRecordBuffer.cpuHandle[0].data.u = u;
			m_RGRecordBuffer.cpuHandle[0].data.v = v;
			m_RGRecordBuffer.cpuHandle[0].data.w = w;
			m_RGRecordBuffer.Upload();
		}
		if (m_ParentApp->IsCameraUpdated() || m_ParentApp->IsFrameResized() || m_ParentApp->IsFrameFlushed() || m_ParentApp->IsTraceChanged() || m_ParentApp->IsLightUpdated())
		{
			m_Params.cpuHandle[0].maxTraceDepth = m_ParentApp->GetMaxTraceDepth();
			m_Params.cpuHandle[0].samplePerALL = 0;
		}
		bool shouldRegen = ((m_SamplePerAll + m_Params.cpuHandle[0].samplePerLaunch) / 1024 != m_SamplePerAll / 1024);
		if (m_ParentApp->IsFrameResized() || shouldRegen)
		{
			std::cout << "Regen!\n";
			std::vector<unsigned int> seedData(m_ParentApp->GetFbWidth() * m_ParentApp->GetFbHeight());
			std::random_device rd;
			std::mt19937 mt(rd());
			std::generate(seedData.begin(), seedData.end(), mt);
			m_SeedBuffer.resize(m_ParentApp->GetFbWidth() * m_ParentApp->GetFbHeight());
			m_SeedBuffer.upload(seedData);
		}
		if (m_ParentApp->IsLightUpdated())
		{
			auto lightColor = make_float4(m_ParentApp->GetBackGroundLight(), 1.0f);
			m_MSRecordBuffers.cpuHandle[RAY_TYPE_RADIANCE].data.bgColor = lightColor;
			RTLIB_CUDA_CHECK(cudaMemcpy(m_MSRecordBuffers.gpuHandle.getDevicePtr() + RAY_TYPE_RADIANCE, &m_MSRecordBuffers.cpuHandle[RAY_TYPE_RADIANCE], sizeof(MissRecord), cudaMemcpyHostToDevice));
		}
	}
	virtual bool ShouldLock()const noexcept { return true; }
	virtual ~TestPG4GuideTracer() {}

private:
	void InitFrameResources()
	{
		std::vector<unsigned int> seedData(m_ParentApp->GetFbWidth() * m_ParentApp->GetFbHeight());
		std::random_device rd;
		std::mt19937 mt(rd());
		std::generate(seedData.begin(), seedData.end(), mt);
		m_SeedBuffer = rtlib::CUDABuffer<uint32_t>(seedData);
	}
	void InitPipeline()
	{
		auto rayGuidePtxFile = std::ifstream(TEST_TEST_PG_CUDA_PATH "/RayGuide.ptx", std::ios::binary);
		if (!rayGuidePtxFile.is_open())
			throw std::runtime_error("Failed To Load RayGuide.ptx!");
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
			guideLinkOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;
#else
			guideLinkOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
#endif
			guideLinkOptions.maxTraceDepth = 2;
		}
		auto guideModuleOptions = OptixModuleCompileOptions{};
		{
#ifndef NDEBUG
			guideModuleOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;
			guideModuleOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
#else
			guideModuleOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
			guideModuleOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
#endif
		}
		guideModuleOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
		m_Pipeline = m_ParentApp->GetOPXContext()->createPipeline(guideCompileOptions);
		m_Modules["RayGuide"] = m_Pipeline.createModule(rayGuidePtxData, guideModuleOptions);
		m_RGProgramGroups["Guide.Default"] = m_Pipeline.createRaygenPG({m_Modules["RayGuide"], RTLIB_RAYGEN_PROGRAM_STR(def)});
		m_RGProgramGroups["Guide.Guiding.Default"] = m_Pipeline.createRaygenPG({m_Modules["RayGuide"], RTLIB_RAYGEN_PROGRAM_STR(pg_def)});
		m_MSProgramGroups["Guide.Radiance"] = m_Pipeline.createMissPG({m_Modules["RayGuide"], RTLIB_MISS_PROGRAM_STR(radiance)});
		m_MSProgramGroups["Guide.Occluded"] = m_Pipeline.createMissPG({m_Modules["RayGuide"], RTLIB_MISS_PROGRAM_STR(occluded)});
		m_HGProgramGroups["Guide.Radiance.Diffuse.Guiding.Default"] = m_Pipeline.createHitgroupPG({m_Modules["RayGuide"], RTLIB_CLOSESTHIT_PROGRAM_STR(radiance_for_diffuse_pg_def)}, {}, {});
		m_HGProgramGroups["Guide.Radiance.Phong.Guiding.Default"] = m_Pipeline.createHitgroupPG({m_Modules["RayGuide"], RTLIB_CLOSESTHIT_PROGRAM_STR(radiance_for_phong_pg_def)}, {}, {});
		m_HGProgramGroups["Guide.Radiance.Emission"] = m_Pipeline.createHitgroupPG({m_Modules["RayGuide"], RTLIB_CLOSESTHIT_PROGRAM_STR(radiance_for_emission)}, {}, {});
		m_HGProgramGroups["Guide.Radiance.Specular"] = m_Pipeline.createHitgroupPG({m_Modules["RayGuide"], RTLIB_CLOSESTHIT_PROGRAM_STR(radiance_for_specular)}, {}, {});
		m_HGProgramGroups["Guide.Radiance.Refraction"] = m_Pipeline.createHitgroupPG({m_Modules["RayGuide"], RTLIB_CLOSESTHIT_PROGRAM_STR(radiance_for_refraction)}, {}, {});
		m_HGProgramGroups["Guide.Occluded"] = m_Pipeline.createHitgroupPG({m_Modules["RayGuide"], RTLIB_CLOSESTHIT_PROGRAM_STR(occluded)}, {}, {});
		m_Pipeline.link(guideLinkOptions);
	}
	void InitShaderBindingTable()
	{
		auto tlas = m_ParentApp->GetTLAS();
		auto camera = m_ParentApp->GetCamera();
		auto &materials = m_ParentApp->GetMaterials();
		m_RGRecordBuffer.cpuHandle.resize(1);
		m_RGRecordBuffer.cpuHandle[0] = m_RGProgramGroups["Guide.Guiding.Default"].getSBTRecord<RayGenData>();
		m_RGRecordBuffer.cpuHandle[0].data.eye = camera.getEye();
		auto [u, v, w] = camera.getUVW();
		m_RGRecordBuffer.cpuHandle[0].data.u = u;
		m_RGRecordBuffer.cpuHandle[0].data.v = v;
		m_RGRecordBuffer.cpuHandle[0].data.w = w;
		m_RGRecordBuffer.Upload();
		m_MSRecordBuffers.cpuHandle.resize(RAY_TYPE_COUNT);
		m_MSRecordBuffers.cpuHandle[RAY_TYPE_RADIANCE] = m_MSProgramGroups["Guide.Radiance"].getSBTRecord<MissData>();
		m_MSRecordBuffers.cpuHandle[RAY_TYPE_RADIANCE].data.bgColor = make_float4(m_ParentApp->GetBackGroundLight(), 1.0f);
		m_MSRecordBuffers.cpuHandle[RAY_TYPE_OCCLUSION] = m_MSProgramGroups["Guide.Occluded"].getSBTRecord<MissData>();
		m_MSRecordBuffers.Upload();
		m_HGRecordBuffers.cpuHandle.resize(tlas->GetSbtCount() * RAY_TYPE_COUNT);
		{
			size_t sbtOffset = 0;
			for (auto &instanceSet : tlas->GetInstanceSets())
			{
				for (auto &baseGASHandle : instanceSet->baseGASHandles)
				{
					for (auto &mesh : baseGASHandle->GetMeshes())
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
						for (size_t i = 0; i <  mesh->GetUniqueResource()->materials.size(); ++i)
						{
							auto materialId = mesh->GetUniqueResource()->materials[i];
							auto &material = materials[materialId];
							HitgroupData radianceHgData = {};
							{
								radianceHgData.vertices    = cudaVertexBuffer->GetHandle().getDevicePtr();
								radianceHgData.normals     = cudaNormalBuffer->GetHandle().getDevicePtr();
								radianceHgData.texCoords   = cudaTexCrdBuffer->GetHandle().getDevicePtr();
								radianceHgData.indices     = cudaTriIndBuffer->GetHandle().getDevicePtr();
								radianceHgData.diffuseTex  = m_ParentApp->GetTexture(material.GetString("diffTex")).getHandle();
								radianceHgData.specularTex = m_ParentApp->GetTexture(material.GetString("specTex")).getHandle();
								radianceHgData.emissionTex = m_ParentApp->GetTexture(material.GetString("emitTex")).getHandle();
								radianceHgData.diffuse     = material.GetFloat3As<float3>("diffCol");
								radianceHgData.specular    = material.GetFloat3As<float3>("specCol");
								radianceHgData.emission    = material.GetFloat3As<float3>("emitCol");
								radianceHgData.shinness    = material.GetFloat1("shinness");
								radianceHgData.transmit    = material.GetFloat3As<float3>("tranCol");
								radianceHgData.refrInd     = material.GetFloat1("refrIndx");
							}
							if (material.GetString("name") == "light")
							{
								m_LightHgRecIndex = RAY_TYPE_COUNT * sbtOffset + RAY_TYPE_COUNT * i + RAY_TYPE_RADIANCE;
							}
							std::string typeString = SpecifyMaterialType(material);
							if (typeString == "Phong" || typeString == "Diffuse")
							{
								typeString += ".Guiding.Default";
							}
							m_HGRecordBuffers.cpuHandle[RAY_TYPE_COUNT * sbtOffset + RAY_TYPE_COUNT * i + RAY_TYPE_RADIANCE] = m_HGProgramGroups[std::string("Guide.Radiance.") + typeString].getSBTRecord<HitgroupData>(radianceHgData);
							m_HGRecordBuffers.cpuHandle[RAY_TYPE_COUNT * sbtOffset + RAY_TYPE_COUNT * i + RAY_TYPE_OCCLUSION] = m_HGProgramGroups["Guide.Occluded"].getSBTRecord<HitgroupData>({});
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
	void InitLaunchParams()
	{
		auto tlas = m_ParentApp->GetTLAS();
		m_Params.cpuHandle.resize(1);
		m_Params.cpuHandle[0].gasHandle = tlas->GetHandle();
		m_Params.cpuHandle[0].sdTree = m_ParentApp->GetSTree()->GetGpuHandle();
		m_Params.cpuHandle[0].width = m_ParentApp->GetFbWidth();
		m_Params.cpuHandle[0].height = m_ParentApp->GetFbHeight();
		m_Params.cpuHandle[0].light = m_ParentApp->GetMeshLightList();
		m_Params.cpuHandle[0].accumBuffer = m_ParentApp->GetAccumBuffer().getDevicePtr();
		m_Params.cpuHandle[0].frameBuffer = nullptr;
		m_Params.cpuHandle[0].seedBuffer = m_SeedBuffer.getDevicePtr();
		m_Params.cpuHandle[0].isBuilt = false;
		m_Params.cpuHandle[0].maxTraceDepth = m_ParentApp->GetMaxTraceDepth();
		m_Params.cpuHandle[0].samplePerLaunch = 1;
		m_Params.Upload();
	}
	void OnLaunchBegin(const test::RTTraceConfig &config)
	{
		UserData *pUserData = (UserData *)config.pUserData;
		if (!pUserData)
		{
			return;
		}
		if (pUserData->samplePerAll == 0)
		{
			m_SamplePerAll = 0;
			m_SamplePerTmp = 0;
			m_SampleForBudget = pUserData->sampleForBudget;
			m_SamplePerLaunch = pUserData->samplePerLaunch;
			m_SampleForRemain = ((m_SampleForBudget - 1 + m_SamplePerLaunch) / m_SamplePerLaunch) * m_SamplePerLaunch;
			m_CurIteration = 0;
			m_SampleForPass = 0;
			std::random_device rd;
			std::mt19937 mt(rd());
			std::vector<unsigned int> seedData(config.width * config.height);
			std::generate(std::begin(seedData), std::end(seedData), mt);
			m_SeedBuffer.upload(seedData);
			m_ParentApp->GetSTree()->Clear();
			m_ParentApp->GetSTree()->Upload();
		}
		if (m_SamplePerTmp == 0)
		{
			//CurIteration > 0 -> Reset
			m_SampleForRemain = m_SampleForRemain - m_SampleForPass;
			m_SampleForPass = std::min<uint32_t>(m_SampleForRemain, (1 << m_CurIteration) * m_SamplePerLaunch);
			if (m_SampleForRemain - m_SampleForPass < 2 * m_SampleForPass)
			{
				m_SampleForPass = m_SampleForRemain;
			}
			/*Remain>Pass -> Not Final Iteration*/
			if (m_SampleForRemain > m_SampleForPass)
			{
				m_ParentApp->GetSTree()->Download();
				m_ParentApp->GetSTree()->Reset(m_CurIteration, m_SamplePerLaunch);
				m_ParentApp->GetSTree()->Upload();
			}
		}
		//std::cout << "CurIteration: " << m_CurIteration << " SamplePerTmp: " << m_SamplePerTmp << std::endl;
	}
	void OnLaunchExecute(const test::RTTraceConfig &config)
	{
		UserData *pUserData = (UserData *)config.pUserData;
		if (!pUserData)
		{
			return;
		}
		m_Params.cpuHandle[0].width = config.width;
		m_Params.cpuHandle[0].height = config.height;
		m_Params.cpuHandle[0].sdTree = m_ParentApp->GetSTree()->GetGpuHandle();
		m_Params.cpuHandle[0].accumBuffer = m_ParentApp->GetAccumBuffer().getDevicePtr();
		m_Params.cpuHandle[0].frameBuffer = pUserData->frameBuffer;
		m_Params.cpuHandle[0].seedBuffer = m_SeedBuffer.getDevicePtr();
		m_Params.cpuHandle[0].isBuilt = m_CurIteration  > pUserData->iterationForBuilt;
		m_Params.cpuHandle[0].isFinal = m_SampleForPass >= m_SampleForRemain;
		m_Params.cpuHandle[0].samplePerALL = m_SamplePerAll;
		m_Params.cpuHandle[0].samplePerLaunch = m_SamplePerLaunch;
		cudaMemcpyAsync(m_Params.gpuHandle.getDevicePtr(), &m_Params.cpuHandle[0], sizeof(RayTraceParams), cudaMemcpyHostToDevice, config.stream);
		//cudaMemcpy(m_Params.gpuHandle.getDevicePtr(), &m_Params.cpuHandle[0], sizeof(RayTraceParams), cudaMemcpyHostToDevice);
		m_Pipeline.launch(config.stream, m_Params.gpuHandle.getDevicePtr(), m_ShaderBindingTable, config.width, config.height, config.depth);
		if (config.isSync)
		{
			RTLIB_CU_CHECK(cuStreamSynchronize(config.stream));
		}
		m_Params.cpuHandle[0].samplePerALL += m_SamplePerLaunch;
		m_SamplePerAll = pUserData->samplePerAll = m_Params.cpuHandle[0].samplePerALL;
		m_SamplePerTmp += m_SamplePerLaunch;
	}
	void OnLaunchEnd(const test::RTTraceConfig &config)
	{
		if (m_SamplePerTmp >= m_SampleForPass)
		{
			m_ParentApp->GetSTree()->Download();
			m_ParentApp->GetSTree()->Build();
			m_ParentApp->GetSTree()->Upload();
			m_SamplePerTmp = 0;
			m_CurIteration++;
		}
	}
	void FreePipeline()
	{
		m_ShaderBindingTable = {};
		m_RGRecordBuffer.Reset();
		m_MSRecordBuffers.Reset();
		m_HGRecordBuffers.Reset();
	}
	void FreeShaderBindingTable()
	{
		m_ShaderBindingTable = {};
		m_RGRecordBuffer.Reset();
		m_MSRecordBuffers.Reset();
		m_HGRecordBuffers.Reset();
	}
	void FreeLaunchParams()
	{
		m_Params.Reset();
	}
	void FreeFrameResources()
	{
		m_SeedBuffer.reset();
	}
private:
	TestPG4Application *m_ParentApp = nullptr;
	Pipeline m_Pipeline = {};
	ModuleMap m_Modules = {};
	RGProgramGroupMap m_RGProgramGroups = {};
	MSProgramGroupMap m_MSProgramGroups = {};
	HGProgramGroupMap m_HGProgramGroups = {};
	OptixShaderBindingTable m_ShaderBindingTable = {};
	rtlib::CUDAUploadBuffer<RayGRecord> m_RGRecordBuffer = {};
	rtlib::CUDAUploadBuffer<MissRecord> m_MSRecordBuffers = {};
	rtlib::CUDAUploadBuffer<HitGRecord> m_HGRecordBuffers = {};
	rtlib::CUDAUploadBuffer<RayTraceParams> m_Params = {};
	rtlib::CUDABuffer<unsigned int> m_SeedBuffer = {};
	unsigned int m_LightHgRecIndex = 0;
	unsigned int m_SamplePerAll = 0;
	unsigned int m_SamplePerTmp = 0;
	unsigned int m_SamplePerLaunch = 0;
	unsigned int m_SampleForBudget = 0;
	unsigned int m_SampleForRemain = 0;
	unsigned int m_SampleForPass = 0;
	unsigned int m_CurIteration = 0;
};
// GuideNEETracer
class TestPG4GuideNEETracer : public test::RTTracer
{
public:
	struct UserData
	{
		uchar4 *frameBuffer;
		unsigned int samplePerAll;
		unsigned int samplePerLaunch;
		unsigned int sampleForBudget;
		unsigned int iterationForBuilt;
	};

private:
	using RayGRecord = rtlib::SBTRecord<RayGenData>;
	using MissRecord = rtlib::SBTRecord<MissData>;
	using HitGRecord = rtlib::SBTRecord<HitgroupData>;
	using Pipeline = rtlib::OPXPipeline;
	using ModuleMap = std::unordered_map<std::string, rtlib::OPXModule>;
	using RGProgramGroupMap = std::unordered_map<std::string, rtlib::OPXRaygenPG>;
	using MSProgramGroupMap = std::unordered_map<std::string, rtlib::OPXMissPG>;
	using HGProgramGroupMap = std::unordered_map<std::string, rtlib::OPXHitgroupPG>;

public:
	TestPG4GuideNEETracer(TestPG4Application *app)
	{
		m_ParentApp = app;
	}
	// RTTracer ����Čp������܂���
	virtual void Initialize() override
	{
		this->InitFrameResources();
		this->InitPipeline();
		this->InitShaderBindingTable();
		this->InitLaunchParams();
	}
	virtual void Launch(const test::RTTraceConfig &config) override
	{
		OnLaunchBegin(config);
		OnLaunchExecute(config);
		OnLaunchEnd(config);
	}
	virtual void CleanUp() override
	{
		m_ParentApp = nullptr;
		this->FreePipeline();
		this->FreeShaderBindingTable();
		this->FreeLaunchParams();
		this->FreeFrameResources();
		m_LightHgRecIndex = 0;
	}
	virtual void Update() override
	{
		if (m_ParentApp->IsCameraUpdated() || m_ParentApp->IsFrameResized())
		{
			auto camera = m_ParentApp->GetCamera();
			m_RGRecordBuffer.cpuHandle[0].data.eye = camera.getEye();
			auto [u, v, w] = camera.getUVW();
			m_RGRecordBuffer.cpuHandle[0].data.u = u;
			m_RGRecordBuffer.cpuHandle[0].data.v = v;
			m_RGRecordBuffer.cpuHandle[0].data.w = w;
			m_RGRecordBuffer.Upload();
		}
		if (m_ParentApp->IsCameraUpdated() || m_ParentApp->IsFrameResized() || m_ParentApp->IsFrameFlushed() || m_ParentApp->IsTraceChanged() || m_ParentApp->IsLightUpdated())
		{
			m_Params.cpuHandle[0].maxTraceDepth = m_ParentApp->GetMaxTraceDepth();
			m_Params.cpuHandle[0].samplePerALL = 0;
		}
		bool shouldRegen = ((m_SamplePerAll + m_Params.cpuHandle[0].samplePerLaunch) / 1024 != m_SamplePerAll / 1024);
		if (m_ParentApp->IsFrameResized() || shouldRegen)
		{
			std::cout << "Regen!\n";
			std::vector<unsigned int> seedData(m_ParentApp->GetFbWidth() * m_ParentApp->GetFbHeight());
			std::random_device rd;
			std::mt19937 mt(rd());
			std::generate(seedData.begin(), seedData.end(), mt);
			m_SeedBuffer.resize(m_ParentApp->GetFbWidth() * m_ParentApp->GetFbHeight());
			m_SeedBuffer.upload(seedData);
		}
		if (m_ParentApp->IsLightUpdated())
		{
			auto lightColor = make_float4(m_ParentApp->GetBackGroundLight(), 1.0f);
			m_MSRecordBuffers.cpuHandle[RAY_TYPE_RADIANCE].data.bgColor = lightColor;
			RTLIB_CUDA_CHECK(cudaMemcpy(m_MSRecordBuffers.gpuHandle.getDevicePtr() + RAY_TYPE_RADIANCE, &m_MSRecordBuffers.cpuHandle[RAY_TYPE_RADIANCE], sizeof(MissRecord), cudaMemcpyHostToDevice));
		}
	}
	virtual bool ShouldLock()const noexcept override { return true; }
	virtual ~TestPG4GuideNEETracer() {}

private:
	void InitPipeline()
	{
		auto rayGuidePtxFile = std::ifstream(TEST_TEST_PG_CUDA_PATH "/RayGuide.ptx", std::ios::binary);
		if (!rayGuidePtxFile.is_open())
			throw std::runtime_error("Failed To Load RayGuide.ptx!");
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
			guideLinkOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;
#else
			guideLinkOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
#endif
			guideLinkOptions.maxTraceDepth = 2;
		}
		auto guideModuleOptions = OptixModuleCompileOptions{};
		{
#ifndef NDEBUG
			guideModuleOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;
			guideModuleOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
#else
			guideModuleOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
			guideModuleOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
#endif
		}
		guideModuleOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
		m_Pipeline = m_ParentApp->GetOPXContext()->createPipeline(guideCompileOptions);
		m_Modules["RayGuide"] = m_Pipeline.createModule(rayGuidePtxData, guideModuleOptions);
		m_RGProgramGroups["Guide.Default"] = m_Pipeline.createRaygenPG({m_Modules["RayGuide"], RTLIB_RAYGEN_PROGRAM_STR(def)});
		m_RGProgramGroups["Guide.Guiding.Default"] = m_Pipeline.createRaygenPG({m_Modules["RayGuide"], RTLIB_RAYGEN_PROGRAM_STR(pg_def)});
		m_RGProgramGroups["Guide.Guiding.NEE"]     = m_Pipeline.createRaygenPG({ m_Modules["RayGuide"], RTLIB_RAYGEN_PROGRAM_STR(pg_nee) });
		m_MSProgramGroups["Guide.Radiance"] = m_Pipeline.createMissPG({m_Modules["RayGuide"], RTLIB_MISS_PROGRAM_STR(radiance)});
		m_MSProgramGroups["Guide.Occluded"] = m_Pipeline.createMissPG({m_Modules["RayGuide"], RTLIB_MISS_PROGRAM_STR(occluded)});
		m_HGProgramGroups["Guide.Radiance.Diffuse.Guiding.NEE.NEE"]     = m_Pipeline.createHitgroupPG({ m_Modules["RayGuide"], RTLIB_CLOSESTHIT_PROGRAM_STR(radiance_for_diffuse_pg_nee_with_nee_light)}, {}, {});
		m_HGProgramGroups["Guide.Radiance.Diffuse.Guiding.NEE.Default"] = m_Pipeline.createHitgroupPG({ m_Modules["RayGuide"], RTLIB_CLOSESTHIT_PROGRAM_STR(radiance_for_diffuse_pg_nee_with_def_light) }, {}, {});
		m_HGProgramGroups["Guide.Radiance.Phong.Guiding.NEE.NEE"]       = m_Pipeline.createHitgroupPG({ m_Modules["RayGuide"], RTLIB_CLOSESTHIT_PROGRAM_STR(radiance_for_phong_pg_nee_with_nee_light)}, {}, {});
		m_HGProgramGroups["Guide.Radiance.Phong.Guiding.NEE.Default"]   = m_Pipeline.createHitgroupPG({ m_Modules["RayGuide"], RTLIB_CLOSESTHIT_PROGRAM_STR(radiance_for_phong_pg_nee_with_def_light) }, {}, {});
		m_HGProgramGroups["Guide.Radiance.Emission.NEE"]     = m_Pipeline.createHitgroupPG({ m_Modules["RayGuide"], RTLIB_CLOSESTHIT_PROGRAM_STR(radiance_for_emission_with_nee_light)}, {}, {});
		m_HGProgramGroups["Guide.Radiance.Emission.Default"] = m_Pipeline.createHitgroupPG({ m_Modules["RayGuide"], RTLIB_CLOSESTHIT_PROGRAM_STR(radiance_for_emission_with_def_light) }, {}, {});
		m_HGProgramGroups["Guide.Radiance.Specular"] = m_Pipeline.createHitgroupPG({m_Modules["RayGuide"], RTLIB_CLOSESTHIT_PROGRAM_STR(radiance_for_specular)}, {}, {});
		m_HGProgramGroups["Guide.Radiance.Refraction"] = m_Pipeline.createHitgroupPG({m_Modules["RayGuide"], RTLIB_CLOSESTHIT_PROGRAM_STR(radiance_for_refraction)}, {}, {});
		m_HGProgramGroups["Guide.Occluded"] = m_Pipeline.createHitgroupPG({m_Modules["RayGuide"], RTLIB_CLOSESTHIT_PROGRAM_STR(occluded)}, {}, {});
		m_Pipeline.link(guideLinkOptions);
	}
	void InitFrameResources()
	{
		std::vector<unsigned int> seedData(m_ParentApp->GetFbWidth() * m_ParentApp->GetFbHeight());
		std::random_device rd;
		std::mt19937 mt(rd());
		std::generate(seedData.begin(), seedData.end(), mt);
		m_SeedBuffer = rtlib::CUDABuffer<uint32_t>(seedData);
	}
	void InitShaderBindingTable()
	{
		auto pipelineName = "Guide"s;
		auto tlas = m_ParentApp->GetTLAS();
		auto camera = m_ParentApp->GetCamera();
		auto &materials = m_ParentApp->GetMaterials();
		m_RGRecordBuffer.cpuHandle.resize(1);
		m_RGRecordBuffer.cpuHandle[0] = m_RGProgramGroups[pipelineName + ".Guiding.NEE"].getSBTRecord<RayGenData>();
		m_RGRecordBuffer.cpuHandle[0].data.eye = camera.getEye();
		auto [u, v, w] = camera.getUVW();
		m_RGRecordBuffer.cpuHandle[0].data.u = u;
		m_RGRecordBuffer.cpuHandle[0].data.v = v;
		m_RGRecordBuffer.cpuHandle[0].data.w = w;
		m_RGRecordBuffer.Upload();
		m_MSRecordBuffers.cpuHandle.resize(RAY_TYPE_COUNT);
		m_MSRecordBuffers.cpuHandle[RAY_TYPE_RADIANCE] = m_MSProgramGroups[pipelineName + ".Radiance"].getSBTRecord<MissData>();
		m_MSRecordBuffers.cpuHandle[RAY_TYPE_RADIANCE].data.bgColor = make_float4(m_ParentApp->GetBackGroundLight(), 1.0f);
		m_MSRecordBuffers.cpuHandle[RAY_TYPE_OCCLUSION] = m_MSProgramGroups[pipelineName + ".Occluded"].getSBTRecord<MissData>();
		m_MSRecordBuffers.Upload();
		m_HGRecordBuffers.cpuHandle.resize(tlas->GetSbtCount() * RAY_TYPE_COUNT);
		{
			size_t sbtOffset = 0;
			for (auto &instanceSet : tlas->GetInstanceSets())
			{
				for (auto &baseGASHandle : instanceSet->baseGASHandles)
				{
					for (auto &mesh : baseGASHandle->GetMeshes())
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
							auto &material = materials[materialId];
							HitgroupData radianceHgData = {};
							{
								radianceHgData.vertices = cudaVertexBuffer->GetHandle().getDevicePtr();
								radianceHgData.normals = cudaNormalBuffer->GetHandle().getDevicePtr();
								radianceHgData.texCoords = cudaTexCrdBuffer->GetHandle().getDevicePtr();
								radianceHgData.indices = cudaTriIndBuffer->GetHandle().getDevicePtr();
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
							if (material.GetString("name") == "light")
							{
								m_LightHgRecIndex = RAY_TYPE_COUNT * sbtOffset + RAY_TYPE_COUNT * i + RAY_TYPE_RADIANCE;
							}
							std::string typeString = SpecifyMaterialType(material);
							if (typeString == "Phong" || typeString == "Diffuse")
							{
								typeString += ".Guiding.NEE";
								if (mesh->GetUniqueResource()->variables.GetBool("hasLight") && mesh->GetUniqueResource()->variables.GetBool("useNEE"))
								{
									typeString += ".NEE";
								}
								else
								{
									typeString += ".Default";
								}
							}
							if (typeString == "Emission")
							{
								if (mesh->GetUniqueResource()->variables.GetBool("hasLight") && mesh->GetUniqueResource()->variables.GetBool("useNEE"))
								{
									typeString += ".NEE";
								}
								else
								{
									typeString += ".Default";
								}
							}
							m_HGRecordBuffers.cpuHandle[RAY_TYPE_COUNT * sbtOffset + RAY_TYPE_COUNT * i + RAY_TYPE_RADIANCE] = m_HGProgramGroups[pipelineName + ".Radiance."s + typeString].getSBTRecord<HitgroupData>(radianceHgData);
							m_HGRecordBuffers.cpuHandle[RAY_TYPE_COUNT * sbtOffset + RAY_TYPE_COUNT * i + RAY_TYPE_OCCLUSION] = m_HGProgramGroups[pipelineName + ".Occluded"s].getSBTRecord<HitgroupData>({});
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
	void InitLaunchParams()
	{
		auto tlas = m_ParentApp->GetTLAS();
		m_Params.cpuHandle.resize(1);
		m_Params.cpuHandle[0].gasHandle = tlas->GetHandle();
		m_Params.cpuHandle[0].sdTree = m_ParentApp->GetSTree()->GetGpuHandle();
		m_Params.cpuHandle[0].width = m_ParentApp->GetFbWidth();
		m_Params.cpuHandle[0].height = m_ParentApp->GetFbHeight();
		m_Params.cpuHandle[0].light = m_ParentApp->GetMeshLightList();
		m_Params.cpuHandle[0].accumBuffer = m_ParentApp->GetAccumBuffer().getDevicePtr();
		m_Params.cpuHandle[0].frameBuffer = nullptr;
		m_Params.cpuHandle[0].seedBuffer = m_SeedBuffer.getDevicePtr();
		m_Params.cpuHandle[0].isBuilt = false;
		m_Params.cpuHandle[0].maxTraceDepth = m_ParentApp->GetMaxTraceDepth();
		m_Params.cpuHandle[0].samplePerLaunch = 1;

		m_Params.Upload();
	}
	void OnLaunchBegin(const test::RTTraceConfig &config)
	{
		UserData *pUserData = (UserData *)config.pUserData;
		if (!pUserData)
		{
			return;
		}
		if (pUserData->samplePerAll == 0)
		{
			m_SamplePerAll = 0;
			m_SamplePerTmp = 0;
			m_SampleForBudget = pUserData->sampleForBudget;
			m_SamplePerLaunch = pUserData->samplePerLaunch;
			m_SampleForRemain = ((m_SampleForBudget - 1 + m_SamplePerLaunch) / m_SamplePerLaunch) * m_SamplePerLaunch;
			m_CurIteration = 0;
			m_SampleForPass = 0;
			std::random_device rd;
			std::mt19937 mt(rd());
			std::vector<unsigned int> seedData(config.width * config.height);
			std::generate(std::begin(seedData), std::end(seedData), mt);
			m_SeedBuffer.upload(seedData);
			m_ParentApp->GetSTree()->Clear();
			m_ParentApp->GetSTree()->Upload();
		}
		if (m_SamplePerTmp == 0)
		{
			//CurIteration > 0 -> Reset
			m_SampleForRemain = m_SampleForRemain - m_SampleForPass;
			m_SampleForPass = std::min<uint32_t>(m_SampleForRemain, (1 << m_CurIteration) * m_SamplePerLaunch);
			if (m_SampleForRemain - m_SampleForPass < 2 * m_SampleForPass)
			{
				m_SampleForPass = m_SampleForRemain;
			}
			/*Remain>Pass -> Not Final Iteration*/
			if (m_SampleForRemain > m_SampleForPass)
			{
				m_ParentApp->GetSTree()->Download();
				m_ParentApp->GetSTree()->Reset(m_CurIteration, m_SamplePerLaunch);
				m_ParentApp->GetSTree()->Upload();
			}
		}
		//std::cout << "CurIteration: " << m_CurIteration << " SamplePerTmp: " << m_SamplePerTmp << std::endl;
	}
	void OnLaunchExecute(const test::RTTraceConfig &config)
	{
		UserData *pUserData = (UserData *)config.pUserData;
		if (!pUserData)
		{
			return;
		}
		m_Params.cpuHandle[0].width = config.width;
		m_Params.cpuHandle[0].height = config.height;
		m_Params.cpuHandle[0].sdTree = m_ParentApp->GetSTree()->GetGpuHandle();
		m_Params.cpuHandle[0].accumBuffer = m_ParentApp->GetAccumBuffer().getDevicePtr();
		m_Params.cpuHandle[0].frameBuffer = pUserData->frameBuffer;
		m_Params.cpuHandle[0].seedBuffer = m_SeedBuffer.getDevicePtr();
		m_Params.cpuHandle[0].isBuilt = m_CurIteration > pUserData->iterationForBuilt;
		m_Params.cpuHandle[0].isFinal = m_SampleForPass >= m_SampleForRemain;
		m_Params.cpuHandle[0].samplePerALL = m_SamplePerAll;
		m_Params.cpuHandle[0].samplePerLaunch = m_SamplePerLaunch;
		cudaMemcpyAsync(m_Params.gpuHandle.getDevicePtr(), &m_Params.cpuHandle[0], sizeof(RayTraceParams), cudaMemcpyHostToDevice, config.stream);
		//cudaMemcpy(m_Params.gpuHandle.getDevicePtr(), &m_Params.cpuHandle[0], sizeof(RayTraceParams), cudaMemcpyHostToDevice);
		m_Pipeline.launch(config.stream, m_Params.gpuHandle.getDevicePtr(), m_ShaderBindingTable, config.width, config.height, config.depth);
		if (config.isSync)
		{
			RTLIB_CU_CHECK(cuStreamSynchronize(config.stream));
		}
		m_Params.cpuHandle[0].samplePerALL += m_SamplePerLaunch;
		m_SamplePerAll = pUserData->samplePerAll = m_Params.cpuHandle[0].samplePerALL;
		m_SamplePerTmp += m_SamplePerLaunch;
	}
	void OnLaunchEnd(const test::RTTraceConfig &config)
	{
		if (m_SamplePerTmp >= m_SampleForPass)
		{
			m_ParentApp->GetSTree()->Download();
			m_ParentApp->GetSTree()->Build();
			m_ParentApp->GetSTree()->Upload();
			m_SamplePerTmp = 0;
			m_CurIteration++;
		}
	}
	void FreePipeline()
	{
		m_ShaderBindingTable = {};
		m_RGRecordBuffer.Reset();
		m_MSRecordBuffers.Reset();
		m_HGRecordBuffers.Reset();
	}
	void FreeShaderBindingTable()
	{
		m_ShaderBindingTable = {};
		m_RGRecordBuffer.Reset();
		m_MSRecordBuffers.Reset();
		m_HGRecordBuffers.Reset();
	}
	void FreeLaunchParams()
	{
		m_Params.Reset();
	}
	void FreeFrameResources()
	{
		m_SeedBuffer.reset();
	}
private:
	TestPG4Application *m_ParentApp = nullptr;
	Pipeline m_Pipeline = {};
	ModuleMap m_Modules = {};
	RGProgramGroupMap m_RGProgramGroups = {};
	MSProgramGroupMap m_MSProgramGroups = {};
	HGProgramGroupMap m_HGProgramGroups = {};
	OptixShaderBindingTable m_ShaderBindingTable = {};
	rtlib::CUDAUploadBuffer<RayGRecord> m_RGRecordBuffer = {};
	rtlib::CUDAUploadBuffer<MissRecord> m_MSRecordBuffers = {};
	rtlib::CUDAUploadBuffer<HitGRecord> m_HGRecordBuffers = {};
	rtlib::CUDAUploadBuffer<RayTraceParams> m_Params = {};
	rtlib::CUDABuffer<unsigned int> m_SeedBuffer = {};
	unsigned int m_LightHgRecIndex = 0;
	unsigned int m_SamplePerAll = 0;
	unsigned int m_SamplePerTmp = 0;
	unsigned int m_SamplePerLaunch = 0;
	unsigned int m_SampleForBudget = 0;
	unsigned int m_SampleForRemain = 0;
	unsigned int m_SampleForPass = 0;
	unsigned int m_CurIteration = 0;
};
// DebugTracer
class TestPG4DebugTracer : public test::RTTracer
{
public:
	struct UserData
	{
		uchar4 *diffuseBuffer; //8
		uchar4 *specularBuffer;
		uchar4 *transmitBuffer;
		uchar4 *emissionBuffer;
		uchar4 *texCoordBuffer;
		uchar4 *normalBuffer;
		uchar4 *depthBuffer;
		uchar4 *sTreeColBuffer;
	};
private:
	using RayGRecord = rtlib::SBTRecord<RayGenData>;
	using MissRecord = rtlib::SBTRecord<MissData>;
	using HitGRecord = rtlib::SBTRecord<HitgroupData>;
	using Pipeline = rtlib::OPXPipeline;
	using ModuleMap = std::unordered_map<std::string, rtlib::OPXModule>;
	using RGProgramGroupMap = std::unordered_map<std::string, rtlib::OPXRaygenPG>;
	using MSProgramGroupMap = std::unordered_map<std::string, rtlib::OPXMissPG>;
	using HGProgramGroupMap = std::unordered_map<std::string, rtlib::OPXHitgroupPG>;

public:
	TestPG4DebugTracer(TestPG4Application *app)
	{
		m_ParentApp = app;
	}
	// RTTracer ����Čp������܂���
	virtual void Initialize() override
	{
		this->InitPipeline();
		this->InitShaderBindingTable();
		this->InitLaunchParams();
	}
	virtual void Launch(const test::RTTraceConfig &config) override
	{
		UserData *pUserData = (UserData *)config.pUserData;
		if (!pUserData)
		{
			return;
		}

		m_Params.cpuHandle[0].width = config.width;
		m_Params.cpuHandle[0].height = config.height;
		m_Params.cpuHandle[0].sdTree = m_ParentApp->GetSTree()->GetGpuHandle();
		m_Params.cpuHandle[0].diffuseBuffer = pUserData->diffuseBuffer;
		m_Params.cpuHandle[0].specularBuffer = pUserData->specularBuffer;
		m_Params.cpuHandle[0].emissionBuffer = pUserData->emissionBuffer;
		m_Params.cpuHandle[0].transmitBuffer = pUserData->transmitBuffer;
		m_Params.cpuHandle[0].normalBuffer = pUserData->normalBuffer;
		m_Params.cpuHandle[0].depthBuffer = pUserData->depthBuffer;
		m_Params.cpuHandle[0].texCoordBuffer = pUserData->texCoordBuffer;
		m_Params.cpuHandle[0].sTreeColBuffer = pUserData->sTreeColBuffer;
		cudaMemcpyAsync(m_Params.gpuHandle.getDevicePtr(), &m_Params.cpuHandle[0], sizeof(RayDebugParams), cudaMemcpyHostToDevice, config.stream);
		m_Pipeline.launch(config.stream, m_Params.gpuHandle.getDevicePtr(), m_ShaderBindingTable, config.width, config.height, config.depth);
		if (config.isSync)
		{
			RTLIB_CU_CHECK(cuStreamSynchronize(config.stream));
		}
	}
	virtual void CleanUp() override
	{
		m_ParentApp = nullptr;
		this->FreePipeline();
		this->FreeShaderBindingTable();
		this->FreeLaunchParams();
		m_LightHgRecIndex = 0;
	}
	virtual void Update() override
	{
		if (m_ParentApp->IsCameraUpdated() || m_ParentApp->IsFrameResized())
		{
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
		if (m_ParentApp->IsLightUpdated())
		{
			auto lightColor = make_float4(m_ParentApp->GetBackGroundLight(),1.0f);
			m_MSRecordBuffers.cpuHandle[RAY_TYPE_RADIANCE].data.bgColor = lightColor;
			RTLIB_CUDA_CHECK(cudaMemcpy(m_MSRecordBuffers.gpuHandle.getDevicePtr() + RAY_TYPE_RADIANCE, &m_MSRecordBuffers.cpuHandle[RAY_TYPE_RADIANCE], sizeof(MissRecord), cudaMemcpyHostToDevice));
		}
	}
	virtual ~TestPG4DebugTracer() {}

private:
	void InitPipeline()
	{
		auto rayDebugPtxFile = std::ifstream(TEST_TEST_PG_CUDA_PATH "/RayDebug.ptx", std::ios::binary);
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
		m_Pipeline = m_ParentApp->GetOPXContext()->createPipeline(debugCompileOptions);
		m_Modules["RayDebug"] = m_Pipeline.createModule(rayDebugPtxData, debugModuleOptions);
		m_RGProgramGroups["Debug.Default"] = m_Pipeline.createRaygenPG({m_Modules["RayDebug"], "__raygen__debug"});
		m_MSProgramGroups["Debug.Default"] = m_Pipeline.createMissPG({m_Modules["RayDebug"], "__miss__debug"});
		m_HGProgramGroups["Debug.Default"] = m_Pipeline.createHitgroupPG({m_Modules["RayDebug"], "__closesthit__debug"}, {}, {});
		m_Pipeline.link(debugLinkOptions);
	}
	void InitShaderBindingTable()
	{
		auto tlas = m_ParentApp->GetTLAS();
		auto camera = m_ParentApp->GetCamera();
		auto &materials = m_ParentApp->GetMaterials();
		m_RGRecordBuffer.cpuHandle.resize(1);
		m_RGRecordBuffer.cpuHandle[0] = m_RGProgramGroups["Debug.Default"].getSBTRecord<RayGenData>();
		m_RGRecordBuffer.cpuHandle[0].data.eye = camera.getEye();
		auto [u, v, w] = camera.getUVW();
		m_RGRecordBuffer.cpuHandle[0].data.u = u;
		m_RGRecordBuffer.cpuHandle[0].data.v = v;
		m_RGRecordBuffer.cpuHandle[0].data.w = w;
		m_RGRecordBuffer.Upload();
		m_MSRecordBuffers.cpuHandle.resize(RAY_TYPE_COUNT);
		m_MSRecordBuffers.cpuHandle[RAY_TYPE_RADIANCE] = m_MSProgramGroups["Debug.Default"].getSBTRecord<MissData>();
		m_MSRecordBuffers.cpuHandle[RAY_TYPE_RADIANCE].data.bgColor = make_float4(m_ParentApp->GetBackGroundLight(), 1.0f);
		m_MSRecordBuffers.cpuHandle[RAY_TYPE_OCCLUSION] = m_MSProgramGroups["Debug.Default"].getSBTRecord<MissData>();
		m_MSRecordBuffers.Upload();
		m_HGRecordBuffers.cpuHandle.resize(tlas->GetSbtCount() * RAY_TYPE_COUNT);
		{
			size_t sbtOffset = 0;
			for (auto &instanceSet : tlas->GetInstanceSets())
			{
				for (auto &baseGASHandle : instanceSet->baseGASHandles)
				{
					for (auto &mesh : baseGASHandle->GetMeshes())
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
							auto &material = materials[materialId];
							HitgroupData radianceHgData = {};
							{
								radianceHgData.vertices = cudaVertexBuffer->GetHandle().getDevicePtr();
								radianceHgData.normals = cudaNormalBuffer->GetHandle().getDevicePtr();
								radianceHgData.texCoords = cudaTexCrdBuffer->GetHandle().getDevicePtr();
								radianceHgData.indices = cudaTriIndBuffer->GetHandle().getDevicePtr();
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
							if (material.GetString("name") == "light")
							{
								m_LightHgRecIndex = RAY_TYPE_COUNT * sbtOffset + RAY_TYPE_COUNT * i + RAY_TYPE_RADIANCE;
							}
							m_HGRecordBuffers.cpuHandle[RAY_TYPE_COUNT * sbtOffset + RAY_TYPE_COUNT * i + RAY_TYPE_RADIANCE] = m_HGProgramGroups["Debug.Default"].getSBTRecord<HitgroupData>(radianceHgData);
							m_HGRecordBuffers.cpuHandle[RAY_TYPE_COUNT * sbtOffset + RAY_TYPE_COUNT * i + RAY_TYPE_OCCLUSION] = m_HGProgramGroups["Debug.Default"].getSBTRecord<HitgroupData>({});
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
	void InitLaunchParams()
	{
		auto tlas = m_ParentApp->GetTLAS();
		m_Params.cpuHandle.resize(1);
		m_Params.cpuHandle[0].gasHandle = tlas->GetHandle();
		m_Params.cpuHandle[0].sdTree = m_ParentApp->GetSTree()->GetGpuHandle();
		m_Params.cpuHandle[0].width = m_ParentApp->GetFbWidth();
		m_Params.cpuHandle[0].height = m_ParentApp->GetFbHeight();
		m_Params.cpuHandle[0].light = m_ParentApp->GetMeshLightList();
		m_Params.cpuHandle[0].diffuseBuffer = nullptr;
		m_Params.cpuHandle[0].specularBuffer = nullptr;
		m_Params.cpuHandle[0].emissionBuffer = nullptr;
		m_Params.cpuHandle[0].transmitBuffer = nullptr;
		m_Params.cpuHandle[0].normalBuffer = nullptr;
		m_Params.cpuHandle[0].depthBuffer = nullptr;
		m_Params.cpuHandle[0].texCoordBuffer = nullptr;
		m_Params.cpuHandle[0].sTreeColBuffer = nullptr;
		m_Params.Upload();
	}
	void FreePipeline()
	{
		m_ShaderBindingTable = {};
		m_RGRecordBuffer.Reset();
		m_MSRecordBuffers.Reset();
		m_HGRecordBuffers.Reset();
	}
	void FreeShaderBindingTable()
	{
		m_ShaderBindingTable = {};
		m_RGRecordBuffer.Reset();
		m_MSRecordBuffers.Reset();
		m_HGRecordBuffers.Reset();
	}
	void FreeLaunchParams()
	{
		m_Params.Reset();
	}
private:
	TestPG4Application *	m_ParentApp       = nullptr;
	Pipeline				m_Pipeline        = {};
	ModuleMap				m_Modules         = {};
	RGProgramGroupMap		m_RGProgramGroups = {};
	MSProgramGroupMap		m_MSProgramGroups = {};
	HGProgramGroupMap		m_HGProgramGroups = {};
	OptixShaderBindingTable m_ShaderBindingTable = {};
	rtlib::CUDAUploadBuffer<RayGRecord> m_RGRecordBuffer = {};
	rtlib::CUDAUploadBuffer<MissRecord> m_MSRecordBuffers = {};
	rtlib::CUDAUploadBuffer<HitGRecord> m_HGRecordBuffers = {};
	rtlib::CUDAUploadBuffer<RayDebugParams> m_Params = {};
	unsigned int m_LightHgRecIndex = 0;
};
// GLFW
void TestPG4Application::InitGLFW()
{
	if (glfwInit() == GLFW_FALSE)
	{
		throw std::runtime_error("Failed To Init GLFW!");
	}
}
// Window
void TestPG4Application::InitWindow()
{
	glfwWindowHint(GLFW_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 4);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
	m_Window = glfwCreateWindow(m_FbWidth, m_FbHeight, m_Title, nullptr, nullptr);
	if (!m_Window)
	{
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
// GLAD
void TestPG4Application::InitGLAD()
{
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		throw std::runtime_error("Failed To Init GLAD!");
	}
}
// CUDA
void TestPG4Application::InitCUDA()
{
	RTLIB_CUDA_CHECK(cudaFree(0));
	RTLIB_OPTIX_CHECK(optixInit());
	m_Context = std::make_shared<rtlib::OPXContext>(rtlib::OPXContext::Desc{0, 0, OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL, 4});
}
// Gui
void TestPG4Application::InitGui()
{
	//Renderer
	m_Renderer = std::make_unique<rtlib::ext::RectRenderer>();
	m_Renderer->init();
	//ImGui
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO &io = ImGui::GetIO();
	(void)io;
	ImGui::StyleColorsDark();
	if (!ImGui_ImplGlfw_InitForOpenGL(m_Window, false))
	{
		throw std::runtime_error("Failed To Init ImGui For GLFW + OpenGL!");
	}
	int major = glfwGetWindowAttrib(m_Window, GLFW_CONTEXT_VERSION_MAJOR);
	int minor = glfwGetWindowAttrib(m_Window, GLFW_CONTEXT_VERSION_MINOR);
	std::string glslVersion = std::string("#version ") + std::to_string(major) + std::to_string(minor) + "0 core";
	if (!ImGui_ImplOpenGL3_Init(glslVersion.c_str()))
	{
		throw std::runtime_error("Failed To Init ImGui For GLFW3");
	}
}
// Assets
void TestPG4Application::InitAssets()
{
	auto objModelPathes = std::vector< std::filesystem::path>{
		//std::filesystem::canonical(std::filesystem::path(TEST_TEST_PG_DATA_PATH "/Models/Lumberyard/Exterior/exterior.obj")),
		//std::filesystem::canonical(std::filesystem::path(TEST_TEST_PG_DATA_PATH "/Models/Lumberyard/Interior/interior.obj"))
		//std::filesystem::canonical(std::filesystem::path(TEST_TEST_PG_DATA_PATH "/Models/CornellBox/CornellBox-Water.obj"))
		//std::filesystem::canonical(std::filesystem::path(TEST_TEST_PG_DATA_PATH "/Models/CornellBox/CornellBox-Original.obj"))
		//std::filesystem::canonical(std::filesystem::path(TEST_TEST_PG_DATA_PATH "/Models/Sponza/Sponza.obj"))
	};
	for (const std::filesystem::directory_entry entry : std::filesystem::directory_iterator(std::filesystem::path(TEST_TEST_PG_DATA_PATH "/Models/Pool/")))
	{
		if (std::filesystem::is_directory(entry.path())) {
			for (const std::filesystem::directory_entry& entry1 : std::filesystem::directory_iterator(entry.path()))
			{
				if (entry1.path().extension() == ".obj")
				{
					objModelPathes.push_back(std::filesystem::canonical(entry1.path()));
				}
			}
		}
	}
	for (auto objModelPath : objModelPathes)
	{
		if (!m_ObjModelAssets.LoadAsset(objModelPath.filename().replace_extension().string(), objModelPath.string()))
		{
			throw std::runtime_error("Failed To Load Obj Model!");
		}
	}
	{
		auto aabb = rtlib::utils::AABB();
		for (auto& [name, objModelAsset] : m_ObjModelAssets.GetAssets())
		{
			for (auto& position : objModelAsset.meshGroup->GetSharedResource()->vertexBuffer) {
				aabb.Update(position);
			}
		}
		/*8点*/
		auto lightBox = rtlib::utils::Box();
		lightBox.x0   = aabb.min.x - 100.0f;
		lightBox.x1   = aabb.max.x + 100.0f;
		lightBox.y0   = aabb.min.y - 100.0f;
		lightBox.y1   = aabb.max.y + 100.0f;
		lightBox.z0   = aabb.min.z - 100.0f;
		lightBox.z1   = aabb.max.z + 100.0f;
		auto vertices = lightBox.getVertices();
		auto indices  = lightBox.getIndices();
		auto lightMeshGroup = rtlib::ext::MeshGroup::New();
		auto lightSharedRes = rtlib::ext::MeshSharedResource::New();
		lightSharedRes->vertexBuffer.Resize(vertices.size());
		std::memcpy(lightSharedRes->vertexBuffer.GetCpuData(), vertices.data(), sizeof(float3) * vertices.size());
		lightSharedRes->texCrdBuffer.Resize(vertices.size());
		for (auto& texCrd : lightSharedRes->texCrdBuffer) { texCrd = { 0.5f,0.5f }; }
		lightSharedRes->normalBuffer.Resize(vertices.size());
		for (auto& normal : lightSharedRes->normalBuffer) { normal = { 0.0f,0.0f,0.0f }; }
		auto lightUniqueRes = rtlib::ext::MeshUniqueResource::New();
		lightUniqueRes->triIndBuffer.Resize(indices.size());
		std::memcpy(lightUniqueRes->triIndBuffer.GetCpuData(), indices.data(),   sizeof(uint3) * indices.size());
		lightUniqueRes->matIndBuffer.Resize(indices.size());
		for (auto& matInd : lightUniqueRes->matIndBuffer) { matInd = 0; }
		lightUniqueRes->materials = std::vector<uint32_t>({ 0 });
		lightUniqueRes->variables.SetBool("hasLight", true);
		auto materials = std::vector<rtlib::ext::VariableMap>();
		materials.resize(1);
		materials[0].SetString("name", "light");
		materials[0].SetUInt32("illum", 2);
		materials[0].SetFloat3("diffCol",
		{
				0.0f,0.0f,0.0f
		});
		materials[0].SetFloat3("specCol",
			{
					0.0f,0.0f,0.0f
			});
		materials[0].SetFloat3("tranCol",
			{
					0.0f,0.0f,0.0f
			});
		materials[0].SetFloat3("emitCol",
			{
					1.0f,1.0f,1.0f
			});
		materials[0].SetString("diffTex", "");
		materials[0].SetString("specTex", "");
		materials[0].SetString("emitTex", "");
		materials[0].SetString("shinTex", "");
		materials[0].SetFloat1("shinness", 10.0f);
		materials[0].SetFloat1("refrIndx", 1.0f);
		lightMeshGroup->SetSharedResource(lightSharedRes);
		lightMeshGroup->SetUniqueResource("light", lightUniqueRes);
		auto objModel = test::RTObjModel();
		objModel.materials = std::move(materials);
		objModel.meshGroup = lightMeshGroup;
		m_ObjModelAssets.GetAssets()["light"] = objModel;
	}
	auto smpTexPath = std::filesystem::canonical(std::filesystem::path(TEST_TEST_PG_DATA_PATH "/Textures/white.png"));
	if (!m_TextureAssets.LoadAsset("", smpTexPath.string()))
	{
		throw std::runtime_error("Failed To Load White Texture!");
	}
	for (auto &[name, objModel] : m_ObjModelAssets.GetAssets())
	{
		for (auto &material : objModel.materials)
		{
			auto diffTexPath = material.GetString("diffTex");
			auto specTexPath = material.GetString("specTex");
			auto emitTexPath = material.GetString("emitTex");
			auto shinTexPath = material.GetString("shinTex");
			if (diffTexPath != "")
			{
				if (!m_TextureAssets.LoadAsset(diffTexPath, diffTexPath))
				{
					std::cout << "DiffTex \"" << diffTexPath << "\" Not Found!\n";
					material.SetString("diffTex", "");
				}
			}
			if (specTexPath != "")
			{
				if (!m_TextureAssets.LoadAsset(specTexPath, specTexPath))
				{
					std::cout << "SpecTex \"" << specTexPath << "\" Not Found!\n";
					material.SetString("specTex", "");
				}
			}
			if (emitTexPath != "")
			{
				if (!m_TextureAssets.LoadAsset(emitTexPath, emitTexPath))
				{
					std::cout << "EmitTex \"" << emitTexPath << "\" Not Found!\n";
					material.SetString("emitTex", "");
				}
				else {
					if (material.GetFloat3("emitCol")[0] == 0.0f &&
						material.GetFloat3("emitCol")[1] == 0.0f &&
						material.GetFloat3("emitCol")[2] == 0.0f) {
						material.SetFloat3("emitCol",{ 1.0f,1.0f,1.0f });
					}
				}
				
			}
			if (shinTexPath != "")
			{
				if (!m_TextureAssets.LoadAsset(shinTexPath, shinTexPath))
				{
					std::cout << "ShinTex \"" << shinTexPath << "\" Not Found!\n";
					material.SetString("shinTex", "");
				}
			}
		}
	}

}
// Acceleration Structure
void TestPG4Application::InitAccelerationStructures()
{
	{
		size_t materialSize = 0;
		for (auto &[name, objModel] : m_ObjModelAssets.GetAssets())
		{
			materialSize += objModel.materials.size();
		}
		m_Materials.resize(materialSize + 1);
		size_t materialOffset = 0;
		for (auto &[name, objModel] : m_ObjModelAssets.GetAssets())
		{
			auto &materials = objModel.materials;
			std::copy(std::begin(materials), std::end(materials), m_Materials.begin() + materialOffset);
			materialOffset += materials.size();
		}
	}

	m_GASHandles["World"] = std::make_shared<rtlib::ext::GASHandle>();
	m_GASHandles["Light"] = std::make_shared<rtlib::ext::GASHandle>();
	{
		OptixAccelBuildOptions accelBuildOptions = {};
		accelBuildOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
		accelBuildOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

		{
			size_t materialOffset = 0;
			for (auto &[name, objModel] : m_ObjModelAssets.GetAssets())
			{
				if (!objModel.meshGroup->GetSharedResource()->vertexBuffer.HasGpuComponent("CUDA"))
				{
					objModel.meshGroup->GetSharedResource()->vertexBuffer.AddGpuComponent<rtlib::ext::resources::CUDABufferComponent<float3>>("CUDA");
				}
				if (!objModel.meshGroup->GetSharedResource()->normalBuffer.HasGpuComponent("CUDA"))
				{
					objModel.meshGroup->GetSharedResource()->normalBuffer.AddGpuComponent<rtlib::ext::resources::CUDABufferComponent<float3>>("CUDA");
				}
				if (!objModel.meshGroup->GetSharedResource()->texCrdBuffer.HasGpuComponent("CUDA"))
				{
					objModel.meshGroup->GetSharedResource()->texCrdBuffer.AddGpuComponent<rtlib::ext::resources::CUDABufferComponent<float2>>("CUDA");
				}
				for (auto &[name, meshUniqueResource] : objModel.meshGroup->GetUniqueResources())
				{
					if (!meshUniqueResource->matIndBuffer.HasGpuComponent("CUDA"))
					{
						meshUniqueResource->matIndBuffer.AddGpuComponent<rtlib::ext::resources::CUDABufferComponent<uint32_t>>("CUDA");
					}
					if (!meshUniqueResource->triIndBuffer.HasGpuComponent("CUDA"))
					{
						meshUniqueResource->triIndBuffer.AddGpuComponent<rtlib::ext::resources::CUDABufferComponent<uint3>>("CUDA");
					}

					auto mesh = rtlib::ext::Mesh::New();
					mesh->SetUniqueResource(meshUniqueResource);
					mesh->SetSharedResource(objModel.meshGroup->GetSharedResource());
					
					for (auto &matIdx : mesh->GetUniqueResource()->materials)
					{
						matIdx += materialOffset;
					}
					if (mesh->GetUniqueResource()->variables.GetBool("hasLight"))
					{
						m_GASHandles["Light"]->AddMesh(mesh);
					}
					else
					{
						m_GASHandles["World"]->AddMesh(mesh);
					}
				}
				materialOffset += objModel.materials.size();
			}
		}
		m_GASHandles["World"]->Build(m_Context.get(), accelBuildOptions);
		m_GASHandles["Light"]->Build(m_Context.get(), accelBuildOptions);
	}
	m_IASHandles["TopLevel"] = std::make_shared<rtlib::ext::IASHandle>();
	{
		OptixAccelBuildOptions accelOptions = {};
		accelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
		accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
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
// Light
void TestPG4Application::InitLight()
{
	auto ChooseNEE = [](const rtlib::ext::MeshPtr& mesh)->bool {
		return !(mesh->GetUniqueResource()->triIndBuffer.Size() < 200 || mesh->GetUniqueResource()->triIndBuffer.Size() > 230);
	};
	auto lightGASHandle = m_GASHandles["Light"];
	for (auto& mesh : lightGASHandle->GetMeshes())
	{
		//Select NEE Light
		if (ChooseNEE(mesh)) {
			mesh->GetUniqueResource()->variables.SetBool("useNEE", false);
		}
		else {
			mesh->GetUniqueResource()->variables.SetBool("useNEE", true);
			std::cout << "Name: " << mesh->GetUniqueResource()->name << " LightCount: " << mesh->GetUniqueResource()->triIndBuffer.Size() << std::endl;
			MeshLight meshLight   = {};
			meshLight.emission    = m_Materials[mesh->GetUniqueResource()->materials[0]].GetFloat3As<float3>("emitCol");
			meshLight.emissionTex = m_TextureAssets.GetAsset(m_Materials[mesh->GetUniqueResource()->materials[0]].GetString("emitTex")).getHandle();
			meshLight.vertices    = mesh->GetSharedResource()->vertexBuffer.GetGpuComponent<rtlib::ext::resources::CUDABufferComponent<float3>>("CUDA")->GetHandle().getDevicePtr();
			meshLight.normals     = mesh->GetSharedResource()->normalBuffer.GetGpuComponent<rtlib::ext::resources::CUDABufferComponent<float3>>("CUDA")->GetHandle().getDevicePtr();
			meshLight.texCoords   = mesh->GetSharedResource()->texCrdBuffer.GetGpuComponent<rtlib::ext::resources::CUDABufferComponent<float2>>("CUDA")->GetHandle().getDevicePtr();
			meshLight.indices     = mesh->GetUniqueResource()->triIndBuffer.GetGpuComponent<rtlib::ext::resources::CUDABufferComponent<uint3>>("CUDA")->GetHandle().getDevicePtr();
			meshLight.indCount    = mesh->GetUniqueResource()->triIndBuffer.Size();
			m_MeshLights.cpuHandle.push_back(meshLight);
		}
	}
	m_MeshLights.Alloc();
	m_MeshLights.Upload();
}
// Camera
void TestPG4Application::InitCamera()
{
	m_CameraController = rtlib::ext::CameraController({0.0f, 1.0f, 5.0f});
	m_MouseSensitity = 0.125f;
	m_MovementSpeed  = 10.0f;
	m_CameraController.SetMouseSensitivity(m_MouseSensitity);
	m_CameraController.SetMovementSpeed(m_MovementSpeed);
}
// STree
void TestPG4Application::InitSTree()
{
	auto worldAABB = rtlib::utils::AABB();

	for (auto &mesh : m_GASHandles["World"]->GetMeshes())
	{
		for (auto &index : mesh->GetUniqueResource()->triIndBuffer)
		{
			worldAABB.Update(mesh->GetSharedResource()->vertexBuffer[index.x]);
			worldAABB.Update(mesh->GetSharedResource()->vertexBuffer[index.y]);
			worldAABB.Update(mesh->GetSharedResource()->vertexBuffer[index.z]);
		}
	}
	m_STree = std::make_shared<test::RTSTreeWrapper>(worldAABB.min, worldAABB.max, 20);
	m_STree->Upload();
}
// FrameResources
void TestPG4Application::InitFrameResources()
{
	//Render
	m_RenderTexture = std::make_unique<rtlib::GLTexture2D<uchar4>>();
	m_RenderTexture->allocate({(size_t)m_FbWidth, (size_t)m_FbHeight, nullptr}, GL_TEXTURE_2D);
	m_RenderTexture->setParameteri(GL_TEXTURE_MAG_FILTER, GL_LINEAR, false);
	m_RenderTexture->setParameteri(GL_TEXTURE_MIN_FILTER, GL_LINEAR, false);
	m_RenderTexture->setParameteri(GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE, false);
	m_RenderTexture->setParameteri(GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE, false);
	//Debug
	m_DebugTexture = std::make_unique<rtlib::GLTexture2D<uchar4>>();
	m_DebugTexture->allocate({(size_t)m_FbWidth, (size_t)m_FbHeight, nullptr}, GL_TEXTURE_2D);
	m_DebugTexture->setParameteri(GL_TEXTURE_MAG_FILTER, GL_LINEAR, false);
	m_DebugTexture->setParameteri(GL_TEXTURE_MIN_FILTER, GL_LINEAR, false);
	m_DebugTexture->setParameteri(GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE, false);
	m_DebugTexture->setParameteri(GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE, false);
	//Frame
	m_AccumBuffer = rtlib::CUDABuffer<float3>(std::vector<float3>(m_FbWidth*m_FbHeight));
	m_FrameBuffer = std::make_unique<test::RTFrameBuffer>(m_FbWidth, m_FbHeight);
	m_FrameBuffer->AddCUGLBuffer("Default");
	m_FrameBuffer->GetCUGLBuffer("Default").upload(
		std::vector<uchar4>(m_FbWidth * m_FbHeight, make_uchar4(0, 0, 0, 255))
	);
	//Debug
	for (auto &debugFrameName : kDebugFrameNames)
	{
		m_FrameBuffer->AddCUGLBuffer(debugFrameName);
		m_FrameBuffer->GetCUGLBuffer(debugFrameName).upload(std::vector<uchar4>(m_FbWidth * m_FbHeight, make_uchar4(0, 0, 0, 255)));
	}
}
// Tracers
void TestPG4Application::InitTracers()
{
	//Simple
	m_SimpleActor = std::shared_ptr<test::RTTracer>(
		new TestPG4SimpleTracer(this));
	m_SimpleActor->Initialize();
	//Simple_NEE
	m_SimpleNEEActor = std::shared_ptr<test::RTTracer>(
		new TestPG4SimpleNEETracer(this));
	m_SimpleNEEActor->Initialize();
	//Guide
	m_GuideActor = std::shared_ptr<test::RTTracer>(
		new TestPG4GuideTracer(this));
	m_GuideActor->Initialize();
	//Guide_NEE
	m_GuideNEEActor = std::shared_ptr<test::RTTracer>(
		new TestPG4GuideNEETracer(this));
	m_GuideNEEActor->Initialize();
	//Debug
	m_DebugActor = std::shared_ptr<test::RTTracer>(
		new TestPG4DebugTracer(this));
	m_DebugActor->Initialize();
}
// Loop: Prepare
void TestPG4Application::InitTimer()
{
	m_CurFrameTime = glfwGetTime();
}
// Loop: Quit
bool TestPG4Application::QuitLoop()
{
	if (!m_Window)
		return false;
	return glfwWindowShouldClose(m_Window);
}
// Trace
void TestPG4Application::Trace()
{
	if (m_LaunchDebug)
	{
		//TestPG4
		TestPG4DebugTracer::UserData userData = {};
		userData.diffuseBuffer  = m_FrameBuffer->GetCUGLBuffer("Diffuse").map();
		userData.specularBuffer = m_FrameBuffer->GetCUGLBuffer("Specular").map();
		userData.emissionBuffer = m_FrameBuffer->GetCUGLBuffer("Emission").map();
		userData.transmitBuffer = m_FrameBuffer->GetCUGLBuffer("Transmit").map();
		userData.normalBuffer   = m_FrameBuffer->GetCUGLBuffer("Normal").map();
		userData.texCoordBuffer = m_FrameBuffer->GetCUGLBuffer("TexCoord").map();
		userData.depthBuffer    = m_FrameBuffer->GetCUGLBuffer("Depth").map();
		userData.sTreeColBuffer = m_FrameBuffer->GetCUGLBuffer("STree").map();

		test::RTTraceConfig traceConfig = {};
		traceConfig.width = m_FbWidth;
		traceConfig.height = m_FbHeight;
		traceConfig.depth = 1;
		traceConfig.isSync = true;
		traceConfig.pUserData = &userData;
		m_DebugActor->Launch(traceConfig);
		m_SampleForPrvDbg = m_SamplePerALL;

		m_FrameBuffer->GetCUGLBuffer("Diffuse").unmap();
		m_FrameBuffer->GetCUGLBuffer("Specular").unmap();
		m_FrameBuffer->GetCUGLBuffer("Emission").unmap();
		m_FrameBuffer->GetCUGLBuffer("Transmit").unmap();
		m_FrameBuffer->GetCUGLBuffer("Normal").unmap();
		m_FrameBuffer->GetCUGLBuffer("TexCoord").unmap();
		m_FrameBuffer->GetCUGLBuffer("Depth").unmap();
		m_FrameBuffer->GetCUGLBuffer("STree").unmap();
	}
	auto beginTraceTime = glfwGetTime();
	{
		test::RTTraceConfig traceConfig = {};
		traceConfig.width  = m_FbWidth;
		traceConfig.height = m_FbHeight;
		traceConfig.depth  = 1;
		traceConfig.isSync = true;

		auto curActor    = std::shared_ptr<test::RTTracer>{};
		auto frameBuffer = m_FrameBuffer->GetCUGLBuffer("Default").map();
		if (!m_TraceGuide)
		{
			if (!m_TraceNEE)
			{
				TestPG4SimpleTracer::UserData userData = {};
				userData.frameBuffer     = frameBuffer;
				userData.samplePerLaunch = m_SamplePerLaunch;
				userData.samplePerAll    = m_SamplePerALL;
				traceConfig.pUserData    = &userData;
				m_SimpleActor->Launch(traceConfig);
				m_SamplePerALL = userData.samplePerAll;
				curActor = m_SimpleActor;
			}
			else
			{
				TestPG4SimpleNEETracer::UserData userData = {};
				userData.frameBuffer     = frameBuffer;
				userData.samplePerLaunch = m_SamplePerLaunch;
				userData.samplePerAll    = m_SamplePerALL;
				traceConfig.pUserData    = &userData;
				m_SimpleNEEActor->Launch(traceConfig);
				m_SamplePerALL = userData.samplePerAll;
				curActor = m_SimpleNEEActor;
			}
		}
		else
		{
			if (!m_TraceNEE)
			{
				TestPG4GuideTracer::UserData userData = {};
				userData.frameBuffer       = frameBuffer;
				userData.samplePerLaunch   = m_SamplePerLaunch;
				userData.samplePerAll      = m_SamplePerALL;
				userData.sampleForBudget   = m_SamplePerBudget;
				userData.iterationForBuilt = 0;
				traceConfig.pUserData      = &userData;
				m_GuideActor->Launch(traceConfig);
				m_SamplePerALL = userData.samplePerAll;
				curActor = m_GuideActor;
			}
			else
			{
				TestPG4GuideNEETracer::UserData userData = {};
				userData.frameBuffer       = frameBuffer;
				userData.samplePerLaunch   = m_SamplePerLaunch;
				userData.samplePerAll      = m_SamplePerALL;
				userData.sampleForBudget   = m_SamplePerBudget;
				userData.iterationForBuilt = 0;
				traceConfig.pUserData      = &userData;
				m_GuideNEEActor->Launch(traceConfig);
				m_SamplePerALL             = userData.samplePerAll;
				curActor = m_GuideNEEActor;
			}
		}
		if (curActor->ShouldLock())
		{
			if (m_SamplePerALL >= m_SamplePerBudget)
			{
				UnLockUpdate();
				m_TraceGuide = false;
				m_FlushFrame = true;
			}
			else
			{
				LockUpdate();
			}
		}
		m_FrameBuffer->GetCUGLBuffer("Default").unmap();
	}
	m_DelTraceTime = glfwGetTime() - beginTraceTime;
	m_CurTraceTime+= m_DelTraceTime;
}
// DrawFrame
void TestPG4Application::DrawFrame()
{
	m_DebugTexture->upload(0, m_FrameBuffer->GetCUGLBuffer(m_CurDebugFrame).getHandle(), 0, 0, m_FbWidth, m_FbHeight);
	m_RenderTexture->upload(0, m_FrameBuffer->GetCUGLBuffer(m_CurRenderFrame).getHandle(), 0, 0, m_FbWidth, m_FbHeight);
	m_Renderer->draw(m_RenderTexture->getID());
}
// DrawGui
void TestPG4Application::DrawGui()
{
	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();

	ImGui::PushStyleColor(ImGuiCol_TitleBgActive, ImVec4(0.0f, 0.7f, 0.2f, 1.0f));
	ImGui::PushStyleColor(ImGuiCol_TitleBg, ImVec4(0.0f, 0.3f, 0.1f, 1.0f));

	ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Once);
	ImGui::SetNextWindowSize(ImVec2(500, 500), ImGuiCond_Once);

	ImGui::Begin("GlobalConfig", nullptr, ImGuiWindowFlags_MenuBar);
	{
		float lightColor[3]    = {m_BgLightColor.x, m_BgLightColor.y, m_BgLightColor.z};
		float cameraFovY       = m_CameraFovY;
		float movementSpeed    = m_MovementSpeed;
		float mouseSensitivity = m_MouseSensitity;
		if (!m_LockUpdate)
		{
			if (ImGui::SliderFloat3("LightColor", lightColor, 0.0f, 10.0f))
			{
				m_BgLightColor = make_float3(lightColor[0], lightColor[1], lightColor[2]);
				m_UpdateLight = true;
			}
			if (ImGui::SliderFloat("CameraFovY", &cameraFovY, -90.0f, 90.0f))
			{
				m_CameraFovY = cameraFovY;
				m_UpdateCamera = true;
			}
			if (ImGui::SliderFloat("MovementSpeed", &movementSpeed, 0.0f, 100.0f))
			{
				m_MovementSpeed = movementSpeed;
				m_CameraController.SetMovementSpeed(movementSpeed);
			}
			if (ImGui::SliderFloat("Sensitivity", &mouseSensitivity, 0.0625, 0.1875))
			{
				m_MouseSensitity = mouseSensitivity;
				m_CameraController.SetMouseSensitivity(mouseSensitivity);
			}
		}
		else
		{
			ImGui::Text("   LightColor: (%f, %f %f)", lightColor[0], lightColor[1], lightColor[2]);
			ImGui::Text("   CameraFovY: %f", cameraFovY);
			ImGui::Text("MovementSpeed: %f", movementSpeed);
			ImGui::Text("  Sensitivity: %f", mouseSensitivity);
		}
		{
			if (ImGui::InputText("SettingDir", m_GlobalSettingPath.data(), m_GlobalSettingPath.size()))
			{
			}
			if (!m_LockUpdate)
			{
				if (ImGui::Button("Load"))
				{
					std::filesystem::path filePath = std::string(m_GlobalSettingPath.data());
					std::ifstream configFile(filePath);
					if (!configFile.fail())
					{
						auto configJson = nlohmann::json();
						configFile >> configJson;
						m_FbWidth = configJson["fbWidth"];
						m_FbHeight = configJson["fbHeight"];
						m_MaxTraceDepth = configJson["maxTraceDepth"];
						m_MovementSpeed = configJson["movementSpeed"];
						m_MouseSensitity = configJson["mouseSensitivity"];
						auto eye = make_float3(
							configJson["camera"]["eye"][0].get<float>(),
							configJson["camera"]["eye"][1].get<float>(),
							configJson["camera"]["eye"][2].get<float>());
						auto lookAt = make_float3(
							configJson["camera"]["lookAt"][0].get<float>(),
							configJson["camera"]["lookAt"][1].get<float>(),
							configJson["camera"]["lookAt"][2].get<float>());
						auto vUp = make_float3(
							configJson["camera"]["vUp"][0].get<float>(),
							configJson["camera"]["vUp"][1].get<float>(),
							configJson["camera"]["vUp"][2].get<float>());
						m_CameraFovY = configJson["camera"]["fovY"].get<float>();
						m_FbAspect = configJson["camera"]["aspect"].get<float>();
						m_BgLightColor = make_float3(
							configJson["light"]["color"][0].get<float>(),
							configJson["light"]["color"][1].get<float>(),
							configJson["light"]["color"][2].get<float>());
						auto imgRenderPath = configJson["imgRenderPath"].get<std::string>();
						m_ImgRenderPath = {};
						std::memcpy(m_ImgRenderPath.data(), imgRenderPath.c_str(), m_ImgRenderPath.size());
						auto imgDebugPath = configJson["imgDebugPath"].get<std::string>();
						std::memcpy(m_ImgDebugPath.data(), imgDebugPath.c_str(), m_ImgDebugPath.size());
						auto camera = rtlib::ext::Camera(eye, lookAt, vUp, m_CameraFovY, m_FbAspect);
						glfwSetWindowSize(m_Window, m_FbWidth, m_FbHeight);
						m_CameraController.SetMouseSensitivity(m_MouseSensitity);
						m_CameraController.SetMovementSpeed(m_MovementSpeed);
						m_CameraController.SetCamera(camera);
						m_ChangeTrace = true;
						m_UpdateCamera = true;
						m_UpdateLight = true;
						m_FlushFrame = true;
					}
					configFile.close();
				}
				ImGui::SameLine();
			}

			if (ImGui::Button("Save"))
			{
				std::filesystem::path filePath = std::string(m_GlobalSettingPath.data());
				auto camera = m_CameraController.GetCamera(m_CameraFovY, m_FbAspect);
				auto eye = camera.getEye();
				auto lookAt = camera.getLookAt();
				auto vUp = camera.getVup();
				auto configJson = nlohmann::json();
				configJson["fbWidth"] = m_FbWidth;
				configJson["fbHeight"] = m_FbHeight;
				configJson["maxTraceDepth"] = m_MaxTraceDepth;
				configJson["movementSpeed"] = m_MovementSpeed;
				configJson["mouseSensitivity"] = m_MouseSensitity;
				configJson["camera"]["eye"] = {eye.x, eye.y, eye.z};
				configJson["camera"]["lookAt"] = {lookAt.x, lookAt.y, lookAt.z};
				configJson["camera"]["vUp"] = {vUp.x, vUp.y, vUp.z};
				configJson["camera"]["fovY"] = camera.getFovY();
				configJson["camera"]["aspect"] = camera.getAspect();
				configJson["light"]["color"] = {lightColor[0], lightColor[1], lightColor[2]};
				configJson["imgRenderPath"] = std::string(m_ImgRenderPath.data());
				configJson["imgDebugPath"] = std::string(m_ImgDebugPath.data());
				std::ofstream configFile(filePath);
				if (!configFile.fail())
				{
					configFile << configJson;
				}
				configFile.close();
			}
		}
		int isLockedUpdate = m_LockUpdate;
		if (!m_TraceGuide && !m_StoreResult)
		{
			if (ImGui::RadioButton("UnLockUpdate", &isLockedUpdate, 0))
			{
				m_LockUpdate = false;
			}
			ImGui::SameLine();
			if (ImGui::RadioButton("  LockUpdate", &isLockedUpdate, 1))
			{
				m_LockUpdate = true;
				m_FlushFrame = true;
			}
		}
	}
	ImGui::End();

	ImGui::Begin("AppConfig", nullptr, ImGuiWindowFlags_MenuBar);
	ImGui::Text("      fps: %4.3lf", (float)1.0f / m_DelFrameTime);
	ImGui::Text("FrameSize: (   %4d,    %4d)", m_FbWidth, m_FbHeight);
	ImGui::Text("CurCursor: (%4.3lf, %4.3lf)", m_CurCursorPos.x, m_CurCursorPos.y);
	ImGui::Text("DelCursor: (%4.3lf, %4.3lf)", m_DelCursorPos.x, m_DelCursorPos.y);
	ImGui::End();

	ImGui::Begin("TraceSetting", nullptr, ImGuiWindowFlags_MenuBar);
	ImGui::Text("        spp/sec: %4.3lf", (float)m_SamplePerLaunch / m_DelTraceTime);
	ImGui::Text("            spp: %4d", m_SamplePerALL);
	{
		if (!m_TraceGuide && !m_LockUpdate)
		{
			int maxTraceDepth = m_MaxTraceDepth;
			if (ImGui::SliderInt("maxTraceDepth", &maxTraceDepth, 1, 100))
			{
				m_MaxTraceDepth = maxTraceDepth;
				m_ChangeTrace = true;
			}
			int samplePerLaunch = m_SamplePerLaunch;
			if (ImGui::SliderInt("samplePerLaunch", &samplePerLaunch, 1, 50))
			{
				m_SamplePerLaunch = samplePerLaunch;
			}
			int samplePerStore = m_SamplePerStore;
			if (ImGui::InputInt("samplePerStore", &samplePerStore, 10, 100))
			{
				m_SamplePerStore = std::max(samplePerStore, 1);
			}
		}
		else
		{
			ImGui::Text("  maxTraceDepth: %d", m_MaxTraceDepth);
			ImGui::Text("samplePerLaunch: %d", m_SamplePerLaunch);
			ImGui::Text(" samplePerStore: %d", m_SamplePerStore);
		}

		int samplePerBudget = m_SamplePerBudget;
		if (ImGui::InputInt("samplePerBudget", &samplePerBudget, 1000, 10000))
		{
			m_SamplePerBudget = samplePerBudget;
		}
		ImGui::InputText("ImgRootDir", m_ImgRenderPath.data(), m_ImgRenderPath.size());
		if (!m_TraceGuide)
		{
			{
				bool traceNEE = m_TraceNEE;
				if (ImGui::RadioButton("NEE", traceNEE))
				{
					m_TraceNEE = !traceNEE;
					m_ChangeTrace = true;
				}
				ImGui::SameLine();
			}

			if (ImGui::Button("Guide"))
			{
				m_TraceGuide = true;
				m_FlushFrame = true;
			}

			ImGui::SameLine();
			if (!m_StoreResult && !m_TraceGuide)
			{
				if (ImGui::Button("Store"))
				{
					m_StoreResult = true;
					m_LockUpdate = true;
					m_FlushFrame = true;
				}
			}
			ImGui::SameLine();
		}
		bool storeImage = false;
		if (m_TraceGuide || m_StoreResult)
		{
			storeImage = m_SamplePerALL % m_SamplePerStore == 0 && m_SamplePerALL;
			if (m_SamplePerALL > m_SamplePerBudget)
			{
				m_StoreResult = false;
				m_FlushFrame = true;
			}
		}
		{
			if (ImGui::Button("Save") || storeImage)
			{
				std::filesystem::path imgRenderPath = std::string(m_ImgRenderPath.data());
				if (std::filesystem::exists(imgRenderPath))
				{

					auto savePath = imgRenderPath / ((m_TraceGuide ? "Guide_"s : "Trace_"s) + (m_TraceNEE ? "NEE_"s : "Def_"s));
					savePath += std::to_string(m_SamplePerALL);
					savePath += "_" + std::to_string(m_CurTraceTime);
					auto saveExrPath = savePath;
					auto savePngPath = savePath;
					saveExrPath += ".exr";
					savePngPath += ".png";
					test::SaveEXRFromCUDA(saveExrPath.string().c_str(), m_AccumBuffer, m_FbWidth, m_FbHeight, m_SamplePerALL);
					test::SavePNGFromGL(savePngPath.string().c_str(), *m_RenderTexture.get());
				}
			}
		}
		ImGui::End();

		ImGui::Begin("DebugSetting", nullptr, ImGuiWindowFlags_MenuBar);
		{
			static int curFrameIdx = 0;
			{
				int i = 0;
				for (auto kDebugFrameName : kDebugFrameNames)
				{
					if (ImGui::RadioButton(kDebugFrameName, &curFrameIdx, i++))
					{
						m_CurDebugFrame = kDebugFrameName;
					}
					if (i != 4 && i != 8)
					{
						ImGui::SameLine();
					}
					else
					{
						ImGui::NewLine();
					}
				}
			}
		}
		if (ImGui::InputText("ImgRootDir", m_ImgDebugPath.data(), m_ImgDebugPath.size()))
		{
		}
		if (ImGui::Button("Launch"))
		{
			m_LaunchDebug = true;
		}
		else
		{
			m_LaunchDebug = false;
		}
		ImGui::SameLine();
		if (ImGui::Button("Save"))
		{
			std::filesystem::path imgDebugPath = std::string(m_ImgDebugPath.data());
			if (std::filesystem::exists(imgDebugPath))
			{
				auto filePath = imgDebugPath / ("Debug_" + m_CurDebugFrame);
				if (m_CurDebugFrame == "STree")
				{
					filePath += "_SppBudget" + std::to_string(m_SamplePerBudget) + "_Spp_" + std::to_string(m_SampleForPrvDbg);
				}
				filePath += ".png";
				test::SavePNGFromGL(filePath.string().c_str(), *m_DebugTexture.get());
			}
		}
		ImGui::NewLine();
		ImGui::Image(reinterpret_cast<void *>(m_DebugTexture->getID()), {256 * m_FbAspect, 256}, {1, 1}, {0, 0});
		ImGui::End();

		ImGui::PopStyleColor();
		ImGui::PopStyleColor();
		ImGui::Render();
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
	}
}
// DrawEvents
void TestPG4Application::PollEvents()
{
	double prvTime = m_CurFrameTime;
	m_CurFrameTime = glfwGetTime();
	m_DelFrameTime = m_CurFrameTime - prvTime;

	if (!m_LockUpdate)
	{
		if (glfwGetWindowAttrib(m_Window, GLFW_RESIZABLE) == GLFW_FALSE)
		{
			glfwSetWindowAttrib(m_Window, GLFW_RESIZABLE, GLFW_TRUE);
		}
		if (glfwGetKey(m_Window, GLFW_KEY_W) == GLFW_PRESS)
		{
			m_CameraController.ProcessKeyboard(rtlib::ext::CameraMovement::eForward, m_DelFrameTime);
			m_UpdateCamera = true;
		}
		if (glfwGetKey(m_Window, GLFW_KEY_S) == GLFW_PRESS)
		{
			m_CameraController.ProcessKeyboard(rtlib::ext::CameraMovement::eBackward, m_DelFrameTime);
			m_UpdateCamera = true;
		}
		if (glfwGetKey(m_Window, GLFW_KEY_A) == GLFW_PRESS)
		{
			m_CameraController.ProcessKeyboard(rtlib::ext::CameraMovement::eLeft, m_DelFrameTime);
			m_UpdateCamera = true;
		}
		if (glfwGetKey(m_Window, GLFW_KEY_D) == GLFW_PRESS)
		{
			m_CameraController.ProcessKeyboard(rtlib::ext::CameraMovement::eRight, m_DelFrameTime);
			m_UpdateCamera = true;
		}
		if (glfwGetKey(m_Window, GLFW_KEY_LEFT) == GLFW_PRESS)
		{
			m_CameraController.ProcessKeyboard(rtlib::ext::CameraMovement::eLeft, m_DelFrameTime);
			m_UpdateCamera = true;
		}
		if (glfwGetKey(m_Window, GLFW_KEY_RIGHT) == GLFW_PRESS)
		{
			m_CameraController.ProcessKeyboard(rtlib::ext::CameraMovement::eRight, m_DelFrameTime);
			m_UpdateCamera = true;
		}
		if (glfwGetKey(m_Window, GLFW_KEY_UP) == GLFW_PRESS)
		{
			m_CameraController.ProcessKeyboard(rtlib::ext::CameraMovement::eUp, m_DelFrameTime);
			m_UpdateCamera = true;
		}
		if (glfwGetKey(m_Window, GLFW_KEY_DOWN) == GLFW_PRESS)
		{
			m_CameraController.ProcessKeyboard(rtlib::ext::CameraMovement::eDown, m_DelFrameTime);
			m_UpdateCamera = true;
		}
		if (glfwGetMouseButton(m_Window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS)
		{
			m_CameraController.ProcessMouseMovement(-m_DelCursorPos.x, m_DelCursorPos.y);
			m_UpdateCamera = true;
		}
	}
	else
	{
		if (glfwGetWindowAttrib(m_Window, GLFW_RESIZABLE) == GLFW_TRUE)
		{
			glfwSetWindowAttrib(m_Window, GLFW_RESIZABLE, GLFW_FALSE);
		}
	}
	glfwPollEvents();
}
// Update
void TestPG4Application::Update()
{
	if (m_ResizeFrame)
	{
		m_FrameBuffer->Resize(m_FbWidth, m_FbHeight);
		m_RenderTexture->reset();
		m_RenderTexture->allocate({(size_t)m_FbWidth, (size_t)m_FbHeight});
		m_RenderTexture->setParameteri(GL_TEXTURE_MAG_FILTER, GL_LINEAR, false);
		m_RenderTexture->setParameteri(GL_TEXTURE_MIN_FILTER, GL_LINEAR, false);
		m_RenderTexture->setParameteri(GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE, false);
		m_RenderTexture->setParameteri(GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE, false);
		m_DebugTexture->reset();
		m_DebugTexture->allocate({(size_t)m_FbWidth, (size_t)m_FbHeight});
		m_DebugTexture->setParameteri(GL_TEXTURE_MAG_FILTER, GL_LINEAR, false);
		m_DebugTexture->setParameteri(GL_TEXTURE_MIN_FILTER, GL_LINEAR, false);
		m_DebugTexture->setParameteri(GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE, false);
		m_DebugTexture->setParameteri(GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE, false);
	}
	if (m_FlushFrame || m_ResizeFrame || m_UpdateCamera || m_UpdateLight || m_ChangeTrace)
	{
		m_AccumBuffer.resize(m_FbWidth * m_FbHeight);
		m_AccumBuffer.upload(std::vector<float3>(m_FbWidth * m_FbHeight, float3{ 0.0f,0.0f,0.0f }));
		m_SamplePerALL = 0;
		m_CurTraceTime = 0.0;
	}
	m_SimpleActor->Update();
	m_SimpleNEEActor->Update();
	m_GuideActor->Update();
	m_GuideNEEActor->Update();
	m_DebugActor->Update();
	m_UpdateCamera = false;
	m_UpdateLight = false;
	m_ResizeFrame = false;
	m_FlushFrame = false;
	m_ChangeTrace = false;
}
// Update: Lock
void TestPG4Application::LockUpdate()
{
	m_LockUpdate = true;
}
// Update: UnLock
void TestPG4Application::UnLockUpdate()
{
	m_LockUpdate = false;
}
// Free: GLFW
void TestPG4Application::FreeGLFW()
{
	glfwTerminate();
}
// Free: Window
void TestPG4Application::FreeWindow()
{
	glfwDestroyWindow(m_Window);
	m_Window = nullptr;
}
// Free: CUDA
void TestPG4Application::FreeCUDA()
{
	m_Context.reset();
}
// Free: Gui
void TestPG4Application::FreeGui()
{
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();
	//Renderer
	m_Renderer->reset();
	m_Renderer.reset();
}
// Free: Assets
void TestPG4Application::FreeAssets()
{
	m_TextureAssets.Reset();
	m_ObjModelAssets.Reset();
}
// Free: Acceleration Structures
void TestPG4Application::FreeAccelerationStructures()
{
	m_GASHandles.clear();
	m_IASHandles.clear();
}
// Free: Light
void TestPG4Application::FreeLight()
{
	m_MeshLights.Reset();
}
// Free: Camera
void TestPG4Application::FreeCamera()
{
	m_CameraController = {};
}
// Free: STree
void TestPG4Application::FreeSTree()
{
	m_STree.reset();
}
// Free: Frame Resources
void TestPG4Application::FreeFrameResources()
{
	m_RenderTexture->reset();
	m_FrameBuffer->CleanUp();
}
// Free: Tracers
void TestPG4Application::FreeTracers()
{
	m_SimpleActor->CleanUp();
	m_SimpleNEEActor->CleanUp();
	m_GuideActor->CleanUp();
	m_GuideNEEActor->CleanUp();
	m_DebugActor->CleanUp();
}
//  Get: OPXContext
auto TestPG4Application::GetOPXContext() const -> std::shared_ptr<rtlib::OPXContext>
{
	return m_Context;
}
//  Get: TLAS
auto TestPG4Application::GetTLAS() const -> rtlib::ext::IASHandlePtr
{
	return m_IASHandles.at("TopLevel");
}
//  Get: Materials
auto TestPG4Application::GetMaterials() const -> const std::vector<rtlib::ext::VariableMap> &
{
	// TODO: return �X�e�[�g�����g�������ɑ}�����܂�
	return m_Materials;
}
//  Get: Camera
auto TestPG4Application::GetCamera() const -> rtlib::ext::Camera
{
	return m_CameraController.GetCamera(m_CameraFovY, m_FbAspect);
}
auto TestPG4Application::GetMeshLightList() const -> MeshLightList
{
	MeshLightList lightList = {};
	lightList.count = m_MeshLights.cpuHandle.size();
	lightList.data  = m_MeshLights.gpuHandle.getDevicePtr();
	return lightList;
}
auto TestPG4Application::GetBackGroundLight() const -> float3
{
	return m_BgLightColor;
}
//  Get: STree
auto TestPG4Application::GetSTree() const -> std::shared_ptr<test::RTSTreeWrapper>
{
	return m_STree;
}
//  Get: Texture
auto TestPG4Application::GetTexture(const std::string &name) const -> const rtlib::CUDATexture2D<uchar4> &
{
	// TODO: return �X�e�[�g�����g�������ɑ}�����܂�
	return m_TextureAssets.GetAsset(name);
}

auto TestPG4Application::GetAccumBuffer() -> rtlib::CUDABuffer<float3>&
{
	// TODO: return ステートメントをここに挿入します
	return m_AccumBuffer;
}

auto TestPG4Application::GetAccumBuffer() const -> const rtlib::CUDABuffer<float3>&
{
	// TODO: return ステートメントをここに挿入します
	return m_AccumBuffer;
}
