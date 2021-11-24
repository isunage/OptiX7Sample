#include "../include/Test22Application.h"
#include <RTLib/Optix.h>
#include <RTLib/Utils.h>
#include <RTLib/ext/Resources/CUDA.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
#include <nlohmann/json.hpp>
#include <random>
//Init
//SimpleTracer
class Test22SimpleTracer : public test::RTTracer
{
public:
	struct UserData
	{
		uchar4*      frameBuffer;
		unsigned int samplePerAll;
		unsigned int samplePerLaunch;
	};

private:
	using RayGRecord = rtlib::SBTRecord<RayGenData>;
	using MissRecord = rtlib::SBTRecord<MissData>;
	using HitGRecord = rtlib::SBTRecord<HitgroupData>;

public:
	Test22SimpleTracer(Test22Application *app)
	{
		m_ParentApp = app;
	}
	// RTTracer ����Čp������܂���
	virtual void Initialize() override
	{
		this->InitFrameResources();
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
		m_Params.cpuHandle[0].accumBuffer = m_AccumBuffer.getDevicePtr();
		m_Params.cpuHandle[0].frameBuffer = pUserData->frameBuffer;
		m_Params.cpuHandle[0].seedBuffer = m_SeedBuffer.getDevicePtr();
		m_Params.cpuHandle[0].samplePerLaunch = pUserData->samplePerLaunch;
		cudaMemcpyAsync(m_Params.gpuHandle.getDevicePtr(), &m_Params.cpuHandle[0], sizeof(RayTraceParams), cudaMemcpyHostToDevice, config.stream);
		auto &pipeline = m_ParentApp->GetOPXPipeline("Trace");
		pipeline.launch(config.stream, m_Params.gpuHandle.getDevicePtr(), m_ShaderBindingTable, config.width, config.height, config.depth);
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
		m_ShaderBindingTable = {};
		m_RGRecordBuffer.Reset();
		m_MSRecordBuffers.Reset();
		m_HGRecordBuffers.Reset();
		m_Params.Reset();
		m_AccumBuffer.reset();
		m_SeedBuffer.reset();
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
			m_AccumBuffer.resize(m_ParentApp->GetFbWidth() * m_ParentApp->GetFbHeight());
			m_AccumBuffer.upload(std::vector<float3>(m_ParentApp->GetFbWidth() * m_ParentApp->GetFbHeight()));
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
			auto light = m_ParentApp->GetLight();
			m_Params.cpuHandle[0].light = light;
			m_HGRecordBuffers.cpuHandle[m_LightHgRecIndex].data.emission = light.emission;
			m_HGRecordBuffers.cpuHandle[m_LightHgRecIndex].data.diffuse = light.emission;
			RTLIB_CUDA_CHECK(cudaMemcpy(m_HGRecordBuffers.gpuHandle.getDevicePtr() + m_LightHgRecIndex, &m_HGRecordBuffers.cpuHandle[m_LightHgRecIndex], sizeof(HitGRecord), cudaMemcpyHostToDevice));
		}
	}
	virtual ~Test22SimpleTracer() {}

private:
	void InitFrameResources()
	{
		std::vector<unsigned int> seedData(m_ParentApp->GetFbWidth() * m_ParentApp->GetFbHeight());
		std::random_device rd;
		std::mt19937 mt(rd());
		std::generate(seedData.begin(), seedData.end(), mt);
		m_AccumBuffer = rtlib::CUDABuffer<float3>(std::vector<float3>(m_ParentApp->GetFbWidth() * m_ParentApp->GetFbHeight()));
		m_SeedBuffer = rtlib::CUDABuffer<uint32_t>(seedData);
	}
	void InitShaderBindingTable()
	{
		auto SpecifyMaterialType = [](const rtlib::ext::VariableMap &material)
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
			else if (specCol.x + specCol.y + specCol.z > 0.0f)
			{
				return "Phong";
			}
			else
			{
				return "Diffuse";
			}
		};
		auto tlas = m_ParentApp->GetTLAS();
		auto camera = m_ParentApp->GetCamera();
		auto &materials = m_ParentApp->GetMaterials();
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
		m_MSRecordBuffers.cpuHandle[RAY_TYPE_RADIANCE].data.bgColor = {0, 0, 0, 0};
		m_MSRecordBuffers.cpuHandle[RAY_TYPE_OCCLUSION] = m_ParentApp->GetMSProgramGroup("Trace.Occluded").getSBTRecord<MissData>();
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
						for (auto& vertexBuffer : mesh->GetSharedResource()->vertexBuffers)
						{
							if (!vertexBuffer.HasGpuComponent("CUDA")) {
							throw std::runtime_error("VertexBuffer of Mesh '" + mesh->GetUniqueResource()->name + "' Has No CUDA Component!");
						}
						}
						if (!mesh->GetUniqueResource()->triIndBuffer.HasGpuComponent("CUDA")) {
							throw std::runtime_error("TriIndBuffer of Mesh '" + mesh->GetUniqueResource()->name + "' Has No CUDA Component!");
						}
						auto cudaVertexBuffer = mesh->GetSharedResource()->vertexBuffers[0].GetGpuComponent<rtlib::ext::resources::CUDABufferComponent<float>>("CUDA");
						auto cudaNormalBuffer = mesh->GetSharedResource()->vertexBuffers[1].GetGpuComponent<rtlib::ext::resources::CUDABufferComponent<float>>("CUDA");
						auto cudaTexCrdBuffer = mesh->GetSharedResource()->vertexBuffers[2].GetGpuComponent<rtlib::ext::resources::CUDABufferComponent<float>>("CUDA");
						auto cudaTriIndBuffer = mesh->GetUniqueResource()->triIndBuffer.GetGpuComponent<rtlib::ext::resources::CUDABufferComponent<uint3>>("CUDA");
						for (size_t i = 0; i < mesh->GetUniqueResource()->materials.size(); ++i)
						{
							auto materialId = mesh->GetUniqueResource()->materials[i];
							auto &material = materials[materialId];
							HitgroupData radianceHgData = {};
							{
								radianceHgData.vertices   =(float3*) cudaVertexBuffer->GetHandle().getDevicePtr();
								radianceHgData.normals    =(float3*) cudaNormalBuffer->GetHandle().getDevicePtr();
								radianceHgData.texCoords  =(float2*) cudaTexCrdBuffer->GetHandle().getDevicePtr();
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
								typeString += ".Default";
							}
							m_HGRecordBuffers.cpuHandle[RAY_TYPE_COUNT * sbtOffset + RAY_TYPE_COUNT * i + RAY_TYPE_RADIANCE ] = m_ParentApp->GetHGProgramGroup(std::string("Trace.Radiance.") + typeString).getSBTRecord<HitgroupData>(radianceHgData);
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
	void InitLaunchParams()
	{
		auto tlas = m_ParentApp->GetTLAS();
		m_Params.cpuHandle.resize(1);
		m_Params.cpuHandle[0].gasHandle = tlas->GetHandle();
		m_Params.cpuHandle[0].width = m_ParentApp->GetFbWidth();
		m_Params.cpuHandle[0].height = m_ParentApp->GetFbHeight();
		m_Params.cpuHandle[0].light = m_ParentApp->GetLight();
		m_Params.cpuHandle[0].accumBuffer = m_AccumBuffer.getDevicePtr();
		m_Params.cpuHandle[0].frameBuffer = nullptr;
		m_Params.cpuHandle[0].seedBuffer = m_SeedBuffer.getDevicePtr();
		m_Params.cpuHandle[0].maxTraceDepth = m_ParentApp->GetMaxTraceDepth();
		m_Params.cpuHandle[0].samplePerLaunch = 1;

		m_Params.Upload();
	}

private:
	Test22Application *m_ParentApp = nullptr;
	OptixShaderBindingTable m_ShaderBindingTable = {};
	rtlib::CUDAUploadBuffer<RayGRecord> m_RGRecordBuffer = {};
	rtlib::CUDAUploadBuffer<MissRecord> m_MSRecordBuffers = {};
	rtlib::CUDAUploadBuffer<HitGRecord> m_HGRecordBuffers = {};
	rtlib::CUDAUploadBuffer<RayTraceParams> m_Params = {};
	rtlib::CUDABuffer<float3> m_AccumBuffer = {};
	rtlib::CUDABuffer<unsigned int> m_SeedBuffer = {};
	unsigned int m_LightHgRecIndex = 0;
	unsigned int m_SamplePerAll = 0;
};
//  NEETracer
class Test22NEETracer : public test::RTTracer
{
public:
	struct UserData
	{
		uchar4* frameBuffer;
		unsigned int samplePerAll;
		unsigned int samplePerLaunch;
	};

private:
	using RayGRecord = rtlib::SBTRecord<RayGenData>;
	using MissRecord = rtlib::SBTRecord<MissData>;
	using HitGRecord = rtlib::SBTRecord<HitgroupData>;

public:
	Test22NEETracer(Test22Application* app)
	{
		m_ParentApp = app;
	}
	// RTTracer ����Čp������܂���
	virtual void Initialize() override
	{
		this->InitFrameResources();
		this->InitShaderBindingTable();
		this->InitLaunchParams();
	}
	virtual void Launch(const test::RTTraceConfig& config) override
	{
		UserData* pUserData = (UserData*)config.pUserData;
		if (!pUserData)
		{
			return;
		}
		m_Params.cpuHandle[0].width       = config.width;
		m_Params.cpuHandle[0].height      = config.height;
		m_Params.cpuHandle[0].accumBuffer = m_AccumBuffer.getDevicePtr();
		m_Params.cpuHandle[0].frameBuffer = pUserData->frameBuffer;
		m_Params.cpuHandle[0].seedBuffer  = m_SeedBuffer.getDevicePtr();
		m_Params.cpuHandle[0].samplePerLaunch = pUserData->samplePerLaunch;
		cudaMemcpyAsync(m_Params.gpuHandle.getDevicePtr(), &m_Params.cpuHandle[0], sizeof(RayTraceParams), cudaMemcpyHostToDevice, config.stream);
		auto& pipeline = m_ParentApp->GetOPXPipeline("Trace");
		pipeline.launch(config.stream, m_Params.gpuHandle.getDevicePtr(), m_ShaderBindingTable, config.width, config.height, config.depth);
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
		m_ShaderBindingTable = {};
		m_RGRecordBuffer.Reset();
		m_MSRecordBuffers.Reset();
		m_HGRecordBuffers.Reset();
		m_Params.Reset();
		m_AccumBuffer.reset();
		m_SeedBuffer.reset();
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
			m_AccumBuffer.resize(m_ParentApp->GetFbWidth() * m_ParentApp->GetFbHeight());
			m_AccumBuffer.upload(std::vector<float3>(m_ParentApp->GetFbWidth() * m_ParentApp->GetFbHeight()));
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
			auto light = m_ParentApp->GetLight();
			m_Params.cpuHandle[0].light = light;
			m_HGRecordBuffers.cpuHandle[m_LightHgRecIndex].data.emission = light.emission;
			m_HGRecordBuffers.cpuHandle[m_LightHgRecIndex].data.diffuse = light.emission;
			RTLIB_CUDA_CHECK(cudaMemcpy(m_HGRecordBuffers.gpuHandle.getDevicePtr() + m_LightHgRecIndex, &m_HGRecordBuffers.cpuHandle[m_LightHgRecIndex], sizeof(HitGRecord), cudaMemcpyHostToDevice));
		}
	}
	virtual ~Test22NEETracer() {}

private:
	void InitFrameResources()
	{
		std::vector<unsigned int> seedData(m_ParentApp->GetFbWidth() * m_ParentApp->GetFbHeight());
		std::random_device rd;
		std::mt19937 mt(rd());
		std::generate(seedData.begin(), seedData.end(), mt);
		m_AccumBuffer = rtlib::CUDABuffer<float3>(std::vector<float3>(m_ParentApp->GetFbWidth() * m_ParentApp->GetFbHeight()));
		m_SeedBuffer = rtlib::CUDABuffer<uint32_t>(seedData);
	}
	void InitShaderBindingTable()
	{
		auto SpecifyMaterialType = [](const rtlib::ext::VariableMap& material)
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
			else if (specCol.x + specCol.y + specCol.z > 0.0f)
			{
				return "Phong";
			}
			else
			{
				return "Diffuse";
			}
		};
		auto tlas = m_ParentApp->GetTLAS();
		auto camera = m_ParentApp->GetCamera();
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
		m_MSRecordBuffers.cpuHandle[RAY_TYPE_RADIANCE].data.bgColor = { 0, 0, 0, 0 };
		m_MSRecordBuffers.cpuHandle[RAY_TYPE_OCCLUSION] = m_ParentApp->GetMSProgramGroup("Trace.Occluded").getSBTRecord<MissData>();
		m_MSRecordBuffers.Upload();
		m_HGRecordBuffers.cpuHandle.resize(tlas->GetSbtCount() * RAY_TYPE_COUNT);
		{
			size_t sbtOffset = 0;
			for (auto& instanceSet : tlas->GetInstanceSets())
			{
				for (auto& baseGASHandle : instanceSet->baseGASHandles)
				{
					for (auto& mesh : baseGASHandle->GetMeshes())
					{
						for (auto& vertexBuffer : mesh->GetSharedResource()->vertexBuffers)
						{
							if (!vertexBuffer.HasGpuComponent("CUDA")) {
							throw std::runtime_error("VertexBuffer of Mesh '" + mesh->GetUniqueResource()->name + "' Has No CUDA Component!");
						}
						}
						if (!mesh->GetUniqueResource()->triIndBuffer.HasGpuComponent("CUDA")) {
							throw std::runtime_error("TriIndBuffer of Mesh '" + mesh->GetUniqueResource()->name + "' Has No CUDA Component!");
						}
						auto cudaVertexBuffer = mesh->GetSharedResource()->vertexBuffers[0].GetGpuComponent<rtlib::ext::resources::CUDABufferComponent<float>>("CUDA");
						auto cudaNormalBuffer = mesh->GetSharedResource()->vertexBuffers[1].GetGpuComponent<rtlib::ext::resources::CUDABufferComponent<float>>("CUDA");
						auto cudaTexCrdBuffer = mesh->GetSharedResource()->vertexBuffers[2].GetGpuComponent<rtlib::ext::resources::CUDABufferComponent<float>>("CUDA");
						auto cudaTriIndBuffer = mesh->GetUniqueResource()->triIndBuffer.GetGpuComponent<rtlib::ext::resources::CUDABufferComponent<uint3>>("CUDA");
						for (size_t i = 0; i < mesh->GetUniqueResource()->materials.size(); ++i)
						{
							auto materialId = mesh->GetUniqueResource()->materials[i];
							auto& material = materials[materialId];
							HitgroupData radianceHgData = {};
							{
								radianceHgData.vertices = (float3*)cudaVertexBuffer->GetHandle().getDevicePtr();
								radianceHgData.normals = (float3*)cudaNormalBuffer->GetHandle().getDevicePtr();
								radianceHgData.texCoords = (float2*)cudaTexCrdBuffer->GetHandle().getDevicePtr();
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
	void InitLaunchParams()
	{
		auto tlas = m_ParentApp->GetTLAS();
		m_Params.cpuHandle.resize(1);
		m_Params.cpuHandle[0].gasHandle = tlas->GetHandle();
		m_Params.cpuHandle[0].width = m_ParentApp->GetFbWidth();
		m_Params.cpuHandle[0].height = m_ParentApp->GetFbHeight();
		m_Params.cpuHandle[0].light = m_ParentApp->GetLight();
		m_Params.cpuHandle[0].accumBuffer = m_AccumBuffer.getDevicePtr();
		m_Params.cpuHandle[0].frameBuffer = nullptr;
		m_Params.cpuHandle[0].seedBuffer = m_SeedBuffer.getDevicePtr();
		m_Params.cpuHandle[0].maxTraceDepth = m_ParentApp->GetMaxTraceDepth();
		m_Params.cpuHandle[0].samplePerLaunch = 1;

		m_Params.Upload();
	}

private:
	Test22Application* m_ParentApp = nullptr;
	OptixShaderBindingTable m_ShaderBindingTable = {};
	rtlib::CUDAUploadBuffer<RayGRecord> m_RGRecordBuffer = {};
	rtlib::CUDAUploadBuffer<MissRecord> m_MSRecordBuffers = {};
	rtlib::CUDAUploadBuffer<HitGRecord> m_HGRecordBuffers = {};
	rtlib::CUDAUploadBuffer<RayTraceParams> m_Params = {};
	rtlib::CUDABuffer<float3> m_AccumBuffer = {};
	rtlib::CUDABuffer<unsigned int> m_SeedBuffer = {};
	unsigned int m_LightHgRecIndex = 0;
	unsigned int m_SamplePerAll = 0;
};
// DebugTracer
class Test22DebugTracer : public test::RTTracer
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
	};

private:
	using RayGRecord = rtlib::SBTRecord<RayGenData>;
	using MissRecord = rtlib::SBTRecord<MissData>;
	using HitGRecord = rtlib::SBTRecord<HitgroupData>;

public:
	Test22DebugTracer(Test22Application *app)
	{
		m_ParentApp = app;
	}
	// RTTracer ����Čp������܂���
	virtual void Initialize() override
	{
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
		m_Params.cpuHandle[0].diffuseBuffer = pUserData->diffuseBuffer;
		m_Params.cpuHandle[0].specularBuffer = pUserData->specularBuffer;
		m_Params.cpuHandle[0].emissionBuffer = pUserData->emissionBuffer;
		m_Params.cpuHandle[0].transmitBuffer = pUserData->transmitBuffer;
		m_Params.cpuHandle[0].normalBuffer = pUserData->normalBuffer;
		m_Params.cpuHandle[0].depthBuffer = pUserData->depthBuffer;
		m_Params.cpuHandle[0].texCoordBuffer = pUserData->texCoordBuffer;
		cudaMemcpyAsync(m_Params.gpuHandle.getDevicePtr(), &m_Params.cpuHandle[0], sizeof(RayDebugParams), cudaMemcpyHostToDevice, config.stream);

		auto &pipeline = m_ParentApp->GetOPXPipeline("Debug");
		pipeline.launch(config.stream, m_Params.gpuHandle.getDevicePtr(), m_ShaderBindingTable, config.width, config.height, config.depth);
		if (config.isSync)
		{
			RTLIB_CU_CHECK(cuStreamSynchronize(config.stream));
		}
	}
	virtual void CleanUp() override
	{
		m_ParentApp = nullptr;
		m_ShaderBindingTable = {};
		m_RGRecordBuffer.Reset();
		m_MSRecordBuffers.Reset();
		m_HGRecordBuffers.Reset();
		m_Params.Reset();
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
			auto light = m_ParentApp->GetLight();
			m_Params.cpuHandle[0].light = light;
			m_HGRecordBuffers.cpuHandle[m_LightHgRecIndex].data.emission = light.emission;
			m_HGRecordBuffers.cpuHandle[m_LightHgRecIndex].data.diffuse = light.emission;
			RTLIB_CUDA_CHECK(cudaMemcpy(m_HGRecordBuffers.gpuHandle.getDevicePtr() + m_LightHgRecIndex, &m_HGRecordBuffers.cpuHandle[m_LightHgRecIndex], sizeof(HitGRecord), cudaMemcpyHostToDevice));
		}
	}
	virtual ~Test22DebugTracer() {}

private:
	void InitShaderBindingTable()
	{
		auto tlas = m_ParentApp->GetTLAS();
		auto camera = m_ParentApp->GetCamera();
		auto &materials = m_ParentApp->GetMaterials();
		m_RGRecordBuffer.cpuHandle.resize(1);
		m_RGRecordBuffer.cpuHandle[0] = m_ParentApp->GetRGProgramGroup("Debug.Default").getSBTRecord<RayGenData>();
		m_RGRecordBuffer.cpuHandle[0].data.eye = camera.getEye();
		auto [u, v, w] = camera.getUVW();
		m_RGRecordBuffer.cpuHandle[0].data.u = u;
		m_RGRecordBuffer.cpuHandle[0].data.v = v;
		m_RGRecordBuffer.cpuHandle[0].data.w = w;
		m_RGRecordBuffer.Upload();
		m_MSRecordBuffers.cpuHandle.resize(RAY_TYPE_COUNT);
		m_MSRecordBuffers.cpuHandle[RAY_TYPE_RADIANCE] = m_ParentApp->GetMSProgramGroup("Debug.Default").getSBTRecord<MissData>();
		m_MSRecordBuffers.cpuHandle[RAY_TYPE_RADIANCE].data.bgColor = {0, 0, 0, 0};
		m_MSRecordBuffers.cpuHandle[RAY_TYPE_OCCLUSION] = m_ParentApp->GetMSProgramGroup("Debug.Default").getSBTRecord<MissData>();
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
						for (auto& vertexBuffer : mesh->GetSharedResource()->vertexBuffers)
						{
							if (!vertexBuffer.HasGpuComponent("CUDA")) {
							throw std::runtime_error("VertexBuffer of Mesh '" + mesh->GetUniqueResource()->name + "' Has No CUDA Component!");
						}
						}
						if (!mesh->GetUniqueResource()->triIndBuffer.HasGpuComponent("CUDA")) {
							throw std::runtime_error("TriIndBuffer of Mesh '" + mesh->GetUniqueResource()->name + "' Has No CUDA Component!");
						}
						auto cudaVertexBuffer = mesh->GetSharedResource()->vertexBuffers[0].GetGpuComponent<rtlib::ext::resources::CUDABufferComponent<float>>("CUDA");
						auto cudaNormalBuffer = mesh->GetSharedResource()->vertexBuffers[1].GetGpuComponent<rtlib::ext::resources::CUDABufferComponent<float>>("CUDA");
						auto cudaTexCrdBuffer = mesh->GetSharedResource()->vertexBuffers[2].GetGpuComponent<rtlib::ext::resources::CUDABufferComponent<float>>("CUDA");
						auto cudaTriIndBuffer = mesh->GetUniqueResource()->triIndBuffer.GetGpuComponent<rtlib::ext::resources::CUDABufferComponent<uint3>>("CUDA");
						for (size_t i = 0; i < mesh->GetUniqueResource()->materials.size(); ++i)
						{
							auto materialId = mesh->GetUniqueResource()->materials[i];
							auto &material = materials[materialId];
							HitgroupData radianceHgData = {};
							{
								radianceHgData.vertices = (float3*)cudaVertexBuffer->GetHandle().getDevicePtr();
								radianceHgData.normals = (float3*)cudaNormalBuffer->GetHandle().getDevicePtr();
								radianceHgData.texCoords = (float2*)cudaTexCrdBuffer->GetHandle().getDevicePtr();
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
							m_HGRecordBuffers.cpuHandle[RAY_TYPE_COUNT * sbtOffset + RAY_TYPE_COUNT * i + RAY_TYPE_RADIANCE] = m_ParentApp->GetHGProgramGroup("Debug.Default").getSBTRecord<HitgroupData>(radianceHgData);
							m_HGRecordBuffers.cpuHandle[RAY_TYPE_COUNT * sbtOffset + RAY_TYPE_COUNT * i + RAY_TYPE_OCCLUSION] = m_ParentApp->GetHGProgramGroup("Debug.Default").getSBTRecord<HitgroupData>({});
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
		m_Params.cpuHandle[0].light = m_ParentApp->GetLight();
		m_Params.cpuHandle[0].diffuseBuffer = nullptr;
		m_Params.cpuHandle[0].specularBuffer = nullptr;
		m_Params.cpuHandle[0].emissionBuffer = nullptr;
		m_Params.cpuHandle[0].transmitBuffer = nullptr;
		m_Params.cpuHandle[0].normalBuffer = nullptr;
		m_Params.cpuHandle[0].depthBuffer = nullptr;
		m_Params.cpuHandle[0].texCoordBuffer = nullptr;
		m_Params.Upload();
	}

private:
	Test22Application *m_ParentApp = nullptr;
	OptixShaderBindingTable m_ShaderBindingTable = {};
	rtlib::CUDAUploadBuffer<RayGRecord> m_RGRecordBuffer = {};
	rtlib::CUDAUploadBuffer<MissRecord> m_MSRecordBuffers = {};
	rtlib::CUDAUploadBuffer<HitGRecord> m_HGRecordBuffers = {};
	rtlib::CUDAUploadBuffer<RayDebugParams> m_Params = {};
	unsigned int m_LightHgRecIndex = 0;
};

void Test22Application::InitGLFW()
{
	if (glfwInit() == GLFW_FALSE)
	{
		throw std::runtime_error("Failed To Init GLFW!");
	}
}

void Test22Application::InitWindow()
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

void Test22Application::InitGLAD()
{
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		throw std::runtime_error("Failed To Init GLAD!");
	}
}

void Test22Application::InitCUDA()
{
	RTLIB_CUDA_CHECK(cudaFree(0));
	RTLIB_OPTIX_CHECK(optixInit());
	m_Context = std::make_shared<rtlib::OPXContext>(
		rtlib::OPXContext::Desc{0, 0, OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL, 4});
}

void Test22Application::InitGui()
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

void Test22Application::InitPipelines()
{
	{
		auto rayTracePtxFile = std::ifstream(TEST_TEST22_CUDA_PATH "/RayTrace.ptx", std::ios::binary);
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
		m_Pipelines["Trace"] = m_Context->createPipeline(traceCompileOptions);
		m_Modules["RayTrace"] = m_Pipelines["Trace"].createModule(rayTracePtxData, traceModuleOptions);
		m_RGProgramGroups["Trace.Default"]                  = m_Pipelines["Trace"].createRaygenPG({m_Modules["RayTrace"], RTLIB_RAYGEN_PROGRAM_STR(def)});
		m_MSProgramGroups["Trace.Radiance"]                 = m_Pipelines["Trace"].createMissPG({ m_Modules["RayTrace"], RTLIB_MISS_PROGRAM_STR(radiance)});
		m_MSProgramGroups["Trace.Occluded"]                 = m_Pipelines["Trace"].createMissPG({ m_Modules["RayTrace"], RTLIB_MISS_PROGRAM_STR(occluded)});
		m_HGProgramGroups["Trace.Radiance.Diffuse.Default"] = m_Pipelines["Trace"].createHitgroupPG({ m_Modules["RayTrace"], RTLIB_CLOSESTHIT_PROGRAM_STR(radiance_for_diffuse_def) }, {}, {});
		m_HGProgramGroups["Trace.Radiance.Diffuse.NEE"]     = m_Pipelines["Trace"].createHitgroupPG({ m_Modules["RayTrace"], RTLIB_CLOSESTHIT_PROGRAM_STR(radiance_for_diffuse_nee) }, {}, {});
		m_HGProgramGroups["Trace.Radiance.Phong.Default"]   = m_Pipelines["Trace"].createHitgroupPG({ m_Modules["RayTrace"], RTLIB_CLOSESTHIT_PROGRAM_STR(radiance_for_phong_def) }, {}, {});
		m_HGProgramGroups["Trace.Radiance.Phong.NEE"]       = m_Pipelines["Trace"].createHitgroupPG({ m_Modules["RayTrace"], RTLIB_CLOSESTHIT_PROGRAM_STR(radiance_for_phong_nee) }, {}, {});
		m_HGProgramGroups["Trace.Radiance.Emission"]        = m_Pipelines["Trace"].createHitgroupPG({ m_Modules["RayTrace"], RTLIB_CLOSESTHIT_PROGRAM_STR(radiance_for_emission)}, {}, {});
		m_HGProgramGroups["Trace.Radiance.Specular"]        = m_Pipelines["Trace"].createHitgroupPG({ m_Modules["RayTrace"], RTLIB_CLOSESTHIT_PROGRAM_STR(radiance_for_specular)}, {}, {});
		m_HGProgramGroups["Trace.Radiance.Refraction"]      = m_Pipelines["Trace"].createHitgroupPG({ m_Modules["RayTrace"], RTLIB_CLOSESTHIT_PROGRAM_STR(radiance_for_refraction)}, {}, {});
		m_HGProgramGroups["Trace.Occluded"]                 = m_Pipelines["Trace"].createHitgroupPG({ m_Modules["RayTrace"], RTLIB_CLOSESTHIT_PROGRAM_STR(occluded)}, {}, {});
		m_Pipelines["Trace"].link(traceLinkOptions);
	}

	{
		auto rayDebugPtxFile = std::ifstream(TEST_TEST22_CUDA_PATH "/RayDebug.ptx", std::ios::binary);
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
		m_Pipelines["Debug"] = m_Context->createPipeline(debugCompileOptions);
		m_Modules["RayDebug"] = m_Pipelines["Debug"].createModule(rayDebugPtxData, debugModuleOptions);
		m_RGProgramGroups["Debug.Default"] = m_Pipelines["Debug"].createRaygenPG({m_Modules["RayDebug"], "__raygen__debug"});
		m_MSProgramGroups["Debug.Default"] = m_Pipelines["Debug"].createMissPG({m_Modules["RayDebug"], "__miss__debug"});
		m_HGProgramGroups["Debug.Default"] = m_Pipelines["Debug"].createHitgroupPG({m_Modules["RayDebug"], "__closesthit__debug"}, {}, {});
		m_Pipelines["Debug"].link(debugLinkOptions);
	}
}

void Test22Application::InitAssets()
{
	auto objModelPathes = std::vector{
		//std::filesystem::canonical(std::filesystem::path(TEST_TEST22_DATA_PATH"/Models/Lumberyard/Exterior/exterior.obj")),
		//std::filesystem::canonical(std::filesystem::path(TEST_TEST22_DATA_PATH"/Models/Lumberyard/Interior/interior.obj"))
		std::filesystem::canonical(std::filesystem::path(TEST_TEST22_DATA_PATH "/Models/Sponza/Sponza.obj"))
	};
	for (auto objModelPath : objModelPathes)
	{
		if (!m_ObjModelAssets.LoadAsset(objModelPath.filename().replace_extension().string(), objModelPath.string()))
		{
			throw std::runtime_error("Failed To Load Obj Model!");
		}
	}
	auto smpTexPath = std::filesystem::canonical(std::filesystem::path(TEST_TEST22_DATA_PATH "/Textures/white.png"));
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
					material.SetString("diffTex", "");
				}
			}
			if (specTexPath != "")
			{
				if (!m_TextureAssets.LoadAsset(specTexPath, specTexPath))
				{
					material.SetString("specTex", "");
				}
			}
			if (emitTexPath != "")
			{
				if (!m_TextureAssets.LoadAsset(emitTexPath, emitTexPath))
				{
					material.SetString("emitTex", "");
				}
			}
			if (shinTexPath != "")
			{
				if (!m_TextureAssets.LoadAsset(shinTexPath, shinTexPath))
				{
					material.SetString("shinTex", "");
				}
			}
		}
	}
}

void Test22Application::InitAccelerationStructures()
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

		bool hasLight = false;
		{
			size_t materialOffset = 0;
			for (auto &[name, objModel] : m_ObjModelAssets.GetAssets())
			{
				for (auto& vertexBuffer : objModel.meshGroup->GetSharedResource()->vertexBuffers) {
					if (!vertexBuffer.HasGpuComponent("CUDA"))
				{
						vertexBuffer.AddGpuComponent<rtlib::ext::resources::CUDABufferComponent<float>>("CUDA");
				}
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
					if (name == "light")
					{
						hasLight = true;
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
		if (!hasLight)
		{
			/*Generate Light*/
			rtlib::utils::AABB aabb = {};
			for (auto &mesh : m_GASHandles["World"]->GetMeshes())
			{
				for (size_t i = 0; i<mesh->GetSharedResource()->vertexBuffers[0].Size() / 3; ++i)
				{
					aabb.Update(make_float3(
						mesh->GetSharedResource()->vertexBuffers[0][3 * i + 0],
						mesh->GetSharedResource()->vertexBuffers[0][3 * i + 1],
						mesh->GetSharedResource()->vertexBuffers[0][3 * i + 2]
					));
				}
			}
			OptixAccelBuildOptions accelOptions = {};
			accelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
			accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
			auto lightMesh = rtlib::ext::Mesh::New();
			lightMesh->SetSharedResource(rtlib::ext::MeshSharedResource::New());
			lightMesh->GetSharedResource()->vertexBuffers.resize(3);
			lightMesh->GetSharedResource()->name = "light";
			lightMesh->GetSharedResource()->vertexBuffers[0] = {
				 aabb.min.x, aabb.max.y - 1e-3f, aabb.min.z,
				 aabb.max.x, aabb.max.y - 1e-3f, aabb.min.z,
				 aabb.max.x, aabb.max.y - 1e-3f, aabb.max.z,
				 aabb.min.x, aabb.max.y - 1e-3f, aabb.max.z };
			lightMesh->GetSharedResource()->vertexBuffers[2] = {
				 0.0f, 0.0f,
				 1.0f, 0.0f,
				 1.0f, 1.0f,
				 0.0f, 1.0f
			};
			lightMesh->GetSharedResource()->vertexBuffers[1] = { 0.0f, -1.0f, 0.0f, 0.0f, -1.0f, 0.0f,0.0f, -1.0f, 0.0f, 0.0f, -1.0f, 0.0f };

			lightMesh->GetSharedResource()->layouts.resize(3);
			lightMesh->GetSharedResource()->layouts[0].stride = sizeof(float3);
			lightMesh->GetSharedResource()->layouts[1].stride = sizeof(float3);
			lightMesh->GetSharedResource()->layouts[2].stride = sizeof(float2);

			lightMesh->GetSharedResource()->attributes.resize(3);
			lightMesh->GetSharedResource()->attributes[0].name = "position";
			lightMesh->GetSharedResource()->attributes[0].format = rtlib::ext::MeshVertexFormat::Float3;
			lightMesh->GetSharedResource()->attributes[0].offset = 0;
			lightMesh->GetSharedResource()->attributes[0].bufferIndex = 0;
			lightMesh->GetSharedResource()->attributes[1].name = "normal";
			lightMesh->GetSharedResource()->attributes[1].format = rtlib::ext::MeshVertexFormat::Float3;
			lightMesh->GetSharedResource()->attributes[1].offset = 0;
			lightMesh->GetSharedResource()->attributes[1].bufferIndex = 1;
			lightMesh->GetSharedResource()->attributes[2].name = "texCoord";
			lightMesh->GetSharedResource()->attributes[2].format = rtlib::ext::MeshVertexFormat::Float2;
			lightMesh->GetSharedResource()->attributes[2].offset = 0;
			lightMesh->GetSharedResource()->attributes[2].bufferIndex = 2;
			auto &lightMaterial = m_Materials.back();
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
			lightMesh->GetSharedResource()->vertexBuffers[0].AddGpuComponent<rtlib::ext::resources::CUDABufferComponent<float>>("CUDA");
			lightMesh->GetSharedResource()->vertexBuffers[1].AddGpuComponent<rtlib::ext::resources::CUDABufferComponent<float>>("CUDA");
			lightMesh->GetSharedResource()->vertexBuffers[2].AddGpuComponent<rtlib::ext::resources::CUDABufferComponent<float>>("CUDA");
			lightMesh->SetUniqueResource(rtlib::ext::MeshUniqueResource::New());
			lightMesh->GetUniqueResource()->name = "light";
			lightMesh->GetUniqueResource()->materials = {(unsigned int)m_Materials.size() - 1};
			lightMesh->GetUniqueResource()->matIndBuffer = { 0, 0 };
			lightMesh->GetUniqueResource()->triIndBuffer = { {0, 1, 2}, {2, 3, 0} };
			lightMesh->GetUniqueResource()->matIndBuffer.AddGpuComponent<rtlib::ext::resources::CUDABufferComponent<uint32_t>>("CUDA");
			lightMesh->GetUniqueResource()->triIndBuffer.AddGpuComponent<rtlib::ext::resources::CUDABufferComponent<uint3>>("CUDA");
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

void Test22Application::InitLight()
{
	auto lightGASHandle = m_GASHandles["Light"];
	auto lightMesh = lightGASHandle->GetMesh(0);
	auto lightVertices = std::vector<float3>();
	for (auto &index : lightMesh->GetUniqueResource()->triIndBuffer)
	{
		lightVertices.push_back(make_float3(
			lightMesh->GetSharedResource()->vertexBuffers[0][3 * index.x + 0],
			lightMesh->GetSharedResource()->vertexBuffers[0][3 * index.x + 1],
			lightMesh->GetSharedResource()->vertexBuffers[0][3 * index.x + 2]
		));
		lightVertices.push_back(make_float3(
			lightMesh->GetSharedResource()->vertexBuffers[0][3 * index.y + 0],
			lightMesh->GetSharedResource()->vertexBuffers[0][3 * index.y + 1],
			lightMesh->GetSharedResource()->vertexBuffers[0][3 * index.y + 2]
		));
		lightVertices.push_back(make_float3(
			lightMesh->GetSharedResource()->vertexBuffers[0][3 * index.z + 0],
			lightMesh->GetSharedResource()->vertexBuffers[0][3 * index.z + 1],
			lightMesh->GetSharedResource()->vertexBuffers[0][3 * index.z + 2]
		));
	}
	auto lightAABB = rtlib::utils::AABB(lightVertices);
	auto lightV3   = lightAABB.max - lightAABB.min;
	std::cout << "AABBMax = (" << lightAABB.max.x << ", " << lightAABB.max.y << ", " << lightAABB.max.z << ")\n";
	std::cout << "AABBMin = (" << lightAABB.min.x << ", " << lightAABB.min.y << ", " << lightAABB.min.z << ")\n";
	m_ParallelLight.corner   = lightAABB.min;
	m_ParallelLight.v1       = make_float3(0.0f, 0.0f, lightV3.z);
	m_ParallelLight.v2       = make_float3(lightV3.x, 0.0f, 0.0f);
	m_ParallelLight.normal   = make_float3(0.0f, -1.0f, 0.0f);
	auto lightMaterial       = m_Materials[lightMesh->GetUniqueResource()->materials[0]];
	m_ParallelLight.emission = lightMaterial.GetFloat3As<float3>("emitCol");
}

void Test22Application::InitCamera()
{
	m_CameraController = rtlib::ext::CameraController({0.0f, 1.0f, 5.0f});
	m_MouseSensitity = 0.125f;
	m_MovementSpeed = 10.0f;
	m_CameraController.SetMouseSensitivity(m_MouseSensitity);
	m_CameraController.SetMovementSpeed(m_MovementSpeed);
}

void Test22Application::InitSTree()
{
	auto worldAABB = rtlib::utils::AABB();

	for (auto &mesh : m_GASHandles["World"]->GetMeshes())
	{
		for (auto &index : mesh->GetUniqueResource()->triIndBuffer)
		{
			worldAABB.Update(make_float3(
				mesh->GetSharedResource()->vertexBuffers[0][3 * index.x + 0],
				mesh->GetSharedResource()->vertexBuffers[0][3 * index.x + 1],
				mesh->GetSharedResource()->vertexBuffers[0][3 * index.x + 2]
			));
			worldAABB.Update(make_float3(
				mesh->GetSharedResource()->vertexBuffers[0][3 * index.y + 0],
				mesh->GetSharedResource()->vertexBuffers[0][3 * index.y + 1],
				mesh->GetSharedResource()->vertexBuffers[0][3 * index.y + 2]
			));
			worldAABB.Update(make_float3(
				mesh->GetSharedResource()->vertexBuffers[0][3 * index.z + 0],
				mesh->GetSharedResource()->vertexBuffers[0][3 * index.z + 1],
				mesh->GetSharedResource()->vertexBuffers[0][3 * index.z + 2]
			));
		}
	}
}

void Test22Application::InitFrameResources()
{
	m_RenderTexture = std::make_unique<rtlib::GLTexture2D<uchar4>>();
	m_RenderTexture->allocate({(size_t)m_FbWidth, (size_t)m_FbHeight, nullptr}, GL_TEXTURE_2D);
	m_RenderTexture->setParameteri(GL_TEXTURE_MAG_FILTER, GL_LINEAR, false);
	m_RenderTexture->setParameteri(GL_TEXTURE_MIN_FILTER, GL_LINEAR, false);
	m_RenderTexture->setParameteri(GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE, false);
	m_RenderTexture->setParameteri(GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE, false);

	m_DebugTexture = std::make_unique<rtlib::GLTexture2D<uchar4>>();
	m_DebugTexture->allocate({(size_t)m_FbWidth, (size_t)m_FbHeight, nullptr}, GL_TEXTURE_2D);
	m_DebugTexture->setParameteri(GL_TEXTURE_MAG_FILTER, GL_LINEAR, false);
	m_DebugTexture->setParameteri(GL_TEXTURE_MIN_FILTER, GL_LINEAR, false);
	m_DebugTexture->setParameteri(GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE, false);
	m_DebugTexture->setParameteri(GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE, false);

	m_FrameBuffer = std::make_unique<test::RTFrameBuffer>(m_FbWidth, m_FbHeight);
	m_FrameBuffer->AddCUGLBuffer("Default");
	m_FrameBuffer->GetCUGLBuffer("Default").upload(
		std::vector<uchar4>(m_FbWidth * m_FbHeight, make_uchar4(0, 0, 0, 255)));
	for (auto &debugFrameName : kDebugFrameNames)
	{
		m_FrameBuffer->AddCUGLBuffer(debugFrameName);
		m_FrameBuffer->GetCUGLBuffer(debugFrameName).upload(std::vector<uchar4>(m_FbWidth * m_FbHeight, make_uchar4(0, 0, 0, 255)));
	}
}
void Test22Application::InitTracers()
{
	m_DebugActor = std::shared_ptr<test::RTTracer>(
		new Test22DebugTracer(this));
	m_DebugActor->Initialize();
	m_SimpleActor = std::shared_ptr<test::RTTracer>(
		new Test22SimpleTracer(this));
	m_SimpleActor->Initialize();
	m_NEEActor = std::shared_ptr<test::RTTracer>(
		new Test22NEETracer(this));
	m_NEEActor->Initialize();
}
//ShouldClose
void Test22Application::PrepareLoop()
{
	m_CurFrameTime = glfwGetTime();
}

bool Test22Application::QuitLoop()
{
	if (!m_Window)
		return false;
	return glfwWindowShouldClose(m_Window);
}

void Test22Application::Trace()
{
	if (m_LaunchDebug)
	{
		Test22DebugTracer::UserData userData = {};
		userData.diffuseBuffer = m_FrameBuffer->GetCUGLBuffer("Diffuse").map();
		userData.specularBuffer = m_FrameBuffer->GetCUGLBuffer("Specular").map();
		userData.emissionBuffer = m_FrameBuffer->GetCUGLBuffer("Emission").map();
		userData.transmitBuffer = m_FrameBuffer->GetCUGLBuffer("Transmit").map();
		userData.normalBuffer = m_FrameBuffer->GetCUGLBuffer("Normal").map();
		userData.texCoordBuffer = m_FrameBuffer->GetCUGLBuffer("TexCoord").map();
		userData.depthBuffer = m_FrameBuffer->GetCUGLBuffer("Depth").map();

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
	}

	auto beginTraceTime = glfwGetTime();
	if (!m_TraceNEE)
	{
		Test22SimpleTracer::UserData userData = {};
		userData.frameBuffer = m_FrameBuffer->GetCUGLBuffer("Default").map();
		userData.samplePerLaunch = m_SamplePerLaunch;
		userData.samplePerAll = m_SamplePerALL;

		test::RTTraceConfig traceConfig = {};
		traceConfig.width = m_FbWidth;
		traceConfig.height = m_FbHeight;
		traceConfig.depth = 1;
		traceConfig.isSync = true;
		traceConfig.pUserData = &userData;

		m_SimpleActor->Launch(traceConfig);

		m_FrameBuffer->GetCUGLBuffer("Default").unmap();
		m_SamplePerALL = userData.samplePerAll;
	}
	else {
		Test22NEETracer::UserData userData = {};
		userData.frameBuffer = m_FrameBuffer->GetCUGLBuffer("Default").map();
		userData.samplePerLaunch = m_SamplePerLaunch;
		userData.samplePerAll = m_SamplePerALL;

		test::RTTraceConfig traceConfig = {};
		traceConfig.width = m_FbWidth;
		traceConfig.height = m_FbHeight;
		traceConfig.depth = 1;
		traceConfig.isSync = true;
		traceConfig.pUserData = &userData;

		m_NEEActor->Launch(traceConfig);

		m_FrameBuffer->GetCUGLBuffer("Default").unmap();
		m_SamplePerALL = userData.samplePerAll;
	}
	m_DelTraceTime = glfwGetTime() - beginTraceTime;
}

void Test22Application::DrawFrame()
{
	 m_DebugTexture->upload(0, m_FrameBuffer->GetCUGLBuffer(m_CurDebugFrame).getHandle(), 0, 0, m_FbWidth, m_FbHeight);
	m_RenderTexture->upload(0, m_FrameBuffer->GetCUGLBuffer(m_CurRenderFrame).getHandle(), 0, 0, m_FbWidth, m_FbHeight);
	m_Renderer->draw(m_RenderTexture->getID());
}
void Test22Application::DrawGui()
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
		float lightColor[3] = {m_ParallelLight.emission.x, m_ParallelLight.emission.y, m_ParallelLight.emission.z};
		float cameraFovY = m_CameraFovY;
		float movementSpeed = m_MovementSpeed;
		float mouseSensitivity = m_MouseSensitity;
		if (!m_LockUpdate)
		{
			if (ImGui::SliderFloat3("LightColor", lightColor, 0.0f, 10.0f))
			{
				m_ParallelLight.emission = make_float3(lightColor[0], lightColor[1], lightColor[2]);
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
						m_ParallelLight.emission = make_float3(
							configJson["light"]["color"][0].get<float>(),
							configJson["light"]["color"][1].get<float>(),
							configJson["light"]["color"][2].get<float>());
						auto camera = rtlib::ext::Camera(eye, lookAt, vUp, m_CameraFovY, m_FbAspect);
						glfwSetWindowSize(m_Window, m_FbWidth, m_FbHeight);
						m_CameraController.SetMouseSensitivity(m_MouseSensitity);
						m_CameraController.SetMovementSpeed(m_MovementSpeed);
						m_CameraController.SetCamera(camera);
						m_ChangeTrace  = true;
						m_UpdateCamera = true;
						m_UpdateLight  = true;
						m_FlushFrame   = true;
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
				std::ofstream configFile(filePath);
				if (!configFile.fail())
				{
					configFile << configJson;
				}
				configFile.close();
			}
		}
		int isLockedUpdate = m_LockUpdate;

		if (ImGui::RadioButton("UnLockUpdate", &isLockedUpdate, 0))
		{
			m_LockUpdate = false;
		}
		ImGui::SameLine();
		if (ImGui::RadioButton("  LockUpdate", &isLockedUpdate, 1))
		{
			m_LockUpdate = true;
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
		if (!m_LockUpdate)
		{
			int maxTraceDepth = m_MaxTraceDepth;
			if (ImGui::SliderInt("maxTraceDepth", &maxTraceDepth, 1, 10))
			{
				m_MaxTraceDepth = maxTraceDepth;
				m_ChangeTrace   = true;
			}
			int samplePerLaunch = m_SamplePerLaunch;
			if (ImGui::SliderInt("samplePerLaunch", &samplePerLaunch, 1, 50))
			{
				m_SamplePerLaunch = samplePerLaunch;
			}
		}
		else
		{
			ImGui::Text("  maxTraceDepth: %d", m_MaxTraceDepth);
			ImGui::Text("samplePerLaunch: %d", m_SamplePerLaunch);
		}

		int samplePerBudget = m_SamplePerBudget;
		if (ImGui::SliderInt("samplePerBudget", &samplePerBudget, 1, 100000))
		{
			m_SamplePerBudget = samplePerBudget;
		}
		if (ImGui::InputText("ImgRootDir", m_ImgRenderPath.data(), m_ImgRenderPath.size()))
		{
		}
		if (!m_TraceNEE)
		{
			if (ImGui::Button("NEE"))
			{
				m_TraceNEE = true;
				m_FlushFrame = true;
			}

			ImGui::SameLine();
		}
		else {
			if (ImGui::Button("Def"))
			{
				m_TraceNEE = false;
				m_FlushFrame = true;
			}

			ImGui::SameLine();
		}
		if (ImGui::Button("Save"))
		{
			std::filesystem::path imgRenderPath = std::string(m_ImgRenderPath.data());
			if (std::filesystem::exists(imgRenderPath))
			{
				std::vector<uchar4> imageData(m_FbWidth * m_FbHeight);
				{
					std::unique_ptr<uchar4[]> imagePixels(new uchar4[m_FbWidth * m_FbHeight]);
					m_RenderTexture->bind();
					glGetTexImage(m_RenderTexture->getTarget(), 0, GL_RGBA, GL_UNSIGNED_BYTE, imagePixels.get());
					m_RenderTexture->unbind();

					for (int i = 0; i < m_FbHeight; ++i)
					{
						std::memcpy(imageData.data() + (m_FbHeight - 1 - i) * m_FbWidth, imagePixels.get() + i * m_FbWidth, sizeof(uchar4) * m_FbWidth);
					}
				}
				auto savePath = imgRenderPath / (m_TraceNEE ? "Trace_NEE_" : "Trace_Def_");
				savePath += std::to_string(m_SamplePerALL);
				savePath += ".png";
				stbi_write_png(savePath.string().c_str(), m_FbWidth, m_FbHeight, 4, imageData.data(), m_FbWidth * 4);
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
				if (i != 4 && i != 7)
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
			std::vector<uchar4> imageData(m_FbWidth * m_FbHeight);
			{
				std::unique_ptr<uchar4[]> imagePixels(new uchar4[m_FbWidth * m_FbHeight]);
				m_DebugTexture->bind();
				glGetTexImage(m_DebugTexture->getTarget(), 0, GL_RGBA, GL_UNSIGNED_BYTE, imagePixels.get());
				m_DebugTexture->unbind();

				for (int i = 0; i < m_FbHeight; ++i)
				{
					std::memcpy(imageData.data() + (m_FbHeight - 1 - i) * m_FbWidth, imagePixels.get() + i * m_FbWidth, sizeof(uchar4) * m_FbWidth);
				}
			}
			auto filePath = imgDebugPath / ("Debug_" + m_CurDebugFrame);
			if (m_CurDebugFrame == "STree")
			{
				filePath += "_SppBudget" + std::to_string(m_SamplePerBudget) + "_Spp_" + std::to_string(m_SampleForPrvDbg);
			}
			filePath += ".png";

			stbi_write_png(filePath.string().c_str(), m_FbWidth, m_FbHeight, 4, imageData.data(), m_FbWidth * 4);
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
void Test22Application::PollEvents()
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
void Test22Application::Update()
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
	if (m_FlushFrame || m_ResizeFrame || m_UpdateCamera || m_UpdateLight||m_ChangeTrace)
	{
		m_SamplePerALL = 0;
	}
	m_SimpleActor->Update();
	m_DebugActor->Update();
	m_NEEActor->Update();
	m_ChangeTrace  = false;
	m_UpdateCamera = false;
	m_UpdateLight  = false;
	m_ResizeFrame  = false;
	m_FlushFrame   = false;
}
void Test22Application::LockUpdate()
{
	m_LockUpdate = true;
}
void Test22Application::UnLockUpdate()
{
	m_LockUpdate = false;
}
//Free
void Test22Application::FreeGLFW()
{
	glfwTerminate();
}

void Test22Application::FreeWindow()
{
	glfwDestroyWindow(m_Window);
	m_Window = nullptr;
}

void Test22Application::FreeCUDA()
{
	m_Context.reset();
}

void Test22Application::FreeGui()
{
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();
	//Renderer
	m_Renderer->reset();
	m_Renderer.reset();
}

void Test22Application::FreePipelines()
{
	m_HGProgramGroups.clear();
	m_MSProgramGroups.clear();
	m_RGProgramGroups.clear();
	m_Modules.clear();
	m_Pipelines.clear();
}

void Test22Application::FreeAssets()
{
	m_TextureAssets.Reset();
	m_ObjModelAssets.Reset();
}

void Test22Application::FreeAccelerationStructures()
{
	m_GASHandles.clear();
	m_IASHandles.clear();
}

void Test22Application::FreeLight()
{
	m_ParallelLight = {};
}

void Test22Application::FreeCamera()
{
	m_CameraController = {};
}

void Test22Application::FreeFrameResources()
{
	m_RenderTexture->reset();
	m_FrameBuffer->CleanUp();
}

void Test22Application::FreeTracers()
{
	m_DebugActor->CleanUp();
	m_SimpleActor->CleanUp();
	m_NEEActor->CleanUp();
}

auto Test22Application::GetOPXContext() const -> std::shared_ptr<rtlib::OPXContext>
{
	return m_Context;
}

auto Test22Application::GetOPXPipeline(const std::string &name) -> rtlib::OPXPipeline &
{
	return m_Pipelines.at(name);
}

auto Test22Application::GetRGProgramGroup(const std::string &name) -> rtlib::OPXRaygenPG &
{
	return m_RGProgramGroups.at(name);
}

auto Test22Application::GetMSProgramGroup(const std::string &name) -> rtlib::OPXMissPG &
{
	return m_MSProgramGroups.at(name);
}

auto Test22Application::GetHGProgramGroup(const std::string &name) -> rtlib::OPXHitgroupPG &
{
	return m_HGProgramGroups.at(name);
}

auto Test22Application::GetTLAS() const -> rtlib::ext::IASHandlePtr
{
	return m_IASHandles.at("TopLevel");
}

auto Test22Application::GetMaterials() const -> const std::vector<rtlib::ext::VariableMap> &
{
	// TODO: return �X�e�[�g�����g�������ɑ}�����܂�
	return m_Materials;
}

auto Test22Application::GetCamera() const -> rtlib::ext::Camera
{
	return m_CameraController.GetCamera(m_CameraFovY, m_FbAspect);
}

auto Test22Application::GetLight() const -> ParallelLight
{
	return m_ParallelLight;
}

auto Test22Application::GetTexture(const std::string &name) const -> const rtlib::CUDATexture2D<uchar4> &
{
	// TODO: return �X�e�[�g�����g�������ɑ}�����܂�
	return m_TextureAssets.GetAsset(name);
}
