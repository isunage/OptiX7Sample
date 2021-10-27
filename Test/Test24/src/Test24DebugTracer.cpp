#include "../include/Tracers/Test24DebugTracer.h"
#include <RTLib/VectorFunction.h>
#include <RTLib/ext/Mesh.h>
#include <RTLib/ext/Resources/CUDA.h>
#include <Test24Config.h>
#include <fstream>

test::tracers::Test24DebugTracer::Test24DebugTracer(ContextPtr context, GuiPtr gui, IASHandlePtr tlas, ObjAssetManagerPtr objAssetManager, ImgAssetManagerPtr imgAssetManager)
{
	m_Context = context;
	m_Gui = gui;
	m_Tlas = tlas;
	m_ObjAssetManager = objAssetManager;
	m_ImgAssetManager = imgAssetManager;
}

void test::tracers::Test24DebugTracer::Initialize()
{
	this->InitPipeline();
	this->InitShaderTable();
	this->InitLaunchParams();
}

void test::tracers::Test24DebugTracer::Launch(const RTTraceConfig& config)
{
	UserData* pUserData = (UserData*)config.pUserData;
	if (!pUserData)
	{
		return;
	}
	m_ParamsBuffer.cpuHandle[0].width          = config.width;
	m_ParamsBuffer.cpuHandle[0].height         = config.height;
	m_ParamsBuffer.cpuHandle[0].diffuseBuffer  = pUserData->diffuseBuffer;
	m_ParamsBuffer.cpuHandle[0].specularBuffer = pUserData->specularBuffer;
	m_ParamsBuffer.cpuHandle[0].emissionBuffer = pUserData->emissionBuffer;
	m_ParamsBuffer.cpuHandle[0].shinnessBuffer = pUserData->shinnessBuffer;
	m_ParamsBuffer.cpuHandle[0].transmitBuffer = pUserData->transmitBuffer;
	m_ParamsBuffer.cpuHandle[0].normalBuffer   = pUserData->normalBuffer;
	m_ParamsBuffer.cpuHandle[0].depthBuffer    = pUserData->depthBuffer;
	m_ParamsBuffer.cpuHandle[0].texCoordBuffer = pUserData->texCoordBuffer;
	m_ParamsBuffer.cpuHandle[0].sTreeColBuffer = pUserData->sTreeColBuffer;
	cudaMemcpyAsync(m_ParamsBuffer.gpuHandle.getDevicePtr(), &m_ParamsBuffer.cpuHandle[0], sizeof(Params), cudaMemcpyHostToDevice, config.stream);
	m_Pipeline.launch(config.stream, m_ParamsBuffer.gpuHandle.getDevicePtr(), m_ShaderTable, config.width, config.height, config.depth);
	if (config.isSync)
	{
		RTLIB_CU_CHECK(cuStreamSynchronize(config.stream));
	}
}

void test::tracers::Test24DebugTracer::CleanUp()
{
	this->FreeLaunchParams();
	this->FreeShaderTable();
	this->FreePipeline();
}

void test::tracers::Test24DebugTracer::Update()
{
}
void test::tracers::Test24DebugTracer::InitPipeline()
{
	auto debugTracerPtxFile = std::ifstream(TEST_TEST24_CUDA_PATH "/Test24DebugTracerKernel.ptx", std::ios::binary);
	if (!debugTracerPtxFile.is_open())
		throw std::runtime_error("Failed To Load Test24DebugTracerKernel.ptx!");
	auto rayDebugPtxData = std::string((std::istreambuf_iterator<char>(debugTracerPtxFile)), (std::istreambuf_iterator<char>()));
	debugTracerPtxFile.close();

	auto debugCompileOptions = OptixPipelineCompileOptions{};
	{
		debugCompileOptions.pipelineLaunchParamsVariableName = "params";
		debugCompileOptions.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_TRIANGLE;
		debugCompileOptions.traversableGraphFlags  = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
		debugCompileOptions.usesMotionBlur         = false;
		debugCompileOptions.numAttributeValues     = 3;
		debugCompileOptions.numPayloadValues       = 8;
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
		debugModuleOptions.optLevel   = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
#else
		debugModuleOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
		debugModuleOptions.optLevel   = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
#endif
		debugModuleOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
	}
	m_Pipeline = m_Context->createPipeline(debugCompileOptions);
	m_Modules["DebugTracer.Kernel"] = m_Pipeline.createModule(rayDebugPtxData, debugModuleOptions);
	m_RgProgramGroups["DebugTracer.RG.Default"] = m_Pipeline.createRaygenPG(  { m_Modules["DebugTracer.Kernel"], "__raygen__debug" });
	m_MsProgramGroups["DebugTracer.MS.Default"] = m_Pipeline.createMissPG(    { m_Modules["DebugTracer.Kernel"], "__miss__debug" });
	m_HgProgramGroups["DebugTracer.HG.Default"] = m_Pipeline.createHitgroupPG({ m_Modules["DebugTracer.Kernel"], "__closesthit__debug" }, {}, {});
	m_Pipeline.link(debugLinkOptions);
}
void test::tracers::Test24DebugTracer::InitShaderTable()
{
	m_RgRecordBuffer.cpuHandle.resize(1);
	m_RgRecordBuffer.cpuHandle[0]                 = m_RgProgramGroups["DebugTracer.RG.Default"].getSBTRecord<RgData>();
	m_RgRecordBuffer.cpuHandle[0].data.camera_eye = m_Gui->GetFloat3As<float3>("Camera.Eye");
	m_RgRecordBuffer.cpuHandle[0].data.camera_u   = m_Gui->GetFloat3As<float3>("Camera.U");
	m_RgRecordBuffer.cpuHandle[0].data.camera_v   = m_Gui->GetFloat3As<float3>("Camera.V");
	m_RgRecordBuffer.cpuHandle[0].data.camera_w   = m_Gui->GetFloat3As<float3>("Camera.W");
	m_RgRecordBuffer.Upload();
	m_MsRecordBuffer.cpuHandle.resize(RAY_TYPE_COUNT);
	m_MsRecordBuffer.cpuHandle[RAY_TYPE_RADIANCE] = m_MsProgramGroups["DebugTracer.MS.Default"].getSBTRecord<MsData>();
	m_MsRecordBuffer.cpuHandle[RAY_TYPE_OCCLUDED] = m_MsProgramGroups["DebugTracer.MS.Default"].getSBTRecord<MsData>();
	m_MsRecordBuffer.Upload();
	m_HgRecordBuffer.cpuHandle.resize(m_Tlas->GetSbtCount() * RAY_TYPE_COUNT);
	{
		size_t sbtOffset = 0;
		for (auto& instanceSet : m_Tlas->GetInstanceSets())
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
						auto& material  = test::assets::ObjAsset::As(m_ObjAssetManager->GetAsset(mesh->GetSharedResource()->variables.GetString("objAssetName")))->GetMaterials()[materialId];
						HgData radianceHgData = {};
						{
							radianceHgData.vertexBuffer = cudaVertexBuffer->GetHandle().getDevicePtr();
							radianceHgData.normalBuffer = cudaNormalBuffer->GetHandle().getDevicePtr();
							radianceHgData.texCrdBuffer = cudaTexCrdBuffer->GetHandle().getDevicePtr();
							radianceHgData.triIndBuffer = cudaTriIndBuffer->GetHandle().getDevicePtr();
							auto diffImgAsset           = test::assets::ImgAsset::As(m_ImgAssetManager->GetAsset(material.GetString("diffTex")));
							auto specImgAsset           = test::assets::ImgAsset::As(m_ImgAssetManager->GetAsset(material.GetString("specTex")));
							auto emitImgAsset           = test::assets::ImgAsset::As(m_ImgAssetManager->GetAsset(material.GetString("emitTex")));
							auto shinImgAsset           = test::assets::ImgAsset::As(m_ImgAssetManager->GetAsset(material.GetString("shinTex")));
							radianceHgData.diffTex      = diffImgAsset->GetImage2D().GetGpuComponent<rtlib::ext::resources::CUDATextureImage2DComponent<uchar4>>("CUDATexture")->GetHandle().getHandle();
							radianceHgData.specTex      = specImgAsset->GetImage2D().GetGpuComponent<rtlib::ext::resources::CUDATextureImage2DComponent<uchar4>>("CUDATexture")->GetHandle().getHandle();
							radianceHgData.emitTex      = emitImgAsset->GetImage2D().GetGpuComponent<rtlib::ext::resources::CUDATextureImage2DComponent<uchar4>>("CUDATexture")->GetHandle().getHandle();
							radianceHgData.shinTex      = shinImgAsset->GetImage2D().GetGpuComponent<rtlib::ext::resources::CUDATextureImage2DComponent<uchar4>>("CUDATexture")->GetHandle().getHandle();
							radianceHgData.diffCol      = material.GetFloat3As<float3>("diffCol");
							radianceHgData.specCol      = material.GetFloat3As<float3>("specCol");
							radianceHgData.emitCol      = material.GetFloat3As<float3>("emitCol");
							radianceHgData.shinness     = material.GetFloat1("shinness");
							radianceHgData.transmit     = material.GetFloat3As<float3>("tranCol");
							radianceHgData.ior          = material.GetFloat1("refrIndx");
						}
						m_HgRecordBuffer.cpuHandle[RAY_TYPE_COUNT * sbtOffset + RAY_TYPE_COUNT * i + RAY_TYPE_RADIANCE] = m_HgProgramGroups["DebugTracer.HG.Default"].getSBTRecord<HgData>(radianceHgData);
						m_HgRecordBuffer.cpuHandle[RAY_TYPE_COUNT * sbtOffset + RAY_TYPE_COUNT * i + RAY_TYPE_OCCLUDED] = m_HgProgramGroups["DebugTracer.HG.Default"].getSBTRecord<HgData>({});
					}
					sbtOffset += mesh->GetUniqueResource()->materials.size();
				}
			}
		}
	}
	m_HgRecordBuffer.Upload();
	m_ShaderTable.raygenRecord    = reinterpret_cast<CUdeviceptr>(m_RgRecordBuffer.gpuHandle.getDevicePtr());
	m_ShaderTable.missRecordBase  = reinterpret_cast<CUdeviceptr>(m_MsRecordBuffer.gpuHandle.getDevicePtr());
	m_ShaderTable.missRecordCount         = m_MsRecordBuffer.cpuHandle.size();
	m_ShaderTable.missRecordStrideInBytes = sizeof(MsRecord);
	m_ShaderTable.hitgroupRecordBase          = reinterpret_cast<CUdeviceptr>(m_HgRecordBuffer.gpuHandle.getDevicePtr());
	m_ShaderTable.hitgroupRecordCount         = m_HgRecordBuffer.cpuHandle.size();
	m_ShaderTable.hitgroupRecordStrideInBytes = sizeof(HgRecord);
}
void test::tracers::Test24DebugTracer::InitLaunchParams()
{
	m_ParamsBuffer.cpuHandle.resize(1);
	m_ParamsBuffer.cpuHandle[0].traversalHandle = m_Tlas->GetHandle();
	m_ParamsBuffer.cpuHandle[0].width           = 0;
	m_ParamsBuffer.cpuHandle[0].height          = 0;
	m_ParamsBuffer.cpuHandle[0].diffuseBuffer   = nullptr;
	m_ParamsBuffer.cpuHandle[0].specularBuffer  = nullptr;
	m_ParamsBuffer.cpuHandle[0].shinnessBuffer  = nullptr;
	m_ParamsBuffer.cpuHandle[0].emissionBuffer  = nullptr;

	m_ParamsBuffer.cpuHandle[0].transmitBuffer  = nullptr;
	m_ParamsBuffer.cpuHandle[0].normalBuffer    = nullptr;
	m_ParamsBuffer.cpuHandle[0].depthBuffer     = nullptr;
	m_ParamsBuffer.cpuHandle[0].texCoordBuffer  = nullptr;

	m_ParamsBuffer.cpuHandle[0].sTreeColBuffer  = nullptr;
	m_ParamsBuffer.Upload();
}
void test::tracers::Test24DebugTracer::FreePipeline()
{
	m_RgProgramGroups.clear();
	m_MsProgramGroups.clear();
	m_HgProgramGroups.clear();
	m_Modules.clear();
	m_Pipeline = {};
}
void test::tracers::Test24DebugTracer::FreeShaderTable()
{
	m_RgRecordBuffer.Reset();
	m_MsRecordBuffer.Reset();
	m_HgRecordBuffer.Reset();
	m_ShaderTable = {};
}
void test::tracers::Test24DebugTracer::FreeLaunchParams()
{
	m_ParamsBuffer.Reset();
}