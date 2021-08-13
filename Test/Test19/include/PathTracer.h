#ifndef PATH_TRACER_H
#define PATH_TRACER_H
#include <RTLib/Optix.h>
#include <RTLib/CUDA.h>
#include <RTLib/Camera.h>
#include <RTLib/ext/TraversalHandle.h>
#include <cuda/RayTrace.h>
#include <fstream>
#include <memory>
#include <string>
#include <stb_image.h>
#include <stb_image_write.h>
#include "../include/SceneBuilder.h"
namespace test {
	template<typename Param_t>
	struct Pipeline {
	public:
		template<typename T>
		using UploadBuffer = rtlib::CUDAUploadBuffer<T>;
		template<typename T>
		using UploadBufferMap = std::unordered_map<std::string, rtlib::CUDAUploadBuffer<T>>;
		struct ProgramDesc {
			std::string moduleName;
			std::string programName;
		};
		struct PipelineSubPass
		{
			OptixShaderBindingTable shaderBindingTable = {};
			UploadBuffer<Param_t>   paramsBuffer = {};
			int                     depth = 0;
		};
		using OPXModuleMap       = std::unordered_map<std::string, rtlib::OPXModule>;
		using PipelineSubPassMap = std::unordered_map<std::string, PipelineSubPass>;
		using RayGRecord         = rtlib::SBTRecord<RayGenData>;
		using MissRecord         = rtlib::SBTRecord<MissData>;
		using HitGRecord         = rtlib::SBTRecord<HitgroupData>;
		using MissPGMap          = std::unordered_map<std::string, rtlib::OPXMissPG>;
		using HitGPGMap          = std::unordered_map<std::string, rtlib::OPXHitgroupPG>;
		using OPXContextPtr      = std::shared_ptr <rtlib::OPXContext >;
		
		OPXContextPtr                     context            = nullptr;
		int								  width				 = 0;
		int 							  height             = 0;
		rtlib::OPXPipeline                pipeline           = {};
		OPXModuleMap                      modules            = {};
		rtlib::OPXRaygenPG                rgProgramGroup     = {};
		MissPGMap                         msProgramGroups    = {};
		HitGPGMap                         hgProgramGroups    = {};
		UploadBufferMap<RayGRecord>	      rayGBuffers		 = {};
		UploadBufferMap<MissRecord>	      missBuffers		 = {};
		UploadBufferMap<HitGRecord>	      hitGBuffers		 = {};
		OptixPipelineCompileOptions       compileOptions     = {};
		OptixPipelineLinkOptions          linkOptions        = {};
		PipelineSubPassMap                subPasses          = {};
		bool                              updateRG           = false;
		bool                              updateMS           = false;
		bool                              updateHG           = false;
		bool                              updatePm           = false;
	public:
		void SetContext(const OPXContextPtr& context_)
		{
			context = context_;
		}
		void InitPipeline(const OptixPipelineCompileOptions& pipelineCompileOptions)
		{
			pipeline = context->createPipeline(pipelineCompileOptions);
			compileOptions = pipelineCompileOptions;
		}
		void LinkPipeline(const OptixPipelineLinkOptions&    pipelineLinkOptions) {
			this->pipeline.link(pipelineLinkOptions);
			linkOptions    = pipelineLinkOptions;
		}
		bool LoadModuleFromPtxFile(const std::string& name,const std::string& ptxFilePath, const OptixModuleCompileOptions& moduleCompileOptions)
		{
			auto ptxFile   = std::ifstream(ptxFilePath, std::ios::binary);
			if ( ptxFile.fail()) {
				return false;
			}
			auto ptxSource = std::string((std::istreambuf_iterator<char>(ptxFile)), (std::istreambuf_iterator<char>()));
			ptxFile.close();
			try {
				auto newModule      = this->pipeline.createModule(ptxSource, moduleCompileOptions);
				this->modules[name] = newModule;
			}
			catch (rtlib::OptixException& err) {
				std::cout << err.what() << std::endl;
				return false;
			}
			return true;
		}
		bool LoadRgProgramGroupFromModule(const ProgramDesc& rgDesc) {
			try {
				auto& rgModule = modules.at(rgDesc.moduleName);
				rgProgramGroup = pipeline.createRaygenPG({ rgModule,rgDesc.programName.c_str() });
			}
			catch (...) {
				return false;
			}
			return true;
		}
		bool LoadMsProgramGroupFromModule(const std::string& pgName,
			                              const ProgramDesc& msDesc) {
			try {
				auto& msModule = modules.at(msDesc.moduleName);
				auto  msProgramGroup = pipeline.createMissPG({ msModule,msDesc.programName.c_str() });
				msProgramGroups[pgName] = msProgramGroup;
			}
			catch (...) {
				return false;
			}
			return true;
		}
		bool LoadHgProgramGroupFromModule(const std::string& pgName,
			                              const ProgramDesc& chDesc,
			                              const ProgramDesc& ahDesc,
			                              const ProgramDesc& isDesc) {
			try {
				rtlib::OPXProgramDesc opxChDesc;
				if (!chDesc.programName.empty() && !chDesc.moduleName.empty()) {
					opxChDesc.module = modules[chDesc.moduleName];
					opxChDesc.entryName = chDesc.programName.c_str();
				}
				rtlib::OPXProgramDesc opxAhDesc;
				if (!ahDesc.programName.empty() && !ahDesc.moduleName.empty()) {
					opxAhDesc.module = modules[ahDesc.moduleName];
					opxAhDesc.entryName = ahDesc.programName.c_str();
				}
				rtlib::OPXProgramDesc opxIsDesc;
				if (!isDesc.programName.empty() && !isDesc.moduleName.empty()) {
					opxIsDesc.module = modules[isDesc.moduleName];
					opxIsDesc.entryName = isDesc.programName.c_str();
				}
				auto hgProgramGroup = pipeline.createHitgroupPG(
					opxChDesc, opxAhDesc, opxIsDesc
				);
				hgProgramGroups[pgName] = hgProgramGroup;
			}
			catch (...) {
				return false;
			}
			return true;
		}
		void InitShaderBindingTable(const std::string& subPassName, const std::string& rgRecName, const std::string& msRecName, const std::string& hgRecName)
		{
			auto& sbt         = subPasses[subPassName].shaderBindingTable;
			auto& rayGBuffer  = rayGBuffers.at(rgRecName);
			auto& missBuffer  = missBuffers.at(msRecName);
			auto& hitGBuffer  = hitGBuffers.at(hgRecName);
			sbt.raygenRecord                = reinterpret_cast<CUdeviceptr>(rayGBuffer.gpuHandle.getDevicePtr());
			sbt.missRecordBase              = reinterpret_cast<CUdeviceptr>(missBuffer.gpuHandle.getDevicePtr());
			sbt.missRecordCount             = missBuffer.cpuHandle.size();
			sbt.missRecordStrideInBytes     = sizeof(missBuffer.cpuHandle[0]);
			sbt.hitgroupRecordBase          = reinterpret_cast<CUdeviceptr>(hitGBuffer.gpuHandle.getDevicePtr());
			sbt.hitgroupRecordCount         = hitGBuffer.cpuHandle.size();
			sbt.hitgroupRecordStrideInBytes = sizeof(hitGBuffer.cpuHandle[0]);
		}
		void Launch(const std::string& sbtName,CUstream stream)noexcept {
			auto& subPass = this->subPasses.at(sbtName);
			this->pipeline.launch(stream, subPass.paramsBuffer.gpuHandle.getDevicePtr(), subPass.shaderBindingTable, this->width, this->height, subPass.depth);
		}
		void Update()noexcept
		{
			if (updateRG)
			{
				for (auto& [name, rayGBuffer] : rayGBuffers) {
					rayGBuffer.Upload();
				}
				updateRG = false;
			}
			if (updateMS)
			{
				for (auto& [name, missBuffer] : missBuffers) {
					missBuffer.Upload();
				}
				updateMS = false;
			}
			if (updateHG)
			{
				for (auto& [name, hitGBuffer] : hitGBuffers) {
					hitGBuffer.Upload();
				}
				updateHG = false;
			}
			if (updatePm)
			{
				for (auto& [name, subPass] : subPasses) {
					subPass.paramsBuffer.Upload();
				}
				updatePm = false;
			}
		}
	};
	class  PathTracer {
	private:
		using OPXContextPtr  = std::shared_ptr <rtlib::OPXContext >;
		using TextureMap     = std::unordered_map<std::string, rtlib::CUDATexture2D<uchar4>>;
	private:
		using GASHandleMap    = std::unordered_map < std::string, std::shared_ptr <rtlib::ext::GASHandle>> ;
		using IASHandleMap    = std::unordered_map<std::string, std::shared_ptr<rtlib::ext::IASHandle>>;
		using TracePipelinePtr= std::shared_ptr<Pipeline<RayTraceParams>>;
		using DebugPipelinePtr= std::shared_ptr<Pipeline<RayDebugParams>>;
	public:
		OPXContextPtr      m_OPXContext    = {};
		GASHandleMap       m_GASHandles    = {};
		IASHandleMap       m_IASHandles    = {};
		TextureMap         m_Textures      = {};
		TracePipelinePtr   m_TracePipeline = {};
		DebugPipelinePtr   m_DebugPipeline = {};
	public:
		void InitCUDA();
		void InitOPX();
	public:
		auto GetOPXContext()const -> const OPXContextPtr&;
		void SetGASHandle(const std::string& keyName, const std::shared_ptr<rtlib::ext::GASHandle>& gasHandle);
		void SetIASHandle(const std::string& keyName, const std::shared_ptr<rtlib::ext::IASHandle>& iasHandle);
		auto GetInstance( const std::string& gasKeyName)const->rtlib::ext::Instance;
		void LoadTexture( const std::string& keyName, const std::string& texPath);
		auto GetTexture(  const std::string& keyName) const ->const rtlib::CUDATexture2D<uchar4>&;
		bool HasTexture(  const std::string& keyName) const noexcept;
		void SetTracePipeline(const TracePipelinePtr& tracePipeline);
		void SetDebugPipeline(const DebugPipelinePtr& debugPipeline);
		auto GetTracePipeline()const->TracePipelinePtr;
		auto GetDebugPipeline()const->DebugPipelinePtr;
	};
}
#endif