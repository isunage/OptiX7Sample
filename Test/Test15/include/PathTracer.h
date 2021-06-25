#ifndef PATH_TRACER_H
#define PATH_TRACER_H
#include <RTLib/Core.h>
#include <RTLib/Camera.h>
#include <cuda/RayTrace.h>
#include <memory>
#include <string>
#include "../include/SceneBuilder.h"
namespace test {
	class PathTracer {
	private:
		using OPXContextPtr  = std::shared_ptr <rtlib::OPXContext >;
		using OPXModuleMap   = std::unordered_map<std::string, rtlib::OPXModule>;
		using TextureMap     = std::unordered_map<std::string, rtlib::CUDATexture2D<uchar4>>;
		using RayGenRecord   = rtlib::SBTRecord<RayGenData>;
		using MissRecord     = rtlib::SBTRecord<MissData>;
		using HitGRecord     = rtlib::SBTRecord<HitgroupData>;
	private:
		struct Pipeline {
			rtlib::OPXPipeline                pipeline           = {};
			rtlib::OPXRaygenPG                raygenPG           = {};
			std::vector<rtlib::OPXMissPG>     missPGs            = {};
			std::vector<rtlib::OPXHitgroupPG> hitGroupPGs        = {};
			OptixShaderBindingTable           shaderbindingTable = {};
			test::ArrayBuffer<RayGenRecord>	  raygenBuffer       = {};
			test::ArrayBuffer<MissRecord>	  missBuffer         = {};
			test::ArrayBuffer<HitGRecord>	  hitGBuffer         = {};
			CUstream                          cuStream           = {};
			test::ArrayBuffer<Params>		  paramsBuffer       = {};
		};
		using  GASHandleMap   = std::unordered_map<std::string, std::shared_ptr<GASHandle>>;
		using  PipelineResMap = std::unordered_map<std::string, Pipeline>;
	public:
		OPXContextPtr    		   m_OPXContext   = {};
		OPXModuleMap			   m_OPXModules   = {};
		IASHandlePtr               m_IASHandle    = {};
		GASHandleMap               m_GASHandles   = {};
		TextureMap                 m_Textures     = {};
		PipelineResMap             m_Pipelines    = {};
	};
}
#endif