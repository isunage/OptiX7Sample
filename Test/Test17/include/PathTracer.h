#ifndef PATH_TRACER_H
#define PATH_TRACER_H
#include <RTLib/Optix.h>
#include <RTLib/CUDA.h>
#include <RTLib/ext/Camera.h>
#include <RTLib/ext/TraversalHandle.h>
#include <cuda/RayTrace.h>
#include <memory>
#include <string>
#include <stb_image.h>
#include <stb_image_write.h>
#include "../include/SceneBuilder.h"
namespace test {
	struct Pipeline {
		template<typename T>
		using UploadBuffer = rtlib::CUDAUploadBuffer<T>;
		using OPXModuleMap = std::unordered_map<std::string, rtlib::OPXModule>;
		using RayGenRecord = rtlib::SBTRecord<RayGenData>;
		using MissRecord   = rtlib::SBTRecord<MissData>;
		using HitGRecord   = rtlib::SBTRecord<HitgroupData>;
		int								  width				 = 0;
		int 							  height             = 0;
		int 							  depth              = 0;
		rtlib::OPXPipeline                pipeline           = {};
		OPXModuleMap                      modules            = {};
		rtlib::OPXRaygenPG                raygenPG           = {};
		std::vector<rtlib::OPXMissPG>     missPGs            = {};
		std::vector<rtlib::OPXHitgroupPG> hitGroupPGs        = {};
		OptixShaderBindingTable           shaderbindingTable = {};
		UploadBuffer<RayGenRecord>	      raygenBuffer		 = {};
		UploadBuffer<MissRecord>	      missBuffer		 = {};
		UploadBuffer<HitGRecord>	      hitGBuffer		 = {};
		UploadBuffer<Params>	    	  paramsBuffer       = {};
	public:
		void Launch(CUstream stream)noexcept;
	};
	class PathTracer {
	private:
		using OPXContextPtr  = std::shared_ptr <rtlib::OPXContext >;
		using TextureMap     = std::unordered_map<std::string, rtlib::CUDATexture2D<uchar4>>;
		
	private:
		using GASHandleMap = std::unordered_map < std::string, std::shared_ptr <rtlib::ext::GASHandle>> ;
		using IASHandleMap   = std::unordered_map<std::string, std::shared_ptr<rtlib::ext::IASHandle>>;
		using PipelineMap    = std::unordered_map<std::string, std::shared_ptr<Pipeline>>;
	public:
		OPXContextPtr m_OPXContext   = {};
		GASHandleMap  m_GASHandles   = {};
		IASHandleMap  m_IASHandles   = {};
		TextureMap    m_Textures     = {};
		PipelineMap   m_Pipelines    = {};
	public:
		void InitCUDA();
		void InitOPX();
	public:
		auto GetOPXContext()const -> const OPXContextPtr&;
		void SetGASHandle(const std::string& keyName, const std::shared_ptr<rtlib::ext::GASHandle>& gasHandle);
		void SetIASHandle(const std::string& keyName, const std::shared_ptr<rtlib::ext::IASHandle>& iasHandle);
		auto  GetInstance( const std::string& gasKeyName)const->rtlib::ext::Instance;
		void  LoadTexture( const std::string& keyName, const std::string& texPath);
		auto   GetTexture( const std::string& keyName) const ->const rtlib::CUDATexture2D<uchar4>&;
		bool   HasTexture( const std::string& keyName) const noexcept;
		void  SetPipeline( const std::string& keyName, const std::shared_ptr<Pipeline>& pipeline);
	};
}
#endif