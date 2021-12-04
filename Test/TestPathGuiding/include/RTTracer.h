#ifndef TEST_RT_TRACER_H
#define TEST_RT_TRACER_H
#include "RTPipeline.h"
#include <cuda/RayTrace.h>
#include <RTLib/core/Optix.h>
#include <RTLib/core/CUDA.h>
#include <RTLib/ext/TraversalHandle.h>
#include <RTLib/ext/VariableMap.h>
namespace test {
	class RTTracer
	{
	public:
        using Context = rtlib::OPXContext;
        using ContextPtr = std::shared_ptr<Context>;
		using RTTracePipeline    = RTPipeline<RayGenData, MissData, HitgroupData, RayTraceParams>;
		using RTDebugPipeline    = RTPipeline<RayGenData, MissData, HitgroupData, RayDebugParams>;
		using RTTracePipelinePtr = std::shared_ptr<RTTracePipeline>;
		using RTDebugPipelinePtr = std::shared_ptr<RTDebugPipeline>;
		using MeshGroupMap       = std::unordered_map<std::string, rtlib::ext::MeshGroupPtr>;
		using GASHandleMap       = std::unordered_map<std::string, rtlib::ext::GASHandlePtr>;
		using IASHandleMap       = std::unordered_map<std::string, rtlib::ext::IASHandlePtr>;
		using MaterialListMap    = std::unordered_map<std::string, rtlib::ext::VariableMapListPtr>;
		using TextureMap         = std::unordered_map<std::string, rtlib::CUDATexture2D<uchar4>>;
	public:
        //Context
        void SetContext(const ContextPtr& context)noexcept;
		auto GetContext()const -> const ContextPtr&;
		//MeshGroup
		void AddMeshGroup(const std::string& mgName, const rtlib::ext::MeshGroupPtr& meshGroup)noexcept;
		auto GetMeshGroup(const std::string& mgName) const->rtlib::ext::MeshGroupPtr;
		auto GetMeshGroups()const -> const MeshGroupMap&;
		//MaterialList
		void AddMaterialList(const std::string& mlName, const rtlib::ext::VariableMapListPtr& materialList)noexcept;
		auto GetMaterialList(const std::string& mlName) const->rtlib::ext::VariableMapListPtr;
		//Texture
		bool LoadTexture(const std::string& keyName, const std::string& texPath);
		auto GetTexture(const std::string& keyName) const ->const rtlib::CUDATexture2D<uchar4>&;
		bool HasTexture(const std::string& keyName) const noexcept;
        //GeometryAS
		void NewGASHandle(const std::string& gasName);
		auto GetGASHandle(const std::string& gasName)const->rtlib::ext::GASHandlePtr;
		//InstanceAS
		void NewIASHandle(const std::string& iasName);
		auto GetIASHandle(const std::string& iasName)const->rtlib::ext::IASHandlePtr;
		//Pipeline
		void SetTracePipeline(const RTTracePipelinePtr& tracePipeline)noexcept;
		void SetDebugPipeline(const RTDebugPipelinePtr& debugPipeline)noexcept;
		auto GetTracePipeline()const -> const RTTracePipelinePtr&;
		auto GetDebugPipeline()const -> const RTDebugPipelinePtr&;
		//TLAS
		void SetTLASName(const std::string& tlasName);
		auto GetTLAS()const->rtlib::ext::IASHandlePtr;
	private:
        ContextPtr         m_Context         = {};
		MeshGroupMap       m_MeshGroups      = {};
		MaterialListMap    m_MaterialLists   = {};
		TextureMap         m_Textures        = {};
		std::string        m_TLASName        = {};
		GASHandleMap       m_GASHandles      = {};
		IASHandleMap       m_IASHandles      = {};
		RTTracePipelinePtr m_TracePipeline   = {};
		RTDebugPipelinePtr m_DebugPipeline   = {};
	};
}
#endif