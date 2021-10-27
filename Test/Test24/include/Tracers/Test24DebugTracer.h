#ifndef TEST_TEST24_DEBUG_TRACER_H
#define TEST_TEST24_DEBUG_TRACER_H
#include <TestLib/RTTracer.h>
#include <TestLib/RTGui.h>
#include <TestLib/RTFrameBuffer.h>
#include <TestLib/Assets/ObjAssets.h>
#include <TestLib/Assets/ImgAssets.h>
#include <RTLib/Optix.h>
#include <RTLib/CUDA.h>
#include <RTLib/ext/TraversalHandle.h>
#include <cuda/Test24DebugTracerDeclare.h>
#include <unordered_map>
#include <memory>
#include <string>
namespace test
{
	namespace tracers
	{
		class Test24DebugTracer : public RTTracer
		{
		private:
			using ContextPtr        = std::shared_ptr<rtlib::OPXContext>;
			using GuiPtr            = test::RTGuiPtr;
			using IASHandlePtr      = rtlib::ext::IASHandlePtr;
			using FrameBufferPtr    = test::RTFrameBufferPtr;
			using Pipeline          = rtlib::OPXPipeline;
			using Params            = test24::tracers::test24_debug::Params;
			using ParamsBuffer      = rtlib::CUDAUploadBuffer<Params>;
			using ModuleMap         = std::unordered_map<std::string, rtlib::OPXModule>;
			using RgProgramGroupMap = std::unordered_map<std::string, rtlib::OPXRaygenPG>;
			using MsProgramGroupMap = std::unordered_map<std::string, rtlib::OPXMissPG>;
			using HgProgramGroupMap = std::unordered_map<std::string, rtlib::OPXHitgroupPG>;
			using RgData			= test24::tracers::test24_debug::RgData;
			using MsData		    = test24::tracers::test24_debug::MsData;
			using HgData			= test24::tracers::test24_debug::HgData;
			using RgRecord          = rtlib::SBTRecord<RgData>;
			using MsRecord          = rtlib::SBTRecord<MsData>;
			using HgRecord          = rtlib::SBTRecord<HgData>;
			using RgRecordBuffer    = rtlib::CUDAUploadBuffer<RgRecord>;
			using MsRecordBuffer    = rtlib::CUDAUploadBuffer<MsRecord>;
			using HgRecordBuffer    = rtlib::CUDAUploadBuffer<HgRecord>;
			using ShaderTable       = OptixShaderBindingTable;
			using ObjAssetManagerPtr= test::assets::ObjAssetManagerPtr;
			using ImgAssetManagerPtr= test::assets::ImgAssetManagerPtr;
		public:
			struct UserData
			{
				uchar4* diffuseBuffer;
				uchar4* specularBuffer;
				uchar4* emissionBuffer;
				uchar4* shinnessBuffer;
				uchar4* transmitBuffer;
				uchar4* normalBuffer;
				uchar4* depthBuffer;
				uchar4* texCoordBuffer;
				uchar4* sTreeColBuffer;
			};
		public:
			Test24DebugTracer(ContextPtr context, GuiPtr gui, IASHandlePtr tlas, ObjAssetManagerPtr objAssetManager, ImgAssetManagerPtr imgAssetManager);
			// RTTracer ÇâÓÇµÇƒåpè≥Ç≥ÇÍÇ‹ÇµÇΩ
			virtual void Initialize() override;
			virtual void Launch(const RTTraceConfig& config) override;
			virtual void CleanUp() override;
			virtual void Update() override;
			virtual ~Test24DebugTracer() {}
		private:
			void InitPipeline();
			void InitShaderTable();
			void InitLaunchParams();
			void FreePipeline();
			void FreeShaderTable();
			void FreeLaunchParams();
		private:
			ContextPtr         m_Context         = nullptr;
			GuiPtr             m_Gui             = {};
			IASHandlePtr       m_Tlas            = nullptr;
			ObjAssetManagerPtr m_ObjAssetManager = {};
			ImgAssetManagerPtr m_ImgAssetManager = {};
			Pipeline           m_Pipeline        = {};
			ModuleMap          m_Modules         = {};
			RgProgramGroupMap  m_RgProgramGroups = {};
			MsProgramGroupMap  m_MsProgramGroups = {};
			HgProgramGroupMap  m_HgProgramGroups = {};
			ParamsBuffer       m_ParamsBuffer    = {};
			RgRecordBuffer     m_RgRecordBuffer  = {};
			MsRecordBuffer     m_MsRecordBuffer  = {};
			HgRecordBuffer     m_HgRecordBuffer  = {};
			ShaderTable        m_ShaderTable     = {};
		};
	}
}
#endif