#ifndef TEST_RT_PIPELINE_H
#define TEST_RT_PIPELINE_H
#include <RTLib/Optix.h>
#include <RTLib/CUDA.h>
#include <RTLib/ext/Camera.h>
#include <RTLib/ext/TraversalHandle.h>
#include "RTRecordBuffer.h"
#include "RTSubPass.h"
#include <fstream>
#include <string>
#include <unordered_map>
namespace test{
    template<typename RayG_t, typename Miss_t, typename HitG_t, typename Params_t>
    class RTPipeline {
    public:
        struct ProgramDesc {
            std::string moduleName = "";
            std::string programName= "";
        };
    private:
        using Context                = rtlib::OPXContext;
        using ContextPtr             = std::shared_ptr<Context>;
        using Module                 = rtlib::OPXModule;
        using ModuleMap              = std::unordered_map<std::string, Module>;
        using Pipeline               = rtlib::OPXPipeline;
        using RayGProgramGroup       = rtlib::OPXRaygenPG;
        using MissProgramGroup       = rtlib::OPXMissPG;
        using HitGProgramGroup       = rtlib::OPXHitgroupPG;
        using RayGProgramGroupMap    = std::unordered_map<std::string, RayGProgramGroup>;
        using MissProgramGroupMap    = std::unordered_map<std::string, MissProgramGroup>;
        using HitGProgramGroupMap    = std::unordered_map<std::string, HitGProgramGroup>;
        using RayGRecordBuffer       = RTRecordBuffer<RayG_t>;
        using MissRecordBuffer       = RTRecordArrayBuffer<Miss_t>;
        using HitGRecordBuffer       = RTRecordArrayBuffer<HitG_t>;
        using RayGRecordBufferPtr    = std::shared_ptr<RayGRecordBuffer>;
        using MissRecordBufferPtr    = std::shared_ptr<MissRecordBuffer>;
        using HitGRecordBufferPtr    = std::shared_ptr<HitGRecordBuffer>;
        using RayGRecordBufferPtrMap = std::unordered_map<std::string, std::shared_ptr<RayGRecordBuffer>>;
        using MissRecordBufferPtrMap = std::unordered_map<std::string, std::shared_ptr<MissRecordBuffer>>;
        using HitGRecordBufferPtrMap = std::unordered_map<std::string, std::shared_ptr<HitGRecordBuffer>>;
        using SubPass                = RTSubPass<RayG_t, Miss_t, HitG_t, Params_t>;
        using SubPassPtr             = std::shared_ptr<SubPass>;
        using SubPassMap             = std::unordered_map<std::string, SubPassPtr>;
        using PipelineCompileOptions = OptixPipelineCompileOptions;
        using PipelineLinkOptions    = OptixPipelineLinkOptions;
    public:
        void Init(const ContextPtr& context, const PipelineCompileOptions& compileOptions,const PipelineLinkOptions& linkOptions) {
            m_Context = context;
            m_PipelineCompileOptions = compileOptions;
            m_PipelineLinkOptions = linkOptions;
            m_Pipeline = context->createPipeline(compileOptions);
        }
        void Link() {
            m_Pipeline.link(m_PipelineLinkOptions);
        }
        void Launch(int fbWidth, int fbHeight, const std::string& subPassName, CUstream stream = nullptr)
        {
            auto subPass = GetSubPass(subPassName);
            subPass->UploadParams();
            m_Pipeline.launch(stream, subPass->GetParamsPtr(), subPass->GetShaderBindingTable(), fbWidth, fbHeight, subPass->GetTraceCallDepth());
            RTLIB_CUDA_CHECK(cudaStreamSynchronize(stream));
        }
        //Module
        bool LoadModuleFromPtxFile(const std::string& moduleName, const std::string& ptxFilePath, const OptixModuleCompileOptions& moduleCompileOptions) {
            auto ptxFile = std::ifstream(ptxFilePath, std::ios::binary);
            if (ptxFile.fail()) {
                return false;
            }
            auto ptxSource = std::string((std::istreambuf_iterator<char>(ptxFile)), (std::istreambuf_iterator<char>()));
            ptxFile.close();
            try {
                auto newModule = this->m_Pipeline.createModule(ptxSource, moduleCompileOptions);
                this->m_Modules[moduleName] = newModule;
            }
            catch (rtlib::OptixException& err) {
                std::cout << err.what() << std::endl;
                return false;
            }
            return true;
        }
        auto GetModule(const std::string& moduleName)const -> const Module& {
            return m_Modules.at(moduleName);
        }
        //ProgramGroup
        bool LoadRayGProgramGroupFromModule(const std::string& pgName, const ProgramDesc& rgDesc) {
            try {
                auto& rgModule = GetModule(rgDesc.moduleName);
                m_RayGProgramGroups[pgName] = m_Pipeline.createRaygenPG({ rgModule,rgDesc.programName.c_str() });
            }
            catch (...) {
                return false;
            }
            return true;
        }
        bool LoadMissProgramGroupFromModule(const std::string& pgName, const ProgramDesc& msDesc) {
            try {
                auto& msModule = GetModule(msDesc.moduleName);
                m_MissProgramGroups[pgName] = m_Pipeline.createMissPG({ msModule,msDesc.programName.c_str() });
            }
            catch (...) {
                return false;
            }
            return true;
        }
        bool LoadHitGProgramGroupFromModule(const std::string& pgName, const ProgramDesc& chDesc, const ProgramDesc& ahDesc, const ProgramDesc& isDesc) {
            try {
                rtlib::OPXProgramDesc opxChDesc;
                if (!chDesc.programName.empty() && !chDesc.moduleName.empty()) {
                    opxChDesc.module    = GetModule(chDesc.moduleName);
                    opxChDesc.entryName = chDesc.programName.c_str();
                }
                rtlib::OPXProgramDesc opxAhDesc;
                if (!ahDesc.programName.empty() && !ahDesc.moduleName.empty()) {
                    opxAhDesc.module    = GetModule(ahDesc.moduleName);
                    opxAhDesc.entryName = ahDesc.programName.c_str();
                }
                rtlib::OPXProgramDesc opxIsDesc;
                if (!isDesc.programName.empty() && !isDesc.moduleName.empty()) {
                    opxIsDesc.module    = GetModule(isDesc.moduleName);
                    opxIsDesc.entryName = isDesc.programName.c_str();
                }
                auto hgProgramGroup = m_Pipeline.createHitgroupPG(
                    opxChDesc, opxAhDesc, opxIsDesc
                );
                m_HitGProgramGroups[pgName] = hgProgramGroup;

            }
            catch (...) {
                return false;
            }
            return true;
        }
        auto GetRayGProgramGroup(const std::string& pgName)const -> const RayGProgramGroup& {
            return m_RayGProgramGroups.at(pgName);
        }
        auto GetMissProgramGroup(const std::string& pgName)const -> const MissProgramGroup& {
            return m_MissProgramGroups.at(pgName);
        }
        auto GetHitGProgramGroup(const std::string& pgName)const -> const HitGProgramGroup& {
            return m_HitGProgramGroups.at(pgName);
        }
        //RecordBuffer
        void NewRayGRecordBuffer(const std::string& rayGRecordName) {
            auto rayGRecordBuffer = RayGRecordBufferPtr(new RayGRecordBuffer());
            rayGRecordBuffer->Alloc();
            m_RayGRecordBuffers[rayGRecordName] = rayGRecordBuffer;
        }
        void NewMissRecordBuffer(const std::string& missRecordName, std::size_t count) {
            auto missRecordBuffer = MissRecordBufferPtr(new MissRecordBuffer());
            missRecordBuffer->Alloc(count);
            m_MissRecordBuffers[missRecordName] = missRecordBuffer;
        }
        void NewHitGRecordBuffer(const std::string& hitGRecordName, std::size_t count) {
            auto hitGRecordBuffer = HitGRecordBufferPtr(new HitGRecordBuffer());
            hitGRecordBuffer->Alloc(count);
            m_HitGRecordBuffers[hitGRecordName] = hitGRecordBuffer;
        }
        auto GetRayGRecordBuffer(const std::string& rayGRecordName)const ->RayGRecordBufferPtr {
            return m_RayGRecordBuffers.at(rayGRecordName);
        }
        auto GetMissRecordBuffer(const std::string& missRecordName)const ->MissRecordBufferPtr {
            return m_MissRecordBuffers.at(missRecordName);
        }
        auto GetHitGRecordBuffer(const std::string& hitGRecordName)const ->HitGRecordBufferPtr {
            return m_HitGRecordBuffers.at(hitGRecordName);
        }
        void AddRayGRecordFromPG(const std::string& rayGRecordName, const std::string& pgName, const RayG_t& rayGData) {
            auto rayGRecordBuffer = GetRayGRecordBuffer(rayGRecordName);
            rayGRecordBuffer->SetRecord(GetRayGProgramGroup(pgName).getSBTRecord(rayGData));
        }
        void AddMissRecordFromPG(const std::string& missRecordName, std::size_t index, const std::string& pgName, const Miss_t& missData) {
            auto missRecordBuffer = GetMissRecordBuffer(missRecordName);
            missRecordBuffer->SetRecord(index,GetMissProgramGroup(pgName).getSBTRecord(missData));
        }
        void AddHitGRecordFromPG(const std::string& hitGRecordName, std::size_t index, const std::string& pgName, const HitG_t& hitGData) {
            auto hitGRecordBuffer = GetHitGRecordBuffer(hitGRecordName);
            hitGRecordBuffer->SetRecord(index, GetHitGProgramGroup(pgName).getSBTRecord(hitGData));
        }
        //SubPass
        void NewSubPass(const std::string& subPassName) {
            m_SubPasses[subPassName] = SubPassPtr(new SubPass());
        }
        auto GetSubPass(const std::string& subPassName)const -> SubPassPtr {
            return m_SubPasses.at(subPassName);
        }
        void AddRayGRecordBufferToSubPass(const std::string& subPassName, const std::string& rayGRecordName) {
            auto subPass          = GetSubPass(subPassName);
            auto rayGRecordBuffer = GetRayGRecordBuffer(rayGRecordName);
            subPass->SetRayGRecordBuffer(rayGRecordBuffer);
        }
        void AddMissRecordBufferToSubPass(const std::string& subPassName, const std::string& missRecordName) {
            auto subPass = GetSubPass(subPassName);
            auto missRecordBuffer = GetMissRecordBuffer(missRecordName);
            subPass->SetMissRecordBuffer(missRecordBuffer);
        }
        void AddHitGRecordBufferToSubPass(const std::string& subPassName, const std::string& hitGRecordName) {
            auto subPass = GetSubPass(subPassName);
            auto hitGRecordBuffer = GetHitGRecordBuffer(hitGRecordName);
            subPass->SetHitGRecordBuffer(hitGRecordBuffer);
        }
    private:
        ContextPtr             m_Context                = {};
        ModuleMap              m_Modules                = {};
        Pipeline               m_Pipeline               = {};
        RayGProgramGroupMap    m_RayGProgramGroups      = {};
        MissProgramGroupMap    m_MissProgramGroups      = {};
        HitGProgramGroupMap    m_HitGProgramGroups      = {};
        RayGRecordBufferPtrMap m_RayGRecordBuffers      = {};
        MissRecordBufferPtrMap m_MissRecordBuffers      = {};
        HitGRecordBufferPtrMap m_HitGRecordBuffers      = {};
        SubPassMap             m_SubPasses              = {};
        PipelineCompileOptions m_PipelineCompileOptions = {};
        PipelineLinkOptions    m_PipelineLinkOptions    = {};
    };
}
#endif