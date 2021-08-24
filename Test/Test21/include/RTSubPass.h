#ifndef TEST_RT_SUBPASS_H
#define TEST_RT_SUBPASS_H
#include "RTRecordBuffer.h"
#include <RTLib/Optix.h>
#include <RTLib/CUDA.h>
#include <cassert>
namespace test{
    template<typename RayG_t, typename Miss_t, typename HitG_t, typename Params_t>
    class RTSubPass {
    public:
        using RayGRecordBuffer = RTRecordBuffer<RayG_t>;
        using MissRecordBuffer = RTRecordArrayBuffer<Miss_t>;
        using HitGRecordBuffer = RTRecordArrayBuffer<HitG_t>;
        using ParamsBuffer     = rtlib::CUDAUploadBuffer<Params_t>;
        using RayGRecordBufferPtr = std::shared_ptr<RayGRecordBuffer>;
        using MissRecordBufferPtr = std::shared_ptr<MissRecordBuffer>;
        using HitGRecordBufferPtr = std::shared_ptr<HitGRecordBuffer>;
        using ParamsBufferPtr     = std::shared_ptr<ParamsBuffer>;
    public:
        //TraceCallDepth
        void SetTraceCallDepth(int depth){
            m_TraceCallDepth = depth;
        }
        auto GetTraceCallDepth()const -> int{
            return m_TraceCallDepth;
        }
        //Params
        void InitParams(const Params_t params){
            if(!m_ParamsBufferPtr){
                m_ParamsBufferPtr = ParamsBufferPtr(new ParamsBuffer());
            }
            m_ParamsBufferPtr->cpuHandle.resize(1);
            m_ParamsBufferPtr->cpuHandle[0] = params;
            m_ParamsBufferPtr->Upload();
        }
        void UploadParams(){
            m_ParamsBufferPtr->Upload();
        }
        auto GetParams()const ->const Params_t& {
            return m_ParamsBufferPtr->cpuHandle[0];
        }
        auto GetParams() -> Params_t& {
            return m_ParamsBufferPtr->cpuHandle[0];
        }
        auto GetParamsPtr()const -> Params_t* {
            return m_ParamsBufferPtr->gpuHandle.getDevicePtr();
        }
        //RecordBuffer
        void SetRayGRecordBuffer(const RayGRecordBufferPtr& rayGRecordBufferPtr){
            m_RayGRecordBufferPtr = rayGRecordBufferPtr;
        }
        void SetMissRecordBuffer(const MissRecordBufferPtr& missRecordBufferPtr){
            m_MissRecordBufferPtr = missRecordBufferPtr;
        }
        void SetHitGRecordBuffer(const HitGRecordBufferPtr& hitGRecordBufferPtr){
            m_HitGRecordBufferPtr = hitGRecordBufferPtr;
        }
        //ShaderBindigTable
        void InitShaderBindingTable(){
            m_ShaderBindingTable.raygenRecord                = reinterpret_cast<CUdeviceptr>(m_RayGRecordBufferPtr->GetDevicePtr());
            m_ShaderBindingTable.missRecordBase              = reinterpret_cast<CUdeviceptr>(m_MissRecordBufferPtr->GetDevicePtr());
            m_ShaderBindingTable.missRecordStrideInBytes     = sizeof(MissRecordBuffer::DataRecord);
            m_ShaderBindingTable.missRecordCount             = m_MissRecordBufferPtr->GetSize();
            m_ShaderBindingTable.hitgroupRecordBase          = reinterpret_cast<CUdeviceptr>(m_HitGRecordBufferPtr->GetDevicePtr());
            m_ShaderBindingTable.hitgroupRecordCount         = m_HitGRecordBufferPtr->GetSize();
            m_ShaderBindingTable.hitgroupRecordStrideInBytes = sizeof(HitGRecordBuffer::DataRecord);
        }
        auto GetShaderBindingTable()const -> const OptixShaderBindingTable& {
            return m_ShaderBindingTable;
        }
        //Launch
    private:
        OptixShaderBindingTable m_ShaderBindingTable;
        RayGRecordBufferPtr     m_RayGRecordBufferPtr;
        MissRecordBufferPtr     m_MissRecordBufferPtr;
        HitGRecordBufferPtr     m_HitGRecordBufferPtr;
        ParamsBufferPtr         m_ParamsBufferPtr;
        int                     m_TraceCallDepth = 0;
    };
}
#endif