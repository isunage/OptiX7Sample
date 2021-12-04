#ifndef TEST_RT_RECORD_BUFFER_H
#define TEST_RT_RECORD_BUFFER_H
#include <RTLib/core/Optix.h>
#include <RTLib/core/CUDA.h>
#include <cassert>
namespace test{
    template<typename Data_t>
    class RTRecordBuffer {
    public:
        using DataRecord   = rtlib::SBTRecord<Data_t>;
        using UploadBuffer = rtlib::CUDAUploadBuffer<DataRecord>;
    public:
        void Alloc() {
            m_UploadBuffer.cpuHandle.resize(1);
            m_UploadBuffer.Upload();
        }
        void SetRecord(const DataRecord& record){
            if (!m_UploadBuffer.cpuHandle.empty()){
                m_UploadBuffer.cpuHandle[0] = record;
            }
        }
        auto GetData()const noexcept -> const Data_t&{
            return m_UploadBuffer.cpuHandle.at(0).data;
        }
        auto GetData() noexcept -> Data_t&{
            return m_UploadBuffer.cpuHandle.at(0).data;
        }
        auto GetDevicePtr()const -> DataRecord*{
            return m_UploadBuffer.gpuHandle.getDevicePtr();
        }
        void Upload() noexcept{
            m_UploadBuffer.Upload();
        }
        
    private:
        UploadBuffer m_UploadBuffer;
    };
    template<typename Data_t>
    class RTRecordArrayBuffer {
    public:
        using DataRecord   = rtlib::SBTRecord<Data_t>;
        using UploadBuffer = rtlib::CUDAUploadBuffer<DataRecord>;
    public:
        void Alloc(std::size_t count) {
            m_UploadBuffer.cpuHandle.resize(count);
            m_UploadBuffer.Upload();
        }
        void SetRecord(std::size_t idx, const DataRecord& record){
            if (!m_UploadBuffer.cpuHandle.empty()){
                m_UploadBuffer.cpuHandle.at(idx) = record;
            }
        }
        auto GetData(std::size_t idx)const -> const Data_t&{
            return m_UploadBuffer.cpuHandle.at(idx).data;
        }
        auto GetData(std::size_t idx) -> Data_t&{
            return m_UploadBuffer.cpuHandle.at(idx).data;
        }
        auto GetDevicePtr()const -> DataRecord*{
            return m_UploadBuffer.gpuHandle.getDevicePtr();
        }
        auto GetSize()const noexcept -> std::size_t{
            return m_UploadBuffer.cpuHandle.size();
        }
        void Upload() noexcept{
            m_UploadBuffer.Upload();
        }
    private:
        UploadBuffer m_UploadBuffer;
    };

}
#endif