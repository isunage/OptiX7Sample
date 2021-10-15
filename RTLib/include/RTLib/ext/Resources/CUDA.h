#ifndef RTLIB_EXT_RESOURCES_CUDA_H
#define RTLIB_EXT_RESOURCES_CUDA_H
#include "../Resources.h"
#include "../../CUDA.h"
namespace rtlib
{
    namespace ext
    {
        namespace resources
        {
            template <typename T>
            class CUDABufferComponent : public BufferComponent
            {
            public:
                static auto New() -> BufferComponentPtr { return BufferComponentPtr(new CUDABufferComponent()); }
                // BufferComponent ����Čp������܂���
                virtual void Init(const void *cpuData, size_t sizeInBytes) override
                {
                    if (m_GpuData)
                    {
                        m_GpuData.reset();
                    }
                    m_GpuData.allocate(sizeInBytes / sizeof(T));
                    m_GpuData.upload((const T *)cpuData, sizeInBytes / sizeof(T));
                }
                virtual void Allocate(size_t sizeInBytes) override
                {
                    if (m_GpuData)
                    {
                        m_GpuData.reset();
                    }
                    m_GpuData.allocate(sizeInBytes / sizeof(T));
                }
                virtual bool Resize(size_t sizeInBytes) override
                {
                    if (!m_GpuData)
                    {
                        return false;
                    }
                    return m_GpuData.resize(sizeInBytes / sizeof(T));
                }
                virtual void Upload(const void *cpuData, size_t sizeInBytes) override
                {
                    if (!m_GpuData)
                    {
                        return;
                    }
                    m_GpuData.upload((const T *)cpuData, sizeInBytes / sizeof(T));
                }
                virtual void Reset() override
                {
                    if (!m_GpuData)
                    {
                        return;
                    }
                    m_GpuData.reset();
                }
                auto GetHandle() const -> const CUDABuffer<T> & { return m_GpuData; }
                auto GetHandle() -> CUDABuffer<T> & { return m_GpuData; }

            private:
                CUDABuffer<T> m_GpuData = {};
            };
        }
    }
}
#endif