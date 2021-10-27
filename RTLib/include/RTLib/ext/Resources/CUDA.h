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
                virtual ~CUDABufferComponent() {}
                auto GetHandle() const -> const CUDABuffer<T> & { return m_GpuData; }
                auto GetHandle() -> CUDABuffer<T> & { return m_GpuData; }

            private:
                CUDABuffer<T> m_GpuData = {};
            };
            template <typename T>
            class CUDABufferImage2DComponent : public Image2DComponent
            {
            public:
                static auto New() -> Image2DComponentPtr { return Image2DComponentPtr(new CUDABufferImage2DComponent()); }
                // Image2DComponent を介して継承されました
                virtual void Init(const void* cpuData, size_t rowPitch, size_t slicePitch) override
                {
                    if (m_GpuData)
                    {
                        m_GpuData.reset();
                    }
                    m_Width  = rowPitch / sizeof(T);
                    m_Height = slicePitch / rowPitch;
                    m_GpuData.allocate(slicePitch / sizeof(T));
                    m_GpuData.upload((const T*)cpuData, slicePitch / sizeof(T));
                }
                virtual void Allocate(size_t rowPitch, size_t slicePitch) override
                {
                    if (m_GpuData)
                    {
                        m_GpuData.reset();
                    }
                    m_GpuData.allocate(slicePitch / sizeof(T));
                }
                virtual bool Resize(size_t rowPitch, size_t slicePitch) override
                {
                    if (!m_GpuData)
                    {
                        return false;
                    }
                    if (!m_GpuData.resize(slicePitch / sizeof(T)))
                    {
                        return false;
                    }
                    m_Width  = rowPitch / sizeof(T);
                    m_Height = slicePitch / rowPitch;
                    return true;
                }
                virtual void Upload(const void* cpuData, size_t rowPitch, size_t slicePitch) override
                {
                    if (!m_GpuData)
                    {
                        return;
                    }
                    m_GpuData.upload((const T*)cpuData, slicePitch / sizeof(T));
                }
                virtual void Reset() override
                {
                    if (!m_GpuData)
                    {
                        return;
                    }
                    m_GpuData.reset();
                }
                virtual ~CUDABufferImage2DComponent(){}
                auto GetHandle() const -> const CUDABuffer<T>& { return m_GpuData; }
                auto GetHandle() -> CUDABuffer<T>& { return m_GpuData; }
            private:
                size_t        m_Width   = 0;
                size_t        m_Height  = 0;
                CUDABuffer<T> m_GpuData = {};
            };
            template <typename T>
            class CUDATextureImage2DComponent : public Image2DComponent
            {
            public:
                static auto New() -> Image2DComponentPtr { return Image2DComponentPtr(new CUDATextureImage2DComponent()); }
                // Image2DComponent を介して継承されました
                virtual void Init(const void* cpuData, size_t rowPitch, size_t slicePitch) override
                {
                    if (m_GpuData)
                    {
                        m_GpuData.reset();
                    }
                    m_GpuData.allocate(rowPitch/sizeof(T), slicePitch/rowPitch, cudaTextureReadMode::cudaReadModeNormalizedFloat);
                    m_GpuData.upload(cpuData, rowPitch/sizeof(T), slicePitch/rowPitch);
                }
                virtual void Allocate(size_t rowPitch, size_t slicePitch) override
                {
                    if (m_GpuData)
                    {
                        m_GpuData.reset();
                    }
                    m_GpuData.allocate(rowPitch / sizeof(T), slicePitch / rowPitch, cudaTextureReadMode::cudaReadModeNormalizedFloat);
                }
                virtual bool Resize(size_t rowPitch, size_t slicePitch) override
                {
                    return m_GpuData.resize(rowPitch / sizeof(T), slicePitch / rowPitch, cudaTextureReadMode::cudaReadModeNormalizedFloat);
                }
                virtual void Upload(const void* cpuData, size_t rowPitch, size_t slicePitch) override
                {
                    if (!m_GpuData)
                    {
                        return;
                    }
                    m_GpuData.upload(cpuData, rowPitch/sizeof(T), slicePitch/rowPitch);
                }
                virtual void Reset() override
                {
                    m_GpuData.reset();
                }
                virtual ~CUDATextureImage2DComponent() {}
                auto GetHandle() const -> const CUDATexture2D<T>& { return m_GpuData; }
                auto GetHandle() -> CUDATexture2D<T>& { return m_GpuData; }
            private:
                CUDATexture2D<T> m_GpuData = {};
            };
        }
    }
}
#endif