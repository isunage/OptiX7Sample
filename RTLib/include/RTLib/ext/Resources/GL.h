#ifndef RTLIB_EXT_RESOURCES_GL_H
#define RTLIB_EXT_RESOURCES_GL_H
#include "../Resources.h"
#include "../../GL.h"
namespace rtlib
{
    namespace ext
    {
        namespace resources
        {
            template <typename T>
            class GLBufferComponent : public BufferComponent
            {
            public:
                static auto New() -> BufferComponentPtr { return BufferComponentPtr(new GLBufferComponent()); }
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
                auto GetHandle() const -> const GLBuffer<T> & { return m_GpuData; }
                auto GetHandle() -> GLBuffer<T> & { return m_GpuData; }

            private:
                GLBuffer<T> m_GpuData = {};
            };

        }
    }
}
#endif