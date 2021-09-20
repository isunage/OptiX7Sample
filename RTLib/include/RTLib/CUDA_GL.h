#ifndef RTLIB_CUDA_GL_H
#define RTLIB_CUDA_GL_H
#include "GL.h"
#include "CUDA.h"
#include <cuda_gl_interop.h>
namespace rtlib{
    //GL Interop Buffer
    //<-委譲-GL Buffer
    //今回はCUDA->OpenGLの一方向にしておく
    template<typename T>
    class GLInteropBuffer{
        GLBuffer<T>            m_Handle;
        cudaStream_t           m_Stream           = nullptr;
        T*                     m_DevicePtr        = nullptr;
        cudaGraphicsResource_t m_GraphicsResource = nullptr;
    private:
        void registerResource() {

            RTLIB_CUDA_CHECK(cudaGraphicsGLRegisterBuffer(
                &m_GraphicsResource,
                getID(),
                cudaGraphicsRegisterFlagsWriteDiscard));
        }
        void unregisterResource() {
            if (m_GraphicsResource) {
                RTLIB_CUDA_CHECK(cudaGraphicsUnregisterResource(m_GraphicsResource));
                m_GraphicsResource = nullptr;
            }
        }
    public:
        GLInteropBuffer() {}
        GLInteropBuffer(const GLInteropBuffer  &)noexcept = delete;
        GLInteropBuffer(GLInteropBuffer&& buffer)noexcept{
            m_Handle = std::move(buffer.m_Handle);
            m_Stream = buffer.m_Stream;
            m_DevicePtr = buffer.m_DevicePtr;
            m_GraphicsResource = buffer.m_GraphicsResource;
            buffer.m_Stream = nullptr;
            buffer.m_DevicePtr = nullptr;
            buffer.m_GraphicsResource = nullptr;
        }
        GLInteropBuffer& operator=(const GLInteropBuffer  &)noexcept = delete;
        GLInteropBuffer& operator=(      GLInteropBuffer && buffer){
            if(this!=&buffer){
                this->reset();
                m_Handle = std::move(buffer.m_Handle);
                m_Stream = buffer.m_Stream;
                m_DevicePtr = buffer.m_DevicePtr;
                m_GraphicsResource = buffer.m_GraphicsResource;
                buffer.m_Stream = nullptr;
                buffer.m_DevicePtr = nullptr;
                buffer.m_GraphicsResource = nullptr;
            }
            return *this;
        }
        explicit GLInteropBuffer(size_t count, GLenum target, GLenum usage, cudaStream_t stream = nullptr)
        :m_Handle(target,usage){
            this->setStream(stream);
            this->resize(count);
        }
        //Get And Set
        RTLIB_DECLARE_GET_AND_SET_BY_VALUE(GLInteropBuffer, cudaStream_t, Stream, m_Stream);
        GLenum getTarget()const noexcept { return m_Handle.getTarget(); };
        void   setTarget(GLenum target)noexcept{ m_Handle.setTarget(target);}
        GLenum getUsage()const noexcept { return m_Handle.getUsage(); };
        void   setUsage(GLenum usage)noexcept{ m_Handle.setUsage(usage);}
        RTLIB_DECLARE_GET_BY_REFERENCE(GLInteropBuffer,GLBuffer<T>,Handle,m_Handle);
        RTLIB_DECLARE_GET_BY_VALUE(GLInteropBuffer,GLuint,ID,m_Handle.getID());
        RTLIB_DECLARE_GET_BY_VALUE(GLInteropBuffer,size_t,Count,m_Handle.getCount());
        RTLIB_DECLARE_GET_BY_VALUE(GLInteropBuffer,size_t,SizeInBytes,m_Handle.getSizeInBytes());
        //Member(GL Buffer)
        void   bind()const{ m_Handle.bind();}
        void   unbind()const{ m_Handle.unbind();}
        //upload
        void  upload(const T* hostPtr, size_t count, bool isBinded = true) {
            m_Handle.upload(hostPtr, count, isBinded);
        }
        void  upload(const std::vector<T>& hostArray, bool isBinded = true) {
            if (isBinded) {
                this->bind();
            }
            m_Handle.upload(hostArray,isBinded);
        }
        //download
        void  download(T* hostPtr, size_t count, bool isBinded = true) {
            m_Handle.download(hostPtr, count, isBinded);
        }
        void  download(std::vector<T>& hostData, bool isBinded = true) {
            m_Handle.download(hostData, isBinded);
        }
        //Member(Unique)
        bool resize(size_t count) {
            if (getCount() != count) {
                std::cout << "RESIZED!\n";
                GLenum target = this->getHandle().getTarget();
                GLenum usage  = this->getHandle().getUsage();
                this->reset();
                this->m_Handle.setTarget(target);
                this->m_Handle.setUsage(usage);
                this->allocate(count);
                return true;

            }
            else {
                std::cout << "NOT RESIZED!\n";
            }
            return false;
        }
        void allocate(size_t count) {
            this->m_Handle.allocate(count);
            std::cout << "ALLOCATED!\n";
            this->registerResource();
        }
        [[nodiscard]] T* map() {
            RTLIB_CUDA_CHECK(cudaGraphicsMapResources(1, &m_GraphicsResource, m_Stream));
            size_t sizeInBytes;
            RTLIB_CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(
                reinterpret_cast<void**>(&m_DevicePtr),
                &sizeInBytes,
                m_GraphicsResource
            ));
            return m_DevicePtr;
        }
        void unmap() {
            RTLIB_CUDA_CHECK(cudaGraphicsUnmapResources(1, &m_GraphicsResource, m_Stream));
            m_DevicePtr = nullptr;
        }
        void reset() {
            //m_Stream = nullptr;
            m_DevicePtr = nullptr;
            this->unregisterResource();
            this->m_Handle.reset();
        }
        ~GLInteropBuffer()noexcept {
            //std::cout << "Destroy GL Interop Buffer" << std::endl;
            try{
                m_Stream = nullptr;
                m_DevicePtr = nullptr;
                this->unregisterResource();
                this->m_Handle.reset();
            }catch(...){
                
            }
            //~GLBuffer();
        }
    };
}
#endif