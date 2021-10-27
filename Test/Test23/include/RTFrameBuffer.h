#ifndef RT_FRAME_BUFFER_H
#define RT_FRAME_BUFFER_H
#include <RTLib/CUDA.h>
#include <RTLib/CUDA_GL.h>
#include <unordered_map>
#include <string>
#include <memory>
namespace test {
    class RTFrameBuffer {
    public:
        RTFrameBuffer() {}
        RTFrameBuffer(int fbWidth, int fbHeight) :m_FbWidth{ fbWidth }, m_FbHeight{fbHeight} {}
        RTFrameBuffer(RTFrameBuffer&& fb)   = default;
        RTFrameBuffer(const RTFrameBuffer&) = delete;
        RTFrameBuffer& operator=(RTFrameBuffer&& fb)   = default;
        RTFrameBuffer& operator=(const RTFrameBuffer&) = delete;
        bool Resize(int fbWidth, int fbHeight) {
            if (m_FbWidth == fbWidth && m_FbHeight == fbHeight) {
                return false;
            }
            m_FbWidth  = fbWidth;
            m_FbHeight = fbHeight;
            for (auto& [name, buffer] : m_CUDABuffers) {
                buffer.resize(fbWidth * fbHeight);
            }
            for (auto& [name, buffer] : m_CUGLBuffers) {
                buffer.resize(fbWidth * fbHeight);
            }
            return true;
        }
        void CleanUp() {
            for (auto& [name, buffer] : m_CUDABuffers) {
                buffer.reset();
            }
            for (auto& [name, buffer] : m_CUGLBuffers) {
                buffer.reset();
            }
            m_CUDABuffers.clear();
        }
        void AddCUDABuffer(const std::string& key) {
            m_CUDABuffers[key] = rtlib::CUDABuffer<uchar4>(std::vector<uchar4>(m_FbWidth * m_FbHeight));
        }
        auto GetCUDABuffer(const std::string& key)->rtlib::CUDABuffer<uchar4>& {
            return m_CUDABuffers.at(key);
        }
        auto GetCUDABuffer(const std::string& key)const -> const rtlib::CUDABuffer<uchar4>& {
            return m_CUDABuffers.at(key);
        }
        void AddCUGLBuffer(const std::string& key) {
            m_CUGLBuffers[key] = rtlib::GLInteropBuffer<uchar4>(m_FbWidth*m_FbHeight,GL_PIXEL_UNPACK_BUFFER,GL_STATIC_DRAW);
        }
        auto GetCUGLBuffer(const std::string& key)->rtlib::GLInteropBuffer<uchar4>& {
            return m_CUGLBuffers.at(key);
        }
        auto GetCUGLBuffer(const std::string& key)const -> const rtlib::GLInteropBuffer<uchar4>& {
            return m_CUGLBuffers.at(key);
        }
        ~RTFrameBuffer()noexcept {
            try
            {
                this->CleanUp();
            }
            catch (...)
            {

            }
        }
    private:
        std::unordered_map<std::string, rtlib::CUDABuffer<uchar4>>      m_CUDABuffers  = {};
        std::unordered_map<std::string, rtlib::GLInteropBuffer<uchar4>> m_CUGLBuffers  = {};
        int                                                             m_FbWidth      = 1;
        int                                                             m_FbHeight     = 1;
    };
}
#endif