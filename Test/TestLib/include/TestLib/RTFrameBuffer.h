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
        RTFrameBuffer(int fbWidth, int fbHeight) :m_FbWidth{ fbWidth }, m_FbHeight{ fbHeight } {}
        RTFrameBuffer(RTFrameBuffer&& fb) = default;
        RTFrameBuffer(const RTFrameBuffer&) = delete;
        RTFrameBuffer& operator=(RTFrameBuffer&& fb) = default;
        RTFrameBuffer& operator=(const RTFrameBuffer&) = delete;
        bool Resize(int fbWidth, int fbHeight) {
            if (m_FbWidth == fbWidth && m_FbHeight == fbHeight) {
                return false;
            }
            m_FbWidth = fbWidth;
            m_FbHeight = fbHeight;
            for (auto& [name, buffer] : m_CUDABuffers) {
                buffer.resize(fbWidth * fbHeight);
            }
            for (auto& [name, buffer] : m_CUGLBuffers) {
                buffer.resize(fbWidth * fbHeight);
            }
            for (auto& [name, texture] : m_GLTextures) {
                texture.reset();
                texture.allocate({ (size_t)m_FbWidth, (size_t)m_FbHeight, nullptr }, GL_TEXTURE_2D);
                texture.setParameteri(GL_TEXTURE_MAG_FILTER, GL_LINEAR, false);
                texture.setParameteri(GL_TEXTURE_MIN_FILTER, GL_LINEAR, false);
                texture.setParameteri(GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE, false);
                texture.setParameteri(GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE, false);
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
            for (auto& [name, texture] : m_GLTextures) {
                texture.reset();
            }
            m_CUDABuffers.clear();
            m_CUGLBuffers.clear();
            m_GLTextures.clear();
        }
        auto GetWidth()const noexcept  -> int { return m_FbWidth; }
        auto GetHeight()const noexcept -> int { return m_FbHeight; }
        auto GetAspect()const noexcept -> float { return (float)m_FbWidth / (float)m_FbHeight; }
        bool HasCUDABuffer(const std::string& key)const noexcept {
            return m_CUDABuffers.count(key) > 0;
        }
        void AddCUDABuffer(const std::string& key) {
            if (HasCUDABuffer(key)) {
                return;
            }
            m_CUDABuffers[key] = rtlib::CUDABuffer<uchar4>(std::vector<uchar4>(m_FbWidth * m_FbHeight));
        }
        auto GetCUDABuffer(const std::string& key)->rtlib::CUDABuffer<uchar4>& {
            return m_CUDABuffers.at(key);
        }
        auto GetCUDABuffer(const std::string& key)const -> const rtlib::CUDABuffer<uchar4>& {
            return m_CUDABuffers.at(key);
        }
        bool HasCUGLBuffer(const std::string& key)const noexcept {
            return m_CUGLBuffers.count(key) > 0;
        }
        void AddCUGLBuffer(const std::string& key) {
            if (HasCUGLBuffer(key)) {
                return;
            }
            m_CUGLBuffers[key] = rtlib::GLInteropBuffer<uchar4>(m_FbWidth * m_FbHeight, GL_PIXEL_UNPACK_BUFFER, GL_STATIC_DRAW);
        }
        auto GetCUGLBuffer(const std::string& key)->rtlib::GLInteropBuffer<uchar4>& {
            return m_CUGLBuffers.at(key);
        }
        auto GetCUGLBuffer(const std::string& key)const -> const rtlib::GLInteropBuffer<uchar4>& {
            return m_CUGLBuffers.at(key);
        }
        bool HasGLTexture(const std::string& key)const noexcept {
            return m_GLTextures.count(key) > 0;
        }
        void AddGLTexture(const std::string& key)
        {
            m_GLTextures[key] = rtlib::GLTexture2D<uchar4>();
            m_GLTextures[key].allocate({ (size_t)m_FbWidth, (size_t)m_FbHeight, nullptr }, GL_TEXTURE_2D);
            m_GLTextures[key].setParameteri(GL_TEXTURE_MAG_FILTER, GL_LINEAR, false);
            m_GLTextures[key].setParameteri(GL_TEXTURE_MIN_FILTER, GL_LINEAR, false);
            m_GLTextures[key].setParameteri(GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE, false);
            m_GLTextures[key].setParameteri(GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE, false);
        }
        auto GetGLTexture(const std::string& key) ->  rtlib::GLTexture2D<uchar4>& {
            return m_GLTextures.at(key);
        }
        auto GetGLTexture(const std::string& key)const -> const rtlib::GLTexture2D<uchar4>& {
            return m_GLTextures.at(key);
        }
        ~RTFrameBuffer()noexcept {}
    private:
        std::unordered_map<std::string, rtlib::CUDABuffer<uchar4>>      m_CUDABuffers = {};
        std::unordered_map<std::string, rtlib::GLInteropBuffer<uchar4>> m_CUGLBuffers = {};
        std::unordered_map<std::string, rtlib::GLTexture2D<uchar4>>     m_GLTextures  = {};
        int                                                             m_FbWidth  = 1;
        int                                                             m_FbHeight = 1;
    };
    using RTFrameBufferPtr = std::shared_ptr<RTFrameBuffer>;
}
#endif