#ifndef TEST_RT_FRAME_BUFFER_H
#define TEST_RT_FRAME_BUFFER_H
#include <RTLib/core/CUDA.h>
#include <RTLib/core/GL.h>
#include <RTLib/core/CUDA_GL.h>
#include <unordered_map>
#include <string>
#include <cassert>
namespace test {
    //大丈夫
    class RTFramebufferComponent
    {
    public:
        virtual void Initialize(int fbWidth, int fbHeight) = 0;
        virtual bool Resize(int fbWidth, int fbHeight) = 0;
        virtual void CleanUp() = 0;
        virtual auto GetIDString()const noexcept -> std::string_view = 0;
        virtual ~RTFramebufferComponent() {}
    };
    class RTFramebuffer {
    public:
        RTFramebuffer() {}
        RTFramebuffer(int fbWidth, int fbHeight) :m_FbWidth{ fbWidth }, m_FbHeight{ fbHeight } {}
        RTFramebuffer(RTFramebuffer&& fb) = default;
        RTFramebuffer(const RTFramebuffer&) = delete;
        RTFramebuffer& operator=(RTFramebuffer&& fb) = default;
        RTFramebuffer& operator=(const RTFramebuffer&) = delete;

        bool Resize(int fbWidth, int fbHeight) {
            if (m_FbWidth == fbWidth && m_FbHeight == fbHeight) {
                return false;
            }
            m_FbWidth = fbWidth;
            m_FbHeight = fbHeight;
            for (auto& [name, component] : m_Components) {
                component->Resize(fbWidth ,fbHeight);
            }
            return true;
        }
        void CleanUp() {
            for (auto& [name, component] : m_Components) {
                component->CleanUp();
            }
            m_Components.clear();
            m_FbWidth  = 0;
            m_FbHeight = 0;
        }

        template<typename S,typename ...Args, bool Cond = std::is_base_of_v<RTFramebufferComponent, S>>
        void AddComponent(const std::string& key, Args&&... args)
        {
            m_Components[key] = std::shared_ptr<S>(new S(std::forward<Args&&>(args)...));
            m_Components[key]->Initialize(m_FbWidth, m_FbHeight);
        }
        template<typename S, bool Cond = std::is_base_of_v<RTFramebufferComponent, S>>
        auto GetComponent(const std::string& key)const -> std::shared_ptr<S> {
            if (m_Components.count(key) == 0) {
                return nullptr;
            }
            auto ptr = m_Components.at(key);
            if (!ptr) { return nullptr; }
            if (S::IDString() != ptr->GetIDString()) { return nullptr; }
            return std::static_pointer_cast<S>(ptr);
        }
        template<typename S, bool Cond = std::is_base_of_v<RTFramebufferComponent, S>>
        auto PopComponent(const std::string& key) -> std::shared_ptr<S> {
            if (m_Components.count(key) == 0) {
                return nullptr;
            }
            auto ptr = m_Components.at(key);
            if (S::IDString() != ptr->GetIDString()) { return nullptr; }
            m_Components.erase(key);
            return std::static_pointer_cast<S>(ptr);
        }
        template<typename S, bool Cond = std::is_base_of_v<RTFramebufferComponent, S>>
        bool HasComponent(const std::string& key)const noexcept { 
            if (m_Components.count(key) == 0) {
                return false;
            }
            auto ptr = m_Components.at(key);
            return S::IDString() == ptr->GetIDString();
        }
        
        ~RTFramebuffer()noexcept {
            if (!m_Components.empty()) {
                CleanUp();
            }
        }
    private:
        std::unordered_map<std::string, std::shared_ptr<test::RTFramebufferComponent>>   m_Components  = {};
        int                                                                              m_FbWidth     = 0;
        int                                                                              m_FbHeight    = 0;
    };
    template<typename T>
    class RTGLTextureFBComponent : public RTFramebufferComponent
    {
    public:
        RTGLTextureFBComponent() noexcept :RTFramebufferComponent() {
            m_Handle   = rtlib::GLTexture2D<uchar4>();
            m_FbWidth  = 0;
            m_FbHeight = 0;
            m_MagFilter = GL_LINEAR;
            m_MinFilter = GL_LINEAR;
            m_WarpSMode = GL_CLAMP_TO_EDGE;
            m_WarpTMode = GL_CLAMP_TO_EDGE;
        }
        RTGLTextureFBComponent(GLint magFilter, GLint minFilter, GLint warpSMode, GLint warpTMode) noexcept : RTGLTextureFBComponent() {
            m_Handle = rtlib::GLTexture2D<uchar4>();
            m_FbWidth = 0;
            m_FbHeight = 0;
            m_MagFilter = magFilter;
            m_MinFilter = minFilter;
            m_WarpSMode = warpSMode;
            m_WarpTMode = warpTMode;
        }

        virtual auto GetIDString()const noexcept -> std::string_view {
            return IDString();
        }
        virtual void Initialize(int fbWidth, int fbHeight) override {
            m_FbWidth = fbWidth;
            m_FbHeight = fbHeight;
            m_Handle.allocate({ (size_t)m_FbWidth,(size_t)m_FbHeight,nullptr }, GL_TEXTURE_2D, false);
            m_Handle.setParameteri(GL_TEXTURE_MAG_FILTER, m_MagFilter, false);
            m_Handle.setParameteri(GL_TEXTURE_MIN_FILTER, m_MinFilter, false);
            m_Handle.setParameteri(GL_TEXTURE_WRAP_S, m_WarpSMode, false);
            m_Handle.setParameteri(GL_TEXTURE_WRAP_T, m_WarpTMode, false);
            m_Handle.unbind();
        }
        virtual bool Resize(int fbWidth, int fbHeight) override
        {
            if (fbWidth != m_FbWidth || fbHeight != m_FbHeight) {
                m_Handle.reset();
                m_Handle.allocate({ (size_t)m_FbWidth,(size_t)m_FbHeight,nullptr }, GL_TEXTURE_2D, false);
                m_Handle.setParameteri(GL_TEXTURE_MAG_FILTER, m_MagFilter, false);
                m_Handle.setParameteri(GL_TEXTURE_MIN_FILTER, m_MinFilter, false);
                m_Handle.setParameteri(GL_TEXTURE_WRAP_S, m_WarpSMode, false);
                m_Handle.setParameteri(GL_TEXTURE_WRAP_T, m_WarpTMode, false);
                m_Handle.unbind();
                m_FbWidth  = fbWidth;
                m_FbHeight = fbHeight;
                return true;
            }
            return false;
        }
        virtual void CleanUp() override
        {
            m_Handle.reset();
            m_FbWidth = 0;
            m_FbHeight = 0;
        }
        auto         GetHandle() noexcept -> rtlib::GLTexture2D<T>& {
            return m_Handle;
        }
        static auto  IDString()   noexcept -> std::string_view
        {
            return "GLTextureFBComponent";
        }

        virtual ~RTGLTextureFBComponent()noexcept {}
    private:
        rtlib::GLTexture2D<T>     m_Handle;
        int                       m_FbWidth ;
        int                       m_FbHeight;
        GLint                     m_MagFilter;
        GLint                     m_MinFilter;
        GLint                     m_WarpSMode;
        GLint                     m_WarpTMode;

    };
    template<typename T>
    class RTCUDABufferFBComponent : public RTFramebufferComponent
    {
    public:
        RTCUDABufferFBComponent() noexcept :RTFramebufferComponent() {}

        virtual auto GetIDString()const noexcept -> std::string_view {
            return IDString();
        }
        virtual void Initialize(int fbWidth, int fbHeight) override {
            m_Handle   = rtlib::CUDABuffer<T>(std::vector<T>(fbWidth * fbHeight));
        }
        virtual bool Resize(int fbWidth, int fbHeight) override
        {
            return m_Handle.resize(fbWidth * fbHeight);
        }
        virtual void CleanUp() override
        {
            m_Handle.reset();
        }
        auto GetHandle() noexcept -> rtlib::CUDABuffer<T>& {
            return m_Handle;
        }
        static auto  IDString()    noexcept -> std::string_view
        {
            return "CUDABufferFBComponent";
        }

        virtual ~RTCUDABufferFBComponent()noexcept {}
    private:
        rtlib::CUDABuffer<T>      m_Handle;

    };
    template<typename T>
    class RTCUGLBufferFBComponent : public RTFramebufferComponent
    {
    public:
        RTCUGLBufferFBComponent() noexcept:RTFramebufferComponent() {}

        virtual auto GetIDString()const noexcept -> std::string_view {
            return IDString();
        }
        virtual void Initialize(int fbWidth, int fbHeight) override {
            m_Handle = rtlib::GLInteropBuffer<T>(fbWidth * fbHeight, GL_PIXEL_UNPACK_BUFFER, GL_STATIC_DRAW);
            m_Handle.unbind();
        }
        virtual bool Resize(int fbWidth, int fbHeight) override
        {
            return m_Handle.resize(fbWidth*fbHeight);
        }
        virtual void CleanUp() override
        {
            m_Handle.reset();
        }
        auto GetHandle() noexcept ->  rtlib::GLInteropBuffer<T>& {
            return m_Handle;
        }
        static auto  IDString()    noexcept -> std::string_view
        {
            return "CUGLBufferFBComponent";
        }

        virtual ~RTCUGLBufferFBComponent()noexcept {}
    private:
        rtlib::GLInteropBuffer<T> m_Handle;

    };
}
#endif
