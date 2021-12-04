#ifndef TEST_TEST24_APPLICATION_H
#define TEST_TEST24_APPLICATION_H
#include <TestLib/RTApplication.h>
#include <TestLib/RTContext.h>
#include <TestLib/RTFrameBuffer.h>
#include <TestLib/RTTracer.h>
#include <TestLib/RTGui.h>
#include <RTLib/core/GL.h>
#include <RTLib/ext/RectRenderer.h>
#include <RTLib/ext/Camera.h>
class Test24Application : public test::RTApplication
{
private:
    using TracerMap   = std::unordered_map < std::string, std::shared_ptr<test::RTTracer>>;
    using ContextPtr  = std::shared_ptr<test::RTContext>;
    using RendererPtr = std::unique_ptr<rtlib::ext::RectRenderer>;
    using FramebufferPtr = std::shared_ptr<test::RTFramebuffer>;
    using CameraControllerPtr = std::shared_ptr<rtlib::ext::CameraController>;
    using GuiPtr = std::shared_ptr<test::RTGui>;
private:
    Test24Application(int fbWidth, int fbHeight, std::string name) noexcept;
public:
    static auto New(int fbWidth, int fbHeight, std::string name) noexcept -> std::shared_ptr<test::RTApplication>;
    // RTApplication を介して継承されました
    virtual void Initialize() override;
    virtual void MainLoop() override;
    virtual void CleanUp() override;
    virtual ~Test24Application() noexcept {}
private:
    void Launch();
    void Render();
    void Update();
    //
    void InitBase();
    void FreeBase();
    void InitGui();
    void FreeGui();
    void InitScene();
    void FreeScene();
    void InitTracers();
    void FreeTracers();

    void RenderGui();
    void RenderFrame(const std::string& name);
    static void FramebufferSizeCallback(GLFWwindow* window, int fbWidth, int fbHeight);
private:
    void CopyRtFrame(const std::string& name);
    void CopyDgFrame(const std::string& name);
private:
    ContextPtr m_Context;
    RendererPtr m_Renderer;
    FramebufferPtr m_Framebuffer;
    GuiPtr m_Gui;
    CameraControllerPtr m_CameraController;
    GLFWwindow *m_Window;
    TracerMap m_Tracers;
    int m_FbWidth;
    int m_FbHeight;
    float m_FovY;
    bool  m_IsResized;
};
#endif