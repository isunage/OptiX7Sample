#ifndef TEST_TEST24_APPLICATION_H
#define TEST_TEST24_APPLICATION_H
#include <TestLib/RTApplication.h>
#include <TestLib/RTContext.h>
#include <TestLib/RTFrameBuffer.h>
#include <TestLib/RTTracer.h>
#include <TestLib/RTGui.h>
#include <TestLib/RTAssets.h>
#include <RTLib/core/GL.h>
#include <RTLib/core/Optix.h>
#include <RTLib/ext/TraversalHandle.h>
#include <RTLib/ext/Mesh.h>
#include <RTLib/ext/RectRenderer.h>
#include <RTLib/ext/Camera.h>
#include <RTLib/ext/VariableMap.h>
#include <Test24Event.h>
class Test24Application : public test::RTApplication
{
private:
    using TracerMap = std::unordered_map<std::string, std::shared_ptr<test::RTTracer>>;
    using ContextPtr = std::shared_ptr<test::RTContext>;
    using RendererPtr= std::unique_ptr<rtlib::ext::RectRenderer>;
    using FramebufferPtr = std::shared_ptr<test::RTFramebuffer>;
    using GuiDelegatePtr = std::unique_ptr<test::RTAppGuiDelegate>;
    using CameraControllerPtr = std::shared_ptr<rtlib::ext::CameraController>;
    using GuiPtr = std::shared_ptr<test::RTGui>;
    using ObjModelAssetManagerPtr = std::shared_ptr<test::RTObjModelAssetManager>;
    using TextureAssetManagerPtr = std::shared_ptr<test::RTTextureAssetManager>;
    using GeometryASMap = std::unordered_map<std::string, rtlib::ext::GASHandlePtr>;
    using InstanceASMap = std::unordered_map<std::string, rtlib::ext::IASHandlePtr>;
    using TracerVariableMap = std::unordered_map<std::string, std::shared_ptr<rtlib::ext::VariableMap>>;
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
    void BegFrame();
    void Render();
    void Update();
    void EndFrame();
    void InitBase();
    void FreeBase();
    void InitGui();
    void FreeGui();
    void InitScene();
    void FreeScene();
    void InitTracers();
    void FreeTracers();
    void PrepareLoop();
    void RenderGui();
    void RenderFrame(const std::string &name);
    static void FramebufferSizeCallback(GLFWwindow *window, int fbWidth, int fbHeight);
    static void CursorPosCallback(GLFWwindow *window, double xpos, double ypos);
    static void ScrollCallback(GLFWwindow *window, double xoff, double yoff);
private:
    void CopyRtFrame(const std::string &name);
    void CopyDgFrame(const std::string &name);
    void ResizeFrame();
    void UpdateCamera();
    void UpdateFrameTime();
private:
    ContextPtr m_Context;
    RendererPtr m_Renderer;
    FramebufferPtr m_Framebuffer;
    CameraControllerPtr m_CameraController;
    GuiDelegatePtr m_GuiDelegate;
    GLFWwindow *m_Window;
    TracerMap m_Tracers;
    ObjModelAssetManagerPtr m_ObjModelManager;
    TextureAssetManagerPtr m_TextureManager;
    std::vector<rtlib::ext::VariableMap> m_Materials;
    GeometryASMap m_GASHandles = {};
    InstanceASMap m_IASHandles = {};
    float3 m_BgLightColor = {};
    unsigned int m_NumRayType;
    int m_FbWidth;
    int m_FbHeight;
    float m_FovY;
    unsigned int             m_SamplePerAll;
    unsigned int             m_SamplePerLaunch;
    unsigned int             m_MaxTraceDepth;
    //TODO Inputの分離
    std::array<float, 2>     m_CurCursorPos;
    std::array<float, 2>     m_DelCursorPos;
    std::array<float, 2>     m_ScrollOffsets;
    float                    m_CurFrameTime;
    float                    m_DelFrameTime;
    unsigned int             m_EventFlags;
    //表示系
    std::vector<std::string> m_FramePublicNames;
    std::string              m_CurMainFrameName;
    //Tracer
    std::vector<std::string> m_TracePublicNames;
    std::string              m_CurMainTraceName;
    TracerVariableMap               m_TracerVariables;
    std::unordered_set<std::string> m_LaunchTracerSet;
    //ObjModel
    std::string m_CurObjModelName;
};
#endif