#include "..\include\Test24Application.h"
#include <Test24Gui.h>
#include <TestGLTracer.h>
#include <Test24Config.h>
Test24Application::Test24Application(int fbWidth, int fbHeight, std::string name) noexcept : test::RTApplication(name)
{
    m_FbWidth          = fbWidth;
    m_FbHeight         = fbHeight;
    m_Window           = nullptr;
    m_FovY             = 0.0f;
    m_IsResized        = false;
    m_UpdateCamera     = false;
    m_CurFrameTime     = 0.0f;
    m_DelFrameTime     = 0.0f;
    m_DelCursorPos[0]  = 0.0f;
    m_DelCursorPos[1]  = 0.0f;
    m_CurCursorPos[0]  = 0.0f;
    m_CurCursorPos[1]  = 0.0f;
    m_ScrollOffsets[0] = 0.0f;
    m_ScrollOffsets[1] = 0.0f;
    m_CurMainFrameName = "";
    m_CurMainTraceName = "";
    m_FramePublicNames = {};
    m_TracePublicNames = {};
}

auto Test24Application::New(int fbWidth, int fbHeight, std::string name) noexcept -> std::shared_ptr<test::RTApplication>
{
    return std::shared_ptr<test::RTApplication>(new Test24Application(fbWidth, fbHeight, name));
}

void Test24Application::Initialize()
{
    InitBase();
    InitGui();
    InitScene();
    InitTracers();
}

void Test24Application::MainLoop()
{
    while (!glfwWindowShouldClose(m_Window))
    {
        this->Launch();
        this->BegFrame();
        this->Render();
        this->Update();
        this->EndFrame();
    }
}

void Test24Application::CleanUp()
{
    FreeTracers();
    FreeScene();
    FreeGui();
    FreeBase();
}

void Test24Application::InitBase()
{
    m_Context = std::shared_ptr<test::RTContext>(new test::RTContext(4, 5));
    m_Window = m_Context->NewWindow(m_FbWidth, m_FbHeight, GetName().c_str());
    glfwSetWindowUserPointer(m_Window, this);
    glfwSetFramebufferSizeCallback(m_Window, FramebufferSizeCallback);
    glfwSetCursorPosCallback(m_Window, CursorPosCallback);
    m_Framebuffer = std::shared_ptr<test::RTFramebuffer>(new test::RTFramebuffer(m_FbWidth, m_FbHeight));
    //GBuffer
    m_Framebuffer->AddComponent<test::RTCUDABufferFBComponent<float3>>("GPosition");
    m_Framebuffer->AddComponent<test::RTCUDABufferFBComponent<float3>>("GNormal");
    m_Framebuffer->AddComponent<test::RTCUDABufferFBComponent<float2>>("GTexCoord");
    m_Framebuffer->AddComponent<test::RTCUDABufferFBComponent<float>>( "GDepth");
    //Render
    m_Framebuffer->AddComponent<test::RTCUDABufferFBComponent<float4>>("RAccum");
    m_Framebuffer->AddComponent<test::RTCUGLBufferFBComponent<uchar4>>("RFrame");
    //Debug
    m_Framebuffer->AddComponent<test::RTCUGLBufferFBComponent<uchar4>>("DNormal");
    m_Framebuffer->AddComponent<test::RTCUGLBufferFBComponent<uchar4>>("DTexCoord");
    m_Framebuffer->AddComponent<test::RTCUGLBufferFBComponent<uchar4>>("DDistance");
    m_Framebuffer->AddComponent<test::RTCUGLBufferFBComponent<uchar4>>("DDiffuse");
    m_Framebuffer->AddComponent<test::RTCUGLBufferFBComponent<uchar4>>("DSpecular");
    m_Framebuffer->AddComponent<test::RTCUGLBufferFBComponent<uchar4>>("DTransmit");
    m_Framebuffer->AddComponent<test::RTCUGLBufferFBComponent<uchar4>>("DShinness");
    m_Framebuffer->AddComponent<test::RTCUGLBufferFBComponent<uchar4>>("DIOR");
    //Texture
    m_Framebuffer->AddComponent<test::RTGLTextureFBComponent<uchar4>>( "RTexture");
    m_Framebuffer->AddComponent<test::RTGLTextureFBComponent<uchar4>>( "DTexture");
    m_Framebuffer->AddComponent<test::RTGLTextureFBComponent<uchar4>>( "PTexture", GL_NEAREST, GL_NEAREST, GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE);
    //Public Frames
    m_FramePublicNames.clear();
    m_FramePublicNames.push_back("RTexture");
    m_FramePublicNames.push_back("DTexture");
    m_FramePublicNames.push_back("PTexture");
    m_FramePublicNames.push_back("RFrame");
    m_FramePublicNames.push_back("DNormal");
    m_FramePublicNames.push_back("DTexCoord");
    m_FramePublicNames.push_back("DDistance");
    m_FramePublicNames.push_back("DDiffuse");
    m_FramePublicNames.push_back("DSpecular");
    m_FramePublicNames.push_back("DTransmit");
    m_FramePublicNames.push_back("DShinness");
    m_FramePublicNames.push_back("DIOR");
    m_CurMainFrameName = "RTexture";
    //Renderer
    m_Renderer = std::make_unique<rtlib::ext::RectRenderer>();
    m_Renderer->init();
    std::cout << m_Framebuffer->GetComponent<test::RTGLTextureFBComponent<uchar4>>("RTexture")->GetIDString() << std::endl;
    //ObjAssetManager
    m_ObjModelManager = std::make_shared<test::RTObjModelAssetManager>();
    //Depth Enable
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
}

void Test24Application::FreeBase()
{
    m_ObjModelManager.reset();
    m_Renderer.reset();
    m_Framebuffer.reset();
    if (m_Window)
    {
        glfwDestroyWindow(m_Window);
        m_Window = nullptr;
    }
    m_Context.reset();
}

void Test24Application::InitGui()
{
    // Gui
    m_GuiDelegate = std::make_unique<Test24GuiDelegate>(m_Window, m_Framebuffer,m_ObjModelManager, m_FramePublicNames, m_TracePublicNames, m_CurMainFrameName, m_CurMainTraceName);
    m_GuiDelegate->Initialize();
    glfwSetScrollCallback(m_Window, ScrollCallback);
}

void Test24Application::FreeGui()
{
    m_GuiDelegate->CleanUp();
    m_GuiDelegate.reset();
}

void Test24Application::RenderGui()
{
    m_GuiDelegate->DrawFrame();
}

void Test24Application::InitScene()
{
    if (m_ObjModelManager->LoadAsset("CornellBox-Water", TEST_TEST24_DATA_PATH"/Models/CornellBox/CornellBox-Water.obj")) {

    }
    m_CameraController = std::make_shared<rtlib::ext::CameraController>(float3{0.0f, 1.0f, 5.0f });
    m_CameraController->SetMouseSensitivity(0.125f);
    m_CameraController->SetMovementSpeed(10.f);
    m_CameraController->SetZoom(40.0f);
}

void Test24Application::FreeScene()
{
}

void Test24Application::InitTracers()
{

    m_Tracers["TestGL"] = std::make_shared<Test24TestGLTracer>(m_FbWidth, m_FbHeight, m_Window,m_ObjModelManager,m_Framebuffer, m_CameraController,m_IsResized,m_UpdateCamera);
    m_Tracers["TestGL"]->Initialize();

    m_TracePublicNames.clear();
    m_TracePublicNames.reserve(m_Tracers.size());
    for (auto& [name, tracer] : m_Tracers)
    {
        m_TracePublicNames.push_back(name);
    }

    m_CurMainTraceName = m_TracePublicNames.front();
}

void Test24Application::FreeTracers()
{
    for (auto &[name, tracer] : m_Tracers)
    {
        tracer->CleanUp();
    }
    m_Tracers.clear();
}

void Test24Application::PrepareLoop()
{
    glfwSetTime(0.0f);
    {
        double xpos, ypos;
        glfwGetCursorPos(m_Window, &xpos, &ypos);
        m_CurCursorPos[0] = xpos;
        m_CurCursorPos[1] = ypos;
    }
}

void Test24Application::RenderFrame(const std::string &name)
{
    if (m_Framebuffer->HasComponent<test::RTGLTextureFBComponent<uchar4>>(name))
    {
        auto rtComponent = m_Framebuffer->GetComponent<test::RTGLTextureFBComponent<uchar4>>(name);
        if (!rtComponent)
        {
            return;
        }
        auto &renderTexture = rtComponent->GetHandle();
        m_Renderer->draw(renderTexture.getID());
        return;
    }
    if (m_Framebuffer->HasComponent<test::RTCUGLBufferFBComponent<uchar4>>(name))
    {
        this->CopyRtFrame(name);
        auto rtComponent = m_Framebuffer->GetComponent<test::RTGLTextureFBComponent<uchar4>>("RTexture");
        if (!rtComponent)
        {
            return;
        }
        auto &renderTexture = rtComponent->GetHandle();
        m_Renderer->draw(renderTexture.getID());
    }
}

void Test24Application::Launch()
{
    m_Tracers["TestGL"]->Launch(m_FbWidth, m_FbHeight, nullptr);
}

void Test24Application::BegFrame()
{
    glFlush();
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
}

void Test24Application::Render()
{
    this->RenderFrame(m_CurMainFrameName);
    this->RenderGui();
}

void Test24Application::Update()
{
    ResizeFrame();
    UpdateCamera();
    for (auto &[name, tracer] : m_Tracers)
    {
        tracer->Update();
    }
    m_IsResized    = false;
    m_UpdateCamera = false;
    glfwPollEvents();
    UpdateFrameTime();
}

void Test24Application::EndFrame()
{
    glfwSwapBuffers(m_Window);
}

void Test24Application::FramebufferSizeCallback(GLFWwindow *window, int fbWidth, int fbHeight)
{
    auto *this_ptr = reinterpret_cast<Test24Application *>(glfwGetWindowUserPointer(window));
    if (this_ptr)
    {
        if (this_ptr->m_FbWidth != fbWidth || this_ptr->m_FbHeight != fbHeight)
        {
            glViewport(0, 0, fbWidth, fbHeight);
            this_ptr->m_FbWidth = fbWidth;
            this_ptr->m_FbHeight = fbHeight;
            this_ptr->m_IsResized = true;
        }
    }
}

void Test24Application::CursorPosCallback(GLFWwindow* window, double xpos, double ypos)
{

    if (!window) {
        return;
    }
    auto* app = reinterpret_cast<Test24Application*>(glfwGetWindowUserPointer(window));
    if (!app) {
        return;
    }
    auto   prvCursorPos = app->m_CurCursorPos;
    app->m_CurCursorPos = std::array<float, 2>{static_cast<float>(xpos), static_cast<float>(ypos)};
    app->m_DelCursorPos[0] = app->m_CurCursorPos[0] - prvCursorPos[0];
    app->m_DelCursorPos[1] = app->m_CurCursorPos[1] - prvCursorPos[1];
}

void Test24Application::ScrollCallback(GLFWwindow* window, double xoff, double yoff)
{
    ImGui_ImplGlfw_ScrollCallback(window, xoff, yoff);
    if (!window) {
        return;
    }
    auto* app = reinterpret_cast<Test24Application*>(glfwGetWindowUserPointer(window));
    if (!app) {
        return;
    }
    if (!ImGui::GetIO().WantCaptureMouse) {
        app->m_ScrollOffsets[0] += xoff;
        app->m_ScrollOffsets[1] += yoff;
    }
}

void Test24Application::CopyRtFrame(const std::string &name)
{
    auto fbComponent = m_Framebuffer->GetComponent<test::RTCUGLBufferFBComponent<uchar4>>(name);
    auto rtComponent = m_Framebuffer->GetComponent<test::RTGLTextureFBComponent<uchar4>>("RTexture");
    if (fbComponent && rtComponent)
    {
        auto &frameBufferGL = fbComponent->GetHandle().getHandle();
        auto &renderTexture = rtComponent->GetHandle();
        renderTexture.upload((size_t)0, frameBufferGL, (size_t)0, (size_t)0, (size_t)m_FbWidth, (size_t)m_FbHeight);
    }
}

void Test24Application::CopyDgFrame(const std::string &name)
{
    auto fbComponent = m_Framebuffer->GetComponent<test::RTCUGLBufferFBComponent<uchar4>>(name);
    auto dgComponent = m_Framebuffer->GetComponent<test::RTGLTextureFBComponent<uchar4>>("DTexture");
    if (fbComponent && dgComponent)
    {
        auto &frameBufferGL = fbComponent->GetHandle().getHandle();
        auto &debugTexture = dgComponent->GetHandle();
        debugTexture.upload((size_t)0, frameBufferGL, (size_t)0, (size_t)0, (size_t)m_FbWidth, (size_t)m_FbHeight);
    }
}

void Test24Application::ResizeFrame()
{
    if (m_IsResized)
    {
        m_Framebuffer->Resize(m_FbWidth, m_FbHeight);
    }
}

void Test24Application::UpdateCamera()
{
    if (m_CurFrameTime != 0.0f)
    {
        if (glfwGetKey(m_Window, GLFW_KEY_W) == GLFW_PRESS)
        {

            m_CameraController->ProcessKeyboard(rtlib::ext::CameraMovement::eForward, m_DelFrameTime);
            m_UpdateCamera = true;
        }
        if (glfwGetKey(m_Window, GLFW_KEY_S) == GLFW_PRESS)
        {
            m_CameraController->ProcessKeyboard(rtlib::ext::CameraMovement::eBackward, m_DelFrameTime);
            m_UpdateCamera = true;
        }
        if (glfwGetKey(m_Window, GLFW_KEY_A) == GLFW_PRESS)
        {
            m_CameraController->ProcessKeyboard(rtlib::ext::CameraMovement::eLeft, m_DelFrameTime);
            m_UpdateCamera = true;
        }
        if (glfwGetKey(m_Window, GLFW_KEY_D) == GLFW_PRESS)
        {
            m_CameraController->ProcessKeyboard(rtlib::ext::CameraMovement::eRight, m_DelFrameTime);
            m_UpdateCamera = true;
        }
        if (glfwGetKey(m_Window, GLFW_KEY_LEFT) == GLFW_PRESS)
        {
            m_CameraController->ProcessKeyboard(rtlib::ext::CameraMovement::eLeft, m_DelFrameTime);
            m_UpdateCamera = true;
        }
        if (glfwGetKey(m_Window, GLFW_KEY_RIGHT) == GLFW_PRESS)
        {
            m_CameraController->ProcessKeyboard(rtlib::ext::CameraMovement::eRight, m_DelFrameTime);
            m_UpdateCamera = true;
        }
        if (glfwGetKey(m_Window, GLFW_KEY_UP) == GLFW_PRESS)
        {
            m_CameraController->ProcessKeyboard(rtlib::ext::CameraMovement::eUp, m_DelFrameTime);
            m_UpdateCamera = true;
        }
        if (glfwGetKey(m_Window, GLFW_KEY_DOWN) == GLFW_PRESS)
        {
            m_CameraController->ProcessKeyboard(rtlib::ext::CameraMovement::eDown, m_DelFrameTime);
            m_UpdateCamera = true;
        }
        if (glfwGetMouseButton(m_Window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS)
        {
            if (!ImGui::GetIO().WantCaptureMouse) {
               
                m_CameraController->ProcessMouseMovement(-m_DelCursorPos[0], m_DelCursorPos[1]);
                m_UpdateCamera = true;
            }
        }
        if (m_ScrollOffsets[0] != 0.0f || m_ScrollOffsets[1] != 0.0f) {
            float yoff = m_ScrollOffsets[1];
            m_CameraController->ProcessMouseScroll(-m_ScrollOffsets[1]);
            m_UpdateCamera = true;
        }
    }
    m_ScrollOffsets[0] = 0.0f;
    m_ScrollOffsets[1] = 0.0f;
}

void Test24Application::UpdateFrameTime()
{
    float newTime  = glfwGetTime();
    float delTime  = newTime - m_CurFrameTime;
    m_CurFrameTime = newTime;
    m_DelFrameTime = delTime;
}
