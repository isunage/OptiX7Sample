#include "..\include\Test24Application.h"
#include <TestGLTracer.h>
Test24Application::Test24Application(int fbWidth, int fbHeight, std::string name) noexcept : test::RTApplication(name)
{
    m_FbWidth = fbWidth;
    m_FbHeight = fbHeight;
    m_Window = nullptr;
    m_FovY = 0.0f;
    m_IsResized = false;
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
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    while (!glfwWindowShouldClose(m_Window))
    {
        this->Launch();
        glFlush();
        glClear(GL_COLOR_BUFFER_BIT);
        this->Render();
        this->Update();
        glfwSwapBuffers(m_Window);
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
    m_Window  = m_Context->NewWindow(m_FbWidth, m_FbHeight, GetName().c_str());
    glfwSetWindowUserPointer(m_Window, this);

    m_Framebuffer = std::shared_ptr<test::RTFramebuffer>(new test::RTFramebuffer(m_FbWidth, m_FbHeight));
    //GBuffer
    m_Framebuffer->AddComponent<test::RTCUDABufferFBComponent<float3>>("GPosition");
    m_Framebuffer->AddComponent<test::RTCUDABufferFBComponent<float3>>("GNormal"  );
    m_Framebuffer->AddComponent<test::RTCUDABufferFBComponent<float2>>("GTexCoord");
    m_Framebuffer->AddComponent<test::RTCUDABufferFBComponent<float> >("GDepth"   );
    //Render
    m_Framebuffer->AddComponent<test::RTCUDABufferFBComponent<float4>>("RAccum");
    m_Framebuffer->AddComponent<test::RTCUGLBufferFBComponent<uchar4>>("RFrame");
    //Debug
    m_Framebuffer->AddComponent<test::RTCUGLBufferFBComponent<uchar4>>("DNormal");
    m_Framebuffer->GetComponent<test::RTCUGLBufferFBComponent<uchar4>>("DNormal")->GetHandle().upload(
        std::vector<uchar4>(m_FbWidth * m_FbHeight, make_uchar4(255, 0, 0, 255))
    );
    m_Framebuffer->AddComponent<test::RTCUGLBufferFBComponent<uchar4>>("DTexCoord");
    m_Framebuffer->AddComponent<test::RTCUGLBufferFBComponent<uchar4>>("DDistance");
    m_Framebuffer->AddComponent<test::RTCUGLBufferFBComponent<uchar4>>("DDiffuse");
    m_Framebuffer->AddComponent<test::RTCUGLBufferFBComponent<uchar4>>("DSpecular");
    m_Framebuffer->AddComponent<test::RTCUGLBufferFBComponent<uchar4>>("DTransmit");
    m_Framebuffer->AddComponent<test::RTCUGLBufferFBComponent<uchar4>>("DShinness");
    m_Framebuffer->AddComponent<test::RTCUGLBufferFBComponent<uchar4>>("DIOR");
    //Texture
    m_Framebuffer->AddComponent<test::RTGLTextureFBComponent<uchar4>>("RTexture");
    m_Framebuffer->AddComponent<test::RTGLTextureFBComponent<uchar4>>("DTexture");
    m_Framebuffer->AddComponent<test::RTGLTextureFBComponent<uchar4>>("TTexture",GL_NEAREST,GL_NEAREST,GL_CLAMP_TO_EDGE,GL_CLAMP_TO_EDGE);
    //Renderer
    m_Renderer = std::make_unique<rtlib::ext::RectRenderer>();
    m_Renderer->init();
}

void Test24Application::FreeBase()
{
    m_Gui.reset();
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
    //Gui
    m_Gui            = std::make_shared<test::RTGui>(m_Window);
    m_Gui->Initialize();
    auto mainMenuBar = m_Gui->AddGuiMainMenuBar();
    auto fileMenu    = mainMenuBar->AddGuiMenu("File");
    auto  newMenu    = fileMenu->AddGuiMenu("New");
    auto projItem    = std::make_shared<test::RTGuiMenuItem>("project"   );
    auto repoItem    = std::make_shared<test::RTGuiMenuItem>("repository");
    auto emitMenu    = mainMenuBar->AddGuiMenu("Edit");
    newMenu->SetGuiMenuItem(projItem);
    newMenu->SetGuiMenuItem(repoItem);
}

void Test24Application::FreeGui()
{
    m_Gui.reset();
}

void Test24Application::RenderGui()
{
    m_Gui->DrawFrame();
}

void Test24Application::InitScene()
{
    m_CameraController = std::make_shared<rtlib::ext::CameraController>();
    m_FovY = 40.0f;
}

void Test24Application::FreeScene()
{

}

void Test24Application::InitTracers()
{
    m_Tracers["TestGL"] = std::make_shared<Test24TestGLTracer>(m_FbWidth, m_FbHeight,m_Window, m_Framebuffer, m_CameraController);
    m_Tracers["TestGL"]->Initialize();
}

void Test24Application::FreeTracers()
{
    for (auto& [name, tracer] : m_Tracers) {
        tracer->CleanUp();
    }
    m_Tracers.clear();
}

void Test24Application::RenderFrame(const std::string &name)
{
    if (m_Framebuffer->HasComponent<test::RTGLTextureFBComponent<uchar4>>(name)) {
        auto rtComponent    = m_Framebuffer->GetComponent<test::RTGLTextureFBComponent<uchar4>>(name);
        if (!rtComponent)
        {
            return;
        }
        auto& renderTexture = rtComponent->GetHandle();
        m_Renderer->draw(renderTexture.getID());
        return;
    }
    if (m_Framebuffer->HasComponent<test::RTCUGLBufferFBComponent<uchar4>>(name)) {
        this->CopyRtFrame(name);
        auto rtComponent = m_Framebuffer->GetComponent<test::RTGLTextureFBComponent<uchar4>>("RTexture");
        if (!rtComponent)
        {
            return;
        }
        auto& renderTexture = rtComponent->GetHandle();
        m_Renderer->draw(renderTexture.getID());
    }
}

void Test24Application::Launch()
{
    m_Tracers["TestGL"]->Launch(m_FbWidth, m_FbHeight, nullptr);
}

void Test24Application::Render()
{
    this->RenderFrame("TTexture");
    this->RenderGui();
}

void Test24Application::Update()
{
    glfwPollEvents();

}

void Test24Application::FramebufferSizeCallback(GLFWwindow* window, int fbWidth, int fbHeight)
{
    auto* this_ptr = reinterpret_cast<Test24Application*>(glfwGetWindowUserPointer(window));
    if (this_ptr) {
        if (this_ptr->m_FbWidth != fbWidth || this_ptr->m_FbHeight != fbHeight)
        {
            glViewport(0, 0, fbWidth, fbHeight);
            this_ptr->m_FbWidth = fbWidth;
            this_ptr->m_FbHeight = fbHeight;
            this_ptr->m_IsResized = true;
        }
    }
}

void Test24Application::CopyRtFrame(const std::string& name)
{
    auto fbComponent = m_Framebuffer->GetComponent<test::RTCUGLBufferFBComponent<uchar4>>(name);
    auto rtComponent = m_Framebuffer->GetComponent<test::RTGLTextureFBComponent<uchar4>>("RTexture");
    if (fbComponent && rtComponent)
    {
        auto& frameBufferGL = fbComponent->GetHandle().getHandle();
        auto& renderTexture = rtComponent->GetHandle();
        renderTexture.upload((size_t)0, frameBufferGL, (size_t)0, (size_t)0, (size_t)m_FbWidth, (size_t)m_FbHeight);
    }
}

void Test24Application::CopyDgFrame(const std::string& name)
{
    auto fbComponent = m_Framebuffer->GetComponent<test::RTCUGLBufferFBComponent<uchar4>>(name);
    auto dgComponent = m_Framebuffer->GetComponent<test::RTGLTextureFBComponent<uchar4>>("DTexture");
    if (fbComponent && dgComponent)
    {
        auto& frameBufferGL = fbComponent->GetHandle().getHandle();
        auto& debugTexture  = dgComponent->GetHandle();
        debugTexture.upload((size_t)0, frameBufferGL, (size_t)0, (size_t)0, (size_t)m_FbWidth, (size_t)m_FbHeight);
    }
}
