#include <TestGLTracer.h>
#include <RTLib/ext/Resources/GL.h>
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
Test24TestGLTracer::Test24TestGLTracer(int fbWidth, int fbHeight, GLFWwindow *window, const std::shared_ptr<test::RTObjModelAssetManager>& objModelManager, const std::shared_ptr<test::RTFramebuffer> &framebuffer, const std::shared_ptr<rtlib::ext::CameraController> &cameraController, const std::string& objModelName,
    const unsigned int& eventFlags) noexcept :m_EventFlags{ eventFlags }, m_NewObjModelName{objModelName}
{
    m_FbWidth = fbWidth;
    m_FbHeight = fbHeight;
    m_Window = window;
    m_ObjModelManager = objModelManager;
    m_CameraController = cameraController;
    m_Framebuffer = framebuffer;
    m_FramebufferGL = 0;
    m_DSTextureGL = 0;
    m_ZNear = 0.01f;
    m_ZFar = 3000.f;
    m_UpdateObjModel  = false;
}

void Test24TestGLTracer::Initialize()
{

    InitProgramGL();
    InitVertexArray();
    InitUniforms();
    InitFramebufferGL();
}

void Test24TestGLTracer::CleanUp()
{
    FreeFramebufferGL();
    FreeUniforms();
    FreeVertexArray();
    FreeProgramGL();
}

void Test24TestGLTracer::Launch(int fbWidth, int fbHeight, void *pUserData)
{
    glBindFramebuffer(GL_FRAMEBUFFER, m_FramebufferGL);
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    glViewport(0, 0, fbWidth, fbHeight );
    if (!m_ObjModelManager->GetAssets().empty()) {
        if (!m_ObjModelManager->HasAsset(m_CurObjModelName))
        {
        }
        else {
            if (m_CurObjModelName != m_NewObjModelName) {
                std::cout << "Update2" << std::endl;
                m_UpdateObjModel = true;
            }
        }
    }
    else {
        if (m_ObjModel.meshGroup) {
            std::cout << "Update3" << std::endl;
            m_UpdateObjModel = true;
        }
    }
    m_GLProgram->use();
    glBindBufferBase(GL_UNIFORM_BUFFER, m_UniformLoc, m_UniformBuffer->getID());
    if (!m_ObjModel.meshGroup) {
        glBindVertexArray(m_VertexArrayGL);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);
    }
    else {
        for (auto& [name, uniqueResource] : m_ObjModel.meshGroup->GetUniqueResources()) {
            glBindVertexArray(uniqueResource->variables.GetUInt32("VertexArrayGL"));
            glDrawElements(GL_TRIANGLES, uniqueResource->triIndBuffer.Size() * 3, GL_UNSIGNED_INT, 0);
            glBindVertexArray(0);
        }
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void Test24TestGLTracer::Update()
{
    ResizeFrame();
    UpdateUniforms();
    UpdateVertexArray();
}

void Test24TestGLTracer::InitProgramGL()
{
    constexpr std::string_view vsSource =
        "#version 330 core\n"
        "layout(std140) uniform Uniforms{\n"
        "   mat4 model;\n"
        "   mat4 viewProj;\n"
        "};\n"
        "layout(location=0) in vec3 position;\n"
        "layout(location=1) in vec2 texCoord;\n"
        "out vec2 uv;\n"
        "void main(){\n"
        "   vec4 pos   = viewProj * model * vec4(position,1.0f);\n"
        "   gl_Position= vec4(pos.x, pos.y,pos.z,pos.w);\n"
        "   uv = texCoord;\n"
        "}\n";
    constexpr std::string_view fsSource =
        "#version 330 core\n"
        "in vec2 uv;\n"
        "layout(location=0) out vec4 color;\n"
        "void main(){\n"
        "   color = vec4(uv.x,uv.y,1.0f-dot(vec2(0.5f),uv),1.0f);\n"
        "}\n";

    m_GLProgram = std::make_unique<rtlib::GLProgram>();
    {
        m_GLProgram->create();
        auto vs = rtlib::GLVertexShader(std::string(vsSource));
        auto fs = rtlib::GLFragmentShader(std::string(fsSource));
        if (!vs.compile())
        {
            std::cout << vs.getLog() << std::endl;
        }
        if (!fs.compile())
        {
            std::cout << fs.getLog() << std::endl;
        }
        m_GLProgram->attach(vs);
        m_GLProgram->attach(fs);
        if (!m_GLProgram->link())
        {
            std::cout << m_GLProgram->getLog() << std::endl;
        }
    }
}

void Test24TestGLTracer::FreeProgramGL()
{
    m_GLProgram->destroy();
    m_GLProgram.reset();
}

void Test24TestGLTracer::InitVertexArray()
{
    constexpr std::array<float3, 4> screenVertices = {
        float3{ -1.0f, -1.0f, 0.5f },
        float3{  1.0f, -1.0f, 0.5f },
        float3{  1.0f,  1.0f, 0.5f },
        float3{ -1.0f,  1.0f, 0.5f } };
    constexpr std::array<float2, 4> screenTexCoords = {
        float2{1.0f, 0.0f},
        float2{0.0f, 0.0f},
        float2{0.0f, 1.0f},
        float2{1.0f, 1.0f}
    };
    constexpr std::array<std::uint32_t, 6> screenIndices = {
        0, 1, 2,
        2, 3, 0 
    };
    
    m_VertexBuffer  = std::make_unique<rtlib::GLBuffer<float3>>( screenVertices, GL_ARRAY_BUFFER, GL_STATIC_DRAW);
    m_TexCrdBuffer  = std::make_unique<rtlib::GLBuffer<float2>>(screenTexCoords, GL_ARRAY_BUFFER, GL_STATIC_DRAW);
    m_IndexBuffer   = std::make_unique<rtlib::GLBuffer<uint32_t>>(screenIndices, GL_ELEMENT_ARRAY_BUFFER, GL_STATIC_DRAW);
    m_VertexArrayGL = 0;
    glGenVertexArrays(1, &m_VertexArrayGL);
    glBindVertexArray(   m_VertexArrayGL);
    m_IndexBuffer ->bind();
    m_VertexBuffer->bind();
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float3), nullptr);
    glEnableVertexAttribArray(0);
    m_TexCrdBuffer->bind();
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(float2), nullptr);
    glEnableVertexAttribArray(1);
    glBindVertexArray(0);
    m_TexCrdBuffer->unbind();
    m_VertexBuffer->unbind();
    m_IndexBuffer ->unbind();
}

void Test24TestGLTracer::FreeVertexArray()
{
    if (m_ObjModel.meshGroup) {
        for (auto& [name, uniqueRes] : m_ObjModel.meshGroup->GetUniqueResources())
        {
            GLuint vertexArrayGL = uniqueRes->variables.GetUInt32("VertexArrayGL");
            glDeleteVertexArrays(1, &vertexArrayGL);
            uniqueRes->variables.SetUInt32("VertexArrayGL", 0);
        }
    }
    m_ObjModel = {};
    if (m_VertexArrayGL) {
        glDeleteVertexArrays(1, &m_VertexArrayGL);
        m_VertexArrayGL = 0;
    }
    m_VertexBuffer->reset();
    m_VertexBuffer.reset();
    m_TexCrdBuffer->reset();
    m_TexCrdBuffer.reset();
    m_IndexBuffer->reset();
    m_IndexBuffer.reset();
    m_UpdateObjModel = false;
}

void Test24TestGLTracer::UpdateVertexArray()
{
    auto& meshGroup = m_ObjModel.meshGroup;
    if (!meshGroup) {
        return;
    }
    if (m_UpdateObjModel) {
        if (meshGroup) {
            for (auto& [name, uniqueRes] : meshGroup->GetUniqueResources())
            {
                GLuint vertexArrayGL =  uniqueRes->variables.GetUInt32("VertexArrayGL");
                glDeleteVertexArrays(1, &vertexArrayGL);
                uniqueRes->variables.SetUInt32("VertexArrayGL",0);
            }
        }
        m_ObjModel = {};
        if (!m_ObjModelManager->GetAssets().empty()) {
            if (!m_ObjModelManager->HasAsset(m_NewObjModelName)) {
                return;
            }
            m_CurObjModelName = m_NewObjModelName;
            m_ObjModel        = m_ObjModelManager->GetAsset(m_CurObjModelName);
            if (meshGroup) {
                auto sharedResource = meshGroup->GetSharedResource();
                if (!sharedResource->vertexBuffer.HasGpuComponent("GLBuffer")) {
                    sharedResource->vertexBuffer.AddGpuComponent<rtlib::ext::resources::GLBufferComponent<float3>>("GLBuffer");
                }
                if (!sharedResource->texCrdBuffer.HasGpuComponent("GLBuffer")) {
                    sharedResource->texCrdBuffer.AddGpuComponent<rtlib::ext::resources::GLBufferComponent<float2>>("GLBuffer");
                }
                if (!sharedResource->normalBuffer.HasGpuComponent("GLBuffer")) {
                    sharedResource->normalBuffer.AddGpuComponent<rtlib::ext::resources::GLBufferComponent<float3>>("GLBuffer");
                }
                for (auto& [name, uniqueRes] : meshGroup->GetUniqueResources()) {
                    if (!uniqueRes->triIndBuffer.HasGpuComponent("GLBuffer")) {
                        uniqueRes->triIndBuffer.AddGpuComponent<rtlib::ext::resources::GLBufferComponent<uint3>>("GLBuffer");
                    }
                    uniqueRes->triIndBuffer.GetGpuComponent<rtlib::ext::resources::GLBufferComponent<uint3>>("GLBuffer")->GetHandle().setTarget(GL_ELEMENT_ARRAY_BUFFER);
                    GLuint vertexArrayGL;
                    glGenVertexArrays(1, &vertexArrayGL);
                    glBindVertexArray(vertexArrayGL);
                    uniqueRes->triIndBuffer.GetGpuComponent<rtlib::ext::resources::GLBufferComponent<uint3>>("GLBuffer")->GetHandle().bind();
                    sharedResource->vertexBuffer.GetGpuComponent<rtlib::ext::resources::GLBufferComponent<float3>>("GLBuffer")->GetHandle().bind();
                    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
                    glEnableVertexAttribArray(0);
                    sharedResource->texCrdBuffer.GetGpuComponent<rtlib::ext::resources::GLBufferComponent<float2>>("GLBuffer")->GetHandle().bind();
                    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
                    glEnableVertexAttribArray(1);
                    glBindVertexArray(0);
                    uniqueRes->triIndBuffer.GetGpuComponent<rtlib::ext::resources::GLBufferComponent<uint3>>("GLBuffer")->GetHandle().unbind();
                    sharedResource->vertexBuffer.GetGpuComponent<rtlib::ext::resources::GLBufferComponent<float3>>("GLBuffer")->GetHandle().unbind();
                    sharedResource->texCrdBuffer.GetGpuComponent<rtlib::ext::resources::GLBufferComponent<float2>>("GLBuffer")->GetHandle().unbind();
                    uniqueRes->variables.SetUInt32("VertexArrayGL", vertexArrayGL);
                }
            }
        }
    }
    m_UpdateObjModel = false;
}

void Test24TestGLTracer::InitFramebufferGL()
{
    FreeFramebufferGL();
    if (m_FbWidth == 0 || m_FbHeight == 0) {
        return;
    }
    glGenTextures(1, &m_DSTextureGL);
    glBindTexture(GL_TEXTURE_2D, m_DSTextureGL);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH24_STENCIL8, m_FbWidth, m_FbHeight, 0, GL_DEPTH_STENCIL, GL_UNSIGNED_INT_24_8, nullptr);
    glGenFramebuffers(1, &m_FramebufferGL);
    glBindFramebuffer(GL_FRAMEBUFFER, m_FramebufferGL);
    auto rtComponent      = m_Framebuffer->GetComponent<test::RTGLTextureFBComponent<uchar4>>("PTexture");
    rtComponent->GetHandle().bind();
    glFramebufferTexture2D(  GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, rtComponent->GetHandle().getID(), 0);
    glFramebufferTexture2D(  GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_TEXTURE_2D, m_DSTextureGL, 0);
    GLenum drawBuffers[2] = {GL_COLOR_ATTACHMENT0,GL_DEPTH_STENCIL_ATTACHMENT };
    glDrawBuffers(2, drawBuffers);
    GLint status          = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (status != GL_FRAMEBUFFER_COMPLETE)
    {
        std::cout << std::hex << status << std::endl;
        std::cout << "Error!\n";
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    rtComponent->GetHandle().unbind();
}

void Test24TestGLTracer::FreeFramebufferGL()
{
    if (m_FramebufferGL)
    {
        glDeleteFramebuffers(1, &m_FramebufferGL);
        m_FramebufferGL = 0;
        glDeleteTextures(1, &m_DSTextureGL);
        m_DSTextureGL = 0;
    }
}

void Test24TestGLTracer::ResizeFrame()
{
    if ((m_EventFlags & TEST24_EVENT_FLAG_RESIZE_FRAME) != TEST24_EVENT_FLAG_RESIZE_FRAME) {
        return;
    }
    if (m_FbWidth != m_Framebuffer->GetWidth() || m_FbHeight != m_Framebuffer->GetHeight())
    {
        m_FbWidth  = m_Framebuffer->GetWidth();
        m_FbHeight = m_Framebuffer->GetHeight();
        InitFramebufferGL();
    }
}

void Test24TestGLTracer::InitUniforms()
{
    Uniforms uniforms = {};
    {
        glm::mat4 model = glm::mat4(1.0f) ;
        std::memcpy(uniforms.model.data(), &model, sizeof(model));
        auto camera = m_CameraController->GetCamera(static_cast<float>(m_FbWidth) / static_cast<float>(m_FbHeight));
        auto eye    = camera.getEye();
        auto center = camera.getLookAt();
        auto vup    = camera.getVup();
        auto view   = glm::lookAt(glm::vec3(eye.x, eye.y, eye.z), glm::vec3(center.x, center.y, center.z), glm::vec3(vup.x, vup.y, vup.z));
        auto proj   = glm::perspective(glm::radians(camera.getFovY()), camera.getAspect(), m_ZNear, m_ZFar);
        glm::mat4 viewProj = proj * view;
        std::memcpy(uniforms.viewProj.data(), &viewProj, sizeof(viewProj));
    }
    GLint blockIdx = m_GLProgram->getUniformBlockIndex("Uniforms");
    m_UniformLoc = 1;
    m_GLProgram->setUniformBlockBinding(blockIdx, m_UniformLoc);
    m_UniformBuffer = std::make_unique<rtlib::GLBuffer<Uniforms>>(std::array<Uniforms, 1>{uniforms}, GL_UNIFORM_BUFFER, GL_DYNAMIC_DRAW);
}

void Test24TestGLTracer::FreeUniforms()
{
    m_UniformBuffer->reset();
    m_UniformBuffer.reset();
    m_UniformLoc = -1;
}

void Test24TestGLTracer::UpdateUniforms()
{
    if ((m_EventFlags&TEST24_EVENT_FLAG_UPDATE_CAMERA)!= TEST24_EVENT_FLAG_UPDATE_CAMERA) {
        return;
    }
    float aspect = static_cast<float>(m_FbWidth) / static_cast<float>(m_FbHeight);
    Uniforms uniforms = {};
    if (aspect > 0.0f) {
        glm::mat4 model = glm::mat4(1.0f);
        std::memcpy(uniforms.model.data(), &model, sizeof(model));
        auto camera = m_CameraController->GetCamera(aspect);
        auto eye = camera.getEye();
        auto center = camera.getLookAt();
        auto vup = camera.getVup();
        auto view = glm::lookAt(glm::vec3(eye.x, eye.y, eye.z), glm::vec3(center.x, center.y, center.z), glm::vec3(vup.x, vup.y, vup.z));
        auto proj = glm::perspective(glm::radians(camera.getFovY()), camera.getAspect(), m_ZNear, m_ZFar);
        glm::mat4 viewProj = proj * view;
        std::memcpy(uniforms.viewProj.data(), &viewProj, sizeof(viewProj));
    }
    m_UniformBuffer->upload(std::vector{ uniforms });
    m_UniformBuffer->unbind();
}
