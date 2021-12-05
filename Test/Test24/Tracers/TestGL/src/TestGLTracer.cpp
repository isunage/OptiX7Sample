#include <TestGLTracer.h>
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
Test24TestGLTracer::Test24TestGLTracer(int fbWidth, int fbHeight, GLFWwindow *window, const std::shared_ptr<test::RTFramebuffer> &framebuffer, const std::shared_ptr<rtlib::ext::CameraController> &cameraController,
    bool& isResizedFrame, bool& updateCamera) noexcept:m_IsResizedFrame{isResizedFrame},m_UpdateCamera{updateCamera}
{
    m_FbWidth = fbWidth;
    m_FbHeight = fbHeight;
    m_Window = window;
    m_CameraController = cameraController;
    m_Framebuffer = framebuffer;
    m_FramebufferGL = 0;
    m_ZNear = 0.01f;
    m_ZFar = 10.f;
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
    glClear(GL_COLOR_BUFFER_BIT);
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    glViewport(0, 0, fbWidth, fbHeight);
    {
        m_GLProgram->use();
        glBindBufferBase(GL_UNIFORM_BUFFER, m_UniformLoc, m_UniformBuffer->getID());
        glBindVertexArray(m_VertexArrayGL);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void Test24TestGLTracer::Update()
{
    ResizeFrame();
    UpdateUniforms();
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
        "   vec4 position = viewProj * model * vec4(position,1.0f);\n"
        "   gl_Position   = vec4(1.0f-position.x, position.y, position.z,position.w);\n"
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
        2, 3, 0 };

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
}

void Test24TestGLTracer::InitFramebufferGL()
{
    FreeFramebufferGL();
    if (m_FbWidth == 0 || m_FbHeight == 0) {
        return;
    }
    glGenFramebuffers(1, &m_FramebufferGL);
    glBindFramebuffer(GL_FRAMEBUFFER, m_FramebufferGL);
    auto rtComponent      = m_Framebuffer->GetComponent<test::RTGLTextureFBComponent<uchar4>>("RTexture");
    rtComponent->GetHandle().bind();
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, rtComponent->GetHandle().getID(), 0);
    GLenum drawBuffers[1] = {GL_COLOR_ATTACHMENT0};
    glDrawBuffers(1, drawBuffers);
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
    }
}

void Test24TestGLTracer::ResizeFrame()
{
    if (!m_IsResizedFrame) {
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
        glm::mat4 model = glm::mat4(1.0f) / 2.0f;
        model[3][3] = 1;
        std::memcpy(uniforms.model.data(), &model, sizeof(model));
        auto camera = m_CameraController->GetCamera(static_cast<float>(m_FbWidth) / static_cast<float>(m_FbHeight));
        auto eye = camera.getEye();
        auto center = camera.getLookAt();
        auto vup = camera.getVup();
        auto view = glm::lookAt(glm::vec3(eye.x, eye.y, eye.z), glm::vec3(center.x, center.y, center.z), glm::vec3(vup.x, vup.y, vup.z));
        auto proj = glm::perspective(glm::radians(camera.getFovY()), camera.getAspect(), m_ZNear, m_ZFar);
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
    if (!m_UpdateCamera&&!m_IsResizedFrame) {
        return;
    }
    float aspect = static_cast<float>(m_FbWidth) / static_cast<float>(m_FbHeight);
    Uniforms uniforms = {};
    if (aspect > 0.0f) {
        glm::mat4 model = glm::mat4(1.0f) / 2.0f;
        model[3][3] = 1;
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