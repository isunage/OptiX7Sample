#include <TestGLTracer.h>
#include <glm/glm.hpp>
Test24TestGLTracer::Test24TestGLTracer(int fbWidth, int fbHeight, GLFWwindow* window, const std::shared_ptr<test::RTFramebuffer>& framebuffer, const std::shared_ptr < rtlib::ext::CameraController >& cameraController) noexcept
{
	m_FbWidth  = fbWidth;
	m_FbHeight = fbHeight;
	m_Window   = window;
	m_CameraController = cameraController;
	m_Framebuffer      = framebuffer;
}

void Test24TestGLTracer::Initialize()
{
    constexpr std::array<TestVertex, 4>    screenVertices = {
                TestVertex{{-1.0f,-1.0f,0.0f},{1.0f, 0.0f}},
                TestVertex{{ 1.0f,-1.0f,0.0f},{0.0f, 0.0f}},
                TestVertex{{ 1.0f, 1.0f,0.0f},{0.0f, 1.0f}},
                TestVertex{{-1.0f, 1.0f,0.0f},{1.0f, 1.0f}}
    };
    constexpr std::array<std::uint32_t,6>  screenIndices  = {
        0,1,2,
        2,3,0
    };
    constexpr std::string_view vsSource =
        "#version 330 core\n"
        //"layout(std140) uniform Uniforms{\n"
        //"   mat4 model;\n"
        //"   mat4 viewProj;\n"
        //"} uniforms;\n"
        "layout(location=0) in vec3 position;\n"
        "layout(location=1) in vec2 texCoord;\n"
        "out vec2 uv;\n"
        "void main(){\n"/*
        "   gl_Position = uniforms.viewProj * uniforms.model * vec4(position,1.0f);\n"*/
        "   gl_Position = vec4(position,1.0f);\n"
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
        if (!vs.compile()) {
            std::cout << vs.getLog() << std::endl;
        }
        if (!fs.compile()) {
            std::cout << fs.getLog() << std::endl;
        }
        m_GLProgram->attach(vs);
        m_GLProgram->attach(fs);
        if (!m_GLProgram->link()) {
            std::cout << m_GLProgram->getLog() << std::endl;
        }
    }

    m_UniformLoc   = m_GLProgram->getUniformLocation("uniforms");
    m_VertexBuffer = std::make_unique<rtlib::GLBuffer<TestVertex>>(screenVertices, GL_ARRAY_BUFFER, GL_STATIC_DRAW);
    m_IndexBuffer  = std::make_unique<rtlib::GLBuffer<uint32_t>>(screenIndices,  GL_ELEMENT_ARRAY_BUFFER, GL_STATIC_DRAW);
    m_VertexArrayGL= 0;

    glGenVertexArrays(1, &m_VertexArrayGL);
    glBindVertexArray(m_VertexArrayGL);

    m_VertexBuffer->bind();
    m_IndexBuffer->bind();

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(TestVertex), &static_cast<TestVertex*>(nullptr)->position);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(TestVertex), &static_cast<TestVertex*>(nullptr)->texCoord);
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glBindVertexArray(0);

    m_VertexBuffer->unbind();
    m_IndexBuffer->unbind();

    auto rtComponent = m_Framebuffer->GetComponent<test::RTGLTextureFBComponent<uchar4>>("TTexture");

    m_FramebufferGL = 0;
    glGenFramebuffers(1, &m_FramebufferGL);
    glBindFramebuffer(GL_FRAMEBUFFER, m_FramebufferGL);
    rtComponent->GetHandle().bind();

    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D, rtComponent->GetHandle().getID(), 0);
    GLenum drawBuffers[1] = { GL_COLOR_ATTACHMENT0 };
    glDrawBuffers(1, drawBuffers);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
    {
        std::cout << "Error!\n";
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    rtComponent->GetHandle().unbind();

}

void Test24TestGLTracer::CleanUp()
{
    m_UniformLoc = -1;
    m_GLProgram->destroy();
    m_GLProgram.reset();
}

void Test24TestGLTracer::Launch(int fbWidth, int fbHeight, void* pUserData)
{
    glBindFramebuffer(GL_FRAMEBUFFER, m_FramebufferGL);
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    glViewport(0, 0, fbWidth, fbHeight);
    {
        m_GLProgram->use();
        glBindVertexArray(m_VertexArrayGL);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void Test24TestGLTracer::Update()
{
}
