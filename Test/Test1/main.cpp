#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <RTLib/CUDA.h>
#include <RTLib/CUDA_GL.h>
#include <RTLib/Utils.h>
#include <RTLib/Math.h>
#include <RTLib/VectorFunction.h>
#include <RTLib/Random.h>
#include <GLFW/glfw3.h>
#include <stb_image.h>
#include <stb_image_resize.h>
#include <stb_image_write.h>
#include <Test1Config.h>
#include <iostream>
#include <string_view>
#include <random>
#include <array>
constexpr std::array<float, 3 * 3> vertices = {
    -1.0f,-1.0f,0.0f,
     1.0f,-1.0f,0.0f,
     0.0f, 1.0f,0.0f
};
constexpr std::array<uint32_t, 3>  indices  = {
    0,1,2
};
constexpr std::string_view vsSource =
"#version 330 core\n"
"layout(location=0) in vec3 position;\n"
"out vec2 uv;\n"
"void main(){\n"
"   gl_Position = vec4(position,1.0f);\n"
"   uv = vec2((position.x+1.0f)/2.0f, (position.y+1.0f)/2.0f);\n"
"}\n";
constexpr std::string_view fsSource =
"#version 330 core\n"
"uniform sampler2D tex;\n"
"in vec2 uv;\n"
"layout(location=0) out vec3 color;\n"
"void main(){\n"
"   color = texture2D(tex,uv).xyz;\n"
"}\n";
int main(){
    if (glfwInit() != GLFW_TRUE) {
        throw std::runtime_error("Failed To Initialize GLFW!");
    }
    glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    GLFWwindow* window = glfwCreateWindow(640, 480, "title", nullptr, nullptr);
    if (!window) {
        throw std::runtime_error("Failed To Create Window!");
    }
    glfwMakeContextCurrent(window);
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        throw std::runtime_error("Failed To Load GLAD!");
    }
    auto program = rtlib::GLProgram();
    {
        program.create();
        auto vs = rtlib::GLVertexShader(  std::string(vsSource));
        auto fs = rtlib::GLFragmentShader(std::string(fsSource));
        if (!vs.compile()) {
            std::cout << vs.getLog() << std::endl;
        }
        if (!fs.compile()) {
            std::cout << fs.getLog() << std::endl;
        }
        program.attach(vs);
        program.attach(fs);
        if (!program.link()) {
            std::cout << program.getLog() << std::endl;
        }
    }
    auto vertexInteropBuffer = rtlib::GLInteropBuffer<float   >(9,         GL_ARRAY_BUFFER, GL_STREAM_DRAW);
    auto  indexInteropBuffer = rtlib::GLInteropBuffer<uint32_t>(3, GL_ELEMENT_ARRAY_BUFFER, GL_STREAM_DRAW);
    //auto vertexBuffer = rtlib::GLBuffer(vertices,        GL_ARRAY_BUFFER, GL_STATIC_DRAW);
       //auto  indexBuffer = rtlib::GLBuffer(indices, GL_ELEMENT_ARRAY_BUFFER, GL_STATIC_DRAW);
    {
        float* d_VertexPtr   = vertexInteropBuffer.map();
        RTLIB_CUDA_CHECK(cudaMemcpy(d_VertexPtr, vertices.data(), sizeof(vertices[0]) * vertices.size(), cudaMemcpyHostToDevice));
        vertexInteropBuffer.unmap();
        uint32_t* d_IndexPtr = indexInteropBuffer.map();
        RTLIB_CUDA_CHECK(cudaMemcpy(d_IndexPtr, indices.data(),   sizeof(indices[0]) * indices.size(), cudaMemcpyHostToDevice));
        indexInteropBuffer.unmap();
    }
    auto cuTexture = rtlib::CUDATexture2D<uchar4>();
    {
        int imgWidth  = 0;
        int imgHeight = 0;
        int imgComp   = 0;
        {
            void* imgData = stbi_load(TEST_TEST1_DATA_PATH"/Textures/sample.png", &imgWidth, &imgHeight, &imgComp, 4);
            cuTexture.allocate(imgWidth, imgHeight);
            cuTexture.upload(imgData, cuTexture.getWidth(), cuTexture.getHeight());
            stbi_image_free(imgData);
        }
        {
            std::vector<uchar4> pixelData(imgWidth / 2 * imgHeight / 2);
            cuTexture.download(reinterpret_cast<void*>(pixelData.data()), imgWidth / 2, imgHeight / 2);
            stbi_write_bmp("tekitou.png", imgWidth / 2, imgHeight / 2, 4, reinterpret_cast<void*>(pixelData.data()));
        }
    }
    auto glTexture = rtlib::GLTexture2D<uchar4>();
    {
        int imgWidth = 0;
        int imgHeight = 0;
        int imgComp = 0;
        {
            unsigned char* imgData_0 = stbi_load(TEST_TEST1_DATA_PATH"/Textures/sample.png", &imgWidth, &imgHeight, &imgComp, 4);
            auto imageBuffer = rtlib::GLBuffer<uchar4>(reinterpret_cast<uchar4*>(imgData_0),imgWidth*imgHeight,GL_PIXEL_UNPACK_BUFFER,GL_STATIC_DRAW);
            glTexture.allocateWithMipLevel({ (size_t)imgWidth, (size_t)imgHeight}, 6);
            glTexture.upload(0,imageBuffer,0,0,glTexture.getViews()[0].width,glTexture.getViews()[0].height);
            glTexture.generateMipmaps(false);
            glTexture.setParameteri(GL_TEXTURE_MAG_FILTER, GL_LINEAR       , false);
            glTexture.setParameteri(GL_TEXTURE_MIN_FILTER, GL_LINEAR       , false);
            glTexture.setParameteri(GL_TEXTURE_WRAP_S    , GL_CLAMP_TO_EDGE, false);
            glTexture.setParameteri(GL_TEXTURE_WRAP_T    , GL_CLAMP_TO_EDGE, false);
            unsigned char* t_ImgData = new unsigned char[imgWidth * imgHeight * 4];
            size_t level = 0;
            for (const auto& view:glTexture.getViews()) {
                bool res = glTexture.download(level, (void*)t_ImgData);
                stbi_write_png((std::string("tekiou") + std::to_string(level) + ".png").c_str(), view.width, view.height, 4, t_ImgData, 0);
                ++level;
            }
            stbi_image_free(imgData_0);
            delete[] t_ImgData;
        }
        
    }
    auto vertexArray  = GLuint(0);
    {
        glGenVertexArrays(1, &vertexArray);
        glBindVertexArray(vertexArray);
        vertexInteropBuffer.bind();
         indexInteropBuffer.bind();
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 3, nullptr);
        glEnableVertexAttribArray(0);
        glBindVertexArray(0);
        vertexInteropBuffer.unbind();
         indexInteropBuffer.unbind();
    }
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    GLint texLoc = program.getUniformLocation("tex");
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        {
            int t_Width, t_Height;
            glfwGetFramebufferSize(window, &t_Width, &t_Height);
            if (t_Width != width || t_Height != height) {
                while (t_Width<1||t_Height<1) {
                    glfwGetFramebufferSize(window, &t_Width, &t_Height);
                }
                width  = t_Width;
                height = t_Height;
                glViewport(0, 0, t_Width, t_Height);
            }
        }
        glClear(GL_COLOR_BUFFER_BIT);
        program.use();
        glActiveTexture(GL_TEXTURE0);
        glTexture.bind();
        glUniform1i(texLoc, 0);
        glBindVertexArray(vertexArray);
        glDrawElements(GL_TRIANGLES, 3, GL_UNSIGNED_INT, nullptr);
        glfwSwapBuffers(window);
    }
    program.destroy();
    glDeleteVertexArrays(1, &vertexArray);
    vertexInteropBuffer.reset();
     indexInteropBuffer.reset();
              glTexture.reset();
    glfwDestroyWindow(window);
    window = nullptr;
    glfwTerminate();
}