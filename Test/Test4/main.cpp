#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <RTLib/Optix.h>
#include <RTLib/Config.h>
#include <RTLib/ext/Math/VectorFunction.h>
#include <RTLib/Exceptions.h>
#include <GLFW/glfw3.h>
#include <stb_image.h>
#include <stb_image_write.h>
#include <iostream>
#include <string_view>
static constexpr std::string_view vsSource =
"#version 430 core\n"
"layout(location = 0) in vec3 position;\n"
"out vec2 uv;\n"
"void main(){\n"
"   gl_Position = vec4(position,1.0f);\n"
"   uv = vec2(position.x,position.y);\n"
"}\n";
static constexpr std::string_view fsSource =
"#version 430 core\n"
"uniform sampler2D sampler;\n"
"in vec2 uv;\n"
"layout(location = 0) out vec3 color;\n"
"void main(){\n"
"   color = texture2D(sampler,uv).rgb;\n"
"}\n";
static constexpr std::string_view csSource =
"#version 430 core\n"
"uniform unsigned int width;\n"
"uniform unsigned int height;\n"
"layout(local_size_x=32,local_size_y=32) in;\n"
"layout(rgba8,binding = 0) readonly  uniform image2D srcTex;\n"
"layout(rgba8,binding = 1) writeonly uniform image2D dstTex;\n"
"void main(){\n"
"   if(gl_GlobalInvocationID.x<width && gl_GlobalInvocationID.y<height){\n"
"       ivec2 pos = ivec2(gl_GlobalInvocationID.xy);\n"
"       vec4 srcPixel = imageLoad(srcTex,pos);\n"
"       vec4 dstPixel = vec4(vec3((srcPixel.x+srcPixel.y+srcPixel.z)/3),255);\n"
"       imageStore(dstTex, pos, dstPixel);\n"
"   }\n"
"}\n";
int main(){
    if(glfwInit()!=GLFW_TRUE){
        std::cerr << "Failed To Init GLFW!\n";
    }
    glfwWindowHint(GLFW_CLIENT_API,GLFW_OPENGL_API);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,3);
    glfwWindowHint(GLFW_OPENGL_PROFILE,GLFW_OPENGL_CORE_PROFILE);
    GLFWwindow* window = glfwCreateWindow(640,480,"Title",nullptr,nullptr);
    if(!window){
        std::cerr << "Failed To Create Window!\n";
    }
    glfwMakeContextCurrent(window);
    if(!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)){
        std::cerr << "Failed To Init OpenGL!\n";
    }
    auto renderProgram = rtlib::GLProgram();
    {
        renderProgram.create();
        auto vs = rtlib::GLVertexShader(  std::string(vsSource));
        if(!vs.compile()){
            std::cout << "Vertex Shader Log\n";
            std::cout << vs.getLog() <<   "\n";
        }
        auto fs = rtlib::GLFragmentShader(std::string(fsSource));
        if(!fs.compile()){
            std::cout << "Fragment Shader Log\n";
            std::cout << fs.getLog() <<   "\n";
        }
        renderProgram.attach(vs);
        renderProgram.attach(fs);
        if(!renderProgram.link()){
            std::cout << "Program Log\n";
            std::cout << renderProgram.getLog() <<   "\n";
        }
    }
    auto computeProgram = rtlib::GLProgram();
    {
        computeProgram.create();
        auto cs = rtlib::GLComputeShader(std::string(csSource));
        if(!cs.compile()){
            std::cout << cs.getLog() << "\n";
        }
        computeProgram.attach(cs);
        if(!computeProgram.link()){
            std::cout << computeProgram.getLog() << std::endl;
        }
    }
    auto workGroupSize = computeProgram.getComputeWorkGroupSize();
    auto srcTex = rtlib::GLTexture2D<uchar4>();
    auto dstTex = rtlib::GLTexture2D<uchar4>();
    {
        int imgWidth, imgHeight, imgComp;
        auto imgData = stbi_load("C:\\Users\\shums\\Desktop\\image.png", &imgWidth, &imgHeight, &imgComp, 4);
        srcTex.allocate({ (size_t)imgWidth,(size_t)imgHeight,(void*)imgData });
        srcTex.setParameteri(GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        srcTex.setParameteri(GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        srcTex.setParameteri(GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        srcTex.setParameteri(GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        srcTex.bindImage(0,0, GL_READ_ONLY);
        dstTex.allocate({ (size_t)imgWidth,(size_t)imgHeight,nullptr });
        dstTex.setParameteri(GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        dstTex.setParameteri(GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        dstTex.setParameteri(GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        dstTex.setParameteri(GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        dstTex.bindImage(1,0,GL_WRITE_ONLY);
    }
    auto  widthLoc = computeProgram.getUniformLocation("width");
    auto heightLoc = computeProgram.getUniformLocation("height");
    computeProgram.use();
    glUniform1ui( widthLoc, srcTex.getViews()[0].width );
    glUniform1ui(heightLoc, srcTex.getViews()[0].height);
    glDispatchCompute((srcTex.getViews()[0].width - 1)/workGroupSize[0] + 1, (srcTex.getViews()[0].height - 1)/workGroupSize[1] + 1, 1);
    std::vector<uchar4> dstImage(srcTex.getViews()[0].width* srcTex.getViews()[0].height);
    dstTex.download(0, dstImage.data());
    stbi_write_png("image.png", dstTex.getViews()[0].width, dstTex.getViews()[0].height, 4, reinterpret_cast<void*>(dstImage.data()), 0);
    return 0;
}