#ifndef TEST_RT_CONTEXT_H
#define TEST_RT_CONTEXT_H
#include <RTLib/core/Optix.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <GLFW/glfw3.h>
#include <unordered_map>
#include <string>
#include <memory>
#include <optional>
namespace test {
    class RTContext
    {
    public:
         RTContext(int gl_version_major, int gl_version_minor)noexcept;
         auto NewWindow(int width, int height, const char* title)const->GLFWwindow*;
         auto GetOPX7Handle()const->std::shared_ptr<rtlib::OPXContext>;
         ~RTContext()noexcept;
    private:
        struct Impl;
        std::unique_ptr < Impl> m_Impl;
    };
    
}
#endif
