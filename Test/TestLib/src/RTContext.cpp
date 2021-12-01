#include "../include/TestLib/RTContext.h"

struct test::RTContext::Impl{
	GLFWwindow*                        m_Client  = nullptr;
	std::shared_ptr<rtlib::OPXContext> m_Opx7Ctx = nullptr;
};

test::RTContext::RTContext(int gl_version_major, int gl_version_minor) noexcept
{
	bool isInitializeGLFW = false;
	GLFWwindow* client    = nullptr;
	std::shared_ptr<rtlib::OPXContext> opx7Ctx;
	try {
		if (glfwInit() == GLFW_FALSE) {
			throw std::runtime_error("Failed To Initialize GLFW!");
		}
		isInitializeGLFW   = true;

		glfwWindowHint(GLFW_VERSION_MAJOR, 3);
		glfwWindowHint(GLFW_VERSION_MINOR, 3);
		glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, gl_version_major);
		glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, gl_version_minor);
		glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API);
		glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
		glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
		glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
		client = glfwCreateWindow(1, 1, "Client", nullptr, nullptr);
		if (!client) {
			throw std::runtime_error("Failed To Create Window!");
		}

		glfwMakeContextCurrent(client);

		if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
			throw std::runtime_error("Failed To Init GLAD!");
		}

		RTLIB_CUDA_CHECK(cudaFree(0));

		RTLIB_OPTIX_CHECK(optixInit());

		opx7Ctx = std::make_shared<rtlib::OPXContext>(rtlib::OPXContext::Desc{
			0,0,OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_OFF,4
		});

		if (!opx7Ctx) {
			throw std::runtime_error("Failed To Init OPX7!");
		}
	}
	catch (std::runtime_error& err) {

		std::cout << err.what() << std::endl;

		if (opx7Ctx) {
			opx7Ctx.reset();
		}

		if (client) {
			glfwDestroyWindow(client);
			client = nullptr;
		}

		if (isInitializeGLFW) {
			glfwTerminate();
			isInitializeGLFW = false;
		}
	}
	m_Impl = std::make_unique<test::RTContext::Impl>();
	m_Impl->m_Client  = client;
	m_Impl->m_Opx7Ctx = opx7Ctx;
}

auto test::RTContext::NewWindow(int width, int height, const char* title) const -> GLFWwindow*
{
	glfwWindowHint(GLFW_VISIBLE, GLFW_TRUE);
	auto window = glfwCreateWindow(width,height,title,nullptr,m_Impl->m_Client);
	glfwMakeContextCurrent(window);
	return window;
}

auto test::RTContext::GetOPX7Handle() const -> std::shared_ptr<rtlib::OPXContext>
{
	return m_Impl->m_Opx7Ctx;
}

test::RTContext::~RTContext() noexcept
{
	try {
		if (m_Impl->m_Opx7Ctx) {
			m_Impl->m_Opx7Ctx.reset();
		}
	}
	catch (std::runtime_error&) {}
	try {

		RTLIB_CUDA_CHECK(cudaFree(0));
	}
	catch (std::runtime_error&) {}
	try {
		if (m_Impl->m_Client) {
			glfwDestroyWindow(m_Impl->m_Client);
			m_Impl->m_Client = nullptr;
		}
	}
	catch (std::runtime_error&) {}
	try {

		glfwTerminate();
	}
	catch (std::runtime_error&) {}
	m_Impl.reset();
}
