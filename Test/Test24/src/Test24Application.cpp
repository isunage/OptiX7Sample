#include "..\include\Test24Application.h"
#include <tiny_obj_loader.h>
#include <RTLib/VectorFunction.h>
#include <RTLib/ext/RectRenderer.h>
#include <RTLib/ext/Camera.h>
#include <RTLib/ext/Mesh.h>
#include <TestLib/RTFramebuffer.h>
#include <TestLib/RTGui.h>
#include <Test24Config.h>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <stdexcept>
#include <memory>
namespace test {
	struct Test24Model
	{
		bool                        isSucceeded;
		rtlib::ext::MeshGroupPtr	meshGroup;
		rtlib::ext::VariableMapList materialList;
		static auto Load(const std::string& filename)-> Test24Model
		{
			bool isSucceeded = false;
			rtlib::ext::MeshGroupPtr	meshGroup    = {};
			rtlib::ext::VariableMapList materialList = {};
			return { isSucceeded, std::move(meshGroup) ,std::move(materialList) };
		}
	};
	struct Test24Scene {
		rtlib::ext::Camera												   camera;
		std::unordered_map<std::string, rtlib::ext::MeshPtr>               meshes;
		std::unordered_map<std::string, rtlib::ext::VariableMapList>       materialLists;
		std::unordered_map<std::string, rtlib::ext::CustomImage2D<uchar4>> textures;
		rtlib::ext::VariableMap                                            traceConfig;
	};
	struct Test24Application::Impl
	{
		using RectRendererPtr = std::unique_ptr<rtlib::ext::RectRenderer>;
		using FramebufferPtr  = std::shared_ptr<test::RTFrameBuffer>;
		using GuiPtr		  = std::unique_ptr<test::RTGui>;
		using ScenePtr		  = std::shared_ptr<test::Test24Scene>;
		using ScenePtrMap     = std::unordered_map<std::string, ScenePtr>;
		void InitWindow();
		void FreeWindow();
		void InitRenderer();
		void FreeRenderer();
		void InitFramebuffer();
		void FreeFramebuffer();
		void InitGui();
		void FreeGui();
		void DrawFramebufferGL(const std::string& texName);
		void DrawFramebufferCUGL(const std::string& texName);
		void DrawFrameGui();
		GLFWwindow*     window		 = nullptr;
		int				width		 = TEST_TEST24_DEF_WIDTH ;
		int				height		 = TEST_TEST24_DEF_HEIGHT;
		std::string		title        = TEST_TEST24_DEF_TITLE ;
		RectRendererPtr renderer     = nullptr;
		FramebufferPtr  framebuffer  = nullptr;
		GuiPtr          gui		     = nullptr;
		ScenePtrMap     scenes       = {};
		bool            isResized    = false;
		float2          curCursorPos = make_float2(FLT_MAX, FLT_MAX);
		float2          delCursorPos = make_float2(   0.0f,    0.0f);
		//WindowSize
		static void WindowSizeCallback( GLFWwindow* window, int width, int height) {
			if (!window) return;
			auto impl = reinterpret_cast<Test24Application::Impl*>(glfwGetWindowUserPointer(window));
			if (!impl) return;
			if (width != impl->width || height != impl->height)
			{
				impl->width     = width;
				impl->height    = height;
				impl->isResized = true;
				glViewport(0, 0, static_cast<GLsizei>(width), static_cast<GLsizei>(height));
			}
		}
		//CursorPos
		static void CursorPosCallback(  GLFWwindow* window, double cursorPosX, double cursorPosY)
		{
			if (!window) return;
			auto impl = reinterpret_cast<Test24Application::Impl*>(glfwGetWindowUserPointer(window));
			if (!impl) return;
			
			float2 curCursorPos = impl->curCursorPos;
			impl->curCursorPos  = make_float2(cursorPosX, cursorPosY);
			if (curCursorPos.x != FLT_MAX || curCursorPos.y != FLT_MAX)
			{
				impl->delCursorPos = impl->curCursorPos - curCursorPos;
			}
		}
	};
	class  Test24DefaultWindow :public test::RTGuiWindow
	{
	public:
		Test24DefaultWindow(const std::string& title,const int& width_, const int& height_, const float2& curCursorPos_, const float2& delCursorPos_)
			:RTGuiWindow(title), width(width_), height(height_), curCursorPos(curCursorPos_), delCursorPos(delCursorPos_)
		{}
		// RTGuiWindow ‚ð‰î‚µ‚ÄŒp³‚³‚ê‚Ü‚µ‚½
		virtual void DrawGui() override
		{
			ImGui::Text("WindowSize   : (%5d, %5d)"      , width		 , height	     );
			ImGui::Text("CurCursorPos : (%3.2f, %3.2f)", curCursorPos.x, curCursorPos.y);
			ImGui::Text("DelCursorPos : (%3.2f, %3.2f)", delCursorPos.x, delCursorPos.y);
		}
		virtual ~Test24DefaultWindow() {}
	private:
		const int   & width;
		const int   & height;
		const float2& curCursorPos;
		const float2& delCursorPos;
	};
}

test::Test24Application::Test24Application():m_Impl(new test::Test24Application::Impl())
{
}

void test::Test24Application::Initialize()
{
	this->InitWindow();
	this->InitRenderer();
	this->InitFramebuffer();
	this->InitGui();
}

void test::Test24Application::MainLoop()
{
	if (!m_Impl || !m_Impl->window) return;
	while (!glfwWindowShouldClose(m_Impl->window))
	{
		Render();
		Update();
		glfwSwapBuffers(m_Impl->window);
	}
}

void test::Test24Application::Terminate()
{
	this->FreeWindow();
}

test::Test24Application::~Test24Application()
{
	this->FreeGui();
	this->FreeFramebuffer();
	this->FreeRenderer();
	this->FreeWindow();
}

void test::Test24Application::InitWindow()
{
	m_Impl->InitWindow();
}

void test::Test24Application::FreeWindow()
{
	m_Impl->FreeWindow();
}

void test::Test24Application::InitFramebuffer()
{
	m_Impl->InitFramebuffer();
}

void test::Test24Application::FreeFramebuffer()
{
	m_Impl->FreeFramebuffer();
}

void test::Test24Application::InitRenderer()
{
	m_Impl->InitRenderer();
}

void test::Test24Application::FreeRenderer()
{
	m_Impl->FreeRenderer();
}

void test::Test24Application::InitGui()
{
	m_Impl->InitGui();
}

void test::Test24Application::FreeGui()
{
	m_Impl->FreeGui();
}

void test::Test24Application::Render()
{
	glClear(GL_COLOR_BUFFER_BIT);
	glClearColor(1.0f, 1.0f, 1.0f, 0.0f);
	m_Impl->DrawFramebufferGL("Default");
	m_Impl->DrawFrameGui();
}

void test::Test24Application::Update()
{
	glfwPollEvents();
}

void test::Test24Application::Impl::InitWindow() {
	if (!glfwInit()) {
		throw std::runtime_error("Failed To Initialize GLFW!");
	}
	glfwWindowHint(GLFW_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
	window = glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr);
	glfwMakeContextCurrent(window);
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		throw std::runtime_error("Failed To Initialize GLAD!");
	}
	glfwSetWindowUserPointer(  window, this);
	glfwSetWindowSizeCallback( window, WindowSizeCallback);
	glfwSetCursorPosCallback(  window, CursorPosCallback);
}

void test::Test24Application::Impl::FreeWindow()
{
	glfwDestroyWindow(window);
	window = nullptr;
}

void test::Test24Application::Impl::InitRenderer()
{
	renderer = std::make_unique<rtlib::ext::RectRenderer>();
	renderer->init();
}

void test::Test24Application::Impl::FreeRenderer()
{
	renderer->reset();
	renderer.reset();
}

void test::Test24Application::Impl::InitFramebuffer()
{
	std::vector<uchar4> pixels(width * height, make_uchar4(255, 255, 0, 255));
	framebuffer = std::make_shared<test::RTFrameBuffer>(width, height);
	framebuffer->AddGLTexture("Default");
	framebuffer->GetGLTexture("Default").upload(
		0, pixels.data(), 0, 0, width, height
	);
}

void test::Test24Application::Impl::FreeFramebuffer()
{
	framebuffer->CleanUp();
	framebuffer.reset();
}

void test::Test24Application::Impl::InitGui()
{
	gui = std::make_unique<test::RTGui>(
		window
	);
	gui->AddGuiWindow(std::shared_ptr<test::RTGuiWindow>(
		new test::Test24DefaultWindow(
			"default", width, height, curCursorPos, delCursorPos
		)
	));
	gui->Initialize();
}

void test::Test24Application::Impl::FreeGui()
{
	gui->CleanUp();
	gui.reset();
}

void test::Test24Application::Impl::DrawFramebufferGL(const std::string& texName)
{
	renderer->draw(framebuffer->GetGLTexture(texName).getID());
}

void test::Test24Application::Impl::DrawFramebufferCUGL(const std::string& texName)
{
	renderer->draw(framebuffer->GetCUGLBuffer(texName).getID());
}

void test::Test24Application::Impl::DrawFrameGui()
{
	gui->DrawFrame();
}
