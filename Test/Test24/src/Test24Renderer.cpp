#include "../include/Test24Renderer.h"
#include <GLFW/glfw3.h>
auto test::Test24Renderer::New(GLFWwindow* window, test::RTFrameBufferPtr framebuffer, test::RTGuiPtr gui) -> RTRendererPtr
{
	return RTRendererPtr(new test::Test24Renderer(window, framebuffer, gui));
}

test::Test24Renderer::Test24Renderer(GLFWwindow* window, test::RTFrameBufferPtr framebuffer, test::RTGuiPtr gui)
{
	m_Window      = window;
	m_Framebuffer = framebuffer;
	m_Gui         = gui;
	glfwGetWindowSize(m_Window, &m_FbWidth, &m_FbHeight);
}

void test::Test24Renderer::Initialize()
{
	m_RectRenderer.init();
	if (!m_Framebuffer->HasGLTexture("Render"))
	{
		m_Framebuffer->AddGLTexture( "Render");
	}
	if (!m_Framebuffer->HasGLTexture("Debug"))
	{
		m_Framebuffer->AddGLTexture( "Debug");
	}
	if (!m_Framebuffer->HasCUGLBuffer("Default"))
	{
		m_Framebuffer->AddCUGLBuffer("Default");
	}
	//Gui
	m_Gui->SetString("RenderFrame", "Default");
	m_Gui->SetString( "DebugFrame", "Default");
	m_Gui->SetBool("Update.RenderFrame", true);
	m_Gui->HasBool("Update.DebugFrame");
	m_Gui->SetBool("Update.DebugFrame" , true);
}

void test::Test24Renderer::Render()
{
	int fbWidth, fbHeight;
	glfwGetWindowSize(m_Window, &fbWidth, &fbHeight);
	if (m_Gui->GetBool("Update.RenderFrame"))
	{
		auto  rtFrameName = m_Gui->GetString("RenderFrame");
		auto& rtFrameData = m_Framebuffer->GetCUGLBuffer("Diffuse");
		m_Framebuffer->GetGLTexture("Render").upload(0, rtFrameData.getHandle(), 0, 0, fbWidth, fbHeight);
		m_Gui->SetBool("Update.RenderFrame", false);
	}
	if (m_Gui->GetBool("Update.DebugFrame"))
	{
		auto  dgFrameName = m_Gui->GetString("DebugFrame");
		auto& dgFrameData = m_Framebuffer->GetCUGLBuffer(dgFrameName);
		m_Framebuffer->GetGLTexture("Debug").upload(0, dgFrameData.getHandle(), 0, 0, fbWidth, fbHeight);
		m_Gui->SetBool("Update.DebugFrame", false);
	}
	m_RectRenderer.draw(m_Framebuffer->GetGLTexture("Render").getID());
	m_Gui->Attach();
}

void test::Test24Renderer::Terminate()
{
	m_RectRenderer.reset();
	m_Window = nullptr;
	m_Framebuffer = nullptr;
	m_Gui = nullptr;
	m_FbWidth = m_FbHeight = 0;
}

bool test::Test24Renderer::Resize(int width, int height)
{
	if (m_FbWidth != width || m_FbHeight != height)
	{
		m_FbWidth  = width;
		m_FbHeight = height;
		return true;
	}
	return false;
}
