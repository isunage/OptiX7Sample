#include "../include/Test24Gui.h"
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <ImGuiFileDialog.h>
#include <stdexcept>

auto test::Test24Gui::New(GLFWwindow* ownerWindow, std::shared_ptr<RTFrameBuffer> frameBuffer) -> test::RTGuiPtr
{
	return test::RTGuiPtr(new Test24Gui(ownerWindow,frameBuffer));
}

test::Test24Gui::Test24Gui(GLFWwindow* ownerWindow, std::shared_ptr<RTFrameBuffer> frameBuffer)
{
	m_Window      = ownerWindow;
	m_Framebuffer = frameBuffer;
}
void test::Test24Gui::Initialize()
{
	glfwSetKeyCallback( m_Window, ImGui_ImplGlfw_KeyCallback);
	glfwSetCharCallback(m_Window, ImGui_ImplGlfw_CharCallback);
	glfwSetCursorEnterCallback(m_Window, ImGui_ImplGlfw_CursorEnterCallback);
	glfwSetMouseButtonCallback(m_Window, ImGui_ImplGlfw_MouseButtonCallback);
	glfwSetMonitorCallback(ImGui_ImplGlfw_MonitorCallback);
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO();
	(void)io;
	ImGui::StyleColorsDark();
	if (!ImGui_ImplGlfw_InitForOpenGL(m_Window, false))
	{
		throw std::runtime_error("Failed To Init ImGui For GLFW + OpenGL!");
	}
	int major = glfwGetWindowAttrib(m_Window, GLFW_CONTEXT_VERSION_MAJOR);
	int minor = glfwGetWindowAttrib(m_Window, GLFW_CONTEXT_VERSION_MINOR);
	std::string glslVersion = std::string("#version ") + std::to_string(major) + std::to_string(minor) + "0 core";
	if (!ImGui_ImplOpenGL3_Init(glslVersion.c_str()))
	{
		throw std::runtime_error("Failed To Init ImGui For GLFW3");
	}
	SetBool("Mode.SettingAsset", true);
	SetBool("Mode.SettingAsset.NewScene", true);
	SetBool("Mode.SettingAsset.PrvScene", false);
	SetBool("Mode.SettingTrace", false);
	SetUInt32("Count.ObjFiles" , 0);
}

void test::Test24Gui::Attach()
{
	
	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();

	ImGui::PushStyleColor(ImGuiCol_TitleBgActive, ImVec4(0.0f, 0.7f, 0.2f, 1.0f));
	ImGui::PushStyleColor(ImGuiCol_TitleBg      , ImVec4(0.0f, 0.3f, 0.1f, 1.0f));

	ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Once);
	ImGui::SetNextWindowSize(ImVec2(500, 500), ImGuiCond_Once);

	if (GetBool("Mode.SettingAsset"))
	{
		AttachOnAsset();
	}
	if (GetBool("Mode.SettingTrace"))
	{
		AttachOnTrace();
	}

	ImGui::PopStyleColor();
	ImGui::PopStyleColor();
	ImGui::Render();
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void test::Test24Gui::Terminate()
{
}

void test::Test24Gui::AttachOnAsset()
{
	bool mode_NewScene = GetBool("Mode.SettingAsset.NewScene");
	bool mode_PrvScene = GetBool("Mode.SettingAsset.PrvScene");
	ImGui::Begin("AssetSetting");
	if (mode_NewScene)
	{

	}
	if (mode_PrvScene)
	{

	}
	if (!mode_NewScene)
	{
		if (ImGui::Button("New Scene"))
		{
			SetBool("Mode.SettingAsset.NewScene", true);
			SetBool("Mode.SettingAsset.PrvScene", false);
		}
	}
	if (!mode_NewScene && !mode_PrvScene) {
		ImGui::SameLine();
	}
	if (!mode_PrvScene) {
		if (ImGui::Button("Prv Scene"))
		{
			SetBool("Mode.SettingAsset.NewScene", false);
			SetBool("Mode.SettingAsset.PrvScene", true);
		}
	}
	ImGui::End();
}

void test::Test24Gui::AttachOnTrace()
{
	auto fbAspect   = m_Framebuffer->GetAspect();
	auto debugTexID = m_Framebuffer->GetGLTexture("Debug").getID();
	ImGui::Begin("TraceSetting", nullptr, ImGuiWindowFlags_MenuBar);
	ImGui::Image(reinterpret_cast<void*>(debugTexID), { 256 * fbAspect, 256 }, { 1, 1 }, { 0, 0 });
	ImGui::End();
}
