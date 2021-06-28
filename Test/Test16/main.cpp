#include <RTLib/GL.h>
#include <RTLib/CUDA.h>
#include <RTLib/CUDA_GL.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <unordered_map>
#include <sstream>
#include <string>

class ImGUIApplication {
public:
	bool InitGLFW(int gl_version_major, int gl_version_minor) {
		if (glfwInit() == GLFW_FALSE) {
			return false;
		}
		glfwWindowHint(GLFW_VERSION_MAJOR, 3);
		glfwWindowHint(GLFW_VERSION_MINOR, 3);
		glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API);
		glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, gl_version_major);
		glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, gl_version_minor);
		glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
		glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
		std::stringstream ss;
		ss << "#version " << gl_version_major << gl_version_minor << "0 core";
		m_GlslVersion = ss.str();
		return true;
	}
	bool InitWindow(int width, int height, const std::string& title) {
		m_Window = glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr);
		if (!m_Window) {
			return false;
		}
		glfwMakeContextCurrent(m_Window);
		glfwSetMouseButtonCallback(m_Window,ImGui_ImplGlfw_MouseButtonCallback);
		glfwSetKeyCallback(m_Window,ImGui_ImplGlfw_KeyCallback);
		glfwSetCharCallback(m_Window,ImGui_ImplGlfw_CharCallback);
		glfwSetScrollCallback(m_Window,ImGui_ImplGlfw_ScrollCallback);
		return true;
	}
	bool InitGLAD() {
		if (!glfwGetCurrentContext()) {
			return false;
		}
		if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
			return false;
		}
		return true;
	}
	bool InitImGui() {
		IMGUI_CHECKVERSION();
		ImGui::CreateContext();
		ImGuiIO& io = ImGui::GetIO(); (void)io;
		ImGui::StyleColorsDark();
		if (!ImGui_ImplGlfw_InitForOpenGL(m_Window, false)) {
			return false;
		}
		if (!ImGui_ImplOpenGL3_Init(m_GlslVersion.c_str())) {
			return false;
		}
		return true;
	}
	void MainLoop() {
		float x = 0.0f;
		float y = 0.0f;
		while (!glfwWindowShouldClose(m_Window)) {
			glfwPollEvents();
			ImGui_ImplOpenGL3_NewFrame();
			ImGui_ImplGlfw_NewFrame();
			ImGui::NewFrame();
			ImGui::PushStyleColor(ImGuiCol_TitleBgActive, ImVec4(0.0f, 0.7f, 0.2f, 1.0f));
			ImGui::PushStyleColor(ImGuiCol_TitleBg, ImVec4(0.0f, 0.3f, 0.1f, 1.0f));
			ImGui::SetNextWindowPos(ImVec2(20, 20));
			ImGui::SetNextWindowSize(ImVec2(280, 300));

			ImGui::Begin("config 1", nullptr, ImGuiWindowFlags_MenuBar);

			if (ImGui::BeginMenuBar()) {
				if (ImGui::BeginMenu("File"))
				{
					if (ImGui::MenuItem("Save")) {

					}
					if (ImGui::MenuItem("Load")) {

					}

					ImGui::EndMenu();
				}
				ImGui::EndMenuBar();
			}


			static std::vector<float> items(10);

			if (ImGui::Button("add")) {
				items.push_back(0.0f);
			}
			if (ImGui::Button("remove")) {
				if (items.empty() == false) {
					items.pop_back();
				}
			}

			ImGui::BeginChild(ImGui::GetID((void*)0), ImVec2(250, 100), ImGuiWindowFlags_NoTitleBar);
			for (int i = 0; i < items.size() ; ++i) {
				char name[16];
				sprintf(name, "item %d", i);
				ImGui::SliderFloat(name, &items[i], 0.0f, 10.0f);
			}
			ImGui::EndChild();

			ImGui::End();

			ImGui::PopStyleColor();
			ImGui::PopStyleColor();
			// Rendering
			ImGui::Render();
			int display_w, display_h;
			glfwGetFramebufferSize(m_Window, &display_w, &display_h);
			glClearColor(0.8f, 0.8f, 0.8f, 0.0f);
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
			glViewport(0, 0, display_w, display_h);
			ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
			glfwSwapBuffers(m_Window);
		}
	}
	void CleanUpImGui() {
		ImGui_ImplOpenGL3_Shutdown();
		ImGui_ImplGlfw_Shutdown();
		ImGui::DestroyContext();
	}
	void CleanUpWindow() {
		glfwDestroyWindow(m_Window);
		m_Window = nullptr;
	}
	void CleanUpGLFW() {
		glfwTerminate();
	}
private:
	GLFWwindow*        m_Window         = nullptr;
	int				   m_Width          = 0;
	int                m_Height         = 0;
	std::string        m_Title          = {};
	std::string        m_GlslVersion    = {};
};
int main() {
	ImGUIApplication app = {};
	app.InitGLFW(4, 4);
	app.InitWindow(640, 480, "title");
	app.InitGLAD();
	app.InitImGui();
	app.MainLoop();
	app.CleanUpImGui();
	app.CleanUpWindow();
	app.CleanUpGLFW();
}