#include "../include/TestLib/RTGui.h"
void test::RTGui::Initialize() {
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
	glfwSetMouseButtonCallback(m_Window, ImGui_ImplGlfw_MouseButtonCallback);
	glfwSetKeyCallback(m_Window, ImGui_ImplGlfw_KeyCallback);
	glfwSetCharCallback(m_Window, ImGui_ImplGlfw_CharCallback);
	glfwSetScrollCallback(m_Window, ImGui_ImplGlfw_ScrollCallback);
}

void test::RTGui::DrawFrame()
{
	BeginFrame();
	for (auto& guiWindow : m_GuiWindows) {
		guiWindow->DrawFrame();
	}
	EndFrame();
}

void test::RTGui::BeginFrame() {
	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();
	for (auto& [idx, col] : m_StyleColors)
	{
		auto i_col = std::get_if<ImU32>(&col);
		if  (i_col) {
			ImGui::PushStyleColor(idx, *i_col);
		}
		auto c_col = std::get_if<ImVec4>(&col);
		if (c_col) {
			ImGui::PushStyleColor(idx, *c_col);
		}
	}
}

void test::RTGui::EndFrame() {
	ImGui::PopStyleColor(m_StyleColors.size());
	ImGui::Render();
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void test::RTGui::CleanUp() {

}

void test::RTGui::AddGuiWindow(std::shared_ptr<RTGuiWindow> guiWindow)
{
	m_GuiWindows.push_back(guiWindow);
}

auto test::RTGui::PopGuiWindow() -> std::shared_ptr<RTGuiWindow>
{
	auto guiWindow = m_GuiWindows.back();
	m_GuiWindows.pop_back();
	return guiWindow;
}

auto test::RTGui::GetGuiWindow(size_t idx) const -> std::shared_ptr<RTGuiWindow>
{
	return m_GuiWindows.at(idx);
}

auto test::RTGui::GetGuiWindows() const -> const std::vector<std::shared_ptr<RTGuiWindow>>&
{
	// TODO: return ステートメントをここに挿入します
	return m_GuiWindows;
}

void test::RTGui::SetStyleColor(ImGuiCol idx, const ImU32 col) {
	m_StyleColors[idx] = col;
}

void test::RTGui::SetStyleColor(ImGuiCol idx, const ImVec4& col)
{
	m_StyleColors[idx] = col;
}

void test::RTGuiWindow::SetTitle(const std::string& title)
{
	m_Title = title;
}

void test::RTGuiWindow::SetNextPos(const ImVec2& pos, ImGuiCond cond, const ImVec2& pivot)
{
	m_NextPosArgs = { pos,cond,pivot };
}

void test::RTGuiWindow::SetNextSize(const ImVec2& size, ImGuiCond cond)
{
	m_NextSizeArgs = { size, cond };
}

void test::RTGuiWindow::DrawFrame()
{
	if (m_NextPosArgs) {
		ImGui::SetNextWindowPos(m_NextPosArgs.value().pos, m_NextPosArgs.value().cond, m_NextPosArgs.value().pivot);
	}
	if (m_NextSizeArgs) {
		ImGui::SetNextWindowSize(m_NextSizeArgs.value().size, m_NextPosArgs.value().cond);
	}
	if (m_IsActive) {
		if (m_WindowArgs.has_value()) {
			ImGui::Begin(m_Title.c_str(), m_WindowArgs.value().p_open, m_WindowArgs.value().flags);
		}
		else {
			ImGui::Begin(m_Title.c_str());
		}
		DrawGui();
		ImGui::End();
	}
}
