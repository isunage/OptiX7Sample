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
	if (HasGuiMainMenuBar()) {
		m_GuiMainMenuBar->DrawFrame();
	}
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

bool test::RTGui::HasGuiMainMenuBar() const noexcept
{
	return m_GuiMainMenuBar!=nullptr;
}

auto test::RTGui::AddGuiMainMenuBar() -> std::shared_ptr<RTGuiMainMenuBar>
{
	m_GuiMainMenuBar = std::make_shared<RTGuiMainMenuBar>();
	return m_GuiMainMenuBar;
}

auto test::RTGui::GetGuiMainMenuBar() const -> std::shared_ptr<RTGuiMainMenuBar>
{
	return m_GuiMainMenuBar;
}

void test::RTGui::SetGuiWindow(std::shared_ptr<RTGuiWindow> guiWindow)
{
	m_GuiWindows.push_back(guiWindow);
}

void test::RTGui::PopGuiWindow()
{
	auto guiWindow = m_GuiWindows.back();
	m_GuiWindows.pop_back();
}

auto test::RTGui::GetGuiWindow(size_t idx) const -> std::shared_ptr<RTGuiWindow>
{
	return m_GuiWindows.at(idx);
}


test::RTGui::RTGui(GLFWwindow* window) noexcept {
	m_Window = window;
}

void test::RTGui::SetStyleColor(ImGuiCol idx, const ImU32 col) {
	m_StyleColors[idx] = col;
}

void test::RTGui::SetStyleColor(ImGuiCol idx, const ImVec4& col)
{
	m_StyleColors[idx] = col;
}

test::RTGuiWindow::RTGuiWindow(std::string title) noexcept :m_Title{ title } {}
test::RTGuiWindow::RTGuiWindow(std::string title, ImGuiWindowFlags flags) :RTGuiWindow(title) {
	m_WindowArgs = { &m_IsActive,flags };
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

void test::RTGuiWindow::SetUserPointer(void* data) noexcept
{
	m_UserPointer = data;
}

auto test::RTGuiWindow::GetUserPointer() const noexcept -> void*
{
	return m_UserPointer;
}

void test::RTGuiWindow::SetDrawCallback(DrawCallback callback) noexcept
{
	if (!callback) { return; }
	m_DrawCallback = callback;
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
		if (HasGuiMenuBar()) {
			m_GuiMenuBar->DrawFrame();
		}
		//Callback->Derive->Subobject
		m_DrawCallback(this);
		DrawGui();
		for (auto& subobject : m_SubObjects) {
			subobject->DrawFrame();
		}
		ImGui::End();
	}
}

bool test::RTGuiWindow::HasGuiMenuBar() const noexcept
{
	return m_GuiMenuBar!=nullptr;
}

auto test::RTGuiWindow::AddGuiMenuBar() -> std::shared_ptr<RTGuiMenuBar>
{
	if (!m_WindowArgs) {
		m_WindowArgs         = WindowArgs{};
		m_WindowArgs->p_open = &m_IsActive;
	}
	m_WindowArgs->flags |= ImGuiWindowFlags_MenuBar;
	m_GuiMenuBar = std::make_shared<RTGuiMenuBar>();
	return m_GuiMenuBar;
}

auto test::RTGuiWindow::GetGuiMenuBar() const -> std::shared_ptr<RTGuiMenuBar>
{
	return m_GuiMenuBar;
}

void test::RTGuiWindow::AddSubObject(const std::shared_ptr<RTGuiSubObject>& subobject)
{
	m_SubObjects.push_back(subobject);
}

void test::RTGuiWindow::DrawGui() {}

test::RTGuiWindow::~RTGuiWindow() noexcept {}

test::RTGuiMainMenuBar::RTGuiMainMenuBar() noexcept
{
}

auto test::RTGuiMainMenuBar::AddGuiMenu(const std::string& name) -> std::shared_ptr<RTGuiMenu>
{
	auto ptr = std::make_shared<test::RTGuiMenu>(name);
	m_GuiMenus.push_back(ptr);
	return ptr;
}


void test::RTGuiMainMenuBar::PopGuiMenu() 
{
	m_GuiMenus.pop_back();
}

auto test::RTGuiMainMenuBar::GetGuiMenu(size_t idx) const -> std::shared_ptr<RTGuiMenu>
{
	return m_GuiMenus.at(idx);
}

void test::RTGuiMainMenuBar::DrawFrame()
{
	if (ImGui::BeginMainMenuBar()) {
		for (auto& menu : m_GuiMenus)
		{
			if (menu) {
				menu->DrawFrame();
			}
		}
		ImGui::EndMainMenuBar();
	}
}

test::RTGuiMainMenuBar::~RTGuiMainMenuBar() noexcept
{
}

test::RTGuiMenu::RTGuiMenu(const std::string& name, bool isEnable) noexcept
{
	m_Name   = name;
	m_Enable = isEnable;
}

void test::RTGuiMenu::SetEnable(bool isEnable) noexcept
{
	m_Enable = isEnable;
}

void test::RTGuiMenu::SetGuiMenuItem(std::shared_ptr<RTGuiMenuItem> guiMenuItem)
{
	m_GuiChilds.push_back(guiMenuItem);
}

auto test::RTGuiMenu::AddGuiMenu(const std::string& name)->std::shared_ptr<RTGuiMenu>
{
	auto ptr = std::make_shared<test::RTGuiMenu>(name);
	m_GuiChilds.push_back(ptr);
	return ptr;
}

void test::RTGuiMenu::PopGuiChild()
{
	m_GuiChilds.pop_back();
}


auto test::RTGuiMenu::GetGuiGuiMenuItem(size_t idx) const -> std::shared_ptr<RTGuiMenuItem>
{
	if (std::get_if<std::shared_ptr<RTGuiMenuItem>>(&m_GuiChilds.at(idx))) {
		return std::get<std::shared_ptr<RTGuiMenuItem>>(m_GuiChilds.at(idx));
	}
	else {
		return nullptr;
	}
}

auto test::RTGuiMenu::GetGuiMenu(size_t idx) const -> std::shared_ptr<RTGuiMenu>
{
	if (std::get_if<std::shared_ptr<RTGuiMenu>>(&m_GuiChilds.at(idx))) {
		return std::get<std::shared_ptr<RTGuiMenu>>(m_GuiChilds.at(idx));
	}
	else {
		return nullptr;
	}
}

void test::RTGuiMenu::DrawFrame()
{

	if (ImGui::BeginMenu(m_Name.c_str(), m_Enable)) {
		for (auto& menuItem: m_GuiChilds) {

			if (std::get_if<0>(&menuItem)) {

				std::get<0>(menuItem)->DrawFrame();
			}
			else {
				std::get<1>(menuItem)->DrawFrame();
			}
		}
		ImGui::EndMenu();
	}
}

test::RTGuiMenu::~RTGuiMenu() noexcept {}

test::RTGuiMenuItem::RTGuiMenuItem(const std::string& name, bool isEnable) noexcept
{
	m_Name   = name;
	m_Enable = isEnable;
}

void test::RTGuiMenuItem::SetEnable(bool isEnable) noexcept
{
	m_Enable = isEnable;
}

void test::RTGuiMenuItem::SetUserPointer(void* data) noexcept
{
	m_UserPointer = data;
}

auto test::RTGuiMenuItem::GetUserPointer() const noexcept -> void*
{
	return m_UserPointer;
}

void test::RTGuiMenuItem::SetClickCallback(ClickCallback callback) noexcept
{
	m_ClickCallback = callback;
}

void test::RTGuiMenuItem::DrawFrame()
{
	if (ImGui::MenuItem(m_Name.c_str(), 0, false, m_Enable)) {
		m_ClickCallback(this);
		OnClick();
		for (auto& subobject : m_SubObjects) {
			if (subobject) {
				subobject->DrawFrame();
			}
		}
	}
}

void test::RTGuiMenuItem::AddSubObject(const std::shared_ptr<RTGuiSubObject>& subobject)
{
	m_SubObjects.push_back(subobject);
}

void test::RTGuiMenuItem::OnClick() {}

test::RTGuiMenuItem::~RTGuiMenuItem() noexcept {}

test::RTGuiMenuBar::RTGuiMenuBar() noexcept
{
}

auto test::RTGuiMenuBar::AddGuiMenu(const std::string& name) -> std::shared_ptr<RTGuiMenu>
{
	auto ptr = std::make_shared<test::RTGuiMenu>(name);
	m_GuiMenus.push_back(ptr);
	return ptr;
}

void test::RTGuiMenuBar::PopGuiMenu()
{
	m_GuiMenus.pop_back();
}

auto test::RTGuiMenuBar::GetGuiMenu(size_t idx) const -> std::shared_ptr<RTGuiMenu>
{
	return m_GuiMenus.at(idx);
}

void test::RTGuiMenuBar::DrawFrame()
{
	if (ImGui::BeginMenuBar()) {
		for (auto& menu : m_GuiMenus)
		{
			if (menu) {
				menu->DrawFrame();
			}
		}
		ImGui::EndMenuBar();
	}
}

test::RTGuiMenuBar::~RTGuiMenuBar() noexcept
{
}
