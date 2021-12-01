#include <TestLib/RTApplication.h>
#include <TestLib/RTFramebuffer.h>
#include <TestLib/RTContext.h>
#include <TestLib/RTGui.h>
#include <RTLib/ext/RectRenderer.h>
#include <TestLibConfig.h>
#include <GLFW/glfw3.h>
#include <glad/glad.h>
#include <memory>
#include <deque>
#include <filesystem>
#include <ImGuiFileDialog.h>
class TestLibTestApplication : public test::RTApplication
{
private:
	class MainGuiWindow : public test::RTGuiWindow
	{
	public:
		MainGuiWindow(std::string title, std::shared_ptr<test::RTFrameBuffer> framebuffer, std::vector<float>& fpsValues, std::vector<std::filesystem::path>& objFilePathes)
			:test::RTGuiWindow(title, ImGuiWindowFlags_MenuBar), 
			m_FpsValues    { fpsValues     },
			m_Framebuffer  { framebuffer   },
			m_ObjFilePathes{ objFilePathes }{
			//MenuBar
			auto menuBar = this->AddGuiMenuBar();
			//FileMenu
			auto fileMenu = menuBar->AddGuiMenu("File");
			//Open
			auto openMenuItem = std::make_shared<test::RTGuiMenuItem>("Open..");
			openMenuItem->SetUserPointer(this);
			openMenuItem->SetClickCallback([](test::RTGuiMenuItem*)->void {
					ImGuiFileDialog::Instance()->OpenDialog("ChooseFileDlgKey", "Choose File", ".obj", TEST_TESTLIB_DATA_PATH"\\");
			});
			fileMenu->SetGuiMenuItem(openMenuItem);
			//Save
			auto saveMenuItem = std::make_shared<test::RTGuiMenuItem>("Save");
			saveMenuItem->SetUserPointer(this);
			fileMenu->SetGuiMenuItem(saveMenuItem);
			//Close
			auto closeMenuItem = std::make_shared<test::RTGuiMenuItem>("Close");
			closeMenuItem->SetUserPointer(this);
			closeMenuItem->SetClickCallback([](test::RTGuiMenuItem* item) ->void {
				//TODO
				auto ptr = reinterpret_cast<MainGuiWindow*>(item->GetUserPointer());
				ptr->SetActive(false);
			});
			fileMenu->SetGuiMenuItem(closeMenuItem);
			this->SetDrawCallback([](RTGuiWindow* window) {
				auto* this_ptr = static_cast<MainGuiWindow*>(window);
				if (ImGuiFileDialog::Instance()->Display("ChooseFileDlgKey"))
				{
					// action if OK
					if (ImGuiFileDialog::Instance()->IsOk())
					{
						std::string filePathName = ImGuiFileDialog::Instance()->GetFilePathName();
						std::string filePath     = ImGuiFileDialog::Instance()->GetCurrentPath();
						// action
						this_ptr->m_ObjFilePathes.push_back(std::filesystem::relative(filePathName));
					}
					// close
					ImGuiFileDialog::Instance()->Close();
				}
				if (!this_ptr->m_ObjFilePathes.empty()) {
					auto objFilePathStrs = std::vector<std::string>();
					auto objFilePathCstrs = std::vector<const char*>();
					objFilePathStrs.reserve(this_ptr->m_ObjFilePathes.size());
					objFilePathCstrs.reserve(this_ptr->m_ObjFilePathes.size());
					for (auto& objFilePath : this_ptr->m_ObjFilePathes)
					{
						auto c_str = objFilePath.c_str();
						objFilePathStrs.push_back(objFilePath.string());
						objFilePathCstrs.push_back(objFilePathStrs.back().c_str());
					}
					static int listbox_item_current = 0;
					ImGui::ListBox("listbox\n(single select)", &listbox_item_current, objFilePathCstrs.data(), objFilePathCstrs.size(), 4);
				}
				// Edit a color (stored as ~4 floats)
				ImGui::ColorEdit4("Color", this_ptr->m_Color);
				// Plot some values
				ImGui::PlotLines("Fps(line)", this_ptr->m_FpsValues.data(), this_ptr->m_FpsValues.size());
				//Text
				ImGui::Text("Fps: %f", this_ptr->m_FpsValues.back());
				// Display contents in a scrolling region
				ImGui::TextColored(ImVec4(1, 1, 0, 1), "Important Stuff");
			});
		}
		virtual ~MainGuiWindow()noexcept {}
	private:
		std::vector<float>&				       m_FpsValues;
		std::vector<std::filesystem::path>&    m_ObjFilePathes;
		float                                  m_Color[4]    = { 0.0f,0.0f,0.0f,0.0f };
		std::shared_ptr<test::RTFrameBuffer>   m_Framebuffer = nullptr;
		bool                                   m_ShowImage   = false;
	};
public:
	TestLibTestApplication(int width, int height, const std::string& title)
	{
		m_Width        = width;
		m_Height       = height;
		m_Title        = title;
		m_SampleOfAll  = 0;
		m_CurFrameTime = 0.0f;
		m_DelFrameTime = 0.0f;
		m_Window       = nullptr;
		m_Gui          = nullptr;
		m_Renderer     = nullptr;
		m_Framebuffer  = nullptr;
		m_FpsValues    = std::vector<float>(5, 0.0f);
	}
	// RTApplication ‚ð‰î‚µ‚ÄŒp³‚³‚ê‚Ü‚µ‚½
	virtual void Initialize() override
	{
		this->InitContext();
		this->InitWindow();
		this->InitRenderer();
		this->InitFramebuffer();
		this->InitGui();
		glfwSetTime(0.0f);
	}
	virtual void MainLoop() override
	{
		while (!glfwWindowShouldClose(m_Window))
		{
			glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
			glClear(GL_COLOR_BUFFER_BIT);
			m_Renderer->draw(m_Framebuffer->GetGLTexture("Default").getID());
			m_Gui->DrawFrame();
			Update();
			glfwPollEvents();
			glfwSwapBuffers(m_Window);
		}
	}
	virtual void Terminate() override
	{
		this->FreeGui();
		this->FreeFramebuffer();
		this->FreeRenderer();
		this->FreeWindow();
		this->FreeContext();
	}
	virtual ~TestLibTestApplication() {}
private:
	void InitContext() {
		m_Context = std::make_shared<test::RTContext>(4, 5);
	}
	void FreeContext() {
		m_Context.reset();
	}
	void InitWindow()
	{
		m_Window = m_Context->NewWindow(m_Width, m_Height, m_Title.c_str());
	}
	void FreeWindow() {
		glfwDestroyWindow(m_Window);
		m_Window = nullptr;
	}
	void InitFramebuffer() {
		std::vector<uchar4> pixels(m_Width * m_Height, make_uchar4(255, 255, 0, 255));
		m_Framebuffer = std::make_shared<test::RTFrameBuffer>(m_Width, m_Height);
		m_Framebuffer->AddGLTexture("Default");
		m_Framebuffer->GetGLTexture("Default").upload(
			0, pixels.data(), 0, 0, m_Width, m_Height
		);
	}
	void FreeFramebuffer() {
		m_Framebuffer->CleanUp();
		m_Framebuffer.reset();
	}
	void InitRenderer() {
		m_Renderer = std::make_unique<rtlib::ext::RectRenderer>();
		m_Renderer->init();
	}
	void FreeRenderer() {
		m_Renderer->reset();
		m_Renderer.reset();
	}
	void InitGui() {
		m_Gui = std::make_unique<test::RTGui>(m_Window);
		m_Gui->Initialize();
		m_Gui->SetStyleColor(ImGuiCol_TitleBgActive, ImVec4(0.0f, 0.7f, 0.2f, 1.0f));
		m_Gui->SetStyleColor(ImGuiCol_TitleBg, ImVec4(0.0f, 0.3f, 0.1f, 1.0f));

		auto guiWindow1 = std::shared_ptr<test::RTGuiWindow>(new MainGuiWindow("Frame1",m_Framebuffer, m_FpsValues,m_ObjFilePathes));
		guiWindow1->SetNextPos( ImVec2(  0,   0), ImGuiCond_Once);
		guiWindow1->SetNextSize(ImVec2(500, 500), ImGuiCond_Once);

		auto guiWindow2 = std::make_shared<test::RTGuiWindow>("Frame2");
		guiWindow2->SetNextPos( ImVec2(500, 500), ImGuiCond_Once);
		guiWindow2->SetNextSize(ImVec2(500, 500), ImGuiCond_Once);

		m_Gui->SetGuiWindow(guiWindow1);
		m_Gui->SetGuiWindow(guiWindow2);
	}
	void FreeGui() {
		m_Gui->CleanUp();
		m_Gui.reset();
	}
	void Update() {
		float prvFrameTime = m_CurFrameTime;
		m_CurFrameTime     = glfwGetTime();
		m_DelFrameTime     = m_CurFrameTime - prvFrameTime;
		if(m_SampleOfAll%10==1){
			std::deque<float> fpsValues(m_FpsValues.begin(), m_FpsValues.end());
			fpsValues.push_back(1.0f / m_DelFrameTime);
			fpsValues.pop_front();
			std::copy(std::begin(fpsValues), std::end(fpsValues), std::begin(m_FpsValues));
		}
		m_SampleOfAll++;
	}
private:
	int                                       m_Width;
	int                                       m_Height;
	std::string	                              m_Title;
	std::shared_ptr<test::RTContext>          m_Context;
	GLFWwindow*							      m_Window;
	unsigned long long                        m_SampleOfAll;
	float                                     m_CurFrameTime;
	float                                     m_DelFrameTime;
	std::vector<std::filesystem::path>        m_ObjFilePathes;
	std::vector<float>                        m_FpsValues;
	std::shared_ptr<test::RTFrameBuffer>      m_Framebuffer;
	std::unique_ptr<rtlib::ext::RectRenderer> m_Renderer;
	std::unique_ptr<test::RTGui>              m_Gui;
};
int main(int argc, const char** argv)
{
	auto app = std::shared_ptr<test::RTApplication>(new TestLibTestApplication(1024, 1024, "TestLibTest"));
	return app->Run(argc,argv);
}