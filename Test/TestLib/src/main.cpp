#include <TestLib/RTApplication.h>
#include <TestLib/RTFramebuffer.h>
#include <TestLib/RTGui.h>
#include <RTLib/ext/RectRenderer.h>
#include <RTLib/ext/Math/Matrix.h>
#include <TestLibConfig.h>
#include <GLFW/glfw3.h>
#include <glad/glad.h>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/string_cast.hpp>
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
			m_ObjFilePathes{ objFilePathes }{}
		virtual void DrawGui()override {
			if (ImGui::BeginMenuBar()) {
				if (ImGui::BeginMenu("File")) {
					if (ImGui::MenuItem("Open..")) {
						/* Do stuff */
						ImGuiFileDialog::Instance()->OpenDialog("ChooseFileDlgKey", "Choose File", ".obj", TEST_TESTLIB_DATA_PATH"\\");
					}
					if (ImGui::MenuItem("Save" )) { /* Do stuff */ }
					if (ImGui::MenuItem("Close")) { /* Do stuff */ }
					ImGui::EndMenu();
				}
				// open Dialog Simple
				// display
				if (ImGuiFileDialog::Instance()->Display("ChooseFileDlgKey"))
				{
					// action if OK
					if (ImGuiFileDialog::Instance()->IsOk())
					{
						std::string filePathName = ImGuiFileDialog::Instance()->GetFilePathName();
						std::string filePath     = ImGuiFileDialog::Instance()->GetCurrentPath();
						// action
						m_ObjFilePathes.push_back(std::filesystem::relative(filePathName));
					}
					// close
					ImGuiFileDialog::Instance()->Close();
				}
				ImGui::EndMenuBar();

				if(!m_ObjFilePathes.empty()){
					auto objFilePathStrs  = std::vector<std::string>();
					auto objFilePathCstrs = std::vector<const char*>();
					objFilePathStrs.reserve(m_ObjFilePathes.size());
					objFilePathCstrs.reserve(m_ObjFilePathes.size());
					for (auto& objFilePath : m_ObjFilePathes)
					{
						auto c_str = objFilePath.c_str();
						 objFilePathStrs.push_back(objFilePath.string());
						objFilePathCstrs.push_back(objFilePathStrs.back().c_str());
					}
					static int listbox_item_current = 0;
					ImGui::ListBox("listbox\n(single select)", &listbox_item_current, objFilePathCstrs.data(), objFilePathCstrs.size(), 4);
				}
				// Edit a color (stored as ~4 floats)
				ImGui::ColorEdit4("Color", m_Color);
				// Plot some values
				ImGui::PlotLines("Fps", m_FpsValues.data(), m_FpsValues.size());
				// Display contents in a scrolling region
				ImGui::TextColored(ImVec4(1, 1, 0, 1), "Important Stuff");
				ImGui::BeginChild("Scrolling");
				for (auto& path : m_ObjFilePathes) {
					auto pathString = path.string();
					ImGui::Text("%s", pathString.data());
				}
				ImGui::EndChild();
			}
		}
		virtual ~MainGuiWindow()noexcept {}
	private:
		std::vector<float>&				       m_FpsValues;
		std::vector<std::filesystem::path>&    m_ObjFilePathes;
		float                                  m_Color[4]    = { 0.0f,0.0f,0.0f,0.0f };
		std::shared_ptr < test::RTFrameBuffer> m_Framebuffer = nullptr;
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
	}
	virtual ~TestLibTestApplication() {}
private:
	void InitWindow()
	{
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
		m_Window = glfwCreateWindow(m_Width, m_Height, m_Title.c_str(), nullptr, nullptr);
		glfwMakeContextCurrent(m_Window);
		if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
		{
			throw std::runtime_error("Failed To Initialize GLAD!");
		}
	}
	void FreeWindow() {
		glfwDestroyWindow(m_Window);
		glfwTerminate();
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

		
		m_Gui->AddGuiWindow(guiWindow1);
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
	auto matrix = rtlib::Matrix3x3(
		make_float3(2.0f, 3.0f, 6.0f),
		make_float3(2.0f, 3.0f, 4.0f),
		make_float3(4.0f, 3.0f, 1.0f)
	);
	 matrix.Show();
	 matrix.Transpose().Show();
	 matrix.Inverse().Show();
	 matrix.InverseTranspose().Show();
	(matrix.Inverse() * matrix).Show();
	(matrix.InverseTranspose() * matrix.Transpose()).Show();
	auto matrix2 = rtlib::Matrix4x4(
		make_float4(2.0f, 3.0f, 6.0f,7.0f),
		make_float4(2.0f, 5.0f, 7.0f,9.0f),
		make_float4(4.0f, 3.0f, 1.0f,5.0f),
		make_float4(7.0f, 8.0f, 9.0f, 5.0f)
	);
	matrix2.Show();
	matrix2.Transpose().Show();
	std::cout << matrix2.Det() << std::endl;
	matrix2.Inverse().Show();
	matrix2.InverseTranspose().Show();
	(matrix2.Inverse() * matrix2).Show();
	(matrix2.InverseTranspose() * matrix2.Transpose()).Show();
	auto m = rtlib::Matrix4x4::Rotate(make_float3(3.0f, 2.0f, 1.0f), (float)RTLIB_M_PI / 6.0f);
	(m * m * m * m * m *m * m * m * m * m * m * m).Show();
	auto m2= glm::rotate(glm::identity<glm::mat4>(), (float)RTLIB_M_PI / 6.0f, glm::vec3(3.0f, 2.0f, 1.0f));
	std::cout << glm::to_string(m2*m2*m2* m2 * m2 * m2* m2 * m2 * m2 * m2 * m2 * m2) << std::endl;
	auto app = std::shared_ptr<test::RTApplication>(new TestLibTestApplication(1024, 1024, "TestLibTest"));
	(m*m*m).Show();
	std::cout << glm::to_string(m2*m2*m2) << std::endl;
	return app->Run(argc,argv);
}