#ifndef TEST_TEST24_GUI_H
#define TEST_TEST24_GUI_H
#include <TestLib/RTAssets.h>
#include <TestLib/RTApplication.h>
#include <TestLib/RTGui.h>
#include <TestLib/RTFrameBuffer.h>
#include <RTLib/ext/Camera.h>
class Test24GuiDelegate : public test::RTAppGuiDelegate
{
public:
	Test24GuiDelegate(
		GLFWwindow* window,
		const std::shared_ptr<rtlib::ext::CameraController>& cameraController,
		const std::shared_ptr<test::RTFramebuffer>& framebuffer,
		const std::shared_ptr<test::RTObjModelAssetManager>& objModelAssetManager,
		const std::vector<std::string>& framePublicNames,
		const std::vector<std::string>& tracePublicNames,
		std::unordered_set<std::string>& launchTracerSet,
		std::string& curMainFrameName,
		std::string& curMainTraceName,
		std::string& curObjModelName,
		bool& updateCamera)
		:
		m_Window{ window }, 
		m_Framebuffer{framebuffer},
		m_CameraController{ cameraController },
		m_ObjModelAssetManager{objModelAssetManager}, 
		m_FramePublicNames{ framePublicNames }, 
		m_TracePublicNames{ tracePublicNames }, 
		m_CurMainFrameName{ curMainFrameName }, 
		m_CurObjModelName{  curObjModelName  },
		m_CurMainTraceName{ curMainTraceName }, 
		m_LaunchTracerSet { launchTracerSet  },
		m_UpdateCamera{updateCamera},
		m_Gui{ std::make_shared<test::RTGui>(window) }
	{}
	// RTAppGuiDelegate を介して継承されました
	virtual void Initialize() override;
	virtual void CleanUp() override;
	virtual void DrawFrame() override;
	virtual auto GetGui() const->std::shared_ptr<test::RTGui> override;
	virtual ~Test24GuiDelegate();
private:
	GLFWwindow* m_Window = nullptr;
	std::shared_ptr<test::RTGui> m_Gui = nullptr;
	std::shared_ptr<rtlib::ext::CameraController> m_CameraController;
	std::shared_ptr<test::RTFramebuffer> m_Framebuffer;
	std::shared_ptr<test::RTObjModelAssetManager> m_ObjModelAssetManager;
	const std::vector<std::string>&  m_FramePublicNames;
	const std::vector<std::string>&  m_TracePublicNames;
	std::unordered_set<std::string>& m_LaunchTracerSet;
	std::string& m_CurObjModelName;
	std::string& m_CurMainFrameName;
	std::string& m_CurMainTraceName;
	bool& m_UpdateCamera;
};
#endif
