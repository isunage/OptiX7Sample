#ifndef TEST_TEST24_GUI_H
#define TEST_TEST24_GUI_H
#include <TestLib/RTAssets.h>
#include <TestLib/RTApplication.h>
#include <TestLib/RTGui.h>
#include <TestLib/RTFrameBuffer.h>
class Test24GuiDelegate : public test::RTAppGuiDelegate
{
public:
	Test24GuiDelegate(
		GLFWwindow* window,
		const std::shared_ptr<test::RTFramebuffer>& framebuffer,
		const std::shared_ptr<test::RTObjModelAssetManager>& objModelAssetManager,
		const std::vector<std::string>& framePublicNames,
		const std::vector<std::string>& tracePublicNames,
		std::string& curMainFrameName,
		std::string& curMainTraceName) 
		:
		m_Window{ window }, 
		m_Framebuffer{framebuffer},
		m_ObjModelAssetManager{objModelAssetManager}, 
		m_FramePublicNames{ framePublicNames }, 
		m_TracePublicNames{ tracePublicNames }, 
		m_CurMainFrameName{ curMainFrameName }, 
		m_CurMainTraceName{ curMainTraceName }, 
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
	std::shared_ptr<test::RTFramebuffer> m_Framebuffer;
	std::shared_ptr<test::RTObjModelAssetManager> m_ObjModelAssetManager;
	const std::vector<std::string>& m_FramePublicNames;
	const std::vector<std::string>& m_TracePublicNames;
	std::string& m_CurMainFrameName;
	std::string& m_CurMainTraceName;
};
#endif
