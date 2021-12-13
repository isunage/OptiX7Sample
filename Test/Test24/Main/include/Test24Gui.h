#ifndef TEST_TEST24_GUI_H
#define TEST_TEST24_GUI_H
#include <TestLib/RTAssets.h>
#include <TestLib/RTApplication.h>
#include <TestLib/RTGui.h>
#include <TestLib/RTFrameBuffer.h>
#include <RTLib/ext/Camera.h>
#include <Test24Event.h>
class Test24GuiDelegate : public test::RTAppGuiDelegate
{
public:
	Test24GuiDelegate(
		GLFWwindow* window,
		const std::shared_ptr<rtlib::ext::CameraController>& cameraController,
		const std::shared_ptr<test::RTFramebuffer>         & framebuffer,
		const std::shared_ptr<test::RTObjModelAssetManager>& objModelAssetManager,
		const std::shared_ptr<test::RTTextureAssetManager> & textureManager,
		const std::array<float, 2>& curCursorPos,
		const std::array<float, 2>& delCursorPos,
		const std::array<float, 2>& scrollOffsets,
		const float& curFrameTime,
		const float& delFrameTime,
		const std::vector<std::string> & framePublicNames,
		const std::vector<std::string> & tracePublicNames,
		std::unordered_set<std::string>& launchTracerSet,
		float3& bgLightColor,
		std::string& curMainFrameName,
		std::string& curMainTraceName,
		std::string& curObjModelName, 
		unsigned int& maxTraceDepth,
		const unsigned int& samplePerAll,
		unsigned int& samplePerLaunch,
		unsigned int& eventFlags):
		m_CurCursorPos {curCursorPos},
		m_DelCursorPos{ delCursorPos },
		m_ScrollOffsets{ scrollOffsets },
		m_CurFrameTime{ curFrameTime },
		m_DelFrameTime{ delFrameTime },
		m_Window{ window }, 
		m_Framebuffer{framebuffer},
		m_CameraController{ cameraController },
		m_ObjModelAssetManager{objModelAssetManager}, 
		m_TextureManager{textureManager},
		m_BgLightColor{bgLightColor},
		m_FramePublicNames{ framePublicNames }, 
		m_TracePublicNames{ tracePublicNames }, 
		m_CurMainFrameName{ curMainFrameName }, 
		m_CurObjModelName {  curObjModelName },
		m_CurMainTraceName{ curMainTraceName }, 
		m_MaxTraceDepth{ maxTraceDepth },
		m_SamplePerAll{ samplePerAll },
		m_SamplePerLaunch{samplePerLaunch},
		m_LaunchTracerSet { launchTracerSet  },
		m_EventFlags{eventFlags},
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
	std::shared_ptr<test::RTGui>                  m_Gui = nullptr;
	std::shared_ptr<rtlib::ext::CameraController> m_CameraController;
	std::shared_ptr<test::RTFramebuffer>          m_Framebuffer;
	std::shared_ptr<test::RTObjModelAssetManager> m_ObjModelAssetManager;
	std::shared_ptr<test::RTTextureAssetManager>  m_TextureManager;
	float3&                     m_BgLightColor;
	const std::array<float, 2>& m_CurCursorPos;
	const std::array<float, 2>& m_DelCursorPos;
	const std::array<float, 2>& m_ScrollOffsets;
	const float& m_CurFrameTime;
	const float& m_DelFrameTime;
	const std::vector<std::string>&  m_FramePublicNames;
	const std::vector<std::string>&  m_TracePublicNames;
	std::unordered_set<std::string>& m_LaunchTracerSet;
	std::string&  m_CurObjModelName;
	std::string&  m_CurMainFrameName;
	std::string&  m_CurMainTraceName;
	unsigned int& m_MaxTraceDepth;
	const unsigned int& m_SamplePerAll;
	unsigned int& m_SamplePerLaunch;
	unsigned int& m_EventFlags;
};
#endif
