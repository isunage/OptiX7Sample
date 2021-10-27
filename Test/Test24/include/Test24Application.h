#ifndef TEST_TEST24_APPLICATION_H
#define TEST_TEST24_APPLICATION_H
#include <Test24Config.h>
#include <TestLib/RTTracer.h>
#include <TestLib/RTRenderer.h>
#include <TestLib/RTFrameBuffer.h>
#include <TestLib/RTApplication.h>
#include <TestLib/RTExceptions.h>
#include <TestLib/RTGui.h>
#include <TestLib/Assets/ImgAssets.h>
#include <TestLib/Assets/ObjAssets.h>
#include <RTLib/ext/TraversalHandle.h>
#include <RTLib/ext/Camera.h>
#include <GLFW/glfw3.h>
namespace test
{
	class Test24Application : public RTApplication
	{
	private:
		using IASHandleMap        = std::unordered_map<std::string, rtlib::ext::IASHandlePtr>;
		using GASHandleMap        = std::unordered_map<std::string, rtlib::ext::GASHandlePtr>;
		using RTTracerMap         = std::unordered_map<std::string, test::RTTracerPtr>;
		using RTFrameBufferPtr    = std::shared_ptr<RTFrameBuffer>;
		using CameraControllerPtr = std::shared_ptr<rtlib::ext::CameraController>;
	public:
		Test24Application(int width, int height, const char* title)noexcept :m_FbWidth{ width }, m_FbHeight{ height }, m_Title{ title }{}
		// 
		static auto New(int width, int height, const char* title) -> RTApplicationPtr { return RTApplicationPtr(new Test24Application(width, height, title)); }
		// RTApplication を介して継承されました
		virtual void Initialize() override;
		// 
		virtual void MainLoop() override;
		// 
		virtual void CleanUp() override;
		// 
		virtual ~Test24Application() {}
	private:
		void InitBase();
		void InitFrameResources();
		void InitGui();
		void InitRenderer();
		void InitAssets();
		void InitAccelerationStructures();
		void InitCamera();
		void InitLight();
		void InitSTree();
		void InitReSTIR();
		void InitTracers();
		//ShouldClose
		void InitTimer();
		bool QuitLoop();
		void Trace();
		void DrawFrame();
		void PollEvents();
		void Update();
		//Free
		void FreeBase();
		void FreeFrameResources();
		void FreeGui();
		void FreeRenderer();
		void FreeAssets();
		void FreeAccelerationStructures();
		void FreeCamera();
		void FreeLight();
		void FreeSTree();
		void FreeTracers();
	private:
		static auto GetHandleFromWindow(GLFWwindow* window)->Test24Application*
		{
			return reinterpret_cast<Test24Application*>(glfwGetWindowUserPointer(window));
		}
		static void CursorPositionCallback(GLFWwindow* window, double xPos, double  yPos) {
			if (!window) {
				return;
			}
			auto* app = GetHandleFromWindow(window);
			if (!app) {
				return;
			}
			auto prvCursorPos = app->m_CurCursorPos;
			app->m_CurCursorPos = make_float2(xPos, yPos);
			app->m_DelCursorPos = app->m_CurCursorPos - prvCursorPos;
		}
		static void FrameBufferSizeCallback(GLFWwindow* window, int fbWidth, int fbHeight) {
			if (!window) {
				return;
			}
			auto* app = GetHandleFromWindow(window);
			if (!app) {
				return;
			}
			if (fbWidth != app->m_FbWidth || fbHeight != app->m_FbHeight) {
				app->m_FbWidth     = fbWidth;
				app->m_FbHeight    = fbHeight;
				app->m_FbAspect    = static_cast<float>(fbWidth) / static_cast<float>(fbHeight);
				app->m_ResizeFrame = true;
				glViewport(0, 0, fbWidth, fbHeight);
			}
		}
	private:
		int								   m_FbWidth         = 1024;
		int						           m_FbHeight        = 1024;
		float                              m_FbAspect        = 1.0f;
		bool                               m_ResizeFrame     = false;
		float2                             m_CurCursorPos    = {};
		float2                             m_DelCursorPos    = {};
		const char*					       m_Title           = "Test24Application";
		GLFWwindow*                        m_Window          = nullptr;
		RTFrameBufferPtr                   m_FrameBuffer     = nullptr;
		RTRendererPtr					   m_Renderer        = {};
		RTGuiPtr                           m_Gui             = {};
		RTTracerMap						   m_Tracers         = {};
		assets::ImgAssetManagerPtr	       m_ImgAssetManager = {};
		assets::ObjAssetManagerPtr         m_ObjAssetManager = {};
		std::shared_ptr<rtlib::OPXContext> m_OptiXContext    = nullptr;
		IASHandleMap					   m_IASHandles      = {};
		GASHandleMap                       m_GASHandles      = {};
		CameraControllerPtr                m_CameraController= nullptr;
	};
}
//どのようにGUI<->Tracer<->Rendererの連携をするかが問題
//一つはGUI側で設定した値、イベントを取得可能にする
#endif
