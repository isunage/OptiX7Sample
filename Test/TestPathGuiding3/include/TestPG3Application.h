#ifndef TEST_TEST_PG3_APPLICATION_H
#define TEST_TEST_PG3_APPLICATION_H
#include <RTApplication.h>
#include <RTTracer.h>
#include <RTFrameBuffer.h>
#include <RTAssets.h>
#include <RTPathGuidingUtils.h>
#include <TestPGConfig.h>
#include <RTLib/GL.h>
#include <RTLib/CUDA.h>
#include <RTLib/CUDA_GL.h>
#include <RTLib/Optix.h>
#include <RTLib/Camera.h>
#include <RTLib/ext/RectRenderer.h>
#include <RTLib/ext/TraversalHandle.h>
#include <RayTrace.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <fstream>
#include <filesystem>
#include <string>
#include <memory>
#include <stdexcept>
class TestPG3Application :public test::RTApplication
{
private:
	static inline constexpr const char* kDebugFrameNames[]  = {
		"Diffuse","Specular","Emission","Transmit","TexCoord","Normal","Depth","STree"
	};
	static inline constexpr std::array<float,3> kDefaultLightColor = { 10.0f,10.0f,10.0f };
private:
	using Renderer          = std::unique_ptr<rtlib::ext::RectRenderer>;
	using RenderTexture     = std::unique_ptr<rtlib::GLTexture2D<uchar4>>;
	using PipelineMap       = std::unordered_map<std::string, rtlib::OPXPipeline>;
	using ModuleMap         = std::unordered_map<std::string, rtlib::OPXModule>;
	using RGProgramGroupMap = std::unordered_map<std::string, rtlib::OPXRaygenPG>;
	using MSProgramGroupMap = std::unordered_map<std::string, rtlib::OPXMissPG>;
	using HGProgramGroupMap = std::unordered_map<std::string, rtlib::OPXHitgroupPG>;
	using GeometryASMap     = std::unordered_map<std::string, rtlib::ext::GASHandlePtr>;
	using InstanceASMap     = std::unordered_map<std::string, rtlib::ext::IASHandlePtr>;
public:
	static auto  New() -> std::shared_ptr<test::RTApplication> { return std::make_shared<TestPG3Application>(); }
	// RTApplication を介して継承されました
	virtual void Initialize() override
	{
		this->InitGLFW();
		this->InitWindow();
		this->InitGLAD();
		this->InitGui();
		this->InitCUDA();
		this->InitPipelines();
		this->InitAssets();
		this->InitAccelerationStructures();
		this->InitLight();
		this->InitCamera();
		this->InitSTree();
		this->InitFrameResources();
		this->InitTracers();
	}
	virtual void MainLoop() override
	{
		this->PrepareLoop();
		while (!this->QuitLoop()) {
			this->Trace();
			glFlush();
			glClear(GL_COLOR_BUFFER_BIT);
			glClearColor(1.0f, 1.0f, 1.0f, 0.0f);
			this->DrawFrame();
			this->DrawGui();
			this->PollEvents();
			this->Update();
			glfwSwapBuffers(m_Window);
		}
	}
	virtual void CleanUp() override
	{
		this->FreeTracers();
		this->FreeFrameResources();
		this->FreeSTree();
		this->FreeCamera();
		this->FreeLight();
		this->FreeAccelerationStructures();
		this->FreeAssets();
		this->FreePipelines();
		this->FreeCUDA();
		this->FreeGui();
		this->FreeWindow();
		this->FreeGLFW();
	}
	virtual ~TestPG3Application() {}
public:
	auto GetFbWidth()const  -> int  { return m_FbWidth; }
	auto GetFbHeight()const -> int  { return m_FbHeight; }
	auto GetMaxTraceDepth()const->unsigned int { return m_MaxTraceDepth; }
	bool IsFrameResized()const { return m_ResizeFrame;  }
	bool IsCameraUpdated()const{ return m_UpdateCamera; }
	bool IsLightUpdated()const { return m_UpdateLight;  }
	auto GetOPXContext() const->std::shared_ptr<rtlib::OPXContext>;
	auto GetOPXPipeline(const std::string& name)->rtlib::OPXPipeline&;
	auto GetRGProgramGroup(const std::string& name)->rtlib::OPXRaygenPG &;
	auto GetMSProgramGroup(const std::string& name)->rtlib::OPXMissPG&;
	auto GetHGProgramGroup(const std::string& name)->rtlib::OPXHitgroupPG&;
	auto GetTLAS()const     -> rtlib::ext::IASHandlePtr;
	auto GetMaterials()const-> const std::vector<rtlib::ext::Material>&;
	auto GetCamera()const -> rtlib::Camera;
	auto GetLight()const  -> ParallelLight;
	auto GetSTree()const  -> std::shared_ptr<test::RTSTreeWrapper>;
	auto GetTexture(const std::string& name)const->const rtlib::CUDATexture2D<uchar4>&;
private:
	//Init
	void InitGLFW();
	void InitWindow();
	void InitGLAD();
	void InitCUDA();
	void InitGui();
	void InitPipelines();
	void InitAssets();
	void InitAccelerationStructures();
	void InitLight();
	void InitCamera();
	void InitSTree();
	void InitFrameResources();
	void InitTracers();
	//ShouldClose
	void PrepareLoop();
	bool QuitLoop();
	void Trace();
	void DrawFrame();
	void DrawGui();
	void PollEvents();
	void Update();
	//Lock
	void   LockUpdate();
	void UnLockUpdate();
	//Free
	void FreeGLFW();
	void FreeWindow();
	void FreeCUDA();
	void FreeGui();
	void FreePipelines();
	void FreeAssets();
	void FreeAccelerationStructures();
	void FreeLight();
	void FreeCamera();
	void FreeSTree();
	void FreeFrameResources();
	void FreeTracers();
private:
	static auto GetHandleFromWindow(GLFWwindow* window)-> TestPG3Application* {
		return reinterpret_cast<TestPG3Application*>(glfwGetWindowUserPointer(window));
	}
	static void CursorPositionCallback(GLFWwindow* window,  double xPos, double  yPos) {
		if (!window) {
			return;
		}
		auto* app = GetHandleFromWindow(window);
		if (!app) {
			return;
		}
		auto prvCursorPos   = app->m_CurCursorPos;
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
	//Window
	GLFWwindow*							 m_Window		   = nullptr;
	int                                  m_FbWidth	       = 1024;
	int                                  m_FbHeight		   = 1024;
	float                                m_FbAspect		   = 1.0f;
	const char*							 m_Title		   = "TestPG3";
	float2                               m_CurCursorPos    = {};
	float2                               m_DelCursorPos    = {};
	double                               m_CurTime         = 0.0;
	double                               m_DelTime         = 0.0;
	float                                m_FovY            = 30.0f;
	bool                                 m_ResizeFrame     = false;
	bool                                 m_UpdateLight     = false;
	bool                                 m_UpdateCamera    = false;
	bool                                 m_LaunchDebug     = false;
	bool                                 m_LockUpdate      = false;
	std::string                          m_CurRenderFrame  = "Default";
	std::string                          m_CurDebugFrame   = "Diffuse";
	unsigned int                         m_SamplePerALL    = 0;
	unsigned int                         m_SamplePerLaunch = 1;
	unsigned int                         m_MaxTraceDepth   = 4;
	//Renderer
	Renderer                             m_Renderer        = nullptr;
	//RenderTexture
	RenderTexture                        m_RenderTexture    = {};
	RenderTexture                        m_DebugTexture     = {};
	//OPTIX
	std::shared_ptr<rtlib::OPXContext>   m_Context          = nullptr;
	PipelineMap	                         m_Pipelines        = {};
	ModuleMap                            m_Modules          = {};
	RGProgramGroupMap                    m_RGProgramGroups  = {};
	MSProgramGroupMap                    m_MSProgramGroups  = {};
	HGProgramGroupMap                    m_HGProgramGroups  = {};
	//FrameBuffer
	std::unique_ptr<test::RTFrameBuffer> m_FrameBuffer      = nullptr;
	//Assets
	test::RTTextureAssetManager          m_TextureAssets    = {};
	test::RTObjModelAssetManager         m_ObjModelAssets   = {};
	//AccelerationStructure
	GeometryASMap                        m_GASHandles       = {};
	InstanceASMap                        m_IASHandles       = {};
	//Materials
	std::vector<rtlib::ext::Material>    m_Materials        = {};
	//Camera
	rtlib::CameraController              m_CameraController = {};
	//Light
	ParallelLight                        m_ParallelLight    = {};
	//STree
	std::shared_ptr<test::RTSTreeWrapper>m_STree            = {};
	//Tracer
	std::shared_ptr<test::RTTracer>      m_TraceActor       = {};
	std::shared_ptr<test::RTTracer>      m_GuideActor       = {};
	std::shared_ptr<test::RTTracer>      m_DebugActor       = {};
};
#endif