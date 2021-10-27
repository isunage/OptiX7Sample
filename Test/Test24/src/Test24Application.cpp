#include "../include/Test24Application.h"
#include "../include/Test24Renderer.h"
#include "../include/Test24Gui.h"
#include "../include/Tracers/Test24DebugTracer.h"
#include <RTLib/ext/Resources/CUDA.h>
#include <TestLib/RTUtils.h>
#include <GLFW/glfw3.h>
void test::Test24Application::Initialize()
{
	this->InitBase();
	this->InitFrameResources();
	this->InitGui();
	this->InitRenderer();
	this->InitAssets();
	this->InitAccelerationStructures();
	this->InitCamera();
	this->InitLight();
	this->InitTracers();
}

void test::Test24Application::MainLoop()
{

	glClearColor(1.0f, 1.0f, 1.0f, 0.0f);
	while (!this->QuitLoop())
	{
		this->Trace();
		//Require Next Frame
		glFlush();
		glClear(GL_COLOR_BUFFER_BIT);
		this->DrawFrame();
		this->PollEvents();
		this->Update();
		glfwSwapBuffers(m_Window);
	}
}

void test::Test24Application::CleanUp()
{
	this->FreeTracers();
	this->FreeLight();
	this->FreeCamera();
	this->FreeAccelerationStructures();
	this->FreeAssets();
	this->FreeRenderer();
	this->FreeGui();
	this->FreeFrameResources();
	this->FreeBase();
}


void test::Test24Application::InitAssets()
{
	//Obj
	m_ObjAssetManager = std::make_shared<test::assets::ObjAssetManager>();

	if (!m_ObjAssetManager->LoadAsset("Sponza", []() {
		auto variables = rtlib::ext::VariableMap();
		variables.SetString("objFilePath", TEST_TEST24_DATA_PATH"/Models/Sponza/sponza.obj");
		variables.SetString("mtlFileDir", TEST_TEST24_DATA_PATH"/Models/Sponza/");
		return variables;
		}())) {
		throw std::runtime_error("Failed To Load ObjAsset!");
	}

	for (auto& [assetName, asset] : m_ObjAssetManager->GetAssets())
	{
		auto objAsset = test::RTAsset::As<test::assets::ObjAsset>(asset);
		objAsset->GetMeshGroup()->GetSharedResource()->vertexBuffer.AddGpuComponent<rtlib::ext::resources::CUDABufferComponent<float3>>("CUDA");
		objAsset->GetMeshGroup()->GetSharedResource()->texCrdBuffer.AddGpuComponent<rtlib::ext::resources::CUDABufferComponent<float3>>("CUDA");
		objAsset->GetMeshGroup()->GetSharedResource()->normalBuffer.AddGpuComponent<rtlib::ext::resources::CUDABufferComponent<float3>>("CUDA");
		for (auto& [uniqueName, uniqueResource] : objAsset->GetMeshGroup()->GetUniqueResources()) {
			uniqueResource->triIndBuffer.AddGpuComponent<rtlib::ext::resources::CUDABufferComponent<uint3>>("CUDA");
			uniqueResource->matIndBuffer.AddGpuComponent<rtlib::ext::resources::CUDABufferComponent<uint32_t>>("CUDA");
		}
	}
	//Img
	m_ImgAssetManager = std::make_shared<test::assets::ImgAssetManager>();

	if (!m_ImgAssetManager->LoadAsset("", []() {
		auto variables = rtlib::ext::VariableMap();
		variables.SetString("path", TEST_TEST24_DATA_PATH"/Textures/white.png");
		return variables;
		}())) {
		throw std::runtime_error("Failed To Load ImgAsset!");
	}

	auto LoadObjTexture = [](auto& imgAssetManager, auto& material, auto name)
	{
		if (!material.HasString(name)) {
			return;
		}
		auto texPath = material.GetString(name);
		if (imgAssetManager->HasAsset(texPath))
		{
			return;
		}
		if (!imgAssetManager->LoadAsset(
			texPath, [&texPath]() {
				auto variables = rtlib::ext::VariableMap();
				variables.SetString("path", texPath);
				return variables;
			}())) {
			material.SetString(name, "");
		}
	};

	for (auto& [assetName, asset] : m_ObjAssetManager->GetAssets())
	{
		auto objAsset = test::RTAsset::As<test::assets::ObjAsset>(asset);
		for (auto& material : objAsset->GetMaterials()) {
			LoadObjTexture(m_ImgAssetManager, material, "diffTex");
			LoadObjTexture(m_ImgAssetManager, material, "emitTex");
			LoadObjTexture(m_ImgAssetManager, material, "specTex");
			LoadObjTexture(m_ImgAssetManager, material, "shinTex");
		}
	}

	for (auto& [assetName, asset] : m_ImgAssetManager->GetAssets())
	{
		auto imgAsset = test::RTAsset::As<test::assets::ImgAsset>(asset);
		imgAsset->GetImage2D().AddGpuComponent<rtlib::ext::resources::CUDATextureImage2DComponent<uchar4>>("CUDATexture");
	}

	for (auto& [assetName, asset] : m_ObjAssetManager->GetAssets())
	{
		auto objAsset = test::RTAsset::As<test::assets::ObjAsset>(asset);
		objAsset->GetMeshGroup()->GetSharedResource()->vertexBuffer.AddGpuComponent<rtlib::ext::resources::CUDABufferComponent<float3>>("CUDA");
		objAsset->GetMeshGroup()->GetSharedResource()->texCrdBuffer.AddGpuComponent<rtlib::ext::resources::CUDABufferComponent<float3>>("CUDA");
		objAsset->GetMeshGroup()->GetSharedResource()->normalBuffer.AddGpuComponent<rtlib::ext::resources::CUDABufferComponent<float3>>("CUDA");
		for (auto& [uniqueName, uniqueResource] : objAsset->GetMeshGroup()->GetUniqueResources()) {
			uniqueResource->triIndBuffer.AddGpuComponent<rtlib::ext::resources::CUDABufferComponent<uint3>>("CUDA");
			uniqueResource->matIndBuffer.AddGpuComponent<rtlib::ext::resources::CUDABufferComponent<uint32_t>>("CUDA");
		}
	}
}

void test::Test24Application::InitBase()
{
	if (glfwInit() == GLFW_FALSE)
	{
		throw test::exceptions::LibraryLoadError("GLFW");
	}
	glfwWindowHint(GLFW_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
	m_Window = glfwCreateWindow(m_FbWidth, m_FbHeight, m_Title, nullptr, nullptr);
	if (!m_Window)
	{
		throw test::exceptions::WindowCreateError(m_Title);
	}
	glfwMakeContextCurrent(m_Window);
	glfwSetWindowUserPointer(m_Window, this);
	glfwSetFramebufferSizeCallback(m_Window, FrameBufferSizeCallback);
	glfwSetCursorPosCallback(m_Window, CursorPositionCallback);

	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		throw test::exceptions::LibraryLoadError("GLAD");
	}

	RTLIB_CUDA_CHECK(cudaFree(0));
	RTLIB_OPTIX_CHECK(optixInit());
	m_OptiXContext = std::make_shared<rtlib::OPXContext>(rtlib::OPXContext::Desc{ 0, 0, OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL, 4 });
}

void test::Test24Application::InitFrameResources()
{
	m_FrameBuffer = std::make_shared<RTFrameBuffer>(m_FbWidth, m_FbHeight);
	m_FrameBuffer->AddCUGLBuffer("Default");
	m_FrameBuffer->GetCUGLBuffer("Default").upload(
		std::vector<uchar4>(m_FbWidth * m_FbHeight, make_uchar4(0, 0, 0, 255))
	);
	m_FrameBuffer->AddCUGLBuffer("Diffuse" );
	m_FrameBuffer->GetCUGLBuffer("Diffuse").upload(
		std::vector<uchar4>(m_FbWidth * m_FbHeight, make_uchar4(0, 0, 0, 255))
	);
	m_FrameBuffer->AddCUGLBuffer("Specular");
	m_FrameBuffer->GetCUGLBuffer("Specular").upload(
		std::vector<uchar4>(m_FbWidth * m_FbHeight, make_uchar4(0, 0, 0, 255))
	);
	m_FrameBuffer->AddCUGLBuffer("Emission");
	m_FrameBuffer->GetCUGLBuffer("Emission").upload(
		std::vector<uchar4>(m_FbWidth * m_FbHeight, make_uchar4(0, 0, 0, 255))
	);
	m_FrameBuffer->AddCUGLBuffer("Shinness");
	m_FrameBuffer->GetCUGLBuffer("Shinness").upload(
		std::vector<uchar4>(m_FbWidth * m_FbHeight, make_uchar4(0, 0, 0, 255))
	);
	m_FrameBuffer->AddCUGLBuffer("Transmit");
	m_FrameBuffer->GetCUGLBuffer("Transmit").upload(
		std::vector<uchar4>(m_FbWidth * m_FbHeight, make_uchar4(0, 0, 0, 255))
	);
	m_FrameBuffer->AddCUGLBuffer("TexCoord");
	m_FrameBuffer->GetCUGLBuffer("TexCoord").upload(
		std::vector<uchar4>(m_FbWidth * m_FbHeight, make_uchar4(0, 0, 0, 255))
	);
	m_FrameBuffer->AddCUGLBuffer("Normal");
	m_FrameBuffer->GetCUGLBuffer("Normal").upload(
		std::vector<uchar4>(m_FbWidth * m_FbHeight, make_uchar4(0, 0, 0, 255))
	);
	m_FrameBuffer->AddCUGLBuffer("Depth");
	m_FrameBuffer->GetCUGLBuffer("Depth").upload(
		std::vector<uchar4>(m_FbWidth * m_FbHeight, make_uchar4(0, 0, 0, 255))
	);
	m_FrameBuffer->AddCUGLBuffer("STreeCol");
	m_FrameBuffer->GetCUGLBuffer("STreeCol").upload(
		std::vector<uchar4>(m_FbWidth * m_FbHeight, make_uchar4(0, 0, 0, 255))
	);
}

void test::Test24Application::InitGui()
{
	m_Gui = test::Test24Gui::New(m_Window,m_FrameBuffer);
	m_Gui->Initialize();
}

void test::Test24Application::InitRenderer()
{
	m_Renderer = test::Test24Renderer::New(m_Window, m_FrameBuffer, m_Gui);
	m_Renderer->Initialize();
	m_Gui->SetString("RenderFrame", "Diffuse");
}

void test::Test24Application::InitAccelerationStructures()
{
	OptixAccelBuildOptions accelBuildOptions = {};
	accelBuildOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
	accelBuildOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

	m_GASHandles["World"] = std::make_shared<rtlib::ext::GASHandle>();

	{
		for (auto& [assetName, asset] : m_ObjAssetManager->GetAssets())
		{
			auto objAsset = test::RTAsset::As<test::assets::ObjAsset>(asset);
			for (auto& [uniqueName, uniqueResource] : objAsset->GetMeshGroup()->GetUniqueResources()) {
				auto mesh = rtlib::ext::Mesh::New();
				mesh->SetSharedResource(objAsset->GetMeshGroup()->GetSharedResource());
				mesh->SetUniqueResource(uniqueName, uniqueResource);
				m_GASHandles["World"]->AddMesh(mesh);
			}
		}
	}

	m_GASHandles["World"]->Build(m_OptiXContext.get(), accelBuildOptions);

	//TopLevel
	m_IASHandles["TopLevel"] = std::make_shared<rtlib::ext::IASHandle>();
	m_IASHandles["TopLevel"]->AddInstanceSet(std::make_shared<rtlib::ext::InstanceSet>());
	rtlib::ext::Instance worldGasInstance = {};
	worldGasInstance.Init(m_GASHandles["World"]);
	worldGasInstance.SetSbtOffset(0);
	m_IASHandles["TopLevel"]->GetInstanceSet(0)->SetInstance(worldGasInstance);
	m_IASHandles["TopLevel"]->Build(m_OptiXContext.get(), accelBuildOptions);
}

void test::Test24Application::InitLight()
{
}

void test::Test24Application::InitCamera()
{
	float fovY = 30.0f;
	float sensitivity = 0.125f;
	float speed = 10.0f;
	m_CameraController = CameraControllerPtr(new rtlib::ext::CameraController({ 0.0f, 1.0f, 5.0f }));
	m_CameraController->SetMouseSensitivity(sensitivity);
	m_CameraController->SetMovementSpeed(speed);
	auto camera = m_CameraController->GetCamera(fovY, m_FbAspect);
	auto camera_eye = camera.getEye();
	auto [camera_u,camera_v,camera_w] = camera.getUVW();
	m_Gui->SetFloat3("Camera.Eye", { camera_eye.x,camera_eye.y,camera_eye.z });
	m_Gui->SetFloat3("Camera.U"  , { camera_u.x,camera_u.y,camera_u.z });
	m_Gui->SetFloat3("Camera.V"  , { camera_v.x,camera_v.y,camera_v.z });
	m_Gui->SetFloat3("Camera.W"  , { camera_w.x,camera_w.y,camera_w.z });
	m_Gui->SetFloat1("Camera.FovY"         , fovY);
	m_Gui->SetFloat1("Camera.FbAspect"     , m_FbAspect);
	m_Gui->SetFloat1("Camera.Sensitivity"  , sensitivity);
	m_Gui->SetFloat1("Camera.MovementSpeed", speed);
}

void test::Test24Application::InitSTree()
{
}

void test::Test24Application::InitReSTIR()
{
}

void test::Test24Application::InitTracers()
{
	m_Tracers["Debug"] = test::RTTracerPtr(new test::tracers::Test24DebugTracer(m_OptiXContext,m_Gui,m_IASHandles["TopLevel"],m_ObjAssetManager,m_ImgAssetManager));
	m_Tracers["Debug"]->Initialize();
}

void test::Test24Application::InitTimer()
{
}

bool test::Test24Application::QuitLoop()
{
	return glfwWindowShouldClose(m_Window);
}

void test::Test24Application::Trace()
{
	RTTraceConfig traceConfig = {};
	traceConfig.width       = m_FbWidth;
	traceConfig.height      = m_FbHeight;
	traceConfig.depth       = 1;
	traceConfig.isSync      = true;
	traceConfig.pUserData   = nullptr;
	traceConfig.stream      = nullptr;
	test::tracers::Test24DebugTracer::UserData userData = {};
	userData.diffuseBuffer  = m_FrameBuffer->GetCUGLBuffer("Diffuse" ).map();
	userData.specularBuffer = m_FrameBuffer->GetCUGLBuffer("Specular").map();
	userData.emissionBuffer = m_FrameBuffer->GetCUGLBuffer("Emission").map();
	userData.transmitBuffer = m_FrameBuffer->GetCUGLBuffer("Transmit").map();
	userData.shinnessBuffer = m_FrameBuffer->GetCUGLBuffer("Shinness").map();
	userData.texCoordBuffer = m_FrameBuffer->GetCUGLBuffer("TexCoord").map();
	userData.normalBuffer   = m_FrameBuffer->GetCUGLBuffer("Normal"  ).map();
	userData.depthBuffer    = m_FrameBuffer->GetCUGLBuffer("Depth"   ).map();
	userData.sTreeColBuffer = m_FrameBuffer->GetCUGLBuffer("STreeCol").map();
	m_Tracers["Debug"]->Launch(traceConfig);
	m_FrameBuffer->GetCUGLBuffer("Diffuse" ).unmap();
	m_FrameBuffer->GetCUGLBuffer("Specular").unmap();
	m_FrameBuffer->GetCUGLBuffer("Emission").unmap();
	m_FrameBuffer->GetCUGLBuffer("Transmit").unmap();
	m_FrameBuffer->GetCUGLBuffer("Shinness").unmap();
	m_FrameBuffer->GetCUGLBuffer("TexCoord").unmap();
	m_FrameBuffer->GetCUGLBuffer("Normal"  ).unmap();
	m_FrameBuffer->GetCUGLBuffer("Depth"   ).unmap();
	m_FrameBuffer->GetCUGLBuffer("STreeCol").unmap();
}

void test::Test24Application::DrawFrame()
{
	m_Renderer->Render();
	test::SavePNGFromGL(TEST_TEST24_CUDA_PATH"/diffuse.png", m_FrameBuffer->GetCUGLBuffer("Default").getHandle(), m_FbWidth, m_FbHeight);
}

void test::Test24Application::PollEvents()
{
	glfwPollEvents();
}

void test::Test24Application::Update()
{
	if (m_ResizeFrame)
	{
		m_FrameBuffer->Resize(m_FbWidth, m_FbHeight);
		m_Renderer->Resize(m_FbWidth, m_FbHeight);
		m_ResizeFrame = false;
	}
}

void test::Test24Application::FreeRenderer()
{
	m_FrameBuffer.reset();
}

void test::Test24Application::FreeGui()
{
	m_Gui.reset();
}

void test::Test24Application::FreeAssets()
{
	m_ObjAssetManager.reset();
	m_ImgAssetManager.reset();
}

void test::Test24Application::FreeAccelerationStructures()
{
	for (auto& [name, gas] : m_GASHandles)
	{
		gas.reset();
	}
	m_GASHandles.clear();
	for (auto& [name, ias] : m_IASHandles)
	{
		ias.reset();
	}
	m_IASHandles.clear();
}

void test::Test24Application::FreeLight()
{
}

void test::Test24Application::FreeCamera()
{
}

void test::Test24Application::FreeSTree()
{
}

void test::Test24Application::FreeBase()
{
	m_OptiXContext.reset();
	glfwDestroyWindow(m_Window);
	m_Window = nullptr;
	glfwTerminate();
}

void test::Test24Application::FreeFrameResources()
{
}

void test::Test24Application::FreeTracers()
{
	m_Tracers["Debug"]->CleanUp();
	m_Tracers.clear();
}
