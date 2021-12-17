#include "..\include\Test24Application.h"
#include <Test24Gui.h>
#include <TestGLTracer.h>
#include <DebugOPXTracer.h>
#include <PathOPXTracer.h>
#include <NEEOPXTracer.h>
#include <GuidePathOPXTracer.h>
#include < GuideNEEOPXTracer.h>
#include <ReSTIROPXTracer.h>
#include <Test24Config.h>
#include <filesystem>
Test24Application::Test24Application(int fbWidth, int fbHeight, std::string name) noexcept : test::RTApplication(name)
{
    m_FbWidth          = fbWidth;
    m_FbHeight         = fbHeight;
    m_Window           = nullptr;
    m_FovY             = 0.0f;
    m_EventFlags       = TEST24_EVENT_FLAG_NONE;
    m_NumRayType       = TEST_TEST24_NUM_RAY_TYPE;
    m_SamplePerAll     = 0;
    m_SamplePerLaunch  = 1;
    m_MaxTraceDepth    = 2;
    m_CurFrameTime     = 0.0f;
    m_DelFrameTime     = 0.0f;
    m_DelCursorPos[0]  = 0.0f;
    m_DelCursorPos[1]  = 0.0f;
    m_CurCursorPos[0]  = 0.0f;
    m_CurCursorPos[1]  = 0.0f;
    m_ScrollOffsets[0] = 0.0f;
    m_ScrollOffsets[1] = 0.0f;
    m_CurMainFrameName = "";
    m_CurMainTraceName = "";
    m_CurObjModelName  = "";
    m_Tracers          = {};
    m_FramePublicNames = {};
    m_TracePublicNames = {};
}

auto Test24Application::New(int fbWidth, int fbHeight, std::string name) noexcept -> std::shared_ptr<test::RTApplication>
{
    return std::shared_ptr<test::RTApplication>(new Test24Application(fbWidth, fbHeight, name));
}

void Test24Application::Initialize()
{
    InitBase();
    InitGui();
    InitScene();
    InitTracers();
}

void Test24Application::MainLoop()      
{
    while (!glfwWindowShouldClose(m_Window))
    {
        this->Launch();
        this->BegFrame();
        this->Render();
        this->Update();
        this->EndFrame();
    }
}

void Test24Application::CleanUp()      
{
    FreeTracers();
    FreeScene();
    FreeGui();
    FreeBase();
}

void Test24Application::InitBase()   
{
    m_Context = std::shared_ptr<test::RTContext>(new test::RTContext(4, 5));
    m_Window = m_Context->NewWindow(m_FbWidth, m_FbHeight, GetName().c_str());
    glfwSetWindowUserPointer(m_Window, this);
    glfwSetFramebufferSizeCallback(m_Window, FramebufferSizeCallback);
    glfwSetCursorPosCallback(m_Window, CursorPosCallback);
    m_Framebuffer = std::shared_ptr<test::RTFramebuffer>(new test::RTFramebuffer(m_FbWidth, m_FbHeight));
    //GBuffer
    m_Framebuffer->AddComponent<test::RTCUDABufferFBComponent<float3>>("GPosition" );
    m_Framebuffer->AddComponent<test::RTCUDABufferFBComponent<float3>>("GNormal"   );
    m_Framebuffer->AddComponent<test::RTCUDABufferFBComponent<float3>>("GDiffuse"  );
    m_Framebuffer->AddComponent<test::RTCUDABufferFBComponent<float3>>("GEmission" );
    m_Framebuffer->AddComponent<test::RTCUDABufferFBComponent<float >>("GDistance" );
    //Render 
    m_Framebuffer->AddComponent<test::RTCUDABufferFBComponent<float3>>("RAccum"    );
    m_Framebuffer->AddComponent<test::RTCUGLBufferFBComponent<uchar4>>("RFrame"    );
    //Debug 
    m_Framebuffer->AddComponent<test::RTCUGLBufferFBComponent<uchar4>>("DNormal"  );
    m_Framebuffer->AddComponent<test::RTCUGLBufferFBComponent<uchar4>>("DTexCoord");
    m_Framebuffer->AddComponent<test::RTCUGLBufferFBComponent<uchar4>>("DDistance");
    m_Framebuffer->AddComponent<test::RTCUGLBufferFBComponent<uchar4>>("DDiffuse" );
    m_Framebuffer->AddComponent<test::RTCUGLBufferFBComponent<uchar4>>("DSpecular");
    m_Framebuffer->AddComponent<test::RTCUGLBufferFBComponent<uchar4>>("DTransmit");
    m_Framebuffer->AddComponent<test::RTCUGLBufferFBComponent<uchar4>>("DEmission");
    m_Framebuffer->AddComponent<test::RTCUGLBufferFBComponent<uchar4>>("DShinness");
    m_Framebuffer->AddComponent<test::RTCUGLBufferFBComponent<uchar4>>("DIOR"     );
    m_Framebuffer->AddComponent<test::RTCUGLBufferFBComponent<uchar4>>("DSTreeCol");
    //Texture
    m_Framebuffer->AddComponent<test::RTGLTextureFBComponent<uchar4>>( "RTexture");
    m_Framebuffer->AddComponent<test::RTGLTextureFBComponent<uchar4>>( "DTexture");
    m_Framebuffer->AddComponent<test::RTGLTextureFBComponent<uchar4>>( "PTexture", GL_NEAREST, GL_NEAREST, GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE);
    //Public Frames
    m_FramePublicNames.clear();
    m_FramePublicNames.push_back("RTexture");
    m_FramePublicNames.push_back("DTexture");
    m_FramePublicNames.push_back("PTexture");
    m_FramePublicNames.push_back("RFrame");
    m_FramePublicNames.push_back("DNormal");
    m_FramePublicNames.push_back("DTexCoord");
    m_FramePublicNames.push_back("DDistance");
    m_FramePublicNames.push_back("DDiffuse");
    m_FramePublicNames.push_back("DSpecular");
    m_FramePublicNames.push_back("DTransmit");
    m_FramePublicNames.push_back("DEmission");
    m_FramePublicNames.push_back("DShinness");
    m_FramePublicNames.push_back("DIOR");
    m_CurMainFrameName = "RFrame";
    //Renderer
    m_Renderer = std::make_unique<rtlib::ext::RectRenderer>();
    m_Renderer->init();
    std::cout << m_Framebuffer->GetComponent<test::RTGLTextureFBComponent<uchar4>>("RTexture")->GetIDString() << std::endl;
    //ObjAssetManager
    m_ObjModelManager = std::make_shared<test::RTObjModelAssetManager>();
    m_TextureManager  = std::make_shared<test::RTTextureAssetManager>();
    m_CameraController = std::make_shared<rtlib::ext::CameraController>(float3{ 0.0f, 1.0f, 5.0f });
    m_CameraController->SetMouseSensitivity(0.125f);
    m_CameraController->SetMovementSpeed(10.f);
    m_CameraController->SetZoom(40.0f);
    //Depth Enable
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
}

void Test24Application::FreeBase()  
{
    m_TextureManager  .reset();
    m_CameraController.reset();
    m_ObjModelManager .reset();
    m_Renderer.reset();
    m_Framebuffer.reset();
    if (m_Window)
    {
        glfwDestroyWindow(m_Window);
        m_Window = nullptr;
    }
    m_Context.reset();
}

void Test24Application::InitGui()
{
     //Gui
    m_GuiDelegate = std::make_unique<Test24GuiDelegate>(
        m_Window,
        m_CameraController, 
        m_Framebuffer,
        m_ObjModelManager,
        m_TextureManager, 
        m_CurCursorPos,
        m_DelCursorPos,
        m_ScrollOffsets,
        m_CurFrameTime,
        m_DelFrameTime,
        m_FramePublicNames, 
        m_TracePublicNames, 
        m_LaunchTracerSet, 
        m_BgLightColor, 
        m_CurMainFrameName, 
        m_CurMainTraceName, 
        m_CurObjModelName,
        m_MaxTraceDepth,
        m_SamplePerAll,
        m_SamplePerLaunch,
        m_EventFlags);
    m_GuiDelegate->Initialize();
    glfwSetScrollCallback(m_Window, ScrollCallback);
}

void Test24Application::FreeGui()
{
    m_GuiDelegate->CleanUp();
    m_GuiDelegate.reset();
}

void Test24Application::RenderGui()
{
    m_GuiDelegate->DrawFrame();
}

void Test24Application::InitScene()
{
    //if (m_ObjModelManager->LoadAsset("CornellBox-Water", TEST_TEST24_DATA_PATH"/Models/CornellBox/CornellBox-Water.obj")) {
    //    m_CurObjModelName = "CornellBox-Water";
    //}
    if (m_ObjModelManager->LoadAsset("Bistro-Exterior", TEST_TEST24_DATA_PATH"/Models/Bistro/Exterior/exterior.obj")) {
        m_CurObjModelName = "Bistro-Exterior";
    }
    {
        size_t materialSize = 0;
        for (auto& [name, objModel] : m_ObjModelManager->GetAssets())
        {
            materialSize += objModel.materials.size();
        }
        m_Materials.resize(materialSize + 1);
        size_t materialOffset = 0;
        for (auto& [name, objModel] : m_ObjModelManager->GetAssets())
        {
            auto& materials = objModel.materials;
            std::copy(std::begin(materials), std::end(materials), m_Materials.begin() + materialOffset);
            materialOffset += materials.size();
        }
    }
    {
        auto smpTexPath = std::filesystem::canonical(std::filesystem::path(TEST_TEST24_DATA_PATH "/Textures/white.png"));
        if (!m_TextureManager->LoadAsset("", smpTexPath.string()))
        {
            throw std::runtime_error("Failed To Load White Texture!");
        }
        for (auto& [name, objModel] : m_ObjModelManager->GetAssets())
        {
            for (auto& material : objModel.materials)
            {
                auto diffTexPath = material.GetString("diffTex");
                auto specTexPath = material.GetString("specTex");
                auto emitTexPath = material.GetString("emitTex");
                auto shinTexPath = material.GetString("shinTex");
                if (diffTexPath != "")
                {
                    if (!m_TextureManager->LoadAsset(diffTexPath, diffTexPath))
                    {
                        std::cout << "DiffTex \"" << diffTexPath << "\" Not Found!\n";
                        material.SetString("diffTex", "");
                    }
                }
                if (specTexPath != "")
                {
                    if (!m_TextureManager->LoadAsset(specTexPath, specTexPath))
                    {
                        std::cout << "SpecTex \"" << specTexPath << "\" Not Found!\n";
                        material.SetString("specTex", "");
                    }
                }
                if (emitTexPath != "")
                {
                    if (!m_TextureManager->LoadAsset(emitTexPath, emitTexPath))
                    {
                        std::cout << "EmitTex \"" << emitTexPath << "\" Not Found!\n";
                        material.SetString("emitTex", "");
                    }
                    else {
                        if (material.GetFloat3("emitCol")[0] == 0.0f &&
                            material.GetFloat3("emitCol")[1] == 0.0f &&
                            material.GetFloat3("emitCol")[2] == 0.0f) {
                            material.SetFloat3("emitCol", { 1.0f,1.0f,1.0f });
                        }
                    }

                }
                if (shinTexPath != "")
                {
                    if (!m_TextureManager->LoadAsset(shinTexPath, shinTexPath))
                    {
                        std::cout << "ShinTex \"" << shinTexPath << "\" Not Found!\n";
                        material.SetString("shinTex", "");
                    }
                }
            }
        }
    }

    m_GASHandles["World"] = std::make_shared<rtlib::ext::GASHandle>();
    m_GASHandles["Light"] = std::make_shared<rtlib::ext::GASHandle>();
    {
        OptixAccelBuildOptions accelBuildOptions = {};
        accelBuildOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
        accelBuildOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

        {
            size_t materialOffset = 0;
            for (auto& [name, objModel] : m_ObjModelManager->GetAssets())
            {
                if (!objModel.meshGroup->GetSharedResource()->vertexBuffer.HasGpuComponent("CUDA"))
                {
                    objModel.meshGroup->GetSharedResource()->vertexBuffer.AddGpuComponent<rtlib::ext::resources::CUDABufferComponent<float3>>("CUDA");
                }
                if (!objModel.meshGroup->GetSharedResource()->normalBuffer.HasGpuComponent("CUDA"))
                {
                    objModel.meshGroup->GetSharedResource()->normalBuffer.AddGpuComponent<rtlib::ext::resources::CUDABufferComponent<float3>>("CUDA");
                }
                if (!objModel.meshGroup->GetSharedResource()->texCrdBuffer.HasGpuComponent("CUDA"))
                {
                    objModel.meshGroup->GetSharedResource()->texCrdBuffer.AddGpuComponent<rtlib::ext::resources::CUDABufferComponent<float2>>("CUDA");
                }
                for (auto& [name, meshUniqueResource] : objModel.meshGroup->GetUniqueResources())
                {
                    if (!meshUniqueResource->matIndBuffer.HasGpuComponent("CUDA"))
                    {
                        meshUniqueResource->matIndBuffer.AddGpuComponent<rtlib::ext::resources::CUDABufferComponent<uint32_t>>("CUDA");
                    }
                    if (!meshUniqueResource->triIndBuffer.HasGpuComponent("CUDA"))
                    {
                        meshUniqueResource->triIndBuffer.AddGpuComponent<rtlib::ext::resources::CUDABufferComponent<uint3>>("CUDA");
                    }

                    auto mesh = rtlib::ext::Mesh::New();
                    mesh->SetUniqueResource(meshUniqueResource);
                    mesh->SetSharedResource(objModel.meshGroup->GetSharedResource());

                    for (auto& matIdx : mesh->GetUniqueResource()->materials)
                    {
                        matIdx += materialOffset;
                    }
                    if (mesh->GetUniqueResource()->variables.GetBool("hasLight"))
                    {
                        m_GASHandles["Light"]->AddMesh(mesh);
                    }
                    else
                    {
                        m_GASHandles["World"]->AddMesh(mesh);
                    }
                }
                materialOffset += objModel.materials.size();
            }
        }
        m_GASHandles["World"]->Build(m_Context->GetOPX7Handle().get(), accelBuildOptions);
        m_GASHandles["Light"]->Build(m_Context->GetOPX7Handle().get(), accelBuildOptions);
    }
    m_IASHandles["TopLevel"] = std::make_shared<rtlib::ext::IASHandle>();
    {
        OptixAccelBuildOptions accelOptions = {};
        accelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
        accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
        //World
        auto worldInstance = rtlib::ext::Instance();
        worldInstance.Init(m_GASHandles["World"]);
        worldInstance.SetSbtOffset(0);
        //Light
        auto lightInstance = rtlib::ext::Instance();
        lightInstance.Init(m_GASHandles["Light"]);
        lightInstance.SetSbtOffset(worldInstance.GetSbtCount() * TEST_TEST24_NUM_RAY_TYPE);
        //InstanceSet
        auto instanceSet = std::make_shared<rtlib::ext::InstanceSet>();
        instanceSet->SetInstance(worldInstance);
        instanceSet->SetInstance(lightInstance);
        instanceSet->Upload();
        //AddInstanceSet
        m_IASHandles["TopLevel"]->AddInstanceSet(instanceSet);
        //Build
        m_IASHandles["TopLevel"]->Build(m_Context->GetOPX7Handle().get(), accelOptions);
    }
}

void Test24Application::FreeScene()
{
    m_GASHandles.clear();
    m_IASHandles.clear();
}

void Test24Application::InitTracers()
{
    m_Tracers.reserve(7);
    m_Tracers["TestGL"] = std::make_shared<Test24TestGLTracer>(
        m_FbWidth, 
        m_FbHeight, 
        m_Window,
        m_ObjModelManager,
        m_Framebuffer, 
        m_CameraController, 
        m_CurObjModelName, 
        m_EventFlags
    );
    m_Tracers["TestGL"]->Initialize();
    m_Tracers["DebugOPX"] = std::make_shared<Test24DebugOPXTracer>(
        m_Context, 
        m_Framebuffer, 
        m_CameraController, 
        m_TextureManager, 
        m_IASHandles["TopLevel"], 
        m_Materials, 
        m_BgLightColor, 
        m_EventFlags
    );
    m_Tracers["DebugOPX"]->Initialize();
    m_Tracers["PathOPX"] = std::make_shared<Test24PathOPXTracer>(
        m_Context,
        m_Framebuffer,
        m_CameraController,
        m_TextureManager,
        m_IASHandles["TopLevel"],
        m_Materials,
        m_BgLightColor,
        m_EventFlags,
        m_MaxTraceDepth
        );
    m_Tracers["PathOPX"]->Initialize();
    m_Tracers[std::string("GuidePathOPX")]=std::make_shared<Test24GuidePathOPXTracer>(
        m_Context,
        m_Framebuffer,
        m_CameraController,
        m_TextureManager,
        m_IASHandles["TopLevel"],
        m_Materials,
        m_BgLightColor,
        m_EventFlags,
        m_MaxTraceDepth
    );
    m_Tracers[std::string("GuidePathOPX")]->Initialize();
    m_Tracers[std::string("GuideNEEOPX")] = std::make_shared<Test24GuideNEEOPXTracer>(
        m_Context,
        m_Framebuffer,
        m_CameraController,
        m_TextureManager,
        m_IASHandles["TopLevel"],
        m_Materials,
        m_BgLightColor,
        m_EventFlags,
        m_MaxTraceDepth
        );
    m_Tracers[std::string("GuideNEEOPX")]->Initialize();
    m_Tracers[std::string("NEEOPX")]     = std::make_shared<Test24NEEOPXTracer>(
        m_Context,
        m_Framebuffer,
        m_CameraController,
        m_TextureManager,
        m_IASHandles["TopLevel"],
        m_Materials,
        m_BgLightColor, 
        m_EventFlags,
        m_MaxTraceDepth
    );
    m_Tracers[std::string("NEEOPX")]->Initialize();
    m_Tracers[std::string("ReSTIROPX")] = std::make_shared<Test24ReSTIROPXTracer>(
        m_Context,
        m_Framebuffer,
        m_CameraController,
        m_TextureManager,
        m_IASHandles["TopLevel"],
        m_Materials,
        m_BgLightColor,
        m_EventFlags
    );
    m_Tracers[std::string("ReSTIROPX")]->Initialize();
    m_TracePublicNames.clear();
    m_TracePublicNames.reserve(m_Tracers.size());
    for (auto& [name, tracer] : m_Tracers)
    {
        m_TracePublicNames.push_back(name);
    }

    m_CurMainTraceName = m_TracePublicNames.front();
}

void Test24Application::FreeTracers()
{
    for (auto &[name, tracer] : m_Tracers)
    {
        tracer->CleanUp();
    }
    m_Tracers.clear();

}

void Test24Application::PrepareLoop()
{
    glfwSetTime(0.0f);
    {
        double xpos, ypos;
        glfwGetCursorPos(m_Window, &xpos, &ypos);
        m_CurCursorPos[0] = xpos;
        m_CurCursorPos[1] = ypos;
    }
}

void Test24Application::RenderFrame(const std::string &name)
{
    if (m_Framebuffer->HasComponent<test::RTGLTextureFBComponent<uchar4>>(name))
    {
        auto rtComponent = m_Framebuffer->GetComponent<test::RTGLTextureFBComponent<uchar4>>(name);
        if (!rtComponent)
        {
            return;
        }
        auto &renderTexture = rtComponent->GetHandle();
        m_Renderer->draw(renderTexture.getID());
        return;
    }
    if (m_Framebuffer->HasComponent<test::RTCUGLBufferFBComponent<uchar4>>(name))
    {
        this->CopyRtFrame(name);
        auto rtComponent = m_Framebuffer->GetComponent<test::RTGLTextureFBComponent<uchar4>>("RTexture");
        if (!rtComponent)
        {
            return;
        }
        auto &renderTexture = rtComponent->GetHandle();
        m_Renderer->draw(renderTexture.getID());
    }
}

void Test24Application::Launch()
{
    m_LaunchTracerSet.insert(m_CurMainTraceName);
    for (auto& name : m_LaunchTracerSet) {
        if (name == "PathOPX")
        {
            Test24PathOPXTracer::UserData  userData = {};
            userData.samplePerLaunch = m_SamplePerLaunch;
            userData.isSync          = true;
            userData.stream          = nullptr;
            m_Tracers[name]->Launch(m_FbWidth, m_FbHeight, &userData);
            m_SamplePerAll           = userData.samplePerAll;
        }
        if (name == "NEEOPX") {
            Test24NEEOPXTracer::UserData userData = {};
            userData.samplePerLaunch    = m_SamplePerLaunch;
            userData.isSync             = true;
            userData.stream             = nullptr;
            m_Tracers[name]->Launch(m_FbWidth, m_FbHeight, &userData);
            m_SamplePerAll              = userData.samplePerAll;
        }
        if (name == "GuidePathOPX")
        {
            //なぜかSDTreeのAccumulationが進まない
            Test24GuidePathOPXTracer::UserData  userData = {};
            userData.samplePerLaunch = m_SamplePerLaunch;
            userData.samplePerAll = m_SamplePerAll;
            userData.sampleForBudget = 4096;
            userData.iterationForBuilt = 0;
            userData.isSync = true;
            userData.stream = nullptr;
            m_Tracers[name]->Launch(m_FbWidth, m_FbHeight, &userData);
            m_SamplePerAll += m_SamplePerLaunch;
            if (m_SamplePerAll >= userData.sampleForBudget) {
                m_CurMainTraceName = "PathOPX";
                m_SamplePerAll = 0;
                m_EventFlags |= TEST24_EVENT_FLAG_CHANGE_TRACE;
            }
        }
        if (name == "GuideNEEOPX")
        {
            //なぜかSDTreeのAccumulationが進まない
            Test24GuideNEEOPXTracer::UserData  userData = {};
            userData.samplePerLaunch = m_SamplePerLaunch;
            userData.samplePerAll = m_SamplePerAll;
            userData.sampleForBudget = 4096;
            userData.iterationForBuilt = 0;
            userData.isSync = true;
            userData.stream = nullptr;
            m_Tracers[name]->Launch(m_FbWidth, m_FbHeight, &userData);
            m_SamplePerAll += m_SamplePerLaunch;
            if (m_SamplePerAll >= userData.sampleForBudget) {
                m_CurMainTraceName = "PathOPX";
                m_SamplePerAll = 0;
                m_EventFlags |= TEST24_EVENT_FLAG_CHANGE_TRACE;
            }
        }
        if (name == "ReSTIROPX")
        {
            Test24ReSTIROPXTracer::UserData  userData = {};
            userData.spatialReuseRange     = 5;
            userData.iterationSpatialReuse = 2;
            userData.numCandidates         = 32;
            userData.isSync                = true;
            userData.stream                = nullptr;
            m_Tracers[name]->Launch(m_FbWidth, m_FbHeight, &userData);
        }
        if (name == "DebugOPX") {
            Test24DebugOPXTracer::UserData userData = {};
            userData.isSync            = true;
            userData.stream            = nullptr;
            m_Tracers[name]->Launch(m_FbWidth, m_FbHeight, &userData);
        }
        if (name == "TestGL") {
            m_Tracers[name]->Launch(m_FbWidth, m_FbHeight, nullptr);
        }
    }
    m_LaunchTracerSet.clear();
}

void Test24Application::BegFrame()
{
    glFlush();
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
}

void Test24Application::Render()
{
    this->RenderFrame(m_CurMainFrameName);
    this->RenderGui();
}

void Test24Application::Update()
{
    ResizeFrame();
    UpdateCamera();
    for (auto& [name, tracer] : m_Tracers)
    {
        tracer->Update();
    }
    m_EventFlags = TEST24_EVENT_FLAG_NONE;
    glfwPollEvents();
    UpdateFrameTime();
}

void Test24Application::EndFrame()
{
    glfwSwapBuffers(m_Window);
}

void Test24Application::FramebufferSizeCallback(GLFWwindow *window, int fbWidth, int fbHeight)
{
    auto *this_ptr = reinterpret_cast<Test24Application *>(glfwGetWindowUserPointer(window));
    if (this_ptr)
    {
        if (this_ptr->m_FbWidth != fbWidth || this_ptr->m_FbHeight != fbHeight)
        {
            glViewport(0, 0, fbWidth, fbHeight);
            this_ptr->m_FbWidth    = fbWidth;
            this_ptr->m_FbHeight   = fbHeight;
            this_ptr->m_EventFlags|= TEST24_EVENT_FLAG_RESIZE_FRAME;
        }
    }
}

void Test24Application::CursorPosCallback(GLFWwindow* window, double xpos, double ypos)
{

    if (!window) {
        return;
    }
    auto* app = reinterpret_cast<Test24Application*>(glfwGetWindowUserPointer(window));
    if (!app) {
        return;
    }
    auto   prvCursorPos = app->m_CurCursorPos;
    app->m_CurCursorPos = std::array<float, 2>{static_cast<float>(xpos), static_cast<float>(ypos)};
    app->m_DelCursorPos[0] = app->m_CurCursorPos[0] - prvCursorPos[0];
    app->m_DelCursorPos[1] = app->m_CurCursorPos[1] - prvCursorPos[1];
}

void Test24Application::ScrollCallback(GLFWwindow* window, double xoff, double yoff)
{
    ImGui_ImplGlfw_ScrollCallback(window, xoff, yoff);
    if (!window) {
        return;
    }
    auto* app = reinterpret_cast<Test24Application*>(glfwGetWindowUserPointer(window));
    if (!app) {
        return;
    }
    if (!ImGui::GetIO().WantCaptureMouse) 
    {
        app->m_ScrollOffsets[0] += xoff;
        app->m_ScrollOffsets[1] += yoff;
    }
}

void Test24Application::CopyRtFrame(const std::string &name)
{
    auto fbComponent = m_Framebuffer->GetComponent<test::RTCUGLBufferFBComponent<uchar4>>(name);
    auto rtComponent = m_Framebuffer->GetComponent<test::RTGLTextureFBComponent<uchar4>>("RTexture");
    if (fbComponent && rtComponent)
    {
        auto &frameBufferGL = fbComponent->GetHandle().getHandle();
        auto &renderTexture = rtComponent->GetHandle();
        renderTexture.upload((size_t)0, frameBufferGL, (size_t)0, (size_t)0, (size_t)m_FbWidth, (size_t)m_FbHeight);
    }
}

void Test24Application::CopyDgFrame(const std::string &name)
{
    auto fbComponent = m_Framebuffer->GetComponent<test::RTCUGLBufferFBComponent<uchar4>>(name);
    auto dgComponent = m_Framebuffer->GetComponent<test::RTGLTextureFBComponent<uchar4>>("DTexture");
    if (fbComponent && dgComponent)
    {
        auto &frameBufferGL = fbComponent->GetHandle().getHandle();
        auto &debugTexture = dgComponent->GetHandle();
        debugTexture.upload((size_t)0, frameBufferGL, (size_t)0, (size_t)0, (size_t)m_FbWidth, (size_t)m_FbHeight);
    }
}

void Test24Application::ResizeFrame()
{
    if ((m_EventFlags&TEST24_EVENT_FLAG_RESIZE_FRAME  )==TEST24_EVENT_FLAG_RESIZE_FRAME)
    {
        m_Framebuffer->Resize(m_FbWidth, m_FbHeight);
    }
    if ((m_EventFlags & TEST24_EVENT_FLAG_FLUSH_FRAME) == TEST24_EVENT_FLAG_FLUSH_FRAME)
    {
        m_SamplePerAll = 0;
    }
}

void Test24Application::UpdateCamera()
{
    if (m_CurFrameTime != 0.0f)
    {
        if (glfwGetKey(m_Window, GLFW_KEY_W) == GLFW_PRESS)
        {

            m_CameraController->ProcessKeyboard(rtlib::ext::CameraMovement::eForward, m_DelFrameTime);
            m_EventFlags |= TEST24_EVENT_FLAG_UPDATE_CAMERA;
        }
        if (glfwGetKey(m_Window, GLFW_KEY_S) == GLFW_PRESS)
        {
            m_CameraController->ProcessKeyboard(rtlib::ext::CameraMovement::eBackward, m_DelFrameTime);
            m_EventFlags |= TEST24_EVENT_FLAG_UPDATE_CAMERA;
        }
        if (glfwGetKey(m_Window, GLFW_KEY_A) == GLFW_PRESS)
        {
            m_CameraController->ProcessKeyboard(rtlib::ext::CameraMovement::eLeft, m_DelFrameTime);
            m_EventFlags     |= TEST24_EVENT_FLAG_UPDATE_CAMERA;
        }
        if (glfwGetKey(m_Window, GLFW_KEY_D) == GLFW_PRESS)
        {
            m_CameraController->ProcessKeyboard(rtlib::ext::CameraMovement::eRight, m_DelFrameTime);
            m_EventFlags     |= TEST24_EVENT_FLAG_UPDATE_CAMERA;
        }
        if (glfwGetKey(m_Window, GLFW_KEY_LEFT) == GLFW_PRESS)
        {
            m_CameraController->ProcessKeyboard(rtlib::ext::CameraMovement::eLeft, m_DelFrameTime);
            m_EventFlags     |= TEST24_EVENT_FLAG_UPDATE_CAMERA;
        }
        if (glfwGetKey(m_Window, GLFW_KEY_RIGHT) == GLFW_PRESS)
        {
            m_CameraController->ProcessKeyboard(rtlib::ext::CameraMovement::eRight, m_DelFrameTime);
            m_EventFlags     |= TEST24_EVENT_FLAG_UPDATE_CAMERA;
        }
        if (glfwGetKey(m_Window, GLFW_KEY_UP) == GLFW_PRESS)
        {
            m_CameraController->ProcessKeyboard(rtlib::ext::CameraMovement::eUp, m_DelFrameTime);
            m_EventFlags     |= TEST24_EVENT_FLAG_UPDATE_CAMERA;
        }
        if (glfwGetKey(m_Window, GLFW_KEY_DOWN) == GLFW_PRESS)
        {
            m_CameraController->ProcessKeyboard(rtlib::ext::CameraMovement::eDown, m_DelFrameTime);
            m_EventFlags     |= TEST24_EVENT_FLAG_UPDATE_CAMERA;
        }
        if (glfwGetMouseButton(m_Window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS)
        {
            if (!ImGui::GetIO().WantCaptureMouse)
            {
               
                m_CameraController->ProcessMouseMovement(-m_DelCursorPos[0], m_DelCursorPos[1]);
                m_EventFlags |= TEST24_EVENT_FLAG_UPDATE_CAMERA;
            }
        }
        if (m_ScrollOffsets[0] != 0.0f || m_ScrollOffsets[1] != 0.0f) {
            float yoff = m_ScrollOffsets[1];
            m_CameraController->ProcessMouseScroll(-m_ScrollOffsets[1]);
            m_EventFlags     |= TEST24_EVENT_FLAG_UPDATE_CAMERA;
        }
    }
    m_ScrollOffsets[0] = 0.0f;
    m_ScrollOffsets[1] = 0.0f;
}

void Test24Application::UpdateFrameTime()
{
    float newTime  = glfwGetTime();
    float delTime  = newTime - m_CurFrameTime;
    m_CurFrameTime = newTime;
    m_DelFrameTime = delTime;
}
