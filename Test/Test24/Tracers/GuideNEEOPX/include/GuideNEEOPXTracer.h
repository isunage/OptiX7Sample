#ifndef TEST_TEST24_GUIDE_NEE_OPX_TRACER_H
#define TEST_TEST24_GUIDE_NEE_OPX_TRACER_H
#include <TestLib/RTTracer.h>
#include <TestLib/RTAssets.h>
#include <TestLib/RTFrameBuffer.h>
#include <TestLib/RTContext.h>
#include <RTLib/core/Optix.h>
#include <RTLib/core/CUDA.h>
#include <RTLib/math/VectorFunction.h>
#include <RTLib/ext/VariableMap.h>
#include <RTLib/ext/Camera.h>
#include <RTLib/ext/TraversalHandle.h>
#include <RTLib/ext/Resources.h>
#include <RTLib/ext/Resources/GL.h>
#include <RTLib/ext/Resources/CUDA.h>
#include <Test24GuideNEEOPXConfig.h>
#include <Test24Event.h>
#include <memory>
//SimpleTracer
// GuideTracer
class Test24GuideNEEOPXTracer : public test::RTTracer
{
public:
    struct UserData
    {
        unsigned int samplePerAll;
        unsigned int samplePerLaunch;
        unsigned int sampleForBudget;
        unsigned int iterationForBuilt;
        bool         isSync;
        CUstream     stream;
    };
private:
    using ContextPtr = std::shared_ptr<test::RTContext>;
    using FramebufferPtr = std::shared_ptr<test::RTFramebuffer>;
    using CameraControllerPtr = std::shared_ptr<rtlib::ext::CameraController>;
    using TextureAssetManager = std::shared_ptr<test::RTTextureAssetManager>;
    using Pipeline = rtlib::OPXPipeline;
    using ModuleMap = std::unordered_map<std::string, rtlib::OPXModule>;
    using RGProgramGroupMap = std::unordered_map<std::string, rtlib::OPXRaygenPG>;
    using MSProgramGroupMap = std::unordered_map<std::string, rtlib::OPXMissPG>;
    using HGProgramGroupMap = std::unordered_map<std::string, rtlib::OPXHitgroupPG>;
public:
    Test24GuideNEEOPXTracer(
        ContextPtr context,
        FramebufferPtr framebuffer,
        CameraControllerPtr cameraController,
        TextureAssetManager textureManager,
        rtlib::ext::IASHandlePtr topLevelAS,
        const std::vector<rtlib::ext::VariableMap>& materials,
        const float3& bgLightColor,
        const unsigned int& eventFlags,
        const unsigned int& maxTraceDepth);
    // RTTracer
    virtual void Initialize() override;
    virtual void Launch(int width, int height, void* pUserData) override;
    virtual void CleanUp() override;
    virtual void Update() override;
    virtual bool ShouldLock()const noexcept;
    virtual ~Test24GuideNEEOPXTracer();

private:
    void InitLight();
    void FreeLight();
    void InitSTree();
    void FreeSTree();
    void InitSdTree();
    void FreeSdTree();
    void InitFrameResources();
    void FreeFrameResources();
    void InitPipeline();
    void FreePipeline();
    void InitShaderBindingTable();
    void FreeShaderBindingTable();
    void InitLaunchParams();
    void FreeLaunchParams();
    void OnLaunchBegin(int width, int height, UserData* pUserData);
    void OnLaunchExecute(int width, int height, UserData* pUserData);
    void OnLaunchEnd(int width, int height, UserData* pUserData);
private:
    struct Impl;
    std::unique_ptr<Impl> m_Impl;
};
#endif