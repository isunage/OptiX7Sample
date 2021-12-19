#ifndef TEST_TEST24_RESTIR_OPX_TRACER_H
#define TEST_TEST24_RESTIR_OPX_TRACER_H
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
#include <Test24Event.h>
#include <memory>
// ReSTIRTracer
class Test24ReSTIROPXTracer : public test::RTTracer
{
private:
    using ContextPtr           = std::shared_ptr<test::RTContext>;
    using FramebufferSharedPtr = std::shared_ptr<test::RTFramebuffer>;
    using FramebufferUniquePtr = std::unique_ptr<test::RTFramebuffer>;
    using CameraControllerPtr  = std::shared_ptr<rtlib::ext::CameraController>;
    using TextureAssetManager  = std::shared_ptr<test::RTTextureAssetManager>;
public:
    struct UserData
    {
        unsigned int numCandidates;
        unsigned int iterationSpatialReuse;
        unsigned int spatialReuseRange;
        bool isSync;
        CUstream stream;
    };

public:
    Test24ReSTIROPXTracer(ContextPtr Context,
        FramebufferSharedPtr Framebuffer,
        CameraControllerPtr CameraController,
        TextureAssetManager TextureManager,
        rtlib::ext::IASHandlePtr TopLevelAS,
        const std::vector<rtlib::ext::VariableMap>& Materials,
        const float3& BgLightColor,
        const unsigned int& eventFlags);
    // RTTracer 
    virtual void Initialize() override;
    virtual void Launch(int width, int height, void* userData) override;
    virtual void CleanUp() override;
    virtual void Update() override;
    virtual ~Test24ReSTIROPXTracer();

private:
    void InitFrameResources();
    void InitPipeline();
    void InitShaderBindingTable();
    void InitLight();
    void InitLaunchParams();

    void FreeFrameResources();
    void FreePipeline();
    void FreeShaderBindingTable();
    void FreeLight();
    void FreeLaunchParams();
private:
    struct Impl;
    std::unique_ptr<Impl> m_Impl;
};
#endif