#include <Test15Config.h>
#include <cuda/RayTrace.h>
#include <RTLib/Optix.h>
#include "../include/PathTracer.h"
int main(){
    test::PathTracer tracer = {};
    RTLIB_CUDA_CHECK(cudaFree(0));
    RTLIB_OPTIX_CHECK(optixInit());
    tracer.m_OPXContext = std::make_shared<rtlib::OPXContext>(rtlib::OPXContext::Desc { 0, 0, OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL, 4 });
    auto objMeshGroup   = std::make_shared<test::ObjMeshGroup>();
    if (!objMeshGroup->Load(TEST_TEST15_DATA_PATH"/Models/CornellBox/CornellBox-Original.obj", TEST_TEST15_DATA_PATH"/Models/CornellBox/")) {
        return -1;
    }
    auto& material = objMeshGroup->GetMaterialSet();
    tracer.m_GASHandles["Sponza"] = std::make_shared<test::GASHandle>();
    {
        auto& gasHandle = tracer.m_GASHandles["Sponza"];
        gasHandle->sbtCount = 0;
        for (auto& name : objMeshGroup->GetMeshGroup()->GetUniqueNames()) {
            gasHandle->meshes.push_back(objMeshGroup->GetMeshGroup()->LoadMesh(name));
        }
        OptixAccelBuildOptions accelOptions = {};
        accelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
        accelOptions.operation  = OPTIX_BUILD_OPERATION_BUILD;
        gasHandle->Build(tracer.m_OPXContext.get(), accelOptions);
    }
    tracer.m_IASHandle = std::make_shared<test::IASHandle>();
    {
        tracer.m_IASHandle->instanceSets.resize(1);
        tracer.m_IASHandle->instanceSets[0] = std::make_shared<test::InstanceSet>();
        tracer.m_IASHandle->instanceSets[0]->instanceBuffer.cpuHandle.resize(1);
        tracer.m_IASHandle->instanceSets[0]->instanceBuffer.cpuHandle[0].traversableHandle = tracer.m_GASHandles["Sponza"]->handle;
        tracer.m_IASHandle->instanceSets[0]->instanceBuffer.cpuHandle[0].instanceId        = 0;
        tracer.m_IASHandle->instanceSets[0]->instanceBuffer.cpuHandle[0].sbtOffset         = 0;
        tracer.m_IASHandle->instanceSets[0]->instanceBuffer.cpuHandle[0].visibilityMask    = 255;
        tracer.m_IASHandle->instanceSets[0]->instanceBuffer.cpuHandle[0].flags             = OPTIX_INSTANCE_FLAG_NONE;
        {
            float transform[12] = {
                1.0f,0.0f,0.0f,0.0f,
                0.0f,1.0f,0.0f,0.0f,
                0.0f,0.0f,1.0f,0.0f
            };
            std::memcpy(tracer.m_IASHandle->instanceSets[0]->instanceBuffer.cpuHandle[0].transform, transform, sizeof(float) * 12);
        }
        tracer.m_IASHandle->instanceSets[0]->instanceBuffer.gpuHandle = rtlib::CUDABuffer<OptixInstance>(tracer.m_IASHandle->instanceSets[0]->instanceBuffer.cpuHandle);
        OptixAccelBuildOptions accelOptions = {};
        accelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
        accelOptions.operation  = OPTIX_BUILD_OPERATION_BUILD;
        tracer.m_IASHandle->Build(tracer.m_OPXContext.get(), accelOptions);
    }
    return 0;
}