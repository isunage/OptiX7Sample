#ifndef RTLIB_EXT_TRAVERSAL_HANDLE_H
#define RTLIB_EXT_TRAVERSAL_HANDLE_H
#include "../Optix.h"
#include "Mesh.h"
namespace rtlib{
    namespace ext {
        struct Instance;
        struct GASHandle {
            OptixTraversableHandle   handle   = {};
            rtlib::CUDABuffer<void>  buffer   = {};
            std::vector<MeshPtr>     meshes   = {};
            size_t                   sbtCount = 0;
        public:
            void Build(const rtlib::OPXContext* context, const OptixAccelBuildOptions& accelOptions);
        };
        using GASHandlePtr = std::shared_ptr<GASHandle>;
        struct Instance {
            OptixInstance              instance      = {};
            std::shared_ptr<GASHandle> baseGASHandle = {};
        public:
            void Init(const GASHandlePtr& gasHandle){
                instance.traversableHandle = gasHandle->handle;
                instance.instanceId        = 0;
                instance.sbtOffset         = 0;
                instance.visibilityMask    = 255;
                instance.flags             = OPTIX_INSTANCE_FLAG_NONE;
                float transform[12] = {
                    1.0f,0.0f,0.0f,0.0f,
                    0.0f,1.0f,0.0f,0.0f,
                    0.0f,0.0f,1.0f,0.0f
                };
                std::memcpy(instance.transform, transform, sizeof(float) * 12);
                baseGASHandle = gasHandle;
            }
        };
        struct InstanceSet {
            CUDAUploadBuffer<OptixInstance>         instanceBuffer = {};
            std::vector<std::shared_ptr<GASHandle>> baseGASHandles = {};
        public:
            void SetInstance(const Instance& instance)noexcept;
        };
        using  InstanceSetPtr = std::shared_ptr<InstanceSet>;
        struct IASHandle {
            OptixTraversableHandle      handle       = {};
            rtlib::CUDABuffer<void>     buffer       = {};
            std::vector<InstanceSetPtr> instanceSets = {};
            size_t                      sbtCount     =  0;
        public:
            void Build(const rtlib::OPXContext* context, const OptixAccelBuildOptions& accelOptions);
        };
        using  IASHandlePtr = std::shared_ptr<IASHandle>;
    }
}
#endif