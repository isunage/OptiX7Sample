#ifndef RTLIB_EXT_TRAVERSAL_HANDLE_H
#define RTLIB_EXT_TRAVERSAL_HANDLE_H
#include "../Optix.h"
#include "Mesh.h"
namespace rtlib{
    namespace ext {
        struct GASHandle {
            OptixTraversableHandle   handle   = {};
            rtlib::CUDABuffer<void>  buffer   = {};
            std::vector<MeshPtr>     meshes   = {};
            size_t                   sbtCount = 0;
        public:
            void Build(const rtlib::OPXContext* context, const OptixAccelBuildOptions& accelOptions);
        };
        struct Instance {
            OptixInstance              instance      = {};
            std::shared_ptr<GASHandle> baseGASHandle = {};
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
        using  IASHandlePtr   = std::shared_ptr<IASHandle>;
    }
}
#endif