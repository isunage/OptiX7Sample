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
        using  GASHandlePtr = std::shared_ptr<GASHandle>;
        enum class InstanceType 
        {
            GAS = 0,
            IAS
        };
        struct IASHandle;
        using  IASHandlePtr = std::shared_ptr<IASHandle>;
        struct Instance {
            InstanceType               type          = InstanceType::GAS;
            OptixInstance              instance      = {};
            std::shared_ptr<GASHandle> baseGASHandle = {};
            std::shared_ptr<IASHandle> baseIASHandle = {};
        public:
            void Init(const GASHandlePtr& gasHandle);
            void Init(const IASHandlePtr& iasHandle);
            auto GetSbtOffset()const noexcept -> size_t;
            void SetSbtOffset(size_t offset)noexcept;
            auto GetSbtCount() const noexcept -> size_t;
        };
        struct InstanceIndex
        {
            InstanceType type;
            size_t       index;
        };
        struct InstanceSet {
            CUDAUploadBuffer<OptixInstance>              instanceBuffer  = {};
            std::vector<InstanceIndex>                   instanceIndices = {};
            std::vector<std::shared_ptr<GASHandle>>      baseGASHandles  = {};
            std::vector<std::shared_ptr<IASHandle>>      baseIASHandles  = {};
        public:
            void Upload()noexcept;
            void SetInstance(const Instance& instance)noexcept;
            auto GetInstance(size_t i)const noexcept -> Instance;
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