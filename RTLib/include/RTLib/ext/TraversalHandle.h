#ifndef RTLIB_EXT_TRAVERSAL_HANDLE_H
#define RTLIB_EXT_TRAVERSAL_HANDLE_H
#include <variant>
#include <RTLib/core/Optix.h>
#include <RTLib/ext/Mesh.h>
namespace rtlib{
    namespace ext {
        enum class InstanceType
        {
            GAS,
            IAS
        };
        struct Instance;
        class  GASHandle {
            OptixTraversableHandle   handle = {};
            rtlib::CUDABuffer<void>  buffer = {};
            std::vector<MeshPtr>     meshes = {};
            size_t                   sbtCount = 0;
        public:
            void Build(const rtlib::OPXContext* context, const OptixAccelBuildOptions& accelOptions);
            //AddMesh
            void AddMesh(const MeshPtr& mesh)noexcept;
            //GetMesh
            auto GetMesh(size_t idx)const noexcept -> MeshPtr
            {
                return meshes[idx];
            }
            auto GetMeshes()const noexcept -> const std::vector<MeshPtr>& {
                return meshes;
            }
            auto GetMeshes() noexcept -> std::vector<MeshPtr>& {
                return meshes;
            }
            //GetHandle
            auto GetHandle()const noexcept -> OptixTraversableHandle
            {
                return handle;
            }
            //GetSbtCount
            auto GetSbtCount()const noexcept -> size_t;
        };
        using  GASHandlePtr   = std::shared_ptr<GASHandle>;
        struct IASHandle;
        using  IASHandlePtr   = std::shared_ptr<IASHandle>;
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
        class  IASHandle{
            OptixTraversableHandle      handle       = {};
            rtlib::CUDABuffer<void>     buffer       = {};
            std::vector<InstanceSetPtr> instanceSets = {};
            size_t                      sbtCount     =  0;
        public:
            void Build(const rtlib::OPXContext* context, const OptixAccelBuildOptions& accelOptions);
            //Add InstanceSet
            void AddInstanceSet(const InstanceSetPtr& instanceSet)noexcept;
            //Get InstanceSet
            auto GetInstanceSet(size_t idx)const noexcept -> const rtlib::ext::InstanceSetPtr&;
            auto GetInstanceSets()      noexcept ->       std::vector<rtlib::ext::InstanceSetPtr >&;
            auto GetInstanceSets()const noexcept -> const std::vector<rtlib::ext::InstanceSetPtr >&;
            //GetHandle
            auto GetHandle()const noexcept -> OptixTraversableHandle
            {
                return handle;
            }
            //Get SbtCount
            auto GetSbtCount()const noexcept -> size_t;
        };
        using  IASHandlePtr   = std::shared_ptr<IASHandle>;
    }
}
#endif