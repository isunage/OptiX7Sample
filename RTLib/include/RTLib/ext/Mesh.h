#ifndef RTLIB_EXT_MESH_MESH_H
#define RTLIB_EXT_MESH_MESH_H
#include "../CUDA.h"
#include "../VectorFunction.h"
#include "Resources.h"
#include "VariableMap.h"
#include <string>
#include <vector>
#include <array>
#include <unordered_map>
#include <memory>
namespace rtlib{
    namespace ext {
        struct MeshSharedResource {
            std::string          name         = {};
            CustomBuffer<float3> vertexBuffer = {};
            CustomBuffer<float3> normalBuffer = {};
            CustomBuffer<float2> texCrdBuffer = {};
            VariableMap          variables    = {};
            static auto New() ->std::shared_ptr<MeshSharedResource> {
                return std::shared_ptr<MeshSharedResource>(new MeshSharedResource());
            }
        };
        using  MeshSharedResourcePtr = std::shared_ptr<MeshSharedResource>;
        struct MeshUniqueResource {
            std::string            name         = {};
            std::vector<uint32_t>  materials    = {};
            CustomBuffer<uint3>    triIndBuffer = {};
            CustomBuffer<uint32_t> matIndBuffer = {};
            VariableMap            variables    = {};
            static auto New() ->std::shared_ptr<MeshUniqueResource> {
                return std::shared_ptr<MeshUniqueResource>(new MeshUniqueResource());
            }
        };
        using  MeshUniqueResourcePtr = std::shared_ptr<MeshUniqueResource>;
        class  MeshGroup;
        class  Mesh{
        public:
            Mesh()noexcept {}
            void SetSharedResource(const  MeshSharedResourcePtr& res)noexcept;
            auto GetSharedResource()const noexcept -> MeshSharedResourcePtr;
            void SetUniqueResource(const  MeshUniqueResourcePtr& res)noexcept;
            void SetUniqueResource(const std::string& name, const MeshUniqueResourcePtr& res)noexcept;
            auto GetUniqueResource()const->MeshUniqueResourcePtr;
            static auto New()->std::shared_ptr<Mesh> {
                return std::shared_ptr<Mesh>(new Mesh());
            }
        private:
            friend class MeshGroup;
            std::string              m_Name           = {};
            MeshSharedResourcePtr    m_SharedResource = {};
            MeshUniqueResourcePtr    m_UniqueResource = {};
        };
        using  MeshPtr      = std::shared_ptr<Mesh>;
        class  MeshGroup {
        public:
            MeshGroup()noexcept {}
            void SetSharedResource(const MeshSharedResourcePtr& res)noexcept;
            auto GetSharedResource()const noexcept -> MeshSharedResourcePtr;
            void SetUniqueResource(const std::string& name, const MeshUniqueResourcePtr& res)noexcept;
            auto GetUniqueResource(const std::string& name) const->MeshUniqueResourcePtr;
            auto GetUniqueResources()const noexcept -> const std::unordered_map<std::string, MeshUniqueResourcePtr>&;
            auto GetUniqueNames()const noexcept -> std::vector<std::string>;
            auto LoadMesh(const std::string& name)const->MeshPtr;
            bool RemoveMesh(const std::string& name);
            static auto New()->std::shared_ptr<MeshGroup> {
                return std::shared_ptr<MeshGroup>(new MeshGroup());
            }
        private:
            using MeshUniqueResourcePtrMap = std::unordered_map<std::string, MeshUniqueResourcePtr>;
            std::string              m_Name            = {};
            MeshSharedResourcePtr    m_SharedResource  = {};
            MeshUniqueResourcePtrMap m_UniqueResources = {};
        };
        using  MeshGroupPtr = std::shared_ptr<MeshGroup>;
    }
}
#endif