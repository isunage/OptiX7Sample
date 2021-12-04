#ifndef RTLIB_EXT_TRAVERSAL_HANDLE2_H
#define RTLIB_EXT_TRAVERSAL_HANDLE2_H
#include <RTLib/core/Optix.h>
#include <unordered_map>
#include <variant>
namespace rtlib {
    namespace ext {
        class  GeometryAccelerationStructure;
        class  BuildInputTriangles
        {
        private:
            using GasWeakPtr = std::weak_ptr<GeometryAccelerationStructure>;
            BuildInputTriangles()noexcept;
        public:
            using UpdateCallback = void(*)(BuildInputTriangles*);
            static auto New()noexcept -> std::shared_ptr< BuildInputTriangles>;
            ~BuildInputTriangles()noexcept;

            bool CheckDirty()const noexcept;
            void Update()noexcept;
            auto GetHandle()const -> const OptixBuildInput&;

            void SetVertexBuffer(void* cudaGpuAddress, size_t sizeInBytes, size_t strideInBytes, OptixVertexFormat format)noexcept;
            void SetVertexBuffer(float2* cudaGpuAddress, size_t numVertices)noexcept;
            void SetVertexBuffer(float3* cudaGpuAddress, size_t numVertices)noexcept;
            void SetVertexBufferGpuAddress(void* cudaGpuAddress)noexcept;
            auto GetVertexBufferGpuAddress()const noexcept-> void*;
            void SetVertexFormat(OptixVertexFormat format)noexcept;
            auto GetVertexFormat()const noexcept->OptixVertexFormat;
            void SetVertexStrideInBytes(size_t strideInBytes)noexcept;
            auto GetVertexStrideInBytes()const noexcept-> size_t;
            void SetNumVertices(size_t numVertices)noexcept;
            auto GetNumVertices()const noexcept->size_t;

            void SetIndexBuffer(void* cudaGpuAddress, size_t sizeInBytes, size_t strideInBytes, OptixIndicesFormat format)noexcept;
            void SetIndexBuffer(ushort3* cudaGpuAddress, size_t numIndices)noexcept;
            void SetIndexBuffer(uint3* cudaGpuAddress, size_t numIndices)noexcept;
            void SetIndexBufferGpuAddress(void* cudaGpuAddress)noexcept;
            auto GetIndexBufferGpuAddress()const noexcept-> void*;
            void SetIndicesFormat(OptixIndicesFormat format)noexcept;
            auto GetIndicesFormat()const noexcept->OptixIndicesFormat;
            void SetIndexStrideInBytes(size_t strideInBytes)noexcept;
            auto GetIndexStrideInBytes()const noexcept->size_t;
            void SetNumIndexTriplets(size_t numIndices)noexcept;
            auto GetNumIndexTriplets()const noexcept->size_t;
            void SetPrimitiveIndexOffset(unsigned int offset)noexcept;
            auto GetPrimitiveIndexOffset()const noexcept->unsigned int;

            void SetSbtOffsetBuffer(void* cudaGpuAddress, size_t sizeInBytes, size_t strideInBytes, size_t formatSize)noexcept;
            void SetSbtOffsetBuffer(uint8_t* cudaGpuAddress, size_t numSbtOffsets)noexcept;
            void SetSbtOffsetBuffer(uint16_t* cudaGpuAddress, size_t numSbtOffsets)noexcept;
            void SetSbtOffsetBuffer(uint32_t* cudaGpuAddress, size_t numSbtOffsets)noexcept;
            void SetSbtOffsetBufferGpuAddress(void* cudaGpuAddress)noexcept;
            auto GetSbtOffsetBufferGpuAddress()const noexcept-> void*;
            void SetSbtOffsetSizeInBytes(size_t sizeInBytes)noexcept;
            auto GetSbtOffsetSizeInBytes()const noexcept->size_t;
            void SetSbtOffsetStrideInBytes(size_t strideInBytes)noexcept;
            auto GetSbtOffsetStrideInBytes()const noexcept->size_t;
            void SetNumSbtOffsets(size_t numIndices)noexcept;
            auto GetNumSbtOffsets()const noexcept->size_t;
            void SetSbtRecordGeometryFlags(size_t idx, unsigned int flags)noexcept;
            auto GetSbtRecordGeometryFlags(size_t idx)const noexcept-> unsigned int;
            void SetNumSbtRecordBases(size_t numSbtRecordBases)noexcept;
            auto GetNumSbtRecordBases()const noexcept -> size_t;

            void SetPreTransformsGpuAddress(void* preTransforms);
            auto GetPreTransformsGpuAddress()const -> void*;

            auto SetUserPointer(void* userPointer)noexcept;
            auto GetUserPointer()const noexcept -> void*;
        private:

            friend class GeometryAccelerationStructure;
            void AddGasReferenceInternal(const std::shared_ptr<GeometryAccelerationStructure>& gas)noexcept;

        private:
            OptixBuildInput                              m_BuildInput              = {};
            std::vector<unsigned int>                    m_SbtRecordGeometryFlags;
            CUdeviceptr                                  m_VertexBufferGpuAddress;
            size_t                                       m_NumSbtRecordBases       = 0;
            bool                                         m_Dirty                   = false;
            std::unordered_map<std::string, GasWeakPtr>  m_GasReferences           = {};
            void*                                        m_UserPointer             = nullptr;
            UpdateCallback                               m_UpdateCallback          = nullptr;
        };
        using  BuildInputTrianglesPtr = std::shared_ptr<BuildInputTriangles>;
        class  BuildInputCustomPrimitives
        {
        private:
            using GasWeakPtr = std::weak_ptr<GeometryAccelerationStructure>;
            BuildInputCustomPrimitives()noexcept;
        public:
            using UpdateCallback = void(*)(BuildInputCustomPrimitives*);
            static auto New()noexcept -> std::shared_ptr< BuildInputCustomPrimitives>;
            ~BuildInputCustomPrimitives()noexcept;

            bool CheckDirty()const noexcept;
            void Update()noexcept;
            auto GetHandle()const -> const OptixBuildInput&;

            void SetAabbBuffer(void* cudaGpuAddress, size_t sizeInBytes, size_t strideInBytes)noexcept;
            void SetAabbBuffer(OptixAabb* cudaGpuAddress, size_t numPrimitives);
            void SetAabbBufferGpuAddress(void* cudaGpuAddress)noexcept;
            auto GetAabbBufferGpuAddress()const noexcept -> void*;
            void SetAabbStrideInBytes(size_t strideInBytes) noexcept;
            auto GetAabbStrideInBytes()const noexcept->size_t;

            void SetNumPrimitives(size_t numPrimitives);
            auto GetNumPrimitives()const noexcept -> size_t;

            void SetPrimitiveIndexOffset(unsigned int offset)noexcept;
            auto GetPrimitiveIndexOffset()const noexcept->unsigned int;

            void SetSbtOffsetBuffer(void* cudaGpuAddress, size_t sizeInBytes, size_t strideInBytes, size_t formatSize)noexcept;
            void SetSbtOffsetBuffer(uint8_t* cudaGpuAddress, size_t numSbtOffsets)noexcept;
            void SetSbtOffsetBuffer(uint16_t* cudaGpuAddress, size_t numSbtOffsets)noexcept;
            void SetSbtOffsetBuffer(uint32_t* cudaGpuAddress, size_t numSbtOffsets)noexcept;
            void SetSbtOffsetBufferGpuAddress(void* cudaGpuAddress)noexcept;
            auto GetSbtOffsetBufferGpuAddress()const noexcept-> void*;
            void SetSbtOffsetSizeInBytes(size_t sizeInBytes)noexcept;
            auto GetSbtOffsetSizeInBytes()const noexcept->size_t;
            void SetSbtOffsetStrideInBytes(size_t strideInBytes)noexcept;
            auto GetSbtOffsetStrideInBytes()const noexcept->size_t;
            void SetNumSbtOffsets(size_t numIndices)noexcept;
            auto GetNumSbtOffsets()const noexcept->size_t;
            void SetSbtRecordGeometryFlags(size_t idx, unsigned int flags)noexcept;
            auto GetSbtRecordGeometryFlags(size_t idx)const noexcept-> unsigned int;
            void SetNumSbtRecordBases(size_t numSbtRecordBases)noexcept;
            auto GetNumSbtRecordBases()const noexcept -> size_t;

            auto SetUserPointer(void* userPointer)noexcept;
            auto GetUserPointer()const noexcept -> void*;
        private:
            friend class GeometryAccelerationStructure;
            void AddGasReferenceInternal(const std::shared_ptr<GeometryAccelerationStructure>& gas)noexcept;
        private:
            OptixBuildInput                              m_BuildInput             = {};
            std::vector<unsigned int>                    m_SbtRecordGeometryFlags;
            CUdeviceptr                                  m_AabbBufferGpuAddress;
            size_t                                       m_NumSbtRecordBases      = 0;
            bool                                         m_Dirty                  = false;
            std::unordered_map<std::string, GasWeakPtr>  m_GasReferences          = {};
            void*                                        m_UserPointer            = nullptr;
            UpdateCallback                               m_UpdateCallback         = nullptr;
        };
        using  BuildInputCustomPrimitivesPtr = std::shared_ptr<BuildInputCustomPrimitives>;
        class  GeometryAccelerationStructure
        {
        private:
            GeometryAccelerationStructure(const std::string& name)noexcept;
        public:
            using UpdateCallback = void (*)(GeometryAccelerationStructure*);
            using BuildInputGeometryPtr = std::variant<BuildInputTrianglesPtr, BuildInputCustomPrimitivesPtr>;
            static auto New(const std::string& name)noexcept -> std::shared_ptr<GeometryAccelerationStructure>;
            ~GeometryAccelerationStructure()noexcept;

            bool CheckDirty()const noexcept;
            auto GetName()const noexcept -> std::string;

            void Build(const rtlib::OPXContext* context, const OptixAccelBuildOptions& accelOptions);
            void Update()noexcept;
            auto GetHandle()const noexcept -> OptixTraversableHandle;

            auto GetBuildInput(size_t idx)const -> const BuildInputGeometryPtr&;
            auto EnumerateBuildInputs()const noexcept -> const std::vector<BuildInputGeometryPtr>&;
            auto GetNumSbtRecordBases()const noexcept -> size_t;

            void SetUserPointer(void* userPointer)noexcept;
            auto GetUserPointer()const noexcept -> void*;

            void SetUpdateCallback(UpdateCallback updateCallback);

            static void AddBuildInput(const std::shared_ptr<GeometryAccelerationStructure>& gas, const BuildInputTrianglesPtr& triangles);
            static void AddBuildInput(const std::shared_ptr<GeometryAccelerationStructure>& gas, const BuildInputCustomPrimitivesPtr& customPrimitives);
        private:
            void AddBuildInputInternal(const BuildInputTrianglesPtr       & triangles       )noexcept;
            void AddBuildInputInternal(const BuildInputCustomPrimitivesPtr& customPrimitives)noexcept;
        private:
            std::string                        m_Name              = "";
            OptixTraversableHandle             m_Handle            = 0;
            rtlib::CUDABuffer<void>            m_Buffer            = {};
            std::vector<BuildInputGeometryPtr> m_BuildInputs       = {};
            size_t                             m_NumSbtRecordBases = 0;
            void*                              m_UserPonter        = nullptr;
            bool                               m_Dirty             = false;
            UpdateCallback                     m_UpdateCallback    = nullptr;
        };
        using  GeometryAccelerationStructurePtr = std::shared_ptr< GeometryAccelerationStructure>;
        class  GASInstance
        {
        private:
            GASInstance(const GeometryAccelerationStructurePtr&)noexcept;
        public:
            using UpdateCallback = void (*)(GASInstance*);
            ~GASInstance()noexcept {}

            bool CheckDirty()const noexcept;
            void Update()noexcept;
            auto GetHandle()const noexcept -> const OptixInstance&;

            void SetInstanceId(unsigned int id)noexcept;
            auto GetInstanceId()const noexcept -> unsigned int;

            void SetNumRayTypes(size_t numRayTypes)noexcept;
            auto GetNumRayTypes()const noexcept-> size_t;

            void SetUserPointer(void* userPointer)noexcept;
            auto GetUserPointer()const noexcept -> void*;

            void SetUpdateCallback(UpdateCallback updateCallback);
        private:
            std::string                        m_Name              = "";
            OptixInstance                      m_Handle            = {};
            GeometryAccelerationStructurePtr   m_Gas               = nullptr;
            size_t                             m_SbtRecordOffset   = 0;
            size_t                             m_NumRayTypes       = 1;
            void*                              m_UserPonter        = nullptr;
            bool                               m_Dirty             = false;
            UpdateCallback                     m_UpdateCallback    = nullptr;
        };
        class IASInstance;
        class InstanceGroup
        {

        };
    }
}
#endif