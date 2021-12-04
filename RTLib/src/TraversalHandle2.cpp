#include <RTLib/ext/TraversalHandle2.h>
rtlib::ext::BuildInputTriangles::BuildInputTriangles() noexcept
{
    this->m_BuildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    this->m_BuildInput.triangleArray = {};
    this->m_BuildInput.triangleArray.vertexBuffers = &m_VertexBufferGpuAddress;
    m_VertexBufferGpuAddress = 0;
}

bool rtlib::ext::BuildInputTriangles::CheckDirty() const noexcept
{
    return m_Dirty;
}

void rtlib::ext::BuildInputTriangles::Update() noexcept
{
    if (m_UpdateCallback) {
        UpdateCallback(this);
    }
}

auto rtlib::ext::BuildInputTriangles::GetHandle() const -> const OptixBuildInput&
{
    // TODO: return �X�e�[�g�����g�������ɑ}�����܂�
    return m_BuildInput;
}

void rtlib::ext::BuildInputTriangles::SetVertexBuffer(void* cudaGpuAddress, size_t sizeInBytes, size_t strideInBytes, OptixVertexFormat format)noexcept
{
    SetVertexBufferGpuAddress(cudaGpuAddress);
    SetVertexFormat(format);
    SetVertexStrideInBytes(strideInBytes);
    SetNumVertices(sizeInBytes / strideInBytes);
}

void rtlib::ext::BuildInputTriangles::SetVertexBuffer(float2* cudaGpuAddress, size_t numVertices)noexcept
{
    SetVertexBufferGpuAddress(static_cast<void*>(cudaGpuAddress));
    SetVertexFormat(OPTIX_VERTEX_FORMAT_FLOAT2);
    SetVertexStrideInBytes(sizeof(float2));
    SetNumVertices(numVertices);
}

void rtlib::ext::BuildInputTriangles::SetVertexBuffer(float3* cudaGpuAddress, size_t numVertices)noexcept
{
    SetVertexBufferGpuAddress(static_cast<void*>(cudaGpuAddress));
    SetVertexFormat(OPTIX_VERTEX_FORMAT_FLOAT3);
    SetVertexStrideInBytes(sizeof(float3));
    SetNumVertices(numVertices);
}

void rtlib::ext::BuildInputTriangles::SetVertexBufferGpuAddress(void* cudaGpuAddress)noexcept
{
    m_VertexBufferGpuAddress = reinterpret_cast<CUdeviceptr>(cudaGpuAddress);
}

auto rtlib::ext::BuildInputTriangles::GetVertexBufferGpuAddress() const noexcept -> void*
{
    return reinterpret_cast<void*>(m_VertexBufferGpuAddress);
}

void rtlib::ext::BuildInputTriangles::SetVertexFormat(OptixVertexFormat format)noexcept
{
    m_BuildInput.triangleArray.vertexFormat = format;
}

auto rtlib::ext::BuildInputTriangles::GetVertexFormat() const noexcept -> OptixVertexFormat
{
    return m_BuildInput.triangleArray.vertexFormat;
}

void rtlib::ext::BuildInputTriangles::SetVertexStrideInBytes(size_t strideInBytes)noexcept
{
    m_BuildInput.triangleArray.vertexStrideInBytes = strideInBytes;
}

auto rtlib::ext::BuildInputTriangles::GetVertexStrideInBytes() const noexcept-> size_t
{
    return m_BuildInput.triangleArray.vertexStrideInBytes;
}

void rtlib::ext::BuildInputTriangles::SetNumVertices(size_t numVertices)noexcept
{
    m_BuildInput.triangleArray.numVertices = numVertices;
}

auto rtlib::ext::BuildInputTriangles::GetNumVertices() const noexcept-> size_t
{
    return m_BuildInput.triangleArray.numVertices;
}

void rtlib::ext::BuildInputTriangles::SetIndexBuffer(void* cudaGpuAddress, size_t sizeInBytes, size_t strideInBytes, OptixIndicesFormat format)noexcept
{
    SetIndexBufferGpuAddress(cudaGpuAddress);
    SetIndicesFormat(format);
    SetIndexStrideInBytes(strideInBytes);
    SetNumIndexTriplets(sizeInBytes / strideInBytes);
}

void rtlib::ext::BuildInputTriangles::SetIndexBuffer(ushort3* cudaGpuAddress, size_t numIndices)noexcept
{
    SetIndexBufferGpuAddress(reinterpret_cast<void*>(cudaGpuAddress));
    SetIndicesFormat(OPTIX_INDICES_FORMAT_UNSIGNED_SHORT3);
    SetIndexStrideInBytes(sizeof(ushort3));
    SetNumIndexTriplets(numIndices);
}

void rtlib::ext::BuildInputTriangles::SetIndexBuffer(uint3* cudaGpuAddress, size_t numIndices)noexcept
{
    SetIndexBufferGpuAddress(reinterpret_cast<void*>(cudaGpuAddress));
    SetIndicesFormat(OPTIX_INDICES_FORMAT_UNSIGNED_INT3);
    SetIndexStrideInBytes(sizeof(uint3));
    SetNumIndexTriplets(numIndices);
}

void rtlib::ext::BuildInputTriangles::SetIndexBufferGpuAddress(void* cudaGpuAddress)noexcept
{
    m_BuildInput.triangleArray.indexBuffer = reinterpret_cast<CUdeviceptr>(cudaGpuAddress);
}

auto rtlib::ext::BuildInputTriangles::GetIndexBufferGpuAddress() const noexcept-> void*
{
    return reinterpret_cast<void*>(m_BuildInput.triangleArray.indexBuffer);
}

void rtlib::ext::BuildInputTriangles::SetIndicesFormat(OptixIndicesFormat format)noexcept
{
    m_BuildInput.triangleArray.indexFormat = format;
}

auto rtlib::ext::BuildInputTriangles::GetIndicesFormat() const noexcept-> OptixIndicesFormat
{
    return m_BuildInput.triangleArray.indexFormat;
}

void rtlib::ext::BuildInputTriangles::SetIndexStrideInBytes(size_t strideInBytes)noexcept {
    m_BuildInput.triangleArray.indexStrideInBytes = strideInBytes;
}

auto rtlib::ext::BuildInputTriangles::GetIndexStrideInBytes() const noexcept-> size_t
{
    return m_BuildInput.triangleArray.indexStrideInBytes;
}

void rtlib::ext::BuildInputTriangles::SetNumIndexTriplets(size_t numIndices)noexcept
{
    m_BuildInput.triangleArray.numIndexTriplets = numIndices;
}

auto rtlib::ext::BuildInputTriangles::GetNumIndexTriplets() const noexcept -> size_t
{
    return m_BuildInput.triangleArray.numIndexTriplets;
}

void rtlib::ext::BuildInputTriangles::SetPrimitiveIndexOffset(unsigned int  offset) noexcept
{
    m_BuildInput.triangleArray.primitiveIndexOffset = offset;
}

auto rtlib::ext::BuildInputTriangles::GetPrimitiveIndexOffset() const noexcept -> unsigned int
{
    return m_BuildInput.triangleArray.primitiveIndexOffset;
}

void rtlib::ext::BuildInputTriangles::SetSbtOffsetBuffer(void* cudaGpuAddress, size_t sizeInBytes, size_t strideInBytes, size_t formatSize)noexcept
{

    SetSbtOffsetBufferGpuAddress(cudaGpuAddress);
    SetSbtOffsetSizeInBytes(formatSize);
    SetSbtOffsetStrideInBytes(strideInBytes);
    SetNumSbtOffsets(sizeInBytes / strideInBytes);
}

void rtlib::ext::BuildInputTriangles::SetSbtOffsetBuffer(uint8_t* cudaGpuAddress, size_t numSbtOffsets)noexcept
{
    SetSbtOffsetBufferGpuAddress(reinterpret_cast<void*>(cudaGpuAddress));
    SetSbtOffsetSizeInBytes(sizeof(uint8_t));
    SetSbtOffsetStrideInBytes(sizeof(uint8_t));
    SetNumSbtOffsets(numSbtOffsets);
}

void rtlib::ext::BuildInputTriangles::SetSbtOffsetBuffer(uint16_t* cudaGpuAddress, size_t numSbtOffsets)noexcept
{
    SetSbtOffsetBufferGpuAddress(reinterpret_cast<void*>(cudaGpuAddress));
    SetSbtOffsetSizeInBytes(sizeof(uint16_t));
    SetSbtOffsetStrideInBytes(sizeof(uint16_t));
    SetNumSbtOffsets(numSbtOffsets);
}

void rtlib::ext::BuildInputTriangles::SetSbtOffsetBuffer(uint32_t* cudaGpuAddress, size_t numSbtOffsets)noexcept
{
    SetSbtOffsetBufferGpuAddress(reinterpret_cast<void*>(cudaGpuAddress));
    SetSbtOffsetSizeInBytes(sizeof(uint32_t));
    SetSbtOffsetStrideInBytes(sizeof(uint32_t));
    SetNumSbtOffsets(numSbtOffsets);
}

void rtlib::ext::BuildInputTriangles::SetSbtOffsetBufferGpuAddress(void* cudaGpuAddress)noexcept
{
    m_BuildInput.triangleArray.sbtIndexOffsetBuffer = reinterpret_cast<CUdeviceptr>(cudaGpuAddress);
}

auto rtlib::ext::BuildInputTriangles::GetSbtOffsetBufferGpuAddress() const noexcept-> void*
{
    return reinterpret_cast<void*>(m_BuildInput.triangleArray.sbtIndexOffsetBuffer);
}

void rtlib::ext::BuildInputTriangles::SetSbtOffsetSizeInBytes(size_t sizeInBytes)noexcept
{
    m_BuildInput.triangleArray.sbtIndexOffsetSizeInBytes = sizeInBytes;
}

auto rtlib::ext::BuildInputTriangles::GetSbtOffsetSizeInBytes() const noexcept-> size_t
{
    return m_BuildInput.triangleArray.sbtIndexOffsetSizeInBytes;
}

void rtlib::ext::BuildInputTriangles::SetSbtOffsetStrideInBytes(size_t strideInBytes)noexcept
{
    m_BuildInput.triangleArray.sbtIndexOffsetStrideInBytes = strideInBytes;
}

auto rtlib::ext::BuildInputTriangles::GetSbtOffsetStrideInBytes() const noexcept-> size_t
{
    return m_BuildInput.triangleArray.sbtIndexOffsetStrideInBytes;
}

void rtlib::ext::BuildInputTriangles::SetNumSbtOffsets(size_t numIndices)noexcept
{
    m_BuildInput.triangleArray.numSbtRecords = numIndices;
    m_SbtRecordGeometryFlags.resize(numIndices);
    m_BuildInput.customPrimitiveArray.flags = m_SbtRecordGeometryFlags.data();
}

auto rtlib::ext::BuildInputTriangles::GetNumSbtOffsets() const noexcept -> size_t
{
    return m_BuildInput.triangleArray.numSbtRecords;
}

void rtlib::ext::BuildInputTriangles::SetSbtRecordGeometryFlags(size_t idx, unsigned int flags)noexcept
{
    if (idx < m_BuildInput.triangleArray.numSbtRecords) {
        m_SbtRecordGeometryFlags[idx] = flags;
    }
}

auto rtlib::ext::BuildInputTriangles::GetSbtRecordGeometryFlags(size_t idx) const noexcept-> unsigned int
{
    if (idx < m_BuildInput.triangleArray.numSbtRecords) {
        return m_SbtRecordGeometryFlags[idx];
    }
    else {
        return OPTIX_GEOMETRY_FLAG_NONE;
    }
}

void rtlib::ext::BuildInputTriangles::SetNumSbtRecordBases(size_t numSbtRecordBases)noexcept
{
    m_NumSbtRecordBases = numSbtRecordBases;
}

auto rtlib::ext::BuildInputTriangles::GetNumSbtRecordBases() const noexcept -> size_t
{
    return m_NumSbtRecordBases;
}

void rtlib::ext::BuildInputTriangles::SetPreTransformsGpuAddress(void* preTransforms)
{
    m_BuildInput.triangleArray.preTransform = reinterpret_cast<CUdeviceptr>(preTransforms);
}

auto rtlib::ext::BuildInputTriangles::GetPreTransformsGpuAddress() const -> void*
{
    return reinterpret_cast<void*>(m_BuildInput.triangleArray.preTransform);
}

auto rtlib::ext::BuildInputTriangles::SetUserPointer(void* userPointer) noexcept
{
    m_UserPointer = userPointer;
}

auto rtlib::ext::BuildInputTriangles::GetUserPointer() const noexcept -> void*
{
    return m_UserPointer;
}

auto rtlib::ext::BuildInputTriangles::New() noexcept -> std::shared_ptr<BuildInputTriangles>
{
    return std::shared_ptr<BuildInputTriangles>(new BuildInputTriangles());
}

rtlib::ext::BuildInputTriangles::~BuildInputTriangles() noexcept {}

void rtlib::ext::BuildInputTriangles::AddGasReferenceInternal(const std::shared_ptr<GeometryAccelerationStructure>& gas) noexcept
{
}

rtlib::ext::BuildInputCustomPrimitives::BuildInputCustomPrimitives() noexcept
{
    m_BuildInput.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    m_BuildInput.customPrimitiveArray = {};
    m_BuildInput.customPrimitiveArray.aabbBuffers = &m_AabbBufferGpuAddress;
}

bool rtlib::ext::BuildInputCustomPrimitives::CheckDirty() const noexcept
{
    return m_Dirty;
}

void rtlib::ext::BuildInputCustomPrimitives::Update() noexcept
{
    if (m_UpdateCallback) {
        UpdateCallback(this);
    }
}

auto rtlib::ext::BuildInputCustomPrimitives::GetHandle() const -> const OptixBuildInput&
{
    // TODO: return �X�e�[�g�����g�������ɑ}�����܂�
    return m_BuildInput;
}

void rtlib::ext::BuildInputCustomPrimitives::SetAabbBuffer(void* cudaGpuAddress, size_t sizeInBytes, size_t strideInBytes) noexcept
{
    SetAabbBufferGpuAddress(cudaGpuAddress);
    SetAabbStrideInBytes(strideInBytes);
    SetNumPrimitives(sizeInBytes / strideInBytes);
}

void rtlib::ext::BuildInputCustomPrimitives::SetAabbBuffer(OptixAabb* cudaGpuAddress, size_t numPrimitives)
{
    SetAabbBufferGpuAddress(static_cast<void*>(cudaGpuAddress));
    SetAabbStrideInBytes(sizeof(OptixAabb));
    SetNumPrimitives(numPrimitives);
}

void rtlib::ext::BuildInputCustomPrimitives::SetAabbBufferGpuAddress(void* cudaGpuAddress) noexcept
{
    m_AabbBufferGpuAddress = reinterpret_cast<CUdeviceptr>(cudaGpuAddress);
}

auto rtlib::ext::BuildInputCustomPrimitives::GetAabbBufferGpuAddress() const noexcept -> void*
{
    return reinterpret_cast<void*>(m_AabbBufferGpuAddress);
}

void rtlib::ext::BuildInputCustomPrimitives::SetAabbStrideInBytes(size_t strideInBytes)  noexcept
{
    m_BuildInput.customPrimitiveArray.strideInBytes = strideInBytes;
}

auto rtlib::ext::BuildInputCustomPrimitives::GetAabbStrideInBytes() const noexcept -> size_t
{
    return m_BuildInput.customPrimitiveArray.strideInBytes;
}

void rtlib::ext::BuildInputCustomPrimitives::SetNumPrimitives(size_t numPrimitives)
{
    m_BuildInput.customPrimitiveArray.numPrimitives = numPrimitives;
}

auto rtlib::ext::BuildInputCustomPrimitives::GetNumPrimitives() const noexcept -> size_t
{
    return m_BuildInput.customPrimitiveArray.numPrimitives;
}

void rtlib::ext::BuildInputCustomPrimitives::SetPrimitiveIndexOffset(unsigned int  offset) noexcept
{
    m_BuildInput.customPrimitiveArray.primitiveIndexOffset = offset;
}

auto rtlib::ext::BuildInputCustomPrimitives::GetPrimitiveIndexOffset() const noexcept -> unsigned int
{
    return m_BuildInput.customPrimitiveArray.primitiveIndexOffset;
}

void rtlib::ext::BuildInputCustomPrimitives::SetSbtOffsetBuffer(void* cudaGpuAddress, size_t sizeInBytes, size_t strideInBytes, size_t formatSize)noexcept
{

    SetSbtOffsetBufferGpuAddress(cudaGpuAddress);
    SetSbtOffsetSizeInBytes(formatSize);
    SetSbtOffsetStrideInBytes(strideInBytes);
    SetNumSbtOffsets(sizeInBytes / strideInBytes);
}

void rtlib::ext::BuildInputCustomPrimitives::SetSbtOffsetBuffer(uint8_t* cudaGpuAddress, size_t numSbtOffsets)noexcept
{
    SetSbtOffsetBufferGpuAddress(reinterpret_cast<void*>(cudaGpuAddress));
    SetSbtOffsetSizeInBytes(sizeof(uint8_t));
    SetSbtOffsetStrideInBytes(sizeof(uint8_t));
    SetNumSbtOffsets(numSbtOffsets);
}

void rtlib::ext::BuildInputCustomPrimitives::SetSbtOffsetBuffer(uint16_t* cudaGpuAddress, size_t numSbtOffsets)noexcept
{
    SetSbtOffsetBufferGpuAddress(reinterpret_cast<void*>(cudaGpuAddress));
    SetSbtOffsetSizeInBytes(sizeof(uint16_t));
    SetSbtOffsetStrideInBytes(sizeof(uint16_t));
    SetNumSbtOffsets(numSbtOffsets);
}

void rtlib::ext::BuildInputCustomPrimitives::SetSbtOffsetBuffer(uint32_t* cudaGpuAddress, size_t numSbtOffsets)noexcept
{
    SetSbtOffsetBufferGpuAddress(reinterpret_cast<void*>(cudaGpuAddress));
    SetSbtOffsetSizeInBytes(sizeof(uint32_t));
    SetSbtOffsetStrideInBytes(sizeof(uint32_t));
    SetNumSbtOffsets(numSbtOffsets);
}

void rtlib::ext::BuildInputCustomPrimitives::SetSbtOffsetBufferGpuAddress(void* cudaGpuAddress)noexcept
{
    m_BuildInput.customPrimitiveArray.sbtIndexOffsetBuffer = reinterpret_cast<CUdeviceptr>(cudaGpuAddress);
}

auto rtlib::ext::BuildInputCustomPrimitives::GetSbtOffsetBufferGpuAddress() const noexcept-> void*
{
    return reinterpret_cast<void*>(m_BuildInput.customPrimitiveArray.sbtIndexOffsetBuffer);
}

void rtlib::ext::BuildInputCustomPrimitives::SetSbtOffsetSizeInBytes(size_t sizeInBytes)noexcept
{
    m_BuildInput.customPrimitiveArray.sbtIndexOffsetSizeInBytes = sizeInBytes;
}

auto rtlib::ext::BuildInputCustomPrimitives::GetSbtOffsetSizeInBytes() const noexcept-> size_t
{
    return m_BuildInput.customPrimitiveArray.sbtIndexOffsetSizeInBytes;
}

void rtlib::ext::BuildInputCustomPrimitives::SetSbtOffsetStrideInBytes(size_t strideInBytes)noexcept
{
    m_BuildInput.customPrimitiveArray.sbtIndexOffsetStrideInBytes = strideInBytes;
}

auto rtlib::ext::BuildInputCustomPrimitives::GetSbtOffsetStrideInBytes() const noexcept-> size_t
{
    return m_BuildInput.customPrimitiveArray.sbtIndexOffsetStrideInBytes;
}

void rtlib::ext::BuildInputCustomPrimitives::SetNumSbtOffsets(size_t numIndices)noexcept
{
    m_BuildInput.customPrimitiveArray.numSbtRecords = numIndices;
    m_SbtRecordGeometryFlags.resize(numIndices);
    m_BuildInput.customPrimitiveArray.flags = m_SbtRecordGeometryFlags.data();
}

auto rtlib::ext::BuildInputCustomPrimitives::GetNumSbtOffsets() const noexcept -> size_t
{
    return m_BuildInput.customPrimitiveArray.numSbtRecords;
}

void rtlib::ext::BuildInputCustomPrimitives::SetSbtRecordGeometryFlags(size_t idx, unsigned int flags)noexcept
{
    if (idx < m_BuildInput.customPrimitiveArray.numSbtRecords) {
        m_SbtRecordGeometryFlags[idx] = flags;
    }
}

auto rtlib::ext::BuildInputCustomPrimitives::GetSbtRecordGeometryFlags(size_t idx) const noexcept-> unsigned int
{
    if (idx < m_BuildInput.customPrimitiveArray.numSbtRecords) {
        return m_SbtRecordGeometryFlags[idx];
    }
    else {
        return OPTIX_GEOMETRY_FLAG_NONE;
    }
}

void rtlib::ext::BuildInputCustomPrimitives::SetNumSbtRecordBases(size_t numSbtRecordBases) noexcept
{
    m_NumSbtRecordBases = numSbtRecordBases;
}

auto rtlib::ext::BuildInputCustomPrimitives::GetNumSbtRecordBases() const noexcept -> size_t
{
    return m_NumSbtRecordBases;
}

auto rtlib::ext::BuildInputCustomPrimitives::SetUserPointer(void* userPointer) noexcept
{
    m_UserPointer = userPointer;
}

auto rtlib::ext::BuildInputCustomPrimitives::GetUserPointer() const noexcept -> void*
{
    return m_UserPointer;
}

auto rtlib::ext::BuildInputCustomPrimitives::New() noexcept -> std::shared_ptr<BuildInputCustomPrimitives>
{
    return std::shared_ptr<BuildInputCustomPrimitives>(new BuildInputCustomPrimitives());
}

rtlib::ext::BuildInputCustomPrimitives::~BuildInputCustomPrimitives() noexcept {}

void rtlib::ext::BuildInputCustomPrimitives::AddGasReferenceInternal(const std::shared_ptr<GeometryAccelerationStructure>& gas) noexcept
{
    if (!gas) {
        return;
    }
    if (m_GasReferences.count(gas->GetName()) == 0) {
        m_GasReferences[gas->GetName()] = gas;
    }
}

rtlib::ext::GeometryAccelerationStructure::GeometryAccelerationStructure(const std::string& name) noexcept
{
    m_Name = name;
}



bool rtlib::ext::GeometryAccelerationStructure::CheckDirty() const noexcept
{
    return m_Dirty;
}

auto rtlib::ext::GeometryAccelerationStructure::GetName() const noexcept -> std::string
{
    return m_Name;
}

void rtlib::ext::GeometryAccelerationStructure::Build(const rtlib::OPXContext* context, const OptixAccelBuildOptions& accelOptions)
{
    this->Update();
    auto buildInputs = std::vector<OptixBuildInput>(m_BuildInputs.size());
    for (auto& buildInput : m_BuildInputs) {
        buildInputs.push_back(std::visit([](const auto& v) { return v->GetHandle(); }, buildInput));
    }
    auto [outputBuffer, gasHandle] = context->buildAccel(accelOptions, buildInputs);
    this->m_Handle = std::move(gasHandle);
    this->m_Buffer = std::move(outputBuffer);
    m_Dirty = true;
}

auto rtlib::ext::GeometryAccelerationStructure::GetHandle() const noexcept -> OptixTraversableHandle
{
    return m_Handle;
}

void rtlib::ext::GeometryAccelerationStructure::AddBuildInputInternal(const BuildInputTrianglesPtr& triangles) noexcept
{
    m_BuildInputs.push_back(triangles);
    m_NumSbtRecordBases += triangles->GetNumSbtRecordBases();
    m_Dirty = false;
}

void rtlib::ext::GeometryAccelerationStructure::AddBuildInputInternal(const BuildInputCustomPrimitivesPtr& customPrimitives) noexcept
{
    m_BuildInputs.push_back(customPrimitives);
    m_NumSbtRecordBases += customPrimitives->GetNumSbtRecordBases();
    m_Dirty = false;
}

auto rtlib::ext::GeometryAccelerationStructure::GetBuildInput(size_t idx) const -> const BuildInputGeometryPtr&
{
    // TODO: return �X�e�[�g�����g�������ɑ}�����܂�
    return m_BuildInputs.at(idx);
}

auto rtlib::ext::GeometryAccelerationStructure::EnumerateBuildInputs() const noexcept -> const std::vector<BuildInputGeometryPtr>&
{
    // TODO: return �X�e�[�g�����g�������ɑ}�����܂�
    return m_BuildInputs;
}

auto rtlib::ext::GeometryAccelerationStructure::GetNumSbtRecordBases() const noexcept -> size_t
{
    return m_NumSbtRecordBases;
}

void rtlib::ext::GeometryAccelerationStructure::Update() noexcept
{
    m_NumSbtRecordBases = 0;
    for (auto& buildInput : m_BuildInputs) {
        m_NumSbtRecordBases += std::visit([](auto& v) {
            v->Update();
            return v->GetNumSbtRecordBases(); 
        },buildInput);
    }
    if (m_UpdateCallback) {
        m_UpdateCallback(this);
    }
}

void rtlib::ext::GeometryAccelerationStructure::SetUserPointer(void* userPointer) noexcept
{
    m_UserPonter = userPointer;
}

auto rtlib::ext::GeometryAccelerationStructure::GetUserPointer() const noexcept -> void*
{
    return m_UserPonter;
}

void rtlib::ext::GeometryAccelerationStructure::SetUpdateCallback(UpdateCallback updateCallback)
{
    m_UpdateCallback = updateCallback;
}

void rtlib::ext::GeometryAccelerationStructure::AddBuildInput(const std::shared_ptr<GeometryAccelerationStructure>& gas, const BuildInputTrianglesPtr& triangles)
{
    gas->AddBuildInputInternal(triangles);
    triangles->AddGasReferenceInternal(gas);
}

void rtlib::ext::GeometryAccelerationStructure::AddBuildInput(const std::shared_ptr<GeometryAccelerationStructure>& gas, const BuildInputCustomPrimitivesPtr& customPrimitives)
{
    gas->AddBuildInputInternal(customPrimitives);
    customPrimitives->AddGasReferenceInternal(gas);
}

auto rtlib::ext::GeometryAccelerationStructure::New(const std::string& name) noexcept -> std::shared_ptr<GeometryAccelerationStructure>
{
    return std::shared_ptr<GeometryAccelerationStructure>(new GeometryAccelerationStructure(name));
}

rtlib::ext::GeometryAccelerationStructure::~GeometryAccelerationStructure() noexcept
{
}

void rtlib::ext::GASInstance::Update() noexcept
{
    
    if (m_UpdateCallback) {
        m_UpdateCallback(this);
    }
}
