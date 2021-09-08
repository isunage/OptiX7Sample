#include "../include/RTLib/ext/TraversalHandle.h"


void rtlib::ext::GASHandle::Build(const rtlib::OPXContext* context, const OptixAccelBuildOptions& accelOptions) {
    auto buildInputs = std::vector<OptixBuildInput>(this->meshes.size());
    auto vertexBuffers = std::vector<CUdeviceptr>(this->meshes.size());
    auto buildFlags = std::vector<std::vector<unsigned int>>(this->meshes.size());
    size_t i = 0;
    size_t sbtCount = 0;
    for (auto& mesh : this->meshes) {
        if (mesh->GetSharedResource()->vertexBuffer.gpuHandle.getSizeInBytes() == 0) {
            mesh->GetSharedResource()->vertexBuffer.Upload();
        }
        if (mesh->GetSharedResource()->normalBuffer.gpuHandle.getSizeInBytes() == 0) {
            mesh->GetSharedResource()->normalBuffer.Upload();
        }
        if (mesh->GetSharedResource()->texCrdBuffer.gpuHandle.getSizeInBytes() == 0) {
            mesh->GetSharedResource()->texCrdBuffer.Upload();
        }
        if (mesh->GetUniqueResource()->triIndBuffer.gpuHandle.getSizeInBytes() == 0) {
            mesh->GetUniqueResource()->triIndBuffer.Upload();
        }
        if (mesh->GetUniqueResource()->matIndBuffer.gpuHandle.getSizeInBytes() == 0) {
            mesh->GetUniqueResource()->matIndBuffer.Upload();
        }
        vertexBuffers[i] = reinterpret_cast<CUdeviceptr>(mesh->GetSharedResource()->vertexBuffer.gpuHandle.getDevicePtr());
        buildFlags[i].resize(mesh->GetUniqueResource()->materials.size());
        std::fill(std::begin(buildFlags[i]), std::end(buildFlags[i]), OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT);
        buildInputs[i].type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        buildInputs[i].triangleArray.flags = buildFlags[i].data();
        buildInputs[i].triangleArray.vertexBuffers = vertexBuffers.data() + i;
        buildInputs[i].triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
        buildInputs[i].triangleArray.vertexStrideInBytes = sizeof(float3);
        buildInputs[i].triangleArray.numVertices = mesh->GetSharedResource()->vertexBuffer.gpuHandle.getCount();
        buildInputs[i].triangleArray.indexBuffer = reinterpret_cast<CUdeviceptr>(mesh->GetUniqueResource()->triIndBuffer.gpuHandle.getDevicePtr());
        buildInputs[i].triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        buildInputs[i].triangleArray.indexStrideInBytes = sizeof(uint3);
        buildInputs[i].triangleArray.numIndexTriplets = mesh->GetUniqueResource()->triIndBuffer.gpuHandle.getCount();
        buildInputs[i].triangleArray.sbtIndexOffsetBuffer = reinterpret_cast<CUdeviceptr>(mesh->GetUniqueResource()->matIndBuffer.gpuHandle.getDevicePtr());
        buildInputs[i].triangleArray.sbtIndexOffsetSizeInBytes = sizeof(uint32_t);
        buildInputs[i].triangleArray.sbtIndexOffsetStrideInBytes = sizeof(uint32_t);
        buildInputs[i].triangleArray.numSbtRecords = mesh->GetUniqueResource()->materials.size();
        sbtCount += buildInputs[i].triangleArray.numSbtRecords;
        ++i;
    }
    auto [outputBuffer, gasHandle] = context->buildAccel(accelOptions, buildInputs);
    this->handle = gasHandle;
    this->buffer = std::move(outputBuffer);
    this->sbtCount = sbtCount;
}

void rtlib::ext::IASHandle::Build(const rtlib::OPXContext* context, const OptixAccelBuildOptions& accelOptions) {
    auto buildInputs = std::vector <OptixBuildInput>(this->instanceSets.size());
    size_t i = 0;
    size_t sbtCount = 0;
    for (auto& instanceSet : this->instanceSets) {
        buildInputs[i].type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
        buildInputs[i].instanceArray.instances = reinterpret_cast<CUdeviceptr>(instanceSet->instanceBuffer.gpuHandle.getDevicePtr());
        buildInputs[i].instanceArray.numInstances = instanceSet->instanceBuffer.gpuHandle.getCount();
        for (auto& baseGasHandle : instanceSet->baseGASHandles) {
            sbtCount += baseGasHandle->sbtCount;
        }
        ++i;
    }
    auto [outputBuffer, iasHandle] = context->buildAccel(accelOptions, buildInputs);
    this->handle = iasHandle;
    this->buffer = std::move(outputBuffer);
    this->sbtCount = sbtCount;
}

auto rtlib::ext::IASHandle::GetSbtCount()const noexcept -> size_t {
    return this->sbtCount;
}

void rtlib::ext::InstanceSet::SetInstance(const Instance& instance) noexcept
{
    instanceBuffer.cpuHandle.push_back(instance.instance);
    if(instance.type==InstanceType::GAS){
        instanceIndices.push_back({ InstanceType::GAS, baseGASHandles.size() });
         baseGASHandles.push_back(instance.baseGASHandle);
    }else{
        instanceIndices.push_back({ InstanceType::IAS, baseIASHandles.size() });
         baseIASHandles.push_back(instance.baseIASHandle);
    }
}

auto rtlib::ext::InstanceSet::GetInstance(size_t i) const noexcept -> Instance
{
    auto instanceIndex  = instanceIndices[i];
    rtlib::ext::Instance instance = {};
    instance.type       = instanceIndex.type;
    instance.instance   = instanceBuffer.cpuHandle[i];
    if (instance.type == InstanceType::GAS)
    {
        instance.baseGASHandle = baseGASHandles[instanceIndex.index];
    }
    else {
        instance.baseIASHandle = baseIASHandles[instanceIndex.index];
    }
    return instance;
}

void rtlib::ext::InstanceSet::Upload() noexcept
{
    instanceBuffer.Upload();
}

void rtlib::ext::Instance::Init(const rtlib::ext::GASHandlePtr& gasHandle) {
    type                       = InstanceType::GAS;
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

void rtlib::ext::Instance::Init(const rtlib::ext::IASHandlePtr& iasHandle) {
    type                       = InstanceType::IAS;
    instance.traversableHandle = iasHandle->handle;
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
    baseIASHandle = iasHandle;
}

auto rtlib::ext::Instance::GetSbtOffset() const noexcept -> size_t
{
    return instance.sbtOffset;
}

void rtlib::ext::Instance::SetSbtOffset(size_t offset) noexcept
{
    instance.sbtOffset = offset;
}

auto rtlib::ext::Instance::GetSbtCount() const noexcept -> size_t
{
    if (type == InstanceType::GAS) {
        return baseGASHandle->sbtCount;
    }
    else
    {
        return baseIASHandle->sbtCount;
    }
}