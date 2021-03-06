#include "../include/RTLib/ext/TraversalHandle.h"
#include "../include/RTLib/ext/Resources/CUDA.h"

void rtlib::ext::GASHandle::Build(const rtlib::OPXContext* context, const OptixAccelBuildOptions& accelOptions) {
    auto buildInputs = std::vector<OptixBuildInput>(this->meshes.size());
    auto vertexBuffers = std::vector<CUdeviceptr>(this->meshes.size());
    auto buildFlags = std::vector<std::vector<unsigned int>>(this->meshes.size());
    size_t i = 0;
    size_t sbtCount = 0;
    for (auto& mesh : this->meshes) {
        if (!mesh->GetSharedResource()->vertexBuffer.HasGpuComponent("CUDA")) {
            throw std::runtime_error("VertexBuffer of Mesh '" + mesh->GetUniqueResource()->name + "' Has No CUDA Component!");
        }
        if (!mesh->GetSharedResource()->normalBuffer.HasGpuComponent("CUDA")) {
            throw std::runtime_error("NormalBuffer of Mesh '" + mesh->GetUniqueResource()->name + "' Has No CUDA Component!");
        }
        if (!mesh->GetSharedResource()->texCrdBuffer.HasGpuComponent("CUDA")) {
            throw std::runtime_error("TexCrdBuffer of Mesh '" + mesh->GetUniqueResource()->name + "' Has No CUDA Component!");
        }
        if (!mesh->GetUniqueResource()->triIndBuffer.HasGpuComponent("CUDA")) {
            throw std::runtime_error("TriIndBuffer of Mesh '" + mesh->GetUniqueResource()->name + "' Has No CUDA Component!");
        }
        if (!mesh->GetUniqueResource()->matIndBuffer.HasGpuComponent("CUDA")) {
            throw std::runtime_error("MatIndBuffer of Mesh '" + mesh->GetUniqueResource()->name + "' Has No CUDA Component!");
        }
        auto cudaVertexBuffer = mesh->GetSharedResource()->vertexBuffer.GetGpuComponent<rtlib::ext::resources::CUDABufferComponent<float3>>("CUDA");
        auto cudaNormalBuffer = mesh->GetSharedResource()->normalBuffer.GetGpuComponent<rtlib::ext::resources::CUDABufferComponent<float3>>("CUDA");
        auto cudaTexCrdBuffer = mesh->GetSharedResource()->texCrdBuffer.GetGpuComponent<rtlib::ext::resources::CUDABufferComponent<float2>>("CUDA");
        auto cudaTriIndBuffer = mesh->GetUniqueResource()->triIndBuffer.GetGpuComponent<rtlib::ext::resources::CUDABufferComponent<uint3>>("CUDA");
        auto cudaMatIndBuffer = mesh->GetUniqueResource()->matIndBuffer.GetGpuComponent<rtlib::ext::resources::CUDABufferComponent<uint32_t>>("CUDA");

        vertexBuffers[i] = reinterpret_cast<CUdeviceptr>(cudaVertexBuffer->GetHandle().getDevicePtr());
        buildFlags[i].resize(mesh->GetUniqueResource()->materials.size());
        std::fill(std::begin(buildFlags[i]), std::end(buildFlags[i]), OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT);
        buildInputs[i].type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        buildInputs[i].triangleArray.flags = buildFlags[i].data();
        buildInputs[i].triangleArray.vertexBuffers = vertexBuffers.data() + i;
        buildInputs[i].triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
        buildInputs[i].triangleArray.vertexStrideInBytes = sizeof(float3);
        buildInputs[i].triangleArray.numVertices = cudaVertexBuffer->GetHandle().getCount();
        buildInputs[i].triangleArray.indexBuffer = reinterpret_cast<CUdeviceptr>(cudaTriIndBuffer->GetHandle().getDevicePtr());
        buildInputs[i].triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        buildInputs[i].triangleArray.indexStrideInBytes = sizeof(uint3);
        buildInputs[i].triangleArray.numIndexTriplets = cudaTriIndBuffer->GetHandle().getCount();
        buildInputs[i].triangleArray.sbtIndexOffsetBuffer = reinterpret_cast<CUdeviceptr>(cudaMatIndBuffer->GetHandle().getDevicePtr());
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

void rtlib::ext::GASHandle::AddMesh(const MeshPtr& mesh)noexcept{
    this->meshes.push_back(mesh);
}

auto rtlib::ext::GASHandle::GetSbtCount()const noexcept -> size_t {
    return this->sbtCount;
}

void rtlib::ext::IASHandle::Build(const rtlib::OPXContext* context, const OptixAccelBuildOptions& accelOptions) {
    auto buildInputs = std::vector <OptixBuildInput>(this->instanceSets.size());
    size_t i = 0;
    size_t sbtCount = 0;
    for (auto& instanceSet : this->instanceSets) {
        buildInputs[i].type                       = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
        buildInputs[i].instanceArray.instances    = reinterpret_cast<CUdeviceptr>(instanceSet->instanceBuffer.gpuHandle.getDevicePtr());
        buildInputs[i].instanceArray.numInstances = instanceSet->instanceBuffer.gpuHandle.getCount();
        for (auto& baseGasHandle : instanceSet->baseGASHandles) {
            sbtCount += baseGasHandle->GetSbtCount();
        }
        ++i;
    }
    auto [outputBuffer, iasHandle] = context->buildAccel(accelOptions, buildInputs);
    this->handle = iasHandle;
    this->buffer = std::move(outputBuffer);
    this->sbtCount = sbtCount;
}

void rtlib::ext::IASHandle::AddInstanceSet(const rtlib::ext::InstanceSetPtr& instanceSet)noexcept {
    instanceSets.push_back(instanceSet);
}

auto rtlib::ext::IASHandle::GetInstanceSet( size_t idx)const noexcept -> const rtlib::ext::InstanceSetPtr& {
    return instanceSets[idx];
}

auto rtlib::ext::IASHandle::GetInstanceSets()const noexcept -> const std::vector<rtlib::ext::InstanceSetPtr > & {
    return instanceSets;
}

auto rtlib::ext::IASHandle::GetInstanceSets() noexcept -> std::vector<rtlib::ext::InstanceSetPtr >& {
    return instanceSets;
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
    instance.traversableHandle = gasHandle->GetHandle();
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
    instance.traversableHandle = iasHandle->GetHandle();
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
        return baseGASHandle->GetSbtCount();
    }
    else
    {
        return baseIASHandle->GetSbtCount();
    }
}