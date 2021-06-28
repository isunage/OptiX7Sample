#ifndef TEST15_SCENE_BUILDER_H
#define TEST15_SCENE_BUILDER_H
#include <tiny_obj_loader.h>
#include <RTLib/Optix.h>
#include <RTLib/CUDA.h>
#include <RTLib/VectorFunction.h>
#include <string>
#include <vector>
#include <array>
#include <unordered_map>
#include <memory>
namespace test {
	enum class PhongMaterialType {
		eDiffuse = 0,
		eSpecular,
		eEmission,
	};
	struct PhongMaterial {
		using Materialtype = PhongMaterialType;
		Materialtype type;
		std::string  name;
		float3       diffCol;
		float3       specCol;
		float3       emitCol;
		float        shinness;
		float        refrInd;
		std::string  diffTex;
		std::string  specTex;
		std::string  emitTex;
		std::string  shinTex;
	};
	struct MaterialSet {
		std::vector<PhongMaterial>        materials = {};
	};
	template<typename T>
	struct ArrayBuffer {
		std::vector<T>       cpuHandle   = {};
		rtlib::CUDABuffer<T> gpuHandle   = {};
	public:
		void Upload() {
			gpuHandle.allocate(cpuHandle.size());
			gpuHandle.upload(cpuHandle);
		}
	};
	struct MeshSharedResource {
		std::string         name         = {};
		ArrayBuffer<float3> vertexBuffer = {};
		ArrayBuffer<float3> normalBuffer = {};
		ArrayBuffer<float2> texCrdBuffer = {};
	};
	using  MeshSharedResourcePtr = std::shared_ptr<MeshSharedResource>;
	struct MeshUniqueResource {
		std::string           name         = {};
		std::vector<uint32_t> materials    = {};
		ArrayBuffer<uint3>    triIndBuffer = {};
		ArrayBuffer<uint32_t> matIndBuffer = {};
	};
	using  MeshUniqueResourcePtr = std::shared_ptr<MeshUniqueResource>;
	class  MeshGroup;
	class  Mesh {
	public:
		Mesh()noexcept {}
		auto GetSharedResource()const noexcept -> MeshSharedResourcePtr;
		void SetUniqueResource(const  MeshUniqueResourcePtr& res)noexcept;
		auto GetUniqueResource()const->MeshUniqueResourcePtr;
	private:
		friend MeshGroup;
		std::string              m_Name           = {};
		MeshSharedResourcePtr    m_SharedResource = {};
		MeshUniqueResourcePtr    m_UniqueResource = {};
	};
	using  MeshPtr = std::shared_ptr<Mesh>;
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
	private:
		using MeshUniqueResourcePtrMap = std::unordered_map<std::string, MeshUniqueResourcePtr>;
		std::string              m_Name            = {};
		MeshSharedResourcePtr    m_SharedResource  = {};
		MeshUniqueResourcePtrMap m_UniqueResources = {};
	};
	class  ObjMeshGroup {
	public:
		bool Load(const std::string& objFilePath, const std::string& mtlFileDir)noexcept;
		auto GetMeshGroup()const noexcept -> std::shared_ptr<MeshGroup>;
		auto GetMaterialSet()const noexcept -> std::shared_ptr<MaterialSet>;
	private:
		std::shared_ptr<MeshGroup>   m_MeshGroup   = {};
		std::shared_ptr<MaterialSet> m_MaterialSet = {};
	};
	struct GASHandle {
		OptixTraversableHandle   handle   = {};
		rtlib::CUDABuffer<void>  buffer   = {};
		std::vector<MeshPtr>     meshes   = {};
		size_t                   sbtCount = 0;
	public:
		void Build(const rtlib::OPXContext* context, const OptixAccelBuildOptions& accelOptions) {
			auto buildInputs   = std::vector<OptixBuildInput>(this->meshes.size());
			auto vertexBuffers = std::vector<CUdeviceptr>(this->meshes.size());
			auto buildFlags    = std::vector<std::vector<unsigned int>>(this->meshes.size());
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
				buildInputs[i].type                                      = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
				buildInputs[i].triangleArray.flags                       = buildFlags[i].data();
				buildInputs[i].triangleArray.vertexBuffers               = vertexBuffers.data() + i;
				buildInputs[i].triangleArray.vertexFormat                = OPTIX_VERTEX_FORMAT_FLOAT3;
				buildInputs[i].triangleArray.vertexStrideInBytes         = sizeof(float3);
				buildInputs[i].triangleArray.numVertices                 = mesh->GetSharedResource()->vertexBuffer.gpuHandle.getCount();
				buildInputs[i].triangleArray.indexBuffer                 = reinterpret_cast<CUdeviceptr>(mesh->GetUniqueResource()->triIndBuffer.gpuHandle.getDevicePtr());
				buildInputs[i].triangleArray.indexFormat                 = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
				buildInputs[i].triangleArray.indexStrideInBytes		     = sizeof(uint3);
				buildInputs[i].triangleArray.numIndexTriplets			 = mesh->GetUniqueResource()->triIndBuffer.gpuHandle.getCount();
				buildInputs[i].triangleArray.sbtIndexOffsetBuffer        = reinterpret_cast<CUdeviceptr>(mesh->GetUniqueResource()->matIndBuffer.gpuHandle.getDevicePtr());
				buildInputs[i].triangleArray.sbtIndexOffsetSizeInBytes   = sizeof(uint32_t);
				buildInputs[i].triangleArray.sbtIndexOffsetStrideInBytes = sizeof(uint32_t);
				buildInputs[i].triangleArray.numSbtRecords               = mesh->GetUniqueResource()->materials.size();
				sbtCount += buildInputs[i].triangleArray.numSbtRecords;
				++i;
			}
			auto [outputBuffer, gasHandle] = context->buildAccel(accelOptions, buildInputs);
			this->handle   = gasHandle;
			this->buffer   = std::move(outputBuffer);
			this->sbtCount = sbtCount;
		}
	};
	struct InstanceSet {
		test::ArrayBuffer<OptixInstance>        instanceBuffer = {};
		std::vector<std::shared_ptr<GASHandle>> baseGASHandles = {};
	};
	using  InstanceSetPtr = std::shared_ptr<InstanceSet>;
	struct IASHandle {
		OptixTraversableHandle      handle       = {};
		rtlib::CUDABuffer<void>     buffer       = {};
		std::vector<InstanceSetPtr> instanceSets = {};
		size_t                      sbtCount     =  0;
	public:
		void Build(const rtlib::OPXContext* context, const OptixAccelBuildOptions& accelOptions) {
			auto buildInputs = std::vector <OptixBuildInput>(this->instanceSets.size());
			size_t i = 0;
			size_t sbtCount = 0;
			for (auto& instanceSet : this->instanceSets) {
				buildInputs[i].type                       = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
				buildInputs[i].instanceArray.instances    = reinterpret_cast<CUdeviceptr>(instanceSet->instanceBuffer.gpuHandle.getDevicePtr());
				buildInputs[i].instanceArray.numInstances = instanceSet->instanceBuffer.gpuHandle.getCount();
				for (auto& baseGasHandle : instanceSet->baseGASHandles) {
					sbtCount += baseGasHandle->sbtCount;
				}
				++i;
			}
			auto [outputBuffer, iasHandle] = context->buildAccel(accelOptions, buildInputs);
			this->handle                   = iasHandle;
			this->buffer                   = std::move(outputBuffer);
			this->sbtCount = sbtCount;
		}
	};
	using  IASHandlePtr   = std::shared_ptr<IASHandle>;
}
#endif