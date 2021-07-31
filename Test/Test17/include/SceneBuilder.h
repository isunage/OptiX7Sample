#ifndef TEST17_SCENE_BUILDER_H
#define TEST17_SCENE_BUILDER_H
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
		eRefract,
		eEmission,
	};
	struct PhongMaterial {
		using Materialtype = PhongMaterialType;
		Materialtype type;
		std::string  name;
		float3       diffCol;
		float3       specCol;
		float3       tranCol;
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
			gpuHandle.resize(cpuHandle.size());
			gpuHandle.upload(cpuHandle);
		}
	};
	struct MeshSharedResource {
		std::string         name         = {};
		ArrayBuffer<float3> vertexBuffer = {};
		ArrayBuffer<float3> normalBuffer = {};
		ArrayBuffer<float2> texCrdBuffer = {};
		static auto New() ->std::shared_ptr<MeshSharedResource> {
			return std::shared_ptr<MeshSharedResource>(new MeshSharedResource());
		}
	};
	using  MeshSharedResourcePtr = std::shared_ptr<MeshSharedResource>;
	struct MeshUniqueResource {
		std::string           name         = {};
		std::vector<uint32_t> materials    = {};
		ArrayBuffer<uint3>    triIndBuffer = {};
		ArrayBuffer<uint32_t> matIndBuffer = {};
		static auto New() ->std::shared_ptr<MeshUniqueResource> {
			return std::shared_ptr<MeshUniqueResource>(new MeshUniqueResource());
		}
	};
	using  MeshUniqueResourcePtr = std::shared_ptr<MeshUniqueResource>;
	class  MeshGroup;
	class  Mesh {
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
		void Build(const rtlib::OPXContext* context, const OptixAccelBuildOptions& accelOptions);
	};
	struct Instance {
		OptixInstance              instance      = {};
		std::shared_ptr<GASHandle> baseGASHandle = {};
	};
	struct InstanceSet {
		test::ArrayBuffer<OptixInstance>        instanceBuffer = {};
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
#endif