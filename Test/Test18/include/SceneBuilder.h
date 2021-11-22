#ifndef TEST17_SCENE_BUILDER_H
#define TEST17_SCENE_BUILDER_H
#include <tiny_obj_loader.h>
#include <RTLib/Optix.h>
#include <RTLib/CUDA.h>
#include <RTLib/ext/Math/VectorFunction.h>
#include <RTLib/ext/Mesh.h>
#include <RTLib/ext/TraversalHandle.h>
#include <RTLib/ext/VariableMap.h>
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
	using MaterialSetPtr = std::shared_ptr<MaterialSet>;
	class  ObjMeshGroup {
	public:
		using MeshGroup = rtlib::ext::MeshGroup;
		bool Load(const std::string& objFilePath, const std::string& mtlFileDir)noexcept;
		auto GetMeshGroup()const noexcept   -> std::shared_ptr<MeshGroup>;
		auto GetMaterialSet()const noexcept -> std::shared_ptr<MaterialSet>;
	private:
		std::shared_ptr<MeshGroup>   m_MeshGroup   = {};
		std::shared_ptr<MaterialSet> m_MaterialSet = {};
	};
}
#endif