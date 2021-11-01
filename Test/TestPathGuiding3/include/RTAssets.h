#ifndef RT_ASSETS_H
#define RT_ASSETS_H
#include <RTLib/CUDA.h>
#include <RTLib/VectorFunction.h>
#include <RTLib/ext/VariableMap.h>
#include <RTLib/ext/Mesh.h>
#include <memory>
#include <vector>
#include <string>
#include <unordered_map>
namespace test {
	class RTTextureAssetManager
	{
	public:
		bool LoadAsset(const std::string& keyName,const std::string& texPath);
		void FreeAsset(const std::string& keyName);
		auto  GetAsset(const std::string& keyName)const -> const rtlib::CUDATexture2D<uchar4>&;
		auto  GetAsset(const std::string& keyName)      ->       rtlib::CUDATexture2D<uchar4>&;
		auto  GetAssets()const -> const std::unordered_map<std::string, rtlib::CUDATexture2D<uchar4>>&;
		auto  GetAssets()      ->       std::unordered_map<std::string, rtlib::CUDATexture2D<uchar4>>&;
		bool  HasAsset(const std::string& keyName)const noexcept;
		void Reset();
		~RTTextureAssetManager() {}
	private:
		std::unordered_map<std::string, rtlib::CUDATexture2D<uchar4>> m_Textures = {};
	};
	struct RTObjModel {
		rtlib::ext::MeshGroupPtr             meshGroup;
		std::vector<rtlib::ext::VariableMap> materials;
	};
	class RTObjModelAssetManager
	{
	public:
		bool LoadAsset(const std::string& keyName, const std::string& objPath);
		void FreeAsset(const std::string& keyName);
		auto  GetAsset(const std::string& keyName)const -> const RTObjModel&;
		auto  GetAsset(const std::string& keyName)->RTObjModel&;
		auto  GetAssets()const -> const std::unordered_map<std::string, RTObjModel>&;
		auto  GetAssets()      ->       std::unordered_map<std::string, RTObjModel>&;
		bool  HasAsset(const std::string& keyName)const noexcept;
		void  Reset();
		~RTObjModelAssetManager() {}
	private:
		std::unordered_map<std::string, RTObjModel> m_ObjModels = {};
	};
}
#endif