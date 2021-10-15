#ifndef RT_ASSETS_H
#define RT_ASSETS_H
#include <RTLib/CUDA.h>
#include <RTLib/VectorFunction.h>
#include <RTLib/ext/VariableMap.h>
#include <RTLib/ext/Mesh.h>
#include <memory>
#include <vector>
#include <string>
#include <array>
#include <type_traits>
#include <unordered_map>
namespace test {
    class RTAsset
    {
    public:
        virtual bool Load()         = 0;
        virtual void Free()         = 0;
        virtual bool IsValid()const = 0;
        virtual ~RTAsset()noexcept{}
    };
    using RTAssetPtr = std::shared_ptr<RTAsset>;
    class RTAssetManager
    {
    public:
		virtual bool LoadAsset(const std::string& key, const std::string& path) = 0;
		virtual void FreeAsset(const std::string& key) = 0;
		auto  GetAsset(const std::string& keyName)const-> RTAssetPtr;
		template<typename T, bool cond = std::is_base_of_v< RTAsset,T>>
		auto  GetAssetAs(const std::string& keyName)const->std::shared_ptr<T>
		{
			return std::static_pointer_cast<T, RTAsset>(GetAsset(keyName));
		}
		auto  GetAssets()const -> const std::unordered_map<std::string, RTAssetPtr>&;
		auto  GetAssets()->std::unordered_map<std::string, RTAssetPtr>&;
		bool  HasAsset(const std::string& keyName)const;
		virtual ~RTAssetManager() {}
	public:
		std::unordered_map<std::string, RTAssetPtr> m_Assets = {};
    };
}
#endif