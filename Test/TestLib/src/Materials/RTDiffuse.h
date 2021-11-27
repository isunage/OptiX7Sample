#ifndef TEST_RT_MATERIAL_DIFFUSE_H
#define TEST_RT_MATERIAL_DIFFUSE_H
#include <TestLib/RTMaterial.h>
#include <TestLib/RTProperties.h>
#include <variant>
#include <string>
namespace test
{
	class RTDiffuseReader;
	class RTDiffuse : public RTMaterial
	{
	public:
		RTDiffuse()noexcept;
		virtual auto GetTypeName()   const noexcept -> RTString override;
		virtual auto GetPluginName() const noexcept -> RTString override;
		virtual auto GetID()         const noexcept -> RTString override;
		virtual auto GetProperties() const noexcept -> const RTProperties & override;
		virtual auto GetJsonAsData() const noexcept ->       nlohmann::json override;
		virtual void SetID(const RTString& id)noexcept;
		auto GetReflectance() const noexcept -> std::variant<RTTexturePtr, RTColor, RTFloat>;
		void SetReflectance(const RTTexturePtr& texture)noexcept;
		void SetReflectance(const RTColor& color)noexcept;
		void SetReflectance(const RTFloat& fv)noexcept;
		virtual ~RTDiffuse() noexcept {}
	private:
		friend class RTDiffuseReader;
		RTProperties m_Properties;
	};
	class RTDiffuseReader : public RTMaterialReader
	{
	public:
		RTDiffuseReader(const std::shared_ptr<RTMaterialCache>& matCache, const std::shared_ptr< RTTextureCache>& texCache)noexcept;
		virtual auto GetPluginName()const noexcept -> RTString override;
		virtual auto LoadJsonFromData(const nlohmann::json& json)noexcept -> RTMaterialPtr override;
		virtual ~RTDiffuseReader()noexcept;
	private:
		struct Impl;
		std::unique_ptr<Impl> m_Impl;
	};

}
#endif