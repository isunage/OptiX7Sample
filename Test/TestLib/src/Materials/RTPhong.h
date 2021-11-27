#ifndef TEST_RT_MATERIAL_PHONG_H
#define TEST_RT_MATERIAL_PHONG_H
#include <TestLib/RTMaterial.h>
#include <TestLib/RTProperties.h>
#include <variant>
#include <string>
namespace test
{
	class RTPhongReader;
	class RTPhong : public RTMaterial
	{
	public:
		RTPhong()noexcept;
		virtual auto GetTypeName()   const noexcept -> RTString override;
		virtual auto GetPluginName() const noexcept -> RTString override;
		virtual auto GetID()         const noexcept -> RTString override;
		virtual auto GetProperties() const noexcept -> const RTProperties & override;
		virtual auto GetJsonAsData() const noexcept ->       nlohmann::json override;
		virtual void SetID(const RTString& id)noexcept;
		auto GetDiffuseReflectance() const noexcept -> std::variant<RTTexturePtr, RTColor, RTFloat>;
		auto GetSpecularReflectance()const noexcept -> std::variant<RTTexturePtr, RTColor, RTFloat>;
		auto GetSpecularExponent()   const noexcept -> std::variant<RTTexturePtr, RTFloat>;
		void SetDiffuseReflectance ( const RTTexturePtr& texture)noexcept;
		void SetDiffuseReflectance ( const RTColor&		 color  )noexcept;
		void SetDiffuseReflectance ( const RTFloat&		 fv     )noexcept;
		void SetSpecularReflectance( const RTTexturePtr& texture)noexcept;
		void SetSpecularReflectance( const RTColor&		 color  )noexcept;
		void SetSpecularReflectance( const RTFloat&		 fv     )noexcept;
		void SetSpecularExponent   ( const RTTexturePtr& texture)noexcept;
		void SetSpecularExponent   ( const RTFloat&      fv     )noexcept;
		virtual ~RTPhong() noexcept {}
	private:
		friend class RTPhongReader;
		RTProperties m_Properties;
	};
	class RTPhongReader : public RTMaterialReader
	{
	public:
		RTPhongReader(const std::shared_ptr<RTMaterialCache>& matCache, const std::shared_ptr< RTTextureCache>& texCache)noexcept;
		virtual auto GetPluginName()const noexcept -> RTString override;
		virtual auto LoadJsonFromData(const nlohmann::json& json)noexcept -> RTMaterialPtr override;
		virtual ~RTPhongReader()noexcept;
	private:
		struct Impl;
		std::unique_ptr<Impl> m_Impl;
	};
}
#endif