#ifndef TEST_RT_MATERIAL_DELTA_REFLECTION_H
#define TEST_RT_MATERIAL_DELTA_REFLECTION_H
#include <TestLib/RTMaterial.h>
#include <TestLib/RTProperties.h>
#include <variant>
#include <string>
namespace test
{
	class RTDeltaReflectionReader;
	class RTDeltaReflection : public RTMaterial
	{
	public:
		RTDeltaReflection()noexcept;
		virtual auto GetTypeName()  const noexcept -> RTString override;
		virtual auto GetPluginName()const noexcept -> RTString override;
		virtual auto GetID()        const noexcept -> RTString override;
		virtual auto GetProperties()const noexcept -> const RTProperties   & override;
		virtual auto GetJsonAsData()const noexcept ->       nlohmann::json   override;
		auto GetReflectance()const noexcept -> std::variant<RTTexturePtr, RTColor, RTFloat>;
		void SetID(const RTString& id)noexcept;
		void SetReflectance( const RTTexturePtr& texture)noexcept;
		void SetReflectance( const RTColor&      color  )noexcept;
		void SetReflectance( const RTFloat&      fv     )noexcept;
		virtual ~RTDeltaReflection() noexcept{}
	private:
		friend class RTDeltaReflectionReader;
		RTProperties m_Properties;
	};
	class RTDeltaReflectionReader : public RTMaterialReader
	{
	public:
		RTDeltaReflectionReader(const std::shared_ptr<RTMaterialCache>& matCache, const std::shared_ptr< RTTextureCache>& texCache)noexcept;
		virtual auto LoadJsonFromData(const nlohmann::json& json)noexcept -> RTMaterialPtr override;
		virtual ~RTDeltaReflectionReader()noexcept;
	private:
		struct Impl;
		std::unique_ptr<Impl> m_Impl;
	};
}
#endif