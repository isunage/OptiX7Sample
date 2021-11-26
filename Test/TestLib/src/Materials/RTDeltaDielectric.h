#ifndef TEST_RT_MATERIAL_DELTA_DIELECTRIC_H
#define TEST_RT_MATERIAL_DELTA_DIELECTRIC_H
#include <TestLib/RTMaterial.h>
#include <TestLib/RTProperties.h>
#include <variant>
#include <string>
namespace test
{
	class RTDeltaDielectricReader;
	class RTDeltaDielectric : public RTMaterial
	{
	public:
		RTDeltaDielectric()noexcept;
		virtual auto GetTypeName()  const noexcept -> RTString override;
		virtual auto GetPluginName()const noexcept -> RTString override;
		virtual auto GetID()        const noexcept -> RTString override;
		virtual auto GetProperties()const noexcept -> const RTProperties & override;
		virtual auto GetJsonAsData()const noexcept ->       nlohmann::json override;
		auto GetReflectance()const noexcept   -> std::variant<RTTexturePtr, RTColor, RTFloat>;
		auto GetTransmittance()const noexcept -> std::variant<RTColor, RTFloat>;
		auto GetIOR()const noexcept -> RTFloat;
		void SetID(const RTString& id)noexcept;
		void SetReflectance(const RTTexturePtr& texture)noexcept;
		void SetReflectance(const RTColor& color)noexcept;
		void SetReflectance(const RTFloat& fv)noexcept;
		void SetTransmittance(const RTColor& color)noexcept;
		void SetTransmittance(const RTFloat& fv)noexcept;
		void SetIOR(const RTFloat& fv)noexcept;
		virtual ~RTDeltaDielectric() noexcept {}
	private:
		friend class RTDeltaDielectricReader;
		RTProperties m_Properties;
	};
	class RTDeltaDielectricReader : public RTMaterialReader
	{
	public:
		RTDeltaDielectricReader(const std::shared_ptr<RTMaterialCache>& matCache, const std::shared_ptr< RTTextureCache>& texCache)noexcept;
		virtual auto LoadJsonFromData(const nlohmann::json& json)noexcept -> RTMaterialPtr override;
		virtual ~RTDeltaDielectricReader()noexcept;
	private:
		struct Impl;
		std::unique_ptr<Impl> m_Impl;
	};
}
#endif