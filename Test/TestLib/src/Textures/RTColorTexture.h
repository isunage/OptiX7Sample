#ifndef TEST_RT_TEXTURE_COLOR_TEXTURE_H
#define TEST_RT_TEXTURE_COLOR_TEXTURE_H
#include <TestLib/RTTexture.h>
#include <TestLib/RTProperties.h>
#include <variant>
#include <string>
namespace test
{
	class RTColorTextureReader;
	class RTColorTexture : public RTTexture
	{
	public:
		RTColorTexture()noexcept;
		virtual auto GetTypeName()   const noexcept -> RTString override;
		virtual auto GetPluginName() const noexcept -> RTString override;
		virtual auto GetID()         const noexcept -> RTString override;
		virtual auto GetProperties() const noexcept -> const RTProperties & override;
		virtual auto GetJsonAsData() const noexcept ->       nlohmann::json override;
		virtual void SetID(const std::string& id)noexcept override;
		auto    GetColor()const noexcept -> RTColor;
		void    SetColor(const RTColor& color)noexcept;
		virtual ~RTColorTexture() noexcept {}
	private:
		friend class RTColorTextureReader;
		RTProperties m_Properties;
	};

	class RTColorTextureReader : public RTTextureReader
	{
	public:
		RTColorTextureReader(const std::shared_ptr< RTTextureCache>& texCache)noexcept;
		virtual auto GetPluginName()const noexcept -> RTString override;
		virtual auto LoadJsonFromData(const nlohmann::json& json)noexcept -> RTTexturePtr override;
		virtual ~RTColorTextureReader()noexcept;
	private:
		struct Impl;
		std::unique_ptr<Impl> m_Impl;
	};
}
#endif