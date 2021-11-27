#ifndef TEST_RT_TEXTURE_BLEND_TEXTURE_H
#define TEST_RT_TEXTURE_BLEND_TEXTURE_H
#include <TestLib/RTTexture.h>
#include <TestLib/RTProperties.h>
#include <variant>
#include <string>
namespace test
{
	class RTBlendTextureReader;
	class RTBlendTexture : public RTTexture
	{
	public:
		RTBlendTexture()noexcept;
		virtual auto GetTypeName() const noexcept -> RTString override;
		virtual auto GetPluginName() const noexcept -> RTString override;
		virtual auto GetID()         const noexcept -> RTString override;
		virtual auto GetProperties() const noexcept -> const RTProperties & override;
		virtual auto GetJsonAsData() const noexcept ->       nlohmann::json override;
		virtual void SetID(const std::string& id)noexcept override;
		auto    GetTexture0()const noexcept -> std::variant<RTColor, RTTexturePtr>;
		auto    GetTexture1()const noexcept -> std::variant<RTColor, RTTexturePtr>;
		auto    GetBlendMode()const noexcept-> RTString;
		void    SetTexture0(const RTColor& color)noexcept;
		void    SetTexture0(const RTTexturePtr& texture)noexcept;
		void    SetTexture1(const RTColor& color)noexcept;
		void    SetTexture1(const RTTexturePtr& texture)noexcept;
		void    SetBlendMode(const RTString& mode)noexcept;
		virtual ~RTBlendTexture() noexcept {}
	private:
		friend class RTBlendTextureReader;
		RTProperties m_Properties;
	};

	class RTBlendTextureReader : public RTTextureReader
	{
	public:
		RTBlendTextureReader(const std::shared_ptr< RTTextureCache>& texCache)noexcept;
		virtual auto GetPluginName()const noexcept -> RTString override;
		virtual auto LoadJsonFromData(const nlohmann::json& json)noexcept -> RTTexturePtr override;
		virtual ~RTBlendTextureReader()noexcept;
	private:
		struct Impl;
		std::unique_ptr<Impl> m_Impl;
	};
}
#endif