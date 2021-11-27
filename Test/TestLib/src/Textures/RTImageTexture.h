#ifndef TEST_RT_TEXTURE_IMAGE_TEXTURE_H
#define TEST_RT_TEXTURE_IMAGE_TEXTURE_H
#include <TestLib/RTTexture.h>
#include <TestLib/RTProperties.h>
#include <variant>
#include <string>
namespace test
{
	class RTImageTextureReader;
	class RTImageTexture : public RTTexture
	{
	public:
		RTImageTexture()noexcept;
		virtual auto GetTypeName  () const noexcept -> RTString override;
		virtual auto GetPluginName() const noexcept -> RTString override;
		virtual auto GetID()         const noexcept -> RTString override;
		virtual auto GetProperties() const noexcept -> const RTProperties & override;
		virtual auto GetJsonAsData() const noexcept ->       nlohmann::json override;
		virtual void SetID(const RTString& id)noexcept;
		auto         GetFilename()   const noexcept -> RTString;
		auto         GetMagFilter()  const noexcept -> RTString;
		auto         GetMinFilter()  const noexcept -> RTString;
		auto         GetWarpModeS()  const noexcept -> RTString;
		auto         GetWarpModeT()  const noexcept -> RTString;
		void         SetFilename (const RTString& filename   )noexcept;
		void         SetMagFilter(const RTString& filtername )noexcept;
		void         SetMinFilter(const RTString& filtername )noexcept;
		void         SetWarpModeS(const RTString& warpModeS  )noexcept;
		void         SetWarpModeT(const RTString& warpModeT  )noexcept;
		virtual ~RTImageTexture() noexcept {}
	private:
		friend class RTImageTextureReader;
		RTProperties m_Properties;
	};
	
	class RTImageTextureReader : public RTTextureReader
	{
	public:
		RTImageTextureReader(const std::shared_ptr< RTTextureCache>& texCache)noexcept;
		virtual auto GetPluginName()const noexcept -> RTString override;
		virtual auto LoadJsonFromData(const nlohmann::json& json)noexcept -> RTTexturePtr override;
		virtual ~RTImageTextureReader()noexcept;
	private:
		struct Impl;
		std::unique_ptr<Impl> m_Impl;
	};
}
#endif