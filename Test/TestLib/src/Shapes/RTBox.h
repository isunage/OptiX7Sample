#ifndef TEST_RT_SHAPE_BOX_H
#define TEST_RT_SHAPE_BOX_H
#include <TestLib/RTMaterial.h>
#include <TestLib/RTShape.h>
#include <TestLib/RTProperties.h>
#include <variant>
#include <string>
namespace test
{
	class RTBoxReader;
	class RTBox : public RTShape
	{
	public:
		RTBox()noexcept;
		virtual auto GetTypeName()   const noexcept -> RTString override;
		virtual auto GetPluginName() const noexcept -> RTString override;
		virtual auto GetID()         const noexcept -> RTString override;
		virtual auto GetProperties() const noexcept -> const RTProperties & override;
		virtual auto GetJsonAsData() const noexcept ->       nlohmann::json override;
        virtual auto GetMaterial()   const noexcept -> RTMaterialPtr override;
		virtual void SetID(const std::string& id)noexcept override;
		auto GetFlipNormal() const noexcept -> RTBool   ;
        auto GetTransforms() const noexcept -> RTMat4x4 ;
		void SetFlipNormal(const RTBool  & val)noexcept ;
		auto SetTransforms(const RTMat4x4& mat)noexcept ;
		virtual ~RTBox() noexcept {}
	private:
		friend class RTBoxReader;
        RTMaterialPtr m_Material;
		RTProperties  m_Properties;
	};

	class RTBoxReader : public RTShapeReader
	{
	public:
		RTBoxReader(const std::shared_ptr< RTMaterialCache>& matCache)noexcept;
		virtual auto GetPluginName()const noexcept -> RTString override;
		virtual auto LoadJsonFromData(const nlohmann::json& json)noexcept -> RTShapePtr override;
		virtual ~RTBoxReader()noexcept;
	private:
		struct Impl;
		std::unique_ptr<Impl> m_Impl;
	};
}
#endif