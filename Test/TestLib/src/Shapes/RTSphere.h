#ifndef TEST_RT_SHAPE_SPHERE_H
#define TEST_RT_SHAPE_SPHERE_H
#include <TestLib/RTMaterial.h>
#include <TestLib/RTShape.h>
#include <TestLib/RTProperties.h>
#include <variant>
#include <string>
namespace test
{
	class RTSphereReader;
	class RTSphere : public RTShape
	{
	public:
		RTSphere()noexcept;
		virtual auto GetTypeName()   const noexcept -> RTString override;
		virtual auto GetPluginName() const noexcept -> RTString override;
		virtual auto GetID()         const noexcept -> RTString override;
		virtual auto GetProperties() const noexcept -> const RTProperties & override;
		virtual auto GetJsonAsData() const noexcept ->       nlohmann::json override;
        virtual auto GetMaterial()   const noexcept -> RTMaterialPtr override;
		virtual void SetID(const std::string& id)noexcept override;
        auto GetCenter () const noexcept -> RTPoint;
        auto GetRadius () const noexcept -> RTFloat;
		auto GetFlipNormals() const noexcept -> RTBool   ;
        auto GetTransforms()  const noexcept -> RTMat4x4 ;
        void SetCenter(const RTPoint& center)noexcept;
        void SetRadius(const RTFloat& radius)noexcept;
		void SetFlipNormals(const RTBool  & val)noexcept ;
		auto SetTransforms(const RTMat4x4& mat)noexcept ;
		virtual ~RTSphere() noexcept {}
	private:
		friend class RTSphereReader;
        RTMaterialPtr m_Material;
		RTProperties  m_Properties;
	};

	class RTSphereReader : public RTShapeReader
	{
	public:
		RTSphereReader(const std::shared_ptr< RTMaterialCache>& matCache)noexcept;
		virtual auto GetPluginName()const noexcept -> RTString override;
		virtual auto LoadJsonFromData(const nlohmann::json& json)noexcept -> RTShapePtr override;
		virtual ~RTSphereReader()noexcept;
	private:
		struct Impl;
		std::unique_ptr<Impl> m_Impl;
	};
}
#endif