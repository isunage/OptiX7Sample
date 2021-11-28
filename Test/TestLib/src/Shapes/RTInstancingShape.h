#ifndef TEST_RT_SHAPE_INSTANCING_SHAPE_H
#define TEST_RT_SHAPE_INSTANCING_SHAPE_H
#include <TestLib/RTMaterial.h>
#include <TestLib/RTShape.h>
#include <TestLib/RTProperties.h>
#include <variant>
#include <string>
namespace test
{
	class RTInstancingShapeReader;
	class RTInstancingShape : public RTShape
	{
	public:
		RTInstancingShape()noexcept;
		virtual auto GetTypeName()   const noexcept -> RTString override;
		virtual auto GetPluginName() const noexcept -> RTString override;
		virtual auto GetID()         const noexcept -> RTString override;
		virtual auto GetProperties() const noexcept -> const RTProperties & override;
		virtual auto GetJsonAsData() const noexcept ->       nlohmann::json override;
        virtual auto GetMaterial()   const noexcept -> RTMaterialPtr override;
		virtual auto GetFlipNormals()const noexcept -> RTBool   override;
		virtual auto GetTransforms() const noexcept -> RTMat4x4 override;
		virtual void SetID(const std::string& id)noexcept override;
        auto GetBaseShape () const noexcept    -> RTShapePtr;
        void SetBaseShape (const RTShapePtr   & shp)noexcept;
		void SetFlipNormals(const RTBool& val)noexcept;
		auto SetTransforms(const RTMat4x4& mat)noexcept;
		virtual ~RTInstancingShape() noexcept {}
	private:
		friend class RTInstancingShapeReader;
        RTMaterialPtr m_Material;
        RTShapePtr    m_BaseShape;
		RTProperties  m_Properties;
	};

	class RTInstancingShapeReader : public RTShapeReader
	{
	public:
		RTInstancingShapeReader(const std::shared_ptr< RTShapeCache>& shpCache,const std::shared_ptr< RTMaterialCache>& matCache)noexcept;
		virtual auto GetPluginName()const noexcept -> RTString override;
		virtual auto LoadJsonFromData(const nlohmann::json& json)noexcept -> RTShapePtr override;
		virtual ~RTInstancingShapeReader()noexcept;
	private:
		struct Impl;
		std::unique_ptr<Impl> m_Impl;
	};
}
#endif