#ifndef TEST_RT_SCENE_GRAPH_SHAPE_ARRAY_H
#define TEST_RT_SCENE_GRAPH_SHAPE_ARRAY_H
#include <TestLib/RTSceneGraph.h>
#include <TestLib/RTShape.h>
#include <TestLib/RTProperties.h>
#include <variant>
#include <string>
namespace test
{
	class RTShapeArrayReader;
	class RTShapeArray : public RTSceneGraph
	{
	public:
		RTShapeArray()noexcept;
		virtual auto GetTypeName()   const noexcept -> RTString override;
		virtual auto GetPluginName() const noexcept -> RTString override;
		virtual auto GetID()         const noexcept -> RTString override;
		virtual auto GetProperties() const noexcept -> const RTProperties & override;
		virtual auto GetJsonAsData() const noexcept ->       nlohmann::json override;
		virtual auto GetTransforms() const noexcept -> RTMat4x4 override;
		virtual void SetID(const std::string& id)noexcept override;
		auto GetShapes() const noexcept    -> const std::vector<RTShapePtr>&;
		auto SetTransforms(const RTMat4x4& mat)noexcept;
		virtual ~RTShapeArray() noexcept {}
	private:
		friend class RTShapeArrayReader;
		std::vector<RTShapePtr> m_Shapes;
		RTProperties            m_Properties;
	};

	class RTShapeArrayReader : public RTSceneGraphReader
	{
	public:
		RTShapeArrayReader(const std::shared_ptr< RTSceneGraphCache>& gphCache, const std::shared_ptr<RTShapeCache>& shpCache)noexcept;
		virtual auto GetPluginName()const noexcept -> RTString override;
		virtual auto LoadJsonFromData(const nlohmann::json& json)noexcept -> RTSceneGraphPtr override;
		virtual ~RTShapeArrayReader()noexcept;
	private:
		struct Impl;
		std::unique_ptr<Impl> m_Impl;
	};
}
#endif