#ifndef TEST_RT_SCENE_GRAPH_INSTANCING_SCENE_GRAPH_H
#define TEST_RT_SCENE_GRAPH_INSTANCING_SCENE_GRAPH_H
#include <TestLib/RTMaterial.h>
#include <TestLib/RTSceneGraph.h>
#include <TestLib/RTProperties.h>
#include <variant>
#include <string>
namespace test
{
	class RTInstancingGraphReader;
	class RTInstancingGraph : public RTSceneGraph
	{
	public:
		RTInstancingGraph()noexcept;
		virtual auto GetTypeName()   const noexcept -> RTString override;
		virtual auto GetPluginName() const noexcept -> RTString override;
		virtual auto GetID()         const noexcept -> RTString override;
		virtual auto GetProperties() const noexcept -> const RTProperties & override;
		virtual auto GetTransforms() const noexcept -> rtlib::Matrix4x4     override;
		virtual auto GetJsonAsData() const noexcept ->       nlohmann::json override;
		virtual void SetID(const std::string& id)noexcept override;
		auto GetMaterials() const noexcept -> const std::vector<RTMaterialPtr> &;
		auto GetBaseGraph() const noexcept -> RTSceneGraphPtr;
		void SetBaseGraph(const RTSceneGraphPtr& shp)noexcept;
		auto SetTransforms(const RTMat4x4& mat)noexcept;
		virtual ~RTInstancingGraph() noexcept {}
	private:
		friend class RTInstancingGraphReader;
		std::vector<RTMaterialPtr> m_Materials;
		RTSceneGraphPtr            m_BaseGraph;
		RTProperties               m_Properties;
	};

	class RTInstancingGraphReader : public RTSceneGraphReader
	{
	public:
		RTInstancingGraphReader(const std::shared_ptr< RTSceneGraphCache>& gphCache, const std::shared_ptr< RTMaterialCache>& matCache)noexcept;
		virtual auto GetPluginName()const noexcept -> RTString override;
		virtual auto LoadJsonFromData(const nlohmann::json& json)noexcept -> RTSceneGraphPtr override;
		virtual ~RTInstancingGraphReader()noexcept;
	private:
		struct Impl;
		std::unique_ptr<Impl> m_Impl;
	};
}
#endif