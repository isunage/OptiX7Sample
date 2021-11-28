#ifndef TEST_RT_SCENE_GRAPH_GRAPH_ARRAY_H
#define TEST_RT_SCENE_GRAPH_GRAPH_ARRAY_H
#include <TestLib/RTSceneGraph.h>
#include <TestLib/RTShape.h>
#include <TestLib/RTProperties.h>
#include <variant>
#include <string>
namespace test
{
	class RTGraphArrayReader;
	class RTGraphArray : public RTSceneGraph
	{
	public:
		RTGraphArray()noexcept;
		virtual auto GetTypeName()   const noexcept -> RTString override;
		virtual auto GetPluginName() const noexcept -> RTString override;
		virtual auto GetID()         const noexcept -> RTString override;
		virtual auto GetProperties() const noexcept -> const RTProperties & override;
		virtual auto GetJsonAsData() const noexcept ->       nlohmann::json override;
		virtual auto GetTransforms() const noexcept -> RTMat4x4 override;
		virtual void SetID(const std::string& id)noexcept override;
		auto GetGraphs() const noexcept    -> const std::vector<RTSceneGraphPtr>&;
		auto SetTransforms(const RTMat4x4& mat)noexcept;
		virtual ~RTGraphArray() noexcept {}
	private:
		friend class RTGraphArrayReader;
		std::vector<RTSceneGraphPtr> m_Graphs;
		RTProperties                 m_Properties;
	};

	class RTGraphArrayReader : public RTSceneGraphReader
	{
	public:
		RTGraphArrayReader(const std::shared_ptr< RTSceneGraphCache>& gphCache)noexcept;
		virtual auto GetPluginName()const noexcept -> RTString override;
		virtual auto LoadJsonFromData(const nlohmann::json& json)noexcept -> RTSceneGraphPtr override;
		virtual ~RTGraphArrayReader()noexcept;
	private:
		struct Impl;
		std::unique_ptr<Impl> m_Impl;
	};
}
#endif