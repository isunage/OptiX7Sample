#include "RTGraphArray.h"

test::RTGraphArray::RTGraphArray() noexcept
{
}

auto test::RTGraphArray::GetTypeName() const noexcept -> RTString
{
	return "SceneGraph";
}

auto test::RTGraphArray::GetPluginName() const noexcept -> RTString
{
	return "GraphArray";
}

auto test::RTGraphArray::GetID() const noexcept -> RTString
{
	if (m_Properties.HasString("ID"))
	{
		return m_Properties.GetString("ID");
	}
	else {
		return "";
	}
}

auto test::RTGraphArray::GetProperties() const noexcept -> const RTProperties&
{
	return m_Properties;
}

auto test::RTGraphArray::GetJsonAsData() const noexcept -> nlohmann::json
{
	nlohmann::json data;
	data = GetProperties().GetJsonAsData();
	data["Type"] = GetTypeName();
	data["Plugin"] = GetPluginName();
	data["Graphs"] = std::vector<nlohmann::json>(m_Graphs.size());
	{
		size_t i = 0;
		for (auto& graph : m_Graphs) {
			if (graph->GetID() != "") {
				data["Graphs"][i] = graph->GetID();
			}
			else {
				data["Graphs"][i] = graph->GetJsonAsData();
			}
			++i;
		}
	}
	return data;
}

auto test::RTGraphArray::GetTransforms() const noexcept -> RTMat4x4
{
	if (m_Properties.HasMat4x4("Transforms")) {
		return m_Properties.GetMat4x4("Transforms");
	}
	else {
		return RTMat4x4::Identity();
	}
}

void test::RTGraphArray::SetID(const std::string& id) noexcept
{
	m_Properties.SetString("ID", id);
}

auto test::RTGraphArray::GetGraphs() const noexcept -> const std::vector<RTSceneGraphPtr>&
{
	// TODO: return �X�e�[�g�����g�������ɑ}�����܂�
	return m_Graphs;
}

auto test::RTGraphArray::SetTransforms(const RTMat4x4& mat) noexcept
{
	m_Properties.SetMat4x4("Transforms", mat);
}

struct test::RTGraphArrayReader::Impl {
	std::weak_ptr<RTSceneGraphCache> gphCache;
};


test::RTGraphArrayReader::RTGraphArrayReader(const std::shared_ptr<RTSceneGraphCache>& gphCache)noexcept
{
	m_Impl = std::make_unique<test::RTGraphArrayReader::Impl>();
	m_Impl->gphCache = gphCache;
}

auto test::RTGraphArrayReader::GetPluginName() const noexcept -> RTString
{
	return "GraphArray";
}

auto test::RTGraphArrayReader::LoadJsonFromData(const nlohmann::json& json) noexcept -> RTSceneGraphPtr
{
	if (!json.contains("Type") || !json["Type"].is_string() || json["Type"].get<std::string>() != "SceneGraph") {
		return nullptr;
	}

	if (!json.contains("Plugin") || !json["Plugin"].is_string() || json["Plugin"].get<std::string>() != "GraphArray") {
		return nullptr;
	}
	if (!json.contains("Graphs") || !json["Graphs"].is_array()) {
		return nullptr;
	}

	auto graphArray = std::make_shared<test::RTGraphArray>();

	graphArray->m_Properties.LoadMat4x4("Transforms", json);

	auto& graphsJson = json["Graphs"];
	if (!graphsJson.is_array()) {
		return nullptr;
	}
	for (auto& graphJson : graphsJson)
	{
		auto graph = m_Impl->gphCache.lock()->LoadJsonFromData(graphJson);
		if (!graph) {
			return nullptr;
		}
		graphArray->m_Graphs.push_back(graph);
	}
	return graphArray;
}

test::RTGraphArrayReader::~RTGraphArrayReader() noexcept
{
}
