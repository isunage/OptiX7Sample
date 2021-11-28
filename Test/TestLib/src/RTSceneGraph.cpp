#include "..\include\TestLib\RTSceneGraph.h"
#include "SceneGraphs/RTGraphArray.h"
#include "SceneGraphs/RTShapeArray.h"
#include "SceneGraphs/RTInstancingGraph.h"

auto test::GetDefaultSceneGraphCache(const std::shared_ptr<RTShapeCache>& shpCache, const std::shared_ptr<RTMaterialCache>& matCache) noexcept -> std::shared_ptr<RTSceneGraphCache>
{
    auto cache = std::make_shared<RTSceneGraphCache>();
    cache->AddReader(std::make_shared<test::RTGraphArrayReader>(cache));
    cache->AddReader(std::make_shared<test::RTShapeArrayReader>(cache, shpCache));
    cache->AddReader(std::make_shared<test::RTInstancingGraphReader>(cache,matCache));
    return cache;
}

test::RTSceneGraphCache::RTSceneGraphCache() noexcept
{
}

bool test::RTSceneGraphCache::AddSceneGraph(const RTSceneGraphPtr& graph) noexcept
{
	if (!graph) { return false; }
	if (!graph->GetProperties().HasString("ID")) {
		return false;
	}
	if (graph->GetProperties().GetString("ID") == "") {
		return false;
	}
	m_Graphs[graph->GetProperties().GetString("ID")] = graph;
	return true;
}

bool test::RTSceneGraphCache::HasSceneGraph(const std::string& id) const noexcept
{
	return m_Graphs.count(id);
}

auto test::RTSceneGraphCache::GetSceneGraph(const std::string& id) const -> RTSceneGraphPtr
{
	return m_Graphs.at(id);
}

bool test::RTSceneGraphCache::AddReader(const RTSceneGraphReaderPtr& reader) noexcept
{
	if (m_Readers.count(reader->GetPluginName()) > 0) {
		return false;
	}
	else {
		m_Readers[reader->GetPluginName()] = reader;
		return true;
	}
}

bool test::RTSceneGraphCache::HasReader(const std::string& id) const noexcept
{
	return m_Readers.count(id) > 0;
}

auto test::RTSceneGraphCache::GetReader(const std::string& id) const -> RTSceneGraphReaderPtr
{
	return m_Readers.at(id);
}

auto test::RTSceneGraphCache::LoadJsonFromData(const nlohmann::json& json) noexcept -> RTSceneGraphPtr
{
	if (json.is_string()) {
		if (HasSceneGraph(json.get<std::string>())) {
			return GetSceneGraph(json.get<std::string>());
		}
		else {
			//�������A�܂��ǂݍ��܂�Ă��Ȃ��\��������
			return nullptr;
		}
	}
	else if (json.is_object()) {
		RTSceneGraphPtr ptr;
		for (auto& [id, reader] : m_Readers) {
			if (ptr = reader->LoadJsonFromData(json)) {
				return ptr;
			}
		}
	}
	return nullptr;
}

test::RTSceneGraphCache::~RTSceneGraphCache() noexcept
{
}
