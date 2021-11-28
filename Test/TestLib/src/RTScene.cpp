#include "..\include\TestLib\RTScene.h"
#include <iostream>
#include <tuple>
#include <queue>
auto test::RTScene::GetCamera() const noexcept -> const RTCameraPtr&
{
	// TODO: return ステートメントをここに挿入します
	return m_Camera;
}
auto test::RTScene::GetWorld() const noexcept -> const RTSceneGraphPtr&
{
	// TODO: return ステートメントをここに挿入します
	return m_World;
}
auto test::RTScene::GetJsonAsString() const noexcept -> std::string
{
	return GetJsonAsData().dump();
}
auto test::RTScene::GetJsonAsData() const noexcept -> nlohmann::json
{
	nlohmann::json data;
	data["Camera"  ]            = m_Camera->GetJsonAsData();
	for (auto& [name, value] : m_Materials) {
		data["Materials"][name] = value->GetJsonAsData();
	}
	for (auto& [name, value] : m_Textures) {
		data["Textures"][name]  = value->GetJsonAsData();
	}
	for (auto& [name, value] : m_Shapes) {
		data["Shapes"][name]    = value->GetJsonAsData();
	}
	for (auto& [name, value] : m_SceneGraphs) {
		data["SceneGraphs"][name] = value->GetJsonAsData();
	}
	if (m_World->GetID() == "") {
		data["World"] = m_World->GetJsonAsData();
	}
	else {
		data["World"] = m_World->GetID();
	}
	return data;
}

test::RTSceneReader::RTSceneReader() noexcept
{
	m_CamCache = test::GetDefaultCameraCache();
	m_TexCache = test::GetDefaultTextureCache();
	m_MatCache = test::GetDefaultMaterialCache(m_TexCache);
	m_ShpCache = test::GetDefaultShapeCache(m_MatCache);
	m_GphCache = test::GetDefaultSceneGraphCache(m_ShpCache, m_MatCache);
}

auto test::RTSceneReader::LoadJsonFromData(const nlohmann::json& json) noexcept -> RTScenePtr
{
	auto scene = std::make_shared<RTScene>();
	if (!json.is_object()) {
		return nullptr;
	}   
	if (!json.contains("Camera")      || 
		!json.contains("Textures")    || !json["Textures"   ].is_object() ||
		!json.contains("Materials")   || !json["Materials"  ].is_object() ||
		!json.contains("Shapes")      || !json["Shapes"     ].is_object() ||
		!json.contains("SceneGraphs") || !json["SceneGraphs"].is_object() ||
		!json.contains("World")) {
		return nullptr;
	}
	{
		auto& camJson = json["Camera"];
		if (!camJson.is_object()) {
			return nullptr;
		}
		auto cam = m_CamCache->LoadJsonFromData(camJson);
		if (!cam) {
			return nullptr;
		}
		scene->m_Camera = cam;
	}
	{
		auto& texsJson = json["Textures"];
		std::queue<std::tuple<std::string, nlohmann::json>> jsonStack;
		for (auto& texJson : texsJson.items()) {
			jsonStack.push({ texJson.key(),texJson.value() });
		}
		size_t i = 0;
		while (!jsonStack.empty()) {
			if (i > 255) { 

				return nullptr;
			}
			auto top     = jsonStack.front();
			jsonStack.pop();
			auto texture = m_TexCache->LoadJsonFromData(std::get<1>(top));
			std::cout << std::get<0>(top) << (texture ? "  : SUCC" : ": FAIL") << std::endl;
			if (!texture) {
				jsonStack.push(top);
			}
			else {
				if (texture->GetID() == "")
				{
					texture->SetID(std::get<0>(top));
					m_TexCache->AddTexture(texture);
				}
				scene->m_Textures[texture->GetID()] = texture;
			}
			++i;
		}
	}
	{
		auto& matsJson = json["Materials"];
		std::queue<std::tuple<std::string, nlohmann::json>> jsonStack;
		for (auto& matJson : matsJson.items()) {
			jsonStack.push({ matJson.key(),matJson.value() });
		}
		size_t i = 0;
		while (!jsonStack.empty()) {
			if (i > 255) { return nullptr; }
			auto top = jsonStack.front();
			jsonStack.pop();
			auto material = m_MatCache->LoadJsonFromData(std::get<1>(top));
			std::cout << std::get<0>(top) << (material ? "  : SUCC" : ": FAIL") << std::endl;
			if (!material) {
				jsonStack.push(top);
			}
			else {
				if (material->GetID() == "")
				{
					material->SetID(std::get<0>(top));
					m_MatCache->AddMaterial(material);
				}
				scene->m_Materials[material->GetID()] = material;
			}
			++i;
		}
	}
	{
		auto& shpsJson = json["Shapes"];
		std::queue<std::tuple<std::string, nlohmann::json>> jsonStack;
		for (auto& shpJson : shpsJson.items()) {
			jsonStack.push({ shpJson.key(),shpJson.value() });
		}
		size_t i = 0;
		while (!jsonStack.empty()) {
			if (i > 255) { return nullptr; }
			auto top = jsonStack.front();
			jsonStack.pop();
			auto shape = m_ShpCache->LoadJsonFromData(std::get<1>(top));
			std::cout << std::get<0>(top) << (shape ? "  : SUCC" : ": FAIL") << std::endl;
			if (!shape) {
				jsonStack.push(top);
			}
			else {
				if (shape->GetID() == "")
				{
					shape->SetID(std::get<0>(top));
					m_ShpCache->AddShape(shape);
				}
				scene->m_Shapes[shape->GetID()] = shape;
			}
			++i;
		}
	}
	{
		auto& gphsJson = json["SceneGraphs"];
		std::queue<std::tuple<std::string, nlohmann::json>> jsonStack;
		for (auto& gphJson : gphsJson.items()) {
			jsonStack.push({ gphJson.key(),gphJson.value() });
		}
		size_t i = 0;
		while (!jsonStack.empty()) {
			if (i > 255) { return nullptr; }
			auto top = jsonStack.front();
			jsonStack.pop();
			auto graph = m_GphCache->LoadJsonFromData(std::get<1>(top));
			std::cout << std::get<0>(top) << (graph ? "  : SUCC" : ": FAIL") << std::endl;
			if (!graph) {
				jsonStack.push(top);
			}
			else {
				if (graph->GetID() == "")
				{
					graph->SetID(std::get<0>(top));
					m_GphCache->AddSceneGraph(graph);
				}
				scene->m_SceneGraphs[graph->GetID()] = graph;
			}
			++i;
		}
	}
	{
		auto& wldJson = json["World"];
		auto world = m_GphCache->LoadJsonFromData(wldJson);
		if (!world) {
			return nullptr;
		}
		scene->m_World = world;
	}
	return scene;
}

test::RTSceneReader::~RTSceneReader() noexcept
{
}
