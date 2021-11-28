#include "RTShapeArray.h"

test::RTShapeArray::RTShapeArray() noexcept
{
}

auto test::RTShapeArray::GetTypeName() const noexcept -> RTString 
{
	return "SceneGraph";
}

auto test::RTShapeArray::GetPluginName() const noexcept -> RTString 
{
	return "ShapeArray";
}

auto test::RTShapeArray::GetID() const noexcept -> RTString 
{
	if (m_Properties.HasString("ID"))
	{
		return m_Properties.GetString("ID");
	}
	else {
		return "";
	}
}

auto test::RTShapeArray::GetProperties() const noexcept -> const RTProperties & 
{
	return m_Properties;
}

auto test::RTShapeArray::GetJsonAsData() const noexcept -> nlohmann::json 
{
	nlohmann::json data;
	data           = GetProperties().GetJsonAsData();
	data["Type"]   = GetTypeName();
	data["Plugin"] = GetPluginName();
	data["Shapes"] = nlohmann::json();
	for (auto& shape : m_Shapes) {
		if (shape->GetID() != "") {
			data["Shapes"].push_back(shape->GetID());
		}
		else {
			data["Shapes"].push_back(shape->GetJsonAsData());
		}
		
	}
	return data;
}

auto test::RTShapeArray::GetTransforms() const noexcept -> RTMat4x4 
{
	if (m_Properties.HasMat4x4("Transforms")) {
		return m_Properties.GetMat4x4("Transforms");
	}
	else {
		return RTMat4x4::Identity();
	}
}

void test::RTShapeArray::SetID(const std::string& id) noexcept
{
	m_Properties.SetString("ID", id);
}

auto test::RTShapeArray::GetShapes() const noexcept -> const std::vector<RTShapePtr>&
{
	// TODO: return ステートメントをここに挿入します
	return m_Shapes;
}

auto test::RTShapeArray::SetTransforms(const RTMat4x4& mat) noexcept
{
	m_Properties.SetMat4x4("Transforms", mat);
}

struct test::RTShapeArrayReader::Impl {
	std::weak_ptr<RTShapeCache     > shpCache;
	std::weak_ptr<RTSceneGraphCache> gphCache;
};


test::RTShapeArrayReader::RTShapeArrayReader(const std::shared_ptr<RTSceneGraphCache>& gphCache, const std::shared_ptr<RTShapeCache>& shpCache) noexcept
{
	m_Impl = std::make_unique<test::RTShapeArrayReader::Impl>();
	m_Impl->gphCache = gphCache;
	m_Impl->shpCache = shpCache;
}

auto test::RTShapeArrayReader::GetPluginName() const noexcept -> RTString 
{
	return "ShapeArray";
}

auto test::RTShapeArrayReader::LoadJsonFromData(const nlohmann::json& json) noexcept -> RTSceneGraphPtr 
{
    if (!json.contains("Type") || !json["Type"].is_string() || json["Type"].get<std::string>() != "SceneGraph") {
        return nullptr;
    }

    if (!json.contains("Plugin") || !json["Plugin"].is_string() || json["Plugin"].get<std::string>() != "ShapeArray") {
        return nullptr;
    }
    if (!json.contains("Shapes") || !json["Shapes"].is_array()) {
        return nullptr;
    }

    auto shapeArray = std::make_shared<test::RTShapeArray>();

    shapeArray->m_Properties.LoadMat4x4("Transforms", json);

	auto& shapesJson = json["Shapes"];
	if (!shapesJson.is_array()) {
		return nullptr;
	}
	for (auto& shapeJson : shapesJson)
	{
		auto shape = m_Impl->shpCache.lock()->LoadJsonFromData(shapeJson);
		if (!shape) {
			return nullptr;
		}
		shapeArray->m_Shapes.push_back(shape);
	}
    return shapeArray;
}

test::RTShapeArrayReader::~RTShapeArrayReader() noexcept
{
}
