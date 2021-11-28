#include "..\include\TestLib\RTShape.h"
#include "Shapes/RTBox.h"
#include "Shapes/RTSphere.h"
#include "Shapes/RTObjMesh.h"
#include "Shapes/RTInstancingShape.h"
test::RTShapeCache::RTShapeCache(const std::shared_ptr<RTMaterialCache>& cache) noexcept
{
    m_MatCache = cache;
}

bool test::RTShapeCache::AddShape(const RTShapePtr& shape) noexcept
{
    if (!shape) { return false; }
    if (!shape->GetProperties().HasString("ID")) {
        return false;
    }
    if (shape->GetProperties().GetString("ID") == "") {
        return false;
    }
    m_Shapes[shape->GetProperties().GetString("ID")] = shape;
    return true;
}

bool test::RTShapeCache::HasShape(const std::string& id) const noexcept
{
    return m_Shapes.count(id)>0;
}

auto test::RTShapeCache::GetShape(const std::string& id) const -> RTShapePtr
{
    return m_Shapes.at(id);
}

bool test::RTShapeCache::AddReader(const RTShapeReaderPtr& reader) noexcept
{
    if (m_Readers.count(reader->GetPluginName())>0)
    {
        return false;
    }
    else {
        m_Readers[reader->GetPluginName()] = reader;
        return true;
    }
}

bool test::RTShapeCache::HasReader(const std::string& id) const noexcept
{
    return m_Readers.count(id) > 0;
}

auto test::RTShapeCache::GetReader(const std::string& id) const -> RTShapeReaderPtr
{
    return m_Readers.at(id);
}

auto test::RTShapeCache::LoadJsonFromData(const nlohmann::json& json) noexcept -> RTShapePtr
{
    if (json.is_string()) {
        if (HasShape(json.get<std::string>())) {
            return GetShape(json.get<std::string>());
        }
        else {
            //‚½‚¾‚µA‚Ü‚¾“Ç‚Ýž‚Ü‚ê‚Ä‚¢‚È‚¢‰Â”\«‚ª‚ ‚é
            return nullptr;
        }
    }
    else if (json.is_object()) {
        RTShapePtr ptr;
        for (auto& [id, reader] : m_Readers) {
            if (ptr = reader->LoadJsonFromData(json)) {
                return ptr;
            }
        }
    }
    return nullptr;
}

auto test::GetDefaultShapeCache(const std::shared_ptr<RTMaterialCache>& matCache) noexcept -> std::shared_ptr<RTShapeCache>
{
    auto shapeCache = std::make_shared<RTShapeCache>(matCache);
    shapeCache->AddReader(std::make_shared<RTBoxReader>(matCache));
    shapeCache->AddReader(std::make_shared<RTSphereReader>(matCache));
    shapeCache->AddReader(std::make_shared<RTObjMeshReader>(matCache));
    shapeCache->AddReader(std::make_shared<RTInstancingShapeReader>(shapeCache,matCache));
    return shapeCache;
}
