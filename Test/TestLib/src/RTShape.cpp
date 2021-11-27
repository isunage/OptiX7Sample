#include "..\include\TestLib\RTShape.h"
#include "Shapes/RTBox.h"
test::RTShapeCache::RTShapeCache(const std::shared_ptr<RTMaterialCache>& cache) noexcept
{
    m_MatCache = cache;
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
    if (json.is_object()) {
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
    return shapeCache;
}
