#include "RTPhong.h"

test::RTPhong::RTPhong() noexcept
{
}

auto test::RTPhong::GetTypeName() const noexcept -> RTString 
{
    return "Material";
}

auto test::RTPhong::GetPluginName() const noexcept -> RTString 
{
    return "Phong";
}

auto test::RTPhong::GetID() const noexcept -> RTString 
{
    if (m_Properties.HasString("ID")){ 
        return m_Properties.GetString("ID");
    }
    return "";
}

auto test::RTPhong::GetProperties() const noexcept -> const RTProperties & 
{
    return m_Properties;
}

auto test::RTPhong::GetJsonAsData() const noexcept -> nlohmann::json 
{
    nlohmann::json data;
    data = GetProperties().GetJsonAsData();
    data["Type"] = GetTypeName();
    data["Plugin"] = GetPluginName();
    return data;
}

auto test::RTPhong::GetDiffuseReflectance() const noexcept -> std::variant<RTTexturePtr, RTColor, RTFloat>
{
    if (m_Properties.HasTexture("DiffuseReflectance")) {
        return m_Properties.GetTexture("DiffuseReflectance");
    }
    if (m_Properties.HasColor("DiffuseReflectance")) {
        return m_Properties.GetColor("DiffuseReflectance");
    }
    if (m_Properties.HasFloat("DiffuseReflectance")) {
        return m_Properties.GetFloat("DiffuseReflectance");
    }
    return 1.0f;
}

auto test::RTPhong::GetSpecularReflectance() const noexcept -> std::variant<RTTexturePtr, RTColor, RTFloat>
{
    if (m_Properties.HasTexture("SpecularReflectance")) {
        return m_Properties.GetTexture("SpecularReflectance");
    }
    if (m_Properties.HasColor(  "SpecularReflectance")) {
        return m_Properties.GetColor("SpecularReflectance");
    }
    if (m_Properties.HasFloat(  "SpecularReflectance")) {
        return m_Properties.GetFloat("SpecularReflectance");
    }
    return 0.0f;
}

auto test::RTPhong::GetSpecularExponent() const noexcept -> std::variant<RTTexturePtr, RTFloat>
{
    if (m_Properties.HasTexture("SpecularExponent")) {
        return m_Properties.GetTexture("SpecularExponent");
    }
    if (m_Properties.HasFloat("SpecularExponent")) {
        return m_Properties.GetFloat("SpecularExponent");
    }
    return 0.0f;
}

void test::RTPhong::SetID(const RTString& id) noexcept
{
    m_Properties.SetString("ID", id);
}

void test::RTPhong::SetDiffuseReflectance(const RTTexturePtr& texture) noexcept
{
    m_Properties.SetTexture("DiffuseReflectance", texture);
}

void test::RTPhong::SetDiffuseReflectance(const RTColor& color) noexcept
{
    m_Properties.SetColor("DiffuseReflectance", color);
}

void test::RTPhong::SetDiffuseReflectance(const RTFloat& fv) noexcept
{
    m_Properties.SetFloat("DiffuseReflectance", fv);
}

void test::RTPhong::SetSpecularReflectance(const RTTexturePtr& texture) noexcept
{
    m_Properties.SetTexture("SpecularReflectance", texture);
}

void test::RTPhong::SetSpecularReflectance(const RTColor& color) noexcept
{
    m_Properties.SetColor("SpecularReflectance", color);
}

void test::RTPhong::SetSpecularReflectance(const RTFloat& fv) noexcept
{
    m_Properties.SetFloat("SpecularReflectance", fv);
}
void test::RTPhong::SetSpecularExponent(const RTTexturePtr& texture) noexcept
{
    m_Properties.SetTexture("SpecularExponent", texture);
}
void test::RTPhong::SetSpecularExponent(const RTFloat& fv) noexcept
{
    m_Properties.SetFloat("SpecularExponent", fv);
}
struct test::RTPhongReader::Impl
{
    std::weak_ptr<RTMaterialCache> matCache;
    std::weak_ptr<RTTextureCache > texCache;
};
test::RTPhongReader::RTPhongReader(const std::shared_ptr<RTMaterialCache>& matCache, const std::shared_ptr<RTTextureCache>& texCache) noexcept:test::RTMaterialReader()
{
    m_Impl = std::make_unique<test::RTPhongReader::Impl>();
    m_Impl->matCache = matCache;
    m_Impl->texCache = texCache;
}

auto test::RTPhongReader::GetPluginName() const noexcept -> RTString
{
    return "Phong";
}


auto test::RTPhongReader::LoadJsonFromData(const nlohmann::json& json) noexcept -> RTMaterialPtr
{
    if (!json.contains("Type")   || !json["Type"].is_string()   || json["Type"].get<std::string>() != "Material") {
        return nullptr;
    }

    if (!json.contains("Plugin") || !json["Plugin"].is_string() || json["Plugin"].get<std::string>() != "Phong") {
        return nullptr;
    }

    auto phong = std::make_shared<test::RTPhong>();
    
    if (!phong->m_Properties.LoadFloat(  "DiffuseReflectance", json) &&
        !phong->m_Properties.LoadColor(  "DiffuseReflectance", json) &&
        !phong->m_Properties.LoadTexture("DiffuseReflectance", json, m_Impl->texCache.lock())) {
        return nullptr;
    }
    
    if (!phong->m_Properties.LoadFloat(  "SpecularReflectance", json) &&
        !phong->m_Properties.LoadColor(  "SpecularReflectance", json) &&
        !phong->m_Properties.LoadTexture("SpecularReflectance", json, m_Impl->texCache.lock())) {
        return nullptr;
    }
    
    if (!phong->m_Properties.LoadFloat(  "SpecularExponent", json) &&
        !phong->m_Properties.LoadTexture("SpecularExponent", json   , m_Impl->texCache.lock())) {
        return nullptr;
    }
    if (phong->m_Properties.LoadString("ID", json)) {
        auto matCache = m_Impl->matCache.lock();
        if (matCache) {
            matCache->AddMaterial(phong);
        }
    }
    return phong;
}

test::RTPhongReader::~RTPhongReader() noexcept
{
}
