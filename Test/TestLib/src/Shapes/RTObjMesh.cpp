#include "RTObjMesh.h"

namespace test {
    namespace internal {
        struct RTObjMeshInternalSharedResource
        {
            std::vector<float3> positions;
            std::vector<float3> normals;
            std::vector<float2> texCoords;
        };
        struct RTObjMeshInternalUniqueResource
        {
            std::vector<uint3>  triIndices;
            RTMaterialPtr       material;
        };
        class RTObjMeshInternalResourceGroup
        {
            std::shared_ptr< RTObjMeshInternalSharedResource>             sharedResource;
            std::unordered_map<RTString, RTObjMeshInternalUniqueResource> uniqueResources;
        };
    }
}

struct test::RTObjMesh::Impl {
    using SharedResourcePtr = std::shared_ptr<internal::RTObjMeshInternalSharedResource>;
    using UniqueResourcePtr = std::shared_ptr<internal::RTObjMeshInternalUniqueResource>;
    RTMaterialPtr     m_Material;
    RTProperties      m_Properties;
    SharedResourcePtr m_SharedResource;
    UniqueResourcePtr m_UniqueResource;
};

test::RTObjMesh::RTObjMesh() noexcept : RTShape()
{
    m_Impl = std::make_unique<RTObjMesh::Impl>();
}

auto test::RTObjMesh::GetTypeName() const noexcept -> RTString 
{
    return "Shape";
}

auto test::RTObjMesh::GetPluginName() const noexcept -> RTString 
{
    return "ObjMesh";
}

auto test::RTObjMesh::GetID() const noexcept -> RTString
{
    if (m_Impl->m_Properties.HasString("ID"))
    {
        return m_Impl->m_Properties.GetString("ID");
    }
    else {
        return "";
    }
}

auto test::RTObjMesh::GetProperties() const noexcept -> const RTProperties & 
{
    return m_Impl->m_Properties;
}

auto test::RTObjMesh::GetJsonAsData() const noexcept -> nlohmann::json 
{
    nlohmann::json data;
    data = GetProperties().GetJsonAsData();
    data["Type"] = GetTypeName();
    data["Plugin"] = GetPluginName();
    if (GetMaterial()) {
        if (GetMaterial()->GetPluginName() != "ObjMtl") {
            if (GetMaterial()->GetID() != "") {
                data["Material"] = GetMaterial()->GetID();
            }
            else {
                data["Material"] = GetMaterial()->GetJsonAsData();
            }
        }
    }
    return data;
}

auto test::RTObjMesh::GetMaterial() const noexcept -> RTMaterialPtr
{
    return m_Impl->m_Material;
}

void test::RTObjMesh::SetID(const std::string& id) noexcept
{
    m_Impl->m_Properties.SetString("ID", id);
}

auto test::RTObjMesh::GetFilename() const noexcept -> RTString
{
    if (m_Impl->m_Properties.HasString("Filename"))
    {
        return m_Impl->m_Properties.GetString("Filename");
    }
    else {
        return "";
    }
}

auto test::RTObjMesh::GetMeshname() const noexcept -> RTString
{
    if (m_Impl->m_Properties.HasString("Meshname"))
    {
        return m_Impl->m_Properties.GetString("Meshname");
    }
    else {
        return "";
    }
}

auto test::RTObjMesh::GetSubMeshname() const noexcept -> RTString
{
    if (m_Impl->m_Properties.HasString("SubMeshname"))
    {
        return m_Impl->m_Properties.GetString("SubMeshname");
    }
    else {
        return "";
    }
}

auto test::RTObjMesh::GetFlipNormals() const noexcept -> RTBool
{
    if (m_Impl->m_Properties.HasBool("FlipNormals")) {
        return m_Impl->m_Properties.GetBool("FlipNormals");
    }
    else
    {
        return false;
    }
}

auto test::RTObjMesh::GetTransforms() const noexcept -> RTMat4x4
{
    if (m_Impl->m_Properties.HasMat4x4("Transforms")) {
        return m_Impl->m_Properties.GetMat4x4("Transforms");
    }
    else
    {
        return RTMat4x4::Identity();
    }
}

auto test::RTObjMesh::SetFilename(const RTString& filename) noexcept
{
    m_Impl->m_Properties.SetString("Filename", filename);
}

auto test::RTObjMesh::SetMeshname(const RTString& meshname) noexcept
{
    m_Impl->m_Properties.SetString("Meshname", meshname);
}

auto test::RTObjMesh::SetSubMeshname(const RTString& submeshname) noexcept
{
    m_Impl->m_Properties.SetString("SubMeshname", submeshname);
}

void test::RTObjMesh::SetFlipNormals(const RTBool& val) noexcept
{
    m_Impl->m_Properties.SetBool("FlipNormals", val);
}

auto test::RTObjMesh::SetTransforms(const RTMat4x4& mat) noexcept
{
    m_Impl->m_Properties.SetMat4x4("Transforms", mat);
}

test::RTObjMesh::~RTObjMesh() noexcept
{
}

test::RTObjMtl::RTObjMtl() noexcept
{
}

auto test::RTObjMtl::GetTypeName() const noexcept -> RTString
{
    return "Material";
}

auto test::RTObjMtl::GetPluginName() const noexcept -> RTString
{
    return "ObjMtl";
}

auto test::RTObjMtl::GetID() const noexcept -> RTString
{
    if (m_Properties.HasString("ID")) {
        return m_Properties.GetString("ID");
    }
    return "";
}

auto test::RTObjMtl::GetProperties() const noexcept -> const RTProperties&
{
    return m_Properties;
}

auto test::RTObjMtl::GetJsonAsData() const noexcept -> nlohmann::json
{
    nlohmann::json data;
    data["Type"] = GetTypeName();
    data["Plugin"] = GetPluginName();
    data["Properties"] = GetProperties().GetJsonAsData();
    return data;
}

auto test::RTObjMtl::GetDiffuseReflectance() const noexcept -> RTTexturePtr
{
    if (m_Properties.HasTexture("DiffuseReflectance")) {
        return m_Properties.GetTexture("DiffuseReflectance");
    }
    return nullptr;
}

auto test::RTObjMtl::GetSpecularReflectance() const noexcept -> RTTexturePtr
{
    if (m_Properties.HasTexture("SpecularReflectance")) {
        return m_Properties.GetTexture("SpecularReflectance");
    }
    return nullptr;
}

auto test::RTObjMtl::GetSpecularExponent() const noexcept -> RTTexturePtr
{
    if (m_Properties.HasTexture("SpecularExponent")) {
        return m_Properties.GetTexture("SpecularExponent");
    }
    return nullptr;
}

auto test::RTObjMtl::GetTransmittance() const noexcept -> RTColor
{
    if (m_Properties.HasColor("Transmittance")) {
        return m_Properties.GetColor("Transmittance");
    }
    else {
        return make_float3(0.0f);
    }
}

auto test::RTObjMtl::GetIOR() const noexcept -> RTFloat
{
    if (m_Properties.HasFloat("IOR")) {
        return m_Properties.GetFloat("IOR");
    }
    else {
        return 1.0f;
    }
}

auto test::RTObjMtl::GetIllumMode() const noexcept -> RTInt32
{
    if (m_Properties.HasInt32("IllumMode")) {
        return m_Properties.GetInt32("IllumMode");
    }
    else {
        return 0;
    }
}

void test::RTObjMtl::SetID(const RTString& id) noexcept
{
    m_Properties.SetString("ID", id);
}

void test::RTObjMtl::SetDiffuseReflectance(const RTTexturePtr& texture) noexcept
{
    m_Properties.SetTexture("DiffuseReflectance", texture);
}

void test::RTObjMtl::SetSpecularReflectance(const RTTexturePtr& texture) noexcept
{
    m_Properties.SetTexture("SpecularReflectance", texture);
}

void test::RTObjMtl::SetSpecularExponent(const RTTexturePtr& texture) noexcept
{
    m_Properties.SetTexture("SpecularExponent", texture);
}

void test::RTObjMtl::SetTransmittance(const RTColor& color) noexcept
{
    m_Properties.SetColor("Transmittance", color);
}

void test::RTObjMtl::SetIOR(const RTFloat& fv) noexcept
{
    m_Properties.SetFloat("IOR", fv);
}

void test::RTObjMtl::SetIllumMode(const RTInt32& iv) noexcept
{
    m_Properties.SetInt32("IllumMode", iv);
}

test::RTObjMtl::~RTObjMtl() noexcept
{
}

struct test::RTObjMeshReader::Impl {
    using ResourceGroupPtr = std::shared_ptr<internal::RTObjMeshInternalResourceGroup>;
    std::unordered_map<std::string, ResourceGroupPtr> resourceGroups;
    std::weak_ptr<RTMaterialCache>                    matCache;
};

test::RTObjMeshReader::RTObjMeshReader(const std::shared_ptr<RTMaterialCache>& matCache) noexcept
{
    m_Impl = std::make_unique<test::RTObjMeshReader::Impl>();
    m_Impl->matCache = matCache;
}

auto test::RTObjMeshReader::GetPluginName() const noexcept -> RTString 
{
    return "ObjMesh";
}

auto test::RTObjMeshReader::LoadJsonFromData(const nlohmann::json& json) noexcept -> RTShapePtr 
{
    if (!json.contains("Type") || !json["Type"].is_string() || json["Type"].get<std::string>() != "Shape") {
        return nullptr;
    }

    if (!json.contains("Plugin") || !json["Plugin"].is_string() || json["Plugin"].get<std::string>() != "ObjMesh") {
        return nullptr;
    }

    auto box = std::make_shared<test::RTObjMesh>();
    if (!box->m_Impl->m_Properties.LoadMat4x4("Transforms" , json)) {
        return nullptr;
    }

    if (!box->m_Impl->m_Properties.LoadString("Filename"   , json)) {
        return nullptr;
    }
    if (!box->m_Impl->m_Properties.LoadString("Meshname"   , json)) {
        return nullptr;
    }

    if (json.contains("Material")) {
        auto& meterialJson = json["Material"];
        auto  material = m_Impl->matCache.lock()->LoadJsonFromData(meterialJson);
        if (!material) {
            return nullptr;
        }
        box->m_Impl->m_Material = material;
    }
    bool useSingleMesh = false;
    if (!box->m_Impl->m_Properties.LoadString("SubMeshname", json)) {
        //この場合複数のMaterialを含む可能性があるMesh全体を一つのShapeとして扱うことを意味する
        //Shapeごとに一つのMaterialを持つことを補償するため、外部定義のMaterialを含むことが必須
        useSingleMesh  = true;
        if (!box->m_Impl->m_Material) {
            return nullptr;
        }
    }


    return box;
}

test::RTObjMeshReader::~RTObjMeshReader() noexcept
{
}
