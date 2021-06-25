#include "../include/SceneBuilder.h"
#include <stdexcept>

void test::MeshGroup::SetSharedResource(const MeshSharedResourcePtr& res) noexcept
{
	this->m_SharedResource = res;
}

auto test::MeshGroup::GetSharedResource() const noexcept -> MeshSharedResourcePtr
{
	return this->m_SharedResource;
}

void test::MeshGroup::SetUniqueResource(const std::string& name, const MeshUniqueResourcePtr& res) noexcept
{
	this->m_UniqueResources[name] = res;
}

auto test::MeshGroup::GetUniqueResource(const std::string& name) const -> MeshUniqueResourcePtr
{
	return this->m_UniqueResources.at(name);
}

auto test::MeshGroup::GetUniqueResources() const noexcept -> const std::unordered_map<std::string, MeshUniqueResourcePtr>&
{
	return m_UniqueResources;
}

auto test::MeshGroup::GetUniqueNames() const noexcept -> std::vector<std::string>
{
    auto uniqueNames = std::vector<std::string>();
    uniqueNames.reserve(m_UniqueResources.size());
    for (auto& [name, res] : m_UniqueResources) {
        uniqueNames.push_back(name);
    }
    return uniqueNames;
}

auto test::MeshGroup::LoadMesh(const std::string& name) const -> MeshPtr
{
	auto mesh = std::make_shared<Mesh>();
	mesh->m_SharedResource = this->GetSharedResource();
	mesh->m_UniqueResource = this->GetUniqueResource(name);
	return mesh;
}

auto test::Mesh::GetSharedResource() const noexcept -> MeshSharedResourcePtr
{
	return m_SharedResource;
}

void test::Mesh::SetUniqueResource(const MeshUniqueResourcePtr& res) noexcept
{
	m_UniqueResource = res;
}

auto test::Mesh::GetUniqueResource() const -> MeshUniqueResourcePtr
{
	return m_UniqueResource;
}

bool test::ObjMeshGroup::Load(const std::string& objFilePath, const std::string& mtlFileDir) noexcept
{
    std::string inputFile = objFilePath;
    tinyobj::ObjReaderConfig readerConfig = {};
    readerConfig.mtl_search_path = mtlFileDir;
    tinyobj::ObjReader reader = {};
    if (!reader.ParseFromFile(inputFile, readerConfig)) {
        if (!reader.Error().empty()) {
            std::cerr << "TinyObjReader: " << reader.Error();
        }
        return false;
    }
    auto meshGroup = std::make_shared<test::MeshGroup>();
    auto phongMaterials = std::vector<test::PhongMaterial>();
    if (!reader.Warning().empty()) {
        std::cout << "TinyObjReader: " << reader.Warning();
    }
    auto& attrib = reader.GetAttrib();
    auto& shapes = reader.GetShapes();
    auto& materials = reader.GetMaterials();
    {
        meshGroup->SetSharedResource(std::make_shared<test::MeshSharedResource>());
        auto& vertexBuffer = meshGroup->GetSharedResource()->vertexBuffer;
        auto& texCrdBuffer = meshGroup->GetSharedResource()->texCrdBuffer;
        auto& normalBuffer = meshGroup->GetSharedResource()->normalBuffer;
        vertexBuffer.cpuHandle.resize(attrib.vertices.size() / 3);
        texCrdBuffer.cpuHandle.resize(vertexBuffer.cpuHandle.size());
        normalBuffer.cpuHandle.resize(vertexBuffer.cpuHandle.size());
        for (size_t i = 0; i < vertexBuffer.cpuHandle.size(); ++i) {
            vertexBuffer.cpuHandle[i] = make_float3(attrib.vertices[3 * i + 0], attrib.vertices[3 * i + 1], attrib.vertices[3 * i + 2]);
        }
        for (size_t i = 0; i < shapes.size(); ++i) {
            std::unordered_map<uint32_t, uint32_t> tmpMaterials = {};
            auto uniqueResource  = std::make_shared<test::MeshUniqueResource>();
            uniqueResource->name = shapes[i].name;
            uniqueResource->triIndBuffer.cpuHandle.resize(shapes[i].mesh.num_face_vertices.size());
            for (size_t j = 0; j < shapes[i].mesh.num_face_vertices.size(); ++j) {
                tinyobj::index_t idxs[3] = {
                    shapes[i].mesh.indices[3 * j + 0],
                    shapes[i].mesh.indices[3 * j + 1],
                    shapes[i].mesh.indices[3 * j + 2]
                };
                uniqueResource->triIndBuffer.cpuHandle[j] = make_uint3(idxs[0].vertex_index, idxs[1].vertex_index, idxs[2].vertex_index);
                for (size_t k = 0; k < 3; ++k) {
                    if (idxs[k].texcoord_index >= 0) {
                        auto tx = attrib.texcoords[2 * size_t(idxs[k].texcoord_index) + 0];
                        auto ty = attrib.texcoords[2 * size_t(idxs[k].texcoord_index) + 1];
                        texCrdBuffer.cpuHandle[idxs[k].vertex_index] = make_float2(tx, ty);
                    }
                    else {
                        texCrdBuffer.cpuHandle[idxs[k].vertex_index] = make_float2(0.5, 0.5);
                    }
                }
                for (size_t k = 0; k < 3; ++k) {
                    if (idxs[k].normal_index >= 0) {
                        auto nx = attrib.normals[3 * size_t(idxs[k].normal_index) + 0];
                        auto ny = attrib.normals[3 * size_t(idxs[k].normal_index) + 1];
                        auto nz = attrib.normals[3 * size_t(idxs[k].normal_index) + 2];
                        normalBuffer.cpuHandle[idxs[k].vertex_index] = make_float3(nx, ny, nz);
                    }
                    else {
                        normalBuffer.cpuHandle[idxs[k].vertex_index] = make_float3(0.0, 1.0, 0.0);
                    }
                }
            }
            uniqueResource->matIndBuffer.cpuHandle.resize(shapes[i].mesh.material_ids.size());
            for (size_t j = 0; j < shapes[i].mesh.material_ids.size();++j){
                if (tmpMaterials.count(shapes[i].mesh.material_ids[j])!= 0) {
                    uniqueResource->matIndBuffer.cpuHandle[j] = tmpMaterials.at(shapes[i].mesh.material_ids[j]);
                }
                else {
                    int newValue                                 = tmpMaterials.size();
                    tmpMaterials[shapes[i].mesh.material_ids[j]] = newValue;
                    uniqueResource->matIndBuffer.cpuHandle[j]    = newValue;
                }
            }
            uniqueResource->materials.resize(tmpMaterials.size());
            for (auto& [Ind, RelInd] : tmpMaterials) {
                uniqueResource->materials[RelInd] = Ind;
            }
            meshGroup->SetUniqueResource(shapes[i].name, uniqueResource);
        }
    }
    {
        phongMaterials.resize(materials.size());
        for (size_t i = 0; i < phongMaterials.size(); ++i) {
            phongMaterials[i].name = materials[i].name;
            phongMaterials[i].diffCol = make_float3(materials[i].diffuse[0],  materials[i].diffuse[1],  materials[i].diffuse[2]);
            phongMaterials[i].specCol = make_float3(materials[i].specular[0], materials[i].specular[1], materials[i].specular[2]);
            phongMaterials[i].emitCol = make_float3(materials[i].emission[0], materials[i].emission[1], materials[i].emission[2]);
            if (!materials[i].diffuse_texname.empty()) {
                phongMaterials[i].diffTex = mtlFileDir+materials[i].diffuse_texname;
            }
            if (!materials[i].specular_texname.empty()) {
                phongMaterials[i].specTex = mtlFileDir + materials[i].specular_texname;
            }
            if (!materials[i].emissive_texname.empty()) {
                phongMaterials[i].emitTex = mtlFileDir + materials[i].emissive_texname;
            }
            if (!materials[i].specular_highlight_texname.empty()) {
                phongMaterials[i].shinTex = mtlFileDir + materials[i].specular_highlight_texname;
            }
            phongMaterials[i].shinness    = materials[i].shininess;
            phongMaterials[i].refrInd     = materials[i].ior;
        }
    }
    m_MeshGroup              = meshGroup;
    m_MaterialSet            = std::make_shared<test::MaterialSet>();
    m_MaterialSet->materials = std::move(phongMaterials);
    return true;
}
auto test::ObjMeshGroup::GetMeshGroup() const noexcept -> std::shared_ptr<MeshGroup>
{
	return m_MeshGroup;
}

auto test::ObjMeshGroup::GetMaterialSet() const noexcept -> std::shared_ptr<MaterialSet>
{
	return m_MaterialSet;
}