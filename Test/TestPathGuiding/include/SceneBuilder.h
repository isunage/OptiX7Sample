#ifndef TEST17_SCENE_BUILDER_H
#define TEST17_SCENE_BUILDER_H
#include <tiny_obj_loader.h>
#include <RTLib/core/Optix.h>
#include <RTLib/core/CUDA.h>
#include <RTLib/math/VectorFunction.h>
#include <RTLib/ext/Mesh.h>
#include <RTLib/ext/TraversalHandle.h>
#include <RTLib/ext/VariableMap.h>
#include <string>
#include <vector>
#include <array>
#include <unordered_map>
#include <memory>
namespace test {
	class  ObjMeshGroup {
    public:
        using MeshGroupPtr = rtlib::ext::MeshGroupPtr;
        using MaterialList = rtlib::ext::VariableMapList;
        using MaterialListPtr = rtlib::ext::VariableMapListPtr;
        bool Load(const std::string& objFilePath, const std::string& mtlFileDir)noexcept {
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
            auto meshGroup = std::make_shared<rtlib::ext::MeshGroup>();
            auto phongMaterials = MaterialListPtr(new MaterialList());
            if (!reader.Warning().empty()) {
                std::cout << "TinyObjReader: " << reader.Warning();
            }
            auto& attrib = reader.GetAttrib();
            auto& shapes = reader.GetShapes();
            auto& materials = reader.GetMaterials();
            {
                meshGroup->SetSharedResource(std::make_shared<rtlib::ext::MeshSharedResource>());
                auto& vertexBuffer = meshGroup->GetSharedResource()->vertexBuffer;
                auto& texCrdBuffer = meshGroup->GetSharedResource()->texCrdBuffer;
                auto& normalBuffer = meshGroup->GetSharedResource()->normalBuffer;

                struct MyHash
                {
                    MyHash()noexcept {}
                    MyHash(const MyHash&)noexcept = default;
                    MyHash(MyHash&&)noexcept = default;
                    ~MyHash()noexcept {}
                    MyHash& operator=(const MyHash&)noexcept = default;
                    MyHash& operator=(MyHash&&)noexcept = default;
                    size_t operator()(tinyobj::index_t key)const
                    {
                        size_t vertexHash = std::hash<int>()(key.vertex_index) & 0x3FFFFF;
                        size_t normalHash = std::hash<int>()(key.normal_index) & 0x1FFFFF;
                        size_t texCrdHash = std::hash<int>()(key.texcoord_index) & 0x1FFFFF;
                        return vertexHash + (normalHash << 22) + (texCrdHash << 43);
                    }
                };
                struct MyEqualTo
                {
                    using first_argument_type  = tinyobj::index_t;
                    using second_argument_type = tinyobj::index_t;
                    using result_type = bool;
                    constexpr bool operator()(const tinyobj::index_t& x, const tinyobj::index_t& y)const
                    {
                        return (x.vertex_index == y.vertex_index) && (x.texcoord_index == y.texcoord_index) && (x.normal_index == y.normal_index);
                    }
                };

                std::vector< tinyobj::index_t> indices = {};
                std::unordered_map<tinyobj::index_t, size_t, MyHash, MyEqualTo> indicesMap = {};
                for (size_t i = 0; i < shapes.size(); ++i) {
                    for (size_t j = 0; j < shapes[i].mesh.num_face_vertices.size(); ++j) {
                        for (size_t k = 0; k < 3; ++k) {
                            //tinyobj::idx
                            tinyobj::index_t idx = shapes[i].mesh.indices[3 * j + k];
                            if (indicesMap.count(idx) == 0) {
                                size_t indicesCount = std::size(indices);
                                indicesMap[idx] = indicesCount;
                                indices.push_back(idx);
                            }
                        }
                    }
                }
                std::cout << "VertexBuffer: " << attrib.vertices.size() / 3  << "->" << indices.size() << std::endl;
                std::cout << "NormalBuffer: " << attrib.normals.size() / 3   << "->" << indices.size() << std::endl;
                std::cout << "TexCrdBuffer: " << attrib.texcoords.size() / 2 << "->" << indices.size() << std::endl;
                vertexBuffer.Resize(indices.size());
                texCrdBuffer.Resize(indices.size());
                normalBuffer.Resize(indices.size());

                for (size_t i = 0; i < indices.size(); ++i) {
                    tinyobj::index_t idx = indices[i];
                    vertexBuffer[i] = make_float3(
                        attrib.vertices[3 * idx.vertex_index + 0],
                        attrib.vertices[3 * idx.vertex_index + 1],
                        attrib.vertices[3 * idx.vertex_index + 2]);
                    if (idx.normal_index >= 0) {
                        normalBuffer[i] = make_float3(
                            attrib.normals[3 * idx.normal_index + 0],
                            attrib.normals[3 * idx.normal_index + 1],
                            attrib.normals[3 * idx.normal_index + 2]);
                    }
                    else {
                        normalBuffer[i] = make_float3(0.0f, 1.0f, 0.0f);
                    }
                    if (idx.texcoord_index >= 0) {
                        texCrdBuffer[i] = make_float2(
                            attrib.texcoords[2 * idx.texcoord_index + 0],
                            attrib.texcoords[2 * idx.texcoord_index + 1]);
                    }
                    else {
                        texCrdBuffer[i] = make_float2(0.5f, 0.5f);
                    }
                }

                std::unordered_map<std::size_t, std::size_t> texCrdMap = {};
                for (size_t i = 0; i < shapes.size(); ++i) {
                    std::unordered_map<uint32_t, uint32_t> tmpMaterials = {};
                    auto uniqueResource = std::make_shared<rtlib::ext::MeshUniqueResource>();
                    uniqueResource->name = shapes[i].name;
                    uniqueResource->triIndBuffer.Resize(shapes[i].mesh.num_face_vertices.size());
                    for (size_t j = 0; j < shapes[i].mesh.num_face_vertices.size(); ++j) {
                        uint32_t idx0 = indicesMap.at(shapes[i].mesh.indices[3 * j + 0]);
                        uint32_t idx1 = indicesMap.at(shapes[i].mesh.indices[3 * j + 1]);
                        uint32_t idx2 = indicesMap.at(shapes[i].mesh.indices[3 * j + 2]);
                        uniqueResource->triIndBuffer[j] = make_uint3(idx0, idx1, idx2);
                    }
                    uniqueResource->matIndBuffer.Resize(shapes[i].mesh.material_ids.size());
                    for (size_t j = 0; j < shapes[i].mesh.material_ids.size(); ++j) {
                        if (tmpMaterials.count(shapes[i].mesh.material_ids[j]) != 0) {
                            uniqueResource->matIndBuffer[j] = tmpMaterials.at(shapes[i].mesh.material_ids[j]);
                        }
                        else {
                            int newValue = tmpMaterials.size();
                            tmpMaterials[shapes[i].mesh.material_ids[j]] = newValue;
                            uniqueResource->matIndBuffer[j] = newValue;
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
                phongMaterials->resize(materials.size());
                for (size_t i = 0; i < phongMaterials->size(); ++i) {
                    (*phongMaterials)[i].SetString("name",  materials[i].name);
                    (*phongMaterials)[i].SetUInt32("illum", materials[i].illum);
                    (*phongMaterials)[i].SetFloat3("diffCol", 
                        { materials[i].diffuse[0],
                          materials[i].diffuse[1],
                          materials[i].diffuse[2]
                        });
                    (*phongMaterials)[i].SetFloat3("specCol", 
                        { materials[i].specular[0],
                          materials[i].specular[1],
                          materials[i].specular[2] 
                        });
                    (*phongMaterials)[i].SetFloat3("tranCol", 
                        { materials[i].transmittance[0],
                          materials[i].transmittance[1] ,
                          materials[i].transmittance[2]
                        });
                    (*phongMaterials)[i].SetFloat3("emitCol",
                        { materials[i].emission[0],
                          materials[i].emission[1] ,
                          materials[i].emission[2]
                        });

                    if (!materials[i].diffuse_texname.empty()) {
                        (*phongMaterials)[i].SetString("diffTex", mtlFileDir + materials[i].diffuse_texname);
                    }
                    else {
                        (*phongMaterials)[i].SetString("diffTex", "");
                    }
                    if (!materials[i].specular_texname.empty()) {
                        (*phongMaterials)[i].SetString("specTex", mtlFileDir + materials[i].specular_texname);
                    }
                    else {
                        (*phongMaterials)[i].SetString("specTex", "");
                    }
                    if (!materials[i].emissive_texname.empty()) {
                        (*phongMaterials)[i].SetString("emitTex", mtlFileDir + materials[i].emissive_texname);
                    }
                    else {
                        (*phongMaterials)[i].SetString("emitTex", "");
                    }
                    if (!materials[i].specular_highlight_texname.empty()) {
                        (*phongMaterials)[i].SetString("shinTex", mtlFileDir + materials[i].specular_highlight_texname);
                    }
                    else {
                        (*phongMaterials)[i].SetString("shinTex", "");
                    }
                    (*phongMaterials)[i].SetFloat1("shinness", materials[i].shininess);
                    (*phongMaterials)[i].SetFloat1("refrIndx", materials[i].ior);
                }
            }
            m_MeshGroup    = meshGroup;
            m_MaterialList = phongMaterials;
            return true;
        }
        bool Load2(const std::string& objFilePath, const std::string& mtlFileDir)noexcept {
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
            auto meshGroup = std::make_shared<rtlib::ext::MeshGroup>();
            auto phongMaterials = MaterialListPtr(new MaterialList());
            if (!reader.Warning().empty()) {
                std::cout << "TinyObjReader: " << reader.Warning();
            }
            auto& attrib = reader.GetAttrib();
            auto& shapes = reader.GetShapes();
            auto& materials = reader.GetMaterials();
            {
                meshGroup->SetSharedResource(std::make_shared<rtlib::ext::MeshSharedResource>());
                auto& vertexBuffer = meshGroup->GetSharedResource()->vertexBuffer;
                auto& texCrdBuffer = meshGroup->GetSharedResource()->texCrdBuffer;
                auto& normalBuffer = meshGroup->GetSharedResource()->normalBuffer;

                struct MyHash
                {
                    MyHash()noexcept {}
                    MyHash(const MyHash&)noexcept = default;
                    MyHash(MyHash&&)noexcept = default;
                    ~MyHash()noexcept {}
                    MyHash& operator=(const MyHash&)noexcept = default;
                    MyHash& operator=(MyHash&&)noexcept = default;
                    size_t operator()(tinyobj::index_t key)const
                    {
                        size_t vertexHash = std::hash<int>()(key.vertex_index) & 0x3FFFFF;
                        size_t normalHash = std::hash<int>()(key.normal_index) & 0x1FFFFF;
                        size_t texCrdHash = std::hash<int>()(key.texcoord_index) & 0x1FFFFF;
                        return vertexHash + (normalHash << 22) + (texCrdHash << 43);
                    }
                };
                struct MyEqualTo
                {
                    using first_argument_type = tinyobj::index_t;
                    using second_argument_type = tinyobj::index_t;
                    using result_type = bool;
                    constexpr bool operator()(const tinyobj::index_t& x, const tinyobj::index_t& y)const
                    {
                        return (x.vertex_index == y.vertex_index) && (x.texcoord_index == y.texcoord_index) && (x.normal_index == y.normal_index);
                    }
                };

                std::vector< tinyobj::index_t> indices = {};
                for (size_t i = 0; i < shapes.size(); ++i) {
                    for (size_t j = 0; j < shapes[i].mesh.num_face_vertices.size(); ++j) {
                        for (size_t k = 0; k < 3; ++k) {
                            //tinyobj::idx
                            tinyobj::index_t idx = shapes[i].mesh.indices[3 * j + k];
                            indices.push_back(idx);
                        }
                    }
                }
                std::cout << "VertexBuffer: " << attrib.vertices.size() / 3 << "->" << indices.size() << std::endl;
                std::cout << "NormalBuffer: " << attrib.normals.size() / 3 << "->" << indices.size() << std::endl;
                std::cout << "TexCrdBuffer: " << attrib.texcoords.size() / 2 << "->" << indices.size() << std::endl;
                vertexBuffer.Resize(indices.size());
                texCrdBuffer.Resize(indices.size());
                normalBuffer.Resize(indices.size());

                for (size_t i = 0; i < indices.size(); ++i) {
                    tinyobj::index_t idx = indices[i];
                    vertexBuffer[i] = make_float3(
                        attrib.vertices[3 * idx.vertex_index + 0],
                        attrib.vertices[3 * idx.vertex_index + 1],
                        attrib.vertices[3 * idx.vertex_index + 2]);
                    if (idx.normal_index >= 0) {
                        normalBuffer[i] = make_float3(
                            attrib.normals[3 * idx.normal_index + 0],
                            attrib.normals[3 * idx.normal_index + 1],
                            attrib.normals[3 * idx.normal_index + 2]);
                    }
                    else {
                        normalBuffer[i] = make_float3(0.0f, 1.0f, 0.0f);
                    }
                    if (idx.texcoord_index >= 0) {
                        texCrdBuffer[i] = make_float2(
                            attrib.texcoords[2 * idx.texcoord_index + 0],
                            attrib.texcoords[2 * idx.texcoord_index + 1]);
                    }
                    else {
                        texCrdBuffer[i] = make_float2(0.5f, 0.5f);
                    }
                }
                size_t offset = 0;
                std::unordered_map<std::size_t, std::size_t> texCrdMap = {};
                for (size_t i = 0; i < shapes.size(); ++i) {
                    std::unordered_map<uint32_t, uint32_t> tmpMaterials = {};
                    auto uniqueResource = std::make_shared<rtlib::ext::MeshUniqueResource>();
                    uniqueResource->name = shapes[i].name;
                    uniqueResource->triIndBuffer.Resize(shapes[i].mesh.num_face_vertices.size());
                    for (size_t j = 0; j < shapes[i].mesh.num_face_vertices.size(); ++j) {
                        uint32_t idx0 = offset + 3 * j + 0;
                        uint32_t idx1 = offset + 3 * j + 1;
                        uint32_t idx2 = offset + 3 * j + 2;
                        uniqueResource->triIndBuffer[j] = make_uint3(idx0, idx1, idx2);
                    }
                    offset += 3 * shapes[i].mesh.num_face_vertices.size();
                    uniqueResource->matIndBuffer.Resize(shapes[i].mesh.material_ids.size());
                    for (size_t j = 0; j < shapes[i].mesh.material_ids.size(); ++j) {
                        if (tmpMaterials.count(shapes[i].mesh.material_ids[j]) != 0) {
                            uniqueResource->matIndBuffer[j] = tmpMaterials.at(shapes[i].mesh.material_ids[j]);
                        }
                        else {
                            int newValue = tmpMaterials.size();
                            tmpMaterials[shapes[i].mesh.material_ids[j]] = newValue;
                            uniqueResource->matIndBuffer[j]    = newValue;
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
                phongMaterials->resize(materials.size());
                for (size_t i = 0; i < phongMaterials->size(); ++i) {
                    (*phongMaterials)[i].SetString("name", materials[i].name);
                    (*phongMaterials)[i].SetUInt32("illum", materials[i].illum);
                    (*phongMaterials)[i].SetFloat3("diffCol",
                        { materials[i].diffuse[0],
                          materials[i].diffuse[1],
                          materials[i].diffuse[2]
                        });
                    (*phongMaterials)[i].SetFloat3("specCol",
                        { materials[i].specular[0],
                          materials[i].specular[1],
                          materials[i].specular[2]
                        });
                    (*phongMaterials)[i].SetFloat3("tranCol",
                        { materials[i].transmittance[0],
                          materials[i].transmittance[1] ,
                          materials[i].transmittance[2]
                        });
                    (*phongMaterials)[i].SetFloat3("emitCol",
                        { materials[i].emission[0],
                          materials[i].emission[1] ,
                          materials[i].emission[2]
                        });

                    if (!materials[i].diffuse_texname.empty()) {
                        (*phongMaterials)[i].SetString("diffTex", mtlFileDir + materials[i].diffuse_texname);
                    }
                    else {
                        (*phongMaterials)[i].SetString("diffTex", "");
                    }
                    if (!materials[i].specular_texname.empty()) {
                        (*phongMaterials)[i].SetString("specTex", mtlFileDir + materials[i].specular_texname);
                    }
                    else {
                        (*phongMaterials)[i].SetString("specTex", "");
                    }
                    if (!materials[i].emissive_texname.empty()) {
                        (*phongMaterials)[i].SetString("emitTex", mtlFileDir + materials[i].emissive_texname);
                    }
                    else {
                        (*phongMaterials)[i].SetString("emitTex", "");
                    }
                    if (!materials[i].specular_highlight_texname.empty()) {
                        (*phongMaterials)[i].SetString("shinTex", mtlFileDir + materials[i].specular_highlight_texname);
                    }
                    else {
                        (*phongMaterials)[i].SetString("shinTex", "");
                    }
                    (*phongMaterials)[i].SetFloat1("shinness", materials[i].shininess);
                    (*phongMaterials)[i].SetFloat1("refrIndx", materials[i].ior);
                }
            }
            m_MeshGroup = meshGroup;
            m_MaterialList = phongMaterials;
            return true;
        }
        auto GetMeshGroup()const noexcept   -> MeshGroupPtr {
            return m_MeshGroup;
        }
        auto GetMaterialList()const noexcept -> MaterialListPtr {
            return m_MaterialList;
        }
    private:
        MeshGroupPtr    m_MeshGroup    = {};
        MaterialListPtr m_MaterialList = {};
    };
}
#endif