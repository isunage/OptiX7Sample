#include "RTObjMesh.h"
#include <TestLib/RTMaterial.h>
#include <RTLib/ext/Math/Matrix.h>
#include <RTLib/ext/VariableMap.h>
#include <tiny_obj_loader.h>
namespace test
{
	namespace internal
	{		//Material
		struct RTObjMeshMtlResource
		{
			std::string name;
			std::string filename;
			std::string ambientTex;
			std::string emissionTex;
			std::string diffuseTex;
			std::string specularTex;
			std::string shinnessTex;
			std::string normalTex;
			std::string bumpTex;
			float3      ambient;
			float3      emission;
			float3      diffuse;
			float3      specular;
			float3      transmittance;
			float       shinness;
			float       ior;
		};
		struct RTObjMeshMtlResourceList
		{
			std::vector<std::shared_ptr<RTObjMeshMtlResource>> data;
		};
		class  RTObjMeshMtlMaterial :public RTMaterial {
		public:
			RTObjMeshMtlMaterial()noexcept;
			virtual auto GetJsonData()const noexcept -> nlohmann::json override;
			virtual ~RTObjMeshMtlMaterial()noexcept;
		private:
			friend class RTObjMeshReader;
			struct Impl;
			std::unique_ptr<Impl> m_Impl;
		};
		struct RTObjMeshMtlMaterial::Impl {
			std::shared_ptr<RTObjMeshMtlResource> internalData;
		};
		RTObjMeshMtlMaterial::RTObjMeshMtlMaterial() noexcept
		{
			m_Impl = std::make_unique<RTObjMeshMtlMaterial::Impl>();
		}
		auto RTObjMeshMtlMaterial::GetJsonData() const noexcept -> nlohmann::json
		{
			nlohmann::json jsonData = {};
			if (m_Impl && m_Impl->internalData) {
				jsonData["type"] = "material";
				jsonData["plugin"] = "objMtl";
				jsonData["filename"] = m_Impl->internalData->filename;
				jsonData["name"] = m_Impl->internalData->name;
			}
			else {
				jsonData["type"] = "material";
				jsonData["plugin"] = "unknown";
			}
			return jsonData;
		}
		RTObjMeshMtlMaterial::~RTObjMeshMtlMaterial() noexcept
		{
			m_Impl.reset();
		}
		//Vertex
		struct RTObjMeshVertex
		{
			float3 position; 
			float3 normal; 
			float2 texCoord;
		};
		//SharedResource
		struct RTObjMeshSharedResource
		{
			using MaterialList = RTObjMeshMtlResourceList;
			std::vector<RTObjMeshVertex> vertexBuffer;
			std::string					 filename;
			MaterialList                 materials;
		};
		using  RTObjMeshSharedResourcePtr = std::shared_ptr<RTObjMeshSharedResource>;
		//UniqueResource
		struct RTObjMeshUniqueResource
		{
			std::vector<uint3>     triIndBuffer;
			std::vector<uint32_t>  matIndBuffer;
			std::string			   name;
		};
		using  RTObjMeshUniqueResourcePtr	 = std::shared_ptr<RTObjMeshUniqueResource>;
		using  RTObjMeshUniqueResourcePtrMap = std::unordered_map<std::string, RTObjMeshUniqueResourcePtr>;
		//ObjModelResource
		struct RTObjModelResource {
			RTObjMeshSharedResourcePtr    sharedResource;
			RTObjMeshUniqueResourcePtrMap uniqueResources;
			bool Load(const std::string& filename) noexcept
            {
                auto mtlBaseDir = std::filesystem::canonical(std::filesystem::path(filename).parent_path());
                tinyobj::ObjReaderConfig readerConfig = {};
                readerConfig.mtl_search_path = mtlBaseDir.string() + "\\";

                tinyobj::ObjReader reader = {};
                if (!reader.ParseFromFile(filename, readerConfig)) {
                    if (!reader.Error().empty()) {
                        std::cerr << "TinyObjReader: " << reader.Error();
                    }
                    return false;
                }
                if (!reader.Warning().empty()) {
                    std::cout << "TinyObjReader: " << reader.Warning();
                }
                auto& attrib    = reader.GetAttrib();
                auto& shapes    = reader.GetShapes();
                auto& materials = reader.GetMaterials();
                {
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

                    this->sharedResource           = std::make_shared<RTObjMeshSharedResource>();
                    this->sharedResource->filename = filename;

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
                    std::cout << "VertexBuffer: " << attrib.vertices.size()  / 3 << "->" << indices.size() << std::endl;
                    std::cout << "NormalBuffer: " << attrib.normals.size()   / 3 << "->" << indices.size() << std::endl;
                    std::cout << "TexCrdBuffer: " << attrib.texcoords.size() / 2 << "->" << indices.size() << std::endl;
                    this->sharedResource->vertexBuffer.resize(indices.size());

                    for (size_t i = 0; i < indices.size(); ++i) {
                        tinyobj::index_t idx = indices[i];
                        this->sharedResource->vertexBuffer[i].position = make_float3(
                            attrib.vertices[3 * idx.vertex_index + 0],
                            attrib.vertices[3 * idx.vertex_index + 1],
                            attrib.vertices[3 * idx.vertex_index + 2]
                        );
                        if (idx.normal_index   >= 0) {
                            this->sharedResource->vertexBuffer[i].normal = make_float3(
                                attrib.normals[3 * idx.normal_index + 0],
                                attrib.normals[3 * idx.normal_index + 1],
                                attrib.normals[3 * idx.normal_index + 2]
                            );
                        }
                        else {
                            this->sharedResource->vertexBuffer[i].normal = make_float3(
                                0.0f,
                                0.0f,
                                0.0f
                            );
                        }
                        if (idx.texcoord_index >= 0) {
                            this->sharedResource->vertexBuffer[i].texCoord = make_float2(
                                attrib.texcoords[2 * idx.texcoord_index + 0],
                                attrib.texcoords[2 * idx.texcoord_index + 1]
                            );
                        }
                        else {
                            this->sharedResource->vertexBuffer[i].texCoord = make_float2(
                                0.5f,0.5f
                            );
                        }
                    }

                    std::unordered_map<std::size_t, std::size_t> texCrdMap = {};
                    for (size_t i = 0; i < shapes.size(); ++i) {
                        auto uniqueResource      = std::make_shared<RTObjMeshUniqueResource>();
                        uniqueResource->name     = shapes[i].name;
                        uniqueResource->triIndBuffer.resize(shapes[i].mesh.num_face_vertices.size());
                        for (size_t j = 0; j < shapes[i].mesh.num_face_vertices.size(); ++j) {
                            uint32_t idx0 = indicesMap.at(shapes[i].mesh.indices[3 * j + 0]);
                            uint32_t idx1 = indicesMap.at(shapes[i].mesh.indices[3 * j + 1]);
                            uint32_t idx2 = indicesMap.at(shapes[i].mesh.indices[3 * j + 2]);
                            uniqueResource->triIndBuffer[j] = make_uint3(idx0, idx1, idx2);
                        }
                        uniqueResource->matIndBuffer.resize(shapes[i].mesh.material_ids.size());
                        for (size_t j = 0; j < shapes[i].mesh.material_ids.size(); ++j) {
                            uniqueResource->matIndBuffer[j] = shapes[i].mesh.material_ids[j];
                        }
                        this->uniqueResources[shapes[i].name] = uniqueResource;
                    }
					this->sharedResource->materials.data.resize(materials.size());
					for (size_t i = 0; i < this->sharedResource->materials.data.size(); ++i) {
						this->sharedResource->materials.data[i] = std::make_shared<RTObjMeshMtlResource>();
						this->sharedResource->materials.data[i]->filename    = mtlBaseDir.string() + "\\";
						this->sharedResource->materials.data[i]->name        = materials[i].name;
						this->sharedResource->materials.data[i]->ambientTex  = materials[i].ambient_texname;
						this->sharedResource->materials.data[i]->diffuseTex  = materials[i].diffuse_texname;
						this->sharedResource->materials.data[i]->emissionTex = materials[i].emissive_texname;
						this->sharedResource->materials.data[i]->specularTex = materials[i].specular_texname;
						this->sharedResource->materials.data[i]->shinnessTex = materials[i].specular_highlight_texname;
						this->sharedResource->materials.data[i]->ambient     = make_float3(
							materials[i].ambient[0], 
							materials[i].ambient[1], 
							materials[i].ambient[2]
						);
						this->sharedResource->materials.data[i]->emission    = make_float3(
							materials[i].emission[0],
							materials[i].emission[1],
							materials[i].emission[2]
						);
						this->sharedResource->materials.data[i]->diffuse     = make_float3(
							materials[i].diffuse[0],
							materials[i].diffuse[1],
							materials[i].diffuse[2]
						);
						this->sharedResource->materials.data[i]->specular    = make_float3(
							materials[i].specular[0],
							materials[i].specular[1],
							materials[i].specular[2]
						);
						this->sharedResource->materials.data[i]->transmittance = make_float3(
							materials[i].transmittance[0],
							materials[i].transmittance[1],
							materials[i].transmittance[2]
						);
						this->sharedResource->materials.data[i]->shinness = materials[i].shininess;
						this->sharedResource->materials.data[i]->ior      = materials[i].ior;
					}
                }
                return true;
            }
		};
		using  RTObjModelResourceMap = std::unordered_map<std::string, RTObjModelResource> ;
}
	//ObjMeshReader
	struct RTObjMeshReader::Impl {
		using RTObjModelResourceMap = internal::RTObjModelResourceMap;
		RTObjModelResourceMap                models;
	};
	RTObjMeshReader::RTObjMeshReader() noexcept
	{
		m_Impl                 = std::make_unique<RTObjMeshReader::Impl>();
	}
	auto RTObjMeshReader::LoadJsonData(const nlohmann::json& jsonData) const noexcept -> RTShapePtr
	{
		std::string     typeStr;
		std::string   pluginStr;
		std::string filenameStr;
		std::string meshnameStr;
		try {
			auto&     typeJson		= jsonData.at("type"    );
			auto&   pluginJson		= jsonData.at("plugin"  );
			auto& filenameJson		= jsonData.at("filename");
			auto& meshnameJson		= jsonData.at("meshname");
			    typeStr =     typeJson.get<std::string>();
			  pluginStr =   pluginJson.get<std::string>();
			filenameStr = filenameJson.get<std::string>();
			meshnameStr = meshnameJson.get<std::string>();
		}
		catch (...) {
			return nullptr;
		}
		if (typeStr != "shape" || pluginStr != "objModel") {
			return nullptr;
		}
		rtlib::Matrix4x4 transforms = rtlib::Matrix4x4::Identity();
		try {
			auto& transformsJson = jsonData.at("transforms");
			auto  transformsData = transformsJson.get<std::vector<float>>();
			transforms(0, 0) = transformsData[4 * 0 + 0];
			transforms(1, 0) = transformsData[4 * 0 + 1];
			transforms(2, 0) = transformsData[4 * 0 + 2];
			transforms(3, 0) = transformsData[4 * 0 + 3];
			transforms(0, 1) = transformsData[4 * 1 + 0];
			transforms(1, 1) = transformsData[4 * 1 + 1];
			transforms(2, 1) = transformsData[4 * 1 + 2];
			transforms(3, 1) = transformsData[4 * 1 + 3];
			transforms(0, 2) = transformsData[4 * 2 + 0];
			transforms(1, 2) = transformsData[4 * 2 + 1];
			transforms(2, 2) = transformsData[4 * 2 + 2];
			transforms(3, 2) = transformsData[4 * 2 + 3];
			transforms(0, 3) = transformsData[4 * 3 + 0];
			transforms(1, 3) = transformsData[4 * 3 + 1];
			transforms(2, 3) = transformsData[4 * 3 + 2];
			transforms(3, 3) = transformsData[4 * 3 + 3];
		}
		catch (...) {}
		if (m_Impl->models.count(filenameStr) == 0) {
			auto modelData = internal::RTObjModelResource();
			if (!modelData.Load(filenameStr))
			{
				return nullptr;
			}
			m_Impl->models[filenameStr] = std::move(modelData);
		}
		if (m_Impl->models[filenameStr].uniqueResources.count(meshnameStr) == 0) {
			return nullptr;
		}
		auto shape = std::make_shared<RTObjMesh>();
		shape->m_Impl->sharedResource = m_Impl->models[filenameStr].sharedResource;
		shape->m_Impl->uniqueResource = m_Impl->models[filenameStr].uniqueResources[meshnameStr];
		shape->m_Impl->transforms	  = transforms;
		return shape;
	}
	RTObjMeshReader::~RTObjMeshReader()
	{
		m_Impl.reset();
	}
	//ObjMesh
	struct RTObjMesh::Impl {
		using RTObjMeshSharedResourcePtr = std::shared_ptr<internal::RTObjMeshSharedResource>;
		using RTObjMeshUniqueResourcePtr = std::shared_ptr<internal::RTObjMeshUniqueResource>;
		RTObjMeshSharedResourcePtr sharedResource;
		RTObjMeshUniqueResourcePtr uniqueResource;
		rtlib::Matrix4x4           transforms;
	};
	RTObjMesh::RTObjMesh() noexcept
	{
		m_Impl = std::make_unique<RTObjMesh::Impl>();
	}
	auto RTObjMesh::GetJsonData() const noexcept -> nlohmann::json 
	{
		// TODO: return ステートメントをここに挿入します
		nlohmann::json jsonData = {};
		if (m_Impl && m_Impl->sharedResource && m_Impl->uniqueResource) {
			jsonData[      "type"] = "shape";
			jsonData[    "plugin"] = "objMesh";
			jsonData[  "filename"] = m_Impl->sharedResource->filename;
			jsonData[      "name"] = m_Impl->uniqueResource->name;
			jsonData["transforms"] = std::vector<float>{
				m_Impl->transforms(0,0),m_Impl->transforms(1,0),m_Impl->transforms(2,0),m_Impl->transforms(3,0),
				m_Impl->transforms(0,1),m_Impl->transforms(1,1),m_Impl->transforms(2,1),m_Impl->transforms(3,1),
				m_Impl->transforms(0,2),m_Impl->transforms(1,2),m_Impl->transforms(2,2),m_Impl->transforms(3,2),
				m_Impl->transforms(0,3),m_Impl->transforms(1,3),m_Impl->transforms(2,3),m_Impl->transforms(3,3)
			};
		}
		else {
			jsonData[  "type"] = "shape";
			jsonData["plugin"] = "unknown";
		}
		return jsonData;
	}
	RTObjMesh::~RTObjMesh() noexcept
	{
		m_Impl.reset();
	}
}

