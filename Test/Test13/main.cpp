#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <stb_image.h>
#include <stb_image_write.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <RTLib/Camera.h>
#include <RTLib/Core.h>
#include <RTLib/Config.h>
#include <RTLib/VectorFunction.h>
#include <RTLib/Exceptions.h>
#include <RTLib/Utils.h>
#include <Test13Config.h>
#include <tiny_obj_loader.h>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <array>
#include <string_view>
#include "cuda/RayTrace.h"
namespace test13{
    struct AABB {
        float3 min = make_float3(FLT_MAX,FLT_MAX,FLT_MAX);
        float3 max = make_float3(   0.0f,   0.0f,   0.0f);
        void Update(const float3& vertex)noexcept{
            this->min = rtlib::min(vertex,this->min);
            this->max = rtlib::max(vertex,this->max);
        }
    };
    struct Material {
        std::string           name         = {};
        float3                diffuseCol   = {};
        std::string           diffuseTex   = {};
        float3                specularCol  = {};
        std::string           specularTex  = {};
        float                 shinness     = {};
        float                 refractiveID = 1.0f;
        float3                emissionCol  = {};
        std::string           emissionTex  = {};
        bool                  isEmission   = false;
        bool                  isSpecular   = false;
        bool                  isTransparent= false;
    };
    struct Mesh {
        std::string           name          = {};
        AABB                  aabb          = {};
        size_t                numPrimitives =  0;//=priIndices.size()=matIndices.size()
        std::vector<uint3>    priIndices    = {};
        std::vector<int>      matIndices    = {};
    }; 
    struct MeshGroup {
        std::string           name         = {};
        AABB                  aabb         = {};
        std::vector<float3>   vertices     = {};
        std::vector<float3>   normals      = {};
        std::vector<float2>   texCoords    = {};
        std::vector<Mesh>     meshes       = {};
        std::vector<Material> materials    = {};
        bool LoadObj(const char* objFilePath, const char* mtlFileDir, const char* defTexDir){
            std::string    warn;
            std::string    err;
            auto attrib    = tinyobj::attrib_t{};
            auto shapes    = std::vector<tinyobj::shape_t>();
            auto tmpMaterials = std::vector<tinyobj::material_t>();
            bool res = tinyobj::LoadObj(&attrib,&shapes,&tmpMaterials,&warn,&err,objFilePath,mtlFileDir);
            if(!warn.empty()){
                std::cout << warn << "\n";
            }
            if(!err.empty()){
                std::cout << err << "\n";
            }
            if(!res){
                return false;
            }
            this->name = objFilePath;
            this->vertices.resize( attrib.vertices.size()/3);
            this->normals.resize(  attrib.vertices.size()/3);
            this->texCoords.resize(attrib.vertices.size()/3);
            this->materials.resize(tmpMaterials.size());
            this->meshes.resize(shapes.size());
            for(size_t s=0;s<this->meshes.size();++s){
                this->meshes[s].name          = shapes[s].name;
                this->meshes[s].aabb          = AABB{};
                this->meshes[s].numPrimitives = shapes[s].mesh.num_face_vertices.size();
                this->meshes[s].priIndices.resize(this->meshes[s].numPrimitives);
                this->meshes[s].matIndices.resize(this->meshes[s].numPrimitives);
                for(size_t f=0;f<this->meshes[s].numPrimitives;++f){
                    tinyobj::index_t idx[3] = {
                        shapes[s].mesh.indices[3*f+0],
                        shapes[s].mesh.indices[3*f+1],
                        shapes[s].mesh.indices[3*f+2]
                    };
                    this->meshes[s].priIndices[f] = make_uint3(
                        idx[0].vertex_index,
                        idx[1].vertex_index,
                        idx[2].vertex_index
                    );
                    this->meshes[s].matIndices[f] = shapes[s].mesh.material_ids[f];
                    for(size_t i=0;i<3;++i){
                        this->meshes[s].aabb.Update(make_float3(
                            attrib.vertices[3*idx[i].vertex_index+0],
                            attrib.vertices[3*idx[i].vertex_index+1],
                            attrib.vertices[3*idx[i].vertex_index+2]
                        ));
                    }
                }
            }
            for(size_t v=0;v<this->vertices.size();++v){
                this->vertices[v] = make_float3(attrib.vertices[3*v+0],attrib.vertices[3*v+1],attrib.vertices[3*v+2]);
                this->aabb.Update(this->vertices[v]);
            }
            
            for(auto& shape:shapes){
                for(auto& meshInd:shape.mesh.indices){
                    if(meshInd.normal_index>0){
                        this->normals[meshInd.vertex_index] = make_float3(
                            attrib.normals[3*meshInd.normal_index+0],
                            attrib.normals[3*meshInd.normal_index+1],
                            attrib.normals[3*meshInd.normal_index+2]
                        );
                    }else{
                        this->normals[meshInd.vertex_index] = make_float3(
                            0.0f,0.0f,0.0f
                        );
                    }
                    if(meshInd.texcoord_index>0){
                        this->texCoords[meshInd.vertex_index] = make_float2(
                             attrib.texcoords[2*meshInd.texcoord_index+0],
                            -attrib.texcoords[2*meshInd.texcoord_index+1]
                        );
                    }else{
                        this->texCoords[meshInd.vertex_index] = make_float2(
                            0.5f,0.5f
                        );
                    }
                    
                }
            }
            for(size_t m=0;m<this->materials.size();++m){
                if(!tmpMaterials[m].diffuse_texname.empty()){
                    this->materials[m].diffuseCol  = make_float3(1.0f,1.0f,1.0f);
                    this->materials[m].diffuseTex  = std::string(mtlFileDir)+tmpMaterials[m].diffuse_texname;
                }else{
                    this->materials[m].diffuseCol  = make_float3(tmpMaterials[m].diffuse[0], tmpMaterials[m].diffuse[1], tmpMaterials[m].diffuse[2]);
                    this->materials[m].diffuseTex  = std::string(defTexDir)+"white.png";
                }
                if(!tmpMaterials[m].specular_texname.empty()){
                    this->materials[m].specularCol = make_float3(1.0f,1.0f,1.0f);
                    this->materials[m].specularTex = std::string(mtlFileDir)+tmpMaterials[m].specular_texname;
                }else{
                    this->materials[m].specularCol = make_float3(tmpMaterials[m].specular[0], tmpMaterials[m].specular[1], tmpMaterials[m].specular[2]);
                    this->materials[m].specularTex = std::string(defTexDir)+"white.png";
                }
                this->materials[m].shinness        = tmpMaterials[m].shininess;
                this->materials[m].refractiveID    = tmpMaterials[m].ior;
                if(!tmpMaterials[m].emissive_texname.empty()){
                    this->materials[m].emissionCol = make_float3(1.0f,1.0f,1.0f);
                    this->materials[m].emissionTex = std::string(mtlFileDir)+tmpMaterials[m].emissive_texname;
                }else{
                    this->materials[m].emissionCol = make_float3(tmpMaterials[m].emission[0], tmpMaterials[m].emission[1], tmpMaterials[m].emission[2]);
                    this->materials[m].emissionTex = std::string(defTexDir)+"white.png";
                }
            }
            return true;
        }
    };
}
int main() {
    auto mg = test13::MeshGroup();
    assert(mg.LoadObj(TEST_TEST13_DATA_PATH"/Models/Sponza/Sponza.obj",TEST_TEST13_DATA_PATH"/Models/Sponza/",TEST_TEST13_DATA_PATH"/Textures/"));
    std::cout << matC;
    return 0;
}