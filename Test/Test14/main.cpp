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
#include <Test14Config.h>
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
struct AABB {
    float3 min = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
    float3 max = make_float3(0.0f, 0.0f, 0.0f);
    void Update(const float3& vertex) {
        min = rtlib::min(min, vertex);
        max = rtlib::max(max, vertex);
    }
};
struct ShapeInfo {
    std::string        name    = {};
    std::vector<uint3> indices = {};
    uint32_t           matID   = 0;
};
struct MaterialInfo {
    std::string        diffTexName  = {};
    float3             diffColor    = {};
    std::string        specTexName  = {};
    float3             specColor    = {};
    std::string        emitTexName  = {};
    float3             emitColor    = {};
    float              shinness     = 0.0f;
    float              refractiveID = 1.0f;
    unsigned int       illum        = 0;
};
struct WindowState {
    float  curTime   = 0.0f;
    float  delTime   = 0.0f;
    float2 curCurPos = {};
    float2 delCurPos = {};
};
int main() {
    //static constexpr float3 vertices[]           = { float3{-0.5f,-0.5f,0.0f},float3{0.5f,-0.5f,0.0f},float3{0.0f,0.5f,0.0f}};
    //static constexpr uint3   indices[]           = {{0,1,2}};
    std::vector<float3>                     vertices      = {};
    std::vector<float2>                     texCoords     = {};
    std::vector<uint3>                      indices       = {};
    std::vector<ShapeInfo>                  shapeInfos    = {};
    std::vector<MaterialInfo>               materialInfos = {};
    std::unordered_map<std::string, size_t> texMap        = {};
    AABB                                    aabb          = {};
    ParallelLight                           light         = {};
    {
        std::string                         err           = {};
        std::string                         warn          = {};
        tinyobj::attrib_t                   attrib        = {};
        std::vector<tinyobj::shape_t>       shapes        = {};
        std::vector<tinyobj::material_t>    materials     = {};
        bool res = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, TEST_TEST14_DATA_PATH"/Models/CornellBox/CornellBox-Mirror.obj", TEST_TEST14_DATA_PATH"/Models/CornellBox/");
        std::cout << warn << "\n";
        std::cout << err << "\n";
        assert(res);
        {
            {
                for (const auto& shape : shapes) {
                    size_t numMeshes = shape.mesh.num_face_vertices.size();
                    std::unordered_map<size_t, size_t> materialIDMap = {};
                    std::vector<ShapeInfo>             tmpShapeInfos = {};
                    for (size_t f = 0; f < numMeshes; ++f) {
                        size_t idx = 0;
                        try {
                            idx = materialIDMap.at(shape.mesh.material_ids[f]);
                        }
                        catch (...) {
                            idx = tmpShapeInfos.size();
                            materialIDMap[shape.mesh.material_ids[f]] = idx;
                            tmpShapeInfos.push_back(ShapeInfo{});
                            tmpShapeInfos[idx].matID = shape.mesh.material_ids[f];
                            tmpShapeInfos[idx].name = shape.name + std::to_string(idx);
                        }
                        tmpShapeInfos[idx].indices.push_back(make_uint3(shape.mesh.indices[3 * f + 0].vertex_index,
                            shape.mesh.indices[3 * f + 1].vertex_index,
                            shape.mesh.indices[3 * f + 2].vertex_index));
                    }
                    for (auto&& tmpShapeInfo : tmpShapeInfos) {
                        shapeInfos.emplace_back(tmpShapeInfo);
                    }
                }
            }
            {
                size_t numVertices = attrib.vertices.size() / 3;
                vertices.resize(numVertices);
                for (size_t v = 0; v < numVertices; ++v) {
                    vertices[v].x = attrib.vertices[3 * v + 0];
                    vertices[v].y = attrib.vertices[3 * v + 1];
                    vertices[v].z = attrib.vertices[3 * v + 2];
                }
            }
            {
                texCoords.resize(vertices.size());
                for (const auto& shape : shapes) {
                    for (auto& meshInd : shape.mesh.indices) {
                        if (meshInd.texcoord_index >= 0) {
                            texCoords[meshInd.vertex_index] = make_float2(
                                attrib.texcoords[2 * meshInd.texcoord_index + 0],
                                -attrib.texcoords[2 * meshInd.texcoord_index + 1]
                            );
                        }
                        else {
                            texCoords[meshInd.vertex_index] = make_float2(
                                0.5f,
                                0.5f
                            );
                        }

                    }
                }
            }
            {
                ShapeInfo  lightShapeInfo = {};
                for (size_t v = 0; v < vertices.size(); ++v) {
                    aabb.Update(vertices[v]);
                }
            }
            {
                materialInfos.resize(materials.size());
                for (size_t i = 0; i < materialInfos.size(); ++i) {
                    if (!materials[i].diffuse_texname.empty()) {
                        materialInfos[i].diffTexName = TEST_TEST14_DATA_PATH"/Models/CornellBox/" + materials[i].diffuse_texname;
                        materialInfos[i].diffColor = make_float3(1.0f);
                    }
                    else {
                        materialInfos[i].diffTexName = TEST_TEST14_DATA_PATH"/Textures/white.png";
                        materialInfos[i].diffColor   = make_float3(materials[i].diffuse[0], materials[i].diffuse[1], materials[i].diffuse[2]);
                    }
                    if (!materials[i].specular_texname.empty()) {
                        materialInfos[i].specTexName = TEST_TEST14_DATA_PATH"/Models/CornellBox/" + materials[i].specular_texname;
                        materialInfos[i].specColor = make_float3(1.0f);
                    }
                    else {
                        materialInfos[i].specTexName = TEST_TEST14_DATA_PATH"/Textures/white.png";
                        materialInfos[i].specColor = make_float3(materials[i].specular[0], materials[i].specular[1], materials[i].specular[2]);
                    }
                    if (!materials[i].emissive_texname.empty()) {
                        materialInfos[i].emitTexName = TEST_TEST14_DATA_PATH"/Models/CornellBox/" + materials[i].emissive_texname;
                        materialInfos[i].emitColor = make_float3(1.0f);
                    }
                    else {
                        materialInfos[i].emitTexName = TEST_TEST14_DATA_PATH"/Textures/white.png";
                        materialInfos[i].emitColor   = make_float3(materials[i].emission[0], materials[i].emission[1], materials[i].emission[2]);
                    }

                    materialInfos[i].shinness       = materials[i].shininess;
                    materialInfos[i].refractiveID   = materials[i].ior;
                    materialInfos[i].illum          = materials[i].illum;
                }
            }
            {
                size_t i = 0;
                for (auto& materialInfo : materialInfos) {
                    if (texMap.count(materialInfo.diffTexName) == 0) {
                        texMap[materialInfo.diffTexName] = i;
                        ++i;
                    }
                    if (texMap.count(materialInfo.specTexName) == 0) {
                        texMap[materialInfo.specTexName] = i;
                        ++i;
                    }
                    if (texMap.count(materialInfo.emitTexName) == 0) {
                        texMap[materialInfo.emitTexName] = i;
                        ++i;
                    }
                }
            }
            //shapeInfos.resize(30);
        }
        {
            AABB      lightAABB = {};
            ShapeInfo lightShapeInfo = {};
            for (size_t i = 0; i < shapeInfos.size(); ++i) {
                if (shapeInfos[i].name == "light0") {
                    lightShapeInfo = shapeInfos[i];
                }
            }
            for (size_t i = 0; i < lightShapeInfo.indices.size(); ++i) {
                auto idx0 = lightShapeInfo.indices[i].x;
                auto idx1 = lightShapeInfo.indices[i].y;
                auto idx2 = lightShapeInfo.indices[i].z;
                auto v0 = vertices[idx0];
                auto v1 = vertices[idx1];
                auto v2 = vertices[idx2];
                lightAABB.Update(v0);
                lightAABB.Update(v1);
                lightAABB.Update(v2);
            }
            light.corner = lightAABB.min;
            light.v1 = make_float3(lightAABB.max.x - lightAABB.min.x, 0.0f, 0.0f);
            light.v2 = make_float3(0.0f, 0.0f, lightAABB.max.z - lightAABB.min.z);
            light.normal = make_float3(0.0f, -1.0f, 0.0f);
            light.emission = materialInfos[lightShapeInfo.matID].emitColor;
        }
        //shapeInfos.resize(100);
    }
    {
        int width   = 768;
        int height  = 768;
        auto cameraController = rtlib::CameraController({ 0.0f,1.0f, 5.0f });
        //��ԍŏ��ɌĂяo��
        RTLIB_CUDA_CHECK(cudaFree(0));
        RTLIB_OPTIX_CHECK(optixInit());
        //OPX�^�͂��ׂĎQ�ƌ^�őΉ�������̂̎Q�Ƃ�ێ�
        //context��copy/move�s��
#ifndef NDEBUG
        auto context = rtlib::OPXContext({ 0, 0, OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL,4 });
#else
        auto context = rtlib::OPXContext({ 0, 0, OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_OFF,4 });
#endif
        
        auto vertexBuffer = rtlib::CUDABuffer<float3>(vertices);
        auto texCrdBuffer = rtlib::CUDABuffer<float2>(texCoords);
        auto textures = std::vector<rtlib::CUDATexture2D<uchar4>>{ texMap.size() };
        {
            {
                for (auto& [TexName, TexId] : texMap) {
                    int  width, height, comp;
                    try {
                        auto img = stbi_load(TexName.c_str(), &width, &height, &comp, 4);
                        textures[TexId].allocate(width, height, cudaTextureReadMode::cudaReadModeElementType);
                        textures[TexId].upload(img, width, height);
                        stbi_image_free(img);
                    }
                    catch (std::runtime_error& err) {
                        std::cout << err.what() << "\n";
                    }
                }
            }
        }
        auto d_pVertices = reinterpret_cast<CUdeviceptr>(vertexBuffer.getDevicePtr());
        auto indexBuffers = std::vector< rtlib::CUDABuffer<uint3>>(shapeInfos.size());
        {
            for (size_t idxBuffID = 0; idxBuffID < indexBuffers.size(); ++idxBuffID) {
                indexBuffers[idxBuffID].allocate(shapeInfos[idxBuffID].indices.size());
                indexBuffers[idxBuffID].upload(shapeInfos[idxBuffID].indices.data(), shapeInfos[idxBuffID].indices.size());
            }
        }
        auto accelBuildOptions = OptixAccelBuildOptions();
        {
            accelBuildOptions.buildFlags     = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
            accelBuildOptions.motionOptions  = {};
            accelBuildOptions.operation      = OPTIX_BUILD_OPERATION_BUILD;
        }
        auto geometryFlags = std::vector<unsigned int>{
            OPTIX_GEOMETRY_FLAG_NONE
        };
        std::vector<OptixBuildInput> buildInputs(indexBuffers.size());
        {
            for (size_t idxBuffID = 0; idxBuffID < indexBuffers.size(); ++idxBuffID) {
                buildInputs[idxBuffID].type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
                buildInputs[idxBuffID].triangleArray.vertexBuffers = &d_pVertices;
                buildInputs[idxBuffID].triangleArray.numVertices = std::size(vertices);
                buildInputs[idxBuffID].triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
                buildInputs[idxBuffID].triangleArray.vertexStrideInBytes = sizeof(float3);
                buildInputs[idxBuffID].triangleArray.indexBuffer = reinterpret_cast<CUdeviceptr>(indexBuffers[idxBuffID].getDevicePtr());
                buildInputs[idxBuffID].triangleArray.numIndexTriplets = indexBuffers[idxBuffID].getCount();
                buildInputs[idxBuffID].triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
                buildInputs[idxBuffID].triangleArray.indexStrideInBytes = sizeof(uint3);
                buildInputs[idxBuffID].triangleArray.numSbtRecords = 1;
                buildInputs[idxBuffID].triangleArray.flags = geometryFlags.data();
            }
        }
        auto pipelineCompileOptions = OptixPipelineCompileOptions{};
        auto [outputBuffer, traversableHandle] = context.buildAccel(accelBuildOptions, buildInputs);
        {
            pipelineCompileOptions.usesMotionBlur = false;
            pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
            pipelineCompileOptions.numAttributeValues = 3;
            pipelineCompileOptions.numPayloadValues = 3;
            pipelineCompileOptions.pipelineLaunchParamsVariableName = "params";
            pipelineCompileOptions.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
            pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
        }
        auto cuSource = std::string();
        {
            auto cuFile = std::ifstream(TEST_TEST14_CUDA_PATH"/RayTrace.cu", std::ios::binary);
            cuSource = std::string((std::istreambuf_iterator<char>(cuFile)), (std::istreambuf_iterator<char>()));

        }
        auto ptxSource = std::string();
        {
            auto ptxFile = std::ifstream(TEST_TEST14_CUDA_PATH"/RayTrace.ptx", std::ios::binary);
            ptxSource = std::string((std::istreambuf_iterator<char>(ptxFile)), (std::istreambuf_iterator<char>()));
        }
        //context��copy�s��
        auto pipeline  = context.createPipeline(pipelineCompileOptions);
        auto program   = rtlib::NVRTCProgram(std::string(cuSource), "sampleProgram");
        {
            try {
                program.compile(rtlib::NVRTCOptions().setIncludeDirs({ RTLIB_INCLUDE_DIR ,RTLIB_OPTIX_INCLUDE_DIR, TEST_TEST14_CUDA_PATH, RTLIB_CUDA_INCLUDE_DIRS }).setOtherOptions({ RTLIB_NVRTC_OPTIONS }).get());
            }
            catch (rtlib::NVRTCException& nvrtcErr) {
                std::cerr << "Failed To NVRTC Compile Program!\n";
                std::cout << program.getLog() << std::endl;
                std::cout << nvrtcErr.what() << std::endl;
            }
            //std::cout << program.getPTX() << "\n";
        }
        auto moduleCompileOptions = OptixModuleCompileOptions{};
        {
            moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
            moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;
            moduleCompileOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
            moduleCompileOptions.numBoundValues = 0;
            moduleCompileOptions.boundValues = 0;
        }
        auto module                       = pipeline.createModule(ptxSource, moduleCompileOptions);
        auto raygenPG                     = pipeline.createRaygenPG({ module,"__raygen__rg" });
        auto missPGForRadiance            = pipeline.createMissPG(  { module,"__miss__radiance" });
        auto missPGForOccluded            = pipeline.createMissPG(  { module,"__miss__occluded" });
        auto hitgroupPGForRadianceDiffuse = pipeline.createHitgroupPG({ module,"__closesthit__radiance_for_diffuse"  }, {}, {});
        auto hitgroupPGForRadianceSpecular= pipeline.createHitgroupPG({ module,"__closesthit__radiance_for_specular" }, {}, {});
        auto hitgroupPGForRadianceEmission= pipeline.createHitgroupPG({ module,"__closesthit__radiance_for_emission" }, {}, {});
        auto hitgroupPGForOccluded        = pipeline.createHitgroupPG({ module,"__closesthit__occluded" }, {}, {});
        auto pipelineLinkOptions          = OptixPipelineLinkOptions{};
        {
            pipelineLinkOptions.maxTraceDepth = 2;
            pipelineLinkOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;
        }
        pipeline.link(pipelineLinkOptions);
        auto      raygenRecord = raygenPG.getSBTRecord<RayGenData>();
        {
            auto camera = cameraController.GetCamera(30.0f, 1.0f);
            raygenRecord.data.eye = camera.getEye();
            auto [u, v, w]        = camera.getUVW();
            raygenRecord.data.u   = u;
            raygenRecord.data.v   = v;
            raygenRecord.data.w   = w;
        }
        std::vector<rtlib::SBTRecord<MissData>> missRecords(RAY_TYPE_COUNT);
        {
            missRecords[RAY_TYPE_RADIANCE] = missPGForRadiance.getSBTRecord<MissData>();
            missRecords[RAY_TYPE_RADIANCE].data.bgColor = float4{ 0.0f,0.0f,0.0f,1.0f };
            missRecords[RAY_TYPE_OCCLUSION] = missPGForOccluded.getSBTRecord<MissData>();
        }
        std::vector<rtlib::SBTRecord<HitgroupData>> hitgroupRecords(indexBuffers.size() * RAY_TYPE_COUNT);
        {
            for (size_t idxBuffID = 0; idxBuffID < indexBuffers.size(); ++idxBuffID) {
                if (materialInfos[shapeInfos[idxBuffID].matID].shinness > 200.0f) {
                    hitgroupRecords[RAY_TYPE_COUNT * idxBuffID + RAY_TYPE_RADIANCE] = hitgroupPGForRadianceSpecular.getSBTRecord<HitgroupData>();
                }else if(idxBuffID==7){
                    hitgroupRecords[RAY_TYPE_COUNT * idxBuffID + RAY_TYPE_RADIANCE] = hitgroupPGForRadianceEmission.getSBTRecord<HitgroupData>();
                }
                else {
                    hitgroupRecords[RAY_TYPE_COUNT * idxBuffID + RAY_TYPE_RADIANCE] = hitgroupPGForRadianceDiffuse.getSBTRecord<HitgroupData>();
                }
                
                hitgroupRecords[RAY_TYPE_COUNT * idxBuffID + RAY_TYPE_RADIANCE].data.vertices  = vertexBuffer.getDevicePtr();
                hitgroupRecords[RAY_TYPE_COUNT * idxBuffID + RAY_TYPE_RADIANCE].data.indices   = indexBuffers[idxBuffID].getDevicePtr();
                hitgroupRecords[RAY_TYPE_COUNT * idxBuffID + RAY_TYPE_RADIANCE].data.texCoords = texCrdBuffer.getDevicePtr();
                hitgroupRecords[RAY_TYPE_COUNT * idxBuffID + RAY_TYPE_RADIANCE].data.diffuse   = materialInfos[shapeInfos[idxBuffID].matID].diffColor;
                auto diffTexName = materialInfos[shapeInfos[idxBuffID].matID].diffTexName;
                hitgroupRecords[RAY_TYPE_COUNT * idxBuffID + RAY_TYPE_RADIANCE].data.diffuseTex = textures[texMap[diffTexName]].getHandle();
                hitgroupRecords[RAY_TYPE_COUNT * idxBuffID + RAY_TYPE_RADIANCE].data.specular   = materialInfos[shapeInfos[idxBuffID].matID].specColor;
                auto specTexName = materialInfos[shapeInfos[idxBuffID].matID].specTexName;
                hitgroupRecords[RAY_TYPE_COUNT * idxBuffID + RAY_TYPE_RADIANCE].data.specularTex = textures[texMap[specTexName]].getHandle();
                hitgroupRecords[RAY_TYPE_COUNT * idxBuffID + RAY_TYPE_RADIANCE].data.shinness    = materialInfos[shapeInfos[idxBuffID].matID].shinness;
                hitgroupRecords[RAY_TYPE_COUNT * idxBuffID + RAY_TYPE_RADIANCE].data.emission    = materialInfos[shapeInfos[idxBuffID].matID].emitColor;
                auto emitTexName = materialInfos[shapeInfos[idxBuffID].matID].emitTexName;
                hitgroupRecords[RAY_TYPE_COUNT * idxBuffID + RAY_TYPE_RADIANCE].data.emissionTex = textures[texMap[emitTexName]].getHandle();
                hitgroupRecords[RAY_TYPE_COUNT * idxBuffID + RAY_TYPE_OCCLUSION] = hitgroupPGForOccluded.getSBTRecord<HitgroupData>();
            }
        }
        
        auto    d_RaygenBuffer          = rtlib::CUDABuffer<rtlib::SBTRecord<RayGenData>>(raygenRecord);
        auto      d_MissBuffer          = rtlib::CUDABuffer<rtlib::SBTRecord<MissData>>(missRecords);
        auto  d_HitgroupBuffer          = rtlib::CUDABuffer<rtlib::SBTRecord<HitgroupData>>(hitgroupRecords);
        OptixShaderBindingTable sbt     = {};
        sbt.raygenRecord                = reinterpret_cast<CUdeviceptr>(d_RaygenBuffer.getDevicePtr());
        sbt.missRecordBase              = reinterpret_cast<CUdeviceptr>(d_MissBuffer.getDevicePtr());
        sbt.missRecordCount             = d_MissBuffer.getCount();
        sbt.missRecordStrideInBytes     = sizeof(rtlib::SBTRecord<MissData>);
        sbt.hitgroupRecordBase          = reinterpret_cast<CUdeviceptr>(d_HitgroupBuffer.getDevicePtr());
        sbt.hitgroupRecordCount         = d_HitgroupBuffer.getCount();
        sbt.hitgroupRecordStrideInBytes = sizeof(rtlib::SBTRecord<HitgroupData>);
        CUstream stream;
        RTLIB_CU_CHECK(cuStreamCreate(&stream, CU_STREAM_DEFAULT));
        auto d_seeds                    = rtlib::CUDABuffer<unsigned int>();
        auto d_params                   = rtlib::CUDABuffer<Params>(Params{});
        auto d_accums                   = rtlib::CUDABuffer<float3>(std::vector<float3>(width * height, make_float3(0.0f, 0.0f, 0.0f)));
        {
            std::random_device rd;
            std::mt19937 mt(rd());
            std::vector<unsigned int> seeds(width * height);
            std::generate(seeds.begin(), seeds.end(), mt);
            d_seeds = rtlib::CUDABuffer<unsigned int>(seeds);
        }
        {
            constexpr std::array<float, 5 * 4> screenVertices = {
                -1.0f,-1.0f,0.0f,0.0f, 1.0f,
                 1.0f,-1.0f,0.0f,1.0f, 1.0f,
                 1.0f, 1.0f,0.0f,1.0f, 0.0f,
                -1.0f, 1.0f,0.0f,0.0f, 0.0f
            };
            constexpr std::array<uint32_t, 6>  screenIndices  = {
                0,1,2,
                2,3,0
            };
            constexpr std::string_view vsSource =
                "#version 330 core\n"
                "layout(location=0) in vec3 position;\n"
                "layout(location=1) in vec2 texCoord;\n"
                "out vec2 uv;\n"
                "void main(){\n"
                "   gl_Position = vec4(position,1.0f);\n"
                "   uv = texCoord;\n"
                "}\n";
            constexpr std::string_view fsSource =
                "#version 330 core\n"
                "uniform sampler2D tex;\n"
                "in vec2 uv;\n"
                "layout(location=0) out vec3 color;\n"
                "void main(){\n"
                "   color = texture2D(tex,uv).xyz;\n"
                "}\n";
            if (glfwInit() != GLFW_TRUE) {
                throw std::runtime_error("Failed To Initialize GLFW!");
            }
            glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API);
            glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
            glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
            glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
            GLFWwindow* window = glfwCreateWindow(width, height, "title", nullptr, nullptr);
            if (!window) {
                throw std::runtime_error("Failed To Create Window!");
            }
            WindowState windowState = {};
            glfwSetWindowUserPointer(window, &windowState);
            glfwMakeContextCurrent(window);
            if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
                throw std::runtime_error("Failed To Load GLAD!");
            }
            auto program = rtlib::GLProgram();
            {
                program.create();
                auto vs = rtlib::GLVertexShader(std::string(vsSource));
                auto fs = rtlib::GLFragmentShader(std::string(fsSource));
                if (!vs.compile()) {
                    std::cout << vs.getLog() << std::endl;
                }
                if (!fs.compile()) {
                    std::cout << fs.getLog() << std::endl;
                }
                program.attach(vs);
                program.attach(fs);
                if (!program.link()) {
                    std::cout << program.getLog() << std::endl;
                }
            }
            auto screenVBO = rtlib::GLBuffer(screenVertices,        GL_ARRAY_BUFFER, GL_STATIC_DRAW);
            auto screenIBO = rtlib::GLBuffer(screenIndices, GL_ELEMENT_ARRAY_BUFFER, GL_STATIC_DRAW);
            auto d_frames  = rtlib::GLInteropBuffer<uchar4>(width * height, GL_PIXEL_UNPACK_BUFFER, GL_DYNAMIC_DRAW);
            auto glTexture = rtlib::GLTexture2D<uchar4>();
            {
                glTexture.allocate({ (size_t)width, (size_t)height });
                glTexture.setParameteri(GL_TEXTURE_MAG_FILTER, GL_LINEAR, false);
                glTexture.setParameteri(GL_TEXTURE_MIN_FILTER, GL_LINEAR, false);
                glTexture.setParameteri(GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE, false);
                glTexture.setParameteri(GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE, false);
            }
            auto screenVAO = GLuint(0);
            {
                glGenVertexArrays(1, &screenVAO);
                glBindVertexArray(screenVAO);
                screenVBO.bind();
                screenIBO.bind();
                glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 5, (void*)(sizeof(float)*0));
                glEnableVertexAttribArray(0);
                glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 5, (void*)(sizeof(float)*3));
                glEnableVertexAttribArray(1);
                glBindVertexArray(0);
                screenVBO.unbind();
                screenIBO.unbind();
            }
            
            glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
            GLint texLoc = program.getUniformLocation("tex");
            auto params            = Params();
            params.accumBuffer     = d_accums.getDevicePtr();
            params.seed            = d_seeds.getDevicePtr();
            params.width           = width;
            params.height          = height;
            params.samplePerLaunch = 10;
            params.samplePerALL    = 0;
            params.gasHandle       = traversableHandle;
            params.light           = light;
            glfwSetCursorPosCallback(window, [](GLFWwindow* wnd,double xPos, double yPos) {
                WindowState* pWindowState = (WindowState*)glfwGetWindowUserPointer(wnd);
                pWindowState->delCurPos.x = xPos - pWindowState->curCurPos.x;
                pWindowState->delCurPos.y = yPos - pWindowState->curCurPos.y;;
                pWindowState->curCurPos.x = xPos;
                pWindowState->curCurPos.y = yPos;
            });
            glfwSetTime(0.0f);
            {
                double xPos, yPos;
                glfwGetCursorPos(window, &xPos, &yPos);
                windowState.curCurPos.x = xPos;
                windowState.curCurPos.y = yPos;
                windowState.delCurPos.x = 0.0f;
                windowState.delCurPos.y = 0.0f;
            }
            bool   isResized     = false;
            bool   isUpdated     = false;
            bool   isFixedLight  = false;
            bool   isMovedCamera = false;
            while (!glfwWindowShouldClose(window)) {
                
                if (isResized) {
                    {
                        std::random_device rd;
                        std::mt19937 mt(rd());
                        std::vector<unsigned int> seeds(width * height);
                        std::generate(seeds.begin(), seeds.end(), mt);
                        d_seeds = rtlib::CUDABuffer<unsigned int>(seeds);
                    }
                    d_frames.resize(width * height);
                    d_accums.resize(width * height);
                    params.accumBuffer = d_accums.getDevicePtr();
                    params.seed   = d_seeds.getDevicePtr();
                    params.width  = width;
                    params.height = height;
                }
                if (isFixedLight) {
                    float  emissionRate = fabsf(fmodf(glfwGetTime(), 20.f) - 10.f) / 10.0f;
                    size_t lightSbtOffset = RAY_TYPE_COUNT * 7 + RAY_TYPE_RADIANCE;
                    rtlib::SBTRecord<HitgroupData>* d_lightHGPtr = d_HitgroupBuffer.getDevicePtr() + lightSbtOffset;
                    rtlib::SBTRecord<HitgroupData>   lightHGData = hitgroupRecords[lightSbtOffset];
                    lightHGData.data.emission *= emissionRate;
                    cudaMemcpy(d_lightHGPtr, &lightHGData, sizeof(rtlib::SBTRecord<HitgroupData>), cudaMemcpyHostToDevice);
                    params.light.emission = light.emission * emissionRate;
                    isUpdated = true;
                }
                if (isMovedCamera) {
                    auto camera = cameraController.GetCamera(30.0f, 1.0f);
                    raygenRecord.data.eye = camera.getEye();
                    auto [u, v, w] = camera.getUVW();
                    raygenRecord.data.u = u;
                    raygenRecord.data.v = v;
                    raygenRecord.data.w = w;
                    d_RaygenBuffer.upload(&raygenRecord, 1);
                    isUpdated = true;
                }
                if (isUpdated) {
                    d_frames.upload(std::vector<uchar4>(width * height));
                    d_accums.upload(std::vector<float3>(width * height));
                    params.samplePerALL = 0;
                }
                {
                    params.frameBuffer = d_frames.map();
                    d_params.upload(&params, 1);
                    pipeline.launch(stream, d_params.getDevicePtr(), sbt, width, height, 2);
                    cuStreamSynchronize(stream);
                    d_frames.unmap();
                    params.samplePerALL += params.samplePerLaunch;
                }
                
                {
                    glfwPollEvents();
                    if (isResized) {
                        glTexture.reset();
                        glTexture.allocate({ (size_t)width,(size_t)height }, GL_TEXTURE_2D);
                        glTexture.setParameteri(GL_TEXTURE_MAG_FILTER,    GL_LINEAR, false);
                        glTexture.setParameteri(GL_TEXTURE_MIN_FILTER,    GL_LINEAR, false);
                        glTexture.setParameteri(GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE, false);
                        glTexture.setParameteri(GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE, false);
                    }
                    glTexture.upload(0, d_frames.getHandle(), 0, 0, width, height);
                    {
                        glViewport(0, 0, width, height);
                    }
                    glClear(GL_COLOR_BUFFER_BIT);
                    program.use();
                    glActiveTexture(GL_TEXTURE0);
                    glTexture.bind();
                    glUniform1i(texLoc, 0);
                    glBindVertexArray(screenVAO);
                    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);
                    glfwSwapBuffers(window);
                    isUpdated            = false;
                    isResized            = false;
                    isMovedCamera        = false;
                    {
                        int tWidth, tHeight;
                        glfwGetWindowSize(window, &tWidth, &tHeight);
                        if (width != tWidth || height != tHeight) {
                            std::cout << width << "->" << tWidth << "\n";
                            std::cout << height << "->" << tHeight << "\n";
                            width = tWidth;
                            height = tHeight;
                            isResized = true;
                            isUpdated = true;
                        }
                        else {
                            isResized = false;
                        }
                        float prevTime = glfwGetTime();
                        windowState.delTime = windowState.curTime - prevTime;
                        windowState.curTime = prevTime;
                        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
                            cameraController.ProcessKeyboard(rtlib::CameraMovement::eForward, windowState.delTime);
                            isMovedCamera = true;
                        }
                        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
                            cameraController.ProcessKeyboard(rtlib::CameraMovement::eBackward, windowState.delTime);
                            isMovedCamera = true;
                        }
                        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
                            cameraController.ProcessKeyboard(rtlib::CameraMovement::eLeft, windowState.delTime);
                            isMovedCamera = true;
                        }
                        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
                            cameraController.ProcessKeyboard(rtlib::CameraMovement::eRight, windowState.delTime);
                            isMovedCamera = true;
                        }
                        if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS) {
                            cameraController.ProcessKeyboard(rtlib::CameraMovement::eLeft, windowState.delTime);
                            isMovedCamera = true;
                        }
                        if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS) {
                            cameraController.ProcessKeyboard(rtlib::CameraMovement::eRight, windowState.delTime);
                            isMovedCamera = true;
                        }
                        if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS) {
                            cameraController.ProcessKeyboard(rtlib::CameraMovement::eUp, windowState.delTime);
                            isMovedCamera = true;
                        }
                        if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS) {
                            cameraController.ProcessKeyboard(rtlib::CameraMovement::eDown, windowState.delTime);
                            isMovedCamera = true;
                        }
                        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
                            cameraController.ProcessMouseMovement(-windowState.delCurPos.x, windowState.delCurPos.y);
                            isMovedCamera = true;
                        }
                    }
                    
                }
            }
            auto img_pixels = std::vector<uchar4>();
            d_frames.download(img_pixels);
            stbi_write_bmp("tekitou.bmp", width, height, 4, img_pixels.data());
            program.destroy();
            glDeleteVertexArrays(1, &screenVAO);
            screenVBO.reset();
            screenIBO.reset();
            glTexture.reset();
            glfwDestroyWindow(window);
            window = nullptr;
            glfwTerminate();
        }
        RTLIB_CU_CHECK(cuStreamDestroy(stream));
    }
}