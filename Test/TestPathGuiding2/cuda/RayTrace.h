#ifndef RAY_TRACE_H
#define RAY_TRACE_H
#include <cuda_runtime.h>
#include <optix.h>
#include <RTLib/Math.h>
#include <RTLib/Random.h>
#include <RTLib/VectorFunction.h>
#include <RTLib/Math.h>
#include <RayTraceConfig.h>
#include <PathGuiding.h>
//#define RAY_GUIDING_SAMPLE_BY_UNIFORM_SPHERE
//#define RAY_GUIDING_SAMPLE_BY_COSINE_SPHERE
//#define TEST_SKIP_TEXTURE_SAMPLE
//#define TEST11_SHOW_EMISSON_COLOR
enum RayType   {
    RAY_TYPE_RADIANCE = 0,
    RAY_TYPE_OCCLUSION,
    RAY_TYPE_COUNT,  
};

enum MaterialType {
    MATERIAL_TYPE_DIFFUSE  = 0,
    MATERIAL_TYPE_SPECULAR,
    MATERIAL_TYPE_REFRACTION,
    MATERIAL_TYPE_EMISSION,
    MATERIAL_TYPE_OCCLUSION,
    MATERIAL_TYPE_COUNT
};

struct ParallelLight {
    float3   corner;
    float3   v1, v2;
    float3   normal;
    float3   emission;
};
struct RadianceRec {
    float3 origin;
    float3 direction;
    float3 data;
};
struct RayTraceParams {
    uchar4*                frameBuffer;
    float3*                accumBuffer;
    float3*                accumBuffer2;
    unsigned int*          seedBuffer;       
    unsigned int           width;      
    unsigned int           height;
    unsigned int           samplePerLaunch;
    unsigned int           samplePerALL;
    unsigned int           samplePerALL2;
    unsigned int           maxTraceDepth;
    OptixTraversableHandle gasHandle;
    STree                  sdTree;
    ParallelLight          light;
    bool                   isBuilt;
};
struct RayDebugParams {
    uchar4*                diffuseBuffer; //8
    uchar4*                specularBuffer;
    uchar4*                transmitBuffer;
    uchar4*                emissionBuffer;
    uchar4*                texCoordBuffer;
    uchar4*                normalBuffer;
    uchar4*                depthBuffer;
    uchar4*                sTreeColBuffer;
    unsigned int           width;
    unsigned int           height;
    OptixTraversableHandle gasHandle;
    STree                  sdTree;
    ParallelLight          light;
};
struct RayGenData{
    float3                u,v,w;
    float3                eye;
};
struct MissData {
    float4  bgColor;
};
struct HitgroupData{
    float3*             vertices;
    float3*             normals;
    float2*             texCoords;
    uint3*              indices;
    float3              diffuse;
    cudaTextureObject_t diffuseTex;
    float3              emission;
    cudaTextureObject_t emissionTex;
    float3              specular;
    cudaTextureObject_t specularTex;
    float3              transmit;
    float               shinness;
    float               refrInd;
#ifdef __CUDACC__
    __forceinline__ __device__ float3 getDiffuseColor(const float2& uv)const noexcept{ 
    #if !RAY_TRACE_ENABLE_SAMPLE
        return this->diffuse;
    #else
        auto diffTC      = tex2D<float4>(this->diffuseTex, uv.x, uv.y);
        auto diffBC      = this->diffuse;
        auto diffColor   = diffBC*make_float3(float(diffTC.x) ,float(diffTC.y) ,float(diffTC.z) );
        return diffColor;
    #endif
    }
    __forceinline__ __device__ float3 getSpecularColor(const float2& uv)const noexcept {
    #if !RAY_TRACE_ENABLE_SAMPLE
        return this->specular;
    #else
        auto specTC      = tex2D<float4>(this->specularTex, uv.x, uv.y);
        auto specBC      = this->specular;
        auto specColor   = specBC * make_float3(float(specTC.x) , float(specTC.y) , float(specTC.z) );
        return specColor;
    #endif
    }
    __forceinline__ __device__ float3 getEmissionColor(const float2& uv)const noexcept {
    #if !RAY_TRACE_ENABLE_SAMPLE
        return this->emission;
    #else
        auto emitTC    = tex2D<float4>(this->emissionTex, uv.x, uv.y);
        auto emitBC    = this->emission;
        auto emitColor = emitBC * make_float3(float(emitTC.x) , float(emitTC.y) , float(emitTC.z) );
        return emitColor;
    #endif
    }
#endif
};
#endif