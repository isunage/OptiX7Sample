#ifndef RAY_TRACE_H
#define RAY_TRACE_H
#include <cuda_runtime.h>
#include <optix.h>
#include <RTLib/Math.h>
#include <RTLib/Random.h>
#include <RTLib/VectorFunction.h>
#include <RTLib/Math.h>
//#define TEST_SKIP_TEXTURE_SAMPLE
//#define   TEST11_SHOW_EMISSON_COLOR
#define TEST_MAX_TRACE_DEPTH 4
#define TEST_SHOW_DIFFUSE_COLOR
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
    float3 emission;
};
struct RadianceRec {
    float3 origin;
    float3 direction;
    float3 data;
};
struct RayTraceParams {
    uchar4*                frameBuffer;
    float3*                accumBuffer;
    unsigned int*          seed;       
    unsigned int           width;      
    unsigned int           height;
    unsigned int           samplePerLaunch;
    unsigned int           samplePerALL;
    unsigned int           maxTraceDepth;
    OptixTraversableHandle gasHandle;
    ParallelLight          light;
};
struct RayDebugParams {
    uchar4*                diffuseBuffer; //8
    uchar4*                specularBuffer;
    uchar4*                transmitBuffer;
    uchar4*                emissionBuffer;
    uchar4*                texCoordBuffer;
    uchar4*                normalBuffer;
    uchar4*                depthBuffer;
    unsigned int           width;
    unsigned int           height;
    OptixTraversableHandle gasHandle;
    ParallelLight          light;
};
struct RayStoreParams {
    uchar4*                frameBuffer;
    float3*                accumBuffer;
    unsigned int*          seed;       
    RadianceRec*           radianceRecData;
    unsigned int*          radianceRecCount;
    unsigned int           width;      
    unsigned int           height;
    unsigned int           samplePerLaunch;
    unsigned int           samplePerALL;
    unsigned int           maxTraceDepth;
    OptixTraversableHandle gasHandle;
    ParallelLight          light;
};
struct RayGenData{
    float3 u,v,w;
    float3 eye;
};
struct MissData {
    float4  bgColor;
};
struct HitgroupData{
    float3*             vertices;
    uint3*              indices;
    float2*             texCoords;
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
    #if defined(TEST_SKIP_TEXTURE_SAMPLE)
        return this->diffuse;
    #else
        auto diffTC      = tex2D<uchar4>(this->diffuseTex, uv.x, uv.y);
        auto diffBC      = this->diffuse;
        auto diffColor   = diffBC*make_float3(float(diffTC.x)/ 255.99f,float(diffTC.y)/ 255.99f,float(diffTC.z)/ 255.99f);
        return diffColor;
    #endif
    }
    __forceinline__ __device__ float3 getSpecularColor(const float2& uv)const noexcept {
    #if defined(TEST_SKIP_TEXTURE_SAMPLE)
        return this->specular;
    #else
        auto specTC      = tex2D<uchar4>(this->specularTex, uv.x, uv.y);
        auto specBC      = this->specular;
        auto specColor   = specBC * make_float3(float(specTC.x) / 255.99f, float(specTC.y) / 255.99f, float(specTC.z) / 255.99f);
        return specColor;
    #endif
    }
    __forceinline__ __device__ float3 getEmissionColor(const float2& uv)const noexcept {
    #if defined(TEST_SKIP_TEXTURE_SAMPLE)
        return this->emission;
    #else
        auto emitTC    = tex2D<uchar4>(this->emissionTex, uv.x, uv.y);
        auto emitBC    = this->emission;
        auto emitColor = emitBC * make_float3(float(emitTC.x) / 255.99f, float(emitTC.y) / 255.99f, float(emitTC.z) / 255.99f);
        return emitColor;
    #endif
    }
#endif
};
#endif