#ifndef RAY_TRACE_H
#define RAY_TRACE_H
#include <cuda_runtime.h>
#include <optix.h>
#include <RTLib/math/Math.h>
#include <RTLib/math/Random.h>
#include <RTLib/math/VectorFunction.h>
#include <RTLib/math/Math.h>
#include <RayTraceConfig.h>
#include <PathGuiding.h>
//#define RAY_GUIDING_SAMPLE_BY_UNIFORM_SPHERE
//#define RAY_GUIDING_SAMPLE_BY_COSINE_SPHERE
//#define TEST_SKIP_TEXTURE_SAMPLE
//#define TEST11_SHOW_EMISSON_COLOR
enum RayType   {
    RAY_TYPE_RADIANCE  = 0,
    RAY_TYPE_OCCLUSION = 1,
    RAY_TYPE_COUNT     = 2,  
};
enum MaterialType {
    MATERIAL_TYPE_DIFFUSE  = 0,
    MATERIAL_TYPE_SPECULAR,
    MATERIAL_TYPE_REFRACTION,
    MATERIAL_TYPE_EMISSION,
    MATERIAL_TYPE_OCCLUSION,
    MATERIAL_TYPE_COUNT
};
struct    LightRec
{
    float3 position;
    float3 emission;
    float  distance;
    float  invPdf;
};
struct    MeshLight
{      
    float3*             vertices;
    float3*             normals;
    float2*             texCoords;
    uint3*              indices;
    unsigned int        indCount;
    float3              emission;
    cudaTextureObject_t emissionTex;
    template<typename RNG>
    RTLIB_INLINE RTLIB_HOST_DEVICE auto Sample(const float3& p_in, LightRec& lRec, RNG& rng)->float3
    {
        auto triIdx = indices[rng.next() % indCount];
        auto v0     = vertices[triIdx.x];
        auto v1     = vertices[triIdx.y];
        auto v2     = vertices[triIdx.z];
        auto t0     = texCoords[triIdx.x];
        auto t1     = texCoords[triIdx.y];
        auto t2     = texCoords[triIdx.z];
        //normal
        auto n0     = rtlib::cross(v1 - v0, v2 - v0);
        //area light
        auto dA     = rtlib::length(n0)/2.0f;
        n0          = rtlib::normalize(n0);
        auto bary   = rtlib::random_in_unit_triangle(make_float3(1.0f,0.0f,0.0f), make_float3(0.0f, 1.0f, 0.0f), make_float3(0.0f, 0.0f, 1.0f), rng);
        auto p      = bary.x * v0 + bary.y * v1 + bary.z * v2;
        auto t      = bary.x * t0 + bary.y * t1 + bary.z * t2;
        auto e      = getEmissionColor(t);
        auto w_out  = p - p_in;
        auto d      = rtlib::length(w_out);
        w_out       = rtlib::normalize(w_out);
        auto lndl   =-rtlib::dot(w_out, n0);
        auto invPdf = lndl * dA * static_cast<float>(indCount)/ (d * d);
        lRec.position = p;
        lRec.emission = e;
        lRec.distance = d;
        lRec.invPdf   = invPdf;
        return w_out;
    };
#ifdef __CUDACC__
    __forceinline__ __device__ float3 getEmissionColor(const float2& uv)const noexcept {
#if !RAY_TRACE_ENABLE_SAMPLE
        return this->emission;
#else
        auto emitTC = tex2D<float4>(this->emissionTex, uv.x, uv.y);
        auto emitBC = this->emission;
        auto emitColor = emitBC * make_float3(float(emitTC.x), float(emitTC.y), float(emitTC.z));
        return emitColor;
#endif
    }
#endif
};
struct MeshLightList
{
    unsigned int count;
    MeshLight*   data;
};
struct   PointLight {
    float3   position;
    float3   emission;
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
    unsigned int*          seedBuffer;       
    unsigned int           width;      
    unsigned int           height;
    unsigned int           samplePerLaunch;
    unsigned int           samplePerALL;
    unsigned int           maxTraceDepth;
    OptixTraversableHandle gasHandle;
    STree2                 sdTree;
    MeshLightList          light;
    bool                   isBuilt;
    bool                   isFinal;
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
    STree2                 sdTree;
    MeshLightList          light;
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