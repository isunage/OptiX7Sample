#ifndef RAY_TRACE_H
#define RAY_TRACE_H
#include <cuda_runtime.h>
#include <optix.h>
#include <RTLib/Math.h>
#include <RTLib/Random.h>
#include <RTLib/VectorFunction.h>
#define TEST11_SHOW_DIFFUSE_COLOR
//#define TEST11_SHOW_NORMAL
enum RayType   {
    RAY_TYPE_RADIANCE = 0,
    RAY_TYPE_OCCLUSION,
    RAY_TYPE_COUNT,  
};
struct ParallelLight {
    float3   corner;
    float3   v1, v2;
    float3   normal;
    float3 emission;
};
struct Params {
    uchar4*                image;
    unsigned int           width;
    unsigned int           height;
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
#ifdef __CUDACC__
    float3 getDiffuseColor(const float2& uv)const noexcept{
        auto diffTC      = tex2D<uchar4>(this->diffuseTex, uv.x, uv.y);
        auto diffBC      = this->diffuse;
        auto diffColor   = diffBC*make_float3(float(diffTC.x)/ 255.99f,float(diffTC.y)/ 255.99f,float(diffTC.z)/ 255.99f);
        return diffColor;
    }
#endif
};
struct RadiancePRD {
    float3       emitted;
    float3       radiance;
    float3       attenuation;
    unsigned int seed;
    int          countEmitted;
    int          done;
    int          pad;
};
#endif