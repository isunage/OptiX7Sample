#ifndef RAY_TRACE_H
#define RAY_TRACE_H
#include <cuda_runtime.h>
#include <optix.h>
#include <RTLib/math/Math.h>
#include <RTLib/math/Random.h>
#include <RTLib/math/VectorFunction.h>
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