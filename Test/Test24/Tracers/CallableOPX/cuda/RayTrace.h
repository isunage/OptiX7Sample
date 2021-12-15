#ifndef RAY_TRACE_H
#define RAY_TRACE_H
#include <cuda_runtime.h>
#include <optix.h>
#include <RTLib/math/Math.h>
#include <RTLib/math/Random.h>
#include <RTLib/math/VectorFunction.h>
#include <RTLib/math/Math.h>
#include <MaterialParameters.h>
//#define RAY_GUIDING_SAMPLE_BY_UNIFORM_SPHERE
//#define RAY_GUIDING_SAMPLE_BY_COSINE_SPHERE
#define RAY_TRACE_ENABLE_SAMPLE 1
//#define TEST11_SHOW_EMISSON_COLOR
namespace test24_callable
{

    enum   RayType {
        RAY_TYPE_RADIANCE = 0,
        RAY_TYPE_OCCLUSION = 1,
        RAY_TYPE_COUNT = 2,
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
        float3* vertices;
        float3* normals;
        float2* texCoords;
        uint3* indices;
        unsigned int        indCount;
        float3              emission;
        cudaTextureObject_t emissionTex;
        template<typename RNG>
        RTLIB_INLINE RTLIB_HOST_DEVICE auto Sample(const float3& p_in, LightRec& lRec, RNG& rng)->float3
        {
            auto triIdx = indices[rng.next() % indCount];
            auto v0 = vertices[triIdx.x];
            auto v1 = vertices[triIdx.y];
            auto v2 = vertices[triIdx.z];
            auto t0 = texCoords[triIdx.x];
            auto t1 = texCoords[triIdx.y];
            auto t2 = texCoords[triIdx.z];
            //normal
            auto n0 = rtlib::cross(v1 - v0, v2 - v0);
            //area light
            auto dA = rtlib::length(n0) / 2.0f;
            n0 = rtlib::normalize(n0);
            auto bary = rtlib::random_in_unit_triangle(make_float3(1.0f, 0.0f, 0.0f), make_float3(0.0f, 1.0f, 0.0f), make_float3(0.0f, 0.0f, 1.0f), rng);
            auto p = bary.x * v0 + bary.y * v1 + bary.z * v2;
            auto t = bary.x * t0 + bary.y * t1 + bary.z * t2;
            auto e = getEmissionColor(t);
            auto w_out = p - p_in;
            auto d = rtlib::length(w_out);
            w_out = rtlib::normalize(w_out);
            auto lndl = -rtlib::dot(w_out, n0);
            auto invPdf = lndl * dA * static_cast<float>(indCount) / (d * d);
            lRec.position = p;
            lRec.emission = e;
            lRec.distance = d;
            lRec.invPdf = invPdf;
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
        MeshLight* data;
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
        MeshLightList          light;
    };
    struct RayGenData {
        float3                u, v, w;
        float3                eye;
    };
    struct MissData {
        float4  bgColor;
    };
    struct HitgroupData {
        float3*             vertices;
        float3*             normals;
        float2*             texCoords;
         uint3*             indices;
        MaterialParameters  matParams;
    };
}

#endif