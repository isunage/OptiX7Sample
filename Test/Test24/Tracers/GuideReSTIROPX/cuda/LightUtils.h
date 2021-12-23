#ifndef LIGHT_UTILS_H
#define LIGHT_UTILS_H
#include <RTLib/math/Math.h>
#include <RTLib/math/Random.h>
#include <RTLib/math/VectorFunction.h>
namespace test24_restir_guide
{
    struct LightRec
    {
        float3 position;
        float3 emission;
        float3 normal;
    };
    struct MeshLight
    {
        float3*             vertices;
        float3*             normals;
        float2*             texCoords;
        uint3*              indices;
        unsigned int        indCount;
        float3              emission;
        cudaTextureObject_t emissionTex;
        template<typename RNG>
        RTLIB_INLINE RTLIB_HOST_DEVICE auto Sample(const float3& p_in, LightRec& lRec, float& distance, float& invAreaProb, RNG& rng)->float3
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
            auto e = GetEmissionColor(t);
            auto w_out = p - p_in;
            auto d = rtlib::length(w_out);
            w_out = rtlib::normalize(w_out);
            lRec.position = p;
            lRec.emission = e;
            lRec.normal = n0;
            distance = d;
            invAreaProb = dA * static_cast<float>(indCount);
            return w_out;
        };
#ifdef __CUDACC__
        __forceinline__ __device__ float3 GetEmissionColor(const float2& uv)const noexcept {
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
        template<typename RNG>
        RTLIB_INLINE RTLIB_HOST_DEVICE auto Sample(const float3& p_in, LightRec& lRec, float& distance, float& invAreaProb, RNG& rng)->float3{
            auto light = data[rng.next()%count];
            auto lgDir = light.Sample(p_in, lRec, distance, invAreaProb, rng);
            invAreaProb *= static_cast<float>(count);
            return lgDir;
        }
    };
}
#endif
