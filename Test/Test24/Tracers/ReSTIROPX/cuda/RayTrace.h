#ifndef RAY_TRACE_H
#define RAY_TRACE_H
#define RAY_TRACE_ENABLE_SAMPLE 1
#include <cuda_runtime.h>
#include <optix.h>
#include <RTLib/math/Math.h>
#include <RTLib/math/Random.h>
#include <RTLib/math/VectorFunction.h>
#include <RTLib/math/Matrix.h>
#include <RayTraceConfig.h>
namespace test24_restir
{
    enum   RayType {
        RAY_TYPE_RADIANCE = 0,
        RAY_TYPE_OCCLUSION = 1,
        RAY_TYPE_COUNT = 2,
    };
    enum   FlameType {
        FRAME_TYPE_CURRENT = 0,
        FRAME_TYPE_PREVIOUS,
        FRAME_TYPE_COUNT
    };
    enum   MaterialType {
        MATERIAL_TYPE_DIFFUSE = 0,
        MATERIAL_TYPE_SPECULAR,
        MATERIAL_TYPE_REFRACTION,
        MATERIAL_TYPE_EMISSION,
        MATERIAL_TYPE_OCCLUSION,
        MATERIAL_TYPE_COUNT
    };
    struct PinholeCameraData
    {
        float3 u, v, w;
        float3 eye;
    };
    struct LightRec
    {
        float3 position;
        float3 emission;
        float3 normal;
    };
    struct MeshLight
    {
        float3* vertices;
        float3* normals;
        float2* texCoords;
        uint3* indices;
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
            auto e = getEmissionColor(t);
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
    template<typename T>
    struct Reservoir
    {
        float        w = 0.0f;
        float        w_sum = 0.0f;
        unsigned int m = 0;
        T            y = {};
        RTLIB_INLINE RTLIB_HOST_DEVICE bool Update(T x_i, float w_i, float rnd01)
        {
            w = 0.0f;
            w_sum += w_i;
            ++m;
            if ((w_i / w_sum) >= rnd01)
            {
                y = x_i;
                return true;
            }
            return false;
        }
    };
    struct ReservoirState
    {
        float targetDensity;
    };
    //NOT OPTIMIZE
    struct RayFirstParams
    {
        unsigned int                 width;
        unsigned int                height;
        OptixTraversableHandle   gasHandle;
        float3* posiBuffer;
        float3* normBuffer;
        float3* emitBuffer;
        float3* diffBuffer;
        float* distBuffer;
        unsigned int* seedBuffer;
        int2* motiBuffer;
        bool                  updateMotion;
    };
    //NOT OPTIMIZE
    struct RaySecondParams
    {
        unsigned int                 width;
        unsigned int                height;
        unsigned int       samplePerLaunch;
        unsigned int          samplePerALL;
        unsigned int         numCandidates;
        OptixTraversableHandle   gasHandle;
        MeshLightList           meshLights;
        float3* accumBuffer;
        uchar4* frameBuffer;
        float3* curPosiBuffer;
        float3* prvPosiBuffer;
        float3* curNormBuffer;
        float3* prvNormBuffer;
        float3* curDiffBuffer;
        float3* prvDiffBuffer;
        float3* emitBuffer;
        float * distBuffer;
        unsigned int* seedBuffer;
        int2* motiBuffer;
        Reservoir<LightRec>* resvBuffer;
        ReservoirState* tempBuffer;
    };
    struct RayGenData
    {
        PinholeCameraData   pinhole[2];
    };
    struct MissData {
        float4  bgColor;
    };
    struct MissData2 {};
    struct HitgroupData {
        float3* vertices;
        float3* normals;
        float2* texCoords;
        uint3* indices;
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
        __forceinline__ __device__ float3 getDiffuseColor(const float2& uv)const noexcept {
#if !RAY_TRACE_ENABLE_SAMPLE
            return this->diffuse;
#else
            auto diffTC = tex2D<float4>(this->diffuseTex, uv.x, uv.y);
            auto diffBC = this->diffuse;
            auto diffColor = diffBC * make_float3(float(diffTC.x), float(diffTC.y), float(diffTC.z));
            return diffColor;
#endif
        }
        __forceinline__ __device__ float3 getSpecularColor(const float2& uv)const noexcept {
#if !RAY_TRACE_ENABLE_SAMPLE
            return this->specular;
#else
            auto specTC = tex2D<float4>(this->specularTex, uv.x, uv.y);
            auto specBC = this->specular;
            auto specColor = specBC * make_float3(float(specTC.x), float(specTC.y), float(specTC.z));
            return specColor;
#endif
        }
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
    struct HitgroupData2 {};
}
#endif