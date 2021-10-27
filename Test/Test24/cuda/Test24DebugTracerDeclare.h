#ifndef TEST24_DEBUG_TRACER_DECLARE_H
#define TEST24_DEBUG_TRACER_DECLARE_H
#include <cuda_runtime.h>
#include <optix.h>
#include <RTLib/Preprocessors.h>
#include <RTLib/VectorFunction.h>
#include "Test24DebugTracerConfig.h"
#include "RayType.h"
namespace test24
{
    namespace tracers
    {
        namespace test24_debug
        {
            struct Params
            {
                unsigned int width;
                unsigned int height;
                OptixTraversableHandle traversalHandle;
                uchar4*  diffuseBuffer;
                uchar4* specularBuffer;
                uchar4* emissionBuffer;
                uchar4* shinnessBuffer;
                uchar4* transmitBuffer;
                uchar4* texCoordBuffer;
                uchar4* sTreeColBuffer;
                uchar4*   normalBuffer;
                uchar4*    depthBuffer;
            };
            struct RgData
            {   
                float3 camera_eye;
                float3 camera_u;
                float3 camera_v;
                float3 camera_w;
            };
            struct MsData
            {
                float3 bgLightColor;
            };
            struct HgData
            {
                float3*             vertexBuffer;
                float3*             normalBuffer;
                float2*             texCrdBuffer;
                uint3*              triIndBuffer;
                float3              diffCol;
                float3              specCol;
                float3              emitCol;
                float3              transmit;
                float               ior;
                float               shinness;
                cudaTextureObject_t diffTex;
                cudaTextureObject_t specTex;
                cudaTextureObject_t emitTex;
                cudaTextureObject_t shinTex;
#ifdef __CUDACC__
                __forceinline__ __device__ float3 getDiffColor(const float2& uv)const noexcept {
#if !TEST24_DEBUG_TRACER_ENABLE_SAMPLE
                    return this->diffCol;
#else
                    auto diffTC = tex2D<float4>(this->diffTex, uv.x, uv.y);
                    auto diffBC = this->diffCol;
                    auto diffColor = diffBC * make_float3(float(diffTC.x), float(diffTC.y), float(diffTC.z));
                    return diffColor;
#endif
            }
                __forceinline__ __device__ float3 getSpecColor(const float2& uv)const noexcept {
#if !TEST24_DEBUG_TRACER_ENABLE_SAMPLE
                    return this->specCol;
#else
                    auto specTC = tex2D<float4>(this->specTex, uv.x, uv.y);
                    auto specBC = this->specCol;
                    auto specColor = specBC * make_float3(float(specTC.x), float(specTC.y), float(specTC.z));
                    return specColor;
#endif
                }
                __forceinline__ __device__ float3 getEmitColor(const float2& uv)const noexcept {
#if !TEST24_DEBUG_TRACER_ENABLE_SAMPLE
                    return this->emitCol;
#else
                    auto emitTC = tex2D<float4>(this->emitTex, uv.x, uv.y);
                    auto emitBC = this->emitTex;
                    auto emitColor = emitBC * make_float3(float(emitTC.x), float(emitTC.y), float(emitTC.z));
                    return emitColor;
#endif
                }
#endif
            };
        }
    }
}
#endif