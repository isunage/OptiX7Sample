#ifndef MATERIAL_PARAMETERS_H
#define MATERIAL_PARAMETERS_H
#include <RTLib/math/Math.h>
#include <RTLib/math/Random.h>
#include <RTLib/math/VectorFunction.h>
namespace test24_restir_guide
{
    enum MaterialType:unsigned int
    {
        MATERIAL_TYPE_DIFFUSE   = 0,
        MATERIAL_TYPE_PHONG     ,
        MATERIAL_TYPE_SPECULAR  ,
        MATERIAL_TYPE_REFRACTION,
        MATERIAL_TYPE_EMISSION  ,
        MATERIAL_TYPE_COUNT
    };
    enum MaterialFlagBits : unsigned int
    {
        MATERIAL_FLAG_USE_NEE_BIT = (1<<0)
    };
    struct MaterialRec
    {
        MaterialType        type;
        unsigned int        flags;
        float3              diffuse;
        float3              specular;
        float3              emission;
        float3              transmit;
        float               shinness;
        float               refrInd;
    };
	struct MaterialParameters
	{
        MaterialType        type;
        unsigned int        flags;
        float3              diffuseCol;
        cudaTextureObject_t diffuseTex;
        float3              specularCol;
        cudaTextureObject_t specularTex;
        float3              emissionCol;
        cudaTextureObject_t emissionTex;
        float3              transmit;
        float               shinness;
        float               refrInd;
#ifdef __CUDACC__
        __forceinline__ __device__ auto   GetRecord(const float2& uv)const noexcept -> MaterialRec
        {
            MaterialRec matRec;
            matRec.type     = type;
            matRec.flags    = flags;
            matRec.diffuse  = GetDiffuse(uv);
            matRec.specular = GetSpecular(uv);
            matRec.emission = GetEmission(uv);
            matRec.transmit = transmit;
            matRec.shinness = shinness;
            matRec.refrInd  = refrInd;
            return matRec;
        }
        __forceinline__ __device__ float3 GetDiffuse(const float2& uv) const noexcept
        {
            auto diffTC = tex2D<float4>(this->diffuseTex, uv.x, uv.y);
            auto diffBC = this->diffuseCol;
            auto diffColor = diffBC * make_float3(float(diffTC.x), float(diffTC.y), float(diffTC.z));
            return diffColor;
        }
        __forceinline__ __device__ float3 GetSpecular(const float2& uv) const noexcept
        {
            auto specTC = tex2D<float4>(this->specularTex, uv.x, uv.y);
            auto specBC = this->specularCol;
            auto specColor = specBC * make_float3(float(specTC.x), float(specTC.y), float(specTC.z));
            return specColor;
        }
        __forceinline__ __device__ float3 GetEmission(const float2& uv) const noexcept
        {
            auto emitTC = tex2D<float4>(this->emissionTex, uv.x, uv.y);
            auto emitBC = this->emissionCol;
            auto emitColor = emitBC * make_float3(float(emitTC.x), float(emitTC.y), float(emitTC.z));
            return emitColor;
        }
#endif
	};
}
#endif