#define __CUDACC__
#include "RayTrace.h"
using namespace test24_nee;
struct RadiancePRD {
    float3        emitted;
    float3        radiance;
    float3        attenuation;
    float3        attenuation2;
    float         distance;
    unsigned int  seed;
    int           countEmitted;
    int           done;
    int           pad;
};
extern "C" {
    __constant__ RayTraceParams params;
}
template<typename RNG>
static __forceinline__ __device__ float3 sampleCosinePDF(const float3& normal, RNG& rng)
{
    rtlib::ONB onb(normal);
    return onb.local(rtlib::random_cosine_direction(rng));
}
template<typename RNG>
static __forceinline__ __device__ float3 samplePhongPDF(const float3& reflectDir, float shinness, RNG& rng)
{
    rtlib::ONB onb(reflectDir);
    const auto cosTht = powf(rtlib::random_float1(0.0f, 1.0f, rng), 1.0f / (shinness + 1.0f));
    const auto sinTht = sqrtf(1.0f - cosTht * cosTht);
    const auto phi = rtlib::random_float1(0.0f, RTLIB_M_2PI, rng);
    return onb.local(make_float3(sinTht * cosf(phi), sinTht * sinf(phi), cosTht));
}
static __forceinline__ __device__ float  getValPhongPDF(const float3& direction, const float3& reflectDir, float shinness)
{

    const auto reflCos = rtlib::max(rtlib::dot(reflectDir, direction), 0.0f);
    return (shinness + 2.0f) * powf(reflCos, shinness) / RTLIB_M_2PI;
}
static __forceinline__ __device__ float3 faceForward(const float3& n, const float3& i, const float3& nref) {
    return copysignf(1.0f, rtlib::dot(n, i)) * nref;
}
static __forceinline__ __device__ void* unpackPointer(unsigned int p0, unsigned int p1) {
    return reinterpret_cast<void*>(rtlib::to_combine(p0, p1));
}
static __forceinline__ __device__ void   packPointer(void* ptr, unsigned int& p0, unsigned int& p1) {
    const unsigned long long llv = reinterpret_cast<const unsigned long long>(ptr);
    p0 = rtlib::to_upper(llv);
    p1 = rtlib::to_lower(llv);
}
static __forceinline__ __device__ float3 unpackFloat3(unsigned int p0, unsigned p1, unsigned int p2)
{
    return make_float3(__uint_as_float(p0), __uint_as_float(p1), __uint_as_float(p2));
}
static __forceinline__ __device__ void   packFloat3(const float3& v, unsigned int& p0, unsigned& p1, unsigned int& p2)
{
    p0 = __float_as_uint(v.x);
    p1 = __float_as_uint(v.y);
    p2 = __float_as_uint(v.z);
}
static __forceinline__ __device__ RadiancePRD* getRadiancePRD() {
    unsigned int p0 = optixGetPayload_0();
    unsigned int p1 = optixGetPayload_1();
    return static_cast<RadiancePRD*>(unpackPointer(p0, p1));
}
static __forceinline__ __device__ void setRadiancePRD(RadiancePRD* prd) {
    unsigned int p0;
    unsigned int p1;
    packPointer(static_cast<void*>(prd), p0, p1);
    optixSetPayload_0(p0);
    optixSetPayload_1(p1);
}
static __forceinline__ __device__ void setRayOrigin(const float3& origin) {
    unsigned int p2, p3, p4;
    packFloat3(origin, p2, p3, p4);
    optixSetPayload_2(p2);
    optixSetPayload_3(p3);
    optixSetPayload_4(p4);
}
static __forceinline__ __device__ void setRayDirection(const float3& direction) {
    unsigned int p5, p6, p7;
    packFloat3(direction, p5, p6, p7);
    optixSetPayload_5(p5);
    optixSetPayload_6(p6);
    optixSetPayload_7(p7);
}
static __forceinline__ __device__ void setPayloadOccluded(bool occluded) {
    optixSetPayload_0(static_cast<unsigned int>(occluded));
}
static __forceinline__ __device__ void traceRadiance(
    OptixTraversableHandle handle,
    float3& rayOrigin,
    float3& rayDirection,
    float tmin, float tmax,
    RadiancePRD* prd) {
    unsigned int p0, p1, p2, p3, p4, p5, p6, p7;
    packPointer(prd, p0, p1);
    optixTrace(handle, rayOrigin, rayDirection, tmin, tmax, 0.0f, OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE, RAY_TYPE_RADIANCE, RAY_TYPE_COUNT, RAY_TYPE_RADIANCE, p0, p1, p2, p3, p4, p5, p6, p7);
    rayOrigin = unpackFloat3(p2, p3, p4);
    rayDirection = unpackFloat3(p5, p6, p7);
}
static __forceinline__ __device__ bool traceOccluded(
    OptixTraversableHandle handle,
    const float3& rayOrigin,
    const float3& rayDirection,
    float tmin, float tmax) {
    unsigned int occluded = false;
    optixTrace(handle, rayOrigin, rayDirection, tmin, tmax, 0.0f, OptixVisibilityMask(255), OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT, RAY_TYPE_OCCLUSION, RAY_TYPE_COUNT, RAY_TYPE_OCCLUSION, occluded);
    return occluded;
}
extern "C" __global__ void __raygen__def() {

    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();
    auto* rgData = reinterpret_cast<RayGenData*>(optixGetSbtDataPointer());
    const float3 u = rgData->u;
    const float3 v = rgData->v;
    const float3 w = rgData->w;
    unsigned int seed = params.seedBuffer[params.width * idx.y + idx.x];
    float3 result = make_float3(0.0f, 0.0f, 0.0f);
    size_t i = params.samplePerLaunch;
    do {
        rtlib::Xorshift32 xor32(seed);
        const float2 jitter = rtlib::random_float2(xor32);
        const float2 d = make_float2(
            ((2.0f * static_cast<float>(idx.x) + jitter.x) / static_cast<float>(dim.x)) - 1.0,
            ((2.0f * static_cast<float>(idx.y) + jitter.y) / static_cast<float>(dim.y)) - 1.0);
        seed = xor32.m_seed;
        float3 rayOrigin = rgData->eye;
        float3 rayDirection = rtlib::normalize(d.x * u + d.y * v + w);
        RadiancePRD prd;
        prd.emitted = make_float3(0.0f, 0.0f, 0.0f);
        prd.radiance = make_float3(0.0f, 0.0f, 0.0f);
        prd.attenuation = make_float3(1.0f, 1.0f, 1.0f);
        prd.attenuation2 = make_float3(0.0f, 0.0f, 0.0f);
        prd.countEmitted = true;
        prd.done = false;
        prd.seed = seed;
        float t_min = 0.01f;
        float t_max = 1e16f;
        int depth = 1;
        for (;;) {
            if (depth >= params.maxTraceDepth) {
                prd.done = true;
            }
            traceRadiance(params.gasHandle, rayOrigin, rayDirection, t_min, t_max, &prd);
            result += prd.emitted;
            result += prd.radiance * prd.attenuation2;
            if (prd.done || isnan(rayDirection.x) || isnan(rayDirection.y) || isnan(rayDirection.z)) {
                break;
            }
            if (isnan(result.x) || isnan(result.y) || isnan(result.z)) {
                printf("Fatal Error\n");
                break;
            }
            depth++;
        }
        seed = prd.seed;
    } while (--i);
    const float3 prevAccumColor = params.accumBuffer[params.width * idx.y + idx.x];
    const float3 accumColor = prevAccumColor + result;
    float3 frameColor = accumColor / (static_cast<float>(params.samplePerALL + params.samplePerLaunch));
    frameColor = frameColor / (make_float3(1.0f, 1.0f, 1.0f) + frameColor);

    params.frameBuffer[params.width * idx.y + idx.x] = make_uchar4(
        static_cast<unsigned char>(255.99 * rtlib::linear_to_gamma(frameColor.x)),
        static_cast<unsigned char>(255.99 * rtlib::linear_to_gamma(frameColor.y)),
        static_cast<unsigned char>(255.99 * rtlib::linear_to_gamma(frameColor.z)), 255);
    params.accumBuffer[params.width * idx.y + idx.x] = accumColor;
    params.seedBuffer[params.width * idx.y + idx.x] = seed;
}
extern "C" __global__ void __miss__radiance() {
    auto* msData = reinterpret_cast<MissData*>(optixGetSbtDataPointer());
    RadiancePRD* prd = getRadiancePRD();
    prd->emitted = make_float3(msData->bgColor.x, msData->bgColor.y, msData->bgColor.z) * prd->attenuation;
    prd->radiance = make_float3(0.0f);
    prd->countEmitted = false;
    prd->done = true;
}
extern "C" __global__ void __miss__occluded() {
    setPayloadOccluded(false);
}
extern "C" __global__ void __closesthit__radiance_for_diffuse_nee_with_nee_light() {

    auto* hgData = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());
    const auto rayDirection = optixGetWorldRayDirection();
    const int    primitiveID = optixGetPrimitiveIndex();
    const auto v0 = hgData->vertices[hgData->indices[primitiveID].x];
    const auto v1 = hgData->vertices[hgData->indices[primitiveID].y];
    const auto v2 = hgData->vertices[hgData->indices[primitiveID].z];
    const auto n0 = optixTransformNormalFromObjectToWorldSpace(rtlib::normalize(rtlib::cross(v1 - v0, v2 - v0)));
    const auto normal = faceForward(n0, make_float3(-rayDirection.x, -rayDirection.y, -rayDirection.z), n0);
    const auto barycentric = optixGetTriangleBarycentrics();
    const auto t0 = hgData->texCoords[hgData->indices[primitiveID].x];
    const auto t1 = hgData->texCoords[hgData->indices[primitiveID].y];
    const auto t2 = hgData->texCoords[hgData->indices[primitiveID].z];
    const auto texCoord = (1.0f - barycentric.x - barycentric.y) * t0 + barycentric.x * t1 + barycentric.y * t2;
    const auto diffuse = hgData->getDiffuseColor(texCoord);
    const auto emission = hgData->getEmissionColor(texCoord);
    const auto position = optixGetWorldRayOrigin() + optixGetRayTmax() * rayDirection;
    RadiancePRD* prd = getRadiancePRD();
    float3 prvAttenuation = prd->attenuation;
    prd->emitted = emission * prvAttenuation * static_cast<float>(prd->countEmitted) * static_cast<float>(rtlib::dot(n0, rayDirection) < 0.0f);
    prd->radiance             = make_float3(0.0f);
    prd->attenuation2         = make_float3(0.0f);
    if (prd->done) {
        return;
    }
    rtlib::Xorshift32 xor32(prd->seed);
    float3 rayOrigin = position + 0.01f * normal;
    {
        float3 newDirection = sampleCosinePDF(normal, xor32);

        setRayOrigin(rayOrigin);
        setRayDirection(newDirection);

        prd->attenuation *= diffuse;
        prd->countEmitted = false;
    }
    {
        LightRec lRec = {};
        auto  lightId  = rtlib::random_float1(0.0f, 1.0f, xor32) * static_cast<float>(params.light.count);
        auto& light    = params.light.data[static_cast<unsigned int>(lightId)];
        auto  lightDir = light.Sample(rayOrigin, lRec, xor32);
        auto  ndl      = rtlib::dot(lightDir, normal);
        float weight   = 0.0f;
        if (ndl > 0.0f && lRec.invPdf > 0.0f) {
            const bool occluded = traceOccluded(params.gasHandle, rayOrigin, lightDir, 0.0f, lRec.distance - 0.01f);
            if (!occluded) {
                weight = ndl * lRec.invPdf;
            }        
        }
        prd->attenuation2 = prvAttenuation * diffuse / RTLIB_M_PI;
        prd->radiance     = lRec.emission * weight * static_cast<float>(params.light.count);
    }
    prd->seed = xor32.m_seed;
}
extern "C" __global__ void __closesthit__radiance_for_diffuse_nee_with_def_light() {

    auto* hgData = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());
    const auto rayDirection = optixGetWorldRayDirection();
    const int    primitiveID = optixGetPrimitiveIndex();
    const auto v0 = hgData->vertices[hgData->indices[primitiveID].x];
    const auto v1 = hgData->vertices[hgData->indices[primitiveID].y];
    const auto v2 = hgData->vertices[hgData->indices[primitiveID].z];
    const auto n0 = optixTransformNormalFromObjectToWorldSpace(rtlib::normalize(rtlib::cross(v1 - v0, v2 - v0)));
    const auto normal = faceForward(n0, make_float3(-rayDirection.x, -rayDirection.y, -rayDirection.z), n0);
    const auto barycentric = optixGetTriangleBarycentrics();
    const auto t0 = hgData->texCoords[hgData->indices[primitiveID].x];
    const auto t1 = hgData->texCoords[hgData->indices[primitiveID].y];
    const auto t2 = hgData->texCoords[hgData->indices[primitiveID].z];
    const auto texCoord = (1.0f - barycentric.x - barycentric.y) * t0 + barycentric.x * t1 + barycentric.y * t2;

    const auto diffuse = hgData->getDiffuseColor(texCoord);
    const auto emission = hgData->getEmissionColor(texCoord);

    const float3 position = optixGetWorldRayOrigin() + optixGetRayTmax() * rayDirection;

    RadiancePRD* prd = getRadiancePRD();
    float3 prvAttenuation = prd->attenuation;
    prd->emitted = emission * prvAttenuation * static_cast<float>(rtlib::dot(n0, rayDirection) < 0.0f);
    prd->radiance = make_float3(0.0f);
    prd->attenuation2 = make_float3(0.0f);
    if (prd->done) {
        return;
    }
    rtlib::Xorshift32 xor32(prd->seed);
    float3 rayOrigin = position + 0.01f * normal;
    {
        float3 newDirection = sampleCosinePDF(normal, xor32);

        setRayOrigin(rayOrigin);
        setRayDirection(newDirection);

        prd->attenuation *= diffuse;
        prd->countEmitted = false;
    }
    {
        LightRec lRec = {};
        auto  lightId  = rtlib::random_float1(0.0f, 1.0f, xor32) * static_cast<float>(params.light.count);
        auto& light    = params.light.data[static_cast<unsigned int>(lightId)];
        auto  lightDir = light.Sample(rayOrigin, lRec, xor32);
        auto  ndl = rtlib::dot(lightDir, normal);
        float weight = 0.0f;
        if (ndl > 0.0f && lRec.invPdf > 0.0f) {
            const bool occluded = traceOccluded(params.gasHandle, rayOrigin, lightDir, 0.0f, lRec.distance - 0.01f);
            if (!occluded) {
                weight = ndl * lRec.invPdf;
            }
        }
        prd->attenuation2 = prvAttenuation * diffuse / RTLIB_M_PI;
        prd->radiance = lRec.emission * weight * static_cast<float>(params.light.count);

    }
    prd->seed = xor32.m_seed;
}
extern "C" __global__ void __closesthit__radiance_for_diffuse_def() {

    auto* hgData = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());
    const float3 rayDirection = optixGetWorldRayDirection();
    const int    primitiveID = optixGetPrimitiveIndex();
    const float3 v0 = hgData->vertices[hgData->indices[primitiveID].x];
    const float3 v1 = hgData->vertices[hgData->indices[primitiveID].y];
    const float3 v2 = hgData->vertices[hgData->indices[primitiveID].z];
    const float3 n0 = optixTransformNormalFromObjectToWorldSpace(rtlib::normalize(rtlib::cross(v1 - v0, v2 - v0)));
    const float3 normal = faceForward(n0, make_float3(-rayDirection.x, -rayDirection.y, -rayDirection.z), n0);
    const float2 barycentric = optixGetTriangleBarycentrics();
    const auto t0 = hgData->texCoords[hgData->indices[primitiveID].x];
    const auto t1 = hgData->texCoords[hgData->indices[primitiveID].y];
    const auto t2 = hgData->texCoords[hgData->indices[primitiveID].z];
    const auto texCoord = (1.0f - barycentric.x - barycentric.y) * t0 + barycentric.x * t1 + barycentric.y * t2;
    const auto diffuse = hgData->getDiffuseColor(texCoord);
    const auto emission = hgData->getEmissionColor(texCoord);
    const float3 position = optixGetWorldRayOrigin() + optixGetRayTmax() * rayDirection;
    RadiancePRD* prd = getRadiancePRD();
    float3 prvAttenuation = prd->attenuation;
    prd->emitted = emission * prvAttenuation * static_cast<float>(rtlib::dot(n0, rayDirection) < 0.0f);
    prd->radiance             = make_float3(0.0f);
    prd->attenuation2         = make_float3(0.0f);
    rtlib::Xorshift32 xor32(prd->seed);
    {
        float3 newDirection = sampleCosinePDF(normal, xor32);

        setRayOrigin(position+0.01f*normal);
        setRayDirection(newDirection);

        prd->attenuation *= diffuse;
        prd->countEmitted = true;
    }
    prd->seed = xor32.m_seed;
}
extern "C" __global__ void __closesthit__radiance_for_specular() {
    auto* hgData = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());
    const float3 rayDirection = optixGetWorldRayDirection();
    const int    primitiveID = optixGetPrimitiveIndex();
    const float2 barycentric = optixGetTriangleBarycentrics();
    const float3 v0 = hgData->vertices[hgData->indices[primitiveID].x];
    const float3 v1 = hgData->vertices[hgData->indices[primitiveID].y];
    const float3 v2 = hgData->vertices[hgData->indices[primitiveID].z];
    float3       n0 = optixTransformNormalFromObjectToWorldSpace(rtlib::normalize(rtlib::cross(v1 - v0, v2 - v0)));
    /*if (hgData->normals) {
        const float3 nv0 = hgData->normals[hgData->indices[primitiveID].x];
        const float3 nv1 = hgData->normals[hgData->indices[primitiveID].y];
        const float3 nv2 = hgData->normals[hgData->indices[primitiveID].z];
        const bool isValidNv0 = !((nv0.x == 0.0f) && (nv0.y == 0.0f) && (nv0.z == 0.0f));
        const bool isValidNv1 = !((nv1.x == 0.0f) && (nv1.y == 0.0f) && (nv1.z == 0.0f));
        const bool isValidNv2 = !((nv2.x == 0.0f) && (nv2.y == 0.0f) && (nv2.z == 0.0f));
        if (isValidNv0 && isValidNv1 && isValidNv2)
        {
            float3 nv = optixTransformNormalFromObjectToWorldSpace(rtlib::normalize((1.0f - barycentric.x - barycentric.y) * nv0 + barycentric.x * nv1 + barycentric.y * nv2));
            if (rtlib::dot(nv, n0) > 0.0f) {
                n0 = nv;
            }
        }
    }*/
    float3 normal = faceForward(n0, make_float3(-rayDirection.x, -rayDirection.y, -rayDirection.z), n0);
    const auto t0 = hgData->texCoords[hgData->indices[primitiveID].x];
    const auto t1 = hgData->texCoords[hgData->indices[primitiveID].y];
    const auto t2 = hgData->texCoords[hgData->indices[primitiveID].z];
    const auto texCoord = (1.0f - barycentric.x - barycentric.y) * t0 + barycentric.x * t1 + barycentric.y * t2;
    const float3 position = optixGetWorldRayOrigin() + optixGetRayTmax() * rayDirection;
    RadiancePRD* prd = getRadiancePRD();
    prd->emitted = make_float3(0.0f, 0.0f, 0.0f);
    prd->radiance     = make_float3(0.0f);
    prd->attenuation2 = make_float3(0.0f);
    {
        float3 specular       = hgData->getSpecularColor(texCoord);
        const auto reflectDir = rtlib::normalize(rtlib::reflect(rayDirection, normal));
        setRayOrigin(position + 0.01f * normal);                
        setRayDirection(reflectDir);
        prd->distance = optixGetRayTmax();
        prd->attenuation *= specular;
        prd->countEmitted = true;
    }
}
extern "C" __global__ void __closesthit__radiance_for_refraction() {
    auto* hgData = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());
    const float3 rayDirection = optixGetWorldRayDirection();
    const int    primitiveID = optixGetPrimitiveIndex();
    const float2 barycentric = optixGetTriangleBarycentrics();
    const float3 v0 = hgData->vertices[hgData->indices[primitiveID].x];
    const float3 v1 = hgData->vertices[hgData->indices[primitiveID].y];
    const float3 v2 = hgData->vertices[hgData->indices[primitiveID].z];
    float3       n0 = optixTransformNormalFromObjectToWorldSpace(rtlib::normalize(rtlib::cross(v1 - v0, v2 - v0)));
    /*if (hgData->normals) {
        const float3 nv0 = hgData->normals[hgData->indices[primitiveID].x];
        const float3 nv1 = hgData->normals[hgData->indices[primitiveID].y];
        const float3 nv2 = hgData->normals[hgData->indices[primitiveID].z];
        const bool isValidNv0 = !((nv0.x == 0.0f) && (nv0.y == 0.0f) && (nv0.z == 0.0f));
        const bool isValidNv1 = !((nv1.x == 0.0f) && (nv1.y == 0.0f) && (nv1.z == 0.0f));
        const bool isValidNv2 = !((nv2.x == 0.0f) && (nv2.y == 0.0f) && (nv2.z == 0.0f));
        if (isValidNv0 && isValidNv1 && isValidNv2)
        {
            float3 nv = optixTransformNormalFromObjectToWorldSpace(rtlib::normalize((1.0f - barycentric.x - barycentric.y) * nv0 + barycentric.x * nv1 + barycentric.y * nv2));
            if (rtlib::dot(nv, n0) > 0.0f) {
                n0 = nv;
            }
        }
    }*/
    float3 normal = {};
    float  refInd = 0.0f;
    if (rtlib::dot(n0, rayDirection) < 0.0f) {
        normal = n0;
        refInd = 1.0f / hgData->refrInd;
    }
    else {
        normal = make_float3(-n0.x, -n0.y, -n0.z);
        refInd = hgData->refrInd;
    }
    const auto t0 = hgData->texCoords[hgData->indices[primitiveID].x];
    const auto t1 = hgData->texCoords[hgData->indices[primitiveID].y];
    const auto t2 = hgData->texCoords[hgData->indices[primitiveID].z];
    const auto texCoord = (1.0f - barycentric.x - barycentric.y) * t0 + barycentric.x * t1 + barycentric.y * t2;
    const float3 position = optixGetWorldRayOrigin() + optixGetRayTmax() * rayDirection;
    RadiancePRD* prd = getRadiancePRD();
    prd->emitted = make_float3(0.0f, 0.0f, 0.0f);
    prd->radiance         = make_float3(0.0f);
    prd->attenuation2     = make_float3(0.0f);
    rtlib::Xorshift32 xor32(prd->seed);
    float3 diffuse = hgData->getDiffuseColor(texCoord);
    float3 specular = hgData->getSpecularColor(texCoord);
    float3 transmit = hgData->transmit;
    {
        const auto reflectDir = rtlib::normalize(rtlib::reflect(rayDirection, normal));
        float  cosine_i   = -rtlib::dot(normal, rayDirection);
        float  sine_o_2   = (1.0f - rtlib::pow2(cosine_i)) * rtlib::pow2(refInd);
#if 0
        float  f0 = rtlib::pow2((1 - refInd) / (1 + refInd));
        float  fresnell = f0 + (1.0f - f0) * rtlib::pow5(1.0f - cosine_i);
#else
        float  fresnell = 0.0f;
        {
            float  cosine_o = sqrtf(rtlib::max(1.0f - sine_o_2, 0.0f));
            float  r_p = (cosine_i - refInd * cosine_o) / (cosine_i + refInd * cosine_o);
            float  r_s = (refInd * cosine_i - cosine_o) / (refInd * cosine_i + cosine_o);
            fresnell = (r_p * r_p + r_s * r_s) / 2.0f;
        }
#endif
        if (rtlib::random_float1(0.0f, 1.0f, xor32) < fresnell || sine_o_2 > 1.0f) {

            //printf("reflect: %lf %lf %lf\n", reflectDir.x, reflectDir.y, reflectDir.z);
            setRayOrigin(position + 0.01f * normal);
            setRayDirection(reflectDir);
            prd->attenuation *= specular;
        }
        else {
            //fix refract direction
            float  cosine_o = sqrtf(1.0f - sine_o_2);
            float3 k = (rayDirection + cosine_i * normal) / sqrtf(1.0f - cosine_i * cosine_i);
            float3 refractDir = rtlib::normalize(sqrtf(sine_o_2) * k - cosine_o * normal);
            //printf("refract: %lf %lf %lf\n", refractDir.x, refractDir.y, refractDir.z);
            setRayOrigin(position - 0.01f * normal);
            setRayDirection(refractDir);
        }
        prd->countEmitted = true;
    }
    prd->seed = xor32.m_seed;
}
extern "C" __global__ void __closesthit__radiance_for_emission_with_nee_light() {
    auto* hgData = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());
    const float3 rayDirection = optixGetWorldRayDirection();
    const int    primitiveID = optixGetPrimitiveIndex();
    const float3 v0 = hgData->vertices[hgData->indices[primitiveID].x];
    const float3 v1 = hgData->vertices[hgData->indices[primitiveID].y];
    const float3 v2 = hgData->vertices[hgData->indices[primitiveID].z];
    const float3 n0 = optixTransformNormalFromObjectToWorldSpace(rtlib::normalize(rtlib::cross(v1 - v0, v2 - v0)));
    const float2 barycentric = optixGetTriangleBarycentrics();
    const auto t0 = hgData->texCoords[hgData->indices[primitiveID].x];
    const auto t1 = hgData->texCoords[hgData->indices[primitiveID].y];
    const auto t2 = hgData->texCoords[hgData->indices[primitiveID].z];
    const auto texCoord = (1.0f - barycentric.x - barycentric.y) * t0 + barycentric.x * t1 + barycentric.y * t2;
    const float3 position = optixGetWorldRayOrigin() + optixGetRayTmax() * rayDirection;
    RadiancePRD* prd = getRadiancePRD();
    if (prd->countEmitted && rtlib::dot(n0, rayDirection) < 0.0f) {
        prd->emitted = hgData->getEmissionColor(texCoord) * prd->attenuation;
    }
    else {
        prd->emitted = make_float3(0.0f);
    }
    prd->radiance = make_float3(0.0f);
	prd->attenuation2=make_float3(0.0f);    
	prd->countEmitted = false;    prd->done = true;
}
extern "C" __global__ void __closesthit__radiance_for_emission_with_def_light() {
    auto* hgData = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());
    const float3 rayDirection = optixGetWorldRayDirection();
    const int    primitiveID = optixGetPrimitiveIndex();
    const float3 v0 = hgData->vertices[hgData->indices[primitiveID].x];
    const float3 v1 = hgData->vertices[hgData->indices[primitiveID].y];
    const float3 v2 = hgData->vertices[hgData->indices[primitiveID].z];
    const float3 n0 = optixTransformNormalFromObjectToWorldSpace(rtlib::normalize(rtlib::cross(v1 - v0, v2 - v0)));
    const float2 barycentric = optixGetTriangleBarycentrics();
    const auto t0 = hgData->texCoords[hgData->indices[primitiveID].x];
    const auto t1 = hgData->texCoords[hgData->indices[primitiveID].y];
    const auto t2 = hgData->texCoords[hgData->indices[primitiveID].z];
    const auto texCoord = (1.0f - barycentric.x - barycentric.y) * t0 + barycentric.x * t1 + barycentric.y * t2;
    const auto distance = optixGetRayTmax();
    const auto position = optixGetWorldRayOrigin() + distance * rayDirection;    
    RadiancePRD* prd = getRadiancePRD();
    if (rtlib::dot(n0, rayDirection) < 0.0f) {
        prd->emitted = hgData->getEmissionColor(texCoord) * prd->attenuation;
    }
    else {
        prd->emitted = make_float3(0.0f);
    }
    prd->radiance = make_float3(0.0f);
    prd->attenuation2 = make_float3(0.0f);
    prd->countEmitted = false;
    prd->done = true;
}
extern "C" __global__ void __closesthit__radiance_for_emission() {
    auto* hgData = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());
    const float3 rayDirection = optixGetWorldRayDirection();
    const int    primitiveID = optixGetPrimitiveIndex();
    const float3 v0 = hgData->vertices[hgData->indices[primitiveID].x];
    const float3 v1 = hgData->vertices[hgData->indices[primitiveID].y];
    const float3 v2 = hgData->vertices[hgData->indices[primitiveID].z];
    const float3 n0 = optixTransformNormalFromObjectToWorldSpace(rtlib::normalize(rtlib::cross(v1 - v0, v2 - v0)));    
    const float2 barycentric = optixGetTriangleBarycentrics();
    const auto t0 = hgData->texCoords[hgData->indices[primitiveID].x];
    const auto t1 = hgData->texCoords[hgData->indices[primitiveID].y];
    const auto t2 = hgData->texCoords[hgData->indices[primitiveID].z];
    const auto texCoord = (1.0f - barycentric.x - barycentric.y) * t0 + barycentric.x * t1 + barycentric.y * t2;
    const auto distance = optixGetRayTmax();
    const auto position = optixGetWorldRayOrigin() + distance * rayDirection;
    RadiancePRD* prd = getRadiancePRD();
    if (rtlib::dot(n0, rayDirection) < 0.0f) {
        prd->emitted = hgData->getEmissionColor(texCoord) * prd->attenuation;
    }
    else {
        prd->emitted = make_float3(0.0f);
    }
    prd->radiance = make_float3(0.0f);
    prd->attenuation2 = make_float3(0.0f);
    prd->countEmitted = false;
    prd->done = true;
}
extern "C" __global__ void __closesthit__radiance_for_phong_nee_with_nee_light() {
    auto* hgData = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());
    const float3 rayDirection = optixGetWorldRayDirection();
    const int    primitiveID = optixGetPrimitiveIndex();
    const float3 v0 = hgData->vertices[hgData->indices[primitiveID].x];
    const float3 v1 = hgData->vertices[hgData->indices[primitiveID].y];
    const float3 v2 = hgData->vertices[hgData->indices[primitiveID].z];
    const float3 n0 = optixTransformNormalFromObjectToWorldSpace(rtlib::normalize(rtlib::cross(v1 - v0, v2 - v0)));
    const float3 normal = faceForward(n0, make_float3(-rayDirection.x, -rayDirection.y, -rayDirection.z), n0);
    const float2 barycentric = optixGetTriangleBarycentrics();
    const auto t0 = hgData->texCoords[hgData->indices[primitiveID].x];
    const auto t1 = hgData->texCoords[hgData->indices[primitiveID].y];
    const auto t2 = hgData->texCoords[hgData->indices[primitiveID].z];
    const auto texCoord = (1.0f - barycentric.x - barycentric.y) * t0 + barycentric.x * t1 + barycentric.y * t2;
    const auto position = optixGetWorldRayOrigin() + optixGetRayTmax() * rayDirection;
    const auto reflectDir = rtlib::normalize(rtlib::reflect(rayDirection, normal));
    const auto diffuse = hgData->getDiffuseColor(texCoord);
    const auto specular = hgData->getSpecularColor(texCoord);
    const auto emission = hgData->getEmissionColor(texCoord);
    const auto shinness = hgData->shinness;
    RadiancePRD* prd = getRadiancePRD();
    const float3 prvAttenuation = prd->attenuation;
    prd->emitted = emission * prvAttenuation * static_cast<float>(prd->countEmitted) * static_cast<float>(rtlib::dot(n0, rayDirection) < 0.0f);
    prd->radiance = make_float3(0.0f);
    prd->attenuation2           = make_float3(0.0f);
    if (prd->done) {
        return;
    }
    rtlib::Xorshift32 xor32(prd->seed);
    {
        const auto rnd = rtlib::random_float1(xor32);
        const auto a_diffuse = (diffuse.x + diffuse.y + diffuse.z) / 3.0f;
        const auto a_specular = (specular.x + specular.y + specular.z) / 3.0f;
        auto  newDirection = make_float3(0.0f);
        auto  weight = make_float3(0.0f);

        if (rnd < a_diffuse) {
            newDirection = sampleCosinePDF(normal, xor32);
            weight = diffuse / a_diffuse;
            prd->countEmitted = false;
            {
                const auto diffuseLobe = diffuse / (a_diffuse * RTLIB_M_PI);
                LightRec lRec = {};
                auto& light = params.light.data[xor32.next() % params.light.count];
                auto  lightDir = light.Sample(position, lRec, xor32);
                auto  ndl = rtlib::dot(lightDir, normal);
                float weight2 = 0.0f;
                if (ndl > 0.0f && lRec.invPdf > 0.0f) {
                    const bool occluded = traceOccluded(params.gasHandle, position, lightDir, 0.01f, lRec.distance - 0.01f);
                    if (!occluded) {
                        weight2 = ndl * lRec.invPdf;
                    }
                }
                prd->attenuation2 = prvAttenuation * diffuseLobe;
                prd->radiance = lRec.emission * weight2 * static_cast<float>(params.light.count);
            }
        }
        else if (rnd < a_diffuse + a_specular) {
            newDirection = samplePhongPDF(reflectDir, shinness, xor32);
            weight = specular * rtlib::max(rtlib::dot(newDirection, normal), 0.0f) / a_specular;
            prd->countEmitted = true;
#if RAY_TRACE_NEE_SPECULAR
            {
                const float2 z = rtlib::random_float2(xor32);
                const auto   light = params.light;
                const float3 lightPos = light.corner + light.v1 * z.x + light.v2 * z.y;
                const float  Ldist = rtlib::distance(lightPos, position);
                const float3 lightDir = rtlib::normalize(lightPos - position);
                const float  ndl = rtlib::dot(normal, lightDir);
                const float  lndl = -rtlib::dot(light.normal, lightDir);
                float weight2 = 0.0f;
                const auto specularLobe = specular * (shinness + 2.0f) * powf(rtlib::max(rtlib::dot(lightDir, reflectDir), 0.0f), shinness) / (a_specular * RTLIB_M_2PI);
                if (ndl > 0.0f && lndl > 0.0f) {
                    const bool occluded = traceOccluded(params.gasHandle, position, lightDir, 0.01f, Ldist - 0.01f);
                    if (!occluded) {
                        //printf("not Occluded!\n");
                        const float A = rtlib::length(rtlib::cross(light.v1, light.v2));
                        weight2 = ndl * lndl * A / (Ldist * Ldist);
                    }
                }
                prd->attenuation2 = prvAttenuation * specularLobe;
                prd->radiance = lRec.emission * weight2;
                prd->countEmitted = false;
            }
#endif
        }
        else {
            prd->attenuation = make_float3(0.0f);
            prd->attenuation2 = make_float3(0.0f);
            prd->countEmitted = false;
            prd->done = true;
        }

        setRayOrigin(position);
        setRayDirection(newDirection);

        prd->attenuation *= weight;
    }
    prd->seed = xor32.m_seed;
}
extern "C" __global__ void __closesthit__radiance_for_phong_nee_with_def_light() {
    auto* hgData = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());
    const float3 rayDirection = optixGetWorldRayDirection();
    const int    primitiveID = optixGetPrimitiveIndex();
    const float3 v0 = hgData->vertices[hgData->indices[primitiveID].x];
    const float3 v1 = hgData->vertices[hgData->indices[primitiveID].y];
    const float3 v2 = hgData->vertices[hgData->indices[primitiveID].z];
    const float3 n0 = optixTransformNormalFromObjectToWorldSpace(rtlib::normalize(rtlib::cross(v1 - v0, v2 - v0)));
    const float3 normal = faceForward(n0, make_float3(-rayDirection.x, -rayDirection.y, -rayDirection.z), n0);
    const float2 barycentric = optixGetTriangleBarycentrics();
    const auto t0 = hgData->texCoords[hgData->indices[primitiveID].x];
    const auto t1 = hgData->texCoords[hgData->indices[primitiveID].y];
    const auto t2 = hgData->texCoords[hgData->indices[primitiveID].z];
    const auto texCoord = (1.0f - barycentric.x - barycentric.y) * t0 + barycentric.x * t1 + barycentric.y * t2;
    const auto position = optixGetWorldRayOrigin() + optixGetRayTmax() * rayDirection;
    const auto reflectDir = rtlib::normalize(rtlib::reflect(rayDirection, normal));
    const auto diffuse = hgData->getDiffuseColor(texCoord);
    const auto specular = hgData->getSpecularColor(texCoord);
    const auto emission = hgData->getEmissionColor(texCoord);
    const auto shinness = hgData->shinness;
    RadiancePRD* prd = getRadiancePRD();
    const float3 prvAttenuation = prd->attenuation;
    prd->emitted = emission * prvAttenuation;
    prd->radiance = make_float3(0.0f);
    if (prd->done) {
        return;
    }
    rtlib::Xorshift32 xor32(prd->seed);
    {
        const auto rnd = rtlib::random_float1(xor32);
        const auto a_diffuse = (diffuse.x + diffuse.y + diffuse.z) / 3.0f;
        const auto a_specular = (specular.x + specular.y + specular.z) / 3.0f;
        auto  newDirection = make_float3(0.0f);
        auto  weight = make_float3(0.0f);

        if (rnd < a_diffuse) {
            newDirection = sampleCosinePDF(normal, xor32);
            weight = diffuse / a_diffuse;
            prd->countEmitted = false;
            {
                const auto diffuseLobe = diffuse / (a_diffuse * RTLIB_M_PI);
                LightRec lRec = {};
                auto& light = params.light.data[xor32.next() % params.light.count];
                auto  lightDir = light.Sample(position, lRec, xor32);
                auto  ndl = rtlib::dot(lightDir, normal);
                float weight2 = 0.0f;
                if (ndl > 0.0f && lRec.invPdf > 0.0f) {
                    const bool occluded = traceOccluded(params.gasHandle, position, lightDir, 0.01f, lRec.distance - 0.01f);
                    if (!occluded) {
                        weight2 = ndl * lRec.invPdf;
                    }
                }
                prd->attenuation2 = prvAttenuation * diffuseLobe;
                prd->radiance = lRec.emission * weight2 * static_cast<float>(params.light.count);
            }
        }
        else if (rnd < a_diffuse + a_specular) {
            newDirection = samplePhongPDF(reflectDir, shinness, xor32);
            weight = specular * rtlib::max(rtlib::dot(newDirection, normal), 0.0f) / a_specular;
            prd->countEmitted = true;
#if RAY_TRACE_NEE_SPECULAR
            {
                const float2 z = rtlib::random_float2(xor32);
                const auto   light = params.light;
                const float3 lightPos = light.corner + light.v1 * z.x + light.v2 * z.y;
                const float  Ldist = rtlib::distance(lightPos, position);
                const float3 lightDir = rtlib::normalize(lightPos - position);
                const float  ndl = rtlib::dot(normal, lightDir);
                const float  lndl = -rtlib::dot(light.normal, lightDir);
                float weight2 = 0.0f;
                const auto specularLobe = specular * (shinness + 2.0f) * powf(rtlib::max(rtlib::dot(lightDir, reflectDir), 0.0f), shinness) / (a_specular * RTLIB_M_2PI);
                if (ndl > 0.0f && lndl > 0.0f) {
                    const bool occluded = traceOccluded(params.gasHandle, position, lightDir, 0.01f, Ldist - 0.01f);
                    if (!occluded) {
                        //printf("not Occluded!\n");
                        const float A = rtlib::length(rtlib::cross(light.v1, light.v2));
                        weight2 = ndl * lndl * A / (Ldist * Ldist);
                    }
                }
                prd->attenuation2 = prvAttenuation * specularLobe;
                prd->radiance = lRec.emission * weight2;
                prd->countEmitted = false;
            }
#endif
        }
        else {
            prd->attenuation = make_float3(0.0f);
            prd->attenuation2 = make_float3(0.0f);
            prd->countEmitted = false;
            prd->done = true;
        }

        setRayOrigin(position);
        setRayDirection(newDirection);

        prd->attenuation *= weight;
    }
    prd->seed = xor32.m_seed;
}
extern "C" __global__ void __closesthit__radiance_for_phong_def() {
    auto* hgData = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());
    const float3 rayDirection = optixGetWorldRayDirection();
    const int    primitiveID = optixGetPrimitiveIndex();
    const float3 v0 = hgData->vertices[hgData->indices[primitiveID].x];
    const float3 v1 = hgData->vertices[hgData->indices[primitiveID].y];
    const float3 v2 = hgData->vertices[hgData->indices[primitiveID].z];
    const float3 n0 = optixTransformNormalFromObjectToWorldSpace(rtlib::normalize(rtlib::cross(v1 - v0, v2 - v0)));
    const float3 normal = faceForward(n0, make_float3(-rayDirection.x, -rayDirection.y, -rayDirection.z), n0);
    const float2 barycentric = optixGetTriangleBarycentrics();
    const auto t0 = hgData->texCoords[hgData->indices[primitiveID].x];
    const auto t1 = hgData->texCoords[hgData->indices[primitiveID].y];
    const auto t2 = hgData->texCoords[hgData->indices[primitiveID].z];
    const auto texCoord = (1.0f - barycentric.x - barycentric.y) * t0 + barycentric.x * t1 + barycentric.y * t2;
    const auto diffuse = hgData->getDiffuseColor(texCoord);
    const auto specular = hgData->getSpecularColor(texCoord);
    const auto emission = hgData->getEmissionColor(texCoord);
    const auto reflectDir = rtlib::normalize(rtlib::reflect(rayDirection, normal));
    const auto shinness = hgData->shinness;
    const float3 position = optixGetWorldRayOrigin() + optixGetRayTmax() * rayDirection;
    RadiancePRD* prd = getRadiancePRD();
    const float3 prvAttenuation = prd->attenuation;
    prd->emitted = emission * prvAttenuation;
    prd->attenuation2 = make_float3(0.0f, 0.0f, 0.0f);
    rtlib::Xorshift32 xor32(prd->seed);
    {
        const auto rnd = rtlib::random_float1(xor32);
        const auto a_diffuse = (diffuse.x + diffuse.y + diffuse.z) / 3.0f;
        const auto a_specular = (specular.x + specular.y + specular.z) / 3.0f;
        auto  newDirection = make_float3(0.0f);
        auto  weight = make_float3(0.0f);

        if (rnd < a_diffuse) {
            newDirection = sampleCosinePDF(normal, xor32);
            const auto cosine = fabsf(rtlib::dot(newDirection, reflectDir));
            weight = diffuse / a_diffuse;
        }
        else if (rnd < a_diffuse + a_specular) {
            newDirection = samplePhongPDF(reflectDir, shinness, xor32);
            weight = specular * fabsf(rtlib::dot(newDirection, normal)) / a_specular;
        }
        setRayOrigin(position);
        setRayDirection(newDirection);
        prd->attenuation *= weight;
        prd->countEmitted = true;
    }
    prd->seed = xor32.m_seed;
}
extern "C" __global__ void __closesthit__occluded() {
    setPayloadOccluded(true);
}
