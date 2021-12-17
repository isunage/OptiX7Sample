#define __CUDACC__
#include "RayTrace.h"
using namespace test24_restir;
struct RadiancePRD {
    float3 position;
    float3 normal;
    float3 emission;
    float3 diffuse;
    float  distance;
};
extern "C" {
    __constant__ RayFirstParams params;
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
static __forceinline__ __device__ RadiancePRD* getRadiancePRD() {
    unsigned int p0 = optixGetPayload_0();
    unsigned int p1 = optixGetPayload_1();
    return static_cast<RadiancePRD*>(unpackPointer(p0, p1));
}
static __forceinline__ __device__ void trace(OptixTraversableHandle handle, const float3& rayOrigin, const float3& rayDirection, float tmin, float tmax, RadiancePRD* prd) {
    unsigned int p0, p1;
    packPointer(prd, p0, p1);
    optixTrace(handle, rayOrigin, rayDirection, tmin, tmax, 0.0f, OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE, RAY_TYPE_RADIANCE, RAY_TYPE_COUNT, RAY_TYPE_RADIANCE, p0, p1);
}
extern "C" __global__ void     __raygen__first() {
    const uint3 idx   = optixGetLaunchIndex();
    const uint3 dim   = optixGetLaunchDimensions();
    auto* rgData      = reinterpret_cast<RayGenData*>(optixGetSbtDataPointer());
    const float3 u    = rgData->pinhole[FRAME_TYPE_CURRENT].u;
    const float3 v    = rgData->pinhole[FRAME_TYPE_CURRENT].v;
    const float3 w    = rgData->pinhole[FRAME_TYPE_CURRENT].w;
    unsigned int seed = params.seedBuffer[params.width * idx.y + idx.x];
    //Jitter ?(possibility of biass)
#if 0
    rtlib::Xorshift32 xor32(seed);
    const float2 jitter = rtlib::random_float2(xor32);
    const float2 d = make_float2(
        ((2.0f * static_cast<float>(idx.x) + jitter.x) / static_cast<float>(dim.x)) - 1.0,
        ((2.0f * static_cast<float>(idx.y) + jitter.y) / static_cast<float>(dim.y)) - 1.0);
    params.seedBuffer[params.width * idx.y + idx.x] = xor32.m_seed;
#else
    const float2 d = make_float2(
        ((2.0f * static_cast<float>(idx.x) ) / static_cast<float>(dim.x)) - 1.0,
        ((2.0f * static_cast<float>(idx.y) ) / static_cast<float>(dim.y)) - 1.0);
#endif
    const float3 origin    = rgData->pinhole[0].eye;
    const float3 direction = rtlib::normalize(d.x * u + d.y * v + w);
    RadiancePRD prd;
    trace(params.gasHandle, origin, direction, 0.0f, 1e16f, &prd);
    params.posiBuffer[params.width * idx.y + idx.x] = prd.position;
    params.normBuffer[params.width * idx.y + idx.x] = prd.normal;
    params.emitBuffer[params.width * idx.y + idx.x] = prd.emission;
    params.diffBuffer[params.width * idx.y + idx.x] = prd.diffuse;
    params.distBuffer[params.width * idx.y + idx.x] = prd.distance;

    if (params.updateMotion)
    {
        float3 curPosition = prd.position;
        float3 prvCamEye   = rgData->pinhole[FRAME_TYPE_PREVIOUS].eye;
        float3 prvCamU     = rgData->pinhole[FRAME_TYPE_PREVIOUS].u;
        float3 prvCamV     = rgData->pinhole[FRAME_TYPE_PREVIOUS].v;
        float3 prvCamW     = rgData->pinhole[FRAME_TYPE_PREVIOUS].w;
        float3 prvEye2Pos  = prvCamEye - curPosition;
        auto   prvDxyz     = rtlib::Matrix3x3(prvCamU, prvCamV, prvEye2Pos).Inverse() * make_float3(-prvCamW.x, -prvCamW.y, -prvCamW.z);
        auto   prvIdx      = make_int2((prvDxyz.x + 1.0f) * static_cast<float>(params.width) / 2.0f, (prvDxyz.y + 1.0f) * static_cast<float>(params.height) / 2.0f);
        params.motiBuffer[params.width * idx.y + idx.x] = make_int2( prvIdx.x - static_cast<int>(idx.x), prvIdx.y - static_cast<int>(idx.y));
    }
    else {
        params.motiBuffer[params.width * idx.y + idx.x] = {0,0};
    }

    rgData->pinhole[FRAME_TYPE_PREVIOUS] = rgData->pinhole[FRAME_TYPE_CURRENT];
}
extern "C" __global__ void       __miss__first() {
    auto* msData        = reinterpret_cast<MissData*>(optixGetSbtDataPointer());
    auto prd            = getRadiancePRD();
    prd->position       = make_float3(0.0f);
    prd->normal         = make_float3(0.0f);
    prd->emission       = make_float3(0.0f);
    prd->diffuse        = make_float3(0.0f);
    prd->distance       = optixGetRayTmax();
}
extern "C" __global__ void __closesthit__first() {
    auto* hgData = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());
    float2 texCoord     = optixGetTriangleBarycentrics();
    auto primitiveID    = optixGetPrimitiveIndex();
    const float3 rayDirection = optixGetWorldRayDirection();
    //printf("%d\n", primitiveId);
    const float3 p      = optixGetWorldRayOrigin() + optixGetRayTmax() * rayDirection;
    const float3 v0     = optixTransformPointFromObjectToWorldSpace (hgData->vertices[hgData->indices[primitiveID].x]);
    const float3 v1     = optixTransformPointFromObjectToWorldSpace (hgData->vertices[hgData->indices[primitiveID].y]);
    const float3 v2     = optixTransformPointFromObjectToWorldSpace (hgData->vertices[hgData->indices[primitiveID].z]);
    const float3 n0     = optixTransformNormalFromObjectToWorldSpace(hgData->normals[ hgData->indices[primitiveID].x]);
    const float3 n1     = optixTransformNormalFromObjectToWorldSpace(hgData->normals[ hgData->indices[primitiveID].y]);
    const float3 n2     = optixTransformNormalFromObjectToWorldSpace(hgData->normals[ hgData->indices[primitiveID].z]);
    const float3 n_base = rtlib::normalize((1.0f - texCoord.x - texCoord.y) * n0 + texCoord.x * n1 + texCoord.y * n2);
    const float3 n_face = rtlib::normalize(rtlib::cross(v1 - v0, v2 - v0));
    const float3 normal = faceForward(n_face, make_float3(-rayDirection.x, -rayDirection.y, -rayDirection.z), n_face);
    auto t0             = hgData->texCoords[hgData->indices[primitiveID].x];
    auto t1             = hgData->texCoords[hgData->indices[primitiveID].y];
    auto t2             = hgData->texCoords[hgData->indices[primitiveID].z];
    auto t              = (1.0f - texCoord.x - texCoord.y) * t0 + texCoord.x * t1 + texCoord.y * t2;
    auto diffuse        = hgData->getDiffuseColor(t);
    auto specular       = hgData->getSpecularColor(t);
    auto transmit       = hgData->transmit;
    auto emission       = hgData->getEmissionColor(t);
    //printf("%f %f\n",t0.x,t0.y);
    auto prd            = getRadiancePRD();
    prd->position       = p+0.01f*normal;
    prd->normal         = normal;
    prd->emission       = emission;
    prd->diffuse        = diffuse;
    prd->distance       = optixGetRayTmax();
}

