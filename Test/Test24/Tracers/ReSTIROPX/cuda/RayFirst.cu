#define __CUDACC__
#include "RayTrace.h"
using namespace test24_restir;
struct RadiancePRD {
    float3 position;
    float3 normal;
    float2 texCoord;
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
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();
    auto* rgData    = reinterpret_cast<RayGenData*>(optixGetSbtDataPointer());
    const float3 u  = rgData->pinhole[0].u;
    const float3 v  = rgData->pinhole[0].v;
    const float3 w  = rgData->pinhole[0].w;
    const float2 d  = make_float2(
        (2.0f * static_cast<float>(idx.x) / static_cast<float>(dim.x)) - 1.0,
        (2.0f * static_cast<float>(idx.y) / static_cast<float>(dim.y)) - 1.0);
    const float3 origin    = rgData->pinhole[0].eye;
    const float3 direction = rtlib::normalize(d.x * u + d.y * v + w);
    //printf("%f, %lf, %lf\n", direction.x, direction.y, direction.z);
    RadiancePRD prd;
    trace(params.gasHandle, origin, direction, 0.0f, 1e16f, &prd);
    params.curPossBuffer[params.width * idx.y + idx.x] = prd.position;
    params.curNormBuffer[params.width * idx.y + idx.x] = prd.normal;
    params.curTexCBuffer[params.width * idx.y + idx.x] = make_float2(prd.texCoord);
    params.curDistBuffer[params.width * idx.y + idx.x] = prd.distance;
}
extern "C" __global__ void       __miss__first() {
    auto* msData  = reinterpret_cast<MissData*>(optixGetSbtDataPointer());
    auto prd      = getRadiancePRD();
    prd->position = make_float3(0.0f);
    prd->normal   = make_float3(0.0f);
    prd->texCoord = make_float2(0.0f);
    prd->distance = optixGetRayTmax();
}
extern "C" __global__ void __closesthit__first() {
    auto* hgData = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());
    float2 texCoord = optixGetTriangleBarycentrics();
    auto primitiveID = optixGetPrimitiveIndex();
    const float3 rayDirection = optixGetWorldRayDirection();
    //printf("%d\n", primitiveId);
    const float3 p  = optixGetWorldRayOrigin() + optixGetRayTmax() * rayDirection;
    const float3 v0 = optixTransformPointFromObjectToWorldSpace( hgData->vertices[hgData->indices[primitiveID].x]);
    const float3 v1 = optixTransformPointFromObjectToWorldSpace( hgData->vertices[hgData->indices[primitiveID].y]);
    const float3 v2 = optixTransformPointFromObjectToWorldSpace( hgData->vertices[hgData->indices[primitiveID].z]);
    const float3 n0 = optixTransformNormalFromObjectToWorldSpace(hgData->normals[ hgData->indices[primitiveID].x]);
    const float3 n1 = optixTransformNormalFromObjectToWorldSpace(hgData->normals[ hgData->indices[primitiveID].y]);
    const float3 n2 = optixTransformNormalFromObjectToWorldSpace(hgData->normals[ hgData->indices[primitiveID].z]);
    const float3 n_base = rtlib::normalize((1.0f - texCoord.x - texCoord.y) * n0 + texCoord.x * n1 + texCoord.y * n2);
    const float3 n_face = rtlib::normalize(rtlib::cross(v1 - v0, v2 - v0));
    const float3 normal = faceForward(n_face, make_float3(-rayDirection.x, -rayDirection.y, -rayDirection.z), n_face);
    auto t0       = hgData->texCoords[hgData->indices[primitiveID].x];
    auto t1       = hgData->texCoords[hgData->indices[primitiveID].y];
    auto t2       = hgData->texCoords[hgData->indices[primitiveID].z];
    auto t        = (1.0f - texCoord.x - texCoord.y) * t0 + texCoord.x * t1 + texCoord.y * t2;
    auto diffuse  = hgData->getDiffuseColor(t);
    auto specular = hgData->getSpecularColor(t);
    auto transmit = hgData->transmit;
    auto emission = hgData->getEmissionColor(t);
    float3 sTreeSize;
    //printf("%f %f\n",t0.x,t0.y);
    auto prd      = getRadiancePRD();
    prd->position = p;
    prd->normal   = normal;
    prd->texCoord = t;
    prd->distance = optixGetRayTmax();
}
extern "C" __global__ void        __anyhit__ah() {
    auto* hgData = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());
}
