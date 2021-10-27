#define __CUDACC__
#include "Test24DebugTracerDeclare.h"
struct RadiancePRD {
    float3        diffuse;
    float3        specular;
    float3        transmit;
    float3        emission;
    float         distance;
    float2        texCoord;
    float3        normal;
    float3        sTreeCol;
};
using RtParams = test24::tracers::test24_debug::Params;
using RgData   = test24::tracers::test24_debug::RgData;
using MsData   = test24::tracers::test24_debug::MsData;
using HgData   = test24::tracers::test24_debug::HgData;
extern "C" {
    __constant__ RtParams params;
}
static __forceinline__ __device__ float3 faceForward(const float3& n, const float3& i, const float3& nref) {
    return copysignf(1.0f, rtlib::dot(n, i)) * nref;
}
static __forceinline__ __device__ void*  unpackPointer(unsigned int p0, unsigned int p1) {
    return reinterpret_cast<void*>(rtlib::to_combine(p0, p1));
}
static __forceinline__ __device__ void   packPointer(void* ptr,unsigned int& p0, unsigned int& p1) {
    const unsigned long long llv = reinterpret_cast<const unsigned long long>(ptr);
    p0 = rtlib::to_upper(llv);
    p1 = rtlib::to_lower(llv);
}
static __forceinline__ __device__ RadiancePRD* getRadiancePRD() {
    unsigned int p0 = optixGetPayload_0();
    unsigned int p1 = optixGetPayload_1();
    return static_cast<RadiancePRD*>(unpackPointer(p0, p1));
}
static __forceinline__ __device__ void trace(OptixTraversableHandle handle,const float3& rayOrigin, const float3& rayDirection,float tmin, float tmax,RadiancePRD* prd) {
    unsigned int p0, p1;
    packPointer(prd,p0,p1);
    optixTrace(handle, rayOrigin, rayDirection, tmin, tmax, 0.0f, OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE, RAY_TYPE_RADIANCE, RAY_TYPE_COUNT, RAY_TYPE_RADIANCE, p0, p1);
}
extern "C" __global__ void     __raygen__debug(){
    const uint3 idx = optixGetLaunchIndex();
	const uint3 dim = optixGetLaunchDimensions();
    auto* rgData    = reinterpret_cast<RgData*>(optixGetSbtDataPointer());
    const float3 u  = rgData->camera_u;
	const float3 v  = rgData->camera_v;
	const float3 w  = rgData->camera_w;
	const float2 d  = make_float2(
		(2.0f * static_cast<float>(idx.x)/static_cast<float>(dim.x)) - 1.0,
		(2.0f * static_cast<float>(idx.y)/static_cast<float>(dim.y)) - 1.0);
	const float3 origin    = rgData->camera_eye;
	const float3 direction = rtlib::normalize(d.x * u + d.y * v + w);
    //printf("%f, %lf, %lf\n", direction.x, direction.y, direction.z);
    RadiancePRD prd;
    trace(params.traversalHandle, origin,direction, 0.0f, 1e16f,&prd);
    auto texCoordColor = make_float3(prd.texCoord.x,prd.texCoord.y,(1.0f-(prd.texCoord.x+prd.texCoord.y)/2.0f));
    auto normalColor   = make_float3((prd.normal.x+1.0f)/2.0f,(prd.normal.y+1.0f)/2.0f,(prd.normal.z+1.0f)/2.0f);
    auto depthColor    = prd.distance / 400.0f;
    //if (idx.x == 100 && idx.y == 100) {
      // printf("prd.sample=%lf\n", prd.sample);
    //}
   // printf("%f, %lf\n", texCoord.x, texCoord.y);
    params.diffuseBuffer[params.width * idx.y + idx.x]  = make_uchar4(static_cast<unsigned char>(255.99 * prd.diffuse.x ), static_cast<unsigned char>(255.99 * prd.diffuse.y ), static_cast<unsigned char>(255.99 * prd.diffuse.z ), 255);
    params.specularBuffer[params.width * idx.y + idx.x] = make_uchar4(static_cast<unsigned char>(255.99 * prd.specular.x), static_cast<unsigned char>(255.99 * prd.specular.y), static_cast<unsigned char>(255.99 * prd.specular.z), 255);
    params.emissionBuffer[params.width * idx.y + idx.x] = make_uchar4(static_cast<unsigned char>(255.99 * prd.emission.x), static_cast<unsigned char>(255.99 * prd.emission.y), static_cast<unsigned char>(255.99 * prd.emission.z), 255);
    params.transmitBuffer[params.width * idx.y + idx.x] = make_uchar4(static_cast<unsigned char>(255.99 * prd.transmit.x), static_cast<unsigned char>(255.99 * prd.transmit.y), static_cast<unsigned char>(255.99 * prd.transmit.z), 255);
    
    params.texCoordBuffer[params.width * idx.y + idx.x] = make_uchar4(static_cast<unsigned char>(255.99 * texCoordColor.x), static_cast<unsigned char>(255.99 * texCoordColor.y), static_cast<unsigned char>(255.99 * texCoordColor.z), 255);
    params.sTreeColBuffer[params.width * idx.y + idx.x] = make_uchar4(static_cast<unsigned char>(255.99 * prd.sTreeCol.x) , static_cast<unsigned char>(255.99 * prd.sTreeCol.y) , static_cast<unsigned char>(255.99 * prd.sTreeCol.z), 255);
    params.normalBuffer[params.width * idx.y + idx.x]   = make_uchar4(static_cast<unsigned char>(255.99 * normalColor.x)  , static_cast<unsigned char>(255.99 * normalColor.y)  , static_cast<unsigned char>(255.99 * normalColor.z), 255);
    params.depthBuffer[params.width * idx.y + idx.x]    = make_uchar4(static_cast<unsigned char>(255.99 * depthColor)     , static_cast<unsigned char>(255.99 * depthColor)     , static_cast<unsigned char>(255.99 * depthColor), 255);
}
extern "C" __global__ void       __miss__debug(){
    auto* msData = reinterpret_cast<MsData*>(optixGetSbtDataPointer());
    auto prd      = getRadiancePRD();
    prd->diffuse  = make_float3(0.0f);
    prd->specular = make_float3(0.0f);
    prd->transmit = make_float3(0.0f);
    prd->emission = make_float3(0.0f);
    prd->distance = optixGetRayTmax();
    prd->sTreeCol = make_float3(0.0f);
    prd->texCoord = make_float2(0.0f);
    prd->normal   = make_float3(0.0f);
}
extern "C" __global__ void __closesthit__debug(){
    auto* hgData     = reinterpret_cast<HgData*>(optixGetSbtDataPointer());
    float2 texCoord  = optixGetTriangleBarycentrics();
    auto primitiveID = optixGetPrimitiveIndex();
    const float3 rayDirection = optixGetWorldRayDirection();
    //printf("%d\n", primitiveId);
    const float3 p      = optixGetWorldRayOrigin() + optixGetRayTmax() * rayDirection;
    const float3 v0     = optixTransformPointFromObjectToWorldSpace( hgData->vertexBuffer[hgData->triIndBuffer[primitiveID].x]);
    const float3 v1     = optixTransformPointFromObjectToWorldSpace( hgData->vertexBuffer[hgData->triIndBuffer[primitiveID].y]);
    const float3 v2     = optixTransformPointFromObjectToWorldSpace( hgData->vertexBuffer[hgData->triIndBuffer[primitiveID].z]);
    const float3 n0     = optixTransformNormalFromObjectToWorldSpace(hgData->normalBuffer[hgData->triIndBuffer[primitiveID].x]);
    const float3 n1     = optixTransformNormalFromObjectToWorldSpace(hgData->normalBuffer[hgData->triIndBuffer[primitiveID].y]);
    const float3 n2     = optixTransformNormalFromObjectToWorldSpace(hgData->normalBuffer[hgData->triIndBuffer[primitiveID].z]);
    const float3 n_base = rtlib::normalize((1.0f - texCoord.x - texCoord.y) * n0 + texCoord.x * n1 + texCoord.y * n2);
    const float3 n_face = rtlib::normalize(rtlib::cross(v1-v0,v2-v0));
    const float3 normal = faceForward(n_face, make_float3(-rayDirection.x,-rayDirection.y,-rayDirection.z), n_face);
    auto t0             = hgData->texCrdBuffer[hgData->triIndBuffer[primitiveID].x];
    auto t1             = hgData->texCrdBuffer[hgData->triIndBuffer[primitiveID].y];
    auto t2             = hgData->texCrdBuffer[hgData->triIndBuffer[primitiveID].z];
    auto t              = (1.0f-texCoord.x-texCoord.y)*t0 + texCoord.x * t1 + texCoord.y * t2;
    auto diffuse        = hgData->getDiffColor(t);
    auto specular       = hgData->getSpecColor(t);
    auto transmit       = hgData->transmit;
    auto emission       = hgData->getEmitColor(t);
    //printf("%f %f\n",t0.x,t0.y);
    auto prd         = getRadiancePRD();
    prd->diffuse     = diffuse;
    prd->specular    = specular;
    prd->transmit    = transmit;
    prd->emission    = emission;
    prd->distance    = optixGetRayTmax();
    prd->sTreeCol    = make_float3(0.0f,0.0f,0.0f);
    //prd->sTreeCol    = make_float3(area);
    prd->texCoord    = t;
    prd->normal      = normal;
}
extern "C" __global__ void        __anyhit__ah(){
    auto* hgData = reinterpret_cast<HgData*>(optixGetSbtDataPointer());
}
