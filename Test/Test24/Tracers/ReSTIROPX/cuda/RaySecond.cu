#define __CUDACC__
#include "RayTrace.h"
using namespace test24_restir;
extern "C" {
    __constant__ RaySecondParams params;
}
struct RadiancePRD
{
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
static __forceinline__ __device__ float3  faceForward(const float3& n, const float3& i, const float3& nref) {
    return copysignf(1.0f, rtlib::dot(n, i)) * nref;
}
static __forceinline__ __device__ void* unpackPointer(unsigned int p0, unsigned int p1) {
    return reinterpret_cast<void*>(rtlib::to_combine(p0, p1));
}
static __forceinline__ __device__ void    packPointer(void* ptr, unsigned int& p0, unsigned int& p1) {
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
extern "C" __global__ void     __raygen__init() {

    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();
    auto* rgData    = reinterpret_cast<RayGenData*>(optixGetSbtDataPointer());
    const float3 u  = rgData->pinhole[0].u;
    const float3 v  = rgData->pinhole[0].v;
    const float3 w  = rgData->pinhole[0].w;
    const float2 d  = make_float2(
        (2.0f * static_cast<float>(idx.x) / static_cast<float>(dim.x)) - 1.0,
        (2.0f * static_cast<float>(idx.y) / static_cast<float>(dim.y)) - 1.0);
    auto   seed      = params.seedBuffer[params.width * idx.y + idx.x];
    auto   normal    = params.normBuffer[params.width * idx.y + idx.x];
    auto   emission  = params.emitBuffer[params.width * idx.y + idx.x];
    auto   diffuse   = params.diffBuffer[params.width * idx.y + idx.x];
    auto   origin    = params.posiBuffer[params.width * idx.y + idx.x] + 0.01f * normal;
    auto   isNotDone =(emission.x == 0.0f) && (emission.y == 0.0f) && (emission.z == 0.0f);
    Reservoir resv   = {};
    if (isNotDone) {
        auto direction = rtlib::normalize(d.x * u + d.y * v + w);
        rtlib::Xorshift32 xor32(seed);
        for (int i = 0; i < params.numCandidates; ++i) {
            auto   lightId  = xor32.next() % params.meshLights.count;
            auto& meshLight = params.meshLights.data[lightId];
            LightRec lrec;
            float3   ldir   = meshLight.Sample(origin, lrec, xor32);
            float3   bsdf   = diffuse * RTLIB_M_INV_PI;
            float3   le     = lrec.emission;
            float    g_over_p = fabs(rtlib::dot(ldir, normal)) * lrec.invPdf;
            float3   lp     = bsdf * le * g_over_p;
            float    lp_a   = (lp.x + lp.y + lp.z) / 3.0f;
            resv.Update(lightId, lp_a * static_cast<float>(params.meshLights.count), rtlib::random_float1(xor32));
        }
        auto& meshLight   = params.meshLights.data[resv.y];
        LightRec lrec;
        float3   ldir     = meshLight.Sample(origin, lrec, xor32);
        float3   bsdf     = diffuse * RTLIB_M_INV_PI;
        float3   le       = lrec.emission;
        float    g_over_p = fabs(rtlib::dot(ldir, normal)) * lrec.invPdf;
        float3   lp       = bsdf * le * g_over_p;
        float    lp_a     = (lp.x + lp.y + lp.z) / 3.0f;
        if (resv.w_sum > 0.0f && lp_a > 0.0f) {
            const bool occluded = traceOccluded(params.gasHandle, origin, ldir, 0.001f, lrec.distance - 0.001f);
            resv.w = occluded ? 0.0f : (resv.w_sum / (lp_a * static_cast<float>(resv.m)));
        }
        seed = xor32.m_seed;
    }
    params.resvBuffer[params.width * idx.y + idx.x] = resv;
    params.seedBuffer[params.width * idx.y + idx.x] = seed;
}
extern "C" __global__ void     __raygen__draw() {
    const uint3 idx  = optixGetLaunchIndex();
    const uint3 dim  = optixGetLaunchDimensions();
    auto* rgData     = reinterpret_cast<RayGenData*>(optixGetSbtDataPointer());
    const float3 u   = rgData->pinhole[0].u;
    const float3 v   = rgData->pinhole[0].v;
    const float3 w   = rgData->pinhole[0].w;
    const float2 d   = make_float2(
        (2.0f * static_cast<float>(idx.x) / static_cast<float>(dim.x)) - 1.0,
        (2.0f * static_cast<float>(idx.y) / static_cast<float>(dim.y)) - 1.0);
    auto resv     = params.resvBuffer[params.width * idx.y + idx.x];
    auto seed     = params.seedBuffer[params.width * idx.y + idx.x];
    auto normal   = params.normBuffer[params.width * idx.y + idx.x];
    auto emission = params.emitBuffer[params.width * idx.y + idx.x];
    auto diffuse  = params.diffBuffer[params.width * idx.y + idx.x];
    auto origin   = params.posiBuffer[params.width * idx.y + idx.x];
    auto result   = params.samplePerLaunch * emission;
    auto direction= rtlib::normalize(d.x * u + d.y * v + w);
    auto isNotDone=(emission.x == 0.0f) && (emission.y == 0.0f) && (emission.z == 0.0f);
    if (isNotDone){
        rtlib::Xorshift32 xor32(seed);
        auto& meshLight = params.meshLights.data[resv.y];
        for (int i = 0; i < params.samplePerLaunch; ++i) {
            //unsigned int ldx  = xor32.next() % params.meshLights.count;
            LightRec lrec;
            float3   ldir     = meshLight.Sample(origin, lrec, xor32);
            float3   bsdf     = diffuse * RTLIB_M_INV_PI;
            float3   le       = lrec.emission;
            float    g_over_p = fabs(rtlib::dot(ldir, normal)) * lrec.invPdf;
            //float3   lp     = bsdf * le * g_over_p * static_cast<float>(params.meshLights.count);
            float3   lp       = bsdf * le * g_over_p;
            float    lp_a     =(lp.x + lp.y + lp.z) / 3.0f;
            if (lp_a > 0.0f) {
                const bool occluded = traceOccluded(params.gasHandle, origin, ldir, 0.001f, lrec.distance - 0.001f);
                if (!occluded) {
                    result   += lp * resv.w;
                }
            }
        }
        seed   = xor32.m_seed;
    }
    const float3 prevAccumColor = params.accumBuffer[params.width * idx.y + idx.x];
    const float3 accumColor     = prevAccumColor + result;
    float3 frameColor           = accumColor / (static_cast<float>(params.samplePerALL + params.samplePerLaunch));
    frameColor                  = frameColor / (make_float3(1.0f, 1.0f, 1.0f) + frameColor);
    params.frameBuffer[params.width * idx.y + idx.x] = make_uchar4(
        static_cast<unsigned char>(255.99 * rtlib::linear_to_gamma(frameColor.x)),
        static_cast<unsigned char>(255.99 * rtlib::linear_to_gamma(frameColor.y)),
        static_cast<unsigned char>(255.99 * rtlib::linear_to_gamma(frameColor.z)), 255);
    params.accumBuffer[params.width * idx.y + idx.x] = accumColor;
    params.seedBuffer[params.width * idx.y + idx.x]  = seed;
}
extern "C" __global__ void __closesthit__occluded() {
    setPayloadOccluded(true);
}
extern "C" __global__ void       __miss__occluded() {
    setPayloadOccluded(false);
}