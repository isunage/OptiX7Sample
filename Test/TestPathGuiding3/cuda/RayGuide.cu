#define __CUDACC__
#include "RayTrace.h"
#include "PathGuiding.h"
struct RadiancePRD {
    DTreeWrapper* dTree;
    float3        dTreeVoxelSize;
    float3        radiance;
    float3        bsdfVal;
    float3        throughPut;
    float         woPdf, bsdfPdf, dTreePdf;
    float         cosine;
    float         distance;
    unsigned int  seed;
    bool          isDelta;
    bool          countEmitted;
    bool          done;
};
extern "C" {
    __constant__ RayTraceParams params;
}
static __forceinline__ __device__ float3       faceForward(const float3& n, const float3& i, const float3& nref) {
    return copysignf(1.0f, rtlib::dot(n, i)) * nref;
}
static __forceinline__ __device__ void*        unpackPointer(unsigned int p0, unsigned int p1) {
    return reinterpret_cast<void*>(rtlib::to_combine(p0, p1));
}
static __forceinline__ __device__ void         packPointer(void* ptr, unsigned int& p0, unsigned int& p1) {
    const unsigned long long llv = reinterpret_cast<const unsigned long long>(ptr);
    p0 = rtlib::to_upper(llv);
    p1 = rtlib::to_lower(llv);
}
static __forceinline__ __device__ float3       unpackFloat3(unsigned int p0, unsigned p1, unsigned int p2)
{
    return make_float3(__uint_as_float(p0), __uint_as_float(p1), __uint_as_float(p2));
}
static __forceinline__ __device__ void         packFloat3(const float3& v, unsigned int& p0, unsigned& p1, unsigned int& p2)
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
static __forceinline__ __device__ void         setRadiancePRD(RadiancePRD* prd) {
    unsigned int p0;
    unsigned int p1;
    packPointer(static_cast<void*>(prd), p0, p1);
    optixSetPayload_0(p0);
    optixSetPayload_1(p1);
}
static __forceinline__ __device__ void         setRayOrigin(const float3& origin) {
    unsigned int p2, p3, p4;
    packFloat3(origin, p2, p3, p4);
    optixSetPayload_2(p2);
    optixSetPayload_3(p3);
    optixSetPayload_4(p4);
}
static __forceinline__ __device__ void         setRayDirection(const float3& direction) {
    unsigned int p5, p6, p7;
    packFloat3(direction, p5, p6, p7);
    optixSetPayload_5(p5);
    optixSetPayload_6(p6);
    optixSetPayload_7(p7);
}
static __forceinline__ __device__ void         setPayloadOccluded(bool occluded) {
    optixSetPayload_0(static_cast<unsigned int>(occluded));
}
static __forceinline__ __device__ void         traceRadiance(
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
static __forceinline__ __device__ bool         traceOccluded(
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
        prd.radiance = make_float3(0.0f);
        prd.bsdfVal  = make_float3(1.0f);
        prd.throughPut = make_float3(1.0f);
        prd.dTreeVoxelSize = make_float3(1.0f);
        prd.woPdf = prd.bsdfPdf = prd.dTreePdf = 0.0f;
        prd.cosine       = 0.0f;
        prd.distance     = 0.0f;
        prd.done         = false;
        prd.countEmitted = true;
        prd.isDelta      = false;
        prd.seed         = seed;
        int depth = 0;
        for (;;) {
            traceRadiance(params.gasHandle, rayOrigin, rayDirection, 0.01f, 1e16f, &prd);
            //vertices�̍X�V
            //Radiance�̍X�V
            //Result�̍X�V
            result += prd.radiance;
            //ThroughPut�̍X�V
            bool isValidThroughPut = (!isnan(prd.throughPut.x) && !isnan(prd.throughPut.y) && !isnan(prd.throughPut.z) &&
                isfinite(prd.throughPut.x) && isfinite(prd.throughPut.y) && isfinite(prd.throughPut.z) &&
                (prd.throughPut.x >= 0.0f && prd.throughPut.y >= 0.0f && prd.throughPut.z >= 0.0f));
            bool isValidDirection  = (!isnan(rayDirection.x) && !isnan(rayDirection.y) && !isnan(rayDirection.z));
            if (prd.done || depth >= params.maxTraceDepth || !isValidThroughPut || !isValidDirection) {
                break;
            }
            depth++;
        }
        seed = prd.seed;
    } while(--i);
    {
        const float3 prevAccumColor = params.accumBuffer[params.width * idx.y + idx.x];
        const float3 accumColor = prevAccumColor + result;
        float3 frameColor = accumColor / (static_cast<float>(params.samplePerALL + params.samplePerLaunch));
        frameColor = frameColor / (make_float3(1.0f, 1.0f, 1.0f) + frameColor);
        params.accumBuffer[params.width * idx.y + idx.x] = accumColor;
        params.frameBuffer[params.width * idx.y + idx.x] = make_uchar4(
            static_cast<unsigned char>(255.99 * rtlib::linear_to_gamma(frameColor.x)),
            static_cast<unsigned char>(255.99 * rtlib::linear_to_gamma(frameColor.y)),
            static_cast<unsigned char>(255.99 * rtlib::linear_to_gamma(frameColor.z)), 255);
    }
    params.seedBuffer[params.width * idx.y + idx.x] = seed;
}
extern "C" __global__ void __raygen__pg_def() {

    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();
    auto* rgData = reinterpret_cast<RayGenData*>(optixGetSbtDataPointer());
    const float3 u = rgData->u;
    const float3 v = rgData->v;
    const float3 w = rgData->w;
    unsigned int seed = params.seedBuffer[params.width * idx.y + idx.x];
    float3 result = make_float3(0.0f, 0.0f, 0.0f);
    size_t i = params.samplePerLaunch;
    TraceVertex vertices[RAY_TRACE_MAX_VERTEX_COUNT] = {};
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
        prd.radiance    = make_float3(0.0f);
        prd.bsdfVal     = make_float3(1.0f);
        prd.throughPut  = make_float3(1.0f);
        prd.dTreeVoxelSize = make_float3(1.0f);
        prd.woPdf = prd.bsdfPdf = prd.dTreePdf = 0.0f;
        prd.distance     = 0.0f;
        prd.done         = false;
        prd.isDelta      = false;
        prd.countEmitted = true;
        prd.seed = seed;
        int depth = 0;
        for (;;) {
            traceRadiance(params.gasHandle, rayOrigin, rayDirection, 0.01f, 1e16f, &prd);
            if(!params.isFinal){
                vertices[depth].rayOrigin      = rayOrigin;
                vertices[depth].rayDirection   = rayDirection;
                vertices[depth].dTree          = prd.dTree;
                vertices[depth].dTreeVoxelSize = prd.dTreeVoxelSize;
                vertices[depth].throughPut     = prd.throughPut;
                vertices[depth].bsdfVal        = prd.bsdfVal;
                vertices[depth].radiance       = make_float3(0.0f);
                vertices[depth].woPdf          = prd.woPdf;
                vertices[depth].bsdfPdf        = prd.bsdfPdf;
                vertices[depth].dTreePdf       = prd.dTreePdf;
                vertices[depth].cosine         = prd.cosine;
                vertices[depth].isDelta        = prd.isDelta;
                for (int j = 0; j < depth; ++j) {
                    vertices[j].Record(prd.radiance);
                }
            }
            //OK
            //Result�̍X�V
            result += prd.radiance;
            bool isValidThroughPut = (!isnan(prd.throughPut.x) && !isnan(prd.throughPut.y) && !isnan(prd.throughPut.z) &&
                                    isfinite(prd.throughPut.x) && isfinite(prd.throughPut.y) && isfinite(prd.throughPut.z) &&
                                            (prd.throughPut.x >= 0.0f && prd.throughPut.y >= 0.0f && prd.throughPut.z >= 0.0f));
            bool isValidDirection  = (!isnan(rayDirection.x) && !isnan(rayDirection.y) && !isnan(rayDirection.z));
            if (prd.done || depth >= params.maxTraceDepth || !isValidThroughPut || !isValidDirection) {
                break;
            }
            depth++;
        }
        if(!params.isFinal){
            for (int j = 0; j < depth; ++j) {
                vertices[j].Commit<RAY_TRACE_S_FILTER, RAY_TRACE_D_FILTER>(params.sdTree, 1.0f);
            }
        }
        seed = prd.seed;
    } while(--i);
    {
        const float3 prevAccumColor = params.accumBuffer[params.width * idx.y + idx.x];
        const float3 accumColor     = prevAccumColor + result;
        float3 frameColor           = accumColor / (static_cast<float>(params.samplePerALL + params.samplePerLaunch));
        frameColor                  = frameColor / (make_float3(1.0f, 1.0f, 1.0f) + frameColor);
        params.accumBuffer[params.width * idx.y + idx.x] = accumColor;
        params.frameBuffer[params.width * idx.y + idx.x] = make_uchar4(
            static_cast<unsigned char>(255.99 * rtlib::linear_to_gamma(frameColor.x)),
            static_cast<unsigned char>(255.99 * rtlib::linear_to_gamma(frameColor.y)),
            static_cast<unsigned char>(255.99 * rtlib::linear_to_gamma(frameColor.z)), 255);
    }
    params.seedBuffer[params.width * idx.y + idx.x] = seed;
}
extern "C" __global__ void __raygen__pg_nee() {
    const uint3 idx   = optixGetLaunchIndex();
    const uint3 dim   = optixGetLaunchDimensions();
    auto* rgData      = reinterpret_cast<RayGenData*>(optixGetSbtDataPointer());
    const float3 u    = rgData->u;
    const float3 v    = rgData->v;
    const float3 w    = rgData->w;
    unsigned int seed = params.seedBuffer[params.width * idx.y + idx.x];
    float3 result = make_float3(0.0f, 0.0f, 0.0f);
    size_t i = params.samplePerLaunch;
    TraceVertex vertices[RAY_TRACE_MAX_VERTEX_COUNT] = {};
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
        prd.radiance = make_float3(0.0f);
        prd.bsdfVal = make_float3(1.0f);
        prd.throughPut = make_float3(1.0f);
        prd.dTreeVoxelSize = make_float3(1.0f);
        prd.woPdf = prd.bsdfPdf = prd.dTreePdf = 0.0f;
        prd.distance = 0.0f;
        prd.done = false;
        prd.isDelta = false;
        prd.countEmitted = true;
        prd.seed = seed;
        int depth = 0;
        for (;;) {
            if (depth >= params.maxTraceDepth) {
                prd.done = true;
            }
            traceRadiance(params.gasHandle, rayOrigin, rayDirection, 0.01f, 1e16f, &prd);
            if (!params.isFinal) {
                vertices[depth].rayOrigin    = rayOrigin;
                vertices[depth].rayDirection = rayDirection;
                vertices[depth].dTree = prd.dTree;
                vertices[depth].dTreeVoxelSize = prd.dTreeVoxelSize;
                vertices[depth].throughPut = prd.throughPut;
                vertices[depth].bsdfVal = prd.bsdfVal;
                vertices[depth].radiance = make_float3(0.0f);
                vertices[depth].woPdf = prd.woPdf;
                vertices[depth].bsdfPdf = prd.bsdfPdf;
                vertices[depth].dTreePdf = prd.dTreePdf;
                vertices[depth].cosine = prd.cosine;
                vertices[depth].isDelta = prd.isDelta;
                for (int j = 0; j < depth; ++j) {
                    vertices[j].Record(prd.radiance);
                }
            }
            //OK
            //Result�̍X�V
            result += prd.radiance;

            bool isValidThroughPut = (!isnan(prd.throughPut.x) && !isnan(prd.throughPut.y)   && !isnan(prd.throughPut.z)   &&
                                    isfinite(prd.throughPut.x) && isfinite(prd.throughPut.y) && isfinite(prd.throughPut.z) &&
                                     (prd.throughPut.x >= 0.0f &&   prd.throughPut.y >= 0.0f && prd.throughPut.z >= 0.0f));
            bool isValidDirection  = (!isnan(rayDirection.x)   && !isnan(rayDirection.y)     && !isnan(rayDirection.z));

            if (prd.done || !isValidThroughPut || !isValidDirection) {
                break;
            }
            depth++;
        }
        if (!params.isFinal) {
            for (int j = 0; j < depth; ++j) {
                vertices[j].Commit<RAY_TRACE_S_FILTER, RAY_TRACE_D_FILTER>(params.sdTree, 1.0f);
            }
        }
        seed = prd.seed;
    } while (--i);
    {
        const float3 prevAccumColor = params.accumBuffer[params.width * idx.y + idx.x];
        const float3 accumColor = prevAccumColor + result;
        float3 frameColor = accumColor / (static_cast<float>(params.samplePerALL + params.samplePerLaunch));
        frameColor = frameColor / (make_float3(1.0f, 1.0f, 1.0f) + frameColor);
        params.accumBuffer[params.width * idx.y + idx.x] = accumColor;
        params.frameBuffer[params.width * idx.y + idx.x] = make_uchar4(
            static_cast<unsigned char>(255.99 * rtlib::linear_to_gamma(frameColor.x)),
            static_cast<unsigned char>(255.99 * rtlib::linear_to_gamma(frameColor.y)),
            static_cast<unsigned char>(255.99 * rtlib::linear_to_gamma(frameColor.z)), 255);
    }
    params.seedBuffer[params.width * idx.y + idx.x] = seed;
}
extern "C" __global__ void __miss__radiance() {
    auto* msData = reinterpret_cast<MissData*>(optixGetSbtDataPointer());
    RadiancePRD* prd = getRadiancePRD();
    prd->radiance = make_float3(msData->bgColor.x, msData->bgColor.y, msData->bgColor.z) * prd->throughPut * static_cast<float>(prd->countEmitted);
    prd->dTree    = nullptr;
    prd->woPdf    = 1.0f;
    prd->bsdfPdf  = 0.0f;
    prd->dTreePdf = 0.0f;
    prd->bsdfVal  = make_float3(1.0f);
    prd->cosine   = 0.0f;
    prd->distance = optixGetRayTmax();
    prd->done     = true;
}
extern "C" __global__ void __miss__occluded() {
    setPayloadOccluded(false);
}
extern "C" __global__ void __closesthit__radiance_for_diffuse_def() {
    auto* hgData = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());
    const float3 rayDirection = optixGetWorldRayDirection();
    const int    primitiveID = optixGetPrimitiveIndex();

    const float3 v0 = optixTransformPointFromObjectToWorldSpace(hgData->vertices[hgData->indices[primitiveID].x]);
    const float3 v1 = optixTransformPointFromObjectToWorldSpace(hgData->vertices[hgData->indices[primitiveID].y]);
    const float3 v2 = optixTransformPointFromObjectToWorldSpace(hgData->vertices[hgData->indices[primitiveID].z]);

    const float3 n0 = optixTransformNormalFromObjectToWorldSpace(rtlib::normalize(rtlib::cross(v1 - v0, v2 - v0)));
    const float3 normal = faceForward(n0, make_float3(-rayDirection.x, -rayDirection.y, -rayDirection.z), n0);

    const float2 barycentric = optixGetTriangleBarycentrics();
    const auto t0 = hgData->texCoords[hgData->indices[primitiveID].x];
    const auto t1 = hgData->texCoords[hgData->indices[primitiveID].y];
    const auto t2 = hgData->texCoords[hgData->indices[primitiveID].z];
    const auto texCoord = (1.0f - barycentric.x - barycentric.y) * t0 + barycentric.x * t1 + barycentric.y * t2;

    const auto diffuse = hgData->getDiffuseColor(texCoord);
    const auto emission = hgData->getEmissionColor(texCoord);

    const auto distance   = optixGetRayTmax();
    const float3 position = optixGetWorldRayOrigin() + distance * rayDirection;
    float3 newDirection   = make_float3(0.0f);
    RadiancePRD* prd      = getRadiancePRD();

    prd->dTree    = nullptr;
    prd->radiance = emission * prd->throughPut;
    prd->distance = distance;

    rtlib::Xorshift32 xor32(prd->seed);
    rtlib::ONB onb(normal);
    newDirection = onb.local(rtlib::random_cosine_direction(xor32));

    const auto cosine = rtlib::dot(newDirection, normal);

    prd->bsdfPdf  = fabsf(cosine) / RTLIB_M_PI;
    prd->dTreePdf = 0.0f;
    prd->woPdf    = prd->bsdfPdf;

    setRayOrigin(position);
    setRayDirection(newDirection);

    prd->cosine      = cosine;
    prd->bsdfVal     = diffuse / RTLIB_M_PI;
    prd->throughPut *= diffuse;
    prd->seed        = xor32.m_seed;

    prd->countEmitted= true;
    prd->isDelta     = false;
}
extern "C" __global__ void __closesthit__radiance_for_diffuse_nee() {
    auto* hgData = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());
    const float3 rayDirection = optixGetWorldRayDirection();
    const int    primitiveID = optixGetPrimitiveIndex();

    const float3 v0 = optixTransformPointFromObjectToWorldSpace(hgData->vertices[hgData->indices[primitiveID].x]);
    const float3 v1 = optixTransformPointFromObjectToWorldSpace(hgData->vertices[hgData->indices[primitiveID].y]);
    const float3 v2 = optixTransformPointFromObjectToWorldSpace(hgData->vertices[hgData->indices[primitiveID].z]);

    const float3 n0 = optixTransformNormalFromObjectToWorldSpace(rtlib::normalize(rtlib::cross(v1 - v0, v2 - v0)));
    const float3 normal = faceForward(n0, make_float3(-rayDirection.x, -rayDirection.y, -rayDirection.z), n0);

    const float2 barycentric = optixGetTriangleBarycentrics();
    const auto t0 = hgData->texCoords[hgData->indices[primitiveID].x];
    const auto t1 = hgData->texCoords[hgData->indices[primitiveID].y];
    const auto t2 = hgData->texCoords[hgData->indices[primitiveID].z];
    const auto texCoord = (1.0f - barycentric.x - barycentric.y) * t0 + barycentric.x * t1 + barycentric.y * t2;

    const auto diffuse = hgData->getDiffuseColor(texCoord);
    const auto emission = hgData->getEmissionColor(texCoord);

    const auto distance = optixGetRayTmax();
    const auto position = optixGetWorldRayOrigin() + distance * rayDirection;
    float3 newDirection = make_float3(0.0f);
    RadiancePRD* prd = getRadiancePRD();

    const auto prvThroughPut = prd->throughPut;
    prd->dTree = nullptr;
    prd->radiance = emission * prvThroughPut * static_cast<float>(prd->countEmitted);
    prd->distance = distance;
    if (prd->done) {
        return;
    }
    rtlib::Xorshift32 xor32(prd->seed);
    rtlib::ONB onb(normal);
    newDirection = onb.local(rtlib::random_cosine_direction(xor32));

    const auto cosine = rtlib::dot(newDirection, normal);

    prd->bsdfPdf = fabsf(cosine) / RTLIB_M_PI;
    prd->dTreePdf = 0.0f;
    prd->woPdf = prd->bsdfPdf;

    setRayOrigin(position);
    setRayDirection(newDirection);

    prd->cosine = cosine;
    prd->bsdfVal = diffuse / RTLIB_M_PI;
    prd->throughPut *= diffuse;

    prd->countEmitted = false;
    prd->isDelta = false;

    {
        const float2 z = rtlib::random_float2(xor32);
        const auto   light = params.light;
        const float3 lightPos = light.corner + light.v1 * z.x + light.v2 * z.y;
        const float  Ldist = rtlib::distance(lightPos, position);
        const float3 lightDir = rtlib::normalize(lightPos - position);
        const float  ndl = rtlib::dot(normal, lightDir);
        const float  lndl = -rtlib::dot(light.normal, lightDir);
        float weight = 0.0f;
        if (ndl > 0.0f && lndl > 0.0f) {
            const bool occluded = traceOccluded(params.gasHandle, position, lightDir, 0.01f, Ldist - 0.01f);
            if (!occluded) {
                //printf("not Occluded!\n");
                const float A = rtlib::length(rtlib::cross(light.v1, light.v2));
                weight = ndl * lndl * A / (Ldist * Ldist);
            }
        }
        prd->radiance += light.emission * prvThroughPut * weight * diffuse / RTLIB_M_PI;
    }

    prd->seed = xor32.m_seed;
}
extern "C" __global__ void __closesthit__radiance_for_diffuse_pg_def() {

    auto* hgData = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());
    const float3 rayDirection = optixGetWorldRayDirection();
    const int    primitiveID = optixGetPrimitiveIndex();

    const float3 v0 = optixTransformPointFromObjectToWorldSpace(hgData->vertices[hgData->indices[primitiveID].x]);
    const float3 v1 = optixTransformPointFromObjectToWorldSpace(hgData->vertices[hgData->indices[primitiveID].y]);
    const float3 v2 = optixTransformPointFromObjectToWorldSpace(hgData->vertices[hgData->indices[primitiveID].z]);

    const float3 n0 = optixTransformNormalFromObjectToWorldSpace(rtlib::normalize(rtlib::cross(v1 - v0, v2 - v0)));
    const float3 normal = faceForward(n0, make_float3(-rayDirection.x, -rayDirection.y, -rayDirection.z), n0);

    const float2 barycentric = optixGetTriangleBarycentrics();
    const auto t0 = hgData->texCoords[hgData->indices[primitiveID].x];
    const auto t1 = hgData->texCoords[hgData->indices[primitiveID].y];
    const auto t2 = hgData->texCoords[hgData->indices[primitiveID].z];
    const auto texCoord = (1.0f - barycentric.x - barycentric.y) * t0 + barycentric.x * t1 + barycentric.y * t2;

    const auto diffuse = hgData->getDiffuseColor(texCoord);
    const auto emission = hgData->getEmissionColor(texCoord);

    const auto distance  = optixGetRayTmax();
    const auto position  = optixGetWorldRayOrigin() + distance * rayDirection;
    auto  dTreeVoxelSize = make_float3(0.0f);
    const auto dTree     = params.sdTree.GetDTreeWrapper(position, dTreeVoxelSize);

    float3 newDirection1 = make_float3(0.0f);
    float3 newDirection2 = make_float3(0.0f);
    float  cosine1      = 0.0f;
    float  cosine2      = 0.0f;

    RadiancePRD* prd    = getRadiancePRD();
    prd->dTree          = dTree;
    prd->dTreeVoxelSize = dTreeVoxelSize;
    prd->radiance       = emission * prd->throughPut;
    prd->distance       = distance;
    prd->bsdfPdf        = 0.0f;
    prd->bsdfVal        = diffuse / RTLIB_M_PI;
    prd->isDelta        = false;
    rtlib::Xorshift32 xor32(prd->seed);

    setRayOrigin(position);
    if(params.isBuilt){
        newDirection1 = dTree->Sample(xor32);
        cosine1 = rtlib::dot(normal, newDirection1);

        if (isnan(newDirection1.x) || isnan(newDirection1.y) || isnan(newDirection1.z)) {
            printf("newDirection1 is nan: new Direction1 = (%f, %f, %f) normal = (%f, %f, %f) n0 = (%f, %f, %f)\n", newDirection1.x, newDirection1.y, newDirection1.z, normal.x, normal.y, normal.z, n0.x, n0.y, n0.z);
        }
    }
    {
        rtlib::ONB onb(normal);
        newDirection2 = onb.local(rtlib::random_cosine_direction(xor32));
        cosine2 = rtlib::dot(normal, newDirection2);
        if (isnan(newDirection2.x) || isnan(newDirection2.y) || isnan(newDirection2.z))
        {
            printf("newDirection2 is nan!\n");
        }
    }

    const float rnd          = rtlib::random_float1(xor32);
    const auto  newDirection = rnd < 0.5f ? newDirection1 : newDirection2;
    const auto  cosine       = rnd < 0.5f ? cosine1 : cosine2;
    const auto  bsdfPdf      = rtlib::max(cosine / RTLIB_M_PI, 0.0f);
    //両方とも正なら
    if (params.isBuilt) {
        const auto  dTreePdf = rtlib::max(dTree->Pdf(newDirection), 0.0f);
        const auto  woPdf = 0.5f * bsdfPdf + 0.5f * dTreePdf;
        prd->bsdfPdf     = bsdfPdf;
        prd->dTreePdf    = dTreePdf;
        prd->woPdf       = woPdf;
        prd->throughPut *= (prd->bsdfVal * rtlib::max(cosine, 0.0f) / woPdf);
        setRayDirection(newDirection);
        prd->cosine = cosine;
    }
    else {
        prd->bsdfPdf     = fabsf(cosine2) / RTLIB_M_PI;
        prd->dTreePdf    = 0.0f;
        prd->woPdf       = prd->bsdfPdf;
        prd->throughPut *= (diffuse);
        setRayDirection(newDirection2);
        prd->cosine      = cosine2;
    }
    prd->countEmitted    = true;
    prd->seed            = xor32.m_seed;
}
extern "C" __global__ void __closesthit__radiance_for_diffuse_pg_nee() {

    auto* hgData = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());
    const float3 rayDirection = optixGetWorldRayDirection();
    const int    primitiveID = optixGetPrimitiveIndex();

    const float3 v0 = optixTransformPointFromObjectToWorldSpace(hgData->vertices[hgData->indices[primitiveID].x]);
    const float3 v1 = optixTransformPointFromObjectToWorldSpace(hgData->vertices[hgData->indices[primitiveID].y]);
    const float3 v2 = optixTransformPointFromObjectToWorldSpace(hgData->vertices[hgData->indices[primitiveID].z]);

    const float3 n0 = optixTransformNormalFromObjectToWorldSpace(rtlib::normalize(rtlib::cross(v1 - v0, v2 - v0)));
    const float3 normal = faceForward(n0, make_float3(-rayDirection.x, -rayDirection.y, -rayDirection.z), n0);

    const float2 barycentric = optixGetTriangleBarycentrics();
    const auto t0 = hgData->texCoords[hgData->indices[primitiveID].x];
    const auto t1 = hgData->texCoords[hgData->indices[primitiveID].y];
    const auto t2 = hgData->texCoords[hgData->indices[primitiveID].z];
    const auto texCoord = (1.0f - barycentric.x - barycentric.y) * t0 + barycentric.x * t1 + barycentric.y * t2;

    const auto diffuse = hgData->getDiffuseColor(texCoord);
    const auto emission = hgData->getEmissionColor(texCoord);

    const auto distance = optixGetRayTmax();
    const auto position = optixGetWorldRayOrigin() + distance * rayDirection;
    auto  dTreeVoxelSize = make_float3(0.0f);
    const auto dTree = params.sdTree.GetDTreeWrapper(position, dTreeVoxelSize);

    float3 newDirection1 = make_float3(0.0f);
    float3 newDirection2 = make_float3(0.0f);
    float  cosine1 = 0.0f;
    float  cosine2 = 0.0f;

    RadiancePRD* prd    = getRadiancePRD();
    auto prvThroughPut  = prd->throughPut;
    prd->dTree          = dTree;
    prd->dTreeVoxelSize = dTreeVoxelSize;
    prd->radiance       = emission * prvThroughPut*static_cast<float>(prd->countEmitted);
    prd->distance       = distance;
    prd->bsdfPdf        = 0.0f;
    prd->bsdfVal        = diffuse / RTLIB_M_PI;
    prd->isDelta        = false;
    //new
    if (prd->done) {
        return;
    }
    rtlib::Xorshift32 xor32(prd->seed);
    setRayOrigin(position);
    if (params.isBuilt) {
        newDirection1 = dTree->Sample(xor32);
        cosine1 = rtlib::dot(normal, newDirection1);

        if (isnan(newDirection1.x) || isnan(newDirection1.y) || isnan(newDirection1.z)) {
            printf("newDirection1 is nan: new Direction1 = (%f, %f, %f) normal = (%f, %f, %f) n0 = (%f, %f, %f)\n", newDirection1.x, newDirection1.y, newDirection1.z, normal.x, normal.y, normal.z, n0.x, n0.y, n0.z);
        }
    }
    {
        rtlib::ONB onb(normal);
        newDirection2 = onb.local(rtlib::random_cosine_direction(xor32));
        cosine2 = rtlib::dot(normal, newDirection2);
        if (isnan(newDirection2.x) || isnan(newDirection2.y) || isnan(newDirection2.z))
        {
            printf("newDirection2 is nan!\n");
        }
    }

    const float rnd = rtlib::random_float1(xor32);
    const auto  newDirection = rnd < 0.5f ? newDirection1 : newDirection2;
    const auto  cosine = rnd < 0.5f ? cosine1 : cosine2;
    const auto  bsdfPdf = rtlib::max(cosine / RTLIB_M_PI, 0.0f);
    //両方とも正なら
    if (params.isBuilt) {
        const auto  dTreePdf = rtlib::max(dTree->Pdf(newDirection), 0.0f);
        const auto  woPdf    = 0.5f * bsdfPdf + 0.5f * dTreePdf;
        prd->bsdfPdf = bsdfPdf;
        prd->dTreePdf = dTreePdf;
        prd->woPdf = woPdf;
        prd->throughPut *= (prd->bsdfVal * rtlib::max(cosine, 0.0f) / woPdf);
        setRayDirection(newDirection);
        prd->cosine = cosine;
    }
    else {
        prd->bsdfPdf = fabsf(cosine2) / RTLIB_M_PI;
        prd->dTreePdf = 0.0f;
        prd->woPdf = prd->bsdfPdf;
        prd->throughPut *= (diffuse);
        setRayDirection(newDirection2);
        prd->cosine = cosine2;
    }
    {
        const float2 z        = rtlib::random_float2(xor32);
        const auto   light    = params.light;
        const float3 lightPos = light.corner + light.v1 * z.x + light.v2 * z.y;
        const float  Ldist    = rtlib::distance(lightPos, position);
        const float3 lightDir = rtlib::normalize(lightPos - position);
        const float ndl       = rtlib::dot(normal, lightDir);
        const float lndl      =-rtlib::dot(light.normal, lightDir);
        const float A         = rtlib::length(rtlib::cross(light.v1, light.v2));
        float lightPdf        = 0.0f;
        float weight          = 0.0f;
        if (ndl > 0.0f && lndl > 0.0f && A>0.0f) {
            const bool occluded = traceOccluded(params.gasHandle, position, lightDir, 0.01f, Ldist - 0.01f);
            if (!occluded) {
                //printf("not Occluded!\n");
                lightPdf      = (Ldist * Ldist) / (lndl * A);
                weight        = ndl * lndl * A / (Ldist * Ldist);
            }
        }
        prd->radiance += light.emission * prvThroughPut * weight * diffuse/RTLIB_M_PI;
    }
    prd->countEmitted = false;
    prd->seed = xor32.m_seed;
}
extern "C" __global__ void __closesthit__radiance_for_specular() {

    auto* hgData = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());
    const float3 rayDirection = optixGetWorldRayDirection();
    const int    primitiveID = optixGetPrimitiveIndex();
    const float2 barycentric = optixGetTriangleBarycentrics();
    const float3 v0 = optixTransformPointFromObjectToWorldSpace(hgData->vertices[hgData->indices[primitiveID].x]);
    const float3 v1 = optixTransformPointFromObjectToWorldSpace(hgData->vertices[hgData->indices[primitiveID].y]);
    const float3 v2 = optixTransformPointFromObjectToWorldSpace(hgData->vertices[hgData->indices[primitiveID].z]);
    float3       n0 = optixTransformNormalFromObjectToWorldSpace(rtlib::normalize(rtlib::cross(v1 - v0, v2 - v0)));
    if (hgData->normals) {
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
    }
    const auto normal = n0;
    const auto t0 = hgData->texCoords[hgData->indices[primitiveID].x];
    const auto t1 = hgData->texCoords[hgData->indices[primitiveID].y];
    const auto t2 = hgData->texCoords[hgData->indices[primitiveID].z];
    const auto texCoord = (1.0f - barycentric.x - barycentric.y) * t0 + barycentric.x * t1 + barycentric.y * t2;
    const float3 position = optixGetWorldRayOrigin() + optixGetRayTmax() * rayDirection;
    RadiancePRD* prd = getRadiancePRD();
    prd->dTree    = nullptr;
    prd->radiance = make_float3(0.0f, 0.0f, 0.0f);
    prd->distance = optixGetRayTmax();
    {
        float3 specular = hgData->getSpecularColor(texCoord);
        float3 reflectDir = rtlib::normalize(rayDirection - 2.0f * rtlib::dot(rayDirection, normal) * normal);
        auto cosine = rtlib::dot(reflectDir, normal);

        prd->woPdf = 0.0f;
        prd->dTreePdf = 0.0f;
        prd->bsdfPdf = std::fabsf(cosine);

        setRayOrigin(position);
        setRayDirection(reflectDir);
        prd->cosine = cosine;

        prd->bsdfVal = specular;
        prd->throughPut *= prd->bsdfVal;
        prd->countEmitted = true;
        prd->isDelta      = true;
    }
}
extern "C" __global__ void __closesthit__radiance_for_refraction() {
    auto* hgData = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());
    const float3 rayDirection = optixGetWorldRayDirection();
    const int    primitiveID = optixGetPrimitiveIndex();
    const float2 barycentric = optixGetTriangleBarycentrics();
    const float3 v0 = optixTransformPointFromObjectToWorldSpace(hgData->vertices[hgData->indices[primitiveID].x]);
    const float3 v1 = optixTransformPointFromObjectToWorldSpace(hgData->vertices[hgData->indices[primitiveID].y]);
    const float3 v2 = optixTransformPointFromObjectToWorldSpace(hgData->vertices[hgData->indices[primitiveID].z]);
    float3       n0 = optixTransformNormalFromObjectToWorldSpace(rtlib::normalize(rtlib::cross(v1 - v0, v2 - v0)));
    if (hgData->normals) {
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
    }
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
    prd->dTree    = nullptr;
    prd->radiance = make_float3(0.0f, 0.0f, 0.0f);
    prd->distance = optixGetRayTmax();
    rtlib::Xorshift32 xor32(prd->seed);
    float3 diffuse = hgData->getDiffuseColor(texCoord);
    float3 specular = hgData->getSpecularColor(texCoord);
    float3 transmit = hgData->transmit;
    {
        float3 reflectDir = rtlib::normalize(rayDirection - 2.0f * rtlib::dot(rayDirection, normal) * normal);
        float  cosine_i = -rtlib::dot(normal, rayDirection);
        float  sine_o_2 = (1.0f - rtlib::pow2(cosine_i)) * rtlib::pow2(refInd);
        float  f0 = rtlib::pow2((1 - refInd) / (1 + refInd));
        float  fresnell = f0 + (1.0f - f0) * rtlib::pow5(1.0f - cosine_i);

        if (rtlib::random_float1(0.0f, 1.0f, xor32) < fresnell || sine_o_2 > 1.0f) {
            float cosine = rtlib::dot(reflectDir, normal);
            prd->woPdf = prd->dTreePdf = 0.0f;
            prd->bsdfPdf = std::fabsf(cosine);
            //printf("reflect: %lf %lf %lf\n", reflectDir.x, reflectDir.y, reflectDir.z);
            setRayOrigin(position + 0.001f * normal);
            setRayDirection(reflectDir);
            prd->cosine = cosine;
            prd->bsdfVal = specular;
            prd->throughPut *= prd->bsdfVal;
        }
        else {
            float  cosine_o = sqrtf(1.0f - sine_o_2);
            float3 k = (rayDirection + cosine_i * normal) / sqrtf(1.0f - cosine_i * cosine_i);
            float3 refractDir = rtlib::normalize(sqrtf(sine_o_2) * k - cosine_o * normal);
            float cosine = rtlib::dot(refractDir, normal);
            prd->woPdf = prd->dTreePdf = 0.0f;
            prd->bsdfPdf = std::fabsf(cosine);
            //printf("refract: %lf %lf %lf\n", refractDir.x, refractDir.y, refractDir.z);
            setRayOrigin(position - 0.001f * normal);
            setRayDirection(refractDir);
            prd->cosine = cosine;
            prd->bsdfVal = make_float3(1.0f);
            prd->throughPut *= prd->bsdfVal;
        }
        prd->isDelta = true;
    }
    prd->countEmitted = true;
    prd->seed = xor32.m_seed;
}
extern "C" __global__ void __closesthit__radiance_for_emission() {
    auto* hgData = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());
    const float3 rayDirection = optixGetWorldRayDirection();
    const int    primitiveID = optixGetPrimitiveIndex();
    const float3 v0 = optixTransformPointFromObjectToWorldSpace(hgData->vertices[hgData->indices[primitiveID].x]);
    const float3 v1 = optixTransformPointFromObjectToWorldSpace(hgData->vertices[hgData->indices[primitiveID].y]);
    const float3 v2 = optixTransformPointFromObjectToWorldSpace(hgData->vertices[hgData->indices[primitiveID].z]);
    const float3 n0 = rtlib::normalize(rtlib::cross(v1 - v0, v2 - v0));
   // const float3 normal = faceForward(n0, make_float3(-rayDirection.x, -rayDirection.y, -rayDirection.z), n0);
    const float2 barycentric = optixGetTriangleBarycentrics();
    const auto t0 = hgData->texCoords[hgData->indices[primitiveID].x];
    const auto t1 = hgData->texCoords[hgData->indices[primitiveID].y];
    const auto t2 = hgData->texCoords[hgData->indices[primitiveID].z];
    const auto texCoord = (1.0f - barycentric.x - barycentric.y) * t0 + barycentric.x * t1 + barycentric.y * t2;
    const auto distance = optixGetRayTmax();
    const float3 position = optixGetWorldRayOrigin() + distance * rayDirection;
    RadiancePRD* prd = getRadiancePRD();
    prd->radiance = hgData->getEmissionColor(texCoord) * prd->throughPut * static_cast<float>(prd->countEmitted)*static_cast<float>(rtlib::dot(n0, rayDirection) < 0.0f);
    prd->bsdfVal = make_float3(1.0f);
    prd->woPdf = 0.0f;
    prd->bsdfPdf = 0.0f;
    prd->dTreePdf = 0.0f;
    prd->dTree = nullptr;
    prd->cosine = 0.0f;
    prd->distance = distance;
    prd->done = true;
}
extern "C" __global__ void __closesthit__occluded() {
    setPayloadOccluded(true);
}
extern "C" __global__ void __closesthit__radiance_for_phong_def() {
    auto* hgData = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());
    const float3 rayDirection = optixGetWorldRayDirection();
    const int    primitiveID = optixGetPrimitiveIndex();

    const float3 v0 = optixTransformPointFromObjectToWorldSpace(hgData->vertices[hgData->indices[primitiveID].x]);
    const float3 v1 = optixTransformPointFromObjectToWorldSpace(hgData->vertices[hgData->indices[primitiveID].y]);
    const float3 v2 = optixTransformPointFromObjectToWorldSpace(hgData->vertices[hgData->indices[primitiveID].z]);

    const float3 n0 = optixTransformNormalFromObjectToWorldSpace(rtlib::normalize(rtlib::cross(v1 - v0, v2 - v0)));
    const float3 normal = faceForward(n0, make_float3(-rayDirection.x, -rayDirection.y, -rayDirection.z), n0);

    const float2 barycentric = optixGetTriangleBarycentrics();
    const auto t0 = hgData->texCoords[hgData->indices[primitiveID].x];
    const auto t1 = hgData->texCoords[hgData->indices[primitiveID].y];
    const auto t2 = hgData->texCoords[hgData->indices[primitiveID].z];

    const auto reflectDir = rtlib::normalize(rayDirection - 2.0f * rtlib::dot(rayDirection, normal) * normal);

    const auto texCoord = (1.0f - barycentric.x - barycentric.y) * t0 + barycentric.x * t1 + barycentric.y * t2;
    const auto diffuse = hgData->getDiffuseColor(texCoord);
    const auto specular = hgData->getSpecularColor(texCoord);
    const auto shinness = hgData->shinness;
    const auto emission = hgData->getEmissionColor(texCoord);
    const auto distance = optixGetRayTmax();
    const float3 position = optixGetWorldRayOrigin() + distance * rayDirection;
    RadiancePRD* prd = getRadiancePRD();

    prd->dTree    = nullptr;
    prd->radiance = emission * prd->throughPut;
    prd->distance = distance;

    rtlib::Xorshift32 xor32(prd->seed);

    auto  newDirection = make_float3(0.0f);
    auto  cosine = 0.0f;

    const auto rnd = rtlib::random_float1(xor32);
    const auto a_diffuse = (diffuse.x + diffuse.y + diffuse.z) / 3.0f;
    const auto a_specular = (specular.x + specular.y + specular.z) / 3.0f;

    if (rnd < a_diffuse) {
        rtlib::ONB onb(normal);
        newDirection = onb.local(rtlib::random_cosine_direction(xor32));
        cosine = rtlib::dot(newDirection, normal);
        prd->bsdfVal = diffuse / (a_diffuse * RTLIB_M_PI);
        prd->bsdfPdf = fabsf(cosine) / RTLIB_M_PI;
        prd->dTreePdf = 0.0f;
        prd->woPdf = prd->bsdfPdf;
        prd->throughPut *= (diffuse / a_diffuse);
        prd->cosine = cosine;
    }
    else if (rnd < a_diffuse + a_specular) {
        const auto cosTht = powf(rtlib::random_float1(0.0f, 1.0f, xor32), 1.0f / (shinness + 1.0f));
        const auto sinTht = sqrtf(1.0f - cosTht * cosTht);
        const auto phi    = rtlib::random_float1(0.0f, RTLIB_M_2PI, xor32);
        rtlib::ONB onb(reflectDir);
        newDirection     = onb.local(make_float3(sinTht * cosf(phi), sinTht * sinf(phi), cosTht));
        cosine           = rtlib::dot(newDirection, normal);
        prd->bsdfVal     = (specular / a_specular) * (shinness + 2.0f) * powf(rtlib::max(rtlib::dot(reflectDir, newDirection),0.0f), shinness) / RTLIB_M_2PI;
        prd->bsdfPdf     = (shinness + 2.0f) * powf(rtlib::max(rtlib::dot(reflectDir, newDirection), 0.0f), shinness) / RTLIB_M_2PI;
        prd->dTreePdf    = 0.0f;
        prd->woPdf       = prd->bsdfPdf;
        prd->throughPut *= (specular * rtlib::max(cosine, 0.0f) / a_specular);
        prd->cosine      = cosine;
    }
    else {
        //printf("Hit!\n");
        //反射しない
        prd->bsdfVal = make_float3(1.0f);
        prd->woPdf = 0.0f;
        prd->bsdfPdf = 0.0f;
        prd->dTreePdf = 0.0f;
        prd->cosine = 0.0f;
        prd->throughPut = make_float3(0.0f);
        prd->dTree = nullptr;
        prd->done  = true;
    }

    setRayOrigin(position);
    setRayDirection(newDirection);
    prd->seed         = xor32.m_seed;
    prd->countEmitted = true;
    prd->isDelta      = false;
}
extern "C" __global__ void __closesthit__radiance_for_phong_pg_def() {
    auto* hgData = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());
    const auto rayDirection = optixGetWorldRayDirection();
    const auto primitiveID = optixGetPrimitiveIndex();

    const auto v0 = optixTransformPointFromObjectToWorldSpace(hgData->vertices[hgData->indices[primitiveID].x]);
    const auto v1 = optixTransformPointFromObjectToWorldSpace(hgData->vertices[hgData->indices[primitiveID].y]);
    const auto v2 = optixTransformPointFromObjectToWorldSpace(hgData->vertices[hgData->indices[primitiveID].z]);

    const auto n0 = optixTransformNormalFromObjectToWorldSpace(rtlib::normalize(rtlib::cross(v1 - v0, v2 - v0)));
    const auto normal = faceForward(n0, make_float3(-rayDirection.x, -rayDirection.y, -rayDirection.z), n0);

    const auto barycentric = optixGetTriangleBarycentrics();
    const auto t0 = hgData->texCoords[hgData->indices[primitiveID].x];
    const auto t1 = hgData->texCoords[hgData->indices[primitiveID].y];
    const auto t2 = hgData->texCoords[hgData->indices[primitiveID].z];


    const auto reflectDir = rtlib::normalize(rayDirection - 2.0f * rtlib::dot(rayDirection, normal) * normal);

    const auto texCoord = (1.0f - barycentric.x - barycentric.y) * t0 + barycentric.x * t1 + barycentric.y * t2;
    const auto emission = hgData->getEmissionColor(texCoord);
    const auto diffuse  = hgData->getDiffuseColor(texCoord);
    const auto specular = hgData->getSpecularColor(texCoord);
    const auto shinness = hgData->shinness;
    const auto distance = optixGetRayTmax();
    const auto position = optixGetWorldRayOrigin() + distance * rayDirection;
    //direction
    float3 newDirection1 = make_float3(0.0f);
    float3 newDirection2 = make_float3(0.0f);
    float3 newDirection3 = make_float3(0.0f);
    //cosine
    float  cosine1 = 0.0f;
    float  cosine2 = 0.0f;
    float  cosine3 = 0.0f;
    //payLoad
    RadiancePRD* prd    = getRadiancePRD();
    auto dTreeVoxelSize = make_float3(0.0f);
    const auto dTree    = params.sdTree.GetDTreeWrapper(position, dTreeVoxelSize);
    prd->dTree          = dTree;
    prd->dTreeVoxelSize = dTreeVoxelSize;
    prd->radiance       = emission*prd->throughPut;
    prd->distance       = distance;
    prd->isDelta        = false;
    rtlib::Xorshift32 xor32(prd->seed);
    //const auto isValid = false;
    setRayOrigin(position);
    if(params.isBuilt){
        newDirection1 = dTree->Sample(xor32);
        cosine1 = rtlib::dot(normal, newDirection1);
    }
    {
        rtlib::ONB onb(normal);
        newDirection2 = onb.local(rtlib::random_cosine_direction(xor32));
        cosine2 = rtlib::dot(normal, newDirection2);

    }
    {
        rtlib::ONB onb(reflectDir);
        const auto cosTht = powf(rtlib::random_float1(0.0f, 1.0f, xor32), 1.0f / (shinness + 1.0f));
        const auto sinTht = sqrtf(1.0f - cosTht * cosTht);
        const auto phi = rtlib::random_float1(0.0f, RTLIB_M_2PI, xor32);
        newDirection3 = onb.local(make_float3(sinTht * cosf(phi), sinTht * sinf(phi), cosTht));
        cosine3 = rtlib::dot(normal, newDirection3);
    }
    const auto  a_diffuse  = (diffuse.x + diffuse.y + diffuse.z) / 3.0f;
    const auto  a_specular = (specular.x + specular.y + specular.z) / 3.0f;
    const float rnd1       = rtlib::random_float1(xor32);
    const float rnd2       = rtlib::random_float1(xor32);

    if (rnd1 < a_diffuse) {
        const auto  newDirection = rnd2 < 0.5f ? newDirection1 : newDirection2;
        const auto  cosine       = rnd2 < 0.5f ?       cosine1 :       cosine2;
        const auto  bsdfPdf      = rtlib::max(cosine / RTLIB_M_PI     , 0.0f);
        const auto  bsdfVal      = diffuse / (RTLIB_M_PI * a_diffuse);
        //両方とも正なら
        if (params.isBuilt) {
            const auto  dTreePdf = rtlib::max(dTree->Pdf(newDirection), 0.0f);
            const auto  woPdf = 0.5f * dTreePdf + 0.5f * bsdfPdf;
            prd->bsdfVal  = bsdfVal;
            prd->dTreePdf = dTreePdf;
            prd->bsdfPdf  = bsdfPdf;
            prd->woPdf    = woPdf;
            prd->throughPut *= (bsdfVal * rtlib::max(cosine, 0.0f) / woPdf);
            prd->cosine = cosine;
            setRayDirection(newDirection);
        }
        else {
            prd->bsdfVal = (diffuse / (RTLIB_M_PI * a_diffuse));
            prd->bsdfPdf = fabsf(cosine2) / RTLIB_M_PI;
            prd->dTreePdf = 0.0f;
            prd->woPdf = prd->bsdfPdf;
            prd->throughPut *= (diffuse / a_diffuse);
            prd->cosine = cosine2;
            setRayDirection(newDirection2);
        }
    }
    else if (rnd1 < a_diffuse + a_specular)
    {
        const auto  newDirection = rnd2 < 0.5f ? newDirection1 : newDirection3;
        const auto  cosine       = rnd2 < 0.5f ? cosine1       : cosine3;
        const auto  bsdfPdf      = (shinness + 2.0f) * powf(rtlib::max(rtlib::dot(reflectDir, newDirection), 0.0f), shinness) / RTLIB_M_2PI;
        const auto  bsdfVal      = specular * bsdfPdf / a_specular;
        //両方とも正なら
        if (params.isBuilt)
        {
            const auto  dTreePdf = rtlib::max(dTree->Pdf(newDirection), 0.0f);
            const auto  woPdf    = 0.5f * dTreePdf + 0.5f * bsdfPdf;
            //printf("Hit1! %f %f\n", woPdf,dTreePdf);
            prd->dTreePdf        = dTreePdf;
            prd->bsdfPdf         = bsdfPdf;
            prd->woPdf           = woPdf;
            prd->bsdfVal         = bsdfVal;
            prd->throughPut     *= (bsdfVal * rtlib::max(cosine, 0.0f) / woPdf);
            prd->cosine          = cosine;
            setRayDirection(newDirection);
        }
        else {
            const auto reflCos   = rtlib::max(rtlib::dot(reflectDir, newDirection3), 0.0f);
            prd->bsdfPdf         = (shinness + 2.0f) * powf(reflCos, shinness) / RTLIB_M_2PI;
            prd->dTreePdf        = 0.0f;
            prd->woPdf           = prd->bsdfPdf;
            prd->bsdfVal         = (specular * prd->bsdfPdf / a_specular);
            prd->throughPut     *= (specular * rtlib::max(cosine3, 0.0f) / a_specular);
            prd->cosine          = cosine3;
            setRayDirection(newDirection3);
        }
    }
    else {
        //printf("Hit!\n");
        //反射しない
        prd->bsdfVal  = make_float3(1.0f);
        prd->woPdf    = 0.0f;
        prd->bsdfPdf  = 0.0f;
        prd->dTreePdf = 0.0f;
        prd->cosine   = 0.0f;
        prd->throughPut = make_float3(0.0f);
        prd->dTree    = nullptr;
        prd->done     = true;
    }
    prd->countEmitted = true;
    prd->seed = xor32.m_seed;
}
extern "C" __global__ void __closesthit__radiance_for_phong_pg_nee() {
    auto* hgData = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());
    const auto rayDirection = optixGetWorldRayDirection();
    const auto primitiveID = optixGetPrimitiveIndex();

    const auto v0 = optixTransformPointFromObjectToWorldSpace(hgData->vertices[hgData->indices[primitiveID].x]);
    const auto v1 = optixTransformPointFromObjectToWorldSpace(hgData->vertices[hgData->indices[primitiveID].y]);
    const auto v2 = optixTransformPointFromObjectToWorldSpace(hgData->vertices[hgData->indices[primitiveID].z]);

    const auto n0 = optixTransformNormalFromObjectToWorldSpace(rtlib::normalize(rtlib::cross(v1 - v0, v2 - v0)));
    const auto normal = faceForward(n0, make_float3(-rayDirection.x, -rayDirection.y, -rayDirection.z), n0);

    const auto barycentric = optixGetTriangleBarycentrics();
    const auto t0 = hgData->texCoords[hgData->indices[primitiveID].x];
    const auto t1 = hgData->texCoords[hgData->indices[primitiveID].y];
    const auto t2 = hgData->texCoords[hgData->indices[primitiveID].z];


    const auto reflectDir = rtlib::normalize(rayDirection - 2.0f * rtlib::dot(rayDirection, normal) * normal);

    const auto texCoord = (1.0f - barycentric.x - barycentric.y) * t0 + barycentric.x * t1 + barycentric.y * t2;
    const auto emission = hgData->getEmissionColor(texCoord);
    const auto diffuse = hgData->getDiffuseColor(texCoord);
    const auto specular = hgData->getSpecularColor(texCoord);
    const auto shinness = hgData->shinness;
    const auto distance = optixGetRayTmax();
    const auto position = optixGetWorldRayOrigin() + distance * rayDirection;
    //direction
    float3 newDirection1 = make_float3(0.0f);
    float3 newDirection2 = make_float3(0.0f);
    float3 newDirection3 = make_float3(0.0f);
    //cosine
    float  cosine1 = 0.0f;
    float  cosine2 = 0.0f;
    float  cosine3 = 0.0f;
    //payLoad
    RadiancePRD* prd    = getRadiancePRD();
    auto dTreeVoxelSize = make_float3(0.0f);
    const auto dTree    = params.sdTree.GetDTreeWrapper(position, dTreeVoxelSize);
    auto prvThroughPut  = prd->throughPut;
    prd->dTree          = dTree;
    prd->dTreeVoxelSize = dTreeVoxelSize;
    prd->radiance       = emission * prvThroughPut * static_cast<float>(prd->countEmitted);
    prd->distance       = distance;
    prd->isDelta        = false;
    if (prd->done) {
        return;
    }
    rtlib::Xorshift32 xor32(prd->seed);
    //const auto isValid = false;
    setRayOrigin(position);
    if (params.isBuilt) {
        newDirection1 = dTree->Sample(xor32);
        cosine1 = rtlib::dot(normal, newDirection1);
    }
    {
        rtlib::ONB onb(normal);
        newDirection2 = onb.local(rtlib::random_cosine_direction(xor32));
        cosine2 = rtlib::dot(normal, newDirection2);

    }
    {
        rtlib::ONB onb(reflectDir);
        const auto cosTht = powf(rtlib::random_float1(0.0f, 1.0f, xor32), 1.0f / (shinness + 1.0f));
        const auto sinTht = sqrtf(1.0f - cosTht * cosTht);
        const auto phi = rtlib::random_float1(0.0f, RTLIB_M_2PI, xor32);
        newDirection3 = onb.local(make_float3(sinTht * cosf(phi), sinTht * sinf(phi), cosTht));
        cosine3 = rtlib::dot(normal, newDirection3);
    }
    const auto  a_diffuse = (diffuse.x + diffuse.y + diffuse.z) / 3.0f;
    const auto  a_specular = (specular.x + specular.y + specular.z) / 3.0f;
    const float rnd1 = rtlib::random_float1(xor32);
    const float rnd2 = rtlib::random_float1(xor32);

    if (rnd1 < a_diffuse) {
        const auto  newDirection = rnd2 < 0.5f ? newDirection1 : newDirection2;
        const auto  cosine = rnd2 < 0.5f ? cosine1 : cosine2;
        const auto  bsdfPdf = rtlib::max(cosine / RTLIB_M_PI, 0.0f);
        const auto  bsdfVal = diffuse / (RTLIB_M_PI * a_diffuse);
        //両方とも正なら
        if (params.isBuilt) {
            const auto  dTreePdf = rtlib::max(dTree->Pdf(newDirection), 0.0f);
            const auto  woPdf = 0.5f * dTreePdf + 0.5f * bsdfPdf;
            prd->bsdfVal = bsdfVal;
            prd->dTreePdf = dTreePdf;
            prd->bsdfPdf = bsdfPdf;
            prd->woPdf = woPdf;
            prd->throughPut *= (bsdfVal * rtlib::max(cosine, 0.0f) / woPdf);
            prd->cosine = cosine;
            setRayDirection(newDirection);
        }
        else {
            prd->bsdfVal = (diffuse / (RTLIB_M_PI * a_diffuse));
            prd->bsdfPdf = fabsf(cosine2) / RTLIB_M_PI;
            prd->dTreePdf = 0.0f;
            prd->woPdf = prd->bsdfPdf;
            prd->throughPut *= (diffuse / a_diffuse);
            prd->cosine = cosine2;
            setRayDirection(newDirection2);
        }
        {
            const float2 z = rtlib::random_float2(xor32);
            const auto   light = params.light;
            const float3 lightPos = light.corner + light.v1 * z.x + light.v2 * z.y;
            const float  Ldist = rtlib::distance(lightPos, position);
            const float3 lightDir = rtlib::normalize(lightPos - position);
            const float  ndl = rtlib::dot(normal, lightDir);
            const float  lndl = -rtlib::dot(light.normal, lightDir);
            const auto  diffuseLobe = diffuse / (a_diffuse*RTLIB_M_PI);
            float weight = 0.0f;
            if (ndl > 0.0f && lndl > 0.0f) {
                const bool occluded = traceOccluded(params.gasHandle, position, lightDir, 0.01f, Ldist - 0.01f);
                if (!occluded) {
                    //printf("not Occluded!\n");
                    const float A = rtlib::length(rtlib::cross(light.v1, light.v2));
                    weight = ndl * lndl * A / (Ldist * Ldist);
                }
            }
            prd->radiance += light.emission * prvThroughPut * weight * diffuseLobe;
        }
        prd->countEmitted = false;
    }
    else if (rnd1 < a_diffuse + a_specular)
    {
        const auto  newDirection = rnd2 < 0.5f ? newDirection1 : newDirection3;
        const auto  cosine = rnd2 < 0.5f ? cosine1 : cosine3;
        const auto  bsdfPdf = (shinness + 2.0f) * powf(rtlib::max(rtlib::dot(reflectDir, newDirection), 0.0f), shinness) / RTLIB_M_2PI;
        const auto  bsdfVal = specular * bsdfPdf / a_specular;
        //両方とも正なら
        if (params.isBuilt)
        {
            const auto  dTreePdf = rtlib::max(dTree->Pdf(newDirection), 0.0f);
            const auto  woPdf = 0.5f * dTreePdf + 0.5f * bsdfPdf;
            //printf("Hit1! %f %f\n", woPdf,dTreePdf);
            prd->dTreePdf = dTreePdf;
            prd->bsdfPdf = bsdfPdf;
            prd->woPdf = woPdf;
            prd->bsdfVal = bsdfVal;
            prd->throughPut *= (bsdfVal * rtlib::max(cosine, 0.0f) / woPdf);
            prd->cosine = cosine;
            setRayDirection(newDirection);
        }
        else {
            const auto reflCos = rtlib::max(rtlib::dot(reflectDir, newDirection3), 0.0f);
            prd->bsdfPdf = (shinness + 2.0f) * powf(reflCos, shinness) / RTLIB_M_2PI;
            prd->dTreePdf = 0.0f;
            prd->woPdf = prd->bsdfPdf;
            prd->bsdfVal = (specular * prd->bsdfPdf / a_specular);
            prd->throughPut *= (specular * rtlib::max(cosine3, 0.0f) / a_specular);
            prd->cosine = cosine3;
            setRayDirection(newDirection3);
        }
        prd->countEmitted = true;
    }
    else {
        //printf("Hit!\n");
        //反射しない
        prd->bsdfVal = make_float3(1.0f);
        prd->woPdf = 0.0f;
        prd->bsdfPdf = 0.0f;
        prd->dTreePdf = 0.0f;
        prd->cosine = 0.0f;
        prd->throughPut = make_float3(0.0f);
        prd->dTree = nullptr;
        prd->done = true;
    }
    prd->seed = xor32.m_seed;
}
