#define __CUDACC__
#include "RayTrace.h"
#include "PathGuiding.h"
struct RadiancePRD {
    DTreeWrapper* dTree;
    float3        emission;
    float3        bsdfVal;
    float3        throughPut;
    float         woPdf, bsdfPdf, dTreePdf;
    float         distance;
    unsigned int  seed;
    bool          isDelta;
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
    unsigned int seed = params.seed[params.width * idx.y + idx.x];
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
        prd.emission = make_float3(0.0f);
        prd.bsdfVal = make_float3(1.0f);
        prd.throughPut = make_float3(1.0f);
        prd.woPdf = prd.bsdfPdf = prd.dTreePdf = 0.0f;
        prd.distance = 0.0f;
        prd.done = false;
        prd.isDelta = false;
        prd.seed = seed;
        int depth = 0;
        for (;;) {
            float3 prvThroughPut = prd.throughPut;
            //
            traceRadiance(params.gasHandle, rayOrigin, rayDirection, 0.01f, 1e16f, &prd);
            //vertices�̍X�V
            //Radiance�̍X�V
            //Result�̍X�V
            result += prvThroughPut * prd.emission;
            //ThroughPut�̍X�V
            if (prd.done || depth >= params.maxTraceDepth) {
                break;
            }
            depth++;
        }
        seed = prd.seed;
    } while (i--);
    const float3 prevAccumColor = params.accumBuffer[params.width * idx.y + idx.x];
    const float3 accumColor = prevAccumColor + result;
    float3 frameColor = accumColor / (static_cast<float>(params.samplePerALL + params.samplePerLaunch));
    frameColor = frameColor / (make_float3(1.0f, 1.0f, 1.0f) + frameColor);
    //if (idx.x == 500 && idx.y  == 500) {
        //printf("%f %f %f\n", frameColor.x, frameColor.y, frameColor.z);
    //}
    params.frameBuffer[params.width * idx.y + idx.x] = make_uchar4(
        static_cast<unsigned char>(255.99 * rtlib::linear_to_gamma(frameColor.x)),
        static_cast<unsigned char>(255.99 * rtlib::linear_to_gamma(frameColor.y)),
        static_cast<unsigned char>(255.99 * rtlib::linear_to_gamma(frameColor.z)), 255);
    params.accumBuffer[params.width * idx.y + idx.x] = accumColor;
    params.seed[params.width * idx.y + idx.x] = seed;
}
extern "C" __global__ void __raygen__pg() {

    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();
    auto* rgData = reinterpret_cast<RayGenData*>(optixGetSbtDataPointer());
    const float3 u = rgData->u;
    const float3 v = rgData->v;
    const float3 w = rgData->w;
    unsigned int seed = params.seed[params.width * idx.y + idx.x];
    float3 result = make_float3(0.0f, 0.0f, 0.0f);
    size_t i = params.samplePerLaunch;
    TraceVertex vertices[32] = {};
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
        prd.emission = make_float3(0.0f);
        prd.bsdfVal = make_float3(1.0f);
        prd.throughPut = make_float3(1.0f);
        prd.woPdf = prd.bsdfPdf = prd.dTreePdf = 0.0f;
        prd.distance = 0.0f;
        prd.done = false;
        prd.isDelta = false;
        prd.seed = seed;
        int depth = 0;
        for (;;) {
            float3 prvThroughPut         = prd.throughPut;
            traceRadiance(params.gasHandle, rayOrigin, rayDirection, 0.01f, 1e16f, &prd);
            vertices[depth].rayOrigin    = rayOrigin;
            vertices[depth].rayDirection = rayDirection;
            vertices[depth].dTree        = prd.dTree;
            vertices[depth].throughPut   = prd.throughPut;
            vertices[depth].bsdfVal      = prd.bsdfVal;
            vertices[depth].radiance     = make_float3(0.0f);
            vertices[depth].woPdf        = 1.0f;
            vertices[depth].bsdfPdf      = prd.bsdfPdf;
            vertices[depth].dTreePdf     = prd.dTreePdf;
            vertices[depth].isDelta      = prd.isDelta;
            for (int j = 0; j < depth; ++j) {
                vertices[j].Record(prvThroughPut * prd.emission);
            }
            //OK
            //Result�̍X�V
            result += prvThroughPut * prd.emission;
            if (isnan(result.x) || isnan(result.y) || isnan(result.z)) {
                printf("result is nan Bug(%d,%d,%d): Result(%f %f %f) prvThroughPut(%lf %lf %lf) prd.emission(%lf %lf %lf) prd.pdf: (%lf %lf %lf) prd.done: %d\n",
                    idx.x, idx.y, depth,
                    result.x, result.y, result.z,
                    prvThroughPut.x, prvThroughPut.y, prvThroughPut.z,
                    prd.emission.x, prd.emission.y, prd.emission.z,
                    prd.woPdf, prd.bsdfPdf, prd.dTreePdf, (int)prd.done
                );
            }
            // if (idx.x==100&&idx.y==100){
            //     printf("(%d,%d,%d)= %f %f %f\n",idx.x,idx.y,depth, prvThroughPut.x,prvThroughPut.y,prvThroughPut.z);
            // }
            bool isValidThroughPut = !(isnan(prd.throughPut.x)   || isnan(prd.throughPut.y)     || isnan(prd.throughPut.z)     ||
                                     !isfinite(prd.throughPut.x) || !isfinite(prd.throughPut.y) || !isfinite(prd.throughPut.z) ||
                                        prd.throughPut.x <= 0.0f || prd.throughPut.y <= 0.0f    || prd.throughPut.z <= 0.0f);
            //ThroughPut�̍X�V
            //if (!isValidThroughPut) {
               // printf("prd.ThroughPut is Invalid Bug(%d,%d,%d): Result(%f %f %f)  prd.throughPut(%lf %lf %lf) prd.emission(%lf %lf %lf) prd.pdf: (%lf %lf %lf) prd.done: %d\n",
                    //idx.x, idx.y, depth,
                   // result.x, result.y, result.z,
                  //  prd.throughPut.x, prd.throughPut.y, prd.throughPut.z,
               //     prd.emission.x, prd.emission.y, prd.emission.z,
                //    prd.woPdf, prd.bsdfPdf, prd.dTreePdf, (int)prd.done
            //    );
      //      }
            if (prd.done || depth >= params.maxTraceDepth || !isValidThroughPut) {
                break;
            }
            depth++;
        }
        for (int j = 0; j < depth; ++j) {
            vertices[j].Commit(1.0f);
        }
        seed = prd.seed;
    } while (i--);
    const float3 prevAccumColor = params.accumBuffer[params.width * idx.y + idx.x];
    const float3 accumColor = prevAccumColor + result;

    float3 frameColor = accumColor / (static_cast<float>(params.samplePerALL + params.samplePerLaunch));
    frameColor = frameColor / (make_float3(1.0f, 1.0f, 1.0f) + frameColor);
    //if (idx.x == 500 && idx.y  == 500) {
        //printf("%f %f %f\n", frameColor.x, frameColor.y, frameColor.z);
    //}
    params.frameBuffer[params.width * idx.y + idx.x] = make_uchar4(
        static_cast<unsigned char>(255.99 * rtlib::linear_to_gamma(frameColor.x)),
        static_cast<unsigned char>(255.99 * rtlib::linear_to_gamma(frameColor.y)),
        static_cast<unsigned char>(255.99 * rtlib::linear_to_gamma(frameColor.z)), 255);
    params.accumBuffer[params.width * idx.y + idx.x] = accumColor;
    params.seed[params.width * idx.y + idx.x] = seed;
}
extern "C" __global__ void __miss__radiance() {
    auto* msData = reinterpret_cast<MissData*>(optixGetSbtDataPointer());
    RadiancePRD* prd = getRadiancePRD();

    prd->emission = make_float3(msData->bgColor.x, msData->bgColor.y, msData->bgColor.z);
    prd->dTree    = nullptr;
    prd->woPdf    = 1.0f;
    prd->bsdfPdf  = 0.0f;
    prd->dTreePdf = 0.0f;
    prd->bsdfVal  = make_float3(1.0f);
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
    const auto texCoord   = (1.0f - barycentric.x - barycentric.y) * t0 + barycentric.x * t1 + barycentric.y * t2;
    const auto distance   = optixGetRayTmax();
    const float3 position = optixGetWorldRayOrigin() + distance * rayDirection;
    float3 newDirection   = make_float3(0.0f);
    RadiancePRD* prd      = getRadiancePRD();

    prd->dTree = nullptr;
    prd->emission = make_float3(0.0f, 0.0f, 0.0f);
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

    float3 diffuse   = hgData->getDiffuseColor(texCoord);
    prd->bsdfVal     = diffuse / RTLIB_M_PI;
    prd->throughPut *= diffuse;
    prd->seed        = xor32.m_seed;
    prd->isDelta     = false;
}
extern "C" __global__ void __closesthit__radiance_for_diffuse_pg() {

    auto* hgData              = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());
    const float3 rayDirection = optixGetWorldRayDirection();
    const int    primitiveID  = optixGetPrimitiveIndex();
    const float3 v0           = optixTransformPointFromObjectToWorldSpace(hgData->vertices[hgData->indices[primitiveID].x]);
    const float3 v1           = optixTransformPointFromObjectToWorldSpace(hgData->vertices[hgData->indices[primitiveID].y]);
    const float3 v2           = optixTransformPointFromObjectToWorldSpace(hgData->vertices[hgData->indices[primitiveID].z]);
    const float3 n0           = optixTransformNormalFromObjectToWorldSpace(rtlib::normalize(rtlib::cross(v1 - v0, v2 - v0)));
    const float3 normal       = faceForward(n0, make_float3(-rayDirection.x, -rayDirection.y, -rayDirection.z), n0);
    const float2 barycentric  = optixGetTriangleBarycentrics();
    const auto t0 = hgData->texCoords[hgData->indices[primitiveID].x];
    const auto t1 = hgData->texCoords[hgData->indices[primitiveID].y];
    const auto t2 = hgData->texCoords[hgData->indices[primitiveID].z];
    const auto texCoord = (1.0f - barycentric.x - barycentric.y) * t0 + barycentric.x * t1 + barycentric.y * t2;
    const auto diffuse = hgData->getDiffuseColor(texCoord);
    const float3 position = optixGetWorldRayOrigin() + optixGetRayTmax() * rayDirection;
    float3 newDirection1 = make_float3(0.0f);
    float3 newDirection2 = make_float3(0.0f);
    float  cosine1 = 0.0f;
    float  cosine2 = 0.0f;
    RadiancePRD* prd = getRadiancePRD();

    prd->dTree = params.sdTree.GetDTreeWrapper(position);
    prd->emission = make_float3(0.0f, 0.0f, 0.0f);
    prd->distance = optixGetRayTmax();
    prd->bsdfPdf = 0.0f;
    prd->bsdfVal = diffuse / RTLIB_M_PI;
    prd->isDelta = false;
    rtlib::Xorshift32 xor32(prd->seed);

    setRayOrigin(position);
    {
        rtlib::ONB onb(normal);
        do {
            newDirection1 = onb.local(rtlib::random_cosine_direction(xor32));
            cosine1 = rtlib::dot(normal, newDirection1);
        } while (cosine1 == 0.0f);

        if (isnan(newDirection1.x) || isnan(newDirection1.y) || isnan(newDirection1.z)) {
            printf("newDirection1 is nan: new Direction1 = (%f, %f, %f) normal = (%f, %f, %f) n0 = (%f, %f, %f)\n", newDirection1.x, newDirection1.y, newDirection1.z, normal.x, normal.y, normal.z, n0.x, n0.y, n0.z);
        }
    }
    do {
        newDirection2 = prd->dTree->Sample(xor32);
        cosine2 = rtlib::dot(normal, newDirection2);
    } while (cosine2 == 0.0f);
    if (isnan(newDirection2.x) || isnan(newDirection2.y) || isnan(newDirection2.z))
    {
        printf("newDirection2 is nan!\n");
    }

    const float rnd = rtlib::random_float1(xor32);
    const auto  newDirection = rnd < 0.5f ? newDirection1 : newDirection2;
    const auto  cosine   = rnd < 0.5f ? cosine1 : cosine2;
    const auto  bsdfPdf  = cosine / RTLIB_M_PI;
    const auto  dTreePdf = prd->dTree->Pdf(newDirection);
    const auto  isValid  = prd->dTree->sampling.GetMean() > 0.0f;
    //両方とも正なら
    if (bsdfPdf > 0.0f && dTreePdf > 0.0f && isValid) {
        prd->bsdfPdf     = bsdfPdf;
        prd->dTreePdf    = dTreePdf;
        prd->woPdf       = (0.5f * prd->dTreePdf + 0.5f * prd->woPdf);
        prd->throughPut *= (prd->bsdfVal * fabsf(cosine)/ prd->woPdf);
        setRayDirection(newDirection);
    }
    else {
        prd->bsdfPdf     = fabsf(cosine1) / RTLIB_M_PI;
        prd->dTreePdf    = 0.0f;
        prd->woPdf       = prd->bsdfPdf;
        prd->throughPut *= (prd->bsdfVal * fabsf(cosine1) / prd->woPdf);
        setRayDirection(newDirection1);
    }
    prd->seed = xor32.m_seed;
}
extern "C" __global__ void __closesthit__radiance_for_specular() {

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
    const float3 position = optixGetWorldRayOrigin() + optixGetRayTmax() * rayDirection;
    RadiancePRD* prd = getRadiancePRD();
    prd->dTree = nullptr;
    prd->emission = make_float3(0.0f, 0.0f, 0.0f);
    prd->distance = optixGetRayTmax();
    {
        float3 specular = hgData->getSpecularColor(texCoord);
        float3 reflectDir = rtlib::normalize(rayDirection - 2.0f * rtlib::dot(rayDirection, normal) * normal);

        prd->woPdf = 0.0f;
        prd->dTreePdf = 0.0f;
        prd->bsdfPdf = std::fabsf(rtlib::dot(reflectDir, normal));

        setRayOrigin(position);
        setRayDirection(reflectDir);


        prd->bsdfVal = specular;
        prd->throughPut *= prd->bsdfVal;
        prd->isDelta = true;
    }
}
extern "C" __global__ void __closesthit__radiance_for_refraction() {
    auto* hgData = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());
    const float3 rayDirection = optixGetWorldRayDirection();
    const int    primitiveID = optixGetPrimitiveIndex();
    const float3 v0 = optixTransformPointFromObjectToWorldSpace(hgData->vertices[hgData->indices[primitiveID].x]);
    const float3 v1 = optixTransformPointFromObjectToWorldSpace(hgData->vertices[hgData->indices[primitiveID].y]);
    const float3 v2 = optixTransformPointFromObjectToWorldSpace(hgData->vertices[hgData->indices[primitiveID].z]);
    const float3 n0 = optixTransformNormalFromObjectToWorldSpace(rtlib::normalize(rtlib::cross(v1 - v0, v2 - v0)));
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
    const float2 barycentric = optixGetTriangleBarycentrics();
    const auto t0 = hgData->texCoords[hgData->indices[primitiveID].x];
    const auto t1 = hgData->texCoords[hgData->indices[primitiveID].y];
    const auto t2 = hgData->texCoords[hgData->indices[primitiveID].z];
    const auto texCoord = (1.0f - barycentric.x - barycentric.y) * t0 + barycentric.x * t1 + barycentric.y * t2;
    const float3 position = optixGetWorldRayOrigin() + optixGetRayTmax() * rayDirection;
    RadiancePRD* prd = getRadiancePRD();
    prd->dTree = nullptr;
    prd->emission = make_float3(0.0f, 0.0f, 0.0f);
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
            prd->woPdf   = prd->dTreePdf = 0.0f;
            prd->bsdfPdf = std::fabsf(rtlib::dot(reflectDir, normal));
            //printf("reflect: %lf %lf %lf\n", reflectDir.x, reflectDir.y, reflectDir.z);
            setRayOrigin(position + 0.001f * normal);
            setRayDirection(reflectDir);

            prd->bsdfVal     = specular;
            prd->throughPut *= prd->bsdfVal;
        }
        else {
            float  cosine_o   = sqrtf(1.0f - sine_o_2);
            float3 refractDir = rtlib::normalize((rayDirection - (cosine_o - cosine_i) * normal) / refInd);
            prd->woPdf   = prd->dTreePdf = 0.0f;
            prd->bsdfPdf = std::fabsf(rtlib::dot(refractDir, normal));
            //printf("refract: %lf %lf %lf\n", refractDir.x, refractDir.y, refractDir.z);
            setRayOrigin(position - 0.001f * normal);
            setRayDirection(refractDir);

            prd->bsdfVal     = make_float3(1.0f);
            prd->throughPut *= prd->bsdfVal;
        }
        prd->isDelta = true;
    }
    prd->seed = xor32.m_seed;
}
extern "C" __global__ void __closesthit__radiance_for_emission() {
    auto* hgData = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());
    const float3 rayDirection = optixGetWorldRayDirection();
    const int    primitiveID = optixGetPrimitiveIndex();
    const float3 v0 = optixTransformPointFromObjectToWorldSpace(hgData->vertices[hgData->indices[primitiveID].x]);
    const float3 v1 = optixTransformPointFromObjectToWorldSpace(hgData->vertices[hgData->indices[primitiveID].y]);
    const float3 v2 = optixTransformPointFromObjectToWorldSpace(hgData->vertices[hgData->indices[primitiveID].z]);
    //const float3 n0 = rtlib::normalize(rtlib::cross(v1 - v0, v2 - v0));
    //const float3 normal = faceForward(n0, make_float3(-rayDirection.x, -rayDirection.y, -rayDirection.z), n0);
    const float2 barycentric = optixGetTriangleBarycentrics();
    const auto t0 = hgData->texCoords[hgData->indices[primitiveID].x];
    const auto t1 = hgData->texCoords[hgData->indices[primitiveID].y];
    const auto t2 = hgData->texCoords[hgData->indices[primitiveID].z];
    const auto texCoord = (1.0f - barycentric.x - barycentric.y) * t0 + barycentric.x * t1 + barycentric.y * t2;
    const auto distance = optixGetRayTmax();
    const float3 position = optixGetWorldRayOrigin() + distance * rayDirection;
    RadiancePRD* prd = getRadiancePRD();
    prd->emission    = hgData->getEmissionColor(texCoord);
    prd->bsdfVal     = make_float3(1.0f);
    prd->woPdf       = 0.0f;
    prd->bsdfPdf     = 0.0f;
    prd->dTreePdf    = 0.0f;
    prd->dTree       = nullptr;
    prd->distance    = distance;
    prd->done        = true;
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
    const auto diffuse  = hgData->getDiffuseColor(texCoord);
    const auto specular = hgData->getSpecularColor(texCoord);
    const auto shinness = hgData->shinness;
    const auto emission = hgData->getEmissionColor(texCoord);
    const auto distance = optixGetRayTmax();
    const float3 position = optixGetWorldRayOrigin() + distance * rayDirection;
    RadiancePRD* prd = getRadiancePRD();

    prd->dTree    = nullptr;
    prd->emission = emission;
    prd->distance = distance;

    rtlib::Xorshift32 xor32(prd->seed);

    auto  newDirection = make_float3(0.0f);
    auto  cosine = 0.0f;
    auto  weight = make_float3(0.0f);

    const auto rnd = rtlib::random_float1(xor32);
    const auto a_diffuse = (diffuse.x + diffuse.y + diffuse.z) / 3.0f;
    const auto a_specular = (specular.x + specular.y + specular.z) / 3.0f;
    if (rnd < a_diffuse) {
        rtlib::ONB onb(normal);
        newDirection  = onb.local(rtlib::random_cosine_direction(xor32));
        cosine        = rtlib::dot(newDirection, normal);
        prd->bsdfVal  = diffuse / (a_diffuse * RTLIB_M_PI);
        prd->bsdfPdf  = fabsf(cosine) / RTLIB_M_PI;
        prd->dTreePdf = 0.0f;
        prd->woPdf    = prd->bsdfPdf;
        prd->throughPut *= (diffuse/a_diffuse);
    }
    else if (rnd < a_diffuse + a_specular) {
        const auto cosTht = powf(rtlib::random_float1(0.0f, 1.0f, xor32), 1.0f / (shinness + 1.0f));
        const auto sinTht = sqrtf(1.0f - cosTht * cosTht);
        const auto phi = rtlib::random_float1(0.0f, RTLIB_M_2PI, xor32);
        rtlib::ONB onb(reflectDir);
        newDirection = onb.local(make_float3(sinTht * cosf(phi), sinTht * sinf(phi), cosTht));
        cosine       = rtlib::dot(newDirection, normal);
        prd->bsdfVal = (specular / a_specular) * (shinness + 2.0f) * powf(fabsf(rtlib::dot(reflectDir, newDirection)), shinness) / RTLIB_M_2PI;
        prd->bsdfPdf = rtlib::max((shinness + 2.0f) * powf(fabsf(rtlib::dot(reflectDir, newDirection)), shinness) / RTLIB_M_2PI, 1e-7f);
        prd->dTreePdf= 0.0f;
        prd->woPdf   = prd->bsdfPdf;
        prd->throughPut *= (specular * fabsf(cosine) / a_specular);
    }

    setRayOrigin(position);
    setRayDirection(newDirection);
    prd->seed = xor32.m_seed;
    prd->isDelta = false;
}
extern "C" __global__ void __closesthit__radiance_for_phong_pg () {
    auto* hgData = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());
    const auto primitiveID  = optixGetPrimitiveIndex();
    const auto v0           = optixTransformPointFromObjectToWorldSpace(hgData->vertices[hgData->indices[primitiveID].x]);
    const auto v1           = optixTransformPointFromObjectToWorldSpace(hgData->vertices[hgData->indices[primitiveID].y]);
    const auto v2           = optixTransformPointFromObjectToWorldSpace(hgData->vertices[hgData->indices[primitiveID].z]);
    const auto n0           = optixTransformNormalFromObjectToWorldSpace(rtlib::normalize(rtlib::cross(v1 - v0, v2 - v0)));
    const auto t0           = hgData->texCoords[hgData->indices[primitiveID].x];
    const auto t1           = hgData->texCoords[hgData->indices[primitiveID].y];
    const auto t2           = hgData->texCoords[hgData->indices[primitiveID].z];
    const auto rayDirection = optixGetWorldRayDirection();
    const auto distance     = optixGetRayTmax();
    const auto position     = optixGetWorldRayOrigin() + distance * rayDirection;
    const auto normal       = faceForward(n0, make_float3(-rayDirection.x, -rayDirection.y, -rayDirection.z), n0);
    const auto reflectDir   = rtlib::normalize(rayDirection - 2.0f * rtlib::dot(rayDirection, normal) * normal);
    const auto barycentric  = optixGetTriangleBarycentrics();
    const auto texCoord     = (1.0f - barycentric.x - barycentric.y) * t0 + barycentric.x * t1 + barycentric.y * t2;
    const auto emission     = hgData->getEmissionColor(texCoord);
    const auto diffuse      = hgData->getDiffuseColor( texCoord);
    const auto specular     = hgData->getSpecularColor(texCoord);
    const auto shinness     = hgData->shinness;
    //direction
    float3 newDirection1    = make_float3(0.0f);
    float3 newDirection2    = make_float3(0.0f);
    float3 newDirection3    = make_float3(0.0f);
    //cosine
    float  cosine1 = 0.0f;
    float  cosine2 = 0.0f;
    float  cosine3 = 0.0f;
    RadiancePRD* prd = getRadiancePRD();

    prd->dTree         = params.sdTree.GetDTreeWrapper(position);
    prd->emission      = emission;
    prd->distance      = distance;
    prd->bsdfPdf       = 0.0f;
    prd->isDelta       = false;
    rtlib::Xorshift32 xor32(prd->seed);
    //const auto isValid = false;
    setRayOrigin(position);
    {
        rtlib::ONB onb(normal);
        //newDirection1   = onb.local(rtlib::random_in_unit_sphere(xor32));
        //cosine1         = rtlib::dot(normal, newDirection1);
        newDirection1 = prd->dTree->Sample(xor32);
        cosine1 = rtlib::dot(normal, newDirection1);
        if (isnan(newDirection1.x) || isnan(newDirection1.y) || isnan(newDirection1.z))
        {
            printf("newDirection1 is nan!\n");
        }
    }
    {
        rtlib::ONB onb(normal);
        newDirection2 = onb.local(rtlib::random_cosine_direction(xor32));
        cosine2       = rtlib::dot(normal, newDirection2);
        if (isnan(newDirection2.x) || isnan(newDirection2.y) || isnan(newDirection2.z)) {
            printf("newDirection2 is nan: newDirection2 = (%f, %f, %f) normal = (%f, %f, %f) n0 = (%f, %f, %f)\n", newDirection2.x, newDirection2.y, newDirection2.z, normal.x, normal.y, normal.z, n0.x, n0.y, n0.z);
        }
    }
    {

        rtlib::ONB onb(reflectDir);
        const auto cosTht = powf(rtlib::random_float1(0.0f, 1.0f, xor32), 1.0f / (shinness + 1.0f));
        const auto sinTht = sqrtf(1.0f - cosTht * cosTht);
        const auto phi    = rtlib::random_float1(0.0f, RTLIB_M_2PI, xor32);
        newDirection3     = onb.local(make_float3(sinTht * cosf(phi), sinTht * sinf(phi), cosTht));
        cosine3 = rtlib::dot(normal, newDirection3);

        if (isnan(newDirection3.x) || isnan(newDirection3.y) || isnan(newDirection3.z))
        {
            printf("newDirection3 is nan: newDirection3 = (%f, %f, %f) normal = (%f, %f, %f) n0 = (%f, %f, %f)\n", newDirection3.x, newDirection3.y, newDirection3.z, normal.x, normal.y, normal.z, n0.x, n0.y, n0.z);
        }
    }
    const auto  a_diffuse    = ( diffuse.x +  diffuse.y +  diffuse.z) / 3.0f;
    const auto  a_specular   = (specular.x + specular.y + specular.z) / 3.0f;
    const float rnd1         = rtlib::random_float1(xor32);
    const float rnd2         = rtlib::random_float1(xor32);

    if (rnd1 < a_diffuse){
        const auto  newDirection = rnd2 < 0.5f ? newDirection1 : newDirection2;
        const auto  cosine       = rnd2 < 0.5f ?       cosine1 :       cosine2;
        const auto  bsdfPdf      = rtlib::max(cosine / RTLIB_M_PI,0.0f);
        const auto  dTreePdf     = rtlib::max(prd->dTree->Pdf(newDirection),0.0f);
        //const auto  dTreePdf     = 1.0f / (4.0f * RTLIB_M_PI);
        //両方とも正なら
        if (params.isBuilt) {
            //printf("Hit1! %f %f\n", woPdf,dTreePdf);
            prd->bsdfVal       = (diffuse /(RTLIB_M_PI*a_diffuse));
            prd->dTreePdf      = dTreePdf;
            prd->bsdfPdf       =  bsdfPdf;
            prd->woPdf         = 0.5f * dTreePdf+0.5f* bsdfPdf;
            prd->throughPut   *= (prd->bsdfVal * rtlib::max(cosine, 0.0f) / prd->woPdf);
            setRayDirection(newDirection);
            if (isnan(prd->throughPut.x) || isnan(prd->throughPut.y) || isnan(prd->throughPut.z)) {
                printf("prd->weight0 is nan: %f %f %f\n", prd->woPdf, prd->bsdfPdf, prd->dTreePdf);
            }
        }
        else {
            prd->bsdfVal       = (diffuse / (RTLIB_M_PI * a_diffuse));
            prd->bsdfPdf       = fabsf(cosine2) / RTLIB_M_PI;
            prd->dTreePdf      = 0.0f;
            prd->woPdf         = prd->bsdfPdf;
            prd->throughPut   *= (diffuse/ a_diffuse);
            setRayDirection(newDirection2);
            if (isnan(prd->throughPut.x) || isnan(prd->throughPut.y) || isnan(prd->throughPut.z)) {
                printf("prd->weight1 is nan: %f %f %f\n", prd->woPdf, prd->bsdfPdf, prd->dTreePdf);
            }
        }
    }
    else if (rnd1 < a_diffuse + a_specular)
    {
        const auto  newDirection = rnd2 < 0.5f ? newDirection1 : newDirection3;
        const auto  cosine       = rnd2 < 0.5f ?       cosine1 :       cosine3;
        const auto  bsdfPdf      = rtlib::max((shinness + 2.0f) * powf(fabsf(rtlib::dot(reflectDir, newDirection)), shinness)/ RTLIB_M_2PI,0.0f);
        const auto  dTreePdf     = rtlib::max(prd->dTree->Pdf(newDirection),0.0f);
        //const auto  dTreePdf     = 1.0f / (4.0f * RTLIB_M_PI);
        //両方とも正なら
        if (params.isBuilt) {
            //printf("Hit1! %f %f\n", woPdf,dTreePdf);
            prd->dTreePdf      = dTreePdf;
            prd->bsdfPdf       = bsdfPdf;
            prd->woPdf         = 0.5f * prd->dTreePdf + 0.5f * prd->bsdfPdf;
            prd->bsdfVal       = (specular * prd->bsdfPdf / a_specular);
            prd->throughPut   *= (prd->bsdfVal * rtlib::max(cosine,0.0f)/prd->woPdf);
            setRayDirection(newDirection);
            if (isnan(prd->throughPut.x) || isnan(prd->throughPut.y) || isnan(prd->throughPut.z)) {
                printf("prd->weight2 is nan: %f %f %f\n", prd->woPdf, prd->bsdfPdf, prd->dTreePdf);
            }
        }
        else {
            //printf("Hit2!\n");
            const auto reflCos = fabsf(rtlib::dot(reflectDir, newDirection3));
            prd->bsdfPdf       = rtlib::max((shinness + 2.0f) * powf(reflCos, shinness) / RTLIB_M_2PI, 1e-7f);
            prd->dTreePdf      = 0.0f;
            prd->woPdf         = prd->bsdfPdf;
            prd->bsdfVal       = (specular * prd->bsdfPdf   / a_specular);
            prd->throughPut   *= (specular * rtlib::max(cosine3, 0.0f) / a_specular);
            setRayDirection(newDirection3);
            if (isnan(prd->throughPut.x) || isnan(prd->throughPut.y) || isnan(prd->throughPut.z)) {
                printf("prd->weight3 is nan: (%f %f %f) reflCos = %f\n", prd->woPdf, prd->bsdfPdf, prd->dTreePdf, reflCos);
            }
        }
    }
    else {
        //printf("Hit!\n");
        //反射しない
        prd->bsdfVal  = make_float3(1.0f);
        prd->woPdf    = 0.0f;
        prd->bsdfPdf  = 0.0f;
        prd->dTreePdf = 0.0f;
        prd->dTree    = nullptr;
        prd->done     = true;
    }
    
    prd->seed = xor32.m_seed;
}
