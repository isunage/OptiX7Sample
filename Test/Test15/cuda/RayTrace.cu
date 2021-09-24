#define __CUDACC__
#include "RayTrace.h"
extern "C" {
    __constant__ Params params;
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
static __forceinline__ __device__ void setRadiancePRD(RadiancePRD* prd) {
    unsigned int p0;
    unsigned int p1;
    packPointer(static_cast<void*>(prd), p0, p1);
    optixSetPayload_0(p0);
    optixSetPayload_1(p1);
}
static __forceinline__ __device__ void  setPayloadOccluded(bool occluded) {
    optixSetPayload_0(static_cast<unsigned int>(occluded));
}
static __forceinline__ __device__ void traceRadiance(
    OptixTraversableHandle handle,
    const float3& rayOrigin, 
    const float3& rayDirection,
    float tmin, float tmax,
    RadiancePRD*  prd) {
    unsigned int p0, p1;
    packPointer(prd, p0, p1);
    optixTrace(handle, rayOrigin, rayDirection, tmin, tmax, 0.0f, OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE, RAY_TYPE_RADIANCE, RAY_TYPE_COUNT, RAY_TYPE_RADIANCE, p0, p1);
}
static __forceinline__ __device__ bool traceOccluded(
    OptixTraversableHandle handle,
    const float3& rayOrigin,
    const float3& rayDirection,
    float tmin, float tmax) {
    unsigned int occluded = false;
    optixTrace(handle, rayOrigin, rayDirection, tmin, tmax, 0.0f, OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE, RAY_TYPE_OCCLUSION, RAY_TYPE_COUNT, RAY_TYPE_OCCLUSION, occluded);
    return occluded;
}
extern "C" __global__ void     __raygen__rg(){
    const uint3 idx             = optixGetLaunchIndex();
	const uint3 dim             = optixGetLaunchDimensions();
    auto* rgData                = reinterpret_cast<RayGenData*>(optixGetSbtDataPointer());
    const float3 u              = rgData->u;
	const float3 v              = rgData->v;
	const float3 w              = rgData->w;
    unsigned int seed           = params.seed[params.width * idx.y + idx.x];
    float3 result               = make_float3(0.0f, 0.0f, 0.0f);
    size_t i                    = params.samplePerLaunch;
    do {
        rtlib::Xorshift32 xor32(seed);
        const float2 jitter = rtlib::random_float2(xor32);
        const float2 d      = make_float2(
            ((2.0f * static_cast<float>(idx.x) + jitter.x) / static_cast<float>(dim.x)) - 1.0,
            ((2.0f * static_cast<float>(idx.y) + jitter.y) / static_cast<float>(dim.y)) - 1.0);
        seed                = xor32.m_seed;
        float3 rayOrigin    = rgData->eye;
        float3 rayDirection = rtlib::normalize(d.x * u + d.y * v + w);
        RadiancePRD prd;
        prd.emitted         = make_float3(0.0f, 0.0f, 0.0f);
        prd.radiance        = make_float3(0.0f, 0.0f, 0.0f);
        prd.attenuation     = make_float3(1.0f, 1.0f, 1.0f);
        prd.countEmitted    = true;
        prd.done            = false;
        prd.seed            = seed;
        int depth = 0;
        for (;;) {
            traceRadiance(params.gasHandle, rayOrigin, rayDirection, 0.01f, 1e16f, &prd);
            result += prd.emitted;
            result += prd.radiance * prd.attenuation;
            if (prd.done || depth >= TEST_MAX_TRACE_DEPTH) {
                break;
            }
            rayOrigin    = prd.origin;
            rayDirection = prd.direction;
            depth++;
        }
        seed = prd.seed;
    } while(--i);
    const float3 prevAccumColor = params.accumBuffer[params.width * idx.y + idx.x];
    const float3 accumColor     = prevAccumColor + result;
    float3 frameColor           = accumColor / (static_cast<float>(params.samplePerALL + params.samplePerLaunch));
    frameColor                  = frameColor / (make_float3(1.0f, 1.0f, 1.0f) + frameColor);
    //if (idx.x == 500 && idx.y  == 500) {
        //printf("%f %f %f\n", frameColor.x, frameColor.y, frameColor.z);
    //}
    params.frameBuffer[params.width * idx.y + idx.x] = make_uchar4(
        static_cast<unsigned char>(255.99 * rtlib::linear_to_gamma(frameColor.x)),
        static_cast<unsigned char>(255.99 * rtlib::linear_to_gamma(frameColor.y)),
        static_cast<unsigned char>(255.99 * rtlib::linear_to_gamma(frameColor.z)), 255);
    params.accumBuffer[params.width * idx.y + idx.x] = accumColor;
    params.seed[params.width * idx.y + idx.x]        = seed;
}
extern "C" __global__ void       __miss__radiance(){
    auto* msData = reinterpret_cast<MissData*>(optixGetSbtDataPointer());
    RadiancePRD* prd = getRadiancePRD();
    prd->radiance    = make_float3(msData->bgColor.x, msData->bgColor.y, msData->bgColor.z);
    prd->done        = true;
}
extern "C" __global__ void       __miss__occluded() {
    setPayloadOccluded(false);
}
extern "C" __global__ void __closesthit__radiance_for_diffuse()  {
    auto*        hgData       = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());
    const float3 rayDirection = optixGetWorldRayDirection();
    const int    primitiveID  = optixGetPrimitiveIndex();
    const float3 v0           = optixTransformPointFromObjectToWorldSpace(hgData->vertices[hgData->indices[primitiveID].x]);
    const float3 v1           = optixTransformPointFromObjectToWorldSpace(hgData->vertices[hgData->indices[primitiveID].y]);
    const float3 v2           = optixTransformPointFromObjectToWorldSpace(hgData->vertices[hgData->indices[primitiveID].z]);
    const float3 n0           = optixTransformNormalFromObjectToWorldSpace(rtlib::normalize(rtlib::cross(v1 - v0, v2 - v0)));
    const float3 normal       = faceForward(n0, make_float3(-rayDirection.x,-rayDirection.y,-rayDirection.z), n0);
    const float2 barycentric  = optixGetTriangleBarycentrics();
    const auto t0             = hgData->texCoords[hgData->indices[primitiveID].x];
    const auto t1             = hgData->texCoords[hgData->indices[primitiveID].y];
    const auto t2             = hgData->texCoords[hgData->indices[primitiveID].z];
    const auto texCoord       = (1.0f - barycentric.x - barycentric.y) * t0 + barycentric.x * t1 + barycentric.y * t2;
    const float3 position     = optixGetWorldRayOrigin() + optixGetRayTmax() * rayDirection;
    RadiancePRD* prd          = getRadiancePRD();
    prd->emitted              = make_float3(0.0f, 0.0f, 0.0f);
    rtlib::Xorshift32 xor32(prd->seed);
    {
        rtlib::ONB onb(normal);
        float3 newDirection = onb.local(rtlib::random_cosine_direction(xor32));
        prd->direction      = newDirection;
        prd->origin         = position;
        float3 diffuse      = hgData->getDiffuseColor(texCoord);
        float3 specular     = hgData->getSpecularColor(texCoord);
        float3 reflectDir   = rtlib::normalize(rayDirection - 2.0f * rtlib::dot(rayDirection, normal) * normal);
        float  shinness     = hgData->shinness;
        float cosine        = fabsf(rtlib::dot(newDirection, reflectDir));
        prd->attenuation   *= diffuse + specular * (shinness + 2.0f) * powf(cosine, shinness) / 2.0f;
        prd->countEmitted   = false;
    }
    {
        const float2 z        = rtlib::random_float2(xor32);
        const auto   light    = params.light;
        const float3 lightPos = light.corner + light.v1 * z.x + light.v2 * z.y;
        const float  Ldist    = rtlib::distance(lightPos, position);
        const float3 lightDir = rtlib::normalize(lightPos - position);
        const float  ndl      = rtlib::dot(normal, lightDir);
        const float  lndl     =-rtlib::dot(light.normal, lightDir);
        float weight = 0.0f;
        if (ndl > 0.0f && lndl > 0.0f) {
            const bool occluded = traceOccluded(params.gasHandle, position, lightDir, 0.01f, Ldist - 0.01f);
            if (!occluded) {
                //printf("not Occluded!\n");
                const float A = rtlib::length(rtlib::cross(light.v1, light.v2));
                weight = ndl * lndl * A / (RTLIB_M_PI * Ldist * Ldist);
            }
        }
        prd->radiance += light.emission * weight;
        
    }
    prd->seed = xor32.m_seed;
}
extern "C" __global__ void __closesthit__radiance_for_specular() {
    auto* hgData = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());
    const float3 rayDirection = optixGetWorldRayDirection();
    const int    primitiveID  = optixGetPrimitiveIndex();
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
    prd->emitted     = make_float3(0.0f, 0.0f, 0.0f);
    {
        float3 specular   = hgData->getSpecularColor(texCoord);
        float3 reflectDir = rtlib::normalize(rayDirection - 2.0f * rtlib::dot(rayDirection, normal) * normal);
        prd->origin       = position;
        prd->direction    = reflectDir;
        prd->attenuation *= specular;
        prd->countEmitted = true;
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
    float3 normal   = {};
    float  refInd   = 0.0f;
    if (rtlib::dot(n0,rayDirection)<0.0f) {
        normal      = n0;
        refInd      = 1.0f / hgData->refrInd; 
    }
    else {
        normal      = make_float3(-n0.x,-n0.y,-n0.z);
        refInd      = hgData->refrInd;
    }
    const float2 barycentric = optixGetTriangleBarycentrics();
    const auto t0 = hgData->texCoords[hgData->indices[primitiveID].x];
    const auto t1 = hgData->texCoords[hgData->indices[primitiveID].y];
    const auto t2 = hgData->texCoords[hgData->indices[primitiveID].z];
    const auto texCoord = (1.0f - barycentric.x - barycentric.y) * t0 + barycentric.x * t1 + barycentric.y * t2;
    const float3 position = optixGetWorldRayOrigin() + optixGetRayTmax() * rayDirection;
    RadiancePRD* prd      = getRadiancePRD();
    prd->emitted          = make_float3(0.0f, 0.0f, 0.0f);
    rtlib::Xorshift32 xor32(prd->seed);
    float3 diffuse        = hgData->getDiffuseColor(texCoord);
    float3 specular       = hgData->getSpecularColor(texCoord);
    float3 transmit       = hgData->transmit;
    {
        prd->origin       = position;
        float3 reflectDir = rtlib::normalize(rayDirection - 2.0f * rtlib::dot(rayDirection, normal) * normal);
        float  cosine_i   = -rtlib::dot(normal, rayDirection);
        float  sine_o_2   = (1.0f - rtlib::pow2(cosine_i)) * rtlib::pow2(refInd);
        float  f0         = rtlib::pow2((1 - refInd) / (1 + refInd));
        float  fresnell   = f0 + (1.0f - f0) * rtlib::pow5(1.0f - cosine_i);
        if (rtlib::random_float1(0.0f, 1.0f, xor32) < fresnell || sine_o_2 > 1.0f) {

            //printf("reflect: %lf %lf %lf\n", reflectDir.x, reflectDir.y, reflectDir.z);
            prd->origin      += 0.001f * normal;
            prd->direction    = reflectDir;
            prd->attenuation *= specular;
        }
        else {
            float  cosine_o   = sqrtf(1.0f - sine_o_2);
            float3 refractDir = (rayDirection - (cosine_o - cosine_i) * normal) / refInd;
            //printf("refract: %lf %lf %lf\n", refractDir.x, refractDir.y, refractDir.z);
            prd->origin      -= 0.001f * normal;
            prd->direction    = refractDir;
            prd->attenuation *= transmit;
        }
        prd->countEmitted   = true;
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
    const auto texCoord   = (1.0f - barycentric.x - barycentric.y) * t0 + barycentric.x * t1 + barycentric.y * t2;
    const float3 position = optixGetWorldRayOrigin() + optixGetRayTmax() * rayDirection;
    RadiancePRD* prd      = getRadiancePRD();
    if (prd->countEmitted) {
        prd->emitted = hgData->getEmissionColor(texCoord) * prd->attenuation;
    }
    prd->countEmitted = false;
}
extern "C" __global__ void __closesthit__occluded() {
    setPayloadOccluded(true);
}
