#define __CUDACC__
#include "RayTrace.h"
#include "PathGuiding.h"
struct RadiancePRD {
    DTreeWrapper* dTree;
    float3        emission;
    float3        bsdfVal;
    float         woWei, bsdfWei, dTreeWei;
    float         woPdf, bsdfPdf, dTreePdf;
    float         cosine;
    float         distance;
    unsigned int  seed;
    bool          isDelta;
    bool          done;
};
extern "C" {
    __constant__ RayTraceParams params;
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
static __forceinline__ __device__ float3 unpackFloat3(unsigned int p0, unsigned p1,unsigned int p2)
{
    return make_float3(__uint_as_float(p0),__uint_as_float(p1),__uint_as_float(p2));
}
static __forceinline__ __device__ void         packFloat3(const float3& v,unsigned int& p0, unsigned& p1,unsigned int& p2)
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
static __forceinline__ __device__ void         setRayOrigin(const float3& origin){
    unsigned int p2,p3,p4;
    packFloat3(origin,p2,p3,p4);
    optixSetPayload_2(p2);
    optixSetPayload_3(p3);
    optixSetPayload_4(p4);
}
static __forceinline__ __device__ void         setRayDirection(const float3& direction){
    unsigned int p5,p6,p7;
    packFloat3(direction,p5,p6,p7);
    optixSetPayload_5(p5);
    optixSetPayload_6(p6);
    optixSetPayload_7(p7);
}
static __forceinline__ __device__ void         setPayloadOccluded(bool occluded) {
    optixSetPayload_0(static_cast<unsigned int>(occluded));
}
static __forceinline__ __device__ void         traceRadiance(
    OptixTraversableHandle handle,
    float3&     rayOrigin, 
    float3&     rayDirection,
    float tmin, float tmax,
    RadiancePRD*  prd) {
    unsigned int p0, p1, p2, p3, p4, p5, p6, p7;
    packPointer(prd, p0, p1);
    optixTrace(handle, rayOrigin, rayDirection, tmin, tmax, 0.0f, OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE, RAY_TYPE_RADIANCE, RAY_TYPE_COUNT, RAY_TYPE_RADIANCE, p0, p1, p2, p3, p4, p5, p6, p7);
    rayOrigin    = unpackFloat3(p2,p3,p4);
    rayDirection = unpackFloat3(p5,p6,p7);
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
        prd.woWei = 1.0f;
        prd.bsdfWei = 0.0f;
        prd.dTreeWei = 0.0f;
        prd.woPdf = 0.0f;
        prd.bsdfPdf = 0.0f;
        prd.dTreePdf = 0.0f;
        prd.distance = 0.0f;
        prd.cosine = 0.0f;
        prd.done = false;
        prd.isDelta = false;
        prd.seed = seed;
        int depth = 0;
        float3 prvThroughPut = make_float3(1.0f);
        for (;;) {
            //���ˈʒu�A���˕���
            //�g���[�X�֐�
            traceRadiance(params.gasHandle, rayOrigin, rayDirection, 0.01f, 1e16f, &prd);
            //vertices�̍X�V
            //Radiance�̍X�V
            //Result�̍X�V
            result += prvThroughPut * prd.emission;
            //ThroughPut�̍X�V
            prvThroughPut *= (prd.cosine * prd.bsdfVal) / (prd.woWei * prd.woPdf + prd.bsdfWei * prd.bsdfPdf + prd.dTreeWei * prd.dTreePdf);
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
extern "C" __global__ void __raygen__pg(){

    const uint3 idx             = optixGetLaunchIndex();
	const uint3 dim             = optixGetLaunchDimensions();
    auto* rgData                = reinterpret_cast<RayGenData*>(optixGetSbtDataPointer());
    const float3 u              = rgData->u;
	const float3 v              = rgData->v;
	const float3 w              = rgData->w;
    unsigned int seed           = params.seed[params.width * idx.y + idx.x];
    float3 result               = make_float3(0.0f, 0.0f, 0.0f);
    size_t i                    = params.samplePerLaunch;
    TraceVertex vertices[32]    = {};
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
        prd.emission   = make_float3(0.0f);
        prd.bsdfVal    = make_float3(1.0f);
        prd.woWei      = 1.0f;
        prd.bsdfWei    = 0.0f;
        prd.dTreeWei   = 0.0f;
        prd.woPdf      = 0.0f;
        prd.bsdfPdf    = 0.0f;
        prd.dTreePdf   = 0.0f;
        prd.distance   = 0.0f;
        prd.cosine     = 0.0f;
        prd.done       = false;
        prd.isDelta    = false;
        prd.seed       = seed;
        int depth      = 0;
        float3 prvThroughPut = make_float3(1.0f);
        for (;;) {
            //���ˈʒu�A���˕���
            vertices[depth].rayOrigin    = rayOrigin;
            vertices[depth].rayDirection = rayDirection;
            //�g���[�X�֐�
            traceRadiance(params.gasHandle, rayOrigin, rayDirection, 0.01f, 1e16f, &prd);
            //vertices�̍X�V
            vertices[depth].dTree        = prd.dTree;
            vertices[depth].throughPut   = prvThroughPut;
            vertices[depth].bsdfVal      = prd.bsdfVal;
            vertices[depth].radiance     = make_float3(0.0f);
            vertices[depth].woPdf        = prd.woPdf;
            vertices[depth].bsdfPdf      = prd.bsdfPdf;
            vertices[depth].dTreePdf     = prd.dTreePdf;
            vertices[depth].isDelta      = prd.isDelta;
            //Radiance�̍X�V
            for (int j = 0; j <= depth; ++j) {
                vertices[j].Record(prvThroughPut * prd.emission);
            }
            //Result�̍X�V
            result         += prvThroughPut * prd.emission;
            // if (idx.x==100&&idx.y==100){
            //     printf("(%d,%d,%d)= %f %f %f\n",idx.x,idx.y,depth, prvThroughPut.x,prvThroughPut.y,prvThroughPut.z);
            // }
            //ThroughPut�̍X�V
            prvThroughPut  *= (prd.cosine * prd.bsdfVal) / (prd.woWei * prd.woPdf + prd.bsdfWei * prd.bsdfPdf + prd.dTreeWei * prd.dTreePdf);
            if (prd.done || depth >= params.maxTraceDepth) {
                if (idx.x==100&&idx.y==100){
                     printf("(%d,%d,%d)= %f %f %f\n",idx.x,idx.y,depth, 
                     vertices[depth].radiance.x/vertices[depth].throughPut.x, 
                     vertices[depth].radiance.y/vertices[depth].throughPut.y, 
                     vertices[depth].radiance.z/vertices[depth].throughPut.z);
                }
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
extern "C" __global__ void __miss__radiance(){
    auto* msData     = reinterpret_cast<MissData*>(optixGetSbtDataPointer());
    RadiancePRD* prd = getRadiancePRD();
    prd->emission    = make_float3(msData->bgColor.x, msData->bgColor.y, msData->bgColor.z);
    prd->done        = true;
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
    const float3 position = optixGetWorldRayOrigin() + optixGetRayTmax() * rayDirection;
    float3 newDirection = make_float3(0.0f);
    RadiancePRD* prd = getRadiancePRD();

    prd->dTree       = nullptr;
    prd->emission    = make_float3(0.0f, 0.0f, 0.0f);
    prd->distance    = optixGetRayTmax();

    rtlib::Xorshift32 xor32(prd->seed);
    {
        prd->woWei    = 1.0f;
        prd->dTreeWei = 0.0f;
        prd->bsdfWei  = 0.0f;

        rtlib::ONB onb(normal);
        newDirection = onb.local(rtlib::random_cosine_direction(xor32));

        prd->woPdf    = rtlib::max(rtlib::dot(newDirection, normal), 0.0f) / RTLIB_M_PI;
        prd->dTreePdf = 0.0f;
        prd->bsdfPdf  = 0.0f;

        setRayOrigin(position);
        setRayDirection(newDirection);

        float3 diffuse = hgData->getDiffuseColor(texCoord);
        prd->bsdfVal   = diffuse / RTLIB_M_PI;
        prd->cosine    = fabsf(rtlib::dot(newDirection, normal));
        prd->seed      = xor32.m_seed;
        prd->isDelta   = false;
    }
    prd->seed = xor32.m_seed;
}
extern "C" __global__ void __closesthit__radiance_for_diffuse_pg() {

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
    float3 newDirection = make_float3(0.0f);
    RadiancePRD* prd = getRadiancePRD();

    prd->dTree = params.sdTree.GetDTreeWrapper(position);
    prd->emission = make_float3(0.0f, 0.0f, 0.0f);
    prd->distance = optixGetRayTmax();

    rtlib::Xorshift32 xor32(prd->seed);
    {
        prd->woWei    = 0.5f;
        prd->dTreeWei = 0.5f;
        prd->bsdfWei  = 0.0f;

        if (rtlib::random_float1(xor32) < prd->woWei) {
            rtlib::ONB onb(normal);
            newDirection = onb.local(rtlib::random_cosine_direction(xor32));
        }
        else {
            newDirection = prd->dTree->Sample(xor32);
        }
        prd->woPdf = rtlib::max(rtlib::dot(newDirection, normal), 0.0f) / RTLIB_M_PI;
        prd->dTreePdf = prd->dTree->Pdf(newDirection);
        prd->bsdfPdf = 0.0f;

        setRayOrigin(position);
        setRayDirection(newDirection);

        float3 diffuse = hgData->getDiffuseColor(texCoord);
        prd->bsdfVal = diffuse / RTLIB_M_PI;
        prd->cosine =  fabsf(rtlib::dot(newDirection, normal));
        prd->seed = xor32.m_seed;
        prd->isDelta = false;
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
    prd->dTree       = nullptr;
    prd->emission    = make_float3(0.0f, 0.0f, 0.0f);
    prd->distance    = optixGetRayTmax();
    {
        prd->woWei    = 0.0f;
        prd->dTreeWei = 0.0f;
        prd->bsdfWei  = 1.0f;

        float3 specular   = hgData->getSpecularColor(texCoord);
        float3 reflectDir = rtlib::normalize(rayDirection - 2.0f * rtlib::dot(rayDirection, normal) * normal);

        prd->woPdf    = 0.0f;
        prd->dTreePdf = 0.0f;
        prd->bsdfPdf  = std::fabsf(rtlib::dot(reflectDir, normal));

        setRayOrigin(position);
        setRayDirection(reflectDir);


        prd->bsdfVal  = specular;
        prd->cosine   = rtlib::dot(reflectDir, normal);
        prd->isDelta  = true;
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
    prd->dTree       = nullptr;
    prd->emission    = make_float3(0.0f, 0.0f, 0.0f);
    prd->distance    = optixGetRayTmax();
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

        prd->woWei    = 0.0f;
        prd->dTreeWei = 0.0f;
        prd->bsdfWei  = 1.0f;

        if (rtlib::random_float1(0.0f, 1.0f, xor32) < fresnell || sine_o_2 > 1.0f) {
            prd->woPdf    = 0.0f;
            prd->dTreePdf = 0.0f;
            prd->bsdfPdf  = std::fabsf(rtlib::dot(reflectDir, normal));
            //printf("reflect: %lf %lf %lf\n", reflectDir.x, reflectDir.y, reflectDir.z);
            setRayOrigin(position + 0.001f * normal);
            setRayDirection(reflectDir);

            prd->bsdfVal  = specular;
            prd->cosine   = rtlib::dot(reflectDir, normal);
        }
        else {
            float  cosine_o = sqrtf(1.0f - sine_o_2);
            float3 refractDir = (rayDirection - (cosine_o - cosine_i) * normal) / refInd;
            prd->woPdf    = 0.0f;
            prd->dTreePdf = 0.0f;
            prd->bsdfPdf  = std::fabsf(rtlib::dot(refractDir, normal));
            //printf("refract: %lf %lf %lf\n", refractDir.x, refractDir.y, refractDir.z);
            setRayOrigin(position - 0.001f * normal);
            setRayDirection(refractDir);

            prd->bsdfVal = transmit;
            prd->cosine  = fabsf(rtlib::dot(refractDir, normal));
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
    const float3 position = optixGetWorldRayOrigin() + optixGetRayTmax() * rayDirection;
    RadiancePRD* prd = getRadiancePRD();
    prd->emission    = hgData->getEmissionColor(texCoord);
    prd->done        = true;
}
extern "C" __global__ void __closesthit__occluded() {
    setPayloadOccluded(true);
}