#define __CUDACC__
#include "RayTrace.h"
extern "C" {
    __constant__ Params params;
}
static __forceinline__ __device__ void trace(OptixTraversableHandle handle,const float3& rayOrigin, const float3& rayDirection,float tmin, float tmax,float3& color) {
    unsigned int p0, p1,p2;
    optixTrace(handle, rayOrigin, rayDirection, tmin, tmax, 0.0f, OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE, RAY_TYPE_RADIANCE, RAY_TYPE_COUNT, RAY_TYPE_RADIANCE, p0, p1, p2);
    color.x = int_as_float(p0);
    color.y = int_as_float(p1);
    color.z = int_as_float(p2);
}
extern "C" __global__ void     __raygen__debug(){
    const uint3 idx = optixGetLaunchIndex();
	const uint3 dim = optixGetLaunchDimensions();
    auto* rgData    = reinterpret_cast<RayGenData*>(optixGetSbtDataPointer());
    const float3 u  = rgData->u;
	const float3 v  = rgData->v;
	const float3 w  = rgData->w;
	const float2 d  = make_float2(
		(2.0f * static_cast<float>(idx.x)/static_cast<float>(dim.x)) - 1.0,
		(2.0f * static_cast<float>(idx.y)/static_cast<float>(dim.y)) - 1.0);
	const float3 origin    = rgData->eye;
	const float3 direction = rtlib::normalize(d.x * u + d.y * v + w);
    //printf("%f, %lf, %lf\n", direction.x, direction.y, direction.z);
    float3 color;
    trace(params.gasHandle, origin,direction, 0.0f, 1e16f,color);
   // printf("%f, %lf\n", texCoord.x, texCoord.y);
    params.frameBuffer[params.width * idx.y + idx.x] = make_uchar4(static_cast<unsigned char>(255.99 * color.x), static_cast<unsigned char>(255.99 * color.y), static_cast<unsigned char>(255.99 * color.z), 255);
}
extern "C" __global__ void       __miss__debug(){
    auto* msData = reinterpret_cast<MissData*>(optixGetSbtDataPointer());
    optixSetPayload_0(float_as_int(msData->bgColor.x));
    optixSetPayload_1(float_as_int(msData->bgColor.y));
    optixSetPayload_2(float_as_int(msData->bgColor.z));
}
extern "C" __global__ void __closesthit__debug(){
    auto* hgData     = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());
    float2 texCoord  = optixGetTriangleBarycentrics();
    auto primitiveId = optixGetPrimitiveIndex();
    //printf("%d\n", primitiveId);
    auto p0          = hgData->vertices[hgData->indices[primitiveId].x];
    auto p1          = hgData->vertices[hgData->indices[primitiveId].y];
    auto p2          = hgData->vertices[hgData->indices[primitiveId].z];
    auto normal      = rtlib::normalize(rtlib::cross(p1 - p0, p2 - p0));
    auto t0          = hgData->texCoords[hgData->indices[primitiveId].x];
    auto t1          = hgData->texCoords[hgData->indices[primitiveId].y];
    auto t2          = hgData->texCoords[hgData->indices[primitiveId].z];
    auto t           = (1.0f-texCoord.x-texCoord.y)*t0 + texCoord.x * t1 + texCoord.y * t2;
    auto diffC       = hgData->getDiffuseColor(t);
    auto specC       = hgData->getSpecularColor(t);
    auto emitC       = hgData->getEmissionColor(t);
    //printf("%f %f\n",t0.x,t0.y);
#ifdef TEST_SHOW_DIFFUSE_COLOR
    optixSetPayload_0(float_as_int(diffC.x));
    optixSetPayload_1(float_as_int(diffC.y));
    optixSetPayload_2(float_as_int(diffC.z));
#endif 
#ifdef TEST_SHOW_SPECULAR_COLOR
    optixSetPayload_0(float_as_int(specC.x));
    optixSetPayload_1(float_as_int(specC.y));
    optixSetPayload_2(float_as_int(specC.z));
#endif 
#ifdef TEST_SHOW_EMISSION_COLOR
    optixSetPayload_0(float_as_int(emitC.x));
    optixSetPayload_1(float_as_int(emitC.y));
    optixSetPayload_2(float_as_int(emitC.z));
#endif 
#ifdef TEST_SHOW_TEXCOORD
    optixSetPayload_0(float_as_int((t.x)));
    optixSetPayload_1(float_as_int((t.y)));
    optixSetPayload_2(float_as_int((2.0f-t.x-t.y)/2.0f));
#endif
#ifdef TEST_SHOW_NORMAL
    optixSetPayload_0(float_as_int((0.5f+0.5f*normal.x)));
    optixSetPayload_1(float_as_int((0.5f+0.5f*normal.y)));
    optixSetPayload_2(float_as_int((0.5f+0.5f*normal.z)));
#endif
}
extern "C" __global__ void     __anyhit__ah(){
    auto* hgData = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());
}
