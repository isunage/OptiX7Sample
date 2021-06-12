#define __CUDACC__
#include "RayTrace.h"
extern "C" {
    __constant__ Params params;
}
static __forceinline__ void trace(OptixTraversableHandle handle,const float3& rayOrigin, const float3& rayDirection,float tmin, float tmax,float3& color) {
    unsigned int p0, p1,p2;
    optixTrace(handle, rayOrigin, rayDirection, tmin, tmax, 0.0f, OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE, 0, 1, 0, p0, p1, p2);
    color.x = int_as_float(p0);
    color.y = int_as_float(p1);
    color.z = int_as_float(p2);
}
extern "C" __global__ void     __raygen__rg(){
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
    params.image[params.width * idx.y + idx.x] = make_uchar4(static_cast<unsigned char>(255.99 * color.x), static_cast<unsigned char>(255.99 * color.y), static_cast<unsigned char>(255.99 * color.z), 255);
}
extern "C" __global__ void       __miss__ms(){
    auto* msData = reinterpret_cast<MissData*>(optixGetSbtDataPointer());
    optixSetPayload_0(float_as_int(msData->bgColor.x));
    optixSetPayload_1(float_as_int(msData->bgColor.y));
    optixSetPayload_2(float_as_int(msData->bgColor.z));
}
extern "C" __global__ void __closesthit__ch(){
    auto* hgData     = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());
    float2 texCoord  = optixGetTriangleBarycentrics();
    auto primitiveId = optixGetPrimitiveIndex();
    //printf("%d\n", primitiveId);
    auto p0          = hgData->vertices[hgData->indices[primitiveId].x];
    auto p1          = hgData->vertices[hgData->indices[primitiveId].y];
    auto p2          = hgData->vertices[hgData->indices[primitiveId].z];
    auto normal      = rtlib::normalize(rtlib::cross(p1 - p0, p2 - p0));
    auto diffTex = hgData->diffuseTex;
    auto t0      = hgData->texCoords[hgData->indices[primitiveId].x];
    auto t1      = hgData->texCoords[hgData->indices[primitiveId].y];
    auto t2      = hgData->texCoords[hgData->indices[primitiveId].z];
    auto t       = (1.0f-texCoord.x-texCoord.y)*t0 + texCoord.x * t1 + texCoord.y * t2;
    auto diffC   = tex2D<uchar4>(diffTex, t.x, t.y);
    //printf("%f %f\n",t0.x,t0.y);
    //optixSetPayload_0(float_as_int(float(diffC.x) / 255.99f));
    //optixSetPayload_1(float_as_int(float(diffC.y) / 255.99f));
    //optixSetPayload_2(float_as_int(float(diffC.z) / 255.99f));
    //optixSetPayload_0(float_as_int((t.x)));
    //optixSetPayload_1(float_as_int((t.y)));
    //optixSetPayload_2(float_as_int((2.0f-t.x-t.y)/2.0f));
    optixSetPayload_0(float_as_int((0.5f+0.5f*normal.x)));
    optixSetPayload_1(float_as_int((0.5f+0.5f*normal.y)));
    optixSetPayload_2(float_as_int((0.5f+0.5f*normal.z)));
}
extern "C" __global__ void     __anyhit__ah(){
    auto* hgData = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());
}
