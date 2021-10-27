#ifndef RAY_TYPE_H
#define RAY_TYPE_H
#include <RTLib/VectorFunction.h>
struct RayState
{
    float3  position;
    float2  texCoord;
    float3  normal_face;
    float3  normal_base;
};
enum RayType
{
    RAY_TYPE_RADIANCE = 0,
    RAY_TYPE_OCCLUDED,
    RAY_TYPE_COUNT
};
#endif