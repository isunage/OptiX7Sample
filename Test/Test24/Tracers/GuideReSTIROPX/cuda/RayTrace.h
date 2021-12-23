#ifndef RAY_TRACE_H
#define RAY_TRACE_H
#include <cuda_runtime.h>
#include <optix.h>
#include <RTLib/math/Math.h>
#include <RTLib/math/Random.h>
#include <RTLib/math/VectorFunction.h>
#include <RTLib/math/Math.h>
#include <RayTraceConfig.h>
#include <Reservoir.h>
#include <SurfaceParameters.h>
#include <MaterialParameters.h>
#include <LightUtils.h>
#include <PathGuiding.h>
//#define RAY_GUIDING_SAMPLE_BY_UNIFORM_SPHERE
//#define RAY_GUIDING_SAMPLE_BY_COSINE_SPHERE
//#define TEST_SKIP_TEXTURE_SAMPLE
//#define TEST11_SHOW_EMISSON_COLOR
namespace test24_restir_guide
{
    enum   RayType
    {
        RAY_TYPE_RADIANCE  = 0,
        RAY_TYPE_OCCLUSION = 1,
        RAY_TYPE_COUNT     = 2,
    };    
    struct ReservoirState
    {
        LightRec     light;
        MaterialRec  material;
        float3       density;
        float        weight;
    };
    struct RayTraceParams
    {
        uchar4*                frameBuffer;
        float3*                accumBuffer;
        unsigned int*          seedBuffer;
        unsigned int           width;
        unsigned int           height;
        unsigned int           samplePerLaunch;
        unsigned int           samplePerALL;
        unsigned int           maxTraceDepth;
        unsigned int           numCandidates;
        OptixTraversableHandle gasHandle;
        STree                  sdTree;
        MeshLightList          light;
        bool                   isBuilt;
        bool                   isFinal;
    };
    struct RayGenData
    {
        float3 u, v, w;
        float3 eye;
    };
    struct MissData
    {
        float4 bgColor;
    };
    struct HitgroupData
    {
        SurfaceParameters    surfParams;
        MaterialParameters    matParams;
    };
}
#endif