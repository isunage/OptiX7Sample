#define __CUDACC__
#include <cuda_runtime.h>
#include <RayTrace.h>
using namespace test24_restir;
extern "C" __global__ void combineSpatialReservoirs(
    Reservoir* inResvBuffer, 
    Reservoir* outResvBuffer, 
    RaySecondParams* params, int width, int height, int range) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < width && j < height) {
        auto origin    = params->posiBuffer[width * j + i];
        auto normal    = params->diffBuffer[width * j + i];
        auto diffuse   = params->diffBuffer[width * j + i];
        auto seed      = params->seedBuffer[width * j + i];
        auto xor32     = rtlib::Xorshift32(seed);
        unsigned int m = 0;
        Reservoir r;
        int s_min  = rtlib::min(i - range, 0);
        int s_max  = rtlib::max(i + range, width-1);
        int t_min  = rtlib::min(j - range, 0);
        int t_max  = rtlib::max(j + range, height-1);
        for (int t = t_min; t <= t_max; ++t) {
            for (int s = s_min; s <= s_max; ++s) {
                auto     r_i       = inResvBuffer[width * t + s];
                auto&    meshLight = params->meshLights.data[r_i.y];
                LightRec lrec;
                float3   ldir      = meshLight.Sample(origin, lrec, xor32);
                float3   bsdf      = diffuse * RTLIB_M_INV_PI;
                float3   le        = lrec.emission;
                float    g_over_p  = fabs(rtlib::dot(ldir, normal)) * lrec.invPdf;
                float3   lp        = bsdf * le * g_over_p;
                float    lp_a      = (lp.x + lp.y + lp.z) / 3.0f;
                r.Update(r_i.y, lp_a * r_i.w * static_cast<float>(r_i.m), rtlib::random_float1(xor32));
                m += r_i.m;
            }
        }
        r.m = m;
        {
            auto&    meshLight= params->meshLights.data[r.y];
            LightRec lrec;
            float3   ldir     = meshLight.Sample(origin, lrec, xor32);
            float3   bsdf     = diffuse * RTLIB_M_INV_PI;
            float3   le       = lrec.emission;
            float    g_over_p = fabs(rtlib::dot(ldir, normal)) * lrec.invPdf;
            float3   lp       = bsdf * le * g_over_p;
            float    lp_a     = (lp.x + lp.y + lp.z) / 3.0f;
            r.w = (lp_a <= 0.0f) ? 0.0f : (r.w_sum / (static_cast<float>(r.m) * lp_a));
        }
        outResvBuffer[width * j+ i]       = r;
        params->seedBuffer[width * j + i] = seed;
    }
}