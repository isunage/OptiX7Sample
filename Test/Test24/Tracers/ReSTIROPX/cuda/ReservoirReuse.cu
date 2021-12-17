#define __CUDACC__
#include <cuda_runtime.h>
#include <RayTrace.h>
using namespace test24_restir;
extern "C" __global__ void combineSpatialReservoirs(
    Reservoir<LightRec> * inResvBuffer,
    Reservoir<LightRec> * outResvBuffer,
    ReservoirState      * tmpStatBuffer,
    RaySecondParams     * params, 
    int                   width, 
    int                   height, 
    int                   sample,
    int                   range) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < width && j < height) {
        auto origin    = params->posiBuffer[width * j + i];
        auto normal    = params->normBuffer[width * j + i];
        auto diffuse   = params->diffBuffer[width * j + i];
        auto seed      = params->seedBuffer[width * j + i];
        auto distance  = params->distBuffer[width * j + i];
        auto xor32     = rtlib::Xorshift32(seed);
        Reservoir<LightRec> r;
        float p_q = tmpStatBuffer[width * j + i].targetDensity;
        //First: Combine CurResv
        Reservoir<LightRec> curResv = inResvBuffer[width * j + i];
        {
            if (curResv.w_sum <= 0.0f) {
                p_q           = 0.0f;
                curResv.w_sum = 0.0f;
            }
            r.Update(curResv.y, p_q * curResv.w * static_cast<float>(curResv.m), rtlib::random_float1(xor32));
            r.m = curResv.m;
        }
        //Second: Combine NearResv
        for (int k = 0; k < sample; ++k)
        {
            /**/
            int s = i + cosf(rtlib::random_float1(xor32) * RTLIB_M_2PI) * static_cast<float>(range);
            int t = j + sinf(rtlib::random_float1(xor32) * RTLIB_M_2PI) * static_cast<float>(range);
            if (s<0 || s > width - 1 || t<0 || t> height-1||((s==i)&&(t==j))) {
                continue;
            }
            /**/
            float3 near_normal   = params->normBuffer[t * width + s];
            float  near_distance = params->distBuffer[t * width + s];
            if (rtlib::dot(near_normal, normal) < 0.90f ||
                fabsf((near_distance - distance)/distance) > 0.10f) {
                continue;
            }

            auto r_i   = inResvBuffer[width * t + s];
            
            float3 ldir  = r_i.y.position - origin;
            //Distance
            float  ldist = rtlib::length(ldir);
                   ldir /= static_cast<float>(ldist);
            //Bsdf
            float3 bsdf  = diffuse * RTLIB_M_INV_PI;
            //Emission
            float3 l_e   = r_i.y.emission;
            //Geometry 
            float  g     = fabsf(rtlib::dot(normal, ldir)) * fabsf(rtlib::dot(r_i.y.normal, ldir)) / (ldist * ldist);
            //Indirect Illumination
            float3 lp    = bsdf * l_e * g;
            float  lp_q  = (lp.x + lp.y + lp.z) / 3.0f;
            if (r.Update(r_i.y, lp_q * r_i.w * static_cast<float>(r_i.m), rtlib::random_float1(xor32))) {
                p_q = lp_q;
            }
            r.m += r_i.m;
        }
        r.w  = (p_q <= 0.0f) ? 0.0f : (r.w_sum / (static_cast<float>(r.m) * p_q));
        outResvBuffer[width * j + i]               = r;
        tmpStatBuffer[width * j + i].targetDensity = p_q;
        params->seedBuffer[width * j + i]          = xor32.m_seed;
    }
}

extern "C" __global__ void combineTemporalReservoirs(
    Reservoir<LightRec> * prvResvBuffer,
    Reservoir<LightRec> * curResvBuffer,
    ReservoirState      * tmpStatBuffer,
    RaySecondParams     * params, 
    int                   width, 
    int                   height) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= width || j >= height) {
        return;
    }
    auto dIdx = params->motiBuffer[width * j + i];
    int  s = i + dIdx.x;
    int  t = j + dIdx.y;
    if (s<0 || s >= width || t< 0 || t >= height) {
        return;
    }
    auto curNormal   = params->normBuffer[width * j + i];
    auto curDistance = params->distBuffer[width * j + i];
    auto prvNormal   = params->normBuffer[width * t + s];
    auto prvDistance = params->distBuffer[width * t + s];
    if (rtlib::dot(curNormal, prvNormal) < 0.90f ||
        fabsf((prvDistance - curDistance) / curDistance) > 0.10f) {
        return;
    }

    auto origin     = params->posiBuffer[width * j + i];
    auto curDiffuse = params->diffBuffer[width * j + i];
    auto seed       = params->seedBuffer[width * j + i];
    auto xor32      = rtlib::Xorshift32(seed);
    Reservoir<LightRec> r;
    float p_q = tmpStatBuffer[width * j + i].targetDensity;
    //First: Combine CurResv
    {
        Reservoir<LightRec> curResv = curResvBuffer[width * j + i];
        if (curResv.w_sum <= 0.0f) {
            p_q = 0.0f;
            curResv.w_sum = 0.0f;
        }
        r.Update(curResv.y, p_q * curResv.w * static_cast<float>(curResv.m), rtlib::random_float1(xor32));
        r.m = curResv.m;
    }
    {
        auto prvResv   = prvResvBuffer[width * t + s];
        prvResv.m      = rtlib::min(prvResv.m, 20 * r.m);
        //selective probability on current pixel
        float3   ldir  = prvResv.y.position - origin;
        float    ldist = rtlib::length(ldir);
        ldir /= static_cast<float>(ldist);
        float3   bsdf = curDiffuse * RTLIB_M_INV_PI;
        float3   le   = prvResv.y.emission;
        float    g    = fabs(rtlib::dot(ldir, curNormal)) * fabs(rtlib::dot(ldir, prvResv.y.normal)) / (ldist * ldist);
        float3   lp   = bsdf * le * g;
        float    lp_q = (lp.x + lp.y + lp.z) / 3.0f;
        if (r.Update(prvResv.y, lp_q * prvResv.w * static_cast<float>(prvResv.m), rtlib::random_float1(xor32))) {
            p_q = lp_q;
        }
        r.m += prvResv.m;
    }
    r.w = (p_q <= 0.0f) ? 0.0f : (r.w_sum / (static_cast<float>(r.m) * p_q));
    curResvBuffer[width * j + i] = r;
    tmpStatBuffer[width * j + i].targetDensity = p_q;
    params->seedBuffer[width * j + i] = xor32.m_seed;
}