#ifndef PAYLOAD_H
#define PAYLOAD_H
#include <RTLib/VectorFunction.h>
//Payload Setting//
//position//w_i//pointer//
struct RayTracePayload
{
    float3       w_o;
    float        distance;

    float3       radiance;
    unsigned int flags;

    float3       f_over_pdf;
    float        pdf;

    void*        p_user_data;
    float        ior;
    unsigned int seed;
};
#endif
