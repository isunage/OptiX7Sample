#ifndef RESERVOIR_H
#define RESERVOIR_H
#include <RTLib/math/Math.h>
#include <RTLib/math/Random.h>
#include <RTLib/math/VectorFunction.h>
namespace test24_restir_guide
{
    template<typename T>
    struct Reservoir
    {
        //Update
        RTLIB_INLINE RTLIB_HOST_DEVICE bool Update(T x_i, float w_i, float rnd01)
        {
            w_sum += w_i;
            ++m;
            if ((w_i / w_sum) >= rnd01)
            {
                y = x_i;
                return true;
            }
            return false;
        }
        float        w_sum = 0.0f;
        unsigned int m     = 0;
        T            y     = {};
    };
}
#endif
