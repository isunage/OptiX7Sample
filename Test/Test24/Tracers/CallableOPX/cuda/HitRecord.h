#ifndef TEST_TEST24_TRACER_CALLABLE_HIT_RECORD_H
#define TEST_TEST24_TRACER_CALLABLE_HIT_RECORD_H
#include <RTLib/math/VectorFunction.h>
namespace test24_callable
{
	struct HitRecord
	{
		float3 s_normal;
		float3 v_normal;
		float2 texcoord;
	};
}
#endif