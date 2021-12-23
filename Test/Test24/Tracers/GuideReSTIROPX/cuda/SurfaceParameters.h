#ifndef SURFACE_PARAMETERS_H
#define SURFACE_PARAMETERS_H
#include <RTLib/math/Math.h>
#include <RTLib/math/Random.h>
#include <RTLib/math/VectorFunction.h>
namespace test24_restir_guide
{
	struct SurfaceParameters
	{
		float3* vertices;
		float3* normals;
		float2* texCoords;
		uint3 * indices;
	};
	struct SurfaceRec
	{
		float3 position;
		float3 sNormal;
		float3 vNormal;
		float2 texCoord;
		float  distance;
	};
}
#endif