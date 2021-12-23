#ifndef CALLABLE_H
#define CALLABLE_H
#include <RTLib/core/Preprocessors.h>
#include <RTLib/math/Math.h>
#include <RTLib/math/Random.h>
#include <RTLib/math/VectorFunction.h>
#include <MaterialParameters.h>
enum CallbableType
{
	CALLABLE_TYPE_BSDF = 0,
	CALLABLE_TYPE_PDF   
};
#endif