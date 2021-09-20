#ifndef TEST_RT_TRACER_H
#define TEST_RT_TRACER_H
#include "RTPipeline.h"
#include <cuda/RayTrace.h>
#include <RTLib/Optix.h>
#include <RTLib/CUDA.h>
#include <RTLib/ext/TraversalHandle.h>
#include <RTLib/ext/Material.h>
namespace test {
	class RTTracer {
	public:
		virtual void Initialize() = 0;
		virtual void Launch()     = 0;
		virtual void CleanUp()    = 0;
		virtual void Update()     = 0;
	};
}
#endif