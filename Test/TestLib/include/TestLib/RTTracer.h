#ifndef TEST_RT_TRACER_H
#define TEST_RT_TRACER_H
#include <string>
namespace test
{
	class RTTracer
	{
	public:
		//   INIT
		virtual void Initialize() = 0;
		//CLEANUP
		virtual void CleanUp()    = 0;
		// LAUNCH
		virtual void Launch(int fbWidth, int fbHeight, void* pUserData) = 0;
		virtual void Update() = 0;
		virtual ~RTTracer()noexcept {}
	};
}
#endif
