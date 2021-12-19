#ifndef TEST_RT_TRACER_H
#define TEST_RT_TRACER_H
#include <RTLib/ext/VariableMap.h>
#include <string>
namespace test
{
	class RTTracer
	{
	public:
		RTTracer()noexcept :m_Variables{ new rtlib::ext::VariableMap() } {}
		//   INIT
		virtual void Initialize() = 0;
		//CLEANUP
		virtual void CleanUp()    = 0;
		// LAUNCH
		virtual void Launch(int fbWidth, int fbHeight, void* pUserData) = 0;
		// UPDATE
		virtual void Update() = 0;
		// VARIABLE
		auto GetVariables()const noexcept -> std::shared_ptr<rtlib::ext::VariableMap> {
			return m_Variables;
		}
		virtual ~RTTracer()noexcept {}
	private:
		std::shared_ptr<rtlib::ext::VariableMap> m_Variables = {};
	};
}
#endif
