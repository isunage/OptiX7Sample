#ifndef RT_GUI_H
#define RT_GUI_H
#include <RTLib/VectorFunction.h>
#include <RTLib/ext/VariableMap.h>
#include <memory>
#include <string>
namespace test
{
	class RTGui: protected rtlib::ext::VariableMap
	{
	public:
		//Set
		using rtlib::ext::VariableMap::SetUInt32;
		using rtlib::ext::VariableMap::SetBool;
		using rtlib::ext::VariableMap::SetFloat1;
		using rtlib::ext::VariableMap::SetFloat2;
		using rtlib::ext::VariableMap::SetFloat3;
		using rtlib::ext::VariableMap::SetFloat4;
		using rtlib::ext::VariableMap::SetString;
		//Get
		using rtlib::ext::VariableMap::GetUInt32;
		using rtlib::ext::VariableMap::GetBool;
		using rtlib::ext::VariableMap::GetFloat1;
		using rtlib::ext::VariableMap::GetFloat2;
		using rtlib::ext::VariableMap::GetFloat3;
		using rtlib::ext::VariableMap::GetFloat4;
		using rtlib::ext::VariableMap::GetString;
		//Has
		using rtlib::ext::VariableMap::HasUInt32;
		using rtlib::ext::VariableMap::HasBool;
		using rtlib::ext::VariableMap::HasFloat1;
		using rtlib::ext::VariableMap::HasFloat2;
		using rtlib::ext::VariableMap::HasFloat3;
		using rtlib::ext::VariableMap::HasFloat4;
		using rtlib::ext::VariableMap::HasString;
		//Get
		using rtlib::ext::VariableMap::GetFloat1As;
		using rtlib::ext::VariableMap::GetFloat2As;
		using rtlib::ext::VariableMap::GetFloat3As;
		using rtlib::ext::VariableMap::GetFloat4As;
		//Init
		virtual void Initialize() = 0;
		//Attach
		virtual void Attach()     = 0;
		//Terminate
		virtual void Terminate()  = 0;
		virtual ~RTGui(){}
	private:
		std::unordered_map<std::string, void*> m_UserPtrMap = {};
	};
	using RTGuiPtr = std::shared_ptr<test::RTGui>;

}
#endif