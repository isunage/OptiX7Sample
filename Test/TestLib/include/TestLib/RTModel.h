#ifndef TEST_RT_MODEL_H
#define TEST_RT_MODEL_H
#include <RTLib/ext/Mesh.h>
#include <RTLib/ext/VariableMap.h>
#include <string>
#include <memory>
namespace test
{
	class RTModel
	{
	public:
		virtual ~RTModel() {}
	private:
		std::unordered_map<std::string, rtlib::ext::MeshGroupPtr> m_MeshGroups  = {};
		rtlib::ext::VariableMapList                               m_Materials   = {};
	};
}
#endif