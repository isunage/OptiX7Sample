#ifndef TEST_RT_SCENE_H
#define TEST_RT_SCENE_H
#include "RTModel.h"
namespace test
{
	class RTScene
	{
	private:
		std::unordered_map<std::string, std::shared_ptr<RTModel>> m_Models = {};
	};
}
#endif