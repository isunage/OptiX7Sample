#ifndef TEST_RT_SCENE_H
#define TEST_RT_SCENE_H
#include "RTModel.h"
#include <unordered_map>
namespace test
{
	class RTScene
	{
	public:
		using ModelMap = std::unordered_map<std::string, std::shared_ptr<RTModel>>;
	public:
		auto GetModel(const std::string& name )const -> const std::shared_ptr<RTModel>&;
		void SetModel(const std::string& name, const std::shared_ptr<RTModel>& model);
		auto GetModels()const noexcept -> const ModelMap&;
		virtual ~RTScene()noexcept {}
	private:
		ModelMap m_Models = {};
	};
}
#endif