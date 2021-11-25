#ifndef TEST_RT_SERIALIZABLE_H
#define TEST_RT_SERIALIZABLE_H
#include <nlohmann/json.hpp>
namespace test{
	class RTSerializable
	{
	public:
		RTSerializable()noexcept {}
		virtual auto GetJsonData()const noexcept -> nlohmann::json = 0;
		virtual ~RTSerializable()noexcept {}
	};
	using RTSerializablePtr = std::shared_ptr<RTSerializable>;
	class RTSerializableReader
	{
	public:
		RTSerializableReader()noexcept {}
		virtual auto LoadJsonData(const nlohmann::json&)const noexcept -> RTSerializablePtr = 0;
		virtual ~RTSerializableReader()noexcept {}
	};
}
#endif
