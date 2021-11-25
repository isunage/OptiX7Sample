#ifndef TEST_RT_MATERIAL_H
#define TEST_RT_MATERIAL_H
#include <RTLib/ext/Mesh.h>
#include <RTLib/ext/Math/Matrix.h>
#include <RTLib/ext/VariableMap.h>
#include <nlohmann/json.hpp>
#include <string>
#include <memory>
namespace test
{

	class RTMaterial
	{
	public:
		RTMaterial()noexcept {}
		virtual auto GetJsonData()const noexcept -> nlohmann::json = 0;
		virtual ~RTMaterial()noexcept {}
	};
	using RTMaterialPtr = std::shared_ptr<RTMaterial>;

	class RTMaterialReader
	{
	public:
		RTMaterialReader()noexcept {}
		virtual auto LoadJsonData(const nlohmann::json&)const noexcept -> RTMaterialPtr = 0;
		virtual ~RTMaterialReader()noexcept {}
	};
	using RTMaterialReaderPtr = std::shared_ptr<RTMaterialReader>;

}
#endif