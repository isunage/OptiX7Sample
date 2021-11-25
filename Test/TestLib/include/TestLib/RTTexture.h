#ifndef TEST_RT_TEXTURE_H
#define TEST_RT_TEXTURE_H
#include <RTLib/ext/Mesh.h>
#include <RTLib/ext/Math/Matrix.h>
#include <RTLib/ext/VariableMap.h>
#include <TestLib/RTMaterial.h>
#include <nlohmann/json.hpp>
#include <string>
#include <memory>
namespace test
{
	class RTTexture
	{
	public:
		RTTexture()noexcept {}
		virtual auto GetJsonData() const noexcept -> nlohmann::json = 0;
		virtual ~RTTexture()noexcept {}
	};
	using RTTexturePtr = std::shared_ptr<RTTexture>;

	class RTTextureReader
	{
	public:
		RTTextureReader()noexcept {}
		virtual auto LoadJsonData(const nlohmann::json&)const noexcept -> RTTexturePtr  = 0;
		virtual ~RTTextureReader() {}
	};
	using RTTextureReaderPtr = std::shared_ptr<RTTextureReader>;
}
#endif