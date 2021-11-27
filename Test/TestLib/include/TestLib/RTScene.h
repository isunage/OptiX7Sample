#ifndef TEST_RT_SCENE_H
#define TEST_RT_SCENE_H
#include <TestLib/RTInterface.h>
#include <TestLib/RTShape.h>
#include <TestLib/RTCamera.h>
#include <TestLib/RTTexture.h>
#include <TestLib/RTMaterial.h>
namespace test
{
	class RTSceneReader;
	class RTScene
	{
	public:
		RTScene()noexcept {}
		auto GetJsonAsString()const noexcept -> std::string;
		auto GetJsonAsData()const noexcept -> nlohmann::json ;
		~RTScene()noexcept {}
	private:
		friend class RTSceneReader;
		RTCameraPtr                                    m_Camera;
		std::unordered_map<std::string, RTTexturePtr>  m_Textures;
		std::unordered_map<std::string, RTMaterialPtr> m_Materials;
		std::unordered_map<std::string, RTShapePtr   > m_Shapes;
	};
	using RTScenePtr = std::shared_ptr<RTScene>;
	class RTSceneReader {
	public:
		 RTSceneReader()noexcept;
		 auto LoadJsonFromData(const nlohmann::json& json)noexcept -> RTScenePtr;
		~RTSceneReader()noexcept;
	private:
		std::shared_ptr<   RTShapeCache> m_ShpCache;
		std::shared_ptr<  RTCameraCache> m_CamCache;
		std::shared_ptr< RTTextureCache> m_TexCache;
		std::shared_ptr<RTMaterialCache> m_MatCache;
	};
	using RTSceneReaderPtr = std::shared_ptr<RTSceneReader>;
}
#endif