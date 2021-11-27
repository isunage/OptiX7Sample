#include "..\include\TestLib\RTCamera.h"
#include "Cameras/RTPinhole.h"
test::RTCameraCache::RTCameraCache() noexcept
{
}

bool test::RTCameraCache::AddReader(const RTCameraReaderPtr& reader) noexcept
{
    if (m_Readers.count(reader->GetPluginName())>0) {
        return false;
    }
    else {
        m_Readers[reader->GetPluginName()] = reader;
        return true;
    }
}

bool test::RTCameraCache::HasReader(const std::string& id) const noexcept
{
    return m_Readers.count(id) > 0;
}

auto test::RTCameraCache::GetReader(const std::string& id) const -> RTCameraReaderPtr
{
    return m_Readers.at(id);
}

auto test::RTCameraCache::LoadJsonFromData(const nlohmann::json& json) noexcept -> RTCameraPtr
{
    if (json.is_object()) {
		RTCameraPtr ptr;
		for (auto& [id, reader] : m_Readers) {
			if (ptr = reader->LoadJsonFromData(json)) {
				return ptr;
			}
		}
	}
	return nullptr;
}

auto test::GetDefaultCameraCache() noexcept -> std::shared_ptr<RTCameraCache>
{
    auto camera = std::make_shared<RTCameraCache>();
    camera->AddReader(std::make_shared<RTPinholeReader>());
    return camera;
}
