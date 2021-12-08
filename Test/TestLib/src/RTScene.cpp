#include "..\include\TestLib\RTScene.h"

auto test::RTScene::GetCameraController() const -> CameraControllerPtr
{
    return m_CameraController;
}
