#ifndef TEST_RT_SCENE_H
#define TEST_RT_SCENE_H
#include <RTLib/ext/Camera.h>
#include <memory>
namespace test
{
    class RTScene
    {
    private:
        using CameraControllerPtr = std::shared_ptr<rtlib::ext::CameraController> ;
    public:
        RTScene()noexcept {}
        /*Camera*/
        auto GetCameraController()const -> CameraControllerPtr;
        /*Mesh  */
        virtual ~RTScene()noexcept {}
    private:
        CameraControllerPtr m_CameraController = nullptr;

    };
}
#endif
