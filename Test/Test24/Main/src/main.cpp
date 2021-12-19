#include <Test24Application.h>
#include <RTLib/math/Matrix.h>
void TestMatrixCalc() {
	auto width = 800;
	auto height = 600;
	auto m_CameraController = std::make_shared<rtlib::ext::CameraController>(float3{ 0.0f, 1.0f, 5.0f });
	m_CameraController->SetMouseSensitivity(0.125f);
	m_CameraController->SetMovementSpeed(10.f);
	m_CameraController->SetZoom(40.0f);
	auto camera = m_CameraController->GetCamera(static_cast<float>(width) / static_cast<float>(height));
	auto eye = camera.getEye();
	auto [u, v, w] = camera.getUVW();
	auto idx = make_int2(12, 12);
	auto d = make_float2(2.0f * static_cast<float>(idx.x) / static_cast<float>(width) - 1.0f, 2.0f * static_cast<float>(idx.y) / static_cast<float>(height) - 1.0f);
	auto direction = rtlib::normalize(d.x * u + d.y * v + w);
	auto position = eye + 100.0f * direction;
	auto eye2pos = position - eye;
	auto stu = rtlib::Matrix3x3(u, v, eye2pos).Inverse() * make_float3(-w.x, -w.y, -w.z);
	std::cout << d.x << "," << d.y << std::endl;
	std::cout << stu.x << "," << stu.y << std::endl;
}
int main(int argc, const char* argv[])
{
	auto app = Test24Application::New(800, 600, "Test24Application");
	return app->Run(argc,argv);

	return 0;
}