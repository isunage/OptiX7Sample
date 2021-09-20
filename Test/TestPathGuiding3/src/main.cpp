#include <TestPGConfig.h>
#include <RTLib/GL.h>
#include <RTLib/CUDA.h>
#include <RTLib/CUDA_GL.h>
#include <RTLib/Camera.h>
#include <RTLib/Utils.h>
#include <RTLib/VectorFunction.h>
#include <RTLib/ext/RectRenderer.h>
#include <cuda/RayTrace.h>
#include <GLFW/glfw3.h>
#include <stb_image_write.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include "../include/RTPathGuidingUtils.h"
#include "../include/RTApplication.h"
#include "../include/RTTracer.h"
#include "../include/SceneBuilder.h"
#include <filesystem>
#include <fstream>
#include <unordered_map>
#include <random>
#include <sstream>
#include <chrono>
#include <string>
namespace test {
	std::string SpecifyMaterialType(const rtlib::ext::Material& material) {
		auto emitCol  = material.GetFloat3As<float3>("emitCol");
		auto tranCol  = material.GetFloat3As<float3>("tranCol");
		auto refrIndx = material.GetFloat1("refrIndx");
		auto shinness = material.GetFloat1("shinness");
		auto illum    = material.GetUInt32("illum");
		if (illum == 7) {
			return "Refraction";
		}else {
			return "Phong";
		}
	}
}
class TestPG3Application : public test::RTApplication
{
public:
	TestPG3Application() :test::RTApplication() {}
	static  auto New()->std::shared_ptr< test::RTApplication>
	{
		return std::make_shared<TestPG3Application>();
	}
	virtual ~TestPG3Application(){}
};
class TestPGContextEvent : public test::RTAppEvent {
public:

};
int main() {
	auto app = TestPG3Application::New();
	app->Initialize();
	app->MainLoop();
	app->CleanUp();
}