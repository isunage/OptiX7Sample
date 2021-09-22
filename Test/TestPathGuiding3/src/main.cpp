#include <TestPG3Application.h>
int main()
{
	auto app = TestPG3Application::New();
	app->Initialize();
	app->MainLoop();
	app->CleanUp();
}