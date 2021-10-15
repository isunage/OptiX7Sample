#include <TestPG4Application.h>
int main()
{
	auto app = TestPG4Application::New();
	app->Initialize();
	app->MainLoop();
	app->CleanUp();
}