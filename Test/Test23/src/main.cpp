#include <Test23Application.h>
int main()
{
	auto app = Test23Application::New();
	app->Initialize();
	app->MainLoop();
	app->CleanUp();
}