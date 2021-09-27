#include <Test22Application.h>
int main()
{
	auto app = Test22Application::New();
	app->Initialize();
	app->MainLoop();
	app->CleanUp();
}