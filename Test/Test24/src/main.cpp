#include <Test24Application.h>
#include <iostream>
int main(int argc, const char* argv[])
{
	auto app = test::Test24Application::New(1024, 1024, "Test24Application");
	try {
		app->Initialize();
		app->MainLoop();
	}
	catch (std::runtime_error& err)
	{
		std::cout << err.what() << std::endl;
	}
	app->CleanUp();
	return 0;
}