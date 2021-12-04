#include <Test24Application.h>
int main(int argc, const char* argv[])
{
	auto app = Test24Application::New(800, 600, "Test24Application");
	return app->Run(argc,argv);
}