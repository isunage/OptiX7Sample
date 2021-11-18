#include <Test24Application.h>
#include <memory>
int main(int argc, const char* argv[])
{
	auto app = test::RTApplicationPtr(
		new test::Test24Application()
	);
	app->Run(argc, argv);
}