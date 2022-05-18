#ifndef TEST_TEST25_APPLICATION_H
#define TEST_TEST25_APPLICATION_H
#include <TestLib/RTApplication.h>
#include <TestLib/RTTracer.h>
namespace test
{
	class RTCamera;
	class RTFrame ;
	class RTFrame {

	};
	class RTFrameCU: public RTFrame
	{

	};
	class RTFrameGL: public RTFrame
	{

	};
}
namespace test25
{
	class ApplicationBuilder {
	public :
	};
	class Application : public test::RTApplication
	{
	public:
		virtual ~Application()noexcept;
		// RTApplication を介して継承されました
		virtual void Initialize() override;
		virtual void MainLoop() override;
		virtual void CleanUp() override;
	private:
		Application()noexcept;
	};
}
#endif
