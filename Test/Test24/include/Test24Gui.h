#ifndef TEST_TEST24_GUI_H
#define TEST_TEST24_GUI_H
#include <TestLib/RTGui.h>
#include <TestLib/RTFrameBuffer.h>
#include <memory>
struct GLFWwindow;
namespace test
{
	class Test24Gui : public test::RTGui
	{
	public:
		static auto New(GLFWwindow* ownerWindow, std::shared_ptr<RTFrameBuffer> frameBuffer)->test::RTGuiPtr;
		//
		Test24Gui(GLFWwindow* ownerWindow, std::shared_ptr<RTFrameBuffer> frameBuffer);
		//Init
		virtual void Initialize() override;
		//Attach
		virtual void Attach() override;
		//Terminate
		virtual void Terminate()override;
		//
		virtual ~Test24Gui()noexcept {}
	private:
		void AttachOnAsset();
		void AttachOnTrace();
	private:
		GLFWwindow*                    m_Window      = nullptr;
		std::shared_ptr<RTFrameBuffer> m_Framebuffer = nullptr;
	};
}
#endif