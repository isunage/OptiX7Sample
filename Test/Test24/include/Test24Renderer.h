#ifndef TEST_TEST24_RENDERER_H
#define TEST_TEST24_RENDERER_H
#include <TestLib/RTFrameBuffer.h>
#include <TestLib/RTRenderer.h>
#include <TestLib/RTGui.h>
#include <RTLib/CUDA.h>
#include <RTLib/ext/RectRenderer.h>
#include <memory>
struct GLFWwindow;
namespace test
{
	class Test24Renderer:public test::RTRenderer
	{
	private:
		using GLTexturePtr = std::unique_ptr<rtlib::GLTexture2D<uchar4>>;
	public:
		static auto New(GLFWwindow* window, test::RTFrameBufferPtr framebuffer, test::RTGuiPtr gui)->RTRendererPtr;
		Test24Renderer( GLFWwindow* window, test::RTFrameBufferPtr framebuffer, test::RTGuiPtr gui);
		// RTRenderer ‚ð‰î‚µ‚ÄŒp³‚³‚ê‚Ü‚µ‚½
		virtual void Initialize() override;
		virtual void Render() override;
		virtual void Terminate() override;
		virtual bool Resize(int width, int height) override;
		virtual ~Test24Renderer() {
			try
			{
				this->Terminate();
			}
			catch (...)
			{

			}
		}
	private:
		GLFWwindow*                          m_Window        = nullptr;
		int                                  m_FbWidth       = 0;
		int                                  m_FbHeight      = 0;
		test::RTFrameBufferPtr               m_Framebuffer   = nullptr;
		test::RTGuiPtr				         m_Gui           = nullptr;
		rtlib::ext::RectRenderer             m_RectRenderer  = {};
	};
}
#endif
