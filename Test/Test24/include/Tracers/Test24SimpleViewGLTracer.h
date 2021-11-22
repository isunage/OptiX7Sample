#ifndef TEST_TEST24_TRACER_SIMPLE_VIEW_GL_TRACER_H
#define TEST_TEST24_TRACER_SIMPLE_VIEW_GL_TRACER_H
#include <Test24Config.h>
#include <RTLib/ext/Math/VectorFunction.h>
#include <RTLib/ext/Resources.h>
#include <RTLib/ext/Resources/GL.h>
#include <TestLib/RTFramebuffer.h>
#include <iostream>
#include <fstream>
#include <string>
#include <memory>
namespace test
{

	class  Test24SimpleViewGLTracer
	{
	public:
		Test24SimpleViewGLTracer(
			const int& width_, const int& height_, const bool& isResized_,
			std::shared_ptr<test::RTFrameBuffer> framebuffer_
		) : m_Width{ width_ }, m_Height{ height_ }, m_IsResized{ isResized_ }
		{
			m_Framebuffer = framebuffer_;
		}
		void Initialize();

		void Terminate();

		~Test24SimpleViewGLTracer() {}
	private:
		void InitProgram();
		void FreeProgram();
		void InitScene();
		void FreeScene();
	private:
		const int& m_Width;
		const int& m_Height;
		const int& m_IsResized;
		std::unique_ptr<rtlib::GLProgram>	 m_GLProgram = nullptr;
		std::shared_ptr<test::RTFrameBuffer> m_Framebuffer = nullptr;
	};
}
#endif
