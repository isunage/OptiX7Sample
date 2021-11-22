#include "../include/Tracers/Test24SimpleViewGLTracer.h"

void test::Test24SimpleViewGLTracer::Initialize()
{
	this->InitProgram();
}

void test::Test24SimpleViewGLTracer::Terminate()
{
	this->FreeProgram();
	m_Framebuffer.reset();
}

void test::Test24SimpleViewGLTracer::InitProgram()
{
	std::string  simpleViewVSString;
	std::string  simpleViewFSString;
	{
		std::fstream simpleViewVSFile(TEST_TEST24_GLSL_PATH"/SimpleView.Vert.glsl", std::ios::binary | std::ios::in);
		if (simpleViewVSFile.is_open())
		{
			simpleViewVSString = std::string(
				(std::istreambuf_iterator<char>(simpleViewVSFile)),
				(std::istreambuf_iterator<char>())
			);
			simpleViewVSFile.close();
		}
		std::fstream simpleViewFSFile(TEST_TEST24_GLSL_PATH"/SimpleView.Frag.glsl", std::ios::binary | std::ios::in);
		if (simpleViewFSFile.is_open())
		{
			simpleViewFSString = std::string(
				(std::istreambuf_iterator<char>(simpleViewFSFile)),
				(std::istreambuf_iterator<char>())
			);
			simpleViewFSFile.close();
		}
	}

	m_GLProgram = std::make_unique<rtlib::GLProgram>();
	m_GLProgram->create();
	auto simpleViewVS = std::make_unique<rtlib::GLVertexShader  >(simpleViewVSString);
	auto simpleViewFS = std::make_unique<rtlib::GLFragmentShader>(simpleViewFSString);
	if (!simpleViewVS->compile()) {
		std::cout << simpleViewVS->getLog() << std::endl;
	}
	if (!simpleViewFS->compile()) {
		std::cout << simpleViewFS->getLog() << std::endl;
	}
	m_GLProgram->attach(*simpleViewVS.get());
	m_GLProgram->attach(*simpleViewFS.get());
	if (!m_GLProgram->link()) {
		std::cout << m_GLProgram->getLog() << std::endl;
	}
	simpleViewVS.reset();
	simpleViewFS.reset();
}

void test::Test24SimpleViewGLTracer::FreeProgram()
{
	m_GLProgram.reset();
}

void test::Test24SimpleViewGLTracer::InitScene()
{
}

void test::Test24SimpleViewGLTracer::FreeScene()
{
}
