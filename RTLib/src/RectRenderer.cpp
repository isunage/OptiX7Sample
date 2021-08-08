#include "../include/RTLib/ext/RectRenderer.h"
#include <iostream>

void rtlib::ext::RectRenderer::initProgram()
{
	m_Program = rtlib::GLProgram();
	{
		m_Program.create();
		auto vs = rtlib::GLVertexShader(std::string(vsSource));
		auto fs = rtlib::GLFragmentShader(std::string(fsSource));
		if (!vs.compile()) {
			std::cout << vs.getLog() << std::endl;
		}
		if (!fs.compile()) {
			std::cout << fs.getLog() << std::endl;
		}
		m_Program.attach(vs);
		m_Program.attach(fs);
		if (!m_Program.link()) {
			std::cout << m_Program.getLog() << std::endl;
		}
	}
	m_TexLoc = m_Program.getUniformLocation("tex");
}

void rtlib::ext::RectRenderer::initRectMesh()
{
	m_RectVBO = rtlib::GLBuffer(screenVertices, GL_ARRAY_BUFFER, GL_STATIC_DRAW);
	m_RectIBO = rtlib::GLBuffer(screenIndices, GL_ELEMENT_ARRAY_BUFFER, GL_STATIC_DRAW);
	m_RectVAO = GLuint(0);
	{
		glGenVertexArrays(1, &m_RectVAO);
		glBindVertexArray(m_RectVAO);
		m_RectVBO.bind();
		m_RectIBO.bind();
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 5, (void*)(sizeof(float) * 0));
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 5, (void*)(sizeof(float) * 3));
		glEnableVertexAttribArray(1);
		glBindVertexArray(0);
		m_RectVBO.unbind();
		m_RectIBO.unbind();
	}

}