#ifndef RTLIB_EXT_RECT_RENDERER_H
#define RTLIB_EXT_RECT_RENDERER_H
#include "../GL.h"
#include <array>
#include <string_view>
namespace rtlib{
    namespace ext {
        class RectRenderer {
        private:
            struct Vertex {
                std::array<float,3> position;
                std::array<float,2> texCoord;
            };
        public:
            RectRenderer() {}
			void init() {
				initProgram();
				initRectMesh();
			}
			void draw(GLuint texID) {
				m_Program.use();
				glActiveTexture(GL_TEXTURE0);
				glBindTexture(GL_TEXTURE_2D,texID);
				glUniform1i(m_TexLoc, 0);
				glBindVertexArray(m_RectVAO);
				glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);
			}
			void reset() {
				glDeleteVertexArrays(1, &m_RectVAO);
				m_RectVAO = 0;
				m_TexLoc  = 0;
				m_RectIBO.reset();
				m_RectVBO.reset();
				m_Program.destroy();
			}
            ~RectRenderer(){}
        private:
            void initProgram();
            void initRectMesh();
        private:
            GLuint                               m_RectVAO = 0;
            rtlib::GLBuffer<Vertex>              m_RectVBO;
            rtlib::GLBuffer<std::uint32_t>       m_RectIBO;
			GLuint                               m_TexLoc = 0;
            rtlib::GLProgram                     m_Program;
        private:
            inline static constexpr std::array<Vertex,4> screenVertices = {
				Vertex{{-1.0f,-1.0f,0.0f},{1.0f, 0.0f}},
				Vertex{{ 1.0f,-1.0f,0.0f},{0.0f, 0.0f}},
				Vertex{{ 1.0f, 1.0f,0.0f},{0.0f, 1.0f}},
				Vertex{{-1.0f, 1.0f,0.0f},{1.0f, 1.0f}}
            };
            inline static constexpr std::array<std::uint32_t, 6>  screenIndices = {
                0,1,2,
                2,3,0
            };
            inline static constexpr std::string_view vsSource =
            "#version 330 core\n"
            "layout(location=0) in vec3 position;\n"
            "layout(location=1) in vec2 texCoord;\n"
            "out vec2 uv;\n"
            "void main(){\n"
            "   gl_Position = vec4(position,1.0f);\n"
            "   uv = texCoord;\n"
            "}\n";
            inline static constexpr std::string_view fsSource =
            "#version 330 core\n"
            "uniform sampler2D tex;\n"
            "in vec2 uv;\n"
            "layout(location=0) out vec3 color;\n"
            "void main(){\n"
            "   color = texture2D(tex,uv).xyz;\n"
            "}\n";
        };
    }
}
#endif