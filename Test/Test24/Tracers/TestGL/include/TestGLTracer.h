#ifndef TEST_TEST24_TRACER_TEST_GL_TRACER_H
#define TEST_TEST24_TRACER_TEST_GL_TRACER_H
#include <TestLib/RTTracer.h>
#include <TestLib/RTFrameBuffer.h>
#include <TestLib/RTAssets.h>
#include <RTLib/core/GL.h>
#include <RTLib/ext/Camera.h>
#include <GLFW/glfw3.h>
#include <Test24Share.h>
class Test24TestGLTracer: public test::RTTracer
{
private:
	struct Uniforms {
		std::array<float, 16>    model;
		std::array<float, 16> viewProj;
	};
public:
	Test24TestGLTracer(
		int fbWidth, int fbHeight, GLFWwindow* window, 
		const std::shared_ptr<test::RTObjModelAssetManager>& objModelManager,
		const std::shared_ptr<test::RTFramebuffer>& framebuffer,
		const std::shared_ptr < rtlib::ext::CameraController >& cameraController ,
		const std::string& objModelName,
		const unsigned int& eventFlags
	)noexcept;
	// RTTracer を介して継承されました
	virtual void Initialize() override;
	virtual void CleanUp()override;
	virtual void Launch(int fbWidth, int fbHeight, void* pUserData) override;
	virtual void Update() override;
	virtual ~Test24TestGLTracer()noexcept {}
private:
	void InitProgramGL();
	void FreeProgramGL();
	void InitVertexArray();
	void FreeVertexArray();
	void UpdateVertexArray();
	void InitFramebufferGL();
	void FreeFramebufferGL();
	void ResizeFrame();
	void InitUniforms();
	void FreeUniforms();
	void UpdateUniforms();
private:
	int                                           m_FbWidth;
	int                                           m_FbHeight;
	GLFWwindow*                                   m_Window;
	std::shared_ptr<test::RTObjModelAssetManager> m_ObjModelManager;
	std::shared_ptr<rtlib::ext::CameraController> m_CameraController;
	std::shared_ptr< test::RTFramebuffer>	      m_Framebuffer;
	std::unique_ptr<rtlib::GLProgram>			  m_GLProgram;
	test::RTObjModel                              m_ObjModel;
	std::string                                   m_CurObjModelName;
	const std::string&                            m_NewObjModelName;
	std::unique_ptr<rtlib::GLBuffer<float3>>      m_VertexBuffer;
	std::unique_ptr<rtlib::GLBuffer<float2>>      m_TexCrdBuffer;
	std::unique_ptr<rtlib::GLBuffer<uint32_t>>    m_IndexBuffer;
	std::unique_ptr<rtlib::GLBuffer<Uniforms>>    m_UniformBuffer;
	GLuint									      m_VertexArrayGL;
	GLuint                                        m_FramebufferGL;
	GLuint                                        m_DSTextureGL;
	GLint                                         m_UniformLoc;
	float                                         m_ZNear;
	float                                         m_ZFar;
	const unsigned int &                          m_EventFlags;
	bool                                          m_UpdateObjModel;
};
#endif