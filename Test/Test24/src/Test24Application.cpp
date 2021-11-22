#include "..\include\Test24Application.h"
#include "..\include\Tracers\Test24SimpleViewGLTracer.h"
#include <tiny_obj_loader.h>
#include <RTLib/ext/Math/VectorFunction.h>
#include <RTLib/GL.h>
#include <RTLib/ext/RectRenderer.h>
#include <RTLib/ext/Camera.h>
#include <RTLib/ext/Mesh.h>
#include <TestLib/RTFramebuffer.h>
#include <TestLib/RTGui.h>
#include <Test24Config.h>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <memory>
namespace test {

	struct Test24Model
	{
		bool                        isSucceeded;
		rtlib::ext::MeshGroupPtr	meshGroup;
		rtlib::ext::VariableMapList materialList;
		static auto LoadObj(const std::string& filename, const std::string& mtlBaseDir = "")-> Test24Model
		{
			bool isSucceeded = false;
			rtlib::ext::MeshGroupPtr	meshGroup    = rtlib::ext::MeshGroupPtr(new rtlib::ext::MeshGroup());
			rtlib::ext::VariableMapList materialList = {};
			{
				tinyobj::ObjReader       reader = {};
				tinyobj::ObjReaderConfig config = {};
				config.mtl_search_path          = mtlBaseDir;
				config.triangulate              = true;
				config.triangulation_method     = "simple";
				if (reader.ParseFromFile(filename, config)) {
					if (!reader.Warning().empty()) {
						std::cout << "TinyObjReader: " << reader.Warning();
					}
					auto& attrib    = reader.GetAttrib();
					auto& shapes    = reader.GetShapes();
					auto& materials = reader.GetMaterials();
					{
						meshGroup->SetSharedResource(std::make_shared<rtlib::ext::MeshSharedResource>());
						auto& vertexBuffer = meshGroup->GetSharedResource()->vertexBuffer;
						auto& texCrdBuffer = meshGroup->GetSharedResource()->texCrdBuffer;
						auto& normalBuffer = meshGroup->GetSharedResource()->normalBuffer;
						auto& sharedVariables = meshGroup->GetSharedResource()->variables;
						sharedVariables.SetString("type"       , "Obj"     );
						sharedVariables.SetString("ObjFilePath", filename  );
						sharedVariables.SetString("MtlBaseDir" , mtlBaseDir);
						struct MyHash
						{
							MyHash()noexcept {}
							MyHash(const MyHash&)noexcept = default;
							MyHash(MyHash&&)noexcept = default;
							~MyHash()noexcept {}
							MyHash& operator=(const MyHash&)noexcept = default;
							MyHash& operator=(MyHash&&)noexcept = default;
							size_t operator()(tinyobj::index_t key)const
							{
								size_t vertexHash = std::hash<int>()(key.vertex_index)   & 0x3FFFFF;
								size_t normalHash = std::hash<int>()(key.normal_index)   & 0x1FFFFF;
								size_t texCrdHash = std::hash<int>()(key.texcoord_index) & 0x1FFFFF;
								return vertexHash + (normalHash << 22) + (texCrdHash << 43);
							}
						};
						struct MyEqualTo
						{
							using first_argument_type = tinyobj::index_t;
							using second_argument_type = tinyobj::index_t;
							using result_type = bool;
							constexpr bool operator()(const tinyobj::index_t& x, const tinyobj::index_t& y)const
							{
								return (x.vertex_index == y.vertex_index) && (x.texcoord_index == y.texcoord_index) && (x.normal_index == y.normal_index);
							}
						};

						std::vector< tinyobj::index_t> indices = {};
						std::unordered_map<tinyobj::index_t, size_t, MyHash, MyEqualTo> indicesMap = {};
						for (size_t i = 0; i < shapes.size(); ++i) {
							for (size_t j = 0; j < shapes[i].mesh.num_face_vertices.size(); ++j) {
								for (size_t k = 0; k < 3; ++k) {
									//tinyobj::idx
									tinyobj::index_t idx = shapes[i].mesh.indices[3 * j + k];
									if (indicesMap.count(idx) == 0) {
										size_t indicesCount = std::size(indices);
										indicesMap[idx] = indicesCount;
										indices.push_back(idx);
									}
								}
							}
						}
						std::cout << "VertexBuffer: " << attrib.vertices.size()  / 3 << "->" << indices.size() << std::endl;
						std::cout << "NormalBuffer: " << attrib.normals.size()   / 3 << "->" << indices.size() << std::endl;
						std::cout << "TexCrdBuffer: " << attrib.texcoords.size() / 2 << "->" << indices.size() << std::endl;
						vertexBuffer.Resize(indices.size());
						texCrdBuffer.Resize(indices.size());
						normalBuffer.Resize(indices.size());

						for (size_t i = 0; i < indices.size(); ++i) {
							tinyobj::index_t idx = indices[i];
							vertexBuffer[i] = make_float3(
								attrib.vertices[3 * idx.vertex_index + 0],
								attrib.vertices[3 * idx.vertex_index + 1],
								attrib.vertices[3 * idx.vertex_index + 2]);
							if (idx.normal_index >= 0) {
								normalBuffer[i] = make_float3(
									attrib.normals[3 * idx.normal_index + 0],
									attrib.normals[3 * idx.normal_index + 1],
									attrib.normals[3 * idx.normal_index + 2]);
							}
							else {
								normalBuffer[i] = make_float3(0.0f, 1.0f, 0.0f);
							}
							if (idx.texcoord_index >= 0) {
								texCrdBuffer[i] = make_float2(
									attrib.texcoords[2 * idx.texcoord_index + 0],
									attrib.texcoords[2 * idx.texcoord_index + 1]);
							}
							else {
								texCrdBuffer[i] = make_float2(0.5f, 0.5f);
							}
						}

						std::unordered_map<std::size_t, std::size_t> texCrdMap = {};
						for (size_t i = 0; i < shapes.size() ; ++i) {
							std::unordered_map<uint32_t, uint32_t> tmpMaterials = {};
							auto uniqueResource = std::make_shared<rtlib::ext::MeshUniqueResource>();
							uniqueResource->name = shapes[i].name;
							uniqueResource->triIndBuffer.Resize(shapes[i].mesh.num_face_vertices.size());
							for (size_t j = 0; j < shapes[i].mesh.num_face_vertices.size(); ++j) {
								uint32_t idx0 = indicesMap.at(shapes[i].mesh.indices[3 * j + 0]);
								uint32_t idx1 = indicesMap.at(shapes[i].mesh.indices[3 * j + 1]);
								uint32_t idx2 = indicesMap.at(shapes[i].mesh.indices[3 * j + 2]);
								uniqueResource->triIndBuffer[j] = make_uint3(idx0, idx1, idx2);
							}
							uniqueResource->matIndBuffer.Resize(shapes[i].mesh.material_ids.size());
							for (size_t j = 0; j < shapes[i].mesh.material_ids.size(); ++j) {
								if (tmpMaterials.count(shapes[i].mesh.material_ids[j]) != 0) {
									uniqueResource->matIndBuffer[j] = tmpMaterials.at(shapes[i].mesh.material_ids[j]);
								}
								else {
									int newValue = tmpMaterials.size();
									tmpMaterials[shapes[i].mesh.material_ids[j]] = newValue;
									uniqueResource->matIndBuffer[j] = newValue;
								}
							}
							uniqueResource->materials.resize(tmpMaterials.size());
							for (auto& [Ind, RelInd] : tmpMaterials) {
								uniqueResource->materials[RelInd] = Ind;
							}
							meshGroup->SetUniqueResource(shapes[i].name, uniqueResource);
						}
					}
					{
						materialList.resize(materials.size());
						for (size_t i = 0; i < materialList.size(); ++i) {
							materialList[i].SetString("name"   , materials[i].name);
							materialList[i].SetString("type"   , "Mtl");
							materialList[i].SetUInt32("illum"  , materials[i].illum);
							materialList[i].SetFloat3("diffCol",
								{ materials[i].diffuse[0],
								  materials[i].diffuse[1],
								  materials[i].diffuse[2]
								});
							materialList[i].SetFloat3("specCol",
								{ materials[i].specular[0],
								  materials[i].specular[1],
								  materials[i].specular[2]
								});
							materialList[i].SetFloat3("tranCol",
								{ materials[i].transmittance[0],
								  materials[i].transmittance[1] ,
								  materials[i].transmittance[2]
								});
							materialList[i].SetFloat3("emitCol",
								{ materials[i].emission[0],
								  materials[i].emission[1] ,
								  materials[i].emission[2]
								});
							if (!materials[i].diffuse_texname.empty()) {
								materialList[i].SetString("diffTex", mtlBaseDir+ "\\" + materials[i].diffuse_texname);
							}
							else {
								materialList[i].SetString("diffTex", "");
							}
							if (!materials[i].specular_texname.empty()) {
								materialList[i].SetString("specTex", mtlBaseDir + "\\" + materials[i].specular_texname);
							}
							else {
								materialList[i].SetString("specTex", "");
							}
							if (!materials[i].emissive_texname.empty()) {
								materialList[i].SetString("emitTex", mtlBaseDir + "\\" + materials[i].emissive_texname);
							}
							else {
								materialList[i].SetString("emitTex", "");
							}
							if (!materials[i].specular_highlight_texname.empty()) {
								materialList[i].SetString("shinTex", mtlBaseDir + "\\" + materials[i].specular_highlight_texname);
							}
							else {
								materialList[i].SetString("shinTex", "");
							}
							materialList[i].SetFloat1("shinness", materials[i].shininess);
							materialList[i].SetFloat1("refrIndx", materials[i].ior);
						}
					}
				}
				else {
					meshGroup.reset();
					if (!reader.Error().empty()) {
						std::cerr << "TinyObjReader: " << reader.Error();
					}
				}
			}
			return { isSucceeded, std::move(meshGroup) ,std::move(materialList) };
		}
	};

	struct Test24Scene {
		rtlib::ext::Camera												   camera;
		std::unordered_map<std::string, rtlib::ext::MeshPtr>               meshes;
		std::unordered_map<std::string, rtlib::ext::VariableMapList>       materialLists;
		std::unordered_map<std::string, rtlib::ext::CustomImage2D<uchar4>> textures;
		rtlib::ext::VariableMap                                            traceConfig;
	};

	struct Test24Application::Impl
	{
		using RectRendererPtr = std::unique_ptr<rtlib::ext::RectRenderer>;
		using FramebufferPtr  = std::shared_ptr<test::RTFrameBuffer>;
		using GuiPtr		  = std::unique_ptr<test::RTGui>;
		using ScenePtr		  = std::shared_ptr<test::Test24Scene>;
		using ScenePtrMap     = std::unordered_map<std::string, ScenePtr>;
		void InitWindow();
		void FreeWindow();
		void InitRenderer();
		void FreeRenderer();
		void InitFramebuffer();
		void FreeFramebuffer();
		void InitGui();
		void FreeGui();
		void DrawFramebufferGL(const std::string& texName);
		void DrawFramebufferCUGL(const std::string& texName);
		void DrawFrameGui();
		GLFWwindow*     window		 = nullptr;
		int				width		 = TEST_TEST24_DEF_WIDTH ;
		int				height		 = TEST_TEST24_DEF_HEIGHT;
		std::string		title        = TEST_TEST24_DEF_TITLE ;
		RectRendererPtr renderer     = nullptr;
		FramebufferPtr  framebuffer  = nullptr;
		GuiPtr          gui		     = nullptr;
		ScenePtrMap     scenes       = {};
		bool            isResized    = false;
		float2          curCursorPos = make_float2(FLT_MAX, FLT_MAX);
		float2          delCursorPos = make_float2(   0.0f,    0.0f);
		//WindowSize
		static void WindowSizeCallback( GLFWwindow* window, int width, int height) {
			if (!window) return;
			auto impl = reinterpret_cast<Test24Application::Impl*>(glfwGetWindowUserPointer(window));
			if (!impl) return;
			if (width != impl->width || height != impl->height)
			{
				impl->width     = width;
				impl->height    = height;
				impl->isResized = true;
				glViewport(0, 0, static_cast<GLsizei>(width), static_cast<GLsizei>(height));
			}
		}
		//CursorPos
		static void CursorPosCallback(  GLFWwindow* window, double cursorPosX, double cursorPosY)
		{
			if (!window) return;
			auto impl = reinterpret_cast<Test24Application::Impl*>(glfwGetWindowUserPointer(window));
			if (!impl) return;
			
			float2 curCursorPos = impl->curCursorPos;
			impl->curCursorPos  = make_float2(cursorPosX, cursorPosY);
			if (curCursorPos.x != FLT_MAX || curCursorPos.y != FLT_MAX)
			{
				impl->delCursorPos = impl->curCursorPos - curCursorPos;
			}
		}
	};

	class  Test24DefaultWindow :public test::RTGuiWindow
	{
	public:
		Test24DefaultWindow(const std::string& title,const int& width_, const int& height_, const float2& curCursorPos_, const float2& delCursorPos_)
			:RTGuiWindow(title), width(width_), height(height_), curCursorPos(curCursorPos_), delCursorPos(delCursorPos_)
		{}
		// RTGuiWindow ����Čp������܂���
		virtual void DrawGui() override
		{
			ImGui::Text("WindowSize   : (%5d, %5d)"      , width		 , height	     );
			ImGui::Text("CurCursorPos : (%3.2f, %3.2f)", curCursorPos.x, curCursorPos.y);
			ImGui::Text("DelCursorPos : (%3.2f, %3.2f)", delCursorPos.x, delCursorPos.y);
		}
		virtual ~Test24DefaultWindow() {}
	private:
		const int   & width;
		const int   & height;
		const float2& curCursorPos;
		const float2& delCursorPos;
	};
}

test::Test24Application::Test24Application():m_Impl(new test::Test24Application::Impl()){}

void test::Test24Application::Initialize()
{
	this->InitWindow();
	this->InitRenderer();
	this->InitFramebuffer();
	this->InitGui();
	auto [isSuccess, meshGroup, materialList] = test::Test24Model::LoadObj(TEST_TEST24_DATA_PATH"/Models/CornellBox/CornellBox-Water.obj");
	auto simpleViewGLTracer = std::make_unique<test::Test24SimpleViewGLTracer>(
		m_Impl->width,m_Impl->height,m_Impl->isResized,m_Impl->framebuffer
	);
	
	simpleViewGLTracer->Initialize();

	simpleViewGLTracer-> Terminate();

}

void test::Test24Application::MainLoop()
{
	if (!m_Impl || !m_Impl->window) return;
	while (!glfwWindowShouldClose(m_Impl->window))
	{
		Render();
		Update();
		glfwSwapBuffers(m_Impl->window);
	}
}

void test::Test24Application::Terminate()
{
	this->FreeWindow();
}

	 test::Test24Application::~Test24Application()
{
	this->FreeGui();
	this->FreeFramebuffer();
	this->FreeRenderer();
	this->FreeWindow();
}

void test::Test24Application::InitWindow()
{
	m_Impl->InitWindow();
}

void test::Test24Application::FreeWindow()
{
	m_Impl->FreeWindow();
}

void test::Test24Application::InitFramebuffer()
{
	m_Impl->InitFramebuffer();
}

void test::Test24Application::FreeFramebuffer()
{
	m_Impl->FreeFramebuffer();
}

void test::Test24Application::InitRenderer()
{
	m_Impl->InitRenderer();
}

void test::Test24Application::FreeRenderer()
{
	m_Impl->FreeRenderer();
}

void test::Test24Application::InitGui()
{
	m_Impl->InitGui();
}

void test::Test24Application::FreeGui()
{
	m_Impl->FreeGui();
}

void test::Test24Application::Render()
{
	glClear(GL_COLOR_BUFFER_BIT);
	glClearColor(1.0f, 1.0f, 1.0f, 0.0f);
	m_Impl->DrawFramebufferGL("Default");
	m_Impl->DrawFrameGui();
}

void test::Test24Application::Update()
{
	glfwPollEvents();
}

void test::Test24Application::Impl::InitWindow() {
	if (!glfwInit()) {
		throw std::runtime_error("Failed To Initialize GLFW!");
	}
	glfwWindowHint(GLFW_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
	window = glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr);
	glfwMakeContextCurrent(window);
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		throw std::runtime_error("Failed To Initialize GLAD!");
	}
	glfwSetWindowUserPointer(  window, this);
	glfwSetWindowSizeCallback( window, WindowSizeCallback);
	glfwSetCursorPosCallback(  window, CursorPosCallback);
}

void test::Test24Application::Impl::FreeWindow()
{
	glfwDestroyWindow(window);
	window = nullptr;
}

void test::Test24Application::Impl::InitRenderer()
{
	renderer = std::make_unique<rtlib::ext::RectRenderer>();
	renderer->init();
}

void test::Test24Application::Impl::FreeRenderer()
{
	renderer->reset();
	renderer.reset();
}

void test::Test24Application::Impl::InitFramebuffer()
{
	std::vector<uchar4> pixels(width * height, make_uchar4(255, 255, 0, 255));
	framebuffer = std::make_shared<test::RTFrameBuffer>(width, height);
	framebuffer->AddGLTexture("Default");
	framebuffer->GetGLTexture("Default").upload(
		0, pixels.data(), 0, 0, width, height
	);
}

void test::Test24Application::Impl::FreeFramebuffer()
{
	framebuffer->CleanUp();
	framebuffer.reset();
}

void test::Test24Application::Impl::InitGui()
{
	gui = std::make_unique<test::RTGui>(
		window
	);
	gui->AddGuiWindow(std::shared_ptr<test::RTGuiWindow>(
		new test::Test24DefaultWindow(
			"default", width, height, curCursorPos, delCursorPos
		)
	));
	gui->Initialize();
}

void test::Test24Application::Impl::FreeGui()
{
	gui->CleanUp();
	gui.reset();
}

void test::Test24Application::Impl::DrawFramebufferGL(const std::string& texName)
{
	renderer->draw(framebuffer->GetGLTexture(texName).getID());
}

void test::Test24Application::Impl::DrawFramebufferCUGL(const std::string& texName)
{
	renderer->draw(framebuffer->GetCUGLBuffer(texName).getID());
}

void test::Test24Application::Impl::DrawFrameGui()
{
	gui->DrawFrame();
}
