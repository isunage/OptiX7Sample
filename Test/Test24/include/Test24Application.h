#ifndef TEST_TEST24_APPLICATION_H
#define TEST_TEST24_APPLICATION_H
#include <TestLib/RTApplication.h>
#include <memory>
namespace test {
	class Test24Application : public test::RTApplication
	{
	public:
		Test24Application();
		// RTApplication ÇâÓÇµÇƒåpè≥Ç≥ÇÍÇ‹ÇµÇΩ
		virtual void Initialize() override;
		virtual void MainLoop() override;
		virtual void Terminate() override;
		virtual ~Test24Application();
	private:
		void InitWindow();
		void FreeWindow();
		void InitFramebuffer();
		void FreeFramebuffer();
		void InitRenderer();
		void FreeRenderer();
		void InitGui();
		void FreeGui();
		//
		void Render();
		void Update();
	private:
		struct Impl;
		std::unique_ptr<Impl> m_Impl;
	};
}
#endif
