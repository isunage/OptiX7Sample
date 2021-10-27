#ifndef RT_RENDERER_H
#define RT_RENDERER_H
#include <memory>
namespace test
{
	class RTRenderer
	{
	public:
		virtual void Initialize() = 0;
		virtual void Render()     = 0;
		virtual void Terminate()  = 0;
		virtual bool Resize(int width, int height) = 0;
		virtual ~RTRenderer()noexcept {}
	};
	using RTRendererPtr = std::shared_ptr<RTRenderer>;
}
#endif
