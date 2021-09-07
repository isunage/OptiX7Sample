#ifndef TEST_RT_APPLICATION_H
#define TEST_RT_APPLICATION_H
namespace test{
    class RTApplication 
    {
    public:
        virtual void Initialize() = 0;
        virtual void MainLoop()   = 0;
        virtual void CleanUp()    = 0;
        virtual ~RTApplication()noexcept{}
    };
}
#endif