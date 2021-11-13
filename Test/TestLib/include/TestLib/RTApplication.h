#ifndef TEST_RT_APPLICATION_H
#define TEST_RT_APPLICATION_H
#include <unordered_map>
#include <string>
#include <memory>
namespace test{
    class RTApplication
    {
    public:
        RTApplication(std::string name = "")noexcept
        {
            m_Name = name;
        }
        int Run(int argc = 0, const char** argv = nullptr);
        virtual void Initialize() = 0;
        virtual void MainLoop()   = 0;
        virtual void Terminate()  = 0;
        virtual ~RTApplication(){}
    protected:
        auto GetName()const noexcept -> std::string;
        auto GetArgc()const noexcept -> int;
        auto GetArgv()const noexcept -> const char**;
    private:
        void SetArgs(int argc, const char** argv);
    private:
        std::string  m_Name = "";
        int          m_Argc = 0;
        const char** m_Argv = nullptr;
    };
    using RTApplicationPtr = std::shared_ptr<RTApplication>;
}
#endif
