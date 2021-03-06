#include "../include/TestLib/RTApplication.h"

int test::RTApplication::Run(int argc, const char** argv)
{
    this->SetArgs(argc, argv);
    this->Initialize();
    this->MainLoop();
    this->Terminate();
    this->SetArgs(0, nullptr);
    return 0;
}

auto test::RTApplication::GetName() const noexcept -> std::string {
    return m_Name;
}

auto test::RTApplication::GetArgc() const noexcept -> int {
    return m_Argc;
}

auto test::RTApplication::GetArgv() const noexcept -> const char** {
    return m_Argv;
}

void test::RTApplication::SetArgs(int argc, const char** argv) {
    m_Argc = argc;
    m_Argv = argv;
}
