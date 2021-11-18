#ifndef TEST_RT_GUI_H
#define TEST_RT_GUI_H
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <unordered_map>
#include <stdexcept>
#include <variant>
#include <vector>
#include <memory>
#include <string>
#include <optional>
namespace test{
    class RTGuiWindow;
    class RTGui
    {
    public:
        RTGui(GLFWwindow* window)noexcept {
            m_Window = window;
        }
        void SetStyleColor(ImGuiCol idx, const ImU32   col);
        void SetStyleColor(ImGuiCol idx, const ImVec4& col);
		void Initialize();
        void DrawFrame();
		void CleanUp();
        //GuiWindow
        void AddGuiWindow(std::shared_ptr<RTGuiWindow> guiWindow);
        auto PopGuiWindow()->std::shared_ptr<RTGuiWindow>;
        auto GetGuiWindow(size_t idx)const->std::shared_ptr<RTGuiWindow>;
        auto GetGuiWindows()const ->const std::vector<std::shared_ptr<RTGuiWindow>>&;
        virtual ~RTGui()noexcept{}
    private:
        void BeginFrame();
        void EndFrame();
        std::unordered_map<ImGuiCol, std::variant<ImU32,ImVec4>> m_StyleColors= {};
        std::vector<std::shared_ptr<RTGuiWindow>>                m_GuiWindows = {};
        GLFWwindow*                                              m_Window     = nullptr;
    };
    class RTGuiWindow
    {
    private:
        struct WindowArgs {
            bool*            p_open;
            ImGuiWindowFlags flags;
        };
        struct PosArgs {
            ImVec2    pos;
            ImGuiCond cond;
            ImVec2    pivot;
        };
        struct SizeArgs {
            ImVec2    size;
            ImGuiCond cond;
        };
    public:
        RTGuiWindow(std::string title = "")noexcept :m_Title{title} {}
        RTGuiWindow(std::string title, ImGuiWindowFlags flags):RTGuiWindow(title){
            m_WindowArgs = { &m_IsActive,flags };
        }
        bool IsActive()const noexcept { return m_IsActive; }
        void SetActive(bool isActive) { m_IsActive = isActive; }
        void SetTitle(const std::string& title);
        void SetNextPos( const ImVec2& pos , ImGuiCond cond = 0, const ImVec2& pivot = ImVec2(0, 0));
        void SetNextSize(const ImVec2& size, ImGuiCond cond = 0);
        void DrawFrame();
        virtual void DrawGui() = 0;
        virtual ~RTGuiWindow()noexcept {}
    private:
        bool                      m_IsActive     = true;
        std::string               m_Title        = "";
        std::optional<WindowArgs> m_WindowArgs   = std::nullopt;
        std::optional<PosArgs>    m_NextPosArgs  = std::nullopt;
        std::optional<SizeArgs>   m_NextSizeArgs = std::nullopt;
    };
}
#endif