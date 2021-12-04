#ifndef TEST_RT_GUI_H
#define TEST_RT_GUI_H
#include <TestLib/RTContext.h>
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
    //Gui->Window,MainMenuBar
    //Window->MenuBar, SubObject
    //MainMenuBar->Menu
    //MenuBar->Menu
    //Menu->MenuItem
    //MenuItem->SubObject
    class RTGui;
    class RTGuiMainMenuBar;
    class RTGuiWindow;
    class RTGuiMenuBar;
    class RTGuiMenu;
    class RTGuiMenuItem;
    class RTGuiSubObject;
    class RTGui
    {
    public:
        RTGui(GLFWwindow* window)noexcept;
        void SetStyleColor(ImGuiCol idx, const ImU32   col);
        void SetStyleColor(ImGuiCol idx, const ImVec4& col);
		void Initialize();
        void DrawFrame();
		void CleanUp();
        //MainMenuBar
        bool HasGuiMainMenuBar()const noexcept;
        auto AddGuiMainMenuBar()->std::shared_ptr<RTGuiMainMenuBar>;
        auto GetGuiMainMenuBar()const -> std::shared_ptr<RTGuiMainMenuBar>;
        //GuiWindow
        void SetGuiWindow(std::shared_ptr<RTGuiWindow> guiWindow);
        void PopGuiWindow();
        auto GetGuiWindow(size_t idx)const->std::shared_ptr<RTGuiWindow>;
        virtual ~RTGui()noexcept;
    private:
        void BeginFrame();
        void EndFrame();
        std::unordered_map<ImGuiCol, std::variant<ImU32,ImVec4>> m_StyleColors    = {};
        std::shared_ptr<RTGuiMainMenuBar>                        m_GuiMainMenuBar = {};
        std::vector<std::shared_ptr<RTGuiWindow>>                m_GuiWindows     = {};
        GLFWwindow*                                              m_Window         = nullptr;
    };
    class RTGuiWindow
    {
    private:
        struct WindowArgs {
            bool* p_open;
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
        using DrawCallback = void(*)(RTGuiWindow*);
        RTGuiWindow(std::string title = "")noexcept;
        RTGuiWindow(std::string title, ImGuiWindowFlags flags);
        bool IsActive()const noexcept { return m_IsActive; }
        void SetActive(bool isActive) { m_IsActive = isActive; }
        void SetTitle(const std::string& title);
        void SetNextPos(const ImVec2& pos, ImGuiCond cond = 0, const ImVec2& pivot = ImVec2(0, 0));
        void SetNextSize(const ImVec2& size, ImGuiCond cond = 0);
        void SetUserPointer(void* data)noexcept;
        auto GetUserPointer()const noexcept -> void*;
        void SetDrawCallback(DrawCallback callback)noexcept;
        void DrawFrame();
        bool HasGuiMenuBar()const noexcept;
        auto AddGuiMenuBar()->std::shared_ptr<RTGuiMenuBar>;
        auto GetGuiMenuBar()const->std::shared_ptr<RTGuiMenuBar>;
        void AddSubObject(const std::shared_ptr<RTGuiSubObject>& subobject);
        virtual void DrawGui();
        virtual ~RTGuiWindow()noexcept;
    private:
        static void DefaultDrawCallback(RTGuiWindow*) {}
        bool                      m_IsActive = true;
        std::string               m_Title = "";
        std::optional<WindowArgs> m_WindowArgs = std::nullopt;
        std::optional<PosArgs>    m_NextPosArgs = std::nullopt;
        std::optional<SizeArgs>   m_NextSizeArgs = std::nullopt;
        void* m_UserPointer = nullptr;
        DrawCallback              m_DrawCallback = DefaultDrawCallback;
        std::shared_ptr<RTGuiMenuBar>  m_GuiMenuBar = {};
        std::vector<std::shared_ptr<RTGuiSubObject>> m_SubObjects = {};
    };
    class RTGuiMainMenuBar
    {
    public:
           RTGuiMainMenuBar()noexcept;
           auto AddGuiMenu(const std::string& name)->std::shared_ptr<RTGuiMenu>;
           void PopGuiMenu();
           auto GetGuiMenu( size_t idx)const->std::shared_ptr<RTGuiMenu>;
           void DrawFrame();
          ~RTGuiMainMenuBar()noexcept;
    private:
        friend class RTGui;
        std::vector<std::shared_ptr<RTGuiMenu>> m_GuiMenus;
    };
    class RTGuiMenuBar
    {
    public:
        RTGuiMenuBar()noexcept;
        auto AddGuiMenu(const std::string& name)->std::shared_ptr<RTGuiMenu>;
        void PopGuiMenu();
        auto GetGuiMenu(size_t idx)const->std::shared_ptr<RTGuiMenu>;
        void DrawFrame();
        ~RTGuiMenuBar()noexcept;
    private:
        friend class RTGuiWindow;
        std::vector<std::shared_ptr< RTGuiMenu>> m_GuiMenus;
    };
    class RTGuiMenu {
    private:
         using RTGuiMenuItemOrMenuPtr = std::variant < std::shared_ptr<RTGuiMenuItem>, std::shared_ptr<RTGuiMenu>>;
    public:
        RTGuiMenu(const std::string& name, bool isEnable = true)noexcept;
        void SetEnable(bool isEnable)noexcept;
        void SetGuiMenuItem(std::shared_ptr<RTGuiMenuItem> guiMenuItem);
        auto AddGuiMenu(const std::string& name)->std::shared_ptr<RTGuiMenu>;
        void PopGuiChild();
        auto GetGuiGuiMenuItem(size_t idx)const ->std::shared_ptr<RTGuiMenuItem>;
        auto GetGuiMenu(size_t idx)const->std::shared_ptr<RTGuiMenu>;
        void DrawFrame();
        ~RTGuiMenu()noexcept;
    private:
        friend class RTGuiMenuBar;
        friend class RTGuiMainMenuBar;
        std::string                         m_Name;
        bool                                m_Enable;
        std::vector<RTGuiMenuItemOrMenuPtr> m_GuiChilds = {};
    };
    class RTGuiMenuItem
    {
    public:
        using ClickCallback = void(*)(RTGuiMenuItem*);
        RTGuiMenuItem(const std::string& name, bool isEnable = true)noexcept;
        void SetEnable(bool isEnable)noexcept;
        void SetUserPointer(void* data)noexcept;
        auto GetUserPointer()const noexcept -> void*;
        void SetClickCallback(ClickCallback callback)noexcept;
        void DrawFrame();
        void AddSubObject(const std::shared_ptr<RTGuiSubObject>& subobject);
        virtual void OnClick();
        virtual ~RTGuiMenuItem()noexcept;
    private:
        static void DefaultClickCallback(RTGuiMenuItem*) {}
        std::string                         m_Name;
        bool                                m_Enable;
        void* m_UserPointer = nullptr;
        ClickCallback m_ClickCallback = DefaultClickCallback;
        std::vector<std::shared_ptr<RTGuiSubObject>> m_SubObjects = {};

    };
    class RTGuiSubObject {
    public:
        RTGuiSubObject()noexcept {}
        virtual void DrawFrame() = 0;
        virtual ~RTGuiSubObject()noexcept {}
    };

}
#endif