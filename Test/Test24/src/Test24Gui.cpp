#include <Test24Gui.h>
#include <Test24Config.h>
#include <filesystem>
class ObjModelConfigGuiWindow : public test::RTGuiWindow
{
public:
    ObjModelConfigGuiWindow(const std::shared_ptr<test::RTFramebuffer>& framebuffer_, const std::shared_ptr<test::RTObjModelAssetManager>& objModelManager_, std::unordered_set<std::string>& launchTracerSet_, std::string& objModelname) :
        test::RTGuiWindow("ObjModelConfig", ImGuiWindowFlags_MenuBar),
        framebuffer{framebuffer_},
        objModelManager{ objModelManager_ },
        launchTracerSet{ launchTracerSet_ }, 
        curObjModelName{objModelname}{
        objFileDialog = std::make_unique<test::RTGuiFileDialog>("Choose ObjFile", ".obj", TEST_TEST24_DATA_PATH);
        objFileDialog->SetUserPointer(this);
        objFileDialog->SetOkCallback([](test::RTGuiFileDialog* fileDialog) {
            auto this_ptr = reinterpret_cast<ObjModelConfigGuiWindow*>(fileDialog->GetUserPointer());
            this_ptr->m_LoadFilePath = fileDialog->GetFilePathName();
            std::cout << this_ptr->m_LoadFilePath << std::endl;
        });
    }
    virtual void DrawGui()override {
        if (!objModelManager) {
            return;
        }
        GLuint pTex = framebuffer->GetComponent<test::RTGLTextureFBComponent<uchar4>>("PTexture")->GetHandle().getID();
        ImGui::Image(((void*)((intptr_t)pTex)), ImVec2(framebuffer->GetWidth()/4, framebuffer->GetHeight()/4),ImVec2(0,1),ImVec2(1,0));
        {
            size_t i = 0;
            int v_selected = 0;
            std::string newObjModelName = curObjModelName;
            for (auto& [name, asset] : objModelManager->GetAssets())
            {
                std::string removeLabel = "Remove###" + std::to_string(i);
                std::string selectLabel = "Select###" + std::to_string(i);
                std::string   openLabel =   "Open###" + std::to_string(i);
                std::string  closeLabel =  "Close###" + std::to_string(i);
                if (ImGui::RadioButton(name.c_str(),name==curObjModelName)) {
                    newObjModelName = name;
                }
                ImGui::SameLine();
                if (ImGui::Button(removeLabel.c_str())) {
                    m_RemoveNameSet.insert(name);
                }
                ImGui::SameLine();
                if (ImGui::Button( openLabel.c_str())) {
                    
                }
                ImGui::SameLine();
                if (ImGui::Button(closeLabel.c_str())) {

                }
                ImGui::NewLine();
                ++i;
            }
            curObjModelName = newObjModelName;
        }
        
        if (ImGui::Button("Add")) {
            if (!objFileDialog->IsOpen()) {
                objFileDialog->Open();
            }
        }
        objFileDialog->DrawFrame();

        if (!m_LoadFilePath.empty()) {
            std::filesystem::path filename = std::filesystem::path(m_LoadFilePath).filename();
            objModelManager->LoadAsset(filename.string(), m_LoadFilePath);
            curObjModelName = filename.string();
            std::cout << "End!\n";
        }
        for (auto& name : m_RemoveNameSet) {
            objModelManager->FreeAsset(name);
        }
        if (!objModelManager->HasAsset(curObjModelName)&&!objModelManager->GetAssets().empty()) {
            curObjModelName = objModelManager->GetAssets().begin()->first;
        }
        m_RemoveNameSet.clear();
        m_LoadFilePath = "";
        launchTracerSet.insert("TestGL");
    }
    virtual ~ObjModelConfigGuiWindow()noexcept {}
private:
    std::string& curObjModelName;
    std::shared_ptr<test::RTFramebuffer>          framebuffer;
    std::shared_ptr<test::RTObjModelAssetManager> objModelManager;
    std::unique_ptr<test::RTGuiFileDialog>        objFileDialog;
    std::unordered_set<std::string>&              launchTracerSet;
public:
    std::unordered_set<std::string> m_RemoveNameSet;
    std::string m_LoadFilePath;
};
class MainTraceConfigGuiWindow : public test::RTGuiWindow
{
public:
    explicit MainTraceConfigGuiWindow(const std::vector<std::string>& tracePublicNames, std::string& traceName)noexcept :
        test::RTGuiWindow("MainTraceConfig", ImGuiWindowFlags_MenuBar), curTraceName{ traceName }, curTraceIdx{ 0 }, traceNames{ tracePublicNames }, isFirst{ true }{}
    virtual void DrawGui()override {
        if (isFirst) {
            bool isFound = false;
            for (auto i = 0; i < traceNames.size(); ++i)
            {
                if (traceNames[i] == curTraceName) {
                    curTraceIdx = i;
                    isFound = true;
                    break;
                }
            }
            if (!isFound) {
                return;
            }
            isFirst = false;
        }
        int val = curTraceIdx;
        for (auto i = 0; i < traceNames.size(); ++i)
        {
            if (ImGui::RadioButton(traceNames[i].c_str(), &val, i)) {
                curTraceIdx = i;
            }
            if (curTraceIdx == i) {
                ImGui::SameLine();
                if (ImGui::Button("Open")) {

                }
                ImGui::SameLine();
                if (ImGui::Button("Close")) {

                }
            }
            ImGui::NewLine();
        }
        curTraceName = traceNames[curTraceIdx];
    }
    virtual ~MainTraceConfigGuiWindow() noexcept {}
private:
    const std::vector<std::string>& traceNames;
    std::string& curTraceName;
    size_t       curTraceIdx;
    bool isFirst;
};
class MainFrameConfigGuiWindow : public test::RTGuiWindow
{
public:
    explicit MainFrameConfigGuiWindow(const std::vector<std::string>& framePublicNames, std::string& frameName)noexcept :
        test::RTGuiWindow("MainFrameConfig", ImGuiWindowFlags_MenuBar), curFrameName{ frameName }, curFrameIdx{ 0 }, frameNames{ framePublicNames }, isFirst{ true }{}
    virtual void DrawGui()override {
        if (isFirst) {
            bool isFound = false;
            for (auto i = 0; i < frameNames.size(); ++i)
            {
                if (frameNames[i] == curFrameName) {
                    curFrameIdx = i;
                    isFound = true;
                    break;
                }
            }
            if (!isFound) {
                return;
            }
            isFirst = false;
        }
        int val = curFrameIdx;
        for (auto i = 0; i < frameNames.size(); ++i)
        {
            if (ImGui::RadioButton(frameNames[i].c_str(), &val, i)) {
                curFrameIdx = i;
            }
            if (i % 4 == 3) {
                ImGui::NewLine();
            }
            else {
                ImGui::SameLine();
            }
        }
        curFrameName = frameNames[curFrameIdx];
    }
    virtual ~MainFrameConfigGuiWindow() noexcept {}
private:
    const std::vector<std::string>& frameNames;
    std::string& curFrameName;
    size_t       curFrameIdx;
    bool isFirst;
};
class    CameraConfigGuiWindow :public test::RTGuiWindow{
public:

    explicit CameraConfigGuiWindow(const std::shared_ptr<test::RTFramebuffer>& framebuffer_, const std::shared_ptr<rtlib::ext::CameraController>& cameraController_,  bool& updateCamera_)noexcept :
        test::RTGuiWindow("CameraConfig", ImGuiWindowFlags_MenuBar), cameraController{ cameraController_ }, updateCamera{ updateCamera_ }, framebuffer{framebuffer_}{}
    virtual void DrawGui()override {
        auto camera = cameraController->GetCamera((float)framebuffer->GetWidth() / (float)framebuffer->GetHeight());
        auto eye = camera.getEye();
        auto atv = camera.getLookAt();
        auto vup = camera.getVup();
        {
            float arr_eye [3]= { eye.x,eye.y,eye.z };
            if (ImGui::InputFloat3("Eye", arr_eye)) {
                camera.setEye(make_float3(arr_eye[0],arr_eye[1],arr_eye[2]));
                updateCamera = true;
            }
            float arr_atv[3] = { atv.x,atv.y,atv.z };
            if (ImGui::InputFloat3("At", arr_atv)) {
                camera.setLookAt(make_float3(arr_atv[0], arr_atv[1], arr_atv[2]));
                updateCamera = true;
            }
            float arr_vup[3] = { vup.x,vup.y,vup.z };
            if (ImGui::InputFloat3("vup", arr_vup)) {
                camera.setLookAt(make_float3(arr_vup[0], arr_vup[1], arr_vup[2]));
                updateCamera = true;
            }
            float zoom = cameraController->GetZoom();
            if (ImGui::InputFloat("zoom", &zoom)) {
                cameraController->SetZoom(zoom);
                updateCamera = true;
            }
        }
        if (updateCamera) {
            cameraController->SetCamera(camera);
        }
    }
    virtual ~CameraConfigGuiWindow() noexcept {}
private:
    std::shared_ptr<test::RTFramebuffer> framebuffer;
    std::shared_ptr<rtlib::ext::CameraController> cameraController;
    bool& updateCamera;
};
void  Test24GuiDelegate::Initialize()
{
    m_Gui->Initialize();
    // MainMenuBar
    auto mainMenuBar = m_Gui->AddGuiMainMenuBar();
    auto fileMenu = mainMenuBar->AddGuiMenu("File");
    {
        auto mdlMenu = fileMenu->AddGuiMenu("Model");
        auto fileDialog = std::make_shared<test::RTGuiFileDialog>("Choose File", ".obj", TEST_TEST24_DATA_PATH"\\");
        auto openMenuItem = std::make_shared < test::RTGuiMenuItem>("Open");
        openMenuItem->SetUserPointer(fileDialog.get());
        openMenuItem->SetClickCallback([](test::RTGuiMenuItem* item) {
            auto fd = reinterpret_cast<test::RTGuiFileDialog*>(item->GetUserPointer());
            if (fd && !fd->IsOpen()) {
                fd->Open();
            }
            });
        auto clseMenuItem = std::make_shared < test::RTGuiMenuItem>("Close");
        clseMenuItem->SetUserPointer(fileDialog.get());
        clseMenuItem->SetClickCallback([](test::RTGuiMenuItem* item) {
            auto fd = reinterpret_cast<test::RTGuiFileDialog*>(item->GetUserPointer());
            if (fd && fd->IsOpen()) {
                fd->Close();
            }
            });
        mdlMenu->SetGuiMenuItem(openMenuItem);
        mdlMenu->SetGuiMenuItem(clseMenuItem);
        m_Gui->SetGuiFileDialog(fileDialog);
    }
    // ConfigMenu
    auto cnfgMenu = mainMenuBar->AddGuiMenu("Config");
    {
        {
            auto fmbfItem = cnfgMenu->AddGuiMenu("Frame");
            // MainFrameConfig
            auto mainFmCnfgWindow = std::make_shared<MainFrameConfigGuiWindow>(m_FramePublicNames, m_CurMainFrameName);
            mainFmCnfgWindow->SetActive(false);   //Default: Invisible
            m_Gui->SetGuiWindow(mainFmCnfgWindow);
            fmbfItem->SetGuiMenuItem(std::make_shared<test::RTGuiOpenWindowMenuItem>(mainFmCnfgWindow));
            fmbfItem->SetGuiMenuItem(std::make_shared<test::RTGuiCloseWindowMenuItem>(mainFmCnfgWindow));
        }
        {
            auto trcrItem = cnfgMenu->AddGuiMenu("Tracer");
            auto mainTcCnfgWindow = std::make_shared<MainTraceConfigGuiWindow>(m_TracePublicNames,m_CurMainTraceName);
            mainTcCnfgWindow->SetActive(false);   //Default: Invisible
            m_Gui->SetGuiWindow(mainTcCnfgWindow);
            trcrItem->SetGuiMenuItem(std::make_shared<test::RTGuiOpenWindowMenuItem>(mainTcCnfgWindow));
            trcrItem->SetGuiMenuItem(std::make_shared<test::RTGuiCloseWindowMenuItem>(mainTcCnfgWindow));
        }
        {
            auto cmrItem = cnfgMenu->AddGuiMenu("Camera");
            auto cmrCnfgWindow = std::make_shared<CameraConfigGuiWindow>(m_Framebuffer, m_CameraController, m_UpdateCamera);
            cmrCnfgWindow->SetActive(false);   //Default: Invisible
            m_Gui->SetGuiWindow(cmrCnfgWindow);
            cmrItem->SetGuiMenuItem(std::make_shared<test::RTGuiOpenWindowMenuItem>(cmrCnfgWindow));
            cmrItem->SetGuiMenuItem(std::make_shared<test::RTGuiCloseWindowMenuItem>(cmrCnfgWindow));
        }
        {
            auto mdlItem = cnfgMenu->AddGuiMenu("Model");
            
            auto mdlCnfgWindow = std::make_shared<ObjModelConfigGuiWindow>(m_Framebuffer, m_ObjModelAssetManager, m_LaunchTracerSet, m_CurObjModelName);
            mdlCnfgWindow->SetActive(false);   //Default: Invisible
            m_Gui->SetGuiWindow(mdlCnfgWindow);
            mdlItem->SetGuiMenuItem(std::make_shared<test::RTGuiOpenWindowMenuItem>(mdlCnfgWindow));
            mdlItem->SetGuiMenuItem(std::make_shared<test::RTGuiCloseWindowMenuItem>(mdlCnfgWindow));
        }
    }
}
void  Test24GuiDelegate::CleanUp()
{
    m_Gui->CleanUp();
    m_Gui.reset();
    m_Window = nullptr;
}
auto  Test24GuiDelegate::GetGui() const -> std::shared_ptr<test::RTGui>
{
	return std::shared_ptr<test::RTGui>();
}
Test24GuiDelegate::~Test24GuiDelegate() {
    if (m_Window) {
        CleanUp();
    }
}
void  Test24GuiDelegate::DrawFrame()
{
    m_Gui->DrawFrame();
}
