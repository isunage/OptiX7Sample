#include <nlohmann/json.hpp>
#include <Test24Gui.h>
#include <Test24Config.h>
#include <Test24DebugOPXConfig.h>
#include <Test24ReSTIROPXConfig.h>
#include <Test24PathOPXConfig.h>
#include <Test24NEEOPXConfig.h>
#include <Test24WRSOPXConfig.h>
#include <Test24GuideReSTIROPXConfig.h>
#include <Test24GuidePathOPXConfig.h>
#include <Test24GuideNEEOPXConfig.h>
#include <Test24GuideWRSOPXConfig.h>
#include <Test24GuideWRS2OPXConfig.h>
#include <filesystem>
#include <chrono>
#include <random>
using namespace std::string_literals;
class       ObjModelConfigGuiWindow : public test::RTGuiWindow
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
class      MainTraceConfigGuiWindow : public test::RTGuiWindow
{
public:
    explicit MainTraceConfigGuiWindow(const std::vector<std::string>& tracePublicNames, std::string& traceName,unsigned int& maxTraceDepth_, unsigned int& samplePerLaunch_, unsigned int& eventFlags_)noexcept :
        test::RTGuiWindow("MainTraceConfig", ImGuiWindowFlags_MenuBar), curTraceName{ traceName }, curTraceIdx{ 0 }, traceNames{ tracePublicNames }, isFirst{ true }, eventFlags{ eventFlags_ }, maxTraceDepth{ maxTraceDepth_ }, samplePerLaunch{ samplePerLaunch_ }, traceWindows{}{}
    virtual void DrawGui()override {
        if (isFirst||(traceNames[curTraceIdx]!=curTraceName)) {
            bool isFound = false;
            for (auto i  = 0; i < traceNames.size(); ++i)
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
        {
            int val = maxTraceDepth;
            if (ImGui::SliderInt("MaxTraceDepth", &val, 1, 100)) {
                maxTraceDepth = val;
                eventFlags |= TEST24_EVENT_FLAG_CHANGE_TRACE;
            }
        }
        {
            int val = curTraceIdx;
            for (auto i = 0; i < traceNames.size(); ++i)
            {
                if (ImGui::RadioButton(traceNames[i].c_str(), &val, i)) {}
                if (val == i) {
                    ImGui::SameLine();
                    if (ImGui::Button("Open")) {
                        if (traceWindows.count(traceNames[i]) > 0) {
                            traceWindows.at(traceNames[i])->SetActive(true);
                        }
                    }
                    ImGui::SameLine();
                    if (ImGui::Button("Close")) {
                        if (traceWindows.count(traceNames[i]) > 0) {
                            traceWindows.at(traceNames[i])->SetActive(false);
                        }
                    }
                }
                ImGui::NewLine();
            }
            if (val != curTraceIdx) {
                curTraceIdx = val;
                eventFlags |= TEST24_EVENT_FLAG_CHANGE_TRACE;
            }
            curTraceName = traceNames[curTraceIdx];
        }
    }
    bool AddTracerWindow(const std::string& traceName, const std::shared_ptr<test::RTGuiWindow>& traceWindow)noexcept
    {
        if (!traceWindow) {
            return false;
        }
        if (std::find(std::begin(traceNames), std::end(traceNames), traceName) == std::end(traceNames)) {
            return false;
        }
        traceWindows[traceName] = traceWindow;
        return true;
    }
    virtual ~MainTraceConfigGuiWindow() noexcept {}
private:
    const std::vector<std::string>& traceNames;
    std::unordered_map<std::string, std::shared_ptr<test::RTGuiWindow>> traceWindows;
    std::string&                    curTraceName;
    unsigned int&                   maxTraceDepth;
    unsigned int&                   samplePerLaunch;
    size_t                          curTraceIdx;
    unsigned int&                   eventFlags;
    bool                            isFirst;
};
class      MainFrameConfigGuiWindow : public test::RTGuiWindow
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
    bool         isFirst;
};
class         CameraConfigGuiWindow :public test::RTGuiWindow{
public:

    explicit CameraConfigGuiWindow(GLFWwindow* window_, const std::shared_ptr<test::RTFramebuffer>& framebuffer_, const std::shared_ptr<rtlib::ext::CameraController>& cameraController_, unsigned int& eventFlags_)noexcept :
        test::RTGuiWindow("CameraConfig", ImGuiWindowFlags_MenuBar), m_Window{ window_ }, cameraController{ cameraController_ }, eventFlags{ eventFlags_ }, framebuffer{ framebuffer_ }, m_FilePath{ "" }{
        glfwGetWindowSize(m_Window, &m_Width, &m_Height);
    }
    virtual void DrawGui()override {
        auto camera = cameraController->GetCamera((float)framebuffer->GetWidth() / (float)framebuffer->GetHeight());
        auto eye    = camera.getEye();
        auto atv    = camera.getLookAt();
        auto vup    = camera.getVup();
        float arr_eye[3] = { eye.x,eye.y,eye.z };
        float arr_atv[3] = { atv.x,atv.y,atv.z };
        float arr_vup[3] = { vup.x,vup.y,vup.z };
        float zoom = cameraController->GetZoom();
        auto speed = cameraController->GetMovementSpeed();
        auto sense = cameraController->GetMouseSensitivity();
        if(!HasEvent(TEST24_EVENT_FLAG_BIT_LOCK)){
            if (ImGui::InputInt("Width: ", &m_Width)) {
            }
            if (ImGui::InputInt("Height: ",&m_Height)) {
            }
            if (ImGui::Button("Resize")) {
                SetEvent(TEST24_EVENT_FLAG_UPDATE_CAMERA | TEST24_EVENT_FLAG_BIT_RESIZE_FRAME);
            }
            if (ImGui::InputFloat3("Eye", arr_eye)) {
                camera.setEye(make_float3(arr_eye[0],arr_eye[1],arr_eye[2]));
                SetEvent(TEST24_EVENT_FLAG_UPDATE_CAMERA);
            }
            if (ImGui::InputFloat3("At", arr_atv)) {
                camera.setLookAt(make_float3(arr_atv[0], arr_atv[1], arr_atv[2]));
                SetEvent(TEST24_EVENT_FLAG_UPDATE_CAMERA);
            }
            if (ImGui::InputFloat3("Vup", arr_vup)) {
                camera.setLookAt(make_float3(arr_vup[0], arr_vup[1], arr_vup[2]));
                SetEvent(TEST24_EVENT_FLAG_UPDATE_CAMERA);
            }
            if (ImGui::InputFloat("Zoom", &zoom)) {
                cameraController->SetZoom(zoom);
                SetEvent(TEST24_EVENT_FLAG_UPDATE_CAMERA);
            }
            if (ImGui::InputFloat("Sensitivity", &sense)) {
                cameraController->SetMouseSensitivity(sense);
            }
            if (ImGui::InputFloat("Speed", &speed)) {
                cameraController->SetMovementSpeed(speed);
            }
            if (HasEvent(TEST24_EVENT_FLAG_BIT_RESIZE_FRAME)) {
                glfwSetWindowSize(m_Window, m_Width, m_Height);
            }
            if (HasEvent(TEST24_EVENT_FLAG_BIT_UPDATE_CAMERA)) {
                cameraController->SetCamera(camera);
            }
        }
        else {
            ImGui::Text(" Eye: (%f, %f, %f)", arr_eye[0], arr_eye[1], arr_eye[2]);
            ImGui::Text("  At: (%f, %f, %f)", arr_atv[0], arr_atv[1], arr_atv[2]);
            ImGui::Text(" Vup: (%f, %f, %f)", arr_atv[0], arr_atv[1], arr_atv[2]);
            ImGui::Text("Zoom:  %f"         , zoom);
            ImGui::Text("Sensitivity:  %f"  , sense);
            ImGui::Text("Speed:  %f"        , speed);
        }

        char saveFilePath[256];
        std::strncpy(saveFilePath, m_FilePath.c_str(), sizeof(saveFilePath) - 1);
        if (ImGui::InputText("SaveFilepath", saveFilePath, sizeof(saveFilePath))) {
            m_FilePath = std::string(saveFilePath);
        }

        if (ImGui::Button("Save")) {
            std::string filePathRoot = std::string(TEST_TEST24_RSLT_PATH) + "/" + saveFilePath;
            if (!std::filesystem::exists(filePathRoot)) {
                std::filesystem::create_directories(filePathRoot);
            }
            std::string filePath     = filePathRoot + "pinhole_camera.json";
            nlohmann::json cameraJson;
            auto camEye           = camera.getEye();
            cameraJson[ "Eye"]   = std::array<float, 3>{arr_eye[0], arr_eye[1], arr_eye[2]};
            cameraJson[ "At" ]   = std::array<float, 3>{arr_atv[0], arr_atv[1], arr_atv[2]};
            cameraJson[ "Vup"]   = std::array<float, 3>{arr_vup[0], arr_vup[1], arr_vup[2]};
            cameraJson["FovY"]   = camera.getFovY();
            cameraJson["Width"]  = framebuffer->GetWidth();
            cameraJson["Height"] = framebuffer->GetHeight();
            cameraJson["Zoom" ]  = zoom;
            cameraJson["Speed"]  = speed;
            cameraJson["Sense"]  = sense;
            auto jsonStr = cameraJson.dump();
            std::ofstream jsonFile(filePath);
            jsonFile << jsonStr;
            jsonFile.close();
        }
        if (!HasEvent(TEST24_EVENT_FLAG_BIT_LOCK)) {
            ImGui::SameLine();
            if (ImGui::Button("Load")) {
                std::string filePathRoot = std::string(TEST_TEST24_RSLT_PATH) + "/" + saveFilePath;
                std::string filePath = filePathRoot + "pinhole_camera.json";
                if (std::filesystem::exists(filePath)) {
                    
                    std::ifstream jsonFile(filePath);
                    std::string jsonStr = std::string((std::istreambuf_iterator<char>(jsonFile)), (std::istreambuf_iterator<char>()));
                    jsonFile.close();

                    auto cameraJson   = nlohmann::json::parse(jsonStr);
                    auto camEyeArr    = cameraJson["Eye"   ].get<std::vector<float>>();
                    auto camAtArr     = cameraJson["At"    ].get<std::vector<float>>();
                    auto camVupArr    = cameraJson["Vup"   ].get<std::vector<float>>();
                    auto camFovY      = cameraJson["FovY"  ].get<float>();
                    auto camWidth     = cameraJson["Width" ].get<int>();
                    auto camHeight    = cameraJson["Height"].get<int>();
                    auto camZoom      = cameraJson["Zoom"  ].get<float>();
                    auto camSpeed     = cameraJson["Speed" ].get<float>();
                    auto camSense     = cameraJson["Sense" ].get<float>();

                    glfwSetWindowSize(m_Window, camWidth, camHeight);

                    cameraController->SetCamera(rtlib::ext::Camera(
                        make_float3(camEyeArr[0], camEyeArr[1],camEyeArr[2]),
                        make_float3( camAtArr[0],  camAtArr[1], camAtArr[2]),
                        make_float3(camVupArr[0], camVupArr[1],camVupArr[2]),
                        camFovY, static_cast<float>(camWidth)/static_cast<float>(camHeight)
                    ));
                    cameraController->SetZoom(camZoom);
                    cameraController->SetMovementSpeed(camSpeed);
                    cameraController->SetMouseSensitivity(camSense);

                    SetEvent(TEST24_EVENT_FLAG_RESIZE_FRAME);
                }
            }
        }

    }
    virtual ~CameraConfigGuiWindow() noexcept {}
private:
    bool HasEvent(unsigned int eventFlag)const noexcept
    {
        return (eventFlags & eventFlag) == eventFlag;
    }
    auto GetEvent()const noexcept -> unsigned int {
        return eventFlags;
    }
    void SetEvent(unsigned int eventFlag)noexcept
    {
        eventFlags |= eventFlag;
    }
    GLFWwindow* m_Window;
    std::string m_FilePath;
    std::shared_ptr<test::RTFramebuffer> framebuffer;
    std::shared_ptr<rtlib::ext::CameraController> cameraController;
    int m_Width;
    int m_Height;
    unsigned int& eventFlags;
};
class          LightConfigGuiWindow: public test::RTGuiWindow {
public:
    LightConfigGuiWindow(
        float3& bgLightColor,
        unsigned int& eventFlags)noexcept
        :test::RTGuiWindow("LightConfig", ImGuiWindowFlags_MenuBar), m_BgLightColor{ bgLightColor }, m_EventFlags{ eventFlags } {}
    virtual void DrawGui()override {
        {
            float arr_bg_light_color[3] = { m_BgLightColor.x,m_BgLightColor.y,m_BgLightColor.z };
            if (ImGui::InputFloat3("BackGroundLight", arr_bg_light_color)) {
                m_BgLightColor = make_float3(arr_bg_light_color[0], arr_bg_light_color[1], arr_bg_light_color[2]);
                SetEvent(TEST24_EVENT_FLAG_UPDATE_LIGHT);
            }
        }
    }
    virtual ~LightConfigGuiWindow()noexcept {}
private:
    //bool HasEvent(unsigned int eventFlag)const noexcept
    //{
    //    return (m_EventFlags & eventFlag) == eventFlag;
    //}
    //auto GetEvent()const noexcept -> unsigned int {
    //    return m_EventFlags;
    //}
    void SetEvent(unsigned int eventFlag)noexcept
    {
        m_EventFlags |= eventFlag;
    }
private:
    float3& m_BgLightColor;
    unsigned int&   m_EventFlags;
};
class          InputConfigGuiWindow : public test::RTGuiWindow {
public:
    InputConfigGuiWindow(
        const std::array<float, 2>& curCursorPos,
        const std::array<float, 2>& delCursorPos,
        const std::array<float, 2>& scrollOffsets,
        const float& curFrameTime,
        const float& delFrameTime,
        const unsigned int& samplePerAll,
        const unsigned int& samplePerLaunch)noexcept
        :test::RTGuiWindow("InputConfig", ImGuiWindowFlags_MenuBar),
        m_CurCursorPos{    curCursorPos },
        m_DelCursorPos{    delCursorPos },
        m_ScrollOffsets{   scrollOffsets },
        m_CurFrameTime{    curFrameTime },
        m_DelFrameTime{    delFrameTime },
        m_SamplePerAll{    samplePerAll },
        m_SamplePerLaunch{ samplePerLaunch }{}
    virtual void DrawGui()override {
        {
            ImGui::Text("SamplePerAll   : %d spp", m_SamplePerAll);
            ImGui::Text("SamplePerLaunch: %d spp", m_SamplePerLaunch);
            ImGui::Text("CurFrameRate :   %f fps", 1.0f/m_DelFrameTime);
            ImGui::Text("CurFrameTime :   %f sec", m_CurFrameTime);
            ImGui::Text("CurCursorPos : (%f, %f)", m_CurCursorPos[0],  m_CurCursorPos[1]);
            ImGui::Text("DelCursorPos : (%f, %f)", m_DelCursorPos[0],  m_DelCursorPos[1]);
            ImGui::Text("ScrollOffsets: (%f, %f)", m_ScrollOffsets[0], m_ScrollOffsets[1]);
        }
    }
    virtual ~InputConfigGuiWindow()noexcept {}
private:
    const unsigned int&         m_SamplePerAll;
    const unsigned int&         m_SamplePerLaunch;
    const std::array<float, 2>& m_CurCursorPos;
    const std::array<float, 2>& m_DelCursorPos;
    const std::array<float, 2>& m_ScrollOffsets;
    const float& m_CurFrameTime;
    const float& m_DelFrameTime;
};
class          TraceConfigGuiWindow:public test::RTGuiWindow {
private:
    using TracerVariableMap = std::unordered_map < std::string, std::shared_ptr<rtlib::ext::VariableMap>>;
public:
    TraceConfigGuiWindow(const std::string& traceName_, const TracerVariableMap& tracerVariables_, const std::shared_ptr<test::RTFramebuffer>& framebuffer_, unsigned int& eventFlags_)noexcept :test::RTGuiWindow(traceName_, ImGuiWindowFlags_MenuBar), traceName{ traceName_ }, tracerVariables{ tracerVariables_ }, framebuffer{ framebuffer_ }, eventFlags{ eventFlags_ }{}
    virtual void DrawGui()override {
        ImGui::Text("TraceName: %s", traceName.c_str());
    }
    virtual ~TraceConfigGuiWindow()noexcept {}
    auto GetName()const noexcept -> std::string {
        return traceName;
    }
    auto GetVariable()const noexcept -> std::shared_ptr<rtlib::ext::VariableMap> {
        if (tracerVariables.count(traceName) > 0) {
            return tracerVariables.at(traceName);
        }
        else {
            return nullptr;
        }
    }
protected:
    auto GetFramebuffer()const noexcept ->  std::shared_ptr < test::RTFramebuffer>
    {
        return framebuffer;
    }
    bool HasEvent(unsigned int eventFlag)const noexcept
    {
        return (eventFlags & eventFlag) == eventFlag;
    }
    auto GetEvent()const noexcept -> unsigned int {
        return eventFlags;
    }
    void SetEvent(unsigned int eventFlag)noexcept
    {
        eventFlags |= eventFlag;
    }
private:
    std::shared_ptr < test::RTFramebuffer> framebuffer;
    std::string                              traceName;
    const TracerVariableMap&           tracerVariables;
    unsigned int&                           eventFlags;
};
class      PathTraceConfigGuiWindow : public TraceConfigGuiWindow
{
private:
    using TracerVariableMap = std::unordered_map < std::string, std::shared_ptr<rtlib::ext::VariableMap>>;
public:
    PathTraceConfigGuiWindow(const TracerVariableMap& tracerVariables_, const std::shared_ptr<test::RTFramebuffer>& framebuffer_, unsigned int& eventFlags_)noexcept : TraceConfigGuiWindow("PathOPX", tracerVariables_, framebuffer_, eventFlags_) {}
    virtual void DrawGui()override {
        auto variable = GetVariable();
        if (variable) {
            int sampleForBudget = variable->GetUInt32("SampleForBudget");
            int samplePerLaunch = variable->GetUInt32("SamplePerLaunch");
            if (!HasEvent(TEST24_EVENT_FLAG_BIT_LOCK)) {
                if (ImGui::InputInt("SampleForBudget", &sampleForBudget)) {
                    variable->SetUInt32("SampleForBudget", sampleForBudget);
                    SetEvent(TEST24_EVENT_FLAG_CHANGE_TRACE);
                }
                if (ImGui::SliderInt("SamplePerLaunch", &samplePerLaunch, 1, 100)) {
                    variable->SetUInt32("SamplePerLaunch", samplePerLaunch);
                    SetEvent(TEST24_EVENT_FLAG_CHANGE_TRACE);
                }
            }
            else {
                ImGui::Text("SamplePerBudget: %d", sampleForBudget);
                ImGui::Text("SamplePerLaunch: %d", samplePerLaunch);
            }

            int samplePerAll = variable->GetUInt32("SamplePerAll");
            ImGui::Text("SamplePerAll: %d", samplePerAll);

            char saveFilePath[256];
            std::strncpy(saveFilePath, m_FilePath.c_str(), sizeof(saveFilePath) - 1);
            if (ImGui::InputText("SaveFilepath", saveFilePath, sizeof(saveFilePath))) {
                m_FilePath = std::string(saveFilePath);
            }
            int samplePerSave = m_SamplePerSave;
            if (ImGui::InputInt("SamplePerSave", &samplePerSave)) {
                m_SamplePerSave = std::max(1, samplePerSave);
            }

            bool isLaunched = variable->GetBool("Launched");
            if (!isLaunched) {
                if (ImGui::Button("Started")) {
                    variable->SetBool("Started", true);
                    SetEvent(TEST24_EVENT_FLAG_CHANGE_TRACE);
                }
                ImGui::SameLine();
            }
            bool pushSave = false;
            if (ImGui::Button("Save")) {
                pushSave = true;
            }
            bool saveIter = isLaunched && samplePerAll && ((samplePerAll % m_SamplePerSave)==0);
            if (pushSave || saveIter) {
                std::cout << "GuidPathTrace Save: " << samplePerAll << std::endl;
                std::string filePathBase = std::string(TEST_TEST24_PATH_OPX_RSLT_PATH) + "/" + saveFilePath + "result_path_" + std::to_string(samplePerAll);
                std::string filePathPng = filePathBase + ".png";
                auto rTexture = GetFramebuffer()->GetComponent<test::RTGLTextureFBComponent<uchar4>>("RTexture");
                if (test::SavePngImgFromGL(filePathPng.c_str(), rTexture->GetHandle())) {
                    std::cout << "Save\n";
                }
                std::string filePathExr = filePathBase + ".exr";
                auto rAccum = GetFramebuffer()->GetComponent<test::RTCUDABufferFBComponent<float3>>("RAccum");
                if (test::SaveExrImgFromCUDA(filePathExr.c_str(), rAccum->GetHandle(), GetFramebuffer()->GetWidth(), GetFramebuffer()->GetHeight(), samplePerAll)) {
                    std::cout << "Save\n";
                }
            }
        }
    }
    virtual ~PathTraceConfigGuiWindow()noexcept {}
private:
    unsigned int m_SamplePerSave = 1000;
    std::string  m_FilePath;
};
class       NEETraceConfigGuiWindow : public TraceConfigGuiWindow
{
private:
    using TracerVariableMap = std::unordered_map < std::string, std::shared_ptr<rtlib::ext::VariableMap>>;
public:
    NEETraceConfigGuiWindow(const TracerVariableMap& tracerVariables_, const std::shared_ptr<test::RTFramebuffer>& framebuffer_, unsigned int& eventFlags_)noexcept : TraceConfigGuiWindow("NEEOPX", tracerVariables_, framebuffer_, eventFlags_) {}
    virtual void DrawGui()override {
        auto variable = GetVariable();
        if (variable) {
            int sampleForBudget = variable->GetUInt32("SampleForBudget");
            int samplePerLaunch = variable->GetUInt32("SamplePerLaunch");
            if (!HasEvent(TEST24_EVENT_FLAG_BIT_LOCK)) {
                if (ImGui::InputInt("SampleForBudget", &sampleForBudget)) {
                    variable->SetUInt32("SampleForBudget", sampleForBudget);
                    SetEvent(TEST24_EVENT_FLAG_CHANGE_TRACE);
                }
                if (ImGui::SliderInt("SamplePerLaunch", &samplePerLaunch, 1, 100)) {
                    variable->SetUInt32("SamplePerLaunch", samplePerLaunch);
                    SetEvent(TEST24_EVENT_FLAG_CHANGE_TRACE);
                }
            }
            else {
                ImGui::Text("SamplePerBudget: %d", sampleForBudget);
                ImGui::Text("SamplePerLaunch: %d", samplePerLaunch);
            }

            int samplePerAll = variable->GetUInt32("SamplePerAll");
            ImGui::Text("SamplePerAll: %d", samplePerAll);

            char saveFilePath[256];
            std::strncpy(saveFilePath, m_FilePath.c_str(), sizeof(saveFilePath) - 1);
            if (ImGui::InputText("SaveFilepath", saveFilePath, sizeof(saveFilePath))) {
                m_FilePath = std::string(saveFilePath);
            }
            int samplePerSave = m_SamplePerSave;
            if (ImGui::InputInt("SamplePerSave", &samplePerSave)) {
                m_SamplePerSave = std::max(1, samplePerSave);
            }

            bool isLaunched = variable->GetBool("Launched");
            if (!isLaunched) {
                if (ImGui::Button("Started")) {
                    variable->SetBool("Started", true);
                    SetEvent(TEST24_EVENT_FLAG_CHANGE_TRACE);
                }
                ImGui::SameLine();
            }
            bool pushSave = false;
            if (ImGui::Button("Save")) {
                pushSave = true;
            }
            bool saveIter = isLaunched && samplePerAll && ((samplePerAll % m_SamplePerSave)==0);
            if (pushSave || saveIter) {
                std::string filePathBase = std::string(TEST_TEST24_NEE_OPX_RSLT_PATH) + "/" + saveFilePath + "result_nee_" + std::to_string(samplePerAll);
                std::string filePathPng = filePathBase + ".png";
                auto rTexture = GetFramebuffer()->GetComponent<test::RTGLTextureFBComponent<uchar4>>("RTexture");
                if (test::SavePngImgFromGL(filePathPng.c_str(), rTexture->GetHandle())) {
                    std::cout << "Save\n";
                }
                std::string filePathExr = filePathBase + ".exr";
                auto rAccum = GetFramebuffer()->GetComponent<test::RTCUDABufferFBComponent<float3>>("RAccum");
                if (test::SaveExrImgFromCUDA(filePathExr.c_str(), rAccum->GetHandle(), GetFramebuffer()->GetWidth(), GetFramebuffer()->GetHeight(), samplePerAll)) {
                    std::cout << "Save\n";
                }
            }
        }
    }
    virtual ~NEETraceConfigGuiWindow()noexcept {}
private:
    unsigned int m_SamplePerSave = 1000;
    std::string m_FilePath;
};
class       WRSTraceConfigGuiWindow : public TraceConfigGuiWindow
{
private:
    using TracerVariableMap = std::unordered_map < std::string, std::shared_ptr<rtlib::ext::VariableMap>>;
public:
    WRSTraceConfigGuiWindow(const TracerVariableMap& tracerVariables_, const std::shared_ptr<test::RTFramebuffer>& framebuffer_, unsigned int& eventFlags_)noexcept : TraceConfigGuiWindow("WRSOPX", tracerVariables_, framebuffer_, eventFlags_), m_FilePath{}{}
    virtual void DrawGui()override {
        auto variable = GetVariable();
        if (variable) {
            int sampleForBudget   = variable->GetUInt32("SampleForBudget");
            int samplePerLaunch   = variable->GetUInt32("SamplePerLaunch");
            int numCandidates     = variable->GetUInt32("NumCandidates");

            if (!HasEvent(TEST24_EVENT_FLAG_BIT_LOCK)) {
                if (ImGui::InputInt("SampleForBudget", &sampleForBudget)) {
                    variable->SetUInt32("SampleForBudget", sampleForBudget);
                    SetEvent(TEST24_EVENT_FLAG_CHANGE_TRACE);
                }
                if (ImGui::SliderInt("SamplePerLaunch", &samplePerLaunch, 1, 100)) {
                    variable->SetUInt32("SamplePerLaunch", samplePerLaunch);
                    SetEvent(TEST24_EVENT_FLAG_CHANGE_TRACE);
                }
                if (ImGui::SliderInt("NumCandidates", &numCandidates, 1, 64)) {
                    variable->SetUInt32("NumCandidates", numCandidates);
                    SetEvent(TEST24_EVENT_FLAG_CHANGE_TRACE);
                }
            }
            else {
                ImGui::Text("  SamplePerBudget: %d", sampleForBudget);
                ImGui::Text("  SamplePerLaunch: %d", samplePerLaunch);
                ImGui::Text("    NumCandidates: %d", numCandidates);
            }

            int samplePerAll = variable->GetUInt32("SamplePerAll");

            ImGui::Text("SamplePerAll: %d", samplePerAll);

            char saveFilePath[256];
            std::strncpy(saveFilePath, m_FilePath.c_str(), sizeof(saveFilePath) - 1);
            if (ImGui::InputText("SaveFilepath", saveFilePath, sizeof(saveFilePath))) {
                m_FilePath = std::string(saveFilePath);
            }
            int samplePerSave = m_SamplePerSave;
            if (ImGui::InputInt("SamplePerSave", &samplePerSave)) {
                m_SamplePerSave = std::max(1, samplePerSave);
            }

            bool isLaunched = variable->GetBool("Launched");
            if (!isLaunched) {
                if (ImGui::Button("Started")) {
                    variable->SetBool("Started", true);
                    SetEvent(TEST24_EVENT_FLAG_CHANGE_TRACE);
                }
                ImGui::SameLine();
            }
            bool pushSave = false;
            if (ImGui::Button("Save")) {
                pushSave = true;
            }
            bool saveIter = isLaunched && samplePerAll && ((samplePerAll % m_SamplePerSave)==0);
            if (pushSave || saveIter) {
                std::string filePathBase = std::string(TEST_TEST24_WRS_OPX_RSLT_PATH) + "/" + saveFilePath + "result_wrs_" + std::to_string(samplePerAll);
                std::string filePathPng = filePathBase + ".png";
                auto rTexture = GetFramebuffer()->GetComponent<test::RTGLTextureFBComponent<uchar4>>("RTexture");
                if (test::SavePngImgFromGL(filePathPng.c_str(), rTexture->GetHandle())) {
                    std::cout << "Save\n";
                }
                std::string filePathExr = filePathBase + ".exr";
                auto rAccum = GetFramebuffer()->GetComponent<test::RTCUDABufferFBComponent<float3>>("RAccum");
                if (test::SaveExrImgFromCUDA(filePathExr.c_str(), rAccum->GetHandle(), GetFramebuffer()->GetWidth(), GetFramebuffer()->GetHeight(), samplePerAll)) {
                    std::cout << "Save\n";
                }
            }
        }
    }
    virtual ~WRSTraceConfigGuiWindow()noexcept {}
private:
    unsigned int m_SamplePerSave = 1000;
    std::string  m_FilePath;
};
class    ReSTIRTraceConfigGuiWindow : public TraceConfigGuiWindow
{
private:
    using TracerVariableMap = std::unordered_map < std::string, std::shared_ptr<rtlib::ext::VariableMap>>;
public:
    ReSTIRTraceConfigGuiWindow(const TracerVariableMap& tracerVariables_, const std::shared_ptr<test::RTFramebuffer>& framebuffer_, unsigned int& eventFlags_)noexcept : TraceConfigGuiWindow("ReSTIROPX", tracerVariables_, framebuffer_,eventFlags_) {}
    virtual void DrawGui()override {
        auto variable = GetVariable();
        if (variable) {
            int sampleForBudget = variable->GetUInt32("SampleForBudget");
            int numCandidates   = variable->GetUInt32("NumCandidates");

            if (!HasEvent(TEST24_EVENT_FLAG_BIT_LOCK)) {
                if (ImGui::InputInt("SampleForBudget", &sampleForBudget)) {
                    variable->SetUInt32("SampleForBudget", sampleForBudget);
                    SetEvent(TEST24_EVENT_FLAG_CHANGE_TRACE);
                }
                if (ImGui::SliderInt("NumCandidates", &numCandidates, 1, 64)) {
                    variable->SetUInt32("NumCandidates", numCandidates);
                    SetEvent(TEST24_EVENT_FLAG_CHANGE_TRACE);
                }
            }
            else {
                ImGui::Text("SampleForBudget: %d", sampleForBudget);
                ImGui::Text("  NumCandidates: %d", numCandidates);
            }
            
            bool temporalReuse = variable->GetBool("ReuseTemporal");
            if (!HasEvent(TEST24_EVENT_FLAG_BIT_LOCK)) {
                if (ImGui::Checkbox("Temporal Reuse", &temporalReuse))
                {
                    variable->SetBool("ReuseTemporal", temporalReuse);
                    SetEvent(TEST24_EVENT_FLAG_CHANGE_TRACE);
                }
            }
            else {
                ImGui::Text("Temporal Reuse: %s", temporalReuse ? "ON" : "OFF");
            }
            

            bool  spatialReuse = variable->GetBool("ReuseSpatial");

            int iter   = variable->GetUInt32("IterationSpatial");
            int range  = variable->GetUInt32("RangeSpatial");
            int sample = variable->GetUInt32("SampleSpatial");

            if (!HasEvent(TEST24_EVENT_FLAG_BIT_LOCK)) {
                if (ImGui::Checkbox("Spatial Reuse", &spatialReuse))
                {
                    variable->SetBool("ReuseSpatial", spatialReuse);
                    SetEvent(TEST24_EVENT_FLAG_CHANGE_TRACE);
                }
                if (spatialReuse) {
                    if (ImGui::SliderInt("Iteration", &iter, 1, 10)) {
                        variable->SetUInt32("IterationSpatial", iter);
                        SetEvent(TEST24_EVENT_FLAG_CHANGE_TRACE);
                    }
                    if (ImGui::SliderInt("Range", &range, 1, 100)) {
                        variable->SetUInt32("RangeSpatial", range);
                        SetEvent(TEST24_EVENT_FLAG_CHANGE_TRACE);
                    }
                    if (ImGui::SliderInt("Sample", &sample, 1, 10)) {
                        variable->SetUInt32("SampleSpatial", sample);
                        SetEvent(TEST24_EVENT_FLAG_CHANGE_TRACE);
                    }
                }
            }
            else {
                if (spatialReuse) {
                    ImGui::Text("Iteration: %d", iter);
                    ImGui::Text("    Range: %d", range);
                    ImGui::Text("   Sample: %d", sample);
                }
            }

            int samplePerAll = variable->GetUInt32("SamplePerAll");
            ImGui::Text("SamplePerAll: %d", samplePerAll);

            char saveFilePath[256];
            std::strncpy(saveFilePath, m_FilePath.c_str(), sizeof(saveFilePath) - 1);
            if (ImGui::InputText("SaveFilepath", saveFilePath, sizeof(saveFilePath))) {
                m_FilePath = std::string(saveFilePath);
            }

            int samplePerSave = m_SamplePerSave;
            if (ImGui::InputInt("SamplePerSave", &samplePerSave)) {
                m_SamplePerSave = std::max(1, samplePerSave);
            }

            bool isLaunched = variable->GetBool("Launched");
            if (!isLaunched) {
                if (ImGui::Button("Started")) {
                    variable->SetBool("Started", true);
                    SetEvent(TEST24_EVENT_FLAG_CHANGE_TRACE);
                }
                ImGui::SameLine();
            }
            bool pushSave = false;
            if (ImGui::Button("Save")) {
                pushSave = true;
            }
            bool saveIter = isLaunched && samplePerAll && ((samplePerAll % m_SamplePerSave) == 0);
            if (pushSave || saveIter) {
                std::string filePathBase = std::string(TEST_TEST24_RESTIR_OPX_RSLT_PATH) + "/" + saveFilePath + "result_restir_" + std::to_string(samplePerAll);
                std::string filePathPng = filePathBase + ".png";
                auto rTexture = GetFramebuffer()->GetComponent<test::RTGLTextureFBComponent<uchar4>>("RTexture");
                if (test::SavePngImgFromGL(filePathPng.c_str(), rTexture->GetHandle())) {
                    std::cout << "Save\n";
                }
                std::string filePathExr = filePathBase + ".exr";
                auto rAccum = GetFramebuffer()->GetComponent<test::RTCUDABufferFBComponent<float3>>("RAccum");
                if (test::SaveExrImgFromCUDA(filePathExr.c_str(), rAccum->GetHandle(), GetFramebuffer()->GetWidth(), GetFramebuffer()->GetHeight(), samplePerAll)) {
                    std::cout << "Save\n";
                }
            }
        }
    }
    virtual ~ReSTIRTraceConfigGuiWindow()noexcept {}
private:
    unsigned int m_SamplePerSave = 1000;
    std::string m_FilePath;
};
class  GuidePathTraceConfigGuiWindow : public TraceConfigGuiWindow
{
private:
    using TracerVariableMap = std::unordered_map < std::string, std::shared_ptr<rtlib::ext::VariableMap>>;
public:
    GuidePathTraceConfigGuiWindow(const TracerVariableMap& tracerVariables_, const std::shared_ptr<test::RTFramebuffer>& framebuffer_, unsigned int& eventFlags_)noexcept : TraceConfigGuiWindow("GuidePathOPX", tracerVariables_, framebuffer_, eventFlags_), m_FilePath(""){}
    virtual void DrawGui()override {
        auto variable = GetVariable();
        if (variable) {
            int sampleForBudget   = variable->GetUInt32("SampleForBudget");
            int samplePerLaunch   = variable->GetUInt32("SamplePerLaunch");
            auto ratioForBudget   = variable->GetFloat1("RatioForBudget");
            int iterationForBuilt = variable->GetUInt32("IterationForBuilt");

            if(!HasEvent(TEST24_EVENT_FLAG_BIT_LOCK)){
                if (ImGui::InputInt("SampleForBudget"      ,&sampleForBudget)) {
                    variable->SetUInt32("SampleForBudget"  , sampleForBudget);
                    SetEvent(TEST24_EVENT_FLAG_CHANGE_TRACE);
                }
                if (ImGui::SliderInt("SamplePerLaunch"     , &samplePerLaunch, 1, 100)) {
                    variable->SetUInt32("SamplePerLaunch"   , samplePerLaunch);
                    SetEvent(TEST24_EVENT_FLAG_CHANGE_TRACE);
                }
                if (ImGui::SliderInt("IterationForBuilt"   , &iterationForBuilt, 1, 10)) {
                    variable->SetUInt32("IterationForBuilt", iterationForBuilt);
                    SetEvent(TEST24_EVENT_FLAG_CHANGE_TRACE);
                }
                if (ImGui::InputFloat(  "RatioForBudget"   , &ratioForBudget)) {
                    variable->SetFloat1("RatioForBudget"   ,  ratioForBudget);
                    SetEvent(TEST24_EVENT_FLAG_CHANGE_TRACE);
                }
            }
            else {
                ImGui::Text("  SamplePerBudget: %d", sampleForBudget  );
                ImGui::Text("  SamplePerLaunch: %d", samplePerLaunch  );
                ImGui::Text("   RatioForBudget: %f", ratioForBudget   );
                ImGui::Text("IterationForBuilt: %f", iterationForBuilt);
            }

            int samplePerAll = variable->GetUInt32("SamplePerAll");
            int curIteration = variable->GetUInt32("CurIteration");
            int samplePerTmp = variable->GetUInt32("SamplePerTmp");

            ImGui::Text("SamplePerAll: %d", samplePerAll);
            ImGui::Text("CurIteration: %d", curIteration);
            ImGui::Text("SamplePerTmp: %d", samplePerTmp);



            char saveFilePath[256];
            std::strncpy(saveFilePath, m_FilePath.c_str(), sizeof(saveFilePath) - 1);
            if (ImGui::InputText("SaveFilepath", saveFilePath, sizeof(saveFilePath))) {
                m_FilePath = std::string(saveFilePath);
            }

            int samplePerSave = m_SamplePerSave;
            if (ImGui::InputInt("SamplePerSave", &samplePerSave)) {
                m_SamplePerSave = std::max(1, samplePerSave);
            }

            bool isLaunched = variable->GetBool("Launched");
            if (!isLaunched) {
                if (ImGui::Button("Started")) {
                    variable->SetBool("Started", true);
                    SetEvent(TEST24_EVENT_FLAG_CHANGE_TRACE);
                }
                ImGui::SameLine();
            }
            bool pushSave = false;
            if (ImGui::Button("Save")) {
                 pushSave = true;
            }
            bool saveIter = isLaunched && samplePerAll && ((samplePerAll % m_SamplePerSave)==0);
            if (pushSave || saveIter){
                std::cout << "GuidePathTrace Save: " << samplePerAll << std::endl;
                std::string filePathBase = std::string(TEST_TEST24_GUIDE_PATH_OPX_RSLT_PATH) + "/" + saveFilePath + "result_guide_path_" + std::to_string(samplePerAll);
                std::string filePathPng = filePathBase + ".png";
                auto rTexture = GetFramebuffer()->GetComponent<test::RTGLTextureFBComponent<uchar4>>("RTexture");
                if (test::SavePngImgFromGL(filePathPng.c_str(), rTexture->GetHandle())) {
                    std::cout << "Save\n";
                }
                std::string filePathExr = filePathBase + ".exr";
                auto rAccum = GetFramebuffer()->GetComponent<test::RTCUDABufferFBComponent<float3>>("RAccum");
                if (test::SaveExrImgFromCUDA(filePathExr.c_str(), rAccum->GetHandle(), GetFramebuffer()->GetWidth(), GetFramebuffer()->GetHeight(), samplePerAll)) {
                    std::cout << "Save\n";
                }
            }
        }
    }
    virtual ~GuidePathTraceConfigGuiWindow()noexcept {}
private:
    unsigned int m_SamplePerSave = 1000;
    std::string  m_FilePath;
};
class  GuideNEETraceConfigGuiWindow : public TraceConfigGuiWindow
{
private:
    using TracerVariableMap = std::unordered_map < std::string, std::shared_ptr<rtlib::ext::VariableMap>>;
public:
    GuideNEETraceConfigGuiWindow(const TracerVariableMap& tracerVariables_, const std::shared_ptr<test::RTFramebuffer>& framebuffer_, unsigned int& eventFlags_)noexcept : TraceConfigGuiWindow("GuideNEEOPX", tracerVariables_, framebuffer_, eventFlags_), m_FilePath("") {}
    virtual void DrawGui()override {
        auto variable = GetVariable();
        if (variable) {
            int sampleForBudget   = variable->GetUInt32("SampleForBudget");
            int samplePerLaunch   = variable->GetUInt32("SamplePerLaunch");
            auto ratioForBudget   = variable->GetFloat1("RatioForBudget");
            int iterationForBuilt = variable->GetUInt32("IterationForBuilt");

            if (!HasEvent(TEST24_EVENT_FLAG_BIT_LOCK)) {
                if (ImGui::InputInt("SampleForBudget", &sampleForBudget)) {
                    variable->SetUInt32("SampleForBudget", sampleForBudget);
                    SetEvent(TEST24_EVENT_FLAG_CHANGE_TRACE);
                }
                if (ImGui::SliderInt("SamplePerLaunch", &samplePerLaunch, 1, 100)) {
                    variable->SetUInt32("SamplePerLaunch", samplePerLaunch);
                    SetEvent(TEST24_EVENT_FLAG_CHANGE_TRACE);
                }
                if (ImGui::SliderInt("IterationForBuilt", &iterationForBuilt, 1, 10)) {
                    variable->SetUInt32("IterationForBuilt", iterationForBuilt);
                    SetEvent(TEST24_EVENT_FLAG_CHANGE_TRACE);
                }
                if (ImGui::InputFloat("RatioForBudget", &ratioForBudget)) {
                    variable->SetFloat1("RatioForBudget", ratioForBudget);
                    SetEvent(TEST24_EVENT_FLAG_CHANGE_TRACE);
                }
            }
            else {
                ImGui::Text("  SamplePerBudget: %d", sampleForBudget);
                ImGui::Text("  SamplePerLaunch: %d", samplePerLaunch);
                ImGui::Text("   RatioForBudget: %f", ratioForBudget);
                ImGui::Text("IterationForBuilt: %f", iterationForBuilt);
            }

            int samplePerAll = variable->GetUInt32("SamplePerAll");
            int curIteration = variable->GetUInt32("CurIteration");
            int samplePerTmp = variable->GetUInt32("SamplePerTmp");

            ImGui::Text("SamplePerAll: %d", samplePerAll);
            ImGui::Text("CurIteration: %d", curIteration);
            ImGui::Text("SamplePerTmp: %d", samplePerTmp);

            char saveFilePath[256];
            std::strncpy(saveFilePath, m_FilePath.c_str(), sizeof(saveFilePath) - 1);
            if (ImGui::InputText("SaveFilepath", saveFilePath, sizeof(saveFilePath))) {
                m_FilePath = std::string(saveFilePath);
            }
            int samplePerSave = m_SamplePerSave;
            if (ImGui::InputInt("SamplePerSave", &samplePerSave)) {
                m_SamplePerSave = std::max(1, samplePerSave);
            }

            bool isLaunched = variable->GetBool("Launched");
            if (!isLaunched) {
                if (ImGui::Button("Started")) {
                    variable->SetBool("Started", true);
                    SetEvent(TEST24_EVENT_FLAG_CHANGE_TRACE);
                }
                ImGui::SameLine();
            }
            bool pushSave = false;
            if (ImGui::Button("Save")) {
                pushSave = true;
            }
            bool saveIter = isLaunched && samplePerAll && ((samplePerAll % m_SamplePerSave)==0);
            if (pushSave || saveIter) {
                std::string filePathBase = std::string(TEST_TEST24_GUIDE_NEE_OPX_RSLT_PATH) + "/" + saveFilePath + "result_guide_nee_" + std::to_string(samplePerAll);
                std::string filePathPng = filePathBase + ".png";
                auto rTexture = GetFramebuffer()->GetComponent<test::RTGLTextureFBComponent<uchar4>>("RTexture");
                if (test::SavePngImgFromGL(filePathPng.c_str(), rTexture->GetHandle())) {
                    std::cout << "Save\n";
                }
                std::string filePathExr = filePathBase + ".exr";
                auto rAccum = GetFramebuffer()->GetComponent<test::RTCUDABufferFBComponent<float3>>("RAccum");
                if (test::SaveExrImgFromCUDA(filePathExr.c_str(), rAccum->GetHandle(), GetFramebuffer()->GetWidth(), GetFramebuffer()->GetHeight(), samplePerAll)) {
                    std::cout << "Save\n";
                }
            }
        }
    }
    virtual ~GuideNEETraceConfigGuiWindow()noexcept {}
private:
    unsigned int m_SamplePerSave = 1000;
    std::string  m_FilePath;
};
class  GuideWRSTraceConfigGuiWindow : public TraceConfigGuiWindow
{
private:
    using TracerVariableMap = std::unordered_map < std::string, std::shared_ptr<rtlib::ext::VariableMap>>;
public:
    GuideWRSTraceConfigGuiWindow(const TracerVariableMap& tracerVariables_, const std::shared_ptr<test::RTFramebuffer>& framebuffer_, unsigned int& eventFlags_)noexcept : TraceConfigGuiWindow("GuideWRSOPX", tracerVariables_, framebuffer_, eventFlags_), m_FilePath{}{}
    virtual void DrawGui()override {
        auto variable = GetVariable();
        if (variable) {
            int sampleForBudget   = variable->GetUInt32("SampleForBudget");
            int samplePerLaunch   = variable->GetUInt32("SamplePerLaunch");
            auto ratioForBudget   = variable->GetFloat1("RatioForBudget");
            int iterationForBuilt = variable->GetUInt32("IterationForBuilt");
            int numCandidates     = variable->GetUInt32("NumCandidates");

            if (!HasEvent(TEST24_EVENT_FLAG_BIT_LOCK)) {
                if (ImGui::InputInt("SampleForBudget", &sampleForBudget)) {
                    variable->SetUInt32("SampleForBudget", sampleForBudget);
                    SetEvent(TEST24_EVENT_FLAG_CHANGE_TRACE);
                }
                if (ImGui::SliderInt("SamplePerLaunch", &samplePerLaunch, 1, 100)) {
                    variable->SetUInt32("SamplePerLaunch", samplePerLaunch);
                    SetEvent(TEST24_EVENT_FLAG_CHANGE_TRACE);
                }
                if (ImGui::SliderInt("IterationForBuilt"   ,&iterationForBuilt, 1, 10)) {
                    variable->SetUInt32("IterationForBuilt", iterationForBuilt);
                    SetEvent(TEST24_EVENT_FLAG_CHANGE_TRACE);
                }
                if (ImGui::InputFloat("RatioForBudget"  ,&ratioForBudget)) {
                    variable->SetFloat1("RatioForBudget", ratioForBudget);
                    SetEvent(TEST24_EVENT_FLAG_CHANGE_TRACE);
                }
                if (ImGui::InputInt("NumCandidates"   ,&numCandidates)) {
                    variable->SetUInt32("NumCandidates", numCandidates);
                    SetEvent(TEST24_EVENT_FLAG_CHANGE_TRACE);
                }
            }
            else {
                ImGui::Text("  SamplePerBudget: %d", sampleForBudget  );
                ImGui::Text("  SamplePerLaunch: %d", samplePerLaunch  );
                ImGui::Text("   RatioForBudget: %f", ratioForBudget   );
                ImGui::Text("IterationForBuilt: %f", iterationForBuilt);
                ImGui::Text("    NumCandidates: %d", numCandidates    );
            }

            int samplePerAll = variable->GetUInt32("SamplePerAll");
            int curIteration = variable->GetUInt32("CurIteration");
            int samplePerTmp = variable->GetUInt32("SamplePerTmp");

            ImGui::Text("SamplePerAll: %d", samplePerAll);
            ImGui::Text("CurIteration: %d", curIteration);
            ImGui::Text("SamplePerTmp: %d", samplePerTmp);

            char saveFilePath[256];
            std::strncpy(saveFilePath, m_FilePath.c_str(), sizeof(saveFilePath) - 1);
            if (ImGui::InputText("SaveFilepath", saveFilePath, sizeof(saveFilePath))) {
                m_FilePath = std::string(saveFilePath);
            }
            int samplePerSave = m_SamplePerSave;
            if (ImGui::InputInt("SamplePerSave", &samplePerSave)) {
                m_SamplePerSave = std::max(1, samplePerSave);
            }

            bool isLaunched = variable->GetBool("Launched");
            if (!isLaunched) {
                if (ImGui::Button("Started")) {
                    variable->SetBool("Started", true);
                    SetEvent(TEST24_EVENT_FLAG_CHANGE_TRACE);
                }
                ImGui::SameLine();
            }
            bool pushSave = false;
            if (ImGui::Button("Save")) {
                pushSave = true;
            }
            bool saveIter = isLaunched && samplePerAll && ((samplePerAll % m_SamplePerSave)==0);
            if (pushSave || saveIter) {
                std::string filePathBase = std::string(TEST_TEST24_GUIDE_WRS_OPX_RSLT_PATH) + "/" + saveFilePath + "result_guide_wrs_" + std::to_string(samplePerAll);
                std::string filePathPng = filePathBase + ".png";
                auto rTexture = GetFramebuffer()->GetComponent<test::RTGLTextureFBComponent<uchar4>>("RTexture");
                if (test::SavePngImgFromGL(filePathPng.c_str(), rTexture->GetHandle())) {
                    std::cout << "Save\n";
                }
                std::string filePathExr = filePathBase + ".exr";
                auto rAccum = GetFramebuffer()->GetComponent<test::RTCUDABufferFBComponent<float3>>("RAccum");
                if (test::SaveExrImgFromCUDA(filePathExr.c_str(), rAccum->GetHandle(), GetFramebuffer()->GetWidth(), GetFramebuffer()->GetHeight(), samplePerAll)) {
                    std::cout << "Save\n";
                }
            }
        }
    }
    virtual ~GuideWRSTraceConfigGuiWindow()noexcept {}
private:
    unsigned int m_SamplePerSave = 1000;
    std::string  m_FilePath;
};
class  GuideWRS2TraceConfigGuiWindow : public TraceConfigGuiWindow
{
private:
    using TracerVariableMap = std::unordered_map <std::string, std::shared_ptr<rtlib::ext::VariableMap>>;
public:
    GuideWRS2TraceConfigGuiWindow(const TracerVariableMap& tracerVariables_, const std::shared_ptr<test::RTFramebuffer>& framebuffer_, unsigned int& eventFlags_)noexcept : TraceConfigGuiWindow("GuideWRS2OPX", tracerVariables_, framebuffer_, eventFlags_), m_FilePath{}{}
    virtual ~GuideWRS2TraceConfigGuiWindow()noexcept {}

    virtual void DrawGui()override {
        auto variable = GetVariable();
        if (variable) {
            int sampleForBudget = variable->GetUInt32("SampleForBudget");
            int samplePerLaunch = variable->GetUInt32("SamplePerLaunch");
            auto ratioForBudget = variable->GetFloat1("RatioForBudget");
            int iterationForBuilt = variable->GetUInt32("IterationForBuilt");
            int numCandidates = variable->GetUInt32("NumCandidates");

            if (!HasEvent(TEST24_EVENT_FLAG_BIT_LOCK)) {
                if (ImGui::InputInt("SampleForBudget", &sampleForBudget)) {
                    variable->SetUInt32("SampleForBudget", sampleForBudget);
                    SetEvent(TEST24_EVENT_FLAG_CHANGE_TRACE);
                }
                if (ImGui::SliderInt("SamplePerLaunch", &samplePerLaunch, 1, 100)) {
                    variable->SetUInt32("SamplePerLaunch", samplePerLaunch);
                    SetEvent(TEST24_EVENT_FLAG_CHANGE_TRACE);
                }
                if (ImGui::SliderInt("IterationForBuilt", &iterationForBuilt, 1, 10)) {
                    variable->SetUInt32("IterationForBuilt", iterationForBuilt);
                    SetEvent(TEST24_EVENT_FLAG_CHANGE_TRACE);
                }
                if (ImGui::InputFloat("RatioForBudget", &ratioForBudget)) {
                    variable->SetFloat1("RatioForBudget", ratioForBudget);
                    SetEvent(TEST24_EVENT_FLAG_CHANGE_TRACE);
                }
                if (ImGui::InputInt("NumCandidates", &numCandidates)) {
                    variable->SetUInt32("NumCandidates", numCandidates);
                    SetEvent(TEST24_EVENT_FLAG_CHANGE_TRACE);
                }
            }
            else {
                ImGui::Text("  SamplePerBudget: %d", sampleForBudget);
                ImGui::Text("  SamplePerLaunch: %d", samplePerLaunch);
                ImGui::Text("   RatioForBudget: %f", ratioForBudget);
                ImGui::Text("IterationForBuilt: %f", iterationForBuilt);
                ImGui::Text("    NumCandidates: %d", numCandidates);
            }

            int samplePerAll = variable->GetUInt32("SamplePerAll");
            int curIteration = variable->GetUInt32("CurIteration");
            int samplePerTmp = variable->GetUInt32("SamplePerTmp");

            ImGui::Text("SamplePerAll: %d", samplePerAll);
            ImGui::Text("CurIteration: %d", curIteration);
            ImGui::Text("SamplePerTmp: %d", samplePerTmp);

            char saveFilePath[256];
            std::strncpy(saveFilePath, m_FilePath.c_str(), sizeof(saveFilePath) - 1);
            if (ImGui::InputText("SaveFilepath", saveFilePath, sizeof(saveFilePath))) {
                m_FilePath = std::string(saveFilePath);
            }
            int samplePerSave = m_SamplePerSave;
            if (ImGui::InputInt("SamplePerSave", &samplePerSave)) {
                m_SamplePerSave = std::max(1, samplePerSave);
            }

            bool isLaunched = variable->GetBool("Launched");
            if (!isLaunched) {
                if (ImGui::Button("Started")) {
                    variable->SetBool("Started", true);
                    SetEvent(TEST24_EVENT_FLAG_CHANGE_TRACE);
                }
                ImGui::SameLine();
            }
            bool pushSave = false;
            if (ImGui::Button("Save")) {
                pushSave = true;
            }
            bool saveIter = isLaunched && samplePerAll && ((samplePerAll % m_SamplePerSave) == 0);
            if (pushSave || saveIter) {
                std::string filePathBase = std::string(TEST_TEST24_GUIDE_WRS2_OPX_RSLT_PATH) + "/" + saveFilePath + "result_guide_wrs2_" + std::to_string(samplePerAll);
                std::string filePathPng = filePathBase + ".png";
                auto rTexture = GetFramebuffer()->GetComponent<test::RTGLTextureFBComponent<uchar4>>("RTexture");
                if (test::SavePngImgFromGL(filePathPng.c_str(), rTexture->GetHandle())) {
                    std::cout << "Save\n";
                }
                std::string filePathExr = filePathBase + ".exr";
                auto rAccum = GetFramebuffer()->GetComponent<test::RTCUDABufferFBComponent<float3>>("RAccum");
                if (test::SaveExrImgFromCUDA(filePathExr.c_str(), rAccum->GetHandle(), GetFramebuffer()->GetWidth(), GetFramebuffer()->GetHeight(), samplePerAll)) {
                    std::cout << "Save\n";
                }
            }
        }
    }
private:
    unsigned int m_SamplePerSave = 1000;
    std::string  m_FilePath;
};
class  GuideReSTIRTraceConfigGuiWindow : public TraceConfigGuiWindow
{
private:
    using TracerVariableMap = std::unordered_map < std::string, std::shared_ptr<rtlib::ext::VariableMap>>;
public:
    GuideReSTIRTraceConfigGuiWindow(const TracerVariableMap& tracerVariables_, const std::shared_ptr<test::RTFramebuffer>& framebuffer_, unsigned int& eventFlags_)noexcept : TraceConfigGuiWindow("GuideReSTIROPX", tracerVariables_, framebuffer_, eventFlags_), m_FilePath{}{}
    virtual void DrawGui()override {
        auto variable = GetVariable();
        if (variable) {
            int sampleForBudget = variable->GetUInt32("SampleForBudget");
            int samplePerLaunch = variable->GetUInt32("SamplePerLaunch");
            auto ratioForBudget = variable->GetFloat1("RatioForBudget");
            int iterationForBuilt = variable->GetUInt32("IterationForBuilt");
            int numCandidates = variable->GetUInt32("NumCandidates");

            if (!HasEvent(TEST24_EVENT_FLAG_BIT_LOCK)) {
                if (ImGui::InputInt("SampleForBudget", &sampleForBudget)) {
                    variable->SetUInt32("SampleForBudget", sampleForBudget);
                    SetEvent(TEST24_EVENT_FLAG_CHANGE_TRACE);
                }
                if (ImGui::SliderInt("SamplePerLaunch", &samplePerLaunch, 1, 100)) {
                    variable->SetUInt32("SamplePerLaunch", samplePerLaunch);
                    SetEvent(TEST24_EVENT_FLAG_CHANGE_TRACE);
                }
                if (ImGui::SliderInt("IterationForBuilt", &iterationForBuilt, 1, 10)) {
                    variable->SetUInt32("IterationForBuilt", iterationForBuilt);
                    SetEvent(TEST24_EVENT_FLAG_CHANGE_TRACE);
                }
                if (ImGui::InputFloat("RatioForBudget", &ratioForBudget)) {
                    variable->SetFloat1("RatioForBudget", ratioForBudget);
                    SetEvent(TEST24_EVENT_FLAG_CHANGE_TRACE);
                }
                if (ImGui::SliderInt("NumCandidates", &numCandidates, 1, 64)) {
                    variable->SetUInt32("NumCandidates", numCandidates);
                    SetEvent(TEST24_EVENT_FLAG_CHANGE_TRACE);
                }
            }
            else {
                ImGui::Text("  SamplePerBudget: %d", sampleForBudget);
                ImGui::Text("  SamplePerLaunch: %d", samplePerLaunch);
                ImGui::Text("   RatioForBudget: %f", ratioForBudget);
                ImGui::Text("IterationForBuilt: %f", iterationForBuilt);
                ImGui::Text("    NumCandidates: %d", numCandidates);
            }

            int samplePerAll = variable->GetUInt32("SamplePerAll");
            int curIteration = variable->GetUInt32("CurIteration");
            int samplePerTmp = variable->GetUInt32("SamplePerTmp");

            ImGui::Text("SamplePerAll: %d", samplePerAll);
            ImGui::Text("CurIteration: %d", curIteration);
            ImGui::Text("SamplePerTmp: %d", samplePerTmp);

            char saveFilePath[256];
            std::strncpy(saveFilePath, m_FilePath.c_str(), sizeof(saveFilePath) - 1);
            if (ImGui::InputText("SaveFilepath", saveFilePath, sizeof(saveFilePath))) {
                m_FilePath = std::string(saveFilePath);
            }
            int samplePerSave = m_SamplePerSave;
            if (ImGui::InputInt("SamplePerSave", &samplePerSave)) {
                m_SamplePerSave = std::max(1, samplePerSave);
            }

            bool isLaunched = variable->GetBool("Launched");
            if (!isLaunched) {
                if (ImGui::Button("Started")) {
                    variable->SetBool("Started", true);
                    SetEvent(TEST24_EVENT_FLAG_CHANGE_TRACE);
                }
                ImGui::SameLine();
            }
            bool pushSave = false;
            if (ImGui::Button("Save")) {
                pushSave = true;
            }
            bool saveIter = isLaunched && samplePerAll && ((samplePerAll % m_SamplePerSave) == 0);
            if (pushSave || saveIter) {
                std::string filePathBase = std::string(TEST_TEST24_GUIDE_RESTIR_OPX_RSLT_PATH) + "/" + saveFilePath + "result_guide_restir_" + std::to_string(samplePerAll);
                std::string filePathPng = filePathBase + ".png";
                auto rTexture = GetFramebuffer()->GetComponent<test::RTGLTextureFBComponent<uchar4>>("RTexture");
                if (test::SavePngImgFromGL(filePathPng.c_str(), rTexture->GetHandle())) {
                    std::cout << "Save\n";
                }
                std::string filePathExr = filePathBase + ".exr";
                auto rAccum = GetFramebuffer()->GetComponent<test::RTCUDABufferFBComponent<float3>>("RAccum");
                if (test::SaveExrImgFromCUDA(filePathExr.c_str(), rAccum->GetHandle(), GetFramebuffer()->GetWidth(), GetFramebuffer()->GetHeight(), samplePerAll)) {
                    std::cout << "Save\n";
                }
            }
        }
    }
    virtual ~GuideReSTIRTraceConfigGuiWindow()noexcept {}
private:
    unsigned int m_SamplePerSave = 1000;
    std::string  m_FilePath;
};
void  Test24GuiDelegate::Initialize()
{
    m_Gui->Initialize();
    // MainMenuBar
    auto mainMenuBar = m_Gui->AddGuiMainMenuBar();
    auto fileMenu    = mainMenuBar->AddGuiMenu("File");
    {
        auto mdlMenu    = fileMenu->AddGuiMenu("Model");
        auto fileDialog = std::make_shared<test::RTGuiFileDialog>("Choose File", ".obj", TEST_TEST24_DATA_PATH"\\");
        //Open
        auto openMenuItem = std::make_shared < test::RTGuiMenuItem>("Open");
        openMenuItem->SetUserPointer(fileDialog.get());
        openMenuItem->SetClickCallback([](test::RTGuiMenuItem* item) {
            auto fd = reinterpret_cast<test::RTGuiFileDialog*>(item->GetUserPointer());
            if (fd && !fd->IsOpen()) {
                fd->Open();
            }
            });
        //Close
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
    auto cnfgMenu    = mainMenuBar->AddGuiMenu("Config");
    {
        // Frame
        {
            auto fmbfItem = cnfgMenu->AddGuiMenu("Frame");
            // MainFrameConfig
            auto mainFmCnfgWindow = std::make_shared<MainFrameConfigGuiWindow>(m_FramePublicNames, m_CurMainFrameName);
            mainFmCnfgWindow->SetActive(false);   //Default: Invisible
            m_Gui->SetGuiWindow(mainFmCnfgWindow);
            fmbfItem->SetGuiMenuItem(std::make_shared<test::RTGuiOpenWindowMenuItem>(mainFmCnfgWindow));
            fmbfItem->SetGuiMenuItem(std::make_shared<test::RTGuiCloseWindowMenuItem>(mainFmCnfgWindow));
        }
        // Trace
        {
            auto trcrItem = cnfgMenu->AddGuiMenu("Tracer");
            auto mainTcCnfgWindow = std::make_shared<MainTraceConfigGuiWindow>(m_TracePublicNames,m_CurMainTraceName,m_MaxTraceDepth,m_SamplePerLaunch,m_EventFlags);
            mainTcCnfgWindow->SetActive(false);   //Default: Invisible
            m_Gui->SetGuiWindow(mainTcCnfgWindow);
            trcrItem->SetGuiMenuItem(std::make_shared<test:: RTGuiOpenWindowMenuItem>(mainTcCnfgWindow));
            trcrItem->SetGuiMenuItem(std::make_shared<test::RTGuiCloseWindowMenuItem>(mainTcCnfgWindow));
            auto   pathTcCnfgWindow = std::make_shared<PathTraceConfigGuiWindow>( m_TracerVariables, m_Framebuffer, m_EventFlags);
            pathTcCnfgWindow->SetActive(false);
            if (mainTcCnfgWindow->AddTracerWindow(pathTcCnfgWindow->GetName(), pathTcCnfgWindow)) {
                m_Gui->SetGuiWindow(pathTcCnfgWindow);
            }
            auto    neeTcCnfgWindow = std::make_shared<NEETraceConfigGuiWindow>(m_TracerVariables, m_Framebuffer, m_EventFlags);
             neeTcCnfgWindow->SetActive(false);
            if (mainTcCnfgWindow->AddTracerWindow(neeTcCnfgWindow->GetName(), neeTcCnfgWindow)) {
                m_Gui->SetGuiWindow(neeTcCnfgWindow);
            }
            auto    wrsTcCnfgWindow = std::make_shared<WRSTraceConfigGuiWindow>(m_TracerVariables, m_Framebuffer, m_EventFlags);
             wrsTcCnfgWindow->SetActive(false);
            if (mainTcCnfgWindow->AddTracerWindow(wrsTcCnfgWindow->GetName(), wrsTcCnfgWindow)) {
                m_Gui->SetGuiWindow(wrsTcCnfgWindow);
            }
            auto restirTcCnfgWindow = std::make_shared<ReSTIRTraceConfigGuiWindow>( m_TracerVariables, m_Framebuffer, m_EventFlags);
            restirTcCnfgWindow->SetActive(false);
            if (mainTcCnfgWindow->AddTracerWindow(restirTcCnfgWindow->GetName(), restirTcCnfgWindow)) {
                m_Gui->SetGuiWindow(restirTcCnfgWindow);
            }
            auto gdPathTcCnfgWindow = std::make_shared<GuidePathTraceConfigGuiWindow>(m_TracerVariables, m_Framebuffer, m_EventFlags);
            gdPathTcCnfgWindow->SetActive(false);
            if (mainTcCnfgWindow->AddTracerWindow(gdPathTcCnfgWindow->GetName(), gdPathTcCnfgWindow)) {
                m_Gui->SetGuiWindow(gdPathTcCnfgWindow);
            }
            auto gdNEETcCnfgWindow = std::make_shared<GuideNEETraceConfigGuiWindow>(m_TracerVariables, m_Framebuffer, m_EventFlags);
            gdNEETcCnfgWindow->SetActive(false);
            if (mainTcCnfgWindow->AddTracerWindow(gdNEETcCnfgWindow->GetName() , gdNEETcCnfgWindow)) {
                m_Gui->SetGuiWindow(gdNEETcCnfgWindow);
            }
            auto gdWRSTcCnfgWindow = std::make_shared<GuideWRSTraceConfigGuiWindow>(m_TracerVariables, m_Framebuffer, m_EventFlags);
            gdWRSTcCnfgWindow->SetActive(false);
            if (mainTcCnfgWindow->AddTracerWindow(gdWRSTcCnfgWindow->GetName(), gdWRSTcCnfgWindow)) {
                m_Gui->SetGuiWindow(gdWRSTcCnfgWindow);
            }
            auto gdWRS2TcCnfgWindow = std::make_shared<GuideWRS2TraceConfigGuiWindow>(m_TracerVariables, m_Framebuffer, m_EventFlags);
            gdWRS2TcCnfgWindow->SetActive(false);
            if (mainTcCnfgWindow->AddTracerWindow(gdWRS2TcCnfgWindow->GetName(), gdWRS2TcCnfgWindow)) {
                m_Gui->SetGuiWindow(gdWRS2TcCnfgWindow);
            }
            auto gdReSTIRTcCnfgWindow = std::make_shared<GuideReSTIRTraceConfigGuiWindow>(m_TracerVariables, m_Framebuffer, m_EventFlags);
            gdReSTIRTcCnfgWindow->SetActive(false);
            if (mainTcCnfgWindow->AddTracerWindow(gdReSTIRTcCnfgWindow->GetName(), gdReSTIRTcCnfgWindow)) {
                m_Gui->SetGuiWindow(gdReSTIRTcCnfgWindow);
            }
        }
        // Input
        {
            auto iptItem = cnfgMenu->AddGuiMenu("Input");
            auto iptCnfgWindow = std::make_shared<InputConfigGuiWindow>(m_CurCursorPos,m_DelCursorPos,m_ScrollOffsets,m_CurFrameTime,m_DelFrameTime,m_SamplePerAll, m_SamplePerLaunch);
            iptCnfgWindow->SetActive(false);   //Default: Invisible
            m_Gui->SetGuiWindow(iptCnfgWindow);
            iptItem->SetGuiMenuItem(std::make_shared<test::RTGuiOpenWindowMenuItem>(iptCnfgWindow));
            iptItem->SetGuiMenuItem(std::make_shared<test::RTGuiCloseWindowMenuItem>(iptCnfgWindow));
        }
        // Camera
        {
            auto cmrItem = cnfgMenu->AddGuiMenu("Camera");
            auto cmrCnfgWindow = std::make_shared<CameraConfigGuiWindow>(m_Window,m_Framebuffer, m_CameraController, m_EventFlags);
            cmrCnfgWindow->SetActive(false);   //Default: Invisible
            m_Gui->SetGuiWindow(cmrCnfgWindow);
            cmrItem->SetGuiMenuItem(std::make_shared<test::RTGuiOpenWindowMenuItem>(cmrCnfgWindow));
            cmrItem->SetGuiMenuItem(std::make_shared<test::RTGuiCloseWindowMenuItem>(cmrCnfgWindow));
        }
        // Light
        {
            auto lhtItem = cnfgMenu->AddGuiMenu("Light");
            auto lhtCnfgWindow = std::make_shared<LightConfigGuiWindow>(m_BgLightColor, m_EventFlags);
            lhtCnfgWindow->SetActive(false);   //Default: Invisible
            m_Gui->SetGuiWindow(lhtCnfgWindow);
            lhtItem->SetGuiMenuItem(std::make_shared<test::RTGuiOpenWindowMenuItem >(lhtCnfgWindow));
            lhtItem->SetGuiMenuItem(std::make_shared<test::RTGuiCloseWindowMenuItem>(lhtCnfgWindow));
        }
        // Model
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
