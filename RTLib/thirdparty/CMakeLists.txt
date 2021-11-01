set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake PARENT_SCOPE)
add_subdirectory(glfw)
add_subdirectory(glad)
add_subdirectory(tinyexr)
add_subdirectory(glm)
add_subdirectory(range-v3)
add_subdirectory(tinyobjloader)
add_subdirectory(boost)
add_subdirectory(json)
add_subdirectory(tinyxml2)
if(NOT tinyexr_lib)
    add_library(tinyexr_lib INTERFACE)
    target_include_directories(tinyexr_lib INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/tinyexr)
    target_link_libraries(tinyexr_lib INTERFACE tinyexr)
endif()
if(NOT stb)
    add_library(stb INTERFACE)
    target_include_directories(stb INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/stb)
endif()
if(NOT imgui)
    set(Imgui_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/imgui)
    add_library(imgui STATIC 
    ${Imgui_INCLUDE_DIR}/imconfig.h
    ${Imgui_INCLUDE_DIR}/imgui.h 
    ${Imgui_INCLUDE_DIR}/imgui_internal.h
    ${Imgui_INCLUDE_DIR}/imstb_rectpack.h
    ${Imgui_INCLUDE_DIR}/imstb_textedit.h
    ${Imgui_INCLUDE_DIR}/imstb_truetype.h

    ${Imgui_INCLUDE_DIR}/imgui.cpp 
    ${Imgui_INCLUDE_DIR}/imgui_demo.cpp
    ${Imgui_INCLUDE_DIR}/imgui_draw.cpp
    ${Imgui_INCLUDE_DIR}/imgui_tables.cpp
    ${Imgui_INCLUDE_DIR}/imgui_widgets.cpp)
    target_include_directories(imgui PUBLIC ${Imgui_INCLUDE_DIR})
endif()
if(NOT imgui-glfw)
    set(Imgui_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/imgui)
    add_library(imgui-glfw STATIC
    ${Imgui_INCLUDE_DIR}/backends/imgui_impl_glfw.h
    ${Imgui_INCLUDE_DIR}/backends/imgui_impl_glfw.cpp
    ${Imgui_INCLUDE_DIR}/backends/imgui_impl_opengl3.h
    ${Imgui_INCLUDE_DIR}/backends/imgui_impl_opengl3.cpp)
    target_include_directories(imgui-glfw PUBLIC ${Imgui_INCLUDE_DIR} ${Imgui_INCLUDE_DIR}/backends)
    target_link_libraries(imgui-glfw imgui glfw glad)
endif()
if(NOT imgui-filedialog)
    set(Imgui_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/imgui)
    set(Imgui_filedialog_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/ImGuiFileDialog)
    add_library(imgui-filedialog STATIC
        ${Imgui_filedialog_INCLUDE_DIR}/ImGuiFileDialog.h
        ${Imgui_filedialog_INCLUDE_DIR}/ImGuiFileDialogConfig.h
        ${Imgui_filedialog_INCLUDE_DIR}/ImGuiFileDialog.cpp
    )
    target_include_directories(imgui-filedialog PUBLIC ${Imgui_filedialog_INCLUDE_DIR})
    target_link_libraries( imgui-filedialog imgui-glfw stb)
endif()