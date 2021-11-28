add_library(TestLib 
    ${CMAKE_CURRENT_SOURCE_DIR}/include/TestLib/RTScene.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/TestLib/RTInterface.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/TestLib/RTProperties.h
    # SceneComponent
    ${CMAKE_CURRENT_SOURCE_DIR}/include/TestLib/RTShape.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/TestLib/RTSceneGraph.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/TestLib/RTMaterial.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/TestLib/RTTexture.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/TestLib/RTCamera.h

    ${CMAKE_CURRENT_SOURCE_DIR}/src/RTScene.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/RTProperties.cpp
    # SceneGraph
    ${CMAKE_CURRENT_SOURCE_DIR}/src/RTSceneGraph.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/SceneGraphs/RTInstancingGraph.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/SceneGraphs/RTInstancingGraph.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/SceneGraphs/RTGraphArray.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/SceneGraphs/RTGraphArray.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/SceneGraphs/RTShapeArray.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/SceneGraphs/RTShapeArray.cpp
    # Shapes
    ${CMAKE_CURRENT_SOURCE_DIR}/src/RTShape.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Shapes/RTObjMesh.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Shapes/RTObjMesh.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Shapes/RTBox.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Shapes/RTBox.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Shapes/RTSphere.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Shapes/RTSphere.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Shapes/RTInstancingShape.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Shapes/RTInstancingShape.cpp
    # Materials
    ${CMAKE_CURRENT_SOURCE_DIR}/src/RTMaterial.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Materials/RTDiffuse.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Materials/RTDiffuse.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Materials/RTPhong.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Materials/RTPhong.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Materials/RTDeltaReflection.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Materials/RTDeltaReflection.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Materials/RTDeltaDielectric.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Materials/RTDeltaDielectric.cpp
    # Textures
    ${CMAKE_CURRENT_SOURCE_DIR}/src/RTTexture.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Textures/RTImageTexture.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Textures/RTImageTexture.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Textures/RTColorTexture.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Textures/RTColorTexture.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Textures/RTCheckTexture.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Textures/RTCheckTexture.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Textures/RTBlendTexture.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Textures/RTBlendTexture.cpp
    # Cameras
    ${CMAKE_CURRENT_SOURCE_DIR}/src/RTCamera.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Cameras/RTPinhole.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Cameras/RTPinhole.cpp

)

target_include_directories(
    TestLib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include 
)

target_link_libraries(
    TestLib PUBLIC RTLib imgui-filedialog  nlohmann_json
)

set(TEST_TESTLIB_DATA_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../Data")
string(REPLACE "/" "\\\\" TEST_TESTLIB_DATA_PATH ${TEST_TESTLIB_DATA_PATH})

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/TestLibConfig.h.in 
    ${CMAKE_CURRENT_BINARY_DIR}/TestLibConfig.h
)

add_executable(TestLib_Test_Texture
    ${CMAKE_CURRENT_BINARY_DIR}/TestLibConfig.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Textures/RTTextureTest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Textures/RTTextureTest.json
)

target_include_directories(
    TestLib_Test_Texture PUBLIC ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/src/Textures
)

target_link_libraries(
    TestLib_Test_Texture PUBLIC TestLib
)

add_executable(TestLib_Test_Material
    ${CMAKE_CURRENT_BINARY_DIR}/TestLibConfig.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Materials/RTMaterialTest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Materials/RTMaterialTest.json
)

target_include_directories(
    TestLib_Test_Material PUBLIC ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/src/Materials
)

target_link_libraries(
    TestLib_Test_Material PUBLIC TestLib
)

add_executable(TestLib_Test_Camera
    ${CMAKE_CURRENT_BINARY_DIR}/TestLibConfig.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Cameras/RTCameraTest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Cameras/RTCameraTest.json
)

target_include_directories(
    TestLib_Test_Camera PUBLIC ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/src/Cameras
)

target_link_libraries(
    TestLib_Test_Camera PUBLIC TestLib
)

add_executable(TestLib_Test_Scene
    ${CMAKE_CURRENT_BINARY_DIR}/TestLibConfig.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/RTSceneTest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/RTSceneTest.json
)

target_include_directories(
    TestLib_Test_Scene PUBLIC ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(
    TestLib_Test_Scene PUBLIC TestLib
)
