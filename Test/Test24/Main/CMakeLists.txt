add_executable(
    Test24
    ${CMAKE_CURRENT_SOURCE_DIR}/include/Test24Application.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/Test24Gui.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Test24Application.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Test24Gui.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
)

target_include_directories(Test24 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(Test24 
    TestLib 
    Test24_Share 
    Test24_Tracer_TestGL 
    Test24_Tracer_DebugOPX 
    Test24_Tracer_PathOPX 
    Test24_Tracer_NEEOPX 
    Test24_Tracer_GuidePathOPX 
    Test24_Tracer_GuideNEEOPX 
    Test24_Tracer_GuideWRSOPX 
    Test24_Tracer_ReSTIROPX
 )
set_target_properties(Test24 PROPERTIES  FOLDER Test24/Main)
