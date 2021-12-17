
add_library(Test24_Share 
    ${CMAKE_CURRENT_SOURCE_DIR}/include/Test24Event.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Test24Event.cpp
)
target_include_directories(Test24_Share PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
set_target_properties(Test24_Share               PROPERTIES  FOLDER Test24/Share)