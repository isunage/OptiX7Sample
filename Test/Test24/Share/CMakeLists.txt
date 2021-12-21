set(TEST_TEST24_NUM_RAY_TYPE 2)
set(TEST_TEST24_DATA_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../Data"  )
set(TEST_TEST24_RSLT_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../Result/Test/Test24")

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/Test24Config.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/Test24Config.h
)

add_library(Test24_Share 
    ${CMAKE_CURRENT_SOURCE_DIR}/include/Test24Share.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Test24Config.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/Test24Config.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Test24Share.cpp
)
target_include_directories(Test24_Share PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(Test24_Share PUBLIC RTLib)
set_target_properties(Test24_Share PROPERTIES  FOLDER Test24/Share)