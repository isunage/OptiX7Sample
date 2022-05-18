
add_executable(Test25 
	${CMAKE_CURRENT_SOURCE_DIR}/include/Test25Application.h
	${CMAKE_CURRENT_SOURCE_DIR}/src/Test25Application.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
)

target_include_directories(Test25 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(Test25 PRIVATE TestLib)

set_target_properties(Test25 PROPERTIES  FOLDER Test25/Main)