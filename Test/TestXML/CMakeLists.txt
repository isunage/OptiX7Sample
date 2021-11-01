set(TEST_TEST_XML_DATA_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../Data")
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/TestXMLConfig.h.in 
    ${CMAKE_CURRENT_BINARY_DIR}/TestXMLConfig.h
)
add_executable(TestXML ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp ${CMAKE_CURRENT_BINARY_DIR}/TestXMLConfig.h)
target_include_directories(TestXML PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(TestXML PUBLIC tinyxml2 glm Boost::spirit Boost::filesystem Boost::thread Boost::iostreams Boost::algorithm)