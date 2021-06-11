unset(OptiX_INCLUDE_DIR)
unset(OptiX_VERSION_MAJOR)
unset(OptiX_VERSION_MINOR)
unset(OptiX_VERSION_PATCH)
unset(OptiX_VERSION_STR)

if(CMAKE_SIZEOF_VOID_P EQUAL 8 AND NOT APPLE)
    set(bit_dest "64")
else()
    set(bit_dest "")
endif()

set(OptiX_INSTALL_DIR CACHE PATH "OptiX Install Directory")

if(NOT OptiX_INSTALL_DIR)
    if(WIN32)
    if($ENV{OptiX_INSTALL_DIR})
        set(OptiX_INSTALL_DIR $ENV{OptiX_INSTALL_DIR})
    else()
        set(OptiX_INSTALL_DIR_LIST 
        "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.2.0"
        "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.1.0"
        "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.0.0")
        foreach(install_dir IN LISTS OptiX_INSTALL_DIR_LIST)
            if(EXISTS ${install_dir})
                set(OptiX_INSTALL_DIR ${install_dir})
                break()
            endif()
        endforeach()
    endif()
    elseif(UNIX)
        set(OptiX_INSTALL_DIR $ENV{OptiX_INSTALL_DIR})
    endif()
endif()

if(NOT OptiX_INSTALL_DIR)
    message(FATAL_ERROR "Failed To Find OptiX Install Directory!")
endif()

find_path(OptiX_INCLUDE_DIR 
    NAMES optix.h 
    PATHS "${OptiX_INSTALL_DIR}/include")

if(OptiX_INCLUDE_DIR STREQUAL "OptiX_INCLUDE_DIR-NOTFOUND")
    message(FATAL_ERROR "Failed To Find OptiX Install Directory!")
endif()

#message(STATUS "OptiX_INCLUDE_DIR=${OptiX_INCLUDE_DIR}")

function(OptiX_Find_Version_From_File include_dir)
    file(READ "${include_dir}/optix.h" optix_h_string)
    #message(STATUS ${optix_h_string})
    string(REGEX MATCH "#define OPTIX_VERSION [0-9]+" define_version_macro ${optix_h_string})
    string(REGEX MATCH "[0-9]+" optix_version_raw_str ${define_version_macro})
    math(EXPR OptiX_VERSION_MAJOR "${optix_version_raw_str}/10000")
    math(EXPR OptiX_VERSION_MINOR "(${optix_version_raw_str}%10000)/100")
    math(EXPR OptiX_VERSION_PATCH "${optix_version_raw_str}%100")
    set(OptiX_VERSION_STR ${OptiX_VERSION_MAJOR}.${OptiX_VERSION_MINOR}.${OptiX_VERSION_PATCH} PARENT_SCOPE)
    set(OptiX_VERSION_MAJOR ${OptiX_VERSION_MAJOR} PARENT_SCOPE)
    set(OptiX_VERSION_MINOR ${OptiX_VERSION_MINOR} PARENT_SCOPE)
    set(OptiX_VERSION_PATCH ${OptiX_VERSION_PATCH} PARENT_SCOPE)
endfunction()

OptiX_Find_Version_From_File(${OptiX_INCLUDE_DIR})

if(NOT OptiX_VERSION_STR)
    message(FATAL_ERROR "Failed To Find OptiX Version!")
endif()
#message(STATUS "${OptiX_VERSION_STR}")

mark_as_advanced(OptiX_INCLUDE_DIR OptiX_VERSION_STR)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OptiX REQUIRED_VARS OptiX_INCLUDE_DIR OptiX_VERSION_STR VERSION_VAR OptiX_VERSION_STR)

if(OptiX_FOUND AND NOT OptiX::OptiX)
    if(OptiX_VERSION_MAJOR EQUAL 7)
        add_library(OptiX::OptiX IMPORTED INTERFACE)
        target_include_directories(OptiX::OptiX INTERFACE ${OptiX_INCLUDE_DIR})
    endif()
endif()