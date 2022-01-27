
cmake_minimum_required(VERSION 3.9)
project(surrogate_toolkit VERSION 0.0.1)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)   # Enable -fPIC for all targets

# Set default standard to C++14, but allow user to override at command line
if(NOT "${CMAKE_CXX_STANDARD}")
    set(CMAKE_CXX_STANDARD 20)
endif()

string(APPEND CMAKE_CXX_FLAGS_DEBUG " -O0 -g -Wall -Wextra -fno-omit-frame-pointer")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

if(NOT CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    # CMake has been explicitly given an install dir, so we use that
elseif(DEFINED ENV{SURROGATE_HOME})
    # CMake hasn't been told of an install dir, but we have an envvar, so we use that instead
    set(CMAKE_INSTALL_PREFIX $ENV{SURROGATE_HOME} CACHE PATH "Comment explaining this nonsense" FORCE)
else()
    # No explicit install dir, so we install to './install' to avoid clobbering /usr/local
    set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/install CACHE PATH "Comment explaining this nonsense" FORCE)
endif()

message(STATUS "Installing to ${CMAKE_INSTALL_PREFIX}")


option(USE_ASAN "Compile with address sanitizer" OFF)
if (${USE_ASAN})
    add_compile_options(-fsanitize=address)
    add_link_options(-fsanitize=address)
endif()

message(STATUS "-----------------------")
message(STATUS "Build type is ${CMAKE_BUILD_TYPE}")
message(STATUS "Installation directory is ${CMAKE_INSTALL_PREFIX}")
if (${USE_ASAN})
    message(STATUS "USE_ASAN    On")
else()
    message(STATUS "USE_ASAN    Off")
endif()
message(STATUS "-----------------------")

include_directories(external)

add_subdirectory(examples)
add_subdirectory(surrogate_library)
add_subdirectory(vacuum_tool)
add_subdirectory(gpu_perf_tester)
add_subdirectory(vacuum_experiments)
