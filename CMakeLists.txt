
cmake_minimum_required(VERSION 3.9)
project(surrogate_toolkit VERSION 0.0.1)

# Expose custom CMake modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

set(CMAKE_POSITION_INDEPENDENT_CODE ON)   # Enable -fPIC for all targets

# Set default standard to C++17, but allow user to override at command line
if(NOT "${CMAKE_CXX_STANDARD}")
    set(CMAKE_CXX_STANDARD 17)
endif()

# Set build type to debug by default, but allow user to override at command line
# Technically CMake does this already, but I want things to be explicit
if(NOT "${CMAKE_BUILD_TYPE}")
    set(CMAKE_BUILD_TYPE Debug)
endif()

string(APPEND CMAKE_CXX_FLAGS_DEBUG " -O0 -g -Wall -Wextra -fno-omit-frame-pointer")

if(NOT CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    # CMake has been explicitly given an install dir, so we use that
elseif(DEFINED ENV{PHASM_HOME})
    # CMake hasn't been told of an install dir, but we have an envvar, so we use that instead
    set(CMAKE_INSTALL_PREFIX $ENV{PHASM_HOME} CACHE PATH "Comment explaining this nonsense" FORCE)
else()
    # No explicit install dir, so we install to './install' to avoid clobbering /usr/local
    set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/install CACHE PATH "Comment explaining this nonsense" FORCE)
endif()


message(STATUS "-----------------------")
message(STATUS "Dependency prefix path is ${CMAKE_PREFIX_PATH}")
message(STATUS "Installation directory is ${CMAKE_INSTALL_PREFIX}")
message(STATUS "Build type is ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard is ${CMAKE_CXX_STANDARD}")
message(STATUS "-----------------------")

option(USE_ASAN "Compile with address sanitizer" OFF)
message(STATUS "USE_ASAN    ${USE_ASAN}")
if (${USE_ASAN})
    add_compile_options(-fsanitize=address)
    add_link_options(-fsanitize=address)
endif()

option(USE_CUDA "Compile with CUDA dependency" OFF)
message(STATUS "USE_CUDA    ${USE_CUDA}")

option(USE_TORCH "Compile with Torch dependency" ON)
message(STATUS "USE_TORCH   ${USE_TORCH}")

option(USE_JULIA "Compile with Julia dependency" OFF)
message(STATUS "USE_JULIA   ${USE_JULIA}")

option(USE_JANA "Compile with JANA2 dependency" ON)
message(STATUS "USE_JANA    ${USE_JANA}")

option(USE_LLVM "Compile with LLVM (e.g. Clang static analyzer) dependency" OFF)
message(STATUS "USE_LLVM    ${USE_LLVM}")

option(USE_PIN "Compile with Intel PIN support" OFF)
message(STATUS "USE_PIN     ${USE_PIN}")

option(USE_DWARF "Compile with libdwarf support" OFF)
message(STATUS "USE_DWARF   ${USE_DWARF}")

message(STATUS "-----------------------")

if (${USE_JULIA})
    find_package(Julia REQUIRED)
    message(STATUS "Found package Julia => ${Julia_DIR}")
endif()

if (${USE_JANA})
    find_package(JANA REQUIRED)
    find_package(Threads REQUIRED)
    message(STATUS "Found package JANA => ${JANA_DIR}")
endif()

if (${USE_TORCH})
    find_package(Torch REQUIRED)
    message(STATUS "Found package Torch => ${Torch_DIR}")
endif()


message(STATUS "-----------------------")

include_directories(external)

add_subdirectory(examples)
add_subdirectory(surrogate)
add_subdirectory(torch_plugin)
add_subdirectory(julia_plugin)
add_subdirectory(memtrace)
add_subdirectory(memtrace_pin)
#add_subdirectory(gpu_perf_tester)

message(STATUS "-----------------------")
