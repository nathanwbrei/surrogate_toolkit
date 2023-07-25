
if (NOT ${USE_LLVM})
    message(STATUS "Skipping target 'phasm-mvdt' because USE_LLVM=Off")
    return()
endif()

message(STATUS "Including target 'phasm-mvdt'")

add_compile_options(
  "$<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-fno-rtti;-Wall>"
  "$<$<AND:$<CXX_COMPILER_ID:GNU>,$<CONFIG:DEBUG>>:-fno-omit-frame-pointer>"
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib")

############## LLVM CONFIGURATION #################

# LLVM_DIR must be set to the prefix of /share/llvm/cmake via commandline
find_package(LLVM REQUIRED CONFIG HINTS ${LLVM_DIR})
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

# We incorporate the CMake features provided by LLVM:
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(AddLLVM)

# LLVM_DIR must be set to the prefix of /share/llvm/cmake via commandline
find_package(Clang REQUIRED)

message("LLVM STATUS:
  Definitions ${LLVM_DEFINITIONS}
  Includes    ${LLVM_INCLUDE_DIRS}
              ${CLANG_INCLUDE_DIRS}
  Libraries   ${LLVM_LIBRARY_DIRS}"
)

# Now set the LLVM header and library paths:
include_directories(SYSTEM ${LLVM_INCLUDE_DIRS} ${CLANG_INCLUDE_DIRS})
link_directories(${LLVM_LIBRARY_DIRS})
add_definitions(${LLVM_DEFINITIONS})

############## FINAL PROJECT CONFIG #################

# And the project header and library paths
set(CMAKE_TEMP_LIBRARY_PATH "${PROJECT_BINARY_DIR}/lib")



# Everything here is for the SimpleStreamChecker example

add_library(streamchecker MODULE
  src/SimpleStreamChecker.cpp
)

set_target_properties(streamchecker PROPERTIES
  CXX_VISIBILITY_PRESET hidden
)

target_include_directories(streamchecker PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(streamchecker PUBLIC 
  clangLex
  clangCrossTU
  clangAST
  clangStaticAnalyzerFrontend
  clangStaticAnalyzerCheckers
  clangStaticAnalyzerCore 
  clangAnalysis
  clangASTMatchers
  clangAST
  clangBasic
  LLVMSupport
  clangCrossTU
  clangLex
)

install(TARGETS streamchecker DESTINATION lib)



# Everything here is for the function-printer example

add_library(function-printer-plugin MODULE
  src/PrintFunctionsAction.cpp
)

target_include_directories(function-printer-plugin
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_library(function-printer STATIC
  src/PrintFunctionsAction.cpp
)

target_include_directories(function-printer
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
)

install(TARGETS function-printer DESTINATION lib)



# Executable PrintFunctions

add_executable(print-functions
  src/PrintFunctionsMain.cpp
)

if (LLVM_ENABLE_ASSERTIONS)
  add_definitions(-DLLVM_ENABLE_ASSERTIONS=${LLVM_ENABLE_ASSERTIONS})
endif()

llvm_map_components_to_libnames(REQ_LLVM_LIBRARIES ${LLVM_TARGETS_TO_BUILD}
                                core support option
)

target_link_libraries(print-functions
  function-printer
  clangAST
  clangBasic
  clangCodeGen
  clangLex
  clangParse
  clangSema
  clangAnalysis
  clangDriver
  clangFrontend
  clangAST
  clangEdit
  clangSerialization
  clangTooling
  clangToolingCore
  clangAST
  clangParse
  clangBasic
  clangSema
  clangBasic
  clangAST
  ${REQ_LLVM_LIBRARIES}
)

target_include_directories(print-functions
  PRIVATE $<TARGET_PROPERTY:function-printer,INTERFACE_INCLUDE_DIRECTORIES>
)

find_package(Threads REQUIRED)
find_package(Curses REQUIRED)
target_link_libraries(print-functions
    ${CMAKE_THREAD_LIBS_INIT}
    ${CMAKE_DL_LIBS}
    ${CURSES_LIBRARIES}
)

set_target_properties(print-functions
                      PROPERTIES
                      LINKER_LANGUAGE CXX
                      PREFIX ""
)

install(TARGETS print-functions
  RUNTIME DESTINATION bin
)



# Executable SimpleStreamChecker


file(GENERATE
     OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/config.h"
     CONTENT
"
#ifndef RUNPLUGIN_CONFIG_H
#define RUNPLUGIN_CONFIG_H

#define PLUGIN_LIBRARY \"$<TARGET_FILE_NAME:streamchecker>\"
#define CMAKE_TEMP_LIBRARY_PATH \"${CMAKE_TEMP_LIBRARY_PATH}\"
#define LLVM_LIBRARY_DIRS \"${LLVM_LIBRARY_DIRS}\"

#endif
"
)
include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_executable(runstreamchecker
  src/SimpleStreamCheckerMain.cpp
)

if (LLVM_ENABLE_ASSERTIONS)
  add_definitions(-DLLVM_ENABLE_ASSERTIONS=${LLVM_ENABLE_ASSERTIONS})
endif()

llvm_map_components_to_libnames(REQ_LLVM_LIBRARIES ${LLVM_TARGETS_TO_BUILD}
                                core support option
)

# NOTE: The order here was determined by configuring a build of clang with CMake
# and extracting the ordering from `libclang/CMakeFiles/libclang.dir/link.txt`
target_link_libraries(runstreamchecker
  PRIVATE
    clangAST
    clangBasic
    clangDriver
    clangFrontend
    clangIndex
    clangLex
    clangSema
    clangSerialization
    clangTooling

    clangFormat
    clangToolingInclusions
    clangToolingCore
    clangFrontend
    clangDriver

    clangRewriteFrontend
    clangStaticAnalyzerFrontend
    clangCrossTU
    clangStaticAnalyzerCheckers
    clangStaticAnalyzerCore
    clangAST

    clangParse
    clangSerialization
    clangSema
    clangEdit
    clangRewrite
    clangAnalysis
    clangASTMatchers
    clangAST
    clangLex
    clangBasic

    ${REQ_LLVM_LIBRARIES}
)

target_include_directories(runstreamchecker
  PRIVATE
    $<TARGET_PROPERTY:streamchecker,INTERFACE_INCLUDE_DIRECTORIES>
)

find_package(Threads REQUIRED)
find_package(Curses REQUIRED)
target_link_libraries(runstreamchecker
PRIVATE
    ${CMAKE_THREAD_LIBS_INIT}
    ${CMAKE_DL_LIBS}
    ${CURSES_LIBRARIES}
)


set_target_properties(runstreamchecker
  PROPERTIES
  LINKER_LANGUAGE CXX
  PREFIX ""
)

install(TARGETS runstreamchecker
  RUNTIME DESTINATION bin
)

# Test cases for the two example plugins

add_library(testlib
  test/files.c
  test/functions.c
)

set_target_properties(testlib PROPERTIES EXPORT_COMPILE_COMMANDS On)


# This is just the skeleton

set(MVDT_SOURCES src/main.cpp src/AnalysisResult.cpp)
add_executable(phasm-mvdt ${MVDT_SOURCES})

target_include_directories(phasm-mvdt PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    # ${CMAKE_CURRENT_SOURCE_DIR}/../../surrogate/include
    # ${CMAKE_CURRENT_SOURCE_DIR}/../../memtrace/include
    )

target_link_libraries(phasm-mvdt)
install(TARGETS phasm-mvdt DESTINATION bin)
