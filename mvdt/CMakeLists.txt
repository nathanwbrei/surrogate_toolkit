
if (NOT ${USE_LLVM})
    message(STATUS "Skipping target 'phasm-mvdt' because USE_LLVM=Off")
    return()
endif()

message(STATUS "Including target 'phasm-mvdt'")
set(MVDT_SOURCES src/main.cpp)
add_executable(phasm-mvdt ${MVDT_SOURCES})

target_include_directories(phasm-mvdt PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    # ${CMAKE_CURRENT_SOURCE_DIR}/../../surrogate/include
    # ${CMAKE_CURRENT_SOURCE_DIR}/../../memtrace/include
    )

target_link_libraries(phasm-mvdt)
install(TARGETS phasm-mvdt DESTINATION bin)

