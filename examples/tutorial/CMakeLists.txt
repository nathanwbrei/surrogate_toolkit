
message(STATUS "Including target 'phasm-example-tutorial'")
set(TUTORIAL_SOURCES tutorial.cpp)
add_executable(phasm-example-tutorial ${TUTORIAL_SOURCES})
target_include_directories(phasm-example-tutorial PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../surrogate/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../../memtrace/include
        )
target_link_libraries(phasm-example-tutorial ${TORCH_LIBRARIES} phasm-surrogate)
install(TARGETS phasm-example-tutorial DESTINATION bin)

