
find_package(Torch REQUIRED)

set(LOADING_PT_EXAMPLE_SOURCES loading_pt.cpp)
add_executable(phasm-example-loading-pt
        ${LOADING_PT_EXAMPLE_SOURCES})
target_compile_definitions(phasm-example-loading-pt PUBLIC PHASM_USE_CUDA=${USE_CUDA})
target_include_directories(phasm-example-loading-pt PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(phasm-example-loading-pt ${TORCH_LIBRARIES})
install(TARGETS phasm-example-loading-pt DESTINATION bin)
