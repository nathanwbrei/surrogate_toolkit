
find_package(Torch REQUIRED)

set(LOADING_PT_EXAMPLE_SOURCES loading_pt.cpp)
add_executable(phasm-example-loading-pt
        ${LOADING_PT_EXAMPLE_SOURCES})
target_link_libraries(phasm-example-loading-pt "${TORCH_LIBRARIES}")
install(TARGETS phasm-example-loading-pt DESTINATION bin)
