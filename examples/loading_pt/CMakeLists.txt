
if (NOT ${USE_TORCH})
    message(STATUS "Skipping target 'phasm-example-loading-pt' because USE_TORCH=Off")
    return()
endif()

message(STATUS "Including target 'phasm-example-loading-pt'")

#find_package(Torch REQUIRED)
# Handled in top-level

set(LOADING_PT_EXAMPLE_SOURCES loading_pt.cpp)
add_executable(phasm-example-loading-pt ${LOADING_PT_EXAMPLE_SOURCES})
set_target_properties(phasm-example-loading-pt PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
target_include_directories(phasm-example-loading-pt PUBLIC ${TORCH_INCLUDE_DIRS})
target_link_libraries(phasm-example-loading-pt "${TORCH_LIBRARIES}" phasm-torch-plugin ${CMAKE_DL_LIBS})
install(TARGETS phasm-example-loading-pt DESTINATION bin)
