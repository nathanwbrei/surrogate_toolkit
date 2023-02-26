
find_package(Torch REQUIRED)

set(PHASM_TORCH_PLUGIN_SOURCES
        src/torch_plugin_main.cc
        src/torch_tensor_utils.cc
        src/feedforward_model.cpp
        src/torchscript_model.cpp
        )

if(${USE_CUDA})
    list(APPEND PHASM_TORCH_PLUGIN_SOURCES
            src/torch_cuda_utils.cc
            )
endif()

add_library(phasm-torch-plugin SHARED ${PHASM_TORCH_PLUGIN_SOURCES})
set_target_properties(phasm-torch-plugin PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
# phasm-torch-plugin.so's (install) rpath should include libtorch, so that we don't have to edit our LD_LIBRARY_PATH
# all the time. We assume that this is neither in the build tree, nor in the system libs. For more details, see
# https://gitlab.kitware.com/cmake/community/-/wikis/doc/cmake/RPATH-handling

target_include_directories(phasm-torch-plugin PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${TORCH_INCLUDE_DIRS})

target_link_libraries(phasm-torch-plugin
        ${TORCH_LIBRARIES}
        phasm-surrogate
        )

set_target_properties(phasm-torch-plugin PROPERTIES PREFIX "" SUFFIX ".so")

install(TARGETS phasm-torch-plugin DESTINATION plugins)


set(PHASM_TORCH_PLUGIN_TEST_SOURCES
        test/pytorch_model_tests.cpp
        test/pytorch_tests.cpp
    )

add_executable("phasm-torch-plugin-tests" ${PHASM_TORCH_PLUGIN_TEST_SOURCES})
target_include_directories(phasm-torch-plugin-tests PRIVATE include ../memtrace/include ${TORCH_INCLUDE_DIRS})
target_link_libraries(phasm-torch-plugin-tests ${TORCH_LIBRARIES} phasm-surrogate phasm-torch-plugin)

