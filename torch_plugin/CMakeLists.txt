
find_package(Torch REQUIRED)

add_library(phasm-torch-plugin SHARED
        src/torch_plugin_main.cc
        src/torch_tensor_utils.cc
        src/feedforward_model.cpp
        src/torchscript_model.cpp
        )

target_include_directories(phasm-torch-plugin PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(phasm-torch-plugin
        ${TORCH_LIBRARIES}
        phasm-surrogate
        )

set_target_properties(phasm-torch-plugin PROPERTIES PREFIX "" SUFFIX ".so")

install(TARGETS phasm-torch-plugin DESTINATION plugins)

