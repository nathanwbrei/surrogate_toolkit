
find_package(Torch REQUIRED)

add_library(phasm-torch-plugin SHARED
        src/torch_plugin_main.cc
        )

target_link_libraries(phasm-torch-plugin
        ${TORCH_LIBRARIES}
        phasm-surrogate
        )

set_target_properties(phasm-torch-plugin PROPERTIES PREFIX "" SUFFIX ".so")

install(TARGETS phasm-torch-plugin DESTINATION plugins)

