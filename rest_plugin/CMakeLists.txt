
if (NOT ${USE_REST})
    message(STATUS "Skipping target 'phasm-rest-plugin' because USE_REST=Off")
    return()
endif()

message(STATUS "Including target 'phasm-rest-plugin'")

set(PHASM_REST_PLUGIN_SOURCES
        src/mlflow_client.cpp
)

add_library(phasm-rest-plugin SHARED ${PHASM_REST_PLUGIN_SOURCES})
set_target_properties(phasm-rest-plugin PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
# For RPATH details, see
# https://gitlab.kitware.com/cmake/community/-/wikis/doc/cmake/RPATH-handling

target_include_directories(phasm-rest-plugin PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        )

target_link_libraries(phasm-rest-plugin
        OpenSSL::SSL OpenSSL::Crypto cpprestsdk::cpprest
        ${CMAKE_DL_LIBS}
        )

install(TARGETS phasm-rest-plugin DESTINATION lib)

#### Unit testing
# set(PHASM_REST_PLUGIN_TEST_SOURCES
#         test/mlflow_rest_tests.cpp
# )

# add_executable("phasm-rest-plugin-tests" ${PHASM_REST_PLUGIN_TEST_SOURCES})
# target_include_directories(phasm-rest-plugin-tests PRIVATE include ${TORCH_INCLUDE_DIRS})
# target_link_libraries(phasm-rest-plugin-tests
        # ${TORCH_LIBRARIES}
        # phasm-surrogate
        # phasm-torch-plugin
        # phasm-rest-plugin
        # )
