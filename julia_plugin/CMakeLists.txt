
if(NOT ${USE_JULIA})

    message(STATUS "Skipping target 'phasm-julia-plugin' because USE_JULIA=Off")
    message(STATUS "Skipping target 'phasm-julia-plugin-tests' because USE_JULIA=Off")

else()

    message(STATUS "Including target 'phasm-julia_plugin'")

    # find_package(Julia)
    # Already happened in top-level CMakeLists

    set(PHASM_JULIA_PLUGIN_SOURCES
            src/julia_plugin_main.cc
            src/julia_model.cc
            src/julia_modelvars.cc
            )

    add_library(phasm-julia-plugin SHARED ${PHASM_JULIA_PLUGIN_SOURCES})
    target_include_directories(phasm-julia-plugin PUBLIC include ${Julia_INCLUDE_DIR})
    target_link_libraries(phasm-julia-plugin ${Julia_LIBRARY} phasm-surrogate)
    target_compile_options(phasm-julia-plugin PUBLIC ${Julia_CFLAGS})
    target_link_options(phasm-julia-plugin PUBLIC ${Julia_LDFLAGS})
    #target_link_directories(phasm-julia-plugin PUBLIC ${Julia_LIBRARY_DIRS})

    set(PHASM_JULIA_PLUGIN_TEST_SOURCES test/julia_tests.cc)
    add_executable("phasm-julia-plugin-tests" ${PHASM_JULIA_PLUGIN_TEST_SOURCES})
    target_include_directories(phasm-julia-plugin-tests PRIVATE include)
    target_link_libraries(phasm-julia-plugin-tests phasm-julia-plugin)

endif()

