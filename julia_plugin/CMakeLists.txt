
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
    set_target_properties(phasm-julia-plugin PROPERTIES INSTALL_RPATH_USE_LINK_PATH True)
    install(TARGETS phasm-julia-plugin DESTINATION lib)
    #target_link_directories(phasm-julia-plugin PUBLIC ${Julia_LIBRARY_DIRS})

    set(PHASM_JULIA_PLUGIN_TEST_SOURCES test/julia_tests.cc)
    add_executable("phasm-julia-plugin-tests" ${PHASM_JULIA_PLUGIN_TEST_SOURCES})
    target_include_directories(phasm-julia-plugin-tests PRIVATE include)
    target_link_libraries(phasm-julia-plugin-tests phasm-julia-plugin)
    set_target_properties(phasm-julia-plugin-tests PROPERTIES INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib)
    
    install(TARGETS phasm-julia-plugin-tests DESTINATION share/tests/julia-plugin-tests)
    install(FILES julia/TestModel.jl DESTINATION share/tests/julia-plugin-tests)
    install(FILES julia/Phasm.jl DESTINATION share/tests/julia-plugin-tests)

endif()

