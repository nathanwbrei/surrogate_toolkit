
if(NOT ${USE_TORCH} OR NOT ${USE_REST})
    message(STATUS "Skipping target 'phasm-example-rest' because USE_TORCH=Off or USE_REST=Off")
else()
    message(STATUS "Including target 'phasm-example-rest'")

    set(MLFLOW_EXAMPLE_SOURCES
        mlflow_rest.cpp
    )

    add_executable(phasm-example-rest ${MLFLOW_EXAMPLE_SOURCES})

    set_target_properties(phasm-example-rest PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)

    target_include_directories(phasm-example-rest
        PUBLIC ${TORCH_INCLUDE_DIRS}
        PRIVATE  ${CMAKE_CURRENT_SOURCE_DIR}/../../install/lib
    )

    target_link_libraries(phasm-example-rest
        "${TORCH_LIBRARIES}"
        phasm-torch-plugin
        phasm-rest-plugin
        ${CMAKE_DL_LIBS}
    )

    install(TARGETS phasm-example-rest DESTINATION bin)

endif()
