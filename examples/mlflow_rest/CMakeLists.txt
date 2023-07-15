
if(NOT ${USE_TORCH})
    message(STATUS "Skipping target 'phasm-example-rest' because USE_TORCH=Off")
else()
    message(STATUS "Including target 'phasm-example-rest'")

    find_package(cpprestsdk REQUIRED)  # TODO (@xmei.jlab.org): Add to top-level options 

    set(MLFLOW_REST_SOURCES
        mlflow_rest.cpp
    )
    add_executable(phasm-example-rest ${MLFLOW_REST_SOURCES})
    set_target_properties(phasm-example-rest PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
    target_include_directories(phasm-example-rest
        PUBLIC ${TORCH_INCLUDE_DIRS}        
    )
    target_link_libraries(phasm-example-rest
        ${TORCH_LIBRARIES} phasm-torch-plugin
        OpenSSL::SSL OpenSSL::Crypto cpprestsdk::cpprest
        ${CMAKE_DL_LIBS}
    )
    install(TARGETS phasm-example-rest DESTINATION bin)
endif()
