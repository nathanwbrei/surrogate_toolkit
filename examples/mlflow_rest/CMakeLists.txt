
if(NOT ${USE_TORCH})

        message(STATUS "Skipping target 'phasm-example-rest' because USE_TORCH=Off")

else()
        message(STATUS "Including target 'phasm-example-rest'")
       
        find_package(cpprestsdk REQUIRED)  # TODO (@xmei.jlab.org): Add to top-level options 

        set(MLFLOW_REST_SOURCES
            mlflow_rest.cpp
            )
        add_executable(phasm-example-rest ${MLFLOW_REST_SOURCES})
        target_include_directories(phasm-example-rest PUBLIC
                ${CMAKE_CURRENT_SOURCE_DIR}/../../surrogate/include
                )
        target_link_libraries(phasm-example-rest
            PRIVATE OpenSSL::SSL OpenSSL::Crypto cpprestsdk::cpprest
            ${TORCH_LIBRARIES} phasm-surrogate
            ${CMAKE_DL_LIBS}
            )
        install(TARGETS phasm-example-rest DESTINATION bin)
endif()
