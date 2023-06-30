
if(NOT ${USE_TORCH})

        message(STATUS "Skipping target 'phasm-example-pdesolver' because USE_TORCH=Off")

else()
        message(STATUS "Including target 'phasm-example-pdesolver'")

        # find_package(Torch REQUIRED)
        # Handled at top-level

        set(DIFFUSION_PDE_SOLVER_SOURCES diffusion_pde_solver.cpp)
        add_executable(phasm-example-pdesolver ${DIFFUSION_PDE_SOLVER_SOURCES})
        target_include_directories(phasm-example-pdesolver PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}/../../surrogate/include
                ${CMAKE_CURRENT_SOURCE_DIR}/../../memtrace/include
                )
        target_link_libraries(phasm-example-pdesolver ${TORCH_LIBRARIES} phasm-surrogate)
        install(TARGETS phasm-example-pdesolver DESTINATION bin)

endif()

