
if (NOT ${USE_TORCH})
    message(STATUS "Skipping target 'phasm-example-pinn-pdesolver' because USE_TORCH=Off")

elseif(NOT ${USE_CUDA})
    message(STATUS "Skipping target 'phasm-example-pinn-pdesolver' because USE_CUDA=Off")

else()
    message(STATUS "Including target 'phasm-example-pinn-pdesolver'")

    # find_package(Torch REQUIRED)
    # Handled at toplevel

    set(PINN_DIFFUSION_PDE_SOLVER_SOURCES pinn_diffusion_pde_solver.cpp)
    add_executable(phasm-example-pinn-pdesolver ${PINN_DIFFUSION_PDE_SOLVER_SOURCES})
    set_target_properties(phasm-example-pinn-pdesolver PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
    target_include_directories(phasm-example-pinn-pdesolver PUBLIC ${TORCH_INCLUDE_DIRS})
    target_link_libraries(phasm-example-pinn-pdesolver "${TORCH_LIBRARIES}" phasm-torch-plugin)
    install(TARGETS phasm-example-pinn-pdesolver DESTINATION bin)

endif()