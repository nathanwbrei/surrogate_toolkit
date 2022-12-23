
add_subdirectory(pde_solver)
add_subdirectory(magnetic_field_map)
add_subdirectory(memtrace)
add_subdirectory(tracking)
add_subdirectory(tutorial)

if(${USE_CUDA})
    add_subdirectory(pinn_pde_solver)
    add_subdirectory(loading_pt)
endif()

