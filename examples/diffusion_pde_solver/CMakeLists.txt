
find_package(Torch REQUIRED)

set(DIFFUSION_PDE_SOLVER_SOURCES diffusion_pde_solver.cpp)
add_executable(diffusion_pde_solver ${DIFFUSION_PDE_SOLVER_SOURCES})
target_include_directories(diffusion_pde_solver PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../surrogate_library/include)
target_link_libraries(diffusion_pde_solver ${TORCH_LIBRARIES} surrogate_library)
install(TARGETS diffusion_pde_solver DESTINATION bin)

