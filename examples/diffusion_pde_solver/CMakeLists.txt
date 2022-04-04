
set(DIFFUSION_PDE_SOLVER_SOURCES diffusion_pde_solver.cpp)
add_executable(diffusion_pde_solver ${DIFFUSION_PDE_SOLVER_SOURCES})
install(TARGETS diffusion_pde_solver DESTINATION bin)

