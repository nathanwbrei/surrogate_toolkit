
find_package(Torch REQUIRED)

set(LIKWID_INCLUDE_DIRS ${LIKWID_DIR}/include)
set(LIKWID_LIBRARY ${LIKWID_DIR}/lib/liblikwid.so)

set(DIFFUSION_PDE_SOLVER_SOURCES diffusion_pde_solver.cpp)
add_executable(phasm-example-pdesolver ${DIFFUSION_PDE_SOLVER_SOURCES})
target_compile_options(phasm-example-pdesolver PRIVATE "-DLIKWID_PERFMON")
target_include_directories(phasm-example-pdesolver PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../surrogate/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../../memtrace/include
        ${LIBKWID_INCLUDE_DIRS}
        )
target_link_libraries(phasm-example-pdesolver ${TORCH_LIBRARIES} phasm-surrogate ${LIKWID_LIBRARY})
install(TARGETS phasm-example-pdesolver DESTINATION bin)

