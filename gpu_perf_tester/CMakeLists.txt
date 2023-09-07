
set(GPU_PERF_TESTER_SOURCES src/main.cpp src/gpu_perf_tester.cpp)
add_executable("phasm-perftest" ${GPU_PERF_TESTER_SOURCES})
target_link_libraries(phasm-perftest phasm-surrogate)

install(TARGETS phasm-perftest DESTINATION bin)
