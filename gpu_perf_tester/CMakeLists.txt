
set(GPU_PERF_TESTER_SOURCES src/main.cpp src/gpu_perf_tester.cpp)
add_executable("gpu_perf_tester" ${GPU_PERF_TESTER_SOURCES})


install(TARGETS gpu_perf_tester DESTINATION bin)
