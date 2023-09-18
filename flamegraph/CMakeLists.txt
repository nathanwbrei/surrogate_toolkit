

set(PHASM_FLAMEGRAPH_SOURCES src/main.cpp)
add_executable("phasm-flamegraph" ${PHASM_FLAMEGRAPH_SOURCES})
target_link_libraries(phasm-flamegraph phasm-surrogate)

install(TARGETS phasm-flamegraph DESTINATION bin)
