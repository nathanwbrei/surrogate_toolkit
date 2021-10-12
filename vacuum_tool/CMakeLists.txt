
set(VACUUM_TOOL_SOURCES src/main.cpp src/vacuum.cpp)
add_executable("vacuum_tool" ${VACUUM_TOOL_SOURCES})

set(VACUUM_TOOL_TEST_SOURCES test/dummy_tests.cpp)
add_executable("vacuum_tool_tests" ${VACUUM_TOOL_TEST_SOURCES})

install(TARGETS vacuum_tool DESTINATION bin)
