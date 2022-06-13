
set(MEMTRACE_TEST_SOURCES
        dummy_tests.cpp
        interpreter_tests.cpp
        )

add_executable("phasm-memtrace-tests" ${MEMTRACE_TEST_SOURCES})
target_include_directories(phasm-memtrace-tests PRIVATE ../include ${LIBDWARF_INCLUDE_DIRS})
target_link_libraries(phasm-memtrace-tests phasm-memtrace ${LIBDWARF_LIBRARY})
