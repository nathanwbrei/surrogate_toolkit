if (NOT ${USE_DWARF})
        message(STATUS "Skipping target 'phasm-memtrace-tests' because USE_DWARF=Off")
        return()
endif()

message(STATUS "Including target 'phasm-memtrace-tests'")

set(MEMTRACE_TEST_SOURCES
        dummy_tests.cpp
        interpreter_tests.cpp
        )

add_executable("phasm-memtrace-tests" ${MEMTRACE_TEST_SOURCES})
target_include_directories(phasm-memtrace-tests PRIVATE ../include ${LIBDWARF_INCLUDE_DIRS})
target_link_libraries(phasm-memtrace-tests phasm-memtrace ${LIBDWARF_LIBRARY})
