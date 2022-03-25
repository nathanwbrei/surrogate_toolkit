
#find_package(libdwarf REQUIRED)
#TODO: Access libdwarf as a CMake target instead
set(LIBDWARF_INCLUDE_DIRS ${LIBDWARF_DIR}/include/libdwarf)
set(LIBDWARF_LIBRARY ${LIBDWARF_DIR}/lib/libdwarf.a)

find_package(ZLIB)
add_executable(dwarf_example src/dwarf_example.cpp)
target_include_directories(dwarf_example PRIVATE ${LIBDWARF_INCLUDE_DIRS})
target_link_libraries(dwarf_example ${LIBDWARF_LIBRARY} ${ZLIB_LIBRARY})

set(VACUUM_TOOL_SOURCES src/vacuum_tool.cpp src/dwarf_utils.cpp)
add_executable("vacuum_tool" ${VACUUM_TOOL_SOURCES})
target_include_directories(vacuum_tool PRIVATE ${LIBDWARF_INCLUDE_DIRS})
target_link_libraries(vacuum_tool ${LIBDWARF_LIBRARY} ${ZLIB_LIBRARY})

set(VACUUM_TOOL_TEST_SOURCES test/dummy_tests.cpp)
add_executable("vacuum_tool_tests" ${VACUUM_TOOL_TEST_SOURCES})
target_include_directories(vacuum_tool_tests PRIVATE src ${LIBDWARF_INCLUDE_DIRS})
target_link_libraries(vacuum_tool_tests ${LIBDWARF_LIBRARY})

install(TARGETS vacuum_tool dwarf_example DESTINATION bin)

add_custom_command(OUTPUT obj-intel64/pinatrace.dylib
        COMMAND make
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src>
        )
