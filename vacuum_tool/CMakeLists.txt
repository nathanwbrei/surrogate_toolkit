
#find_package(libdwarf REQUIRED)
#TODO: Access libdwarf as a CMake target instead
set(LIBDWARF_INCLUDE_DIRS ${LIBDWARF_DIR}/include/libdwarf ${LIBDWARF_DIR}/include/libdwarf-0)
set(LIBDWARF_LIBRARY ${LIBDWARF_DIR}/lib/libdwarf.a)

find_package(ZLIB)
add_executable(dwarf_example src/dwarf_example.cpp)
target_include_directories(dwarf_example PRIVATE ${LIBDWARF_INCLUDE_DIRS})
target_link_libraries(dwarf_example ${LIBDWARF_LIBRARY} ${ZLIB_LIBRARY})

set(VACUUM_TOOL_SOURCES src/vacuum_backend.cpp src/dwarf_utils.cpp)
add_executable("vacuum_backend" ${VACUUM_TOOL_SOURCES})
target_include_directories(vacuum_backend PRIVATE ${LIBDWARF_INCLUDE_DIRS})
target_link_libraries(vacuum_backend ${LIBDWARF_LIBRARY} ${ZLIB_LIBRARY})

set(VACUUM_TOOL_TEST_SOURCES test/dummy_tests.cpp)
add_executable("vacuum_tool_tests" ${VACUUM_TOOL_TEST_SOURCES})
target_include_directories(vacuum_tool_tests PRIVATE src ${LIBDWARF_INCLUDE_DIRS})
target_link_libraries(vacuum_tool_tests ${LIBDWARF_LIBRARY})

set(TARGET_SOURCES src/vacuum_target.cpp)
add_executable(vacuum_target ${TARGET_SOURCES})
target_compile_options(vacuum_target PUBLIC "-g" "-O0")
target_link_options(vacuum_target PUBLIC "-g")
install(TARGETS vacuum_target DESTINATION bin)

configure_file(src/vacuum_tool.in vacuum_tool
               FILE_PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ)

install(TARGETS vacuum_backend dwarf_example DESTINATION bin)
install(FILES ${CMAKE_BINARY_DIR}/vacuum_tool/vacuum_tool DESTINATION bin)

set(VACUUM_PIN_PLUGIN_SOURCES vacuum_pin_plugin.cpp)
add_custom_command(
        OUTPUT src/obj-intel64/vacuum_pin_plugin.dylib
        COMMAND echo "Running make Vacuum tool plugin" #make
        DEPENDS ${VACUUM_PIN_PLUGIN_SOURCES}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src>
        VERBATIM
        )

#add_dependencies(vacuum_backend src/obj-intel64/vacuum_pin_plugin.dylib)

install(FILES src/obj-intel64/vacuum_pin_plugin.dylib DESTINATION bin)