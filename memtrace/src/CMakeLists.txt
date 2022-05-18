
#find_package(libdwarf REQUIRED)
#TODO: Access libdwarf as a CMake target instead?
set(LIBDWARF_INCLUDE_DIRS ${LIBDWARF_DIR}/include/libdwarf ${LIBDWARF_DIR}/include/libdwarf-0)
set(LIBDWARF_LIBRARY ${LIBDWARF_DIR}/lib/libdwarf.a)
find_package(ZLIB)

add_executable(dwarf_example dwarf_example.cpp)
target_include_directories(dwarf_example PRIVATE ${LIBDWARF_INCLUDE_DIRS})
target_link_libraries(dwarf_example ${LIBDWARF_LIBRARY} ${ZLIB_LIBRARY})


set(MEMTRACE_SOURCES
        interpreter.cpp
        dwarf_utils.cpp
        )
add_library(phasm-memtrace ${MEMTRACE_SOURCES})
target_include_directories(phasm-memtrace PRIVATE ../include ${LIBDWARF_INCLUDE_DIRS})
target_link_libraries(phasm-memtrace ${LIBDWARF_LIBRARY} ${ZLIB_LIBRARY})
install(TARGETS phasm-memtrace DESTINATION lib)


set(MEMTRACE_BACKEND_SOURCES
        memtrace_backend.cpp
        )
add_executable("phasm-memtrace-backend" ${MEMTRACE_BACKEND_SOURCES})
target_include_directories(phasm-memtrace-backend PRIVATE ../include ${LIBDWARF_INCLUDE_DIRS})
target_link_libraries(phasm-memtrace-backend phasm-memtrace ${LIBDWARF_LIBRARY} ${ZLIB_LIBRARY})
install(TARGETS phasm-memtrace-backend DESTINATION bin)

