
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


configure_file(memtrace_pin.in phasm-memtrace-pin)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/phasm-memtrace-pin
        PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
        DESTINATION bin
        )

set(PINPLUGIN ${CMAKE_CURRENT_SOURCE_DIR}/../../memtrace_pin/obj-intel64/memtrace_pin_frontend.dylib)
if(EXISTS ${PINPLUGIN})
    message(${PINPLUGIN})
    install(FILES ${PINPLUGIN} DESTINATION lib)
endif()

set(PINPLUGIN ${CMAKE_CURRENT_SOURCE_DIR}/../../memtrace_pin/obj-intel64/memtrace_pin_frontend.so)
message(${PINPLUGIN})
if(EXISTS ${PINPLUGIN})
    install(FILES ${PINPLUGIN} DESTINATION lib)
endif()
