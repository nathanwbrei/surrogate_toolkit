
configure_file(memtrace_pin.in phasm-memtrace-pin)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/phasm-memtrace-pin
        PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
        DESTINATION bin
        )

if (APPLE)
    set(PINPLUGIN ${CMAKE_CURRENT_SOURCE_DIR}/obj-intel64/memtrace_pin_frontend.dylib)
else()
    set(PINPLUGIN ${CMAKE_CURRENT_SOURCE_DIR}/obj-intel64/memtrace_pin_frontend.so)
endif()


add_custom_command(OUTPUT ${PINPLUGIN}
        COMMAND PIN_ROOT=${PIN_ROOT} make
        COMMENT "Building phasm_memtrace_pin plugin: ${PINPLUGIN}"
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
add_custom_target(phasm_memtrace_pin ALL
            DEPENDS ${PINPLUGIN} phasm-memtrace-pin
)
install(FILES ${PINPLUGIN} DESTINATION lib)




