
add_subdirectory(src)
add_subdirectory(test)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
        DESTINATION include/phasm)