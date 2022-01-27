

set(TARGET_SOURCES main.cpp)
add_executable("vacuum_target" ${TARGET_SOURCES})

install(TARGETS vacuum_target DESTINATION bin)
