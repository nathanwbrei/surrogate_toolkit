

set(TARGET_SOURCES main.cpp)
add_executable("vacuum_target" ${TARGET_SOURCES})

# TODO: Include debug symbols all the time, regardless of project settings?

install(TARGETS vacuum_target DESTINATION bin)