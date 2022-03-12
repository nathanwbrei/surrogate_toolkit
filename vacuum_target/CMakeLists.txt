
set(TARGET_SOURCES vacuum_target.cpp)
add_executable(vacuum_target ${TARGET_SOURCES})
target_compile_options(vacuum_target PUBLIC "-g" "-O0")
target_link_options(vacuum_target PUBLIC "-g")
install(TARGETS vacuum_target DESTINATION bin)

