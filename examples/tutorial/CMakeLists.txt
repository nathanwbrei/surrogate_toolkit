
find_package(Torch REQUIRED)

set(TUTORIAL_SOURCES tutorial.cpp)
add_executable(tutorial ${TUTORIAL_SOURCES})
target_include_directories(tutorial PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../surrogate_library/include)
target_link_libraries(tutorial ${TORCH_LIBRARIES} surrogate_library)
install(TARGETS tutorial DESTINATION bin)

