
find_package(Torch REQUIRED)


set(SURROGATE_LIBRARY_SOURCES src/surrogate_builder.cpp)
add_library(surrogate_library STATIC ${SURROGATE_LIBRARY_SOURCES})
target_include_directories(surrogate_library PRIVATE include ${TORCH_INCLUDE_DIRS})
target_link_libraries(surrogate_library ${TORCH_LIBRARIES})
install(TARGETS surrogate_library DESTINATION lib)


set(SURROGATE_LIBRARY_TEST_SOURCES
        test/experiment_tests.cpp
        test/capturing_tests.cpp
        test/sampling_tests.cpp
        test/sampling_tests.cpp
        test/pytorch_model_tests.cpp
        )
add_executable("surrogate_library_tests" ${SURROGATE_LIBRARY_TEST_SOURCES})
target_include_directories(surrogate_library_tests PRIVATE include)
target_link_libraries(surrogate_library_tests surrogate_library)


