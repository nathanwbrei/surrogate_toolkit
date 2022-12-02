
find_package(Torch REQUIRED)

set(SURROGATE_LIBRARY_SOURCES
        src/surrogate.cpp
        src/model.cpp
        src/surrogate_builder.cpp
        src/tensor.cpp
        src/feedforward_model.cpp
        src/torchscript_model.cpp
        src/plugin_loader.cc
        )

add_library(phasm-surrogate STATIC ${SURROGATE_LIBRARY_SOURCES})
target_include_directories(phasm-surrogate
        PUBLIC include
        PRIVATE ../memtrace/include ${TORCH_INCLUDE_DIRS})

target_link_libraries(phasm-surrogate ${TORCH_LIBRARIES})
install(TARGETS phasm-surrogate DESTINATION lib)


set(SURROGATE_LIBRARY_TEST_SOURCES
        test/capturing_tests.cpp
        test/sampling_tests.cpp
        test/sampling_tests.cpp
        test/pytorch_model_tests.cpp
        test/pytorch_tests.cpp
        test/range_tests.cpp
        test/memorizing_tests.cpp
        test/prism_tests.cpp
        test/optics_tests.cpp
        test/fluent_tests.cpp
        test/tutorial_tests.cpp
        )
add_executable("phasm-surrogate-tests" ${SURROGATE_LIBRARY_TEST_SOURCES})
target_include_directories(phasm-surrogate-tests PRIVATE include ../memtrace/include)
target_link_libraries(phasm-surrogate-tests phasm-surrogate ${TORCH_LIBRARIES})


install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
        DESTINATION include/phasm)

