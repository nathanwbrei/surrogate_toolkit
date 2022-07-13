

find_package(JANA REQUIRED)
find_package(Threads REQUIRED)
find_package(Torch REQUIRED)

set(MAGNETIC_FIELD_MAP_SOURCES
        magnetic_field_map.cpp
        DMagneticFieldMapFineMesh.cc
        HDEVIO.cc
        JCalibrationGenerator_TestFixture.cpp
        JCalibration_TestFixture.cpp
        )

add_executable(phasm-example-magfieldmap ${MAGNETIC_FIELD_MAP_SOURCES})
target_include_directories(phasm-example-magfieldmap PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../../surrogate/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../../memtrace/include
        ${JANA_INCLUDE_DIR}
        )
target_link_libraries(phasm-example-magfieldmap ${TORCH_LIBRARIES} phasm-surrogate ${JANA_LIBRARY} Threads::Threads ${CMAKE_DL_LIBS})
install(TARGETS phasm-example-magfieldmap DESTINATION bin)

