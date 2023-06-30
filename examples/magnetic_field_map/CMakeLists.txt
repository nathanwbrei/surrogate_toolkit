
if (NOT ${USE_TORCH})

        message(STATUS "Skipping target 'phasm-example-magfieldmap' because USE_TORCH=Off")

elseif (NOT ${USE_JANA})

        message(STATUS "Skipping target 'phasm-example-magfieldmap' because USE_JANA=Off")

else()
        message(STATUS "Including target 'phasm-example-magfieldmap'")

        # find_package(JANA REQUIRED)
        # find_package(Threads REQUIRED)
        # find_package(Torch REQUIRED)

        set(MAGNETIC_FIELD_MAP_SOURCES
                magnetic_field_map.cpp
                halld_recon/DMagneticFieldMapFineMesh.cc
                halld_recon/HDEVIO.cc
                halld_recon/JCalibrationGenerator_TestFixture.cpp
                halld_recon/JCalibration_TestFixture.cpp
                )

        add_executable(phasm-example-magfieldmap ${MAGNETIC_FIELD_MAP_SOURCES})
        target_include_directories(phasm-example-magfieldmap PUBLIC
                ${CMAKE_CURRENT_SOURCE_DIR}/../../surrogate/include
                ${CMAKE_CURRENT_SOURCE_DIR}/../../memtrace/include
                ${JANA_INCLUDE_DIR}
                )
        target_link_libraries(phasm-example-magfieldmap ${TORCH_LIBRARIES} phasm-surrogate ${JANA_LIBRARY} Threads::Threads ${CMAKE_DL_LIBS})
        install(TARGETS phasm-example-magfieldmap DESTINATION bin)

endif()