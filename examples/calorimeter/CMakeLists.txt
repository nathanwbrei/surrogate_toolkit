
if (NOT ${USE_GEANT4})
    message(STATUS "Skipping target 'phasm-example-calorimeter' because USE_GEANT4=Off")
else()
    message(STATUS "Including target 'phasm-example-calorimeter'")
    add_subdirectory(B4a)
    add_subdirectory(B4d)
    add_custom_target(phasm-example-calorimeter DEPENDS phasm-example-calorimeter-B4a phasm-example-calorimeter-B4d)
endif()


