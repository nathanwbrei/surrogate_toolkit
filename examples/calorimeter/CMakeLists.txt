
if (NOT ${USE_GEANT4})
    message(STATUS "Skipping target 'phasm-example-calorimeter' because USE_GEANT4=Off")
else()
    message(STATUS "Including target 'phasm-example-calorimeter'")
    add_subdirectory(B4a)
    add_subdirectory(B4d)
    add_custom_target(phasm_example_calorimeter DEPENDS phasm_example_calorimeter_B4a phasm_example_calorimeter_B4d)
endif()
