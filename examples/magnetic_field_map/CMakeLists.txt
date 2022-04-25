

find_package(JANA REQUIRED)
find_package(Threads REQUIRED)
find_package(Torch REQUIRED)

set(MAGNETIC_FIELD_MAP_SOURCES
        magnetic_field_map.cpp
        DMagneticFieldMapFineMesh.cc
        HDEVIO.cc
        )

add_executable(magnetic_field_map ${MAGNETIC_FIELD_MAP_SOURCES})
target_include_directories(magnetic_field_map PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../surrogate_library/include ${JANA_INCLUDE_DIR})
target_link_libraries(magnetic_field_map ${TORCH_LIBRARIES} surrogate_library ${JANA_LIBRARY} Threads::Threads ${CMAKE_DL_LIBS})
install(TARGETS magnetic_field_map DESTINATION bin)

