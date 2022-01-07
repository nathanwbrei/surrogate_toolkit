

find_package(JANA REQUIRED)

set(MAGNETIC_FIELD_MAP_SOURCES
        main.cpp
        DMagneticFieldMapFineMesh.cc
        HDEVIO.cc
        TVector3.cxx
        )

add_executable(magnetic_field_map ${MAGNETIC_FIELD_MAP_SOURCES})
target_include_directories(magnetic_field_map PUBLIC ${JANA_INCLUDE_DIR})
target_link_libraries(magnetic_field_map ${JANA_LIBRARY})
install(TARGETS magnetic_field_map DESTINATION bin)

