
set(Julia_VERSION 1.9)

find_path(Julia_INCLUDE_DIR
        NAMES "julia.h"
        PATHS ${Julia_DIR}/include/julia
        )

find_path(Julia_LIBRARY_DIR
        NAMES "libjulia.so"
        PATHS ${Julia_DIR}/lib
        )

set(Julia_LIBRARY_DIRS ${Julia_LIBRARY} ${Julia_LIBRARY}/julia)
set(Julia_LIBRARY ${Julia_LIBRARY_DIR}/libjulia.so)
set(Julia_INCLUDE_DIRS ${Julia_INCLUDE_DIR})


include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Julia
        FOUND_VAR Julia_FOUND
        VERSION_VAR Julia_VERSION
        REQUIRED_VARS Julia_DIR Julia_INCLUDE_DIR Julia_LIBRARY
        )

set(Julia_CFLAGS "-fPIC")
set(Julia_LDFLAGS "-Wl,--export-dynamic")
