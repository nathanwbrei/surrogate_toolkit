

# Experimenting with creating a vacuum tool using Pin

## To build:

1. Download Pin from [1]
2. Set envvar PIN_ROOT=$PATH_TO_PIN [2]
3. `make`. No CMake support yet. Builds in-source to `obj-intel64` directory
4. $PIN_ROOT/pin -t obj-intel64/pinatrace.dylib -- ../../cmake-build-debug-local/vacuum_target/vaccum_target

## Sources

[1]: https://www.intel.com/content/www/us/en/developer/articles/tool/pin-a-binary-instrumentation-tool-downloads.html
[2]: https://software.intel.com/sites/landingpage/pintool/docs/98547/Pin/html/index.html#BuildingInsideKit

