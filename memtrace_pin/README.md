

# Experimenting with creating a vacuum tool using Pin

## To build:

1. Download Pin from [1]
2. Set envvar PIN_ROOT=$PATH_TO_PIN [2]
3. `make`. No CMake support yet. Builds in-source to `obj-intel64` directory
4. $PIN_ROOT/pin -t obj-intel64/pinatrace.dylib -- ../../cmake-build-debug-local/vacuum_target/vaccum_target

```bash
export PIN_ROOT=/Users/nbrei/projects/surrogate/Pin/pin-3.22-98547-g7a303a835-clang-mac/
export VACUUMTARGET=/Users/nbrei/projects/surrogate/surrogate_toolkit/cmake-build-debug-local/vacuum_tool/vacuum_target
export VACUUMTOOL=/Users/nbrei/projects/surrogate/surrogate_toolkit/vacuum_tool/src/obj-intel64/vacuum_pin_plugin.dylib
make
$PIN_ROOT/pin -t $VACUUMTOOL -- $VACUUMTARGET
```

## Sources

[1]: https://www.intel.com/content/www/us/en/developer/articles/tool/pin-a-binary-instrumentation-tool-downloads.html
[2]: https://software.intel.com/sites/landingpage/pintool/docs/98547/Pin/html/index.html#BuildingInsideKit

