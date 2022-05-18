

# Experimenting with creating a vacuum tool using Pin

## To run:
```bash
# From the phasm install directory
bin/phasm-memtrace-pin bin/phasm-example-memtrace
```

## What the run scripts are essentially doing:

```bash
export VACUUMTARGET=`pwd`/../build/examples/memtrace/phasm-example-memtrace
export VACUUMTOOL=`pwd`/obj-intel64/memtrace_pin_frontend.dylib
# Note: Should be memtrace_pin_frontend.so on Linux

# To run the vacuum tool against vacuum_target
$PIN_ROOT/pin -t $VACUUMTOOL -- $VACUUMTARGET
```


## Sources

[1]: https://www.intel.com/content/www/us/en/developer/articles/tool/pin-a-binary-instrumentation-tool-downloads.html
[2]: https://software.intel.com/sites/landingpage/pintool/docs/98547/Pin/html/index.html#BuildingInsideKit

