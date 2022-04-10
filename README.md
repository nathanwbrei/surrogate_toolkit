
# Readme

## How to build
```bash
git clone https://github.com/nathanwbrei/surrogate_toolkit
cd surrogate_toolkit

# Install PyTorch, Intel Pin, and JANA2 dependencies
./download_deps.sh
export DEPS=`pwd`/deps

# Build everything except vacuum tool
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH="$DEPS/libtorch;$DEPS/JANA2/install" -DLIBDWARF_DIR="$DEPS/libdwarf-0.3.4/install" ..
make install

# Build vacuum tool 
# This is tricky because it uses a complicated Makefile thanks to PIN
cd ..
cd vacuum_tool/src
export PIN_ROOT=$DEPS/Pin/pin-3.22-98547-g7a303a835-clang-mac/
make
export VACUUMTARGET=`pwd`/../build/vacuum_target/vacuum_target
export VACUUMTOOL=`pwd`/obj-intel64/vacuum_pin_plugin.dylib

# To run the vacuum tool against vacuum_target
# $PIN_ROOT/pin -t $VACUUMTOOL -- $VACUUMTARGET

```
