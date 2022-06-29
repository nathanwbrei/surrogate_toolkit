
# Readme

## How to build
```bash
git clone https://github.com/nathanwbrei/phasm
cd phasm

# Install PyTorch, Intel Pin, and JANA2 dependencies
./download_deps.sh
export DEPS=`pwd`/deps

# Build everything except vacuum tool
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH="$DEPS/libtorch;$DEPS/JANA2/install" -DLIBDWARF_DIR="$DEPS/libdwarf-0.3.4/installdir" ..
make install

# To run one of the examples:
export LD_LIBRARY_PATH=$DEPS/libtorch/lib:$LD_LIBRARY_PATH
PHASM_CALL_MODE=CaptureAndDump bin/phasm-example-pdesolver


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

## Special directions for building on ifarm

You can't download pytorch from the ifarm nodes because the site is
blocked. Instead, download it from a different CUE machine or copy 
it over via scp. 

You need a compiler that supports C++14 for pytorch. The default gcc on
ifarm is 4.8.5, which isn't good enough. Obtain a newer gcc via:
```bash
module unload gcc
module load gcc/10.2.0
```
CMake chooses the gcc/g++ specified by $CC/$CXX, not by whatever is on the $PATH. 
The module files don't appear to be setting these environment variables correctly,
so we have to set them ourselves:
```bash
export CC=`which gcc`
export CXX=`which g++`
```

Farm machines need the pre-cxx11 ABI version of the precompiled PyTorch. 
Go into download_deps.sh and comment/uncomment the following lines:

```bash
#wget --no-check-certificate https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.11.0%2Bcpu.zip
wget --no-check-certificate https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-1.11.0%2Bcpu.zip
```

Finally, you might get an error when trying to run phasm saying that it 
can't find libtorch.so. In this case, you need to add it to your LD_LIBRARY_PATH
like so:
```bash
export LD_LIBRARY_PATH=$DEPS/libtorch/lib:$LD_LIBRARY_PATH
```

Now you can proceed with the generic directions for building and running phasm.
