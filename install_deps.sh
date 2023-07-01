#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Error: Wrong number of arguments"
    echo "Usage: $0 <DOWNLOAD_DIR> <INSTALL_DIR>"
    echo ""
    exit 1
fi

DOWNLOAD_DIR=$(readlink -f $1)
INSTALL_DIR=$(readlink -f $2)
echo "Download dir: $DOWNLOAD_DIR"
echo "Install dir:  $INSTALL_DIR"


# exit when any command fails in the script
set -e

function finish {
  RESULT=$?
  if [[ ! $RESULT -eq 0 ]]; then
    echo "Error: '${last_command}' returned with exit code $RESULT."
  fi
}

# Record the last command
trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG
# echo the error message given before exiting
trap finish EXIT


# Figure out whether we are on linux or macos
if [[ "$OSTYPE" == "darwin"* ]]; then
  MACOS=1
  echo "Detected system = macOS"
else
  MACOS=0
  echo "Assuming system = Linux"
fi

mkdir -p $DOWNLOAD_DIR
mkdir -p $INSTALL_DIR
cd $DOWNLOAD_DIR

# Install PyTorch
rm -rf $INSTALL_DIR/Torch
unzip libtorch.zip
mv libtorch $INSTALL_DIR/Torch

# Install PIN
rm -rf $INSTALL_DIR/PIN
mkdir $INSTALL_DIR/PIN
tar -xf pin.tar.gz -C $INSTALL_DIR/PIN --strip-components 1

# Build and install JANA2
rm -rf JANA
unzip JANA2.zip
mv -f JANA2-2.0.6 JANA
mkdir -p JANA/install
export JANA_HOME=$INSTALL_DIR
mkdir JANA/build
cd JANA/build
cmake .. -DCMAKE_INSTALL_PREFIX=$JANA_HOME
make -j8 install
cd ../..

# Build and install libdwarf
rm -rf libdwarf
mkdir libdwarf
tar -xf libdwarf.tar.xz -C libdwarf --strip-components 1
mkdir libdwarf/build
cd libdwarf/build
cmake .. -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR
make install

echo "Success!"
echo "Pass the following variables to CMake:"
echo "-DCMAKE_PREFIX_PATH=\"$INSTALL_DIR\""
