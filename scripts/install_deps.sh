#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Error: Wrong number of arguments"
    echo "Usage: $0 <DOWNLOAD_DIR> <INSTALL_DIR>"
    echo ""
    exit 1
fi

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


declare_use_dep() {
    local usevar=$1
    local default=$2
    if [[ -v $usevar ]]; then
        echo "$usevar=${!usevar} (user override)"
    else
        export $usevar=$default
        echo "$usevar=${!usevar} (default)"
    fi
}

declare_use_dep USE_TORCH ON
declare_use_dep USE_JULIA ON
declare_use_dep USE_JANA ON
declare_use_dep USE_PIN OFF
declare_use_dep USE_DWARF OFF

DOWNLOAD_DIR=$(readlink -f $1)
INSTALL_DIR=$(readlink -f $2)
echo "Download dir: $DOWNLOAD_DIR"
echo "Install dir:  $INSTALL_DIR"
mkdir -p $DOWNLOAD_DIR
mkdir -p $INSTALL_DIR
cd $DOWNLOAD_DIR

# Install PyTorch
if [[ $USE_TORCH ~= [oO][nN] ]]; then
  rm -rf $INSTALL_DIR/Torch
  unzip libtorch.zip
  mv libtorch $INSTALL_DIR/Torch
fi

# Install Julia
if [[ $USE_JULIA ~= [oO][nN] ]]; then
  rm -rf $INSTALL_DIR/julia
  mkdir $INSTALL_DIR/julia
  tar -xf julia.tar.gz -C $INSTALL_DIR/julia --strip-components 1
fi

# Install PIN
if [[ $USE_PIN ~= [oO][nN] ]]; then
  rm -rf $INSTALL_DIR/PIN
  mkdir $INSTALL_DIR/PIN
  tar -xf pin.tar.gz -C $INSTALL_DIR/PIN --strip-components 1
fi

# Build and install JANA2
if [[ $USE_JANA ~= [oO][nN] ]]; then
  rm -rf JANA
  unzip JANA.zip
  mv -f JANA2-2.0.6 JANA
  mkdir -p JANA/install
  export JANA_HOME=$INSTALL_DIR/JANA
  mkdir JANA/build
  cd JANA/build
  cmake .. -DCMAKE_INSTALL_PREFIX=$JANA_HOME
  make -j8 install
  cd ../..
fi 

# Build and install libdwarf
if [[ $USE_DWARF ~= [oO][nN] ]]; then
  rm -rf libdwarf
  mkdir libdwarf
  tar -xf libdwarf.tar.xz -C libdwarf --strip-components 1
  mkdir libdwarf/build
  cd libdwarf/build
  cmake .. -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR
  make install
fi

echo "Success!"
echo "Pass the following variables to CMake:"
echo "-DCMAKE_PREFIX_PATH=\"$INSTALL_DIR\""
