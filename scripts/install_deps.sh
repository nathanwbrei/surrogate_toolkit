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


# Pull in the environment variables describing which deps we need
# These should match PHASM's CMake flags exactly
source $(dirname "$0")/collect_deps.sh


DOWNLOAD_DIR=$(readlink -f $1)
INSTALL_DIR=$(readlink -f $2)
echo "Download dir: $DOWNLOAD_DIR"
echo "Install dir:  $INSTALL_DIR"
mkdir -p $DOWNLOAD_DIR
mkdir -p $INSTALL_DIR
cd $DOWNLOAD_DIR

# Install PyTorch
if [ $PHASM_USE_TORCH -eq 1 ]; then
  echo "Installing Torch"
  rm -rf $INSTALL_DIR/Torch
  unzip libtorch.zip
  mv libtorch $INSTALL_DIR/Torch
fi

# Install Julia
if [ $PHASM_USE_JULIA -eq 1 ]; then
  echo "Installing Julia"
  rm -rf $INSTALL_DIR/julia
  mkdir $INSTALL_DIR/julia
  tar -xf julia.tar.gz -C $INSTALL_DIR/julia --strip-components 1
fi

# Install PIN
if [ $PHASM_USE_PIN -eq 1 ]; then
  echo "Installing PIN"
  rm -rf $INSTALL_DIR/PIN
  mkdir $INSTALL_DIR/PIN
  tar -xf pin.tar.gz -C $INSTALL_DIR/PIN --strip-components 1
fi

# Build and install JANA2
if [ $PHASM_USE_JANA -eq 1 ]; then
  echo "Building and installing JANA"
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
if [ $PHASM_USE_DWARF -eq 1 ]; then
  echo "Building and installing libdwarf"
  rm -rf libdwarf
  mkdir libdwarf
  tar -xf libdwarf.tar.xz -C libdwarf --strip-components 1
  mkdir libdwarf/build
  cd libdwarf/build
  cmake .. -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR/libdwarf
  make install
fi

if [ $PHASM_USE_GEANT4 -eq 1 ]; then
  echo "Building and installing geant4"
  rm -rf geant4
  mkdir geant4
  tar -xf geant4.tar.gz -C geant4 --strip-components 1
  mkdir geant4/build
  cd geant4/build
  cmake .. -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR/geant4
  make -j8 install
fi

echo "Success!"
echo "Pass the following variables to CMake:"
echo "-DCMAKE_PREFIX_PATH=\"$INSTALL_DIR\""
