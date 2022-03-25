#!/bin/bash

# Figure out whether we are on linux or macos
if [[ "$OSTYPE" == "darwin"* ]]; then
  MACOS=1
  echo "Detected your system is macOS"
else
  MACOS=0
  echo "Assuming your system is Linux"
fi

# Give the user the opportunity to cancel
DEPSDIR=`pwd`/deps
echo "This will download PyTorch, PIN, and JANA2 to $DEPSDIR."
read -p "Are you sure this is what you want to do? " REPLY
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    exit 1
fi

mkdir deps
cd deps

# Download PyTorch (detailed instructions on https://pytorch.org/)
if [[ $MACOS -eq 1 ]]; then
    wget --no-check-certificate https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.10.1.zip
    unzip libtorch-macos-1.10.1.zip
else
    wget --no-check-certificate https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.11.0%2Bcpu.zip
    unzip libtorch-cxx11-abi-shared-with-deps-1.11.0+cpu.zip
fi

# Download PIN
if [[ $MACOS -eq 1 ]]; then
    wget https://software.intel.com/sites/landingpage/pintool/downloads/pin-3.22-98547-g7a303a835-clang-mac.tar.gz
    unzip pin-3.22-98547-g7a303a835-gcc-linux.tar.gz
else
    wget https://software.intel.com/sites/landingpage/pintool/downloads/pin-3.22-98547-g7a303a835-gcc-linux.tar.gz
    unzip pin-3.22-98547-g7a303a835-gcc-linux.tar.gz
fi

# Build and install JANA2
git clone http://github.com/JeffersonLab/JANA2
mkdir JANA2/install
export JANA_HOME=JANA2/install
mkdir JANA2/build
cd JANA2/build
cmake .. -DCMAKE_INSTALL_PREFIX=$JANA_HOME
make -j8 install

echo "Download succeeded!"
echo "Pass these variables to CMake:"
