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
if [[ $MACOS -eq 1 ]]; then
    # Download macOS version
    unzip $DOWNLOAD_DIR/libtorch-macos-1.10.1.zip
    mv libtorch-macos-1.10.1 libtorch
else
    # Download Linux version
    read -p "Download the cxx11 ABI version? (Choose yes unless you are running something like CentOS7) [y/n]: " REPLY
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        wget --no-check-certificate https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.11.0%2Bcpu.zip
        unzip libtorch-cxx11-abi-shared-with-deps-1.11.0+cpu.zip
    else
        wget --no-check-certificate https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-1.11.0%2Bcpu.zip
        unzip libtorch-shared-with-deps-1.11.0+cpu.zip
    fi
fi

# Download PIN
DOWNLOAD_PIN=1
if [[ -d "pin" ]]; then
    read -p "Intel PIN has already been downloaded. Re-download? [y/n]: " REPLY
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        DOWNLOAD_PIN=0
    fi
fi
if [[ $DOWNLOAD_PIN -eq 1 ]]; then
    echo "Downloading Intel PIN"
    if [[ $MACOS -eq 1 ]]; then
        wget https://software.intel.com/sites/landingpage/pintool/downloads/pin-3.22-98547-g7a303a835-clang-mac.tar.gz
        tar -xf pin-3.22-98547-g7a303a835-clang-mac.tar.gz
        mv pin-3.22-98547-g7a303a835-clang-mac pin
    else
        wget https://software.intel.com/sites/landingpage/pintool/downloads/pin-3.22-98547-g7a303a835-gcc-linux.tar.gz
        tar -xf pin-3.22-98547-g7a303a835-gcc-linux.tar.gz
        mv pin-3.22-98547-g7a303a835-gcc-linux pin
    fi
fi

# Build and install JANA2
DOWNLOAD_JANA=1
if [[ -d "JANA2" ]]; then
    read -p "JANA2 has already been downloaded. Re-download? [y/n]: " REPLY
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        DOWNLOAD_JANA=0
    fi
fi
if [[ $DOWNLOAD_JANA -eq 1 ]]; then
    echo "Downloading JANA2"
    git clone http://github.com/JeffersonLab/JANA2 --branch=v2.0.6
    mkdir JANA2/install
    export JANA_HOME=$INSTALL_DIR
    mkdir JANA2/build
    cd JANA2/build
    cmake .. -DCMAKE_INSTALL_PREFIX=$JANA_HOME
    make -j8 install
    cd ../..
fi

# Build and install libdwarf
DOWNLOAD_LIBDWARF=1
if [[ -d "libdwarf-0.3.4" ]]; then
    read -p "libdwarf has already been downloaded. Re-download? [y/n]: " REPLY
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        DOWNLOAD_LIBDWARF=0
    fi
fi
if [[ $DOWNLOAD_LIBDWARF -eq 1 ]]; then
    echo "Downloading libdwarf"
    #git clone https://github.com/davea42/libdwarf-code libdwarf
    wget --no-check-certificate https://github.com/davea42/libdwarf-code/releases/download/v0.3.4/libdwarf-0.3.4.tar.xz
    tar -xf libdwarf-0.3.4.tar.xz
    mkdir libdwarf-0.3.4/build
    cd libdwarf-0.3.4/build
    cmake .. -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR
    make install
fi

echo "Success!"
echo "Pass the following variables to CMake:"
echo "-DCMAKE_PREFIX_PATH=\"$INSTALL_DIR\""
