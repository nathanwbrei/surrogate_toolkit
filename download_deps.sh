#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Error: Wrong number of arguments"
    echo "Usage: $0 <DOWNLOAD_DIR>"
    echo ""
    exit 1
fi

DOWNLOAD_DIR=$(readlink -f $1)
echo "Download dir: $DOWNLOAD_DIR"


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
cd $DOWNLOAD_DIR

# Download Julia
DOWNLOAD_JULIA=1
if [[ -f "julia.tar.gz" ]]; then
    read -p "Julia has already been downloaded. Re-download? [y/n]: " REPLY
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        DOWNLOAD_JULIA=0
    fi
fi
if [[ $DOWNLOAD_JULIA -eq 1 ]]; then
    echo "Downloading Julia"
    if [[ $MACOS -eq 1 ]]; then
        # Download macOS version
        echo "Skipping download of Julia for macOS until we figure out which package we need"
    else
        # Download Linux version
        wget --no-check-certificate -O julia.tar.gz https://julialang-s3.julialang.org/bin/linux/x64/1.9/julia-1.9.1-linux-x86_64.tar.gz
    fi
fi

# Download PyTorch
DOWNLOAD_LIBTORCH=1
if [[ -f "libtorch.zip" ]]; then
    read -p "Torch has already been downloaded. Re-download? [y/n]: " REPLY
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        DOWNLOAD_LIBTORCH=0
    fi
fi
if [[ $DOWNLOAD_LIBTORCH -eq 1 ]]; then
    echo "Downloading Torch"
    if [[ $MACOS -eq 1 ]]; then
        # Download macOS version
        wget --no-check-certificate -O libtorch.zip https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.10.1.zip libtorch-macos-1.10.1.zip
    else
        # Download Linux version
        wget --no-check-certificate -O libtorch.zip https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.0.0%2Bcu118.zip
    fi
fi

# Download PIN
DOWNLOAD_PIN=1
if [[ -f "pin.tar.gz" ]]; then
    read -p "Intel PIN has already been downloaded. Re-download? [y/n]: " REPLY
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        DOWNLOAD_PIN=0
    fi
fi
if [[ $DOWNLOAD_PIN -eq 1 ]]; then
    echo "Downloading Intel PIN"
    if [[ $MACOS -eq 1 ]]; then
        wget --no-check-certificate -O pin.tar.gz https://software.intel.com/sites/landingpage/pintool/downloads/pin-3.22-98547-g7a303a835-clang-mac.tar.gz
    else
        wget --no-check-certificate -O pin.tar.gz https://software.intel.com/sites/landingpage/pintool/downloads/pin-3.22-98547-g7a303a835-gcc-linux.tar.gz
    fi
fi

# Download JANA2
DOWNLOAD_JANA=1
if [[ -f "JANA2.zip" ]]; then
    read -p "JANA2 has already been downloaded. Re-download? [y/n]: " REPLY
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        DOWNLOAD_JANA=0
    fi
fi
if [[ $DOWNLOAD_JANA -eq 1 ]]; then
    echo "Downloading JANA2"
    wget --no-check-certificate -O JANA2.zip https://github.com/JeffersonLab/JANA2/archive/refs/tags/v2.0.6.zip
fi

# Download libdwarf
DOWNLOAD_LIBDWARF=1
if [[ -f "libdwarf.tar.xz" ]]; then
    read -p "libdwarf has already been downloaded. Re-download? [y/n]: " REPLY
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        DOWNLOAD_LIBDWARF=0
    fi
fi
if [[ $DOWNLOAD_LIBDWARF -eq 1 ]]; then
    echo "Downloading libdwarf"
    wget --no-check-certificate -O libdwarf.tar.xz https://github.com/davea42/libdwarf-code/releases/download/v0.3.4/libdwarf-0.3.4.tar.xz
fi

echo "Success!"
