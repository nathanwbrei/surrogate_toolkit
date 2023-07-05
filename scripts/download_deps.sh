#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Error: Wrong number of arguments"
    echo "Usage: $0 <DOWNLOAD_DIR>"
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




conditional_download() {
    local name=$1
    local usevar=$2
    local filename=$3
    local url=$4
    local do_download=1
    if [[ ${!usevar} =~ [Oo][Nn] ]]; then
        if [[ -f $filename ]]; then
            read -p "$name has already been downloaded. Re-download? [y/n]: " REPLY
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                do_download=0
            fi
        fi
        if [[ $do_download -eq 1 ]]; then
            echo "Downloading $name"
            wget --no-check-certificate -O $filename $url
        fi
    fi
}

# Pull in the environment variables describing which deps we need
# These should match PHASM's CMake flags exactly
source $(dirname "$0")/collect_deps.sh

echo ""
DOWNLOAD_DIR=$(readlink -f $1)
echo "DOWNLOAD_DIR $DOWNLOAD_DIR"

mkdir -p $DOWNLOAD_DIR
cd $DOWNLOAD_DIR

if [[ "$OSTYPE" == "darwin"* ]]; then
  MACOS=1
  echo "Detected system = macOS"

  conditional_download Torch USE_TORCH libtorch.zip https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.10.1.zip libtorch-macos-1.10.1.zip
  #conditional_download Julia USE_JULIA julia.tar.gz JULIA_URL
  conditional_download JANA USE_JANA JANA.zip https://github.com/JeffersonLab/JANA2/archive/refs/tags/v2.0.6.zip
  conditional_download PIN USE_PIN pin.tar.gz https://software.intel.com/sites/landingpage/pintool/downloads/pin-3.22-98547-g7a303a835-clang-mac.tar.gz
  conditional_download Libdwarf USE_DWARF libdwarf.tar.xz https://github.com/davea42/libdwarf-code/releases/download/v0.3.4/libdwarf-0.3.4.tar.xz
  conditional_download geant4 USE_GEANT4 geant4.tar.gz https://gitlab.cern.ch/geant4/geant4/-/archive/v11.0.2/geant4-v11.0.2.tar.gz

else
  MACOS=0
  echo "Assuming system = Linux"

  conditional_download Torch USE_TORCH libtorch.zip https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.0.0%2Bcu118.zip
  conditional_download Julia USE_JULIA julia.tar.gz https://julialang-s3.julialang.org/bin/linux/x64/1.9/julia-1.9.1-linux-x86_64.tar.gz
  conditional_download JANA USE_JANA JANA.zip https://github.com/JeffersonLab/JANA2/archive/refs/tags/v2.0.6.zip
  conditional_download PIN USE_PIN pin.tar.gz https://software.intel.com/sites/landingpage/pintool/downloads/pin-3.22-98547-g7a303a835-gcc-linux.tar.gz
  conditional_download Libdwarf USE_DWARF libdwarf.tar.xz https://github.com/davea42/libdwarf-code/releases/download/v0.3.4/libdwarf-0.3.4.tar.xz
  conditional_download geant4 USE_GEANT4 geant4.tar.gz https://gitlab.cern.ch/geant4/geant4/-/archive/v11.0.2/geant4-v11.0.2.tar.gz

fi

echo "Success!"
