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
    if [ ${!usevar} -eq 1 ]; then
        if [[ -f $filename ]]; then
            read -p "$name has already been downloaded. Re-download? [y/n]: " REPLY
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                do_download=0
            fi
        fi
        if [[ $do_download -eq 1 ]]; then
            echo "Downloading $name"
            if [[ $HOST_IS_MACOS -eq 1 ]]; then
              curl --insecure -o $filename -L $url
            else
              wget --no-check-certificate -O $filename $url
            fi
        fi
    fi
}

# Pull in the environment variables describing which deps we need
# These should match PHASM's CMake flags exactly
source $(dirname "$0")/collect_deps.sh

echo ""
DOWNLOAD_DIR=$(readlink -f $1)
echo "DOWNLOAD_DIR $DOWNLOAD_DIR"

if [[ "$OSTYPE" == "darwin"* ]]; then
  HOST_IS_MACOS=1
else
  HOST_IS_MACOS=0
fi
echo "HOST_IS_MACOS = $HOST_IS_MACOS"


if [[ ! -n "$TARGET_IS_MACOS" ]]
then
  export TARGET_IS_MACOS=0
fi

echo "TARGET_IS_MACOS = $TARGET_IS_MACOS"


mkdir -p $DOWNLOAD_DIR
cd $DOWNLOAD_DIR

if [ $TARGET_IS_MACOS -eq 1 ]; then
  conditional_download Torch PHASM_USE_TORCH libtorch.zip https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.10.1.zip libtorch-macos-1.10.1.zip
  conditional_download Julia USE_JULIA julia.tar.gz https://julialang-s3.julialang.org/bin/mac/aarch64/1.9/julia-1.9.2-macaarch64.tar.gz
  conditional_download JANA PHASM_USE_JANA JANA.zip https://github.com/JeffersonLab/JANA2/archive/refs/tags/v2.0.6.zip
  conditional_download PIN PHASM_USE_PIN pin.tar.gz https://software.intel.com/sites/landingpage/pintool/downloads/pin-3.22-98547-g7a303a835-clang-mac.tar.gz
  conditional_download Libdwarf PHASM_USE_DWARF libdwarf.tar.xz https://github.com/davea42/libdwarf-code/releases/download/v0.3.4/libdwarf-0.3.4.tar.xz
  conditional_download geant4 PHASM_USE_GEANT4 geant4.tar.gz https://gitlab.cern.ch/geant4/geant4/-/archive/v11.0.2/geant4-v11.0.2.tar.gz

else
  if [ $PHASM_USE_CUDA -eq 1 ]; then
    conditional_download Torch PHASM_USE_TORCH libtorch_cuda.zip https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.0.0%2Bcu118.zip
  else
    conditional_download Torch PHASM_USE_TORCH libtorch_cpu.zip https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.0.1%2Bcpu.zip
  fi

  conditional_download Julia PHASM_USE_JULIA julia.tar.gz https://julialang-s3.julialang.org/bin/linux/x64/1.9/julia-1.9.1-linux-x86_64.tar.gz
  conditional_download JANA PHASM_USE_JANA JANA.zip https://github.com/JeffersonLab/JANA2/archive/refs/tags/v2.0.6.zip
  conditional_download PIN PHASM_USE_PIN pin.tar.gz https://software.intel.com/sites/landingpage/pintool/downloads/pin-3.22-98547-g7a303a835-gcc-linux.tar.gz
  conditional_download Libdwarf PHASM_USE_DWARF libdwarf.tar.xz https://github.com/davea42/libdwarf-code/releases/download/v0.3.4/libdwarf-0.3.4.tar.xz
  conditional_download geant4 PHASM_USE_GEANT4 geant4.tar.gz https://gitlab.cern.ch/geant4/geant4/-/archive/v11.0.2/geant4-v11.0.2.tar.gz
fi

echo "Success!"
